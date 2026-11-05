"""Portable MusicGen model bundles and trusted-checkpoint export.

A *bundle* is a self-contained, torch-free directory that carries a trained
AudioCraft ``transformer_lm`` snapshot. Metadata inspection needs only the base
tools; generation and perturbation need the MusicGen runtime. A bundle contains:

- ``config.json`` — ``macchina_ricordi.model_bundle.v1``: who the model is
  (``name``), the minimal builder config (``model_config``), the codec contract
  (``codec``, ``audio``), the trained ``segment_duration``, checksums
  (``weights``), and safe provenance (``provenance``);
- ``model.safetensors`` — the LM weights as a flat ``{name: array}`` mapping,
  with a SHA-256 recorded in the manifest;
- ``README.md`` and ``LICENSE`` — a human summary and an honest licence note.

The bundle is deliberately *portable and safe*:

- The weights are numpy/safetensors only — no pickle, so a public bundle can be
  read by any loader without ``torch.load``.
- ``read_bundle`` validates the schema, the checksum and the safetensors shape
  header *before* anything large is allocated.
- Only v1 *unconditional* ``transformer_lm`` models on the standard EnCodec
  (``facebook/encodec_32khz``, 32 kHz / 1 ch / 50 Hz) are supported. A custom
  codec or any conditioning fails loudly instead of silently writing a bundle
  the loader cannot rebuild.
- Provenance is scrubbed for local paths and logs; ``epoch`` / ``sig`` /
  ``recipe`` / ``parents`` survive.
- Writes are staged in a temp dir and renamed into place; overwrite is
  opt-in via ``force``, and unrelated files already in an existing output dir
  are never deleted.

``export_checkpoint_bundle`` turns a trusted legacy AudioCraft ``.th``
checkpoint into a bundle: it resolves the builder config from the Hydra source,
selects the requested weights (``model``, ``ema`` or ``best``), and writes a
bundle. Legacy checkpoints are trusted only because the exporter owns them; the
load always passes ``weights_only=False`` explicitly.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..musicgen.shared import sha256_file, write_json

BUNDLE_SCHEMA = "macchina_ricordi.model_bundle.v1"
MANIFEST_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
README_FILENAME = "README.md"
LICENSE_FILENAME = "LICENSE"

# Which snapshot of a training run a bundle carries: the raw model, an EMA average, or the best-valid-ce one.
WEIGHTS_STATES = ("model", "ema", "best")

# The one codec this format knows how to describe; anything else fails loudly.
ENCODEC_32KHZ_CODEC = {"name": "facebook/encodec_32khz"}
ENCODEC_32KHZ_AUDIO = {"sample_rate": 32000, "channels": 1, "frame_rate": 50.0}

SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")
_SUMMABLE_TRANSFORMER_KEYS = ("n_q", "card", "dim", "num_heads", "num_layers", "hidden_scale")

# Upper bounds on the LM architecture. A bundle manifests a config that the loader
# rebuilds strictly, so a hostile manifest must not be able to demand an absurdly
# large model (millions of modules, gigabytes allocated). These bounds comfortably
# cover every official AudioCraft LM (largest: dim 3072, num_layers 48) while still
# rejecting anything a believable public bundle would legitimately carry.
ARCHITECTURE_BOUNDS = {
    "dim": (8, 4096),
    "num_heads": (1, 256),
    "num_layers": (1, 128),
    "hidden_scale": (1, 64),
}
_BUILDER_COPY_KEYS = (
    "lm_model",
    "transformer_lm",
    "codebooks_pattern",
    "attribute_dropout",
    "classifier_free_guidance",
    "fuser",
    "conditioners",
    "dtype",
)

_SAFETENSORS_DTYPES = {
    "F64",
    "F32",
    "F16",
    "BF16",
    "I64",
    "I32",
    "I16",
    "I8",
    "U8",
    "BOOL",
    "U16",
    "U32",
    "U64",
}
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class BundleError(ValueError):
    """A model bundle is malformed, unsupported, or refused for safety."""


@dataclass(frozen=True)
class ModelBundle:
    """A bundle on disk: its top-level directory and its parsed ``config.json``."""

    path: Path
    config: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.config.get("name") or self.path.name)

    @property
    def model_config(self) -> dict[str, Any]:
        value = self.config.get("model_config")
        return value if isinstance(value, dict) else {}

    @property
    def codec(self) -> str | None:
        codec = self.config.get("codec")
        name = codec.get("name") if isinstance(codec, Mapping) else None
        return str(name) if name else None

    @property
    def audio(self) -> dict[str, Any]:
        value = self.config.get("audio")
        return value if isinstance(value, dict) else {}

    @property
    def segment_duration(self) -> float | None:
        try:
            value = self.config.get("segment_duration")
            return float(value) if value else None
        except (TypeError, ValueError):
            return None

    @property
    def weights(self) -> dict[str, Any]:
        value = self.config.get("weights")
        return value if isinstance(value, dict) else {}

    @property
    def weights_file(self) -> str:
        return str(self.weights.get("file") or WEIGHTS_FILENAME)

    @property
    def sha256(self) -> str | None:
        value = self.weights.get("sha256")
        return str(value) if value else None

    @property
    def selection(self) -> str:
        return str(self.weights.get("selection") or "model")

    @property
    def provenance(self) -> dict[str, Any]:
        value = self.config.get("provenance")
        return value if isinstance(value, dict) else {}


# ── the manifest ────────────────────────────────────────────────────────


def is_bundle(path: str | Path) -> bool:
    """Whether ``path`` (a bundle dir, or its ``config.json``) is a valid bundle.

    Never raises: it returns ``False`` for anything that is not a readable,
    matching-schema bundle with a ``model.safetensors`` present.
    """
    folder = _try_bundle_dir(path)
    if folder is None:
        return False
    manifest = folder / MANIFEST_FILENAME
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema") != BUNDLE_SCHEMA:
        return False
    return (folder / WEIGHTS_FILENAME).is_file()


def read_bundle(path: str | Path, verify: bool = True) -> ModelBundle:
    """Read and optionally verify a bundle at ``path``.

    ``path`` may be the bundle directory or its ``config.json``. With
    ``verify=True`` (default) the checksum and the safetensors shape header are
    validated before any tensor is loaded. Raises ``BundleError`` for an
    unsupported schema, malformed manifest, unsafe identity, or a failed
    checksum / header.
    """
    folder = _require_bundle_dir(path)
    manifest_path = folder / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BundleError(f"{manifest_path} is missing; {folder} is not a bundle")
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise BundleError("config.json must be a regular, bounded-size file, not a symlink")
    for filename in (README_FILENAME, LICENSE_FILENAME):
        asset = folder / filename
        if asset.is_symlink() or not asset.is_file():
            raise BundleError(f"{filename} must be a regular file in the bundle")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"{manifest_path} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"{manifest_path} must hold a JSON object")
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unsupported bundle schema {payload.get('schema')!r}; expected {BUNDLE_SCHEMA!r}")

    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise BundleError("weights must be an object")
    if weights.get("file") != WEIGHTS_FILENAME:
        raise BundleError(f"weights.file must be {WEIGHTS_FILENAME!r}, got {weights.get('file')!r}")
    digest = weights.get("sha256")
    if not isinstance(digest, str) or not _SHA_RE.match(digest):
        raise BundleError("weights.sha256 must be a lowercase 64-hex sha256")
    if weights.get("selection") not in WEIGHTS_STATES:
        raise BundleError(
            f"weights.selection {weights.get('selection')!r} is unsupported; "
            f"expected one of {', '.join(WEIGHTS_STATES)}"
        )
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise BundleError("model_config must be an object")
    _check_identity_fields(payload)
    _validate_config_fields(model_config, payload.get("codec"), payload.get("audio"), payload.get("segment_duration"))

    weights_path = folder / WEIGHTS_FILENAME
    if not weights_path.is_file():
        raise BundleError(f"weights file is missing: {weights_path}")
    if verify:
        if weights_path.is_symlink():
            raise BundleError(f"weights file must not be a symlink: {weights_path}")
        header, file_size = _read_safetensors_header(weights_path)
        _validate_safetensors_header(header, weights_path, file_size)
        _check_byte_counts(header, file_size)
        if sha256_file(weights_path) != digest:
            raise BundleError(f"weights checksum mismatch for {weights_path.name}")
    return ModelBundle(path=folder, config=payload)


def load_bundle_tensors(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load the bundle weights as ``{name: tensor}``; ``torch`` is used when present.

    ``torch`` is an optional, lazy dependency: when installed the tensors come
    back as ``torch.Tensor`` moved to ``device``; without torch (the pure
    numpy/safetensors base) they come back as ``numpy.ndarray`` and only
    ``device="cpu"`` is supported.
    """
    bundle = read_bundle(path, verify=True)
    weights_path = bundle.path / WEIGHTS_FILENAME
    torch = _import_torch_or_none()
    if torch is not None:
        from safetensors.torch import load_file as _sf_torch

        return _sf_torch(str(weights_path), device=device)
    if device != "cpu":
        raise BundleError("without torch, load_bundle_tensors only supports device='cpu'")
    from safetensors.numpy import load_file as _sf_numpy

    return _sf_numpy(str(weights_path))


def bundle_tensor_shapes(path: str | Path) -> dict[str, tuple[list[int], str]]:
    """Map each weight name to ``(shape, safetensors-dtype)`` from the header, without loading tensor data.

    Reads and verifies the bundle (same checks as ``read_bundle(verify=True)``),
    then returns each tensor's ``shape`` and ``dtype`` straight from the
    safetensors header. The loader uses this to confirm a bundle's weights match
    the declared architecture *before* anything is allocated into memory.
    """
    bundle = read_bundle(path, verify=True)
    header, _ = _read_safetensors_header(bundle.path / WEIGHTS_FILENAME)
    return {
        name: (list(entry.get("shape") or []), str(entry.get("dtype")))
        for name, entry in header.items()
        if name != "__metadata__"
    }


def validate_model_config(
    model_config: Mapping[str, Any],
    codec: Mapping[str, Any],
    audio: Mapping[str, Any],
    segment_duration: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Check a bundle config against the v1 contract; returns normalized values.

    Raises ``BundleError`` when the model is conditioned, the LM is not a v1
    ``transformer_lm``, a required shape is not a positive integer, the dtype is
    unknown, the codec is not the standard EnCodec 32 kHz, or the segment
    duration is not a finite positive number.
    """
    if not isinstance(model_config, dict):
        raise BundleError("model_config must be an object")
    if model_config.get("lm_model") != "transformer_lm":
        raise BundleError(
            f"unsupported lm_model {model_config.get('lm_model')!r}: this format bundles v1 "
            "AudioCraft 'transformer_lm' models only"
        )
    conditioners = model_config.get("conditioners")
    if conditioners and not _empty_mapping(conditioners):
        raise BundleError(
            "conditioned models are unsupported: found a non-empty 'conditioners' section; "
            "export an unconditional transformer_lm model"
        )
    transformer = model_config.get("transformer_lm")
    if not isinstance(transformer, dict):
        raise BundleError("model_config.transformer_lm must be an object")
    for key in _SUMMABLE_TRANSFORMER_KEYS:
        value = transformer.get(key)
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        if key == "hidden_scale":
            valid = _optional_number(value) is not None and math.isfinite(float(value)) and value > 0
        if not valid:
            raise BundleError(f"model_config.transformer_lm.{key} must be a positive number (got {value!r})")
    if transformer["dim"] % transformer["num_heads"]:
        raise BundleError("transformer_lm.dim must be divisible by num_heads")
    if transformer["n_q"] != 4 or transformer["card"] != 2048:
        raise BundleError("The standard EnCodec codec requires n_q=4 and card=2048")
    for key, (low, high) in ARCHITECTURE_BOUNDS.items():
        number = _optional_number(transformer.get(key))
        if number is None or not (low <= number <= high):
            raise BundleError(
                f"model_config.transformer_lm.{key} is {transformer.get(key)!r}; this loader bounds it to "
                f"[{low}, {high}] so a hostile manifest cannot demand an absurdly large architecture"
            )
    dtype = model_config.get("dtype")
    if dtype not in SUPPORTED_DTYPES:
        raise BundleError(f"unsupported dtype {dtype!r}; expected one of {', '.join(SUPPORTED_DTYPES)}")

    codec = dict(codec or {})
    if codec.get("name") != ENCODEC_32KHZ_CODEC["name"]:
        raise BundleError(f"unsupported codec {codec.get('name')!r}; only {ENCODEC_32KHZ_CODEC['name']!r} is supported")
    audio = dict(audio or {})
    if int(audio.get("sample_rate") or 0) != 32000:
        raise BundleError("audio.sample_rate must be 32000 for the standard EnCodec codec")
    if int(audio.get("channels") or 0) != 1:
        raise BundleError("audio.channels must be 1 for the standard EnCodec codec")
    if (
        not math.isfinite(float(audio.get("frame_rate") or 0.0))
        or abs(float(audio.get("frame_rate") or 0.0) - 50.0) > 1e-9
    ):
        raise BundleError("audio.frame_rate must be 50.0 for the standard EnCodec codec")

    segment = _optional_number(segment_duration)
    if segment is None or not math.isfinite(segment) or segment <= 0:
        raise BundleError(f"segment_duration must be a finite number > 0 (got {segment_duration!r})")
    dataset = model_config.get("dataset")
    declared = _optional_number(dataset.get("segment_duration")) if isinstance(dataset, dict) else None
    if declared is not None and abs(declared - segment) > 1e-9:
        raise BundleError(
            f"model_config.dataset.segment_duration {declared!r} does not match segment_duration {segment!r}"
        )
    return dict(codec), dict(audio), segment


def write_bundle(
    output_dir: str | Path,
    tensors: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any],
    codec: Mapping[str, Any],
    audio: Mapping[str, Any],
    segment_duration: float,
    name: str,
    provenance: Mapping[str, Any] | None = None,
    weights_state: str = "model",
    force: bool = False,
    license_text: str | None = None,
    readme_text: str | None = None,
) -> ModelBundle:
    """Write a self-contained bundle to ``output_dir`` and return the verified result.

    ``tensors`` is a flat ``{name: array}`` mapping (numpy arrays, or torch
    tensors that are detached to CPU). ``model_config`` is the minimal builder
    config, validated against the v1 unconditional ``transformer_lm``/EnCodec
    contract. ``license_text`` writes an explicit ``LICENSE`` (a perturbed or
    derivative bundle should pass the source bundle's ``LICENSE`` verbatim so
    attributions survive; the default note invents no model licence). ``readme_text``
    replaces the generated ``README.md``. Writes are staged in a temp dir and
    renamed in; ``force=False`` refuses to overwrite an existing output dir, and
    unrelated files already there are never deleted.
    """
    model_config = dict(model_config)
    codec_norm, audio_norm, segment = validate_model_config(model_config, codec, audio, segment_duration)
    stored_config = _minimal_config_copy(model_config)
    stored_config["dataset"] = {"segment_duration": segment}
    _validate_config_fields(stored_config, codec_norm, audio_norm, segment)
    weights_state = _check_weights_state(weights_state)
    output_dir = Path(output_dir)
    _check_name(name)
    provenance = _safe_provenance(provenance)

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"bundle output path is not a directory: {output_dir}")
    if output_dir.exists() and output_dir.is_symlink():
        raise BundleError(f"refusing to write into a symlinked output dir: {output_dir}")
    if output_dir.exists() and not force:
        raise FileExistsError(f"output bundle already exists: {output_dir}. Pass force=True to overwrite.")

    flat = _coerce_tensors(tensors)
    if not flat:
        raise BundleError("cannot write a bundle with no weights")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name or 'bundle'}.tmp_", dir=output_dir.parent))
    try:
        staged = tmp / WEIGHTS_FILENAME
        _save_safetensors(flat, staged)
        digest = sha256_file(staged)
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "name": name,
            "model_config": stored_config,
            "codec": dict(codec_norm),
            "audio": dict(audio_norm),
            "segment_duration": segment,
            "weights": {"file": WEIGHTS_FILENAME, "sha256": digest, "selection": weights_state},
            "provenance": provenance,
        }
        write_json(tmp / MANIFEST_FILENAME, manifest)
        readme_body = readme_text if readme_text is not None else _readme(manifest)
        (tmp / README_FILENAME).write_text(readme_body, encoding="utf-8")
        (tmp / LICENSE_FILENAME).write_text(_license(license_text), encoding="utf-8")
        _stage_into(tmp, output_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return read_bundle(output_dir, verify=True)


# ── trusted checkpoint export ───────────────────────────────────────────


def select_checkpoint_weights(state: dict[str, Any], weights_state: str = "model") -> dict[str, Any]:
    """Pick a weight mapping out of a legacy AudioCraft checkpoint ``state``.

    ``state`` is the dict ``torch.load`` produced. ``weights_state`` selects:

    - ``"model"`` (default) — the raw current model weights (``state["model"]``,
      or the bare ``state`` itself when the checkpoint *is* a model ``state_dict``);
    - ``"best"`` — best-state weights, at ``best_state.model`` or, failing that,
      ``fsdp_best_state.model``;
    - ``"ema"`` — EMA weights, at ``ema.model`` / ``ema.state.model``.

    A requested ``ema``/``best`` that is absent raises — it never falls back to
    another state.
    """
    if not isinstance(state, dict):
        raise TypeError("checkpoint must load to a top-level dictionary")
    weights_state = _check_weights_state(weights_state)
    if weights_state == "model":
        model = state.get("model")
        if isinstance(model, dict) and _weights_mapping(model):
            return model
        if _weights_mapping(state):
            return state
        raise BundleError("no raw model weights found: expected state['model'] (or the bare state dict itself)")
    if weights_state == "best":
        for nested in ("best_state.model", "fsdp_best_state.model"):
            candidate = _nested_lookup(state, nested)
            if isinstance(candidate, dict) and _weights_mapping(candidate):
                return candidate
        best_state = state.get("best_state")
        if isinstance(best_state, dict) and _weights_mapping(best_state):
            return best_state
        raise BundleError(
            "no best-state weights found: expected state['best_state']['model'] or state['fsdp_best_state']['model']"
        )
    # ema
    for nested in ("ema.model", "ema.state.model"):
        candidate = _nested_lookup(state, nested)
        if isinstance(candidate, dict) and _weights_mapping(candidate):
            return candidate
    ema = state.get("ema")
    if isinstance(ema, dict) and _weights_mapping(ema):
        return ema
    raise BundleError("no EMA weights found: expected state['ema']['model'] or state['ema']['state']['model']")


def export_checkpoint_bundle(
    checkpoint_path: str | Path,
    config_path: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    *,
    name: str,
    weights_state: str = "model",
    provenance: Mapping[str, Any] | None = None,
    force: bool = False,
    torch_module: Any | None = None,
    loader: Any | None = None,
    license_text: str | None = None,
    readme_text: str | None = None,
) -> ModelBundle:
    """Export a trusted legacy AudioCraft ``.th`` checkpoint as a portable bundle.

    ``config_path`` is the resolved Hydra builder source (an ``.hydra/config.yaml``
    path, a plain YAML path, or a dict); only the whitelisted builder fields are
    recorded, never datasource/training/cache/paths. ``loader`` injects the
    checkpoint reader (default ``torch.load(..., weights_only=False)`` on CPU);
    ``torch_module`` supplies the torch instance used to detach tensors.
    ``license_text``/``readme_text`` pass through to ``write_bundle``.
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    raw_config = _load_source_config(config_path)
    model_config = resolve_minimal_model_config(raw_config)
    codec, audio, segment_duration = _derive_audio(raw_config)
    loader = loader or _make_default_loader(torch_module)
    state = loader(checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must load to a top-level dictionary")
    weights = select_checkpoint_weights(state, weights_state)
    tensors = _flatten_to_numpy(weights, torch_module)
    return write_bundle(
        output_dir,
        tensors,
        model_config=model_config,
        codec=codec,
        audio=audio,
        segment_duration=segment_duration,
        name=name,
        provenance=provenance,
        weights_state=weights_state,
        force=force,
        license_text=license_text,
        readme_text=readme_text,
    )


def resolve_minimal_model_config(source: Any) -> dict[str, Any]:
    """The whitelisted build-config fields out of a resolved Hydra source.

    Only what ``get_lm_model`` needs to rebuild the LM is kept; every datasource,
    cache, training and path key is dropped.
    """
    raw = _load_source_config(source)
    out: dict[str, Any] = {}
    for key in _BUILDER_COPY_KEYS:
        if key in raw:
            out[key] = _jsonable(raw[key])
    out.setdefault("lm_model", "")
    segment = _nested_number(raw, "dataset", "segment_duration")
    if segment is not None:
        out["dataset"] = {"segment_duration": segment}
    if isinstance(out.get("transformer_lm"), dict):
        out["transformer_lm"] = dict(out["transformer_lm"])
    return out


# ── internals: config ───────────────────────────────────────────────────


def _load_source_config(source: Any) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    # OmegaConf resolves `${...}` interpolations ("resolved source Hydra").
    try:
        from omegaconf import OmegaConf  # runtime-only; the base passes dicts

        loaded = OmegaConf.load(path)
        return OmegaConf.to_container(loaded, resolve=True) or {}
    except Exception:
        pass
    import yaml

    payload = yaml.safe_load(text)
    return payload if isinstance(payload, dict) else {}


def _derive_audio(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    segment = _nested_number(raw, "dataset", "segment_duration")
    checkpoint = raw.get("compression_model_checkpoint", "//pretrained/facebook/encodec_32khz")
    if checkpoint not in ("//pretrained/facebook/encodec_32khz", "facebook/encodec_32khz"):
        raise BundleError("Unsupported codec checkpoint; v1 export requires the standard facebook/encodec_32khz codec")
    if (raw.get("interleave_stereo_codebooks") or {}).get("use", False):
        raise BundleError("Stereo codebook interleaving is not supported by bundle v1")
    codec = dict(ENCODEC_32KHZ_CODEC)
    audio = dict(ENCODEC_32KHZ_AUDIO)
    if isinstance(raw.get("sample_rate"), int | float):
        audio["sample_rate"] = int(raw["sample_rate"])
    if isinstance(raw.get("channels"), int):
        audio["channels"] = int(raw["channels"])
    if isinstance(raw.get("compression_model_framerate"), int | float):
        audio["frame_rate"] = float(raw["compression_model_framerate"])
    return codec, audio, segment


def _minimal_config_copy(model_config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _BUILDER_COPY_KEYS:
        if key in model_config:
            out[key] = _jsonable(model_config[key])
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_jsonable(v) for v in value]
    return None


def _nested_number(raw: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = raw
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return _optional_number(current)


def _nested_lookup(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


# ── internals: manifest safety ──────────────────────────────────────────


def _check_identity_fields(payload: dict[str, Any]) -> None:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise BundleError("manifest must carry a non-empty 'name'")
    _check_name(name)


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name.strip():
        raise BundleError("bundle 'name' must be a non-empty string")
    if "\x00" in name or "/" in name or "\\" in name:
        raise BundleError(f"bundle 'name' must not contain path separators or NUL: {name!r}")
    if name in (".", ".."):
        raise BundleError(f"invalid bundle name: {name!r}")


def _validate_config_fields(model_config: dict[str, Any], codec: Any, audio: Any, segment_duration: Any) -> None:
    if codec is None or audio is None:
        raise BundleError("manifest must carry 'codec' and 'audio'")
    allowed = set(_BUILDER_COPY_KEYS) | {"dataset"}
    if set(model_config) - allowed:
        raise BundleError("model_config contains non-builder fields")
    for required in _BUILDER_COPY_KEYS:
        if required not in model_config:
            raise BundleError(f"model_config is missing builder field {required!r}")
    if _optional_number(segment_duration) is None:
        raise BundleError("manifest must carry a numeric segment_duration")
    try:
        serialized = json.dumps(model_config, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise BundleError("model_config must contain finite JSON values") from exc
    if "${" in serialized:
        raise BundleError("model_config must be resolved; interpolations are not allowed")
    if not isinstance(codec, dict) or not isinstance(audio, dict):
        raise BundleError("'codec' and 'audio' must be objects")
    dataset = model_config.get("dataset")
    nested = _optional_number(dataset.get("segment_duration")) if isinstance(dataset, dict) else None
    recorded = _optional_number(segment_duration)
    if nested is not None and recorded is not None and abs(nested - recorded) > 1e-9:
        raise BundleError(
            f"model_config.dataset.segment_duration {nested!r} does not match segment_duration {recorded!r}"
        )
    validate_model_config(model_config, codec, audio, nested if nested is not None else recorded)


# ── internals: provenance scrubbing ─────────────────────────────────────


def _safe_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if provenance is None:
        return {}
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be a dict")
    cleaned: dict[str, Any] = {}
    for key, value in provenance.items():
        safe = _safe_prov_value(value)
        if safe is not None:
            cleaned[str(key)] = safe
    return cleaned


def _safe_prov_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {str(k): c for k, child in value.items() if (c := _safe_prov_value(child)) is not None}
        return cleaned or None
    if isinstance(value, (list, tuple)):
        items = [c for child in value if (c := _safe_prov_value(child)) is not None]
        return items or None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or "\n" in stripped or "\r" in stripped:
            return None
        if _looks_like_path(stripped):
            return None
        return stripped
    return None


def _looks_like_path(value: str) -> bool:
    if value.startswith(("/", "~", "..", "\\\\", "//")):
        return True
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True  # Windows drive, e.g. C:\…
    return "/" in value or "\\" in value


# ── internals: safetensors ──────────────────────────────────────────────


def _coerce_tensors(tensors: Mapping[str, Any]) -> dict[str, Any]:
    torch = _import_torch_or_none()
    if not tensors:
        return {}
    flat: dict[str, Any] = {}
    for key, value in tensors.items():
        if not isinstance(key, str) or not key:
            raise BundleError("weight names must be non-empty strings")
        if "/" in key or "\\" in key or "\x00" in key:
            raise BundleError(f"refusing weight name with path separators: {key!r}")
        if torch is not None and _is_torch_tensor(value):
            flat[key] = value.detach().cpu().contiguous().clone()
        else:
            flat[key] = np.asarray(value).copy(order="C")
    return flat


def _flatten_to_numpy(weights: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    for key, value in weights.items():
        if not _is_torch_tensor(value) and not isinstance(value, np.ndarray):
            raise BundleError(f"weight {key!r} is not a tensor or numpy array")
    return _coerce_tensors(weights)


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "dim") and not isinstance(value, np.ndarray)


def _weights_mapping(mapping: dict[str, Any]) -> bool:
    if not mapping:
        return False
    for value in mapping.values():
        if isinstance(value, np.ndarray) and value.ndim >= 0:
            return True
        if _is_torch_tensor(value):
            return True
    return False


def _save_safetensors(tensors: dict[str, Any], path: Path) -> None:
    if any(_is_torch_tensor(value) for value in tensors.values()):
        import torch
        from safetensors.torch import save_file

        values = {
            name: value if _is_torch_tensor(value) else torch.from_numpy(value) for name, value in tensors.items()
        }
        save_file(values, str(path))
    else:
        from safetensors.numpy import save_file

        save_file(tensors, str(path))


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise BundleError(f"{path.name} is too short to be a safetensors file")
        length = struct.unpack("<Q", head)[0]
        if length > _MAX_HEADER_BYTES:
            raise BundleError(f"{path.name} safetensors header is unreasonably large")
        header_bytes = handle.read(length)
        if len(header_bytes) < length:
            raise BundleError(f"{path.name} safetensors header is truncated")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f"{path.name} has a malformed safetensors header") from exc
    if not isinstance(header, dict):
        raise BundleError(f"{path.name} safetensors header must be an object")
    return header, path.stat().st_size


def _validate_safetensors_header(header: dict[str, Any], path: Path, file_size: int) -> None:
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise BundleError(f"{path.name}: tensor {key!r} header must be an object")
        if entry.get("dtype") not in _SAFETENSORS_DTYPES:
            raise BundleError(f"{path.name}: tensor {key!r} has unsupported dtype {entry.get('dtype')!r}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape):
            raise BundleError(f"{path.name}: tensor {key!r} has invalid shape {shape!r}")


def _check_byte_counts(header: dict[str, Any], file_size: int) -> None:
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise BundleError(f"tensor {key!r} has missing data offsets")
        start, end = offsets
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= file_size):
            raise BundleError(f"tensor {key!r} data range {offsets} exceeds the file")


# ── internals: atomic staging ───────────────────────────────────────────


def _stage_into(tmp: Path, output_dir: Path) -> None:
    if not output_dir.exists():
        os.replace(tmp, output_dir)  # atomic rename on the same filesystem
        return
    # Refuse unsafe destinations before replacing anything. Commit the manifest
    # last; readers verify the weights hash and never accept a mixed snapshot.
    entries = sorted(tmp.iterdir(), key=lambda entry: entry.name == MANIFEST_FILENAME)
    for entry in entries:
        target = output_dir / entry.name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise BundleError(f"Refusing unsafe existing bundle asset: {target}")
    for entry in entries:
        os.replace(entry, output_dir / entry.name)


# ── internals: misc ─────────────────────────────────────────────────────


def _try_bundle_dir(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.is_file() and candidate.name == MANIFEST_FILENAME:
        return candidate.parent
    if candidate.is_dir():
        return candidate
    return None


def _require_bundle_dir(path: str | Path) -> Path:
    folder = _try_bundle_dir(path)
    if folder is None:
        raise BundleError(f"not a bundle directory: {path}")
    return folder.resolve()


def _empty_mapping(value: Any) -> bool:
    return isinstance(value, dict | list | tuple) and len(value) == 0


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _check_weights_state(weights_state: str) -> str:
    if weights_state not in WEIGHTS_STATES:
        raise BundleError(f"unsupported weights_state {weights_state!r}; expected one of {', '.join(WEIGHTS_STATES)}")
    return weights_state


def _import_torch_or_none() -> Any:
    try:
        import torch

        return torch
    except Exception:
        return None


def _make_default_loader(torch_module: Any | None) -> Any:
    def load(path: Path) -> Any:
        torch = torch_module or _import_torch_or_none()
        if torch is None:
            raise BundleError("export requires torch to read a legacy .th checkpoint; install torch in the runtime")
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # very old torch without weights_only
            return torch.load(path, map_location="cpu")

    return load


# ── internals: sidecar text ─────────────────────────────────────────────


def _readme(manifest: dict[str, Any]) -> str:
    name = manifest.get("name", "model")
    selection = manifest.get("weights", {}).get("selection", "model")
    model_config = manifest.get("model_config", {}) if isinstance(manifest.get("model_config"), dict) else {}
    shape = model_config.get("transformer_lm", {}) if isinstance(model_config.get("transformer_lm"), dict) else {}
    dim, layers = shape.get("dim"), shape.get("num_layers")
    head = f"{name} ({int(dim)} dim, {int(layers)} layers)" if dim and layers else str(name)
    return f"""# {head}

A portable AudioCraft model bundle (schema `{BUNDLE_SCHEMA}`).

* `config.json` — identity, the minimal builder config, the codec contract and
  provenance; the `weights.sha256` pins `model.safetensors`.
* `model.safetensors` — the transformer-LM weights, a flat `{{name: array}}`
  mapping of the `{selection}` state, readable with `safetensors` without torch
  or pickle.
* `LICENSE` — where this model's licence stands; see its text.

The bundle covers only the LM. The audio codec is the separate standard EnCodec
(`facebook/encodec_32khz`, 32 kHz / 1 ch / 50 Hz), which this bundle does not
carry. The trained segment duration is `{manifest.get("segment_duration")!r}s`;
device is chosen at load time; the saved dtype and exact tensor values are retained.

## Builder config (the whitelisted fields)

```json
{json.dumps(model_config, indent=2)}
```
"""


def _license(license_text: str | None) -> str:
    if license_text is not None:
        return license_text
    return (
        "MODEL BUNDLE LICENSE NOTE\n"
        "=========================\n"
        "\n"
        "This bundle ships a trained transformer-LM weights file "
        f"(`{WEIGHTS_FILENAME}`) and its manifest. This file does NOT grant, claim or assert\n"
        "any licence over the model itself: the model's weights belong to whoever trained them\n"
        "and are governed by whatever licence that party holds. If you redistribute this bundle\n"
        "and you hold an explicit licence for the model (your own trained weights, or a licence\n"
        "you were granted), attach that licence here or re-export with an explicit license_text.\n"
        "\n"
        "AUTOGENERATED DEFAULT: no licence is asserted by this file alone.\n"
        "\n"
        "THIRD-PARTY CODEC\n"
        "-----------------\n"
        "The audio codec is the standard Meta EnCodec (`facebook/encodec_32khz`), a separate\n"
        "component this bundle does NOT include. Meta releases EnCodec weights under\n"
        "CC-BY-NC 4.0; the codec's licence is Meta's and must be satisfied separately. This\n"
        "bundle makes no claim over the codec.\n"
        "\n"
        "If you add model weights to a bundle, keep their licence alongside and make sure it\n"
        "permits what you are doing — never rely on this default note."
    )
