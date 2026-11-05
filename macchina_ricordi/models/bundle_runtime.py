"""Load a portable model bundle into a playable ``MusicGen`` without Dora, a solver, or any XP.

A bundle is a directory written by ``macchina_ricordi.models.bundle`` (schema
``macchina_ricordi.model_bundle.v1``): a JSON manifest plus the LM weights in a
safetensors file, enough to rebuild a model from the resolved builder fields
only. This module is the *loading* half of that contract and is deliberately
independent of Dora: it builds the LM directly from the bundled config and
loads the EnCodec codec from its pretrained checkpoint, so a bundle plays on a
fresh AudioCraft install with no experiment records present.

Usage::

    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    model, source = load_bundle_model(path, device="cpu")

``load_bundle_model`` returns ``(MusicGen, sourceMetadata)``; the caller
applies its own inference optimisations (debug checkpointing off, fp16, ...).
The loader never creates a solver, XP, or dataloader.

All heavy imports (torch, audiocraft, omegaconf, the shared bundle module) are
lazy inside the call so that importing this module — and the CLI that only
occasionally reaches it — requires no torch and does no audiocraft work.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Matches scripts/run_musicgen_generate.py: AudioCraft's MusicGen constructor
# rejects a max_duration below the default extend stride (18.0s + 1). We build
# with a larger value then clamp to the bundle's real segment length.
_AUDIOCRAFT_CONSTRUCTOR_EXTEND_STRIDE = 18.0


class NotABundleError(ValueError):
    """Raised when a path is not a recognized model bundle."""


class BundleRuntimeError(RuntimeError):
    """Raised when bundle model loading is requested but the shared bundle API is unavailable."""


class _BundleDict:
    """Normalise whatever ``read_bundle`` returns into a stable mapping.

    The shared core owns ``read_bundle``'s return type; the runtime only
    depends on a mapping with the documented schema fields, so it accepts a
    plain dict, an object exposing ``to_dict()``, or the core's ``ModelBundle``
    (which carries the raw manifest as ``.config``).
    """

    __slots__ = ("_values",)

    def __init__(self, raw: Any) -> None:
        if hasattr(raw, "to_dict"):
            values = raw.to_dict()
            values = values.get("payload") if isinstance(values, dict) and "payload" in values else values
        elif isinstance(getattr(raw, "config", None), dict):
            values = raw.config
        else:
            values = raw
        if not isinstance(values, dict):
            raise BundleRuntimeError(f"Bundle reader returned an unsupported value: {type(raw).__name__}")
        # Accept both the raw manifest dict and a ModelBundle's `.config`.
        if "model_config" in values and "weights" in values:
            self._values = values
        else:
            self._values = getattr(values, "config", values) if isinstance(values, dict) else values
        if not isinstance(self._values, dict) or "model_config" not in self._values:
            raise BundleRuntimeError(f"Bundle reader did not return a v1 manifest mapping: {type(raw).__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def to_dict(self) -> dict[str, Any]:
        return self._values


def _load_shared_bundle_api():
    """Import the shared core bundle functions, or raise for an unsupported install."""
    try:
        from .bundle import (
            bundle_tensor_shapes as _bundle_tensor_shapes,
        )
        from .bundle import (
            is_bundle as _is_bundle,
        )
        from .bundle import (
            load_bundle_tensors as _load_bundle_tensors,
        )
        from .bundle import (
            read_bundle as _read_bundle,
        )
    except ImportError as exc:  # bundle.py is owned by the core worker; not present on bare installs
        raise BundleRuntimeError(
            "Bundle support is not available in this install: "
            "the shared helper macchina_ricordi.models.bundle could not be imported."
        ) from exc
    return _is_bundle, _read_bundle, _load_bundle_tensors, _bundle_tensor_shapes


def is_bundle_dir(path: str | Path) -> bool:
    """True when ``path`` is a recognized model bundle directory (safe to call anywhere).

    Returns False (never raises) when the shared bundle API is unavailable, so
    callers that can also fall back to the legacy XP/checkpoint flow degrade
    gracefully on installs without bundle support.
    """
    try:
        is_bundle, _read, _load, _shapes = _load_shared_bundle_api()
    except BundleRuntimeError:
        return False
    try:
        return bool(is_bundle(path))
    except Exception:  # noqa: BLE001 - a malformed/inaccessible path is simply not a bundle here
        return False


def bundle_dir_or_none(value: str | Path | None) -> Path | None:
    """``value`` as a resolved bundle dir, or ``None`` when it isn't one."""
    if value in (None, ""):
        return None
    candidate = Path(value).expanduser()
    if not candidate.exists() or not candidate.is_dir():
        return None
    if not is_bundle_dir(candidate):
        return None
    return candidate.resolve()


def bundle_segment_duration(path: str | Path) -> float | None:
    """The trained ``segment_duration`` of a bundle, or ``None`` when it isn't one.

    Lets a caller that cannot (or must not) load the whole model still read the
    trained horizon off an existing bundle — e.g. the stream session's horizon
    guard. Safe to call anywhere; returns ``None`` for non-bundles.
    """
    try:
        _is_bundle, read_bundle, _load, _shapes = _load_shared_bundle_api()
    except BundleRuntimeError:
        return None
    try:
        if not _is_bundle(path):
            return None
        bundle = _BundleDict(read_bundle(path, verify=False))
        return float(_segment_duration(bundle))
    except Exception:  # noqa: BLE001 - a malformed bundle has no horizon we can trust
        return None


def _lm_config(model_config: dict[str, Any], device: str) -> dict[str, Any]:
    """The resolved builder config for ``builders.get_lm_model``, plus the device."""
    config = dict(model_config)
    config["device"] = device
    dataset = config.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        dataset = dict(dataset)
        config["dataset"] = dataset
    return config


def _codec_config(device: str) -> dict[str, Any]:
    """The minimal wrapping config for the codec: wrapped model only needs the device/I/O dims.

    ``CompressionSolver.wrapped_model_from_checkpoint`` is a static method that
    delegates to ``model_from_checkpoint`` for the actual weights and uses the
    config only for optional wrapping (interleave / ``compression_model_n_q``),
    both of which a standard mono bundle leaves off.
    """
    return {"device": device}


# Bounds on the LM architecture the runtime is willing to rebuild. Kept in sync
# with ``macchina_ricordi.models.bundle.ARCHITECTURE_BOUNDS``; the copy here is
# a belt-and-suspenders so the runtime never starts an (even meta) build on a
# manifest an older/absent shared bundle module did not bound.
_ARCH_BOUNDS = {
    "dim": (8, 4096),
    "num_heads": (1, 256),
    "num_layers": (1, 128),
    "hidden_scale": (1, 64),
}


def _validate_arch_bounds(model_config: dict[str, Any]) -> None:
    """Reject an untrusted bundle whose declared architecture is implausibly large.

    Called *before* any model construction, even the allocation-free meta build,
    so a hostile manifest cannot make the loader spin up millions of modules.
    """
    transformer = model_config.get("transformer_lm")
    if not isinstance(transformer, dict):
        raise BundleRuntimeError("model_config.transformer_lm must be an object")
    if transformer.get("n_q") != 4 or transformer.get("card") != 2048:
        raise BundleRuntimeError(
            f"unsupported codebook geometry n_q={transformer.get('n_q')!r}, "
            f"card={transformer.get('card')!r}; only n_q=4 / card=2048 (standard EnCodec) is supported"
        )
    for key, (low, high) in _ARCH_BOUNDS.items():
        value = transformer.get(key)
        number = value if isinstance(value, int | float) and not isinstance(value, bool) else None
        if number is None or not math.isfinite(number) or not (low <= number <= high):
            raise BundleRuntimeError(
                f"refusing an unreasonably large architecture: transformer_lm.{key} is "
                f"{value!r} (allowed [{low}, {high}])"
            )


def _meta_validate_lm(
    builders: Any,
    model_config: dict[str, Any],
    tensor_shapes: dict[str, tuple[list[int], str]],
) -> None:
    """Instantiate the declared LM on the meta device and require the weights to match exactly.

    The meta device allocates no CPU/GPU storage, so this validates a hostile
    bundle's full tensor-key set and every shape against the described
    architecture before the loader commits any real memory. Raises
    ``BundleRuntimeError`` on any key or shape mismatch.
    """
    from omegaconf import OmegaConf

    meta_lm = builders.get_lm_model(OmegaConf.create(_lm_config(model_config, "meta")))
    try:
        expected = meta_lm.state_dict()
    finally:
        del meta_lm
    expected_keys = set(expected)
    file_keys = set(tensor_shapes)
    if expected_keys != file_keys:
        missing = sorted(expected_keys - file_keys)
        extra = sorted(file_keys - expected_keys)
        detail = []
        if missing:
            detail.append(f"missing from the bundle: {', '.join(missing[:8])}")
        if extra:
            detail.append(f"not part of the declared architecture: {', '.join(extra[:8])}")
        raise BundleRuntimeError("bundle weights do not match the declared architecture (" + "; ".join(detail) + ")")
    for name in expected_keys:
        expected_shape = tuple(expected[name].shape)
        file_shape = tuple(tensor_shapes[name][0])
        if expected_shape != file_shape:
            raise BundleRuntimeError(
                f"bundle weight {name} has shape {file_shape}, but the declared architecture requires {expected_shape}"
            )


def load_bundle_model(path: str | Path, device: str = "cpu") -> tuple[Any, dict[str, object]]:
    """Load a bundle into ``(MusicGen, sourceMetadata)``.

    Builds the LM directly from the bundle's resolved builder fields
    (``audiocraft.models.builders.get_lm_model``), loads its selected weights
    strictly, and attaches the standard EnCodec codec via
    ``CompressionSolver.wrapped_model_from_checkpoint``. The returned
    ``MusicGen.max_duration`` is the bundle's ``segment_duration``.

    Raises ``NotABundleError`` when ``path`` is not a bundle, and
    ``BundleRuntimeError`` when the shared bundle API is unavailable.
    """
    bundle_path = Path(path).expanduser().resolve()
    is_bundle, read_bundle, load_bundle_tensors, bundle_tensor_shapes = _load_shared_bundle_api()

    if not is_bundle(bundle_path):
        raise NotABundleError(
            f"Not a model bundle: {bundle_path}. Expected a directory written by "
            "macchina_ricordi.models.bundle (schema macchina_ricordi.model_bundle.v1)."
        )
    bundle = _BundleDict(read_bundle(bundle_path, verify=True))

    model_config = bundle.get("model_config") or {}
    codec_spec = bundle.get("codec") or {}
    audio = bundle.get("audio") or {}
    weights = bundle.get("weights") or {}
    name = str(bundle.get("name") or "")
    provenance = bundle.get("provenance") or {}
    if isinstance(provenance, dict):
        provenance = dict(provenance)
    else:
        provenance = {}

    # codec is either a mapping {"name": ...} or the core's bare codec string.
    codec_name = (
        codec_spec.get("name") if isinstance(codec_spec, dict) else (str(codec_spec) if codec_spec else None)
    ) or "facebook/encodec_32khz"

    # ---- pre-flight: bound the architecture and prove the weights fit before allocating ----.
    # A bundle manifest is untrusted input: bound the declared geometry before any
    # construction (even the allocation-free meta build), then build the model on the
    # meta device and require the safetensors header's key set and shapes to match the
    # architecture exactly. Only a valid, self-consistent bundle proceeds to a real
    # CPU/GPU allocation and a codec load.
    _validate_arch_bounds(model_config)

    # ---- LM: build directly from the resolved builder fields, no solver ----
    from audiocraft.models import MusicGen, builders
    from audiocraft.solvers.compression import CompressionSolver
    from omegaconf import OmegaConf

    _meta_validate_lm(builders, model_config, bundle_tensor_shapes(bundle_path))

    lm = builders.get_lm_model(OmegaConf.create(_lm_config(model_config, device)))
    # A bundle's safetensors already carries the selected snapshot (model/ema/best)
    # as a flat {name: tensor} mapping, chosen at write time.
    tensors = load_bundle_tensors(bundle_path, device=device)
    # assign=True keeps the bundled tensor's own dtype instead of upcasting to the
    # builder's default float32; read_bundle already proved the key set and shapes.
    lm.load_state_dict(tensors, strict=True, assign=True)
    lm.eval()

    # ---- Codec: pretrained EnCodec via the solver's static loader, no instance ----
    codec_obj = CompressionSolver.wrapped_model_from_checkpoint(
        OmegaConf.create(_codec_config(device)),
        f"//pretrained/{codec_name}",
        device,
    )
    codec_obj.eval()
    _validate_codec(codec_obj, audio, codec_name)

    # ---- MusicGen: constructor then clamp to the real segment length ----
    segment_duration = _segment_duration(bundle)
    constructor_max_duration = max(segment_duration, _AUDIOCRAFT_CONSTRUCTOR_EXTEND_STRIDE + 1.0)
    model = MusicGen(name or codec_name, codec_obj, lm, max_duration=constructor_max_duration)
    model.max_duration = segment_duration

    source: dict[str, object] = {
        "kind": "bundle",
        "bundle_path": str(bundle_path),
        "name": name,
        "model_id": name or codec_name,
        "weights_digest": weights.get("sha256"),
        "weights_state": str(weights.get("selection") or "model"),
        "provenance": provenance,
        "codec": codec_name,
        "segment_duration": segment_duration,
        "segment_duration_seconds": segment_duration,
        "max_duration": segment_duration,
        "sample_rate": _audio_int(audio, "sample_rate", 32_000),
        "channels": _audio_int(audio, "channels", 1),
        "frame_rate": _audio_float(audio, "frame_rate", 50.0),
    }
    # A profile is optional and only carried when the bundle records one. It is
    # never inferred by poking at an original XP's folder.
    if provenance.get("recipe") or provenance.get("profile"):
        source["profile"] = str(provenance.get("recipe") or provenance["profile"])
    return model, source


def bundle_to_source_metadata(bundle: Any, bundle_path: str | Path) -> dict[str, object]:
    """Source metadata for a bundle already read by the caller, without loading the model.

    Shared by callers (e.g. the export path) that need provenance for producing
    output but already hold the ``MusicGen``. Keeps the source dict shape in one
    place so every consumer reports bundles identically.
    """
    wrapped = _BundleDict(bundle)
    weights = wrapped.get("weights") or {}
    audio = wrapped.get("audio") or {}
    provenance = wrapped.get("provenance") or {}
    if isinstance(provenance, dict):
        provenance = dict(provenance)
    else:
        provenance = {}
    name = str(wrapped.get("name") or "")
    codec_spec = wrapped.get("codec") or {}
    codec_name = (
        codec_spec.get("name") if isinstance(codec_spec, dict) else (str(codec_spec) if codec_spec else None)
    ) or "facebook/encodec_32khz"
    path = Path(bundle_path).expanduser().resolve()
    source: dict[str, object] = {
        "kind": "bundle",
        "bundle_path": str(path),
        "name": name,
        "weights_digest": weights.get("sha256"),
        "weights_state": str(weights.get("selection") or "model"),
        "provenance": provenance,
        "codec": codec_name,
        "segment_duration": _segment_duration(wrapped),
        "segment_duration_seconds": _segment_duration(wrapped),
        "max_duration": _segment_duration(wrapped),
        "sample_rate": _audio_int(audio, "sample_rate", 32_000),
        "channels": _audio_int(audio, "channels", 1),
        "frame_rate": _audio_float(audio, "frame_rate", 50.0),
    }
    if provenance.get("recipe") or provenance.get("profile"):
        source["profile"] = str(provenance.get("recipe") or provenance["profile"])
    return source


def _segment_duration(bundle: _BundleDict) -> float:
    raw = bundle.get("segment_duration")
    if raw is None:
        model_config = bundle.get("model_config") or {}
        dataset = model_config.get("dataset") if isinstance(model_config, dict) else None
        raw = (dataset or {}).get("segment_duration") if isinstance(dataset, dict) else None
    value = float(raw) if raw is not None else 10.0
    return value if value > 0 else 10.0


def _audio_int(audio: dict[str, Any], key: str, default: int) -> int:
    raw = audio.get(key) if isinstance(audio, dict) else None
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _audio_float(audio: dict[str, Any], key: str, default: float) -> float:
    raw = audio.get(key) if isinstance(audio, dict) else None
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _validate_codec(codec_obj: Any, audio: dict[str, Any], codec_name: str) -> None:
    """Confirm the loaded codec matches the bundle's declared I/O geometry.

    Raises if the pretrained codec disagrees with the bundle manifest — the
    bundle promises ``audio.sample_rate/channels/frame_rate`` and a model is
    only playable against the codec it was trained with.
    """
    expected_sample = _audio_int(audio, "sample_rate", 32_000)
    expected_channels = _audio_int(audio, "channels", 1)
    expected_frame = _audio_float(audio, "frame_rate", 50.0)

    actual_sample = _codec_attr_int(codec_obj, "sample_rate", expected_sample)
    actual_channels = _codec_attr_int(codec_obj, "channels", expected_channels)
    actual_frame = _codec_attr_float(codec_obj, "frame_rate", expected_frame)

    problems: list[str] = []
    if actual_sample is not None and actual_sample != expected_sample:
        problems.append(f"sample_rate {actual_sample} != {expected_sample}")
    if actual_channels is not None and actual_channels != expected_channels:
        problems.append(f"channels {actual_channels} != {expected_channels}")
    if actual_frame is not None and abs(actual_frame - expected_frame) > 1e-3:
        problems.append(f"frame_rate {actual_frame} != {expected_frame}")
    if problems:
        raise ValueError(f"codec {codec_name} disagrees with the bundle manifest: " + "; ".join(problems))


def _codec_attr_int(codec_obj: Any, name: str, default: int) -> int | None:
    try:
        value = getattr(codec_obj, name)
        if callable(value):
            value = value()
        return int(value) if value is not None else default
    except Exception:  # noqa: BLE001 - a codec that cannot report a property isn't a hard mismatch
        return default


def _codec_attr_float(codec_obj: Any, name: str, default: float) -> float | None:
    try:
        value = getattr(codec_obj, name)
        if callable(value):
            value = value()
        return float(value) if value is not None else default
    except Exception:  # noqa: BLE001
        return default


__all__ = [
    "BundleRuntimeError",
    "NotABundleError",
    "bundle_dir_or_none",
    "bundle_segment_duration",
    "bundle_to_source_metadata",
    "is_bundle_dir",
    "load_bundle_model",
]
