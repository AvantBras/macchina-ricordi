"""Models: what a trained model is, what it is called, and how to point at one.

A model is a Dora XP, ``.dora/xps/<sig>/``. Everything training knows about it is read from the files training
already writes there (``.hydra/config.yaml``, ``macchina_ricordi_train_metadata.json``, ``history.json``, the
checkpoints); nothing is ever written into an XP folder. What training cannot know (a name, a title, notes,
measured behaviour) lives in ``musicgen/models.toml``, a small Git-tracked registry next to the recipes.

A model reference is one string, accepted wherever a command takes a model:

- a registry name (``midwife``), a sig (``4ae63477``) or a unique sig prefix (``4ae6``), a Dora label
  (``slerp_a07_warm_midwife_50ep``), or a path to an XP folder or a checkpoint file;
- optionally followed by a checkpoint selector: ``@best`` (lowest valid ce among checkpoints on disk),
  ``@last``/``@latest`` (newest), ``@final`` (``checkpoint.th``) or ``@<epoch>``.

A portable model bundle (a directory under ``<repo>/models/<name>``, see ``bundle``) also resolves, to a synthetic
``Model`` of ``kind='bundle'`` with a ``bundle-<sha256[:12]>`` id. A bundle carries one selected snapshot, so an
``@selector`` on a bundle is refused.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..musicgen.config import get_repo_root, load_musicgen_project_config
from ..musicgen.shared import read_json, write_text_atomic
from .bundle import ModelBundle, is_bundle, read_bundle
from .provenance import Origin, resolve_init, xp_sig_from_path

REGISTRY_RELATIVE_PATH = Path("musicgen/models.toml")
REGISTRY_SCHEMA = "macchina_ricordi.models.v1"
TRAIN_METADATA_FILENAME = "macchina_ricordi_train_metadata.json"
HISTORY_FILENAME = "history.json"
XP_MARKERS = (".argv.json", HISTORY_FILENAME, TRAIN_METADATA_FILENAME)
SHAPE_KEYS = ("dim", "num_heads", "num_layers", "hidden_scale", "n_q", "card")
SELECTORS = ("best", "last", "latest", "final")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SIG_LIKE_RE = re.compile(r"^[0-9a-f]{8}$")
_SELECTOR_RE = re.compile(r"^(best|last|latest|final|\d+)$")
_EPOCH_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.th$")
_MIN_PREFIX = 4


class ModelNotFound(LookupError):
    pass


class RegistryError(ValueError):
    pass


# ── the registry ──


@dataclass(frozen=True)
class ModelNotes:
    """One ``[models.<sig>]`` entry of ``musicgen/models.toml``: the facts a person adds."""

    sig: str
    name: str | None = None
    title: str | None = None
    notes: str | None = None
    measured: Mapping[str, Any] = field(default_factory=dict)
    study: Mapping[str, Any] = field(default_factory=dict)
    """Architecture as one study described the model (``MODEL_SPECS``); ``epochs`` there is the epoch studied."""


def registry_path(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root or get_repo_root()) / REGISTRY_RELATIVE_PATH


def load_registry(repo_root: str | Path | None = None) -> dict[str, ModelNotes]:
    path = registry_path(repo_root)
    if not path.exists():
        return {}
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    if not isinstance(models, dict):
        raise RegistryError(f"{path}: [models] must be a table of [models.<sig>] entries")
    entries: dict[str, ModelNotes] = {}
    names: dict[str, str] = {}
    for sig, raw in models.items():
        if not isinstance(raw, dict):
            raise RegistryError(f"{path}: [models.{sig}] must be a table")
        entry = ModelNotes(
            sig=str(sig),
            name=_optional_str(raw.get("name")),
            title=_optional_str(raw.get("title")),
            notes=_optional_str(raw.get("notes")),
            measured=dict(raw.get("measured") or {}),
            study=dict(raw.get("study") or {}),
        )
        if entry.name is not None:
            if entry.name in names:
                raise RegistryError(f"{path}: name {entry.name!r} is used by both {names[entry.name]} and {sig}")
            names[entry.name] = entry.sig
        entries[entry.sig] = entry
    return entries


def save_registry(entries: Mapping[str, ModelNotes], repo_root: str | Path | None = None) -> Path:
    """Rewrite the registry, keeping its header comments; refused unless the text parses back to ``entries``."""
    path = registry_path(repo_root)
    header = _header_comments(path.read_text(encoding="utf-8")) if path.exists() else ""
    text = render_registry(entries, header=header)
    if _parse_registry_text(text) != {sig: _entry_dict(entry) for sig, entry in entries.items()}:
        raise RegistryError("refusing to write models.toml: the rendered file does not parse back to the same data")
    write_text_atomic(path, text)
    return path


def render_registry(entries: Mapping[str, ModelNotes], *, header: str = "") -> str:
    lines = [header.rstrip("\n"), ""] if header.strip() else []
    lines += [f'schema = "{REGISTRY_SCHEMA}"']
    for entry in entries.values():
        data = _entry_dict(entry)
        lines += ["", f"[models.{entry.sig}]"]
        for key in ("name", "title", "notes"):
            if data.get(key) is not None:
                lines.append(f"{key} = {_toml_value(data[key])}")
        for key in ("measured", "study"):
            if data.get(key):
                body = ", ".join(f"{k} = {_toml_value(v)}" for k, v in data[key].items())
                lines.append(f"{key} = {{ {body} }}")
    return "\n".join(lines) + "\n"


def set_model_name(sig: str, name: str | None, *, repo_root: str | Path | None = None) -> ModelNotes:
    """Give a model a name (or clear it with ``None``), after checking it cannot be mistaken for anything else."""
    entries = load_registry(repo_root)
    if name is not None:
        name = name.strip()
        if not _NAME_RE.match(name):
            raise RegistryError(
                f"invalid name {name!r}: use lowercase letters, digits, '.', '_' or '-', starting with a letter"
            )
        if _SIG_LIKE_RE.match(name):
            raise RegistryError(f"{name!r} looks like a sig; pick a name that cannot be confused with one")
        for other in entries.values():
            if other.name == name and other.sig != sig:
                raise RegistryError(f"{name!r} already names {other.sig}")
        clash = _labels(dora_xps_dir(repo_root)).get(name)
        if clash is not None and clash != {sig}:
            raise RegistryError(f"{name!r} is already the Dora label of {', '.join(sorted(clash))}")
    current = entries.get(sig, ModelNotes(sig=sig))
    entries[sig] = ModelNotes(
        sig=sig, name=name, title=current.title, notes=current.notes, measured=current.measured, study=current.study
    )
    if not any(_entry_dict(entries[sig]).get(key) for key in ("name", "title", "notes", "measured", "study")):
        del entries[sig]
    save_registry(entries, repo_root)
    return entries.get(sig, ModelNotes(sig=sig))


# ── the model ──


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    epoch: int | None
    """The epoch it holds; for ``checkpoint.th`` that is the number of epochs in ``history.json``."""
    final: bool


@dataclass(frozen=True)
class Model:
    sig: str
    folder: Path
    name: str | None
    title: str | None
    label: str | None
    recipe: str | None
    dataset: str | None
    """The workspace the model trained on, read from its Hydra config (what training actually read)."""
    dataset_path: str | None
    shape: Mapping[str, Any]
    segment_duration: float | None
    codec: str | None
    conditioned: bool
    epochs_planned: int | None
    epochs_done: int
    best_epoch: int | None
    best_valid_ce: float | None
    checkpoints: tuple[Checkpoint, ...]
    parents: tuple[str, ...]
    origin: str | None
    """``warm-start``, ``merge`` or ``perturb`` when the model started from another's weights; else ``None``."""
    notes: str | None
    measured: Mapping[str, Any]
    warnings: tuple[str, ...]
    valid_ce: tuple[tuple[int, float], ...] = field(default=(), repr=False)
    init: Origin | None = field(default=None, repr=False)
    """Where the starting weights came from, as a tree through merges and perturbations (``provenance``)."""
    pool: Mapping[str, Any] | None = None
    """The token pool it trained from (``train run --pool``): path, crops, repeats, epochs; ``None`` for fresh crops."""
    kind: str = "xp"
    """
    ``xp`` for a trained Dora XP, ``bundle`` for a portable model bundle (``models/<name>``). A bundle is not a Dora
    record: it has no Hydra config, no history and no trained checkpoints — one selected snapshot only.
    """
    selection: str | None = None
    """The snapshot the record's single checkpoint is: ``model``, ``ema`` or ``best`` (bundle records only)."""

    @property
    def selection_text(self) -> str:
        return (self.selection or "model") if self.kind == "bundle" else ""

    @property
    def crops(self) -> str:
        """How training drew its crops: ``fresh`` every step, or from a ``pool`` encoded once."""
        if self.kind == "bundle":
            return "bundle"
        return "pool" if self.pool is not None else "fresh"

    @property
    def display_name(self) -> str:
        return self.name or self.label or self.sig

    def checkpoint(self, selector: str = "last") -> Checkpoint:
        """The checkpoint a selector picks: ``best``, ``last``/``latest``, ``final`` or an epoch number."""
        selector = selector.strip().lower()
        if not self.checkpoints:
            raise ModelNotFound(f"{self.sig} has no checkpoint on disk")
        if selector == "final":
            final = [c for c in self.checkpoints if c.final]
            if not final:
                raise ModelNotFound(f"{self.sig} has no checkpoint.th")
            return final[0]
        if selector in ("last", "latest"):
            return max(self.checkpoints, key=lambda c: (c.epoch or 0, not c.final))
        if selector == "best":
            ce = dict(self.valid_ce)
            scored = [c for c in self.checkpoints if c.epoch in ce]
            if not scored:
                raise ModelNotFound(f"{self.sig}: no checkpoint on disk has a valid ce in history.json")
            return min(scored, key=lambda c: (ce[c.epoch], c.final))
        if selector.isdigit():
            epoch = int(selector)
            matches = sorted((c for c in self.checkpoints if c.epoch == epoch), key=lambda c: c.final)
            if not matches:
                available = ", ".join(str(c.epoch) for c in self.checkpoints if c.epoch is not None)
                raise ModelNotFound(f"{self.sig} has no checkpoint for epoch {epoch} (on disk: {available or 'none'})")
            return matches[0]
        raise ModelNotFound(f"unknown checkpoint selector {selector!r}; use best, last, final or an epoch number")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sig": self.sig,
            "name": self.name,
            "title": self.title,
            "label": self.label,
            "folder": str(self.folder),
            "recipe": self.recipe,
            "dataset": self.dataset,
            "dataset_path": self.dataset_path,
            "shape": dict(self.shape),
            "segment_duration": self.segment_duration,
            "codec": self.codec,
            "conditioned": self.conditioned,
            "epochs_planned": self.epochs_planned,
            "epochs_done": self.epochs_done,
            "best_epoch": self.best_epoch,
            "best_valid_ce": self.best_valid_ce,
            "checkpoints": [{"file": c.path.name, "epoch": c.epoch, "final": c.final} for c in self.checkpoints],
            "parents": list(self.parents),
            "origin": self.origin,
            "init": self.init.to_dict() if self.init is not None else None,
            "notes": self.notes,
            "measured": dict(self.measured),
            "warnings": list(self.warnings),
            "crops": self.crops,
            "pool": dict(self.pool) if self.pool is not None else None,
            "kind": self.kind,
            "selection": self.selection,
        }


def dora_xps_dir(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root) / ".dora" / "xps"
    env = os.getenv("AUDIOCRAFT_DORA_DIR")
    if env:
        return Path(env).expanduser().resolve() / "xps"
    return load_musicgen_project_config().dora_dir / "xps"


def dora_dir_or_default(explicit: str | None = None) -> Path:
    """An explicit ``--dora-dir``, else the project's (``AUDIOCRAFT_DORA_DIR`` or the configured one)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    from .checkpoints import resolve_dora_dir

    return resolve_dora_dir()


def xp_folder_for_sig(sig: str, *, dora_dir: Path | None = None) -> Path:
    xp_folder = (dora_dir or dora_dir_or_default()) / "xps" / sig
    if not xp_folder.is_dir():
        raise FileNotFoundError(f"XP folder does not exist: {xp_folder}")
    return xp_folder


def is_xp_folder(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in XP_MARKERS)


def load_model(folder: str | Path, *, registry: Mapping[str, ModelNotes] | None = None) -> Model:
    folder = Path(folder).expanduser().resolve()
    if not is_xp_folder(folder):
        raise ModelNotFound(f"not an XP folder: {folder}")
    sig = folder.name
    notes = (registry if registry is not None else load_registry()).get(sig, ModelNotes(sig=sig))
    metadata = read_json(folder / TRAIN_METADATA_FILENAME)
    metadata = metadata if isinstance(metadata, dict) else {}
    config = _read_hydra_config(folder)
    history = load_history(folder)
    warnings: list[str] = []

    dataset_path = _nested(config, "datasource", "train")
    dataset = _workspace_of(dataset_path)
    recorded = metadata.get("workspace_dir")
    if isinstance(recorded, str) and recorded and dataset and Path(recorded).name != dataset:
        warnings.append(
            f"metadata says it trained on {Path(recorded).name}, its Hydra config says {dataset}; the config wins"
        )
    if dataset is None and isinstance(recorded, str) and recorded:
        dataset = Path(recorded).name

    transformer = config.get("transformer_lm") if isinstance(config.get("transformer_lm"), dict) else {}
    valid_ce = tuple(valid_ce_by_epoch(history).items())
    best = min(valid_ce, key=lambda item: item[1]) if valid_ce else None
    init = resolve_init(metadata, xps_dir=folder.parent)
    if init is None or not init.parent_sigs():
        # A merge or perturbation written straight into an XP-shaped folder (the morph walks) was never trained;
        # its sidecar is the only record of where its weights came from.
        operations = {
            key: payload
            for key in ("merged_from", "perturbed_from")
            if isinstance(payload := read_json(folder / f"{key}.json"), dict)
        }
        if operations:
            init = resolve_init(
                {"init_from_checkpoint": {"user_input": str(folder), "source_resume_metadata": operations}},
                xps_dir=folder.parent,
            )
    parents = init.parent_sigs() if init is not None else ()
    origin = init.operation if init is not None else None
    label = metadata.get("effective_label") or config.get("label") or _label_from_argv(folder)
    return Model(
        sig=sig,
        folder=folder,
        name=notes.name,
        title=notes.title,
        label=str(label) if label else None,
        recipe=_optional_str(metadata.get("recipe") or metadata.get("profile")),
        dataset=dataset,
        dataset_path=str(dataset_path) if dataset_path else None,
        shape={key: transformer.get(key) for key in SHAPE_KEYS if key in transformer},
        segment_duration=_optional_float(_nested(config, "dataset", "segment_duration")),
        codec=_optional_str(config.get("compression_model_checkpoint")),
        conditioned=bool(config.get("conditioners")),
        epochs_planned=_planned_epochs(folder, config, metadata),
        epochs_done=len(history),
        best_epoch=best[0] if best else None,
        best_valid_ce=best[1] if best else None,
        checkpoints=_checkpoints(folder, epochs_done=len(history)),
        parents=parents,
        origin=origin,
        notes=notes.notes,
        measured=dict(notes.measured),
        warnings=tuple(warnings),
        valid_ce=valid_ce,
        init=init,
        pool=_pool_summary(config),
    )


def list_models(repo_root: str | Path | None = None) -> list[Model]:
    xps = dora_xps_dir(repo_root)
    if not xps.is_dir():
        return []
    registry = load_registry(repo_root)
    return [load_model(path, registry=registry) for path in sorted(xps.iterdir()) if is_xp_folder(path)]


# ── model bundles (portable, ``models/<name>``) ──
#
# A stable reference for a portable bundle, derived from its weights' sha256. It never looks like an 8-hex Dora XP
# sig, so it cannot collide with XP names; bundle folder names under models/ may still share a name with an XP.
BUNDLE_ID_PREFIX = "bundle-"
BUNDLES_RELATIVE_DIR = "models"


def bundles_dir(repo_root: str | Path | None) -> Path:
    return Path(repo_root or get_repo_root()) / BUNDLES_RELATIVE_DIR


def list_bundles(repo_root: str | Path | None) -> tuple[Path, ...]:
    """Portable bundle folders under ``<repo>/models/``, in name order; cheap (no manifest off the disk)."""
    base = bundles_dir(repo_root)
    if not base.is_dir():
        return ()
    return tuple(sorted(path for path in base.iterdir() if is_bundle(path)))


def load_bundle_model(path: str | Path, *, verify: bool = False) -> Model:
    """A synthetic ``Model`` record for a portable bundle: one selected snapshot, not a trained Dora XP.

    The sig is a stable ``bundle-<sha256[:12]>`` id (never an 8-hex XP sig, so it cannot collide with XP names).
    ``verify`` hashes the weights file; leave it off when listing many bundles (the GUI does).
    """
    return _bundle_to_model(read_bundle(path, verify=verify))


def _bundle_to_model(bundle: ModelBundle) -> Model:
    # The format-core bundle config stores the LM shape under model_config.transformer_lm; only that is shape data.
    transformer = bundle.model_config.get("transformer_lm") if isinstance(bundle.model_config, Mapping) else {}
    transformer = transformer if isinstance(transformer, Mapping) else {}
    shape = {key: transformer.get(key) for key in SHAPE_KEYS if key in transformer}
    provenance = bundle.provenance
    epoch = _optional_int(provenance.get("epoch"))
    weights_path = bundle.path / bundle.weights_file
    checkpoint = Checkpoint(path=weights_path, epoch=epoch, final=True)
    name = bundle.name
    return Model(
        sig=BUNDLE_ID_PREFIX + (bundle.sha256[:12] if bundle.sha256 else "unknown"),
        folder=bundle.path,
        name=name,
        title=None,
        label=name,
        recipe=_optional_str(provenance.get("recipe")),
        dataset=_optional_str(provenance.get("dataset")),
        dataset_path=None,
        shape=shape,
        segment_duration=bundle.segment_duration,
        codec=bundle.codec,
        conditioned=bool(bundle.model_config.get("conditioners")),
        epochs_planned=None,
        epochs_done=1,
        best_epoch=epoch,
        best_valid_ce=None,
        checkpoints=(checkpoint,),
        parents=tuple(str(p) for p in provenance.get("parents")) if provenance.get("parents") else (),
        origin="bundle",
        notes=None,
        measured={},
        warnings=(),
        valid_ce=(),
        init=None,
        pool=None,
        kind="bundle",
        selection=bundle.selection,
    )


def _bundle_ref(raw: str, bundle: ModelBundle) -> ModelRef:
    """A reference to a bundle: its synthetic model plus the single selected snapshot as the checkpoint."""
    model = _bundle_to_model(bundle)
    snapshot = model.checkpoints[0]
    return ModelRef(raw, model, snapshot.path, snapshot.epoch)


def list_bundle_models(repo_root: str | Path | None = None) -> list[Model]:
    """Every portable bundle under ``<repo>/models/`` as a synthetic ``Model`` (fast: no weights hashing)."""
    root = repo_root or get_repo_root()
    return [load_bundle_model(folder, verify=False) for folder in list_bundles(root)]


# ── references ──


@dataclass(frozen=True)
class ModelRef:
    text: str
    model: Model | None
    """``None`` for a checkpoint file outside ``.dora/xps`` (a merge or perturbation output)."""
    checkpoint: Path | None
    """Set when the reference names a checkpoint: a selector, a default selector, or a checkpoint path."""
    epoch: int | None

    @property
    def checkpoint_selector(self) -> str | None:
        """The checkpoint as the older ``--checkpoint`` flags spell it: ``final`` or an epoch number."""
        if self.checkpoint is None:
            return None
        if self.checkpoint.name == "checkpoint.th":
            return "final"
        match = _EPOCH_CHECKPOINT_RE.match(self.checkpoint.name)
        return match.group(1) if match else None


def split_selector(text: str) -> tuple[str, str | None]:
    base, sep, selector = text.rpartition("@")
    if sep and base and _SELECTOR_RE.match(selector.lower()):
        return base, selector.lower()
    return text, None


def resolve_model(
    text: str,
    *,
    default_selector: str | None = None,
    repo_root: str | Path | None = None,
) -> ModelRef:
    """Resolve a model reference (see the module docstring). Raises ``ModelNotFound`` with suggestions.

    A portable bundle (a directory under ``<repo>/models/``) resolves to a synthetic ``Model`` of ``kind='bundle'``.
    A bundle holds one selected snapshot, so an ``@selector`` on a bundle is refused.
    """
    raw = str(text).strip()
    if not raw:
        raise ModelNotFound("empty model reference")
    base, typed_selector = split_selector(raw)
    selector = typed_selector or default_selector
    registry = load_registry(repo_root)
    root = Path(repo_root or get_repo_root())

    path = Path(base).expanduser()
    candidate = path if path.is_absolute() else root / path if (root / path).exists() else path
    if candidate.is_file():
        sig = xp_sig_from_path(candidate)
        resolved = candidate.resolve()
        if typed_selector is not None:
            raise ModelNotFound(f"{raw!r}: a checkpoint path already names a checkpoint; drop the @{typed_selector}")
        if sig is not None and is_xp_folder(resolved.parent):
            model = load_model(resolved.parent, registry=registry)
            epoch = next((c.epoch for c in model.checkpoints if c.path == resolved), None)
            return ModelRef(raw, model, resolved, epoch)
        return ModelRef(raw, None, resolved, None)
    if candidate.is_dir():
        if is_bundle(candidate):
            if typed_selector is not None:
                raise ModelNotFound(f"{raw!r}: a bundle has one selected snapshot; drop the @{typed_selector}")
            return _bundle_ref(raw, read_bundle(candidate))
        if not is_xp_folder(candidate):
            raise ModelNotFound(f"{base!r} is a folder but not an XP or a model bundle")
        return _with_selector(raw, load_model(candidate, registry=registry), selector)

    bundle = _lookup_bundle(base, root)
    if bundle is not None:
        if typed_selector is not None:
            raise ModelNotFound(f"{raw!r}: a bundle has one selected snapshot; drop the @{typed_selector}")
        return _bundle_ref(raw, bundle)

    xps = dora_xps_dir(repo_root)
    sig = _lookup_sig(base, xps, registry)
    return _with_selector(raw, load_model(xps / sig, registry=registry), selector)


def resolve_checkpoint_path(
    text: str,
    *,
    default_selector: str = "last",
    repo_root: str | Path | None = None,
) -> str:
    """For flags that take a checkpoint file: an existing file passes through, a model reference is resolved.

    A named bundle reference resolves to the bundle *directory* (a bundle's loading handle is its folder, not the
    safetensors file inside it), so ``train --from NAME`` and ``model perturb NAME`` read the bundle as a whole.
    A raw train XP reference resolves to its checkpoint file.
    """
    if Path(text).expanduser().is_file():
        return text
    ref = resolve_model(text, default_selector=default_selector, repo_root=repo_root)
    if ref.model is not None and ref.model.kind == "bundle":
        return str(ref.model.folder)
    assert ref.checkpoint is not None
    return str(ref.checkpoint)


def resolve_sig(text: str, *, repo_root: str | Path | None = None) -> str:
    """For commands that act on a whole model (resume, lineage, watch): the sig a reference points at."""
    base, selector = split_selector(str(text).strip())
    if selector is not None:
        raise ModelNotFound(f"{text!r}: this command acts on a whole model; drop the @{selector}")
    ref = resolve_model(base, repo_root=repo_root)
    if ref.model is None:
        raise ModelNotFound(f"{text!r} is a checkpoint file, not a model")
    return ref.model.sig


# ── shared readers (the GUI uses these too) ──


def load_history(xp_folder: Path) -> list[dict[str, Any]]:
    payload = read_json(Path(xp_folder) / HISTORY_FILENAME)
    return [entry for entry in payload if isinstance(entry, dict)] if isinstance(payload, list) else []


def valid_ce_by_epoch(history: Iterable[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for epoch, entry in enumerate(history, start=1):
        valid = entry.get("valid")
        ce = valid.get("ce") if isinstance(valid, dict) else None
        if isinstance(ce, int | float) and math.isfinite(ce):
            out[epoch] = float(ce)
    return out


def model_display_names(repo_root: str | Path | None = None) -> dict[str, str]:
    """Each sig's name, or else its Dora label, without loading every model."""
    names: dict[str, str] = {}
    for label, sigs in _labels(dora_xps_dir(repo_root)).items():
        for sig in sigs:
            names[sig] = label
    for sig, entry in load_registry(repo_root).items():
        if entry.name:
            names[sig] = entry.name
    return names


# ── internals ──


def _with_selector(raw: str, model: Model, selector: str | None) -> ModelRef:
    if selector is None:
        return ModelRef(raw, model, None, None)
    checkpoint = model.checkpoint(selector)
    return ModelRef(raw, model, checkpoint.path, checkpoint.epoch)


def _lookup_bundle(text: str, root: Path) -> ModelBundle | None:
    """A bundle by its folder name under ``<repo>/models/``, or by a unique ``bundle-<hex>`` id prefix."""
    base_dir = bundles_dir(root)
    named = base_dir / text
    if is_bundle(named):
        return read_bundle(named, verify=False)
    if base_dir.is_dir():
        matches = [folder for folder in base_dir.iterdir() if is_bundle(folder) and _bundle_matches(folder, text)]
        if len(matches) == 1:
            return read_bundle(matches[0], verify=False)
        if len(matches) > 1:
            raise ModelNotFound(f"'{text}' matches several bundles: {', '.join(sorted(Path(m).name for m in matches))}")
        if matches:
            raise ModelNotFound(f"'{text}' matches no identifier of the bundle folders on disk")
    return None


def _bundle_matches(folder: Path, text: str) -> bool:
    low = text.lower()
    if low == folder.name.lower():
        return True
    if not low.startswith(BUNDLE_ID_PREFIX):
        return False
    bundle = read_bundle(folder, verify=False)
    bundle_id = BUNDLE_ID_PREFIX + (bundle.sha256[:12] if bundle.sha256 else "unknown")
    return bundle_id.startswith(low)


def _lookup_sig(text: str, xps: Path, registry: Mapping[str, ModelNotes]) -> str:
    named = {entry.name: sig for sig, entry in registry.items() if entry.name}
    if text in named:
        return named[text]
    lowered = text.lower()
    if lowered in named:
        return named[lowered]
    if (xps / text).is_dir() and is_xp_folder(xps / text):
        return text
    labels = _labels(xps)
    label = text if text in labels else _sanitized_or_none(text)
    if label in labels:
        sigs = labels[label]
        if len(sigs) > 1:
            raise ModelNotFound(f"label {label!r} is shared by {', '.join(sorted(sigs))}; use a sig or give one a name")
        return next(iter(sigs))
    if xps.is_dir() and len(text) >= _MIN_PREFIX and re.fullmatch(r"[0-9a-f]+", lowered):
        matches = [path.name for path in xps.iterdir() if path.name.startswith(lowered) and is_xp_folder(path)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ModelNotFound(f"{text!r} matches several sigs: {', '.join(sorted(matches))}")
    candidates = [*named, *labels, *(path.name for path in (xps.iterdir() if xps.is_dir() else ()))]
    close = difflib.get_close_matches(text, candidates, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(close)}?" if close else " `model list` shows every model."
    raise ModelNotFound(f"no model called {text!r}.{hint}")


def sanitize_label(raw_label: str) -> str:
    """A run label as Dora stores it: ``--run-label nz-deep`` trains a model labelled ``nz_deep``."""
    label = re.sub(r"[^A-Za-z0-9]+", "_", raw_label.strip()).strip("_").lower()
    if not label:
        raise ValueError("Training run label must contain at least one letter or digit.")
    return label[:96]


def _sanitized_or_none(text: str) -> str | None:
    try:
        return sanitize_label(text)
    except ValueError:
        return None


def _labels(xps: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    if not xps.is_dir():
        return labels
    for folder in xps.iterdir():
        if not is_xp_folder(folder):
            continue
        metadata = read_json(folder / TRAIN_METADATA_FILENAME)
        label = metadata.get("effective_label") if isinstance(metadata, dict) else None
        label = label or _label_from_argv(folder)
        if label:
            labels.setdefault(str(label), set()).add(folder.name)
    return labels


def _label_from_argv(folder: Path) -> str | None:
    argv = read_json(folder / ".argv.json")
    if isinstance(argv, list):
        for item in reversed(argv):
            if isinstance(item, str) and item.startswith("label="):
                return item.split("=", 1)[1]
    return None


def _checkpoints(folder: Path, *, epochs_done: int) -> tuple[Checkpoint, ...]:
    found: list[Checkpoint] = []
    for path in folder.glob("checkpoint*.th"):
        match = _EPOCH_CHECKPOINT_RE.match(path.name)
        if match:
            found.append(Checkpoint(path.resolve(), int(match.group(1)), False))
        elif path.name == "checkpoint.th":
            found.append(Checkpoint(path.resolve(), epochs_done or None, True))
    return tuple(sorted(found, key=lambda c: (c.epoch or 0, c.final)))


def _planned_epochs(folder: Path, config: Mapping[str, Any], metadata: Mapping[str, Any]) -> int | None:
    # `train resume` raises the target in-process; .hydra keeps the original one.
    planned = _optional_int(_nested(config, "optim", "epochs"))
    if planned is None:
        planned = _epochs_from_command(metadata.get("command"))
    resume = read_json(folder / "resume_status.json")
    target = _optional_int(resume.get("target_epochs")) if isinstance(resume, dict) else None
    if target is not None and (planned is None or target > planned):
        return target
    return planned


def _epochs_from_command(command: object) -> int | None:
    epochs = None
    for part in command if isinstance(command, list) else ():
        if isinstance(part, str) and part.startswith("optim.epochs="):
            value = part.split("=", 1)[1]
            epochs = int(value) if value.isdigit() else epochs
    return epochs


def _workspace_of(train_path: object) -> str | None:
    if not isinstance(train_path, str) or not train_path:
        return None
    parts = Path(train_path).parts
    if len(parts) >= 4 and parts[-3] == "egs":
        return parts[-4]
    if len(parts) >= 4 and parts[-4] == "weights":
        # artifacts/weights/<dataset>/<overlay>/train: trained on a weights overlay of <dataset>
        return parts[-3]
    return Path(train_path).parent.name or None


def _pool_summary(config: Mapping[str, Any]) -> dict[str, Any] | None:
    from ..training.pool import pool_of_config, read_pool_status

    pool = pool_of_config(config)
    if pool is None:
        return None
    status = read_pool_status(pool) or {}
    return {
        "path": str(pool),
        "crops": status.get("crops"),
        "repeats": status.get("repeats_per_crop"),
        "epochs": status.get("epochs"),
    }


def _read_hydra_config(folder: Path) -> dict[str, Any]:
    path = folder / ".hydra" / "config.yaml"
    if not path.exists():
        return {}
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _optional_str(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _entry_dict(entry: ModelNotes) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key in ("name", "title", "notes"):
        if getattr(entry, key) is not None:
            data[key] = getattr(entry, key)
    for key in ("measured", "study"):
        if getattr(entry, key):
            data[key] = dict(getattr(entry, key))
    return data


def _parse_registry_text(text: str) -> dict[str, dict[str, Any]]:
    payload = tomllib.loads(text)
    return {str(sig): dict(raw) for sig, raw in payload.get("models", {}).items()}


def _header_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
        else:
            break
    return "\n".join(lines).strip("\n") + "\n" if any(line.startswith("#") for line in lines) else ""


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise RegistryError(f"cannot write {value!r} to models.toml")
