"""Export a model as something to hand over: a ``.nrbox`` the CPU appliance plays (``docs/appliance.md``).

A box is a model operation, read from the model record. ``model export <model>@best --box`` resolves the reference,
writes the record as a card, and runs the appliance's two steps in their own environments:

1. ``appliance.export`` in ``.venv-musicgen`` (AudioCraft): lifts the weights and the decoder into a ``.nrsrc``,
   with the card as the box's ``spec.json`` name, note and ``model``, and refuses if the checkpoint's architecture is
   not the record's;
2. ``appliance.pack`` in ``.venv-appliance`` (torch, onnx): converts to ONNX, checks it against torch, probes it.

Each run writes ``<name>.export_status.json`` beside the box.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.config import load_musicgen_project_config
from ..musicgen.shared import write_text_atomic
from .bundle import MANIFEST_FILENAME, WEIGHTS_FILENAME, WEIGHTS_STATES
from .record import Checkpoint, Model, ModelNotFound, resolve_model

EXPORT_STATUS_SCHEMA = "macchina_ricordi.model_export.v1"
DEFAULT_BOX_DIR = Path("boxes")
DEFAULT_BUNDLE_DIR = Path("models")
APPLIANCE_ENV = ".venv-appliance"
# File selection and the weight state stored within that file are independent.
WEIGHTS_STATE_SELECTOR = {"model": "last", "best": "last", "ema": "last"}


def _require_weights_state(state: str) -> str:
    if state not in WEIGHTS_STATES:
        raise ValueError(f"weights-state must be one of {', '.join(WEIGHTS_STATES)}, not {state!r}")
    return state


@dataclass(frozen=True)
class BoxExportPlan:
    sig: str
    name: str
    checkpoint: str
    epoch: int | None
    card: dict[str, Any]
    card_path: str
    source_path: str
    box_path: str
    status_path: str
    export_command: tuple[str, ...]
    pack_command: tuple[str, ...] | None
    missing: tuple[str, ...]
    """Environments the commands need that are not there."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def box_card(model: Model, checkpoint: Checkpoint, *, note: str = "") -> dict[str, Any]:
    """The record a box carries: who the model is, what it trained on, and which checkpoint this is."""
    ce = dict(model.valid_ce)
    return {
        "schema": "macchina_ricordi.model_card.v1",
        "sig": model.sig,
        "kind": model.kind,
        "name": model.display_name,
        "title": model.title,
        "label": model.label,
        "note": note or model.title or model.notes or "",
        "recipe": model.recipe,
        "dataset": model.dataset,
        "shape": dict(model.shape),
        "segment_duration": model.segment_duration,
        "parents": list(model.parents),
        "origin": model.origin,
        "checkpoint": checkpoint.path.name,
        "epoch": checkpoint.epoch,
        "valid_ce": ce.get(checkpoint.epoch) if checkpoint.epoch is not None else None,
        "best_epoch": model.best_epoch,
        "measured": dict(model.measured),
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def box_name(model: Model, checkpoint: Checkpoint) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", model.display_name).strip("_") or model.sig
    return f"{base}_ep{checkpoint.epoch}" if checkpoint.epoch is not None else base


def plan_box_export(
    reference: str,
    *,
    out_dir: str | Path = DEFAULT_BOX_DIR,
    name: str | None = None,
    note: str = "",
    pack: bool = True,
    probe: bool = True,
    weights_state: str = "model",
    repo_root: str | Path | None = None,
) -> BoxExportPlan:
    """What ``model export --box`` would run. A reference without ``@`` exports the named snapshot."""
    _require_weights_state(weights_state)
    ref = resolve_model(reference, default_selector=WEIGHTS_STATE_SELECTOR[weights_state], repo_root=repo_root)
    if ref.model is None or ref.checkpoint is None:
        raise ModelNotFound(f"{reference!r} is a loose checkpoint; a box is exported from a model in .dora/xps")
    model = ref.model
    if model.kind == "bundle" and weights_state not in ("model", model.selection):
        raise ModelNotFound("A bundle has one selected snapshot; the requested alternative weight state is unavailable")
    checkpoint = next(c for c in model.checkpoints if c.path == ref.checkpoint)
    config = load_musicgen_project_config()
    root = Path(repo_root or config.repo_root)
    boxes = Path(out_dir)
    boxes = boxes if boxes.is_absolute() else root / boxes
    box = name or box_name(model, checkpoint)
    card_path = boxes / "src" / f"{box}.card.json"
    source_path = boxes / "src" / f"{box}.nrsrc"
    box_path = boxes / f"{box}.nrbox"
    export_command = (
        str(config.musicgen_python),
        "-m",
        "appliance.export",
        str(model.folder) if model.kind == "bundle" else model.sig,
        "--out",
        str(source_path),
        "--checkpoint",
        checkpoint.path.name,
        "--name",
        box,
        "--card",
        str(card_path),
        "--weights-state",
        weights_state,
    )
    appliance_python = root / APPLIANCE_ENV / "bin" / "python"
    pack_command = None
    if pack:
        pack_command = (str(appliance_python), "-m", "appliance.pack", str(source_path), "--out", str(box_path))
        if not probe:
            pack_command += ("--no-probe",)
    missing = []
    if not config.musicgen_python.exists():
        missing.append(f"{config.musicgen_python} (`runtime setup` makes it)")
    if pack and not appliance_python.exists():
        missing.append(f"{appliance_python} (docs/appliance.md, step 2, makes it)")
    return BoxExportPlan(
        sig=model.sig,
        name=box,
        checkpoint=str(checkpoint.path),
        epoch=checkpoint.epoch,
        card=box_card(model, checkpoint, note=note),
        card_path=str(card_path),
        source_path=str(source_path),
        box_path=str(box_path),
        status_path=str(boxes / f"{box}.export_status.json"),
        export_command=export_command,
        pack_command=pack_command,
        missing=tuple(missing),
    )


def run_box_export(
    plan: BoxExportPlan,
    *,
    repo_root: str | Path | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Run the plan's steps in order, recording each in the status sidecar; stops at the first that fails."""
    if plan.missing:
        raise FileNotFoundError("missing: " + "; ".join(plan.missing))
    root = Path(repo_root or load_musicgen_project_config().repo_root)
    card_path = Path(plan.card_path)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(plan.card, indent=2) + "\n", encoding="utf-8")
    env = {"PYTHONPATH": str(root)}
    if plan.card.get("kind") != "bundle":
        env["AUDIOCRAFT_DORA_DIR"] = str(Path(plan.checkpoint).parents[2])
    status: dict[str, Any] = {
        "schema": EXPORT_STATUS_SCHEMA,
        "started_at": _now(),
        "plan": plan.to_dict(),
        "steps": [],
        "status": "running",
    }
    _write_status(plan, status)
    steps = [("export", plan.export_command)] + ([("pack", plan.pack_command)] if plan.pack_command else [])
    for step, command in steps:
        result = run(list(command), cwd=root, env={**os.environ, **env}, check=False)
        status["steps"].append({"step": step, "returncode": result.returncode, "finished_at": _now()})
        if result.returncode != 0:
            status["status"] = "failed"
            status["failed_step"] = step
            break
    else:
        status["status"] = "completed"
    status["finished_at"] = _now()
    _write_status(plan, status)
    return status


def _write_status(plan: BoxExportPlan, status: dict[str, Any]) -> None:
    write_text_atomic(Path(plan.status_path), json.dumps(status, indent=2) + "\n")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── portable model bundles (``model export --bundle``) ──


HELPER_SCRIPT = Path("scripts") / "export_musicgen_bundle.py"


@dataclass(frozen=True)
class BundleExportPlan:
    """What ``model export --bundle`` would run: the export helper lifted by the AudioCraft runtime.

    Planning is read-only; the actual tensor lift (checkpoint -> safetensors) is the bundle worker
    (``macchina_ricordi.models.bundle.export_checkpoint_bundle``) invoked through
    ``scripts/export_musicgen_bundle.py`` inside the AudioCraft runtime, which keeps torch out of this module and
    the root GUI. The worker writes ``config.json`` (the format-core manifest) and ``model.safetensors``.
    """

    name: str
    weights_state: str
    bundle_dir: str
    manifest_path: str
    weights_path: str
    provenance: dict[str, Any]
    checkpoint: str
    helper_command: tuple[str, ...]
    missing: tuple[str, ...]
    force: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bundle_basename(model: Model) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", model.display_name).strip("_") or model.sig
    return base


def _bundle_provenance_payload(model: Model, checkpoint: Checkpoint, weights_state: str) -> dict[str, Any]:
    """Provenance the bundle worker scrubs and records: identity + epoch/state, never paths or logs."""
    prov: dict[str, Any] = {"weights_state": weights_state}
    if model.sig:
        prov["sig"] = model.sig
    if checkpoint.epoch is not None:
        prov["epoch"] = checkpoint.epoch
    if model.recipe:
        prov["recipe"] = model.recipe
    if model.dataset:
        prov["dataset"] = model.dataset
    if model.parents:
        prov["parents"] = list(model.parents)
    return prov


def plan_bundle_export(
    reference: str,
    *,
    out_dir: str | Path = DEFAULT_BUNDLE_DIR,
    name: str | None = None,
    weights_state: str = "model",
    force: bool = False,
    repo_root: str | Path | None = None,
) -> BundleExportPlan:
    """What ``model export --bundle`` would run. A reference with no ``@`` exports the named snapshot.

    The bundle is written to ``<out_dir>/<name>/`` (a folder per bundle), default ``<repo>/models/<name>/``. The
    planner never writes: it validates the reference/snapshot and builds the helper command the worker would run.
    """
    _require_weights_state(weights_state)
    ref = resolve_model(reference, default_selector=WEIGHTS_STATE_SELECTOR[weights_state], repo_root=repo_root)
    if ref.model is None or ref.checkpoint is None:
        raise ModelNotFound(f"{reference!r} is a loose checkpoint; a bundle is exported from a model in .dora/xps")
    model = ref.model
    if model.kind != "xp":
        raise ModelNotFound(f"{reference!r} is itself a bundle with one snapshot; nothing to lift into a new one")
    checkpoint = next(c for c in model.checkpoints if c.path == ref.checkpoint)
    config = load_musicgen_project_config()
    root = Path(repo_root or config.repo_root)
    base = Path(out_dir)
    out = base if base.is_absolute() else root / base
    bundle_name = name or _bundle_basename(model)
    bundle_dir = out / bundle_name
    provenance = _bundle_provenance_payload(model, checkpoint, weights_state)
    config_path = model.folder / ".hydra" / "config.yaml"
    helper_command = (
        str(config.musicgen_python),
        "-m",
        "scripts.export_musicgen_bundle",
        str(checkpoint.path),
        str(config_path),
        str(bundle_dir),
        "--name",
        bundle_name,
        "--weights-state",
        weights_state,
    )
    if force:
        helper_command += ("--force",)
    if checkpoint.epoch is not None:
        helper_command += ("--epoch", str(checkpoint.epoch))
    if model.sig:
        helper_command += ("--sig", model.sig)
    if model.recipe:
        helper_command += ("--recipe", model.recipe)
    if model.parents:
        helper_command += ("--parents", ",".join(model.parents))
    missing = []
    if not config.musicgen_python.exists():
        missing.append(f"{config.musicgen_python} (`runtime setup` makes it)")
    if not (root / HELPER_SCRIPT).exists():
        missing.append(f"{root / HELPER_SCRIPT} is missing")
    return BundleExportPlan(
        name=bundle_name,
        weights_state=weights_state,
        bundle_dir=str(bundle_dir),
        manifest_path=str(bundle_dir / MANIFEST_FILENAME),
        weights_path=str(bundle_dir / WEIGHTS_FILENAME),
        provenance=provenance,
        checkpoint=str(checkpoint.path),
        helper_command=helper_command,
        missing=tuple(missing),
        force=force,
    )


def run_bundle_export(
    plan: BundleExportPlan,
    *,
    repo_root: str | Path | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Run the export helper, then verifies the written bundle (manifest + weights hash). Stops on the first failure."""
    from .bundle import read_bundle

    if plan.missing:
        raise FileNotFoundError("missing: " + "; ".join(plan.missing))
    root = Path(repo_root or load_musicgen_project_config().repo_root)
    bundle_dir = Path(plan.bundle_dir)
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = run(list(plan.helper_command), cwd=root, env=env, check=False)
    if result.returncode != 0:
        return {"status": "failed", "failed_step": "export", "returncode": result.returncode}
    try:
        read_bundle(bundle_dir, verify=True)
    except ValueError as exc:
        return {"status": "failed", "failed_step": "verify", "error": str(exc)}
    return {
        "status": "completed",
        "bundle_dir": plan.bundle_dir,
        "weights_path": plan.weights_path,
        "manifest_path": plan.manifest_path,
    }
