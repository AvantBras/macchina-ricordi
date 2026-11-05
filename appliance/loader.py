"""Open a box and hand back something that plays.

Nothing here imports torch. A ``.nrbox`` opens on ONNX Runtime; only a ``.nrsrc``
source reaches for torch, and only when one is actually opened — so a packaged
build that never sees a source never needs it.
"""

from __future__ import annotations

import os
from pathlib import Path

from .box import BOX_SUFFIX, read_box, read_spec
from .places import app_dirs, models_dir
from .player import Knobs, Player
from .spec import ModelSpec

SOURCE_SUFFIX = ".nrsrc"


def default_roots() -> list[Path]:
    """Where a box is looked for when nobody says: the app, the person's models
    folder, then the working directory."""
    roots: list[Path] = []
    env = os.environ.get("MACCHINA_RICORDI_BOXES")
    if env:
        roots.append(Path(env).expanduser())
    roots += app_dirs()
    roots.append(models_dir())
    roots += [Path.cwd() / "boxes", Path.cwd()]
    return roots


def find_boxes(*roots: Path) -> list[Path]:
    """Boxes directly inside the given folders, plus any files named outright.

    Deliberately not recursive: run from this repository, a recursive scan of the
    working directory walks 249 GB of experiment folders.
    """
    search = [Path(r).expanduser() for r in roots] if roots else default_roots()
    # Keyed by file name: the same box built into the app and sitting in the
    # models folder is one model, and the first place searched wins.
    found: dict[str, Path] = {}
    for root in search:
        if root.is_file() and root.suffix in (BOX_SUFFIX, SOURCE_SUFFIX):
            found.setdefault(root.name, root.resolve())
        elif root.is_dir():
            for path in sorted(root.glob(f"*{BOX_SUFFIX}")):
                found.setdefault(path.name, path.resolve())
    return list(found.values())


def open_backend(path: Path, *, threads: int | None = None, fast: bool = False):
    """``fast`` keeps ONNX Runtime's memory arena: about 30% quicker, about 850 MB more."""
    path = Path(path)
    if path.suffix == SOURCE_SUFFIX:
        try:
            from .lm import load_source_backend
        except ImportError as exc:
            raise ValueError(
                f"{path.name} is a source file and needs torch to play. "
                "Pack it into a .nrbox with `python -m appliance.pack` first."
            ) from exc
        return load_source_backend(path, threads=threads)
    from .runtime_onnx import OnnxBackend

    return OnnxBackend(read_box(path), threads=threads, memory_arena=fast)


def read_any_spec(path: Path) -> ModelSpec:
    path = Path(path)
    if path.suffix == SOURCE_SUFFIX:
        from .format import read_source

        return read_source(path)[0]
    return read_spec(path)


def load_player(
    path: Path,
    *,
    threads: int | None = None,
    fast: bool = False,
    context_seconds: float = 5.0,
    round_seconds: float = 4.0,
    seed: int | None = None,
    **player_options,
) -> tuple[Player, ModelSpec]:
    backend = open_backend(path, threads=threads, fast=fast)
    player = Player(
        backend,
        context_seconds=context_seconds,
        round_seconds=round_seconds,
        seed=seed,
        **player_options,
    )
    return player, backend.spec


def recommended_knobs(spec: ModelSpec, base: Knobs | None = None) -> Knobs:
    """The box's own tuning, where it has any.

    A model that goes silent at temperature 1.0 and stays alive at 1.4 should
    open at 1.4 without anyone having to know that about it.
    """
    knobs = base or Knobs()
    temperature = spec.tuning.get("temperature") if spec.tuning else None
    if temperature:
        knobs = Knobs(**(knobs.__dict__ | {"temperature": float(temperature)}))
    return knobs.clamped()


MODEL_KEYS = ("sig", "dataset", "recipe", "epoch", "parents")


def describe(spec: ModelSpec, path: Path) -> dict:
    return {
        "name": spec.name,
        "path": str(path),
        "file": path.name,
        "origin": spec.origin,
        "note": spec.note,
        "dim": spec.dim,
        "layers": spec.num_layers,
        "sample_rate": spec.sample_rate,
        "frame_rate": spec.frame_rate,
        "segment_duration": spec.segment_duration,
        "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
        "tuning": dict(spec.tuning or {}),
        "model": {key: spec.model.get(key) for key in MODEL_KEYS} if spec.model else None,
    }
