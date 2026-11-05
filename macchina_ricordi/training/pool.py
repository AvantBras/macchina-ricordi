"""Token pools: training crops encoded once and served to training as cached batches.

`train run --pool` builds one with ``scripts/build_token_pool.py`` (or reuses it) and trains with
``cache.path=<pool>``; without ``--pool`` training draws and encodes fresh crops as always.
Pool sizing uses the recipe family and the requested training schedule.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POOLS_DIR = Path("artifacts") / "token_pools"
NANO_REPEATS = 8
LARGER_REPEATS = 4  # bigger models memorize more of what comes back (Carlini et al. 2022)
INFO_BYTES = 550  # per crop in infos.pt (527 measured); its codes add 400 bytes per second of segment
ENCODE_AUDIO_SECONDS_PER_SECOND = 290  # fp32, 4 crops at a time, RTX 3060 at its power cap (2026-10-24)
DISK_MARGIN_BYTES = 10 * 10**9


def max_repeats(recipe_parent: str | None) -> int:
    return NANO_REPEATS if recipe_parent in ("nano-family", "pico-family") else LARGER_REPEATS


def pool_size(dataset_seconds: float, *, epochs: int, batch_size: int, updates_per_epoch: int, repeats: int) -> int:
    """One crop per second of training audio, or few enough repeats, whichever needs more; rounded up to 1,000."""
    draws = epochs * updates_per_epoch * batch_size
    return math.ceil(max(dataset_seconds, draws / repeats) / 1000) * 1000


@dataclass(frozen=True)
class PoolSpec:
    """What a pool must match to serve a run. The batch files are laid out per epoch, update and batch slot."""

    dataset: str
    fingerprint: str
    train_dir: str
    segment_duration: float
    codec: str
    batch_size: int
    updates_per_epoch: int
    epochs: int
    crops: int
    seed: int = 0

    @property
    def name(self) -> str:
        seed = f"-seed{self.seed}" if self.seed else ""
        return (
            f"{self.dataset.lstrip('.')}-{self.fingerprint[:8]}-{self.segment_duration:g}s-b{self.batch_size}"
            f"-{self.crops // 1000}k-{self.epochs}ep{seed}"
        )

    @property
    def repeats(self) -> float:
        return self.epochs * self.updates_per_epoch * self.batch_size / self.crops

    def pool_bytes(self) -> int:
        """tokens.npy + infos.pt + order.npy."""
        frames = int(self.segment_duration * 50)
        return self.crops * (frames * 4 * 2 + INFO_BYTES) + self.epochs * self.updates_per_epoch * self.batch_size * 4


def read_pool_status(pool_dir: str | Path) -> dict[str, Any] | None:
    path = Path(pool_dir) / "pool_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def pool_problems(status: Mapping[str, Any] | None, spec: PoolSpec) -> list[str]:
    """Why a built pool cannot serve this run; empty when it can."""
    if status is None:
        return ["it has no readable pool_status.json (an interrupted build?)"]
    checks = (
        ("dataset_fingerprint", spec.fingerprint, "dataset fingerprint"),
        ("dataset", spec.train_dir, "train manifest folder (dataset or weights overlay)"),
        ("segment_duration", spec.segment_duration, "segment duration"),
        ("codec", spec.codec, "codec"),
        ("batch_size", spec.batch_size, "batch size"),
        ("updates_per_epoch", spec.updates_per_epoch, "updates per epoch"),
        ("crop_seed", spec.seed, "crop seed"),
    )
    problems = [
        f"{label}: pool has {status.get(key)!r}, the run needs {want!r}"
        for key, want, label in checks
        if status.get(key, 0 if key == "crop_seed" else None) != want
    ]
    if int(status.get("epochs") or 0) < spec.epochs:
        problems.append(f"it holds batches for {status.get('epochs')} epochs, the run needs {spec.epochs}")
    return problems


def _draws_same_crops(status: Mapping[str, Any] | None, spec: PoolSpec) -> bool:
    """A pool with the same dataset, segment and crop seed drew the same crops, whatever its size or epochs."""
    return status is not None and all(
        status.get(key, 0 if key == "crop_seed" else None) == want
        for key, want in (("dataset_fingerprint", spec.fingerprint), ("dataset", spec.train_dir),
                          ("segment_duration", spec.segment_duration), ("crop_seed", spec.seed))
    )  # fmt: skip


def _last(overrides: Sequence[str], key: str) -> str | None:
    values = [item.split("=", 1)[1] for item in overrides if item.startswith(f"{key}=")]
    return values[-1] if values else None


def spec_for_plan(plan: Any, *, seed: int, repeats: int) -> PoolSpec:
    """The pool a planned run needs, read from its Dora overrides and its rendered dataset config."""
    import yaml

    from ..data.clip_stats import read_manifest_durations

    overrides = [str(item) for item in plan.command[2:]]  # after `dora run`
    config = yaml.safe_load(Path(plan.dataset_config_path).read_text(encoding="utf-8")) or {}
    train_dir = str(config["datasource"]["train"])
    epochs = int(float(_last(overrides, "optim.epochs") or 0))
    batch_size = int(_last(overrides, "dataset.batch_size") or 0)
    if not epochs or not batch_size:
        raise RuntimeError("--pool needs optim.epochs and dataset.batch_size in the planned command")
    updates = int(_last(overrides, "optim.updates_per_epoch") or 2000)  # MusicGen's default
    seconds = sum(read_manifest_durations(Path(train_dir) / "data.jsonl.gz") or [])
    if not seconds:
        raise RuntimeError(f"--pool cannot read clip durations from {train_dir}/data.jsonl.gz")
    return PoolSpec(
        dataset=Path(plan.workspace_dir).name,
        fingerprint=str(plan.dataset_fingerprint_digest),
        train_dir=train_dir,
        segment_duration=float(_last(overrides, "dataset.segment_duration") or 0),
        codec=str(_last(overrides, "compression_model_checkpoint")),
        batch_size=batch_size,
        updates_per_epoch=updates,
        epochs=epochs,
        crops=pool_size(seconds, epochs=epochs, batch_size=batch_size, updates_per_epoch=updates, repeats=repeats),
        seed=seed,
    )


def build_pool(
    spec: PoolSpec,
    pool_dir: Path,
    *,
    overrides: Sequence[str],
    musicgen_python: str,
    audiocraft_dir: Path,
    repo_root: Path,
) -> int:
    """Run ``scripts/build_token_pool.py`` in the AudioCraft venv; it writes ``<pool_dir>`` only when complete."""
    import subprocess

    from ..musicgen.config import musicgen_script_env

    dora_overrides = [item for item in overrides if not item.startswith(("label=", "cache."))]
    command = [
        musicgen_python, str(repo_root / "scripts" / "build_token_pool.py"), "dora", str(pool_dir),
        "--crops", str(spec.crops), "--epochs", str(spec.epochs), "--crop-seed", str(spec.seed),
        "--fingerprint", spec.fingerprint, "--", *dora_overrides,
    ]  # fmt: skip
    return subprocess.run(command, cwd=audiocraft_dir, env=musicgen_script_env(), check=False).returncode


def resume_problem(xp_folder: Path, epochs: int) -> str | None:
    """Why resuming this XP to ``epochs`` would fail: its pool has no batches past the epochs it was built for."""
    pool = pool_of_xp(xp_folder)
    if pool is None:
        return None
    built = int((read_pool_status(pool) or {}).get("epochs") or 0)
    if epochs <= built:
        return None
    return (
        f"{xp_folder.name} trains from the token pool {pool}, which holds batches for {built} epochs, not {epochs}. "
        f"Start a new model from it instead: `train run --from {xp_folder.name}@last --pool --pool-seed 1` "
        "(new crops), or without --pool (fresh crops)."
    )


def pool_of_xp(folder: Path) -> Path | None:
    """The pool an XP folder trained from, read from its ``.hydra/config.yaml``; ``None`` for fresh crops."""
    import yaml

    try:
        config = yaml.safe_load((folder / ".hydra" / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return pool_of_config(config) if isinstance(config, Mapping) else None


def pool_of_config(config: Mapping[str, Any]) -> Path | None:
    """The pool a Hydra config trains from: the parent of its ``cache.path``, unless it writes a cache instead."""
    cache = config.get("cache")
    if not isinstance(cache, Mapping) or not cache.get("path") or cache.get("write"):
        return None
    path = Path(str(cache["path"]))
    return path if (path / "pool_status.json").exists() else path.parent  # before 2026-10-25: <pool>/cache


def cache_override(pool_dir: Path) -> str:
    """The ``cache.path`` override that trains from a pool; older pools keep their batches in ``cache/``."""
    return f"cache.path={pool_dir / 'cache' if (pool_dir / 'cache').is_dir() else pool_dir}"


@dataclass(frozen=True)
class PoolPlan:
    """The pool a `--pool` run trains from, and why it cannot (``problem``, ``None`` when it can)."""

    spec: PoolSpec
    path: Path
    built: bool
    build_hours: float
    problem: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.path.name,
            "crops": self.spec.crops,
            "repeats": round(self.spec.repeats, 2),
            "epochs": self.spec.epochs,
            "built": self.built,
            "build_hours": round(self.build_hours, 2),
            "bytes": self.spec.pool_bytes(),
            "problem": self.problem,
        }


def plan_pool(recipe: str, base_plan: Any, *, seed: int, init_from_checkpoint: str | None, repo_root: Path) -> PoolPlan:
    import shutil

    from ..musicgen.config import get_recipe

    spec = spec_for_plan(base_plan, seed=seed, repeats=max_repeats(get_recipe(recipe).extends))
    pool_dir = repo_root / POOLS_DIR / spec.name
    hours = spec.crops * spec.segment_duration / ENCODE_AUDIO_SECONDS_PER_SECOND / 3600

    def plan(problem: str | None) -> PoolPlan:
        return PoolPlan(spec, pool_dir, pool_dir.exists(), hours, problem)

    init = init_from_checkpoint
    init_pool = pool_of_xp(Path(init).parent) if init else None
    if init_pool is not None and _draws_same_crops(read_pool_status(init_pool), spec):
        return plan(f"the warm start {init} trained on a pool of the same crops, so they would come back again; "
                    "use another pool seed (--pool-seed N) to draw new ones")  # fmt: skip
    if pool_dir.exists():
        problems = pool_problems(read_pool_status(pool_dir), spec)
        return plan(f"{pool_dir.name} cannot serve this run: " + "; ".join(problems) if problems else None)
    need = spec.pool_bytes()
    pool_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(pool_dir.parent).free
    if need + DISK_MARGIN_BYTES > free:
        return plan(f"not enough disk for the pool: it needs {need / 1e9:.1f} GB plus a "
                    f"{DISK_MARGIN_BYTES / 1e9:.0f} GB margin, and {free / 1e9:.1f} GB is free")  # fmt: skip
    return plan(None)
