from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_LAYOUT_FLAT = "flat"
OUTPUT_LAYOUT_DATED_RUN_DIR = "dated-run-dir"
OUTPUT_LAYOUT_SESSION = "session"
DEFAULT_PREFIX = "musicgen"
DEFAULT_SESSIONS_DIR = Path("artifacts/sessions")
DEFAULT_RUNS_DIR = Path("artifacts/runs")
DEFAULT_SWEEPS_DIR = Path("artifacts/sweeps")
LEGACY_GENERATED_AUDIO_DIR = Path("artifacts/generated_audio")


@dataclass(frozen=True)
class GenerationRunPaths:
    run_id: str
    prefix: str
    layout: str
    output_dir: Path
    run_dir: Path
    samples_dir: Path
    metadata_path: Path


def utc_compact_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_output_prefix(prefix: str) -> str:
    return prefix.strip().replace(" ", "_") or "musicgen"


def run_id_to_date_bucket(run_id: str) -> str:
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").date().isoformat()
    except ValueError:
        return "unknown-date"


def build_generation_run_paths(
    output_dir: str | Path,
    *,
    prefix: str,
    run_id: str,
    output_layout: str,
    kind: str = "generate",
) -> GenerationRunPaths:
    safe_prefix = safe_output_prefix(prefix)
    root = Path(output_dir).expanduser().resolve()
    if output_layout == OUTPUT_LAYOUT_SESSION:
        # artifacts/sessions/<day>/generate_<run id>[_<prefix>]/{run.json, takes/000.wav}:
        # see macchina_ricordi/sessions/
        label = "" if safe_prefix == DEFAULT_PREFIX else f"_{safe_prefix}"
        run_dir = root / run_id_to_date_bucket(run_id) / f"{kind}_{run_id}{label}"
        return GenerationRunPaths(
            run_id=run_id,
            prefix=safe_prefix,
            layout=output_layout,
            output_dir=root,
            run_dir=run_dir,
            samples_dir=run_dir / "takes",
            metadata_path=run_dir / "run.json",
        )
    if output_layout == OUTPUT_LAYOUT_FLAT:
        return GenerationRunPaths(
            run_id=run_id,
            prefix=safe_prefix,
            layout=output_layout,
            output_dir=root,
            run_dir=root,
            samples_dir=root,
            metadata_path=root / f"{safe_prefix}_{run_id}.json",
        )
    if output_layout != OUTPUT_LAYOUT_DATED_RUN_DIR:
        raise ValueError(f"Unsupported output layout: {output_layout}")
    run_dir = root / run_id_to_date_bucket(run_id) / f"{safe_prefix}_{run_id}"
    return GenerationRunPaths(
        run_id=run_id,
        prefix=safe_prefix,
        layout=output_layout,
        output_dir=root,
        run_dir=run_dir,
        samples_dir=run_dir / "samples",
        metadata_path=run_dir / "run.json",
    )


def sample_audio_path(paths: GenerationRunPaths, index: int) -> Path:
    if paths.layout == OUTPUT_LAYOUT_FLAT:
        return paths.samples_dir / f"{paths.prefix}_{paths.run_id}_{index:03d}.wav"
    return paths.samples_dir / f"{index:03d}.wav"


def generation_kind(mode: str) -> str:
    """The session kind a generation request makes: ``continue`` for a continuation, else ``generate``."""
    return "continue" if mode in ("continuation", "continue") else "generate"


def relative_to_metadata(path: Path, metadata_path: Path) -> str:
    try:
        return path.resolve().relative_to(metadata_path.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_metadata_file_path(value: object, *, metadata_path: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path.resolve()


def iter_generation_metadata_paths(repo_root: str | Path) -> tuple[Path, ...]:
    root = Path(repo_root).expanduser().resolve()
    paths: list[Path] = []
    paths.extend(_session_day_files(root, "run.json"))
    runs_root = root / DEFAULT_RUNS_DIR
    if runs_root.is_dir():
        paths.extend(sorted(runs_root.glob("**/run.json")))
    legacy_root = root / LEGACY_GENERATED_AUDIO_DIR
    if legacy_root.is_dir():
        paths.extend(sorted(legacy_root.glob("*.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def iter_sweep_index_paths(repo_root: str | Path) -> tuple[Path, ...]:
    root = Path(repo_root).expanduser().resolve()
    paths: list[Path] = []
    paths.extend(_session_day_files(root, "index.csv"))
    for sweep_root in (root / DEFAULT_SWEEPS_DIR, root / LEGACY_GENERATED_AUDIO_DIR / "sweeps"):
        if sweep_root.is_dir():
            # Historical sweeps use both ``sweeps/<id>/index.csv`` and
            # ``sweeps/<model>/<id>/index.csv`` layouts.
            paths.extend(sorted(sweep_root.rglob("index.csv")))
    return tuple(path.resolve() for path in paths)


def _session_day_files(root: Path, name: str) -> list[Path]:
    """``artifacts/sessions/<YYYY-MM-DD>/<session>/<name>``; the notebook folders beside the days are skipped."""
    home = root / DEFAULT_SESSIONS_DIR
    if not home.is_dir():
        return []
    return sorted(
        path
        for day in home.iterdir()
        if len(day.name) == 10 and day.name[4] == "-" and day.name[:4].isdigit()
        for path in day.glob(f"*/{name}")
    )
