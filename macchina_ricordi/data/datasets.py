"""Datasets: named sets of clips, split into train and valid, that a recipe trains on.

A new dataset lives in ``datasets/<name>/`` and is made in one step by ``build_dataset`` (``data build``):
the clips gathered from its sources (``clips/`` and ``corpus.json``, hard-linked or sliced), the split as hard
links in ``data/train`` and ``data/valid``, AudioCraft's manifests in ``egs/``, and ``dataset.json`` describing
the whole build. Nothing rewrites a dataset afterwards: a different cut is a new dataset under a new name.

Workspaces prepared before datasets existed (``.workspace_*``, ``.macchina_ricordi_*`` and a few older names) are frozen
and read through the same ``open_dataset``; their name is their folder name, leading dot included. Trainings record
themselves in their job's folder (``train_jobs``), never in a dataset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..musicgen.config import get_repo_root
from ..musicgen.shared import read_json

DATASETS_DIRNAME = "datasets"
DATASET_FILENAME = "dataset.json"
BUILD_STATUS_FILENAME = "build_status.json"
DATASET_SCHEMA = "macchina_ricordi.dataset.v1"
# where workspaces were made before datasets; older ones are recognised by their egs/ folder
LEGACY_PREFIXES = (".workspace", ".macchina_ricordi")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    legacy: bool
    """A workspace prepared before datasets existed: frozen, read from its prepare records."""
    egs_name: str | None
    created_at: str | None
    sources: tuple[str, ...]
    slicing: Mapping[str, Any] = field(default_factory=dict)
    valid_fraction: float | None = None
    seed: int | None = None
    train_clips: int | None = None
    valid_clips: int | None = None
    train_seconds: float | None = None
    slice_length_s: float | None = None
    speed_ratios: tuple[float, ...] = ()
    fingerprint_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "legacy": self.legacy,
            "egs_name": self.egs_name,
            "created_at": self.created_at,
            "sources": list(self.sources),
            "slicing": dict(self.slicing),
            "valid_fraction": self.valid_fraction,
            "seed": self.seed,
            "train_clips": self.train_clips,
            "valid_clips": self.valid_clips,
            "train_hours": round(self.train_seconds / 3600, 3) if self.train_seconds is not None else None,
            "slice_length_s": self.slice_length_s,
            "speed_ratios": list(self.speed_ratios),
            "fingerprint_digest": self.fingerprint_digest,
        }


def datasets_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root or get_repo_root()) / DATASETS_DIRNAME


def dataset_dirs(repo_root: str | Path | None = None) -> list[Path]:
    """Every dataset folder: ``datasets/*`` first, then the legacy dot-folders that hold prepared manifests."""
    root = Path(repo_root or get_repo_root())
    found = (
        sorted(path for path in datasets_root(root).glob("*") if path.is_dir()) if datasets_root(root).is_dir() else []
    )
    found += sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(".")
        and path.is_dir()
        and (path.name.startswith(LEGACY_PREFIXES) or (path / "egs").is_dir())
    )
    return found


def dataset_name(path: Path) -> str:
    return path.name


def find_dataset_dir(name_or_path: str | Path, *, repo_root: str | Path | None = None) -> Path | None:
    """The folder a dataset name (``bear_bones``, ``.workspace_x``) or path refers to, if it exists."""
    text = str(name_or_path).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    root = Path(repo_root or get_repo_root())
    for path in (datasets_root(root) / text, root / text):
        if path.is_dir():
            return path.resolve()
    return None


def open_dataset(name_or_path: str | Path, *, repo_root: str | Path | None = None) -> Dataset:
    path = find_dataset_dir(name_or_path, repo_root=repo_root)
    if path is None:
        raise DatasetError(f"no dataset called {str(name_or_path)!r}; `data list` shows them")
    record = read_json(path / DATASET_FILENAME)
    if isinstance(record, dict) and record.get("schema") == DATASET_SCHEMA:
        return _from_record(path, record)
    return _from_legacy(path)


def list_datasets(repo_root: str | Path | None = None) -> list[Dataset]:
    datasets = []
    for path in dataset_dirs(repo_root):
        try:
            datasets.append(open_dataset(path, repo_root=repo_root))
        except (DatasetError, OSError, ValueError):
            continue
    return datasets


def build_dataset(
    name: str,
    sources: Sequence[str | Path],
    *,
    slice_seconds: float | None = None,
    hop_seconds: float | None = None,
    min_clip_seconds: float | None = None,
    sample_rate: int = 32_000,
    channels: int = 1,
    exclude: Sequence[str | Path] = (),
    skip_duplicates: bool = False,
    valid_fraction: float | None = None,
    seed: int | None = None,
    max_silence_fraction: float | None = None,
    speed_ratios: str | Sequence[float] | None = None,
    speed_mode: str | None = None,
    musicgen_python: str | None = None,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    repo_root: str | Path | None = None,
) -> Dataset:
    """Gather clips from ``sources``, split them and write the manifests, into a new ``datasets/<name>/``."""
    from ..augmentation import DEFAULT_SPEED_MODE
    from ..musicgen.shared import TaskSidecar, utc_now_iso
    from .corpus import CorpusError, CorpusSpec, build_corpus, plan_corpus, read_corpus_manifest
    from .prepare import (
        DEFAULT_MAX_SILENCE_FRACTION,
        prepare_musicgen_dataset,
    )

    folder = plan_dataset_folder(name, repo_root=repo_root)
    spec = CorpusSpec(
        name=name,
        sources=tuple(Path(source).expanduser().resolve() for source in sources),
        slice_seconds=slice_seconds,
        hop_seconds=hop_seconds,
        min_clip_seconds=min_clip_seconds,
        sample_rate=sample_rate,
        channels=channels,
        exclude=frozenset(Path(path).expanduser().resolve() for path in exclude),
        skip_duplicates=skip_duplicates,
    )
    plan = plan_corpus(spec, folder)
    if not plan.clips:
        raise CorpusError("no clip would be written")

    folder.mkdir(parents=True)
    with TaskSidecar(folder / BUILD_STATUS_FILENAME, name=name) as status:
        corpus = build_corpus(plan, workers=workers, progress=progress)
        if not corpus.written:
            raise CorpusError("no clip could be written; see build_status.json")
        status.update(clips_written=corpus.written, clips_failed=len(corpus.failed))
        status.write()
        prepared = prepare_musicgen_dataset(
            input_dir=folder / "clips",
            workspace_dir=folder,
            musicgen_python=musicgen_python,
            valid_fraction=valid_fraction,  # None → prepare honours the checkout's configured default at call time
            seed=seed,  # None → prepare honours the checkout's configured default at call time
            max_silence_fraction=DEFAULT_MAX_SILENCE_FRACTION if max_silence_fraction is None else max_silence_fraction,
            speed_ratios=speed_ratios,
            speed_mode=speed_mode or DEFAULT_SPEED_MODE,
            link=True,
        )
        prepare_log = read_json(folder / "logs" / "prepare_dataset.json") or {}
        corpus_manifest = read_corpus_manifest(folder) or {}
        record = {
            "schema": DATASET_SCHEMA,
            "name": name,
            "created_at": utc_now_iso(),
            "sources": [str(source) for source in spec.sources],
            "slicing": corpus_manifest.get("slicing") or {},
            "clips": {"written": corpus.written, "failed": len(corpus.failed)},
            "split": {
                "valid_fraction": prepare_log.get("valid_fraction"),
                "seed": prepare_log.get("seed"),
                "train": prepared.train_count,
                "valid": prepared.valid_count,
                "dropped_silent": (prepare_log.get("train_dropped_silent") or 0)
                + (prepare_log.get("valid_dropped_silent") or 0),
            },
            "speed": {"ratios": prepare_log.get("speed_ratios") or [], "mode": prepare_log.get("speed_mode")},
            "egs_name": prepared.train_manifest.parent.parent.name,
            "train_seconds": _manifest_seconds(prepared.train_manifest),
            "clip_duration_stats": prepare_log.get("clip_duration_stats"),
            "fingerprint_digest": _fingerprint_digest(folder),
        }
        (folder / DATASET_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return open_dataset(folder)


def plan_dataset_folder(name: str, *, repo_root: str | Path | None = None) -> Path:
    """Where a new dataset called ``name`` goes; refused if the name is invalid or taken."""
    if not _NAME_RE.match(name or ""):
        raise DatasetError(f"invalid dataset name {name!r}: letters, digits, '.', '_' or '-', not starting with '.'")
    folder = datasets_root(repo_root) / name
    if folder.exists():
        if (folder / DATASET_FILENAME).exists():
            raise DatasetError(f"a dataset called {name!r} exists; datasets are never rebuilt, pick another name")
        raise DatasetError(f"{folder} holds an unfinished build ({BUILD_STATUS_FILENAME}); delete it and build again")
    return folder


def _from_record(path: Path, record: Mapping[str, Any]) -> Dataset:
    split = record.get("split") or {}
    stats = record.get("clip_duration_stats") or {}
    speed = record.get("speed") or {}
    return Dataset(
        name=path.name,
        path=path,
        legacy=False,
        egs_name=record.get("egs_name"),
        created_at=record.get("created_at"),
        sources=tuple(str(s) for s in record.get("sources") or ()),
        slicing=dict(record.get("slicing") or {}),
        valid_fraction=split.get("valid_fraction"),
        seed=split.get("seed"),
        train_clips=split.get("train"),
        valid_clips=split.get("valid"),
        train_seconds=record.get("train_seconds"),
        slice_length_s=stats.get("slice_length_s"),
        speed_ratios=tuple(float(r) for r in speed.get("ratios") or ()),
        fingerprint_digest=record.get("fingerprint_digest"),
    )


def _from_legacy(path: Path) -> Dataset:
    from .corpus import read_corpus_manifest

    prepared = read_json(path / "logs" / "prepare_dataset.json")
    prepared = prepared if isinstance(prepared, dict) else {}
    egs_name = _egs_name(path)
    input_dir = prepared.get("input_dir")
    sources: tuple[str, ...] = (str(input_dir),) if input_dir else ()
    slicing: dict[str, Any] = {}
    if isinstance(input_dir, str):
        corpus = read_corpus_manifest(Path(input_dir).parent) if Path(input_dir).name == "clips" else None
        if corpus:
            sources = tuple(str(s) for s in corpus.get("sources") or sources)
            slicing = dict(corpus.get("slicing") or {})
    stats = prepared.get("clip_duration_stats") or {}
    train_manifest = path / "egs" / egs_name / "train" / "data.jsonl.gz" if egs_name else None
    if not stats and train_manifest is not None and train_manifest.exists():
        # prepared before prepare recorded clip stats: read them off the manifest
        from .clip_stats import compute_clip_stats, read_manifest_durations

        stats = compute_clip_stats(read_manifest_durations(train_manifest) or []) or {}
    created = (path / "logs" / "prepare_dataset.json").stat().st_mtime if prepared else None
    return Dataset(
        name=path.name,
        path=path,
        legacy=True,
        egs_name=egs_name,
        created_at=_iso(created) if created else None,
        sources=sources,
        slicing=slicing,
        valid_fraction=prepared.get("valid_fraction"),
        seed=prepared.get("seed"),
        train_clips=prepared.get("train_count"),
        valid_clips=prepared.get("valid_count"),
        train_seconds=_manifest_seconds(train_manifest) if train_manifest and train_manifest.exists() else None,
        slice_length_s=stats.get("slice_length_s") if isinstance(stats, dict) else None,
        speed_ratios=tuple(float(r) for r in prepared.get("speed_ratios") or ()),
        fingerprint_digest=_fingerprint_digest(path),
    )


def _egs_name(path: Path) -> str | None:
    names = sorted(child.name for child in (path / "egs").glob("*") if (child / "train" / "data.jsonl.gz").exists())
    return names[0] if len(names) == 1 else (names[0] if names else None)


def _manifest_seconds(manifest: Path) -> float | None:
    from .clip_stats import read_manifest_durations

    durations = read_manifest_durations(manifest)
    return round(sum(durations), 1) if durations else None


def _fingerprint_digest(path: Path) -> str | None:
    from .prepare import _dataset_fingerprint_digest

    fingerprint = read_json(path / "logs" / "dataset_fingerprint.json")
    if not isinstance(fingerprint, dict):
        return None
    try:
        return _dataset_fingerprint_digest(fingerprint)
    except RuntimeError:
        return None


def _iso(timestamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, UTC).isoformat()
