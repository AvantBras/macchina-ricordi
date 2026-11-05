"""Datasets in the GUI: the source library, the prepared workspaces, new corpora, and rebalancing.

- The library is every folder under ``library`` (the home of sources), ``examples/sample_data`` (where the
  albums collected before October 2026 stay, because records point into it) and ``training_datasets``.
- A new corpus is ``data corpus`` (clips from source folders, whole or sliced), then
  optionally ``data prepare`` into a new workspace, run detached in one shell, so it
  keeps going if the GUI stops. Each run leaves ``artifacts/data_jobs/<id>.log`` and
  ``<id>_status.json``; the log ends with an exit marker the page reads the outcome from.
- ``prepare`` wipes a workspace's ``data/``, ``egs/`` and ``logs/`` first, so the GUI only ever
  prepares into a workspace that does not exist yet.
- Rebalancing computes the weights as a job (``plan_rebalance``), shows the clusters with
  example clips, and saves them as a named overlay only when asked (``save_weights_overlay``),
  refusing if the manifest changed since. Overlays live outside the workspace.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shlex
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from macchina_ricordi.data.clip_stats import compute_clip_stats, extract_override_float, pad_share
from macchina_ricordi.data.corpus import (
    AUDIO_EXTENSIONS,
    CLIPS_DIRNAME,
    AudioProbe,
    AudioProbeCache,
    CorpusError,
    CorpusSpec,
    find_duplicate_recordings,
    plan_corpus,
    plan_digest,
    read_corpus_manifest,
    sanitize,
)
from macchina_ricordi.data.datasets import (
    DATASET_FILENAME,
    DATASETS_DIRNAME,
    DatasetError,
    dataset_dirs,
    find_dataset_dir,
    plan_dataset_folder,
)
from macchina_ricordi.data.weights import list_weights_overlays
from macchina_ricordi.musicgen.shared import read_json, write_text_atomic
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME

from .data import SERVED_AUDIO_SUFFIXES
from .jobs import DetachedJobs
from .jobs import now_iso as _now_iso
from .jobs import relative as _relative

LIBRARY_ROOTS = ("library", "examples/sample_data", "training_datasets")
CORPORA_ROOT = "training_datasets"
JOBS_DIR = "artifacts/data_jobs"
JOB_SCHEMA = "macchina_ricordi.gui.data_job.v1"
EXIT_MARKER = "[macchina_ricordi data job exit"
PREVIEW_TTL_SECONDS = 30 * 60
SKIPPED_SHOWN = 60
PAD_TOLERANCE_SECONDS = 0.05
PAD_WARN_SHARE = 0.05
PROBE_CACHE = "artifacts/gui_cache/audio_probe.json"
SELECTIONS_FILE = "artifacts/corpus_selections.json"
SELECTIONS_SCHEMA = "macchina_ricordi.gui.corpus_selections.v1"
LIBRARY_TAGS_SHOWN = 12
_ID3V1_GENRE = re.compile(r"\(\d+\)")
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_WORKSPACE_NAME = re.compile(r"^\.[A-Za-z0-9._-]+$")
_DATASET_NAME = re.compile(r"^\.?[A-Za-z0-9][A-Za-z0-9._-]*$")


class CorpusRequestError(ValueError):
    pass


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, EOFError, json.JSONDecodeError):
        return []


def clip_group(path: Path, split_root: Path) -> str:
    """The source a clip came from: the part of its name before ``__``, else its folder under the split."""
    if "__" in path.stem:
        return path.stem.split("__", 1)[0]
    try:
        parts = path.relative_to(split_root).parts
    except ValueError:
        parts = ()
    return parts[0] if len(parts) > 1 else "—"


@dataclass(frozen=True)
class CorpusRequest:
    name: str
    sources: tuple[str, ...]
    slice_seconds: float | None
    hop_seconds: float | None
    min_clip_seconds: float | None
    prepare: bool
    dataset: str
    workspace: str
    valid_fraction: float
    seed: int
    max_silence_fraction: float
    speed_ratios: str
    exclude: tuple[str, ...] = ()
    skip_duplicates: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CorpusRequest:
        def number(key: str, default: float | None = None) -> float | None:
            value = payload.get(key)
            if value in (None, ""):
                return default
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise CorpusRequestError(f"{key} must be a number") from exc

        name = str(payload.get("name") or "").strip()
        sources = payload.get("sources") or []
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise CorpusRequestError("sources must be a list of library folders")
        exclude = payload.get("exclude") or []
        if not isinstance(exclude, list) or not all(isinstance(e, str) for e in exclude):
            raise CorpusRequestError("exclude must be a list of files")
        speed = str(payload.get("speed_ratios") or "").strip()
        if speed and not re.fullmatch(r"[0-9.]+(\s*,\s*[0-9.]+)*", speed):
            raise CorpusRequestError("speed ratios are numbers separated by commas, e.g. 0.9,1.1")
        valid_fraction = number("valid_fraction", 0.1)
        silence = number("max_silence_fraction", 0.85)
        assert valid_fraction is not None and silence is not None
        if not 0.0 < valid_fraction < 0.5:
            raise CorpusRequestError("the validation share must be between 0 and 0.5")
        if not 0.0 < silence <= 1.0:
            raise CorpusRequestError("the silence threshold must be between 0 and 1")
        # a built dataset is named after its corpus: datasets/<name>
        workspace = name
        return cls(
            name=name,
            sources=tuple(dict.fromkeys(sources)),
            slice_seconds=number("slice_seconds"),
            hop_seconds=number("hop_seconds"),
            min_clip_seconds=number("min_clip_seconds"),
            prepare=bool(payload.get("prepare", True)),
            dataset=str(payload.get("dataset") or ""),
            workspace=workspace,
            valid_fraction=valid_fraction,
            seed=int(number("seed", 42) or 0),
            max_silence_fraction=silence,
            speed_ratios=speed.replace(" ", ""),
            exclude=tuple(sorted(set(exclude))),
            skip_duplicates=bool(payload.get("skip_duplicates", False)),
        )

    def key(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()[:20]


def _stored_selections(path: Path) -> dict[str, Any]:
    """The saved selections; a file that exists but cannot be parsed is an error, never an empty list."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except ValueError as exc:
        raise CorpusRequestError(f"{path.name} is damaged; fix or move it, or saving would erase it") from exc


@dataclass
class CorpusManager:
    repo_root: Path
    xps_by_workspace: Callable[[], dict[str, list[str]]] = dict
    """workspace folder name -> the XP sigs trained on it (from the catalogue)."""
    generation_dirs: Callable[[Collection[str]], list[dict[str, Any]]] = lambda sigs: []
    """Folders of generated audio made by these XPs, for coverage rebalancing."""
    active_workspaces: Callable[[], Collection[str]] = lambda: ()
    cli_prefix: tuple[str, ...] | None = None
    musicgen_check: Callable[[], str | None] | None = None
    """Returns why `data prepare` cannot run here (no AudioCraft runtime), or None."""
    profiles_loader: Callable[[], list[Any]] | None = None
    _durations: AudioProbeCache = field(default_factory=AudioProbeCache)
    _probe_cache_loaded: bool = False
    _twins: dict[Path, list[Path]] = field(default_factory=dict)
    _previews: dict[str, tuple[CorpusRequest, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self._jobs = DetachedJobs(self.repo_root, JOBS_DIR, JOB_SCHEMA, EXIT_MARKER)

    # ── recipes ──

    def _profiles(self) -> list[Any]:
        if self.profiles_loader is not None:
            return self.profiles_loader()
        from macchina_ricordi.musicgen.config import load_musicgen_project_config

        return list(load_musicgen_project_config(self.repo_root).recipes.values())

    def _profile_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "name": recipe.name,
                "segment_duration": extract_override_float(recipe.overrides, "dataset.segment_duration"),
            }
            for recipe in self._profiles()
        ]

    def _dataset_rows(self) -> list[dict[str, Any]]:
        """What a new workspace is prepared as: one dataset any recipe can train on, at each recipe's segment."""
        rows = self._profile_rows()
        segments = sorted({row["segment_duration"] for row in rows if row["segment_duration"]})
        return [{"name": DATASET_CONFIG_NAME, "profiles": [row["name"] for row in rows], "segments": segments}]

    def _padding_audit(self, durations: list[float], dataset: str | None = None) -> list[dict[str, Any]]:
        """For each segment length a recipe trains at: the share of clips shorter than it."""
        by_segment: dict[float, list[str]] = {}
        for row in self._profile_rows():
            segment = row["segment_duration"]
            if segment and durations:
                by_segment.setdefault(float(segment), []).append(row["name"])
        audit = []
        for segment, names in sorted(by_segment.items()):
            # decoded slices land a few ms short of their nominal length; that is not padding
            padded = pad_share(durations, segment - PAD_TOLERANCE_SECONDS)
            audit.append(
                {
                    "segment_duration": segment,
                    "profiles": names,
                    "padded_share": round(padded, 4),
                    "padded_clips": round(padded * len(durations)),
                    "median": round(statistics.median(durations), 3),
                }
            )
        return audit

    # ── the library ──

    def library(self) -> dict[str, Any]:
        started = time.monotonic()
        self._load_probe_cache()
        used_by = self._sources_in_use()
        folders = []
        for root_name in LIBRARY_ROOTS:
            root = self.repo_root / root_name
            if not root.is_dir():
                continue
            for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not folder.is_dir() or folder.name.startswith("."):
                    continue
                folders.append(self._library_folder(folder, used_by))
        self._mark_duplicates(folders)
        self._save_probe_cache()
        return {
            "roots": list(LIBRARY_ROOTS),
            "folders": folders,
            "datasets": self._dataset_rows(),
            "scan_seconds": round(time.monotonic() - started, 2),
        }

    def _load_probe_cache(self) -> None:
        with self._lock:
            if not self._probe_cache_loaded:
                self._durations.load(self.repo_root / PROBE_CACHE)
                self._probe_cache_loaded = True

    def _save_probe_cache(self) -> None:
        try:
            self._durations.save(self.repo_root / PROBE_CACHE)
        except OSError:
            pass

    def _audio_files(self, folder: Path) -> list[Path]:
        return sorted(
            path
            for path in folder.rglob("*")
            if path.suffix.lower() in AUDIO_EXTENSIONS and not path.name.startswith(".") and path.is_file()
        )

    def _library_folder(self, folder: Path, used_by: dict[Path, list[str]]) -> dict[str, Any]:
        formats: Counter[str] = Counter()
        artists: Counter[str] = Counter()
        albums: Counter[str] = Counter()
        genres: Counter[str] = Counter()
        years: set[int] = set()
        seconds = 0.0
        files = 0
        prefixed = 0
        unreadable = 0
        modified = folder.stat().st_mtime
        for path in self._audio_files(folder):
            files += 1
            formats[path.suffix.lower().lstrip(".")] += 1
            prefixed += "__" in path.stem
            modified = max(modified, path.stat().st_mtime)
            probe = self._durations.probe(path)
            if probe is None or probe.duration is None:
                unreadable += 1
                continue
            seconds += probe.duration
            if probe.artist:
                artists[probe.artist] += 1
            if probe.tags.get("album"):
                albums[probe.tags["album"]] += 1
            # "(49)" is an ID3v1 genre number, not a name
            if probe.tags.get("genre") and not _ID3V1_GENRE.fullmatch(probe.tags["genre"]):
                genres[probe.tags["genre"]] += 1
            if (year := _year(probe.tags.get("date"))) is not None:
                years.add(year)
        corpus = read_corpus_manifest(folder)
        workspaces = sorted(
            {
                name
                for source, names in used_by.items()
                if source == folder or folder in source.parents
                for name in names
            }
        )
        return {
            "path": _relative(folder, self.repo_root),
            "name": folder.name,
            "root": _relative(folder.parent, self.repo_root),
            "files": files,
            "seconds": round(seconds, 1),
            "formats": dict(formats.most_common()),
            "unreadable": unreadable,
            "modified_at": datetime.fromtimestamp(modified, tz=UTC).isoformat(timespec="seconds"),
            # most frequent first; album artist over track artist, so a compilation reads as one
            "artists": [name for name, _ in artists.most_common(LIBRARY_TAGS_SHOWN)],
            "artist_count": len(artists),
            "tagged": sum(artists.values()),
            "albums": [name for name, _ in albums.most_common(LIBRARY_TAGS_SHOWN)],
            "genres": [name for name, _ in genres.most_common(4)],
            "years": [min(years), max(years)] if years else None,
            # a folder of `<source>__<file>` names was assembled from other folders
            "assembled": bool(corpus) or (files > 0 and prefixed / files > 0.8),
            "corpus": {"slicing": corpus.get("slicing"), "sources": corpus.get("sources")} if corpus else None,
            "workspaces": workspaces,
        }

    def _mark_duplicates(self, folders: list[dict[str, Any]]) -> None:
        """Which folders hold the same recordings, among the ones not assembled from others."""
        owner: dict[Path, str] = {}
        for row in folders:
            if not row["assembled"]:
                for path in self._audio_files(self.repo_root / row["path"]):
                    owner[path] = row["path"]
        twins: dict[Path, list[Path]] = {}
        for group in find_duplicate_recordings(list(owner), self._durations):
            for path in group:
                twins[path] = [other for other in group if other != path]
        self._twins = twins
        for row in folders:
            shared: Counter[str] = Counter()
            for path, others in twins.items():
                if owner[path] == row["path"]:
                    shared.update({owner[other] for other in others if owner[other] != row["path"]})
            row["duplicates"] = [{"path": path, "files": count} for path, count in shared.most_common()]

    def library_folder(self, relative: str) -> dict[str, Any]:
        """The audio files of one library folder, with their tags, to listen before picking it."""
        folder = self._source_folder(relative)
        self._load_probe_cache()
        files = []
        for path in self._audio_files(folder):
            probe = self._durations.probe(path) or AudioProbe(None)
            files.append(
                {
                    "path": _relative(path, self.repo_root),
                    "name": path.relative_to(folder).as_posix(),
                    "duration": round(probe.duration, 2) if probe.duration is not None else None,
                    "artist": probe.artist,
                    "title": probe.tags.get("title"),
                    "album": probe.tags.get("album"),
                    "duplicates": [_relative(other, self.repo_root) for other in self._twins.get(path, [])],
                    # what the browser can play from /files/
                    "playable": path.suffix.lower() in SERVED_AUDIO_SUFFIXES,
                }
            )
        self._save_probe_cache()
        return {"path": _relative(folder, self.repo_root), "files": files}

    def _sources_in_use(self) -> dict[Path, list[str]]:
        used: dict[Path, list[str]] = {}
        for workspace in self._workspace_dirs():
            prepared = read_json(workspace / "logs" / "prepare_dataset.json") or {}
            if isinstance(prepared.get("input_dir"), str):
                used.setdefault(Path(prepared["input_dir"]).resolve(), []).append(workspace.name)
        return used

    # ── saved selections ──

    def selections(self) -> dict[str, Any]:
        """Saved sets of library folders (and files left out), and the recipes of the built corpora."""
        stored = read_json(self.repo_root / SELECTIONS_FILE) or {}
        saved = []
        for name, entry in sorted((stored.get("selections") or {}).items(), key=lambda item: item[0].lower()):
            if not isinstance(entry, dict):
                continue
            sources = [str(path) for path in entry.get("sources") or []]
            saved.append(
                {
                    "name": name,
                    "sources": sources,
                    "exclude": [str(path) for path in entry.get("exclude") or []],
                    "saved_at": entry.get("saved_at"),
                    "missing": [path for path in sources if not (self.repo_root / path).is_dir()],
                }
            )
        corpora = []
        root = self.repo_root / CORPORA_ROOT
        for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()) if root.is_dir() else ():
            manifest = read_corpus_manifest(folder)
            if not manifest:
                continue
            sources = [_relative(path, self.repo_root) for path in manifest.get("sources") or []]
            corpora.append(
                {
                    "name": manifest.get("name") or folder.name,
                    "sources": sources,
                    "exclude": [_relative(path, self.repo_root) for path in manifest.get("excluded") or []],
                    "saved_at": manifest.get("created_at"),
                    "slicing": manifest.get("slicing"),
                    "skip_duplicates": bool(manifest.get("skip_duplicates")),
                    "missing": [path for path in sources if not (self.repo_root / path).is_dir()],
                }
            )
        return {"saved": saved, "corpora": corpora}

    def save_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name or len(name) > 80:
            raise CorpusRequestError("a selection needs a name of at most 80 characters")
        sources = payload.get("sources") or []
        exclude = payload.get("exclude") or []
        if not isinstance(sources, list) or not sources or not all(isinstance(x, str) for x in sources):
            raise CorpusRequestError("tick at least one folder to save a selection")
        if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
            raise CorpusRequestError("exclude must be a list of files")
        folders = [_relative(self._source_folder(source), self.repo_root) for source in dict.fromkeys(sources)]
        kept_out = sorted(path for path in set(exclude) if any(path.startswith(f"{folder}/") for folder in folders))
        with self._lock:
            path = self.repo_root / SELECTIONS_FILE
            stored = _stored_selections(path)
            entries = dict(stored.get("selections") or {})
            entries[name] = {"sources": folders, "exclude": kept_out, "saved_at": _now_iso()}
            stored = {"schema": SELECTIONS_SCHEMA, "selections": entries}
            write_text_atomic(path, json.dumps(stored, indent=2, ensure_ascii=False))
        return self.selections()

    def delete_selection(self, name: str) -> dict[str, Any]:
        with self._lock:
            path = self.repo_root / SELECTIONS_FILE
            stored = _stored_selections(path)
            entries = dict(stored.get("selections") or {})
            if entries.pop(name, None) is None:
                raise CorpusRequestError(f"no saved selection named {name!r}")
            stored = {"schema": SELECTIONS_SCHEMA, "selections": entries}
            write_text_atomic(path, json.dumps(stored, indent=2, ensure_ascii=False))
        return self.selections()

    # ── workspaces ──

    def _workspace_dirs(self) -> list[Path]:
        return dataset_dirs(self.repo_root)

    def _workspace(self, name: str) -> Path:
        """A dataset by its name: ``bear_bones`` is ``datasets/bear_bones``, ``.workspace_x`` a legacy folder."""
        path = find_dataset_dir(name, repo_root=self.repo_root) if _DATASET_NAME.match(name or "") else None
        if path is None or path not in self._workspace_dirs():
            raise CorpusRequestError(f"no dataset named {name!r} in this repo")
        return path

    def workspaces(self) -> dict[str, Any]:
        xps = self.xps_by_workspace()
        active = set(self.active_workspaces())
        rows = []
        for workspace in self._workspace_dirs():
            prepared = read_json(workspace / "logs" / "prepare_dataset.json") or {}
            splits: dict[str, Any] = {}
            datasets = []
            for manifest in sorted((workspace / "egs").glob("*/*/data.jsonl.gz")):
                dataset, split = manifest.parent.parent.name, manifest.parent.name
                if dataset not in datasets:
                    datasets.append(dataset)
                durations = [float(row.get("duration") or 0) for row in _read_manifest(manifest)]
                stats = compute_clip_stats(durations)
                splits.setdefault(split, {"clips": 0, "seconds": 0.0, "stats": None})
                splits[split]["clips"] += len(durations)
                splits[split]["seconds"] += round(sum(durations), 1)
                if split == "train":
                    splits[split]["stats"] = stats
            rebalanced = [
                dataset for dataset in datasets if (workspace / "egs" / dataset / "train" / "rebalance.json").exists()
            ]
            input_dir = prepared.get("input_dir")
            rows.append(
                {
                    "name": workspace.name,
                    "datasets": datasets,
                    "splits": splits,
                    "source": _relative(input_dir, self.repo_root) if isinstance(input_dir, str) else None,
                    "source_exists": isinstance(input_dir, str) and Path(input_dir).is_dir(),
                    "corpus": self._corpus_of(input_dir),
                    "prepared_at": _mtime_iso(workspace / "logs" / "prepare_dataset.json"),
                    "speed_variants": prepared.get("train_speed_variants") or 0,
                    "dropped_silent": (prepared.get("train_dropped_silent") or 0)
                    + (prepared.get("valid_dropped_silent") or 0),
                    "rebalanced": rebalanced,
                    "xps": xps.get(workspace.name, []),
                    "training": workspace.name in active,
                }
            )
        return {"workspaces": rows, "profiles": self._profile_rows(), "jobs": self.jobs()}

    def _corpus_of(self, input_dir: Any) -> str | None:
        if not isinstance(input_dir, str):
            return None
        folder = Path(input_dir)
        corpus = read_corpus_manifest(folder.parent) if folder.name == CLIPS_DIRNAME else None
        return _relative(folder.parent, self.repo_root) if corpus else None

    def workspace(self, name: str) -> dict[str, Any]:
        workspace = self._workspace(name)
        prepared = read_json(workspace / "logs" / "prepare_dataset.json") or {}
        datasets = []
        for dataset_dir in sorted((workspace / "egs").glob("*")):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            splits = {}
            groups: dict[str, dict[str, float]] = {}
            for split in ("train", "valid"):
                manifest = dataset_dir / split / "data.jsonl.gz"
                split_root = workspace / "data" / split
                clips = []
                for row in _read_manifest(manifest):
                    path = Path(str(row.get("path")))
                    group = clip_group(path, split_root)
                    duration = float(row.get("duration") or 0)
                    clips.append(
                        {
                            "path": _relative(path, self.repo_root),
                            "name": path.name,
                            "group": group,
                            "duration": round(duration, 3),
                            "weight": row.get("weight"),
                            "speed": "__speed" in path.stem,
                        }
                    )
                    tally = groups.setdefault(
                        group, {"train": 0, "train_seconds": 0.0, "valid": 0, "valid_seconds": 0.0}
                    )
                    tally[split] += 1
                    tally[f"{split}_seconds"] += duration
                splits[split] = {
                    "clips": clips,
                    "stats": compute_clip_stats([c["duration"] for c in clips]),
                    "manifest": _relative(manifest, self.repo_root) if manifest.exists() else None,
                }
            audit = self._padding_audit([c["duration"] for c in splits["train"]["clips"]], dataset)
            sidecar = read_json(dataset_dir / "train" / "rebalance.json")
            if sidecar:
                sidecar = self._relative_examples(sidecar)
            datasets.append(
                {
                    "name": dataset,
                    "splits": splits,
                    "groups": {g: {k: round(v, 1) for k, v in t.items()} for g, t in sorted(groups.items())},
                    "audit": audit,
                    "rebalance": sidecar,
                    "weights": [
                        {
                            **overlay.summary(),
                            "folder": _relative(overlay.folder, self.repo_root),
                            "stale": overlay.is_stale(),
                            "cluster_table": self._relative_examples({"cluster_table": list(overlay.cluster_table)})[
                                "cluster_table"
                            ],
                        }
                        for overlay in list_weights_overlays(self.repo_root, name)
                        if overlay.dataset_name == dataset
                    ],
                }
            )
        xps = self.xps_by_workspace().get(name, [])
        input_dir = prepared.get("input_dir")
        return {
            "name": name,
            "prepared": prepared or None,
            "prepared_at": _mtime_iso(workspace / "logs" / "prepare_dataset.json"),
            "source": _relative(input_dir, self.repo_root) if isinstance(input_dir, str) else None,
            "source_exists": isinstance(input_dir, str) and Path(input_dir).is_dir(),
            "corpus": self._corpus_of(input_dir),
            "datasets": datasets,
            "xps": xps,
            "training": name in set(self.active_workspaces()),
            "generation_dirs": self.generation_dirs(xps),
            "rebalance_job": self.rebalance_state(name),
        }

    def _relative_examples(self, sidecar: dict[str, Any]) -> dict[str, Any]:
        table = []
        for row in sidecar.get("cluster_table") or []:
            if isinstance(row, dict) and isinstance(row.get("examples"), list):
                row = {**row, "examples": [_relative(p, self.repo_root) for p in row["examples"]]}
            table.append(row)
        return {**sidecar, "cluster_table": table}

    # ── new corpora ──

    def _source_folder(self, relative: str) -> Path:
        path = (self.repo_root / relative).resolve()
        roots = [(self.repo_root / root).resolve() for root in LIBRARY_ROOTS]
        if not path.is_dir() or path.parent not in roots:
            raise CorpusRequestError(f"{relative!r} is not a folder of the library")
        return path

    def _spec(self, request: CorpusRequest) -> tuple[CorpusSpec, Path]:
        spec = CorpusSpec(
            name=request.name,
            sources=tuple(self._source_folder(source) for source in request.sources),
            slice_seconds=request.slice_seconds,
            hop_seconds=request.hop_seconds,
            min_clip_seconds=request.min_clip_seconds,
            exclude=frozenset((self.repo_root / path).resolve() for path in request.exclude),
            skip_duplicates=request.skip_duplicates,
        )
        if request.prepare:
            return spec, self.repo_root / DATASETS_DIRNAME / request.name
        return spec, self.repo_root / CORPORA_ROOT / request.name

    def corpus_preview(self, request: CorpusRequest) -> dict[str, Any]:
        try:
            spec, output_dir = self._spec(request)
            plan = plan_corpus(spec, output_dir, durations=self._durations)
        except CorpusError as exc:
            raise CorpusRequestError(str(exc)) from exc
        durations = [clip.duration or 0.0 for clip in plan.clips]
        blockers, warnings = [], []
        if (output_dir / CLIPS_DIRNAME).exists() and any((output_dir / CLIPS_DIRNAME).iterdir()):
            blockers.append(f"{_relative(output_dir, self.repo_root)} already holds a corpus; pick another name")
        if not plan.clips:
            blockers.append("no clip would be written")
        dataset, padding = None, []
        if request.prepare:
            dataset = next((row for row in self._dataset_rows() if row["name"] == request.dataset), None)
            if dataset is None:
                blockers.append(f"new workspaces are prepared as {DATASET_CONFIG_NAME!r}, not {request.dataset!r}")
            try:
                plan_dataset_folder(request.name, repo_root=self.repo_root)
            except DatasetError as exc:
                blockers.append(str(exc))
            problem = self.musicgen_check() if self.musicgen_check else None
            if problem:
                blockers.append(problem)
            padding = self._padding_audit(durations)
            if padding and all(row["padded_share"] > PAD_WARN_SHARE for row in padding):
                shortest = padding[0]
                warnings.append(
                    f"{shortest['padded_share']:.0%} of the clips are shorter than {shortest['segment_duration']:g} s, "
                    "the shortest segment any recipe trains at: training would pad them with "
                    "silence (and refuses when the median is)"
                )
        if sum(1 for counts in plan.per_source.values() if counts["clips"]) > 1:
            top, counts = max(plan.per_source.items(), key=lambda item: item[1]["clips"])
            share = counts["clips"] / len(plan.clips)
            if share > 0.5:
                warnings.append(
                    f"{top} is {share:.0%} of the clips, so about {share:.0%} of what training draws "
                    "(it draws each clip equally often); rebalancing the workspace can even the sources out"
                )
        duplicates = self._duplicate_pairs(plan)
        if duplicates and not request.skip_duplicates:
            count = sum(pair["files"] for pair in duplicates)
            warnings.append(
                f"{count} file{'s are' if count > 1 else ' is'} probably the same recording as a file in another "
                "ticked folder (same title and length, matching audio), so they would count twice; "
                "tick 'leave probable duplicates out', or leave one copy out by hand"
            )
        key = hashlib.sha256(f"{request.key()}:{plan_digest(plan)}".encode()).hexdigest()[:20]
        if not blockers:
            with self._lock:
                self._previews[key] = (request, time.time())
        return {
            "name": request.name,
            "output_dir": _relative(output_dir, self.repo_root),
            "per_source": plan.per_source,
            "clips": len(plan.clips),
            "seconds": round(plan.total_seconds, 1),
            "stats": compute_clip_stats(durations),
            "examples": [clip.name for clip in plan.clips[:6]],
            "skipped": [{"path": _relative(p, self.repo_root), "reason": r} for p, r in plan.skipped[:SKIPPED_SHOWN]],
            "skipped_count": len(plan.skipped),
            "excluded_count": len(plan.excluded),
            "duplicates": duplicates,
            "prepare": request.prepare,
            "workspace": request.workspace if request.prepare else None,
            "dataset": dataset,
            "padding": padding,
            "commands": [shlex.join(command) for command in self._commands(request, output_dir)],
            "blockers": blockers,
            "warnings": warnings,
            "key": None if blockers else key,
        }

    def _duplicate_pairs(self, plan: Any) -> list[dict[str, Any]]:
        """Probable duplicates by pair of folders: how many, and which copy is (or would be) kept."""
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for path, keeper in plan.duplicates:
            folders = (_relative(path.parent, self.repo_root), _relative(keeper.parent, self.repo_root))
            row = pairs.setdefault(
                folders, {"folder": folders[0], "kept_folder": folders[1], "files": 0, "examples": []}
            )
            row["files"] += 1
            if len(row["examples"]) < 4:
                row["examples"].append({"path": path.name, "kept": keeper.name})
        return sorted(pairs.values(), key=lambda row: -row["files"])

    def _cli(self) -> list[str]:
        return list(self.cli_prefix or (sys.executable, "-m", "macchina_ricordi.cli"))

    def _commands(self, request: CorpusRequest, output_dir: Path) -> list[list[str]]:
        verb = "build" if request.prepare else "corpus"
        corpus = [*self._cli(), "data", verb, request.name]
        corpus += [_relative(self._source_folder(source), self.repo_root) for source in request.sources]
        if not request.prepare:
            corpus += ["--output-dir", _relative(output_dir, self.repo_root)]
        for flag, value in (
            ("--slice", request.slice_seconds),
            ("--hop", request.hop_seconds),
            ("--min-clip", request.min_clip_seconds),
        ):
            if value is not None:
                corpus += [flag, f"{value:g}"]
        if request.skip_duplicates:
            corpus.append("--skip-duplicates")
        for path in request.exclude:
            corpus += ["--exclude", path]
        if request.prepare:
            corpus += [
                "--valid-fraction", f"{request.valid_fraction:g}", "--seed", str(request.seed),
                "--max-silence-fraction", f"{request.max_silence_fraction:g}",
            ]  # fmt: skip
            if request.speed_ratios:
                corpus += ["--speed-ratios", request.speed_ratios]
        return [corpus]

    def corpus_launch(self, request: CorpusRequest, key: str) -> dict[str, Any]:
        with self._lock:
            preview = self._previews.get(key)
            if preview is None or preview[0] != request or time.time() - preview[1] > PREVIEW_TTL_SECONDS:
                raise CorpusRequestError("the form changed or the preview is old; preview again")
            self._refuse_if_busy()
        fresh = self.corpus_preview(request)
        if fresh["key"] != key:
            raise CorpusRequestError("the sources changed since the preview; preview again")
        _, output_dir = self._spec(request)
        title = f"corpus {request.name}" + (f" → {request.workspace}" if request.prepare else "")
        return self._launch(
            "corpus",
            title,
            self._commands(request, output_dir),
            {
                "corpus": _relative(output_dir, self.repo_root),
                "workspace": request.workspace if request.prepare else None,
            },
        )

    def _dataset_ready(self, name: object) -> bool:
        path = find_dataset_dir(str(name), repo_root=self.repo_root) if name else None
        return path is not None and (
            (path / DATASET_FILENAME).exists() or (path / "logs" / "prepare_dataset.json").exists()
        )

    # ── jobs ──

    def _refuse_if_busy(self) -> None:
        running = [job for job in self.jobs() if job["active"]]
        if running:
            raise CorpusRequestError(f"another data job is running ({running[0]['title']}); wait for it")

    def _launch(
        self, kind: str, title: str, commands: list[list[str]], meta: dict[str, Any], job_id: str | None = None
    ) -> dict[str, Any]:
        return self.job(self._jobs.launch(kind, title, commands, meta, job_id)["id"])

    def jobs(self) -> list[dict[str, Any]]:
        return [self._summarize_job(record) for _, record in self._jobs.records()]

    def _summarize_job(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._jobs.summarize(record),
            "corpus": record.get("corpus"),
            "workspace": record.get("workspace"),
            "workspace_ready": self._dataset_ready(record.get("workspace")),
        }

    def job(self, job_id: str) -> dict[str, Any]:
        record = self._jobs.record(job_id)
        if record is None:
            raise CorpusRequestError(f"no data job {job_id!r}")
        return {**self._summarize_job(record), "log": self._jobs.log(record)}

    # ── rebalancing ──
    # Weights are computed by `data rebalance --plan-out` as a data job (a process of its own,
    # so feature extraction's worker pool never forks this threaded server) and written by
    # `save_weights_overlay` only when asked.

    def rebalance_start(self, workspace_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        workspace = self._workspace(workspace_name)
        dataset = str(payload.get("dataset") or "")
        if not (workspace / "egs" / dataset / "train" / "data.jsonl.gz").exists():
            raise CorpusRequestError(f"{workspace_name} has no train manifest for {dataset!r}")
        try:
            k = int(payload.get("k") or 10)
            alpha = float(payload.get("alpha") or 0.5)
            cap = float(payload.get("cap") or 4.0)
            seed = int(payload.get("seed") or 0)
        except (TypeError, ValueError) as exc:
            raise CorpusRequestError("k, alpha, cap and seed must be numbers") from exc
        if not 2 <= k <= 64 or not 0 < alpha <= 2 or not 1 < cap <= 20:
            raise CorpusRequestError("k is 2–64, alpha 0–2, cap 1–20")
        gens = []
        for folder in payload.get("gens_dirs") or []:
            path = (self.repo_root / str(folder)).resolve()
            if not path.is_dir() or not path.is_relative_to(self.repo_root):
                raise CorpusRequestError(f"not a folder in this repo: {folder}")
            gens.append(_relative(path, self.repo_root))
        with self._lock:
            self._refuse_if_busy()
            job_id = self._jobs.new_id("rebalance")
            plan_path = self._jobs.path / f"{job_id}_plan.json"
            command = [
                *self._cli(), "data", "rebalance",
                "--workspace-dir", workspace_name, "--dataset-name", dataset,
                "--k", str(k), "--alpha", f"{alpha:g}", "--cap", f"{cap:g}",
                "--seed", str(seed), "--plan-out", _relative(plan_path, self.repo_root),
            ]  # fmt: skip
            for folder in gens:
                command += ["--gens-dir", folder]
            return self._launch(
                "rebalance",
                f"weights for {workspace_name}",
                [command],
                {
                    "workspace": workspace_name,
                    "dataset": dataset,
                    "plan": _relative(plan_path, self.repo_root),
                    "params": {"k": k, "alpha": alpha, "cap": cap, "seed": seed},
                    "gens_dirs": gens,
                },
                job_id=job_id,
            )

    def _rebalance_records(self, workspace_name: str) -> list[tuple[Path, dict[str, Any]]]:
        records = self._jobs.records("*_rebalance*_status.json")
        return [(path, record) for path, record in records if record.get("workspace") == workspace_name]

    def rebalance_state(self, workspace_name: str) -> dict[str, Any] | None:
        """The latest weights computed for this workspace: the job, and the clusters once it is done."""
        records = self._rebalance_records(workspace_name)
        if not records:
            return None
        _, record = records[0]
        state = {
            **self._summarize_job(record),
            "dataset": record.get("dataset"),
            "params": record.get("params"),
            "gens_dirs": record.get("gens_dirs", []),
            "applied_at": record.get("applied_at"),
            "saved_as": record.get("saved_as"),
        }
        plan = self._load_plan(record)
        if plan is not None:
            state["result"] = {
                "mode": plan.mode,
                "digest": plan.weights_digest,
                "segments": plan.segments,
                "readable_segments": plan.readable_segments,
                "generations": plan.generations,
                "cluster_table": self._relative_examples({"cluster_table": plan.cluster_table})["cluster_table"],
                "stale": _manifest_changed(plan),
            }
        return state

    def _load_plan(self, record: dict[str, Any]) -> Any:
        from macchina_ricordi.data.weights import load_rebalance_plan

        plan_path = self.repo_root / str(record.get("plan") or "")
        if not record.get("plan") or not plan_path.is_file():
            return None
        try:
            return load_rebalance_plan(plan_path)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def rebalance_save(self, workspace_name: str, digest: str, name: str) -> dict[str, Any]:
        """Keep the latest computed weights as a named overlay; the workspace itself is not written."""
        from macchina_ricordi.data.weights import save_weights_overlay

        self._workspace(workspace_name)
        records = self._rebalance_records(workspace_name)
        plan = self._load_plan(records[0][1]) if records else None
        if plan is None or plan.weights_digest != digest:
            raise CorpusRequestError("compute the weights first (or again)")
        try:
            overlay = save_weights_overlay(plan, name=name.strip(), repo_root=self.repo_root)
        except (ValueError, FileExistsError, RuntimeError) as exc:
            raise CorpusRequestError(str(exc)) from exc
        path, record = records[0]
        record["saved_as"] = overlay.name
        record["applied_at"] = _now_iso()
        path.write_text(json.dumps(record, indent=2) + "\n")
        folder = _relative(overlay.folder, self.repo_root)
        return {"name": overlay.name, "digest": overlay.weights_digest, "folder": folder}

    def mix_save(self, workspace_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Save a mix of source folders as a named overlay. It reads the train manifest only, so it runs at once."""
        from macchina_ricordi.data.weights import plan_folder_mix, save_weights_overlay

        workspace = self._workspace(workspace_name)
        dataset = str(payload.get("dataset") or "")
        manifest = workspace / "egs" / dataset / "train" / "data.jsonl.gz"
        if not manifest.exists():
            raise CorpusRequestError(f"{workspace_name} has no train manifest for {dataset!r}")
        try:
            mix = {str(folder): float(share) for folder, share in dict(payload.get("mix") or {}).items()}
            plan = plan_folder_mix(manifest, mix)
            overlay = save_weights_overlay(plan, name=str(payload.get("name") or "").strip(), repo_root=self.repo_root)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise CorpusRequestError(str(exc)) from exc
        folder = _relative(overlay.folder, self.repo_root)
        return {"name": overlay.name, "digest": overlay.weights_digest, "folder": folder}

    def weights_delete(self, workspace_name: str, name: str) -> dict[str, Any]:
        from macchina_ricordi.data.weights import delete_weights_overlay

        self._workspace(workspace_name)
        try:
            folder = delete_weights_overlay(self.repo_root, workspace_name, name)
        except FileNotFoundError as exc:
            raise CorpusRequestError(str(exc)) from exc
        return {"deleted": _relative(folder, self.repo_root)}

    def rebalance_reset(self, workspace_name: str, dataset: str) -> dict[str, Any]:
        from macchina_ricordi.data.weights import clear_manifest_weights

        workspace = self._workspace(workspace_name)
        manifest = workspace / "egs" / dataset / "train" / "data.jsonl.gz"
        if not manifest.exists():
            raise CorpusRequestError(f"{workspace_name} has no train manifest for {dataset!r}")
        if workspace_name in set(self.active_workspaces()):
            raise CorpusRequestError("a training run is using this workspace; clear the weights after it ends")
        clear_manifest_weights(manifest)
        return {"cleared": _relative(manifest, self.repo_root)}


def _manifest_changed(plan: Any) -> bool:
    try:
        stat = plan.manifest_path.stat()
    except OSError:
        return True
    return (stat.st_size, stat.st_mtime) != tuple(plan.manifest_signature)


def _year(date: str | None) -> int | None:
    match = _YEAR.search(date or "")
    return int(match.group(1)) if match else None


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
    except OSError:
        return None


__all__ = ["CorpusManager", "CorpusRequest", "CorpusRequestError", "clip_group", "sanitize"]
