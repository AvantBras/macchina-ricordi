"""Read-only views over the zoo for the local GUI.

Models, parents, children, sessions and experiments come from one ``zoo.build_zoo`` per refresh, the same graph
``model show``, ``model lineage`` and the catalog read. Nothing is written back.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from macchina_ricordi.judging.store import DEFAULT_IMPRESSION_PATH, load_impressions
from macchina_ricordi.models.catalog import Catalog, build_catalog
from macchina_ricordi.models.lineage import build_lineage_graph_for_sig, lineage_graph_to_dict
from macchina_ricordi.models.record import Model, load_history
from macchina_ricordi.models.zoo import Zoo, build_zoo, session_row
from macchina_ricordi.musicgen.shared import read_json

TRAIN_METADATA_FILENAME = "macchina_ricordi_train_metadata.json"
HISTORY_FILENAME = "history.json"
ACTIVE_WINDOW_SECONDS = 15 * 60
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg"})
# the library also holds m4a/opus sources, which browsers play too
SERVED_AUDIO_SUFFIXES = AUDIO_SUFFIXES | {".m4a", ".opus"}
TEXT_SUFFIXES = frozenset({".md", ".json", ".txt", ".log", ".toml"})
CURVE_POINTS = 48


@dataclass(frozen=True)
class HistorySummary:
    epochs_done: int
    best_valid_ce: float | None
    best_valid_epoch: int | None
    last_valid_ce: float | None
    last_train_ce: float | None
    valid_curve: tuple[float, ...]


@dataclass(frozen=True)
class XpSummary:
    sig: str
    label: str | None
    profile: str | None
    workspace: str | None
    dataset_name: str | None
    train_count: int | None
    epochs_planned: int | None
    epochs_done: int
    best_valid_ce: float | None
    best_valid_epoch: int | None
    last_valid_ce: float | None
    last_train_ce: float | None
    valid_curve: tuple[float, ...]
    parents: tuple[str, ...]
    origin: str | None
    source_checkpoint: str | None
    written_at: str | None
    last_activity: float | None
    active: bool
    has_metadata: bool
    checkpoint_count: int
    sample_epochs: int
    generation_runs: int
    experiments: tuple[str, ...]
    name: str | None = None
    sessions: int = 0
    children: tuple[str, ...] = ()
    kind: str = "xp"
    """``xp`` for a trained Dora XP, ``bundle`` for a portable bundle (models/<name>). The GUI uses this to refuse
    Dora-only operations (resume training, checkpoint cruft) on a bundle."""
    folder: str | None = None
    """The model's folder, repo-relative: ``.dora/xps/<sig>`` for a trained XP, ``models/<name>`` for a bundle. The GUI
    shows it on the model page and hands it to warm start in place of a fabricated ``.dora/xps`` path."""


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    epoch: int | None
    size_mb: float
    modified: float


@dataclass(frozen=True)
class AudioFile:
    name: str
    path: str


@dataclass(frozen=True)
class GenerationRun:
    run_id: str
    mode: str
    prefix: str
    checkpoint: str | None
    duration_seconds: float
    temperature: float | None
    top_k: int | None
    seed: int | None
    written_at: str | None
    listening_summary: str | None
    listening_score: float | None
    files: tuple[AudioFile, ...]
    metadata_path: str


def _finite(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def summarize_history(history: list[dict[str, Any]]) -> HistorySummary:
    best_ce: float | None = None
    best_epoch: int | None = None
    last_valid: float | None = None
    last_train: float | None = None
    valid_curve: list[float] = []
    for epoch, entry in enumerate(history, start=1):
        valid = entry.get("valid")
        train = entry.get("train")
        if isinstance(valid, dict) and (ce := _finite(valid.get("ce"))) is not None:
            last_valid = ce
            valid_curve.append(ce)
            if best_ce is None or ce < best_ce:
                best_ce, best_epoch = ce, epoch
        if isinstance(train, dict) and (ce := _finite(train.get("ce"))) is not None:
            last_train = ce
    return HistorySummary(len(history), best_ce, best_epoch, last_valid, last_train, _downsample(valid_curve))


def _downsample(values: list[float], points: int = CURVE_POINTS) -> tuple[float, ...]:
    if len(values) <= points:
        return tuple(round(v, 4) for v in values)
    step = (len(values) - 1) / (points - 1)
    return tuple(round(values[round(i * step)], 4) for i in range(points))


def history_series(history: list[dict[str, Any]]) -> dict[str, dict[str, list[float | None]]]:
    """Per-stage metric series indexed by epoch-1, with gaps as None (NaN-safe for JSON)."""
    series: dict[str, dict[str, list[float | None]]] = {}
    for index, entry in enumerate(history):
        for stage, metrics in entry.items():
            if not isinstance(metrics, dict):
                continue
            for name, value in metrics.items():
                column = series.setdefault(stage, {}).setdefault(name, [])
                column.extend([None] * (index - len(column)))
                column.append(_finite(value))
    for metrics in series.values():
        for column in metrics.values():
            column.extend([None] * (len(history) - len(column)))
    return series


def list_checkpoints(xp_folder: Path) -> tuple[CheckpointFile, ...]:
    files: list[CheckpointFile] = []
    for path in xp_folder.glob("checkpoint*.th"):
        stem = path.stem.removeprefix("checkpoint").removeprefix("_")
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            CheckpointFile(
                name=path.name,
                epoch=int(stem) if stem.isdigit() else None,
                size_mb=round(stat.st_size / 1e6, 1),
                modified=stat.st_mtime,
            )
        )
    return tuple(sorted(files, key=lambda item: (item.epoch is None, item.epoch or 0, item.name)))


def list_sample_epochs(xp_folder: Path) -> tuple[str, ...]:
    samples = xp_folder / "samples"
    if not samples.is_dir():
        return ()
    names = [entry.name for entry in samples.iterdir() if entry.is_dir()]
    return tuple(sorted(names, key=lambda name: (not name.isdigit(), int(name) if name.isdigit() else 0, name)))


def list_audio(folder: Path, repo_root: Path) -> tuple[AudioFile, ...]:
    if not folder.is_dir():
        return ()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file())
    return tuple(AudioFile(name=str(p.relative_to(folder)), path=_repo_relative(p, repo_root)) for p in files)


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _last_activity(xp_folder: Path) -> float | None:
    times = []
    for name in (HISTORY_FILENAME, "train.log", "solver.log.0"):
        try:
            times.append((xp_folder / name).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


class GuiIndex:
    """Caches the zoo, the catalog built on it and the XP summaries; call ``refresh`` to rescan."""

    def __init__(self, repo_root: Path, *, ttl_seconds: float = 60.0) -> None:
        self.repo_root = repo_root.resolve()
        self.dora_xps = self.repo_root / ".dora" / "xps"
        self.ttl_seconds = ttl_seconds
        self._zoo: Zoo | None = None
        self._catalog: Catalog | None = None
        self._summaries: tuple[XpSummary, ...] = ()
        self._loaded_at = 0.0

    def refresh(self) -> None:
        self._zoo = build_zoo(self.repo_root)
        self._catalog = build_catalog(include_lineage=False, zoo=self._zoo)
        self._summaries = tuple(self._build_summaries(self._zoo, self._catalog))
        self._loaded_at = time.monotonic()

    def _ensure(self) -> Catalog:
        if self._catalog is None or time.monotonic() - self._loaded_at > self.ttl_seconds:
            self.refresh()
        assert self._catalog is not None
        return self._catalog

    @property
    def catalog(self) -> Catalog:
        return self._ensure()

    @property
    def zoo(self) -> Zoo:
        self._ensure()
        assert self._zoo is not None
        return self._zoo

    def summaries(self) -> tuple[XpSummary, ...]:
        self._ensure()
        return self._summaries

    def _build_summaries(self, zoo: Zoo, catalog: Catalog) -> list[XpSummary]:
        generation_counts: dict[str, int] = {}
        for generation in catalog.generations:
            if generation.model_sig:
                generation_counts[generation.model_sig] = generation_counts.get(generation.model_sig, 0) + 1
        now = time.time()
        return [
            self._summarize(model, zoo, now=now, generation_runs=generation_counts.get(model.sig, 0))
            for model in zoo.models
        ]

    def _summarize(self, model: Model, zoo: Zoo, *, now: float, generation_runs: int) -> XpSummary:
        xp_folder = model.folder
        metadata = read_json(xp_folder / TRAIN_METADATA_FILENAME)
        has_metadata = isinstance(metadata, dict)
        md: dict[str, Any] = metadata if isinstance(metadata, dict) else {}
        fingerprint = md.get("dataset_fingerprint") if isinstance(md.get("dataset_fingerprint"), dict) else {}
        history = summarize_history(load_history(xp_folder))
        last_activity = _last_activity(xp_folder)
        train_count = fingerprint.get("train_count")
        init = model.init
        return XpSummary(
            sig=model.sig,
            label=model.label,
            profile=model.recipe,
            workspace=model.dataset,
            dataset_name=fingerprint.get("dataset_name") or md.get("dataset_name"),
            train_count=int(train_count) if isinstance(train_count, int) else None,
            epochs_planned=model.epochs_planned,
            epochs_done=history.epochs_done,
            best_valid_ce=history.best_valid_ce,
            best_valid_epoch=history.best_valid_epoch,
            last_valid_ce=history.last_valid_ce,
            last_train_ce=history.last_train_ce,
            valid_curve=history.valid_curve,
            parents=zoo.parents(model.sig),
            origin=model.origin,
            source_checkpoint=Path(init.checkpoint).name if init is not None and init.checkpoint else None,
            written_at=md.get("written_at"),
            last_activity=last_activity,
            active=last_activity is not None and now - last_activity < ACTIVE_WINDOW_SECONDS,
            has_metadata=has_metadata,
            checkpoint_count=len(model.checkpoints),
            sample_epochs=len(list_sample_epochs(xp_folder)),
            generation_runs=generation_runs,
            experiments=tuple(entry.experiment_id for entry in zoo.experiments_for(model.sig)),
            name=model.name,
            sessions=len(zoo.sessions_for(model.sig)),
            children=zoo.children(model.sig),
            kind=model.kind,
            folder=_repo_relative(model.folder, self.repo_root),
        )

    def xp_folder(self, sig: str) -> Path | None:
        if not sig or "/" in sig or sig.startswith("."):
            return None
        folder = self.dora_xps / sig
        return folder if folder.is_dir() else None

    def xp_detail(self, sig: str) -> dict[str, Any] | None:
        folder = self.xp_folder(sig)
        if folder is None:
            return self._bundle_detail(sig)
        catalog = self._ensure()
        summary = next((item for item in self._summaries if item.sig == sig), None)
        if summary is None:
            self.refresh()
            summary = next((item for item in self._summaries if item.sig == sig), None)
        metadata = read_json(folder / TRAIN_METADATA_FILENAME)
        overrides = _command_overrides(metadata.get("command") if isinstance(metadata, dict) else None)
        parent_overrides = None
        if summary and summary.parents:
            parent_metadata = read_json(self.dora_xps / summary.parents[0] / TRAIN_METADATA_FILENAME)
            if isinstance(parent_metadata, dict):
                parent_overrides = _command_overrides(parent_metadata.get("command"))
        try:
            lineage = lineage_graph_to_dict(
                build_lineage_graph_for_sig(sig, dora_dir=self.repo_root / ".dora", repo_root=self.repo_root)
            )
        except (FileNotFoundError, ValueError):
            lineage = None
        zoo = self.zoo
        children = list(zoo.children(sig))
        experiments = [asdict(e) for e in catalog.experiments if sig in e.xp_sigs]
        streams = [asdict(s) for s in catalog.streams if sig in s.model_sigs]
        sessions = [
            {**session_row(session), "path": _repo_relative(Path(session.path), self.repo_root)}
            for session in zoo.sessions_for(sig)
        ]
        return {
            "summary": asdict(summary) if summary else None,
            "metadata": metadata if isinstance(metadata, dict) else None,
            "overrides": overrides,
            "override_diff": _override_diff(overrides, parent_overrides) if parent_overrides is not None else None,
            "history": history_series(load_history(folder)),
            "checkpoints": [asdict(c) for c in list_checkpoints(folder)],
            "sample_epochs": list(list_sample_epochs(folder)),
            "lineage": lineage,
            "children": children,
            "generations": [asdict(g) for g in self.generations_for(sig)],
            "experiments": experiments,
            "streams": streams,
            "sessions": sessions,
            "impressions": [asdict(r) for r in load_impressions(self.repo_root / DEFAULT_IMPRESSION_PATH, model=sig)],
            "descendants": list(zoo.descendants(sig)),
        }

    def _bundle_detail(self, sig: str) -> dict[str, Any] | None:
        """The model page for a portable bundle: a single snapshot, no Dora record.

        A bundle has no Hydra config, history or epoch checkpoints, so the Dora-only sections are empty and the
        checkpoint list carries the one ``model.safetensors`` (a directory, not a legacy ``.th`` merge/perturb input).
        """
        model = self.zoo.model(sig)
        if model is None or model.kind != "bundle":
            return None
        summary = next((item for item in self._summaries if item.sig == sig), None)
        if summary is None:
            summary = self._summarize(model, self.zoo, now=time.time(), generation_runs=len(self.generations_for(sig)))
        checkpoints: list[dict[str, Any]] = []
        for ckpt in model.checkpoints:
            try:
                stat = ckpt.path.stat()
            except OSError:
                stat = None
            checkpoints.append(
                asdict(
                    CheckpointFile(
                        name=ckpt.path.name,
                        epoch=ckpt.epoch,
                        size_mb=round(stat.st_size / 1e6, 1) if stat else 0.0,
                        modified=stat.st_mtime if stat else 0.0,
                    )
                )
            )
        sessions = [
            {**session_row(session), "path": _repo_relative(Path(session.path), self.repo_root)}
            for session in self.zoo.sessions_for(sig)
        ]
        return {
            "summary": asdict(summary) if summary else None,
            "metadata": None,
            "overrides": {},
            "override_diff": None,
            "history": {},
            "checkpoints": checkpoints,
            "sample_epochs": [],
            "lineage": None,
            "children": list(self.zoo.children(sig)),
            "generations": [asdict(g) for g in self.generations_for(sig)],
            "experiments": [],
            "streams": [],
            "sessions": sessions,
            "impressions": [asdict(r) for r in load_impressions(self.repo_root / DEFAULT_IMPRESSION_PATH, model=sig)],
            "descendants": list(self.zoo.descendants(sig)),
        }

    def samples(self, sig: str, epoch: str) -> tuple[AudioFile, ...] | None:
        folder = self.xp_folder(sig)
        if folder is None or not epoch or "/" in epoch or epoch.startswith("."):
            return None
        return list_audio(folder / "samples" / epoch, self.repo_root)

    def generations_for(self, sig: str | None = None) -> tuple[GenerationRun, ...]:
        runs = []
        for entry in self._ensure().generations:
            if sig is not None and entry.model_sig != sig:
                continue
            metadata_path = Path(entry.metadata_path)
            if not metadata_path.is_absolute():
                metadata_path = self.repo_root / metadata_path
            runs.append(
                GenerationRun(
                    run_id=entry.run_id,
                    mode=entry.mode,
                    prefix=entry.prefix,
                    checkpoint=Path(entry.checkpoint_path).name if entry.checkpoint_path else None,
                    duration_seconds=entry.duration_seconds,
                    temperature=entry.temperature,
                    top_k=entry.top_k,
                    seed=entry.seed,
                    written_at=entry.written_at,
                    listening_summary=entry.listening_summary,
                    listening_score=entry.listening_score,
                    files=self._generation_files(metadata_path),
                    metadata_path=_repo_relative(metadata_path, self.repo_root),
                )
            )
        return tuple(sorted(runs, key=lambda run: run.run_id, reverse=True))

    def _generation_files(self, metadata_path: Path) -> tuple[AudioFile, ...]:
        payload = read_json(metadata_path)
        if not isinstance(payload, dict):
            return ()
        files = []
        for value in payload.get("files", []):
            if not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = metadata_path.parent / path
            if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file():
                files.append(AudioFile(name=path.name, path=_repo_relative(path, self.repo_root)))
        return tuple(files)

    def experiments(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._ensure().experiments]

    def resolve_file(self, relative: str) -> Path | None:
        """Resolve a repo-relative path to a servable audio/text file, refusing anything outside the repo."""
        if not relative or "\x00" in relative:
            return None
        candidate = (self.repo_root / relative).resolve()
        if not candidate.is_relative_to(self.repo_root) or not candidate.is_file():
            return None
        if candidate.suffix.lower() not in SERVED_AUDIO_SUFFIXES | TEXT_SUFFIXES:
            return None
        return candidate


def _command_overrides(command: Any) -> dict[str, str]:
    """Effective ``key=value`` overrides of a Dora command; later duplicates win, as in Hydra."""
    overrides: dict[str, str] = {}
    for part in command if isinstance(command, list) else ():
        if isinstance(part, str) and "=" in part and not part.startswith("-"):
            key, value = part.split("=", 1)
            overrides[key] = value
    return overrides


def _override_diff(child: dict[str, str], parent: dict[str, str]) -> dict[str, Any]:
    return {
        "removed": {key: parent[key] for key in parent if key not in child},
        "added": {key: child[key] for key in child if key not in parent},
        "changed": {key: [parent[key], child[key]] for key in child if key in parent and parent[key] != child[key]},
    }


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    return value
