"""Tidy-dataset loader over stream session logs.

A stream session writes one JSON object per round to ``performance.jsonl``, with the
control snapshot that produced the round nested under ``active_control``, the scorer
payload under ``score``, and the best-of-N candidate table under ``score.candidates``.
That shape is convenient to append to and awkward to analyse. This module flattens a
set of sessions into three long tables — rounds, candidates, events — that every
statistical phase of the stream investigation reads from.

Two things it deliberately does *not* do: it does not drop or impute missing
indicators (coverage is a finding, so it is reported instead), and it does not
normalise duration-dependent fields in place (``onset_count`` scales with
``duration_seconds``; the derived ``*_per_second`` columns are added alongside, never
substituted).

Comparability caveats the caller must respect are surfaced as columns rather than
enforced: ``novelty_calibration_path`` (novelty is only comparable within one
calibration) and ``candidates`` (indicator distributions are censored by best-of-N
selection whenever it exceeds 1).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "SessionMeta",
    "StreamDataset",
    "CONTROL_FIELDS",
    "DURATION_DEPENDENT_FIELDS",
    "discover_sessions",
    "load_session_meta",
    "load_stream_dataset",
    "coverage_table",
    "write_stream_dataset",
]

# Control knobs snapshotted per round. Kept explicit so a schema change in the
# stream worker shows up as a missing column rather than as silently absent data.
CONTROL_FIELDS = (
    "command",
    "pin_target",
    "tau_seconds",
    "completion_duration_seconds",
    "round_duration_seconds",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "cfg_coef",
    "extend_stride",
    "model_xp",
    "candidates",
    "novelty_weight",
    "stuck_weight",
    "stuck_kick_rounds",
    "motif_dir",
    "motif_period",
    "motif_mix",
    "tail_rescue",
)

# Indicators that scale with the scored window and must be normalised or conditioned
# on ``duration_seconds`` before being compared across rounds.
DURATION_DEPENDENT_FIELDS = ("onset_count",)

_MIN_SCORED_ROUNDS = 5


@dataclass(frozen=True)
class SessionMeta:
    """Session-level facts drawn from ``manifest.json`` plus a row scan."""

    run_id: str
    path: Path
    rounds_written: int
    scored_rounds: int
    novelty_rounds: int
    seed_mode: str | None
    started_at: str | None
    finished_at: str | None
    stopped_by: str | None
    manifest_xp: str | None
    row_xps: tuple[str, ...]
    novelty_calibration_path: str | None
    novelty_calibration_dims: int | None
    has_manifest: bool

    @property
    def model_xp(self) -> str | None:
        """The session's model, resolved from rows when the manifest lacks one."""
        if self.manifest_xp:
            return self.manifest_xp
        if len(self.row_xps) == 1:
            return self.row_xps[0]
        return None


@dataclass(frozen=True)
class StreamDataset:
    """Three aligned long tables plus the session metadata they were built from.

    ``rounds`` is one row per scored round: identity, control snapshot (``ctl_*``),
    indicators (``score`` payload, flattened at top level), and derived helpers.
    ``candidates`` is one row per best-of-N take, joinable on ``(run_id, round_idx)``.
    ``events`` is the ``events.jsonl`` sidecar, joinable on ``(run_id, round_idx)``.
    """

    rounds: pd.DataFrame
    candidates: pd.DataFrame
    events: pd.DataFrame
    sessions: tuple[SessionMeta, ...]

    @property
    def indicator_columns(self) -> tuple[str, ...]:
        skip = {"candidates", "duration_seconds"}
        return tuple(
            c
            for c in self.rounds.columns
            if not c.startswith(("ctl_", "meta_", "d_"))
            and c not in skip
            and c not in _IDENTITY_COLUMNS
            and pd.api.types.is_numeric_dtype(self.rounds[c])
        )

    @property
    def control_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.rounds.columns if c.startswith("ctl_"))


_IDENTITY_COLUMNS = (
    "run_id",
    "round_idx",
    "take_idx",
    "timestamp",
    "model",
    "model_xp",
    "round_wav",
    "command_consumed",
    "seed_used",
    "elapsed_seconds",
    "playback_seconds",
    "buffer_seconds_ahead",
    "prompt_start_seconds",
    "prompt_duration_seconds",
    "prompt_audio",
    "motif_source",
    "motif_injection_weight",
    "tail_rescued_from_seconds",
)


def _short_xp(value: object) -> str | None:
    """Reduce an xp path (absolute or relative) to its signature."""
    if value in (None, ""):
        return None
    return Path(str(value)).name or None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_session_meta(session_dir: Path) -> SessionMeta:
    """Read one session's manifest and scan its rows for model / coverage facts."""
    manifest_path = session_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    calibration = manifest.get("novelty_calibration") or {}
    rows = _read_jsonl(session_dir / "performance.jsonl")

    scored = 0
    novelty = 0
    xps: set[str] = set()
    for row in rows:
        score = row.get("score") or {}
        if "onset_density" in score:
            scored += 1
        if score.get("novelty_whitened") is not None:
            novelty += 1
        short = _short_xp(row.get("model_xp") or (row.get("active_control") or {}).get("model_xp"))
        if short:
            xps.add(short)

    backend = manifest.get("backend") or {}
    manifest_xp = _short_xp(
        manifest.get("final_xp_path")
        or backend.get("xp_path")
        or (manifest.get("initial_control") or {}).get("model_xp")
    )

    return SessionMeta(
        run_id=manifest.get("run_id") or session_dir.name,
        path=session_dir,
        rounds_written=len(rows),
        scored_rounds=scored,
        novelty_rounds=novelty,
        seed_mode=manifest.get("seed_mode"),
        started_at=manifest.get("started_at"),
        finished_at=manifest.get("finished_at"),
        stopped_by=manifest.get("stopped_by"),
        manifest_xp=manifest_xp,
        row_xps=tuple(sorted(xps)),
        novelty_calibration_path=calibration.get("path"),
        novelty_calibration_dims=calibration.get("dims"),
        has_manifest=manifest_path.exists(),
    )


def discover_sessions(
    streams_root: Path | None = None,
    *,
    min_scored_rounds: int = _MIN_SCORED_ROUNDS,
) -> tuple[SessionMeta, ...]:
    """Find every session under ``streams_root`` carrying usable scored rounds.

    Without a root, every stream session in both homes (``artifacts/streams/`` and the dated
    ``artifacts/sessions/``). Sessions whose rounds were logged without a scorer payload are
    the pre-2026-07-28 corpus; they are excluded here rather than loaded and filtered later,
    so that "in scope" has exactly one definition.
    """
    if streams_root is None:
        from ...sessions.store import stream_session_dirs

        children = stream_session_dirs()
    elif streams_root.exists():
        children = sorted(streams_root.iterdir())
    else:
        return ()
    metas = []
    for child in children:
        if not child.is_dir() or not (child / "performance.jsonl").exists():
            continue
        meta = load_session_meta(child)
        if meta.scored_rounds >= min_scored_rounds:
            metas.append(meta)
    return tuple(metas)


def _round_record(row: dict, meta: SessionMeta) -> dict | None:
    score = dict(row.get("score") or {})
    if "onset_density" not in score:
        return None
    score.pop("candidates", None)

    control = row.get("active_control") or {}
    record: dict = {
        "run_id": meta.run_id,
        "round_idx": row.get("round_idx"),
        "take_idx": row.get("take_idx"),
        "timestamp": row.get("timestamp"),
        "command_consumed": row.get("command_consumed"),
        "round_wav": row.get("round_wav"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "playback_seconds": row.get("playback_seconds"),
        "buffer_seconds_ahead": row.get("buffer_seconds_ahead"),
        "prompt_start_seconds": row.get("prompt_start_seconds"),
        "prompt_duration_seconds": row.get("prompt_duration_seconds"),
        "motif_source": row.get("motif_source"),
        "motif_injection_weight": row.get("motif_injection_weight"),
        "tail_rescued_from_seconds": row.get("tail_rescued_from_seconds"),
    }
    record["model"] = _short_xp(row.get("model_xp") or control.get("model_xp")) or meta.model_xp
    for field in CONTROL_FIELDS:
        value = control.get(field)
        if field == "model_xp":
            value = _short_xp(value)
        elif field == "motif_dir":
            value = None if value in (None, "") else str(value)
        record[f"ctl_{field}"] = value
    record.update(score)

    record["meta_novelty_calibration"] = meta.novelty_calibration_path
    record["meta_seed_mode"] = meta.seed_mode
    return record


def _candidate_records(row: dict, meta: SessionMeta) -> list[dict]:
    score = row.get("score") or {}
    out = []
    for cand in score.get("candidates") or []:
        out.append(
            {
                "run_id": meta.run_id,
                "round_idx": row.get("round_idx"),
                "model": _short_xp(row.get("model_xp") or (row.get("active_control") or {}).get("model_xp"))
                or meta.model_xp,
                "candidate": cand.get("candidate"),
                "selected": bool(cand.get("selected")),
                "silence_fraction": cand.get("silence_fraction"),
                "arc_coherence_score": cand.get("arc_coherence_score"),
                "tail_silence_fraction": cand.get("tail_silence_fraction"),
                "novelty": cand.get("novelty"),
            }
        )
    return out


def _event_records(session_dir: Path, meta: SessionMeta) -> list[dict]:
    out = []
    for event in _read_jsonl(session_dir / "events.jsonl"):
        out.append(
            {
                "run_id": meta.run_id,
                "round_idx": event.get("round_idx"),
                "timestamp": event.get("timestamp"),
                "kind": event.get("kind"),
                "level": event.get("level"),
                "message": event.get("message"),
                "data": json.dumps(event.get("data")) if event.get("data") is not None else None,
            }
        )
    return out


def _add_derived(rounds: pd.DataFrame) -> pd.DataFrame:
    """Append derived columns: duration normalisations and per-session round time."""
    if rounds.empty:
        return rounds
    duration = rounds.get("duration_seconds")
    if duration is not None:
        for field in DURATION_DEPENDENT_FIELDS:
            if field in rounds.columns:
                rounds[f"d_{field}_per_second"] = rounds[field] / duration.replace(0, pd.NA)
        if "max_silence_gap_seconds" in rounds.columns:
            rounds["d_max_silence_gap_fraction"] = rounds["max_silence_gap_seconds"] / duration.replace(0, pd.NA)

    # Sessions before 2026-07-31 logged the whole window as ``round_duration_seconds``;
    # later ones log the generated part as ``completion_duration_seconds``. Reconcile
    # both into one pair of columns so the knob is analysable across the whole corpus.
    tau = pd.to_numeric(rounds["ctl_tau_seconds"], errors="coerce")
    total = pd.to_numeric(rounds["ctl_round_duration_seconds"], errors="coerce")
    completion = pd.to_numeric(rounds["ctl_completion_duration_seconds"], errors="coerce")
    rounds["d_round_duration_seconds"] = total.fillna(tau + completion)
    rounds["d_completion_duration_seconds"] = completion.fillna(total - tau)

    times = pd.to_datetime(rounds["timestamp"], errors="coerce", utc=True)
    rounds["d_wall_seconds"] = (times - times.groupby(rounds["run_id"]).transform("min")).dt.total_seconds()
    rounds["d_round_seq"] = rounds.groupby("run_id").cumcount()
    return rounds


def load_stream_dataset(
    streams_root: Path | str | None = None,
    *,
    sessions: Sequence[str] | None = None,
    min_scored_rounds: int = _MIN_SCORED_ROUNDS,
    metas: Iterable[SessionMeta] | None = None,
) -> StreamDataset:
    """Flatten scored stream sessions into rounds / candidates / events tables.

    ``sessions`` restricts to the given run ids; by default every session with at
    least ``min_scored_rounds`` scored rounds is loaded.
    """
    root = Path(streams_root) if streams_root is not None else None
    selected = tuple(metas) if metas is not None else discover_sessions(root, min_scored_rounds=min_scored_rounds)
    if sessions is not None:
        wanted = set(sessions)
        selected = tuple(m for m in selected if m.run_id in wanted)

    round_rows: list[dict] = []
    candidate_rows: list[dict] = []
    event_rows: list[dict] = []
    for meta in selected:
        raw = _read_jsonl(meta.path / "performance.jsonl")
        for row in raw:
            record = _round_record(row, meta)
            if record is None:
                continue
            round_rows.append(record)
            candidate_rows.extend(_candidate_records(row, meta))
        event_rows.extend(_event_records(meta.path, meta))

    rounds = pd.DataFrame(round_rows)
    if not rounds.empty:
        rounds = rounds.sort_values(["run_id", "round_idx"], ignore_index=True)
        rounds = _add_derived(rounds)
    return StreamDataset(
        rounds=rounds,
        candidates=pd.DataFrame(candidate_rows),
        events=pd.DataFrame(event_rows),
        sessions=selected,
    )


def coverage_table(dataset: StreamDataset) -> pd.DataFrame:
    """Per-field non-null coverage, overall and per session.

    Returned long: one row per (column, run_id) plus an ``__all__`` pseudo-session,
    so a field that exists only in later sessions is visible as such.
    """
    rounds = dataset.rounds
    if rounds.empty:
        return pd.DataFrame(columns=["column", "run_id", "n", "non_null", "coverage", "n_unique"])

    frames = []
    groups: list[tuple[str, pd.DataFrame]] = [("__all__", rounds)]
    groups += [(str(run_id), part) for run_id, part in rounds.groupby("run_id")]
    for run_id, part in groups:
        described = pd.DataFrame(
            {
                "column": part.columns,
                "run_id": run_id,
                "n": len(part),
                "non_null": [int(part[c].notna().sum()) for c in part.columns],
                "n_unique": [int(part[c].nunique(dropna=True)) for c in part.columns],
            }
        )
        frames.append(described)
    out = pd.concat(frames, ignore_index=True)
    out["coverage"] = out["non_null"] / out["n"].replace(0, pd.NA)
    return out


def write_stream_dataset(dataset: StreamDataset, out_dir: Path | str) -> Path:
    """Materialise the tables as CSV plus a ``stream_dataset_status.json`` sidecar."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset.rounds.to_csv(out / "rounds.csv", index=False)
    dataset.candidates.to_csv(out / "candidates.csv", index=False)
    dataset.events.to_csv(out / "events.csv", index=False)
    coverage_table(dataset).to_csv(out / "coverage.csv", index=False)

    status = {
        "schema": "macchina_ricordi.stream_dataset.v1",
        "rounds": int(len(dataset.rounds)),
        "candidates": int(len(dataset.candidates)),
        "events": int(len(dataset.events)),
        "sessions": [
            {
                "run_id": m.run_id,
                "model": m.model_xp,
                "row_xps": list(m.row_xps),
                "rounds_written": m.rounds_written,
                "scored_rounds": m.scored_rounds,
                "novelty_rounds": m.novelty_rounds,
                "seed_mode": m.seed_mode,
                "started_at": m.started_at,
                "finished_at": m.finished_at,
                "stopped_by": m.stopped_by,
                "novelty_calibration": m.novelty_calibration_path,
                "novelty_calibration_dims": m.novelty_calibration_dims,
                "has_manifest": m.has_manifest,
            }
            for m in dataset.sessions
        ],
        "indicator_columns": list(dataset.indicator_columns),
        "control_columns": list(dataset.control_columns),
        "duration_dependent_fields": list(DURATION_DEPENDENT_FIELDS),
    }
    status_path = out / "stream_dataset_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status_path
