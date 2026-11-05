"""Parallel re-scoring of existing audio, and per-group summaries.

Re-scoring is how a new metric earns trust: run it over sweeps that have already been
listened to and check whether the ranking it produces matches what the ear said. The work
is librosa/CPU-bound and embarrassingly parallel, so it runs in a process pool.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class RescoreRow:
    path: str
    group: str
    metrics: dict[str, float]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "group": self.group, **self.metrics, "rescore_error": self.error}


def score_one(path: str) -> dict[str, Any]:
    """Arc + structure metrics for one file. Runs in a worker process."""
    from .arc import compute_arc_coherence
    from .structure import compute_structure

    metrics: dict[str, Any] = {}
    error = ""
    try:
        metrics.update(compute_arc_coherence(path).to_dict())
    except Exception as exc:  # noqa: BLE001
        error = f"arc: {exc}"
    try:
        metrics.update(compute_structure(path).to_dict())
    except Exception as exc:  # noqa: BLE001
        error = f"{error}; structure: {exc}" if error else f"structure: {exc}"
    return {"path": path, "metrics": metrics, "error": error}


def rescore_paths(
    items: Sequence[tuple[str, str]],
    *,
    jobs: int = 0,
    progress: Any = None,
    base_metrics: Mapping[str, Mapping[str, float]] | None = None,
) -> list[RescoreRow]:
    """Score ``(path, group)`` items in parallel.

    ``jobs=0`` picks a sensible default; ``jobs=1`` runs inline, which keeps tracebacks
    readable when a metric is misbehaving. ``base_metrics`` carries columns the sweep
    already recorded (the arrhythmic ``score``, sampling parameters) into the output, so
    the result is a complete table for preference analysis rather than one that silently
    lacks whichever metric you wanted to compare against.
    """
    import os

    if not items:
        return []
    worker_count = jobs if jobs > 0 else max(1, min(len(items), (os.cpu_count() or 4)))
    groups = dict(items)

    def _build(payload: Mapping[str, Any]) -> RescoreRow:
        path = str(payload["path"])
        metrics = dict(base_metrics.get(path, {})) if base_metrics else {}
        metrics.update(payload["metrics"])
        return RescoreRow(
            path=path,
            group=groups.get(path, ""),
            metrics=metrics,
            error=str(payload["error"]),
        )

    rows: list[RescoreRow] = []
    if worker_count == 1:
        for path, _group in items:
            rows.append(_build(score_one(path)))
            if progress is not None:
                progress(len(rows), len(items))
        return rows

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(score_one, path) for path, _group in items]
        for future in as_completed(futures):
            rows.append(_build(future.result()))
            if progress is not None:
                progress(len(rows), len(items))
    rows.sort(key=lambda row: (row.group, row.path))
    return rows


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def summarize_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    group_key: str = "group",
    gate_key: str = "silence_gate",
    min_gate: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Mean of each metric per group, over clips that pass the silence gate.

    Filtering here rather than at the call site is deliberate: reading an unfiltered mean
    off a table of generated audio has produced a wrong conclusion in this project more
    than once, because near-silent clips post extreme values on every development axis.
    """
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(group_key, "")), []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(buckets.items()):
        gate_values = [_numeric(row.get(gate_key)) for row in group_rows]
        usable = [row for row, gate in zip(group_rows, gate_values, strict=True) if gate is None or gate >= min_gate]
        entry: dict[str, Any] = {
            "clips": len(group_rows),
            "usable": len(usable),
            "dropped_silent": len(group_rows) - len(usable),
        }
        for metric in metrics:
            values = [value for row in usable if (value := _numeric(row.get(metric))) is not None]
            entry[metric] = float(np.mean(values)) if values else None
        summary[group] = entry
    return summary


def rank_groups(
    summary: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> list[tuple[str, float]]:
    ranked = [(group, float(entry[metric])) for group, entry in summary.items() if entry.get(metric) is not None]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def read_metric_lookup(
    path: str | Path,
    *,
    path_column: str = "path",
) -> dict[str, dict[str, float]]:
    """Read a scored CSV into ``{path: {metric: value}}`` for preference analysis."""
    source = Path(path).expanduser()
    lookup: dict[str, dict[str, float]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get(path_column) or "").strip()
            if not key:
                continue
            metrics: dict[str, float] = {}
            for column, value in row.items():
                if column == path_column:
                    continue
                number = _numeric(value)
                if number is not None:
                    metrics[column] = number
            lookup[key] = metrics
    return lookup


def iter_sweep_items(
    sweep_dirs: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
) -> list[tuple[str, str]]:
    """``(path, group)`` pairs for every playable take in the given sessions (sweeps, streams, any kind)."""
    from ..sessions.takes import playable_takes

    return [(take.path, take.group) for take in playable_takes(sweep_dirs, group_column=group_column)]


def sweep_base_metrics(
    sweep_dirs: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
) -> dict[str, dict[str, float]]:
    """Numbers the sessions already recorded (a sweep's ``index.csv``, a round's live score), keyed by path."""
    from ..sessions.takes import playable_takes

    return {take.path: dict(take.metrics) for take in playable_takes(sweep_dirs, group_column=group_column)}
