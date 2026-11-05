from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

SPARKLINE_BARS = "▁▂▃▄▅▆▇█"
AGGREGATE_CHOICES = ("mean", "median", "p90", "max", "min")


@dataclass(frozen=True)
class ScoreRow:
    epoch: int
    sample_id: str
    wav_path: str
    metric_value: float | None
    score_error: str | None


@dataclass(frozen=True)
class EpochAggregate:
    epoch: int
    n: int
    value: float
    min_value: float
    max_value: float


def load_score_rows(scores_path: Path, *, metric: str = "score") -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    if not scores_path.exists():
        return rows
    with scores_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            epoch = payload.get("epoch")
            sample_id = payload.get("sample_id")
            wav_path = payload.get("wav_path", "")
            score_error = payload.get("score_error")
            metric_value: float | None
            try:
                raw_metric = payload.get(metric)
                metric_value = float(raw_metric) if raw_metric is not None else None
            except (TypeError, ValueError):
                metric_value = None
            if not isinstance(epoch, int) or not isinstance(sample_id, str):
                continue
            rows.append(
                ScoreRow(
                    epoch=epoch,
                    sample_id=sample_id,
                    wav_path=str(wav_path),
                    metric_value=metric_value,
                    score_error=score_error if isinstance(score_error, str) else None,
                )
            )
    return rows


def aggregate_by_epoch(
    rows: Sequence[ScoreRow],
    *,
    aggregate: str = "median",
) -> list[EpochAggregate]:
    if aggregate not in AGGREGATE_CHOICES:
        raise ValueError(f"Unsupported aggregate: {aggregate!r}. Expected one of: {', '.join(AGGREGATE_CHOICES)}.")
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.score_error or row.metric_value is None:
            continue
        grouped.setdefault(row.epoch, []).append(row.metric_value)

    points: list[EpochAggregate] = []
    for epoch in sorted(grouped):
        values = grouped[epoch]
        if not values:
            continue
        points.append(
            EpochAggregate(
                epoch=epoch,
                n=len(values),
                value=_apply_aggregate(values, aggregate),
                min_value=min(values),
                max_value=max(values),
            )
        )
    return points


def _apply_aggregate(values: Sequence[float], aggregate: str) -> float:
    if aggregate == "mean":
        return float(sum(values) / len(values))
    if aggregate == "median":
        return float(median(values))
    if aggregate == "max":
        return float(max(values))
    if aggregate == "min":
        return float(min(values))
    if aggregate == "p90":
        return _percentile(values, 90.0)
    raise ValueError(f"Unsupported aggregate: {aggregate!r}.")


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def render_sparkline(values: Sequence[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return SPARKLINE_BARS[0]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return SPARKLINE_BARS[0] * len(values)
    bars = []
    last_index = len(SPARKLINE_BARS) - 1
    for value in values:
        normalized = (value - lo) / span
        index = int(round(normalized * last_index))
        index = max(0, min(last_index, index))
        bars.append(SPARKLINE_BARS[index])
    return "".join(bars)


def format_score_curve_table(
    *,
    sig: str,
    metric: str,
    aggregate: str,
    points: Sequence[EpochAggregate],
    n_total: int,
    n_failed: int,
    include_sparkline: bool = True,
) -> str:
    lines = [
        f"sig: {sig}",
        f"metric: {metric}",
        f"aggregate: {aggregate}",
        f"samples_total: {n_total} (failed: {n_failed})",
        "",
    ]
    if not points:
        lines.append("no scored samples yet")
        return "\n".join(lines)

    header = f"{'epoch':>6}  {'n':>3}  {aggregate:>8}  {'min':>8}  {'max':>8}"
    lines.append(header)
    for point in points:
        lines.append(
            f"{point.epoch:>6}  {point.n:>3}  {point.value:>8.4f}  {point.min_value:>8.4f}  {point.max_value:>8.4f}"
        )
    if include_sparkline:
        lines.append("")
        lines.append(render_sparkline([point.value for point in points]))
    return "\n".join(lines)


def format_score_curve_json(
    *,
    sig: str,
    metric: str,
    aggregate: str,
    points: Sequence[EpochAggregate],
    n_total: int,
    n_failed: int,
) -> str:
    payload: dict[str, Any] = {
        "sig": sig,
        "metric": metric,
        "aggregate": aggregate,
        "samples_total": n_total,
        "samples_failed": n_failed,
        "points": [
            {
                "epoch": point.epoch,
                "n": point.n,
                "value": point.value,
                "min": point.min_value,
                "max": point.max_value,
            }
            for point in points
        ],
    }
    return json.dumps(payload, indent=2)


def summarize_score_rows(rows: Sequence[ScoreRow]) -> tuple[int, int]:
    total = len(rows)
    failed = sum(1 for row in rows if row.score_error)
    return total, failed
