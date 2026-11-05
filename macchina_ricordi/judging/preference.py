"""Blind A/B preference store, metric agreement, and composite weight fitting.

Every metric in this project has been hand-designed and hand-weighted, and twice now a
hand-weighted metric has confidently disagreed with the listener: the ``mega-mid`` ranking
(fixed by adding an arc axis) and the 25 Hz stack (fixed by adding the axes in
``structure.py``). The fix for the *next* one is not a cleverer formula — it is ground
truth. This module stores blind pairwise judgements and turns them into two things:

- ``metric_agreement`` — for each metric, how often "higher value" picked the same winner
  the listener did. A metric near 0.5 is noise; below 0.5 is actively inverted.
- ``fit_preference_weights`` — a Bradley-Terry / logistic fit of a composite over metric
  differences, so the weights come from the ear rather than from a guess.

The store is JSONL so a session can be appended to safely and read back partially.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .questions import get_question, question_ids

PREFERENCE_SCHEMA = "macchina_ricordi.judging.preference.v1"
DEFAULT_PREFERENCE_PATH = Path("artifacts/preferences/preferences.jsonl")
DEFAULT_SESSIONS_DIRNAME = "sessions"

VALID_VERDICTS = ("a", "b", "tie")
_EPS = 1e-9


@dataclass(frozen=True)
class PreferenceRecord:
    """One blind verdict. A v0 record read back is a v1 record whose question is ``better``.

    ``question`` is what makes an evening poolable: the store now holds several different
    experiments, and an attribute verdict must never be counted as if it were a preference.
    ``seconds_played`` rides along because a verdict on 8-second excerpts and a verdict on
    whole clips are different data.
    """

    pair_id: str
    path_a: str
    path_b: str
    winner: str
    listener: str = "me"
    created_at: str = ""
    notes: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    question: str = "better"
    context_paths: list[str] = field(default_factory=list)
    session_id: str = ""
    seconds_played: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreferenceRecord:
        seconds = payload.get("seconds_played")
        try:
            seconds_played = None if seconds is None else float(seconds)
        except (TypeError, ValueError):
            seconds_played = None
        return cls(
            pair_id=str(payload.get("pair_id", "")),
            path_a=str(payload.get("path_a", "")),
            path_b=str(payload.get("path_b", "")),
            winner=str(payload.get("winner", "")),
            listener=str(payload.get("listener", "")),
            created_at=str(payload.get("created_at", "")),
            notes=str(payload.get("notes", "")),
            context=dict(payload.get("context", {}) or {}),
            question=str(payload.get("question") or "better"),
            context_paths=[str(item) for item in (payload.get("context_paths") or [])],
            session_id=str(payload.get("session_id", "")),
            seconds_played=seconds_played,
        )


@dataclass(frozen=True)
class MetricAgreement:
    metric: str
    pairs: int
    agreement: float
    z_score: float
    verdict: str
    question: str = "better"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FittedPreferenceModel:
    metrics: list[str]
    weights: list[float]
    scales: list[float]
    pairs: int
    accuracy: float
    log_loss: float
    l2: float
    dropped_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def score(self, values: Mapping[str, float]) -> float:
        total = 0.0
        for metric, weight, scale in zip(self.metrics, self.weights, self.scales, strict=True):
            value = values.get(metric)
            if value is None:
                continue
            total += weight * float(value) / max(scale, _EPS)
        return float(total)


# ── store ─────────────────────────────────────────────────────────────


def pair_key(path_a: str | Path, path_b: str | Path) -> str:
    """Order-independent identity for a pair, so A/B and B/A are the same comparison."""
    first, second = sorted([str(path_a), str(path_b)])
    return f"{first}||{second}"


def append_preference(
    record: PreferenceRecord,
    *,
    path: str | Path = DEFAULT_PREFERENCE_PATH,
) -> Path:
    if record.winner not in VALID_VERDICTS:
        raise ValueError(f"winner must be one of {VALID_VERDICTS}, got {record.winner!r}")
    # A question outside the catalog cannot be pooled or analysed, so it is refused at the
    # door rather than discovered as an unreadable line months later.
    get_question(record.question)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": PREFERENCE_SCHEMA, **record.to_dict()}
    if not payload["created_at"]:
        payload["created_at"] = datetime.now(UTC).isoformat()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return target


def remove_last_preference(
    record: PreferenceRecord,
    *,
    path: str | Path = DEFAULT_PREFERENCE_PATH,
) -> Path:
    """Remove one just-appended record, refusing to rewrite any other store line."""
    target = Path(path).expanduser()
    if not target.exists():
        raise ValueError(f"preference store does not exist: {target}")
    raw = target.read_bytes()
    lines = raw.splitlines(keepends=True)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError(f"preference store is empty: {target}")
    try:
        payload = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"last preference store line is not valid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"last preference store line is not a preference record: {target}")
    matches = all(
        payload.get(field) == value
        for field, value in {
            "pair_id": record.pair_id,
            "path_a": record.path_a,
            "path_b": record.path_b,
            "winner": record.winner,
            "session_id": record.session_id,
        }.items()
    )
    if not matches:
        raise ValueError("can only undo the most recently appended preference")
    new_size = sum(len(line) for line in lines[:-1])
    with target.open("r+b") as handle:
        handle.truncate(new_size)
    return target


def load_preferences(path: str | Path = DEFAULT_PREFERENCE_PATH) -> list[PreferenceRecord]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    records: list[PreferenceRecord] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("winner") in VALID_VERDICTS:
            records.append(PreferenceRecord.from_dict(payload))
    return records


def filter_by_question(
    records: Iterable[PreferenceRecord],
    question: str,
) -> list[PreferenceRecord]:
    return [record for record in records if record.question == question]


def recorded_questions(records: Iterable[PreferenceRecord]) -> list[str]:
    """Questions present in the store, ``better`` first so the main table leads."""
    found = {record.question for record in records}
    ordered = [question for question in question_ids() if question in found]
    return ordered + sorted(found.difference(ordered))


def judged_pair_keys(
    records: Iterable[PreferenceRecord],
    *,
    question: str | None = None,
) -> set[str]:
    """Pairs already judged — scoped to one question when given.

    A pair judged for ``better`` carries no answer to "which one went somewhere?", so an
    attribute session must not treat it as spent. Without ``question`` this is the old
    whole-store behaviour.
    """
    return {
        pair_key(record.path_a, record.path_b) for record in records if question is None or record.question == question
    }


def write_session_sidecar(
    payload: Mapping[str, Any],
    *,
    session_id: str,
    store_path: str | Path = DEFAULT_PREFERENCE_PATH,
) -> Path:
    """Record how a session was assembled, next to the store it appended to.

    The store holds verdicts; this holds the sources, question, seed, targeting and the
    full pair list *including skipped pairs*, which is what makes an evening reproducible
    and a later re-analysis honest about what was presented but not answered.
    """
    target = Path(store_path).expanduser().parent / DEFAULT_SESSIONS_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{session_id}_session.json"
    body = {"schema": PREFERENCE_SCHEMA, "session_id": session_id, **dict(payload)}
    path.write_text(json.dumps(body, indent=2))
    return path


def new_session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ── analysis ──────────────────────────────────────────────────────────


def _decided(records: Sequence[PreferenceRecord]) -> list[PreferenceRecord]:
    return [record for record in records if record.winner in ("a", "b")]


def _normalised_lookup(
    metric_lookup: Mapping[str, Mapping[str, float]],
) -> dict[str, Mapping[str, float]]:
    """Index the scored table under both the path as written and its absolute form.

    A verdict on a stream round is stored under the path the session was given
    (``artifacts/streams/…``) while the scored CSV may hold the absolute one, or the
    reverse. Matching on the raw string alone would report every metric as uncovered and
    look exactly like "no data yet". No filesystem access: ``abspath`` is string work.
    """
    normalised = dict(metric_lookup)
    for key, row in metric_lookup.items():
        normalised.setdefault(os.path.abspath(key), row)
    return normalised


def _lookup(
    metric_lookup: Mapping[str, Mapping[str, float]],
    path: str,
    metric: str,
) -> float | None:
    row = metric_lookup.get(path)
    if row is None:
        row = metric_lookup.get(os.path.abspath(path))
    if row is None:
        return None
    value = row.get(metric)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def metric_agreement(
    records: Sequence[PreferenceRecord],
    metric_lookup: Mapping[str, Mapping[str, float]],
    *,
    metrics: Sequence[str] | None = None,
    min_pairs: int = 8,
    question: str = "better",
) -> list[MetricAgreement]:
    """How often each metric's higher value picks the listener's winner.

    Ties in the listener's verdict are excluded; ties in the metric count as half.
    ``z_score`` is the normal approximation against a coin flip — with the handful of
    pairs an evening of listening produces, treat anything under |z| ≈ 2 as undecided.

    One question at a time, and only the metrics that question bears on. ``better``
    verdicts are pooled over every metric as before; an attribute verdict is pooled over
    its family only, because "which felt more new?" is not evidence about
    ``dev_dynamics`` and mixing the two would produce a number that means neither.
    """
    asked = get_question(question)
    metric_lookup = _normalised_lookup(metric_lookup)
    decided = _decided(filter_by_question(records, question))
    if metrics is None:
        names: set[str] = set()
        for row in metric_lookup.values():
            names.update(row.keys())
        metrics = sorted(names)
    metrics = [metric for metric in metrics if asked.validates(metric)]

    results: list[MetricAgreement] = []
    for metric in metrics:
        wins = 0.0
        total = 0
        for record in decided:
            value_a = _lookup(metric_lookup, record.path_a, metric)
            value_b = _lookup(metric_lookup, record.path_b, metric)
            if value_a is None or value_b is None:
                continue
            total += 1
            if abs(value_a - value_b) < _EPS:
                wins += 0.5
            elif (value_a > value_b) == (record.winner == "a"):
                wins += 1.0
        if total == 0:
            continue
        agreement = wins / total
        z_score = float((agreement - 0.5) * 2.0 * np.sqrt(total))
        if total < min_pairs:
            verdict = "insufficient"
        elif abs(z_score) < 2.0:
            verdict = "undecided"
        elif z_score > 0:
            verdict = "agrees"
        else:
            verdict = "inverted"
        results.append(
            MetricAgreement(
                metric=metric,
                pairs=total,
                agreement=float(agreement),
                z_score=z_score,
                verdict=verdict,
                question=question,
            )
        )
    results.sort(key=lambda item: abs(item.z_score), reverse=True)
    return results


def session_agreement(
    session_records: Sequence[PreferenceRecord],
    store_records: Sequence[PreferenceRecord],
    metric_lookup: Mapping[str, Mapping[str, float]],
    *,
    question: str,
    metrics: Sequence[str],
    testing: str | None = None,
) -> list[tuple[MetricAgreement, MetricAgreement | None]]:
    """Each metric's agreement over one session, beside the store's (whose verdict needs ``min_pairs``).

    A ``metric`` session is pooled only with other ``metric`` verdicts: a verdict across models or knobs credits
    the metric with whatever else differed between the clips.
    """
    if testing == "metric":
        store_records = [record for record in store_records if record.context.get("testing") == "metric"]
    session = metric_agreement(session_records, metric_lookup, metrics=metrics, min_pairs=0, question=question)
    store = metric_agreement(store_records, metric_lookup, metrics=metrics, question=question)
    by_metric = {item.metric: item for item in store}
    return [(item, by_metric.get(item.metric)) for item in session]


def _design_matrix(
    records: Sequence[PreferenceRecord],
    metric_lookup: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for record in records:
        values_a = [_lookup(metric_lookup, record.path_a, metric) for metric in metrics]
        values_b = [_lookup(metric_lookup, record.path_b, metric) for metric in metrics]
        if any(value is None for value in values_a) or any(value is None for value in values_b):
            continue
        rows.append([float(a) - float(b) for a, b in zip(values_a, values_b, strict=True)])
        targets.append(1.0 if record.winner == "a" else 0.0 if record.winner == "b" else 0.5)
    if not rows:
        return np.zeros((0, len(metrics))), np.zeros(0)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _metric_scales(
    metric_lookup: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
) -> np.ndarray:
    scales: list[float] = []
    for metric in metrics:
        values = [value for path in metric_lookup if (value := _lookup(metric_lookup, path, metric)) is not None]
        std = float(np.std(values)) if len(values) > 1 else 0.0
        scales.append(std if std > _EPS else 1.0)
    return np.asarray(scales, dtype=np.float64)


def _metric_coverage(
    records: Sequence[PreferenceRecord],
    metric_lookup: Mapping[str, Mapping[str, float]],
    metric: str,
) -> int:
    return sum(
        1
        for record in records
        if _lookup(metric_lookup, record.path_a, metric) is not None
        and _lookup(metric_lookup, record.path_b, metric) is not None
    )


def fit_preference_weights(
    records: Sequence[PreferenceRecord],
    metric_lookup: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
    *,
    l2: float = 1.0,
    iterations: int = 50,
    coverage_ratio: float = 0.8,
) -> FittedPreferenceModel:
    """Fit ``P(a wins) = sigmoid(w · (x_a - x_b))`` by regularised IRLS.

    Metric differences are divided by each metric's spread across the candidate pool, so
    the fitted weights are directly comparable to one another. ``l2`` is not optional in
    practice: with a few dozen pairs the fit separates easily and unregularised weights
    run away.

    The design matrix needs every metric present on both sides of a pair, so one metric
    that is missing from the scored CSV would otherwise silently drop *every* row and
    return a zeroed model. Metrics covering less than ``coverage_ratio`` of the
    best-covered metric's pairs are dropped instead, and reported in ``dropped_metrics``.

    Feed this ``better`` records only (``filter_by_question``). Attribute questions
    validate axes; they do not define taste, and a ``harder`` verdict in the design
    matrix would fit "hits harder" into the composite as if it were preference.
    """
    metrics = list(metrics)
    metric_lookup = _normalised_lookup(metric_lookup)
    coverage = {metric: _metric_coverage(records, metric_lookup, metric) for metric in metrics}
    best_coverage = max(coverage.values(), default=0)
    if best_coverage > 0:
        threshold = coverage_ratio * best_coverage
        kept = [metric for metric in metrics if coverage[metric] >= threshold]
        dropped = [metric for metric in metrics if coverage[metric] < threshold]
    else:
        kept, dropped = metrics, []

    scales = _metric_scales(metric_lookup, kept)
    design, targets = _design_matrix(records, metric_lookup, kept)
    if design.shape[0] == 0:
        return FittedPreferenceModel(
            metrics=kept,
            weights=[0.0] * len(kept),
            scales=[float(scale) for scale in scales],
            pairs=0,
            accuracy=0.0,
            log_loss=float("nan"),
            l2=float(l2),
            dropped_metrics=dropped,
        )
    metrics = kept

    design = design / np.maximum(scales, _EPS)
    n_pairs, n_metrics = design.shape
    weights = np.zeros(n_metrics, dtype=np.float64)
    identity = np.eye(n_metrics, dtype=np.float64)

    for _ in range(iterations):
        logits = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = design.T @ (probabilities - targets) / n_pairs + l2 * weights
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        hessian = (design.T * variance) @ design / n_pairs + l2 * identity
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights = weights - step
        if float(np.max(np.abs(step))) < 1e-8:
            break

    logits = design @ weights
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    decided_mask = np.abs(targets - 0.5) > _EPS
    if decided_mask.any():
        predicted = probabilities[decided_mask] > 0.5
        actual = targets[decided_mask] > 0.5
        accuracy = float(np.mean(predicted == actual))
    else:
        accuracy = 0.0
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    log_loss = float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))

    return FittedPreferenceModel(
        metrics=metrics,
        weights=[float(value) for value in weights],
        scales=[float(scale) for scale in scales],
        pairs=int(n_pairs),
        accuracy=accuracy,
        log_loss=log_loss,
        l2=float(l2),
        dropped_metrics=dropped,
    )


# ── composite weights ─────────────────────────────────────────────────

_STRUCTURE_METRIC_KEYS = {
    "dev_top2": "development",
    "ssm_novelty": "novelty",
    "ssm_return": "return",
}


def structure_weights_from_model(model: FittedPreferenceModel) -> dict[str, float]:
    """Project a fitted model onto the ``structure_score`` composite weights.

    Negative fitted weights are clipped to zero rather than kept: a negative weight means
    the axis is anti-correlated with preference, which is a finding to investigate, not
    something to silently bake into a score that is supposed to reward development.
    """
    raw: dict[str, float] = {}
    for metric, weight in zip(model.metrics, model.weights, strict=True):
        key = _STRUCTURE_METRIC_KEYS.get(metric)
        if key is not None:
            raw[key] = max(0.0, float(weight))
    total = sum(raw.values())
    if total <= _EPS:
        from ..scoring.structure import DEFAULT_STRUCTURE_WEIGHTS

        return dict(DEFAULT_STRUCTURE_WEIGHTS)
    return {key: value / total for key, value in raw.items()}


def save_preference_model(
    model: FittedPreferenceModel,
    *,
    path: str | Path,
    structure_weights: Mapping[str, float] | None = None,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": PREFERENCE_SCHEMA,
        "fitted_at": datetime.now(UTC).isoformat(),
        "model": model.to_dict(),
        "structure_weights": dict(structure_weights or structure_weights_from_model(model)),
    }
    target.write_text(json.dumps(payload, indent=2))
    return target
