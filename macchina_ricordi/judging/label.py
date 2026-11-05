"""Single-clip alarm labeling, stratified sampling, and detection analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..models.specs import CLEAN_SHARE_EXEMPT, resolve_segment_duration, spec_for
from ..sessions.takes import PlayableTake, numeric, playable_takes
from .questions import SINGLE, get_question

LABEL_SCHEMA = "macchina_ricordi.judging.label.v1"
DEFAULT_LABEL_PATH = Path("artifacts/preferences/labels.jsonl")
DEFAULT_LABEL_SESSIONS_DIRNAME = "sessions"
VALID_ANSWERS = ("yes", "no", "unsure")

Comparator = Literal["below", "at_or_below", "at_or_above"]


@dataclass(frozen=True)
class AlarmStratum:
    label: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = False

    def contains(self, value: float) -> bool:
        if self.lower is not None:
            if value < self.lower or (value == self.lower and not self.lower_inclusive):
                return False
        if self.upper is not None:
            if value > self.upper or (value == self.upper and not self.upper_inclusive):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlarmSpec:
    id: str
    question: str
    predictor: str
    eligibility: str
    comparator: Comparator
    strata: tuple[AlarmStratum, ...]
    reference_thresholds: tuple[float, ...]
    primary_threshold: float
    stream_only: bool = False

    def stratum_index(self, value: float) -> int | None:
        for index, stratum in enumerate(self.strata):
            if stratum.contains(value):
                return index
        return None

    def predicts_yes(self, value: float, threshold: float) -> bool:
        if self.comparator == "below":
            return value < threshold
        if self.comparator == "at_or_below":
            return value <= threshold
        return value >= threshold


ALARMS: tuple[AlarmSpec, ...] = (
    AlarmSpec(
        id="timbral_coherence",
        question="broke",
        predictor="timbral_coherence",
        eligibility="playable stream row or scored sweep row with the predictor",
        comparator="below",
        strata=(
            AlarmStratum("<0.95", upper=0.95),
            AlarmStratum("0.95–0.985", lower=0.95, upper=0.985),
            AlarmStratum("≥0.985", lower=0.985),
        ),
        reference_thresholds=(0.985,),
        primary_threshold=0.985,
    ),
    AlarmSpec(
        id="silence_gate",
        question="silent",
        predictor="arc_silence_fraction",
        eligibility="playable stream row or scored sweep row with the predictor",
        comparator="at_or_above",
        strata=(
            AlarmStratum("<0.30", upper=0.30),
            AlarmStratum("0.30–0.60", lower=0.30, upper=0.60),
            AlarmStratum("≥0.60", lower=0.60),
        ),
        reference_thresholds=(0.30, 0.60),
        primary_threshold=0.60,
    ),
    AlarmSpec(
        id="temperature_floor",
        question="gapped",
        predictor="temperature",
        eligibility="stream model with known clean_share below 0.99",
        comparator="below",
        strata=(
            AlarmStratum("<1.0", upper=1.0),
            AlarmStratum("1.0–1.4", lower=1.0, upper=1.4),
            AlarmStratum("≥1.4", lower=1.4),
        ),
        reference_thresholds=(1.4,),
        primary_threshold=1.4,
        stream_only=True,
    ),
    AlarmSpec(
        id="horizon",
        question="gapped",
        predictor="horizon_ratio",
        eligibility="stream model with a resolvable training horizon",
        comparator="at_or_above",
        strata=(
            AlarmStratum("<0.90", upper=0.90),
            AlarmStratum("0.90–1.0", lower=0.90, upper=1.0),
            AlarmStratum("≥1.0", lower=1.0),
        ),
        reference_thresholds=(0.90, 1.0),
        primary_threshold=0.90,
        stream_only=True,
    ),
    AlarmSpec(
        id="top_k_cliff",
        question="gapped",
        predictor="top_k_cliff_ratio",
        eligibility="stream model with a configured top-k cliff",
        comparator="at_or_below",
        strata=(
            AlarmStratum("≤1.0", upper=1.0, upper_inclusive=True),
            AlarmStratum("1.0–2.0", lower=1.0, upper=2.0, lower_inclusive=False),
            AlarmStratum("≥2.0", lower=2.0),
        ),
        reference_thresholds=(1.0,),
        primary_threshold=1.0,
        stream_only=True,
    ),
)

_ALARMS_BY_ID = {alarm.id: alarm for alarm in ALARMS}


def alarm_ids() -> tuple[str, ...]:
    return tuple(alarm.id for alarm in ALARMS)


def get_alarm(alarm_id: str) -> AlarmSpec:
    try:
        return _ALARMS_BY_ID[alarm_id]
    except KeyError:
        known = ", ".join(alarm_ids())
        raise KeyError(f"unknown alarm {alarm_id!r}; known alarms: {known}") from None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class LabelCandidate:
    path: str
    source_type: Literal["stream", "sweep", "session"]
    audit_values: dict[str, float] = field(default_factory=dict)
    logged_scores: dict[str, Any] = field(default_factory=dict)
    active_control: dict[str, Any] = field(default_factory=dict)
    prompt_duration_seconds: float = 0.0
    source_session_id: str = ""
    source_position: int | None = None
    group: str = ""
    model_id: str = ""
    playback_path: str = field(default="", compare=False, repr=False)

    @property
    def playable_path(self) -> str:
        return self.playback_path or self.path


def _stream_audit_values(take: PlayableTake) -> tuple[dict[str, float], bool]:
    values = numeric(take.scores)
    temperature = _finite_float(take.controls.get("temperature"))
    top_k = _finite_float(take.controls.get("top_k"))
    duration = _finite_float(take.duration_seconds)
    model = str(take.model or take.controls.get("model_xp") or "")
    model_spec = spec_for(model)
    horizon = resolve_segment_duration(model)
    if temperature is not None:
        values["temperature"] = temperature
    if top_k is not None:
        values["top_k"] = top_k
    if duration is not None and horizon is not None and horizon > 0:
        values["horizon_ratio"] = duration / horizon
        values["segment_duration"] = horizon
    if top_k is not None and model_spec is not None and model_spec.top_k_cliff:
        values["top_k_cliff"] = float(model_spec.top_k_cliff)
        values["top_k_cliff_ratio"] = top_k / model_spec.top_k_cliff
    temperature_eligible = bool(
        model_spec is not None and model_spec.clean_share is not None and model_spec.clean_share < CLEAN_SHARE_EXEMPT
    )
    if model_spec is not None and model_spec.clean_share is not None:
        values["model_clean_share"] = model_spec.clean_share
    return values, temperature_eligible


def _label_candidate(take: PlayableTake) -> LabelCandidate:
    if take.kind == "stream":
        values, temperature_eligible = _stream_audit_values(take)
        values["temperature_floor_eligible"] = 1.0 if temperature_eligible else 0.0
        model = Path(str(take.model)).name if take.model else ""
        return LabelCandidate(
            path=take.path,
            source_type="stream",
            audit_values=values,
            logged_scores=dict(take.scores),
            active_control=dict(take.controls),
            prompt_duration_seconds=take.prompt_duration_seconds,
            source_session_id=take.session_id,
            source_position=take.position,
            group=model or take.session_id,
            model_id=model,
        )
    return LabelCandidate(
        path=take.path,
        source_type="sweep" if take.kind == "sweep" else "session",
        audit_values=dict(take.metrics),
        logged_scores={**take.controls, **take.scores},
        source_session_id=take.session_id,
        group=take.group,
        model_id=Path(str(take.model)).name if take.model else "",
    )


def load_stream_label_candidates(session_dirs: Iterable[str | Path]) -> list[LabelCandidate]:
    return [_label_candidate(take) for take in playable_takes(session_dirs) if take.kind == "stream"]


def load_label_candidates(
    sources: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
) -> list[LabelCandidate]:
    """Every playable take of the sources (any session kind, by folder or id; ``sessions.takes``)."""
    return [_label_candidate(take) for take in playable_takes(sources, group_column=group_column)]


def _candidate_is_eligible(candidate: LabelCandidate, alarm: AlarmSpec) -> bool:
    if alarm.stream_only and candidate.source_type != "stream":
        return False
    if alarm.id == "temperature_floor" and candidate.audit_values.get("temperature_floor_eligible") != 1.0:
        return False
    return alarm.predictor in candidate.audit_values


@dataclass(frozen=True)
class LabelSelection:
    candidate: LabelCandidate
    alarm_id: str
    question: str
    stratum_index: int
    stratum_label: str
    stratum_population: int
    stratum_offered: int
    inclusion_probability: float
    sampling_frame_id: str


def label_identity(path: str | Path, question: str, alarm_id: str) -> tuple[str, str, str]:
    return str(path), question, alarm_id


def _sampling_frame_id(alarm: AlarmSpec, buckets: Sequence[Sequence[LabelCandidate]]) -> str:
    payload = {
        "schema": LABEL_SCHEMA,
        "alarm": alarm.id,
        "population": [
            sorted((candidate.path, candidate.audit_values[alarm.predictor]) for candidate in bucket)
            for bucket in buckets
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def select_label_candidates(
    candidates: Sequence[LabelCandidate],
    *,
    alarm_id: str,
    count: int = 30,
    seed: int = 0,
    exclude: Iterable[tuple[str, str, str]] = (),
) -> list[LabelSelection]:
    alarm = get_alarm(alarm_id)
    if count < len(alarm.strata):
        raise ValueError(f"clips must be at least {len(alarm.strata)} for three-stratum coverage")
    excluded = set(exclude)
    buckets: list[list[LabelCandidate]] = [[] for _ in alarm.strata]
    seen_paths: set[str] = set()
    for candidate in candidates:
        if candidate.path in seen_paths:
            continue
        seen_paths.add(candidate.path)
        if not _candidate_is_eligible(candidate, alarm):
            continue
        if label_identity(candidate.path, alarm.question, alarm.id) in excluded:
            continue
        value = candidate.audit_values[alarm.predictor]
        index = alarm.stratum_index(value)
        if index is not None:
            buckets[index].append(candidate)
    empty = [alarm.strata[index].label for index, bucket in enumerate(buckets) if not bucket]
    if empty:
        raise ValueError(f"alarm {alarm.id} has no eligible clips in strata: {', '.join(empty)}")
    total = sum(len(bucket) for bucket in buckets)
    if total < count:
        raise ValueError(f"alarm {alarm.id} has only {total} eligible unlabeled clips; requested {count}")

    target = [count // len(buckets)] * len(buckets)
    for index in range(count % len(buckets)):
        target[index] += 1
    offered = [min(target[index], len(bucket)) for index, bucket in enumerate(buckets)]
    remaining = count - sum(offered)
    while remaining:
        progressed = False
        for index, bucket in enumerate(buckets):
            if offered[index] < len(bucket):
                offered[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError(f"alarm {alarm.id} cannot allocate {count} clips across its strata")

    frame_id = _sampling_frame_id(alarm, buckets)
    rng = random.Random(seed)
    chosen: list[list[LabelCandidate]] = []
    for index, bucket in enumerate(buckets):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        chosen.append(shuffled[: offered[index]])

    selections: list[LabelSelection] = []
    for offset in range(max(offered)):
        for index, members in enumerate(chosen):
            if offset >= len(members):
                continue
            population = len(buckets[index])
            selections.append(
                LabelSelection(
                    candidate=members[offset],
                    alarm_id=alarm.id,
                    question=alarm.question,
                    stratum_index=index,
                    stratum_label=alarm.strata[index].label,
                    stratum_population=population,
                    stratum_offered=offered[index],
                    inclusion_probability=offered[index] / population,
                    sampling_frame_id=frame_id,
                )
            )
    return selections


@dataclass(frozen=True)
class LabelRecord:
    path: str
    alarm_id: str
    question: str
    answer: str
    listener: str = "me"
    created_at: str = ""
    notes: str = ""
    session_id: str = ""
    seconds_played: float | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LabelRecord:
        seconds = _finite_float(payload.get("seconds_played"))
        return cls(
            path=str(payload.get("path", "")),
            alarm_id=str(payload.get("alarm_id", "")),
            question=str(payload.get("question", "")),
            answer=str(payload.get("answer", "")),
            listener=str(payload.get("listener", "")),
            created_at=str(payload.get("created_at", "")),
            notes=str(payload.get("notes", "")),
            session_id=str(payload.get("session_id", "")),
            seconds_played=seconds,
            context=dict(payload.get("context", {}) or {}),
        )


def append_label(record: LabelRecord, *, path: str | Path = DEFAULT_LABEL_PATH) -> Path:
    if record.answer not in VALID_ANSWERS:
        raise ValueError(f"answer must be one of {VALID_ANSWERS}, got {record.answer!r}")
    alarm = get_alarm(record.alarm_id)
    question = get_question(record.question)
    if question.type != SINGLE or alarm.question != question.id:
        raise ValueError(f"alarm {alarm.id!r} requires single-clip question {alarm.question!r}")
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": LABEL_SCHEMA, **record.to_dict()}
    if not payload["created_at"]:
        payload["created_at"] = datetime.now(UTC).isoformat()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return target


def load_labels(path: str | Path = DEFAULT_LABEL_PATH) -> list[LabelRecord]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    records: list[LabelRecord] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, Mapping)
            and payload.get("schema") == LABEL_SCHEMA
            and payload.get("answer") in VALID_ANSWERS
        ):
            records.append(LabelRecord.from_dict(payload))
    return records


def remove_last_label(record: LabelRecord, *, path: str | Path = DEFAULT_LABEL_PATH) -> Path:
    target = Path(path).expanduser()
    if not target.exists():
        raise ValueError(f"label store does not exist: {target}")
    raw = target.read_bytes()
    lines = raw.splitlines(keepends=True)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError(f"label store is empty: {target}")
    try:
        payload = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"last label store line is not valid JSON: {target}") from exc
    matches = all(
        payload.get(key) == value
        for key, value in {
            "path": record.path,
            "alarm_id": record.alarm_id,
            "question": record.question,
            "answer": record.answer,
            "session_id": record.session_id,
        }.items()
    )
    if not matches:
        raise ValueError("can only undo the most recently appended label")
    with target.open("r+b") as handle:
        handle.truncate(sum(len(line) for line in lines[:-1]))
    return target


def labeled_identities(records: Iterable[LabelRecord]) -> set[tuple[str, str, str]]:
    return {label_identity(record.path, record.question, record.alarm_id) for record in records}


def write_label_sidecar(
    payload: Mapping[str, Any],
    *,
    session_id: str,
    store_path: str | Path = DEFAULT_LABEL_PATH,
) -> Path:
    target = Path(store_path).expanduser().parent / DEFAULT_LABEL_SESSIONS_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{session_id}_label_session.json"
    body = {"schema": LABEL_SCHEMA, "session_id": session_id, **dict(payload)}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def label_context(selection: LabelSelection) -> dict[str, Any]:
    candidate = selection.candidate
    return {
        "source_type": candidate.source_type,
        "source_session_id": candidate.source_session_id,
        "source_position": candidate.source_position,
        "group": candidate.group,
        "model_id": candidate.model_id,
        "logged_scores": dict(candidate.logged_scores),
        "active_control": dict(candidate.active_control),
        "playback": "fresh_audio_after_prompt" if candidate.source_type == "stream" else "source_audio",
        "prompt_duration_seconds": candidate.prompt_duration_seconds,
        "audit_values": dict(candidate.audit_values),
        "predictor": get_alarm(selection.alarm_id).predictor,
        "stratum_index": selection.stratum_index,
        "stratum_label": selection.stratum_label,
        "stratum_population": selection.stratum_population,
        "stratum_offered": selection.stratum_offered,
        "inclusion_probability": selection.inclusion_probability,
        "sampling_frame_id": selection.sampling_frame_id,
    }


@dataclass(frozen=True)
class DetectionPoint:
    threshold: float
    raw_tp: int
    raw_fp: int
    raw_tn: int
    raw_fn: int
    weighted_tp: float
    weighted_fp: float
    weighted_tn: float
    weighted_fn: float
    raw_precision: float | None
    raw_recall: float | None
    raw_f1: float | None
    weighted_precision: float | None
    weighted_recall: float | None
    weighted_f1: float | None
    is_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionFrame:
    alarm_id: str
    session_id: str
    sampling_frame_id: str
    labels: int
    decided: int
    unsure: int
    unsure_weight: float
    unsure_weight_share: float
    sufficient: bool
    recommended_threshold: float | None
    points: tuple[DetectionPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "points": [point.to_dict() for point in self.points]}


@dataclass(frozen=True)
class DetectionReport:
    alarm_id: str
    labels: int
    decided: int
    unsure: int
    frames: tuple[DetectionFrame, ...]
    raw_aggregate: DetectionFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LABEL_SCHEMA,
            "alarm_id": self.alarm_id,
            "labels": self.labels,
            "decided": self.decided,
            "unsure": self.unsure,
            "frames": [frame.to_dict() for frame in self.frames],
            "raw_aggregate": self.raw_aggregate.to_dict() if self.raw_aggregate else None,
        }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _scores(tp: float, fp: float, fn: float) -> tuple[float | None, float | None, float | None]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None:
        return precision, recall, None
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _thresholds(values: Sequence[float], alarm: AlarmSpec) -> list[float]:
    unique = sorted(set(values))
    thresholds = set(alarm.reference_thresholds)
    if unique:
        thresholds.add(math.nextafter(unique[0], -math.inf))
        thresholds.add(math.nextafter(unique[-1], math.inf))
        thresholds.update((left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False))
    return sorted(thresholds)


def _point(
    records: Sequence[LabelRecord],
    alarm: AlarmSpec,
    threshold: float,
    *,
    weighted: bool,
) -> DetectionPoint:
    raw = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    totals = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    for record in records:
        if record.answer not in ("yes", "no"):
            continue
        values = record.context.get("audit_values") or {}
        value = _finite_float(values.get(alarm.predictor)) if isinstance(values, Mapping) else None
        if value is None:
            continue
        predicted = alarm.predicts_yes(value, threshold)
        actual = record.answer == "yes"
        key = "tp" if predicted and actual else "fp" if predicted else "fn" if actual else "tn"
        raw[key] += 1
        probability = _finite_float(record.context.get("inclusion_probability")) if weighted else 1.0
        weight = 1.0 / probability if probability is not None and probability > 0 else 1.0
        totals[key] += weight
    raw_precision, raw_recall, raw_f1 = _scores(raw["tp"], raw["fp"], raw["fn"])
    weighted_precision, weighted_recall, weighted_f1 = _scores(totals["tp"], totals["fp"], totals["fn"])
    return DetectionPoint(
        threshold=threshold,
        raw_tp=raw["tp"],
        raw_fp=raw["fp"],
        raw_tn=raw["tn"],
        raw_fn=raw["fn"],
        weighted_tp=totals["tp"],
        weighted_fp=totals["fp"],
        weighted_tn=totals["tn"],
        weighted_fn=totals["fn"],
        raw_precision=raw_precision,
        raw_recall=raw_recall,
        raw_f1=raw_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        is_reference=threshold in alarm.reference_thresholds,
    )


def _frame(
    records: Sequence[LabelRecord],
    alarm: AlarmSpec,
    *,
    min_labels: int,
    weighted: bool,
    session_id: str,
    frame_id: str,
) -> DetectionFrame:
    usable = []
    values: list[float] = []
    for record in records:
        audit_values = record.context.get("audit_values") or {}
        value = _finite_float(audit_values.get(alarm.predictor)) if isinstance(audit_values, Mapping) else None
        if value is not None:
            usable.append(record)
            values.append(value)
    points = tuple(_point(usable, alarm, threshold, weighted=weighted) for threshold in _thresholds(values, alarm))
    decided = [record for record in usable if record.answer in ("yes", "no")]
    answers = {record.answer for record in decided}
    sufficient = len(decided) >= min_labels and answers == {"yes", "no"}
    recommended: float | None = None
    if sufficient:
        scored = [point for point in points if point.weighted_f1 is not None]
        if scored:
            best = max(
                scored,
                key=lambda point: (
                    point.weighted_f1 or 0.0,
                    point.weighted_precision or 0.0,
                    -abs(point.threshold - alarm.primary_threshold),
                ),
            )
            recommended = best.threshold
    unsure_records = [record for record in usable if record.answer == "unsure"]
    unsure_weight = 0.0
    total_weight = 0.0
    for record in usable:
        probability = _finite_float(record.context.get("inclusion_probability")) if weighted else 1.0
        weight = 1.0 / probability if probability is not None and probability > 0 else 1.0
        total_weight += weight
        if record.answer == "unsure":
            unsure_weight += weight
    return DetectionFrame(
        alarm_id=alarm.id,
        session_id=session_id,
        sampling_frame_id=frame_id,
        labels=len(usable),
        decided=len(decided),
        unsure=len(unsure_records),
        unsure_weight=unsure_weight,
        unsure_weight_share=unsure_weight / total_weight if total_weight > 0 else 0.0,
        sufficient=sufficient,
        recommended_threshold=recommended,
        points=points,
    )


def analyze_detection(
    records: Sequence[LabelRecord],
    *,
    alarm_id: str,
    min_labels: int = 8,
    session_id: str | None = None,
) -> DetectionReport:
    if min_labels < 1:
        raise ValueError("min_labels must be at least 1")
    alarm = get_alarm(alarm_id)
    selected = [
        record
        for record in records
        if record.alarm_id == alarm.id and (session_id is None or record.session_id == session_id)
    ]
    grouped: dict[tuple[str, str], list[LabelRecord]] = {}
    for record in selected:
        frame_id = str(record.context.get("sampling_frame_id", ""))
        grouped.setdefault((record.session_id, frame_id), []).append(record)
    frames = tuple(
        _frame(
            members,
            alarm,
            min_labels=min_labels,
            weighted=True,
            session_id=key[0],
            frame_id=key[1],
        )
        for key, members in sorted(grouped.items())
    )
    aggregate = None
    if selected:
        aggregate = replace(
            _frame(
                selected,
                alarm,
                min_labels=min_labels,
                weighted=False,
                session_id="all",
                frame_id="raw_sample_only",
            ),
            sufficient=False,
            recommended_threshold=None,
        )
    return DetectionReport(
        alarm_id=alarm.id,
        labels=len(selected),
        decided=sum(record.answer in ("yes", "no") for record in selected),
        unsure=sum(record.answer == "unsure" for record in selected),
        frames=frames,
        raw_aggregate=aggregate,
    )


def with_playback_path(selection: LabelSelection, path: str) -> LabelSelection:
    return replace(selection, candidate=replace(selection.candidate, playback_path=path))
