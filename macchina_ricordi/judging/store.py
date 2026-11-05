"""Where verdicts land, and one reader over all of them.

Four kinds of verdict, four append-only logs side by side in ``artifacts/preferences/``:

- ``preferences.jsonl``: blind A/B verdicts (``judge ab``, the GUI's Écoute page), ``preference.py``;
- ``labels.jsonl``: blind single-clip alarm labels (``judge label``), ``label.py``;
- ``witness.jsonl``: a captioner witness's ratings, asked on 1–5 and stored as 0–1 (``judge witness``), this module;
- ``impressions.jsonl``: sighted, gut-feeling impressions of a whole model, a 1–5 rating and/or a note, optionally
  tied to a session (the GUI's model page), this module. Never mixed with the blind verdicts above.

Each log keeps its own schema, because ``judge prefs`` and ``judge labels`` analyse each kind its own way.
``load_verdicts`` reads all four as one list of ``Verdict`` rows, which is what ``judge log`` prints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .label import DEFAULT_LABEL_PATH, load_labels
from .preference import DEFAULT_PREFERENCE_PATH, load_preferences

VERDICTS_DIR = DEFAULT_PREFERENCE_PATH.parent
DEFAULT_WITNESS_PATH = VERDICTS_DIR / "witness.jsonl"
WITNESS_SCHEMA = "macchina_ricordi.judging.witness.v1"
DEFAULT_IMPRESSION_PATH = VERDICTS_DIR / "impressions.jsonl"
IMPRESSION_SCHEMA = "macchina_ricordi.judging.impression.v1"
VERDICT_KINDS = ("ab", "label", "witness", "impression")


@dataclass(frozen=True)
class WitnessRecord:
    """One clip rated by a captioner witness: the four axes, their mean, and how many answers parsed."""

    path: str
    backend: str
    model_id: str = ""
    created_at: str = ""
    score: float = 0.0
    ratings: dict[str, float] = field(default_factory=dict)
    answered: int = 0
    asked: int = 0
    paraphrases: int = 1
    error: str = ""
    answers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WitnessRecord:
        return cls(
            path=str(payload.get("path", "")),
            backend=str(payload.get("backend", "")),
            model_id=str(payload.get("model_id", "")),
            created_at=str(payload.get("created_at", "")),
            score=float(payload.get("score") or 0.0),
            ratings={str(k): float(v) for k, v in (payload.get("ratings") or {}).items()},
            answered=int(payload.get("answered") or 0),
            asked=int(payload.get("asked") or 0),
            paraphrases=int(payload.get("paraphrases") or 1),
            error=str(payload.get("error", "")),
            answers={str(k): [str(a) for a in v] for k, v in (payload.get("answers") or {}).items()},
        )


def append_witness(record: WitnessRecord, *, path: str | Path = DEFAULT_WITNESS_PATH) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": WITNESS_SCHEMA, **record.to_dict()}
    if not payload["created_at"]:
        payload["created_at"] = datetime.now(UTC).isoformat()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return target


def load_witness(path: str | Path = DEFAULT_WITNESS_PATH) -> list[WitnessRecord]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and payload.get("schema") == WITNESS_SCHEMA:
            records.append(WitnessRecord.from_dict(payload))
    return records


@dataclass(frozen=True)
class ImpressionRecord:
    """How playing a model felt: a 1–5 rating, a note, or both; optionally the session it is about."""

    model: str
    rating: int | None = None
    note: str = ""
    session: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.rating is not None and (isinstance(self.rating, bool) or self.rating not in range(1, 6)):
            raise ValueError(f"rating must be 1–5, got {self.rating!r}")
        if not self.model:
            raise ValueError("an impression needs a model")
        if self.rating is None and not self.note.strip():
            raise ValueError("an impression needs a rating, a note, or both")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ImpressionRecord:
        rating = payload.get("rating")
        return cls(
            model=str(payload.get("model", "")),
            rating=None if rating in (None, "") else int(rating),
            note=str(payload.get("note") or ""),
            session=str(payload.get("session") or ""),
            created_at=str(payload.get("created_at", "")),
        )


def append_impression(record: ImpressionRecord, *, path: str | Path = DEFAULT_IMPRESSION_PATH) -> ImpressionRecord:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not record.created_at:
        record = replace(record, created_at=datetime.now(UTC).isoformat())
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": IMPRESSION_SCHEMA, **asdict(record)}) + "\n")
    return record


def load_impressions(path: str | Path = DEFAULT_IMPRESSION_PATH, *, model: str = "") -> list[ImpressionRecord]:
    source = Path(path).expanduser()
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
            if isinstance(payload, Mapping) and payload.get("schema") == IMPRESSION_SCHEMA:
                record = ImpressionRecord.from_dict(payload)
                if not model or record.model == model:
                    records.append(record)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return records


@dataclass(frozen=True)
class Verdict:
    """Any verdict, in one shape: who decided what about which takes."""

    kind: str
    """``ab``, ``label``, ``witness`` or ``impression``."""
    created_at: str
    question: str
    """The A/B or label question id; ``witness`` for a witness rating."""
    paths: tuple[str, ...]
    answer: str
    """``a``, ``b`` or ``tie``; a label's ``yes``/``no``/``unsure``; a witness's mean rating (0–1); an impression's
    rating (1–5, empty for a note alone)."""
    by: str
    """The listener, or the witness backend."""
    session_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "paths": list(self.paths)}


def load_verdicts(
    *,
    preferences_path: str | Path = DEFAULT_PREFERENCE_PATH,
    labels_path: str | Path = DEFAULT_LABEL_PATH,
    witness_path: str | Path = DEFAULT_WITNESS_PATH,
    impressions_path: str | Path = DEFAULT_IMPRESSION_PATH,
    kinds: Iterable[str] = VERDICT_KINDS,
) -> list[Verdict]:
    """Every verdict in the four logs, oldest first."""
    wanted = set(kinds)
    verdicts: list[Verdict] = []
    if "ab" in wanted:
        verdicts += [
            Verdict(
                kind="ab",
                created_at=record.created_at,
                question=record.question,
                paths=(record.path_a, record.path_b),
                answer=record.winner,
                by=record.listener,
                session_id=record.session_id,
                detail={"pair_id": record.pair_id, "notes": record.notes, "seconds_played": record.seconds_played},
            )
            for record in load_preferences(preferences_path)
        ]
    if "label" in wanted:
        verdicts += [
            Verdict(
                kind="label",
                created_at=record.created_at,
                question=record.question,
                paths=(record.path,),
                answer=record.answer,
                by=record.listener,
                session_id=record.session_id,
                detail={"alarm_id": record.alarm_id, "notes": record.notes, "seconds_played": record.seconds_played},
            )
            for record in load_labels(labels_path)
        ]
    if "witness" in wanted:
        verdicts += [
            Verdict(
                kind="witness",
                created_at=record.created_at,
                question="witness",
                paths=(record.path,),
                answer=f"{record.score:.3f}",
                by=record.backend + (f" ({record.model_id})" if record.model_id else ""),
                detail={"ratings": record.ratings, "answered": record.answered, "asked": record.asked}
                | ({"error": record.error} if record.error else {}),
            )
            for record in load_witness(witness_path)
        ]
    if "impression" in wanted:
        verdicts += [
            Verdict(
                kind="impression",
                created_at=record.created_at,
                question=record.model,
                paths=(),
                answer="" if record.rating is None else str(record.rating),
                by="you",
                session_id=record.session,
                detail={"note": record.note},
            )
            for record in load_impressions(impressions_path)
        ]
    verdicts.sort(key=lambda verdict: verdict.created_at)
    return verdicts


def verdict_counts(verdicts: Iterable[Verdict]) -> dict[str, dict[str, int]]:
    """How many verdicts of each kind, per question."""
    counts: dict[str, dict[str, int]] = {}
    for verdict in verdicts:
        by_question = counts.setdefault(verdict.kind, {})
        by_question[verdict.question] = by_question.get(verdict.question, 0) + 1
    return counts
