"""Captioner-witness judging — a semantic score orthogonal to every DSP metric.

The arc/structure metrics measure signal statistics. A captioner witness (Audio Flamingo 3
by default) has heard music, so it can answer "does this have a shape" in a way that does
not reduce to chroma total variation. That makes it a genuinely independent axis, useful
exactly where the DSP metrics have already been caught disagreeing with the listener.

Two deliberate constraints:

- **No temporal questions.** This project has established that Music Flamingo's claims
  about order and timing ("the piece opens with…", "at 0:12…") are not reliable. Every
  question here is holistic.
- **Ratings, not prose.** Each question demands a single 1-5 number, which is parsed and
  normalised to ``[0, 1]``. Prose answers are kept in the report for reading, but only the
  number enters the score.

Prompt sensitivity is real, so each axis ships two phrasings; ``paraphrases=2`` asks both
and averages, which is the cheap way to stop a score from being an artefact of wording.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_RATING_SCALE = 5

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

_RATING_PATTERNS = (
    re.compile(r"\b([1-5])\s*(?:/|\s*out\s+of\s*)\s*5\b", re.IGNORECASE),
    re.compile(r"\brating\b[^0-9]{0,12}([1-5])\b", re.IGNORECASE),
    re.compile(r"\b(?:score|rate|rated)\b[^0-9]{0,12}([1-5])\b", re.IGNORECASE),
    # Sentence-final "…is 1." must parse, while "1.5" and "44" must not.
    re.compile(r"(?<![\d.])([1-5])(?!\.?\d)"),
)

_ANSWER_SUFFIX = "Reply with only a single number from 1 to 5 and nothing else."


@dataclass(frozen=True)
class JudgeQuestion:
    name: str
    prompts: tuple[str, ...]


JUDGE_QUESTIONS: tuple[JudgeQuestion, ...] = (
    JudgeQuestion(
        name="shape",
        prompts=(
            "Does this piece of music have an overall shape — does it develop and go "
            "somewhere, instead of staying in one place or wandering aimlessly? "
            "1 means no shape at all, 5 means a clear and satisfying shape. " + _ANSWER_SUFFIX,
            "Rate how strongly this music develops as a whole, from 1 (completely static "
            "or directionless) to 5 (a strong sense of development). " + _ANSWER_SUFFIX,
        ),
    ),
    JudgeQuestion(
        name="coherence",
        prompts=(
            "Does this sound like one coherent instrument or voice, rather than unrelated "
            "sounds stitched together? 1 means completely incoherent, 5 means a single "
            "consistent voice throughout. " + _ANSWER_SUFFIX,
            "Rate the timbral consistency of this audio, from 1 (the sound lurches "
            "between unrelated materials) to 5 (one consistent sound world). " + _ANSWER_SUFFIX,
        ),
    ),
    JudgeQuestion(
        name="intent",
        prompts=(
            "Does this sound deliberate, like a musician making choices, rather than "
            "random noise or a malfunction? 1 means entirely accidental, 5 means clearly "
            "intentional. " + _ANSWER_SUFFIX,
            "Rate how much this sounds like intentional music rather than noise or an "
            "artefact, from 1 (pure noise) to 5 (clearly composed). " + _ANSWER_SUFFIX,
        ),
    ),
    JudgeQuestion(
        name="engagement",
        prompts=(
            "As a listener, how interesting is this to keep listening to? 1 means "
            "completely uninteresting, 5 means compelling. " + _ANSWER_SUFFIX,
            "Rate how much this audio holds your attention, from 1 (dull) to 5 (captivating). " + _ANSWER_SUFFIX,
        ),
    ),
)


@dataclass(frozen=True)
class JudgeComponents:
    judge_score: float = 0.0
    judge_shape: float = 0.0
    judge_coherence: float = 0.0
    judge_intent: float = 0.0
    judge_engagement: float = 0.0
    judge_answered: int = 0
    judge_asked: int = 0
    judge_error: str = ""
    answers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_row(self) -> dict[str, Any]:
        """Flat CSV-friendly view (drops the raw prose)."""
        payload = self.to_dict()
        payload.pop("answers", None)
        return payload

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [item.name for item in fields(cls) if item.name != "answers"]


def parse_rating(text: str) -> float | None:
    """Extract a 1-5 rating from a free-text answer, or ``None`` if there isn't one."""
    if not text:
        return None
    for pattern in _RATING_PATTERNS:
        match = pattern.search(text)
        if match:
            value = int(match.group(1))
            if 1 <= value <= _RATING_SCALE:
                return float(value)
    lowered = text.lower()
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b(?:\s+out\s+of\s+five)?", lowered):
            return float(value)
    return None


def normalize_rating(rating: float) -> float:
    return float((rating - 1.0) / (_RATING_SCALE - 1.0))


def judge_audio(
    audio_path: str | Path,
    *,
    questions: Sequence[JudgeQuestion] = JUDGE_QUESTIONS,
    paraphrases: int = 1,
    captioner_backend: Any = None,
    backend: str = "audio-flamingo-3",
    model_id: str = "",
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "auto",
    max_new_tokens: int = 24,
    max_audio_seconds: float = 120.0,
    trust_remote_code: bool = True,
) -> JudgeComponents:
    """Ask a captioner witness to rate one clip. Never raises — errors land in the result."""
    path = Path(audio_path).expanduser().resolve()
    try:
        resolved_backend = captioner_backend or create_judge_backend(
            backend=backend,
            model_id=model_id,
            device=device,
            dtype=dtype,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001 — the judge is optional, never fatal
        return JudgeComponents(judge_error=str(exc))

    per_axis: dict[str, float] = {}
    answers: dict[str, list[str]] = {}
    asked = 0
    answered = 0
    error = ""

    for question in questions:
        ratings: list[float] = []
        answers[question.name] = []
        for prompt in question.prompts[: max(1, paraphrases)]:
            asked += 1
            try:
                reply = resolved_backend.generate_caption(
                    path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    max_audio_seconds=max_audio_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                error = error or str(exc)
                continue
            answers[question.name].append(reply)
            rating = parse_rating(reply)
            if rating is not None:
                ratings.append(normalize_rating(rating))
                answered += 1
        if ratings:
            per_axis[question.name] = float(sum(ratings) / len(ratings))

    judge_score = float(sum(per_axis.values()) / len(per_axis)) if per_axis else 0.0
    return JudgeComponents(
        judge_score=judge_score,
        judge_shape=per_axis.get("shape", 0.0),
        judge_coherence=per_axis.get("coherence", 0.0),
        judge_intent=per_axis.get("intent", 0.0),
        judge_engagement=per_axis.get("engagement", 0.0),
        judge_answered=answered,
        judge_asked=asked,
        judge_error=error,
        answers=answers,
    )


def create_judge_backend(
    *,
    backend: str = "audio-flamingo-3",
    model_id: str = "",
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "auto",
    trust_remote_code: bool = True,
) -> Any:
    """Build one captioner backend to reuse across many clips (it loads slowly)."""
    from ..listening.captioner import create_captioner_backend

    return create_captioner_backend(
        backend=backend,
        model_id=model_id,
        device=device,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
    )


_BACKEND_KEYS = ("backend", "model_id", "device", "dtype", "quantization", "trust_remote_code")
