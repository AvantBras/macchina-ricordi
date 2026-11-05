"""The question catalog — what a listening session is allowed to ask.

A blind judgement is only interpretable if the question that produced it is recorded
with it, and only poolable if the question is drawn from a fixed set. Free text would
make both impossible: "which is better?" and "which one went somewhere?" are different
experiments, and a ``newer`` verdict says nothing about ``dev_dynamics``.

Each question names the metrics it bears on. ``better`` names none, which means *all* of
them — it is the taste question, the only one the Bradley-Terry fit consumes. Every other
question is an attribute probe: cleaner signal for its own family, useless for the
composite.

Adding a question is adding a row here. No schema change.
"""

from __future__ import annotations

from dataclasses import dataclass

PAIRWISE = "pairwise"
PAIRWISE_CONTEXT = "pairwise_context"
SINGLE = "single"

DEFAULT_QUESTION = "better"


@dataclass(frozen=True)
class Question:
    """One row of the catalog: what is asked, how it is asked, what it validates."""

    id: str
    prompt: str
    type: str
    metrics: tuple[str, ...] = ()
    keys: tuple[str, ...] = ("a", "b", "tie")

    @property
    def validates_everything(self) -> bool:
        """``better`` bears on every metric; an attribute question bears on its family."""
        return not self.metrics

    def validates(self, metric: str) -> bool:
        return self.validates_everything or metric in self.metrics


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="better",
        prompt="which is better?",
        type=PAIRWISE,
    ),
    Question(
        id="newer",
        prompt="relative to the context — which felt more new?",
        type=PAIRWISE_CONTEXT,
        metrics=("novelty", "novelty_raw", "novelty_whitened"),
    ),
    Question(
        id="more_developed",
        prompt="which one went somewhere?",
        type=PAIRWISE,
        metrics=(
            "dev_harmonic",
            "dev_timbre",
            "dev_dynamics",
            "dev_rhythm",
            "dev_top2",
            "structure_score",
            "ssm_novelty",
        ),
    ),
    Question(
        id="harder",
        prompt="which hits harder?",
        type=PAIRWISE,
        metrics=(
            "onset_strength_mean",
            "onset_strength_p90",
            "onset_strength_std",
            "percussive_onset_strength_mean",
            "percussive_onset_strength_p90",
            "percussive_energy_ratio",
            "strong_onset_fraction",
            "spectral_change_mean",
            "spectral_change_p90",
        ),
    ),
    Question(
        id="returns",
        prompt="which came back to earlier material?",
        type=PAIRWISE,
        metrics=("ssm_return", "ssm_mean"),
    ),
    Question(
        id="broke",
        prompt="did this collapse / go static?",
        type=SINGLE,
        metrics=("timbral_coherence",),
        keys=("yes", "no", "unsure"),
    ),
    Question(
        id="silent",
        prompt="is this effectively silence?",
        type=SINGLE,
        metrics=(
            "silence_fraction",
            "tail_silence_fraction",
            "arc_silence_fraction",
            "max_silence_gap_seconds",
        ),
        keys=("yes", "no", "unsure"),
    ),
    Question(
        id="gapped",
        prompt="did this go silent or leave disruptive gaps?",
        type=SINGLE,
        metrics=(
            "temperature",
            "horizon_ratio",
            "top_k_cliff_ratio",
            "arc_silence_fraction",
            "max_silence_gap_seconds",
        ),
        keys=("yes", "no", "unsure"),
    ),
)

_BY_ID = {question.id: question for question in QUESTIONS}


def get_question(question_id: str) -> Question:
    try:
        return _BY_ID[question_id]
    except KeyError:
        known = ", ".join(sorted(_BY_ID))
        raise KeyError(f"unknown question {question_id!r}; known questions: {known}") from None


def question_ids(*, type: str | None = None) -> tuple[str, ...]:
    return tuple(question.id for question in QUESTIONS if type is None or question.type == type)
