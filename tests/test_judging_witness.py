from __future__ import annotations

import pytest

from macchina_ricordi.judging.witness import (
    JUDGE_QUESTIONS,
    JudgeComponents,
    JudgeQuestion,
    judge_audio,
    normalize_rating,
    parse_rating,
)


class _FakeBackend:
    """Stands in for a captioner witness; records what it was asked."""

    def __init__(self, replies: list[str] | str = "3", *, fail_on: str = "") -> None:
        self._replies = replies if isinstance(replies, list) else None
        self._single = replies if isinstance(replies, str) else None
        self._fail_on = fail_on
        self.prompts: list[str] = []
        self.calls = 0

    def generate_caption(self, path, *, prompt, max_new_tokens, max_audio_seconds):  # noqa: ANN001
        self.prompts.append(prompt)
        if self._fail_on and self._fail_on in prompt:
            raise RuntimeError("backend exploded")
        reply = self._single if self._single is not None else self._replies[self.calls % len(self._replies)]
        self.calls += 1
        return reply


# ── rating parsing ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4", 4.0),
        ("I would rate this 4 out of 5.", 4.0),
        ("4/5", 4.0),
        ("Rating: 2", 2.0),
        ("My score is 5, because it is compelling.", 5.0),
        ("three", 3.0),
        ("Three out of five.", 3.0),
        ("The answer is 1.", 1.0),
    ],
)
def test_parse_rating_reads_common_phrasings(text, expected):
    assert parse_rating(text) == expected


@pytest.mark.parametrize("text", ["", "I cannot rate this.", "somewhere around 8", "0"])
def test_parse_rating_returns_none_when_absent(text):
    assert parse_rating(text) is None


def test_parse_rating_ignores_out_of_range_numbers():
    assert parse_rating("This is 44 seconds long") is None


def test_normalize_rating_maps_to_unit_interval():
    assert normalize_rating(1.0) == pytest.approx(0.0)
    assert normalize_rating(3.0) == pytest.approx(0.5)
    assert normalize_rating(5.0) == pytest.approx(1.0)


# ── judging ───────────────────────────────────────────────────────────


def test_judge_audio_scores_every_axis(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    backend = _FakeBackend("5")
    components = judge_audio(clip, captioner_backend=backend)
    assert components.judge_score == pytest.approx(1.0)
    assert components.judge_shape == pytest.approx(1.0)
    assert components.judge_coherence == pytest.approx(1.0)
    assert components.judge_intent == pytest.approx(1.0)
    assert components.judge_engagement == pytest.approx(1.0)
    assert components.judge_answered == 4
    assert components.judge_asked == 4
    assert not components.judge_error


def test_paraphrases_ask_two_phrasings_and_average(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    backend = _FakeBackend(["1", "5"])
    components = judge_audio(clip, captioner_backend=backend, paraphrases=2)
    assert backend.calls == 8
    assert len(set(backend.prompts)) == 8
    assert components.judge_shape == pytest.approx(0.5)


def test_questions_never_ask_about_time_or_order():
    """The witness's temporal claims are known-unreliable; the questions must be holistic."""
    banned = ("first", "begin", "opens", "ends", "then", "timestamp", "second half", "order")
    for question in JUDGE_QUESTIONS:
        for prompt in question.prompts:
            lowered = prompt.lower()
            assert not any(word in lowered for word in banned), prompt


def test_unparseable_answers_do_not_score(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    components = judge_audio(clip, captioner_backend=_FakeBackend("I would rather not say."))
    assert components.judge_score == pytest.approx(0.0)
    assert components.judge_answered == 0
    assert components.judge_asked == 4


def test_backend_failure_is_captured_not_raised(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    backend = _FakeBackend("4", fail_on="coherent instrument")
    components = judge_audio(clip, captioner_backend=backend)
    assert components.judge_error
    assert components.judge_coherence == pytest.approx(0.0)
    assert components.judge_shape == pytest.approx(0.75)


def test_missing_backend_is_reported_not_raised(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.touch()

    def _boom(**_kwargs):
        raise RuntimeError("no captioner installed")

    monkeypatch.setattr("macchina_ricordi.judging.witness.create_judge_backend", _boom)
    components = judge_audio(clip)
    assert components.judge_error == "no captioner installed"
    assert components.judge_score == pytest.approx(0.0)


def test_custom_questions_are_respected(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    backend = _FakeBackend("2")
    question = JudgeQuestion(name="shape", prompts=("Rate the shape 1-5.",))
    components = judge_audio(clip, questions=[question], captioner_backend=backend)
    assert backend.calls == 1
    assert components.judge_shape == pytest.approx(0.25)
    assert components.judge_coherence == pytest.approx(0.0)


def test_to_row_drops_prose_and_matches_fieldnames(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.touch()
    components = judge_audio(clip, captioner_backend=_FakeBackend("3"))
    row = components.to_row()
    assert "answers" not in row
    assert set(row) == set(JudgeComponents.fieldnames())
    assert components.answers["shape"], "prose is still kept on the dataclass"
