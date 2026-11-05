from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.judging.ab import AbCandidate, AbPair, run_ab_session
from macchina_ricordi.judging.ab_context import (
    ContextAudioError,
    ContextAudioPreparer,
    FreshAudioPreparer,
    read_pcm_frames,
)
from macchina_ricordi.judging.preference import load_preferences


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate: int = 1000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _pair(tmp_path: Path, *, context_seconds: float = 4.0) -> AbPair:
    context_paths: list[str] = []
    for index in range(5):
        path = tmp_path / f"context_{index}.wav"
        _write_wav(path, np.full(6000, (index + 1) * 1000, dtype=np.int16))
        context_paths.append(str(path))
    first_path = tmp_path / "candidate_a.wav"
    second_path = tmp_path / "candidate_b.wav"
    _write_wav(first_path, np.arange(6000, dtype=np.int16))
    _write_wav(second_path, np.full(6000, -2000, dtype=np.int16))
    return AbPair(
        pair_id="pair_000",
        first=AbCandidate(
            str(first_path),
            "low",
            session_id="session",
            session_position=8,
            prompt_duration_seconds=2.0,
        ),
        second=AbCandidate(
            str(second_path),
            "high",
            session_id="session",
            session_position=10,
            prompt_duration_seconds=2.0,
        ),
        context_paths=tuple(context_paths),
        context_prompt_durations_seconds=(1.0,) * 5,
        context_excerpt_seconds=context_seconds,
    )


def test_context_audio_trims_candidates_and_builds_ordered_faded_reel(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    preparer = ContextAudioPreparer()
    temporary_directory = preparer.directory

    with preparer:
        prepared = preparer.prepare([pair])[0]
        fmt, fresh_bytes, fresh_frames = read_pcm_frames(prepared.first.playable_path)
        _, reel_bytes, reel_frames = read_pcm_frames(prepared.context_reel_path)
        fresh = np.frombuffer(fresh_bytes, dtype="<i2")
        reel = np.frombuffer(reel_bytes, dtype="<i2")

        assert fresh_frames == 4000
        assert fresh[0] == 2000
        assert reel_frames == 20_000
        assert fmt.sample_rate == 1000 and fmt.channels == 1 and fmt.sample_width == 2
        for index in range(5):
            start = index * 4000
            assert reel[start] == 0
            assert reel[start + 100] == (index + 1) * 1000
            assert reel[start + 3999] == 0

    assert not temporary_directory.exists()


def test_context_audio_cleanup_runs_after_preparation_failure(tmp_path: Path) -> None:
    pair = _pair(tmp_path, context_seconds=7.0)
    preparer = ContextAudioPreparer()
    temporary_directory = preparer.directory

    with pytest.raises(ContextAudioError, match="needs 7.000s"):
        with preparer:
            preparer.prepare([pair])

    assert not temporary_directory.exists()


def test_fresh_audio_preparer_trims_and_cleans_single_clip(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, np.arange(5000, dtype=np.int16))
    preparer = FreshAudioPreparer()
    temporary_directory = preparer.directory

    with preparer:
        prepared = preparer.prepare_fresh_wav(source, 2.0)
        _, frames, count = read_pcm_frames(prepared)
        assert count == 3000
        assert np.frombuffer(frames, dtype="<i2")[0] == 2000

    assert not temporary_directory.exists()


def test_line_context_session_plays_temporaries_but_persists_only_sources(tmp_path: Path) -> None:
    pair = _pair(tmp_path)
    store = tmp_path / "preferences.jsonl"
    calls: list[str] = []
    answers = iter(["c", "r", "a"])
    preparer = ContextAudioPreparer()
    temporary_directory = preparer.directory

    with preparer:
        prepared = preparer.prepare([pair])[0]
        records = run_ab_session(
            [prepared],
            store_path=store,
            question="newer",
            session_id="context-line",
            seconds=2.0,
            play_fn=lambda path: calls.append(str(path)),
            input_fn=lambda _prompt: next(answers),
            print_fn=lambda *_args: None,
        )
        assert calls == [
            prepared.context_reel_path,
            prepared.first.playable_path,
            prepared.second.playable_path,
            prepared.context_reel_path,
            prepared.context_reel_path,
            prepared.first.playable_path,
            prepared.second.playable_path,
        ]

    record = load_preferences(store)[0]
    sidecar = json.loads((tmp_path / "sessions" / "context-line_session.json").read_text())
    serialized = store.read_text() + json.dumps(sidecar)
    assert len(records) == 1 and records[0].winner == record.winner
    assert record.path_a == pair.first.path and record.path_b == pair.second.path
    assert record.context_paths == list(pair.context_paths)
    assert record.context["context_protocol"] == "shared_memory.v1"
    assert record.context["shared_round_count"] == 5
    assert str(temporary_directory) not in serialized


@pytest.mark.parametrize("answer", ["q", "interrupt"])
def test_context_audio_cleanup_runs_after_quit_or_interrupt(tmp_path: Path, answer: str) -> None:
    preparer = ContextAudioPreparer()
    temporary_directory = preparer.directory

    def respond(_prompt: str) -> str:
        if answer == "interrupt":
            raise KeyboardInterrupt
        return answer

    if answer == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            with preparer:
                run_ab_session(
                    preparer.prepare([_pair(tmp_path)]),
                    store_path=tmp_path / "preferences.jsonl",
                    question="newer",
                    play_fn=lambda _path: None,
                    input_fn=respond,
                    print_fn=lambda *_args: None,
                )
    else:
        with preparer:
            assert (
                run_ab_session(
                    preparer.prepare([_pair(tmp_path)]),
                    store_path=tmp_path / "preferences.jsonl",
                    question="newer",
                    play_fn=lambda _path: None,
                    input_fn=respond,
                    print_fn=lambda *_args: None,
                )
                == []
            )

    assert not temporary_directory.exists()
