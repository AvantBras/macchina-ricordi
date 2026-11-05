from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import macchina_ricordi.judging.ab as ab
from macchina_ricordi.judging.ab import AbCandidate, AbPair
from macchina_ricordi.judging.ab_tui import AbTui
from macchina_ricordi.judging.preference import load_preferences


class FakePlayback:
    player_name = "fake-player"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def play(self, path: str | Path, *, seconds: float | None = None) -> bool:
        self.calls.append(("play", (str(path), seconds)))
        return True

    def play_sequence(self, paths, *, seconds: float | None = None) -> bool:  # type: ignore[no-untyped-def]
        normalized = [
            (str(path.path), path.seconds) if isinstance(path, ab.PlaybackClip) else str(path) for path in paths
        ]
        self.calls.append(("sequence", (normalized, seconds)))
        return True

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def close(self) -> None:
        self.calls.append(("close", None))


class FakeTerminal:
    active = True

    def read_keys(self, _timeout: float) -> list[str]:
        return []


def _pairs() -> list[AbPair]:
    return [
        AbPair(
            "pair_000",
            AbCandidate("a.wav", "left", metrics={"shape": 1.0}),
            AbCandidate("b.wav", "right", metrics={"shape": 0.0}),
        ),
        AbPair(
            "pair_001",
            AbCandidate("c.wav", "left", metrics={"shape": 0.0}),
            AbCandidate("d.wav", "right", metrics={"shape": 1.0}),
        ),
    ]


def test_tui_keeps_playback_interruptible_and_records_a_note(tmp_path: Path) -> None:
    playback = FakePlayback()
    tui = AbTui(
        _pairs(),
        store_path=tmp_path / "preferences.jsonl",
        seconds=8.0,
        session_id="20260810T200000Z",
        stream=io.StringIO(),
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=playback,
        note_reader=lambda _prompt: "felt more open",
    )

    tui.handle_key("1")
    tui.handle_key("2")
    tui.handle_key("m")
    tui.handle_key("a")

    records = load_preferences(tmp_path / "preferences.jsonl")
    assert records[0].winner == "a"
    assert records[0].notes == "felt more open"
    assert records[0].seconds_played == 8.0
    assert ("play", ("a.wav", 8.0)) in playback.calls
    assert ("play", ("b.wav", 8.0)) in playback.calls
    assert tui.current_pair == _pairs()[1]


def test_tui_undo_requeues_the_last_pair_and_removes_only_that_line(tmp_path: Path) -> None:
    store = tmp_path / "preferences.jsonl"
    tui = AbTui(
        _pairs(),
        store_path=store,
        session_id="20260810T200001Z",
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=FakePlayback(),
    )

    tui.handle_key("a")
    tui.handle_key("s")
    assert tui.done is True
    tui.handle_key("z")

    assert tui.done is False
    assert tui.current_pair is not None and tui.current_pair.pair_id == "pair_000"
    assert load_preferences(store) == []
    tui.handle_key("b")
    assert load_preferences(store)[0].winner == "b"


def test_tui_reveal_is_blind_until_after_the_session_and_writes_sidecar(tmp_path: Path) -> None:
    stream = io.StringIO()
    tui = AbTui(
        _pairs()[:1],
        store_path=tmp_path / "preferences.jsonl",
        session_id="20260810T200002Z",
        stream=stream,
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=FakePlayback(),
    )

    frame = tui.frame(width=80)
    assert "a.wav" not in frame and "left" not in frame
    assert "which is better?" in frame
    tui.handle_key("a")
    reveal = tui.reveal()

    assert "left vs right" in reveal
    assert "A" in reveal
    sidecar = tmp_path / "sessions" / "20260810T200002Z_session.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["pairs"][0]["outcome"] == "a"
    assert payload["player"] == "fake-player"


def test_tui_seconds_keys_change_the_recording_setting(tmp_path: Path) -> None:
    tui = AbTui(
        _pairs()[:1],
        store_path=tmp_path / "preferences.jsonl",
        seconds=8.0,
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=FakePlayback(),
    )
    tui.handle_key(",")
    assert tui.seconds == 7.0
    tui.handle_key(".")
    assert tui.seconds == 8.0


def _context_pairs() -> list[AbPair]:
    return [
        AbPair(
            "pair_000",
            AbCandidate(
                "source_a.wav",
                "low",
                session_id="session",
                session_position=8,
                prompt_duration_seconds=2.0,
                playback_path="fresh_a.wav",
            ),
            AbCandidate(
                "source_b.wav",
                "high",
                session_id="session",
                session_position=10,
                prompt_duration_seconds=2.0,
                playback_path="fresh_b.wav",
            ),
            context_paths=tuple(f"source_context_{index}.wav" for index in range(6)),
            context_prompt_durations_seconds=(1.0,) * 6,
            context_excerpt_seconds=3.5,
            context_reel_path="context_reel.wav",
        )
    ]


def test_context_tui_autoplays_full_reel_then_limited_fresh_candidates(tmp_path: Path) -> None:
    playback = FakePlayback()
    tui = AbTui(
        _context_pairs(),
        store_path=tmp_path / "preferences.jsonl",
        question="newer",
        seconds=8.0,
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=playback,
    )

    tui._autoplay_current()
    tui.handle_key("c")
    tui.handle_key("1")
    tui.handle_key("2")

    assert (
        "sequence",
        ([("context_reel.wav", None), ("fresh_a.wav", 8.0), ("fresh_b.wav", 8.0)], None),
    ) in playback.calls
    assert ("sequence", ([("context_reel.wav", None)], None)) in playback.calls
    assert ("play", ("fresh_a.wav", 8.0)) in playback.calls
    assert ("play", ("fresh_b.wav", 8.0)) in playback.calls


def test_context_tui_indicator_and_record_use_runtime_and_source_paths_respectively(tmp_path: Path) -> None:
    class TrackingPlayback(FakePlayback):
        current: Path | None = Path("context_reel.wav")

        def current_path(self) -> Path | None:
            return self.current

    playback = TrackingPlayback()
    store = tmp_path / "preferences.jsonl"
    tui = AbTui(
        _context_pairs(),
        store_path=store,
        question="newer",
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=playback,
    )

    assert "▶ context" in tui.frame()
    playback.current = Path("fresh_b.wav")
    assert "▶ B" in tui.frame()
    tui.handle_key("a")

    record = load_preferences(store)[0]
    assert record.path_a == "source_a.wav" and record.path_b == "source_b.wav"
    assert record.context_paths == [f"source_context_{index}.wav" for index in range(6)]
    assert record.context["round_gap"] == 2


def test_nonblocking_player_terminates_the_previous_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        def __init__(self) -> None:
            self.ended = False

        def poll(self) -> int | None:
            return 0 if self.ended else None

        def terminate(self) -> None:
            self.ended = True

        def kill(self) -> None:
            self.ended = True

        def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
            self.ended = True
            return 0

    processes: list[Process] = []

    def fake_popen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr(ab.subprocess, "Popen", fake_popen)
    player = ab.NonBlockingClipPlayer(player=["ffplay"])
    assert player.play("a.wav", seconds=8.0) is True
    assert player.play("b.wav", seconds=8.0) is True
    assert processes[0].ended is True
    player.close()
