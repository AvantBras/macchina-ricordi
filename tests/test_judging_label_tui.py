from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from macchina_ricordi.judging.label import LabelCandidate, LabelSelection, load_labels
from macchina_ricordi.judging.label_session import run_label_session
from macchina_ricordi.judging.label_tui import LabelTui


class FakePlayback:
    player_name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.current: Path | None = None

    def play(self, path: str | Path, *, seconds: float | None = None) -> bool:
        self.current = Path(path)
        self.calls.append(("play", (str(path), seconds)))
        return True

    def stop(self) -> None:
        self.current = None
        self.calls.append(("stop", None))

    def close(self) -> None:
        self.calls.append(("close", None))

    def current_path(self) -> Path | None:
        return self.current


class FakeTerminal:
    active = True

    def read_keys(self, _timeout: float) -> list[str]:
        return []


def _selections() -> list[LabelSelection]:
    selections = []
    for index, value in enumerate((0.90, 0.97, 0.99)):
        selections.append(
            LabelSelection(
                candidate=LabelCandidate(
                    path=f"source_{index}.wav",
                    source_type="stream",
                    audit_values={"timbral_coherence": value},
                    prompt_duration_seconds=2.0,
                    playback_path=f"fresh_{index}.wav",
                ),
                alarm_id="timbral_coherence",
                question="broke",
                stratum_index=index,
                stratum_label=("low", "mid", "high")[index],
                stratum_population=10,
                stratum_offered=1,
                inclusion_probability=0.1,
                sampling_frame_id="frame",
            )
        )
    return selections


def test_label_tui_is_blind_autoplays_records_notes_and_advances(tmp_path: Path) -> None:
    playback = FakePlayback()
    stream = io.StringIO()
    store = tmp_path / "labels.jsonl"
    tui = LabelTui(
        _selections(),
        store_path=store,
        seconds=8.0,
        session_id="label-session",
        stream=stream,
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=playback,
        note_reader=lambda _prompt: "heard a lurch",
    )

    frame = tui.frame()
    assert "source_0" not in frame and "timbral" not in frame and "0.9" not in frame
    tui._autoplay_current()
    tui.handle_key(" ")
    tui.handle_key("m")
    tui.handle_key("y")

    assert ("play", ("fresh_0.wav", 8.0)) in playback.calls
    assert ("play", ("fresh_1.wav", 8.0)) in playback.calls
    record = load_labels(store)[0]
    assert record.path == "source_0.wav" and record.answer == "yes"
    assert record.notes == "heard a lurch"
    assert record.context["playback"] == "fresh_audio_after_prompt"
    assert "fresh_0.wav" not in store.read_text()


def test_label_tui_undo_skip_unsure_seconds_indicator_and_sidecar(tmp_path: Path) -> None:
    playback = FakePlayback()
    store = tmp_path / "labels.jsonl"
    tui = LabelTui(
        _selections(),
        store_path=store,
        seconds=8.0,
        session_id="label-sidecar",
        terminal=FakeTerminal(),  # type: ignore[arg-type]
        playback=playback,
    )
    tui._autoplay_current()
    assert "▶ clip" in tui.frame()
    tui.handle_key(",")
    assert tui.seconds == 7.0
    tui.handle_key("n")
    tui.handle_key("s")
    tui.handle_key("u")
    assert tui.done is True
    tui.handle_key("z")
    assert tui.done is False and tui.current_selection is not None
    assert tui.current_selection.candidate.path == "source_2.wav"
    tui.handle_key("y")
    tui.finalize()

    assert [record.answer for record in load_labels(store)] == ["no", "yes"]
    sidecar = json.loads((tmp_path / "sessions" / "label-sidecar_label_session.json").read_text())
    assert [item["outcome"] for item in sidecar["clips"]] == ["no", "skipped", "yes"]


def test_line_label_session_initial_replay_answers_and_persists_sources(tmp_path: Path) -> None:
    store = tmp_path / "labels.jsonl"
    calls: list[str] = []
    answers = iter(["r", "yes", "no", "unsure"])
    records = run_label_session(
        _selections(),
        store_path=store,
        session_id="line-label",
        play_fn=lambda path: calls.append(str(path)),
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda *_args: None,
    )

    assert calls == ["fresh_0.wav", "fresh_0.wav", "fresh_1.wav", "fresh_2.wav"]
    assert [record.answer for record in records] == ["yes", "no", "unsure"]
    assert [record.path for record in records] == ["source_0.wav", "source_1.wav", "source_2.wav"]
    serialized = store.read_text() + (tmp_path / "sessions" / "line-label_label_session.json").read_text()
    assert "fresh_0.wav" not in serialized


def test_line_label_quit_writes_unreached_sidecar(tmp_path: Path) -> None:
    records = run_label_session(
        _selections(),
        store_path=tmp_path / "labels.jsonl",
        session_id="line-quit",
        play_fn=lambda _path: None,
        input_fn=lambda _prompt: "q",
        print_fn=lambda *_args: None,
    )
    assert records == []
    sidecar = json.loads((tmp_path / "sessions" / "line-quit_label_session.json").read_text())
    assert all(item["outcome"] == "unreached" for item in sidecar["clips"])


def test_line_label_interrupt_writes_unreached_sidecar(tmp_path: Path) -> None:
    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_label_session(
            _selections(),
            store_path=tmp_path / "labels.jsonl",
            session_id="line-interrupt",
            play_fn=lambda _path: None,
            input_fn=interrupt,
            print_fn=lambda *_args: None,
        )
    sidecar = json.loads((tmp_path / "sessions" / "line-interrupt_label_session.json").read_text())
    assert all(item["outcome"] == "unreached" for item in sidecar["clips"])
