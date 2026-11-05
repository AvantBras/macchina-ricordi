from __future__ import annotations

import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.instrument.stream.monitor import MonitorSnapshot, render_frame
from macchina_ricordi.instrument.stream.session import (
    StreamControl,
    build_stream_session_paths,
    read_control,
    write_control,
)
from macchina_ricordi.instrument.stream.tui import (
    HELP_LINES,
    StreamStatus,
    StreamTui,
    TerminalSession,
    apply_key,
    decode_keys,
)

# ── key decoding ──────────────────────────────────────────────────────


def test_decode_keys_returns_every_key_in_a_burst() -> None:
    # A held-down arrow arrives as repeats inside one read; dropping all but
    # the first would make sweeping a knob feel broken.
    assert decode_keys("\x1b[A\x1b[A\x1b[A") == ["up", "up", "up"]
    assert decode_keys("rp?") == ["r", "p", "?"]
    assert decode_keys("\x1b[Ar\x1b[B") == ["up", "r", "down"]


def test_decode_keys_drops_unknown_escapes_without_leaking_their_payload() -> None:
    assert decode_keys("\x1b[Zr") == ["r"]  # not ['[', 'Z', 'r']
    assert decode_keys("\x1b") == []
    assert decode_keys("") == []


@pytest.mark.parametrize("data", ["", "\x1b", "\x1b[", "\x1b[Z"])
def test_decode_keys_discards_incomplete_escapes(data: str) -> None:
    # A bare or unknown escape must not wedge the input loop waiting for more.
    assert decode_keys(data) == []


def test_decode_keys_drops_whole_escape_sequences() -> None:
    assert decode_keys("\x1b[1;5C") == []  # Ctrl+Right: its final C is not a key
    assert decode_keys("\x1b[5~") == []  # Page Up
    assert decode_keys("\x1bOA") == ["up"]  # arrows in application mode
    assert decode_keys("\x1b[C\x1b[D") == ["right", "left"]


# ── keymap ────────────────────────────────────────────────────────────


def test_apply_key_nudges_temperature_both_ways() -> None:
    control = StreamControl(temperature=1.0)
    up = apply_key("up", control)
    assert up.control.temperature == pytest.approx(1.05)
    assert up.handled is True
    assert "temperature" in (up.message or "")
    assert apply_key("down", control).control.temperature == pytest.approx(0.95)


def test_apply_key_nudges_tau_round_topk_and_novelty() -> None:
    control = StreamControl(tau_seconds=5.0, completion_duration_seconds=15.0, top_k=250, novelty_weight=0.3)
    assert apply_key("right", control).control.tau_seconds == pytest.approx(5.5)
    assert apply_key("left", control).control.tau_seconds == pytest.approx(4.5)
    assert apply_key("]", control).control.completion_duration_seconds == pytest.approx(16.0)
    assert apply_key("[", control).control.completion_duration_seconds == pytest.approx(14.0)
    assert apply_key(".", control).control.top_k == 260
    assert apply_key(",", control).control.top_k == 240
    assert apply_key("N", control).control.novelty_weight == pytest.approx(0.35)
    assert apply_key("n", control).control.novelty_weight == pytest.approx(0.25)


def test_apply_key_clamps_at_both_ends() -> None:
    assert apply_key("down", StreamControl(temperature=0.05)).control.temperature == pytest.approx(0.05)
    assert apply_key("up", StreamControl(temperature=3.0)).control.temperature == pytest.approx(3.0)
    assert apply_key("n", StreamControl(novelty_weight=0.0)).control.novelty_weight == 0.0
    assert apply_key("N", StreamControl(novelty_weight=1.0)).control.novelty_weight == 1.0
    assert apply_key(",", StreamControl(top_k=0)).control.top_k == 0
    assert apply_key("[", StreamControl(completion_duration_seconds=1.0)).control.completion_duration_seconds == 1.0
    assert apply_key("]", StreamControl(completion_duration_seconds=60.0)).control.completion_duration_seconds == 60.0


def test_apply_key_says_when_a_knob_is_already_at_its_limit() -> None:
    result = apply_key("up", StreamControl(temperature=3.0))
    assert result.handled is True  # still worth flashing, so the key does not feel dead
    assert "at limit" in (result.message or "")


def test_apply_key_allows_tau_longer_than_completion() -> None:
    control = StreamControl(tau_seconds=4.0, completion_duration_seconds=4.0)
    updated = apply_key("right", control).control
    assert updated.tau_seconds == pytest.approx(4.5)
    assert updated.completion_duration_seconds == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("key", "command"),
    [("r", "reroll"), ("p", "pin"), ("i", "inject_light"), ("I", "inject_heavy"), ("Q", "stop")],
)
def test_apply_key_queues_one_shot_commands(key: str, command: str) -> None:
    result = apply_key(key, StreamControl())
    assert result.control.command == command
    assert result.handled is True


def test_lowercase_q_does_not_stop_the_session() -> None:
    # A stray keystroke must never end a performance.
    assert apply_key("q", StreamControl()).handled is False


@pytest.mark.parametrize("key", ["z", "1", " ", None])
def test_apply_key_ignores_unbound_keys(key: str | None) -> None:
    # "1" is a playlist key only when a playlist was given; without one it must
    # not rewrite the control file.
    control = StreamControl()
    result = apply_key(key, control)
    assert result.handled is False
    assert result.control == control


# ── candidates ────────────────────────────────────────────────────────


def test_apply_key_nudges_candidates_both_ways_and_stops_at_the_batch_limits() -> None:
    control = StreamControl(candidates=4)
    assert apply_key("C", control).control.candidates == 5
    assert apply_key("c", control).control.candidates == 3
    # Below 1 there is nothing to generate; above 8 the batch stops being free.
    assert apply_key("c", StreamControl(candidates=1)).control.candidates == 1
    capped = apply_key("C", StreamControl(candidates=8))
    assert capped.control.candidates == 8
    assert "at limit" in (capped.message or "")


# ── buffer thresholds ─────────────────────────────────────────────────


def test_apply_key_nudges_the_buffer_cap_and_keeps_the_floor_ratio() -> None:
    control = StreamControl(max_buffer_ahead_seconds=60.0, resume_buffer_ahead_seconds=45.0)
    lowered = apply_key("-", control).control
    assert lowered.max_buffer_ahead_seconds == pytest.approx(55.0)
    assert lowered.resume_buffer_ahead_seconds == pytest.approx(55.0 * 0.75)  # the 0.75 ratio survives

    raised = apply_key("+", control).control
    assert raised.max_buffer_ahead_seconds == pytest.approx(65.0)
    assert raised.resume_buffer_ahead_seconds == pytest.approx(65.0 * 0.75)


def test_floor_keys_move_the_resume_threshold_as_a_fraction_of_the_cap() -> None:
    control = StreamControl(max_buffer_ahead_seconds=60.0, resume_buffer_ahead_seconds=30.0)
    assert apply_key("}", control).control.resume_buffer_ahead_seconds == pytest.approx(36.0)
    assert apply_key("{", control).control.resume_buffer_ahead_seconds == pytest.approx(24.0)
    full = StreamControl(max_buffer_ahead_seconds=60.0, resume_buffer_ahead_seconds=60.0)
    assert "at limit" in (apply_key("}", full).message or "")
    assert "no throttle configured" in (apply_key("}", StreamControl()).message or "")


def test_apply_key_turns_the_throttle_on_from_unset() -> None:
    result = apply_key("-", StreamControl())
    assert result.handled is True
    assert result.control.max_buffer_ahead_seconds == pytest.approx(30.0)
    assert result.control.resume_buffer_ahead_seconds == pytest.approx(20.0)
    assert "throttle on" in (result.message or "")


def test_apply_key_clamps_the_buffer_cap() -> None:
    floored = apply_key("-", StreamControl(max_buffer_ahead_seconds=5.0))
    assert floored.control.max_buffer_ahead_seconds == pytest.approx(5.0)
    assert "at limit" in (floored.message or "")
    ceiled = apply_key("+", StreamControl(max_buffer_ahead_seconds=300.0))
    assert ceiled.control.max_buffer_ahead_seconds == pytest.approx(300.0)


def test_t_holds_the_throttle_open_without_losing_the_cap() -> None:
    # The cap must survive the toggle: switching back has to restore this
    # session's own thresholds, not a default.
    control = StreamControl(max_buffer_ahead_seconds=60.0, resume_buffer_ahead_seconds=45.0)
    opened = apply_key("t", control).control
    assert opened.throttle_enabled is False
    assert opened.max_buffer_ahead_seconds == pytest.approx(60.0)
    assert opened.resume_buffer_ahead_seconds == pytest.approx(45.0)

    closed = apply_key("t", opened).control
    assert closed == control


def test_t_on_an_unthrottled_session_says_so_instead_of_doing_nothing() -> None:
    result = apply_key("t", StreamControl())
    assert result.control == StreamControl()
    assert "no throttle configured" in (result.message or "")


def test_nudging_the_cap_does_not_silently_re_arm_a_held_open_throttle() -> None:
    # Re-throttling mid-cushion, unasked, is the worst possible moment for it.
    control = StreamControl(max_buffer_ahead_seconds=60.0, throttle_enabled=False)
    result = apply_key("+", control)
    assert result.control.throttle_enabled is False
    assert "throttle still off" in (result.message or "")


def test_buffer_resume_never_exceeds_the_cap_it_was_derived_from() -> None:
    # A control whose resume sits above its max (hand-edited) must not produce
    # a resume the loop can never reach.
    result = apply_key("-", StreamControl(max_buffer_ahead_seconds=20.0, resume_buffer_ahead_seconds=40.0))
    assert result.control.resume_buffer_ahead_seconds <= result.control.max_buffer_ahead_seconds


# ── prompt steering ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cut", "boost", "field"),
    [("j", "J", "prompt_tilt_low_db"), ("k", "K", "prompt_tilt_mid_db"), ("l", "L", "prompt_tilt_high_db")],
)
def test_band_keys_tilt_their_own_band_only(cut: str, boost: str, field: str) -> None:
    control = StreamControl()
    boosted = apply_key(boost, control).control
    assert getattr(boosted, field) == pytest.approx(1.5)
    cut_result = apply_key(cut, control).control
    assert getattr(cut_result, field) == pytest.approx(-1.5)
    # The other two bands stay where they were.
    others = [
        f
        for _c, _b, f in [
            ("j", "J", "prompt_tilt_low_db"),
            ("k", "K", "prompt_tilt_mid_db"),
            ("l", "L", "prompt_tilt_high_db"),
        ]
        if f != field
    ]
    assert all(getattr(boosted, other) == 0.0 for other in others)


def test_tilt_clamps_at_twelve_db() -> None:
    assert apply_key("J", StreamControl(prompt_tilt_low_db=12.0)).control.prompt_tilt_low_db == 12.0
    assert apply_key("j", StreamControl(prompt_tilt_low_db=-12.0)).control.prompt_tilt_low_db == -12.0
    assert "at limit" in (apply_key("J", StreamControl(prompt_tilt_low_db=12.0)).message or "")


def test_tilt_flash_is_signed_so_a_cut_reads_as_a_cut() -> None:
    assert "-1.5" in (apply_key("k", StreamControl()).message or "")
    assert "+1.5" in (apply_key("K", StreamControl()).message or "")


def test_rate_keys_nudge_prompt_rate_and_clamp() -> None:
    assert apply_key(">", StreamControl()).control.prompt_rate == pytest.approx(1.02)
    assert apply_key("<", StreamControl()).control.prompt_rate == pytest.approx(0.98)
    assert apply_key(">", StreamControl(prompt_rate=2.0)).control.prompt_rate == 2.0
    assert apply_key("<", StreamControl(prompt_rate=0.5)).control.prompt_rate == 0.5


def test_zero_returns_both_steering_controls_to_neutral() -> None:
    # Counting keypresses back to flat mid-performance is not playing.
    steered = StreamControl(
        prompt_tilt_low_db=9.0,
        prompt_tilt_mid_db=-3.0,
        prompt_tilt_high_db=6.0,
        prompt_rate=1.14,
    )
    flat = apply_key("0", steered).control
    assert (flat.prompt_tilt_low_db, flat.prompt_tilt_mid_db, flat.prompt_tilt_high_db) == (0.0, 0.0, 0.0)
    assert flat.prompt_rate == 1.0


def test_zero_leaves_every_other_control_alone() -> None:
    steered = StreamControl(prompt_rate=1.1, temperature=1.4, candidates=4, model_xp="xp")
    flat = apply_key("0", steered).control
    assert (flat.temperature, flat.candidates, flat.model_xp) == (1.4, 4, "xp")


def test_zero_on_an_already_flat_session_says_so() -> None:
    result = apply_key("0", StreamControl())
    assert result.control == StreamControl()
    assert "already flat" in (result.message or "")


def test_zero_is_not_a_playlist_key() -> None:
    # Digits 1-9 select models; 0 must stay the steering reset even then.
    result = apply_key("0", StreamControl(prompt_rate=1.1), playlist=(".dora/xps/aaaaaaaa",))
    assert result.control.prompt_rate == 1.0
    assert result.control.model_xp is None


# ── model playlist ────────────────────────────────────────────────────

_PLAYLIST = (".dora/xps/aaaaaaaa", ".dora/xps/bbbbbbbb", ".dora/xps/cccccccc")


def test_digit_keys_select_from_the_playlist() -> None:
    result = apply_key("2", StreamControl(), playlist=_PLAYLIST)
    assert result.handled is True
    assert result.control.model_xp == ".dora/xps/bbbbbbbb"
    assert "bbbbbbbb" in (result.message or "")


def test_digit_past_the_end_of_the_playlist_flashes_instead_of_swapping() -> None:
    result = apply_key("9", StreamControl(model_xp="keep-me"), playlist=_PLAYLIST)
    assert result.control.model_xp == "keep-me"
    assert "3 models" in (result.message or "")


def test_selecting_the_loaded_model_says_so_rather_than_pretending_to_swap() -> None:
    control = StreamControl(model_xp=_PLAYLIST[0])
    result = apply_key("1", control, playlist=_PLAYLIST)
    assert result.control.model_xp == _PLAYLIST[0]
    assert "already loaded" in (result.message or "")


def test_tab_cycles_the_playlist_and_wraps() -> None:
    control = StreamControl(model_xp=_PLAYLIST[2])
    assert apply_key("\t", control, playlist=_PLAYLIST).control.model_xp == _PLAYLIST[0]
    assert apply_key("\t", StreamControl(), playlist=_PLAYLIST).control.model_xp == _PLAYLIST[0]


def test_tab_matches_the_current_model_the_way_the_round_loop_does() -> None:
    # The loop compares resolved paths before deciding a swap is a swap, so an
    # unresolved model_xp must still be recognised as playlist slot 1.
    control = StreamControl(model_xp="./.dora/xps/aaaaaaaa")
    assert apply_key("\t", control, playlist=_PLAYLIST).control.model_xp == _PLAYLIST[1]


# ── in-flight status ──────────────────────────────────────────────────


def test_stream_status_describes_the_current_phase() -> None:
    status = StreamStatus()
    status.set_phase("generating", round_idx=214)
    described = status.describe()
    assert "generating" in described and "round 214" in described

    status.set_phase("throttled", round_idx=215)
    assert "buffer full" in status.describe()

    status.set_phase("generating", round_idx=216, take_idx=2)
    assert "take 2" in status.describe()


# ── terminal handling ─────────────────────────────────────────────────


def test_terminal_session_is_a_noop_without_a_tty() -> None:
    # pytest captures stdin/stdout, so this is the non-interactive path.
    terminal = TerminalSession(stream=io.StringIO())
    assert terminal.is_interactive() is False
    with terminal as entered:
        assert entered.active is False
    assert terminal.active is False
    assert terminal.read_keys(0.01) == []


def test_terminal_session_restores_on_an_exception() -> None:
    terminal = TerminalSession(stream=io.StringIO())
    with pytest.raises(RuntimeError):
        with terminal:
            raise RuntimeError("render blew up")
    assert terminal.active is False


def test_read_keys_finishes_an_escape_sequence_split_across_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # one read takes at most 64 bytes, so the 11th Ctrl+Right is cut in two and its final C must not leak
    read_end, write_end = os.pipe()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(fileno=lambda: read_end))
    terminal = TerminalSession(stream=io.StringIO())
    terminal.active, terminal._fd = True, read_end
    try:
        os.write(write_end, b"\x1b[1;5C" * 11)
        assert terminal.read_keys(0.5) == []
        assert terminal.read_keys(0.5) == []
        os.write(write_end, b"\x1b[A")
        assert terminal.read_keys(0.5) == ["up"]
        os.write(write_end, b"\x1b")  # a lone Esc is held for one read, then dropped once input goes quiet
        assert terminal.read_keys(0.5) == []
        assert terminal.read_keys(0.01) == []
        os.write(write_end, b"r")
        assert terminal.read_keys(0.5) == ["r"]
    finally:
        os.close(read_end)
        os.close(write_end)


# ── the TUI ───────────────────────────────────────────────────────────


def _session(tmp_path: Path, control: StreamControl | None = None) -> Path:
    paths = build_stream_session_paths(tmp_path, run_id="20260729T120000Z")
    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.performance_path.write_text("", encoding="utf-8")
    write_control(paths.control_path, control or StreamControl())
    return paths.session_dir


def _tui(
    tmp_path: Path,
    control: StreamControl | None = None,
    *,
    playback=None,
    playlist=(),
) -> StreamTui:
    session_dir = _session(tmp_path, control)
    return StreamTui(
        session_dir,
        session_dir / "control.json",
        stream=io.StringIO(),
        terminal=TerminalSession(stream=io.StringIO()),
        playback=playback,
        playlist=playlist,
    )


def test_tui_key_writes_through_to_the_control_file(tmp_path: Path) -> None:
    tui = _tui(tmp_path, StreamControl(temperature=1.0))
    assert tui.handle_key("up") is True
    assert read_control(tui.control_path).temperature == pytest.approx(1.05)
    assert "temperature" in (tui.flash or "")


def test_tui_pin_captures_the_current_playback_path(tmp_path: Path) -> None:
    target = tmp_path / "rounds" / "round_004.wav"

    class Playback:
        def current_path(self) -> Path:
            return target

    tui = _tui(tmp_path, playback=Playback())
    assert tui.handle_key("p") is True
    control = read_control(tui.control_path)
    assert control.command == "pin"
    assert control.pin_target == str(target)


@pytest.mark.parametrize("key", ["z", "2"])  # "2" is a model key only when a playlist was given
def test_tui_leaves_the_control_file_alone_for_unbound_keys(tmp_path: Path, key: str) -> None:
    tui = _tui(tmp_path)
    before = tui.control_path.read_bytes()
    assert tui.handle_key(key) is False
    assert tui.control_path.read_bytes() == before


def test_tui_preserves_fields_it_has_no_key_for(tmp_path: Path) -> None:
    # The read-modify-write must not drop knobs that are only editable by hand.
    tui = _tui(tmp_path, StreamControl(candidates=6, motif_period=4, extend_stride=6.0))
    tui.handle_key("up")
    after = read_control(tui.control_path)
    assert (after.candidates, after.motif_period, after.extend_stride) == (6, 4, 6.0)


def test_tui_help_key_toggles_without_touching_control(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    before = tui.control_path.read_bytes()
    assert tui.handle_key("?") is True
    assert tui.show_help is True
    assert HELP_LINES[0] in tui.status_lines()
    assert tui.handle_key("?") is True
    assert tui.show_help is False
    assert tui.control_path.read_bytes() == before


def test_tui_playlist_digit_writes_the_model_swap_through(tmp_path: Path) -> None:
    tui = _tui(tmp_path, playlist=(".dora/xps/aaaaaaaa", ".dora/xps/bbbbbbbb"))
    assert tui.handle_key("2") is True
    assert read_control(tui.control_path).model_xp == ".dora/xps/bbbbbbbb"


def test_tui_help_names_the_playlist_models(tmp_path: Path) -> None:
    tui = _tui(tmp_path, playlist=(".dora/xps/aaaaaaaa", ".dora/xps/bbbbbbbb"))
    tui.handle_key("?")
    lines = tui.status_lines()
    assert any("1  aaaaaaaa" in line for line in lines)
    assert any("2  bbbbbbbb" in line for line in lines)


def test_tui_buffer_key_writes_both_thresholds_through(tmp_path: Path) -> None:
    tui = _tui(tmp_path, StreamControl(max_buffer_ahead_seconds=60.0, resume_buffer_ahead_seconds=45.0))
    assert tui.handle_key("-") is True
    control = read_control(tui.control_path)
    assert control.max_buffer_ahead_seconds == pytest.approx(55.0)
    assert control.resume_buffer_ahead_seconds == pytest.approx(55.0 * 0.75)


def test_tui_throttle_toggle_writes_through(tmp_path: Path) -> None:
    tui = _tui(tmp_path, StreamControl(max_buffer_ahead_seconds=60.0))
    assert tui.handle_key("t") is True
    assert read_control(tui.control_path).throttle_enabled is False
    assert tui.handle_key("t") is True
    assert read_control(tui.control_path).throttle_enabled is True


def test_tui_steering_keys_write_through(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    for key in ("J", "J", "l", ">"):
        assert tui.handle_key(key) is True
    control = read_control(tui.control_path)
    assert control.prompt_tilt_low_db == pytest.approx(3.0)
    assert control.prompt_tilt_high_db == pytest.approx(-1.5)
    assert control.prompt_rate == pytest.approx(1.02)

    assert tui.handle_key("0") is True
    flat = read_control(tui.control_path)
    assert flat.prompt_tilt_low_db == 0.0
    assert flat.prompt_rate == 1.0


def test_tui_reports_a_broken_control_file_instead_of_crashing(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    tui.control_path.write_text("{not json", encoding="utf-8")
    assert tui.handle_key("up") is True
    assert "control edit failed" in (tui.flash or "")


def test_tui_frame_carries_the_status_line_and_its_own_title(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    tui.status.set_phase("generating", round_idx=3)
    frame = tui.frame(width=84)
    assert "stream — 20260729T120000Z" in frame
    assert "? for keys" in frame
    assert "generating round 3" in frame


def test_tui_draw_survives_a_render_error(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    tui.session_dir = Path("/definitely/not/a/session")

    def explode(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("boom")

    tui.frame = explode  # type: ignore[method-assign]
    tui.draw(width=84)
    assert "render_error" in tui.stream.getvalue()  # type: ignore[union-attr]


def test_tui_start_and_stop_are_idempotent(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    tui.start()
    tui.start()
    tui.stop()
    tui.stop()


# ── render_frame status lines ─────────────────────────────────────────


def test_render_frame_places_status_lines_under_the_header() -> None:
    snapshot = MonitorSnapshot(session_dir=Path("/tmp/s"), rows=[], control=None, read_at=0.0)
    frame = render_frame(snapshot, status_lines=["generating round 9 · 2.1s"])
    assert "generating round 9" in frame
    assert "waiting for the first round" in frame


def test_render_frame_status_lines_are_clipped_to_width() -> None:
    snapshot = MonitorSnapshot(session_dir=Path("/tmp/s"), rows=[], control=None, read_at=0.0)
    frame = render_frame(snapshot, width=40, status_lines=["x" * 200])
    assert all(len(line) <= 40 for line in frame.splitlines())


# ── worker stderr redirection ─────────────────────────────────────────


def test_persistent_generator_sends_worker_stderr_to_a_file(tmp_path: Path, monkeypatch) -> None:
    from macchina_ricordi.instrument.stream import worker as stream_worker

    captured: dict[str, object] = {}

    def fake_popen(_cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return type("P", (), {"poll": lambda self: None})()

    monkeypatch.setattr(stream_worker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(stream_worker, "require_musicgen_python", lambda: sys.executable)
    monkeypatch.setattr(stream_worker, "musicgen_script_env", lambda: {"PATH": "/usr/bin"})
    config = type("C", (), {"repo_root": Path.cwd(), "dora_dir": tmp_path / "dora"})()
    monkeypatch.setattr(stream_worker, "load_musicgen_project_config", lambda: config)

    log = tmp_path / "worker.log"
    generator = stream_worker.PersistentStreamGenerator(stderr_path=log)
    generator._ensure_process()

    handle = captured["stderr"]
    assert handle is not None and not isinstance(handle, int)
    handle.write("audiocraft noise\n")
    handle.flush()
    assert "audiocraft noise" in log.read_text(encoding="utf-8")


def test_persistent_generator_inherits_stderr_by_default() -> None:
    from macchina_ricordi.instrument.stream.worker import PersistentStreamGenerator

    # None means "inherit the parent's stderr" — today's behaviour, unchanged.
    assert PersistentStreamGenerator()._open_stderr() is None


# ── follow mode toggle ────────────────────────────────────────────────


def test_tui_f_key_toggles_follow_without_touching_control(tmp_path: Path) -> None:
    tui = _tui(tmp_path)
    before = tui.control_path.read_bytes()
    assert tui.follow == "generated"

    assert tui.handle_key("f") is True
    assert tui.follow == "playing"
    assert "playing" in (tui.flash or "")

    assert tui.handle_key("f") is True
    assert tui.follow == "generated"
    # Display-only: a performance is never affected by looking at it differently.
    assert tui.control_path.read_bytes() == before


def test_tui_follow_reaches_the_frame(tmp_path: Path) -> None:
    session_dir = _session(tmp_path)
    tui = StreamTui(
        session_dir,
        session_dir / "control.json",
        stream=io.StringIO(),
        terminal=TerminalSession(stream=io.StringIO()),
        follow="playing",
    )
    (session_dir / "performance.jsonl").write_text(
        json.dumps(
            {
                "round_idx": 1,
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "round_wav": "rounds/round_001.wav",
                "playback_seconds": 10.0,
                "buffer_seconds_ahead": None,
                "score": {"novelty_vs_memory": 0.4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert "nothing playing" in tui.frame(width=84)


# ── blind toggle ──────────────────────────────────────────────────────


def test_b_toggles_blind_without_touching_the_control_file(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    control = session / "control.json"
    write_control(control, StreamControl(temperature=1.2))
    before = control.read_text(encoding="utf-8")

    tui = StreamTui(session, control, stream=io.StringIO())
    assert tui.blind is False

    assert tui.handle_key("b") is True
    assert tui.blind is True
    assert "blind" in (tui.flash or "")
    # Display-only: a performance can never be affected by looking away.
    assert control.read_text(encoding="utf-8") == before

    assert tui.handle_key("b") is True
    assert tui.blind is False


def test_blind_tui_frame_carries_no_readings(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    write_control(session / "control.json", StreamControl())
    (session / "performance.jsonl").write_text(
        json.dumps(
            {
                "round_idx": 1,
                "take_idx": 0,
                "timestamp": "2026-08-06T10:00:00Z",
                "round_wav": "rounds/round_001.wav",
                "duration_seconds": 10.0,
                "elapsed_seconds": 2.0,
                "playback_seconds": 10.0,
                "buffer_seconds_ahead": 12.0,
                "score": {"arc_coherence_score": 0.777, "novelty_whitened": 0.321},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tui = StreamTui(session, session / "control.json", stream=io.StringIO(), blind=True)
    frame = tui.frame(width=90)
    assert "playing blind" in frame
    assert "0.777" not in frame and "0.321" not in frame
