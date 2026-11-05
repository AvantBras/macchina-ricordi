from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from macchina_ricordi.instrument.stream.monitor import (
    _read_jsonl_tail,
    derive_events,
    latest_session_dir,
    meter,
    playback_head,
    read_snapshot,
    render_events,
    render_frame,
    run_monitor,
    sparkline,
)


def _stamp(offset_seconds: float = 0.0) -> str:
    moment = datetime.now(UTC) + timedelta(seconds=offset_seconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(round_idx: int, **overrides) -> dict:
    row = {
        "round_idx": round_idx,
        "take_idx": 0,
        "timestamp": _stamp(),
        "command_consumed": "continue",
        "active_control": {
            "temperature": 1.2,
            "tau_seconds": 5.0,
            "completion_duration_seconds": 15.0,
            "candidates": 2,
            "novelty_weight": 0.3,
            "model_xp": ".dora/xps/abc123",
        },
        "round_wav": f"rounds/round_{round_idx:03d}.wav",
        "duration_seconds": 15.0,
        "elapsed_seconds": 4.0,
        "playback_seconds": 10.0,
        "buffer_seconds_ahead": 20.0,
        "score": {
            "arc_coherence_score": 0.5,
            "silence_fraction": 0.1,
            "score": 4.0,
            "onset_density": 6.0,
        },
    }
    row.update(overrides)
    return row


def _write_session(tmp_path: Path, rows: list[dict], *, control: dict | None = None) -> Path:
    session = tmp_path / "20260729T000000Z"
    session.mkdir(parents=True, exist_ok=True)
    (session / "performance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if control is not None:
        (session / "control.json").write_text(json.dumps(control), encoding="utf-8")
    return session


# ── session discovery / reading ───────────────────────────────────────


def test_latest_session_dir_picks_newest_with_a_log(tmp_path: Path) -> None:
    for name in ("20260101T000000Z", "20260729T000000Z"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "performance.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "20261231T000000Z").mkdir()  # no log: not a session yet
    assert latest_session_dir(tmp_path) == tmp_path / "20260729T000000Z"


def test_latest_session_dir_returns_none_when_empty(tmp_path: Path) -> None:
    assert latest_session_dir(tmp_path) is None
    assert latest_session_dir(tmp_path / "missing") is None


def test_read_jsonl_tail_skips_a_half_written_row(tmp_path: Path) -> None:
    path = tmp_path / "performance.jsonl"
    path.write_text(json.dumps(_row(1)) + "\n" + '{"round_idx": 2, "partial', encoding="utf-8")
    rows = _read_jsonl_tail(path, limit=10)
    assert [row["round_idx"] for row in rows] == [1]


def test_read_jsonl_tail_honours_limit(tmp_path: Path) -> None:
    path = tmp_path / "performance.jsonl"
    path.write_text("".join(json.dumps(_row(i)) + "\n" for i in range(10)), encoding="utf-8")
    assert [row["round_idx"] for row in _read_jsonl_tail(path, limit=3)] == [7, 8, 9]


def test_read_jsonl_tail_tolerates_a_missing_file(tmp_path: Path) -> None:
    assert _read_jsonl_tail(tmp_path / "nope.jsonl", limit=5) == []


def test_read_snapshot_tolerates_malformed_control(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)])
    (session / "control.json").write_text("{not json", encoding="utf-8")
    snapshot = read_snapshot(session)
    assert snapshot.control is None
    assert len(snapshot.rows) == 1


# ── playback head ─────────────────────────────────────────────────────


def test_playback_head_locates_the_audible_round_behind_the_generator() -> None:
    now = datetime.now(UTC).timestamp()
    rows = [_row(i, buffer_seconds_ahead=25.0, playback_seconds=10.0) for i in range(1, 6)]
    head = playback_head(rows, now=now)
    # 25s buffered / 10s per round: the player is three rounds back from round 5.
    assert head.audible_round == 3
    assert head.rounds_behind == 2
    assert head.edit_lands_at == 6
    assert head.drained is False


def test_playback_head_reports_drained_when_the_log_went_quiet() -> None:
    now = datetime.now(UTC).timestamp()
    rows = [_row(4, timestamp=_stamp(-600), buffer_seconds_ahead=20.0)]
    head = playback_head(rows, now=now)
    assert head.drained is True
    assert head.seconds_ahead_now == 0.0
    assert head.edit_lands_at == 5


def test_playback_head_without_buffer_telemetry_still_reports_the_next_round() -> None:
    now = datetime.now(UTC).timestamp()
    head = playback_head([_row(2, buffer_seconds_ahead=None)], now=now)
    assert head.audible_round is None
    assert head.edit_lands_at == 3


def test_playback_head_on_an_empty_log() -> None:
    head = playback_head([], now=0.0)
    assert head.audible_round is None and head.edit_lands_at is None


# ── drawing primitives ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("values", "width", "expected"),
    [
        ([0.0, None, 1.0], 3, "▁ █"),  # a gap stays blank between the floor and the ceiling
        ([0.5, 0.5, 0.5], 3, "▁▁▁"),  # a flat series sits on the floor
        ([None, None], 2, "  "),  # no data at all is blank
    ],
)
def test_sparkline_draws_gaps_flat_series_and_no_data(values: list[float | None], width: int, expected: str) -> None:
    assert sparkline(values, width=width) == expected


def test_meter_clamps_out_of_range_values() -> None:
    assert meter(99.0, lo=0.0, hi=1.0, width=4) == "████"
    assert meter(-5.0, lo=0.0, hi=1.0, width=4) == "····"
    assert meter(None, lo=0.0, hi=1.0, width=4) == "    "


# ── frame rendering ───────────────────────────────────────────────────


def test_render_frame_without_rounds_says_it_is_waiting(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [])
    frame = render_frame(read_snapshot(session))
    assert "waiting for the first round" in frame


def test_render_frame_shows_last_round_metrics_and_both_heads(tmp_path: Path) -> None:
    rows = [_row(i, buffer_seconds_ahead=25.0) for i in range(1, 6)]
    session = _write_session(tmp_path, rows, control={"temperature": 1.4, "candidates": 2})
    frame = render_frame(read_snapshot(session))
    assert "round 5" in frame
    assert "hearing round 3" in frame
    assert "an edit now lands at round 6" in frame
    assert "round_005.wav" in frame
    assert "arc" in frame and "0.500" in frame
    assert "temp 1.40" in frame


def test_render_frame_derives_completion_for_legacy_rows(tmp_path: Path) -> None:
    legacy_control = {
        "temperature": 1.4,
        "tau_seconds": 5.0,
        "round_duration_seconds": 15.0,
    }
    session = _write_session(tmp_path, [_row(1, active_control=legacy_control)])
    frame = render_frame(read_snapshot(session))
    assert "completion 10.0" in frame
    assert "round 15.0" not in frame


def test_render_frame_marks_the_selected_candidate(tmp_path: Path) -> None:
    score = {
        "arc_coherence_score": 0.4,
        "silence_fraction": 0.0,
        "candidates": [
            {"candidate": 0, "selected": False, "arc_coherence_score": 0.2, "novelty": 0.01},
            {"candidate": 1, "selected": True, "arc_coherence_score": 0.4, "novelty": 0.05},
        ],
    }
    session = _write_session(tmp_path, [_row(1, score=score)])
    frame = render_frame(read_snapshot(session))
    selected_line = next(line for line in frame.splitlines() if "#1" in line)
    other_line = next(line for line in frame.splitlines() if "#0" in line)
    assert "◀" in selected_line
    assert "◀" not in other_line
    # The kept candidate's novelty is what the round-level novelty meter reports.
    assert "0.0500" in selected_line


def test_render_frame_surfaces_rescue_and_motif_flags(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [_row(1, tail_rescued_from_seconds=9.5, motif_source="/bank/motif_a.wav")],
    )
    frame = render_frame(read_snapshot(session))
    assert "tail_rescue" in frame
    assert "motif_a.wav" in frame


def test_render_frame_warns_when_generation_cannot_keep_up(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1, elapsed_seconds=14.0, playback_seconds=10.0)])
    frame = render_frame(read_snapshot(session))
    assert "1.40x rt" in frame
    assert "starving" in frame


def test_render_frame_reports_a_drained_buffer(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1, timestamp=_stamp(-900))])
    frame = render_frame(read_snapshot(session))
    assert "playback drained" in frame


def test_render_frame_meters_the_buffer_against_the_cap(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [_row(1, buffer_seconds_ahead=30.0)],
        control={"max_buffer_ahead_seconds": 60.0},
    )
    frame = render_frame(read_snapshot(session))
    assert "max 60s" in frame


def test_render_frame_keeps_saying_the_throttle_is_held_open(tmp_path: Path) -> None:
    # A forgotten toggle rebuilds a huge buffer and every other knob goes numb,
    # so the state has to outlive the keypress flash that set it.
    session = _write_session(
        tmp_path,
        [_row(1, buffer_seconds_ahead=180.0)],
        control={"max_buffer_ahead_seconds": 60.0, "throttle_enabled": False},
    )
    frame = render_frame(read_snapshot(session))
    assert "throttle off" in frame
    assert "60s parked" in frame
    assert "max 60s" not in frame  # no ceiling to meter against while it is open


def test_control_row_wraps_instead_of_losing_its_tail(tmp_path: Path) -> None:
    # Ten knobs do not fit 80 columns, and the ones that fell off the end were
    # always the newest ones.
    session = _write_session(
        tmp_path,
        [_row(1)],
        control={
            "temperature": 1.25,
            "tau_seconds": 5.0,
            "completion_duration_seconds": 10.0,
            "candidates": 4,
            "novelty_weight": 0.3,
            "min_p": 0.05,
            "top_k": 250,
            "prompt_rate": 1.08,
            "max_buffer_ahead_seconds": 60.0,
        },
    )
    frame = render_frame(read_snapshot(session), width=84)
    assert all(len(line) <= 84 for line in frame.splitlines())
    for expected in ("temp 1.25", "top_k 250", "rate 1.080", "max_ahead  60.0"):
        assert expected in frame
    narrow = render_frame(read_snapshot(session), width=70)
    assert all(len(line) <= 70 for line in narrow.splitlines())


def test_control_row_hides_controls_that_are_switched_off(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [_row(1)],
        control={"temperature": 1.0, "novelty_weight": 0.0, "min_p": 0.0, "prompt_rate": 1.0},
    )
    frame = render_frame(read_snapshot(session))
    assert "temp 1.00" in frame
    assert "novelty 0.00" not in frame
    assert "rate 1.000" not in frame


def test_control_row_shows_an_engaged_prompt_tilt(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [_row(1)],
        control={"prompt_tilt_low_db": 6.0, "prompt_tilt_mid_db": 0.0, "prompt_tilt_high_db": -4.5},
    )
    frame = render_frame(read_snapshot(session))
    assert "tilt +6.0/+0.0/-4.5" in frame


def test_flags_name_the_steering_a_round_was_generated_under(tmp_path: Path) -> None:
    session = _write_session(
        tmp_path,
        [
            _row(
                1,
                prompt_rate=1.08,
                prompt_rate_mode="tempo",
                prompt_tilt_db={"low_db": 6.0, "mid_db": 0.0, "high_db": -4.5},
            )
        ],
    )
    frame = render_frame(read_snapshot(session))
    assert "prompt_rate 1.080 (tempo)" in frame
    assert "tilt +6.0/+0.0/-4.5dB" in frame


def test_render_frame_survives_rows_with_no_score(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1, score=None, buffer_seconds_ahead=None)])
    frame = render_frame(read_snapshot(session))
    assert "round 1" in frame
    assert "—" in frame  # missing numbers render as a dash, not a crash


def test_run_monitor_once_writes_a_single_frame(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)])
    buffer = io.StringIO()
    run_monitor(session, once=True, stream=buffer)
    assert buffer.getvalue().count("stream monitor") == 1


def test_monitor_never_writes_to_the_session(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)], control={"temperature": 1.0})
    before = {p.name: p.read_bytes() for p in session.iterdir()}
    run_monitor(session, once=True, stream=io.StringIO())
    after = {p.name: p.read_bytes() for p in session.iterdir()}
    assert before == after


def test_sparkline_bucket_averages_a_long_history() -> None:
    line = sparkline([float(i) for i in range(100)], width=10)
    assert len(line) == 10
    assert line[0] == "▁" and line[-1] == "█"


def test_sparkline_keeps_gaps_when_downsampling() -> None:
    values: list[float | None] = [None] * 10 + [1.0] * 10
    line = sparkline(values, width=2)
    # One bucket has no readings at all; the other is flat, so it sits on the floor.
    assert line == " ▁"


def test_render_frame_labels_a_binned_history(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(i) for i in range(200)])
    frame = render_frame(read_snapshot(session, history=200), width=78)
    assert "last 200 rounds" in frame
    assert "per column" in frame


# ── events pane ───────────────────────────────────────────────────────


def _event(kind: str, message: str, *, level: str = "info", round_idx: int = 3) -> dict:
    return {
        "timestamp": _stamp(),
        "round_idx": round_idx,
        "kind": kind,
        "level": level,
        "message": message,
    }


def _write_events(session: Path, events: list[dict]) -> None:
    (session / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_events_box_is_always_the_requested_height() -> None:
    for count in (0, 1, 5, 20):
        events = [_event("throttle", f"event {i}") for i in range(count)]
        rendered = render_events(events, height=8, width=80)
        assert len(rendered) == 9  # title + a box that never changes size


def test_events_box_keeps_the_newest_events() -> None:
    events = [_event("throttle", f"event {i}") for i in range(20)]
    rendered = "\n".join(render_events(events, height=3, width=80))
    assert "event 19" in rendered
    assert "event 16" not in rendered


def test_events_box_marks_severity() -> None:
    rendered = render_events([_event("playback_error", "stream write failed", level="error")], height=1, width=80)
    assert "✗" in rendered[1]


def test_render_frame_shows_events_from_the_sidecar(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)])
    _write_events(session, [_event("model_swap", ".dora/xps/a -> .dora/xps/b")])
    frame = render_frame(read_snapshot(session))
    assert "model_swap: .dora/xps/a -> .dora/xps/b" in frame
    assert "derived from round rows" not in frame


def test_render_frame_keeps_a_constant_height_as_events_arrive(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)])
    _write_events(session, [_event("throttle", "waiting")])
    short = render_frame(read_snapshot(session)).splitlines()
    _write_events(session, [_event("throttle", f"event {i}") for i in range(50)])
    long = render_frame(read_snapshot(session)).splitlines()
    assert len(short) == len(long)


def test_events_box_can_be_switched_off(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1)])
    _write_events(session, [_event("throttle", "waiting")])
    frame = render_frame(read_snapshot(session), events_height=0)
    assert "events" not in frame


# ── derived events (sessions predating the sidecar) ───────────────────


def test_derive_events_reconstructs_what_round_rows_record() -> None:
    rows = [
        _row(1, model_xp=".dora/xps/a"),
        _row(2, model_xp=".dora/xps/b", tail_rescued_from_seconds=10.0),
        _row(3, model_xp=".dora/xps/b", motif_source="/bank/motif_a.wav"),
        _row(
            4,
            model_xp=".dora/xps/b",
            elapsed_seconds=16.0,
            playback_seconds=10.0,
            active_control={"completion_duration_seconds": 15.0},
        ),
    ]
    kinds = [event["kind"] for event in derive_events(rows)]
    assert kinds == ["model_swap", "tail_rescue", "motif_recap", "generation_slow_warning"]


def test_derive_events_recognizes_all_injection_commands() -> None:
    for command in ("inject", "inject_light", "inject_heavy"):
        events = derive_events([_row(1, command_consumed=command, motif_source="/bank/motif.wav")])
        assert [event["kind"] for event in events] == ["pinned_injection"]


def test_derive_events_ignores_a_steady_session() -> None:
    assert derive_events([_row(i) for i in range(5)]) == []


def test_snapshot_falls_back_to_derived_events(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1), _row(2, tail_rescued_from_seconds=9.0)])
    snapshot = read_snapshot(session)
    assert snapshot.events_derived is True
    assert snapshot.events[-1]["kind"] == "tail_rescue"
    assert "derived from round rows" in render_frame(snapshot)


def test_sidecar_wins_over_derived_events(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1, tail_rescued_from_seconds=9.0)])
    _write_events(session, [_event("throttle", "waiting")])
    snapshot = read_snapshot(session)
    assert snapshot.events_derived is False
    assert [event["kind"] for event in snapshot.events] == ["throttle"]


# ── follow mode: which round the metrics describe ─────────────────────


def _played_session(tmp_path: Path) -> Path:
    # 25s buffered at 10s per round puts the player three rounds back from 5.
    rows = [_row(i, buffer_seconds_ahead=25.0, playback_seconds=10.0) for i in range(1, 6)]
    rows[2]["score"] = {"arc_coherence_score": 0.111, "novelty_whitened": 0.222}
    rows[4]["score"] = {"arc_coherence_score": 0.999, "novelty_whitened": 0.888}
    return _write_session(tmp_path, rows)


def test_follow_generated_is_the_default_and_describes_the_newest_round(tmp_path: Path) -> None:
    frame = render_frame(read_snapshot(_played_session(tmp_path)))
    assert "metrics · round 5 · last generated" in frame
    assert "0.999" in frame and "0.111" not in frame


def test_follow_playing_describes_the_audible_round(tmp_path: Path) -> None:
    frame = render_frame(read_snapshot(_played_session(tmp_path)), follow="playing")
    assert "round 3" in frame
    assert "NOW PLAYING" in frame
    assert "0.111" in frame and "0.999" not in frame  # the played round's metrics, not the newest


def test_follow_playing_says_it_lags(tmp_path: Path) -> None:
    # The whole risk of this mode is mistaking a lagging number for a live one.
    frame = render_frame(read_snapshot(_played_session(tmp_path)), follow="playing")
    line = next(line for line in frame.splitlines() if "metrics ·" in line)
    assert "lags the generator" in line
    assert "2 behind" in line


def test_follow_playing_falls_back_when_nothing_is_playing(tmp_path: Path) -> None:
    session = _write_session(tmp_path, [_row(1, buffer_seconds_ahead=None)])
    frame = render_frame(read_snapshot(session), follow="playing")
    assert "last generated (nothing playing)" in frame


def test_follow_playing_admits_when_the_buffer_predates_the_loaded_history(tmp_path: Path) -> None:
    # 200s buffered but only 3 rounds (30s) loaded: the true audible round is
    # older than anything on hand, so the label must not claim to know it.
    rows = [_row(i, buffer_seconds_ahead=200.0, playback_seconds=10.0) for i in range(40, 46)]
    session = _write_session(tmp_path, rows)
    frame = render_frame(read_snapshot(session, history=3), follow="playing")
    assert "or earlier" in frame
    assert "predates --history" in frame
    assert "NOW PLAYING" not in frame


def test_raw_novelty_has_no_row_of_its_own(tmp_path: Path) -> None:
    # ρ = +0.64 with whitened and a dynamic range of ~0.004: a research
    # diagnostic, kept in the sidecar, off the panel.
    session = _write_session(tmp_path, [_row(1, score={"novelty_whitened": 0.30, "novelty_raw": 0.004})])
    frame = render_frame(read_snapshot(session))
    assert "novelty_raw" not in frame
    assert "0.300" in frame


def test_novelty_row_falls_back_to_raw_when_uncalibrated(tmp_path: Path) -> None:
    # No calibration, so `novelty_whitened` is null: the row shows the raw
    # cosine rather than going blank.
    session = _write_session(tmp_path, [_row(1, score={"novelty_whitened": None, "novelty_raw": 0.004})])
    frame = render_frame(read_snapshot(session))
    novelty_line = next(line for line in frame.splitlines() if line.strip().startswith("novelty "))
    assert "0.004" in novelty_line


# ── blind mode ────────────────────────────────────────────────────────


def _blind_session(tmp_path: Path) -> Path:
    rows = [_row(i, buffer_seconds_ahead=25.0, playback_seconds=10.0) for i in range(1, 4)]
    rows[-1]["score"] = {
        "arc_coherence_score": 0.777,
        "timbral_coherence": 0.942,
        "novelty_whitened": 0.321,
        "silence_fraction": 0.05,
        "candidates": [{"candidate": 0, "selected": True, "arc_coherence_score": 0.777, "novelty": 0.321}],
    }
    return _write_session(tmp_path, rows, control={"temperature": 1.5, "tau_seconds": 5.0})


def test_blind_mode_hides_every_per_round_reading(tmp_path: Path) -> None:
    frame = render_frame(read_snapshot(_blind_session(tmp_path)), show_metrics=False)
    assert "playing blind" in frame
    for leaked in ("0.777", "0.942", "0.321", "metrics ·", "candidates", "lurching", "steady"):
        assert leaked not in frame, f"blind frame leaked {leaked!r}"


def test_blind_mode_keeps_everything_that_is_not_a_judgement(tmp_path: Path) -> None:
    frame = render_frame(read_snapshot(_blind_session(tmp_path)), show_metrics=False, events_height=0)
    assert "buffer" in frame  # buffer health is operational, not a reading
    assert "control" in frame
    assert "hearing round" in frame


def test_metrics_are_shown_by_default(tmp_path: Path) -> None:
    frame = render_frame(read_snapshot(_blind_session(tmp_path)))
    assert "0.777" in frame
    assert "playing blind" not in frame


# ── timbral coherence as an alarm ─────────────────────────────────────


@pytest.mark.parametrize(
    ("timbral", "expected", "unexpected"),
    [
        # 63% of rounds live up here, where the three decimals carry nothing.
        (0.9993, ["steady"], ["0.999"]),
        # Below the line the number is back exactly where it means something.
        (0.942, ["lurching", "0.942"], ["steady"]),
        (None, [], ["steady", "lurching"]),  # unrecorded: the row stays blank
    ],
)
def test_timbral_coherence_row_is_steady_lurching_or_blank(
    tmp_path: Path, timbral: float | None, expected: list[str], unexpected: list[str]
) -> None:
    score = {"arc_coherence_score": 0.5} if timbral is None else {"timbral_coherence": timbral}
    session = _write_session(tmp_path, [_row(1, score=score)])
    line = next(line for line in render_frame(read_snapshot(session)).splitlines() if "timbral_coh" in line)
    for word in expected:
        assert word in line
    for word in unexpected:
        assert word not in line


# ── guidance block ────────────────────────────────────────────────────


def _frame_with_control(tmp_path: Path, control: dict) -> str:
    session = _write_session(tmp_path, [_row(1)], control=control)
    return render_frame(read_snapshot(session), width=120, events_height=0)


def test_frame_warns_when_the_window_overruns_the_horizon(tmp_path: Path) -> None:
    frame = _frame_with_control(
        tmp_path,
        {
            "model_xp": ".dora/xps/17087d6a",  # trained at 10s
            "tau_seconds": 5.0,
            "completion_duration_seconds": 10.0,
            "temperature": 1.6,
            "top_k": 250,
        },
    )
    assert "guidance" in frame
    assert "overruns" in frame and "10s horizon" in frame


def test_frame_surfaces_the_temperature_floor_and_top_k_cliff(tmp_path: Path) -> None:
    frame = _frame_with_control(
        tmp_path,
        {
            "model_xp": ".dora/xps/8aa68bbe",
            "tau_seconds": 5.0,
            "completion_duration_seconds": 5.0,
            "temperature": 1.0,
            "top_k": 25,
        },
    )
    assert "below the 1.4 floor" in frame
    assert "silence cliff" in frame


def test_frame_stays_quiet_on_settled_settings(tmp_path: Path) -> None:
    frame = _frame_with_control(
        tmp_path,
        {
            "model_xp": ".dora/xps/4ae63477",
            "tau_seconds": 5.0,
            "completion_duration_seconds": 5.0,
            "temperature": 1.0,
            "top_k": 250,
        },
    )
    assert "guidance" not in frame
