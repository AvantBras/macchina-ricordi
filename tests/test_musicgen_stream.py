from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.instrument.saving import _write_wav
from macchina_ricordi.instrument.stream.playback import NullPlaybackWorker
from macchina_ricordi.instrument.stream.session import (
    LiveStatusWriter,
    StreamControl,
    StreamStatus,
    _rest_before_round,
    build_stream_session_paths,
    consume_command,
    extract_tail,
    read_control,
    run_stream_session,
    write_control,
)
from macchina_ricordi.instrument.stream.worker import StreamGenerationResult, StreamModelSource


def _dummy_generator() -> _DummyStreamGenerator:
    return _DummyStreamGenerator()


class _DummyStreamGenerator:
    """Writes a fixed 1s WAV and records every request received."""

    sample_rate = 32_000
    frames = 32_000

    def __init__(self) -> None:
        self.requests: list = []
        self.sources: list[StreamModelSource] = []
        self.output_paths: list[Path] = []
        self.closed = False

    def generate(self, request, *, output_path: Path, source: StreamModelSource):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        self.sources.append(source)
        self.output_paths.append(output_path)
        _write_wav(output_path, np.zeros((1, self.frames), dtype=np.float32), self.sample_rate)
        return StreamGenerationResult(
            output_path=output_path,
            sample_rate=self.sample_rate,
            backend="audiocraft",
            model_id="dummy/micro",
            device="cpu",
            warnings=[],
            metadata={"mode": request.mode, "source": {"kind": "xp"}},
        )

    def close(self) -> None:
        self.closed = True


class _LongRoundGenerator(_DummyStreamGenerator):
    """Writes rounds long enough that a large tau is not clamped by file length."""

    frames = 32_000 * 20


def test_control_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    original = StreamControl(
        command="reroll",
        tau_seconds=4.0,
        completion_duration_seconds=12.0,
        temperature=1.3,
        top_k=80,
        top_p=0.1,
        cfg_coef=2.5,
        extend_stride=6.0,
        model_xp="/dora/xps/abcd1234",
    )
    write_control(path, original)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["completion_duration_seconds"] == 12.0
    assert "round_duration_seconds" not in payload
    parsed = read_control(path)
    assert parsed == original


def test_read_control_fills_defaults(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"temperature": 1.5}), encoding="utf-8")
    parsed = read_control(path)
    assert parsed.temperature == 1.5
    assert parsed.command == "continue"
    assert parsed.completion_duration_seconds == pytest.approx(10.0)
    assert parsed.top_k == StreamControl().top_k
    assert parsed.model_xp is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not json", "not valid JSON"),
        (json.dumps({"command": "destroy"}), "invalid command"),
        (json.dumps({"round_duration_seconds": 15.0}), "replace it with 'completion_duration_seconds'"),
    ],
)
def test_read_control_rejects_bad_input(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "control.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_control(path)


def test_consume_command_resets_one_shot_and_keeps_every_other_setting(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    control = StreamControl(
        command="pin",
        pin_target="/tmp/round_004.wav",
        temperature=1.7,
        model_xp="/dora/xps/abcd1234",
        candidates=3,
        min_p=0.08,
        novelty_weight=0.4,
        motif_period=5,
        motif_mix=0.3,
    )
    write_control(path, control)
    consume_command(path, control)
    parsed = read_control(path)
    assert parsed.command == "continue"
    assert parsed.pin_target is None
    assert parsed.temperature == 1.7
    assert parsed.model_xp == "/dora/xps/abcd1234"
    assert parsed.candidates == 3
    assert parsed.min_p == 0.08
    assert parsed.novelty_weight == 0.4
    assert parsed.motif_period == 5
    assert parsed.motif_mix == 0.3


def test_consume_command_keeps_what_changed_since_the_round_read_the_file(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    round_copy = StreamControl(command="pin", pin_target="/tmp/round_004.wav")
    # A knob moved after the round read the file: resetting the pin must not undo it.
    write_control(path, StreamControl(command="pin", pin_target="/tmp/round_004.wav", temperature=1.4))
    consume_command(path, round_copy)
    parsed = read_control(path)
    assert (parsed.command, parsed.temperature) == ("continue", 1.4)

    # A newer one-shot is left for the next round to act on.
    write_control(path, StreamControl(command="reroll"))
    consume_command(path, round_copy)
    assert read_control(path).command == "reroll"


@pytest.mark.parametrize(
    ("samples", "tau", "expected_start", "expected_duration"),
    [
        (32_000, 0.3, 0.7, 0.3),  # the last window of a 1 s file
        (16_000, 2.0, 0.0, 0.5),  # a tau longer than the 0.5 s file is clamped to it
    ],
)
def test_extract_tail_returns_the_last_window_clamped_to_the_file(
    tmp_path: Path, samples: int, tau: float, expected_start: float, expected_duration: float
) -> None:
    wav_path = tmp_path / "source.wav"
    _write_wav(wav_path, np.zeros((1, samples), dtype=np.float32), 32_000)
    start, dur = extract_tail(wav_path, tau_seconds=tau)
    assert dur == pytest.approx(expected_duration, rel=1e-3)
    assert start == pytest.approx(expected_start, abs=1e-3)


def test_stream_session_writes_artifacts(tmp_path: Path) -> None:
    generator = _dummy_generator()
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260525T100000Z")

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
            temperature=1.0,
            top_k=250,
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=42,
        max_rounds=2,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    assert (paths.rounds_dir / "round_000.wav").exists()
    assert (paths.rounds_dir / "round_001.wav").exists()
    assert (paths.rounds_dir / "round_002.wav").exists()
    assert result.rounds_written == 3
    assert result.stopped_by == "max_rounds"

    perf_lines = paths.performance_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(perf_lines) == 3
    indices = [json.loads(line)["round_idx"] for line in perf_lines]
    assert indices == [0, 1, 2]

    # Round 0 = unconditional, rounds 1+ = continuation with prompt_audio pointing back.
    requests = generator.requests
    assert requests[0].mode == "unconditional"
    assert requests[0].duration_seconds == pytest.approx(1.0)
    for k in range(1, len(requests)):
        assert requests[k].mode == "continuation"
        assert requests[k].prompt_duration_seconds == pytest.approx(0.3)
        assert requests[k].duration_seconds == pytest.approx(1.3)
        assert requests[k].prompt_audio_path is not None
        assert Path(requests[k].prompt_audio_path).name == f"round_{k - 1:03d}.wav"

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "macchina_ricordi.stream_session.v1"
    assert manifest["initial_control"]["completion_duration_seconds"] == 1.0
    assert "round_duration_seconds" not in manifest["initial_control"]
    assert manifest["rounds_written"] == 3
    assert manifest["stopped_by"] == "max_rounds"
    assert manifest["seed_mode"] == "uncond"
    assert len(generator.sources) == 3
    assert generator.closed is False

    live = json.loads((paths.session_dir / "live_status.json").read_text(encoding="utf-8"))
    assert (live["phase"], live["playing"]) == ("stopped", None)


def test_live_status_tells_another_process_what_only_the_loop_knows(tmp_path: Path) -> None:
    playing = tmp_path / "rounds" / "round_007.wav"

    class _Player(NullPlaybackWorker):
        def current_path(self) -> Path | None:
            return playing

        def buffer_seconds_ahead(self) -> float | None:
            return 12.5

    status = StreamStatus()
    status.set_phase("throttled", round_idx=9)
    emitted: list[str] = []
    writer = LiveStatusWriter(
        tmp_path / "live_status.json",
        status=status,
        player=_Player(),
        emit=lambda kind, *_args, **_data: emitted.append(kind),
    )
    writer.write()
    live = json.loads((tmp_path / "live_status.json").read_text(encoding="utf-8"))
    assert live["pid"] == os.getpid()
    assert (live["phase"], live["round_idx"], live["playing"], live["buffer_seconds_ahead"]) == (
        "throttled",
        9,
        str(playing),
        12.5,
    )

    # A write that fails is reported once and never raised into the round loop.
    (tmp_path / "blocked").write_text("", encoding="utf-8")
    writer.path = tmp_path / "blocked" / "live_status.json"
    writer.write()
    writer.write()
    assert emitted == ["live_status_failed"]


def test_stream_session_seed_excerpt_mode_copies_seed(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.wav"
    _write_wav(seed_path, np.zeros((1, 32_000), dtype=np.float32), 32_000)

    generator = _dummy_generator()
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260525T100001Z")

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
        ),
        seed_mode="excerpt",
        seed_audio=seed_path,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=None,
        max_rounds=1,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    assert (paths.rounds_dir / "round_000.wav").exists()
    assert (paths.rounds_dir / "round_001.wav").exists()
    assert result.rounds_written == 2
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed_mode"] == "excerpt"
    assert manifest["seed_audio_source"] == str(seed_path)
    assert [request.mode for request in generator.requests] == ["continuation"]
    assert generator.requests[0].prompt_duration_seconds == pytest.approx(0.3)
    assert generator.requests[0].duration_seconds == pytest.approx(1.3)


def test_reroll_creates_take_file(tmp_path: Path) -> None:
    generator = _dummy_generator()
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260525T100002Z")

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="reroll",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=7,
        max_rounds=2,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    assert (paths.rounds_dir / "round_000.wav").exists()
    assert (paths.rounds_dir / "round_001.wav").exists()
    assert (paths.rounds_dir / "round_001_take_1.wav").exists()
    assert (paths.rounds_dir / "round_002.wav").exists()
    # Seed + 3 continuation writes (round 1 take 0, round 1 take 1, round 2 take 0)
    assert result.rounds_written == 4
    # After the reroll consume, control file should be back to continue.
    assert read_control(paths.control_path).command == "continue"


def test_max_rounds_zero_treated_as_infinite_until_command(tmp_path: Path) -> None:
    """When max_rounds=0 and command stays continue, the loop should keep going.

    We trigger a stop after the first continuation by patching the backend to flip the
    control file mid-flight.
    """
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260525T100004Z")

    class StoppingGenerator(_DummyStreamGenerator):
        def __init__(self, control_path: Path) -> None:
            super().__init__()
            self.control_path = control_path
            self.calls = 0

        def generate(self, request, *, output_path: Path, source: StreamModelSource):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 2:  # after round 1 continuation
                stopper = StreamControl(command="stop", tau_seconds=0.3, completion_duration_seconds=1.0)
                write_control(self.control_path, stopper)
            return super().generate(request, output_path=output_path, source=source)

    generator = StoppingGenerator(paths.control_path)

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(command="continue", tau_seconds=0.3, completion_duration_seconds=1.0),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=None,
        max_rounds=0,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    assert result.rounds_written == 2  # seed + 1 continuation, then stop took effect
    assert result.stopped_by == "command"


def test_stream_session_closes_owned_generator(monkeypatch, tmp_path: Path) -> None:
    generator = _dummy_generator()
    monkeypatch.setattr(
        "macchina_ricordi.instrument.stream.session.PersistentStreamGenerator", lambda **_kwargs: generator
    )
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260525T100005Z")

    run_stream_session(
        paths=paths,
        initial_control=StreamControl(command="continue", tau_seconds=0.3, completion_duration_seconds=1.0),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=None,
        max_rounds=1,
        score_rounds=False,
        player=NullPlaybackWorker(),
    )

    assert generator.closed is True


def test_stream_session_swaps_xp_when_control_changes(tmp_path: Path) -> None:
    initial_xp = tmp_path / "xp_initial"
    initial_xp.mkdir()
    (initial_xp / "checkpoint.th").write_text("", encoding="utf-8")
    new_xp = tmp_path / "xp_swapped"
    new_xp.mkdir()
    (new_xp / "checkpoint.th").write_text("", encoding="utf-8")

    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260526T100000Z")

    # Rewrite the control file after the seed's backend call so round 1 sees the swap request.
    class SwappingGenerator(_DummyStreamGenerator):
        def __init__(self, control_path: Path, target_xp: str) -> None:
            super().__init__()
            self.control_path = control_path
            self.target_xp = target_xp
            self.calls = 0

        def generate(self, request, *, output_path: Path, source: StreamModelSource):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                write_control(
                    self.control_path,
                    StreamControl(
                        command="continue",
                        tau_seconds=0.3,
                        completion_duration_seconds=1.0,
                        model_xp=self.target_xp,
                    ),
                )
            return super().generate(request, output_path=output_path, source=source)

    generator = SwappingGenerator(paths.control_path, str(new_xp))

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
            model_xp=str(initial_xp),
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name="micro-unconditional",
        xp_path=str(initial_xp),
        checkpoint="latest",
        checkpoint_path=None,
        base_seed=None,
        max_rounds=2,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    assert len(generator.sources) == 3
    assert generator.sources[0].xp_path == str(initial_xp.resolve())
    assert generator.sources[1].xp_path == str(new_xp.resolve())
    assert generator.sources[2].xp_path == str(new_xp.resolve())
    assert generator.sources[1].profile_name is not None

    # Manifest captures the swap.
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_xp_path"] == str(new_xp)
    assert manifest["swap_count"] == 1
    assert result.rounds_written == 3

    # Performance log records the active model per round.
    rows = [json.loads(line) for line in paths.performance_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["model_xp"] == str(initial_xp)
    assert rows[1]["model_xp"] == str(new_xp)
    assert rows[2]["model_xp"] == str(new_xp)


def test_stream_session_invalid_model_xp_is_warning_only(tmp_path: Path, capsys) -> None:
    initial_xp = tmp_path / "xp_initial"
    initial_xp.mkdir()
    (initial_xp / "checkpoint.th").write_text("", encoding="utf-8")
    bogus = tmp_path / "does_not_exist"  # not created

    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260526T100001Z")

    class WriteBogusGenerator(_DummyStreamGenerator):
        def __init__(self, control_path: Path, target: str) -> None:
            super().__init__()
            self.control_path = control_path
            self.target = target
            self.calls = 0

        def generate(self, request, *, output_path: Path, source: StreamModelSource):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                write_control(
                    self.control_path,
                    StreamControl(
                        command="continue",
                        tau_seconds=0.3,
                        completion_duration_seconds=1.0,
                        model_xp=self.target,
                    ),
                )
            return super().generate(request, output_path=output_path, source=source)

    generator = WriteBogusGenerator(paths.control_path, str(bogus))

    result = run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
            model_xp=str(initial_xp),
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name="micro-unconditional",
        xp_path=str(initial_xp),
        checkpoint="latest",
        checkpoint_path=None,
        base_seed=None,
        max_rounds=2,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )

    captured = capsys.readouterr().out
    assert "model_swap_skipped" in captured
    # Generator should keep using the original xp_path.
    assert generator.sources[0].xp_path == str(initial_xp.resolve())
    assert generator.sources[1].xp_path == str(initial_xp.resolve())
    assert result.rounds_written == 3

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_xp_path"] == str(initial_xp)
    assert manifest["swap_count"] == 0


def test_stream_session_passes_tau_as_skip_seconds(tmp_path: Path) -> None:
    generator = _dummy_generator()
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260526T100002Z")

    player = NullPlaybackWorker()

    run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=0.3,
            completion_duration_seconds=1.0,
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name=None,
        xp_path=None,
        checkpoint=None,
        checkpoint_path=None,
        base_seed=None,
        max_rounds=2,
        score_rounds=False,
        player=player,
        generator=generator,
    )

    # Seed enqueued with skip_seconds=0; continuation rounds with skip_seconds == tau.
    assert len(player.enqueued) == 3
    seed_entry, round1_entry, round2_entry = player.enqueued
    assert seed_entry[1] == 0.0
    assert round1_entry[1] == pytest.approx(0.3, rel=1e-3)
    assert round2_entry[1] == pytest.approx(0.3, rel=1e-3)


# ── novelty as a reading: always computed, raw + whitened ─────────────


def _horizon_session(tmp_path: Path, *, tau_seconds: float, segment_duration: float) -> tuple:
    xp = tmp_path / "xps" / "seg_model"
    xp.mkdir(parents=True)
    (xp / ".argv.json").write_text(json.dumps([f"dataset.segment_duration={segment_duration:g}"]), encoding="utf-8")
    (xp / "checkpoint.th").write_text("", encoding="utf-8")
    paths = build_stream_session_paths(tmp_path / "streams", run_id="20260806T090000Z")
    generator = _LongRoundGenerator()
    run_stream_session(
        paths=paths,
        initial_control=StreamControl(
            command="continue",
            tau_seconds=tau_seconds,
            completion_duration_seconds=2.0,
            model_xp=str(xp),
        ),
        seed_mode="uncond",
        seed_audio=None,
        model_id=None,
        device=None,
        profile_name="micro-unconditional",
        xp_path=str(xp),
        checkpoint="latest",
        checkpoint_path=None,
        base_seed=None,
        max_rounds=2,
        score_rounds=False,
        player=NullPlaybackWorker(),
        generator=generator,
    )
    return paths, generator


@pytest.mark.parametrize(
    ("tau_seconds", "segment_duration", "prompt_seconds", "guard_count"),
    [
        # A 12 s tau into a 10 s-segment model is the AudioCraft abort, reachable from the
        # TUI's keys: it must never leave the process, and it reports once, not every round.
        (12.0, 10.0, 9.0, 1),
        (5.0, 15.0, 5.0, 0),
    ],
)
def test_horizon_guard_clamps_only_a_prompt_that_would_abort(
    tmp_path: Path, tau_seconds: float, segment_duration: float, prompt_seconds: float, guard_count: int
) -> None:
    paths, generator = _horizon_session(tmp_path, tau_seconds=tau_seconds, segment_duration=segment_duration)

    continuations = [r for r in generator.requests if r.mode == "continuation"]
    assert continuations
    for request in continuations:
        assert request.prompt_duration_seconds == pytest.approx(prompt_seconds)

    events = [json.loads(line) for line in paths.events_path.read_text().splitlines() if line.strip()]
    guards = [e for e in events if e["kind"] == "horizon_guard"]
    assert len(guards) == guard_count
    if guard_count:
        assert guards[0]["level"] == "warning"
        assert guards[0]["data"]["clamped_tau"] == prompt_seconds

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["segment_duration"] == segment_duration


def test_drone_guard_reads_held_notes_ranks_them_down_and_kicks(tmp_path: Path) -> None:
    from macchina_ricordi.instrument.stream.session import _select_best_candidate, drone_readings, stuck_kick

    sr = 16_000
    t = np.arange(3 * sr) / sr
    drone = drone_readings(0.5 * np.sin(2 * np.pi * 660 * t))
    texture = drone_readings(0.2 * np.random.default_rng(0).standard_normal(t.size))
    assert drone["drone_score"] > 0.6 > 0.26 > texture["drone_score"]

    control = StreamControl(temperature=1.0, top_k=200, stuck_kick_rounds=2)
    assert stuck_kick(control, [0.1, 0.65, 0.7], rounds_since_kick=5) == (1.4, 400)
    assert stuck_kick(control, [0.7, 0.2, 0.7], rounds_since_kick=5) is None
    assert stuck_kick(control, [0.65, 0.7], rounds_since_kick=1) is None  # no back-to-back kicks
    assert stuck_kick(StreamControl(), [0.9, 0.9], rounds_since_kick=5) is None  # off by default

    paths = [tmp_path / "held.wav", tmp_path / "moving.wav"]
    stationarity = {paths[0]: 0.98, paths[1]: 0.80}
    pick = lambda weight: _select_best_candidate(  # noqa: E731
        paths,
        prompt_duration_seconds=None,
        scorer=lambda p, **_: {"silence_fraction": 0.0, "arc_coherence_score": 1.0 if p == paths[0] else 0.9},
        tail_silence_extractor=lambda *_a, **_k: 0.0,
        drone_extractor=lambda p, **_: {"stationarity": stationarity[p], "chroma_top": 0.5, "drone_score": 0.0},
        stuck_weight=weight,
    )[0]
    assert pick(0.0) == 0
    assert pick(0.6) == 1


def test_rest_before_round_waits_cuts_short_when_the_buffer_runs_low_and_hears_stop(tmp_path: Path) -> None:
    import time

    class _Player(NullPlaybackWorker):
        ahead = 30.0

        def buffer_seconds_ahead(self) -> float | None:
            return self.ahead

    control_path = tmp_path / "control.json"
    write_control(control_path, StreamControl(round_delay_seconds=0.2))
    player = _Player()
    started = time.monotonic()
    assert _rest_before_round(player=player, control_path=control_path, delay=0.2, refill_seconds=2.0) is None
    assert time.monotonic() - started >= 0.2

    player.ahead = 1.5  # less than the 2s the next round takes to make: resting would go silent
    events: list[str] = []
    started = time.monotonic()
    _rest_before_round(
        player=player,
        control_path=control_path,
        delay=3.0,
        refill_seconds=2.0,
        events=type("E", (), {"emit": lambda self, kind, *_a, **_k: events.append(kind)})(),
    )
    assert events == ["rest_cut"] and time.monotonic() - started < 0.1

    player.ahead = 30.0
    write_control(control_path, StreamControl(command="stop", round_delay_seconds=3.0))
    stop = _rest_before_round(player=player, control_path=control_path, delay=3.0, refill_seconds=2.0)
    assert stop is not None and stop.command == "stop"
