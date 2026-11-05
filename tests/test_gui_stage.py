from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from macchina_ricordi.gui import stage
from macchina_ricordi.gui.data import GuiIndex
from macchina_ricordi.gui.server import serve
from macchina_ricordi.gui.stage import LaunchRequest, StageError, StageManager, session_state
from macchina_ricordi.instrument.stream.session import StreamControl, read_control, write_control

SESSION = "stream_20261022T120000Z"

# Stands in for `play stream`: announces its session folder like the CLI, then keeps a live status fresh, and
# ignores the stop command, so only the escalation's Ctrl-C ends it.
FAKE_STREAM = """
import json, os, sys, time
from datetime import datetime, timezone
folder = sys.argv[1]
print("session_dir: " + folder)
while True:
    status = {"pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), "phase": "generating"}
    with open(os.path.join(folder, "live_status.tmp"), "w") as handle:
        json.dump(status, handle)
    os.replace(os.path.join(folder, "live_status.tmp"), os.path.join(folder, "live_status.json"))
    time.sleep(0.1)
"""


def _xp(repo: Path, sig: str = "aaaa1111") -> None:
    folder = repo / ".dora" / "xps" / sig
    folder.mkdir(parents=True)
    (folder / ".argv.json").write_text(json.dumps(["dataset.segment_duration=15"]), encoding="utf-8")
    (folder / "checkpoint.th").write_bytes(b"")


def _session(repo: Path, *, live: dict | None = None, rounds: int = 6) -> Path:
    """A stream session folder with ``rounds`` rounds; ``live`` fields make this process its live status writer."""
    folder = repo / "artifacts" / "sessions" / "2026-10-22" / SESSION
    (folder / "rounds").mkdir(parents=True)
    now = datetime.now(UTC)
    rows = [
        {
            "round_idx": i,
            "take_idx": 0,
            "timestamp": (now - timedelta(seconds=10 * (rounds - i))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "round_wav": f"rounds/round_{i:03d}.wav",
            "playback_seconds": 10.0,
            "elapsed_seconds": 2.0,
            "buffer_seconds_ahead": 30.0,
            "score": {"arc_coherence_score": 0.1 * i, "silence_fraction": 0.0},
        }
        for i in range(rounds)
    ]
    (folder / "performance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_control(folder / "control.json", StreamControl())
    if live is not None:
        status = {"pid": os.getpid(), "updated_at": now.isoformat(), **live}
        (folder / "live_status.json").write_text(json.dumps(status), encoding="utf-8")
    return folder


def test_a_session_is_live_only_while_its_process_keeps_its_status_fresh(tmp_path: Path) -> None:
    folder = _session(tmp_path, live={"phase": "generating"})
    status = json.loads((folder / "live_status.json").read_text(encoding="utf-8"))
    now = time.time()
    assert session_state(folder, status, None, now=now) == "live"
    assert session_state(folder, {**status, "phase": "stopped"}, None, now=now) == "stopping"
    assert session_state(folder, status, None, now=now + 60) == "unfinished"  # stopped writing, never finished
    assert session_state(folder, status, {"stopped_by": "command"}, now=now + 60) == "ended"
    gone = subprocess.Popen(["true"])
    gone.wait()
    assert session_state(folder, {**status, "pid": gone.pid}, None, now=now) == "unfinished"
    # Started before live_status.json existed: a log still being appended to is the only sign of life.
    assert session_state(folder, None, None, now=now) == "live"
    assert session_state(folder, None, None, now=now + 600) == "unfinished"


def test_keys_go_through_the_tui_keymap_and_pin_what_is_playing(tmp_path: Path) -> None:
    folder = _session(tmp_path, live={"phase": "generating", "playing": "/x/rounds/round_004.wav"})
    stage = StageManager(tmp_path)
    result = stage.press(SESSION, ["up", "up", "N", "x"])
    control = read_control(folder / "control.json")
    assert (control.temperature, control.novelty_weight) == (pytest.approx(1.1), pytest.approx(0.05))
    assert result["message"] == "novelty_weight 0.00 → 0.05"

    stage.press(SESSION, ["p"])
    control = read_control(folder / "control.json")
    assert (control.command, control.pin_target) == ("pin", "/x/rounds/round_004.wav")


def test_a_slider_keeps_to_its_knob_limits_and_an_ended_session_is_only_watched(tmp_path: Path) -> None:
    folder = _session(tmp_path, live={"phase": "generating"})
    stage = StageManager(tmp_path)
    stage.set(SESSION, "top_k", 5000)
    assert read_control(folder / "control.json").top_k == 2048
    with pytest.raises(StageError):
        stage.set(SESSION, "cfg_coef", 1.0)  # no knob, so no key either

    (folder / "live_status.json").unlink()
    (folder / "manifest.json").write_text(json.dumps({"stopped_by": "command"}), encoding="utf-8")
    with pytest.raises(StageError, match="not playing"):
        stage.press(SESSION, ["up"])
    assert read_control(folder / "control.json").temperature == 1.0


def test_the_frame_hears_the_round_the_live_status_names(tmp_path: Path) -> None:
    _session(tmp_path, live={"phase": "throttled", "playing": "/x/rounds/round_003.wav", "buffer_seconds_ahead": 21.5})
    frame = StageManager(tmp_path).frame(SESSION, follow="playing")
    assert (frame["state"], frame["exact"]) == ("live", True)
    head = frame["head"]
    assert (head["audible_round"], head["rounds_behind"], head["edit_lands_at"]) == (3, 2, 6)
    assert frame["focus"]["round_idx"] == 3
    json.dumps(frame, allow_nan=False)  # what the server sends


def test_the_frame_says_where_an_edit_lands_and_what_each_round_changed(tmp_path: Path) -> None:
    live = {"phase": "generating", "round_idx": 6, "playing": "/x/rounds/round_004.wav", "buffer_seconds_ahead": 15.0}
    folder = _session(tmp_path, live=live)
    rows = [json.loads(line) for line in (folder / "performance.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, row in enumerate(rows):
        row["active_control"] = {"temperature": 1.2 if i >= 4 else 1.0, "command": "pin" if i == 4 else "continue"}
    rows[3]["command_consumed"] = "reroll"
    (folder / "performance.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    frame = StageManager(tmp_path).frame(SESSION)
    assert frame["head"]["edit_lands_at"] == 7  # round 6 already read its controls
    history = frame["history"]
    assert history["edits"][4] == [["temperature", 1.0, 1.2]] and history["edits"][5] == []
    assert history["marks"][3] == ["reroll"]

    status = json.loads((folder / "live_status.json").read_text(encoding="utf-8"))
    (folder / "live_status.json").write_text(
        json.dumps({**status, "phase": "throttled", "round_idx": 7}), encoding="utf-8"
    )
    assert StageManager(tmp_path).frame(SESSION)["head"]["edit_lands_at"] == 7  # waiting to read them, so this round


def test_a_preview_resolves_the_models_and_names_what_else_holds_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _xp(tmp_path)
    _session(tmp_path, live={"phase": "generating"})  # a stream already playing
    monkeypatch.setattr(stage, "gpu_sample", lambda: {"available": False})
    manager = StageManager(tmp_path, active_trainings=lambda: ["king_night"])
    plan = manager.preview(
        {"models": ["aaaa1111", "aaaa1111@final", "nope"], "tau_seconds": 6, "completion_duration_seconds": 12}
    )
    assert len(plan["errors"]) == 1 and plan["errors"][0].startswith("3: no model called 'nope'")
    assert plan["models"][0]["segment_duration"] == 15.0
    assert "@final is ignored" in plan["models"][1]["note"]
    assert any("overruns" in advisory["message"] for advisory in plan["guidance"])
    assert any("king_night" in w for w in plan["warnings"]) and any(SESSION in w for w in plan["warnings"])
    assert "--playlist aaaa1111,aaaa1111@final,nope" in plan["command"]
    with pytest.raises(StageError, match="temperature"):
        LaunchRequest.from_payload({"models": ["aaaa1111"], "temperature": 9})


def test_a_bundle_is_selectable_for_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from macchina_ricordi.models.bundle import write_bundle

    write_bundle(
        tmp_path / "models" / "drummer",
        {"w": np.zeros((2,), dtype="float32")},
        model_config={
            "lm_model": "transformer_lm",
            "transformer_lm": {"dim": 192, "num_heads": 3, "num_layers": 10, "hidden_scale": 4, "n_q": 4, "card": 2048},
            "dtype": "float32",
            "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
            "attribute_dropout": {},
            "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
            "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
            "conditioners": {},
        },
        codec={"name": "facebook/encodec_32khz"},
        audio={"sample_rate": 32000, "channels": 1, "frame_rate": 50.0},
        segment_duration=7.0,
        name="drummer",
    )
    monkeypatch.setattr(stage, "gpu_sample", lambda: {"available": True})
    plan = StageManager(tmp_path).preview({"models": ["drummer"]})
    assert plan["errors"] == []
    assert plan["models"][0]["kind"] == "bundle"
    assert plan["models"][0]["folder"].endswith("models/drummer")
    assert plan["models"][0]["segment_duration"] == 7.0


def test_a_launch_runs_detached_and_a_stop_it_ignores_gets_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _xp(tmp_path)
    session = tmp_path / "artifacts" / "sessions" / "2026-10-22" / "stream_20261022T130000Z"
    session.mkdir(parents=True)
    write_control(session / "control.json", StreamControl())
    script = tmp_path / "fake_stream.py"
    script.write_text(FAKE_STREAM, encoding="utf-8")
    monkeypatch.setattr(stage, "cli_argv", lambda cli: [sys.executable, "-u", str(script), str(session)])
    monkeypatch.setattr(stage, "STOP_ESCALATION", ((0.3, signal.SIGINT), (3.0, signal.SIGTERM), (3.0, signal.SIGKILL)))
    manager = StageManager(tmp_path)
    with pytest.raises(StageError, match="needs one"):
        LaunchRequest.from_payload({"models": ["aaaa1111"], "buffer_floor_ratio": 0.5})

    launched = manager.launch(
        {
            "models": ["aaaa1111"],
            "temperature": 1.4,
            "round_delay_seconds": 0.5,
            "max_buffer_ahead_seconds": 30,
            "buffer_floor_ratio": 0.5,
        }
    )
    assert launched["id"] == session.name
    assert "--temperature 1.4" in launched["command"]
    assert "--round-delay-seconds 0.5 --max-buffer-ahead-seconds 30 --buffer-floor-ratio 0.5" in launched["command"]
    assert json.loads((session / "launch.json").read_text(encoding="utf-8"))["job"] == launched["job"]
    process = manager._jobs.child(launched["job"])
    assert process is not None and process.poll() is None

    manager.press(session.name, ["Q"])
    assert read_control(session / "control.json").command == "stop"
    process.wait(timeout=10)


def test_a_stream_that_never_starts_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _xp(tmp_path)
    failing = [sys.executable, "-c", "print('playback_unavailable: no PortAudio here'); raise SystemExit(1)"]
    monkeypatch.setattr(stage, "cli_argv", lambda cli: failing)
    with pytest.raises(StageError, match="no PortAudio here"):
        StageManager(tmp_path).launch({"models": ["aaaa1111"]})


def test_the_page_reads_and_steers_through_the_server(tmp_path: Path) -> None:
    _session(tmp_path, live={"phase": "generating"})
    server = serve(GuiIndex(tmp_path), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/stage"
    try:
        with urllib.request.urlopen(base) as response:
            assert json.load(response)["sessions"][0]["state"] == "live"
        request = urllib.request.Request(
            f"{base}/{SESSION}/keys",
            data=json.dumps({"keys": ["]"]}).encode(),
            headers={"Content-Type": "application/json", "X-Macchina-Ricordi": "1"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["control"]["completion_duration_seconds"] == 11.0
    finally:
        server.shutdown()
        server.server_close()
