from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from macchina_ricordi.training.plateau import (
    build_plateau_watcher_config,
    classify_plateau,
    extract_metric_value,
    read_history,
    run_plateau_watcher,
)


def test_extract_metric_value_supports_flat_and_nested_keys() -> None:
    assert extract_metric_value({"loss": 1.5}, "loss") == 1.5
    assert extract_metric_value({"valid": {"loss": 0.7}}, "valid.loss") == 0.7
    assert extract_metric_value({"valid": {"loss": 0.7}}, "loss") is None
    assert extract_metric_value({"valid": {"loss": "nope"}}, "valid.loss") is None
    assert extract_metric_value({}, "missing.path") is None


def test_classify_plateau_detects_no_improvement_after_patience() -> None:
    history = [
        {"valid": {"loss": 1.0}},
        {"valid": {"loss": 0.9}},
        {"valid": {"loss": 0.95}},
        {"valid": {"loss": 0.95}},
        {"valid": {"loss": 0.95}},
    ]
    report = classify_plateau(
        history,
        metric="valid.loss",
        patience_epochs=3,
        min_improvement=1e-4,
    )
    assert report.best_value == pytest.approx(0.9)
    assert report.best_epoch_index == 1
    assert report.latest_epoch_index == 4
    assert report.epochs_since_best == 3
    assert report.plateau_detected is True


def test_classify_plateau_respects_min_improvement() -> None:
    history = [
        {"valid": {"loss": 0.9}},
        {"valid": {"loss": 0.8999}},  # improvement < min_improvement
        {"valid": {"loss": 0.8998}},
    ]
    report = classify_plateau(
        history,
        metric="valid.loss",
        patience_epochs=2,
        min_improvement=0.001,
    )
    assert report.best_epoch_index == 0
    assert report.epochs_since_best == 2
    assert report.plateau_detected is True


def test_classify_plateau_no_history_is_not_a_plateau() -> None:
    report = classify_plateau([], metric="valid.loss", patience_epochs=1, min_improvement=1e-6)
    assert report.plateau_detected is False
    assert report.total_entries == 0


def test_classify_plateau_rejects_invalid_patience() -> None:
    with pytest.raises(ValueError, match="patience_epochs must be >= 1"):
        classify_plateau([], metric="x", patience_epochs=0, min_improvement=0)


def test_read_history_returns_empty_for_missing_or_invalid(tmp_path: Path) -> None:
    assert read_history(tmp_path / "missing.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert read_history(bad) == []
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"oops": True}), encoding="utf-8")
    assert read_history(wrong_shape) == []


def test_build_plateau_watcher_config_rejects_stop_without_pid(tmp_path: Path) -> None:
    xp_folder = tmp_path / "xp"
    xp_folder.mkdir()
    with pytest.raises(ValueError, match="action=stop requires --stop-pid"):
        build_plateau_watcher_config(
            sig="sig",
            xp_folder=xp_folder,
            action="stop",
            stop_pid=None,
        )


def test_build_plateau_watcher_config_rejects_invalid_action(tmp_path: Path) -> None:
    xp_folder = tmp_path / "xp"
    xp_folder.mkdir()
    with pytest.raises(ValueError, match="Invalid action"):
        build_plateau_watcher_config(sig="sig", xp_folder=xp_folder, action="explode")


def test_run_plateau_watcher_once_writes_status_and_events(tmp_path: Path) -> None:
    xp_folder = tmp_path / "xp"
    xp_folder.mkdir()
    history_path = xp_folder / "history.json"
    history_path.write_text(
        json.dumps([{"valid": {"loss": 1.0}}, {"valid": {"loss": 1.0}}]),
        encoding="utf-8",
    )

    config = build_plateau_watcher_config(
        sig="some_sig",
        xp_folder=xp_folder,
        patience_epochs=4,
        min_improvement=1e-4,
        once=True,
    )
    exit_code = run_plateau_watcher(config)

    assert exit_code == 0
    assert config.status_path.exists()
    status = json.loads(config.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "one_shot_done"
    assert status["sig"] == "some_sig"

    events = config.events_path.read_text(encoding="utf-8").splitlines()
    event_types = [json.loads(line)["event"] for line in events]
    assert event_types == ["watch_started", "progress", "watch_stopped"]


def test_run_plateau_watcher_triggers_stop_action_on_plateau(tmp_path: Path) -> None:
    xp_folder = tmp_path / "xp"
    xp_folder.mkdir()
    history_path = xp_folder / "history.json"
    history_path.write_text(
        json.dumps(
            [
                {"valid": {"loss": 1.0}},
                {"valid": {"loss": 0.9}},
                {"valid": {"loss": 0.9}},
                {"valid": {"loss": 0.9}},
                {"valid": {"loss": 0.9}},
            ]
        ),
        encoding="utf-8",
    )

    config = build_plateau_watcher_config(
        sig="plateau_sig",
        xp_folder=xp_folder,
        patience_epochs=3,
        min_improvement=1e-4,
        action="stop",
        stop_pid=12345,
        poll_seconds=0.1,
    )

    kill_calls: list[tuple[int, float]] = []

    def fake_kill(pid: int, grace: float) -> dict[str, Any]:
        kill_calls.append((pid, grace))
        return {"pid": pid, "sigterm_sent": True, "sigkill_sent": False}

    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    exit_code = run_plateau_watcher(
        config,
        sleep_fn=fake_sleep,
        kill_fn=fake_kill,
    )

    assert exit_code == 0
    assert kill_calls == [(12345, config.grace_seconds)]
    events = [json.loads(line) for line in config.events_path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["event"] for event in events]
    assert "plateau_detected" in event_types
    assert "stop_sent" in event_types
    stop_event = next(event for event in events if event["event"] == "stop_sent")
    assert stop_event["stop_pid"] == 12345
    status = json.loads(config.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "plateau_detected"


def test_run_plateau_watcher_loops_when_not_once_until_plateau(tmp_path: Path) -> None:
    xp_folder = tmp_path / "xp"
    xp_folder.mkdir()
    history_path = xp_folder / "history.json"
    history_path.write_text(
        json.dumps([{"valid": {"loss": 1.0}}]),
        encoding="utf-8",
    )

    config = build_plateau_watcher_config(
        sig="loop_sig",
        xp_folder=xp_folder,
        patience_epochs=2,
        min_improvement=1e-4,
        poll_seconds=0.1,
    )

    sleep_count = {"count": 0}

    def fake_sleep(seconds: float) -> None:
        sleep_count["count"] += 1
        if sleep_count["count"] == 1:
            history_path.write_text(
                json.dumps(
                    [
                        {"valid": {"loss": 1.0}},
                        {"valid": {"loss": 1.0}},
                        {"valid": {"loss": 1.0}},
                    ]
                ),
                encoding="utf-8",
            )

    exit_code = run_plateau_watcher(config, sleep_fn=fake_sleep)

    assert exit_code == 0
    assert sleep_count["count"] == 1
    events = [json.loads(line) for line in config.events_path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["event"] for event in events]
    assert event_types.count("progress") == 2
    assert "plateau_detected" in event_types
