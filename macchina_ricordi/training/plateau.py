from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..musicgen.shared import utc_now_iso, write_json

PLATEAU_EVENT_SCHEMA_VERSION = 1
DEFAULT_METRIC = "valid.loss"
VALID_ACTIONS = ("log", "stop")


@dataclass(frozen=True)
class PlateauWatcherConfig:
    sig: str
    xp_folder: Path
    history_path: Path
    metric: str
    patience_epochs: int
    min_improvement: float
    poll_seconds: float
    action: str
    stop_pid: int | None
    events_path: Path
    status_path: Path
    grace_seconds: float
    once: bool


@dataclass(frozen=True)
class PlateauReport:
    total_entries: int
    latest_value: float | None
    best_value: float | None
    best_epoch_index: int | None
    latest_epoch_index: int | None
    epochs_since_best: int
    plateau_detected: bool
    metric: str
    patience_epochs: int
    min_improvement: float


def read_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def extract_metric_value(entry: dict[str, Any], metric: str) -> float | None:
    """Extract a metric value using either a dotted nested path or a flat key."""
    if metric in entry:
        value = entry[metric]
        return float(value) if _is_number(value) else None
    current: Any = entry
    for part in metric.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if _is_number(current) else None


def classify_plateau(
    history: Sequence[dict[str, Any]],
    *,
    metric: str,
    patience_epochs: int,
    min_improvement: float,
) -> PlateauReport:
    if patience_epochs < 1:
        raise ValueError("patience_epochs must be >= 1.")
    if min_improvement < 0:
        raise ValueError("min_improvement must be >= 0.")

    best_value: float | None = None
    best_epoch_index: int | None = None
    latest_value: float | None = None
    latest_epoch_index: int | None = None

    for index, entry in enumerate(history):
        value = extract_metric_value(entry, metric)
        if value is None:
            continue
        latest_value = value
        latest_epoch_index = index
        if best_value is None or value < best_value - min_improvement:
            best_value = value
            best_epoch_index = index

    if latest_epoch_index is None or best_epoch_index is None:
        epochs_since_best = 0
    else:
        epochs_since_best = max(0, latest_epoch_index - best_epoch_index)

    plateau_detected = latest_value is not None and best_value is not None and epochs_since_best >= patience_epochs

    return PlateauReport(
        total_entries=len(history),
        latest_value=latest_value,
        best_value=best_value,
        best_epoch_index=best_epoch_index,
        latest_epoch_index=latest_epoch_index,
        epochs_since_best=epochs_since_best,
        plateau_detected=plateau_detected,
        metric=metric,
        patience_epochs=patience_epochs,
        min_improvement=min_improvement,
    )


def build_plateau_watcher_config(
    *,
    sig: str,
    xp_folder: Path,
    metric: str = DEFAULT_METRIC,
    patience_epochs: int = 4,
    min_improvement: float = 1e-4,
    poll_seconds: float = 60.0,
    action: str = "log",
    stop_pid: int | None = None,
    events_path: Path | None = None,
    status_path: Path | None = None,
    grace_seconds: float = 10.0,
    once: bool = False,
) -> PlateauWatcherConfig:
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action!r}. Expected one of: {sorted(VALID_ACTIONS)}.")
    if action == "stop" and stop_pid is None:
        raise ValueError("action=stop requires --stop-pid to identify which training process to terminate.")
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be >= 0.")
    if poll_seconds < 0.1:
        raise ValueError("poll_seconds must be >= 0.1.")
    return PlateauWatcherConfig(
        sig=sig,
        xp_folder=xp_folder,
        history_path=xp_folder / "history.json",
        metric=metric,
        patience_epochs=patience_epochs,
        min_improvement=min_improvement,
        poll_seconds=poll_seconds,
        action=action,
        stop_pid=stop_pid,
        events_path=events_path or (xp_folder / "plateau_events.jsonl"),
        status_path=status_path or (xp_folder / "plateau_watch_status.json"),
        grace_seconds=grace_seconds,
        once=once,
    )


def run_plateau_watcher(
    config: PlateauWatcherConfig,
    *,
    sleep_fn: Callable[[float], None] | None = None,
    kill_fn: Callable[[int, float], dict[str, Any]] | None = None,
) -> int:
    sleeper = sleep_fn or time.sleep
    terminator = kill_fn or _send_stop_signal

    _append_event(
        config.events_path,
        {
            "event": "watch_started",
            "sig": config.sig,
            "metric": config.metric,
            "patience_epochs": config.patience_epochs,
            "min_improvement": config.min_improvement,
            "action": config.action,
            "stop_pid": config.stop_pid,
            "once": config.once,
        },
    )

    while True:
        history = read_history(config.history_path)
        report = classify_plateau(
            history,
            metric=config.metric,
            patience_epochs=config.patience_epochs,
            min_improvement=config.min_improvement,
        )
        _write_status(config, report, status="watching")
        _append_event(
            config.events_path,
            {
                "event": "progress",
                "sig": config.sig,
                **_report_summary(report),
            },
        )

        if report.plateau_detected:
            _append_event(
                config.events_path,
                {
                    "event": "plateau_detected",
                    "sig": config.sig,
                    **_report_summary(report),
                },
            )
            if config.action == "stop" and config.stop_pid is not None:
                kill_payload = terminator(config.stop_pid, config.grace_seconds)
                _append_event(
                    config.events_path,
                    {
                        "event": "stop_sent",
                        "sig": config.sig,
                        "stop_pid": config.stop_pid,
                        "kill": kill_payload,
                    },
                )
            _write_status(config, report, status="plateau_detected")
            _append_event(
                config.events_path,
                {
                    "event": "watch_stopped",
                    "sig": config.sig,
                    "reason": "plateau_detected",
                },
            )
            return 0

        if config.once:
            _write_status(config, report, status="one_shot_done")
            _append_event(
                config.events_path,
                {
                    "event": "watch_stopped",
                    "sig": config.sig,
                    "reason": "once_mode",
                },
            )
            return 0

        sleeper(config.poll_seconds)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    event = dict(payload)
    event["schema_version"] = PLATEAU_EVENT_SCHEMA_VERSION
    event["timestamp"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event))
        handle.write("\n")


def _write_status(
    config: PlateauWatcherConfig,
    report: PlateauReport,
    *,
    status: str,
) -> None:
    payload = {
        "schema_version": PLATEAU_EVENT_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "status": status,
        "sig": config.sig,
        "xp_folder": str(config.xp_folder),
        "history_path": str(config.history_path),
        "events_path": str(config.events_path),
        "metric": config.metric,
        "patience_epochs": config.patience_epochs,
        "min_improvement": config.min_improvement,
        "action": config.action,
        "stop_pid": config.stop_pid,
        **_report_summary(report),
    }
    write_json(config.status_path, payload)


def _report_summary(report: PlateauReport) -> dict[str, Any]:
    return {
        "total_entries": report.total_entries,
        "latest_value": report.latest_value,
        "best_value": report.best_value,
        "best_epoch_index": report.best_epoch_index,
        "latest_epoch_index": report.latest_epoch_index,
        "epochs_since_best": report.epochs_since_best,
        "plateau_detected": report.plateau_detected,
    }


def _send_stop_signal(pid: int, grace_seconds: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pid": pid,
        "sigterm_sent": False,
        "sigkill_sent": False,
    }
    try:
        pgid = os.getpgid(pid)
        payload["process_group_id"] = pgid
    except OSError as exc:
        payload["getpgid_error"] = str(exc)
        pgid = None

    target = pgid if pgid is not None else pid
    send_group = pgid is not None

    try:
        if send_group:
            os.killpg(target, signal.SIGTERM)
        else:
            os.kill(target, signal.SIGTERM)
        payload["sigterm_sent"] = True
    except ProcessLookupError:
        payload["sigterm_error"] = "process already exited"
        return payload
    except OSError as exc:
        payload["sigterm_error"] = str(exc)

    time.sleep(max(0.0, grace_seconds))
    if not _pid_running(pid):
        return payload

    try:
        if send_group:
            os.killpg(target, signal.SIGKILL)
        else:
            os.kill(target, signal.SIGKILL)
        payload["sigkill_sent"] = True
    except ProcessLookupError:
        payload["sigkill_error"] = "process already exited"
    except OSError as exc:
        payload["sigkill_error"] = str(exc)
    return payload


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
