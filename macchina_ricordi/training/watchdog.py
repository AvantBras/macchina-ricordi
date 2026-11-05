from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..musicgen.shared import utc_now_iso


@dataclass(frozen=True)
class MusicGenGpuWatchdogConfig:
    gpu_index: int = 0
    poll_interval_seconds: float = 1.0
    temperature_trip_c: float | None = None
    events_path: Path | None = None


@dataclass(frozen=True)
class GpuWatchdogSample:
    raw_output: str
    temperature_c: float | None
    gpu_lost: bool
    query_failed: bool
    query_error: str | None = None


def parse_gpu_watchdog_sample(raw_output: str) -> GpuWatchdogSample:
    text = raw_output.strip()
    line = next((item.strip() for item in text.splitlines() if item.strip()), "")
    parts = [part.strip() for part in line.split(",")] if line else []

    temperature_c: float | None = None
    if parts:
        try:
            temperature_c = float(parts[0].split()[0])
        except (IndexError, ValueError):
            temperature_c = None

    return GpuWatchdogSample(
        raw_output=text,
        temperature_c=temperature_c,
        gpu_lost="[GPU is lost]" in text,
        query_failed=False,
    )


def query_gpu_watchdog_sample(*, gpu_index: int) -> GpuWatchdogSample:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,pstate",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return GpuWatchdogSample(
            raw_output="",
            temperature_c=None,
            gpu_lost=False,
            query_failed=True,
            query_error=str(exc),
        )

    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return GpuWatchdogSample(
            raw_output=stdout,
            temperature_c=None,
            gpu_lost="[GPU is lost]" in stdout or "[GPU is lost]" in stderr,
            query_failed=True,
            query_error=stderr or f"nvidia-smi exited with code {completed.returncode}",
        )
    return parse_gpu_watchdog_sample(stdout)


def monitor_musicgen_process(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    config: MusicGenGpuWatchdogConfig,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    while process.poll() is None:
        sample = query_gpu_watchdog_sample(gpu_index=config.gpu_index)
        event: dict[str, object] | None = None

        if sample.gpu_lost:
            event = {
                "event": "gpu_lost",
                "observed_at": utc_now_iso(),
                "gpu_index": config.gpu_index,
                "raw_output": sample.raw_output,
            }
        elif (
            config.temperature_trip_c is not None
            and sample.temperature_c is not None
            and sample.temperature_c >= config.temperature_trip_c
        ):
            event = {
                "event": "temperature_trip",
                "observed_at": utc_now_iso(),
                "gpu_index": config.gpu_index,
                "temperature_c": sample.temperature_c,
                "temperature_trip_c": config.temperature_trip_c,
                "raw_output": sample.raw_output,
            }

        if event is not None:
            event["kill"] = _kill_musicgen_process(process)
            events.append(event)
            _append_watchdog_event(config.events_path, event)
            break

        time.sleep(config.poll_interval_seconds)

    return events


def _kill_musicgen_process(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempted_signal": "SIGKILL",
        "killed_process_group": False,
    }
    try:
        pgid = os.getpgid(process.pid)
    except OSError as exc:
        payload["kill_error"] = str(exc)
        return payload

    try:
        os.killpg(pgid, signal.SIGKILL)
        payload["killed_process_group"] = True
        payload["process_group_id"] = pgid
        return payload
    except ProcessLookupError:
        payload["process_group_id"] = pgid
        payload["kill_error"] = "process group already exited"
        return payload
    except OSError as exc:
        payload["process_group_id"] = pgid
        payload["kill_error"] = str(exc)

    try:
        os.kill(process.pid, signal.SIGKILL)
        payload["killed_pid_only"] = True
    except ProcessLookupError:
        payload["kill_error"] = "process already exited"
    except OSError as exc:
        payload["kill_error"] = str(exc)
    return payload


def _append_watchdog_event(path: Path | None, event: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
