"""Scène: start a stream session and play it from the browser.

The page is a control surface, never an audio path: the sound stays in the stream's own process, so nothing the
browser does can make it drop out (the appliance panel's rule). It reads what ``play monitor`` reads plus
``live_status.json``, and writes only ``control.json``, the way the TUI's keys do: ``apply_key`` under
``control_file_lock``, so a key means the same thing on both and neither can drop the other's edit. A stream it
starts is the ``play stream`` a terminal would run, detached like a training job, so it outlives the page.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from macchina_ricordi.instrument.stream.monitor import FOLLOW_GENERATED, FOLLOW_MODES, frame_data, read_snapshot
from macchina_ricordi.instrument.stream.session import (
    LIVE_STATUS_FILENAME,
    PLAYLIST_FILENAME,
    StreamControl,
    control_file_lock,
    read_control,
    write_control,
)
from macchina_ricordi.instrument.stream.tui import (
    BUFFER_AHEAD_CEILING_SECONDS,
    BUFFER_AHEAD_FLOOR_SECONDS,
    KNOBS,
    apply_key,
)
from macchina_ricordi.models.record import ModelNotFound, resolve_model, split_selector
from macchina_ricordi.models.specs import advisories, resolve_segment_duration
from macchina_ricordi.sessions.record import count_lines, iso_from_run_id, read_json
from macchina_ricordi.sessions.store import stream_folders
from macchina_ricordi.training.jobs import pid_alive

from .jobs import DetachedJobs, tail
from .training import gpu_sample

# The status file is rewritten every half second: this long without a write means its process stopped writing.
LIVE_STALE_SECONDS = 5.0
# A session started before live_status.json existed shows it is running only by appending rounds, and a throttled
# round that reloads a model can take more than a minute.
LEGACY_IDLE_SECONDS = 120.0
MAX_KEYS = 64
HISTORY_ROUNDS = 60
EVENT_ROWS = 12
PLAYING = ("live", "stopping")
_RUN_ID = re.compile(r"\d{8}T\d{6}Z")

JOBS_DIR = "artifacts/stage_jobs"
JOB_SCHEMA = "macchina_ricordi.stage_job.v1"
EXIT_MARKER = "[macchina_ricordi stream exit"
LAUNCH_FILENAME = "launch.json"
MAX_MODELS = 9  # one per number key
START_TIMEOUT_SECONDS = 30.0
# A stop the loop has not acted on, for a stream this server started: Ctrl-C (the stream's own clean stop, manifest
# and all) after a minute, enough for a model load to finish, then SIGTERM, then SIGKILL.
STOP_ESCALATION = ((60.0, signal.SIGINT), (20.0, signal.SIGTERM), (20.0, signal.SIGKILL))
# The knobs a launch starts at, and the `play stream` flag that sets each; every one stays live afterwards.
START_FLAGS = {
    "completion_duration_seconds": "--completion-duration",
    "tau_seconds": "--tau",
    "temperature": "--temperature",
    "top_k": "--top-k",
    "candidates": "--candidates",
    "round_delay_seconds": "--round-delay-seconds",
}

Change = Callable[[StreamControl, Path, dict[str, Any] | None], tuple[StreamControl, str | None]]


class StageError(ValueError):
    """A request refused before anything was written."""


class StageNotFound(LookupError):
    pass


def cli_argv(cli: list[str]) -> list[str]:
    """How this server runs a CLI command: its own interpreter, unbuffered so the log keeps up."""
    return [sys.executable, "-u", "-m", "macchina_ricordi.cli", *cli]


@dataclass(frozen=True)
class LaunchRequest:
    """What the page asks to play: models in number-key order (the first one starts) and the knobs to start at."""

    models: tuple[str, ...]
    knobs: dict[str, float]
    max_buffer_ahead_seconds: float | None
    buffer_floor_ratio: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LaunchRequest:
        raw = payload.get("models")
        models = tuple(str(text).strip() for text in raw if str(text).strip()) if isinstance(raw, list) else ()
        if not 0 < len(models) <= MAX_MODELS:
            raise StageError(f"pick 1 to {MAX_MODELS} models: the first one starts, the number keys swap between them")
        if any(set(text) & set(",\r\n") or len(text) > 300 for text in models):
            raise StageError("a model reference holds no comma and no line break")
        knobs: dict[str, float] = {}
        for knob in KNOBS:
            value = _number(payload, knob.field) if knob.field in START_FLAGS else None
            if value is None:
                continue
            if not knob.low <= value <= knob.high:
                raise StageError(f"{knob.field} goes from {knob.low:g} to {knob.high:g}")
            knobs[knob.field] = int(round(value)) if knob.integer else value
        cap = _number(payload, "max_buffer_ahead_seconds")
        if cap is not None and not BUFFER_AHEAD_FLOOR_SECONDS <= cap <= BUFFER_AHEAD_CEILING_SECONDS:
            raise StageError(
                f"the buffer cap goes from {BUFFER_AHEAD_FLOOR_SECONDS:g} to {BUFFER_AHEAD_CEILING_SECONDS:g} s"
            )
        floor = _number(payload, "buffer_floor_ratio")
        if floor is not None and not 0 <= floor <= 1:
            raise StageError("the buffer floor goes from 0 to 1 × the buffer cap")
        if floor is not None and cap is None:
            raise StageError("the buffer floor is a share of the buffer cap, so it needs one")
        return cls(models, knobs, cap, floor)

    def cli(self) -> list[str]:
        """The ``play stream`` a terminal would run, without ``--tui``: only the page and the log watch this one."""
        args = ["play", "stream", "--xp-path", self.models[0]]
        if len(self.models) > 1:
            args += ["--playlist", ",".join(self.models)]
        for field, value in self.knobs.items():
            args += [START_FLAGS[field], f"{value:g}"]
        if self.max_buffer_ahead_seconds is not None:
            args += ["--max-buffer-ahead-seconds", f"{self.max_buffer_ahead_seconds:g}"]
        if self.buffer_floor_ratio is not None:
            args += ["--buffer-floor-ratio", f"{self.buffer_floor_ratio:g}"]
        return args


def session_state(folder: Path, live: dict[str, Any] | None, manifest: dict[str, Any] | None, *, now: float) -> str:
    """``live``, ``stopping`` (playing out its buffer), ``ended``, or ``unfinished`` (gone without a manifest)."""
    pid = live.get("pid") if live is not None else None
    if isinstance(pid, int) and pid_alive(pid) and now - _timestamp(live.get("updated_at")) <= LIVE_STALE_SECONDS:
        return "stopping" if live.get("phase") == "stopped" else "live"
    if manifest is not None:
        return "ended"
    if live is None:
        try:
            if now - (folder / "performance.jsonl").stat().st_mtime <= LEGACY_IDLE_SECONDS:
                return "live"
        except OSError:
            pass
    return "unfinished"


class StageManager:
    def __init__(self, repo_root: Path, *, active_trainings: Callable[[], list[str]] = list) -> None:
        self.repo_root = repo_root
        self.active_trainings = active_trainings
        self._jobs = DetachedJobs(repo_root, JOBS_DIR, JOB_SCHEMA, EXIT_MARKER)
        self._stopping: set[str] = set()

    def overview(self) -> dict[str, Any]:
        """The sessions, and what the launch form starts from: the knobs' limits and the stream's own defaults."""
        return {
            "sessions": self.sessions(),
            "knobs": [asdict(knob) for knob in KNOBS],
            "defaults": asdict(StreamControl()),
        }

    def sessions(self) -> list[dict[str, Any]]:
        """Every stream session: the ones playing first, then newest first."""
        now = time.time()
        found = [
            self._summary(folder, now)
            for folder in reversed(stream_folders(self.repo_root))
            if (folder / "performance.jsonl").is_file() or (folder / LIVE_STATUS_FILENAME).is_file()
        ]
        return sorted(found, key=lambda session: session["state"] not in PLAYING)

    def frame(self, session_id: str, *, follow: str = FOLLOW_GENERATED) -> dict[str, Any]:
        folder = self._folder(session_id)
        snapshot = read_snapshot(folder, history=HISTORY_ROUNDS, events=EVENT_ROWS)
        manifest = read_json(folder / "manifest.json")
        state = session_state(folder, snapshot.live, manifest or None, now=snapshot.read_at)
        live = snapshot.live if state in PLAYING else None
        launch = read_json(folder / LAUNCH_FILENAME)
        return {
            "id": folder.name,
            "state": state,
            "exact": live is not None,
            "status": live,
            "ended": {key: manifest.get(key) for key in ("finished_at", "stopped_by", "rounds_written")},
            "playlist": _playlist(folder),
            "knobs": [asdict(knob) for knob in KNOBS],
            "launch": {**launch, "log_tail": tail(self.repo_root / str(launch.get("log")), 40)} if launch else None,
            **frame_data(snapshot, follow=follow if follow in FOLLOW_MODES else FOLLOW_GENERATED, live=live),
        }

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """What Play would start, checked first: each model resolved, the guidance, and what else holds the GPU."""
        request = LaunchRequest.from_payload(payload)
        models, errors = self._resolve(request)
        first = models[0].get("folder")
        guidance = (
            advisories(
                first,
                tau_seconds=request.knobs.get("tau_seconds"),
                completion_duration_seconds=request.knobs.get("completion_duration_seconds"),
                temperature=request.knobs.get("temperature"),
                top_k=request.knobs.get("top_k"),
            )
            if first
            else ()
        )
        warnings = [
            f"{label} is training: two jobs on one 12 GB GPU usually means both crawl or one dies"
            for label in self.active_trainings()
        ]
        warnings += [
            f"{session['id']} is playing: both would share the GPU and the speakers"
            for session in self.sessions()
            if session["state"] in PLAYING
        ]
        return {
            "models": models,
            "errors": errors,
            "warnings": warnings,
            "guidance": [asdict(advisory) for advisory in guidance],
            "command": "python -m macchina_ricordi.cli " + shlex.join(request.cli()),
            "gpu": gpu_sample(),
        }

    def launch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start ``play stream`` detached, like a training job, and wait until its loop runs (its status exists)."""
        request = LaunchRequest.from_payload(payload)
        _models, errors = self._resolve(request)
        if errors:
            raise StageError("; ".join(errors))
        cli = request.cli()
        record = self._jobs.launch(
            "stream", f"play stream {request.models[0]}", [cli_argv(cli)], {"models": list(request.models)}
        )
        folder = self._wait_for_start(record)
        launch = {
            "job": record["id"],
            "log": record["log"],
            "command": "python -m macchina_ricordi.cli " + shlex.join(cli),
            "launched_at": record["created_at"],
        }
        (folder / LAUNCH_FILENAME).write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
        return {"id": folder.name, **launch}

    def _resolve(self, request: LaunchRequest) -> tuple[list[dict[str, Any]], list[str]]:
        """Each model as the stream will load it: the first at its checkpoint, a swap at the latest one."""
        models: list[dict[str, Any]] = []
        errors: list[str] = []
        for slot, text in enumerate(request.models, start=1):
            try:
                ref = resolve_model(text, repo_root=self.repo_root)
            except ModelNotFound as exc:
                errors.append(f"{slot}: {exc}")
                models.append({"text": text})
                continue
            if ref.model is None:
                errors.append(f"{slot}: {text} is a checkpoint file, and a stream needs a model in .dora/xps")
                models.append({"text": text})
                continue
            selector = split_selector(text)[1] if slot > 1 else None
            segment_duration = (
                ref.model.segment_duration
                if ref.model.kind == "bundle"
                else resolve_segment_duration(str(ref.model.folder))
            )
            models.append(
                {
                    "text": text,
                    "sig": ref.model.sig,
                    "folder": str(ref.model.folder),
                    "kind": ref.model.kind,
                    "checkpoint": ref.checkpoint.name if slot == 1 and ref.checkpoint else "latest",
                    "segment_duration": segment_duration,
                    "note": f"a swap loads the latest checkpoint, so @{selector} is ignored" if selector else None,
                }
            )
        return models, errors

    def _wait_for_start(self, record: dict[str, Any]) -> Path:
        """The new session's folder once its loop runs; the end of its log when it never gets there."""
        log = self.repo_root / record["log"]
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            announced = next((line for line in tail(log, 200) if line.startswith("session_dir: ")), None)
            folder = Path(announced.removeprefix("session_dir: ").strip()) if announced else None
            if folder is not None and (folder / LIVE_STATUS_FILENAME).is_file():
                return folder
            if not pid_alive(record["pid"]):
                break
            time.sleep(0.2)
        reason = (
            f"not playing after {START_TIMEOUT_SECONDS:.0f} s: it may still show up under En scène"
            if pid_alive(record["pid"])
            else "the stream stopped before its first round"
        )
        raise StageError(f"{reason}. The end of {record['log']}:\n" + "\n".join(tail(log, 20)))

    def _escalate_stop(self, folder: Path) -> None:
        """A stream this server started gets Ctrl-C, then harder, if the stop command is not acted on in time."""
        job = str(read_json(folder / LAUNCH_FILENAME).get("job") or "")
        process = self._jobs.child(job)
        if process is None or process.poll() is not None or job in self._stopping:
            return  # started from a terminal, or by a GUI since restarted: the stop command is all it gets
        self._stopping.add(job)
        threading.Thread(target=_escalate, args=(process,), daemon=True).start()

    def press(self, session_id: str, keys: object) -> dict[str, Any]:
        """Apply keys as the TUI does, in one write: a held key arrives as a burst, as from a terminal."""
        if not isinstance(keys, list) or not 0 < len(keys) <= MAX_KEYS or not all(isinstance(k, str) for k in keys):
            raise StageError(f"keys must be a list of 1 to {MAX_KEYS} key names")

        def change(
            control: StreamControl, folder: Path, live: dict[str, Any] | None
        ) -> tuple[StreamControl, str | None]:
            playlist = _playlist(folder)
            message = None
            for key in keys:
                result = apply_key(key, control, playlist=playlist)
                if not result.handled:
                    continue
                control, message = result.control, result.message
                if key == "p":
                    # What the speakers are on now, as the TUI captures it; with no status the loop picks.
                    playing = (live or {}).get("playing")
                    control = replace(control, pin_target=playing)
                    message = f"pin queued: {Path(playing).name}" if playing else message
            return control, message

        result = self._edit(session_id, change)
        if "Q" in keys:
            self._escalate_stop(self._folder(session_id))
        return result

    def set(self, session_id: str, field: str, value: object) -> dict[str, Any]:
        """Put one knob at a value, within the limits its keys keep to."""
        knob = next((k for k in KNOBS if k.field == field), None)
        try:
            target = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            target = math.nan
        if knob is None or not math.isfinite(target):
            raise StageError(f"cannot set {field!r} to {value!r}")

        def change(control: StreamControl, _folder: Path, _live: dict[str, Any] | None) -> tuple[StreamControl, str]:
            before = getattr(control, field)
            after: float = min(knob.high, max(knob.low, target))
            after = int(round(after)) if knob.integer else after
            message = f"{field} {format(before, knob.fmt)} → {format(after, knob.fmt)}"
            return replace(control, **{field: after}), message

        return self._edit(session_id, change)

    def _edit(self, session_id: str, change: Change) -> dict[str, Any]:
        folder = self._folder(session_id)
        live = read_json(folder / LIVE_STATUS_FILENAME) or None
        if session_state(folder, live, read_json(folder / "manifest.json") or None, now=time.time()) != "live":
            raise StageError("this session is not playing: it can be watched, not steered")
        path = folder / "control.json"
        with control_file_lock(path):
            before = read_control(path)
            control, message = change(before, folder, live)
            if control != before:
                write_control(path, control)
        return {"control": asdict(control), "message": message}

    def _folder(self, session_id: str) -> Path:
        found = next((folder for folder in stream_folders(self.repo_root) if folder.name == session_id), None)
        if found is None:
            raise StageNotFound(session_id)
        return found

    def _summary(self, folder: Path, now: float) -> dict[str, Any]:
        live = read_json(folder / LIVE_STATUS_FILENAME) or None
        manifest = read_json(folder / "manifest.json")
        state = session_state(folder, live, manifest or None, now=now)
        model = manifest.get("final_xp_path") or read_json(folder / "control.json").get("model_xp")
        run_id = _RUN_ID.search(folder.name)
        return {
            "id": folder.name,
            "state": state,
            "phase": live.get("phase") if live and state in PLAYING else None,
            "started_at": manifest.get("started_at") or (iso_from_run_id(run_id.group(0)) if run_id else None),
            "finished_at": manifest.get("finished_at"),
            "rounds": manifest.get("rounds_written") or count_lines(folder / "performance.jsonl"),
            "model": Path(str(model)).name if model else None,
            "stopped_by": manifest.get("stopped_by"),
        }


def _escalate(process: subprocess.Popen[bytes]) -> None:
    """Ctrl-C, then SIGTERM, then SIGKILL, each after a wait, to the job's whole session: shell, stream and worker."""
    for wait, signum in STOP_ESCALATION:
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return


def _number(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise StageError(f"{name} must be a number, not {value!r}")
    return number


def _timestamp(value: object) -> float:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return -math.inf


def _playlist(folder: Path) -> list[str]:
    try:
        entries = json.loads((folder / PLAYLIST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(entry) for entry in entries] if isinstance(entries, list) else []
