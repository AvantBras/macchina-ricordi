"""Training jobs, and the queue that runs them one at a time.

Every training is a job, and a job's folder ``artifacts/train_jobs/<id>/`` is that training's whole record:

- ``job.json``: what to run, written once. A training job keeps ``train run``'s own arguments; a resume job keeps
  the model and its new epoch target.
- ``job_status.json``: where it is. ``queued``, then ``starting`` and ``running``, then ``completed``, ``failed`` or
  ``stopped``; ``cancelled`` if it never started.
- ``train_plan.json`` and ``train_run_status.json``, written by ``train run`` as it starts: the resolved plan, then
  the Dora process and how it ended (``gpu_watchdog_events.jsonl`` beside them with the watchdog on).
- ``train.log``: everything the job printed, when the queue or the GUI started it.

``train run`` makes a job and runs it in the foreground. ``train queue add`` makes one and leaves it queued, and
``train queue run`` (the worker) starts queued jobs in the order they were added, one at a time, and only while no
other job is running. One worker at a time: it holds an flock on ``worker.lock`` for as long as it runs.

Datasets never receive run records; older ones in a workspace's ``logs/`` are still read as a fallback.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.config import get_repo_root
from ..musicgen.shared import read_json, write_text_atomic

JOBS_DIR = Path("artifacts") / "train_jobs"
JOB_FILENAME = "job.json"
STATUS_FILENAME = "job_status.json"
LOG_FILENAME = "train.log"
TRAIN_PLAN_FILENAME = "train_plan.json"
TRAIN_RUN_STATUS_FILENAME = "train_run_status.json"
WATCHDOG_EVENTS_FILENAME = "gpu_watchdog_events.jsonl"
WORKER_LOCK_FILENAME = "worker.lock"
WORKER_LOG_FILENAME = "worker.log"
JOB_SCHEMA = "macchina_ricordi.train_job.v1"
KINDS = ("train", "resume")
RUNNING_STATES = ("starting", "building", "running")  # building: a `--pool` run encoding its token pool
ACTIVE_STATES = frozenset({*RUNNING_STATES, "stopping"})
FINAL_STATES = frozenset({"completed", "failed", "stopped", "cancelled"})
STOP_GRACE_SECONDS = 30.0
CLI_PREFIX = ("-m", "macchina_ricordi.cli")
_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:_[a-z0-9_]+)?(?:-\d+)?$")


class JobError(RuntimeError):
    """A job that cannot be found, added, started or stopped; the message says why."""


def jobs_dir(repo_root: str | Path | None = None) -> Path:
    return get_repo_root(repo_root) / JOBS_DIR


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Job:
    id: str
    path: Path
    kind: str
    argv: tuple[str, ...]
    request: dict[str, Any]
    origin: str
    created_at: str
    status: dict[str, Any]

    @property
    def state(self) -> str:
        """The recorded status, corrected for a process that died without saying so."""
        status = str(self.status.get("status") or "queued")
        if status not in RUNNING_STATES:
            return status
        if not _is_job_process(self.status.get("pid")):
            return "stopped" if self.status.get("stop") else "failed"
        return "stopping" if self.status.get("stop") else status

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def log_path(self) -> Path:
        return self.path / LOG_FILENAME

    @property
    def xp_sig(self) -> str | None:
        return self.status.get("xp_sig") or self.request.get("xp_sig")

    def run_status(self) -> dict[str, Any]:
        return read_json(self.path / TRAIN_RUN_STATUS_FILENAME) or {}

    def command(self, prefix: Sequence[str] | None = None) -> list[str]:
        """What starts it: ``train run`` with the job's arguments, or ``admin resume-xp``."""
        head = list(prefix) if prefix else [sys.executable, *CLI_PREFIX]
        if self.kind == "resume":
            return [*head, "train", "resume", str(self.request["model"]),
                    "--epochs", str(self.request["epochs"])]  # fmt: skip
        return [*head, "train", "run", *self.argv, "--job", self.id]

    def describe(self) -> str:
        request = self.request
        if self.kind == "resume":
            return f"resume {request.get('model')} to {request.get('epochs')} epochs"
        text = f"{request.get('recipe')} on {request.get('dataset')}"
        if request.get("weights"):
            text += f" (weights {request['weights']})"
        if request.get("init_from"):
            text += f" from {request['init_from']}"
        if request.get("epochs"):
            text += f", {request['epochs']} epoch{'' if request['epochs'] == 1 else 's'}"
        return text

    def to_dict(self) -> dict[str, Any]:
        status = self.status
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "state": self.state,
            "active": self.active,
            "description": self.describe(),
            "request": self.request,
            "argv": list(self.argv),
            "created_at": self.created_at,
            "started_at": status.get("started_at"),
            "finished_at": status.get("finished_at"),
            "returncode": status.get("returncode"),
            "error": status.get("error"),
            "xp_sig": self.xp_sig,
            "effective_label": status.get("effective_label") or self.request.get("run_label"),
            "epochs": status.get("epochs") or self.request.get("epochs"),
            "stop": status.get("stop"),
            "path": str(self.path),
            "log": str(self.log_path) if self.log_path.exists() else None,
        }


# ── reading ──


def load_job(job: str | Path, *, repo_root: str | Path | None = None) -> Job:
    path = Path(job) if isinstance(job, Path) else jobs_dir(repo_root) / str(job)
    if isinstance(job, str) and not _ID_RE.fullmatch(job):
        raise JobError(f"no job {job!r} (ids look like 20261006T101500Z_name; see `train queue list`)")
    record = read_json(path / JOB_FILENAME)
    if record is None or record.get("schema") != JOB_SCHEMA:
        raise JobError(f"no job {path.name!r} (see `train queue list`)")
    return Job(
        id=path.name,
        path=path,
        kind=str(record.get("kind") or "train"),
        argv=tuple(str(arg) for arg in record.get("argv") or ()),
        request=dict(record.get("request") or {}),
        origin=str(record.get("origin") or "cli"),
        created_at=str(record.get("created_at") or ""),
        status=read_json(path / STATUS_FILENAME) or {"status": "queued"},
    )


def list_jobs(repo_root: str | Path | None = None) -> list[Job]:
    """Every job, oldest first: the queue's order."""
    root = jobs_dir(repo_root)
    if not root.is_dir():
        return []
    jobs = []
    for path in root.iterdir():
        if path.is_dir() and (path / JOB_FILENAME).exists():
            try:
                jobs.append(load_job(path))
            except JobError:
                continue
    return sorted(jobs, key=lambda job: (job.created_at, job.id))


def queued_jobs(repo_root: str | Path | None = None) -> list[Job]:
    return [job for job in list_jobs(repo_root) if job.state == "queued"]


def active_jobs(repo_root: str | Path | None = None) -> list[Job]:
    return [job for job in list_jobs(repo_root) if job.active]


def training_sigs(repo_root: str | Path | None = None) -> set[str]:
    """Models a job is training right now; cleanup leaves them alone."""
    return {job.xp_sig for job in active_jobs(repo_root) if job.xp_sig}


def find_train_plan(
    workspace_dir: str | Path, *, xp_sig: str | None = None, repo_root: str | Path | None = None
) -> Path | None:
    """The plan of the job that trained ``xp_sig``, else of the latest job on this dataset.

    Falls back to the workspace's own ``logs/train_plan.json``, where runs recorded themselves before jobs.
    """
    workspace = Path(workspace_dir).resolve()
    on_dataset = []
    for job in reversed(list_jobs(repo_root)):
        plan_path = job.path / TRAIN_PLAN_FILENAME
        plan = read_json(plan_path)
        if plan is None:
            continue
        if xp_sig is not None and plan.get("xp_sig") == xp_sig:
            return plan_path
        if Path(str(plan.get("workspace_dir") or "")).resolve() == workspace:
            on_dataset.append(plan_path)
    legacy = workspace / "logs" / TRAIN_PLAN_FILENAME
    if xp_sig is not None and (legacy_plan := read_json(legacy)) and legacy_plan.get("xp_sig") == xp_sig:
        return legacy
    if on_dataset:
        return on_dataset[0]
    return legacy if legacy.exists() else None


def last_job_on(dataset_name: str, repo_root: str | Path | None = None) -> Job | None:
    return next((job for job in reversed(list_jobs(repo_root)) if job.request.get("dataset") == dataset_name), None)


# ── writing ──


def create_job(
    kind: str,
    *,
    request: dict[str, Any],
    argv: Sequence[str] = (),
    origin: str = "cli",
    started_by: int | None = None,
    repo_root: str | Path | None = None,
) -> Job:
    """Write a new job: queued, or already started by process ``started_by`` (a foreground ``train run``)."""
    if kind not in KINDS:
        raise JobError(f"unknown job kind {kind!r}")
    root = jobs_dir(repo_root)
    hint = request.get("run_label") or request.get("model") or request.get("recipe") or kind
    with _locked(root):
        job_id = _new_id(root, str(hint))
        path = root / job_id
        path.mkdir(parents=True)
        created = now_iso()
        _write_json(
            path / JOB_FILENAME,
            {"schema": JOB_SCHEMA, "kind": kind, "origin": origin, "created_at": created,
             "argv": list(argv), "request": request},
        )  # fmt: skip
        status: dict[str, Any] = {"status": "queued", "updated_at": created}
        if started_by is not None:
            status.update(status="starting", pid=started_by, started_at=created)
        _write_json(path / STATUS_FILENAME, status)
    return load_job(path)


def parse_train_argv(argv: Sequence[str]) -> argparse.Namespace:
    """``train run``'s arguments, parsed by ``train run``'s own parser."""
    from .cli import _add_train_run_arguments

    parser = argparse.ArgumentParser(prog="train run", add_help=False, exit_on_error=False)
    _add_train_run_arguments(parser)
    try:
        args, unknown = parser.parse_known_args(list(argv))
    except (argparse.ArgumentError, SystemExit) as exc:
        raise JobError(f"not a valid training: {exc}") from None
    if unknown:
        raise JobError(f"not a valid training: unknown arguments {' '.join(unknown)}")
    return args


def train_job_argv(args: argparse.Namespace, *, repo_root: str | Path | None = None) -> list[str]:
    """The arguments a job keeps: ``train run``'s, canonical, with the dataset by name and never a token."""
    argv = ["--recipe", args.profile, "--dataset", _dataset_token(args.workspace_dir, repo_root)]
    if args.epochs:
        argv += ["--epochs", str(args.epochs)]
    for override in args.override:
        argv += ["--override", override]
    for flag, value in (
        ("--weights", args.weights),
        ("--run-label", args.run_label),
        ("--from", args.init_from_checkpoint),
        ("--audiocraft-dir", args.audiocraft_dir),
        ("--musicgen-python", args.musicgen_python),
    ):
        if value:
            argv += [flag, str(value)]
    if getattr(args, "pool", False):
        argv.append("--pool")
        if args.pool_seed:
            argv += ["--pool-seed", str(args.pool_seed)]
    if args.no_auto_dataset_label:
        argv.append("--no-auto-dataset-label")
    if args.allow_segment_padding:
        argv.append("--allow-segment-padding")
    if args.gpu_watchdog:
        argv.append("--gpu-watchdog")
        if args.gpu_watchdog_temp_c is not None:
            argv += ["--gpu-watchdog-temp-c", f"{args.gpu_watchdog_temp_c:g}"]
        if args.gpu_watchdog_gpu_index:
            argv += ["--gpu-watchdog-gpu-index", str(args.gpu_watchdog_gpu_index)]
        if args.gpu_watchdog_poll_seconds != 1.0:
            argv += ["--gpu-watchdog-poll-seconds", f"{args.gpu_watchdog_poll_seconds:g}"]
    return argv


def train_job_request(args: argparse.Namespace, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """What a training job is, for people and agents: the sentence `recipe on dataset [from model]`."""
    return {
        "dataset": _dataset_token(args.workspace_dir, repo_root),
        "recipe": args.profile,
        "weights": args.weights,
        "init_from": args.init_from_checkpoint,
        "run_label": args.run_label,
        "epochs": args.epochs or epochs_in(args.override),
        "overrides": list(args.override),
        "pool": bool(getattr(args, "pool", False)),
    }


def add_train_job(
    argv: Sequence[str] | argparse.Namespace, *, origin: str = "cli", repo_root: str | Path | None = None
) -> Job:
    """Queue ``train run <argv>`` after checking what can be checked now (see ``check_train_job``)."""
    args = argv if isinstance(argv, argparse.Namespace) else parse_train_argv(argv)
    check_train_job(args, repo_root=repo_root)
    return create_job(
        "train",
        request=train_job_request(args, repo_root=repo_root),
        argv=train_job_argv(args, repo_root=repo_root),
        origin=origin,
        repo_root=repo_root,
    )


def add_resume_job(model: str, *, epochs: int, origin: str = "cli", repo_root: str | Path | None = None) -> Job:
    if epochs < 1:
        raise JobError("--epochs must be at least 1")
    _check_model_reference(model, repo_root=repo_root, whole_model=True)
    return create_job("resume", request={"model": model, "epochs": epochs}, origin=origin, repo_root=repo_root)


def check_train_job(args: argparse.Namespace, *, repo_root: str | Path | None = None) -> None:
    """Fail now, not hours later: the dataset and weights exist, and the model to start from exists or is the
    run label of a job queued before this one (it is then resolved when this job starts)."""
    from ..data.datasets import find_dataset_dir

    dataset = find_dataset_dir(args.workspace_dir, repo_root=repo_root)
    if dataset is None:
        raise JobError(f"no dataset called {args.workspace_dir!r} (see `data list`)")
    if args.weights:
        from ..data.weights import load_weights_overlay

        try:
            overlay = load_weights_overlay(get_repo_root(repo_root), dataset.name, args.weights)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise JobError(f"--weights {args.weights}: {exc}") from None
        if overlay.is_stale():
            raise JobError(f"--weights {args.weights} is stale: its dataset's manifest changed since")
    init = args.init_from_checkpoint
    if init and "://" in init:
        raise JobError("--from accepts local model bundles or checkpoints; download the model first.")
    if init and not Path(init).expanduser().exists():
        _check_model_reference(init, repo_root=repo_root, whole_model=False)


def epochs_in(overrides: Sequence[Any]) -> int | None:
    value = None
    for part in overrides:
        if isinstance(part, str) and part.startswith("optim.epochs="):
            value = part.split("=", 1)[1]
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _check_model_reference(text: str, *, repo_root: str | Path | None, whole_model: bool) -> None:
    from ..models.record import ModelNotFound, resolve_checkpoint_path, resolve_sig, split_selector

    try:
        resolve_sig(text, repo_root=repo_root) if whole_model else resolve_checkpoint_path(text, repo_root=repo_root)
        return
    except ModelNotFound as exc:
        problem = str(exc)
    label = _label_slug(split_selector(text)[0])
    upcoming = {_label_slug(job.request.get("run_label") or "") for job in list_jobs(repo_root)
                if job.kind == "train" and (job.state == "queued" or job.active)}  # fmt: skip
    if label and label in upcoming:
        return
    raise JobError(f"{problem} It is not the run label of a queued job either.")


def _label_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()


def _dataset_token(text: str, repo_root: str | Path | None) -> str:
    """A dataset by name when it is in this repo (so the job reads the same from anywhere), else its full path."""
    from ..data.datasets import find_dataset_dir

    found = find_dataset_dir(text, repo_root=repo_root)
    if found is None:
        return text
    root = get_repo_root(repo_root).resolve()
    return found.name if found.parent in (root, root / "datasets") else str(found)


def claim(job: Job, *, pid: int) -> Job:
    """Mark a queued job as started by ``pid``. Refuses one that someone else started or cancelled."""
    with _locked(job.path.parent):
        current = load_job(job.path)
        if current.state != "queued":
            raise JobError(f"job {job.id} is {current.state}, not queued")
        _write_status(job.path, current.status, status="starting", pid=pid, started_at=now_iso())
    return load_job(job.path)


def update_job(job: Job, **fields: Any) -> Job:
    with _locked(job.path.parent):
        _write_status(job.path, load_job(job.path).status, **fields)
    return load_job(job.path)


def finish(job: Job, *, returncode: int | None = None, error: str | None = None) -> Job:
    """Record how a started job ended, unless that is already recorded. A stop asked for reads as ``stopped``."""
    with _locked(job.path.parent):
        current = load_job(job.path)
        if current.status.get("status") in FINAL_STATES:
            return current
        if current.status.get("stop"):
            status = "stopped"
        else:
            status = "completed" if returncode == 0 and error is None else "failed"
        fields: dict[str, Any] = {"status": status, "finished_at": now_iso(), "returncode": returncode}
        if error is not None:
            fields["error"] = error
        _write_status(job.path, current.status, **fields)
    return load_job(job.path)


def stop(job: Job) -> dict[str, Any]:
    """Cancel a queued job, or SIGTERM a running one's training processes.

    The job's own process (``train run`` or ``admin resume-xp``) is not signalled: it sees its child exit and records
    the outcome itself. Pair with ``escalate`` to SIGKILL what ignores the SIGTERM.
    """
    with _locked(job.path.parent):
        current = load_job(job.path)
        if current.state == "queued":
            _write_status(job.path, current.status, status="cancelled", finished_at=now_iso())
            return {"cancelled": True}
        if current.state not in RUNNING_STATES:
            raise JobError(f"job {job.id} is {current.state}; there is nothing to stop")
        run_status = current.run_status()
        request: dict[str, Any] = {"requested_at": now_iso(), "signal": "SIGTERM"}
        group = run_status.get("train_process_group_id")
        job_pid = current.status.get("pid")
        if group and group != job_pid and pid_alive(run_status.get("train_pid")):
            request["target"] = {"process_group": group}
            request["sent"] = signal_group(int(group), signal.SIGTERM)
        elif job_pid:
            targets = descendants(int(job_pid))
            request["target"] = {"descendants_of": job_pid, "pids": targets}
            request["sent"] = bool(signal_all(targets, signal.SIGTERM))
        else:
            raise JobError(f"job {job.id} has no process to signal yet; try again in a moment")
        _write_status(job.path, current.status, stop=request)
    return request


def escalate(job: Job, request: dict[str, Any], *, grace_seconds: float = STOP_GRACE_SECONDS) -> bool:
    """SIGKILL whatever ignored a stop's SIGTERM for ``grace_seconds``. True if something had to be killed."""
    target = request.get("target") or {}
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if "process_group" in target and not group_alive(int(target["process_group"])):
            return False
        if "pids" in target and not any(pid_alive(pid) for pid in target["pids"]):
            return False
        time.sleep(0.5)
    if "process_group" in target:
        killed = signal_group(int(target["process_group"]), signal.SIGKILL)
    else:
        killed = bool(signal_all([pid for pid in target.get("pids", []) if pid_alive(pid)], signal.SIGKILL))
    current = load_job(job.path)
    update_job(job, stop={**(current.status.get("stop") or {}), "escalated_at": now_iso(), "sigkill_sent": killed})
    return killed


# ── running ──


def start(job: Job, *, prefix: Sequence[str] | None = None, cwd: Path | None = None) -> subprocess.Popen[bytes]:
    """Start a claimed job detached (its own session), its output appended to ``train.log``."""
    with job.log_path.open("ab") as log:
        process = subprocess.Popen(
            job.command(prefix),
            cwd=cwd or get_repo_root(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
    update_job(job, pid=process.pid)
    return process


def start_now(job: Job, *, prefix: Sequence[str] | None = None, repo_root: str | Path | None = None) -> Job:
    """Run a queued job at once, beside whatever else is running (the GUI's "start now")."""
    _prepare(job, repo_root=repo_root)
    claimed = claim(job, pid=os.getpid())
    process = start(claimed, prefix=prefix, cwd=get_repo_root(repo_root))
    # reap it, so a finished job does not linger as a zombie that still looks alive
    import threading

    threading.Thread(target=_reap, args=(claimed, process), daemon=True).start()
    return load_job(claimed.path)


def run_queue(
    *,
    repo_root: str | Path | None = None,
    watch: bool = False,
    poll_seconds: float = 5.0,
    prefix: Sequence[str] | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """The worker: start queued jobs in order, one at a time, while no other job runs.

    Returns when the queue is empty (or keeps waiting for new jobs with ``watch``). A failed job does not stop the
    queue; a later job that starts from its model fails fast when that model is missing. Returns 1 if any job failed.
    """
    root = jobs_dir(repo_root)
    with _worker_lock(root) as holder:
        if holder is not None:
            echo(f"a queue worker is already running (pid {holder}); it will pick up new jobs")
            return 1
        failed = False
        waiting_on: str | None = None
        while True:
            busy = active_jobs(repo_root)
            pending = queued_jobs(repo_root)
            if not pending and not watch:
                break
            if busy or not pending:
                if busy and busy[0].id != waiting_on:
                    waiting_on = busy[0].id
                    echo(f"[{_clock()}] waiting for {waiting_on} ({busy[0].state}) before {len(pending)} queued")
                time.sleep(poll_seconds)
                continue
            waiting_on = None
            job = pending[0]
            try:
                claimed = claim(job, pid=os.getpid())
            except JobError:
                continue  # cancelled or started by someone else meanwhile
            echo(f"[{_clock()}] start {job.id}: {job.describe()}")
            try:
                _prepare(claimed, repo_root=repo_root)
            except JobError as exc:
                done = finish(claimed, error=str(exc))
            else:
                process = start(claimed, prefix=prefix, cwd=get_repo_root(repo_root))
                done = finish(claimed, returncode=process.wait())
            failed = failed or done.state == "failed"
            echo(f"[{_clock()}] {done.state} {job.id}" + (f" ({done.xp_sig})" if done.xp_sig else "")
                 + (f": {done.status['error']}" if done.status.get("error") else ""))  # fmt: skip
        return 1 if failed else 0


def worker_pid(repo_root: str | Path | None = None) -> int | None:
    """The running worker's pid, or None when no worker holds the lock."""
    if not (jobs_dir(repo_root) / WORKER_LOCK_FILENAME).exists():
        return None
    with _worker_lock(jobs_dir(repo_root), hold=False) as holder:
        return holder


def ensure_worker(*, prefix: Sequence[str] | None = None, repo_root: str | Path | None = None) -> int:
    """Start a worker in the background unless one is running; its pid either way."""
    running = worker_pid(repo_root)
    if running is not None:
        return running
    root = jobs_dir(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    head = list(prefix) if prefix else [sys.executable, *CLI_PREFIX]
    with (root / WORKER_LOG_FILENAME).open("ab") as log:
        process = subprocess.Popen(
            [*head, "train", "queue", "run"],
            cwd=get_repo_root(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
    import threading

    threading.Thread(target=process.wait, daemon=True).start()
    return process.pid


def _prepare(job: Job, *, repo_root: str | Path | None = None) -> None:
    """What can only be settled when a job starts: the model a resume job names."""
    if job.kind != "resume":
        return
    from ..models.record import ModelNotFound, resolve_sig

    try:
        sig = resolve_sig(str(job.request["model"]), repo_root=repo_root)
    except ModelNotFound as exc:
        raise JobError(str(exc)) from None
    update_job(job, xp_sig=sig, epochs=job.request.get("epochs"))


def _reap(job: Job, process: subprocess.Popen[bytes]) -> None:
    finish(job, returncode=process.wait())


# ── processes ──


def _proc_stat(pid: int) -> tuple[str, int] | None:
    """(state, ppid) from /proc, or None when the process is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    return fields[0], int(fields[1])


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    stat = _proc_stat(int(pid))
    return stat is not None and stat[0] not in ("Z", "X")


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _is_job_process(pid: Any) -> bool:
    """Alive and still ours: a recycled pid running something else does not count."""
    return pid_alive(pid) and "macchina_ricordi.cli" in cmdline(int(pid))


def descendants(pid: int) -> list[int]:
    """Every live process below ``pid``, parents before children."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        stat = _proc_stat(int(entry.name))
        if stat is not None:
            children.setdefault(stat[1], []).append(int(entry.name))
    ordered: list[int] = []
    frontier = [pid]
    while frontier:
        current = frontier.pop(0)
        for child in sorted(children.get(current, [])):
            if child not in ordered:
                ordered.append(child)
                frontier.append(child)
    return ordered


def signal_all(pids: Sequence[int], sig: signal.Signals) -> list[int]:
    sent = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except OSError:
            continue
    return sent


def signal_group(pgid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except OSError:
        return False


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except OSError:
        return False


# ── files ──


def _new_id(root: Path, hint: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", hint.lower()).strip("_")[:40]
    base = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + (f"_{slug}" if slug else "")
    job_id, n = base, 1
    while (root / job_id).exists():
        n += 1
        job_id = f"{base}-{n}"
    return job_id


def _write_status(path: Path, current: dict[str, Any], **fields: Any) -> None:
    _write_json(path / STATUS_FILENAME, {**current, **fields, "updated_at": now_iso()})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, default=str) + "\n")


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    """Serialise status changes between the worker, the GUI and the CLI."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _worker_lock(root: Path, *, hold: bool = True) -> Iterator[int | None]:
    """Yields None when this process got the worker lock, else the pid of the worker that holds it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / WORKER_LOCK_FILENAME
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            text = handle.read().strip()
            yield int(text) if text.isdigit() else -1
            return
        try:
            if hold:
                handle.seek(0)
                handle.truncate()
                handle.write(str(os.getpid()))
                handle.flush()
            yield None
        finally:
            if hold:
                handle.seek(0)
                handle.truncate()
            fcntl.flock(handle, fcntl.LOCK_UN)


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")
