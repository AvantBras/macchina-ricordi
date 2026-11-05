"""Resume a model in place to a new epoch target, now or once a running epoch is written."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..musicgen.runtime import (
    build_musicgen_subprocess_env,
    load_musicgen_project_config,
    normalize_python_executable_path,
)
from ..musicgen.shared import TaskSidecar, bootstrap_audiocraft_dora_dir, format_command, utc_now_iso, write_json
from ..musicgen.xp_datasource import XpDatasetConfig, restore_xp_dataset_config
from .pool import resume_problem as pool_resume_problem


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _history_entry_count(xp_folder: Path) -> int | None:
    history_path = xp_folder / "history.json"
    if not history_path.exists():
        return None
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        return len(payload)
    return None


def _file_snapshot(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return {
        "path": str(path),
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
        "size_bytes": path.stat().st_size,
    }


def _failure_hints(exc: Exception) -> list[str]:
    text = str(exc).lower()
    hints: list[str] = []
    if "weights only load failed" in text:
        hints.append(
            "PyTorch refused to unpickle a trusted local XP checkpoint with weights_only=True; "
            "resume-xp sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for in-place resumes."
        )
    if "history.json" in text and "winerror 183" in text:
        hints.append("Dora hit a Windows atomic-rename issue while updating history.json.")
    if "checkpoint.th" in text and "winerror 183" in text:
        hints.append("AudioCraft hit a Windows atomic-rename issue while saving checkpoint.th.")
    if ("file not found" in text or "winerror 2" in text) and shutil.which("ffmpeg") is None:
        hints.append("ffmpeg is not on PATH; AudioCraft sample export can fail during generate stages or final epochs.")
    if not hints:
        hints.append("Inspect the traceback below plus history.json/checkpoint timestamps to see how far the run got.")
    return hints


def _log_name(cfg) -> str:  # type: ignore[no-untyped-def]
    if cfg.execute_only:
        return f"{cfg.execute_only}.log.{{rank}}"
    return "solver.log.{rank}"


def _runtime_modules_available() -> bool:
    return all(importlib.util.find_spec(module_name) is not None for module_name in ("audiocraft", "flashy", "hydra"))


def _resolve_runtime_python(explicit_python: str | None = None) -> Path:
    if explicit_python:
        runtime_python = normalize_python_executable_path(explicit_python)
    else:
        runtime_python = normalize_python_executable_path(load_musicgen_project_config().musicgen_python)
    if not runtime_python.exists():
        raise FileNotFoundError(
            f"MusicGen runtime Python does not exist: {runtime_python}. "
            "Run `make setup-musicgen` before resuming training."
        )
    return runtime_python


def _resolve_resume_command_python(explicit_python: str | None = None) -> str:
    if explicit_python:
        return str(normalize_python_executable_path(explicit_python))
    return sys.executable


def _run_runtime_command(command: list[str]) -> int:
    config = load_musicgen_project_config()
    env = build_musicgen_subprocess_env(base_env=dict(os.environ))
    print(f"runtime_command: {format_command(command)}")
    completed = subprocess.run(
        command,
        cwd=config.repo_root,
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _delegate_resume_xp(sig: str, *, epochs: int, resume_python: str | None = None) -> int:
    runtime_python = _resolve_runtime_python(resume_python)
    if normalize_python_executable_path(sys.executable) == runtime_python:
        raise RuntimeError(
            f"AudioCraft is not importable in the selected resume Python. Selected interpreter: {runtime_python}"
        )
    command = [
        str(runtime_python),
        "-m",
        "macchina_ricordi.cli",
        "train",
        "resume",
        sig,
        "--epochs",
        str(epochs),
        "--in-process",
    ]
    return _run_runtime_command(command)


def _dora_dir_from_env_or_config() -> Path:
    dora_env = os.getenv("AUDIOCRAFT_DORA_DIR")
    if dora_env:
        return Path(dora_env).expanduser().resolve()
    return load_musicgen_project_config().dora_dir


def _xp_folder_for_sig(sig: str) -> Path:
    return _dora_dir_from_env_or_config() / "xps" / sig


def _restore_xp_dataset_config(sig: str) -> XpDatasetConfig:
    restored = restore_xp_dataset_config(
        _xp_folder_for_sig(sig),
        audiocraft_dir=load_musicgen_project_config().audiocraft_dir,
        purpose="resume",
    )
    if restored.status == "restored":
        print(f"restored_dataset_config: {restored.config_path}")
        print(f"dataset_train_dir: {restored.train_dir}")
        if restored.changed:
            print(f"dataset_config_was: {restored.previous_train_dir}")
    return restored


def _allow_trusted_xp_checkpoint_loads() -> str:
    return os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _run_resume_xp_in_process(sig: str, *, epochs: int) -> int:
    bootstrap_audiocraft_dora_dir()
    _restore_xp_dataset_config(sig)
    trusted_checkpoint_load = _allow_trusted_xp_checkpoint_loads()

    import flashy
    import hydra
    from audiocraft import train
    from omegaconf import DictConfig, OmegaConf

    if "USER" not in os.environ and os.getenv("USERNAME"):
        os.environ["USER"] = os.environ["USERNAME"]

    xp = train.main.get_xp_from_sig(sig)
    cfg = OmegaConf.merge(xp.cfg, DictConfig({"optim": {"epochs": epochs}}))
    xp_folder = Path(xp.folder)
    status_path = xp_folder / "resume_status.json"

    print(f"resume_sig: {xp.sig}")
    print(f"xp_folder: {xp.folder}")
    print(f"target_epochs: {epochs}")

    sidecar = TaskSidecar(
        status_path,
        resume_sig=xp.sig,
        xp_folder=str(xp_folder),
        target_epochs=epochs,
        device=str(cfg.device),
        execute_only=str(cfg.execute_only or ""),
        ffmpeg_on_path=shutil.which("ffmpeg"),
        torch_force_no_weights_only_load=trusted_checkpoint_load,
    )

    try:
        with sidecar, xp.enter():
            train.init_seed_and_system(cfg)
            flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=_log_name(cfg))
            flashy.distrib.init()
            solver = train.get_solver(cfg)
            if cfg.show:
                solver.show()
            else:
                solver.run()

            sidecar.update(
                history_entries=_history_entry_count(xp_folder),
                checkpoint=_file_snapshot(xp_folder / "checkpoint.th"),
                history=_file_snapshot(xp_folder / "history.json"),
                solver_log=_file_snapshot(xp_folder / "solver.log.0"),
            )
    except Exception as exc:
        sidecar.update(
            history_entries=_history_entry_count(xp_folder),
            checkpoint=_file_snapshot(xp_folder / "checkpoint.th"),
            history=_file_snapshot(xp_folder / "history.json"),
            solver_log=_file_snapshot(xp_folder / "solver.log.0"),
            hints=_failure_hints(exc),
        )
        sidecar.write()
        print(f"resume_status: {status_path}", file=sys.stderr)
        for hint in sidecar.payload.get("hints", []):
            print(f"hint: {hint}", file=sys.stderr)
        raise
    finally:
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    return 0


def run_resume_xp(
    sig: str,
    *,
    epochs: int,
    resume_python: str | None = None,
    in_process: bool = False,
) -> int:
    if problem := pool_resume_problem(_xp_folder_for_sig(sig), epochs):
        print(f"error: {problem}", file=sys.stderr)
        return 1
    if not in_process and not _runtime_modules_available():
        return _delegate_resume_xp(sig, epochs=epochs, resume_python=resume_python)
    return _run_resume_xp_in_process(sig, epochs=epochs)


def _is_pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _solver_log_tail(solver_log: Path) -> str | None:
    if not solver_log.exists():
        return None
    try:
        lines = solver_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.strip():
            return line
    return None


def _log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def run_handoff_resume(
    sig: str,
    *,
    wait_for_epoch: int,
    epochs: int,
    active_pid: int | None = None,
    grace_seconds: int = 600,
    poll_seconds: int = 120,
    resume_python: str | None = None,
    resume_log_path: str | Path | None = None,
    status_path: str | Path | None = None,
    in_process: bool = False,
) -> int:
    if wait_for_epoch < 1:
        raise ValueError("--wait-for-epoch must be >= 1.")
    if epochs <= wait_for_epoch:
        raise ValueError("--epochs must be greater than --wait-for-epoch.")
    if grace_seconds < 0:
        raise ValueError("--grace-seconds must be >= 0.")
    if poll_seconds < 1:
        raise ValueError("--poll-seconds must be >= 1.")
    if problem := pool_resume_problem(_xp_folder_for_sig(sig), epochs):
        raise ValueError(problem)

    if not in_process and not _runtime_modules_available():
        return _delegate_handoff_resume(
            sig,
            wait_for_epoch=wait_for_epoch,
            epochs=epochs,
            active_pid=active_pid,
            grace_seconds=grace_seconds,
            poll_seconds=poll_seconds,
            resume_python=resume_python,
            resume_log_path=resume_log_path,
            status_path=status_path,
        )

    bootstrap_audiocraft_dora_dir()
    from audiocraft import train

    xp = train.main.get_xp_from_sig(sig)
    xp_folder = Path(xp.folder)
    checkpoint_path = xp_folder / f"checkpoint_{wait_for_epoch}.th"
    solver_log = xp_folder / "solver.log.0"
    repo_root = Path(__file__).resolve().parents[2]
    resolved_status_path = (
        Path(status_path).expanduser().resolve() if status_path else xp_folder / "handoff_resume_status.json"
    )
    resolved_resume_log_path = (
        Path(resume_log_path).expanduser().resolve()
        if resume_log_path
        else xp_folder / f"resume_to_{epochs}_stdout.log"
    )
    resolved_resume_python = _resolve_resume_command_python(resume_python)

    status_payload: dict[str, object] = {
        "status": "waiting",
        "started_at": utc_now_iso(),
        "resume_sig": xp.sig,
        "xp_folder": str(xp_folder),
        "wait_for_epoch": wait_for_epoch,
        "target_epochs": epochs,
        "active_pid": active_pid,
        "grace_seconds": grace_seconds,
        "poll_seconds": poll_seconds,
        "checkpoint_path": str(checkpoint_path),
        "solver_log": str(solver_log),
        "resume_log_path": str(resolved_resume_log_path),
        "resume_python": resolved_resume_python,
    }
    write_json(resolved_status_path, status_payload)
    _log(
        f"Handoff watcher armed for XP {xp.sig}; waiting for epoch {wait_for_epoch} "
        f"before resuming to {epochs} total epochs."
    )

    grace_deadline: datetime | None = None
    while True:
        pid_running = _is_pid_running(active_pid)
        checkpoint_ready = checkpoint_path.exists()
        latest_line = _solver_log_tail(solver_log)

        status_payload.update(
            {
                "last_checked_at": utc_now_iso(),
                "active_pid_running": pid_running,
                "checkpoint_ready": checkpoint_ready,
                "latest_solver_log_line": latest_line,
            }
        )

        if pid_running:
            status_payload["status"] = "waiting"
            write_json(resolved_status_path, status_payload)
            _log(f"Current run still active; latest={latest_line}")
            time.sleep(poll_seconds)
            continue

        if not checkpoint_ready:
            status_payload["status"] = "waiting_for_checkpoint"
            write_json(resolved_status_path, status_payload)
            _log(f"Current pid is gone, but {checkpoint_path.name} is not present yet; latest={latest_line}")
            time.sleep(poll_seconds)
            continue

        if grace_deadline is None:
            grace_deadline = _utc_now() + timedelta(seconds=grace_seconds)
            status_payload["status"] = "waiting_grace"
            status_payload["grace_until"] = grace_deadline.isoformat()
            write_json(resolved_status_path, status_payload)
            _log(
                f"Epoch {wait_for_epoch} completion confirmed. Waiting an extra "
                f"{grace_seconds} seconds before resuming."
            )

        remaining_seconds = (grace_deadline - _utc_now()).total_seconds()
        if remaining_seconds > 0:
            time.sleep(min(poll_seconds, max(1, int(remaining_seconds))))
            continue

        status_payload["status"] = "launching_resume"
        status_payload["resume_started_at"] = utc_now_iso()
        write_json(resolved_status_path, status_payload)
        _log(f"Launching in-place resume for XP {xp.sig} to total epoch target {epochs}.")

        command = [
            resolved_resume_python,
            "-m",
            "macchina_ricordi.cli",
            "train",
            "resume",
            xp.sig,
            "--epochs",
            str(epochs),
        ]
        resolved_resume_log_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_resume_log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        status_payload["finished_at"] = utc_now_iso()
        status_payload["resume_returncode"] = completed.returncode
        if completed.returncode == 0:
            status_payload["status"] = "completed"
            write_json(resolved_status_path, status_payload)
            _log("Resume process completed successfully.")
            return 0

        status_payload["status"] = "failed"
        write_json(resolved_status_path, status_payload)
        _log(f"Resume process failed with exit code {completed.returncode}. See {resolved_resume_log_path}.")
        return int(completed.returncode)


def _delegate_handoff_resume(
    sig: str,
    *,
    wait_for_epoch: int,
    epochs: int,
    active_pid: int | None,
    grace_seconds: int,
    poll_seconds: int,
    resume_python: str | None,
    resume_log_path: str | Path | None,
    status_path: str | Path | None,
) -> int:
    runtime_python = _resolve_runtime_python(None)
    if normalize_python_executable_path(sys.executable) == runtime_python:
        raise RuntimeError(
            f"AudioCraft is not importable in the configured MusicGen runtime. Selected interpreter: {runtime_python}"
        )

    command = [
        str(runtime_python),
        "-m",
        "macchina_ricordi.cli",
        "train",
        "handoff",
        sig,
        "--wait-for-epoch",
        str(wait_for_epoch),
        "--epochs",
        str(epochs),
        "--grace-seconds",
        str(grace_seconds),
        "--poll-seconds",
        str(poll_seconds),
        "--in-process",
    ]
    if active_pid is not None:
        command.extend(["--active-pid", str(active_pid)])
    if resume_python is not None:
        command.extend(["--resume-python", str(resume_python)])
    if resume_log_path is not None:
        command.extend(["--resume-log-path", str(resume_log_path)])
    if status_path is not None:
        command.extend(["--status-path", str(status_path)])
    return _run_runtime_command(command)
