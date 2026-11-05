"""Run a planned training, recording it in the job's folder (``train_plan.json``, ``train_run_status.json``)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ..musicgen.runtime import (
    build_musicgen_subprocess_env,
)
from ..musicgen.shared import TaskSidecar, utc_now_iso, write_json
from .plan import MusicGenTrainingPlan
from .watchdog import MusicGenGpuWatchdogConfig, monitor_musicgen_process


def _write_training_metadata(plan: MusicGenTrainingPlan) -> None:
    if plan.train_metadata_path is None or plan.train_metadata is None:
        return
    payload = dict(plan.train_metadata)
    payload["written_at"] = utc_now_iso()
    write_json(plan.train_metadata_path, payload)


def _training_status_payload(
    plan: MusicGenTrainingPlan,
    *,
    torch_force_no_weights_only_load: bool,
) -> dict[str, object]:
    return {
        "workspace_dir": str(plan.workspace_dir),
        "audiocraft_dir": str(plan.audiocraft_dir),
        "dataset_config_path": str(plan.dataset_config_path),
        "command": plan.command,
        "xp_sig": plan.xp_sig,
        "xp_folder": str(plan.xp_folder) if plan.xp_folder is not None else None,
        "effective_label": plan.effective_label,
        "auto_dataset_label": plan.auto_dataset_label,
        "dataset_fingerprint_digest": plan.dataset_fingerprint_digest,
        "torch_force_no_weights_only_load": torch_force_no_weights_only_load,
    }


def _resolve_continue_from_override(command: Sequence[str]) -> str | None:
    for part in reversed(command):
        if part.startswith("continue_from="):
            return part.split("=", 1)[1]
    return None


def _should_force_non_weights_only_checkpoint_loads(plan: MusicGenTrainingPlan) -> bool:
    if plan.init_from_checkpoint is not None:
        return True
    continue_from = _resolve_continue_from_override(plan.command)
    if continue_from is None:
        return False
    return not continue_from.startswith("//")


def _build_training_subprocess_env(plan: MusicGenTrainingPlan) -> dict[str, str]:
    env = build_musicgen_subprocess_env()
    if _should_force_non_weights_only_checkpoint_loads(plan):
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return env


def _record_plan(plan: MusicGenTrainingPlan, record_dir: Path) -> None:
    if plan.plan_record is not None:
        write_json(record_dir / "train_plan.json", plan.plan_record)


def run_musicgen_training(plan: MusicGenTrainingPlan, *, record_dir: Path) -> int:
    """Run the planned training, recording ``train_plan.json`` and ``train_run_status.json`` in ``record_dir``
    (the job's folder; datasets never receive run records)."""
    forced_non_weights_only = _should_force_non_weights_only_checkpoint_loads(plan)
    status_path = record_dir / "train_run_status.json"
    sidecar = TaskSidecar(
        status_path,
        **_training_status_payload(
            plan,
            torch_force_no_weights_only_load=forced_non_weights_only,
        ),
    )

    if plan.dry_run:
        sidecar.update(status="dry_run")
        sidecar.write()
        return 0

    _record_plan(plan, record_dir)
    _write_training_metadata(plan)
    with sidecar:
        try:
            completed = subprocess.run(
                plan.command,
                check=True,
                cwd=plan.audiocraft_dir,
                env=_build_training_subprocess_env(plan),
            )
            sidecar.update(returncode=int(completed.returncode))
        except FileNotFoundError as exc:
            sidecar.update(
                missing_executable=exc.filename,
                hints=[
                    "Verify the selected musicgen Python has dora installed and available next to it.",
                    "If later-stage sample export fails, ensure ffmpeg is installed and on PATH.",
                ],
            )
            raise RuntimeError(f"MusicGen training failed before launch. See {status_path} for details.") from exc
        except subprocess.CalledProcessError as exc:
            sidecar.update(
                returncode=exc.returncode,
                hints=[
                    "Inspect the training traceback/output near the failure point for the real cause.",
                    (
                        "If the traceback mentions ffmpeg during generate/sample export, "
                        "install ffmpeg or disable generate stages."
                    ),
                    (
                        "If the traceback mentions WinError 183 on checkpoint/history writes, "
                        "a Windows-safe atomic write patch may still be needed in external training dependencies."
                    ),
                ],
            )
            raise RuntimeError(
                f"MusicGen training failed with exit code {exc.returncode}. See {status_path} for a structured summary."
            ) from exc

    return int(completed.returncode)


def run_musicgen_training_with_watchdog(
    plan: MusicGenTrainingPlan,
    *,
    record_dir: Path,
    gpu_watchdog: MusicGenGpuWatchdogConfig,
) -> int:
    forced_non_weights_only = _should_force_non_weights_only_checkpoint_loads(plan)
    status_path = record_dir / "train_run_status.json"
    events_path = gpu_watchdog.events_path or record_dir / "gpu_watchdog_events.jsonl"
    sidecar = TaskSidecar(
        status_path,
        **_training_status_payload(
            plan,
            torch_force_no_weights_only_load=forced_non_weights_only,
        ),
        gpu_watchdog={
            "enabled": True,
            "gpu_index": gpu_watchdog.gpu_index,
            "poll_interval_seconds": gpu_watchdog.poll_interval_seconds,
            "temperature_trip_c": gpu_watchdog.temperature_trip_c,
            "events_path": str(events_path),
        },
    )

    if plan.dry_run:
        sidecar.update(status="dry_run")
        sidecar.write()
        return 0

    resolved_watchdog = MusicGenGpuWatchdogConfig(
        gpu_index=gpu_watchdog.gpu_index,
        poll_interval_seconds=gpu_watchdog.poll_interval_seconds,
        temperature_trip_c=gpu_watchdog.temperature_trip_c,
        events_path=events_path,
    )

    _record_plan(plan, record_dir)
    _write_training_metadata(plan)
    with sidecar:
        try:
            process = subprocess.Popen(
                plan.command,
                cwd=plan.audiocraft_dir,
                env=_build_training_subprocess_env(plan),
                start_new_session=True,
            )
            sidecar.update(train_pid=process.pid)
            try:
                sidecar.update(train_process_group_id=os.getpgid(process.pid))
            except OSError:
                pass
            sidecar.write()

            watchdog_events = monitor_musicgen_process(process, config=resolved_watchdog)
            returncode = int(process.wait())
            if watchdog_events:
                gpu_wd = sidecar.payload.get("gpu_watchdog")
                if isinstance(gpu_wd, dict):
                    gpu_wd["events"] = watchdog_events

            sidecar.update(returncode=returncode)

            if returncode != 0:
                sidecar.update(
                    hints=[
                        "Inspect the training traceback/output near the failure point for the real cause.",
                        "If the GPU watchdog tripped, check the structured events log alongside nvidia-smi telemetry.",
                        (
                            "If the traceback mentions ffmpeg during generate/sample export, "
                            "install ffmpeg or disable generate stages."
                        ),
                        (
                            "If the traceback mentions WinError 183 on checkpoint/history writes, "
                            "a Windows-safe atomic write patch may still be needed in external training dependencies."
                        ),
                    ],
                )
                raise RuntimeError(
                    f"MusicGen training failed with exit code {returncode}. See {status_path} for a structured summary."
                )
        except FileNotFoundError as exc:
            sidecar.update(
                missing_executable=exc.filename,
                hints=[
                    "Verify the selected musicgen Python has dora installed and available next to it.",
                    "If later-stage sample export fails, ensure ffmpeg is installed and on PATH.",
                ],
            )
            raise RuntimeError(f"MusicGen training failed before launch. See {status_path} for details.") from exc

    return returncode
