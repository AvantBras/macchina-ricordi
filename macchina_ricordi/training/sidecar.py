from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models.checkpoints import discover_xp_folder
from ..musicgen.config import musicgen_script_env
from ..musicgen.runtime import load_musicgen_project_config, require_musicgen_python
from ..musicgen.shared import utc_now_iso

SIDECAR_STATE_VERSION = 1
SCORES_JSONL_SCHEMA_VERSION = 1
CHECKPOINT_EPOCH_PATTERN = re.compile(r"^checkpoint_(\d+)\.th$")
OPTIM_EPOCHS_OVERRIDE_PATTERN = re.compile(r"^\s*-\s*optim\.epochs\s*=\s*(\d+)\s*$")
STAGE_MODE_TO_STAGES: dict[str, tuple[str, ...]] = {
    "generate": ("generate",),
    "evaluate": ("evaluate",),
    "both": ("evaluate", "generate"),
}
GENERATED_SAMPLES_DIRNAME = "samples"
GENERATED_SAMPLE_GLOB = "*.wav"
GENERATED_SAMPLES_RESERVED_DIRS: frozenset[str] = frozenset({"prompt", "reference"})


@dataclass(frozen=True)
class MusicGenCheckpointEpoch:
    epoch: int
    path: Path


@dataclass(frozen=True)
class MusicGenCpuSidecarPlan:
    xp_path: Path
    xp_sig: str
    dora_dir: Path
    repo_root: Path
    musicgen_python: str
    stages: tuple[str, ...]
    poll_seconds: float
    num_threads: int
    batch_size: int | None
    once: bool
    artifacts_dir: Path
    state_path: Path
    events_path: Path
    terminal_epoch: int | None
    score_samples: bool
    scores_path: Path


def plan_musicgen_cpu_sidecar(
    *,
    xp_path: str | Path | None = None,
    musicgen_python: str | None = None,
    stage_mode: str = "both",
    poll_seconds: float = 30.0,
    num_threads: int | None = None,
    batch_size: int | None = None,
    once: bool = False,
    score_samples: bool = True,
) -> MusicGenCpuSidecarPlan:
    resolved_xp_path = resolve_sidecar_xp_path(xp_path)
    config = load_musicgen_project_config()
    resolved_musicgen_python = require_musicgen_python(musicgen_python)
    resolved_num_threads = max(1, num_threads or (os.cpu_count() or 1))
    stages = STAGE_MODE_TO_STAGES[stage_mode]
    artifacts_dir = resolved_xp_path / "sidecar_cpu"
    terminal_epoch = resolve_terminal_epoch_from_hydra_overrides(resolved_xp_path)

    return MusicGenCpuSidecarPlan(
        xp_path=resolved_xp_path,
        xp_sig=resolved_xp_path.name,
        dora_dir=resolved_xp_path.parent.parent,
        repo_root=config.repo_root,
        musicgen_python=resolved_musicgen_python,
        stages=stages,
        poll_seconds=poll_seconds,
        num_threads=resolved_num_threads,
        batch_size=batch_size,
        once=once,
        artifacts_dir=artifacts_dir,
        state_path=artifacts_dir / "state.json",
        events_path=artifacts_dir / "events.jsonl",
        terminal_epoch=terminal_epoch,
        score_samples=score_samples,
        scores_path=artifacts_dir / "scores.jsonl",
    )


def resolve_sidecar_xp_path(xp_path: str | Path | None = None) -> Path:
    if xp_path is not None:
        path = Path(xp_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"XP directory does not exist: {path}")
        return path
    return discover_xp_folder(xp_path)


def run_musicgen_cpu_sidecar(plan: MusicGenCpuSidecarPlan) -> int:
    plan.artifacts_dir.mkdir(parents=True, exist_ok=True)
    state = load_musicgen_cpu_sidecar_state(plan.state_path)

    try:
        while True:
            candidate = select_latest_checkpoint_for_sidecar(
                list_epoch_checkpoints(plan.xp_path),
                state=state,
                stages=plan.stages,
            )
            if candidate is None:
                if plan.once:
                    print("pending_checkpoint: none")
                    return 0
                time.sleep(plan.poll_seconds)
                continue

            print(f"pending_checkpoint_epoch: {candidate.epoch}")
            print(f"pending_checkpoint_path: {candidate.path}")
            for stage in plan.stages:
                if candidate.epoch <= get_last_processed_epoch(state, stage):
                    continue
                if stage == "generate" and plan.terminal_epoch is not None and candidate.epoch == plan.terminal_epoch:
                    mark_stage_epoch_processed(state, stage=stage, epoch=candidate.epoch)
                    save_musicgen_cpu_sidecar_state(plan.state_path, state)
                    append_musicgen_sidecar_event(
                        plan.events_path,
                        {
                            "event": "stage_skipped_terminal_generate",
                            "xp_sig": plan.xp_sig,
                            "stage": stage,
                            "epoch": candidate.epoch,
                            "terminal_epoch": plan.terminal_epoch,
                            "checkpoint_path": str(candidate.path),
                        },
                    )
                    print(f"skipped_stage: {stage}")
                    print(f"skipped_epoch: {candidate.epoch}")
                    print("skip_reason: terminal_epoch_generate")
                    continue
                log_path = plan.artifacts_dir / f"{stage}_epoch_{candidate.epoch:04d}.log"
                command = build_musicgen_sidecar_runner_command(
                    plan=plan,
                    checkpoint=candidate,
                    stage=stage,
                )
                append_musicgen_sidecar_event(
                    plan.events_path,
                    {
                        "event": "stage_started",
                        "xp_sig": plan.xp_sig,
                        "stage": stage,
                        "epoch": candidate.epoch,
                        "checkpoint_path": str(candidate.path),
                        "log_path": str(log_path),
                        "command": command,
                    },
                )
                with log_path.open("w", encoding="utf-8") as log_file:
                    completed = subprocess.run(
                        command,
                        check=False,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=musicgen_script_env(dora_dir=plan.dora_dir, repo_root=plan.repo_root),
                        cwd=plan.repo_root,
                    )
                if completed.returncode != 0:
                    append_musicgen_sidecar_event(
                        plan.events_path,
                        {
                            "event": "stage_failed",
                            "xp_sig": plan.xp_sig,
                            "stage": stage,
                            "epoch": candidate.epoch,
                            "checkpoint_path": str(candidate.path),
                            "log_path": str(log_path),
                            "returncode": int(completed.returncode),
                        },
                    )
                    raise RuntimeError(f"CPU sidecar stage {stage} failed for epoch {candidate.epoch}. See {log_path}.")

                mark_stage_epoch_processed(state, stage=stage, epoch=candidate.epoch)
                save_musicgen_cpu_sidecar_state(plan.state_path, state)
                append_musicgen_sidecar_event(
                    plan.events_path,
                    {
                        "event": "stage_completed",
                        "xp_sig": plan.xp_sig,
                        "stage": stage,
                        "epoch": candidate.epoch,
                        "checkpoint_path": str(candidate.path),
                        "log_path": str(log_path),
                    },
                )
                print(f"completed_stage: {stage}")
                print(f"completed_epoch: {candidate.epoch}")
                print(f"stage_log: {log_path}")

                if stage == "generate" and plan.score_samples:
                    score_samples_for_epoch(
                        plan=plan,
                        epoch=candidate.epoch,
                    )

            if plan.once:
                return 0
    except KeyboardInterrupt:
        append_musicgen_sidecar_event(
            plan.events_path,
            {
                "event": "sidecar_interrupted",
                "xp_sig": plan.xp_sig,
            },
        )
        return 130


def list_epoch_checkpoints(xp_path: Path) -> list[MusicGenCheckpointEpoch]:
    checkpoints: list[MusicGenCheckpointEpoch] = []
    for path in xp_path.glob("checkpoint_*.th"):
        epoch = parse_musicgen_checkpoint_epoch(path)
        if epoch is not None:
            checkpoints.append(MusicGenCheckpointEpoch(epoch=epoch, path=path.resolve()))
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.epoch)


def parse_musicgen_checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_EPOCH_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_terminal_epoch_from_hydra_overrides(xp_path: Path) -> int | None:
    overrides_path = xp_path / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return None
    try:
        lines = overrides_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    terminal_epoch: int | None = None
    for line in lines:
        match = OPTIM_EPOCHS_OVERRIDE_PATTERN.match(line)
        if match is not None:
            terminal_epoch = int(match.group(1))
    return terminal_epoch


def select_latest_checkpoint_for_sidecar(
    checkpoints: Sequence[MusicGenCheckpointEpoch],
    *,
    state: dict[str, object],
    stages: Sequence[str],
) -> MusicGenCheckpointEpoch | None:
    if not checkpoints:
        return None
    if any(checkpoints[-1].epoch > get_last_processed_epoch(state, stage) for stage in stages):
        return checkpoints[-1]
    return None


def build_musicgen_sidecar_runner_command(
    *,
    plan: MusicGenCpuSidecarPlan,
    checkpoint: MusicGenCheckpointEpoch,
    stage: str,
) -> list[str]:
    command = [
        plan.musicgen_python,
        "-m",
        "macchina_ricordi.training.sidecar_runner",
        "--sig",
        plan.xp_sig,
        "--checkpoint-path",
        str(checkpoint.path),
        "--stage",
        stage,
        "--num-threads",
        str(plan.num_threads),
    ]
    if plan.batch_size is not None:
        command.extend(["--batch-size", str(plan.batch_size)])
    return command


def load_musicgen_cpu_sidecar_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "version": SIDECAR_STATE_VERSION,
            "last_processed_epoch": {},
            "updated_at": utc_now_iso(),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != SIDECAR_STATE_VERSION:
        raise RuntimeError(f"Unsupported CPU sidecar state version in {path}")
    if not isinstance(payload.get("last_processed_epoch"), dict):
        raise RuntimeError(f"Malformed CPU sidecar state file at {path}")
    return payload


def save_musicgen_cpu_sidecar_state(path: Path, state: dict[str, object]) -> None:
    payload = dict(state)
    payload["version"] = SIDECAR_STATE_VERSION
    payload["updated_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_last_processed_epoch(state: dict[str, object], stage: str) -> int:
    last_processed = state.get("last_processed_epoch", {})
    if not isinstance(last_processed, dict):
        return 0
    raw_value = last_processed.get(stage, 0)
    if isinstance(raw_value, int):
        return raw_value
    return 0


def mark_stage_epoch_processed(
    state: dict[str, object],
    *,
    stage: str,
    epoch: int,
) -> None:
    last_processed = state.setdefault("last_processed_epoch", {})
    if not isinstance(last_processed, dict):
        raise RuntimeError("CPU sidecar state is malformed: last_processed_epoch is not a dict")
    last_processed[stage] = epoch


def append_musicgen_sidecar_event(path: Path, payload: dict[str, object]) -> None:
    event = dict(payload)
    event["timestamp"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event))
        handle.write("\n")


def discover_generated_samples(xp_path: Path, epoch: int) -> list[Path]:
    epoch_dir = xp_path / GENERATED_SAMPLES_DIRNAME / str(epoch)
    if not epoch_dir.exists() or not epoch_dir.is_dir():
        return []
    samples: list[Path] = []
    for entry in sorted(epoch_dir.iterdir()):
        if entry.is_dir():
            continue
        if entry.suffix.lower() != ".wav":
            continue
        samples.append(entry.resolve())
    return samples


def load_existing_scored_sample_ids(scores_path: Path, epoch: int) -> set[str]:
    if not scores_path.exists():
        return set()
    scored: set[str] = set()
    with scores_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("epoch") != epoch:
                continue
            sample_id = payload.get("sample_id")
            if isinstance(sample_id, str):
                scored.add(sample_id)
    return scored


def append_score_row(scores_path: Path, payload: dict[str, Any]) -> None:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def score_samples_for_epoch(
    *,
    plan: MusicGenCpuSidecarPlan,
    epoch: int,
    scorer: Callable[[Path], Any] | None = None,
    reward_components_cls: type[Any] | None = None,
) -> dict[str, Any]:
    try:
        if reward_components_cls is None:
            from macchina_ricordi.scoring.reward import RewardComponents

            reward_components_cls = RewardComponents
        if scorer is None:
            from macchina_ricordi.scoring.reward import score_arrhythmic_percussive

            scorer = score_arrhythmic_percussive
    except Exception as exc:
        append_musicgen_sidecar_event(
            plan.events_path,
            {
                "event": "samples_scoring_failed",
                "xp_sig": plan.xp_sig,
                "epoch": epoch,
                "error": str(exc),
            },
        )
        return {
            "epoch": epoch,
            "n_total": 0,
            "n_failed": 0,
            "n_skipped": 0,
            "scores_path": str(plan.scores_path),
        }

    samples = discover_generated_samples(plan.xp_path, epoch)
    already_scored = load_existing_scored_sample_ids(plan.scores_path, epoch)
    fresh_samples = [path for path in samples if path.stem not in already_scored]
    if not fresh_samples:
        append_musicgen_sidecar_event(
            plan.events_path,
            {
                "event": "samples_scored",
                "xp_sig": plan.xp_sig,
                "epoch": epoch,
                "n_total": 0,
                "n_failed": 0,
                "n_skipped": len(already_scored),
                "scores_path": str(plan.scores_path),
            },
        )
        return {"epoch": epoch, "n_total": 0, "n_failed": 0, "n_skipped": len(already_scored)}

    default_components = reward_components_cls().to_dict()
    failures = 0
    score_values: list[float] = []
    for path in fresh_samples:
        row: dict[str, Any] = {
            "schema_version": SCORES_JSONL_SCHEMA_VERSION,
            "epoch": int(epoch),
            "sample_id": path.stem,
            "wav_path": str(path),
            "scored_at": utc_now_iso(),
        }
        row.update(default_components)
        row["score_error"] = None
        try:
            components = scorer(path)
            row.update(components.to_dict())
            score_values.append(float(row.get("score", 0.0)))
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            row["score_error"] = str(exc)
            append_musicgen_sidecar_event(
                plan.events_path,
                {
                    "event": "sample_score_failed",
                    "xp_sig": plan.xp_sig,
                    "epoch": int(epoch),
                    "sample_id": path.stem,
                    "wav_path": str(path),
                    "error": str(exc),
                },
            )
        append_score_row(plan.scores_path, row)

    summary: dict[str, Any] = {
        "event": "samples_scored",
        "xp_sig": plan.xp_sig,
        "epoch": int(epoch),
        "n_total": len(fresh_samples),
        "n_failed": failures,
        "n_skipped": len(already_scored),
        "scores_path": str(plan.scores_path),
    }
    if score_values:
        summary["mean_score"] = float(sum(score_values) / len(score_values))
        summary["min_score"] = float(min(score_values))
        summary["max_score"] = float(max(score_values))
    append_musicgen_sidecar_event(plan.events_path, summary)
    print(f"scored_epoch: {epoch}")
    print(f"scored_samples: {len(fresh_samples)}")
    if failures:
        print(f"score_failures: {failures}")
    return {
        "epoch": int(epoch),
        "n_total": len(fresh_samples),
        "n_failed": failures,
        "n_skipped": len(already_scored),
    }
