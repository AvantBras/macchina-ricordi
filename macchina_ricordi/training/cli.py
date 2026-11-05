"""`train`: run, queue, resume and watch trainings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..cli.common import MacchinaRicordiHelpFormatter, clip
from ..models.cli import (
    apply_model_reference,
    resolve_sig_argument,
)
from ..models.record import dora_dir_or_default, xp_folder_for_sig
from ..musicgen.config import (
    default_recipe_name,
    recipe_choices,
)
from ..training.plateau import DEFAULT_METRIC as DEFAULT_PLATEAU_METRIC
from .plateau import build_plateau_watcher_config, run_plateau_watcher
from .resume import run_handoff_resume, run_resume_xp


def cmd_train_resume(args: argparse.Namespace) -> int:
    resolve_sig_argument(args)
    return run_resume_xp(
        args.sig,
        epochs=args.epochs,
        resume_python=args.resume_python,
        in_process=args.in_process,
    )


def cmd_train_watch(args: argparse.Namespace) -> int:
    resolve_sig_argument(args)
    xp_folder = xp_folder_for_sig(
        args.sig,
        dora_dir=dora_dir_or_default(args.dora_dir) if args.dora_dir else None,
    )
    events_path = Path(args.events_path).expanduser().resolve() if args.events_path else None
    status_path = Path(args.status_path).expanduser().resolve() if args.status_path else None
    config = build_plateau_watcher_config(
        sig=args.sig,
        xp_folder=xp_folder,
        metric=args.metric,
        patience_epochs=args.patience_epochs,
        min_improvement=args.min_improvement,
        poll_seconds=args.poll_seconds,
        action=args.action,
        stop_pid=args.stop_pid,
        events_path=events_path,
        status_path=status_path,
        grace_seconds=args.grace_seconds,
        once=args.once,
    )

    print(f"sig: {config.sig}")
    print(f"xp_folder: {config.xp_folder}")
    print(f"history_path: {config.history_path}")
    print(f"metric: {config.metric}")
    print(f"patience_epochs: {config.patience_epochs}")
    print(f"min_improvement: {config.min_improvement}")
    print(f"action: {config.action}")
    print(f"events_path: {config.events_path}")
    print(f"status_path: {config.status_path}")
    return run_plateau_watcher(config)


def cmd_train_handoff(args: argparse.Namespace) -> int:
    resolve_sig_argument(args)
    return run_handoff_resume(
        args.sig,
        wait_for_epoch=args.wait_for_epoch,
        epochs=args.epochs,
        active_pid=args.active_pid,
        grace_seconds=args.grace_seconds,
        poll_seconds=args.poll_seconds,
        resume_python=args.resume_python,
        resume_log_path=args.resume_log_path,
        status_path=args.status_path,
        in_process=args.in_process,
    )


def cmd_train_run(args: argparse.Namespace) -> int:
    """Every real training is a job (``artifacts/train_jobs/<id>/``): this one, or the queued one ``--job`` names."""
    import os
    import sys

    from ..training import jobs as train_jobs

    if args.job and args.dry_run:
        print("error: --job runs a queued job; a dry run has nothing to record", file=sys.stderr)
        return 1
    job = None
    if args.job:
        try:
            job = train_jobs.load_job(args.job)
            if job.state == "queued":
                job = train_jobs.claim(job, pid=os.getpid())
            elif job.state == "starting":
                job = train_jobs.update_job(job, pid=os.getpid())
            else:
                raise train_jobs.JobError(f"job {job.id} is {job.state}, not queued")
        except train_jobs.JobError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    try:
        return _train_run(args, job)
    except KeyboardInterrupt:
        if job is not None:
            train_jobs.update_job(job, stop={"requested_at": train_jobs.now_iso(), "signal": "SIGINT"})
            train_jobs.finish(job, returncode=130)
        raise
    except BaseException as exc:
        if job is not None:
            train_jobs.finish(job, error=f"{type(exc).__name__}: {exc}")
        raise


def _train_run(args: argparse.Namespace, job: Any) -> int:
    import sys
    from pathlib import Path

    from ..data.clip_stats import SegmentDurationMismatch
    from ..models.record import ModelNotFound, resolve_checkpoint_path
    from ..musicgen.shared import format_command
    from ..training import jobs as train_jobs
    from ..training.plan import plan_musicgen_training
    from ..training.run import run_musicgen_training, run_musicgen_training_with_watchdog
    from ..training.watchdog import MusicGenGpuWatchdogConfig

    def fail(message: str) -> int:
        print(f"error: {message}", file=sys.stderr)
        if job is not None:
            train_jobs.finish(job, error=message)
        return 1

    job_argv = train_jobs.train_job_argv(args)
    job_request = train_jobs.train_job_request(args)
    init = args.init_from_checkpoint
    if init and "://" in init:
        return fail("--from accepts local model bundles or checkpoints. Download and extract the model first.")
    if init and not Path(init).expanduser().exists():
        try:
            args.init_from_checkpoint = resolve_checkpoint_path(init)
        except ModelNotFound as exc:
            return fail(f"--from: {exc}")
        print(f"init_from_checkpoint_ref: {init} -> {args.init_from_checkpoint}")

    overrides = [*args.override, *([f"optim.epochs={args.epochs}"] if args.epochs else [])]

    def make_plan(plan_overrides: list[str]) -> Any:
        return plan_musicgen_training(
            audiocraft_dir=args.audiocraft_dir,
            profile_name=args.profile,
            workspace_dir=args.workspace_dir,
            musicgen_python=args.musicgen_python,
            overrides=plan_overrides,
            run_label=args.run_label,
            no_auto_dataset_label=args.no_auto_dataset_label,
            dry_run=args.dry_run,
            init_from_checkpoint=args.init_from_checkpoint,
            allow_segment_padding=args.allow_segment_padding,
            weights=args.weights,
        )

    pool_record = None
    try:
        if args.pool:
            from ..training import pool as token_pool

            if any(item.startswith("cache.") for item in overrides):
                return fail("--pool sets cache.path itself; drop the cache.* override")
            base_plan = make_plan(overrides)
            spec, pool_dir, problem = _prepare_pool(args, base_plan)
            if problem:
                return fail(problem)
            pool_record = {"path": str(pool_dir), "crops": spec.crops, "repeats": round(spec.repeats, 2),
                           "epochs": spec.epochs, "crop_seed": spec.seed}  # fmt: skip
            if not pool_dir.exists() and not args.dry_run:
                job = _start_job(job, job_request, job_argv, status="building")
                if _build_pool(args, base_plan, spec, pool_dir) != 0:
                    return fail(f"token pool build failed; nothing was trained ({pool_dir}.partial is left to inspect)")
            overrides = [*overrides, token_pool.cache_override(pool_dir)]
        plan = make_plan(overrides)
    except SegmentDurationMismatch as exc:
        return fail(f"segment_duration mismatch: {exc}")
    except (FileNotFoundError, RuntimeError) as exc:
        if args.weights is None and not args.pool:
            raise
        return fail(f"{'--weights ' + args.weights if args.weights else '--pool'}: {exc}")
    if pool_record is not None and plan.train_metadata is not None:
        plan.train_metadata["pool"] = pool_record

    for warning in plan.segment_warnings:
        print(f"segment_duration_warning: {warning}")

    print(f"recipe: {args.profile}")
    if args.weights:
        print(f"weights: {args.weights}")
    print(f"audiocraft_dir: {plan.audiocraft_dir}")
    print(f"dataset_config: {plan.dataset_config_path}")
    if plan.xp_sig is not None:
        print(f"xp_sig: {plan.xp_sig}")
    if plan.xp_folder is not None:
        print(f"xp_folder: {plan.xp_folder}")
    if plan.effective_label is not None:
        print(f"effective_label: {plan.effective_label}")
    print(f"command: {format_command(plan.command)}")
    if plan.init_from_checkpoint is not None:
        print(f"init_from_checkpoint_origin: {plan.init_from_checkpoint['origin']}")
        print(f"init_from_checkpoint_path: {plan.init_from_checkpoint['resolved_local_path']}")
        fingerprint_match = plan.init_from_checkpoint.get("dataset_fingerprint_matches_current_workspace")
        if fingerprint_match is not None:
            print(f"init_from_checkpoint_dataset_match: {fingerprint_match}")
        for warning in plan.init_from_checkpoint.get("warnings", []):
            print(f"init_from_checkpoint_warning: {warning}")
    if args.dry_run:
        print("dry_run: command validated but not launched")
        return 0

    job = _start_job(job, job_request, job_argv)
    job = train_jobs.update_job(
        job,
        status="running",
        xp_sig=plan.xp_sig,
        effective_label=plan.effective_label,
        epochs=train_jobs.epochs_in(plan.command),
    )
    print(f"job: {job.id}")

    if args.gpu_watchdog:
        returncode = run_musicgen_training_with_watchdog(
            plan,
            record_dir=job.path,
            gpu_watchdog=MusicGenGpuWatchdogConfig(
                gpu_index=args.gpu_watchdog_gpu_index,
                poll_interval_seconds=args.gpu_watchdog_poll_seconds,
                temperature_trip_c=args.gpu_watchdog_temp_c,
                events_path=None,
            ),
        )
    else:
        returncode = run_musicgen_training(plan, record_dir=job.path)
    train_jobs.finish(job, returncode=returncode)
    return returncode


def _start_job(job: Any, job_request: dict[str, Any], job_argv: list[str], **fields: Any) -> Any:
    """The job this training records into: the queued one, or a new one."""
    import os

    from ..training import jobs as train_jobs

    if job is None:
        others = [other.id for other in train_jobs.active_jobs()]
        if others:
            print(f"warning: training beside {', '.join(others)}; `train queue add` waits its turn instead")
        job = train_jobs.create_job("train", request=job_request, argv=job_argv, started_by=os.getpid())
    return train_jobs.update_job(job, **fields) if fields else job


def _prepare_pool(args: argparse.Namespace, base_plan: Any) -> tuple[Any, Any, str | None]:
    """The pool `--pool` trains from: its spec, its folder, and why it cannot be used (``None`` when it can)."""
    from ..musicgen.config import get_repo_root
    from ..training import pool as token_pool

    pool = token_pool.plan_pool(
        args.profile, base_plan, seed=args.pool_seed, init_from_checkpoint=args.init_from_checkpoint,
        repo_root=get_repo_root(),
    )  # fmt: skip
    spec = pool.spec
    print(f"pool: {pool.path}")
    print(f"pool_crops: {spec.crops:,}, each used {spec.repeats:.1f} times over {spec.epochs} epochs")
    if pool.built:
        print("pool_state: built, reused")
    else:
        gigabytes = spec.pool_bytes() / 1e9
        print(f"pool_state: to build, about {pool.build_hours:.1f} h of encoding and {gigabytes:.1f} GB on disk")
    return spec, pool.path, pool.problem


def _build_pool(args: argparse.Namespace, base_plan: Any, spec: Any, pool_dir: Path) -> int:
    from ..musicgen.config import get_repo_root
    from ..musicgen.runtime import require_musicgen_python
    from ..training import pool as token_pool

    return token_pool.build_pool(
        spec,
        pool_dir,
        overrides=[str(item) for item in base_plan.command[2:]],
        musicgen_python=require_musicgen_python(args.musicgen_python),
        audiocraft_dir=base_plan.audiocraft_dir,
        repo_root=get_repo_root(),
    )


def cmd_train_queue(args: argparse.Namespace) -> int:
    import sys

    from ..training import jobs as train_jobs

    command = args.queue_command
    try:
        if command == "add":
            job = train_jobs.add_train_job(args)
        elif command == "resume":
            job = train_jobs.add_resume_job(args.model, epochs=args.epochs)
        elif command == "run":
            return train_jobs.run_queue(watch=args.watch, poll_seconds=args.poll_seconds)
        elif command == "list":
            return _print_queue(train_jobs.list_jobs(), every=args.all, as_json=args.json)
        else:
            job = train_jobs.load_job(args.job_id)
    except train_jobs.JobError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if command == "cancel":
        try:
            request = train_jobs.stop(job)
        except train_jobs.JobError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if request.get("cancelled"):
            print(f"cancelled {job.id}")
            return 0
        print(f"stopping {job.id}: SIGTERM sent to {request.get('target')}; waiting up to "
              f"{train_jobs.STOP_GRACE_SECONDS:.0f} s before SIGKILL")  # fmt: skip
        killed = train_jobs.escalate(job, request)
        print("had to SIGKILL" if killed else "stopped")
        return 0
    if command == "show":
        payload = _job_payload(job)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        for key in ("id", "state", "description", "origin", "created_at", "started_at", "finished_at",
                    "xp_sig", "effective_label", "epochs_done", "returncode", "error", "path"):  # fmt: skip
            if payload.get(key) is not None:
                print(f"{key}: {payload[key]}")
        print(f"command: {' '.join(job.command())}")
        if payload["log_tail"]:
            print("\n".join(["--- train.log (tail) ---", *payload["log_tail"]]))
        return 0

    queued = train_jobs.queued_jobs()
    if getattr(args, "json", False):
        print(json.dumps({**_job_payload(job), "position": len(queued)}, indent=2))
    else:
        print(f"queued {job.id}: {job.describe()} (position {len(queued)})")
    if getattr(args, "start", False):
        pid = train_jobs.ensure_worker()
        if not getattr(args, "json", False):
            print(f"worker: pid {pid}, log {train_jobs.jobs_dir() / train_jobs.WORKER_LOG_FILENAME}")
    elif train_jobs.worker_pid() is None and not getattr(args, "json", False):
        print("no worker is running: start one with `train queue run` (or add --start)")
    return 0


def _job_payload(job: Any) -> dict[str, Any]:
    from ..models.record import dora_xps_dir, load_history

    payload = job.to_dict()
    folder = dora_xps_dir() / job.xp_sig if job.xp_sig else None
    payload["epochs_done"] = len(load_history(folder)) if folder is not None and folder.is_dir() else None
    payload["log_tail"] = _tail_lines(job.log_path, 20)
    return payload


def _tail_lines(path: Any, count: int) -> list[str]:
    try:
        text = path.read_bytes()[-32_768:].decode("utf-8", errors="replace")
    except OSError:
        return []
    rows = [row.rsplit("\r", 1)[-1] for row in text.splitlines()]
    return [row for row in rows if row.strip()][-count:]


def _print_queue(jobs: list[Any], *, every: bool, as_json: bool) -> int:
    from ..training.jobs import worker_pid

    shown = jobs if every else [job for job in jobs if job.state not in ("cancelled",)][-20:]
    if as_json:
        print(json.dumps({"worker_pid": worker_pid(), "jobs": [_job_payload(job) for job in shown]}, indent=2))
        return 0
    if not shown:
        print("no training jobs yet: `train queue add --dataset <name> --recipe <recipe> ...`")
        return 0
    rows = []
    for job in shown:
        payload = _job_payload(job)
        done = payload["epochs_done"]
        progress = f"{done}/{payload['epochs'] or '?'}" if done is not None else ""
        rows.append((job.id, job.state, clip(job.describe(), 60), job.xp_sig or "", progress))
    header = ("job", "state", "what", "model", "epochs")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    pid = worker_pid()
    queued = sum(job.state == "queued" for job in jobs)
    print(f"\n{queued} queued · worker " + (f"running (pid {pid})" if pid else "not running"))
    return 0


def cmd_train_sidecar(args: argparse.Namespace) -> int:
    from ..training.sidecar import plan_musicgen_cpu_sidecar, run_musicgen_cpu_sidecar

    apply_model_reference(args, checkpoint_attr=None, checkpoint_path_attr=None)
    plan = plan_musicgen_cpu_sidecar(
        xp_path=args.xp_path,
        musicgen_python=args.musicgen_python,
        stage_mode=args.stage,
        poll_seconds=args.poll_seconds,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        once=args.once,
        score_samples=args.score_samples,
    )

    print(f"xp_path: {plan.xp_path}")
    print(f"xp_sig: {plan.xp_sig}")
    print(f"musicgen_python: {plan.musicgen_python}")
    print(f"stages: {','.join(plan.stages)}")
    print(f"poll_seconds: {plan.poll_seconds}")
    print(f"num_threads: {plan.num_threads}")
    if plan.batch_size is not None:
        print(f"batch_size: {plan.batch_size}")
    if plan.terminal_epoch is not None:
        print(f"terminal_epoch: {plan.terminal_epoch}")
    print(f"state_path: {plan.state_path}")
    print(f"events_path: {plan.events_path}")
    print(f"score_samples: {plan.score_samples}")
    if plan.score_samples:
        print(f"scores_path: {plan.scores_path}")

    return run_musicgen_cpu_sidecar(plan)


def _add_train_run_arguments(parser: argparse.ArgumentParser) -> None:
    """What a training takes: shared by `train run` and `train queue add`, so a queued job is the same command."""
    parser.add_argument(
        "--recipe",
        "--profile",
        dest="profile",
        choices=recipe_choices(),
        default=default_recipe_name(),
        help="Recipe: how to train (shape, codec, schedule), from musicgen/pyproject.toml",
    )
    parser.add_argument(
        "--audiocraft-dir",
        default=None,
        help="Path to an AudioCraft checkout; defaults to the repo-managed .audiocraft checkout",
    )
    parser.add_argument(
        "--dataset",
        "--workspace-dir",
        dest="workspace_dir",
        required=True,
        help="Dataset to train on, by name (`data list`) or folder: the recipe never names data",
    )
    parser.add_argument(
        "--musicgen-python",
        default=None,
        help=(
            "Python executable whose environment contains audiocraft and dora; "
            "defaults to the repo-managed .venv-musicgen runtime"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="How many epochs, instead of the recipe's (the same as --override optim.epochs=N)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional dora override, e.g. optim.lr=5e-5",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Friendly label for this Dora run; emits label=<sanitized-label>",
    )
    parser.add_argument(
        "--no-auto-dataset-label",
        action="store_true",
        help="Disable the default dataset-fingerprint label and use legacy Dora signature behavior",
    )
    parser.add_argument(
        "--allow-segment-padding",
        action="store_true",
        help=(
            "Train even when dataset.segment_duration exceeds the workspace's clip "
            "length (downgrades the hard error to a recorded warning). Most examples "
            "will be silence-padded with crop augmentation disabled."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Train on a weights overlay saved by `data rebalance` or `data mix` (see `data weights list`)",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help=(
            "Train from a token pool: crops encoded once, each used up to 8 times (nano and pico families; 4 above), "
            "instead of fresh crops every step. Builds it under artifacts/token_pools/ first, or reuses it"
        ),
    )
    parser.add_argument(
        "--pool-seed",
        type=int,
        default=0,
        help="With --pool: draw another set of crops (a new pool), e.g. to continue a model trained on one",
    )
    parser.add_argument(
        "--from",
        "--init-from-checkpoint",
        dest="init_from_checkpoint",
        default=None,
        help=(
            "Warm-start from another model: a model reference (`deep@best`, a sig, a label), a checkpoint "
            "file, or local portable bundle directory. Injects continue_from=<resolved-path>; "
            "cannot be combined with --override continue_from=..."
        ),
    )
    parser.add_argument(
        "--gpu-watchdog",
        action="store_true",
        help=(
            "Poll nvidia-smi during training and kill the training process group "
            "if the GPU is lost or a temp trip is hit"
        ),
    )
    parser.add_argument(
        "--gpu-watchdog-temp-c",
        type=float,
        default=None,
        help="Optional temperature trip point for --gpu-watchdog",
    )
    parser.add_argument(
        "--gpu-watchdog-gpu-index",
        type=int,
        default=0,
        help="GPU index to poll with --gpu-watchdog (default: 0)",
    )
    parser.add_argument(
        "--gpu-watchdog-poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval in seconds for --gpu-watchdog (default: 1.0)",
    )


def register_train_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train = subparsers.add_parser(
        "train",
        help="Train a recipe on a dataset: run, queue, resume, watch",
        description="Every training is a job in artifacts/train_jobs/<id>/. `run` trains now; `queue` lines up "
        "jobs for one worker. `resume` and `handoff` train a model further; `watch` and `sidecar` follow a run.",
        epilog="""examples:
  python -m macchina_ricordi.cli train run --dataset noisy_10s --recipe reshape-nano --dry-run
  python -m macchina_ricordi.cli train queue add --dataset noisy_10s --recipe reshape-deep --epochs 30 --start
  python -m macchina_ricordi.cli train resume midwife --epochs 80""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    train_subparsers = train.add_subparsers(dest="train_command", required=True)
    train_run = train_subparsers.add_parser(
        "run",
        help="Render dataset config and launch Dora training from this repo",
        description="Render the AudioCraft dataset config and launch a Dora MusicGen training run.",
        epilog="""examples:
  python -m macchina_ricordi.cli train run --dry-run
  python -m macchina_ricordi.cli train run --profile micro-unconditional --run-label sketch-a
  python -m macchina_ricordi.cli train run --dataset my_dataset --from models/my-model""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    _add_train_run_arguments(train_run)
    train_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the dora command without launching it",
    )
    train_run.add_argument("--job", default=None, help=argparse.SUPPRESS)  # set by the queue worker and the GUI
    train_run.set_defaults(func=cmd_train_run)

    queue = train_subparsers.add_parser(
        "queue",
        help="Line trainings up; one worker runs them in order, one GPU job at a time",
        description=(
            "Every training is a job in artifacts/train_jobs/<id>/. `add` queues one with exactly `train run`'s "
            "arguments; `run` is the worker that starts queued jobs in order while nothing else trains. A later "
            "job can start from an earlier one's model by its run label: --from <label>@best."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli train queue add --dataset bz --recipe reshape-deep --epochs 30 --run-label bz-deep
  python -m macchina_ricordi.cli train queue add --dataset bz --recipe reshape-deep --from bz-deep@best --run-label bz2
  python -m macchina_ricordi.cli train queue resume bz-deep --epochs 60
  python -m macchina_ricordi.cli train queue run
  python -m macchina_ricordi.cli train queue list --json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    queue_subparsers = queue.add_subparsers(dest="queue_command", required=True)
    queue_add = queue_subparsers.add_parser(
        "add",
        help="Queue a training: the same arguments as `train run`",
        description="Queue a training. The dataset, weights and --from model are checked now; the rest when it starts.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    _add_train_run_arguments(queue_add)
    queue_resume = queue_subparsers.add_parser(
        "resume",
        help="Queue a resume: train a model on, to --epochs in total",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    queue_resume.add_argument("model", help="The model: a name, sig, label, or the run label of a queued job")
    queue_resume.add_argument("--epochs", type=int, required=True, help="The new total epoch target")
    for parser in (queue_add, queue_resume):
        parser.add_argument("--start", action="store_true", help="Also start a background worker unless one runs")
        parser.add_argument("--json", action="store_true", help="Print the job as JSON")
    queue_list = queue_subparsers.add_parser("list", help="Jobs, oldest first: queued, running and done")
    queue_list.add_argument("--all", action="store_true", help="Every job, cancelled ones too (default: last 20)")
    queue_list.add_argument("--json", action="store_true", help="Machine-readable, with the worker's pid")
    queue_show = queue_subparsers.add_parser("show", help="One job: its request, state, model and log tail")
    queue_show.add_argument("job_id", help="The job's id (`train queue list`)")
    queue_show.add_argument("--json", action="store_true", help="Print as JSON")
    queue_cancel = queue_subparsers.add_parser(
        "cancel", help="Cancel a queued job, or stop a running one (its checkpoints so far are kept)"
    )
    queue_cancel.add_argument("job_id", help="The job's id (`train queue list`)")
    queue_run = queue_subparsers.add_parser(
        "run",
        help="The worker: run queued jobs in order until none is left",
        description="Start queued jobs in order, one at a time, waiting while any other job trains. "
        "Jobs run detached, so stopping the worker leaves the current one training.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    queue_run.add_argument("--watch", action="store_true", help="Keep waiting for new jobs instead of exiting")
    queue_run.add_argument("--poll-seconds", type=float, default=5.0, help="How often to look (default: 5)")
    for parser in (queue_add, queue_resume, queue_list, queue_show, queue_cancel, queue_run):
        parser.set_defaults(func=cmd_train_queue)

    resume = train_subparsers.add_parser(
        "resume",
        help="Resume an existing AudioCraft XP in place",
        description="Resume an existing Dora XP in place by increasing its total epoch target.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    resume.add_argument("sig", help="The model to resume: a sig, name or label")
    resume.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="New total epoch target for the existing XP",
    )
    resume.add_argument(
        "--resume-python",
        default=None,
        help="Python executable for AudioCraft resume work; defaults to the repo-managed .venv-musicgen runtime",
    )
    resume.add_argument(
        "--in-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    resume.set_defaults(func=cmd_train_resume)

    handoff = train_subparsers.add_parser(
        "handoff",
        help="Wait for a checkpointed epoch, then launch an in-place resume",
        description="Wait for an XP to reach a checkpointed epoch, then launch an in-place resume helper.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    handoff.add_argument("sig", help="The model to watch and resume: a sig, name or label")
    handoff.add_argument(
        "--wait-for-epoch",
        type=int,
        required=True,
        help="Epoch that must complete before the resume is launched",
    )
    handoff.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="New total epoch target for the resumed XP",
    )
    handoff.add_argument(
        "--active-pid",
        type=int,
        default=None,
        help="Optional currently running training PID to wait on before resuming",
    )
    handoff.add_argument(
        "--grace-seconds",
        type=int,
        default=600,
        help="Extra settle time after completion before launching resume",
    )
    handoff.add_argument(
        "--poll-seconds",
        type=int,
        default=120,
        help="Polling interval while waiting for the current run to finish",
    )
    handoff.add_argument(
        "--resume-python",
        default=None,
        help="Python executable to use for the actual resume command; defaults to the watcher runtime",
    )
    handoff.add_argument(
        "--resume-log-path",
        default=None,
        help="Optional file path to capture stdout/stderr from the resume helper",
    )
    handoff.add_argument(
        "--status-path",
        default=None,
        help="Optional JSON status path for this watcher",
    )
    handoff.add_argument(
        "--in-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    handoff.set_defaults(func=cmd_train_handoff)

    plateau = train_subparsers.add_parser(
        "watch",
        help="Watch an XP's history.json and emit events when training plateaus",
        description="Watch an XP history.json for plateau events, optionally stopping a training PID.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    plateau.add_argument("sig", help="The model to watch: a sig, name or label")
    plateau.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the AudioCraft dora directory (defaults to AUDIOCRAFT_DORA_DIR or the repo-managed .dora)",
    )
    plateau.add_argument(
        "--metric",
        default=DEFAULT_PLATEAU_METRIC,
        help=f"Dotted metric path inside each history entry (default: {DEFAULT_PLATEAU_METRIC})",
    )
    plateau.add_argument(
        "--patience-epochs",
        type=int,
        default=4,
        help="Number of history entries without improvement before reporting a plateau (default: 4)",
    )
    plateau.add_argument(
        "--min-improvement",
        type=float,
        default=1e-4,
        help="Minimum metric delta to count as an improvement (default: 1e-4)",
    )
    plateau.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="Polling interval while watching history.json (default: 60.0)",
    )
    plateau.add_argument(
        "--action",
        choices=("log", "stop"),
        default="log",
        help="Action to take on plateau detection (default: log)",
    )
    plateau.add_argument(
        "--stop-pid",
        type=int,
        default=None,
        help="Training PID to terminate when action=stop; required with --action=stop",
    )
    plateau.add_argument(
        "--grace-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait between SIGTERM and SIGKILL when action=stop (default: 10.0)",
    )
    plateau.add_argument(
        "--events-path",
        default=None,
        help="Override events JSONL path (default: <xp>/plateau_events.jsonl)",
    )
    plateau.add_argument(
        "--status-path",
        default=None,
        help="Override status JSON path (default: <xp>/plateau_watch_status.json)",
    )
    plateau.add_argument(
        "--once",
        action="store_true",
        help="Take a single sample, write status, and exit (no polling loop)",
    )
    plateau.set_defaults(func=cmd_train_watch)

    sidecar = train_subparsers.add_parser(
        "sidecar",
        help="Watch epoch checkpoints and run CPU-only generate/evaluate",
        description="Watch an XP for immutable checkpoints and run CPU-side generate/evaluate sidecars.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    sidecar.add_argument(
        "--xp-path",
        default=None,
        help="Explicit dora XP folder to watch; auto-discovered if omitted",
    )
    sidecar.add_argument(
        "--musicgen-python",
        default=None,
        help=(
            "Python executable whose environment contains audiocraft and dora; "
            "defaults to the repo-managed .venv-musicgen runtime"
        ),
    )
    sidecar.add_argument(
        "--stage",
        choices=("generate", "evaluate", "both"),
        default="both",
        help="CPU stage(s) to run for each newest unseen checkpoint (default: both)",
    )
    sidecar.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval in seconds while waiting for the next immutable epoch checkpoint",
    )
    sidecar.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Torch/OMP thread count for CPU stage execution (default: host CPU count)",
    )
    sidecar.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional CPU-side batch size override for generate/evaluate",
    )
    sidecar.add_argument(
        "--once",
        action="store_true",
        help="Process at most the current newest unseen checkpoint once, then exit",
    )
    sidecar.add_argument(
        "--score-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each generate stage, score the newly produced samples with the arrhythmic/percussive "
            "scorer and append per-sample rows to <xp>/sidecar_cpu/scores.jsonl (default: on)"
        ),
    )
    sidecar.set_defaults(func=cmd_train_sidecar)
