"""Queue, watch and stop MusicGen training from the GUI.

Training goes through the same jobs as the CLI (``macchina_ricordi.training.jobs``): Launch queues the previewed
training as a job in ``artifacts/train_jobs/<id>/`` and makes sure a queue worker runs, so it starts as soon as
nothing else trains ("start now" runs it beside whatever is running instead). Jobs run detached, so they outlive
the GUI, and a training started from a terminal or by an agent appears here exactly like one started from this page.

Previews call the CLI's ``--dry-run`` planner: they may write dataset configuration and a derived bundle
warm-start cache, but never launch training.

A warm start (``init_from_checkpoint``) is resolved the same way the CLI does: it may be a legacy ``.th``
checkpoint, a merged or perturbed output file, or the directory of a portable bundle under ``models/``.

Stopping signals the Dora process tree only, never the job's own CLI process: that one sees its child exit and
records the outcome, which a signal to it would prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from macchina_ricordi.musicgen.config import load_musicgen_project_config
from macchina_ricordi.musicgen.shared import read_json
from macchina_ricordi.training import jobs as train_jobs
from macchina_ricordi.training.watchdog import query_gpu_watchdog_sample

from .data import TRAIN_METADATA_FILENAME, history_series, load_history
from .jobs import relative

PREVIEW_TTL_SECONDS = 30 * 60
LOG_TAIL_LINES = 80
SAMPLING_MODES = ("file", "duration")
# Train split only: validation keeps sampling every file equally, so valid ce stays comparable across runs.
# `+` because the per-split section does not declare the key (same as `data rebalance`'s weight switch).
DURATION_SAMPLING_OVERRIDE = "+dataset.train.sample_on_duration=true"

# The keys worth knowing, with AudioCraft's defaults as this checkout's configs set them
# (solver/default.yaml < solver/musicgen/default.yaml < musicgen_base_32khz.yaml). Profiles
# override many of them; the page shows the selected profile's value next to each.
OVERRIDE_GUIDE: tuple[dict[str, str], ...] = (
    {
        "group": "Length and schedule",
        "key": "optim.epochs",
        "default": "500",
        "what": "How many epochs. An epoch is optim.updates_per_epoch updates, not one pass over the data.",
    },
    {
        "group": "Length and schedule",
        "key": "optim.updates_per_epoch",
        "default": "2000",
        "what": "Batches per epoch. Each batch is batch_size random crops, so an epoch sees the same amount of "
        "audio however large the dataset is.",
    },
    {
        "group": "Length and schedule",
        "key": "optim.lr",
        "default": "1 (with dadam)",
        "what": "Learning rate. With adamw (what the profiles use) 1e-4 is the usual; warm starts often go lower.",
    },
    {
        "group": "Length and schedule",
        "key": "optim.optimizer",
        "default": "dadam",
        "what": "adamw or dadam (D-Adaptation, which picks its own step size; lr is then a multiplier, 1).",
    },
    {"group": "Length and schedule", "key": "optim.adam.weight_decay", "default": "0", "what": "AdamW weight decay."},
    {
        "group": "Length and schedule",
        "key": "schedule.cosine.warmup",
        "default": "4000",
        "what": "Linear warm-up, in updates (4000 = two epochs at 2000 updates each).",
    },
    {
        "group": "Length and schedule",
        "key": "schedule.cosine.lr_min_ratio",
        "default": "0.0",
        "what": "Where the cosine decay ends, as a fraction of lr. The decay spans the whole run, so changing "
        "optim.epochs also stretches it.",
    },
    {
        "group": "Length and schedule",
        "key": "optim.ema.use",
        "default": "true",
        "what": "Keep an exponential moving average of the weights. Validation runs on it, so valid ce and the "
        "saved best state are the averaged model.",
    },
    {
        "group": "Data",
        "key": "dataset.segment_duration",
        "default": "30",
        "what": "Seconds of audio per training example (a random crop). Longer than the clips means silence "
        "padding, which train run refuses unless padding is allowed.",
    },
    {
        "group": "Data",
        "key": "dataset.batch_size",
        "default": "192",
        "what": "Crops per update. The main VRAM knob with segment_duration; the profiles use 1 to 4 on the 3060.",
    },
    {
        "group": "Data",
        "key": "dataset.min_segment_ratio",
        "default": "0.8",
        "what": "How much of each crop must be real audio: a crop may start as late as the file's duration minus "
        "segment_duration × this, and the rest is padded with silence. Lower means more silent crop endings.",
    },
    {
        "group": "Data",
        "key": "dataset.valid.num_samples",
        "default": "10000",
        "what": "Validation crops per epoch; the valid ce you compare runs on. Profiles use 50.",
    },
    {
        "group": "Data",
        "key": "dataset.train.shuffle_seed",
        "default": "0",
        "what": "Draw a different random stream of crops with otherwise identical settings (and a new signature).",
    },
    {
        "group": "Data",
        "key": "dataset.num_workers",
        "default": "10",
        "what": "Data-loading processes. Profiles use 2; more helps only if the GPU waits on data.",
    },
    {
        "group": "Model shape (from scratch only)",
        "key": "transformer_lm.dim",
        "default": "512",
        "what": "Width. A warm start must keep the checkpoint's shape.",
    },
    {"group": "Model shape (from scratch only)", "key": "transformer_lm.num_layers", "default": "8", "what": "Depth."},
    {
        "group": "Model shape (from scratch only)",
        "key": "transformer_lm.num_heads",
        "default": "8",
        "what": "Attention heads; dim must divide by it.",
    },
    {
        "group": "Model shape (from scratch only)",
        "key": "transformer_lm.dropout",
        "default": "0.",
        "what": "Dropout inside the transformer; a regulariser for small datasets.",
    },
    {
        "group": "Model shape (from scratch only)",
        "key": "transformer_lm.checkpointing",
        "default": "none",
        "what": "torch trades speed for memory (recomputes activations); what the larger profiles use.",
    },
    {
        "group": "Checkpoints and side jobs",
        "key": "checkpoint.save_every",
        "default": "50",
        "what": "Write checkpoint_<epoch>.th every N epochs (profiles: 5). Stopping loses what came after the last.",
    },
    {
        "group": "Checkpoints and side jobs",
        "key": "checkpoint.keep_last",
        "default": "10",
        "what": "How many of those epoch checkpoints to keep.",
    },
    {
        "group": "Checkpoints and side jobs",
        "key": "generate.every",
        "default": "25",
        "what": "Generate samples every N epochs on the GPU. Profiles set 10000 (off) and leave it to the CPU sidecar.",
    },
    {
        "group": "Checkpoints and side jobs",
        "key": "evaluate.every",
        "default": "25",
        "what": "Evaluation stage every N epochs; off (10000) in the profiles.",
    },
    {
        "group": "Checkpoints and side jobs",
        "key": "autocast",
        "default": "true",
        "what": "Mixed precision (float16). Off is slower and uses more memory.",
    },
)


class TrainingRequestError(ValueError):
    """A launch or preview that cannot go ahead; the message says why."""


@dataclass(frozen=True)
class LaunchRequest:
    profile: str
    workspace: str
    overrides: tuple[str, ...] = ()
    run_label: str | None = None
    init_from_checkpoint: str | None = None
    gpu_watchdog: bool = True
    gpu_watchdog_temp_c: float | None = None
    allow_segment_padding: bool = False
    sampling: str = "file"
    weights: str | None = None
    pool: bool = False
    pool_seed: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LaunchRequest:
        overrides_raw = payload.get("overrides") or ()
        if isinstance(overrides_raw, str):
            overrides_raw = overrides_raw.splitlines()
        overrides = tuple(line.strip() for line in overrides_raw if str(line).strip())
        for override in overrides:
            if "=" not in override or override.startswith("-") or any(ch.isspace() for ch in override):
                raise TrainingRequestError(f"override {override!r} must look like key=value, with no spaces")
            if payload.get("pool") and override.startswith("cache."):
                raise TrainingRequestError("a token pool sets cache.path itself; drop the cache.* override")
            if "sample_on_duration" in override.split("=", 1)[0]:
                raise TrainingRequestError("set duration sampling with the sampling control, not an override")
        sampling = str(payload.get("sampling") or "file")
        if sampling not in SAMPLING_MODES:
            raise TrainingRequestError(f"sampling must be one of {', '.join(SAMPLING_MODES)}")
        temp = payload.get("gpu_watchdog_temp_c")
        try:
            temp_c = float(temp) if temp not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise TrainingRequestError("watchdog temperature must be a number") from exc
        try:
            pool_seed = int(payload.get("pool_seed") or 0)
        except (TypeError, ValueError) as exc:
            raise TrainingRequestError("pool seed must be a whole number") from exc
        return cls(
            profile=str(payload.get("profile") or "").strip(),
            workspace=str(payload.get("workspace") or "").strip(),
            overrides=overrides,
            run_label=str(payload.get("run_label") or "").strip() or None,
            init_from_checkpoint=str(payload.get("init_from_checkpoint") or "").strip() or None,
            gpu_watchdog=bool(payload.get("gpu_watchdog", True)),
            gpu_watchdog_temp_c=temp_c,
            allow_segment_padding=bool(payload.get("allow_segment_padding", False)),
            sampling=sampling,
            weights=str(payload.get("weights") or "").strip() or None,
            pool=bool(payload.get("pool", False)),
            pool_seed=pool_seed,
        )

    def cli_args(self) -> list[str]:
        """Arguments after ``train run``, as a terminal user would type them."""
        args = ["--recipe", self.profile, "--dataset", self.workspace]
        if self.weights:
            args += ["--weights", self.weights]
        for override in self.overrides:
            args += ["--override", override]
        if self.sampling == "duration":
            args += ["--override", DURATION_SAMPLING_OVERRIDE]
        if self.run_label:
            args += ["--run-label", self.run_label]
        if self.init_from_checkpoint:
            args += ["--init-from-checkpoint", self.init_from_checkpoint]
        if self.allow_segment_padding:
            args.append("--allow-segment-padding")
        if self.pool:
            args.append("--pool")
            if self.pool_seed:
                args += ["--pool-seed", str(self.pool_seed)]
        if self.gpu_watchdog:
            args.append("--gpu-watchdog")
            if self.gpu_watchdog_temp_c is not None:
                args += ["--gpu-watchdog-temp-c", f"{self.gpu_watchdog_temp_c:g}"]
        return args

    def effective_overrides(self) -> list[str]:
        return [*self.overrides, *([DURATION_SAMPLING_OVERRIDE] if self.sampling == "duration" else [])]

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class _Preview:
    key: str
    created: float
    payload: dict[str, Any]


def _tail(path: Path, lines: int = LOG_TAIL_LINES, max_bytes: int = 64 * 1024) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    # tqdm redraws a line with \r; keep only the last state of each line
    rows = [row.rsplit("\r", 1)[-1] for row in text.split("\n")]
    return [row for row in rows if row.strip()][-lines:]


def gpu_sample() -> dict[str, Any]:
    sample = query_gpu_watchdog_sample(gpu_index=0)
    parts = [part.strip() for part in sample.raw_output.split(",")] if sample.raw_output else []

    def number(index: int) -> float | None:
        try:
            return float(parts[index].split()[0])
        except (IndexError, ValueError):
            return None

    return {
        "available": not sample.query_failed,
        "error": sample.query_error,
        "gpu_lost": sample.gpu_lost,
        "temperature_c": sample.temperature_c,
        "power_w": number(1),
        "utilization_pct": number(2),
        "memory_used_mib": number(3),
        "pstate": parts[4] if len(parts) > 4 else None,
    }


# ── the manager ───────────────────────────────────────────────────────


@dataclass
class TrainingManager:
    repo_root: Path
    python: str = field(default_factory=lambda: sys.executable)
    planner: Callable[..., Any] | None = None
    cli_prefix: tuple[str, ...] | None = None
    stop_grace_seconds: float = train_jobs.STOP_GRACE_SECONDS
    _previews: dict[str, _Preview] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()

    @property
    def _prefix(self) -> tuple[str, ...]:
        return self.cli_prefix or (self.python, *train_jobs.CLI_PREFIX)

    # options

    def options(self) -> dict[str, Any]:
        from macchina_ricordi.data.clip_stats import clip_stats_for_workspace
        from macchina_ricordi.data.prepare import find_workspace_manifest
        from macchina_ricordi.data.weights import list_weights_overlays, read_rebalance_sidecar

        config = load_musicgen_project_config(self.repo_root)
        profiles = [
            {
                "name": recipe.name,
                "extends": recipe.extends,
                "overrides": list(recipe.overrides),
                "epochs": _override_value(recipe.overrides, "optim.epochs"),
                "segment_duration": _override_value(recipe.overrides, "dataset.segment_duration"),
            }
            for recipe in config.recipes.values()
        ]
        workspaces = []
        for workspace in self._workspace_dirs():
            datasets = sorted(
                path.name
                for path in (workspace / "egs").glob("*")
                if (path / "train" / "data.jsonl.gz").exists() and (path / "valid" / "data.jsonl.gz").exists()
            )
            if not datasets:
                continue
            manifest = find_workspace_manifest(workspace)
            stats = clip_stats_for_workspace(manifest) if manifest is not None else None
            last = train_jobs.last_job_on(workspace.name, self.repo_root)
            rebalance = {
                dataset: {k: sidecar.get(k) for k in ("mode", "k", "alpha", "cap", "written_at")}
                for dataset in datasets
                if (sidecar := read_rebalance_sidecar(workspace / "egs" / dataset / "train"))
            }
            workspaces.append(
                {
                    "name": workspace.name,
                    "datasets": datasets,
                    "clips": (stats or {}).get("count"),
                    "slice_length_s": (stats or {}).get("slice_length_s"),
                    "median_s": (stats or {}).get("median"),
                    "min_s": (stats or {}).get("min"),
                    "p90_s": (stats or {}).get("p90"),
                    "max_s": (stats or {}).get("max"),
                    "rebalance": rebalance,
                    "weights": [
                        {**overlay.summary(), "stale": overlay.is_stale()}
                        for overlay in list_weights_overlays(self.repo_root, workspace.name)
                    ],
                    "last_run": {
                        "status": last.state,
                        "xp_sig": last.xp_sig,
                        "effective_label": last.to_dict()["effective_label"],
                        "finished_at": last.status.get("finished_at"),
                    }
                    if last
                    else None,
                }
            )
        checkpoints_dir = self.repo_root / "artifacts" / "checkpoints"
        # loose files, and merged or perturbed ones in a folder of their own
        loose = sorted(relative(path, self.repo_root) for path in checkpoints_dir.rglob("*.th"))
        return {
            "profiles": profiles,
            "workspaces": workspaces,
            "loose_checkpoints": loose,
            "override_guide": list(OVERRIDE_GUIDE),
        }

    def _workspace_dirs(self) -> list[Path]:
        from macchina_ricordi.data.datasets import dataset_dirs

        return [path.resolve() for path in dataset_dirs(self.repo_root)]

    def _workspace(self, name: str) -> Path:
        from macchina_ricordi.data.datasets import find_dataset_dir

        path = find_dataset_dir(name, repo_root=self.repo_root) if name and "/" not in name.strip("/") else None
        if path is None or path not in self._workspace_dirs():
            raise TrainingRequestError(f"no dataset named {name!r} in this repo")
        return path

    # preview / launch

    def preview(self, request: LaunchRequest) -> dict[str, Any]:
        if not request.profile:
            raise TrainingRequestError("pick a recipe")
        workspace = self._workspace(request.workspace)
        planner = self.planner
        if planner is None:
            from macchina_ricordi.training.plan import plan_musicgen_training as planner
        from macchina_ricordi.data.clip_stats import SegmentDurationMismatch

        def make_plan(overrides: list[str]) -> Any:
            return planner(
                audiocraft_dir=None,
                profile_name=request.profile,
                workspace_dir=str(workspace),
                overrides=overrides,
                run_label=request.run_label,
                dry_run=True,
                init_from_checkpoint=request.init_from_checkpoint,
                allow_segment_padding=request.allow_segment_padding,
                **({"weights": request.weights, "repo_root": self.repo_root} if request.weights else {}),
            )

        pool = None
        try:
            overrides = request.effective_overrides()
            if request.pool:
                from macchina_ricordi.training import pool as token_pool

                base_plan = make_plan(overrides)
                pool = token_pool.plan_pool(
                    request.profile, base_plan, seed=request.pool_seed,
                    init_from_checkpoint=(base_plan.init_from_checkpoint or {}).get("resolved_local_path"),
                    repo_root=self.repo_root,
                )  # fmt: skip
                if pool.problem:
                    raise TrainingRequestError(pool.problem)
                overrides = [*overrides, token_pool.cache_override(pool.path)]
            plan = make_plan(overrides)
        except SegmentDurationMismatch as exc:
            raise TrainingRequestError(
                f"segment_duration mismatch: {exc} — shorten dataset.segment_duration or allow padding"
            ) from exc
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            raise TrainingRequestError(str(exc)) from exc

        xp_folder = Path(plan.xp_folder) if plan.xp_folder else None
        existing = None
        if xp_folder is not None and xp_folder.is_dir():
            metadata = read_json(xp_folder / TRAIN_METADATA_FILENAME) or {}
            existing = {
                "label": metadata.get("effective_label"),
                "epochs_done": len(load_history(xp_folder)),
                "checkpoints": sorted(path.name for path in xp_folder.glob("checkpoint*.th")),
            }
        runs = self.runs()
        payload = {
            "key": request.key(),
            "request": asdict(request),
            "xp_sig": plan.xp_sig,
            "xp_folder": relative(xp_folder, self.repo_root) if xp_folder else None,
            "existing_xp": existing,
            "effective_label": plan.effective_label,
            "command": list(plan.command),
            "cli": ["python", "-m", "macchina_ricordi.cli", "train", "queue", "add", *request.cli_args()],
            "epochs": _override_value(plan.command, "optim.epochs"),
            "segment_duration": (plan.train_metadata or {}).get("segment_duration"),
            "clip_slice_length_s": (plan.train_metadata or {}).get("clip_slice_length_s"),
            "segment_warnings": list(plan.segment_warnings),
            "init_from_checkpoint": plan.init_from_checkpoint,
            "sampling": request.sampling,
            "rebalance": (plan.train_metadata or {}).get("rebalance"),
            "pool": pool.to_dict() if pool else None,
            "dataset_fingerprint_digest": plan.dataset_fingerprint_digest,
            "active_runs": [run["id"] for run in runs if run["active"]],
            "queued_runs": [run["id"] for run in runs if run["state"] == "queued"],
        }
        with self._lock:
            now = time.monotonic()
            self._previews = {k: v for k, v in self._previews.items() if now - v.created < PREVIEW_TTL_SECONDS}
            self._previews[payload["key"]] = _Preview(payload["key"], now, payload)
        return payload

    def launch(self, request: LaunchRequest, *, preview_key: str, start_now: bool = False) -> dict[str, Any]:
        """Queue the previewed training, then start a worker (it waits its turn), or start it now beside the rest."""
        key = request.key()
        with self._lock:
            preview = self._previews.get(key)
        if preview is None or preview_key != key or time.monotonic() - preview.created > PREVIEW_TTL_SECONDS:
            raise TrainingRequestError("the form changed since the preview (or it is older than 30 min); preview again")
        self._workspace(request.workspace)
        try:
            job = train_jobs.add_train_job(request.cli_args(), origin="gui", repo_root=self.repo_root)
            if start_now:
                job = train_jobs.start_now(job, prefix=self._prefix, repo_root=self.repo_root)
            else:
                train_jobs.ensure_worker(prefix=self._prefix, repo_root=self.repo_root)
        except train_jobs.JobError as exc:
            raise TrainingRequestError(str(exc)) from exc
        with self._lock:
            self._previews.pop(key, None)
        return self.run_detail(job.id)

    # runs

    def runs(self) -> list[dict[str, Any]]:
        """Every training job, newest first; queued ones carry their place in the queue."""
        jobs = train_jobs.list_jobs(self.repo_root)
        queued = [job.id for job in jobs if job.state == "queued"]
        runs = [self._summarize(job, queued) for job in jobs]
        return list(reversed(runs))

    def _summarize(self, job: train_jobs.Job, queued: list[str]) -> dict[str, Any]:
        record = job.to_dict()
        return {
            **record,
            "source": job.origin,
            "workspace": job.request.get("dataset") or job.request.get("model"),
            "profile": job.request.get("recipe"),
            "stoppable": job.state in ("queued", *train_jobs.RUNNING_STATES),
            "queue_position": queued.index(job.id) + 1 if job.id in queued else None,
            "started_at": record["started_at"] or record["created_at"],
            "epochs_done": self._epochs_done(job.xp_sig),
        }

    def _epochs_done(self, xp_sig: Any) -> int:
        if not xp_sig:
            return 0
        return len(load_history(self.repo_root / ".dora" / "xps" / str(xp_sig)))

    def _job(self, job_id: str) -> train_jobs.Job:
        try:
            return train_jobs.load_job(job_id, repo_root=self.repo_root)
        except train_jobs.JobError as exc:
            raise TrainingRequestError(str(exc)) from exc

    def run_detail(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        queued = [other.id for other in train_jobs.queued_jobs(self.repo_root)]
        summary = self._summarize(job, queued)
        xp_folder = self.repo_root / ".dora" / "xps" / str(job.xp_sig) if job.xp_sig else None
        history = load_history(xp_folder) if xp_folder and xp_folder.is_dir() else []
        xp_log = []
        if xp_folder is not None:
            for name in ("solver.log.0", "train.log"):
                if (xp_folder / name).exists():
                    xp_log = _tail(xp_folder / name, lines=40)
                    break
        return {
            **summary,
            "record": {"argv": job.command(self._prefix), "path": relative(job.path, self.repo_root)},
            "run_status": job.run_status() or None,
            "history": history_series(history),
            "eta_seconds": _eta_seconds(history, summary.get("epochs")),
            "seconds_per_epoch": _seconds_per_epoch(history),
            "log": _tail(job.log_path),
            "log_path": relative(job.log_path, self.repo_root),
            "xp_log": xp_log,
            "xp_exists": bool(xp_folder and xp_folder.is_dir()),
        }

    def stop(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job, or stop a running one (SIGKILL after the grace period if need be)."""
        job = self._job(job_id)
        try:
            request = train_jobs.stop(job)
        except train_jobs.JobError as exc:
            raise TrainingRequestError(str(exc)) from exc
        if not request.get("cancelled"):
            threading.Thread(
                target=train_jobs.escalate, args=(job, request), kwargs={"grace_seconds": self.stop_grace_seconds},
                daemon=True,
            ).start()  # fmt: skip
        return self.run_detail(job_id)

    def status(self) -> dict[str, Any]:
        return {"gpu": gpu_sample(), "runs": self.runs(), "worker_pid": train_jobs.worker_pid(self.repo_root)}


def _override_value(command: Sequence[Any], key: str) -> Any:
    value = None
    for part in command:
        if isinstance(part, str) and part.startswith(f"{key}="):
            value = part.split("=", 1)[1]
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _seconds_per_epoch(history: list[dict[str, Any]]) -> float | None:
    """Wall time of the last few epochs, summed over every stage Dora timed."""
    totals = []
    for entry in history[-3:]:
        total = sum(
            float(stage.get("duration", 0.0))
            for stage in entry.values()
            if isinstance(stage, dict) and isinstance(stage.get("duration"), int | float)
        )
        if total > 0:
            totals.append(total)
    return sum(totals) / len(totals) if totals else None


def _eta_seconds(history: list[dict[str, Any]], planned: Any) -> float | None:
    per_epoch = _seconds_per_epoch(history)
    if per_epoch is None or not isinstance(planned, int) or planned <= len(history):
        return None
    return per_epoch * (planned - len(history))
