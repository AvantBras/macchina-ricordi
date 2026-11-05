"""Housekeeping: what ``.dora/xps`` holds, and the three ways to trim it.

Each action goes through its admin command's own planner and executor, so a GUI commit does
what the terminal command does and writes the same reports:

- epoch checkpoints: ``tidy clean`` (a ``checkpoint_cleanup_report.json``
  in each XP it touches). Checkpoints anything refers to are always kept here.
- whole XPs: ``tidy prune``, restricted to the XPs picked on the page
  (``<dora>/prune_report_<ts>.json``).
- archive: ``tidy archive`` (copies to another disk, never deletes;
  ``<dora>/checkpoint_archive_report_<ts>.json``). Runs in a background thread with progress.

A preview returns a key over exactly what would change; a commit re-plans and refuses unless the
key still matches, so nothing is deleted that the page did not show. XPs being trained (an active
run, or files written in the last 15 minutes) are left out of every action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from macchina_ricordi.tidy.archive import (
    ArchiveDecision,
    ArchivePlan,
    execute_archive_plan,
    plan_archive,
    validate_archive_destination,
)
from macchina_ricordi.tidy.clean import (
    CleanupPlan,
    best_epoch_checkpoint,
    cleanup_protections,
    execute_cleanup,
    plan_cleanup,
    scan_xp_for_checkpoint_candidates,
)
from macchina_ricordi.tidy.prune import PrunePlan, execute_prune_plan, plan_prune
from macchina_ricordi.tidy.refs import CheckpointReferences, scan_checkpoint_references

from .data import ACTIVE_WINDOW_SECONDS, HISTORY_FILENAME, TRAIN_METADATA_FILENAME, _last_activity
from .jobs import now_iso

ACTIONS = ("checkpoints", "xps", "archive")
ARCHIVE_MODES = ("latest", "epochs", "all")
ARCHIVE_ENV_FILE = Path("~/.config/macchina_ricordi/checkpoint-archive.env")
KEEP_MARKERS = ("logs/keep.json", ".keep", "keep.json")
XP_MARKERS = (TRAIN_METADATA_FILENAME, HISTORY_FILENAME, ".argv.json")
_EPOCH_CHECKPOINT = re.compile(r"^checkpoint_(\d+)\.th$")
_SIG = re.compile(r"^[0-9A-Za-z_.-]+$")
REFERENCE_EXAMPLES = 4


class HousekeepingError(ValueError):
    pass


@dataclass(frozen=True)
class HouseRequest:
    action: str
    sigs: tuple[str, ...] = ()
    keep_last: int = 1
    keep_best: bool = True
    include_referenced: bool = False
    destination: str = ""
    checkpoint_mode: str = "latest"
    min_age: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HouseRequest:
        action = str(payload.get("action") or "")
        if action not in ACTIONS:
            raise HousekeepingError(f"unknown action {action!r}")
        sigs = payload.get("sigs") or []
        if not isinstance(sigs, list) or not all(isinstance(sig, str) and _SIG.match(sig) for sig in sigs):
            raise HousekeepingError("sigs must be a list of XP folder names")
        try:
            keep_last = int(payload.get("keep_last", 1))
        except (TypeError, ValueError) as exc:
            raise HousekeepingError("keep_last must be a whole number") from exc
        if not 0 <= keep_last <= 100:
            raise HousekeepingError("keep_last must be between 0 and 100")
        mode = str(payload.get("checkpoint_mode") or "latest")
        if mode not in ARCHIVE_MODES:
            raise HousekeepingError(f"checkpoint_mode must be one of {', '.join(ARCHIVE_MODES)}")
        return cls(
            action=action,
            sigs=tuple(dict.fromkeys(sigs)),
            keep_last=keep_last,
            keep_best=bool(payload.get("keep_best", True)),
            include_referenced=bool(payload.get("include_referenced")),
            destination=str(payload.get("destination") or "").strip(),
            checkpoint_mode=mode,
            min_age=str(payload.get("min_age") or "").strip(),
        )


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _nearest_existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def read_archive_destination(env_file: Path) -> str | None:
    """The destination the systemd archive timer is configured with, if this machine has one."""
    try:
        text = env_file.expanduser().read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, _, value = line.strip().partition("=")
        if name == "MACCHINA_RICORDI_ARCHIVE" and value.strip():
            return value.strip().strip("'\"")
    return None


@dataclass
class HousekeepingManager:
    repo_root: Path
    dora_dir: Path | None = None
    active_sigs: Callable[[], Collection[str]] = lambda: ()
    # tests archive into a temporary folder; the page never lifts the other-disk guard
    allow_same_filesystem: bool = False
    archive_env_file: Path = ARCHIVE_ENV_FILE
    _job: dict[str, Any] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()
        self.dora_dir = (self.dora_dir or self.repo_root / ".dora").resolve()

    @property
    def xps_root(self) -> Path:
        assert self.dora_dir is not None
        return self.dora_dir / "xps"

    def _folders(self) -> list[Path]:
        if not self.xps_root.is_dir():
            return []
        return sorted(path for path in self.xps_root.iterdir() if path.is_dir())

    def _live(self, now: float | None = None) -> dict[str, str]:
        now = time.time() if now is None else now
        live = {sig: "an active training run" for sig in self.active_sigs() if sig}
        for folder in self._folders():
            if folder.name in live:
                continue
            latest = _latest_write(folder)
            if latest is not None and now - latest < ACTIVE_WINDOW_SECONDS:
                live[folder.name] = "written to in the last 15 minutes (training?)"
        return live

    # ── the inventory ──

    def inventory(self) -> dict[str, Any]:
        started = time.monotonic()
        references = scan_checkpoint_references(self.repo_root, self.dora_dir)
        live = self._live()
        folders = [self._folder(folder, references, live) for folder in self._folders()]
        kinds = ("latest", "epoch", "other", "rest")
        totals = {kind: sum(entry[f"{kind}_bytes"] for entry in folders) for kind in kinds}
        totals["all"] = sum(totals.values())
        assert self.dora_dir is not None
        usage = shutil.disk_usage(self.dora_dir)
        return {
            "dora_dir": _relative(self.dora_dir, self.repo_root),
            "filesystem": {"total": usage.total, "used": usage.used, "free": usage.free},
            "totals": totals,
            "folders": folders,
            "archive_default": read_archive_destination(self.archive_env_file),
            "job": self.job(),
            "scan_seconds": round(time.monotonic() - started, 2),
        }

    def _folder(self, folder: Path, references: CheckpointReferences, live: dict[str, str]) -> dict[str, Any]:
        sig = folder.name
        latest_bytes = epoch_bytes = other_bytes = rest_bytes = 0
        latest: dict[str, Any] | None = None
        epochs: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []
        for directory, _, files in os.walk(folder):
            top = Path(directory) == folder
            for name in files:
                path = Path(directory) / name
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                match = _EPOCH_CHECKPOINT.match(name) if top else None
                if top and name == "checkpoint.th":
                    latest_bytes += size
                    latest = {"bytes": size, "refs": len(references.for_checkpoint(sig, name))}
                elif match:
                    epoch_bytes += size
                    files_referring = references.for_checkpoint(sig, name)
                    epochs.append(
                        {
                            "epoch": int(match.group(1)),
                            "name": name,
                            "bytes": size,
                            "refs": len(files_referring),
                            "ref": files_referring[0] if files_referring else None,
                        }
                    )
                elif name.endswith(".th"):
                    other_bytes += size
                    others.append({"name": _relative(path, folder), "bytes": size})
                else:
                    rest_bytes += size
        best = best_epoch_checkpoint(folder, scan_xp_for_checkpoint_candidates(folder))
        for entry in epochs:
            entry["best"] = best is not None and best[0].epoch == entry["epoch"]
        epochs.sort(key=lambda entry: entry["epoch"])
        referring = references.for_xp(sig)
        latest_write = _latest_write(folder)
        return {
            "sig": sig,
            "is_xp": any((folder / name).exists() for name in XP_MARKERS),
            "bytes": latest_bytes + epoch_bytes + other_bytes + rest_bytes,
            "latest_bytes": latest_bytes,
            "epoch_bytes": epoch_bytes,
            "other_bytes": other_bytes,
            "rest_bytes": rest_bytes,
            "latest": latest,
            "epochs": epochs,
            "others": others,
            "refs": len(referring),
            "ref_examples": list(_examples(referring)),
            "children": sorted(_children(referring)),
            "keep_marker": _has_keep_marker(folder),
            "live": live.get(sig),
            "last_write": latest_write,
        }

    # ── plans ──

    def plan(self, request: HouseRequest) -> dict[str, Any]:
        return self._plan(request)[0]

    def _plan(self, request: HouseRequest) -> tuple[dict[str, Any], Any]:
        known = {folder.name for folder in self._folders()}
        unknown = [sig for sig in request.sigs if sig not in known]
        if unknown:
            raise HousekeepingError(f"no such XP folder: {', '.join(unknown)}")
        if request.action == "checkpoints":
            return self._plan_checkpoints(request, known)
        if request.action == "xps":
            return self._plan_xps(request, known)
        return self._plan_archive(request, known)

    def _scope(self, request: HouseRequest, known: set[str]) -> tuple[list[str], list[dict[str, str]]]:
        live = self._live()
        targets = list(request.sigs) or sorted(known)
        skipped = [{"sig": sig, "reason": live[sig]} for sig in targets if sig in live]
        return [sig for sig in targets if sig not in live], skipped

    def _plan_checkpoints(self, request: HouseRequest, known: set[str]) -> tuple[dict[str, Any], CleanupPlan | None]:
        sigs, skipped = self._scope(request, known)
        payload: dict[str, Any] = {"action": "checkpoints", "rows": [], "skipped": skipped, "key": None}
        if not sigs:
            payload.update(remove_count=0, remove_bytes=0)
            return payload, None
        assert self.dora_dir is not None
        protected = cleanup_protections(
            dora_dir=self.dora_dir, repo_root=self.repo_root, referenced=True, best=request.keep_best
        )
        plan = plan_cleanup(dora_dir=self.dora_dir, keep_last=request.keep_last, sigs=sigs, protected=protected)
        removes = []
        for decision in plan.decisions:
            if decision.skipped_reason:
                skipped.append({"sig": decision.sig, "reason": decision.skipped_reason})
                continue
            if not decision.keep and not decision.remove:
                continue
            rescued = {candidate.path for candidate, _ in decision.protected}
            payload["rows"].append(
                {
                    "sig": decision.sig,
                    "keep": [
                        {
                            "epoch": c.epoch,
                            "bytes": c.size_bytes,
                            "reason": "; ".join(
                                ([] if c.path in rescued else [f"among the last {request.keep_last}"])
                                + ([protected[c.path]] if c.path in protected else [])
                            ),
                        }
                        for c in sorted(decision.keep, key=lambda c: c.epoch)
                    ],
                    "remove": [{"epoch": c.epoch, "bytes": c.size_bytes} for c in decision.remove],
                }
            )
            removes.extend((str(c.path), c.size_bytes, c.mtime) for c in decision.remove)
        payload.update(
            remove_count=len(removes),
            remove_bytes=sum(size for _, size, _ in removes),
            key=_digest(["checkpoints", sorted(removes)]) if removes else None,
        )
        return payload, plan

    def _plan_xps(self, request: HouseRequest, known: set[str]) -> tuple[dict[str, Any], PrunePlan | None]:
        if not request.sigs:
            raise HousekeepingError("pick the XPs to delete first")
        live = self._live()
        references = scan_checkpoint_references(self.repo_root, self.dora_dir)
        held: dict[str, list[str]] = {}
        notes: dict[str, list[str]] = {}
        for sig in request.sigs:
            folder = self.xps_root / sig
            reasons, info = [], []
            if sig in live:
                reasons.append(live[sig])
            if _has_keep_marker(folder):
                reasons.append("has a keep marker (.keep or logs/keep.json)")
            referring = references.for_xp(sig)
            if referring:
                children = sorted(_children(referring))
                others = [path for path in referring if not path.startswith(".dora/xps/")]
                bits = []
                if children:
                    bits.append(f"parent of {', '.join(children[:5])}{'…' if len(children) > 5 else ''}")
                if len(others) == 1:
                    bits.append(f"mentioned in {others[0]}")
                elif others:
                    bits.append(f"mentioned in {len(others)} files, e.g. {others[0]}")
                text = "; ".join(bits)
                (info if request.include_referenced else reasons).append(text)
            held[sig] = reasons
            notes[sig] = info
        wanted = [sig for sig in request.sigs if not held[sig]]
        plan: PrunePlan | None = None
        decided: dict[str, tuple[str, list[str], int]] = {}
        if wanted:
            assert self.dora_dir is not None
            plan = plan_prune(
                dora_dir=self.dora_dir,
                keep_last=0,
                keep_labeled=True,
                keep_sigs=tuple(sorted(known - set(wanted))),
            )
            for decision in plan.decisions:
                if decision.entry.sig in wanted:
                    reasons = [] if decision.action == "remove" else list(decision.reasons)
                    decided[decision.entry.sig] = (decision.action, reasons, decision.entry.size_bytes)
        rows = []
        for sig in request.sigs:
            action, reasons, size = decided.get(sig, ("keep", held[sig], 0))
            if not size:
                size = _tree_size(self.xps_root / sig)
            rows.append(
                {
                    "sig": sig,
                    "action": "remove" if action == "remove" else "keep",
                    "bytes": size,
                    "reasons": reasons,
                    "notes": notes[sig],
                }
            )
        removes = sorted((row["sig"], row["bytes"]) for row in rows if row["action"] == "remove")
        payload = {
            "action": "xps",
            "rows": rows,
            "remove_count": len(removes),
            "remove_bytes": sum(size for _, size in removes),
            "key": _digest(["xps", removes]) if removes else None,
        }
        return payload, plan

    def _plan_archive(self, request: HouseRequest, known: set[str]) -> tuple[dict[str, Any], ArchivePlan | None]:
        payload: dict[str, Any] = {
            "action": "archive",
            "destination": request.destination,
            "destination_error": None,
            "rows": [],
            "skipped": [],
            "key": None,
            "copy_count": 0,
            "copy_bytes": 0,
            "already_count": 0,
            "free_bytes": None,
        }
        if not request.destination:
            raise HousekeepingError("choose where to archive to")
        destination = Path(request.destination).expanduser()
        if not destination.is_absolute():
            raise HousekeepingError("the destination must be an absolute path, on the other disk")
        assert self.dora_dir is not None
        try:
            resolved = validate_archive_destination(
                dora_dir=self.dora_dir, destination=destination, allow_same_filesystem=self.allow_same_filesystem
            )
        except ValueError as exc:
            payload["destination_error"] = str(exc)
            return payload, None
        payload["destination"] = str(resolved)
        payload["free_bytes"] = shutil.disk_usage(_nearest_existing(resolved)).free
        sigs, skipped = self._scope(request, known)
        payload["skipped"] = skipped
        if not sigs:
            return payload, None
        plan = plan_archive(
            dora_dir=self.dora_dir,
            destination=resolved,
            checkpoint_mode=request.checkpoint_mode,
            sigs=sigs,
            min_age=request.min_age or None,
        )
        copies = []
        for decision in plan.decisions:
            candidate = decision.candidate
            payload["rows"].append(
                {
                    "sig": candidate.sig,
                    "checkpoint": candidate.checkpoint_name,
                    "epoch": candidate.epoch,
                    "bytes": candidate.size_bytes,
                    "action": decision.action,
                    "reasons": list(decision.reasons),
                }
            )
            if decision.action == "archive":
                copies.append((str(candidate.checkpoint_path), candidate.size_bytes, candidate.mtime))
        payload.update(
            copy_count=len(copies),
            copy_bytes=sum(size for _, size, _ in copies),
            already_count=sum(1 for row in payload["rows"] if row["action"] == "already_archived"),
            key=_digest(["archive", str(resolved), sorted(copies)]) if copies else None,
        )
        return payload, plan

    # ── commits ──

    def commit(self, request: HouseRequest, key: str) -> dict[str, Any]:
        with self._lock:
            if self._job is not None and self._job["state"] == "running":
                raise HousekeepingError("an archive copy is running; wait for it to finish")
            payload, plan = self._plan(request)
            if not key or payload.get("key") != key:
                raise HousekeepingError("what is on disk changed since the preview; preview again")
            if request.action == "checkpoints":
                return self._commit_checkpoints(plan)
            if request.action == "xps":
                return self._commit_xps(plan)
            return self._start_archive(plan)

    def _commit_checkpoints(self, plan: CleanupPlan) -> dict[str, Any]:
        # only the XPs losing a file get a cleanup report
        touched = replace(plan, decisions=tuple(decision for decision in plan.decisions if decision.remove))
        report = execute_cleanup(touched, commit=True)
        return {
            "action": "checkpoints",
            "removed_count": report.removed_count,
            "freed_bytes": report.freed_bytes,
            "reports": [
                _relative(decision.xp_path / "checkpoint_cleanup_report.json", self.repo_root)
                for decision in touched.decisions
            ],
        }

    def _commit_xps(self, plan: PrunePlan) -> dict[str, Any]:
        report = execute_prune_plan(plan, commit=True)
        freed = sum(
            decision.entry.size_bytes for decision in plan.decisions if decision.entry.sig in report.removed_sigs
        )
        return {
            "action": "xps",
            "removed": list(report.removed_sigs),
            "failed": [sig for sig in report.skipped_sigs if "rmtree failed" in sig],
            "freed_bytes": freed,
            "report": _relative(report.report_path, self.repo_root),
        }

    def _start_archive(self, plan: ArchivePlan) -> dict[str, Any]:
        copies = [decision for decision in plan.decisions if decision.action == "archive"]
        job: dict[str, Any] = {
            "id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "state": "running",
            "destination": str(plan.destination),
            "started_at": now_iso(),
            "finished_at": None,
            "decisions": len(plan.decisions),
            "done": 0,
            "to_copy": len(copies),
            "copied": 0,
            "bytes_total": sum(decision.candidate.size_bytes for decision in copies),
            "bytes_copied": 0,
            "already": 0,
            "failed": [],
            "current": _describe(copies[0]) if copies else None,
            "report": None,
            "error": None,
        }
        self._job = job

        def progress(done: int, total: int, result: ArchiveDecision) -> None:
            job["done"] = done
            if result.action == "archived":
                job["copied"] += 1
                job["bytes_copied"] += result.candidate.size_bytes
            elif result.action == "already_archived":
                job["already"] += 1
            elif result.action == "failed":
                job["failed"].append({"checkpoint": _describe(result), "reason": "; ".join(result.reasons)})
            upcoming = next((d for d in plan.decisions[done:] if d.action == "archive"), None)
            job["current"] = _describe(upcoming) if upcoming else None

        def run() -> None:
            try:
                report = execute_archive_plan(
                    plan, commit=True, allow_same_filesystem=self.allow_same_filesystem, progress=progress
                )
                job["report"] = _relative(report.report_path, self.repo_root)
                job["state"] = "finished"
            except Exception as exc:  # the page shows it; the thread must not die silently
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["state"] = "failed"
            finally:
                job["finished_at"] = now_iso()
                job["current"] = None

        threading.Thread(target=run, name="archive-checkpoints", daemon=True).start()
        return {"action": "archive", "job": self.job()}

    def job(self) -> dict[str, Any] | None:
        return dict(self._job, failed=list(self._job["failed"])) if self._job is not None else None


def _describe(decision: ArchiveDecision) -> str:
    return f"{decision.candidate.sig}/{decision.candidate.checkpoint_name}"


def _latest_write(folder: Path) -> float | None:
    times = [t for t in (_last_activity(folder),) if t is not None]
    for path in folder.glob("checkpoint*.th"):
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def _has_keep_marker(folder: Path) -> bool:
    return any((folder / name).exists() for name in KEEP_MARKERS)


def _tree_size(folder: Path) -> int:
    total = 0
    for directory, _, files in os.walk(folder):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                continue
    return total


def _children(referring: Collection[str]) -> set[str]:
    """XPs whose own records mention this one: warm starts from it, merges or perturbations of it."""
    return {parts[2] for parts in (path.split("/") for path in referring) if parts[:2] == [".dora", "xps"]}


def _examples(referring: tuple[str, ...]) -> tuple[str, ...]:
    # a child XP first (the strongest tie), then whatever else, a few at most
    ordered = sorted(referring, key=lambda path: (not path.startswith(".dora/xps/"), path))
    return tuple(ordered[:REFERENCE_EXAMPLES])
