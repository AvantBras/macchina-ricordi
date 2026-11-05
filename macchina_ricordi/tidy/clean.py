from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..musicgen.shared import utc_now_iso, write_json
from .prune import format_bytes

CLEANUP_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointCandidate:
    """A single epoch checkpoint file that may be deleted."""

    epoch: int
    path: Path
    size_bytes: int
    mtime: float


@dataclass(frozen=True)
class XpCleanupDecision:
    """Per-XP result: which checkpoints to keep / remove, and why."""

    sig: str
    xp_path: Path
    running: bool
    keep: tuple[CheckpointCandidate, ...]
    remove: tuple[CheckpointCandidate, ...]
    skipped_reason: str | None  # None if actionable, non-None if skipped
    protected: tuple[tuple[CheckpointCandidate, str], ...] = ()  # kept although keep_last would drop them


@dataclass(frozen=True)
class CleanupPlan:
    dora_dir: Path
    xps_root: Path
    decisions: tuple[XpCleanupDecision, ...]
    keep_last: int
    force: bool


@dataclass(frozen=True)
class CleanupExecutionReport:
    plan: CleanupPlan
    committed: bool
    removed_count: int
    skipped_count: int
    freed_bytes: int


def _is_xp_running(xp_path: Path) -> bool:
    from ..training.jobs import training_sigs

    if xp_path.name in training_sigs():
        return True
    status_path = xp_path / "train_run_status.json"
    if not status_path.exists():
        return False
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("status") == "running"


def scan_xp_for_checkpoint_candidates(xp_path: Path) -> list[CheckpointCandidate]:
    """Return deletable epoch checkpoint candidates, sorted by epoch ascending.

    checkpoint.th and checkpoint_merged.th are never included.
    """
    from ..training.sidecar import list_epoch_checkpoints

    epoch_checkpoints = list_epoch_checkpoints(xp_path)
    candidates: list[CheckpointCandidate] = []
    for ec in epoch_checkpoints:
        try:
            st = ec.path.stat()
        except OSError:
            continue
        candidates.append(
            CheckpointCandidate(
                epoch=ec.epoch,
                path=ec.path,
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            )
        )
    return candidates


def classify_checkpoints(
    candidates: Sequence[CheckpointCandidate],
    *,
    keep_last: int,
) -> tuple[tuple[CheckpointCandidate, ...], tuple[CheckpointCandidate, ...]]:
    """Split candidates into (keep, remove) based on --keep-last.

    The keep set is the N candidates with the highest epoch numbers.
    If there are <= keep_last candidates, all are kept and remove is empty.
    """
    if keep_last < 0:
        raise ValueError("keep_last must be >= 0")
    if keep_last == 0:
        return (), tuple(candidates)

    sorted_desc = sorted(candidates, key=lambda c: c.epoch, reverse=True)
    keep = tuple(sorted_desc[:keep_last])
    remove = tuple(sorted(candidates, key=lambda c: c.epoch)[: len(candidates) - keep_last])
    return keep, remove


def plan_cleanup(
    *,
    dora_dir: Path,
    keep_last: int = 2,
    sigs: Sequence[str] = (),
    force: bool = False,
    protected: Mapping[Path, str] | None = None,
) -> CleanupPlan:
    """``protected`` maps checkpoint paths to the reason they must stay, whatever ``keep_last`` says."""
    xps_root = dora_dir / "xps"
    sig_set = {s.strip() for s in sigs if s.strip()}
    protected_reasons = {Path(path).resolve(): reason for path, reason in (protected or {}).items()}

    if not xps_root.exists() or not xps_root.is_dir():
        return CleanupPlan(
            dora_dir=dora_dir,
            xps_root=xps_root,
            decisions=(),
            keep_last=keep_last,
            force=force,
        )

    decisions: list[XpCleanupDecision] = []
    for xp_folder in sorted(xps_root.iterdir(), key=lambda p: p.name):
        if not xp_folder.is_dir():
            continue
        sig = xp_folder.name
        if sig_set and sig not in sig_set:
            continue

        running = _is_xp_running(xp_folder)
        if running and not force:
            decisions.append(
                XpCleanupDecision(
                    sig=sig,
                    xp_path=xp_folder,
                    running=True,
                    keep=(),
                    remove=(),
                    skipped_reason="status=running; use --force to override",
                )
            )
            continue

        candidates = scan_xp_for_checkpoint_candidates(xp_folder)
        keep, remove = classify_checkpoints(candidates, keep_last=keep_last)
        rescued = tuple((c, protected_reasons[c.path]) for c in remove if c.path in protected_reasons)
        if rescued:
            rescued_paths = {c.path for c, _ in rescued}
            keep = tuple(sorted((*keep, *(c for c, _ in rescued)), key=lambda c: c.epoch, reverse=True))
            remove = tuple(c for c in remove if c.path not in rescued_paths)
        decisions.append(
            XpCleanupDecision(
                sig=sig,
                xp_path=xp_folder,
                running=running,
                keep=keep,
                remove=remove,
                skipped_reason=None,
                protected=rescued,
            )
        )

    return CleanupPlan(
        dora_dir=dora_dir,
        xps_root=xps_root,
        decisions=tuple(decisions),
        keep_last=keep_last,
        force=force,
    )


def cleanup_protections(
    *,
    dora_dir: Path,
    repo_root: Path,
    referenced: bool = True,
    best: bool = False,
) -> dict[Path, str]:
    """Epoch checkpoints cleanup should keep: those something refers to, and each XP's best one.

    The best is the epoch checkpoint on disk whose epoch has the lowest valid ce in ``history.json``.
    """
    xps_root = dora_dir / "xps"
    if not xps_root.is_dir():
        return {}
    references = None
    if referenced:
        from .refs import scan_checkpoint_references

        references = scan_checkpoint_references(repo_root, dora_dir)
    protected: dict[Path, str] = {}
    for xp_folder in sorted(xps_root.iterdir()):
        if not xp_folder.is_dir():
            continue
        candidates = scan_xp_for_checkpoint_candidates(xp_folder)
        if best:
            winner = best_epoch_checkpoint(xp_folder, candidates)
            if winner is not None:
                candidate, ce = winner
                protected[candidate.path] = f"best valid ce on disk ({ce:.3f} at epoch {candidate.epoch})"
        if references is not None:
            for candidate in candidates:
                files = references.for_checkpoint(xp_folder.name, candidate.path.name)
                if files:
                    more = f" and {len(files) - 1} more" if len(files) > 1 else ""
                    protected[candidate.path] = f"referenced by {files[0]}{more}"
    return protected


def best_epoch_checkpoint(
    xp_folder: Path, candidates: Sequence[CheckpointCandidate]
) -> tuple[CheckpointCandidate, float] | None:
    try:
        history = json.loads((xp_folder / "history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list):
        return None
    best: tuple[CheckpointCandidate, float] | None = None
    for candidate in candidates:
        if not 1 <= candidate.epoch <= len(history):
            continue
        entry = history[candidate.epoch - 1]
        valid = entry.get("valid") if isinstance(entry, dict) else None
        ce = valid.get("ce") if isinstance(valid, dict) else None
        if not isinstance(ce, int | float) or not math.isfinite(ce):
            continue
        if best is None or ce <= best[1]:  # a tie goes to the later, longer-trained epoch
            best = (candidate, float(ce))
    return best


def _write_cleanup_report(
    plan: CleanupPlan,
    decision: XpCleanupDecision,
    *,
    commit: bool,
    removed: tuple[CheckpointCandidate, ...],
    error: str | None,
) -> Path:
    report_path = decision.xp_path / "checkpoint_cleanup_report.json"
    payload = {
        "schema_version": CLEANUP_REPORT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "sig": decision.sig,
        "xp_path": str(decision.xp_path),
        "committed": commit,
        "keep_last": plan.keep_last,
        "force": plan.force,
        "skipped_reason": decision.skipped_reason,
        "keep": [{"epoch": c.epoch, "path": str(c.path), "size_bytes": c.size_bytes} for c in decision.keep],
        "removed": [{"epoch": c.epoch, "path": str(c.path), "size_bytes": c.size_bytes} for c in removed],
        "protected": [
            {"epoch": c.epoch, "path": str(c.path), "size_bytes": c.size_bytes, "reason": reason}
            for c, reason in decision.protected
        ],
        "error": error,
    }
    write_json(report_path, payload)
    return report_path


def execute_cleanup(
    plan: CleanupPlan,
    *,
    commit: bool,
) -> CleanupExecutionReport:
    removed_count = 0
    skipped_count = 0
    freed_bytes = 0

    for decision in plan.decisions:
        if decision.skipped_reason is not None:
            skipped_count += 1
            _write_cleanup_report(plan, decision, commit=False, removed=(), error=None)
            continue

        if commit:
            actually_removed: list[CheckpointCandidate] = []
            for candidate in decision.remove:
                try:
                    candidate.path.unlink()
                    actually_removed.append(candidate)
                    freed_bytes += candidate.size_bytes
                    removed_count += 1
                except OSError:
                    pass
            _write_cleanup_report(plan, decision, commit=True, removed=tuple(actually_removed), error=None)
        else:
            _write_cleanup_report(plan, decision, commit=False, removed=decision.remove, error=None)
            freed_bytes += sum(c.size_bytes for c in decision.remove)

    return CleanupExecutionReport(
        plan=plan,
        committed=commit,
        removed_count=removed_count,
        skipped_count=skipped_count,
        freed_bytes=freed_bytes,
    )


def format_cleanup_plan_table(plan: CleanupPlan) -> str:
    if not plan.decisions:
        return "no XP folders found"

    lines = [
        f"{'sig':<10} {'action':<14} {'keep':>6} {'remove':>6} {'freed':>10}  protected",
    ]
    for decision in plan.decisions:
        if decision.skipped_reason:
            action = f"skip ({decision.skipped_reason[:30]})"
            keep_n = "-"
            remove_n = "-"
            freed = "-"
        elif decision.remove:
            action = "clean"
            keep_n = str(len(decision.keep))
            remove_n = str(len(decision.remove))
            freed = format_bytes(sum(c.size_bytes for c in decision.remove))
        else:
            action = "nothing-to-do"
            keep_n = str(len(decision.keep))
            remove_n = "0"
            freed = "0B"
        held = ", ".join(f"{c.path.name}: {reason}" for c, reason in decision.protected)
        lines.append(f"{decision.sig:<10} {action:<14} {keep_n:>6} {remove_n:>6} {freed:>10}  {held}".rstrip())
    return "\n".join(lines)
