from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..musicgen.shared import utc_now_iso, write_json

PRUNE_REPORT_SCHEMA_VERSION = 1
DURATION_PATTERN = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>h|d|w)$",
    re.IGNORECASE,
)
DURATION_UNIT_SECONDS = {
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}


@dataclass(frozen=True)
class XpScanEntry:
    sig: str
    path: Path
    size_bytes: int
    mtime: float
    has_checkpoint: bool
    running: bool
    labeled: bool


@dataclass(frozen=True)
class PruneDecision:
    entry: XpScanEntry
    action: str  # "keep" | "remove" | "skip_running"
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PrunePlan:
    dora_dir: Path
    xps_root: Path
    decisions: tuple[PruneDecision, ...]
    keep_last: int
    keep_labeled: bool
    keep_sigs: tuple[str, ...]
    keep_younger_than_seconds: float | None
    force: bool


@dataclass(frozen=True)
class PruneExecutionReport:
    plan: PrunePlan
    committed: bool
    removed_sigs: tuple[str, ...]
    skipped_sigs: tuple[str, ...]
    report_path: Path


def parse_duration_expression(expression: str | None) -> float | None:
    if expression is None:
        return None
    text = expression.strip().lower()
    if not text:
        return None
    match = DURATION_PATTERN.match(text)
    if match is None:
        raise ValueError(f"Invalid duration expression: {expression!r}. Expected e.g. '24h', '7d', '2w'.")
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    return value * DURATION_UNIT_SECONDS[unit]


def scan_xp_folders(xps_root: Path) -> list[XpScanEntry]:
    if not xps_root.exists() or not xps_root.is_dir():
        return []
    entries: list[XpScanEntry] = []
    for xp_folder in sorted(xps_root.iterdir(), key=lambda path: path.name):
        if not xp_folder.is_dir():
            continue
        entries.append(_build_xp_scan_entry(xp_folder))
    return entries


def classify_xps(
    entries: Sequence[XpScanEntry],
    *,
    keep_last: int,
    keep_labeled: bool,
    keep_sigs: Sequence[str],
    keep_younger_than_seconds: float | None,
    force: bool,
    now_seconds: float | None = None,
) -> list[PruneDecision]:
    if keep_last < 0:
        raise ValueError("keep_last must be >= 0.")
    keep_sig_set = {sig.strip() for sig in keep_sigs if sig.strip()}
    current_time = now_seconds if now_seconds is not None else _current_epoch_seconds()

    ordered_by_recent = sorted(
        entries,
        key=lambda entry: entry.mtime,
        reverse=True,
    )
    keep_last_set = {entry.sig for entry in ordered_by_recent[:keep_last]}

    decisions: list[PruneDecision] = []
    for entry in entries:
        reasons: list[str] = []
        if entry.sig in keep_last_set:
            reasons.append(f"among --keep-last {keep_last}")
        if keep_labeled and entry.labeled:
            reasons.append("labeled (keep.json or .keep present)")
        if entry.sig in keep_sig_set:
            reasons.append("explicit --keep-sig")
        if keep_younger_than_seconds is not None and current_time - entry.mtime <= keep_younger_than_seconds:
            reasons.append(f"younger than --keep-younger-than ({keep_younger_than_seconds:.0f}s)")
        if entry.running and not force:
            decisions.append(
                PruneDecision(
                    entry=entry,
                    action="skip_running",
                    reasons=("status=running; use --force to override",),
                )
            )
            continue
        if reasons:
            decisions.append(
                PruneDecision(
                    entry=entry,
                    action="keep",
                    reasons=tuple(reasons),
                )
            )
            continue
        decisions.append(
            PruneDecision(
                entry=entry,
                action="remove",
                reasons=("no keep rule matched",),
            )
        )
    return decisions


def plan_prune(
    *,
    dora_dir: Path,
    keep_last: int = 3,
    keep_labeled: bool = True,
    keep_sigs: Sequence[str] = (),
    keep_younger_than: str | None = None,
    force: bool = False,
) -> PrunePlan:
    xps_root = dora_dir / "xps"
    entries = scan_xp_folders(xps_root)
    keep_younger_than_seconds = parse_duration_expression(keep_younger_than)
    decisions = classify_xps(
        entries,
        keep_last=keep_last,
        keep_labeled=keep_labeled,
        keep_sigs=keep_sigs,
        keep_younger_than_seconds=keep_younger_than_seconds,
        force=force,
    )
    return PrunePlan(
        dora_dir=dora_dir,
        xps_root=xps_root,
        decisions=tuple(decisions),
        keep_last=keep_last,
        keep_labeled=keep_labeled,
        keep_sigs=tuple(keep_sigs),
        keep_younger_than_seconds=keep_younger_than_seconds,
        force=force,
    )


def execute_prune_plan(
    plan: PrunePlan,
    *,
    commit: bool,
    report_path: Path | None = None,
) -> PruneExecutionReport:
    removed: list[str] = []
    skipped: list[str] = []
    for decision in plan.decisions:
        if decision.action == "remove" and commit:
            try:
                shutil.rmtree(decision.entry.path)
                removed.append(decision.entry.sig)
            except OSError as exc:
                skipped.append(f"{decision.entry.sig} (rmtree failed: {exc})")
        elif decision.action == "skip_running":
            skipped.append(decision.entry.sig)

    resolved_report_path = report_path or _default_prune_report_path(plan.dora_dir)
    payload = build_prune_report_payload(
        plan=plan,
        committed=commit,
        removed=removed,
        skipped=skipped,
    )
    write_json(resolved_report_path, payload)
    return PruneExecutionReport(
        plan=plan,
        committed=commit,
        removed_sigs=tuple(removed),
        skipped_sigs=tuple(skipped),
        report_path=resolved_report_path,
    )


def build_prune_report_payload(
    *,
    plan: PrunePlan,
    committed: bool,
    removed: Sequence[str],
    skipped: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": PRUNE_REPORT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "dora_dir": str(plan.dora_dir),
        "xps_root": str(plan.xps_root),
        "committed": committed,
        "keep_last": plan.keep_last,
        "keep_labeled": plan.keep_labeled,
        "keep_sigs": list(plan.keep_sigs),
        "keep_younger_than_seconds": plan.keep_younger_than_seconds,
        "force": plan.force,
        "removed_sigs": list(removed),
        "skipped_sigs": list(skipped),
        "decisions": [
            {
                "sig": decision.entry.sig,
                "path": str(decision.entry.path),
                "action": decision.action,
                "size_bytes": decision.entry.size_bytes,
                "mtime": decision.entry.mtime,
                "labeled": decision.entry.labeled,
                "running": decision.entry.running,
                "has_checkpoint": decision.entry.has_checkpoint,
                "reasons": list(decision.reasons),
            }
            for decision in plan.decisions
        ],
    }


def format_prune_plan_table(plan: PrunePlan) -> str:
    if not plan.decisions:
        return "no XP folders found"
    lines = [
        f"{'action':<13} {'size':>10} {'age':>10} {'label':<6} {'running':<8} sig",
    ]
    now = _current_epoch_seconds()
    for decision in plan.decisions:
        age_seconds = max(0.0, now - decision.entry.mtime)
        lines.append(
            f"{decision.action:<13} "
            f"{format_bytes(decision.entry.size_bytes):>10} "
            f"{_format_age(age_seconds):>10} "
            f"{'yes' if decision.entry.labeled else '-':<6} "
            f"{'yes' if decision.entry.running else '-':<8} "
            f"{decision.entry.sig}"
        )
    return "\n".join(lines)


def _build_xp_scan_entry(xp_folder: Path) -> XpScanEntry:
    size_bytes = 0
    for path in xp_folder.rglob("*"):
        try:
            if path.is_file():
                size_bytes += path.stat().st_size
        except OSError:
            continue
    checkpoint_path = xp_folder / "checkpoint.th"
    if checkpoint_path.exists():
        mtime = checkpoint_path.stat().st_mtime
        has_checkpoint = True
    else:
        mtime = xp_folder.stat().st_mtime
        has_checkpoint = False
    labeled = _is_xp_labeled(xp_folder)
    running = _is_xp_status_running(xp_folder)
    return XpScanEntry(
        sig=xp_folder.name,
        path=xp_folder,
        size_bytes=size_bytes,
        mtime=mtime,
        has_checkpoint=has_checkpoint,
        running=running,
        labeled=labeled,
    )


def _is_xp_labeled(xp_folder: Path) -> bool:
    candidates: Iterable[Path] = (
        xp_folder / "logs" / "keep.json",
        xp_folder / ".keep",
        xp_folder / "keep.json",
    )
    return any(candidate.exists() for candidate in candidates)


def _is_xp_status_running(xp_folder: Path) -> bool:
    from ..training.jobs import training_sigs

    if xp_folder.name in training_sigs():
        return True
    status_paths = [
        xp_folder / "train_run_status.json",
        xp_folder.parent.parent / "logs" / "train_run_status.json" if xp_folder.parent.name == "xps" else None,
    ]
    for status_path in status_paths:
        if status_path is None or not status_path.exists():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "running":
            return True
    return False


def _default_prune_report_path(dora_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return dora_dir / f"prune_report_{timestamp}.json"


def _current_epoch_seconds() -> float:
    return datetime.now(UTC).timestamp()


def format_bytes(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f}TiB"


def _format_age(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
