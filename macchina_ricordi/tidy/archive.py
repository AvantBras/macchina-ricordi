from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..musicgen.shared import read_json, sha256_file, utc_now_iso, write_json, write_text_atomic
from .prune import format_bytes, parse_duration_expression

ARCHIVE_REPORT_SCHEMA_VERSION = 1
ARCHIVE_MANIFEST_SCHEMA_VERSION = 1
ARCHIVE_INDEX_SCHEMA_VERSION = 1
CHECKPOINT_NAME_PATTERN = re.compile(r"^checkpoint_(?P<epoch>\d+)\.th$")
CHECKPOINT_CHUNK_SIZE = 1024 * 1024
ARCHIVE_COMPANION_FILES = (
    ".argv.json",
    "history.json",
    "macchina_ricordi_train_metadata.json",
    "outputs/hyperparams.json",
)
ARCHIVE_COMPANION_DIRECTORIES = (".hydra",)


@dataclass(frozen=True)
class ArchiveCandidate:
    sig: str
    xp_path: Path
    checkpoint_path: Path
    checkpoint_name: str
    epoch: int | None
    size_bytes: int
    mtime: float
    label: str | None
    profile: str | None
    metadata_present: bool
    running: bool


@dataclass(frozen=True)
class ArchiveDecision:
    candidate: ArchiveCandidate
    action: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArchivePlan:
    dora_dir: Path
    xps_root: Path
    destination: Path
    decisions: tuple[ArchiveDecision, ...]
    checkpoint_mode: str
    sigs: tuple[str, ...]
    labels: tuple[str, ...]
    profiles: tuple[str, ...]
    min_age_seconds: float | None
    min_epoch: int | None
    require_metadata: bool


@dataclass(frozen=True)
class ArchiveExecutionReport:
    plan: ArchivePlan
    committed: bool
    archived_count: int
    already_archived_count: int
    skipped_count: int
    failed_count: int
    archived_bytes: int
    report_path: Path


class ArchiveError(RuntimeError):
    """Raised when a checkpoint cannot be archived safely."""


def plan_archive(
    *,
    dora_dir: Path,
    destination: Path,
    checkpoint_mode: str = "latest",
    sigs: Sequence[str] = (),
    labels: Sequence[str] = (),
    profiles: Sequence[str] = (),
    min_age: str | None = None,
    min_epoch: int | None = None,
    require_metadata: bool = False,
    now_seconds: float | None = None,
) -> ArchivePlan:
    if checkpoint_mode not in {"latest", "epochs", "all"}:
        raise ValueError("checkpoint_mode must be one of: latest, epochs, all")
    if min_epoch is not None and min_epoch < 0:
        raise ValueError("min_epoch must be >= 0")

    resolved_dora = dora_dir.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    xps_root = resolved_dora / "xps"
    sig_set = _normalised_values(sigs)
    label_values = _normalised_values(labels)
    profile_values = _normalised_values(profiles)
    min_age_seconds = parse_duration_expression(min_age)
    current_time = now_seconds if now_seconds is not None else datetime.now(UTC).timestamp()

    decisions: list[ArchiveDecision] = []
    if not xps_root.is_dir():
        return ArchivePlan(
            dora_dir=resolved_dora,
            xps_root=xps_root,
            destination=resolved_destination,
            decisions=(),
            checkpoint_mode=checkpoint_mode,
            sigs=tuple(sig_set),
            labels=tuple(label_values),
            profiles=tuple(profile_values),
            min_age_seconds=min_age_seconds,
            min_epoch=min_epoch,
            require_metadata=require_metadata,
        )

    for xp_path in sorted(xps_root.iterdir(), key=lambda path: path.name):
        if not xp_path.is_dir() or (sig_set and xp_path.name not in sig_set):
            continue
        metadata = _read_metadata(xp_path / "macchina_ricordi_train_metadata.json")
        label = _metadata_string(metadata, "effective_label") or _metadata_string(metadata, "label")
        profile = _metadata_string(metadata, "profile")
        running = _is_xp_running(xp_path, metadata)
        candidates = _scan_checkpoint_candidates(xp_path, checkpoint_mode)
        for candidate in candidates:
            archive_candidate = _build_candidate(
                xp_path=xp_path,
                checkpoint_path=candidate,
                label=label,
                profile=profile,
                metadata_present=metadata is not None,
                running=running,
            )
            reasons = _selection_reasons(
                archive_candidate,
                labels=label_values,
                profiles=profile_values,
                min_age_seconds=min_age_seconds,
                min_epoch=min_epoch,
                require_metadata=require_metadata,
                now_seconds=current_time,
            )
            if reasons:
                decisions.append(ArchiveDecision(archive_candidate, "skip", tuple(reasons)))
                continue

            if _is_already_archived(resolved_destination, archive_candidate):
                decisions.append(
                    ArchiveDecision(
                        archive_candidate,
                        "already_archived",
                        ("source size and mtime match an archive index entry",),
                    )
                )
            else:
                decisions.append(ArchiveDecision(archive_candidate, "archive", ("matches archive policy",)))

    return ArchivePlan(
        dora_dir=resolved_dora,
        xps_root=xps_root,
        destination=resolved_destination,
        decisions=tuple(decisions),
        checkpoint_mode=checkpoint_mode,
        sigs=tuple(sig_set),
        labels=tuple(label_values),
        profiles=tuple(profile_values),
        min_age_seconds=min_age_seconds,
        min_epoch=min_epoch,
        require_metadata=require_metadata,
    )


def execute_archive_plan(
    plan: ArchivePlan,
    *,
    commit: bool,
    report_path: Path | None = None,
    allow_same_filesystem: bool = False,
    progress: Callable[[int, int, ArchiveDecision], None] | None = None,
) -> ArchiveExecutionReport:
    """``progress(done, total, result)`` is called after each decision when committing."""
    if commit:
        _validate_destination(
            dora_dir=plan.dora_dir,
            destination=plan.destination,
            allow_same_filesystem=allow_same_filesystem,
        )
        plan.destination.mkdir(parents=True, exist_ok=True)

    archived_count = 0
    already_archived_count = 0
    skipped_count = 0
    failed_count = 0
    archived_bytes = 0
    execution_decisions: list[ArchiveDecision] = []

    if commit:
        with _archive_lock(plan.destination):
            for decision in plan.decisions:
                result = _execute_decision(decision, plan.destination)
                execution_decisions.append(result)
                if progress is not None:
                    progress(len(execution_decisions), len(plan.decisions), result)
                if result.action == "archived":
                    archived_count += 1
                    archived_bytes += result.candidate.size_bytes
                elif result.action == "already_archived":
                    already_archived_count += 1
                elif result.action == "failed":
                    failed_count += 1
                else:
                    skipped_count += 1
    else:
        for decision in plan.decisions:
            execution_decisions.append(decision)
            if decision.action == "archive":
                skipped_count += 1
            elif decision.action == "already_archived":
                already_archived_count += 1
            else:
                skipped_count += 1

    resolved_report_path = report_path or _default_report_path(plan.dora_dir)
    report_payload = build_archive_report_payload(
        plan=plan,
        committed=commit,
        decisions=execution_decisions,
        archived_count=archived_count,
        already_archived_count=already_archived_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        archived_bytes=archived_bytes,
    )
    write_json(resolved_report_path, report_payload)
    return ArchiveExecutionReport(
        plan=plan,
        committed=commit,
        archived_count=archived_count,
        already_archived_count=already_archived_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        archived_bytes=archived_bytes,
        report_path=resolved_report_path,
    )


def validate_archive_destination(
    *,
    dora_dir: Path,
    destination: Path,
    allow_same_filesystem: bool = False,
) -> Path:
    """Validate the destination without creating it.

    The nearest existing parent must be on a different filesystem by default.
    This prevents an unmounted HDD path from silently becoming a directory on
    the system disk.
    """

    return _validate_destination(
        dora_dir=dora_dir,
        destination=destination,
        allow_same_filesystem=allow_same_filesystem,
    )


def build_archive_report_payload(
    *,
    plan: ArchivePlan,
    committed: bool,
    decisions: Sequence[ArchiveDecision],
    archived_count: int,
    already_archived_count: int,
    skipped_count: int,
    failed_count: int,
    archived_bytes: int,
) -> dict[str, object]:
    return {
        "schema_version": ARCHIVE_REPORT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "dora_dir": str(plan.dora_dir),
        "xps_root": str(plan.xps_root),
        "destination": str(plan.destination),
        "committed": committed,
        "checkpoint_mode": plan.checkpoint_mode,
        "sigs": list(plan.sigs),
        "labels": list(plan.labels),
        "profiles": list(plan.profiles),
        "min_age_seconds": plan.min_age_seconds,
        "min_epoch": plan.min_epoch,
        "require_metadata": plan.require_metadata,
        "archived_count": archived_count,
        "already_archived_count": already_archived_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "archived_bytes": archived_bytes,
        "decisions": [
            {
                "sig": decision.candidate.sig,
                "xp_path": str(decision.candidate.xp_path),
                "checkpoint_path": str(decision.candidate.checkpoint_path),
                "checkpoint_name": decision.candidate.checkpoint_name,
                "epoch": decision.candidate.epoch,
                "size_bytes": decision.candidate.size_bytes,
                "mtime": decision.candidate.mtime,
                "label": decision.candidate.label,
                "profile": decision.candidate.profile,
                "metadata_present": decision.candidate.metadata_present,
                "running": decision.candidate.running,
                "action": decision.action,
                "reasons": list(decision.reasons),
            }
            for decision in decisions
        ],
    }


def format_archive_plan_table(plan: ArchivePlan) -> str:
    if not plan.decisions:
        return "no matching checkpoint candidates found"
    lines = [f"{'action':<17} {'size':>10} {'profile':<24} {'label':<24} checkpoint"]
    for decision in plan.decisions:
        candidate = decision.candidate
        lines.append(
            f"{decision.action:<17} "
            f"{format_bytes(candidate.size_bytes):>10} "
            f"{(candidate.profile or '-'):24.24} "
            f"{(candidate.label or '-'):24.24} "
            f"{candidate.sig}/{candidate.checkpoint_name}"
        )
        if decision.action == "skip":
            lines.append(f"  reason: {', '.join(decision.reasons)}")
    return "\n".join(lines)


def _normalised_values(values: Sequence[str]) -> list[str]:
    return sorted({value.strip() for value in values if value.strip()})


def _scan_checkpoint_candidates(xp_path: Path, checkpoint_mode: str) -> list[Path]:
    candidates: list[Path] = []
    if checkpoint_mode in {"latest", "all"}:
        latest = xp_path / "checkpoint.th"
        if latest.is_file():
            candidates.append(latest)
    if checkpoint_mode in {"epochs", "all"}:
        epoch_paths = [
            path for path in xp_path.iterdir() if path.is_file() and CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
        ]
        candidates.extend(
            sorted(
                epoch_paths,
                key=lambda path: int(CHECKPOINT_NAME_PATTERN.fullmatch(path.name)["epoch"]),
            )
        )
    return candidates


def _build_candidate(
    *,
    xp_path: Path,
    checkpoint_path: Path,
    label: str | None,
    profile: str | None,
    metadata_present: bool,
    running: bool,
) -> ArchiveCandidate:
    stat = checkpoint_path.stat()
    match = CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_path.name)
    epoch = int(match["epoch"]) if match else None
    return ArchiveCandidate(
        sig=xp_path.name,
        xp_path=xp_path,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_path.name,
        epoch=epoch,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        label=label,
        profile=profile,
        metadata_present=metadata_present,
        running=running,
    )


def _selection_reasons(
    candidate: ArchiveCandidate,
    *,
    labels: Sequence[str],
    profiles: Sequence[str],
    min_age_seconds: float | None,
    min_epoch: int | None,
    require_metadata: bool,
    now_seconds: float,
) -> list[str]:
    reasons: list[str] = []
    if candidate.running:
        reasons.append("XP status is running")
    if require_metadata and not candidate.metadata_present:
        reasons.append("metadata is missing or invalid")
    if labels and candidate.label not in labels:
        reasons.append(f"label does not match {', '.join(labels)}")
    if profiles and candidate.profile not in profiles:
        reasons.append(f"profile does not match {', '.join(profiles)}")
    if min_age_seconds is not None and now_seconds - candidate.mtime < min_age_seconds:
        reasons.append(f"checkpoint is younger than {min_age_seconds:.0f}s")
    if min_epoch is not None and candidate.epoch is not None and candidate.epoch < min_epoch:
        reasons.append(f"epoch is below {min_epoch}")
    return reasons


def _read_metadata(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _metadata_string(metadata: dict[str, object] | None, key: str) -> str | None:
    if metadata is None:
        return None
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_xp_running(xp_path: Path, metadata: dict[str, object] | None) -> bool:
    from ..training.jobs import training_sigs

    if xp_path.name in training_sigs():
        return True
    status_paths = [xp_path / "train_run_status.json"]
    if xp_path.parent.name == "xps":
        status_paths.append(xp_path.parent.parent / "logs" / "train_run_status.json")
    workspace_dir = _metadata_string(metadata, "workspace_dir")
    if workspace_dir:
        status_paths.append(Path(workspace_dir).expanduser() / "logs" / "train_run_status.json")
    for status_path in status_paths:
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") == "running":
            return True
    return False


def _is_already_archived(destination: Path, candidate: ArchiveCandidate) -> bool:
    index_path = destination / candidate.sig / "archive_index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("checkpoint_name") != candidate.checkpoint_name:
            continue
        if entry.get("size_bytes") != candidate.size_bytes or entry.get("source_mtime") != candidate.mtime:
            continue
        archive_path = entry.get("archive_path")
        source_sha256 = entry.get("source_sha256")
        if not isinstance(archive_path, str) or not isinstance(source_sha256, str):
            continue
        manifest = read_json(destination / candidate.sig / archive_path / "archive_manifest.json") or {}
        if manifest.get("source_sha256") == source_sha256:
            return True
    return False


def _execute_decision(decision: ArchiveDecision, destination: Path) -> ArchiveDecision:
    if decision.action != "archive":
        return decision
    candidate = decision.candidate
    try:
        if not candidate.checkpoint_path.is_file():
            raise ArchiveError("source checkpoint disappeared")
        if candidate.running:
            raise ArchiveError("XP became running before the copy started")
        if _is_already_archived(destination, candidate):
            return ArchiveDecision(candidate, "already_archived", ("archive index matches source",))
        if shutil.disk_usage(destination).free < candidate.size_bytes:
            raise ArchiveError(
                f"not enough free space for {candidate.checkpoint_name} ({candidate.size_bytes} bytes required)"
            )
        bundle_root = destination / candidate.sig
        staging = bundle_root / f".staging-{uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            checkpoint_destination = staging / candidate.checkpoint_name
            source_sha256 = _copy_file_with_hash(candidate.checkpoint_path, checkpoint_destination)
            _assert_source_unchanged(candidate)
            copied_files = [
                _file_manifest_entry(
                    source_path=candidate.checkpoint_path,
                    destination_path=checkpoint_destination,
                    relative_path=candidate.checkpoint_name,
                    sha256=source_sha256,
                )
            ]
            copied_files.extend(_copy_companions(candidate.xp_path, staging))
            manifest = {
                "schema_version": ARCHIVE_MANIFEST_SCHEMA_VERSION,
                "archived_at": utc_now_iso(),
                "source_xp": str(candidate.xp_path),
                "source_checkpoint": str(candidate.checkpoint_path),
                "source_checkpoint_name": candidate.checkpoint_name,
                "epoch": candidate.epoch,
                "source_sha256": source_sha256,
                "source_size_bytes": candidate.size_bytes,
                "source_mtime": candidate.mtime,
                "sig": candidate.sig,
                "label": candidate.label,
                "profile": candidate.profile,
                "files": copied_files,
            }
            write_text_atomic(staging / "archive_manifest.json", json.dumps(manifest, indent=2))
            versions_dir = bundle_root / "versions"
            versions_dir.mkdir(parents=True, exist_ok=True)
            final_dir = versions_dir / source_sha256[:16]
            if final_dir.exists():
                existing_manifest = read_json(final_dir / "archive_manifest.json") or {}
                if existing_manifest.get("source_sha256") != source_sha256:
                    raise ArchiveError(f"archive hash collision at {final_dir}")
                shutil.rmtree(staging)
            else:
                staging.rename(final_dir)
            _update_archive_index(bundle_root, manifest, final_dir)
            return ArchiveDecision(candidate, "archived", (str(final_dir),))
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (ArchiveError, OSError, ValueError) as exc:
        return ArchiveDecision(candidate, "failed", (str(exc),))


def _copy_companions(xp_path: Path, staging: Path) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for relative_name in ARCHIVE_COMPANION_FILES:
        source_path = xp_path / relative_name
        if not source_path.is_file():
            continue
        destination_path = staging / relative_name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        sha256 = _copy_file_with_hash(source_path, destination_path)
        copied.append(
            _file_manifest_entry(
                source_path=source_path,
                destination_path=destination_path,
                relative_path=relative_name,
                sha256=sha256,
            )
        )
    for relative_name in ARCHIVE_COMPANION_DIRECTORIES:
        source_path = xp_path / relative_name
        if not source_path.is_dir():
            continue
        destination_path = staging / relative_name
        shutil.copytree(source_path, destination_path)
        for copied_path in sorted(path for path in destination_path.rglob("*") if path.is_file()):
            relative_path = copied_path.relative_to(staging).as_posix()
            copied.append(
                _file_manifest_entry(
                    source_path=source_path / copied_path.relative_to(destination_path),
                    destination_path=copied_path,
                    relative_path=relative_path,
                    sha256=sha256_file(copied_path),
                )
            )
    return copied


def _file_manifest_entry(
    *,
    source_path: Path,
    destination_path: Path,
    relative_path: str,
    sha256: str,
) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "source_path": str(source_path),
        "size_bytes": destination_path.stat().st_size,
        "sha256": sha256,
    }


def _copy_file_with_hash(source_path: Path, destination_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as source, destination_path.open("wb") as destination:
        while chunk := source.read(CHECKPOINT_CHUNK_SIZE):
            digest.update(chunk)
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    return digest.hexdigest()


def _assert_source_unchanged(candidate: ArchiveCandidate) -> None:
    try:
        stat = candidate.checkpoint_path.stat()
    except OSError as exc:
        raise ArchiveError(f"source checkpoint disappeared: {exc}") from exc
    if stat.st_size != candidate.size_bytes or stat.st_mtime != candidate.mtime:
        raise ArchiveError("source checkpoint changed during copy")


def _update_archive_index(bundle_root: Path, manifest: dict[str, object], final_dir: Path) -> None:
    index_path = bundle_root / "archive_index.json"
    existing = read_json(index_path) or {}
    entries = existing.get("entries") if isinstance(existing.get("entries"), list) else []
    entry = {
        "archived_at": manifest["archived_at"],
        "archive_path": str(final_dir.relative_to(bundle_root)),
        "checkpoint_name": manifest["source_checkpoint_name"],
        "epoch": manifest["epoch"],
        "source_sha256": manifest["source_sha256"],
        "size_bytes": manifest["source_size_bytes"],
        "source_mtime": manifest["source_mtime"],
    }
    if not any(
        isinstance(previous, dict)
        and previous.get("checkpoint_name") == entry["checkpoint_name"]
        and previous.get("source_sha256") == entry["source_sha256"]
        for previous in entries
    ):
        entries.append(entry)
    index = {
        "schema_version": ARCHIVE_INDEX_SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "sig": manifest["sig"],
        "entries": entries,
    }
    write_text_atomic(index_path, json.dumps(index, indent=2))


def _validate_destination(*, dora_dir: Path, destination: Path, allow_same_filesystem: bool) -> Path:
    resolved_dora = dora_dir.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if not resolved_dora.exists():
        raise ValueError(f"Dora directory does not exist: {resolved_dora}")
    existing_parent = _nearest_existing_parent(resolved_destination)
    if not existing_parent.is_dir():
        raise ValueError(f"Destination parent is not a directory: {existing_parent}")
    if not allow_same_filesystem and os.stat(resolved_dora).st_dev == os.stat(existing_parent).st_dev:
        raise ValueError(
            f"Destination appears to be on the same filesystem as {resolved_dora}; "
            "refusing to archive there. Mount the other HDD or pass --allow-same-filesystem."
        )
    return resolved_destination


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


@contextmanager
def _archive_lock(destination: Path) -> Iterator[None]:
    lock_path = destination / ".checkpoint_archive.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        handle.close()


def _default_report_path(dora_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return dora_dir / f"checkpoint_archive_report_{timestamp}.json"


__all__ = [
    "ArchiveCandidate",
    "ArchiveDecision",
    "ArchiveError",
    "ArchiveExecutionReport",
    "ArchivePlan",
    "build_archive_report_payload",
    "execute_archive_plan",
    "format_archive_plan_table",
    "plan_archive",
    "validate_archive_destination",
]
