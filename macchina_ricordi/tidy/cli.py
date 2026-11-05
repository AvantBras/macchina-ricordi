"""`tidy`: prune XP folders, clean and archive checkpoints, backfill old metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cli.common import MacchinaRicordiHelpFormatter
from ..models.record import dora_dir_or_default
from ..musicgen.runtime import (
    get_repo_root,
    load_musicgen_project_config,
)
from .archive import execute_archive_plan, format_archive_plan_table, plan_archive, validate_archive_destination
from .backfill import BackfillReport, backfill_xps, format_backfill_report
from .clean import cleanup_protections, execute_cleanup, format_cleanup_plan_table, plan_cleanup
from .prune import execute_prune_plan, format_bytes, format_prune_plan_table, plan_prune


def run_prune(
    *,
    dora_dir: Path,
    keep_last: int,
    keep_labeled: bool,
    keep_sigs: tuple[str, ...] | list[str],
    keep_younger_than: str | None,
    force: bool,
    commit: bool,
    report_path: Path | None = None,
) -> int:
    plan = plan_prune(
        dora_dir=dora_dir,
        keep_last=keep_last,
        keep_labeled=keep_labeled,
        keep_sigs=tuple(keep_sigs),
        keep_younger_than=keep_younger_than,
        force=force,
    )
    print(f"dora_dir: {plan.dora_dir}")
    print(f"xps_root: {plan.xps_root}")
    print(format_prune_plan_table(plan))

    report = execute_prune_plan(plan, commit=commit, report_path=report_path)
    print(f"report: {report.report_path}")
    print(f"committed: {report.committed}")
    print(f"removed: {len(report.removed_sigs)}")
    print(f"skipped: {len(report.skipped_sigs)}")
    for sig in report.removed_sigs:
        print(f"removed_sig: {sig}")
    for sig in report.skipped_sigs:
        print(f"skipped_sig: {sig}")
    if not commit:
        print("dry_run: no XP folders were deleted (pass --commit to apply)")
    return 0


def run_clean_checkpoints(
    *,
    dora_dir: Path,
    keep_last: int,
    sigs: tuple[str, ...],
    force: bool,
    commit: bool,
    protect_referenced: bool = False,
    keep_best: bool = False,
) -> int:
    protected = (
        cleanup_protections(dora_dir=dora_dir, repo_root=get_repo_root(), referenced=protect_referenced, best=keep_best)
        if protect_referenced or keep_best
        else None
    )
    plan = plan_cleanup(
        dora_dir=dora_dir,
        keep_last=keep_last,
        sigs=tuple(sigs),
        force=force,
        protected=protected,
    )
    print(f"dora_dir: {plan.dora_dir}")
    print(f"xps_root: {plan.xps_root}")
    print(format_cleanup_plan_table(plan))

    report = execute_cleanup(plan, commit=commit)
    print(f"committed: {report.committed}")
    print(f"checkpoints_removed: {report.removed_count}")
    print(f"xps_skipped: {report.skipped_count}")
    print(f"freed: {format_bytes(report.freed_bytes)}")
    if not commit:
        print("dry_run: no checkpoint files were deleted (pass --commit to apply)")
    return 0


def run_archive_checkpoints(
    *,
    dora_dir: Path,
    destination: Path,
    checkpoint_mode: str,
    sigs: tuple[str, ...],
    labels: tuple[str, ...],
    profiles: tuple[str, ...],
    min_age: str | None,
    min_epoch: int | None,
    require_metadata: bool,
    allow_same_filesystem: bool,
    commit: bool,
    report_path: Path | None = None,
) -> int:
    try:
        resolved_destination = validate_archive_destination(
            dora_dir=dora_dir,
            destination=destination,
            allow_same_filesystem=allow_same_filesystem,
        )
        plan = plan_archive(
            dora_dir=dora_dir,
            destination=resolved_destination,
            checkpoint_mode=checkpoint_mode,
            sigs=sigs,
            labels=labels,
            profiles=profiles,
            min_age=min_age,
            min_epoch=min_epoch,
            require_metadata=require_metadata,
        )
        print(f"dora_dir: {plan.dora_dir}")
        print(f"xps_root: {plan.xps_root}")
        print(f"destination: {plan.destination}")
        print(format_archive_plan_table(plan))
        report = execute_archive_plan(
            plan,
            commit=commit,
            report_path=report_path,
            allow_same_filesystem=allow_same_filesystem,
        )
    except (OSError, ValueError) as exc:
        print(f"archive_error: {exc}", file=sys.stderr)
        return 1

    print(f"report: {report.report_path}")
    print(f"committed: {report.committed}")
    print(f"archived: {report.archived_count}")
    print(f"already_archived: {report.already_archived_count}")
    print(f"skipped: {report.skipped_count}")
    print(f"failed: {report.failed_count}")
    print(f"bytes_archived: {format_bytes(report.archived_bytes)}")
    if not commit:
        print("dry_run: no checkpoint files were copied (pass --commit to apply)")
    return 1 if report.failed_count else 0


def cmd_tidy_prune(args: argparse.Namespace) -> int:
    dora_dir = dora_dir_or_default(args.dora_dir)
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else None
    return run_prune(
        dora_dir=dora_dir,
        keep_last=args.keep_last,
        keep_labeled=args.keep_labeled,
        keep_sigs=tuple(args.keep_sig or ()),
        keep_younger_than=args.keep_younger_than,
        force=args.force,
        commit=args.commit,
        report_path=report_path,
    )


def cmd_tidy_clean(args: argparse.Namespace) -> int:
    dora_dir = dora_dir_or_default(args.dora_dir)
    return run_clean_checkpoints(
        dora_dir=dora_dir,
        keep_last=args.keep_last,
        sigs=tuple(args.sig or ()),
        force=args.force,
        commit=args.commit,
        protect_referenced=args.protect_referenced,
        keep_best=args.keep_best,
    )


def cmd_tidy_archive(args: argparse.Namespace) -> int:
    dora_dir = dora_dir_or_default(args.dora_dir)
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else None
    return run_archive_checkpoints(
        dora_dir=dora_dir,
        destination=Path(args.destination).expanduser().resolve(),
        checkpoint_mode=args.checkpoint,
        sigs=tuple(args.sig or ()),
        labels=tuple(args.label or ()),
        profiles=tuple(args.profile or ()),
        min_age=args.min_age,
        min_epoch=args.min_epoch,
        require_metadata=args.require_metadata,
        allow_same_filesystem=args.allow_same_filesystem,
        commit=args.commit,
        report_path=report_path,
    )


def cmd_tidy_backfill(args: argparse.Namespace) -> int:
    config = load_musicgen_project_config()
    dora_dir = Path(args.dora_dir) if args.dora_dir else config.dora_dir
    report = backfill_xps(dora_dir, dry_run=not args.commit, force=args.force)

    if args.json:
        import json as _json

        print(_json.dumps(_report_to_dict(report), indent=2))
    else:
        print(format_backfill_report(report))

    return 1 if report.errors > 0 else 0


def _report_to_dict(report: BackfillReport) -> dict:
    return {
        "dry_run": report.dry_run,
        "total_xps": report.total,
        "written": report.written,
        "skipped": report.skipped,
        "errors": report.errors,
        "results": [
            {
                "xp_sig": r.xp_sig,
                "action": r.action,
                "metadata_path": r.metadata_path,
                "workspace_dir": r.workspace_dir,
                "label": r.label,
                "merged_from": r.merged_from,
                "lineage_sidecar": r.lineage_sidecar,
                "error": r.error,
            }
            for r in report.results
        ],
    }


def register_tidy_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tidy = subparsers.add_parser(
        "tidy",
        help="Housekeeping: prune XPs, clean and archive checkpoints",
        description="Reclaim disk space without losing what something still points at. "
        "Every command is a dry run until --commit.",
        epilog="""examples:
  python -m macchina_ricordi.cli tidy prune
  python -m macchina_ricordi.cli tidy clean --keep-every 10
  python -m macchina_ricordi.cli tidy archive --dest /mnt/archive""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    tidy_subparsers = tidy.add_subparsers(dest="tidy_command", required=True)

    prune = tidy_subparsers.add_parser(
        "prune",
        help="Enumerate .dora/xps/ folders and prune old runs (dry-run by default)",
        description="Plan XP cleanup under .dora/xps. This is a dry-run unless --commit is supplied.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    prune.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the AudioCraft dora directory (defaults to AUDIOCRAFT_DORA_DIR or the repo-managed .dora)",
    )
    prune.add_argument(
        "--keep-last",
        type=int,
        default=3,
        help="Keep the N most recently updated XPs (default: 3)",
    )
    prune.add_argument(
        "--keep-labeled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep XPs that have a logs/keep.json or .keep marker (default: on)",
    )
    prune.add_argument(
        "--keep-sig",
        action="append",
        default=[],
        help="Explicit XP signature to keep; can be passed multiple times",
    )
    prune.add_argument(
        "--keep-younger-than",
        default=None,
        help="Keep XPs whose checkpoint.th is newer than this duration (e.g. 24h, 7d, 2w)",
    )
    prune.add_argument(
        "--force",
        action="store_true",
        help="Allow pruning XPs whose train_run_status.json says status=running",
    )
    prune.add_argument(
        "--commit",
        action="store_true",
        help="Actually delete the XPs marked for removal (default: dry-run)",
    )
    prune.add_argument(
        "--report-path",
        default=None,
        help="Optional explicit path for the prune_report JSON (default: <dora-dir>/prune_report_<ts>.json)",
    )
    prune.set_defaults(func=cmd_tidy_prune)

    clean = tidy_subparsers.add_parser(
        "clean",
        help="Delete intermediate epoch checkpoints from kept XP folders",
        description=(
            "Scan XPs and remove old checkpoint_N.th files, keeping only the last N by epoch. Dry-run by default."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    clean.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the AudioCraft dora directory (defaults to AUDIOCRAFT_DORA_DIR or the repo-managed .dora)",
    )
    clean.add_argument(
        "--keep-last",
        type=int,
        default=2,
        help="Keep the N highest-epoch checkpoint files per XP (default: 2)",
    )
    clean.add_argument(
        "--sig",
        action="append",
        default=[],
        dest="sig",
        help="Limit cleanup to specific XP signature(s); can be passed multiple times",
    )
    clean.add_argument(
        "--protect-referenced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep epoch checkpoints that anything outside their XP mentions: generation runs, sweeps, "
            "sessions, merges, child XPs, notes (default: on)"
        ),
    )
    clean.add_argument(
        "--keep-best",
        action="store_true",
        help="Also keep each XP's epoch checkpoint with the lowest valid ce in history.json",
    )
    clean.add_argument(
        "--force",
        action="store_true",
        help="Allow cleanup of XPs whose train_run_status.json says status=running",
    )
    clean.add_argument(
        "--commit",
        action="store_true",
        help="Actually delete the checkpoint files (default: dry-run)",
    )
    clean.set_defaults(func=cmd_tidy_clean)

    archive = tidy_subparsers.add_parser(
        "archive",
        help="Copy matching checkpoints to another filesystem (dry-run by default)",
        description=(
            "Plan and optionally copy immutable checkpoint bundles to an archive destination. "
            "The source XPs are never modified."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    archive.add_argument(
        "--destination",
        required=True,
        help="Archive directory on the other HDD; its nearest existing parent must already be mounted",
    )
    archive.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the AudioCraft dora directory (defaults to AUDIOCRAFT_DORA_DIR or the repo-managed .dora)",
    )
    archive.add_argument(
        "--checkpoint",
        choices=("latest", "epochs", "all"),
        default="latest",
        help="Checkpoint files to consider: checkpoint.th, checkpoint_N.th, or both (default: latest)",
    )
    archive.add_argument(
        "--sig",
        action="append",
        default=[],
        help="Limit to an XP signature; can be passed multiple times",
    )
    archive.add_argument(
        "--label",
        action="append",
        default=[],
        help="Match effective training label; can be passed multiple times",
    )
    archive.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Match training profile; can be passed multiple times",
    )
    archive.add_argument(
        "--min-age",
        default=None,
        help="Only archive checkpoints at least this old (e.g. 1h, 7d, 2w)",
    )
    archive.add_argument(
        "--min-epoch",
        type=int,
        default=None,
        help="For epoch mode, ignore checkpoint_N.th files below this epoch",
    )
    archive.add_argument(
        "--require-metadata",
        action="store_true",
        help="Skip XPs without valid macchina_ricordi_train_metadata.json",
    )
    archive.add_argument(
        "--allow-same-filesystem",
        action="store_true",
        help="Allow the destination to be on the same filesystem as the source (unsafe for HDD offload)",
    )
    archive.add_argument(
        "--commit",
        action="store_true",
        help="Actually copy and verify matching checkpoints (default: dry-run)",
    )
    archive.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the JSON audit report (default: <dora-dir>/checkpoint_archive_report_<ts>.json)",
    )
    archive.set_defaults(func=cmd_tidy_archive)

    backfill_meta = tidy_subparsers.add_parser(
        "backfill",
        help="Backfill macchina_ricordi_train_metadata.json for existing XPs that lack it (dry-run by default)",
        description="Plan or write missing macchina_ricordi_train_metadata.json files for existing XPs.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    backfill_meta.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the .dora directory (defaults to the repo-managed .dora)",
    )
    backfill_meta.add_argument(
        "--commit",
        action="store_true",
        help="Actually write metadata files (default: dry-run)",
    )
    backfill_meta.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metadata files",
    )
    backfill_meta.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report",
    )
    backfill_meta.set_defaults(func=cmd_tidy_backfill)
