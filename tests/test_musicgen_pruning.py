from __future__ import annotations

import json
import os
from pathlib import Path

from macchina_ricordi.tidy.prune import (
    classify_xps,
    execute_prune_plan,
    plan_prune,
    scan_xp_folders,
)


def _make_xp(
    xps_root: Path,
    sig: str,
    *,
    mtime: float | None = None,
    labeled: bool = False,
    running: bool = False,
    with_checkpoint: bool = True,
) -> Path:
    xp_folder = xps_root / sig
    xp_folder.mkdir(parents=True, exist_ok=True)
    if with_checkpoint:
        checkpoint = xp_folder / "checkpoint.th"
        checkpoint.write_bytes(b"weights")
        if mtime is not None:
            os.utime(checkpoint, (mtime, mtime))
    if labeled:
        (xp_folder / "logs").mkdir(parents=True, exist_ok=True)
        (xp_folder / "logs" / "keep.json").write_text("{}", encoding="utf-8")
    if running:
        (xp_folder / "train_run_status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    return xp_folder


def test_classify_xps_keeps_last_n_and_labeled(tmp_path: Path) -> None:
    xps_root = tmp_path / "xps"
    _make_xp(xps_root, "a_old", mtime=1000.0)
    _make_xp(xps_root, "b_labeled", mtime=2000.0, labeled=True)
    _make_xp(xps_root, "c_recent", mtime=3000.0)
    _make_xp(xps_root, "d_newest", mtime=4000.0)

    entries = scan_xp_folders(xps_root)
    decisions = classify_xps(
        entries,
        keep_last=2,
        keep_labeled=True,
        keep_sigs=(),
        keep_younger_than_seconds=None,
        force=False,
    )

    by_sig = {decision.entry.sig: decision for decision in decisions}
    assert by_sig["d_newest"].action == "keep"
    assert by_sig["c_recent"].action == "keep"
    assert by_sig["b_labeled"].action == "keep"
    assert "labeled" in " ".join(by_sig["b_labeled"].reasons)
    assert by_sig["a_old"].action == "remove"


def test_execute_prune_plan_dry_run_never_deletes(tmp_path: Path) -> None:
    dora_dir = tmp_path / "dora"
    xps_root = dora_dir / "xps"
    _make_xp(xps_root, "a", mtime=1000.0)
    _make_xp(xps_root, "b", mtime=2000.0)
    _make_xp(xps_root, "c", mtime=3000.0)

    plan = plan_prune(dora_dir=dora_dir, keep_last=1)
    report = execute_prune_plan(plan, commit=False)

    assert report.committed is False
    assert report.removed_sigs == ()
    assert (xps_root / "a").exists()
    assert (xps_root / "b").exists()
    assert (xps_root / "c").exists()
    assert report.report_path.exists()
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["committed"] is False
    remove_sigs = {d["sig"] for d in payload["decisions"] if d["action"] == "remove"}
    assert remove_sigs == {"a", "b"}


def test_execute_prune_plan_skips_running_by_default(tmp_path: Path) -> None:
    dora_dir = tmp_path / "dora"
    xps_root = dora_dir / "xps"
    _make_xp(xps_root, "old_running", mtime=500.0, running=True)
    _make_xp(xps_root, "recent", mtime=5000.0)

    plan = plan_prune(dora_dir=dora_dir, keep_last=0)
    report = execute_prune_plan(plan, commit=True)

    assert "old_running" in report.skipped_sigs
    assert (xps_root / "old_running").exists()
    assert not (xps_root / "recent").exists()
