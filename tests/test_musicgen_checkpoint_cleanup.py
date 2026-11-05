from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from macchina_ricordi.tidy.clean import (
    CheckpointCandidate,
    classify_checkpoints,
    execute_cleanup,
    format_cleanup_plan_table,
    plan_cleanup,
)

# --- classify_checkpoints ---


@pytest.mark.parametrize(
    ("epochs", "keep_last", "kept", "removed"),
    [
        ([5, 10, 15, 20], 2, [20, 15], [5, 10]),  # the newest n survive
        ([5, 10], 5, [10, 5], []),  # under the limit, nothing goes
        ([5, 10], 0, [], [5, 10]),  # keep none, remove all
        ([], 3, [], []),
    ],
)
def test_classify_checkpoints_keeps_the_newest_n(
    epochs: list[int], keep_last: int, kept: list[int], removed: list[int]
) -> None:
    cands = [CheckpointCandidate(epoch=e, path=Path(f"c{e}.th"), size_bytes=100, mtime=float(e)) for e in epochs]
    keep, remove = classify_checkpoints(cands, keep_last=keep_last)
    assert [c.epoch for c in keep] == kept
    assert sorted(c.epoch for c in remove) == removed


# --- plan_cleanup ---


def test_plan_cleanup_skips_running(tmp_path: Path):
    dora = tmp_path / "dora"
    xps = dora / "xps"
    xp = xps / "live"
    xp.mkdir(parents=True)
    (xp / "checkpoint_5.th").write_text("a")
    (xp / "train_run_status.json").write_text(json.dumps({"status": "running"}))

    plan = plan_cleanup(dora_dir=dora, keep_last=1, force=False)
    assert plan.decisions[0].skipped_reason is not None

    forced = plan_cleanup(dora_dir=dora, keep_last=1, force=True)
    assert forced.decisions[0].skipped_reason is None


# --- execute_cleanup ---


def test_execute_cleanup_dry_run_does_not_delete(tmp_path: Path):
    dora = tmp_path / "dora"
    xps = dora / "xps"
    xp = xps / "test"
    xp.mkdir(parents=True)
    (xp / "checkpoint_5.th").write_text("a")
    (xp / "checkpoint_10.th").write_text("b")
    (xp / "checkpoint_15.th").write_text("c")

    plan = plan_cleanup(dora_dir=dora, keep_last=1)
    report = execute_cleanup(plan, commit=False)

    assert report.committed is False
    assert report.removed_count == 0
    assert (xp / "checkpoint_5.th").exists()
    assert (xp / "checkpoint_10.th").exists()
    assert (xp / "checkpoint_15.th").exists()
    assert (xp / "checkpoint_cleanup_report.json").exists()


def test_execute_cleanup_commit_deletes_expected(tmp_path: Path):
    dora = tmp_path / "dora"
    xps = dora / "xps"
    xp = xps / "test"
    xp.mkdir(parents=True)
    c5 = xp / "checkpoint_5.th"
    c5.write_text("a")
    c10 = xp / "checkpoint_10.th"
    c10.write_text("b")
    c15 = xp / "checkpoint_15.th"
    c15.write_text("c")

    plan = plan_cleanup(dora_dir=dora, keep_last=1)
    report = execute_cleanup(plan, commit=True)

    assert report.committed is True
    assert report.removed_count == 2
    assert not c5.exists()
    assert not c10.exists()
    assert c15.exists()


def test_execute_cleanup_never_touches_final_checkpoint(tmp_path: Path):
    dora = tmp_path / "dora"
    xps = dora / "xps"
    xp = xps / "test"
    xp.mkdir(parents=True)
    (xp / "checkpoint_5.th").write_text("a")
    final = xp / "checkpoint.th"
    final.write_text("final")
    merged = xp / "checkpoint_merged.th"
    merged.write_text("merged")

    plan = plan_cleanup(dora_dir=dora, keep_last=0)
    execute_cleanup(plan, commit=True)

    assert final.exists()
    assert merged.exists()
    assert not (xp / "checkpoint_5.th").exists()


# --- CLI integration tests ---


def test_cli_clean_dry_runs_until_committed(tmp_path: Path):
    """The CLI only reports by default; --commit removes all but the newest checkpoint."""
    dora = tmp_path / "dora"
    xp = dora / "xps" / "test123"
    xp.mkdir(parents=True)
    (xp / "checkpoint_5.th").write_text("a" * 100)
    (xp / "checkpoint_10.th").write_text("b" * 100)
    (xp / "checkpoint_15.th").write_text("c" * 100)
    (xp / "checkpoint.th").write_text("final")
    base = [sys.executable, "-m", "macchina_ricordi.cli", "tidy", "clean", "--dora-dir", str(dora), "--keep-last", "1"]

    dry = subprocess.run(base, capture_output=True, text=True)
    assert dry.returncode == 0
    assert "dry_run" in dry.stdout
    assert "test123" in dry.stdout
    assert (xp / "checkpoint_5.th").exists()
    assert (xp / "checkpoint_15.th").exists()
    assert (xp / "checkpoint_cleanup_report.json").exists()

    commit = subprocess.run([*base, "--commit"], capture_output=True, text=True)
    assert commit.returncode == 0
    assert "committed: True" in commit.stdout
    assert not (xp / "checkpoint_5.th").exists()
    assert not (xp / "checkpoint_10.th").exists()
    assert (xp / "checkpoint_15.th").exists()
    assert (xp / "checkpoint.th").exists()


# --- protections ---


def _xp_with_epochs(dora: Path, sig: str, epochs: list[int], valid_ce: list[float] | None = None) -> Path:
    xp = dora / "xps" / sig
    xp.mkdir(parents=True)
    for epoch in epochs:
        (xp / f"checkpoint_{epoch}.th").write_text("x" * epoch)
    (xp / "checkpoint.th").write_text("final")
    if valid_ce is not None:
        (xp / "history.json").write_text(json.dumps([{"valid": {"ce": ce}} for ce in valid_ce]))
    return xp


def test_protected_checkpoints_survive_keep_last(tmp_path: Path):
    dora = tmp_path / "dora"
    xp = _xp_with_epochs(dora, "abc", [5, 10, 15])
    plan = plan_cleanup(dora_dir=dora, keep_last=1, protected={xp / "checkpoint_5.th": "referenced by a sweep"})
    decision = plan.decisions[0]
    assert [c.epoch for c in decision.remove] == [10]
    assert [c.epoch for c in decision.keep] == [15, 5]
    assert [(c.epoch, reason) for c, reason in decision.protected] == [(5, "referenced by a sweep")]

    execute_cleanup(plan, commit=True)
    assert (xp / "checkpoint_5.th").exists()
    report = json.loads((xp / "checkpoint_cleanup_report.json").read_text())
    assert report["protected"][0]["reason"] == "referenced by a sweep"
    assert "referenced by a sweep" in format_cleanup_plan_table(plan)
