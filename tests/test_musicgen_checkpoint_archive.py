from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.tidy.archive import (
    execute_archive_plan,
    plan_archive,
    validate_archive_destination,
)


def _write_xp(
    dora: Path,
    sig: str,
    *,
    label: str = "demo",
    profile: str = "micro-unconditional",
    checkpoint_bytes: bytes = b"checkpoint",
) -> Path:
    xp = dora / "xps" / sig
    xp.mkdir(parents=True)
    (xp / "checkpoint.th").write_bytes(checkpoint_bytes)
    (xp / "checkpoint_5.th").write_bytes(b"epoch-5")
    (xp / "history.json").write_text("[]", encoding="utf-8")
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps(
            {
                "effective_label": label,
                "profile": profile,
                "workspace_dir": str(xp / "workspace"),
            }
        ),
        encoding="utf-8",
    )
    return xp


def test_execute_archive_copies_verified_bundle_and_is_idempotent(tmp_path: Path) -> None:
    dora = tmp_path / "dora"
    xp = _write_xp(dora, "abc123", checkpoint_bytes=b"a" * 8192)
    (xp / ".hydra").mkdir()
    (xp / ".hydra" / "config.yaml").write_text("demo: true\n", encoding="utf-8")
    destination = tmp_path / "archive"
    report_path = tmp_path / "archive-report.json"

    plan = plan_archive(dora_dir=dora, destination=destination)
    report = execute_archive_plan(
        plan,
        commit=True,
        report_path=report_path,
        allow_same_filesystem=True,
    )

    assert report.archived_count == 1
    assert report.failed_count == 0
    assert report.report_path == report_path
    versions = list((destination / "abc123" / "versions").iterdir())
    assert len(versions) == 1
    bundle = versions[0]
    assert (bundle / "checkpoint.th").read_bytes() == b"a" * 8192
    assert (bundle / "history.json").exists()
    assert (bundle / "macchina_ricordi_train_metadata.json").exists()
    assert (bundle / ".hydra" / "config.yaml").exists()
    manifest = json.loads((bundle / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_checkpoint_name"] == "checkpoint.th"
    assert manifest["source_size_bytes"] == 8192
    assert json.loads((destination / "abc123" / "archive_index.json").read_text(encoding="utf-8"))["entries"]

    second_plan = plan_archive(dora_dir=dora, destination=destination)
    assert second_plan.decisions[0].action == "already_archived"
    second_report = execute_archive_plan(second_plan, commit=True, allow_same_filesystem=True)
    assert second_report.archived_count == 0
    assert second_report.already_archived_count == 1

    archive_index = destination / "abc123" / "archive_index.json"
    index_payload = json.loads(archive_index.read_text(encoding="utf-8"))
    archived_bundle = destination / "abc123" / index_payload["entries"][0]["archive_path"]
    (archived_bundle / "archive_manifest.json").unlink()
    third_plan = plan_archive(dora_dir=dora, destination=destination)
    assert third_plan.decisions[0].action == "archive"


def test_destination_guard_rejects_same_filesystem_by_default(tmp_path: Path) -> None:
    dora = tmp_path / "dora"
    (dora / "xps").mkdir(parents=True)

    with pytest.raises(ValueError, match="same filesystem"):
        validate_archive_destination(dora_dir=dora, destination=tmp_path / "archive")
