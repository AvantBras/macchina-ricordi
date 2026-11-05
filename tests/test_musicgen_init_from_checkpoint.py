from __future__ import annotations

import json
from pathlib import Path

import pytest

import macchina_ricordi.models.checkpoints as local_checkpoints
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME
from macchina_ricordi.training.plan import (
    build_training_command,
    plan_musicgen_training,
)

WORKSPACE_DIRNAME = ".macchina_ricordi_micro_uncond"


def test_build_training_command_injects_continue_from_for_micro_unconditional(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "env" / "python.exe"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("", encoding="utf-8")

    command = build_training_command(
        musicgen_python=str(fake_python),
        profile_name="micro-unconditional",
        init_from_checkpoint_path="/custom/ckpt.th",
    )

    assert "continue_from=/custom/ckpt.th" in command
    assert sum(1 for part in command if part.startswith("continue_from=")) == 1


def test_parse_init_checkpoint_uri_local_path() -> None:
    origin, parsed = local_checkpoints.parse_init_checkpoint_uri(
        "/tmp/checkpoint.th", profile_name="micro-unconditional"
    )
    assert origin == "local"
    assert parsed == {"path": "/tmp/checkpoint.th"}


def test_plan_musicgen_training_with_init_from_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_dir = tmp_path / WORKSPACE_DIRNAME
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "train").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "valid").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "train" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "valid" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)

    source_checkpoint = tmp_path / "source_xp" / "checkpoint.th"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"weights")
    (source_checkpoint.parent / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "profile": "micro-unconditional",
                "dataset_name": "macchina_ricordi_micro_uncond",
                "train_aggregate_sha256": "source-train",
                "valid_aggregate_sha256": "source-valid",
            }
        ),
        encoding="utf-8",
    )

    (workspace_dir / "logs").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "logs" / "dataset_fingerprint.json").write_text(
        json.dumps(
            {
                "profile": "micro-unconditional",
                "dataset_name": "macchina_ricordi_micro_uncond",
                "train_aggregate_sha256": "current-train",
                "valid_aggregate_sha256": "current-valid",
            }
        ),
        encoding="utf-8",
    )

    plan = plan_musicgen_training(
        audiocraft_dir=str(audiocraft_dir),
        profile_name="micro-unconditional",
        workspace_dir=workspace_dir,
        musicgen_python=str(python_exe),
        dry_run=True,
        init_from_checkpoint=str(source_checkpoint),
    )

    assert plan.init_from_checkpoint is not None
    assert plan.init_from_checkpoint["origin"] == "local"
    assert plan.init_from_checkpoint["dataset_fingerprint_matches_current_workspace"] is False
    assert plan.init_from_checkpoint["warnings"], "expected mismatch warning"

    continue_from_entries = [override for override in plan.command if override.startswith("continue_from=")]
    assert continue_from_entries == [f"continue_from={source_checkpoint.resolve()}"]

    persisted_plan = plan.plan_record
    assert persisted_plan["init_from_checkpoint"]["origin"] == "local"
    assert persisted_plan["init_from_checkpoint"]["dataset_fingerprint_matches_current_workspace"] is False

    assert plan.train_metadata is not None
    assert "init_from_checkpoint" in plan.train_metadata
    assert plan.train_metadata["init_from_checkpoint"]["origin"] == "local"


def test_plan_musicgen_training_rejects_user_continue_from_with_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_dir = tmp_path / WORKSPACE_DIRNAME
    audiocraft_dir = tmp_path / "audiocraft"
    python_dir = tmp_path / "env"
    python_exe = python_dir / "python.exe"
    dora_exe = python_dir / "dora.exe"
    python_dir.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")
    dora_exe.write_text("", encoding="utf-8")
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "train").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "valid").mkdir(parents=True)
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "train" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (workspace_dir / "egs" / DATASET_CONFIG_NAME / "valid" / "data.jsonl.gz").write_text("{}", encoding="utf-8")
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)

    source_checkpoint = tmp_path / "source_xp" / "checkpoint.th"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"weights")

    with pytest.raises(RuntimeError, match="Cannot combine --init-from-checkpoint"):
        plan_musicgen_training(
            audiocraft_dir=str(audiocraft_dir),
            profile_name="micro-unconditional",
            workspace_dir=workspace_dir,
            musicgen_python=str(python_exe),
            overrides=["continue_from=/other/ckpt.th"],
            init_from_checkpoint=str(source_checkpoint),
        )
