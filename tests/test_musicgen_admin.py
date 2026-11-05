from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.musicgen.xp_datasource import XpDatasourceMissing
from macchina_ricordi.training import resume


def _runtime_config(tmp_path: Path) -> SimpleNamespace:
    runtime_python = tmp_path / ".venv-musicgen" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("", encoding="utf-8")
    return SimpleNamespace(
        repo_root=tmp_path,
        dora_dir=tmp_path / ".dora",
        audiocraft_dir=tmp_path / ".audiocraft",
        musicgen_python=runtime_python,
    )


def test_resume_xp_delegates_to_musicgen_runtime_when_audiocraft_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _runtime_config(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run(command, cwd, env, check):  # type: ignore[no-untyped-def]
        calls.append({"command": list(command), "cwd": cwd, "env": env, "check": check})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(resume, "_runtime_modules_available", lambda: False)
    monkeypatch.setattr(resume, "load_musicgen_project_config", lambda: config)
    monkeypatch.setattr(resume, "build_musicgen_subprocess_env", lambda base_env=None: {"PATH": "demo"})
    monkeypatch.setattr(resume.subprocess, "run", fake_run)

    assert resume.run_resume_xp("abcd1234", epochs=60) == 0

    assert calls == [
        {
            "command": [
                str(config.musicgen_python),
                "-m",
                "macchina_ricordi.cli",
                "train",
                "resume",
                "abcd1234",
                "--epochs",
                "60",
                "--in-process",
            ],
            "cwd": tmp_path,
            "env": {"PATH": "demo"},
            "check": False,
        }
    ]


def _write_xp_with_datasource(xp_folder: Path, data_root: Path) -> None:
    (xp_folder / ".hydra").mkdir(parents=True)
    (xp_folder / ".argv.json").write_text(
        '["solver=musicgen/musicgen_base_32khz", "dset=audio/macchina_ricordi_micro_uncond", "optim.epochs=50"]',
        encoding="utf-8",
    )
    (xp_folder / ".hydra" / "config.yaml").write_text(
        f"""
datasource:
  max_sample_rate: 32000
  max_channels: 1
  train: {data_root / "train"}
  valid: {data_root / "valid"}
  evaluate: {data_root / "evaluate"}
  generate: {data_root / "generate"}
""".lstrip(),
        encoding="utf-8",
    )


def test_restore_xp_dataset_config_from_saved_hydra_config(monkeypatch, tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    xp_folder = config.dora_dir / "xps" / "abcd1234"
    data_root = tmp_path / "ws_a" / "egs"
    for split in ("train", "valid"):
        (data_root / split).mkdir(parents=True)
    _write_xp_with_datasource(xp_folder, data_root)
    monkeypatch.setattr(resume, "load_musicgen_project_config", lambda: config)

    restored = resume._restore_xp_dataset_config("abcd1234")

    assert restored.status == "restored"
    assert (
        restored.config_path
        == config.audiocraft_dir / "config" / "dset" / "audio" / "macchina_ricordi_micro_uncond.yaml"
    )
    payload = restored.config_path.read_text(encoding="utf-8")
    assert f'train: "{data_root / "train"}"' in payload
    assert f'valid: "{data_root / "valid"}"' in payload
    assert f'evaluate: "{data_root / "evaluate"}"' in payload
    assert f'generate: "{data_root / "generate"}"' in payload
    status = json.loads((xp_folder / "dataset_config_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "restored"
    assert status["purpose"] == "resume"


def test_restore_xp_dataset_config_refuses_when_training_data_is_gone(monkeypatch, tmp_path: Path) -> None:
    config = _runtime_config(tmp_path)
    xp_folder = config.dora_dir / "xps" / "abcd1234"
    _write_xp_with_datasource(xp_folder, tmp_path / "deleted_ws" / "egs")
    shared = config.audiocraft_dir / "config" / "dset" / "audio" / "macchina_ricordi_micro_uncond.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text('datasource:\n  train: "/elsewhere/ws_b/train"\n', encoding="utf-8")
    monkeypatch.setattr(resume, "load_musicgen_project_config", lambda: config)

    with pytest.raises(XpDatasourceMissing, match="no longer exists"):
        resume._restore_xp_dataset_config("abcd1234")

    assert "/elsewhere/ws_b/train" in shared.read_text(encoding="utf-8")
    status = json.loads((xp_folder / "dataset_config_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
