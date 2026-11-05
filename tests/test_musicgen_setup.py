from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from macchina_ricordi.musicgen.setup import (
    run_musicgen_preflight,
    setup_repo_managed_musicgen_runtime,
)


# Left behind in .audiocraft by the removed text-conditioned profile.
def _write_musicgen_project(repo_root: Path, *, commit_sha: str = "1ca32f28d2d6c223b4e2335042ac558c308f2ea2") -> None:
    musicgen_dir = repo_root / "musicgen"
    musicgen_dir.mkdir(parents=True)
    (musicgen_dir / "pyproject.toml").write_text(
        f"""
[project]
name = "macchina_ricordi_musicgen_runtime"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = []

[tool.uv]
package = false

[tool.macchina_ricordi.runtime]
project_environment = ".venv-musicgen"
checkout_dir = ".audiocraft"
dora_dir = ".dora"
python_request = "3.11"

[tool.macchina_ricordi.audiocraft]
repo_url = "https://github.com/aaronabebe/audiocraft"
commit_sha = "{commit_sha}"

[tool.macchina_ricordi.defaults]
recipe = "micro-unconditional"
valid_fraction = 0.1
split_seed = 42

[tool.macchina_ricordi.recipes.base]
abstract = true
overrides = ["solver=musicgen/musicgen_base_32khz"]

[tool.macchina_ricordi.recipes."micro-unconditional"]
extends = "base"
overrides = ["dataset.segment_duration=15"]

[tool.macchina_ricordi.recipes."reshape-nano"]
extends = "base"
overrides = ["transformer_lm.dim=192"]
""".strip(),
        encoding="utf-8",
    )


def test_musicgen_preflight_reports_ready_for_editable_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    python_dir = tmp_path / "env"
    python_dir.mkdir()
    python_exe = python_dir / "python.exe"
    python_exe.write_text("", encoding="utf-8")
    (python_dir / "dora.exe").write_text("", encoding="utf-8")

    audiocraft_dir = tmp_path / "audiocraft-micro"
    (audiocraft_dir / "config" / "dset" / "audio").mkdir(parents=True)
    (audiocraft_dir / "audiocraft").mkdir()
    origin = audiocraft_dir / "audiocraft" / "__init__.py"
    origin.write_text("", encoding="utf-8")

    monkeypatch.setattr("macchina_ricordi.musicgen.setup.shutil.which", lambda name: "C:/ffmpeg/bin/ffmpeg.exe")

    def fake_run(command, capture_output, check, text, timeout):  # type: ignore[no-untyped-def]
        payload = {
            "python_executable": str(python_exe),
            "python_version": "3.11.0",
            "modules": {
                "audiocraft": {"available": True, "origin": str(origin)},
                "dora": {"available": True, "origin": str(python_dir / "Lib" / "site-packages" / "dora.py")},
            },
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("macchina_ricordi.musicgen.setup.subprocess.run", fake_run)

    report = run_musicgen_preflight(
        explicit_python=str(python_exe),
        explicit_audiocraft_dir=str(audiocraft_dir),
    )

    assert report.is_ready is True
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["musicgen_python"] == "ok"
    assert statuses["dora_cli"] == "ok"
    assert statuses["ffmpeg_cli"] == "ok"
    assert statuses["audiocraft_import"] == "ok"
    assert statuses["audiocraft_checkout"] == "ok"
    assert statuses["audiocraft_checkout_match"] == "ok"


def test_setup_repo_managed_musicgen_runtime_fails_on_unrelated_dirty_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_musicgen_project(tmp_path)
    checkout_dir = tmp_path / ".audiocraft"
    (checkout_dir / ".git").mkdir(parents=True)

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        if command[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, stdout=" M pyproject.toml\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("macchina_ricordi.musicgen.setup.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="pyproject.toml"):
        setup_repo_managed_musicgen_runtime(explicit_root=tmp_path)
