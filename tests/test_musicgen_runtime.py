from __future__ import annotations

import os
from pathlib import Path

from macchina_ricordi.musicgen.runtime import (
    load_musicgen_project_config,
    resolve_musicgen_python,
)


def _write_musicgen_project(repo_root: Path) -> None:
    musicgen_dir = repo_root / "musicgen"
    musicgen_dir.mkdir(parents=True)
    (musicgen_dir / "pyproject.toml").write_text(
        """
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
commit_sha = "1ca32f28d2d6c223b4e2335042ac558c308f2ea2"

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


def test_load_musicgen_project_config_reads_repo_local_defaults(tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)

    config = load_musicgen_project_config(tmp_path)
    expected_python = (
        tmp_path / ".venv-musicgen" / "Scripts" / "python.exe"
        if os.name == "nt"
        else tmp_path / ".venv-musicgen" / "bin" / "python"
    )

    assert config.project_dir == tmp_path / "musicgen"
    assert config.env_dir == tmp_path / ".venv-musicgen"
    assert config.musicgen_python == expected_python
    assert config.audiocraft_dir == tmp_path / ".audiocraft"
    assert config.python_request == "3.11"
    assert config.audiocraft_repo_url == "https://github.com/aaronabebe/audiocraft"
    assert config.default_recipe == "micro-unconditional"
    assert config.recipe_choices == ("micro-unconditional", "reshape-nano")
    assert config.recipes["micro-unconditional"].overrides == (
        "solver=musicgen/musicgen_base_32khz",
        "dataset.segment_duration=15",
    )


def test_resolve_musicgen_python_defaults_to_repo_managed_runtime(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    config = load_musicgen_project_config(tmp_path)
    monkeypatch.setattr("macchina_ricordi.musicgen.runtime.load_musicgen_project_config", lambda: config)

    assert resolve_musicgen_python() == str(config.musicgen_python)
