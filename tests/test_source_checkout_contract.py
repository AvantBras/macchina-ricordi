"""The source-checkout installation contract.

Installing from a source checkout is the only supported install: the workspace commands read the
checkout's ``musicgen/pyproject.toml``. This module pins the behaviour the other way too — without a
checkout, imports, ``--help`` and the audio/analysis commands still work, and any command that needs the
checkout prints actionable clone/install guidance instead of a traceback.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from macchina_ricordi.musicgen.config import (
    REPO_ROOT_ENV,
    WorkspaceMissingError,
    default_recipe_name,
    find_workspace_root,
    get_repo_root,
    load_musicgen_project_config,
    musicgen_default_split_seed,
    musicgen_default_valid_fraction,
    recipe_choices,
)

MINI_PROJECT = """\
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
repo_url = "https://github.com/example/audiocraft"
commit_sha = "0000000000000000000000000000000000000000"

[tool.macchina_ricordi.defaults]
recipe = "my-recipe"
valid_fraction = 0.25
split_seed = 7

[tool.macchina_ricordi.codecs]
encodec_32khz = "//pretrained/facebook/encodec_32khz"

[tool.macchina_ricordi.recipes."my-recipe"]
overrides = ["solver=musicgen/musicgen_base_32khz", "transformer_lm.dim=192"]
"""


def _write_mini_checkout(root: Path) -> Path:
    (root / "musicgen").mkdir(parents=True)
    (root / "musicgen" / "pyproject.toml").write_text(MINI_PROJECT, encoding="utf-8")
    return root


# ── get_repo_root / find_workspace_root ──────────────────────────────────────────────


def test_get_repo_root_uses_checkout_backstop(monkeypatch: pytest.MonkeyPatch) -> None:
    # In the editable dev install used here, __file__.parents[2] is the checkout root.
    real_root = get_repo_root()
    assert (real_root / "musicgen" / "pyproject.toml").is_file()
    # MACCHINA_RICORDI_ROOT overrides it for a non-editable install.
    monkeypatch.setenv(REPO_ROOT_ENV, "/tmp/somewhere-else")
    assert get_repo_root() == Path("/tmp/somewhere-else").resolve()
    # An explicit root always wins, even over the env override.
    monkeypatch.setenv(REPO_ROOT_ENV, "/tmp/ignored-root")
    assert get_repo_root(real_root) == real_root.resolve()


def test_find_workspace_root_returns_none_without_checkout(tmp_path: Path) -> None:
    assert find_workspace_root(tmp_path) is None
    assert find_workspace_root(_write_mini_checkout(tmp_path)) is not None


# ── load_musicgen_project_config: actionable error, no traceback ────────────────────


def test_load_config_raises_actionable_error_without_checkout(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceMissingError) as excinfo:
        load_musicgen_project_config(tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path.resolve()) in message
    for clue in ("git clone", "cd macchina-ricordi", "uv sync --extra dev", REPO_ROOT_ENV):
        assert clue in message, f"guidance should mention {clue!r}"


def test_load_config_still_reads_a_valid_checkout(tmp_path: Path) -> None:
    _write_mini_checkout(tmp_path)
    config = load_musicgen_project_config(tmp_path)
    assert config.default_recipe == "my-recipe"
    assert config.default_valid_fraction == 0.25
    assert config.default_split_seed == 7


# ── graceful default/recipe helpers ─────────────────────────────────────────────────


def test_graceful_defaults_fall_back_without_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))  # a dir with no checkout
    assert musicgen_default_valid_fraction() == 0.1
    assert musicgen_default_split_seed() == 42
    assert default_recipe_name() == "reshape-nano"
    assert recipe_choices() == ("reshape-nano", "micro-unconditional")


def test_graceful_defaults_honor_custom_values_in_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_mini_checkout(tmp_path)
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert musicgen_default_valid_fraction() == 0.25
    assert musicgen_default_split_seed() == 7
    assert default_recipe_name() == "my-recipe"
    assert recipe_choices() == ("my-recipe",)


# ── CLI behaviour without a checkout ────────────────────────────────────────────────


def _run_cli(argv: list[str], checkout_root: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env[REPO_ROOT_ENV] = str(checkout_root)
    return subprocess.run(
        [sys.executable, "-m", "macchina_ricordi.cli", *argv],
        capture_output=True,
        text=True,
        cwd=str(checkout_root),
        env=env,
    )


def test_build_parser_and_imports_succeed_without_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    import macchina_ricordi.cli  # noqa: F401  (imports must not load the workspace)
    from macchina_ricordi.cli import build_parser

    parser = build_parser()
    assert parser.get_default("command") is None  # built without needing a checkout


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["score", "--help"],
        ["listen", "--help"],
        ["segment", "--help"],
        ["data", "prepare", "--help"],
        ["train", "run", "--help"],
        ["model", "list", "--help"],
    ],
)
def test_help_commands_work_without_checkout(argv: list[str], tmp_path: Path) -> None:
    result = _run_cli(argv, tmp_path)
    assert result.returncode == 0, f"{argv} failed without checkout:\n{result.stderr}"
    assert "Traceback" not in result.stderr


def test_data_prepare_help_shows_split_defaults_without_checkout(tmp_path: Path) -> None:
    result = _run_cli(["data", "prepare", "--help"], tmp_path)
    assert result.returncode == 0
    assert "Deterministic split seed (default: 42)" in result.stdout


def test_workspace_command_prints_guidance_without_checkout(tmp_path: Path) -> None:
    result = _run_cli(["runtime", "check"], tmp_path)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "since ignored" not in result.stderr
    for clue in ("musicgen/pyproject.toml", "git clone", "cd macchina-ricordi", "uv sync --extra dev"):
        assert clue in result.stderr, f"guidance should mention {clue!r}"
    assert str(tmp_path.resolve()) in result.stderr


def test_train_command_prints_guidance_without_checkout(tmp_path: Path) -> None:
    result = _run_cli(["train", "run", "--dataset", "x", "--dry-run"], tmp_path)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "git clone" in result.stderr


# ── central workspace-noun guard: actionable error before the body touches a bare install ─────────


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "list"],
        ["data", "audit"],
        ["data", "weights", "list", "--workspace-dir", "x"],
        ["model", "list"],
        ["session", "list"],
        ["tidy", "clean", "--dora-dir", "x"],
        ["runtime", "check"],
        ["gui"],  # must NOT start a server without a checkout
    ],
)
def test_workspace_noun_commands_are_guarded_without_checkout(argv: list[str], tmp_path: Path) -> None:
    result = _run_cli(argv, tmp_path)
    assert result.returncode == 2, f"{argv} should be refused without a checkout, got {result.returncode}"
    assert "Traceback" not in result.stderr
    for clue in ("musicgen/pyproject.toml", "git clone", "uv sync --extra dev", REPO_ROOT_ENV):
        assert clue in result.stderr, f"guidance should mention {clue!r}\n{result.stderr}"


def test_workspace_guard_allows_explicit_repo_root(tmp_path: Path) -> None:
    """`gui --repo-root <valid checkout>` passes the guard and reaches its body."""
    _write_mini_checkout(tmp_path)
    import macchina_ricordi.cli as cli_mod

    args = cli_mod.build_parser().parse_args(["gui", "--repo-root", str(tmp_path), "--port", "0"])
    assert cli_mod.workspace_guard_error(args) is None


def test_workspace_noun_help_still_works_without_checkout(tmp_path: Path) -> None:
    """The guard is after argparse, so --help on any workspace noun / sub-noun is answered, not blocked."""
    for argv in [
        ["gui", "--help"],
        ["play", "--help"],
        ["play", "stream", "--help"],
        ["tidy", "--help"],
        ["data", "list", "--help"],
    ]:
        result = _run_cli(argv, tmp_path)
        assert result.returncode == 0, f"{argv} --help must work without a checkout: {result.stderr}"


# ── import never reads the workspace config; defaults defer to it at call time ─────────────────


def test_import_is_safe_with_malformed_workspace_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Importing config/data/training modules must never read the workspace (malformed config here)."""
    (tmp_path / "musicgen").mkdir(parents=True)
    (tmp_path / "musicgen" / "pyproject.toml").write_text("[tool.macchina_ricordi] not: [valid", encoding="utf-8")
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))

    # Fresh interpreter: assert the import chain tolerates a malformed checkout at import time.
    code = (
        "import macchina_ricordi.data.prepare, macchina_ricordi.data.datasets, "
        "macchina_ricordi.training.plan, macchina_ricordi.cli\n"
        "print('imported')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=dict(os.environ, **{REPO_ROOT_ENV: str(tmp_path)}),
    )
    assert result.returncode == 0, f"import chain failed on malformed config:\n{result.stderr}"
    assert "imported" in result.stdout


def test_published_defaults_are_stable_at_import_and_defer_to_custom_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Module-level published defaults do not read config; runtime functions defer to a custom checkout."""
    from macchina_ricordi.data import prepare as prepare_mod
    from macchina_ricordi.training import plan as plan_mod

    # Imported defaults are the stable published ones, not the configured (custom) values.
    assert prepare_mod.DEFAULT_VALID_FRACTION == 0.1
    assert prepare_mod.DEFAULT_SPLIT_SEED == 42
    assert plan_mod.DEFAULT_TRAINING_PROFILE == "reshape-nano"

    # Point at a custom checkout: the runtime default lookup now honours it, even after import.
    _write_mini_checkout(tmp_path)
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
    assert musicgen_default_valid_fraction() == 0.25
    assert musicgen_default_split_seed() == 7
    assert default_recipe_name() == "my-recipe"


def test_prepare_musicgen_dataset_defers_defaults_to_custom_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """prepare_musicgen_dataset honours the checkout's split defaults at call time, not at import."""
    import macchina_ricordi.data.prepare as prepare_mod

    _write_mini_checkout(tmp_path)
    monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))

    captured: dict[str, object] = {}
    real_split = prepare_mod.split_audio_files

    def fake_split(files, *, valid_fraction, seed):  # type: ignore[no-untyped-def]
        captured["valid_fraction"] = valid_fraction
        captured["seed"] = seed
        return real_split(files, valid_fraction=valid_fraction, seed=seed)

    monkeypatch.setattr(prepare_mod, "split_audio_files", fake_split)
    monkeypatch.setattr(
        prepare_mod.subprocess, "run", lambda command, check=False, cwd=None, env=None, **kw: subprocess.CompletedProcess(command, 0)
    )  # fmt: skip

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    for index in range(4):
        (input_dir / f"clip_{index}.wav").write_bytes(b"fake audio")
    python_exe = tmp_path / "env" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("", encoding="utf-8")

    prepare_mod.prepare_musicgen_dataset(
        input_dir=input_dir,
        workspace_dir=tmp_path / "workspace",
        musicgen_python=str(python_exe),
    )
    # No valid_fraction/seed given → the custom checkout's defaults (0.25, 7) are used, not the
    # import-time published ones (0.1, 42).
    assert captured["valid_fraction"] == 0.25
    assert captured["seed"] == 7
