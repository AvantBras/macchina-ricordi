from __future__ import annotations

from pathlib import Path

from .config import (
    MusicGenProjectConfig,
    bootstrap_audiocraft_dora_dir,
    build_musicgen_subprocess_env,
    get_repo_root,
    load_musicgen_project_config,
    normalize_python_executable_path,
)

__all__ = [
    "MusicGenProjectConfig",
    "bootstrap_audiocraft_dora_dir",
    "build_musicgen_subprocess_env",
    "get_repo_root",
    "load_musicgen_project_config",
    "normalize_python_executable_path",
]


def resolve_musicgen_python(explicit_python: str | None = None) -> str:
    if explicit_python:
        return str(normalize_python_executable_path(explicit_python))
    return str(normalize_python_executable_path(load_musicgen_project_config().musicgen_python))


def require_musicgen_python(explicit_python: str | None = None, *, flag: str = "--musicgen-python") -> str:
    musicgen_python = Path(resolve_musicgen_python(explicit_python)).expanduser()
    if musicgen_python.exists() and musicgen_python.is_file():
        return str(musicgen_python)
    if explicit_python:
        raise FileNotFoundError(
            f"Python executable not found: {musicgen_python}. "
            f"Pass a valid {flag} or run `make setup-musicgen` "
            "to use the repo-managed runtime."
        )
    raise FileNotFoundError(
        f"Repo-managed MusicGen Python executable not found: {musicgen_python}. "
        f"Run `make setup-musicgen` or pass {flag}."
    )


def resolve_audiocraft_dir(explicit_dir: str | None = None) -> Path:
    config = load_musicgen_project_config()
    path = Path(explicit_dir).expanduser().resolve() if explicit_dir else config.audiocraft_dir.resolve()
    config_dir = path / "config" / "dset" / "audio"
    if not path.exists() or not path.is_dir():
        if explicit_dir:
            raise FileNotFoundError(f"AudioCraft checkout does not exist: {path}")
        raise FileNotFoundError(
            f"Repo-managed AudioCraft checkout does not exist: {path}. "
            "Run `make setup-musicgen` or pass --audiocraft-dir."
        )
    if not config_dir.exists():
        if explicit_dir:
            raise FileNotFoundError(f"AudioCraft checkout is missing config/dset/audio: {path}")
        raise FileNotFoundError(
            f"Repo-managed AudioCraft checkout is missing config/dset/audio: {path}. "
            "Run `make setup-musicgen` to recreate it or pass --audiocraft-dir."
        )
    return path
