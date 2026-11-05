from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import bootstrap_audiocraft_dora_dir, load_musicgen_project_config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    """The file's JSON, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_text_atomic(path: Path, text: str) -> None:
    """Readers see the old file or the new one, never half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TaskSidecar:
    """
    Context manager to handle standard 'status' sidecar reporting.
    Writes 'running' on enter, and 'completed' or 'failed' (with traceback) on exit.
    """

    def __init__(self, path: Path, **initial_payload: Any) -> None:
        self.path = path
        self.payload = dict(initial_payload)

    def update(self, **kwargs: Any) -> None:
        """Update the internal payload. Does not write to disk immediately."""
        self.payload.update(kwargs)

    def write(self) -> None:
        """Persist the current payload to the sidecar file."""
        write_json(self.path, self.payload)

    def __enter__(self) -> TaskSidecar:
        self.payload.update(
            {
                "status": "running",
                "started_at": utc_now_iso(),
            }
        )
        self.write()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.payload["finished_at"] = utc_now_iso()
        if exc_type:
            self.payload.update(
                {
                    "status": "failed",
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    "traceback": "".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
                }
            )
        elif self.payload.get("status") == "running":
            self.payload["status"] = "completed"
        self.write()


def format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def dora_candidates(musicgen_python: Path) -> tuple[Path, ...]:
    script_dir = musicgen_python.parent
    return (
        script_dir / "dora.exe",
        script_dir / "dora-script.py",
        script_dir / "dora",
        script_dir / "Scripts" / "dora.exe",
        script_dir / "Scripts" / "dora-script.py",
        script_dir / "Scripts" / "dora",
    )


def resolve_dora_command(musicgen_python: str) -> list[str]:
    python_path = Path(musicgen_python).expanduser()
    for candidate in dora_candidates(python_path):
        if candidate.exists():
            if candidate.suffix == ".py":
                return [musicgen_python, str(candidate)]
            return [str(candidate)]
    return ["dora"]


def run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=True,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


__all__ = [
    "bootstrap_audiocraft_dora_dir",
    "dora_candidates",
    "format_command",
    "load_musicgen_project_config",
    "read_json",
    "resolve_dora_command",
    "run_checked",
    "sha256_file",
    "utc_now_iso",
    "write_json",
    "write_text_atomic",
]
