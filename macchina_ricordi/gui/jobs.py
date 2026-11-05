"""Detached jobs: a chain of CLI commands run through ``/bin/sh`` in a session of its own.

A job outlives the page and the server. It is a log file and a ``<id>_status.json`` record in its
jobs folder; the shell appends an exit marker to the log, so its state is read back from the log alone.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from macchina_ricordi.musicgen.shared import read_json
from macchina_ricordi.training.jobs import pid_alive

LOG_TAIL_LINES = 60


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def relative(path: Path | str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    # ffmpeg/mpg123 chatter about ID3 tags is noise here
    kept = [line for line in text.splitlines() if "libmpg123" not in line and "Xing stream" not in line]
    return kept[-lines:]


@dataclass
class DetachedJobs:
    repo_root: Path
    jobs_dir: str
    schema: str
    exit_marker: str
    _children: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.repo_root / self.jobs_dir

    def new_id(self, kind: str) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        job_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{kind}"
        while (self.path / f"{job_id}_status.json").exists():
            job_id += "_"
        return job_id

    def launch(
        self, kind: str, title: str, commands: list[list[str]], meta: dict[str, Any], job_id: str | None = None
    ) -> dict[str, Any]:
        job_id = job_id or self.new_id(kind)
        log_path = self.path / f"{job_id}.log"
        script = " && ".join(shlex.join(command) for command in commands)
        shell = f'({script}); code=$?; echo "{self.exit_marker} $code]"; exit $code'
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                ["/bin/sh", "-c", shell],
                cwd=self.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        record = {
            "schema": self.schema,
            "id": job_id,
            "kind": kind,
            "title": title,
            "commands": [shlex.join(command) for command in commands],
            "pid": process.pid,
            "created_at": now_iso(),
            "log": relative(log_path, self.repo_root),
            **meta,
        }
        (self.path / f"{job_id}_status.json").write_text(json.dumps(record, indent=2) + "\n")
        self._children[job_id] = process
        # reap it so it does not linger as a zombie that still looks alive
        threading.Thread(target=process.wait, daemon=True).start()
        return record

    def child(self, job_id: str) -> subprocess.Popen[bytes] | None:
        """The process this server started for ``job_id``; ``None`` when an earlier server started it."""
        return self._children.get(job_id)

    def records(self, pattern: str = "*_status.json") -> list[tuple[Path, dict[str, Any]]]:
        """Newest first."""
        if not self.path.is_dir():
            return []
        found = []
        for path in sorted(self.path.glob(pattern), reverse=True):
            record = read_json(path)
            if record and record.get("schema") == self.schema:
                found.append((path, record))
        return found

    def record(self, job_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[\w-]+", job_id):
            return None
        record = read_json(self.path / f"{job_id}_status.json")
        return record if record and record.get("schema") == self.schema else None

    def summarize(self, record: dict[str, Any]) -> dict[str, Any]:
        log = tail(self.repo_root / str(record.get("log")), 400)
        exit_code = None
        for line in reversed(log):
            if line.startswith(self.exit_marker):
                exit_code = int(line[len(self.exit_marker) :].strip(" ]") or 1)
                break
        alive = pid_alive(record.get("pid")) and exit_code is None
        state = (
            "running" if alive else "completed" if exit_code == 0 else "failed" if exit_code is not None else "exited"
        )
        progress = next((line for line in reversed(log) if line.startswith("progress: ")), None)
        return {
            "id": record["id"],
            "kind": record.get("kind"),
            "title": record.get("title"),
            "state": state,
            "active": alive,
            "exit_code": exit_code,
            "progress": progress.removeprefix("progress: ") if progress else None,
            "created_at": record.get("created_at"),
            "commands": record.get("commands", []),
            "log_path": record.get("log"),
        }

    def log(self, record: dict[str, Any]) -> list[str]:
        return tail(self.repo_root / str(record.get("log")))
