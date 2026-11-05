"""The session record: what one ``play`` command made, as a list of takes.

A session folder holds ``session.json`` (this record's header) and its takes. Generate, continue, sweep, duet
and variations list theirs in ``takes.jsonl``; a stream's takes are the rounds of its ``performance.jsonl``,
which it appends to live. The producer's own detail (``run.json``, ``index.csv``, a stream's ``events.jsonl``)
stays beside them. See docs/sessions.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models.record import xp_sig_from_path

SESSION_SCHEMA = "macchina_ricordi.session.v1"
SESSION_FILENAME = "session.json"
TAKES_FILENAME = "takes.jsonl"
KINDS = ("generate", "continue", "sweep", "stream", "duet", "variations")
# Kinds `play replay` can make again byte for byte. A sweep's takes come from the sweep helper (fp32, no CUDA
# graphs) while replay runs the generate engine (fp16, fastgen), so their bytes would differ.
REPLAYABLE_KINDS = ("generate", "continue")


class SessionNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Take:
    """One clip a session made. ``index`` counts from 0, like a stream's rounds and a run's samples."""

    index: int
    audio: Path | None
    label: str = ""
    role: str = "take"  # take · mix · voice · reference
    model: str | None = None
    seed: int | None = None
    duration_seconds: float | None = None
    controls: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    replay: Path | None = None  # the metadata `play replay` regenerates this take from
    error: str | None = None
    prompt_duration_seconds: float | None = None  # a continuation's real prompt, after the horizon guard

    @property
    def sig(self) -> str | None:
        return xp_sig_from_path(self.model)

    def to_json(self, base: Path | None = None) -> dict[str, Any]:
        return {
            "take": self.index,
            "audio": _relative(self.audio, base),
            "label": self.label,
            "role": self.role,
            "model": self.model,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "controls": self.controls,
            "scores": self.scores,
            "parent": self.parent,
            "replay": _relative(self.replay, base),
            "error": self.error,
            "prompt_duration_seconds": self.prompt_duration_seconds,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any], base: Path) -> Take:
        return cls(
            index=int(payload.get("take", 0)),
            audio=_absolute(payload.get("audio"), base),
            label=str(payload.get("label") or ""),
            role=str(payload.get("role") or "take"),
            model=payload.get("model"),
            seed=_optional_int(payload.get("seed")),
            duration_seconds=_optional_float(payload.get("duration_seconds")),
            controls=dict(payload.get("controls") or {}),
            scores=dict(payload.get("scores") or {}),
            parent=payload.get("parent"),
            replay=_absolute(payload.get("replay"), base),
            error=payload.get("error"),
            prompt_duration_seconds=_optional_float(payload.get("prompt_duration_seconds")),
        )


@dataclass(frozen=True)
class Session:
    """One output of ``play``: a kind, the models it used, its starting controls and its takes.

    ``sealed`` says the folder has a ``session.json``; sessions made before October 2026 are read from their
    producer's own files (run.json, sweep_plan.json, manifest.json) by the adapters in ``read.py``.
    """

    id: str
    kind: str
    path: Path
    created_at: str | None = None
    finished_at: str | None = None
    label: str | None = None
    models: tuple[str, ...] = ()
    engine: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    take_count: int = 0
    takes_log: str | None = None
    native: str | None = None
    sealed: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    """What the producer recorded that is particular to its kind: a run id, a generation's prefix and model label,
    a sweep's recipe and grid size, a stream's stop reason. Listings (the catalog) read these instead of reopening
    the producer's files."""

    @property
    def sigs(self) -> tuple[str, ...]:
        found: list[str] = []
        for model in self.models:
            sig = xp_sig_from_path(model)
            if sig and sig not in found:
                found.append(sig)
        return tuple(found)

    @property
    def folder(self) -> Path:
        return self.path if self.path.is_dir() else self.path.parent

    def takes(self) -> tuple[Take, ...]:
        from .read import load_takes

        return load_takes(self)

    def take(self, index: int) -> Take:
        for take in self.takes():
            if take.index == index:
                return take
        raise SessionNotFound(f"{self.id} has no take {index} (takes: {self.take_count})")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SESSION_SCHEMA,
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "label": self.label,
            "models": list(self.models),
            "engine": self.engine,
            "controls": self.controls,
            "parent": self.parent,
            "take_count": self.take_count,
            "takes_log": self.takes_log,
            "native": self.native,
            "facts": self.facts,
        }

    def summary(self) -> dict[str, Any]:
        return {**self.to_json(), "path": str(self.path), "sigs": list(self.sigs), "sealed": self.sealed}

    @classmethod
    def from_json(cls, payload: dict[str, Any], folder: Path) -> Session:
        return cls(
            id=str(payload.get("id") or folder.name),
            kind=str(payload.get("kind") or "generate"),
            path=folder,
            created_at=payload.get("created_at"),
            finished_at=payload.get("finished_at"),
            label=payload.get("label"),
            models=tuple(str(model) for model in payload.get("models") or () if model),
            engine=dict(payload.get("engine") or {}),
            controls=dict(payload.get("controls") or {}),
            parent=payload.get("parent"),
            take_count=int(payload.get("take_count") or 0),
            takes_log=payload.get("takes_log"),
            native=payload.get("native"),
            sealed=True,
            facts=dict(payload.get("facts") or {}),
        )


def iso_from_run_id(run_id: object) -> str | None:
    """``20261005T141908Z`` → ``2026-10-05T14:19:08+00:00``; anything else → None."""
    try:
        return datetime.strptime(str(run_id), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _relative(path: Path | None, base: Path | None) -> str | None:
    if path is None:
        return None
    if base is not None:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def _absolute(value: object, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _optional_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
