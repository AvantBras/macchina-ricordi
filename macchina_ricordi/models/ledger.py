"""Read and edit the Git-tracked experiment ledger.

The ledger is deliberately small and human-authored.  Generated model,
generation, sweep, and stream metadata remains the source of machine facts;
ledger entries add the question, status, interpretation, and pointers that
cannot be recovered from those sidecars.

Edits rewrite the whole file from its parsed form, keeping the header comments
and the layout a hand-written entry has. The file is Git-tracked, so Git is its
history; a write is refused unless the new text parses back to exactly the
intended data.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.shared import write_text_atomic

LEDGER_RELATIVE_PATH = Path("research/training/experiment_ledger.toml")
LEDGER_SCHEMA = "macchina_ricordi.experiment_ledger.v1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_STATUSES = {"planned", "running", "paused", "completed", "abandoned"}


@dataclass(frozen=True)
class ExperimentLedgerEntry:
    """One human-authored study annotation."""

    experiment_id: str
    title: str
    status: str
    hypothesis: str
    tags: tuple[str, ...]
    xp_sigs: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    sweep_ids: tuple[str, ...]
    stream_ids: tuple[str, ...]
    generation_run_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    notes_paths: tuple[str, ...]
    verdict: str
    next_action: str
    updated_at: str | None
    annotations: dict[str, Any]
    source_path: str


@dataclass(frozen=True)
class ExperimentLedger:
    """The parsed ledger and non-fatal validation warnings."""

    path: str
    schema: str | None
    entries: tuple[ExperimentLedgerEntry, ...]
    warnings: tuple[str, ...]


def ledger_path(repo_root: Path) -> Path:
    return repo_root / LEDGER_RELATIVE_PATH


def _string(value: object, *, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _resolve_paths(repo_root: Path, values: tuple[str, ...]) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        resolved.append(str(path.resolve()))
    return tuple(resolved)


def _parse_entry(
    raw: object,
    *,
    index: int,
    repo_root: Path,
    source_path: Path,
) -> tuple[ExperimentLedgerEntry | None, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"experiments[{index}] must be a table"]

    experiment_id = _string(raw.get("id"))
    if not experiment_id:
        warnings.append(f"experiments[{index}] is missing id")
    elif not _ID_RE.fullmatch(experiment_id):
        warnings.append(f"experiments[{index}] has invalid id {experiment_id!r}")

    status = _string(raw.get("status"), default="planned")
    if status not in _STATUSES:
        warnings.append(f"{experiment_id or f'experiments[{index}]'} has unknown status {status!r}")

    for field in (
        "tags",
        "xp_sigs",
        "checkpoint_paths",
        "sweep_ids",
        "stream_ids",
        "generation_run_ids",
        "artifact_paths",
        "notes_paths",
    ):
        value = raw.get(field)
        if value is not None and not isinstance(value, list):
            warnings.append(f"{experiment_id or f'experiments[{index}]'}.{field} must be an array")

    annotations = raw.get("annotations", {})
    if not isinstance(annotations, dict):
        warnings.append(f"{experiment_id or f'experiments[{index}]'}.annotations must be a table")
        annotations = {}

    entry = ExperimentLedgerEntry(
        experiment_id=experiment_id,
        title=_string(raw.get("title"), default=experiment_id),
        status=status,
        hypothesis=_string(raw.get("hypothesis")),
        tags=_string_tuple(raw.get("tags")),
        xp_sigs=_string_tuple(raw.get("xp_sigs")),
        checkpoint_paths=_resolve_paths(repo_root, _string_tuple(raw.get("checkpoint_paths"))),
        sweep_ids=_string_tuple(raw.get("sweep_ids")),
        stream_ids=_string_tuple(raw.get("stream_ids")),
        generation_run_ids=_string_tuple(raw.get("generation_run_ids")),
        artifact_paths=_resolve_paths(repo_root, _string_tuple(raw.get("artifact_paths"))),
        notes_paths=_resolve_paths(repo_root, _string_tuple(raw.get("notes_paths"))),
        verdict=_string(raw.get("verdict")),
        next_action=_string(raw.get("next_action")),
        updated_at=_string(raw.get("updated_at")) or None,
        annotations=dict(annotations),
        source_path=str(source_path.resolve()),
    )
    return entry, warnings


def load_experiment_ledger(repo_root: Path) -> ExperimentLedger:
    """Load the ledger, returning warnings instead of failing catalog builds."""
    path = ledger_path(repo_root)
    if not path.exists():
        return ExperimentLedger(path=str(path.resolve()), schema=None, entries=(), warnings=())

    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return ExperimentLedger(
            path=str(path.resolve()),
            schema=None,
            entries=(),
            warnings=(f"could not read ledger {path}: {exc}",),
        )

    schema = payload.get("schema") if isinstance(payload.get("schema"), str) else None
    warnings: list[str] = []
    if schema != LEDGER_SCHEMA:
        warnings.append(f"ledger schema is {schema!r}; expected {LEDGER_SCHEMA!r}")

    raw_entries = payload.get("experiments", [])
    if not isinstance(raw_entries, list):
        warnings.append("ledger experiments must be an array of tables")
        raw_entries = []

    entries: list[ExperimentLedgerEntry] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        entry, entry_warnings = _parse_entry(raw, index=index, repo_root=repo_root, source_path=path)
        warnings.extend(entry_warnings)
        if entry is None or not entry.experiment_id:
            continue
        if not _ID_RE.fullmatch(entry.experiment_id):
            continue
        if entry.experiment_id in seen_ids:
            warnings.append(f"duplicate experiment id {entry.experiment_id!r}")
            continue
        seen_ids.add(entry.experiment_id)
        entries.append(entry)

    return ExperimentLedger(
        path=str(path.resolve()),
        schema=schema,
        entries=tuple(entries),
        warnings=tuple(warnings),
    )


# ── editing ───────────────────────────────────────────────────────────

_ENTRY_KEY_ORDER = (
    "id",
    "title",
    "status",
    "hypothesis",
    "tags",
    "xp_sigs",
    "checkpoint_paths",
    "sweep_ids",
    "stream_ids",
    "generation_run_ids",
    "notes_paths",
    "artifact_paths",
    "verdict",
    "next_action",
    "updated_at",
)
_TEXT_FIELDS = ("title", "status", "hypothesis", "verdict", "next_action", "updated_at")
_LIST_FIELDS = (
    "tags",
    "xp_sigs",
    "checkpoint_paths",
    "sweep_ids",
    "stream_ids",
    "generation_run_ids",
    "notes_paths",
    "artifact_paths",
)
_INLINE_ARRAY_WIDTH = 88


class LedgerEditError(ValueError):
    """An edit that would leave the ledger invalid or ambiguous."""


@dataclass(frozen=True)
class LedgerWriteResult:
    path: str
    experiment_id: str
    created: bool
    changed_fields: tuple[str, ...]


def save_experiment(
    repo_root: Path,
    experiment_id: str,
    fields: Mapping[str, Any],
    *,
    create: bool = False,
    today: str | None = None,
) -> LedgerWriteResult:
    """Create or update one ``[[experiments]]`` entry and rewrite the ledger.

    Text and list fields in ``fields`` replace the stored value; an empty string or
    empty list removes the key. Fields not mentioned are kept as they are, including
    ones this module does not know. ``updated_at`` is set to today unless given.
    """
    if not _ID_RE.fullmatch(experiment_id):
        raise LedgerEditError(f"invalid experiment id {experiment_id!r}: use lowercase letters, digits, - and _")
    unknown = sorted(set(fields) - set(_TEXT_FIELDS) - set(_LIST_FIELDS))
    if unknown:
        raise LedgerEditError(f"cannot edit {', '.join(unknown)} from here")

    path = ledger_path(repo_root)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            payload = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise LedgerEditError(f"{path} does not parse, fix it by hand first: {exc}") from exc
    else:
        text = ""
        payload = {"schema": LEDGER_SCHEMA, "experiments": []}

    experiments = payload.setdefault("experiments", [])
    if not isinstance(experiments, list) or not all(isinstance(item, dict) for item in experiments):
        raise LedgerEditError(f"{path}: experiments must be an array of tables")
    matches = [item for item in experiments if item.get("id") == experiment_id]
    if len(matches) > 1:
        raise LedgerEditError(f"{path} has {len(matches)} entries with id {experiment_id!r}; fix it by hand first")
    if create and matches:
        raise LedgerEditError(f"experiment {experiment_id!r} already exists")
    if not create and not matches:
        raise LedgerEditError(f"no experiment {experiment_id!r} in {path}")

    entry: dict[str, Any] = matches[0] if matches else {"id": experiment_id, "status": "planned"}
    before = json.dumps(entry, sort_keys=True, default=str)
    changed: list[str] = []
    for key, value in fields.items():
        cleaned = _clean_field(key, value)
        if cleaned in ("", []):
            if key in entry:
                del entry[key]
                changed.append(key)
        elif entry.get(key) != cleaned:
            entry[key] = cleaned
            changed.append(key)
    if entry.get("status", "planned") not in _STATUSES:
        raise LedgerEditError(f"status must be one of {', '.join(sorted(_STATUSES))}")
    if "updated_at" not in fields and (create or changed):
        entry["updated_at"] = today or datetime.now(UTC).date().isoformat()
    if create:
        experiments.append(entry)
    elif json.dumps(entry, sort_keys=True, default=str) == before:
        return LedgerWriteResult(str(path), experiment_id, False, ())

    rendered = render_ledger(payload, header=_header_comments(text))
    if tomllib.loads(rendered) != payload:
        raise LedgerEditError("the rewritten ledger would not read back identically; nothing was written")
    write_text_atomic(path, rendered)
    return LedgerWriteResult(str(path), experiment_id, create, tuple(changed))


def _clean_field(key: str, value: Any) -> str | list[str]:
    if key in _LIST_FIELDS:
        if isinstance(value, str):
            value = [part for part in re.split(r"[,\n]", value)]
        if not isinstance(value, list | tuple):
            raise LedgerEditError(f"{key} must be a list of strings")
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise LedgerEditError(f"{key} must be a list of strings")
            item = item.strip()
            if item and item not in items:
                items.append(item)
        return items
    if value is None:
        return ""
    if not isinstance(value, str):
        raise LedgerEditError(f"{key} must be text")
    return value.strip()


def _header_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("#") or (not line.strip() and lines):
            lines.append(line)
            continue
        break
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def render_ledger(payload: Mapping[str, Any], *, header: str = "") -> str:
    """Serialize a ledger payload the way a hand-written ledger is laid out."""
    blocks: list[str] = []
    if header:
        blocks.append(header)
    top = [_toml_line(key, value) for key, value in payload.items() if key != "experiments"]
    if top:
        blocks.append("\n".join(top))
    for entry in payload.get("experiments", []):
        blocks.append(_render_entry(entry))
    return "\n\n".join(blocks) + "\n"


def _render_entry(entry: Mapping[str, Any]) -> str:
    known = [key for key in _ENTRY_KEY_ORDER if key in entry]
    extra = [key for key in entry if key not in _ENTRY_KEY_ORDER]
    scalars = [key for key in known + extra if not isinstance(entry[key], dict)]
    tables = [key for key in known + extra if isinstance(entry[key], dict)]
    parts = ["[[experiments]]\n" + "\n".join(_toml_line(key, entry[key]) for key in scalars)]
    for key in tables:
        body = "\n".join(_toml_line(k, v) for k, v in entry[key].items())
        parts.append(f"[experiments.{_toml_key(key)}]" + (f"\n{body}" if body else ""))
    return "\n\n".join(parts)


def _toml_line(key: str, value: Any) -> str:
    rendered = _toml_value(value)
    line = f"{_toml_key(key)} = {rendered}"
    if isinstance(value, list) and len(line) > _INLINE_ARRAY_WIDTH:
        items = "".join(f"  {_toml_value(item)},\n" for item in value)
        return f"{_toml_key(key)} = [\n{items}]"
    return line


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise LedgerEditError(f"cannot write a {type(value).__name__} value to the ledger")
