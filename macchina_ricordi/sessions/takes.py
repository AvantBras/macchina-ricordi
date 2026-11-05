"""Every playable take of some sessions, with the numbers they recorded and a group: what A/B and rescore read.

A source is a session folder or id (``resolve_session_folders``). Takes come from ``Session.takes()`` only, so a
sweep, a stream, a generation, a duet and a variations session are all read by the same adapters as
``session show``. A duet's voice stems and a variations session's codec round trip are skipped: they are parts of
another take, not takes to judge or score on their own.

``group_by`` decides what a take's group is: ``model_xp`` (the model that made it, the default), ``session``, or
anything else (``bucket:<metric>``, which ``judging.ab`` relabels into terciles) meaning the session. A sweep's
takes are grouped by an ``index.csv`` column instead (``group_column``, default ``sweep_id``). ``condition`` groups
every kind by the model plus the knobs that shape the audio, seeds excepted: takes in one condition differ only by
chance, which is what testing a metric against the ear needs. ``condition:<knob>`` leaves that one knob out, so a
group holds takes that differ only by chance and by that knob; ``condition:model`` leaves the model out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .read import _sweep_rows, facts_of, load_session
from .record import Session, Take
from .store import resolve_session_folders

GROUP_BY_MODEL = "model_xp"
GROUP_BY_SESSION = "session"
GROUP_BY_CONDITION = "condition"
# controls that do not change what a take sounds like (or, like the seed, are the chance a condition varies over)
_NOT_CONDITION = frozenset(
    {
        "seed",
        "grid",
        "command",
        "pin_target",
        "model_xp",
        "max_buffer_ahead_seconds",
        "resume_buffer_ahead_seconds",
        "throttle_enabled",
        "round_delay_seconds",
    }
)
UNPLAYED_ROLES = ("voice", "reference")


@dataclass(frozen=True)
class PlayableTake:
    path: str
    group: str
    metrics: dict[str, float] = field(default_factory=dict)
    label: str = ""
    session_id: str = ""
    kind: str = ""
    round_idx: int | None = None
    position: int | None = None
    """For a stream round: its place among the session's playable rounds, scored or not."""
    prompt_duration_seconds: float = 0.0
    model: str | None = None
    controls: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    """What the session recorded, as recorded (``metrics`` keeps only the numbers)."""
    duration_seconds: float | None = None
    model_name: str = ""
    """The model as ``model_name`` names it; empty when the take does not say."""


def playable_takes(
    sources: Iterable[str | Path],
    *,
    group_column: str = "sweep_id",
    group_by: str = GROUP_BY_MODEL,
    scored_only: bool = False,
) -> list[PlayableTake]:
    """``scored_only`` drops stream rounds logged without a score (other kinds keep every take)."""
    found: list[PlayableTake] = []
    for folder in resolve_session_folders(sources):
        try:
            # streams and sweeps keep the path spelling they were given, as the preference store recorded them
            session = load_session(folder, resolve=False)
            if session.kind not in ("stream", "sweep"):
                session = load_session(folder)
        except (LookupError, OSError, ValueError):
            continue
        found += _session_takes(session, group_column=group_column, group_by=group_by, scored_only=scored_only)
    return found


def _session_takes(session: Session, *, group_column: str, group_by: str, scored_only: bool) -> list[PlayableTake]:
    facts = facts_of(session)
    if session.kind == "stream":
        # a stream named by its manifest keeps its run id; a live one, without a manifest yet, its folder name
        session_id = str(facts.get("run_id") or session.id) if session.native else session.folder.name
    else:
        session_id = session.id
    takes: list[PlayableTake] = []
    position = 0
    # group_column may name any index.csv column, text ones included, which takes do not keep
    rows = _sweep_rows(session.folder) if session.kind == "sweep" else []
    for take in session.takes():
        if take.role in UNPLAYED_ROLES or take.audio is None or take.error or not take.audio.exists():
            continue
        if session.kind == "stream":
            take_position = position
            position += 1
            if scored_only and not take.scores:
                continue
            takes.append(_stream_take(take, session_id, group_by, take_position))
        elif session.kind == "sweep":
            row = rows[take.index] if take.index < len(rows) else {}
            takes.append(_sweep_take(take, session, facts, group_column, row, group_by))
        else:
            group = (take.sig or session_id) if group_by == GROUP_BY_MODEL else session_id
            name = model_name(take.sig or take.model, facts.get("checkpoint_path"))
            if _by_condition(group_by):
                group = condition_of(name or session_id, take.controls, free=_free(group_by))
            takes.append(
                PlayableTake(
                    path=str(take.audio),
                    group=group,
                    metrics=numeric(take.scores),
                    label=f"{session_id} {take.label}",
                    session_id=session_id,
                    kind=session.kind,
                    model=take.model,
                    controls=dict(take.controls),
                    scores=dict(take.scores),
                    duration_seconds=take.duration_seconds,
                    model_name=name,
                )
            )
    return takes


def _stream_take(take: Take, session_id: str, group_by: str, position: int) -> PlayableTake:
    model = Path(str(take.model)).name if take.model else ""
    round_idx = _round_idx(take.label)
    group = (model or session_id) if group_by == GROUP_BY_MODEL else session_id
    if _by_condition(group_by):
        group = condition_of(model or session_id, take.controls, free=_free(group_by))
    return PlayableTake(
        path=str(take.audio),
        group=group,
        metrics=numeric(take.scores),
        label=f"{session_id} round {round_idx}" if round_idx is not None else session_id,
        session_id=session_id,
        kind="stream",
        round_idx=round_idx,
        position=position,
        prompt_duration_seconds=_nonnegative(take.prompt_duration_seconds),
        model=take.model,
        controls=dict(take.controls),
        scores=dict(take.scores),
        duration_seconds=take.duration_seconds,
        model_name=model,
    )


def _sweep_take(
    take: Take, session: Session, facts: Mapping[str, Any], column: str, row: Mapping[str, str], group_by: str
) -> PlayableTake:
    combo = take.label.split(" #", 1)[0] if take.label.startswith("combo") else ""
    columns: dict[str, Any] = {
        "sweep_id": facts.get("run_id"),
        "profile": facts.get("profile"),
        "model_id": facts.get("model_id"),
        "combo_id": combo,
        "xp_path": take.model,
        **take.controls,
        **take.scores,
    }
    group = str(row.get(column) or columns.get(column) or facts.get("profile") or session.folder.name).strip()
    name = model_name(take.model or facts.get("model_id"), row.get("checkpoint_path"))
    if _by_condition(group_by):
        group = condition_of(name or session.folder.name, take.controls, free=_free(group_by))
    temperature = take.controls.get("temperature", "")
    return PlayableTake(
        path=str(take.audio),
        group=group or session.folder.name,
        metrics=numeric({**take.controls, **take.scores}),
        label=" ".join(bit for bit in (combo, f"temp={temperature}") if bit.strip()),
        session_id=session.id,
        kind="sweep",
        model=take.model,
        controls=dict(take.controls),
        scores=dict(take.scores),
        duration_seconds=take.duration_seconds,
        model_name=name,
    )


def model_name(model: Any, checkpoint: Any = None) -> str:
    """A model by its folder name, ``@<epoch>`` added when the take names a numbered checkpoint: ``c5140082@20``."""
    from ..musicgen.backend import CHECKPOINT_EPOCH_PATTERN

    if not model:
        return ""
    epoch = CHECKPOINT_EPOCH_PATTERN.match(Path(str(checkpoint or "")).name)
    name = Path(str(model)).name
    return f"{name}@{epoch.group(1)}" if epoch else name


def condition_of(model: Any, controls: Mapping[str, Any], *, free: str = "") -> str:
    """``<model> knob=value …``: the model by its folder name (unless ``free`` is ``model``), then every
    audio-shaping scalar knob but ``free``."""
    knobs = (
        f"{key}={knob_text(value)}"
        for key, value in sorted(controls.items())
        if key not in _NOT_CONDITION and key != free and not isinstance(value, dict | list)
    )
    return " ".join(knobs if free == "model" else (Path(str(model)).name, *knobs))


def knob_text(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _by_condition(group_by: str) -> bool:
    return group_by == GROUP_BY_CONDITION or group_by.startswith(f"{GROUP_BY_CONDITION}:")


def _free(group_by: str) -> str:
    return group_by.partition(":")[2]


def numeric(values: Mapping[str, Any]) -> dict[str, float]:
    """The finite numbers among ``values`` (booleans are not numbers here)."""
    out: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        number = float(value)
        if math.isfinite(number):
            out[key] = number
    return out


def _round_idx(label: str) -> int | None:
    parts = label.split()
    if len(parts) >= 2 and parts[0] == "round" and parts[1].isdigit():
        return int(parts[1])
    return None


def _nonnegative(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0
