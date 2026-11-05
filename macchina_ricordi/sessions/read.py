"""Read any session folder, sealed or not.

A sealed folder has a ``session.json``. Everything made before October 2026, and a stream that is still playing,
is read from its producer's own files: a generation's ``run.json``, a sweep's ``sweep_plan.json`` and
``index.csv``, a stream's ``manifest.json`` and ``performance.jsonl``, a duet's or variations' ``manifest.json``.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .record import (
    SESSION_FILENAME,
    TAKES_FILENAME,
    Session,
    SessionNotFound,
    Take,
    _optional_float,
    _optional_int,
    count_lines,
    iso_from_run_id,
    read_json,
    read_jsonl,
)

SWEEP_CONTROL_COLUMNS = (
    "mode",
    "prompt_audio_path",
    "prompt_start_seconds",
    "prompt_duration_seconds",
    "duration_seconds",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "cfg_coef",
    "extend_stride",
    "use_sampling",
)
_SWEEP_TEXT_COLUMNS = (
    "sweep_id",
    "combo_id",
    "combo_index",
    "sample_index",
    "path",
    "error",
    "model_id",
    "profile",
    "xp_path",
    "checkpoint_path",
    "num_samples",
    "listening_report_path",
    "listening_markdown_path",
    "listening_summary",
    "listening_labels",
    "listening_error",
    "score_error",
)
_GENERATION_CONTROLS = (
    "mode",
    "duration_seconds",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "cfg_coef",
    "extend_stride",
    "use_sampling",
    "prompt_audio_path",
    "prompt_start_seconds",
    "prompt_duration_seconds",
)
_SAMPLING_KEYS = ("temperature", "top_k", "top_p", "min_p")


def load_session(path: str | Path, *, resolve: bool = True) -> Session:
    """The session at ``path``: a session folder, its ``session.json``, or a generation's metadata JSON.

    ``resolve=False`` keeps ``path`` as spelled, so take paths come back spelled from it too (the preference store
    keys stream rounds by those strings).
    """
    target = Path(path).expanduser()
    target = target.resolve() if resolve else target
    if target.is_file():
        if target.name == SESSION_FILENAME:
            target = target.parent
        elif target.suffix == ".json" and _is_generation_metadata(read_json(target)):
            return _generation_session(target)
        else:
            raise SessionNotFound(f"{target} is not a session")
    if not target.is_dir():
        raise SessionNotFound(f"no session folder at {target}")
    record = target / SESSION_FILENAME
    if record.is_file():
        payload = read_json(record)
        if payload:
            return Session.from_json(payload, target)
    return describe_native(target)


def describe_native(folder: Path) -> Session:
    """Read a session from its producer's own files, ignoring any ``session.json``."""
    if (folder / "run.json").is_file():
        return _generation_session(folder / "run.json")
    if (folder / "sweep_plan.json").is_file() or (folder / "index.csv").is_file():
        return _sweep_session(folder)
    manifest = read_json(folder / "manifest.json")
    schema = str(manifest.get("schema") or "")
    if (folder / "performance.jsonl").is_file() or schema.startswith("macchina_ricordi.stream_session"):
        return _stream_session(folder, manifest)
    if schema.startswith("macchina_ricordi.duet_session"):
        return _duet_session(folder, manifest)
    if schema.startswith("macchina_ricordi.variations"):
        return _variations_session(folder, manifest)
    raise SessionNotFound(f"{folder} is not a session folder")


def load_takes(session: Session) -> tuple[Take, ...]:
    folder = session.folder
    if session.takes_log == TAKES_FILENAME and (folder / TAKES_FILENAME).is_file():
        return tuple(Take.from_json(row, folder) for row in read_jsonl(folder / TAKES_FILENAME))
    if session.kind == "stream":
        return _stream_takes(folder)
    if session.kind in ("generate", "continue"):
        metadata = session.path if session.path.is_file() else folder / "run.json"
        return _generation_takes(metadata, read_json(metadata))
    if session.kind == "sweep":
        return _sweep_takes(folder)
    if session.kind == "duet":
        return _duet_takes(folder, read_json(folder / "manifest.json"))
    if session.kind == "variations":
        return _variations_takes(folder, read_json(folder / "manifest.json"))
    return ()


# ── generate / continue: run.json, or a pre-2026-05 flat ``<prefix>_<run_id>.json`` ──


def _is_generation_metadata(payload: dict[str, Any]) -> bool:
    return {"run_id", "request", "files"} <= set(payload)


def _generation_kind(request: dict[str, Any]) -> str:
    return "continue" if str(request.get("mode") or "") in ("continuation", "continue") else "generate"


def _generation_source(payload: dict[str, Any]) -> dict[str, Any]:
    source = (payload.get("generation") or {}).get("source")
    return source if isinstance(source, dict) else {}


def _source_model(source: dict[str, Any], fallback: object = None) -> str | None:
    value = source.get("xp_path") or source.get("checkpoint_path") or source.get("model_id") or fallback
    return str(value) if value else None


def _generation_session(metadata_path: Path) -> Session:
    payload = read_json(metadata_path)
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    source = _generation_source(payload)
    model = _source_model(source, payload.get("model_id"))
    in_folder = metadata_path.name == "run.json"
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    prefix = output.get("prefix")
    run_id = payload.get("run_id")
    model_block = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    facts = {
        "run_id": str(run_id) if run_id else None,
        "prefix": _generation_prefix(metadata_path, output, run_id),
        "model_label": model_block.get("label"),
        "model_workspace": model_block.get("workspace"),
        "xp_path": source.get("xp_path"),
        "checkpoint_path": source.get("checkpoint_path"),
        "num_samples": request.get("num_samples", len(payload.get("files") or ())),
    }
    return Session(
        id=metadata_path.parent.name if in_folder else metadata_path.stem,
        kind=_generation_kind(request),
        path=metadata_path.parent if in_folder else metadata_path,
        created_at=iso_from_run_id(payload.get("run_id")),
        label=str(prefix) if prefix and prefix != "musicgen" else None,
        models=(model,) if model else (),
        engine=_engine(payload),
        controls={key: request[key] for key in _GENERATION_CONTROLS if key in request},
        parent=request.get("prompt_audio_path"),
        take_count=len(payload.get("files") or ()),
        native=metadata_path.name,
        facts=facts,
    )


def _generation_prefix(metadata_path: Path, output: dict[str, Any], run_id: object) -> str:
    """The name a run was saved under: its ``output.prefix``, else its file or folder name without the run id."""
    if isinstance(output.get("prefix"), str):
        return str(output["prefix"])
    if metadata_path.name == "run.json":
        name = metadata_path.parent.name
        return name[: -(len(str(run_id)) + 1)] if run_id and name.endswith(f"_{run_id}") else name
    name = metadata_path.name
    return name[: -(len(str(run_id)) + 6)] if run_id and name.endswith(f"_{run_id}.json") else name


def _engine(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in ("backend", "model_id", "device", "sample_rate") if key in payload}


def _generation_takes(metadata_path: Path, payload: dict[str, Any]) -> tuple[Take, ...]:
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    model = _source_model(_generation_source(payload), payload.get("model_id"))
    controls = {key: request[key] for key in _GENERATION_CONTROLS if key in request}
    takes = []
    for index, value in enumerate(payload.get("files") or ()):
        audio = _resolve(value, metadata_path.parent)
        takes.append(
            Take(
                index=index,
                audio=audio,
                label=f"sample {index}",
                model=model,
                seed=_optional_int(request.get("seed")),
                duration_seconds=_optional_float(request.get("duration_seconds")),
                controls=controls,
                scores=_listening_scores(audio),
                parent=request.get("prompt_audio_path"),
                replay=metadata_path,
                error=payload.get("error") or None,
            )
        )
    return tuple(takes)


def _listening_scores(audio: Path | None) -> dict[str, Any]:
    from ..listening.reports import read_listening_report

    report = read_listening_report(audio)
    return {"listening_score": report.score} if report is not None and report.score is not None else {}


# ── sweep: sweep_plan.json + index.csv + combo_NNN/metadata.json ──


def _sweep_rows(folder: Path) -> list[dict[str, str]]:
    path = folder / "index.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _sweep_session(folder: Path) -> Session:
    plan = read_json(folder / "sweep_plan.json")
    report = read_json(folder / "sweep_report.json")
    rows = _sweep_rows(folder)
    source = plan.get("source") if isinstance(plan.get("source"), dict) else {}
    models = []
    for value in [_source_model(source), *(row.get("xp_path") or row.get("checkpoint_path") for row in rows)]:
        if value and value not in models:
            models.append(value)
    if not models and plan.get("model_id"):
        models.append(str(plan["model_id"]))
    combinations = plan.get("combinations") if isinstance(plan.get("combinations"), list) else []
    grid: dict[str, list[Any]] = {}
    for combination in combinations:
        for key, value in combination.items():
            if key in ("combo_index", "combo_id"):
                continue
            values = grid.setdefault(key, [])
            if value not in values:
                values.append(value)
    controls: dict[str, Any] = {
        key: plan[key] for key in ("mode", "num_samples", "prompt_audio_path", "prompt_start_seconds") if key in plan
    }
    controls.update({key: values[0] for key, values in grid.items() if len(values) == 1})
    controls["grid"] = {key: values for key, values in grid.items() if len(values) > 1}
    status = report.get("status")
    created_at = plan.get("created_at")
    if not isinstance(created_at, str):
        created_at = _mtime_iso(folder / "index.csv")
    facts = {
        "run_id": str(plan.get("sweep_id") or _first(rows, "sweep_id") or folder.name),
        "profile": plan.get("profile") or next((row["profile"] for row in rows if row.get("profile")), None),
        "model_id": plan.get("model_id") or next((row["model_id"] for row in rows if row.get("model_id")), None),
        "checkpoint_path": source.get("checkpoint_path")
        or next((row["checkpoint_path"] for row in rows if row.get("checkpoint_path")), None),
        "mode": plan.get("mode") or next((row["mode"] for row in rows if row.get("mode")), None),
        "combo_count": len(combinations) or len({row.get("combo_id") for row in rows if row.get("combo_id")}),
        "listening_rows": sum(1 for row in rows if row.get("listening_report_path") or row.get("listening_summary")),
    }
    return Session(
        id=str(plan.get("sweep_id") or folder.name),
        kind="sweep",
        path=folder,
        created_at=created_at,
        finished_at=_mtime_iso(folder / "sweep_report.json") if status in ("completed", "failed") else None,
        models=tuple(models),
        engine={key: plan[key] for key in ("backend", "model_id", "device") if key in plan},
        controls=controls,
        parent=plan.get("prompt_audio_path"),
        take_count=len(rows),
        native="sweep_plan.json" if plan else "index.csv",
        facts=facts,
    )


def _sweep_takes(folder: Path) -> tuple[Take, ...]:
    takes = []
    for index, row in enumerate(_sweep_rows(folder)):
        audio = _resolve(row.get("path"), folder)
        combo = (row.get("combo_id") or "").strip()
        sample = (row.get("sample_index") or "").strip()
        label = combo + (f" #{sample}" if sample and row.get("num_samples") not in (None, "", "1") else "")
        controls = {key: _cell(row.get(key)) for key in SWEEP_CONTROL_COLUMNS if row.get(key) not in (None, "")}
        scores = {}
        for key, value in row.items():
            if key in SWEEP_CONTROL_COLUMNS or key in _SWEEP_TEXT_COLUMNS or value in (None, ""):
                continue
            number = _optional_float(value)
            if number is not None:
                scores[key] = number
        takes.append(
            Take(
                index=index,
                audio=audio,
                label=label or f"take {index}",
                model=row.get("xp_path") or row.get("checkpoint_path") or row.get("model_id") or None,
                seed=_optional_int(row.get("seed")),
                duration_seconds=_optional_float(row.get("duration_seconds")),
                controls=controls,
                scores=scores,
                parent=row.get("prompt_audio_path") or None,
                error=row.get("error") or None,
            )
        )
    return tuple(takes)


def _first(rows: list[dict[str, str]], column: str) -> str | None:
    return next((row[column] for row in rows if row.get(column)), None)


def _cell(value: str | None) -> Any:
    text = (value or "").strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    number = _optional_float(text)
    if number is None:
        return text
    return int(number) if number.is_integer() and "." not in text else number


# ── stream: manifest.json (written at the end) + performance.jsonl (one row per round, live) ──


def _stream_session(folder: Path, manifest: dict[str, Any]) -> Session:
    performance = folder / "performance.jsonl"
    initial = manifest.get("initial_control")
    if not isinstance(initial, dict):
        initial = read_json(folder / "control.json")
    backend = manifest.get("backend") if isinstance(manifest.get("backend"), dict) else {}
    models: list[str] = []
    for value in (backend.get("xp_path"), initial.get("model_xp"), manifest.get("final_xp_path")):
        if value and str(value) not in models and not _same_xp(str(value), models):
            models.append(str(value))
    run_id = manifest.get("run_id") or folder.name.removeprefix("stream_")
    rows = count_lines(performance)
    facts = {
        "run_id": str(run_id),
        "stopped_by": manifest.get("stopped_by"),
        "seed_mode": manifest.get("seed_mode"),
        "rounds_written": int(manifest.get("rounds_written") or rows),
    }
    return Session(
        id=folder.name,
        kind="stream",
        path=folder,
        created_at=_iso(manifest.get("started_at")) or iso_from_run_id(run_id),
        finished_at=_iso(manifest.get("finished_at")),
        models=tuple(models),
        engine={key: backend[key] for key in ("model_id", "device", "checkpoint", "checkpoint_path") if key in backend},
        controls=dict(initial),
        parent=manifest.get("seed_audio_source"),
        take_count=rows,
        takes_log="performance.jsonl",
        native="manifest.json" if manifest else None,
        facts=facts,
    )


def _same_xp(value: str, models: list[str]) -> bool:
    name = Path(value).name
    return any(Path(model).name == name for model in models)


def _stream_takes(folder: Path) -> tuple[Take, ...]:
    """One take per logged row. A reroll logs another row for the same round, so takes are numbered by row: the
    take number is the round number unless the stream rerolled."""
    takes = []
    for row in read_jsonl(folder / "performance.jsonl"):
        round_idx = row.get("round_idx")
        if not isinstance(round_idx, int):
            continue
        control = row.get("active_control") if isinstance(row.get("active_control"), dict) else {}
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        reroll = row.get("take_idx") if isinstance(row.get("take_idx"), int) else 0
        takes.append(
            Take(
                index=len(takes),
                audio=_resolve(row.get("round_wav"), folder),
                label=f"round {round_idx}" + (f" · take {reroll}" if reroll else ""),
                model=row.get("model_xp") or control.get("model_xp"),
                seed=_optional_int(row.get("seed_used")),
                duration_seconds=_optional_float(row.get("duration_seconds")),
                controls=control,
                scores={key: value for key, value in score.items() if key != "candidates"},
                parent=row.get("prompt_audio"),
                prompt_duration_seconds=_optional_float(row.get("prompt_duration_seconds")),
            )
        )
    return tuple(takes)


# ── duet: manifest.json with two voices, a stereo mix and a stem per voice ──


def _duet_session(folder: Path, manifest: dict[str, Any]) -> Session:
    voices = manifest.get("voices") if isinstance(manifest.get("voices"), dict) else {}
    models = tuple(model for model in (_source_model(voice.get("source") or {}) for voice in voices.values()) if model)
    controls = {
        key: manifest[key]
        for key in ("rounds", "round_duration_seconds", "tau_seconds", "candidates", "seed")
        if key in manifest
    }
    controls["voices"] = {name: _voice_controls(voice) for name, voice in voices.items()}
    return Session(
        id=folder.name,
        kind="duet",
        path=folder,
        created_at=_folder_time(folder),
        finished_at=_mtime_iso(folder / "manifest.json"),
        models=models,
        engine={"sample_rate": manifest.get("sample_rate")},
        controls=controls,
        take_count=len(_duet_takes(folder, manifest)),
        native="manifest.json",
    )


def _voice_controls(voice: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in voice.items() if key not in ("source", "stem")}


def _duet_takes(folder: Path, manifest: dict[str, Any]) -> tuple[Take, ...]:
    voices = manifest.get("voices") if isinstance(manifest.get("voices"), dict) else {}
    takes: list[Take] = []
    if manifest.get("mix"):
        takes.append(
            Take(
                index=0,
                audio=_resolve(manifest["mix"], folder),
                label="mix",
                role="mix",
                model=None,
                seed=_optional_int(manifest.get("seed")),
            )
        )
    for name, voice in sorted(voices.items()):
        if not voice.get("stem"):
            continue
        takes.append(
            Take(
                index=len(takes),
                audio=_resolve(voice["stem"], folder),
                label=f"voice {name}",
                role="voice",
                model=_source_model(voice.get("source") or {}),
                seed=_optional_int(manifest.get("seed")),
                controls=_voice_controls(voice),
            )
        )
    return tuple(takes)


# ── variations: manifest.json, N re-performances of one input take, plus its codec round trip ──


def _variations_session(folder: Path, manifest: dict[str, Any]) -> Session:
    model = _source_model(manifest.get("source") or {})
    return Session(
        id=folder.name,
        kind="variations",
        path=folder,
        created_at=_folder_time(folder),
        finished_at=_mtime_iso(folder / "manifest.json"),
        models=(model,) if model else (),
        engine={"sample_rate": manifest.get("sample_rate"), "frame_rate": manifest.get("frame_rate")},
        controls=_variation_controls(manifest),
        parent=manifest.get("input"),
        take_count=len(manifest.get("files") or ()) + (1 if manifest.get("reference_roundtrip") else 0),
        native="manifest.json",
    )


def _variation_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = ("seconds", "n", "freeze_books", "keep_intro_seconds", *_SAMPLING_KEYS, "seed")
    return {key: manifest[key] for key in keys if key in manifest}


def _variations_takes(folder: Path, manifest: dict[str, Any]) -> tuple[Take, ...]:
    model = _source_model(manifest.get("source") or {})
    controls = _variation_controls(manifest)
    parent = manifest.get("input")
    takes = [
        Take(
            index=index,
            audio=_resolve(value, folder),
            label=Path(str(value)).stem.replace("_", " "),
            model=model,
            seed=_optional_int(manifest.get("seed")),
            duration_seconds=_optional_float(manifest.get("seconds")),
            controls=controls,
            parent=parent,
        )
        for index, value in enumerate(manifest.get("files") or ())
    ]
    if manifest.get("reference_roundtrip"):
        takes.append(
            Take(
                index=len(takes),
                audio=_resolve(manifest["reference_roundtrip"], folder),
                label="reference round trip",
                role="reference",
                parent=parent,
            )
        )
    return tuple(takes)


# ── helpers ──


def _resolve(value: object, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else (base / path)


def _iso(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def _folder_time(folder: Path) -> str | None:
    """Duet and variations folders were named ``<utc>`` or ``<yyyymmdd>_<name>``; else the folder's mtime."""
    parts = folder.name.split("_")
    for part in parts:
        iso = iso_from_run_id(part)
        if iso:
            return iso
    for part in parts:
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d").replace(tzinfo=UTC).isoformat()
            except ValueError:
                pass
    return _mtime_iso(folder)


def facts_of(session: Session) -> dict[str, Any]:
    """A session's ``facts``; a session sealed before they were recorded is read again from its producer's files."""
    if session.facts or not session.sealed:
        return session.facts
    try:
        return describe_native(session.folder).facts
    except (SessionNotFound, OSError, ValueError):
        return {}
