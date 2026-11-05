from __future__ import annotations

import csv
import html
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from ..musicgen.backend import resolve_generation_checkpoint
from ..musicgen.config import musicgen_script_env
from ..musicgen.runtime import load_musicgen_project_config, require_musicgen_python
from ..musicgen.shared import write_json
from .provenance import resolve_generation_profile

SWEEP_SCHEMA_VERSION = 1
SWEEP_GRID_FIELDS = (
    "duration_seconds",
    "prompt_duration_seconds",
    "temperature",
    "top_k",
    "top_p",
    "cfg_coef",
    "extend_stride",
    "seed",
    "use_sampling",
)
SWEEP_INDEX_FIELDNAMES = [
    "sweep_id",
    "combo_id",
    "combo_index",
    "sample_index",
    "path",
    "error",
    "mode",
    "model_id",
    "profile",
    "xp_path",
    "checkpoint_path",
    "prompt_audio_path",
    "prompt_start_seconds",
    "prompt_duration_seconds",
    "duration_seconds",
    "num_samples",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "cfg_coef",
    "extend_stride",
    "use_sampling",
    "listening_report_path",
    "listening_markdown_path",
    "listening_summary",
    "listening_score",
    "listening_labels",
    "listening_error",
]
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class SweepRun:
    sweep_id: str
    output_dir: Path
    plan_path: Path
    report_path: Path
    index_csv_path: Path
    index_html_path: Path | None
    combo_count: int
    sample_count: int
    dry_run: bool


def normalize_sweep_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "continue":
        return "continuation"
    if normalized in {"continuation", "unconditional"}:
        return normalized
    raise ValueError("Sweep mode must be continue or unconditional.")


def parse_float_grid(raw: str | None, *, default: float | None, name: str) -> tuple[float | None, ...]:
    if raw is None:
        return (default,)
    values: list[float | None] = []
    for token in _split_grid_values(raw, name=name):
        if token.lower() in {"none", "null"}:
            values.append(None)
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {name}: {token!r}") from exc
    return tuple(values)


def parse_int_grid(raw: str | None, *, default: int | None, name: str) -> tuple[int | None, ...]:
    if raw is None:
        return (default,)
    values: list[int | None] = []
    for token in _split_grid_values(raw, name=name):
        if token.lower() in {"none", "null"}:
            values.append(None)
            continue
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in {name}: {token!r}") from exc
    return tuple(values)


def parse_bool_grid(raw: str | None, *, default: bool, name: str) -> tuple[bool, ...]:
    if raw is None:
        return (default,)
    values: list[bool] = []
    for token in _split_grid_values(raw, name=name):
        normalized = token.lower()
        if normalized in TRUE_VALUES:
            values.append(True)
        elif normalized in FALSE_VALUES:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean in {name}: {token!r}")
    return tuple(values)


def build_sweep_combinations(
    *,
    mode: str,
    duration_seconds: float,
    prompt_duration_seconds: float | None,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
    extend_stride: float,
    seed: int | None,
    use_sampling: bool,
    duration_seconds_values: str | None = None,
    prompt_duration_seconds_values: str | None = None,
    temperature_values: str | None = None,
    top_k_values: str | None = None,
    top_p_values: str | None = None,
    cfg_coef_values: str | None = None,
    extend_stride_values: str | None = None,
    seed_values: str | None = None,
    use_sampling_values: str | None = None,
    max_combinations: int = 64,
) -> tuple[dict[str, object], ...]:
    normalized_mode = normalize_sweep_mode(mode)
    if max_combinations < 1:
        raise ValueError("max_combinations must be at least 1.")

    prompt_default = None if normalized_mode == "unconditional" else prompt_duration_seconds
    grid = {
        "duration_seconds": parse_float_grid(
            duration_seconds_values, default=duration_seconds, name="duration_seconds_values"
        ),
        "prompt_duration_seconds": parse_float_grid(
            prompt_duration_seconds_values,
            default=prompt_default,
            name="prompt_duration_seconds_values",
        ),
        "temperature": parse_float_grid(temperature_values, default=temperature, name="temperature_values"),
        "top_k": parse_int_grid(top_k_values, default=top_k, name="top_k_values"),
        "top_p": parse_float_grid(top_p_values, default=top_p, name="top_p_values"),
        "cfg_coef": parse_float_grid(cfg_coef_values, default=cfg_coef, name="cfg_coef_values"),
        "extend_stride": parse_float_grid(extend_stride_values, default=extend_stride, name="extend_stride_values"),
        "seed": parse_int_grid(seed_values, default=seed, name="seed_values"),
        "use_sampling": parse_bool_grid(use_sampling_values, default=use_sampling, name="use_sampling_values"),
    }
    if normalized_mode == "unconditional":
        grid["prompt_duration_seconds"] = (None,)

    combination_count = 1
    for values in grid.values():
        combination_count *= len(values)
    if combination_count > max_combinations:
        raise ValueError(
            f"Sweep expands to {combination_count} combinations, above --max-combinations={max_combinations}."
        )

    combinations: list[dict[str, object]] = []
    for index, values in enumerate(product(*(grid[field] for field in SWEEP_GRID_FIELDS))):
        combo = dict(zip(SWEEP_GRID_FIELDS, values, strict=True))
        _validate_combination(normalized_mode, combo)
        combo["combo_index"] = index
        combo["combo_id"] = f"combo_{index:03d}"
        combinations.append(combo)
    return tuple(combinations)


def run_musicgen_sweep(
    *,
    mode: str,
    model_id: str | None,
    profile_name: str | None,
    device: str | None,
    xp_path: str | Path | None,
    checkpoint: str | None,
    checkpoint_path: str | Path | None,
    prompt_audio_path: str | Path | None,
    prompt_start_seconds: float,
    prompt_duration_seconds: float | None,
    num_samples: int,
    duration_seconds: float,
    seed: int | None,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_coef: float,
    use_sampling: bool,
    extend_stride: float,
    output_dir: str | Path | None,
    output_prefix: str = "sweep",
    duration_seconds_values: str | None = None,
    prompt_duration_seconds_values: str | None = None,
    temperature_values: str | None = None,
    top_k_values: str | None = None,
    top_p_values: str | None = None,
    cfg_coef_values: str | None = None,
    extend_stride_values: str | None = None,
    seed_values: str | None = None,
    use_sampling_values: str | None = None,
    max_combinations: int = 64,
    dry_run: bool = False,
    score: bool = False,
    index_html: bool = True,
    listen: bool = False,
    now: datetime | None = None,
) -> SweepRun:
    normalized_mode = normalize_sweep_mode(mode)
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if prompt_start_seconds < 0:
        raise ValueError("prompt_start_seconds must be non-negative.")

    resolved_prompt_audio = _resolve_prompt_audio(
        mode=normalized_mode,
        prompt_audio_path=prompt_audio_path,
    )
    combinations = build_sweep_combinations(
        mode=normalized_mode,
        duration_seconds=duration_seconds,
        prompt_duration_seconds=prompt_duration_seconds,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef,
        extend_stride=extend_stride,
        seed=seed,
        use_sampling=use_sampling,
        duration_seconds_values=duration_seconds_values,
        prompt_duration_seconds_values=prompt_duration_seconds_values,
        temperature_values=temperature_values,
        top_k_values=top_k_values,
        top_p_values=top_p_values,
        cfg_coef_values=cfg_coef_values,
        extend_stride_values=extend_stride_values,
        seed_values=seed_values,
        use_sampling_values=use_sampling_values,
        max_combinations=max_combinations,
    )
    resolved_checkpoint = resolve_generation_checkpoint(
        xp_path=xp_path,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    resolved_xp = Path(xp_path).expanduser().resolve() if xp_path is not None else None
    resolved_profile = resolve_generation_profile(
        profile_name=profile_name,
        xp_path=resolved_xp,
        checkpoint_path=resolved_checkpoint,
    )
    source = {
        "kind": "xp" if resolved_checkpoint is not None else "model_id",
        "model_id": model_id or "facebook/musicgen-small",
        "profile": resolved_profile,
        "xp_path": str(resolved_xp) if resolved_xp is not None else None,
        "checkpoint": checkpoint or "latest",
        "checkpoint_path": str(resolved_checkpoint) if resolved_checkpoint is not None else None,
    }
    sweep_id, sweep_dir = _next_sweep_dir(output_dir=output_dir, output_prefix=output_prefix, now=now)
    plan_path = sweep_dir / "sweep_plan.json"
    report_path = sweep_dir / "sweep_report.json"
    index_csv_path = sweep_dir / "index.csv"
    index_html_path = sweep_dir / "index.html" if index_html else None
    plan = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "backend": "audiocraft",
        "mode": normalized_mode,
        "model_id": model_id or "facebook/musicgen-small",
        "device": device,
        "profile": resolved_profile,
        "num_samples": int(num_samples),
        "prompt_audio_path": str(resolved_prompt_audio) if resolved_prompt_audio is not None else None,
        "prompt_start_seconds": float(prompt_start_seconds),
        "source": source,
        "combinations": list(combinations),
    }
    sweep_dir.mkdir(parents=True, exist_ok=False)
    write_json(plan_path, plan)

    if dry_run:
        _write_csv(index_csv_path, SWEEP_INDEX_FIELDNAMES, [])
        report = _build_report(
            plan=plan,
            status="dry_run",
            report_path=report_path,
            index_csv_path=index_csv_path,
            index_html_path=index_html_path,
            samples_written=0,
            combo_failures=0,
        )
        report["score_enabled"] = bool(score)
        report["listen_enabled"] = bool(listen)
        report["listening_reports"] = 0
        report["listening_failed"] = 0
        write_json(report_path, report)
        if index_html_path is not None:
            write_sweep_index_html(index_csv_path=index_csv_path, output_path=index_html_path, title=sweep_id)
        _seal(sweep_dir)
        return SweepRun(
            sweep_id=sweep_id,
            output_dir=sweep_dir,
            plan_path=plan_path,
            report_path=report_path,
            index_csv_path=index_csv_path,
            index_html_path=index_html_path,
            combo_count=len(combinations),
            sample_count=0,
            dry_run=True,
        )

    _run_sweep_helper(plan_path)
    rows = _read_csv(index_csv_path)
    if score:
        rows = score_sweep_rows(rows)
        rows = arc_score_sweep_rows(rows)
        rows = structure_score_sweep_rows(rows)
    if listen:
        rows = listen_sweep_rows(rows)
    if score or listen:
        _write_csv(index_csv_path, _fieldnames_for_rows(rows), rows)
    if index_html_path is not None:
        write_sweep_index_html(index_csv_path=index_csv_path, output_path=index_html_path, title=sweep_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["score_enabled"] = bool(score)
    report["listen_enabled"] = bool(listen)
    report["listening_reports"] = sum(1 for row in rows if row.get("listening_report_path"))
    report["listening_failed"] = sum(1 for row in rows if row.get("listening_error"))
    report["index_html_path"] = str(index_html_path) if index_html_path is not None else None
    write_json(report_path, report)
    _seal(sweep_dir)

    return SweepRun(
        sweep_id=sweep_id,
        output_dir=sweep_dir,
        plan_path=plan_path,
        report_path=report_path,
        index_csv_path=index_csv_path,
        index_html_path=index_html_path,
        combo_count=len(combinations),
        sample_count=sum(1 for row in rows if row.get("path")),
        dry_run=False,
    )


def score_sweep_rows(
    rows: Sequence[dict[str, str]],
    *,
    scorer: Callable[..., Any] | None = None,
    reward_components_cls: type[Any] | None = None,
) -> list[dict[str, str]]:
    if reward_components_cls is None:
        from macchina_ricordi.scoring.reward import RewardComponents

        reward_components_cls = RewardComponents
    if scorer is None:
        from macchina_ricordi.scoring.reward import score_arrhythmic_percussive

        scorer = score_arrhythmic_percussive

    scored_rows: list[dict[str, str]] = []
    default_components = reward_components_cls().to_dict()
    for row in rows:
        scored = dict(row)
        scored.update({key: str(value) for key, value in default_components.items()})
        scored["score_error"] = ""
        path = scored.get("path", "")
        if not path or scored.get("error"):
            scored_rows.append(scored)
            continue
        try:
            components = scorer(Path(path))
            scored.update({key: str(value) for key, value in components.to_dict().items()})
        except Exception as exc:  # pylint: disable=broad-except
            scored["score_error"] = str(exc)
        scored_rows.append(scored)
    return scored_rows


def arc_score_sweep_rows(
    rows: Sequence[dict[str, str]],
    *,
    scorer: Callable[..., Any] | None = None,
    arc_components_cls: type[Any] | None = None,
) -> list[dict[str, str]]:
    """Add arc/coherence columns to scored sweep rows (the development/coherence axis the
    arrhythmic scorer is blind to). Rides along with ``--score``; never raises per-row."""
    if arc_components_cls is None:
        from macchina_ricordi.scoring.arc import ArcCoherenceComponents

        arc_components_cls = ArcCoherenceComponents
    if scorer is None:
        from macchina_ricordi.scoring.arc import compute_arc_coherence

        scorer = compute_arc_coherence

    scored_rows: list[dict[str, str]] = []
    default_components = arc_components_cls().to_dict()
    for row in rows:
        scored = dict(row)
        scored.update({key: str(value) for key, value in default_components.items()})
        scored.setdefault("arc_score_error", "")
        scored["arc_score_error"] = ""
        path = scored.get("path", "")
        if not path or scored.get("error"):
            scored_rows.append(scored)
            continue
        try:
            components = scorer(
                Path(path),
                prompt_duration_seconds=scored.get("prompt_duration_seconds") or None,
            )
            scored.update({key: str(value) for key, value in components.to_dict().items()})
        except Exception as exc:  # pylint: disable=broad-except
            scored["arc_score_error"] = str(exc)
        scored_rows.append(scored)
    return scored_rows


def structure_score_sweep_rows(
    rows: Sequence[dict[str, str]],
    *,
    scorer: Callable[..., Any] | None = None,
    structure_components_cls: type[Any] | None = None,
) -> list[dict[str, str]]:
    """Add multi-axis development + SSM structure columns to scored sweep rows.

    ``arc_*`` only sees harmonic movement; these columns also see timbral, dynamic and
    rhythmic development, which is how percussive models actually move. Rides along with
    ``--score``; never raises per-row."""
    if structure_components_cls is None:
        from macchina_ricordi.scoring.structure import StructureComponents

        structure_components_cls = StructureComponents
    if scorer is None:
        from macchina_ricordi.scoring.structure import compute_structure

        scorer = compute_structure

    scored_rows: list[dict[str, str]] = []
    default_components = structure_components_cls().to_dict()
    for row in rows:
        scored = dict(row)
        scored.update({key: str(value) for key, value in default_components.items()})
        scored["structure_score_error"] = ""
        path = scored.get("path", "")
        if not path or scored.get("error"):
            scored_rows.append(scored)
            continue
        try:
            components = scorer(
                Path(path),
                prompt_duration_seconds=scored.get("prompt_duration_seconds") or None,
            )
            scored.update({key: str(value) for key, value in components.to_dict().items()})
        except Exception as exc:  # pylint: disable=broad-except
            scored["structure_score_error"] = str(exc)
        scored_rows.append(scored)
    return scored_rows


def listen_sweep_rows(
    rows: Sequence[dict[str, str]],
    *,
    listener: Callable[..., Any] | None = None,
) -> list[dict[str, str]]:
    if listener is None:
        from .listening import write_listening_sidecars

        listener = write_listening_sidecars

    listened_rows: list[dict[str, str]] = []
    defaults = {
        "listening_report_path": "",
        "listening_markdown_path": "",
        "listening_summary": "",
        "listening_score": "",
        "listening_labels": "",
        "listening_error": "",
    }
    for row in rows:
        listened = dict(row)
        listened.update(defaults)
        path = listened.get("path", "")
        if not path or listened.get("error"):
            listened_rows.append(listened)
            continue
        try:
            result = listener(Path(path))
            error = str(getattr(result, "error", "") or "")
            if error:
                listened["listening_error"] = error
            else:
                listened["listening_report_path"] = str(getattr(result, "report_path", "") or "")
                listened["listening_markdown_path"] = str(getattr(result, "markdown_path", "") or "")
                listened["listening_summary"] = str(getattr(result, "summary", "") or "")
                listened["listening_labels"] = str(getattr(result, "labels", "") or "")
                score = getattr(result, "score", None)
                listened["listening_score"] = "" if score is None else str(score)
        except Exception as exc:
            listened["listening_error"] = str(exc)
        listened_rows.append(listened)
    return listened_rows


def write_sweep_index_html(*, index_csv_path: Path, output_path: Path, title: str) -> None:
    rows = _read_csv(index_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "combo_id",
        "sample_index",
        "score",
        "temperature",
        "top_k",
        "top_p",
        "cfg_coef",
        "duration_seconds",
        "prompt_duration_seconds",
        "seed",
        "use_sampling",
        "listening_score",
        "listening_summary",
        "listening_error",
        "error",
    ]
    header_cells = "".join(f"<th>{html.escape(column)}</th>" for column in ["audio", *columns])
    body_rows = []
    for row in rows:
        source = ""
        if row.get("path"):
            source = os.path.relpath(row["path"], output_path.parent)
        audio_cell = f'<audio controls preload="none" src="{html.escape(source)}"></audio>' if source else ""
        value_cells = "".join(f"<td>{html.escape(row.get(column, ''))}</td>" for column in columns)
        body_rows.append(f"<tr><td>{audio_cell}</td>{value_cells}</tr>")
    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{html.escape(title)}</title>",
            "<style>",
            "body{font-family:sans-serif;margin:24px;}",
            "table{border-collapse:collapse;width:100%;}",
            "th,td{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top;}",
            "audio{width:220px;}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
            "<table>",
            f"<thead><tr>{header_cells}</tr></thead>",
            f"<tbody>{''.join(body_rows)}</tbody>",
            "</table>",
            "</body>",
            "</html>",
        ]
    )
    output_path.write_text(document, encoding="utf-8")


def _split_grid_values(raw: str, *, name: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _validate_combination(mode: str, combo: dict[str, object]) -> None:
    duration = _required_float(combo["duration_seconds"], "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds values must be positive.")
    prompt_duration = combo.get("prompt_duration_seconds")
    if mode == "continuation":
        if prompt_duration is None:
            raise ValueError("continuation sweeps require prompt_duration_seconds.")
        prompt_duration_float = _required_float(prompt_duration, "prompt_duration_seconds")
        if prompt_duration_float <= 0:
            raise ValueError("prompt_duration_seconds values must be positive.")
        if prompt_duration_float >= duration:
            raise ValueError("prompt_duration_seconds values must be shorter than duration_seconds values.")
    if _required_float(combo["temperature"], "temperature") <= 0:
        raise ValueError("temperature values must be positive.")
    if _required_int(combo["top_k"], "top_k") < 0:
        raise ValueError("top_k values must be non-negative.")
    if _required_float(combo["top_p"], "top_p") < 0:
        raise ValueError("top_p values must be non-negative.")
    if _required_float(combo["cfg_coef"], "cfg_coef") < 0:
        raise ValueError("cfg_coef values must be non-negative.")
    if _required_float(combo["extend_stride"], "extend_stride") <= 0:
        raise ValueError("extend_stride values must be positive.")


def _required_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} cannot be none.")
    return float(value)


def _required_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} cannot be none.")
    return int(value)


def _resolve_prompt_audio(*, mode: str, prompt_audio_path: str | Path | None) -> Path | None:
    if mode != "continuation":
        return None
    if prompt_audio_path is None:
        raise ValueError("continuation sweeps require --prompt-audio.")
    resolved = Path(prompt_audio_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Prompt audio does not exist: {resolved}")
    return resolved


def _next_sweep_dir(
    *,
    output_dir: str | Path | None,
    output_prefix: str,
    now: datetime | None,
) -> tuple[str, Path]:
    safe_prefix = output_prefix.strip().replace(" ", "_") or "sweep"
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        from ..sessions.store import new_session_dir

        folder = new_session_dir("sweep", run_id=timestamp, label=None if safe_prefix == "sweep" else safe_prefix)
        return folder.name, folder
    root = Path(output_dir).expanduser().resolve()
    base_id = f"{safe_prefix}_{timestamp}"
    for suffix in ["", *[f"_{index:03d}" for index in range(1, 1000)]]:
        sweep_id = f"{base_id}{suffix}"
        candidate = root / sweep_id
        if not candidate.exists():
            return sweep_id, candidate
    raise RuntimeError(f"Could not find an unused sweep directory under {root}")


def _seal(sweep_dir: Path) -> None:
    from ..sessions.store import seal_session

    seal_session(sweep_dir)


def _run_sweep_helper(plan_path: Path) -> None:
    config = load_musicgen_project_config()
    musicgen_python = Path(require_musicgen_python()).expanduser()
    helper_script = config.repo_root / "scripts" / "run_musicgen_sweep.py"
    if not helper_script.exists():
        raise FileNotFoundError(f"AudioCraft sweep helper script not found: {helper_script}")
    env = musicgen_script_env()
    command = [str(musicgen_python), str(helper_script), "--plan-json", str(plan_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        message = f"AudioCraft MusicGen sweep helper failed with exit code {exc.returncode}."
        if stderr:
            message = f"{message} {stderr}"
        raise RuntimeError(message) from exc


def _build_report(
    *,
    plan: dict[str, object],
    status: str,
    report_path: Path,
    index_csv_path: Path,
    index_html_path: Path | None,
    samples_written: int,
    combo_failures: int,
) -> dict[str, object]:
    combinations = plan["combinations"]
    assert isinstance(combinations, list)
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sweep_id": plan["sweep_id"],
        "status": status,
        "mode": plan["mode"],
        "combo_count": len(combinations),
        "combo_failures": combo_failures,
        "samples_written": samples_written,
        "plan_path": str(report_path.parent / "sweep_plan.json"),
        "report_path": str(report_path),
        "index_csv_path": str(index_csv_path),
        "index_html_path": str(index_html_path) if index_html_path is not None else None,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames_for_rows(rows: Sequence[dict[str, str]]) -> list[str]:
    fieldnames = list(SWEEP_INDEX_FIELDNAMES)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
