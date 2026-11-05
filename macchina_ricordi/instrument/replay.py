from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.backend import DEFAULT_EXTEND_STRIDE_SECONDS, GenerationRequest
from ..musicgen.shared import sha256_file, utc_now_iso, write_json
from .run_paths import OUTPUT_LAYOUT_FLAT, resolve_metadata_file_path
from .saving import SavedAudioRun, save_generation_run

REPLAY_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplaySourceRun:
    metadata_path: Path
    run_id: str
    backend: str
    model_id: str
    device: str
    sample_rate: int
    prompts: tuple[str, ...]
    batch_size: int
    mode: str
    duration_seconds: float
    max_new_tokens: int
    seed: int | None
    prompt_audio_path: str | None
    prompt_start_seconds: float
    prompt_duration_seconds: float | None
    use_sampling: bool
    top_k: int
    top_p: float
    temperature: float
    cfg_coef: float
    extend_stride: float
    generation: dict[str, Any]
    audio_paths: tuple[Path, ...]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class ReplaySampleResult:
    index: int
    source_path: Path
    replay_path: Path
    source_sha256: str
    replay_sha256: str
    matches: bool


@dataclass(frozen=True)
class ReplayReport:
    source: ReplaySourceRun
    replay_run: SavedAudioRun
    samples: tuple[ReplaySampleResult, ...]
    all_match: bool
    report_path: Path


def resolve_replay_metadata(ref: str | Path) -> Path:
    """What to replay: a generate or continue session (folder or id), one of its takes, or a metadata JSON."""
    from ..sessions.record import REPLAYABLE_KINDS, SessionNotFound
    from ..sessions.store import TAKE_SEPARATOR, find_session, find_take

    path = Path(ref).expanduser()
    if path.is_file():
        return path
    if (path / "run.json").is_file():
        return path / "run.json"
    text = str(ref)
    if TAKE_SEPARATOR in text:
        session, take = find_take(text)
        if take.replay is None:
            raise SessionNotFound(f"{text}: {_not_replayable(session.kind)}")
        return take.replay
    session = find_session(text)
    if session.kind not in REPLAYABLE_KINDS:
        raise SessionNotFound(f"{session.id}: {_not_replayable(session.kind)}")
    return session.path if session.path.is_file() else session.folder / "run.json"


def _not_replayable(kind: str) -> str:
    if kind == "stream":
        return "a stream round depends on what played before it, so it cannot be replayed"
    if kind == "sweep":
        return (
            "a sweep's takes come from the sweep engine (fp32), and replay runs the generate engine (fp16, CUDA "
            "graphs), so the bytes would not match; to compare anyway, pass its combo_NNN/metadata.json"
        )
    return f"a {kind} take cannot be replayed (only generate and continue sessions can)"


def load_source_run(metadata_path: Path) -> ReplaySourceRun:
    resolved = Path(metadata_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Replay source metadata not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    required_keys = {"run_id", "backend", "model_id", "device", "sample_rate", "request", "files"}
    missing = required_keys - set(payload)
    if missing:
        raise RuntimeError(f"Replay source metadata is missing required keys: {sorted(missing)}")
    request_payload = payload["request"]
    prompts = request_payload.get("prompts") or []
    audio_paths = [
        path
        for path in (resolve_metadata_file_path(value, metadata_path=resolved) for value in payload["files"])
        if path is not None
    ]
    for audio_path in audio_paths:
        if not audio_path.exists():
            raise FileNotFoundError(f"Replay source audio file is missing: {audio_path}")
    seed = request_payload.get("seed")
    prompt_audio_path = request_payload.get("prompt_audio_path")
    prompt_duration_seconds = request_payload.get("prompt_duration_seconds")
    return ReplaySourceRun(
        metadata_path=resolved,
        run_id=str(payload["run_id"]),
        backend=str(payload["backend"]),
        model_id=str(payload["model_id"]),
        device=str(payload["device"]),
        sample_rate=int(payload["sample_rate"]),
        prompts=tuple(str(prompt) for prompt in prompts),
        batch_size=int(request_payload.get("batch_size", max(1, len(prompts)))),
        mode=str(request_payload.get("mode", "text")),
        duration_seconds=float(request_payload.get("duration_seconds", 10.0)),
        max_new_tokens=int(request_payload.get("max_new_tokens", 256)),
        seed=int(seed) if seed is not None else None,
        prompt_audio_path=str(prompt_audio_path) if prompt_audio_path is not None else None,
        prompt_start_seconds=float(request_payload.get("prompt_start_seconds", 0.0)),
        prompt_duration_seconds=(float(prompt_duration_seconds) if prompt_duration_seconds is not None else None),
        use_sampling=bool(request_payload.get("use_sampling", True)),
        top_k=int(request_payload.get("top_k", 250)),
        top_p=float(request_payload.get("top_p", 0.0)),
        temperature=float(request_payload.get("temperature", 1.0)),
        cfg_coef=float(request_payload.get("cfg_coef", 3.0)),
        extend_stride=float(request_payload.get("extend_stride", DEFAULT_EXTEND_STRIDE_SECONDS)),
        generation=dict(payload.get("generation", {})),
        audio_paths=tuple(audio_paths),
        raw_payload=payload,
    )


def build_replay_request(source: ReplaySourceRun) -> GenerationRequest:
    return GenerationRequest(
        prompts=list(source.prompts) or [""],
        batch_size=source.batch_size,
        mode=source.mode,
        duration_seconds=source.duration_seconds,
        max_new_tokens=source.max_new_tokens,
        device=source.device,
        seed=source.seed,
        prompt_audio_path=source.prompt_audio_path,
        prompt_start_seconds=source.prompt_start_seconds,
        prompt_duration_seconds=source.prompt_duration_seconds,
        use_sampling=source.use_sampling,
        top_k=source.top_k,
        top_p=source.top_p,
        temperature=source.temperature,
        cfg_coef=source.cfg_coef,
        extend_stride=source.extend_stride,
    )


def run_replay_audit(
    source: ReplaySourceRun,
    *,
    output_dir: str | Path,
    prefix: str = "replay",
    backend_factory: Callable[[ReplaySourceRun], Any] | None = None,
    now_iso: Callable[[], str] | None = None,
    report_path: Path | None = None,
) -> ReplayReport:
    factory = backend_factory or _default_replay_backend_factory
    backend = factory(source)
    request = build_replay_request(source)
    result = backend.generate(request)

    output_root = Path(output_dir).expanduser().resolve()
    timestamp = (now_iso or _compact_utc_now)()
    replay_dir = output_root / f"replay_{source.run_id}_{timestamp}"
    replay_dir.mkdir(parents=True, exist_ok=True)

    replay_run = save_generation_run(
        result,
        request,
        output_dir=replay_dir,
        prefix=prefix,
        output_layout=OUTPUT_LAYOUT_FLAT,
    )

    samples: list[ReplaySampleResult] = []
    for index, source_path in enumerate(source.audio_paths):
        replay_path = replay_run.audio_paths[index] if index < len(replay_run.audio_paths) else None
        if replay_path is None:
            source_sha = sha256_file(source_path)
            samples.append(
                ReplaySampleResult(
                    index=index,
                    source_path=source_path,
                    replay_path=replay_dir / f"MISSING_INDEX_{index:03d}",
                    source_sha256=source_sha,
                    replay_sha256="",
                    matches=False,
                )
            )
            continue
        source_sha = sha256_file(source_path)
        replay_sha = sha256_file(replay_path)
        samples.append(
            ReplaySampleResult(
                index=index,
                source_path=source_path,
                replay_path=replay_path,
                source_sha256=source_sha,
                replay_sha256=replay_sha,
                matches=source_sha == replay_sha,
            )
        )

    all_match = (
        len(samples) == len(source.audio_paths)
        and len(samples) == len(replay_run.audio_paths)
        and all(sample.matches for sample in samples)
    )

    resolved_report_path = report_path or (replay_dir / "replay_report.json")
    write_json(
        resolved_report_path,
        build_replay_report_payload(
            source=source,
            replay_run=replay_run,
            samples=samples,
            all_match=all_match,
        ),
    )
    return ReplayReport(
        source=source,
        replay_run=replay_run,
        samples=tuple(samples),
        all_match=all_match,
        report_path=resolved_report_path,
    )


def build_replay_report_payload(
    *,
    source: ReplaySourceRun,
    replay_run: SavedAudioRun,
    samples: list[ReplaySampleResult],
    all_match: bool,
) -> dict[str, object]:
    return {
        "schema_version": REPLAY_REPORT_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "all_match": all_match,
        "source": {
            "metadata_path": str(source.metadata_path),
            "run_id": source.run_id,
            "backend": source.backend,
            "model_id": source.model_id,
            "device": source.device,
            "sample_rate": source.sample_rate,
            "prompts": list(source.prompts),
            "batch_size": source.batch_size,
            "mode": source.mode,
            "duration_seconds": source.duration_seconds,
            "max_new_tokens": source.max_new_tokens,
            "seed": source.seed,
            "prompt_audio_path": source.prompt_audio_path,
            "prompt_start_seconds": source.prompt_start_seconds,
            "prompt_duration_seconds": source.prompt_duration_seconds,
            "use_sampling": source.use_sampling,
            "top_k": source.top_k,
            "top_p": source.top_p,
            "temperature": source.temperature,
            "cfg_coef": source.cfg_coef,
            "extend_stride": source.extend_stride,
            "generation": source.generation,
            "audio_paths": [str(path) for path in source.audio_paths],
        },
        "replay_run": {
            "run_id": replay_run.run_id,
            "output_dir": str(replay_run.output_dir),
            "metadata_path": str(replay_run.metadata_path),
            "audio_paths": [str(path) for path in replay_run.audio_paths],
        },
        "samples": [
            {
                "index": sample.index,
                "source_path": str(sample.source_path),
                "replay_path": str(sample.replay_path),
                "source_sha256": sample.source_sha256,
                "replay_sha256": sample.replay_sha256,
                "matches": sample.matches,
            }
            for sample in samples
        ],
    }


def _default_replay_backend_factory(source: ReplaySourceRun):  # type: ignore[no-untyped-def]
    source_payload = source.generation.get("source")
    if source.backend != "audiocraft":
        raise ValueError(
            f"This run was generated with the {source.backend!r} backend, which is not supported by this checkout. "
            "Replay requires a compatible implementation of the original backend."
        )
    from ..musicgen.backend import build_audiocraft_backend

    source_dict = source_payload if isinstance(source_payload, dict) else {}
    if source_dict.get("kind") == "bundle":
        from ..models.bundle import read_bundle

        path = source_dict.get("bundle_path")
        if not isinstance(path, str) or not path:
            raise ValueError("The recorded bundle path is missing; replay cannot select the original model.")
        bundle = read_bundle(path)
        if source_dict.get("weights_digest") != bundle.config["weights"]["sha256"]:
            raise ValueError("The bundle weights changed since this session; restore the recorded snapshot to replay.")
        return build_audiocraft_backend(model_id=bundle.config["name"], device=source.device, xp_path=bundle.path)
    return build_audiocraft_backend(
        model_id=source.model_id,
        device=source.device,
        profile_name=str(source_dict["profile"]) if source_dict.get("profile") else None,
        xp_path=str(source_dict["xp_path"]) if source_dict.get("xp_path") else None,
        checkpoint_path=(str(source_dict["checkpoint_path"]) if source_dict.get("checkpoint_path") else None),
    )


def _compact_utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
