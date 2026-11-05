from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..musicgen.backend import GenerationRequest, GenerationResult
from .run_paths import (
    OUTPUT_LAYOUT_FLAT,
    OUTPUT_LAYOUT_SESSION,
    build_generation_run_paths,
    generation_kind,
    relative_to_metadata,
    sample_audio_path,
)


@dataclass(frozen=True)
class SavedAudioRun:
    run_id: str
    output_dir: Path
    audio_paths: list[Path]
    metadata_path: Path


def _to_numpy_audio_batch(audio_values: Any) -> np.ndarray:
    array = audio_values
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()

    np_audio = np.asarray(array)
    if np_audio.ndim == 1:
        # [T] -> [B=1, C=1, T]
        np_audio = np_audio[None, None, :]
    elif np_audio.ndim == 2:
        # [B, T] -> [B, C=1, T]
        np_audio = np_audio[:, None, :]
    elif np_audio.ndim == 3:
        # [B, C, T]
        pass
    else:
        raise ValueError(f"Unsupported audio tensor rank: ndim={np_audio.ndim}. Expected 1D, 2D, or 3D audio values.")

    if np_audio.shape[-1] <= 0:
        raise ValueError("Audio values must contain at least one sample.")
    return np_audio.astype(np.float32, copy=False)


def _write_wav(path: Path, audio_ch_t: np.ndarray, sample_rate: int) -> None:
    channels, frames = audio_ch_t.shape
    clipped = np.clip(audio_ch_t, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2", copy=False)
    interleaved = np.ascontiguousarray(pcm.T)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(channels))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.tobytes())
    _validate_written_wav(path, channels=channels, frames=frames, sample_rate=sample_rate)


def _validate_written_wav(path: Path, *, channels: int, frames: int, sample_rate: int) -> None:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != int(channels):
            raise ValueError(f"WAV channel count mismatch for {path}.")
        if wav_file.getframerate() != int(sample_rate):
            raise ValueError(f"WAV sample rate mismatch for {path}.")
        if wav_file.getnframes() != int(frames):
            raise ValueError(f"WAV frame count mismatch for {path}.")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"WAV sample width mismatch for {path}.")


def _summarize_train_metadata(train_md: dict[str, object]) -> dict[str, object]:
    """Extract a compact model-identity summary from macchina_ricordi_train_metadata.json."""
    summary: dict[str, object] = {}

    label = train_md.get("effective_label")
    if label:
        summary["label"] = label

    ws_dir = train_md.get("workspace_dir")
    if isinstance(ws_dir, str):
        summary["workspace"] = Path(ws_dir).name

    fingerprint = train_md.get("dataset_fingerprint")
    if isinstance(fingerprint, dict):
        summary["dataset"] = {
            k: fingerprint[k] for k in ("train_count", "valid_count", "train_aggregate_sha256") if k in fingerprint
        }

    ic = train_md.get("init_from_checkpoint")
    if isinstance(ic, dict):
        lineage: dict[str, object] = {}
        user_input = ic.get("user_input")
        if user_input:
            lineage["continue_from"] = user_input
        src_md = ic.get("source_resume_metadata")
        if isinstance(src_md, dict):
            mf = src_md.get("merged_from")
            if isinstance(mf, dict):
                lineage["merged_from"] = {k: mf[k] for k in ("method", "alpha", "sources") if k in mf}
            pf = src_md.get("perturbed_from")
            if isinstance(pf, dict):
                lineage["perturbed_from"] = {k: pf[k] for k in ("sigma", "layer_scope", "seed", "sources") if k in pf}
        summary["init_from_checkpoint"] = lineage

    return summary


def _inject_model_metadata(metadata: dict[str, object], result: GenerationResult) -> None:
    """Inline a summary of the model's training metadata into generation output."""
    source = result.metadata.get("source")
    if not isinstance(source, dict):
        return

    if source.get("kind") == "bundle":
        metadata["model"] = {
            key: source[key]
            for key in ("name", "weights_digest", "weights_state", "codec", "provenance")
            if key in source
        }
        metadata["model"]["kind"] = "bundle"
        return

    xp_path = source.get("xp_path")
    if not isinstance(xp_path, str):
        return

    train_md_path = Path(xp_path) / "macchina_ricordi_train_metadata.json"
    if not train_md_path.exists():
        return

    try:
        train_md = json.loads(train_md_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    model_summary = _summarize_train_metadata(train_md)
    if model_summary:
        metadata["model"] = model_summary


def save_generation_run(
    result: GenerationResult,
    request: GenerationRequest,
    *,
    output_dir: str | Path,
    prefix: str = "musicgen",
    output_layout: str = OUTPUT_LAYOUT_FLAT,
    now: datetime | None = None,
) -> SavedAudioRun:
    if result.sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    run_id = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    paths = build_generation_run_paths(
        output_dir,
        prefix=prefix,
        run_id=run_id,
        output_layout=output_layout,
        kind=generation_kind(request.mode),
    )

    paths.samples_dir.mkdir(parents=True, exist_ok=True)

    batch_audio = _to_numpy_audio_batch(result.audio_values)
    audio_paths: list[Path] = []

    for idx, sample_ch_t in enumerate(batch_audio):
        audio_path = sample_audio_path(paths, idx)
        _write_wav(audio_path, sample_ch_t, result.sample_rate)
        audio_paths.append(audio_path)

    metadata_path = paths.metadata_path
    metadata: dict[str, object] = {
        "run_id": run_id,
        "backend": result.backend,
        "model_id": result.model_id,
        "device": result.device,
        "sample_rate": result.sample_rate,
        "request": {
            "prompts": list(request.prompts),
            "batch_size": request.batch_size,
            "num_samples": request.batch_size,
            "mode": request.mode,
            "duration_seconds": request.duration_seconds,
            "max_new_tokens": request.max_new_tokens,
            "seed": request.seed,
            "prompt_audio_path": request.prompt_audio_path,
            "prompt_start_seconds": request.prompt_start_seconds,
            "prompt_duration_seconds": request.prompt_duration_seconds,
            "use_sampling": request.use_sampling,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "cfg_coef": request.cfg_coef,
            "extend_stride": request.extend_stride,
        },
        "generation": dict(result.metadata),
        "warnings": list(result.warnings),
        "files": [relative_to_metadata(path, metadata_path) for path in audio_paths],
        "output": {
            "layout": paths.layout,
            "prefix": paths.prefix,
        },
    }

    _inject_model_metadata(metadata, result)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if paths.layout == OUTPUT_LAYOUT_SESSION:
        from ..sessions.store import seal_session

        seal_session(paths.run_dir)

    return SavedAudioRun(
        run_id=run_id,
        output_dir=paths.run_dir,
        audio_paths=audio_paths,
        metadata_path=metadata_path,
    )
