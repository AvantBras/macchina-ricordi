from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np

EPS = 1e-8


@dataclass(frozen=True)
class SegmentInterval:
    track_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _write_wav(path: Path, audio_ch_t: np.ndarray, sample_rate: int) -> None:
    channels, frames = audio_ch_t.shape
    clipped = np.clip(audio_ch_t, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    interleaved = pcm16.T.reshape(frames * channels)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(channels))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.tobytes())


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None

    if start is not None:
        runs.append((start, int(mask.size) - 1))

    return runs


def _to_channel_first(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform[None, :]
    if waveform.ndim == 2:
        return waveform
    raise ValueError(f"Unsupported waveform rank: {waveform.ndim}")


def compute_silence_fraction(
    waveform: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    frame_threshold_db: float,
) -> float:
    if waveform.size == 0:
        return 1.0
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return 1.0
    peak = float(np.max(rms))
    if peak <= EPS:
        return 1.0
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=peak)
    return float(np.mean(rms_db < frame_threshold_db))


def detect_track_segments(
    audio_path: str | Path,
    *,
    analysis_sr: int = 8_000,
    frame_length: int = 4_096,
    hop_length: int = 1_024,
    min_silence_seconds: float = 1.5,
    silence_threshold_db: float = -32.0,
    min_track_seconds: float = 30.0,
    res_type: str = "soxr_hq",
) -> list[SegmentInterval]:
    waveform, sr = librosa.load(
        audio_path,
        sr=analysis_sr,
        mono=True,
        res_type=res_type,
    )
    total_duration = float(len(waveform) / sr) if sr > 0 else 0.0
    if total_duration <= 0.0:
        return []

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    if rms.size == 0:
        return [
            SegmentInterval(
                track_index=1,
                start_seconds=0.0,
                end_seconds=total_duration,
                duration_seconds=total_duration,
            )
        ]

    rms_db = librosa.amplitude_to_db(np.maximum(rms, EPS), ref=np.max)
    quiet_mask = rms_db <= silence_threshold_db
    frame_times = librosa.frames_to_time(
        np.arange(rms.size),
        sr=sr,
        hop_length=hop_length,
    )

    candidate_boundaries: list[float] = []
    for start_frame, end_frame in _contiguous_true_runs(quiet_mask):
        start_time = float(frame_times[start_frame])
        end_time = float(frame_times[min(end_frame + 1, rms.size - 1)] if rms.size > 1 else total_duration)
        if end_time - start_time < min_silence_seconds:
            continue

        quietest_offset = int(np.argmin(rms[start_frame : end_frame + 1]))
        quietest_frame = start_frame + quietest_offset
        candidate_boundaries.append(float(frame_times[quietest_frame]))

    boundaries = [0.0]
    for boundary in sorted(candidate_boundaries):
        if boundary - boundaries[-1] < min_track_seconds:
            continue
        if total_duration - boundary < min_track_seconds:
            continue
        boundaries.append(boundary)
    boundaries.append(total_duration)

    segments: list[SegmentInterval] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False), start=1):
        duration = float(end - start)
        if duration <= 0.0:
            continue
        segments.append(
            SegmentInterval(
                track_index=index,
                start_seconds=float(start),
                end_seconds=float(end),
                duration_seconds=duration,
            )
        )

    return segments


def export_track_segments(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    analysis_sr: int = 8_000,
    frame_length: int = 4_096,
    hop_length: int = 1_024,
    min_silence_seconds: float = 1.5,
    silence_threshold_db: float = -32.0,
    min_track_seconds: float = 30.0,
    output_sr: int = 32_000,
    mono: bool = True,
    res_type: str = "soxr_hq",
    max_silence_fraction: float = 0.85,
    silence_frame_db: float = -35.0,
) -> list[dict[str, float | int | str]]:
    source_path = Path(audio_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    segments = detect_track_segments(
        source_path,
        analysis_sr=analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        min_silence_seconds=min_silence_seconds,
        silence_threshold_db=silence_threshold_db,
        min_track_seconds=min_track_seconds,
        res_type=res_type,
    )

    rows: list[dict[str, float | int | str]] = []
    for segment in segments:
        waveform, loaded_sr = librosa.load(
            source_path,
            sr=output_sr,
            mono=mono,
            offset=segment.start_seconds,
            duration=segment.duration_seconds,
            res_type=res_type,
        )

        mono_for_silence = waveform if waveform.ndim == 1 else np.mean(waveform, axis=0)
        silence_fraction = compute_silence_fraction(
            mono_for_silence,
            frame_length=frame_length,
            hop_length=hop_length,
            frame_threshold_db=silence_frame_db,
        )
        if silence_fraction > max_silence_fraction:
            continue

        segment_name = f"{segment.track_index:03d}.wav"
        segment_path = destination / segment_name
        _write_wav(segment_path, _to_channel_first(waveform), loaded_sr)

        rows.append(
            {
                "source_path": str(source_path),
                "segment_path": str(segment_path),
                **segment.to_dict(),
                "silence_fraction": silence_fraction,
            }
        )

    return rows
