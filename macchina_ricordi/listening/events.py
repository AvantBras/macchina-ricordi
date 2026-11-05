from __future__ import annotations

import librosa
import numpy as np

from ..scoring.features import analyze_onsets
from ..segmentation import _contiguous_true_runs
from .schema import NotableEvent

EPS = 1e-8


def _silence_frame_mask(
    waveform: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    frame_threshold_db: float,
) -> np.ndarray:
    if waveform.size == 0:
        return np.asarray([], dtype=bool)
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return np.asarray([], dtype=bool)
    peak = float(np.max(rms))
    if peak <= EPS:
        return np.ones(rms.size, dtype=bool)
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=peak)
    return rms_db < frame_threshold_db


def detect_notable_events(
    waveform: np.ndarray,
    *,
    sr: int,
    hop_length: int = 512,
    max_events: int = 12,
    min_silence_seconds: float = 0.5,
    silence_frame_db: float = -35.0,
    silence_frame_length: int = 4_096,
    silence_hop_length: int = 1_024,
) -> list[NotableEvent]:
    if waveform.size == 0 or sr <= 0:
        return []

    events: list[NotableEvent] = []

    onsets = analyze_onsets(waveform, sr=sr, hop_length=hop_length)
    if onsets.times.size > 0 and onsets.envelope.size > 0:
        envelope_ref = float(np.percentile(onsets.envelope, 99.0))
        if envelope_ref <= EPS:
            envelope_ref = float(np.max(onsets.envelope)) or 1.0
        for frame, time in zip(onsets.frames, onsets.times, strict=False):
            if frame < 0 or frame >= onsets.envelope.size:
                continue
            raw_strength = float(onsets.envelope[frame])
            normalized = max(0.0, min(raw_strength / envelope_ref, 1.0))
            events.append(
                NotableEvent(
                    time_seconds=float(time),
                    end_seconds=None,
                    kind="onset",
                    strength=normalized,
                    detail="strong onset",
                )
            )

    silence_mask = _silence_frame_mask(
        waveform,
        frame_length=silence_frame_length,
        hop_length=silence_hop_length,
        frame_threshold_db=silence_frame_db,
    )
    if silence_mask.size > 0:
        frame_times = librosa.frames_to_time(
            np.arange(silence_mask.size),
            sr=sr,
            hop_length=silence_hop_length,
        )
        total_duration = float(waveform.size / sr)
        for start_frame, end_frame in _contiguous_true_runs(silence_mask):
            start_time = float(frame_times[start_frame])
            if end_frame + 1 < frame_times.size:
                end_time = float(frame_times[end_frame + 1])
            else:
                end_time = total_duration
            if end_time - start_time < min_silence_seconds:
                continue
            events.append(
                NotableEvent(
                    time_seconds=start_time,
                    end_seconds=end_time,
                    kind="silence_start",
                    strength=1.0,
                    detail="silence begins",
                )
            )
            events.append(
                NotableEvent(
                    time_seconds=end_time,
                    end_seconds=None,
                    kind="silence_end",
                    strength=1.0,
                    detail="silence ends",
                )
            )

    if len(events) <= max_events:
        events.sort(key=lambda event: event.time_seconds)
        return events

    silence_events = [event for event in events if event.kind != "onset"]
    onset_events = sorted(
        (event for event in events if event.kind == "onset"),
        key=lambda event: event.strength,
        reverse=True,
    )
    remaining = max_events - len(silence_events)
    if remaining < 0:
        kept = silence_events[:max_events]
    else:
        kept = silence_events + onset_events[:remaining]
    kept.sort(key=lambda event: event.time_seconds)
    return kept
