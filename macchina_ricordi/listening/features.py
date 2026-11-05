from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import librosa
import numpy as np

from ..scoring.features import EPS, summarize_waveform
from ..scoring.reward import score_arrhythmic_percussive_waveform
from .describe import describe_feature_summary, load_default_thresholds
from .events import detect_notable_events
from .schema import (
    ANALYSIS_VERSION,
    SCHEMA_VERSION,
    AnalysisInfo,
    AnalysisParameters,
    AudioInfo,
    ListeningDossier,
    ListeningSummary,
    ProvenanceInfo,
    Sketches,
    Timeline,
    TimelineWindow,
    WholeClip,
)
from .sketch import render_numeric_spectrogram, render_spectral_sketch
from .timeline import build_timeline_regions, label_window, summarize_time_contour


def _duration_seconds(waveform: np.ndarray, sr: int) -> float:
    return float(len(waveform) / sr) if sr > 0 else 0.0


def _silence_fraction(waveform: np.ndarray, *, frame_length: int, hop_length: int) -> float:
    if waveform.size == 0:
        return 1.0
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return 1.0
    peak = float(np.max(rms))
    if peak <= EPS:
        return 1.0
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=peak)
    return float(np.mean(rms_db < -35.0))


def _band_ratios(waveform: np.ndarray, *, sr: int, n_fft: int, hop_length: int) -> tuple[float, float, float]:
    if waveform.size == 0:
        return 0.0, 0.0, 0.0
    spectrum = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)) ** 2
    if spectrum.size == 0:
        return 0.0, 0.0, 0.0
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low = frequencies < 250.0
    mid = (frequencies >= 250.0) & (frequencies < 2_000.0)
    high = frequencies >= 2_000.0
    total = float(np.sum(spectrum))
    if total <= EPS:
        return 0.0, 0.0, 0.0
    return (
        float(np.sum(spectrum[low]) / total),
        float(np.sum(spectrum[mid]) / total),
        float(np.sum(spectrum[high]) / total),
    )


def _spectral_flux_stats(waveform: np.ndarray, *, sr: int, n_fft: int, hop_length: int) -> tuple[float, float]:
    if waveform.size == 0:
        return 0.0, 0.0
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=32,
        power=2.0,
    )
    if mel.shape[1] <= 1:
        return 0.0, 0.0
    log_mel = librosa.power_to_db(mel + EPS, ref=np.max)
    frame_changes = np.mean(np.abs(np.diff(log_mel, axis=1)), axis=0)
    if frame_changes.size == 0:
        return 0.0, 0.0
    return float(np.mean(frame_changes)), float(np.percentile(frame_changes, 90.0))


def _window_features(waveform: np.ndarray, *, sr: int, hop_length: int) -> dict[str, float]:
    n_fft = max(2, min(len(waveform), max(1_024, hop_length * 2)))
    summary = summarize_waveform(waveform, sr=sr, hop_length=hop_length)
    low_ratio, mid_ratio, high_ratio = _band_ratios(waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_flux_mean, spectral_flux_p90 = _spectral_flux_stats(
        waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return {
        "duration_seconds": summary.duration_seconds,
        "rms_mean": summary.rms_mean,
        "peak_amplitude": summary.peak_amplitude,
        "spectral_centroid_mean": summary.spectral_centroid_mean,
        "spectral_bandwidth_mean": summary.spectral_bandwidth_mean,
        "spectral_flatness_mean": summary.spectral_flatness_mean,
        "onset_density": summary.onset_density,
        "pulse_instability": summary.pulse_instability,
        "low_band_ratio": low_ratio,
        "mid_band_ratio": mid_ratio,
        "high_band_ratio": high_ratio,
        "spectral_flux_mean": spectral_flux_mean,
        "spectral_flux_p90": spectral_flux_p90,
        "silence_fraction": _silence_fraction(waveform, frame_length=n_fft, hop_length=hop_length),
    }


def _build_windows(
    waveform: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    window_seconds: float,
    window_hop_seconds: float,
) -> list[TimelineWindow]:
    duration = _duration_seconds(waveform, sr)
    if duration <= 0.0:
        features = _window_features(waveform, sr=sr, hop_length=hop_length)
        return [TimelineWindow(start_seconds=0.0, end_seconds=0.0, label=label_window(features), features=features)]

    window_samples = max(1, int(round(window_seconds * sr)))
    hop_samples = max(1, int(round(window_hop_seconds * sr)))
    starts = list(range(0, max(len(waveform) - window_samples, 0) + 1, hop_samples))
    if not starts:
        starts = [0]
    last_start = max(len(waveform) - window_samples, 0)
    if starts[-1] != last_start and duration > window_seconds:
        starts.append(last_start)

    windows: list[TimelineWindow] = []
    for start in starts:
        end = min(start + window_samples, len(waveform))
        segment = waveform[start:end]
        features = _window_features(segment, sr=sr, hop_length=hop_length)
        windows.append(
            TimelineWindow(
                start_seconds=float(start / sr),
                end_seconds=float(end / sr),
                label=label_window(features),
                features=features,
            )
        )
    return windows


def _global_summary(description: dict[str, str], reward: dict[str, object], features: dict[str, float]) -> str:
    return (
        f"A {description['loudness']}, {description['brightness']} sound with {description['attack']} attack, "
        f"{description['noisiness']} texture, {description['density']} events, {description['pulse']} pulse, "
        f"and {description['resonance']} resonance. The arrhythmic/percussive score is "
        f"{float(reward.get('score', 0.0)):.3f}, with silence covering {features['silence_fraction']:.2%} of the clip."
    )


def analyze_waveform(
    waveform: np.ndarray,
    *,
    path: str | Path,
    sr: int,
    hop_length: int = 512,
    res_type: str = "soxr_hq",
    tempo_method: Literal["autocorr", "tempogram"] = "tempogram",
    window_seconds: float = 2.0,
    window_hop_seconds: float = 1.0,
) -> ListeningDossier:
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D mono array")

    path = Path(path).expanduser().resolve()
    parameters = AnalysisParameters(
        sr=sr,
        hop_length=hop_length,
        res_type=res_type,
        tempo_method=tempo_method,
        window_seconds=window_seconds,
        window_hop_seconds=window_hop_seconds,
    )
    analysis = AnalysisInfo(
        version=ANALYSIS_VERSION,
        analyzed_at=datetime.now(UTC).isoformat(),
        parameters=parameters,
    )
    reward = score_arrhythmic_percussive_waveform(
        waveform,
        sr=sr,
        hop_length=hop_length,
        tempo_penalty_method=tempo_method,
    )
    load_default_thresholds()
    raw_summary = summarize_waveform(waveform, sr=sr, hop_length=hop_length)
    description = describe_feature_summary(raw_summary)
    whole_features = asdict(raw_summary)
    whole_features["silence_fraction"] = reward.silence_fraction
    whole_features["max_silence_gap_seconds"] = reward.max_silence_gap_seconds

    windows = _build_windows(
        waveform,
        sr=sr,
        hop_length=hop_length,
        window_seconds=window_seconds,
        window_hop_seconds=window_hop_seconds,
    )
    regions = build_timeline_regions(windows)
    time_contour = summarize_time_contour(windows, regions)
    notable_events = detect_notable_events(waveform, sr=sr, hop_length=hop_length)
    return ListeningDossier(
        schema=SCHEMA_VERSION,
        analysis=analysis,
        audio=AudioInfo(
            path=str(path),
            duration_seconds=_duration_seconds(waveform, sr),
            sample_rate=sr,
            channels=1,
            mono=True,
        ),
        provenance=ProvenanceInfo(
            source="file",
            audio_path=str(path),
        ),
        whole_clip=WholeClip(
            reward=reward.to_dict(),
            description=description.to_dict(),
            features=whole_features,
        ),
        timeline=Timeline(
            window_seconds=window_seconds,
            window_hop_seconds=window_hop_seconds,
            windows=windows,
            regions=regions,
        ),
        sketches=Sketches(
            spectral=render_spectral_sketch(windows),
            numeric_spectrogram=render_numeric_spectrogram(waveform, sr=sr),
        ),
        summary=ListeningSummary(
            global_summary=_global_summary(description.to_dict(), reward.to_dict(), whole_features),
            time_contour=time_contour,
        ),
        warnings=[],
        notable_events=notable_events,
    )


def analyze_audio(
    path: str | Path,
    *,
    sr: int = 32_000,
    hop_length: int = 512,
    res_type: str = "soxr_hq",
    tempo_method: Literal["autocorr", "tempogram"] = "tempogram",
    window_seconds: float = 2.0,
    window_hop_seconds: float = 1.0,
) -> ListeningDossier:
    audio_path = Path(path).expanduser().resolve()
    waveform, loaded_sr = librosa.load(audio_path, sr=sr, mono=True, res_type=res_type)
    return analyze_waveform(
        waveform,
        path=audio_path,
        sr=int(loaded_sr),
        hop_length=hop_length,
        res_type=res_type,
        tempo_method=tempo_method,
        window_seconds=window_seconds,
        window_hop_seconds=window_hop_seconds,
    )
