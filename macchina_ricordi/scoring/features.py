from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

EPS = 1e-8


@dataclass(frozen=True)
class OnsetFeatures:
    envelope: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    count: int
    density: float
    instability: float
    strength_mean: float
    strength_std: float
    strength_p90: float
    strong_onset_fraction: float


@dataclass(frozen=True)
class AudioFeatureSummary:
    duration_seconds: float
    rms_mean: float
    peak_amplitude: float
    attack_db: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_flatness_mean: float
    resonance_ratio: float
    onset_density: float
    pulse_instability: float


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def ioi_coefficient_of_variation(onset_times: np.ndarray) -> float:
    if onset_times.size < 3:
        return 0.0

    ioi = np.diff(onset_times)
    ioi = ioi[ioi > 1e-3]
    if ioi.size < 2:
        return 0.0

    mean_ioi = float(np.mean(ioi))
    if mean_ioi <= EPS:
        return 0.0
    return float(np.std(ioi) / mean_ioi)


def duration_seconds(waveform: np.ndarray, sr: int) -> float:
    return float(len(waveform) / sr) if sr > 0 else 0.0


def validate_mono_waveform(waveform: np.ndarray, *, name: str = "waveform") -> None:
    if waveform.ndim != 1:
        raise ValueError(f"{name} must be a 1D mono array")


def analyze_onsets(
    waveform: np.ndarray,
    *,
    sr: int,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> OnsetFeatures:
    validate_mono_waveform(waveform)
    duration = duration_seconds(waveform, sr)
    if duration <= 0.0:
        return OnsetFeatures(
            envelope=np.asarray([], dtype=np.float64),
            frames=np.asarray([], dtype=np.int64),
            times=np.asarray([], dtype=np.float64),
            count=0,
            density=0.0,
            instability=0.0,
            strength_mean=0.0,
            strength_std=0.0,
            strength_p90=0.0,
            strong_onset_fraction=0.0,
        )

    onset_kwargs: dict[str, object] = {
        "y": waveform,
        "sr": sr,
        "hop_length": hop_length,
    }
    if n_fft is not None:
        onset_kwargs["n_fft"] = n_fft
    envelope = librosa.onset.onset_strength(**onset_kwargs)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )
    times = librosa.frames_to_time(
        frames,
        sr=sr,
        hop_length=hop_length,
    )
    count = int(len(times))
    density = float(count / max(duration, EPS))
    strength_mean = float(np.mean(envelope)) if envelope.size else 0.0
    strength_std = float(np.std(envelope)) if envelope.size else 0.0
    strength_p90 = safe_percentile(envelope, 90.0)
    strong_threshold = strength_mean + strength_std
    strong_onset_fraction = float(np.mean(envelope >= strong_threshold)) if envelope.size else 0.0

    return OnsetFeatures(
        envelope=envelope,
        frames=frames,
        times=times,
        count=count,
        density=density,
        instability=ioi_coefficient_of_variation(times),
        strength_mean=strength_mean,
        strength_std=strength_std,
        strength_p90=strength_p90,
        strong_onset_fraction=strong_onset_fraction,
    )


def summarize_waveform(
    waveform: np.ndarray,
    *,
    sr: int,
    hop_length: int = 512,
) -> AudioFeatureSummary:
    validate_mono_waveform(waveform)
    duration = duration_seconds(waveform, sr)
    if waveform.size == 0 or duration <= 0.0:
        return AudioFeatureSummary(
            duration_seconds=duration,
            rms_mean=0.0,
            peak_amplitude=0.0,
            attack_db=0.0,
            spectral_centroid_mean=0.0,
            spectral_bandwidth_mean=0.0,
            spectral_flatness_mean=0.0,
            resonance_ratio=0.0,
            onset_density=0.0,
            pulse_instability=0.0,
        )

    rms = librosa.feature.rms(y=waveform)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    peak_amplitude = float(np.max(np.abs(waveform)))
    if rms_mean < EPS or peak_amplitude < EPS:
        attack_db = 0.0
    else:
        attack_db = float(20.0 * np.log10(peak_amplitude / rms_mean))

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=waveform)[0]
    spectral_centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0
    spectral_bandwidth_mean = float(np.mean(bandwidth)) if bandwidth.size else 0.0
    spectral_flatness_mean = float(np.mean(flatness)) if flatness.size else 0.0
    if spectral_centroid_mean < EPS:
        resonance_ratio = 1.0
    else:
        resonance_ratio = float(np.clip(spectral_bandwidth_mean / spectral_centroid_mean, 0.0, 2.0))

    onsets = analyze_onsets(waveform, sr=sr, hop_length=hop_length)
    return AudioFeatureSummary(
        duration_seconds=duration,
        rms_mean=rms_mean,
        peak_amplitude=peak_amplitude,
        attack_db=attack_db,
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_bandwidth_mean=spectral_bandwidth_mean,
        spectral_flatness_mean=spectral_flatness_mean,
        resonance_ratio=resonance_ratio,
        onset_density=onsets.density,
        pulse_instability=onsets.instability,
    )
