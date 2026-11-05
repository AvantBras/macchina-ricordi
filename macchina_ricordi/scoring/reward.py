from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np

from .features import (
    EPS,
    analyze_onsets,
    ioi_coefficient_of_variation,
    safe_percentile,
    validate_mono_waveform,
)
from .features import duration_seconds as _duration_seconds

AUXILIARY_FEATURE_SR = 8_000


@dataclass(frozen=True)
class RewardComponents:
    score: float = 0.0
    tempo_instability: float = 0.0
    onset_density: float = 0.0
    tempo_penalty: float = 0.0
    duration_seconds: float = 0.0
    onset_count: int = 0
    percussive_energy_ratio: float = 0.0
    onset_strength_mean: float = 0.0
    onset_strength_std: float = 0.0
    onset_strength_p90: float = 0.0
    strong_onset_fraction: float = 0.0
    percussive_onset_strength_mean: float = 0.0
    percussive_onset_strength_p90: float = 0.0
    percussive_onset_density: float = 0.0
    percussive_tempo_instability: float = 0.0
    band_onset_density_low: float = 0.0
    band_onset_density_mid: float = 0.0
    band_onset_density_high: float = 0.0
    onset_high_frequency_ratio: float = 0.0
    tempogram_peak_ratio: float = 0.0
    tempogram_peak_sharpness: float = 0.0
    dominant_tempo_consistency: float = 0.0
    windowed_tempo_instability_mean: float = 0.0
    windowed_tempo_instability_std: float = 0.0
    silence_fraction: float = 0.0
    max_silence_gap_seconds: float = 0.0
    spectral_change_mean: float = 0.0
    spectral_change_p90: float = 0.0
    spectral_change_event_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return safe_percentile(values, percentile)


def _ioi_coefficient_of_variation(onset_times: np.ndarray) -> float:
    return ioi_coefficient_of_variation(onset_times)


def _windowed_instability_stats(
    onset_times: np.ndarray,
    *,
    duration_seconds: float,
    window_seconds: float = 8.0,
    step_seconds: float = 4.0,
) -> tuple[float, float]:
    if onset_times.size < 3 or duration_seconds <= 0.0:
        return 0.0, 0.0

    if duration_seconds <= window_seconds:
        values = np.asarray([_ioi_coefficient_of_variation(onset_times)], dtype=np.float64)
    else:
        starts = np.arange(0.0, max(duration_seconds - window_seconds, 0.0) + step_seconds, step_seconds)
        samples: list[float] = []
        for start in starts:
            end = start + window_seconds
            local = onset_times[(onset_times >= start) & (onset_times < end)]
            if local.size >= 3:
                samples.append(_ioi_coefficient_of_variation(local))
        values = np.asarray(samples, dtype=np.float64)

    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _tempogram_summary(
    onset_envelope: np.ndarray,
    *,
    sr: int,
    hop_length: int,
) -> tuple[float, float, float]:
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
    )
    if tempogram.size == 0 or tempogram.shape[0] <= 1 or tempogram.shape[1] == 0:
        return 0.0, 0.0, 0.0

    non_dc = np.clip(tempogram[1:], a_min=0.0, a_max=None)
    frame_energy = np.sum(non_dc, axis=0)
    valid = frame_energy > EPS
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    active = non_dc[:, valid]
    active_energy = frame_energy[valid]
    dominant = np.max(active, axis=0)
    peak_ratio = float(np.mean(dominant / (active_energy + EPS)))

    if active.shape[0] >= 2:
        top2 = np.partition(active, kth=max(active.shape[0] - 2, 0), axis=0)[-2:]
        sharpness = float(np.mean((top2[-1] - top2[-2]) / (top2[-1] + EPS)))
    else:
        sharpness = 0.0

    dominant_bins = np.argmax(active, axis=0)
    bin_counts = np.bincount(dominant_bins)
    consistency = float(np.max(bin_counts) / max(len(dominant_bins), 1))
    return peak_ratio, sharpness, consistency


def _tempo_clarity_penalty_tempogram_from_tempogram(tempogram: np.ndarray) -> float:
    if tempogram.size == 0 or tempogram.shape[0] <= 1:
        return 0.0

    mean_tempogram = np.mean(tempogram, axis=1)
    non_dc = np.clip(mean_tempogram[1:], a_min=0.0, a_max=None)
    total_energy = float(np.sum(non_dc))
    if total_energy <= EPS:
        return 0.0

    dominant_peak = float(np.max(non_dc))
    return dominant_peak / (total_energy + EPS)


def _tempo_clarity_penalty_tempogram(
    onset_envelope: np.ndarray,
    *,
    sr: int,
    hop_length: int,
) -> float:
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
    )
    return _tempo_clarity_penalty_tempogram_from_tempogram(tempogram)


def _tempo_clarity_penalty_autocorr(
    onset_envelope: np.ndarray,
    *,
    max_size: int = 2048,
) -> float:
    if onset_envelope.size <= 1:
        return 0.0

    ac = librosa.autocorrelate(
        onset_envelope,
        max_size=min(max_size, int(onset_envelope.size)),
    )
    if ac.size <= 1:
        return 0.0

    non_dc = np.clip(ac[1:], a_min=0.0, a_max=None)
    total_energy = float(np.sum(non_dc))
    if total_energy <= EPS:
        return 0.0

    dominant_peak = float(np.max(non_dc))
    return dominant_peak / (total_energy + EPS)


def _band_onset_densities(
    waveform: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    n_fft: int,
    duration_seconds: float,
) -> tuple[float, float, float]:
    if duration_seconds <= 0.0:
        return 0.0, 0.0, 0.0

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=96,
        power=2.0,
    )
    if mel.size == 0:
        return 0.0, 0.0, 0.0

    log_mel = librosa.power_to_db(mel + EPS, ref=np.max)
    band_edges = np.linspace(0, log_mel.shape[0], num=4, dtype=int)
    channels = [
        slice(int(band_edges[0]), int(band_edges[1])),
        slice(int(band_edges[1]), int(band_edges[2])),
        slice(int(band_edges[2]), int(band_edges[3])),
    ]
    multi = librosa.onset.onset_strength_multi(
        S=log_mel,
        sr=sr,
        hop_length=hop_length,
        channels=channels,
        detrend=False,
        center=True,
    )
    if multi.ndim == 1:
        multi = multi[np.newaxis, :]

    densities: list[float] = []
    for envelope in multi:
        frames = librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=sr,
            hop_length=hop_length,
            units="frames",
        )
        densities.append(float(len(frames) / max(duration_seconds, EPS)))
    while len(densities) < 3:
        densities.append(0.0)
    return densities[0], densities[1], densities[2]


def _silence_stats(
    waveform: np.ndarray,
    *,
    sr: int,
    frame_length: int,
    hop_length: int,
    silence_threshold_db: float = -35.0,
) -> tuple[float, float]:
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return 0.0, 0.0

    peak_rms = float(np.max(rms))
    if peak_rms <= EPS:
        total_span = float(len(rms) * hop_length / max(sr, 1))
        return 1.0, total_span

    ref = peak_rms
    rms_db = librosa.amplitude_to_db(rms + EPS, ref=ref)
    silence_mask = rms_db < silence_threshold_db
    silence_fraction = float(np.mean(silence_mask))

    max_run = 0
    current_run = 0
    for is_silent in silence_mask:
        if is_silent:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    max_gap_seconds = float(max_run * hop_length / max(sr, 1))
    return silence_fraction, max_gap_seconds


def _spectral_change_stats(
    waveform: np.ndarray,
    *,
    sr: int,
    n_fft: int,
    hop_length: int,
    duration_seconds: float,
) -> tuple[float, float, float]:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=96,
        power=2.0,
    )
    if mel.shape[1] <= 1:
        return 0.0, 0.0, 0.0

    log_mel = librosa.power_to_db(mel + EPS, ref=np.max)
    frame_changes = np.mean(np.abs(np.diff(log_mel, axis=1)), axis=0)
    if frame_changes.size == 0:
        return 0.0, 0.0, 0.0

    mean_change = float(np.mean(frame_changes))
    p90_change = _safe_percentile(frame_changes, 90.0)
    threshold = mean_change + float(np.std(frame_changes))
    event_density = float(np.sum(frame_changes >= threshold) / max(duration_seconds, EPS))
    return mean_change, p90_change, event_density


def _high_frequency_ratio(
    waveform: np.ndarray,
    *,
    sr: int,
    n_fft: int,
    hop_length: int,
    onset_frames: np.ndarray,
) -> float:
    spectrum = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)) ** 2
    if spectrum.size == 0 or spectrum.shape[1] == 0:
        return 0.0

    valid_frames = onset_frames[(onset_frames >= 0) & (onset_frames < spectrum.shape[1])]
    if valid_frames.size == 0:
        selected = spectrum
    else:
        selected = spectrum[:, valid_frames]

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_mask = frequencies >= 2_000.0
    total_energy = float(np.sum(selected))
    if total_energy <= EPS:
        return 0.0
    return float(np.sum(selected[high_mask]) / total_energy)


@dataclass(frozen=True)
class AuxiliaryFeatures:
    """The downsampled waveform and FFT geometry every auxiliary feature shares.

    Exposed rather than derived inline because `macchina_ricordi.scoring.live` computes a
    subset of these same fields for the stream, and must land on identical
    numbers. Deriving the geometry in two places is exactly how two scorers drift
    apart without anyone noticing.
    """

    waveform: np.ndarray
    sr: int
    hop_length: int
    n_fft: int


def primary_n_fft(waveform: np.ndarray, *, hop_length: int = 512) -> int:
    """FFT size for the full-rate onset analysis."""
    return max(2, min(len(waveform), max(2_048, hop_length * 4)))


def auxiliary_features(waveform: np.ndarray, *, sr: int, hop_length: int = 512) -> AuxiliaryFeatures:
    """Resample to `AUXILIARY_FEATURE_SR` and derive the shared FFT geometry."""
    feature_waveform = waveform
    feature_sr = sr
    if sr > AUXILIARY_FEATURE_SR:
        feature_waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=AUXILIARY_FEATURE_SR,
            res_type="soxr_hq",
        )
        feature_sr = AUXILIARY_FEATURE_SR
    return AuxiliaryFeatures(
        waveform=feature_waveform,
        sr=feature_sr,
        hop_length=max(128, int(round(hop_length * feature_sr / max(sr, 1)))),
        n_fft=max(2, min(len(feature_waveform), max(1_024, hop_length * 2))),
    )


def score_arrhythmic_percussive_waveform(
    waveform: np.ndarray,
    *,
    sr: int = 32_000,
    hop_length: int = 512,
    tempo_penalty_method: Literal["autocorr", "tempogram"] = "autocorr",
) -> RewardComponents:
    validate_mono_waveform(waveform)

    duration_seconds = _duration_seconds(waveform, sr)
    if duration_seconds <= 0.0:
        return RewardComponents()

    n_fft = primary_n_fft(waveform, hop_length=hop_length)

    onset_features = analyze_onsets(
        waveform,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    onset_envelope = onset_features.envelope
    onset_count = onset_features.count
    onset_density = onset_features.density
    tempo_instability = onset_features.instability
    onset_strength_mean = onset_features.strength_mean
    onset_strength_std = onset_features.strength_std
    onset_strength_p90 = onset_features.strength_p90
    strong_onset_fraction = onset_features.strong_onset_fraction

    aux = auxiliary_features(waveform, sr=sr, hop_length=hop_length)
    feature_waveform = aux.waveform
    feature_sr = aux.sr
    feature_n_fft = aux.n_fft
    feature_hop_length = aux.hop_length

    harmonic_waveform, percussive_waveform = librosa.effects.hpss(feature_waveform)
    del harmonic_waveform
    total_energy = float(np.sum(np.square(feature_waveform)))
    percussive_energy = float(np.sum(np.square(percussive_waveform)))
    percussive_energy_ratio = percussive_energy / max(total_energy, EPS)

    percussive_onset_features = analyze_onsets(
        percussive_waveform,
        sr=feature_sr,
        hop_length=feature_hop_length,
        n_fft=feature_n_fft,
    )
    percussive_onset_envelope = percussive_onset_features.envelope
    percussive_onset_frames = percussive_onset_features.frames
    percussive_onset_times = percussive_onset_features.times
    percussive_onset_density = percussive_onset_features.density
    percussive_tempo_instability = percussive_onset_features.instability
    percussive_onset_strength_mean = percussive_onset_features.strength_mean
    percussive_onset_strength_p90 = percussive_onset_features.strength_p90

    band_onset_density_low, band_onset_density_mid, band_onset_density_high = _band_onset_densities(
        percussive_waveform,
        sr=feature_sr,
        hop_length=feature_hop_length,
        n_fft=feature_n_fft,
        duration_seconds=duration_seconds,
    )

    onset_high_frequency_ratio = _high_frequency_ratio(
        percussive_waveform,
        sr=feature_sr,
        n_fft=feature_n_fft,
        hop_length=feature_hop_length,
        onset_frames=percussive_onset_frames,
    )
    tempogram_peak_ratio, tempogram_peak_sharpness, dominant_tempo_consistency = _tempogram_summary(
        percussive_onset_envelope,
        sr=feature_sr,
        hop_length=feature_hop_length,
    )
    windowed_tempo_instability_mean, windowed_tempo_instability_std = _windowed_instability_stats(
        percussive_onset_times,
        duration_seconds=duration_seconds,
    )
    silence_fraction, max_silence_gap_seconds = _silence_stats(
        feature_waveform,
        sr=feature_sr,
        frame_length=feature_n_fft,
        hop_length=feature_hop_length,
    )
    spectral_change_mean, spectral_change_p90, spectral_change_event_density = _spectral_change_stats(
        feature_waveform,
        sr=feature_sr,
        n_fft=feature_n_fft,
        hop_length=feature_hop_length,
        duration_seconds=duration_seconds,
    )

    if tempo_penalty_method == "autocorr":
        tempo_penalty = _tempo_clarity_penalty_autocorr(onset_envelope)
    elif tempo_penalty_method == "tempogram":
        tempo_penalty = _tempo_clarity_penalty_tempogram(
            onset_envelope,
            sr=sr,
            hop_length=hop_length,
        )
    else:
        raise ValueError(f"Unknown tempo_penalty_method: {tempo_penalty_method}")

    score = tempo_instability * onset_density * max(0.0, 1.0 - tempo_penalty)

    return RewardComponents(
        score=float(score),
        tempo_instability=float(tempo_instability),
        onset_density=float(onset_density),
        tempo_penalty=float(tempo_penalty),
        duration_seconds=float(duration_seconds),
        onset_count=onset_count,
        percussive_energy_ratio=float(percussive_energy_ratio),
        onset_strength_mean=float(onset_strength_mean),
        onset_strength_std=float(onset_strength_std),
        onset_strength_p90=float(onset_strength_p90),
        strong_onset_fraction=float(strong_onset_fraction),
        percussive_onset_strength_mean=float(percussive_onset_strength_mean),
        percussive_onset_strength_p90=float(percussive_onset_strength_p90),
        percussive_onset_density=float(percussive_onset_density),
        percussive_tempo_instability=float(percussive_tempo_instability),
        band_onset_density_low=float(band_onset_density_low),
        band_onset_density_mid=float(band_onset_density_mid),
        band_onset_density_high=float(band_onset_density_high),
        onset_high_frequency_ratio=float(onset_high_frequency_ratio),
        tempogram_peak_ratio=float(tempogram_peak_ratio),
        tempogram_peak_sharpness=float(tempogram_peak_sharpness),
        dominant_tempo_consistency=float(dominant_tempo_consistency),
        windowed_tempo_instability_mean=float(windowed_tempo_instability_mean),
        windowed_tempo_instability_std=float(windowed_tempo_instability_std),
        silence_fraction=float(silence_fraction),
        max_silence_gap_seconds=float(max_silence_gap_seconds),
        spectral_change_mean=float(spectral_change_mean),
        spectral_change_p90=float(spectral_change_p90),
        spectral_change_event_density=float(spectral_change_event_density),
    )


def score_arrhythmic_percussive(
    audio_path: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    hop_length: int = 512,
    res_type: str = "soxr_qq",
    tempo_penalty_method: Literal["autocorr", "tempogram"] = "autocorr",
) -> RewardComponents:
    waveform, loaded_sr = librosa.load(
        audio_path,
        sr=sr,
        mono=mono,
        res_type=res_type,
    )
    return score_arrhythmic_percussive_waveform(
        waveform,
        sr=loaded_sr,
        hop_length=hop_length,
        tempo_penalty_method=tempo_penalty_method,
    )
