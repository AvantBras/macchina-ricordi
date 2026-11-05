from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .reward import score_arrhythmic_percussive_waveform

EPS = 1e-8

# Per-field min-max bounds for rhythm feature normalisation.
_RHYTHM_BOUNDS: dict[str, tuple[float, float]] = {
    "onset_density": (0.0, 25.0),
    "tempo_instability": (0.0, 3.0),
    "percussive_onset_density": (0.0, 25.0),
    "percussive_tempo_instability": (0.0, 3.0),
    "percussive_energy_ratio": (0.0, 1.0),
    "tempogram_peak_ratio": (0.0, 1.0),
    "tempogram_peak_sharpness": (0.0, 1.0),
}

# ``dominant_tempo_consistency`` was dropped 2026-08-06: 87% of rounds in the
# stream corpus sit pinned at 1.0, so it contributed a near-constant dimension.
# The scorer still computes it — this is about what the *similarity* vector pools.

_RHYTHM_FIELDS = list(_RHYTHM_BOUNDS.keys())

# Z-score calibration loaded lazily.  ``None`` means uncalibrated
# (raw cosine).  Set via ``load_calibration`` or ``calibrate_from_dir``.
_CALIBRATION: dict[str, dict[str, np.ndarray]] | None = None


@dataclass(frozen=True)
class SimilarityComponents:
    mfcc_cosine: float = 0.0
    spectral_cosine: float = 0.0
    rhythm_cosine: float = 0.0
    combined: float = 0.0
    self_similarity_a: float | None = None
    self_similarity_b: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < EPS or norm_b < EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _zscore_normalize(vec: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (vec - means) / np.maximum(stds, EPS)


def _percentiles_stacked(stacked: np.ndarray) -> np.ndarray:
    """Return p10, p50, p90 per row, concatenated into a flat vector."""
    if stacked.size == 0 or stacked.shape[1] == 0:
        return np.zeros(stacked.shape[0] * 3, dtype=np.float64)
    p10 = np.percentile(stacked, 10, axis=1)
    p50 = np.percentile(stacked, 50, axis=1)
    p90 = np.percentile(stacked, 90, axis=1)
    return np.concatenate([p10, p50, p90])


def _calibrated_cosine(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    key: str,
) -> float:
    """Cosine similarity, optionally z-scored if calibration is loaded."""
    if _CALIBRATION is not None and key in _CALIBRATION:
        means, stds = _CALIBRATION[key]["means"], _CALIBRATION[key]["stds"]
        vec_a = _zscore_normalize(vec_a, means, stds)
        vec_b = _zscore_normalize(vec_b, means, stds)
    return _cosine_similarity(vec_a, vec_b)


# ── feature vectors ───────────────────────────────────────────────────


def _mfcc_stats_vector(waveform: np.ndarray, *, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Aggregate MFCC + delta + delta2 frames via p10/p50/p90 per coefficient.

    Returns a flat vector of length ``3 * n_mfcc * 3`` (117 with defaults).
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    if mfcc.size == 0 or mfcc.shape[1] <= 1:
        return np.zeros(n_mfcc * 9, dtype=np.float64)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta, delta2])
    return _percentiles_stacked(stacked)


def _spectral_stats_vector(waveform: np.ndarray, *, sr: int, n_mels: int = 32) -> np.ndarray:
    """Aggregate mel-spectrogram (mean per band) + spectral contrast (mean per band).

    Returns a flat vector of length ``n_mels + (n_bands + 1)`` (39 with defaults).
    """
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6)

    mel_means = np.mean(mel, axis=1) if mel.shape[1] > 0 else np.zeros(n_mels)
    c_means = np.mean(contrast, axis=1) if contrast.shape[1] > 0 else np.zeros(7)

    return np.concatenate([mel_means, c_means]).astype(np.float64)


def _rhythm_features_vector(waveform: np.ndarray, *, sr: int) -> np.ndarray:
    """Extract rhythm features via the existing arrhythmicity scorer and min-max normalise."""
    if waveform.size == 0:
        return np.zeros(len(_RHYTHM_FIELDS), dtype=np.float64)

    components = score_arrhythmic_percussive_waveform(waveform, sr=sr)
    d = components.to_dict()

    values: list[float] = []
    for field in _RHYTHM_FIELDS:
        raw = float(d.get(field, 0.0))
        lo, hi = _RHYTHM_BOUNDS[field]
        if hi - lo < EPS:
            values.append(0.0)
        else:
            values.append(float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0)))

    return np.asarray(values, dtype=np.float64)


# ── calibration ───────────────────────────────────────────────────────


def calibrate_from_dir(directory: str | Path, *, sr: int = 16_000, max_files: int = 200) -> dict[str, Any]:
    """Compute feature-wise means and stds from a corpus of audio files.

    Returns a calibration dict suitable for ``load_calibration``.
    Scans ``directory`` recursively for WAV/FLAC/MP3 files.
    """
    from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files

    directory = Path(directory).expanduser().resolve()
    files = list(iter_audio_files(directory, recursive=True, extensions=DEFAULT_EXTENSIONS))
    if max_files and len(files) > max_files:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in indices]

    accum: dict[str, list[np.ndarray]] = {"mfcc": [], "spectral": [], "rhythm": []}

    for path in files:
        try:
            waveform, _ = librosa.load(path, sr=sr, mono=True)
            if waveform.size < sr:
                continue
            accum["mfcc"].append(_mfcc_stats_vector(waveform, sr=sr))
            accum["spectral"].append(_spectral_stats_vector(waveform, sr=sr))
            accum["rhythm"].append(_rhythm_features_vector(waveform, sr=sr))
        except Exception:
            continue

    calibration: dict[str, dict[str, list[float]]] = {}
    for key, vecs in accum.items():
        if not vecs:
            continue
        matrix = np.stack(vecs, axis=0)
        calibration[key] = {
            "means": matrix.mean(axis=0).tolist(),
            "stds": matrix.std(axis=0).tolist(),
        }

    return {
        "source": str(directory),
        "num_files": len(accum["mfcc"]),
        "sr": sr,
        "calibration": calibration,
    }


def expected_dims() -> dict[str, int]:
    """Length of each feature vector, for validating a calibration against the code."""
    return {"mfcc": 13 * 9, "spectral": 32 + 7, "rhythm": len(_RHYTHM_FIELDS)}


def load_calibration(data: dict[str, Any]) -> None:
    """Install a calibration dict (from ``calibrate_from_dir`` or a JSON file).

    After calling this, all similarity computations run z-score-normalised
    cosine similarity using the supplied feature statistics.

    Blocks whose length does not match the current feature vector are skipped
    rather than installed: a calibration written before a field was added to or
    removed from a vector would otherwise broadcast-error deep inside a cosine,
    or worse, silently z-score against the wrong dimensions. Regenerate those
    with ``score sim-calibrate``.
    """
    global _CALIBRATION
    cal = data.get("calibration", data)
    dims = expected_dims()
    installed: dict[str, dict[str, np.ndarray]] = {}
    for key, block in cal.items():
        if key not in dims:
            continue
        means = np.asarray(block["means"], dtype=np.float64)
        stds = np.asarray(block["stds"], dtype=np.float64)
        if means.shape != stds.shape or means.size != dims[key]:
            continue
        installed[key] = {"means": means, "stds": stds}
    _CALIBRATION = installed


def clear_calibration() -> None:
    """Revert to raw (uncalibrated) cosine similarity."""
    global _CALIBRATION
    _CALIBRATION = None


def calibration_loaded() -> bool:
    return _CALIBRATION is not None


# ── public API ────────────────────────────────────────────────────────


def feature_vectors(waveform: np.ndarray, *, sr: int = 16_000) -> dict[str, np.ndarray]:
    """The three vectors similarity compares (MFCC, spectral and rhythm statistics) for one mono waveform.

    The listening archive stores these per clip, so ``score similar`` and ``score sim`` compare the same thing.
    """
    return {
        "mfcc": _mfcc_stats_vector(waveform, sr=sr),
        "spectral": _spectral_stats_vector(waveform, sr=sr),
        "rhythm": _rhythm_features_vector(waveform, sr=sr),
    }


def similarity_from_vectors(vectors_a: Mapping[str, Any], vectors_b: Mapping[str, Any]) -> SimilarityComponents:
    """Compare two sets of ``feature_vectors``: one cosine per group (z-scored when calibrated), and their mean."""
    cosines = {
        key: _calibrated_cosine(
            np.asarray(vectors_a.get(key, []), dtype=float), np.asarray(vectors_b.get(key, []), dtype=float), key=key
        )
        for key in ("mfcc", "spectral", "rhythm")
    }
    return SimilarityComponents(
        mfcc_cosine=float(cosines["mfcc"]),
        spectral_cosine=float(cosines["spectral"]),
        rhythm_cosine=float(cosines["rhythm"]),
        combined=float(sum(cosines.values()) / 3.0),
    )


def compute_similarity_waveform(
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    *,
    sr: int = 16_000,
) -> SimilarityComponents:
    """Compute multi-axis cosine similarity between two raw 1-D mono waveforms."""
    if waveform_a.ndim != 1 or waveform_b.ndim != 1:
        raise ValueError("waveforms must be 1D mono arrays")
    return similarity_from_vectors(feature_vectors(waveform_a, sr=sr), feature_vectors(waveform_b, sr=sr))


def compute_similarity(
    path_a: str | Path,
    path_b: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    res_type: str = "soxr_qq",
) -> SimilarityComponents:
    """Load two audio files and compute cross-length similarity."""
    waveform_a, loaded_sr_a = librosa.load(path_a, sr=sr, mono=mono, res_type=res_type)
    waveform_b, loaded_sr_b = librosa.load(path_b, sr=sr, mono=mono, res_type=res_type)
    return compute_similarity_waveform(waveform_a, waveform_b, sr=int(max(loaded_sr_a, loaded_sr_b)))


def compute_self_similarity(
    path: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    res_type: str = "soxr_qq",
) -> SimilarityComponents:
    """Split an audio file in half and measure similarity between the two halves."""
    waveform, loaded_sr = librosa.load(path, sr=sr, mono=mono, res_type=res_type)
    mid = len(waveform) // 2
    if mid < loaded_sr:
        return SimilarityComponents(combined=1.0)
    return compute_similarity_waveform(waveform[:mid], waveform[mid:], sr=int(loaded_sr))


def compute_similarity_with_baseline(
    path_a: str | Path,
    path_b: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    res_type: str = "soxr_qq",
) -> SimilarityComponents:
    """Compute cross-similarity between two files and their self-similarity baselines."""
    cross = compute_similarity(path_a, path_b, sr=sr, mono=mono, res_type=res_type)
    self_a = compute_self_similarity(path_a, sr=sr, mono=mono, res_type=res_type)
    self_b = compute_self_similarity(path_b, sr=sr, mono=mono, res_type=res_type)
    return SimilarityComponents(
        mfcc_cosine=cross.mfcc_cosine,
        spectral_cosine=cross.spectral_cosine,
        rhythm_cosine=cross.rhythm_cosine,
        combined=cross.combined,
        self_similarity_a=self_a.combined,
        self_similarity_b=self_b.combined,
    )
