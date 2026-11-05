"""Multi-axis development + self-similarity structure scoring.

``arc.py`` measures development on a single axis — total variation of *chroma*, i.e.
harmonic movement. That axis is near-blind to music whose development is rhythmic or
timbral, which is exactly how the 25 Hz / percussive models move: they scored ~0.28 on
``arc_coherence_score`` against ~0.78 for the 50 Hz deep model while sounding fine.

This module measures four development axes in parallel and adds structure — whether the
piece has *sections* and whether it *returns* to material — which plain total variation
cannot see (a random walk maximises total variation without composing anything).

Axes, all on overlapping windows over the generated tail, all in ``[0, 1]``:

- ``dev_harmonic`` — cosine distance between consecutive windows' mean chroma, computed
  on the HPSS **harmonic** component so percussion stops smearing the pitch estimate.
- ``dev_timbre`` — cosine distance between consecutive windows' mean MFCC (c1..c12).
- ``dev_dynamics`` — mean absolute level change in dB, normalised by 12 dB.
- ``dev_rhythm`` — mean absolute change of a scaled onset/percussive feature vector,
  computed on the HPSS **percussive** component.
- ``dev_top2`` — mean of the two strongest axes: a model that develops on *some* axis
  should score, without demanding it develop on all of them at once.

Structure, from a cosine self-similarity matrix (SSM) over combined chroma+MFCC frames:

- ``ssm_novelty`` — peak (p95) contrast of the Foote checkerboard novelty curve, i.e.
  within-section similarity minus cross-section similarity at the strongest boundary.
  Uniform texture gives ~0, a hard section boundary approaches 1.
- ``ssm_return`` — how much better the best *distant* match is than the average distant
  match. A drone matches everything equally and scores 0; music that comes back to
  earlier material scores high. Note this rewards *contrast*: material that recurs
  against a varied background scores higher than the same recurrence against a uniform
  one, which is the intended reading of "comes back".
- ``ssm_mean`` — average distant-frame similarity, i.e. global coherence (one voice).
- ``boundaries_per_minute`` — picked novelty peaks, as a rate.

``structure_score`` is a provisional composite, deliberately kept fittable: weights come
from ``preference.py`` once enough blind A/B judgements exist (``load_structure_weights``).
The silence gate is baked *into* the score rather than left to the caller, because every
previous analysis in this project has been burned at least once by forgetting to filter
silent clips before reading a mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .arc import _to_float

STRUCTURE_SR = 16_000
DEFAULT_WINDOW_COUNT = 8
SSM_FPS = 5.0
HOP_LENGTH = 512
_EPS = 1e-9

# Silence gate: full credit at or below ``_GATE_FULL``, zero at or above ``_GATE_ZERO``.
_GATE_FULL = 0.30
_GATE_ZERO = 0.60

# Level change (dB) that counts as a full-scale dynamic move between windows.
_DYNAMICS_REFERENCE_DB = 12.0

DEFAULT_STRUCTURE_WEIGHTS: dict[str, float] = {
    "development": 0.4,
    "novelty": 0.3,
    "return": 0.3,
}

_STRUCTURE_WEIGHTS: dict[str, float] | None = None


@dataclass(frozen=True)
class StructureComponents:
    structure_score: float = 0.0
    dev_top2: float = 0.0
    dev_harmonic: float = 0.0
    dev_timbre: float = 0.0
    dev_dynamics: float = 0.0
    dev_rhythm: float = 0.0
    ssm_novelty: float = 0.0
    ssm_return: float = 0.0
    ssm_mean: float = 0.0
    boundaries_per_minute: float = 0.0
    structure_silence_fraction: float = 0.0
    silence_gate: float = 0.0
    window_count: int = 0
    structure_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


# ── weights ───────────────────────────────────────────────────────────


def load_structure_weights(path: str | Path) -> dict[str, float]:
    """Activate composite weights fitted from blind A/B preferences."""
    import json

    global _STRUCTURE_WEIGHTS
    payload = json.loads(Path(path).expanduser().read_text())
    weights = payload.get("structure_weights", payload)
    resolved = {key: float(weights[key]) for key in DEFAULT_STRUCTURE_WEIGHTS if key in weights}
    if not resolved:
        raise ValueError(f"No structure weights found in {path}")
    _STRUCTURE_WEIGHTS = resolved
    return dict(resolved)


def clear_structure_weights() -> None:
    global _STRUCTURE_WEIGHTS
    _STRUCTURE_WEIGHTS = None


def active_structure_weights() -> dict[str, float]:
    return dict(_STRUCTURE_WEIGHTS or DEFAULT_STRUCTURE_WEIGHTS)


# ── helpers ───────────────────────────────────────────────────────────


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS or not np.isfinite(norm):
        return None
    return vector / norm


def _cosine_distance_series(vectors: list[np.ndarray | None]) -> float:
    """Mean ``1 - cos`` between consecutive non-degenerate vectors."""
    distances: list[float] = []
    previous: np.ndarray | None = None
    for vector in vectors:
        unit = None if vector is None else _unit(vector)
        if unit is not None and previous is not None:
            distances.append(max(0.0, 1.0 - float(np.dot(previous, unit))))
        previous = unit
    return float(np.mean(distances)) if distances else 0.0


def _absolute_change_series(values: list[np.ndarray], *, reference: float = 1.0) -> float:
    """Mean absolute per-dimension change between consecutive vectors, scaled to [0, 1]."""
    if len(values) < 2:
        return 0.0
    stacked = np.vstack(values)
    deltas = np.abs(np.diff(stacked, axis=0))
    return float(np.clip(np.mean(deltas) / max(reference, _EPS), 0.0, 1.0))


def silence_gate(silence_fraction: float) -> float:
    """1.0 for clean audio, ramping to 0.0 for mostly-silent clips."""
    if not np.isfinite(silence_fraction):
        return 0.0
    if silence_fraction <= _GATE_FULL:
        return 1.0
    if silence_fraction >= _GATE_ZERO:
        return 0.0
    return float((_GATE_ZERO - silence_fraction) / (_GATE_ZERO - _GATE_FULL))


def _window_slices(n_frames: int, window_count: int) -> list[tuple[int, int]]:
    """Overlapping (50 %) frame windows covering the whole signal.

    Overlap matters: with hard cuts a section boundary landing mid-window smears the
    change across two windows and reads as no change at all.
    """
    if n_frames <= 0:
        return []
    span = 2.0 * n_frames / (window_count + 1)
    hop = span / 2.0
    if span < 2.0:
        return [(0, n_frames)]
    slices: list[tuple[int, int]] = []
    for index in range(window_count):
        start = int(round(index * hop))
        stop = int(round(start + span))
        if stop > n_frames:
            stop = n_frames
        if stop - start >= 2:
            slices.append((start, stop))
        if stop >= n_frames:
            break
    return slices


def _resolve_window_count(duration: float, requested: int) -> int:
    """Shrink the window count so each window keeps at least ~1 s of audio."""
    if duration <= 0:
        return 0
    affordable = int(2.0 * duration - 1.0)
    return int(max(2, min(requested, affordable))) if affordable >= 2 else 0


def _mean_pool(matrix: np.ndarray, block: int) -> np.ndarray:
    """Mean-pool along the time axis (columns) into blocks of ``block`` frames."""
    if block <= 1 or matrix.shape[1] < block:
        return matrix
    usable = (matrix.shape[1] // block) * block
    pooled = matrix[:, :usable].reshape(matrix.shape[0], usable // block, block).mean(axis=2)
    return pooled


def _checkerboard_kernel(half: int) -> np.ndarray:
    axis = np.arange(-half, half + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    taper = np.exp(-0.5 * (grid_x**2 + grid_y**2) / max((half / 2.0) ** 2, _EPS))
    return taper * np.sign(grid_x) * np.sign(grid_y)


def _foote_novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    """Checkerboard novelty, scaled so a perfect boundary reads ~1.0.

    The kernel's positive and negative lobes each carry half its absolute mass, so
    dividing by ``sum(|K|) / 2`` turns the raw response into "within-section similarity
    minus cross-section similarity" — an absolute contrast rather than an arbitrary unit.
    That matters: a relative shape statistic over this curve reads ~1.0 for a drone,
    whose novelty curve is pure numerical noise around zero.
    """
    n_frames = ssm.shape[0]
    if n_frames < 2 * half + 1:
        return np.asarray([], dtype=np.float64)
    kernel = _checkerboard_kernel(half)
    scale = float(np.sum(np.abs(kernel))) / 2.0
    novelty = np.zeros(n_frames - 2 * half, dtype=np.float64)
    for offset in range(novelty.size):
        block = ssm[offset : offset + 2 * half + 1, offset : offset + 2 * half + 1]
        novelty[offset] = float(np.sum(kernel * block))
    return np.maximum(novelty, 0.0) / max(scale, _EPS)


# A boundary weaker than this is indistinguishable from texture wobble.
_NOVELTY_FLOOR = 0.05


def _pick_peaks(curve: np.ndarray, *, min_distance: int, min_height: float = 0.0) -> list[int]:
    if curve.size < 3:
        return []
    q25, median, q75 = np.percentile(curve, [25, 50, 75])
    threshold = max(median + 1.5 * max(float(q75 - q25), _EPS), min_height)
    peaks: list[int] = []
    for index in range(1, curve.size - 1):
        value = curve[index]
        if value < threshold or value < curve[index - 1] or value <= curve[index + 1]:
            continue
        if peaks and index - peaks[-1] < min_distance:
            if value > curve[peaks[-1]]:
                peaks[-1] = index
            continue
        peaks.append(index)
    return peaks


# ── feature extraction ────────────────────────────────────────────────


def _rhythm_vector(
    onset_envelope: np.ndarray,
    percussive_ratio: float,
    *,
    frames_per_second: float,
    envelope_reference: float,
) -> np.ndarray:
    """Scaled rhythmic character of one window, each dimension in [0, 1].

    Tempograms are the usual choice here and are the wrong one for this project: the
    music is arrhythmic by construction, so a tempo axis is mostly noise. Onset
    density/strength shape is what actually moves when the rhythm develops.
    """
    if onset_envelope.size == 0:
        return np.zeros(5, dtype=np.float64)
    reference = max(envelope_reference, _EPS)
    strengths = onset_envelope / reference
    strong = float(np.mean(onset_envelope >= (np.mean(onset_envelope) + np.std(onset_envelope))))
    duration = onset_envelope.size / max(frames_per_second, _EPS)
    peaks = librosa.util.peak_pick(
        onset_envelope,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.05 * reference,
        wait=2,
    )
    density = float(len(peaks) / max(duration, _EPS))
    return np.asarray(
        [
            np.clip(density / 25.0, 0.0, 1.0),
            np.clip(float(np.mean(strengths)), 0.0, 1.0),
            np.clip(float(np.std(strengths)), 0.0, 1.0),
            np.clip(strong, 0.0, 1.0),
            np.clip(percussive_ratio, 0.0, 1.0),
        ],
        dtype=np.float64,
    )


def _build_ssm(chroma: np.ndarray, mfcc: np.ndarray, *, sr: int) -> np.ndarray:
    """Cosine self-similarity over combined chroma + timbre frames at ``SSM_FPS``."""
    block = max(1, int(round((sr / HOP_LENGTH) / SSM_FPS)))
    pooled_chroma = _mean_pool(chroma, block)
    pooled_mfcc = _mean_pool(mfcc, block)
    n_frames = min(pooled_chroma.shape[1], pooled_mfcc.shape[1])
    if n_frames < 4:
        return np.zeros((0, 0), dtype=np.float64)

    def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        return matrix / np.maximum(norms, _EPS)

    features = np.vstack(
        [
            _normalize_columns(pooled_chroma[:, :n_frames]),
            _normalize_columns(pooled_mfcc[:, :n_frames]),
        ]
    ) / np.sqrt(2.0)
    features = _normalize_columns(features)
    return np.clip(features.T @ features, -1.0, 1.0)


def _lag_profile(ssm: np.ndarray, min_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean similarity along each diagonal — ``d(L)`` = how alike frames ``L`` apart are."""
    n_frames = ssm.shape[0]
    max_lag = int(0.75 * n_frames)
    if max_lag <= min_lag:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    lags = np.arange(min_lag, max_lag, dtype=np.int64)
    profile = np.asarray([float(np.mean(np.diagonal(ssm, offset=int(lag)))) for lag in lags])
    return lags, profile


def _return_from_lag_profile(profile: np.ndarray) -> float:
    """How far similarity climbs back up after having fallen away.

    Similarity-versus-lag decays monotonically for music that keeps moving on, and dips
    then *recovers* for music that returns to earlier material. Measuring the recovery —
    ``max(d(L) - running_min(d)(L))`` — is what makes A-B-A separate from A-B-C.

    Two things that look right and are not:

    - "best distant match minus average distant match" collapses to ``1 - mean``, because
      the exclusion band is necessarily shorter than a section, so every frame matches
      itself later on. That *penalises* returning, which raises the mean.
    - peak prominence against a local median finds short repeated phrases but not
      sections: a three-part piece produces a broad plateau at large lag, and any local
      baseline sits inside the plateau and cancels it.
    """
    if profile.size < 5:
        return 0.0
    smoothed = np.convolve(profile, np.ones(3) / 3.0, mode="same")
    smoothed[0], smoothed[-1] = profile[0], profile[-1]
    return float(np.clip(np.max(smoothed - np.minimum.accumulate(smoothed)), 0.0, 1.0))


def _structure_from_ssm(ssm: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(novelty, return, mean, boundaries_per_minute)``."""
    n_frames = ssm.shape[0]
    if n_frames < 8:
        return 0.0, 0.0, 0.0, 0.0

    min_lag = max(2, int(round(2.0 * SSM_FPS)))
    if min_lag >= n_frames - 1:
        min_lag = max(1, n_frames // 4)
    lag = np.abs(np.subtract.outer(np.arange(n_frames), np.arange(n_frames)))
    far = lag >= min_lag
    if not far.any():
        return 0.0, 0.0, 0.0, 0.0

    far_mean = float(np.mean(ssm[far]))
    _lags, profile = _lag_profile(ssm, min_lag)
    ssm_return = _return_from_lag_profile(profile)

    half = max(2, int(round(2.0 * SSM_FPS)))
    while 2 * half + 1 >= n_frames and half > 2:
        half -= 1
    novelty = _foote_novelty(ssm, half)
    if novelty.size < 3:
        return 0.0, ssm_return, far_mean, 0.0

    ssm_novelty = float(np.clip(np.percentile(novelty, 95), 0.0, 1.0))
    peaks = _pick_peaks(novelty, min_distance=half, min_height=_NOVELTY_FLOOR)
    minutes = n_frames / SSM_FPS / 60.0
    boundaries = float(len(peaks) / max(minutes, _EPS))
    return ssm_novelty, ssm_return, far_mean, boundaries


# ── public API ────────────────────────────────────────────────────────


def compute_structure_waveform(
    waveform: np.ndarray,
    sr: int = STRUCTURE_SR,
    *,
    window_count: int = DEFAULT_WINDOW_COUNT,
    weights: dict[str, float] | None = None,
) -> StructureComponents:
    """Multi-axis development + SSM structure for a mono generated tail."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D mono array")

    duration = float(waveform.size / sr) if sr else 0.0
    resolved_windows = _resolve_window_count(duration, window_count)
    if resolved_windows == 0 or waveform.size < sr:
        return StructureComponents(structure_seconds=duration)

    from .reward import _silence_stats

    silence_fraction, _ = _silence_stats(
        waveform,
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    gate = silence_gate(silence_fraction)

    harmonic, percussive = librosa.effects.hpss(waveform)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)[1:]
    rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms + _EPS, ref=1.0)
    onset_envelope = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=HOP_LENGTH)
    percussive_rms = librosa.feature.rms(y=percussive, frame_length=2048, hop_length=HOP_LENGTH)[0]

    n_frames = min(chroma.shape[1], mfcc.shape[1], rms.size, onset_envelope.size, percussive_rms.size)
    if n_frames < 4:
        return StructureComponents(
            structure_silence_fraction=float(silence_fraction),
            silence_gate=gate,
            structure_seconds=duration,
        )

    frames_per_second = sr / HOP_LENGTH
    envelope_reference = float(np.percentile(onset_envelope[:n_frames], 95)) or 1.0
    slices = _window_slices(n_frames, resolved_windows)

    chroma_vectors: list[np.ndarray | None] = []
    timbre_vectors: list[np.ndarray | None] = []
    level_vectors: list[np.ndarray] = []
    rhythm_vectors: list[np.ndarray] = []
    for start, stop in slices:
        chroma_vectors.append(chroma[:, start:stop].mean(axis=1))
        timbre_vectors.append(mfcc[:, start:stop].mean(axis=1))
        level_vectors.append(np.asarray([float(np.mean(rms_db[start:stop]))], dtype=np.float64))
        window_rms = float(np.mean(rms[start:stop]))
        window_percussive = float(np.mean(percussive_rms[start:stop]))
        rhythm_vectors.append(
            _rhythm_vector(
                onset_envelope[start:stop],
                window_percussive / max(window_rms, _EPS),
                frames_per_second=frames_per_second,
                envelope_reference=envelope_reference,
            )
        )

    dev_harmonic = _cosine_distance_series(chroma_vectors)
    dev_timbre = _cosine_distance_series(timbre_vectors)
    dev_dynamics = _absolute_change_series(level_vectors, reference=_DYNAMICS_REFERENCE_DB)
    dev_rhythm = _absolute_change_series(rhythm_vectors)

    axes = sorted([dev_harmonic, dev_timbre, dev_dynamics, dev_rhythm], reverse=True)
    dev_top2 = float(np.mean(axes[:2]))

    ssm = _build_ssm(chroma[:, :n_frames], mfcc[:, :n_frames], sr=sr)
    ssm_novelty, ssm_return, ssm_mean, boundaries = _structure_from_ssm(ssm)

    resolved_weights = weights or active_structure_weights()
    blend = (
        resolved_weights.get("development", 0.0) * dev_top2
        + resolved_weights.get("novelty", 0.0) * ssm_novelty
        + resolved_weights.get("return", 0.0) * ssm_return
    )
    coherence_factor = 0.5 + 0.5 * float(np.clip(ssm_mean, 0.0, 1.0))
    structure_score = float(100.0 * gate * coherence_factor * max(blend, 0.0))

    return StructureComponents(
        structure_score=structure_score,
        dev_top2=dev_top2,
        dev_harmonic=float(dev_harmonic),
        dev_timbre=float(dev_timbre),
        dev_dynamics=float(dev_dynamics),
        dev_rhythm=float(dev_rhythm),
        ssm_novelty=float(ssm_novelty),
        ssm_return=float(ssm_return),
        ssm_mean=float(ssm_mean),
        boundaries_per_minute=float(boundaries),
        structure_silence_fraction=float(silence_fraction),
        silence_gate=float(gate),
        window_count=len(slices),
        structure_seconds=duration,
    )


def compute_structure(
    audio_path: str | Path,
    *,
    prompt_duration_seconds: float | str | None = None,
    sr: int = STRUCTURE_SR,
    res_type: str = "soxr_qq",
    window_count: int = DEFAULT_WINDOW_COUNT,
    weights: dict[str, float] | None = None,
) -> StructureComponents:
    """Load a generated file, trim the prompt echo, and score structure.

    Never raises on degenerate audio — returns zeroed components instead.
    """
    waveform, loaded_sr = librosa.load(audio_path, sr=sr, mono=True, res_type=res_type)
    loaded_sr = int(loaded_sr)

    echo = _to_float(prompt_duration_seconds)
    if echo and echo > 0:
        start = int(echo * loaded_sr)
        if start < len(waveform):
            waveform = waveform[start:]

    if waveform.size < loaded_sr:
        return StructureComponents(structure_seconds=float(waveform.size / max(loaded_sr, 1)))

    return compute_structure_waveform(
        waveform,
        sr=loaded_sr,
        window_count=window_count,
        weights=weights,
    )
