"""Arc / coherence scoring — the axis the arrhythmic scorer is blind to.

Motivation from the author's experiments: deeper reshaped models can produce a
*coherent voice developing along an arc*, where shallower/wider models pile up busy
sound "all at the same time". The arrhythmic/percussive scorer rewards the busyness and
cannot see the development (corr with arc ~ +0.14). These metrics capture it:

- ``arc_tv4`` — total variation of mean-chroma across 4 equal windows (harmonic
  development; "the music goes somewhere").
- ``chroma_drift`` — ``1 - cos`` between the mean-chroma of the first and second halves.
- ``timbral_coherence`` — MFCC-stats cosine between the two halves (one coherent voice;
  low = timbre lurches around).
- ``arc_coherence_score`` — provisional composite ``10 * arc_tv4 * timbral_coherence``
  (development gated by coherence: wandering without a coherent voice does not count).
- ``arc_coherence_gated`` — the same composite multiplied by a silence gate. Mostly-silent
  clips post huge ``arc_tv4`` (chroma of near-noise wanders freely) and have repeatedly
  faked good arc numbers in this project's analyses; the gated field is safe to average
  without remembering to filter first. ``arc_coherence_score`` is left ungated so it stays
  comparable with every number recorded before 2026-07-28.

All measures are computed on the generated **tail** — for continuation samples the first
``prompt_duration_seconds`` of prompt echo are trimmed first.

For development on non-harmonic axes (timbre, dynamics, rhythm) and for section
structure, see ``macchina_ricordi.scoring.structure``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from .similarity import _calibrated_cosine, _mfcc_stats_vector

ARC_SR = 16_000
_EPS = 1e-9


@dataclass(frozen=True)
class ArcCoherenceComponents:
    arc_coherence_score: float = 0.0
    arc_coherence_gated: float = 0.0
    arc_tv4: float = 0.0
    chroma_drift: float = 0.0
    timbral_coherence: float = 0.0
    arc_silence_fraction: float = 0.0
    arc_tail_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fieldnames(cls) -> list[str]:
        return [field.name for field in fields(cls)]


def _mean_chroma_unit(waveform: np.ndarray, sr: int) -> np.ndarray | None:
    if waveform.size < sr // 4:
        return None
    chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
    vec = chroma.mean(axis=1)
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        return None
    return vec / norm


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are unit vectors


def timbral_coherence(a: np.ndarray, b: np.ndarray, *, sr: int) -> float:
    """MFCC-stats cosine between two segments: one coherent voice, or a lurch.

    Deliberately not ``compute_similarity_waveform(a, b).mfcc_cosine``, which is
    what this used to be. That builds the spectral and rhythm vectors as well and
    throws both away — and the rhythm vector runs the entire arrhythmic scorer on
    each half. Identical number, ~255 ms cheaper on a 10 s round, which was 30%
    of the stream's whole per-round scoring budget.
    """
    if min(len(a), len(b)) < sr // 4:
        return 0.0
    return float(_calibrated_cosine(_mfcc_stats_vector(a, sr=sr), _mfcc_stats_vector(b, sr=sr), key="mfcc"))


def arc_tv4(waveform: np.ndarray, *, sr: int) -> float:
    """Total variation of mean-chroma across 4 equal windows — "it goes somewhere"."""
    total = 0.0
    w = len(waveform) // 4
    if w < sr // 4:
        return 0.0
    prev = None
    for i in range(4):
        cv = _mean_chroma_unit(waveform[i * w : (i + 1) * w], sr)
        if cv is None:
            prev = None
            continue
        if prev is not None:
            total += max(0.0, 1.0 - _cos(prev, cv))
        prev = cv
    return total


def compute_arc_coherence_waveform(
    waveform: np.ndarray,
    sr: int = ARC_SR,
) -> ArcCoherenceComponents:
    """Arc/coherence metrics for a mono generated tail (prompt echo already trimmed)."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D mono array")

    tail_seconds = float(len(waveform) / sr) if sr else 0.0

    # Halves -> chroma_drift + timbral_coherence
    mid = len(waveform) // 2
    a, b = waveform[:mid], waveform[mid:]
    ca, cb = _mean_chroma_unit(a, sr), _mean_chroma_unit(b, sr)
    chroma_drift = 0.0 if ca is None or cb is None else max(0.0, 1.0 - _cos(ca, cb))

    coherence = timbral_coherence(a, b, sr=sr)
    tv4 = arc_tv4(waveform, sr=sr)
    arc_coherence_score = 10.0 * tv4 * max(0.0, coherence)

    from .reward import _silence_stats
    from .structure import silence_gate

    silence_fraction, _ = _silence_stats(waveform, sr=sr, frame_length=2048, hop_length=512)

    return ArcCoherenceComponents(
        arc_coherence_score=float(arc_coherence_score),
        arc_coherence_gated=float(arc_coherence_score * silence_gate(silence_fraction)),
        arc_tv4=float(tv4),
        chroma_drift=float(chroma_drift),
        timbral_coherence=float(coherence),
        arc_silence_fraction=float(silence_fraction),
        arc_tail_seconds=float(tail_seconds),
    )


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_arc_coherence(
    audio_path: str | Path,
    *,
    prompt_duration_seconds: float | str | None = None,
    sr: int = ARC_SR,
    res_type: str = "soxr_qq",
) -> ArcCoherenceComponents:
    """Load a generated WAV, trim the prompt echo, and score arc/coherence.

    Never raises on degenerate audio — returns zeroed components instead.
    """
    waveform, loaded_sr = librosa.load(audio_path, sr=sr, mono=True, res_type=res_type)
    loaded_sr = int(loaded_sr)

    echo = _to_float(prompt_duration_seconds)
    if echo and echo > 0:
        start = int(echo * loaded_sr)
        if start < len(waveform):
            waveform = waveform[start:]

    if waveform.size < loaded_sr // 4:
        return ArcCoherenceComponents(arc_tail_seconds=float(waveform.size / loaded_sr))

    return compute_arc_coherence_waveform(waveform, sr=loaded_sr)
