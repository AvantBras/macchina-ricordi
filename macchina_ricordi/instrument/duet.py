"""Pure audio helpers for the two-model duet renderer.

The duet loop lives in ``scripts/run_musicgen_duet.py`` (AudioCraft venv);
everything numerical enough to unit-test on CPU lives here: prompt blending,
candidate picking by silence, stem assembly, and the stereo master. numpy
only — importable from both the dev venv and the musicgen venv.
"""

from __future__ import annotations

import math

import numpy as np

# Matches the arrhythmic scorer's silence notion (RMS frames below -35 dB).
SILENCE_THRESHOLD_DB = -35.0
_FRAME = 2048
_HOP = 512
_EPS = 1e-10


def silence_fraction(waveform: np.ndarray, *, threshold_db: float = SILENCE_THRESHOLD_DB) -> float:
    """Fraction of RMS frames below ``threshold_db`` for a 1-D waveform."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1-D mono")
    if waveform.size < _FRAME:
        rms = float(np.sqrt(np.mean(waveform**2))) if waveform.size else 0.0
        return 1.0 if 20.0 * math.log10(rms + _EPS) < threshold_db else 0.0
    n_frames = 1 + (waveform.size - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n_frames)[:, None]
    frames = waveform[idx]
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + _EPS)
    return float(np.mean(rms_db < threshold_db))


def pick_least_silent(candidates: np.ndarray) -> tuple[int, list[float]]:
    """Pick the candidate row with the lowest silence fraction.

    ``candidates`` is ``[N, T]``. Returns ``(winner_index, fractions)``.
    """
    if candidates.ndim != 2 or candidates.shape[0] < 1:
        raise ValueError("candidates must be [N, T] with N >= 1")
    fractions = [silence_fraction(row) for row in candidates]
    return int(np.argmin(fractions)), fractions


def mix_tails(own: np.ndarray, partner: np.ndarray, *, listen: float) -> np.ndarray:
    """Blend a voice's own tail with its partner's tail, end-aligned.

    ``listen`` is the partner's share (0 = deaf, 1 = only hears partner). A
    shorter partner tail is zero-padded at the front — the model continues
    from the *end* of the prompt, so ends must line up.
    """
    if own.ndim != 1 or partner.ndim != 1:
        raise ValueError("tails must be 1-D mono")
    if not 0.0 <= listen <= 1.0:
        raise ValueError(f"listen must be in [0, 1], got {listen}")
    aligned = partner[-own.size :] if partner.size >= own.size else partner
    if aligned.size < own.size:
        aligned = np.concatenate([np.zeros(own.size - aligned.size, dtype=own.dtype), aligned])
    blend = (1.0 - listen) * own + listen * aligned
    peak = float(np.max(np.abs(blend))) if blend.size else 0.0
    if peak > 0.99:
        blend = blend * (0.99 / peak)
    return blend.astype(np.float32, copy=False)


def crossfade_append(track: np.ndarray, contribution: np.ndarray, *, crossfade_samples: int) -> np.ndarray:
    """Append ``contribution`` with a short linear crossfade at the junction.

    Linear (equal-gain) because the two sides are a continuation of the same
    material — correlated signals keep constant amplitude under a linear fade.
    """
    if track.size == 0 or crossfade_samples <= 0:
        return np.concatenate([track, contribution]).astype(np.float32, copy=False)
    xf = min(crossfade_samples, track.size, contribution.size)
    ramp = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)
    head = track[: track.size - xf]
    blended = track[track.size - xf :] * (1.0 - ramp) + contribution[:xf] * ramp
    return np.concatenate([head, blended, contribution[xf:]]).astype(np.float32, copy=False)


def equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan of a mono signal. ``pan`` in [-1 (left), 1 (right)].

    Returns ``[2, T]`` (L, R).
    """
    if mono.ndim != 1:
        raise ValueError("mono must be 1-D")
    if not -1.0 <= pan <= 1.0:
        raise ValueError(f"pan must be in [-1, 1], got {pan}")
    angle = (pan + 1.0) * math.pi / 4.0
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)]).astype(np.float32, copy=False)


def stereo_master(
    stem_a: np.ndarray,
    stem_b: np.ndarray,
    *,
    pan_a: float = -0.35,
    pan_b: float = 0.35,
    gain_a: float = 1.0,
    gain_b: float = 0.8,
    ceiling: float = 0.89,
) -> np.ndarray:
    """Pan, gain and sum two mono stems into a peak-safe stereo master ``[2, T]``."""
    length = max(stem_a.size, stem_b.size)
    padded_a = np.pad(stem_a, (0, length - stem_a.size))
    padded_b = np.pad(stem_b, (0, length - stem_b.size))
    mix = equal_power_pan(padded_a * gain_a, pan_a) + equal_power_pan(padded_b * gain_b, pan_b)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > ceiling:
        mix = mix * (ceiling / peak)
    return mix.astype(np.float32, copy=False)
