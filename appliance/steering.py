"""Changing what the model believes its recent past was.

The MusicGen stream's prompt steering, ported to plain numpy for a runtime with no
librosa. Each round, the tail the model is about to continue from is decoded,
transformed and re-encoded, so the model continues from a past that never
quite happened. What is *played* is never touched: only what is fed back.

Two things follow, and they are the whole character of these controls. They
compound — each round's output becomes the next round's prompt, so a tilt held
for a few rounds drags the stream's whole spectrum. And returning to neutral
does not snap back: the stream stays wherever it drifted, because the model is
now genuinely making that music.

- **Tilt**: a three-band EQ, crossovers at 250 Hz and 4 kHz, one octave wide and
  smooth, normalised so equal settings are a flat gain of exactly that.
- **Rate**: tape varispeed. ``rate × tau`` seconds of tail are resampled to
  *exactly* tau, so the prompt's length never moves; above 1 the past comes back
  faster and higher, below 1 slower and lower.
- **Motif blend**: a pinned moment mixed into the tail, end-aligned.

Order matches the stream: blend first, then rate, then tilt, so what the model
hears is what the controls say. See ``macchina_ricordi/instrument/prompt_transform.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macchina_ricordi.instrument import controls

# One vocabulary with the stream (macchina_ricordi/instrument/controls.py), so a gesture means the same thing on both.
MAX_TILT_DB = controls.MAX_TILT_DB
MIN_RATE, MAX_RATE = controls.TAPE_SPEED.appliance
LOW_CROSSOVER_HZ = controls.TILT_LOW_CROSSOVER_HZ
HIGH_CROSSOVER_HZ = controls.TILT_HIGH_CROSSOVER_HZ
TRANSITION_OCTAVES = controls.TILT_TRANSITION_OCTAVES
PEAK_CEILING = 0.99

INJECT_LIGHT = controls.BRING_BACK_LIGHTLY.weight
INJECT_HEAVY = controls.BRING_BACK_STRONGLY.weight
RECAP_MIX = controls.RECALL_MIX


@dataclass(frozen=True)
class Tilt:
    low_db: float = 0.0
    mid_db: float = 0.0
    high_db: float = 0.0

    @property
    def is_flat(self) -> bool:
        return not any(abs(v) > 1e-9 for v in (self.low_db, self.mid_db, self.high_db))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def band_gain_curve(freqs: np.ndarray, tilt: Tilt) -> np.ndarray:
    """Linear gain per frequency — the stream's curve, point for point."""
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    half = TRANSITION_OCTAVES / 2.0
    low_w = 1.0 - _smoothstep((np.log2(f / LOW_CROSSOVER_HZ) + half) / (2.0 * half))
    high_w = _smoothstep((np.log2(f / HIGH_CROSSOVER_HZ) + half) / (2.0 * half))
    mid_w = np.clip(1.0 - low_w - high_w, 0.0, 1.0)
    total = low_w + mid_w + high_w
    db = (low_w * tilt.low_db + mid_w * tilt.mid_db + high_w * tilt.high_db) / total
    return np.power(10.0, db / 20.0)


def apply_tilt(audio: np.ndarray, sample_rate: int, tilt: Tilt) -> np.ndarray:
    """Three-band EQ as one zero-phase filter over the whole window.

    The stream scales an STFT; with a gain curve this smooth, one FFT over the
    window is the same filter without the frame bookkeeping. The window is
    reflected at both ends first, so the filter's wrap-around lands in padding
    rather than at the end of the prompt — which is the part the model continues
    from.
    """
    if tilt.is_flat or audio.size == 0:
        return audio
    pad = min(4096, audio.size - 1) if audio.size > 1 else 0
    padded = np.pad(audio.astype(np.float64), pad, mode="reflect") if pad else audio.astype(np.float64)
    n = int(2 ** np.ceil(np.log2(padded.size)))
    spectrum = np.fft.rfft(padded, n=n)
    spectrum *= band_gain_curve(np.fft.rfftfreq(n, 1.0 / sample_rate), tilt)
    filtered = np.fft.irfft(spectrum, n=n)[: padded.size]
    return filtered[pad : pad + audio.size].astype(np.float32)


def resample_to(audio: np.ndarray, target_samples: int, *, taps: int = 16) -> np.ndarray:
    """Tape-style resample of ``audio`` into exactly ``target_samples``.

    Windowed-sinc interpolation, band-limited to the lower of the two rates so a
    speed-up does not alias. For the ±10% this is used over, 16 zero crossings a
    side is transparent.
    """
    if target_samples <= 0 or audio.size == 0:
        return np.zeros(max(0, target_samples), dtype=np.float32)
    if audio.size == target_samples:
        return audio.astype(np.float32)
    ratio = audio.size / float(target_samples)
    cutoff = min(1.0, 1.0 / ratio)
    positions = (np.arange(target_samples) + 0.5) * ratio - 0.5
    base = np.floor(positions).astype(np.int64)
    offsets = np.arange(-taps + 1, taps + 1)
    idx = base[:, None] + offsets[None, :]
    distance = positions[:, None] - idx
    window = 0.5 * (1.0 + np.cos(np.pi * np.clip(distance / taps, -1.0, 1.0)))
    kernel = cutoff * np.sinc(cutoff * distance) * window
    kernel /= kernel.sum(axis=1, keepdims=True)
    samples = audio[np.clip(idx, 0, audio.size - 1)].astype(np.float64)
    return (samples * kernel).sum(axis=1).astype(np.float32)


def end_aligned(audio: np.ndarray, length: int) -> np.ndarray:
    """Exactly ``length`` samples, keeping the *end*: the model continues from there."""
    if audio.size >= length:
        return audio[audio.size - length :]
    return np.concatenate([np.zeros(length - audio.size, dtype=audio.dtype), audio])


def blend(current: np.ndarray, motif: np.ndarray, mix: float) -> np.ndarray:
    """Mix a motif into the tail, end-aligned, and keep the peak under full scale."""
    mixed = (1.0 - mix) * current + mix * end_aligned(motif, current.size).astype(current.dtype)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    return mixed * (PEAK_CEILING / peak) if peak > PEAK_CEILING else mixed
