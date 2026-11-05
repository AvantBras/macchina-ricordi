"""Transforms applied to the continuation prompt before the model sees it.

The stream re-prompts on its own tail every round, so the prompt window is the
one place where a control can *intervene* rather than measure: change what the
model believes its recent past was, and it continues from that instead. Both
transforms here therefore compound — hold one for a few rounds and it drags the
stream, because each round's output becomes the next round's prompt — and both
stop compounding the moment they return to neutral, leaving the stream wherever
it drifted to.

- **Tilt** (`PromptTilt`) — a three-band EQ on the prompt. The model hears a
  bass- or treble-heavy past and follows it. The interesting failure is the
  model fighting back toward its comfort zone, which is itself audible.
- **Rate** (`prompt_rate`) — tape-style varispeed of the prompt window only.
  Crucially the *output* is always exactly the requested duration: `rate × tau`
  seconds of tail are resampled down to `tau`, so prompt token count, the
  horizon guard and the buffer arithmetic all stay put. Only what is fed back
  is shifted, never what is played, so the drift arrives a round later as the
  music genuinely moving rather than as an effect pedal.

The numerical parts take and return arrays so they can be unit-tested without
audio files; only `render_prompt_transform` touches the disk.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .controls import MAX_TILT_DB, TAPE_SPEED, TILT_HIGH_CROSSOVER_HZ, TILT_LOW_CROSSOVER_HZ, TILT_TRANSITION_OCTAVES

RATE_MODES = ("tape", "tempo")
PromptRateMode = Literal["tape", "tempo"]

MIN_PROMPT_RATE, MAX_PROMPT_RATE = TAPE_SPEED.limits
# Past roughly ±15% EnCodec's round-trip starts contributing its own artefacts;
# ±10% is the range the rate nudge was designed around.
CLEAN_PROMPT_RATE_MARGIN = 0.15

# The tilt's limits and crossovers are shared with the appliance (``controls.py``). The crossovers are smooth
# rather than a brick wall: a hard band edge rings, and the ringing would be indistinguishable from the model
# responding to the tilt.

_STFT_N_FFT = 2048
_STFT_HOP = 512
_PEAK_CEILING = 0.99


@dataclass(frozen=True)
class PromptTilt:
    """Three-band gain in dB applied to the prompt window."""

    low_db: float = 0.0
    mid_db: float = 0.0
    high_db: float = 0.0

    @property
    def is_flat(self) -> bool:
        return not any(abs(v) > 1e-9 for v in (self.low_db, self.mid_db, self.high_db))

    def to_dict(self) -> dict[str, float]:
        return {"low_db": self.low_db, "mid_db": self.mid_db, "high_db": self.high_db}

    def describe(self) -> str:
        return f"{self.low_db:+.1f}/{self.mid_db:+.1f}/{self.high_db:+.1f}dB"


FLAT_TILT = PromptTilt()


@dataclass(frozen=True)
class PromptTransformResult:
    """What the transform actually managed to do, for the performance log."""

    duration_seconds: float
    rate_requested: float
    rate_applied: float
    tilt: PromptTilt

    @property
    def rate_was_clamped(self) -> bool:
        return abs(self.rate_applied - self.rate_requested) > 1e-6


def _smoothstep(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def band_gain_curve(
    freqs: np.ndarray,
    tilt: PromptTilt,
    *,
    low_crossover_hz: float = TILT_LOW_CROSSOVER_HZ,
    high_crossover_hz: float = TILT_HIGH_CROSSOVER_HZ,
    transition_octaves: float = TILT_TRANSITION_OCTAVES,
) -> np.ndarray:
    """Linear gain per frequency bin for a three-band tilt.

    The band weights cross over smoothly in log-frequency and are renormalised
    to sum to 1, so a tilt with all three bands set to the same value is a flat
    gain of exactly that many dB at every frequency, whatever the crossovers.
    """
    f = np.maximum(np.asarray(freqs, dtype=float), 1e-6)
    half = max(1e-6, float(transition_octaves) / 2.0)
    low_w = 1.0 - _smoothstep((np.log2(f / low_crossover_hz) + half) / (2.0 * half))
    high_w = _smoothstep((np.log2(f / high_crossover_hz) + half) / (2.0 * half))
    mid_w = np.clip(1.0 - low_w - high_w, 0.0, 1.0)
    total = low_w + mid_w + high_w
    db = (low_w * tilt.low_db + mid_w * tilt.mid_db + high_w * tilt.high_db) / total
    return np.power(10.0, db / 20.0)


def apply_tilt(waveform: np.ndarray, *, sr: int, tilt: PromptTilt) -> np.ndarray:
    """Three-band EQ via STFT magnitude scaling. Returns the same length."""
    if tilt.is_flat or waveform.size == 0:
        return waveform
    import librosa

    n_fft = min(_STFT_N_FFT, max(64, 2 ** int(np.floor(np.log2(max(waveform.size, 64))))))
    spectrum = librosa.stft(waveform, n_fft=n_fft, hop_length=_STFT_HOP)
    gains = band_gain_curve(librosa.fft_frequencies(sr=sr, n_fft=n_fft), tilt)
    return librosa.istft(spectrum * gains[:, None], hop_length=_STFT_HOP, n_fft=n_fft, length=waveform.size)


def apply_rate(waveform: np.ndarray, *, sr: int, target_samples: int, mode: PromptRateMode = "tape") -> np.ndarray:
    """Fit ``waveform`` into exactly ``target_samples``, tape or tempo-only.

    ``tape`` resamples, so pitch and tempo move together and transients survive
    — analog varispeed. ``tempo`` phase-vocodes, preserving pitch but smearing
    transients, which on arrhythmic percussive material is very audible and may
    be either the point or exactly wrong.
    """
    if waveform.size == 0 or target_samples <= 0:
        return np.zeros(max(0, target_samples), dtype=np.float32)
    if waveform.size == target_samples:
        return waveform

    import librosa

    ratio = waveform.size / float(target_samples)
    if mode == "tempo":
        shifted = librosa.effects.time_stretch(waveform, rate=ratio)
    else:
        target_sr = max(1, int(round(sr / ratio)))
        shifted = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return _fit_length(shifted, target_samples)


def _fit_length(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Trim or zero-pad to an exact length, keeping the *end* of the audio.

    The model continues from the end of the prompt, so a rounding error has to
    be paid at the front where it is inaudible rather than at the splice point.
    """
    if waveform.size == target_samples:
        return waveform
    if waveform.size > target_samples:
        return waveform[waveform.size - target_samples :]
    return np.concatenate([np.zeros(target_samples - waveform.size, dtype=waveform.dtype), waveform])


def _write_mono_wav(path: Path, waveform: np.ndarray, *, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sr))
        wav_file.writeframes(pcm.tobytes())


def render_prompt_transform(
    source_wav: Path,
    out_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    rate: float = 1.0,
    mode: PromptRateMode = "tape",
    tilt: PromptTilt = FLAT_TILT,
) -> PromptTransformResult:
    """Write the transformed prompt window, always ``duration_seconds`` long.

    A rate above 1 needs ``rate × duration`` seconds of source ending where the
    plain window ends. When the source cannot reach that far back the window is
    clamped to what exists and the *applied* rate comes back lower than the
    requested one — the caller reports that rather than silently generating at
    a rate the player did not ask for.
    """
    import librosa

    rate = float(rate)
    window_end = float(start_seconds) + float(duration_seconds)
    source_start = max(0.0, window_end - rate * float(duration_seconds))
    source_window = max(0.0, window_end - source_start)
    if source_window <= 0.0 or duration_seconds <= 0.0:
        raise ValueError(f"Empty prompt window for {source_wav}")

    waveform, sr = librosa.load(source_wav, sr=None, mono=True, offset=source_start, duration=source_window)
    sr = int(sr)
    if waveform.size == 0 or sr <= 0:
        raise ValueError(f"Empty prompt audio from {source_wav}")

    target_samples = max(1, int(round(float(duration_seconds) * sr)))
    # What the source could actually supply, which is what the model will hear.
    applied_rate = (waveform.size / float(sr)) / float(duration_seconds)

    shifted = apply_rate(waveform, sr=sr, target_samples=target_samples, mode=mode)
    tilted = apply_tilt(shifted, sr=sr, tilt=tilt)

    peak = float(np.max(np.abs(tilted))) if tilted.size else 0.0
    if peak > _PEAK_CEILING:
        tilted = tilted * (_PEAK_CEILING / peak)

    _write_mono_wav(out_path, tilted, sr=sr)
    return PromptTransformResult(
        duration_seconds=tilted.size / float(sr),
        rate_requested=rate,
        rate_applied=applied_rate,
        tilt=tilt,
    )


def clamp_prompt_rate(value: Any, default: float = 1.0) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(rate):
        return default
    return min(MAX_PROMPT_RATE, max(MIN_PROMPT_RATE, rate))


def clamp_tilt_db(value: Any, default: float = 0.0) -> float:
    try:
        db = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(db):
        return default
    return min(MAX_TILT_DB, max(-MAX_TILT_DB, db))
