"""Shared helpers for the voice-loop scripts documented in docs/looping.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from macchina_ricordi.looping import VocalActivity, measure_vocal_activity
from macchina_ricordi.looping.core import DEFAULT_VOICE_THRESHOLD_DB

SR = 22_050
FEATURE_HOP = 220


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(str(path), sr=sr, mono=True)
    return waveform


def load_stems(report: dict, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    return load_mono(report["stems"]["vocals"], sr), load_mono(report["stems"]["instrumental"], sr)


def vocal_activity(report: dict, vocals: np.ndarray, sr: int = SR) -> VocalActivity:
    threshold = report.get("settings", {}).get("voice_threshold_db", DEFAULT_VOICE_THRESHOLD_DB)
    return measure_vocal_activity(vocals, sr, threshold_db=threshold)


def fmt_time(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    minutes, rest = divmod(abs(seconds), 60.0)
    return f"{sign}{int(minutes)}:{rest:05.2f}"


def features(waveform: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float]:
    """Chroma (melody/harmony) stacked with normalised MFCCs (timbre), 10 ms frames. Returns (features, fps)."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr, hop_length=FEATURE_HOP)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, hop_length=FEATURE_HOP, n_mfcc=20)[1:]
    mfcc = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-9)
    frames = min(chroma.shape[1], mfcc.shape[1])
    return np.vstack([chroma[:, :frames] * 3.0, mfcc[:, :frames] / 3.0]), sr / FEATURE_HOP


def window(feats: np.ndarray, fps: float, start: float, seconds: float) -> np.ndarray:
    first = max(0, int(round(start * fps)))
    return feats[:, first : first + int(seconds * fps)]


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equally sized feature windows; 1.0 means the ear would hear no jump."""
    frames = min(a.shape[-1], b.shape[-1])
    if frames == 0:
        return float("nan")
    x, y = a[:, :frames].ravel(), b[:, :frames].ravel()
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def beat_deviation_ms(waveform: np.ndarray, sr: int, beat_seconds: float, repeats: int) -> np.ndarray:
    """Beat phase of each repeat against one continuous grid, relative to the first repeat, in ms.

    Wrapped to +-half a beat. A loop that is a whole number of bars reads ~0 everywhere. A single repeat at
    ~half a beat is the reading flipping to the off-beat (hi-hats), not drift; drift grows repeat after repeat.
    """
    import librosa

    hop = 64
    envelope = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop)
    times = np.arange(envelope.shape[0]) * hop / sr
    total = waveform.shape[-1] / sr
    span = total / repeats
    phases = np.linspace(0.0, beat_seconds, 300, endpoint=False)
    grid = np.arange(0.0, total, beat_seconds)
    best = []
    for index in range(repeats):
        lo, hi = index * span + 0.3, (index + 1) * span - 0.3
        inside = grid[(grid > lo - beat_seconds) & (grid < hi)]
        scores = []
        for phase in phases:
            ticks = inside + phase
            ticks = ticks[(ticks > lo) & (ticks < hi)]
            scores.append(np.interp(ticks, times, envelope).sum())
        best.append(phases[int(np.argmax(scores))])
    relative = (np.asarray(best) - best[0] + beat_seconds / 2.0) % beat_seconds - beat_seconds / 2.0
    return relative * 1000.0
