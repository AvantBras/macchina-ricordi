from __future__ import annotations

from pathlib import Path

import numpy as np

from macchina_ricordi.listening.features import analyze_waveform
from macchina_ricordi.listening.spectrogram import (
    attach_spectrogram_path,
    default_spectrogram_path,
    render_mel_spectrogram_png,
)


def _sine_wave(*, sr: int = 16_000, duration: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / sr, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_default_spectrogram_path_uses_audio_stem(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    assert default_spectrogram_path(audio_path) == tmp_path / "clip_spectrogram.png"


def test_render_mel_spectrogram_png_writes_valid_png(tmp_path: Path) -> None:
    output_path = tmp_path / "out_spectrogram.png"

    returned = render_mel_spectrogram_png(_sine_wave(), sr=16_000, output_path=output_path)

    assert returned == output_path
    assert output_path.exists()
    data = output_path.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_attach_spectrogram_path_returns_new_dossier_with_path_set(tmp_path: Path) -> None:
    dossier = analyze_waveform(_sine_wave(), path=tmp_path / "clip.wav", sr=16_000)

    assert dossier.sketches.mel_spectrogram_path is None

    attached = attach_spectrogram_path(dossier, path="clip_spectrogram.png")

    assert attached.sketches.mel_spectrogram_path == "clip_spectrogram.png"
    assert dossier.sketches.mel_spectrogram_path is None
