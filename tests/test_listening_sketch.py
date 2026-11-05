from __future__ import annotations

import librosa
import numpy as np

from macchina_ricordi.listening.sketch import render_numeric_spectrogram


def _grid_digit_rows(grid: str) -> list[str]:
    rows: list[str] = []
    for line in grid.splitlines():
        if "| " not in line:
            continue
        body = line.split("| ", 1)[1]
        if body and all(char.isdigit() for char in body):
            rows.append(body)
    return rows


def test_render_numeric_spectrogram_empty_returns_marker() -> None:
    assert render_numeric_spectrogram(np.asarray([], dtype=np.float32), sr=16_000) == "(no audio)"


def test_render_numeric_spectrogram_silence_is_all_zeros() -> None:
    waveform = np.zeros(16_000, dtype=np.float32)

    grid = render_numeric_spectrogram(waveform, sr=16_000)
    digit_rows = _grid_digit_rows(grid)

    assert len(digit_rows) == 16
    assert all(len(row) == 40 for row in digit_rows)
    assert all(set(row) == {"0"} for row in digit_rows)


def test_render_numeric_spectrogram_sine_peaks_in_target_band() -> None:
    sr = 16_000
    duration = 1.0
    freq = 440.0
    t = np.arange(0, duration, 1.0 / sr, dtype=np.float32)
    waveform = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    grid = render_numeric_spectrogram(waveform, sr=sr)
    digit_rows = _grid_digit_rows(grid)

    assert len(digit_rows) == 16
    assert all(len(row) == 40 for row in digit_rows)
    assert any("9" in row for row in digit_rows)

    centers = librosa.mel_frequencies(n_mels=16, fmin=50.0, fmax=sr / 2.0)
    band_index = int(np.argmin(np.abs(centers - freq)))
    grid_row_index_from_bottom = band_index  # rendered with low freqs at the bottom
    # _grid_digit_rows is in the printed top-down order, so highest freq is index 0.
    row_index_top_down = (16 - 1) - grid_row_index_from_bottom
    target_row = digit_rows[row_index_top_down]
    target_mean = sum(int(c) for c in target_row) / len(target_row)
    top_mean = sum(int(c) for c in digit_rows[0]) / len(digit_rows[0])
    bottom_mean = sum(int(c) for c in digit_rows[-1]) / len(digit_rows[-1])

    assert target_mean > top_mean
    assert target_mean > bottom_mean


def test_render_numeric_spectrogram_white_noise_is_broad_band() -> None:
    rng = np.random.default_rng(42)
    waveform = rng.normal(0.0, 0.25, 32_000).astype(np.float32)

    grid = render_numeric_spectrogram(waveform, sr=16_000)
    digit_rows = _grid_digit_rows(grid)

    assert len(digit_rows) == 16
    row_means = [sum(int(c) for c in row) / len(row) for row in digit_rows]
    # White noise has energy in every band, so every row should be loud and none should be silent.
    assert min(row_means) > 5.0


def test_render_numeric_spectrogram_includes_header_and_legend() -> None:
    sr = 16_000
    t = np.arange(0, 1.0, 1.0 / sr, dtype=np.float32)
    waveform = (0.3 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)

    grid = render_numeric_spectrogram(waveform, sr=sr)

    assert "0.0s" in grid
    assert "1.0s" in grid
    assert "legend:" in grid
    assert "per-clip normalized" in grid
