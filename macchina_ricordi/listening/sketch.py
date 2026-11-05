from __future__ import annotations

import librosa
import numpy as np

from .schema import TimelineWindow


def _bucket(value: float, *, maximum: float = 1.0) -> str:
    if maximum <= 0.0:
        maximum = 1.0
    scaled = max(0.0, min(value / maximum, 1.0))
    if scaled < 0.12:
        return ".."
    if scaled < 0.35:
        return "::"
    if scaled < 0.60:
        return "=="
    if scaled < 0.82:
        return "++"
    return "##"


def render_spectral_sketch(windows: list[TimelineWindow]) -> str:
    if not windows:
        return "time\nhigh\nmid\nlow\nonsets"

    max_onsets = max((window.features["onset_density"] for window in windows), default=1.0)
    time_cells = [f"{window.start_seconds:.0f}s".rjust(4) for window in windows]
    high_cells = [_bucket(window.features["high_band_ratio"]) for window in windows]
    mid_cells = [_bucket(window.features["mid_band_ratio"]) for window in windows]
    low_cells = [_bucket(window.features["low_band_ratio"]) for window in windows]
    onset_cells = [_bucket(window.features["onset_density"], maximum=max_onsets) for window in windows]

    return "\n".join(
        [
            "time   " + " ".join(time_cells),
            "high   " + " ".join(high_cells),
            "mid    " + " ".join(mid_cells),
            "low    " + " ".join(low_cells),
            "onsets " + " ".join(onset_cells),
        ]
    )


def _resample_columns(matrix: np.ndarray, target_columns: int) -> np.ndarray:
    rows, cols = matrix.shape
    if cols == target_columns:
        return matrix
    if cols == 0:
        return np.zeros((rows, target_columns), dtype=matrix.dtype)
    if cols < target_columns:
        positions = np.linspace(0, cols - 1, target_columns)
        indices = np.clip(np.round(positions).astype(int), 0, cols - 1)
        return matrix[:, indices]
    edges = np.linspace(0, cols, target_columns + 1, dtype=int)
    out = np.empty((rows, target_columns), dtype=matrix.dtype)
    for column in range(target_columns):
        start = edges[column]
        end = max(edges[column + 1], start + 1)
        out[:, column] = matrix[:, start:end].mean(axis=1)
    return out


def _format_frequency_label(freq_hz: float) -> str:
    if freq_hz >= 10_000:
        return f"{freq_hz / 1000:.0f}k"
    if freq_hz >= 1_000:
        return f"{freq_hz / 1000:.1f}k"
    return f"{freq_hz:.0f}"


def _format_time_header(duration_seconds: float, time_columns: int, freq_label_width: int) -> str:
    if duration_seconds <= 0.0:
        duration_seconds = 0.0
    marker_columns = [0, time_columns // 4, time_columns // 2, (3 * time_columns) // 4, time_columns - 1]
    cells = [" "] * time_columns
    for column in marker_columns:
        time_at_column = duration_seconds * column / max(time_columns - 1, 1)
        label = f"{time_at_column:.1f}s"
        start = column if column + len(label) <= time_columns else time_columns - len(label)
        for offset, char in enumerate(label):
            target = start + offset
            if 0 <= target < time_columns:
                cells[target] = char
    return " " * freq_label_width + "| " + "".join(cells)


def render_numeric_spectrogram(
    waveform: np.ndarray,
    *,
    sr: int,
    n_mels: int = 16,
    time_columns: int = 40,
    fmin: float = 50.0,
    fmax: float | None = None,
    db_floor: float = -80.0,
) -> str:
    if waveform.size == 0 or sr <= 0:
        return "(no audio)"

    effective_fmax = float(fmax) if fmax is not None else float(sr) / 2.0
    duration_seconds = float(waveform.size / sr)

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=effective_fmax,
        power=2.0,
    )
    if mel.size == 0 or float(np.max(mel)) <= 0.0:
        digits_per_row = "0" * time_columns
    else:
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_resampled = _resample_columns(mel_db, time_columns)
        scaled = (mel_db_resampled - db_floor) / (-db_floor)
        digits = np.clip(np.round(scaled * 9.0), 0, 9).astype(int)
        digits_per_row = None

    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=effective_fmax)
    freq_labels = [_format_frequency_label(freq) for freq in centers]
    freq_label_width = max(len(label) for label in freq_labels)
    freq_labels = [label.rjust(freq_label_width) for label in freq_labels]

    lines = [_format_time_header(duration_seconds, time_columns, freq_label_width)]
    for row_index in range(n_mels - 1, -1, -1):
        if digits_per_row is None:
            row_chars = "".join(str(value) for value in digits[row_index])
        else:
            row_chars = digits_per_row
        lines.append(f"{freq_labels[row_index]}| {row_chars}")

    lines.append(f"legend: 0=silent .. 9=clip peak (db_floor={db_floor:.0f}dB, per-clip normalized)")
    return "\n".join(lines)
