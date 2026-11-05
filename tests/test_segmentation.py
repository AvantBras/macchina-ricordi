from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from macchina_ricordi.segmentation import (
    detect_track_segments,
    export_track_segments,
)


def _write_test_wav(path: Path, waveform: np.ndarray, sr: int) -> None:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm16.tobytes())


def _make_album_like_waveform(sr: int = 4_000) -> np.ndarray:
    first_track_t = np.arange(0, 6.0, 1.0 / sr, dtype=np.float32)
    second_track_t = np.arange(0, 6.5, 1.0 / sr, dtype=np.float32)

    first_track = 0.45 * np.sin(2.0 * np.pi * 220.0 * first_track_t)
    gap = np.zeros(int(2.5 * sr), dtype=np.float32)
    second_track = 0.4 * np.sin(2.0 * np.pi * 330.0 * second_track_t)
    return np.concatenate([first_track, gap, second_track]).astype(np.float32)


def test_detect_track_segments_finds_quiet_gap(tmp_path: Path) -> None:
    audio_path = tmp_path / "album.wav"
    _write_test_wav(audio_path, _make_album_like_waveform(), sr=4_000)

    segments = detect_track_segments(
        audio_path,
        analysis_sr=4_000,
        frame_length=1_024,
        hop_length=256,
        min_silence_seconds=1.0,
        silence_threshold_db=-28.0,
        min_track_seconds=4.0,
    )

    assert len(segments) == 2
    assert 6.0 <= segments[0].end_seconds <= 8.6
    assert 6.0 <= segments[1].start_seconds <= 8.6
    assert segments[0].duration_seconds >= 5.0
    assert segments[1].duration_seconds >= 5.0


def test_export_track_segments_writes_split_wavs(tmp_path: Path) -> None:
    audio_path = tmp_path / "album.wav"
    output_dir = tmp_path / "segments"
    _write_test_wav(audio_path, _make_album_like_waveform(), sr=4_000)

    rows = export_track_segments(
        audio_path,
        output_dir,
        analysis_sr=4_000,
        frame_length=1_024,
        hop_length=256,
        min_silence_seconds=1.0,
        silence_threshold_db=-28.0,
        min_track_seconds=4.0,
        output_sr=8_000,
        mono=True,
    )

    assert len(rows) == 2
    written_files = sorted(output_dir.glob("*.wav"))
    assert [path.name for path in written_files] == ["001.wav", "002.wav"]
    assert all(Path(row["segment_path"]).exists() for row in rows)
    assert rows[0]["track_index"] == 1
    assert rows[1]["track_index"] == 2


def _make_three_track_with_silent_middle(sr: int = 4_000) -> np.ndarray:
    sine_t = np.arange(0, 5.0, 1.0 / sr, dtype=np.float32)
    sine1 = 0.45 * np.sin(2.0 * np.pi * 220.0 * sine_t)
    sine3 = 0.4 * np.sin(2.0 * np.pi * 330.0 * sine_t)
    outer_gap = np.zeros(int(1.5 * sr), dtype=np.float32)

    blip_t = np.arange(0, 0.4, 1.0 / sr, dtype=np.float32)
    blip = 0.45 * np.sin(2.0 * np.pi * 220.0 * blip_t)
    inner_gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    middle = np.tile(np.concatenate([blip, inner_gap]), 5)

    return np.concatenate([sine1, outer_gap, middle, outer_gap, sine3]).astype(np.float32)


def test_export_track_segments_drops_mostly_silent_segments(tmp_path: Path) -> None:
    sr = 4_000
    audio_path = tmp_path / "album.wav"
    output_dir = tmp_path / "segments"
    _write_test_wav(audio_path, _make_three_track_with_silent_middle(sr), sr=sr)

    rows = export_track_segments(
        audio_path,
        output_dir,
        analysis_sr=sr,
        frame_length=1_024,
        hop_length=256,
        min_silence_seconds=1.0,
        silence_threshold_db=-28.0,
        min_track_seconds=4.0,
        output_sr=sr,
        mono=True,
        max_silence_fraction=0.4,
        silence_frame_db=-35.0,
    )

    assert len(rows) == 2
    track_indices = [row["track_index"] for row in rows]
    assert 2 not in track_indices
    written_files = sorted(output_dir.glob("*.wav"))
    assert [path.name for path in written_files] == ["001.wav", "003.wav"]
    for row in rows:
        assert row["silence_fraction"] < 0.4
