from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import macchina_ricordi.segmentation.cli as cli_segmentation


def test_cmd_segment_file_writes_segment_manifest(monkeypatch, capsys, tmp_path: Path) -> None:
    audio_path = tmp_path / "album.wav"
    output_dir = tmp_path / "segments"
    audio_path.write_bytes(b"audio")
    written: dict[str, object] = {}

    monkeypatch.setattr(
        cli_segmentation,
        "_segment_single_file",
        lambda *args, **kwargs: (
            [
                {
                    "source_path": str(audio_path.resolve()),
                    "segment_path": str((output_dir / "album" / "001.wav").resolve()),
                    "track_index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "duration_seconds": 10.0,
                }
            ],
            0,
        ),
    )
    monkeypatch.setattr(
        cli_segmentation,
        "write_csv",
        lambda path, fieldnames, rows: written.update(path=path, fieldnames=fieldnames, rows=rows),
    )

    exit_code = cli_segmentation.cmd_segment_file(
        argparse.Namespace(
            audio_path=str(audio_path),
            output_dir=str(output_dir),
            analysis_sr=8_000,
            frame_length=4_096,
            hop_length=1_024,
            min_silence_seconds=1.5,
            silence_threshold_db=-32.0,
            min_track_seconds=30.0,
            output_sr=32_000,
            mono=True,
            res_type="soxr_hq",
            max_silence_fraction=0.85,
            silence_frame_db=-35.0,
        )
    )

    stdout = capsys.readouterr().out
    manifest_path = (output_dir / "album" / "segments.csv").resolve()

    assert exit_code == 0
    assert written["path"] == manifest_path
    assert len(written["rows"]) == 1
    assert "Wrote 1 segments" in stdout
    assert f"Manifest: {manifest_path}" in stdout


def test_cmd_segment_file_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli_segmentation.cmd_segment_file(
            argparse.Namespace(
                audio_path=str(tmp_path / "missing.wav"),
                output_dir=str(tmp_path / "segments"),
                analysis_sr=8_000,
                frame_length=4_096,
                hop_length=1_024,
                min_silence_seconds=1.5,
                silence_threshold_db=-32.0,
                min_track_seconds=30.0,
                output_sr=32_000,
                mono=True,
                res_type="soxr_hq",
            )
        )
