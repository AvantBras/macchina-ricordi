from __future__ import annotations

import argparse
import json
from pathlib import Path

from macchina_ricordi.listening import cli as listening_cli
from macchina_ricordi.listening.schema import (
    SCHEMA_VERSION,
    AnalysisInfo,
    AnalysisParameters,
    AudioInfo,
    ListeningDossier,
    ListeningSummary,
    ProvenanceInfo,
    Sketches,
    Timeline,
    TimelineRegion,
    TimelineWindow,
    WholeClip,
)


def _file_args(audio_path: Path, **overrides: object) -> argparse.Namespace:
    data = {
        "audio_path": str(audio_path),
        "sr": 16_000,
        "hop_length": 512,
        "res_type": "soxr_hq",
        "tempo_method": "tempogram",
        "window_seconds": 2.0,
        "window_hop_seconds": 1.0,
        "output": "",
        "markdown": "",
        "sidecars": False,
        "spectrogram": False,
        "json": False,
        "semantic_probes": False,
        "semantic_backend": "msclap",
        "semantic_model_version": "2023",
        "semantic_device": "cpu",
        "semantic_model_path": "",
        "captioner_witness": False,
        "captioner_backend": "qwen2-audio",
        "captioner_model_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "captioner_device": "cpu",
        "captioner_dtype": "float32",
        "captioner_quantization": "none",
        "captioner_max_new_tokens": 64,
        "captioner_max_audio_seconds": 30.0,
        "captioner_prompt_profile": "experimental_texture_v1",
        "captioner_trust_remote_code": True,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def _fake_dossier(path: Path) -> ListeningDossier:
    return ListeningDossier(
        schema=SCHEMA_VERSION,
        analysis=AnalysisInfo(
            version="test",
            analyzed_at="2026-05-15T00:00:00+00:00",
            parameters=AnalysisParameters(
                sr=16_000,
                hop_length=512,
                res_type="soxr_hq",
                tempo_method="tempogram",
                window_seconds=2.0,
                window_hop_seconds=1.0,
            ),
        ),
        audio=AudioInfo(path=str(path.resolve()), duration_seconds=1.0, sample_rate=16_000, channels=1, mono=True),
        provenance=ProvenanceInfo(source="file", audio_path=str(path.resolve())),
        whole_clip=WholeClip(
            reward={"score": 1.25, "silence_fraction": 0.0, "max_silence_gap_seconds": 0.0},
            description={
                "loudness": "moderate",
                "brightness": "bright",
                "attack": "sharp",
                "noisiness": "rough",
                "density": "dense",
                "pulse": "unstable",
                "resonance": "room",
            },
            features={"silence_fraction": 0.0, "max_silence_gap_seconds": 0.0},
        ),
        timeline=Timeline(
            window_seconds=2.0,
            window_hop_seconds=1.0,
            windows=[
                TimelineWindow(
                    start_seconds=0.0,
                    end_seconds=1.0,
                    label="dense bright rough broad-band texture, unstable pulse",
                    features={"onset_density": 1.0},
                )
            ],
            regions=[
                TimelineRegion(
                    start_seconds=0.0,
                    end_seconds=1.0,
                    label="dense rough noise field",
                    evidence={"onset_density": "medium"},
                    window_indices=[0],
                )
            ],
        ),
        sketches=Sketches(spectral="time     0s\nhigh   ##"),
        summary=ListeningSummary(global_summary="Global summary.", time_contour="Time contour."),
        warnings=[],
    )


def test_cmd_listen_file_prints_markdown_by_default(monkeypatch, capsys, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    monkeypatch.setattr(listening_cli, "analyze_audio", lambda *args, **kwargs: _fake_dossier(audio_path))

    exit_code = listening_cli.cmd_listen_file(_file_args(audio_path))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert stdout.startswith("# Listening Dossier:")
    assert "Global summary." in stdout


def test_cmd_listen_file_prints_json(monkeypatch, capsys, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    monkeypatch.setattr(listening_cli, "analyze_audio", lambda *args, **kwargs: _fake_dossier(audio_path))

    listening_cli.cmd_listen_file(_file_args(audio_path, json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["whole_clip"]["reward"]["score"] == 1.25


def test_cmd_listen_file_sidecars_write_default_paths(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    monkeypatch.setattr(listening_cli, "analyze_audio", lambda *args, **kwargs: _fake_dossier(audio_path))

    listening_cli.cmd_listen_file(_file_args(audio_path, sidecars=True))

    assert (tmp_path / "clip_listening_report.json").exists()
    assert (tmp_path / "clip_listening.md").exists()
