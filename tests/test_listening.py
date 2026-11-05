from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from macchina_ricordi.listening.features import analyze_waveform
from macchina_ricordi.listening.render import (
    render_dossier_json,
    render_dossier_markdown,
)
from macchina_ricordi.listening.schema import SCHEMA_VERSION, ListeningDossier


def _make_click_track(
    onset_times: np.ndarray,
    *,
    duration: float = 3.0,
    sr: int = 16_000,
) -> np.ndarray:
    y = np.zeros(int(duration * sr), dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset in onset_times:
        idx = int(onset * sr)
        if 0 <= idx < len(y) - len(click):
            y[idx : idx + len(click)] += click
    return y


def test_listening_dossier_to_dict_has_stable_top_level_keys(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)

    payload = dossier.to_dict()

    assert isinstance(dossier, ListeningDossier)
    assert list(payload) == [
        "schema",
        "analysis",
        "audio",
        "provenance",
        "whole_clip",
        "timeline",
        "sketches",
        "summary",
        "notable_events",
        "warnings",
    ]
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["provenance"]["source"] == "file"
    assert isinstance(payload["notable_events"], list)
    assert payload["sketches"]["mel_spectrogram_path"] is None
    assert isinstance(payload["sketches"]["numeric_spectrogram"], str)
    assert payload["sketches"]["numeric_spectrogram"].strip() != ""


def test_silent_waveform_produces_valid_dossier(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)

    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)

    assert dossier.whole_clip.reward["score"] == 0.0
    assert dossier.whole_clip.description["loudness"] == "whisper"
    assert dossier.whole_clip.description["density"] == "sparse"
    assert dossier.timeline.windows
    assert dossier.timeline.regions
    assert dossier.timeline.windows[0].label == "near-silent"
    assert "near-silence" in dossier.summary.time_contour


def test_markdown_render_includes_core_sections(tmp_path: Path) -> None:
    waveform = _make_click_track(np.arange(0.2, 2.0, 0.3), duration=2.0)
    dossier = analyze_waveform(waveform, path=tmp_path / "clip.wav", sr=16_000)

    markdown = render_dossier_markdown(dossier)

    assert "# Listening Dossier:" in markdown
    assert "clip.wav" in markdown
    assert "duration:" in markdown
    assert "score:" in markdown
    assert "## Source" in markdown
    assert "## Form Regions" in markdown
    assert "## Timeline" in markdown
    assert "## Evidence" in markdown
    assert "## Sketch" in markdown
    sketch_section = markdown.split("## Sketch", 1)[1]
    next_section_index = sketch_section.find("\n## ")
    sketch_body = sketch_section if next_section_index < 0 else sketch_section[:next_section_index]
    assert sketch_body.count("```text") == 2
    assert "legend: 0=silent" in sketch_body


def test_json_render_round_trips(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)

    payload = json.loads(render_dossier_json(dossier))

    assert payload["schema"] == SCHEMA_VERSION
    assert payload["audio"]["path"].endswith("silence.wav")
    assert payload["provenance"]["audio_path"].endswith("silence.wav")
    assert payload["timeline"]["windows"]
    assert payload["timeline"]["regions"]
