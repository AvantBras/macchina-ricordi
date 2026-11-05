from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.listening.features import analyze_waveform
from macchina_ricordi.listening.render import render_dossier_markdown
from macchina_ricordi.listening.semantic import (
    DEFAULT_SEMANTIC_PROBES,
    MsClapSemanticProbeBackend,
    SemanticBackendUnavailable,
    attach_semantic_probes,
    create_semantic_probe_backend,
    run_semantic_probes,
)


class FakeSemanticBackend:
    name = "fake"
    version = "test"
    device = "cpu"
    model_path = None

    def __init__(self, scores_by_key: dict[str, float]) -> None:
        self.scores_by_key = scores_by_key

    def score_audio_text(self, audio_path: str | Path, prompts: list[str]) -> list[float]:
        prompt_to_key = {definition.prompt: definition.key for definition in DEFAULT_SEMANTIC_PROBES}
        return [self.scores_by_key.get(prompt_to_key[prompt], 0.1) for prompt in prompts]


def _fake_backend() -> FakeSemanticBackend:
    return FakeSemanticBackend(
        {
            "metallic_scrape": 0.90,
            "granular_cloud": 0.78,
            "unstable_pulse": 0.72,
            "stable_beat": 0.70,
            "clear_melody": 0.0,
            "sung_voice": 0.02,
        }
    )


def test_semantic_vocabulary_keys_and_groups_are_stable() -> None:
    keys = [definition.key for definition in DEFAULT_SEMANTIC_PROBES]
    groups = {definition.group for definition in DEFAULT_SEMANTIC_PROBES}

    assert keys[:9] == [
        "metallic",
        "wooden",
        "plastic",
        "glassy",
        "dusty",
        "granular",
        "brittle",
        "smeared",
        "corroded",
    ]
    assert "metallic_scrape" in keys
    assert "stable_beat" in keys
    assert groups == {"material", "gesture", "temporal_form", "composite", "negative_control"}


def test_mock_backend_produces_strong_weak_and_uncertain_results(tmp_path: Path) -> None:
    report = run_semantic_probes(tmp_path / "clip.wav", probe_backend=_fake_backend())

    strong_keys = [score.key for score in report.strong]
    weak_keys = [score.key for score in report.weak]
    uncertain_keys = [score.key for score in report.uncertain]

    assert strong_keys[:2] == ["metallic_scrape", "granular_cloud"]
    assert "clear_melody" in weak_keys
    assert "sung_voice" in weak_keys
    assert uncertain_keys == ["stable_beat"]
    assert report.backend.name == "fake"
    assert report.scores[0].prompt


def test_missing_msclap_backend_raises_without_real_model_import(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "msclap":
            raise ModuleNotFoundError("No module named 'msclap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SemanticBackendUnavailable) as exc_info:
        create_semantic_probe_backend(device="cpu")

    assert "pip install -e" in str(exc_info.value)


def test_msclap_backend_caches_text_embeddings_between_files(tmp_path: Path) -> None:
    class FakeMsClapModel:
        def __init__(self) -> None:
            self.text_calls = 0

        def get_text_embeddings(self, prompts):  # type: ignore[no-untyped-def]
            self.text_calls += 1
            return np.asarray([[1.0, 0.0], [0.0, 1.0]])

        def get_audio_embeddings(self, paths, resample=True):  # type: ignore[no-untyped-def]
            return np.asarray([[1.0, 0.0]])

        def compute_similarity(self, audio_embeddings, text_embeddings):  # type: ignore[no-untyped-def]
            return audio_embeddings @ text_embeddings.T

    model = FakeMsClapModel()
    backend = MsClapSemanticProbeBackend.__new__(MsClapSemanticProbeBackend)
    backend.version = "test"
    backend.device = "cpu"
    backend.model_path = None
    backend._model = model
    backend._text_embedding_cache = {}

    prompts = ["metallic", "voice"]
    first = backend.score_audio_text(tmp_path / "a.wav", prompts)
    second = backend.score_audio_text(tmp_path / "b.wav", prompts)

    assert first == second == [1.0, 0.0]
    assert model.text_calls == 1


def test_dossier_to_dict_includes_semantic_data_when_present(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)
    enriched = attach_semantic_probes(dossier, probe_backend=_fake_backend())

    payload = enriched.to_dict()
    round_trip = json.loads(json.dumps(payload))

    assert "semantic_probes" not in dossier.to_dict()
    assert round_trip["semantic_probes"]["backend"]["name"] == "fake"
    assert round_trip["semantic_probes"]["strong"][0]["key"] == "metallic_scrape"
    assert round_trip["warnings"] == []


def test_markdown_renders_semantic_section_only_when_present(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)
    enriched = attach_semantic_probes(dossier, probe_backend=_fake_backend())

    plain_markdown = render_dossier_markdown(dossier)
    enriched_markdown = render_dossier_markdown(enriched)

    assert "## Semantic Probes" not in plain_markdown
    assert "## Semantic Probes" in enriched_markdown
    assert "metallic_scrape" in enriched_markdown
    assert "stable_beat" in enriched_markdown
