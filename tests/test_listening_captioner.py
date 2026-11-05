from __future__ import annotations

import builtins
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.listening.captioner import (
    DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID,
    DEFAULT_QWEN2_AUDIO_MODEL_ID,
    AudioFlamingo3CaptionerBackend,
    CaptionerBackendUnavailable,
    Qwen2AudioCaptionerBackend,
    _resolve_captioner_quantization,
    attach_captioner_witness,
    captioner_summary_from_payload,
    create_captioner_backend,
    default_captioner_model_id,
)
from macchina_ricordi.listening.features import analyze_waveform
from macchina_ricordi.listening.render import render_dossier_markdown
from macchina_ricordi.listening.schema import (
    SemanticBackendInfo,
    SemanticProbeReport,
    SemanticProbeScore,
    WholeClip,
)


class FakeCaptionerBackend:
    name = "fake-captioner"
    model_id = "fake/model"
    device = "cpu"
    dtype = "float32"
    quantization = "none"
    trust_remote_code = False

    def __init__(self, response: str) -> None:
        self.response = response

    def generate_caption(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        max_new_tokens: int,
        max_audio_seconds: float,
    ) -> str:
        assert "experimental music archive" in prompt
        assert max_new_tokens > 0
        assert max_audio_seconds > 0
        return self.response


def _weak_semantic_report() -> SemanticProbeReport:
    weak_scores = [
        SemanticProbeScore(
            key=key,
            group="negative_control",
            label=key.replace("_", " "),
            prompt=key.replace("_", " "),
            raw_similarity=0.0,
            score=0.0,
        )
        for key in ("sung_voice", "piano_chords", "acoustic_guitar", "pop_song")
    ]
    return SemanticProbeReport(
        backend=SemanticBackendInfo(name="fake", version="test", device="cpu"),
        vocabulary_version="macchina_ricordi.semantic_probes.v0",
        scores=weak_scores,
        strong=[],
        weak=weak_scores,
        uncertain=[],
    )


def test_fake_captioner_backend_attaches_witness_without_downloading_models(tmp_path: Path) -> None:
    waveform = np.zeros(16_000, dtype=np.float32)
    dossier = analyze_waveform(waveform, path=tmp_path / "silence.wav", sr=16_000)

    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeCaptionerBackend("A sparse granular texture with no obvious pulse."),
    )
    payload = enriched.to_dict()

    assert "captioner_witness" not in dossier.to_dict()
    assert payload["captioner_witness"]["backend"]["name"] == "fake-captioner"
    assert payload["captioner_witness"]["summary"].startswith("A sparse granular texture")
    assert payload["captioner_witness"]["agreement"]["with_features"] == "unknown"


def test_captioner_witness_json_round_trips(tmp_path: Path) -> None:
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeCaptionerBackend("Noise bursts move in a loose uneven texture."),
    )

    round_trip = json.loads(json.dumps(enriched.to_dict()))

    assert round_trip["captioner_witness"]["raw_response"].startswith("Noise bursts")
    assert round_trip["warnings"] == []


def test_cross_exam_warnings_fire_for_conventional_claims(tmp_path: Path) -> None:
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    dossier = replace(
        dossier,
        whole_clip=WholeClip(
            reward=dossier.whole_clip.reward,
            description={**dossier.whole_clip.description, "pulse": "unstable"},
            features={**dossier.whole_clip.features, "pulse_instability": 0.9},
        ),
        semantic_probes=_weak_semantic_report(),
    )

    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeCaptionerBackend(
            "A steady beat with sung voice, piano chords, guitar, drums, and a pop song feel."
        ),
    )
    warnings = enriched.captioner_witness.warnings

    assert any("steady beat" in warning for warning in warnings)
    assert any("voice/singing" in warning for warning in warnings)
    assert any("melody/harmony/chords" in warning for warning in warnings)
    assert any("piano" in warning for warning in warnings)
    assert any("guitar" in warning for warning in warnings)
    assert any("pop" in warning for warning in warnings)
    assert any("drums" in warning for warning in warnings)
    assert enriched.captioner_witness.agreement["with_features"] == "low"
    assert enriched.captioner_witness.agreement["with_semantic_probes"] == "low"


def test_captioner_backend_unavailable_raises_dedicated_error() -> None:
    with pytest.raises(CaptionerBackendUnavailable):
        create_captioner_backend(backend="not-a-backend")


def test_markdown_renders_captioner_section_only_when_present(tmp_path: Path) -> None:
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeCaptionerBackend("A quiet texture with uncertain source identity."),
    )

    assert "## Captioner Witness" not in render_dossier_markdown(dossier)
    markdown = render_dossier_markdown(enriched)
    assert "## Captioner Witness" in markdown
    assert "Captioner output is a model witness" in markdown
    assert "A quiet texture" in markdown


def test_captioner_summary_from_payload_is_compact(tmp_path: Path) -> None:
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeCaptionerBackend("A brittle ticking sound, perhaps metallic."),
    )

    summary = captioner_summary_from_payload(enriched.to_dict())

    assert summary is not None
    assert summary["backend"] == "fake-captioner"
    assert summary["quantization"] == "none"
    assert summary["summary"].startswith("A brittle ticking")


def _install_fake_captioner_modules(
    monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool = True
) -> dict[str, object]:
    captured: dict[str, object] = {}

    torch_module = types.ModuleType("torch")
    torch_module.float32 = "torch.float32"  # type: ignore[attr-defined]
    torch_module.float16 = "torch.float16"  # type: ignore[attr-defined]
    torch_module.bfloat16 = "torch.bfloat16"  # type: ignore[attr-defined]
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]

    transformers_module = types.ModuleType("transformers")

    class FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16_000)

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeProcessor:
            captured["processor_model_id"] = model_id
            captured["processor_kwargs"] = kwargs
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            captured["model_id"] = model_id
            captured["model_kwargs"] = kwargs
            return cls()

        def eval(self) -> FakeModel:
            return self

        def to(self, device: str) -> FakeModel:
            captured["model_to"] = device
            return self

        def parameters(self):  # type: ignore[no-untyped-def]
            return iter([types.SimpleNamespace(device="cuda:0" if cuda_available else "cpu")])

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["bnb_kwargs"] = kwargs

    transformers_module.AutoProcessor = FakeProcessor  # type: ignore[attr-defined]
    transformers_module.Qwen2AudioForConditionalGeneration = FakeModel  # type: ignore[attr-defined]
    transformers_module.AudioFlamingo3ForConditionalGeneration = FakeModel  # type: ignore[attr-defined]
    transformers_module.BitsAndBytesConfig = FakeBitsAndBytesConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return captured


def test_captioner_quantization_auto_resolves_by_device() -> None:
    assert _resolve_captioner_quantization("auto", device="cuda") == "bnb-4bit"
    assert _resolve_captioner_quantization("auto", device="cpu") == "none"
    assert _resolve_captioner_quantization("none", device="cuda") == "none"


def test_qwen2_audio_captioner_none_preserves_non_quantized_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_captioner_modules(monkeypatch, cuda_available=True)

    backend = Qwen2AudioCaptionerBackend(device="cuda", dtype="float16", quantization="none")

    kwargs = captured["model_kwargs"]
    assert isinstance(kwargs, dict)
    assert backend.quantization == "none"
    assert kwargs["dtype"] == "torch.float16"
    assert kwargs["device_map"] == "auto"
    assert "quantization_config" not in kwargs


def test_qwen2_audio_captioner_bnb_4bit_missing_bitsandbytes_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_captioner_modules(monkeypatch, cuda_available=True)
    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "bitsandbytes":
            raise ImportError("bitsandbytes missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(CaptionerBackendUnavailable, match="bitsandbytes"):
        Qwen2AudioCaptionerBackend(device="cuda", dtype="float16", quantization="bnb-4bit")


def test_default_captioner_model_id_resolves_per_backend() -> None:
    assert default_captioner_model_id("qwen2-audio") == DEFAULT_QWEN2_AUDIO_MODEL_ID
    assert default_captioner_model_id("audio-flamingo-3") == DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID
    with pytest.raises(CaptionerBackendUnavailable, match="not-a-backend"):
        default_captioner_model_id("not-a-backend")


def test_factory_resolves_backend_default_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_captioner_modules(monkeypatch, cuda_available=False)

    backend = create_captioner_backend(backend="audio-flamingo-3")
    assert backend.name == "audio-flamingo-3"
    assert captured["model_id"] == DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID

    backend = create_captioner_backend(backend="qwen2-audio")
    assert backend.name == "qwen2-audio"
    assert captured["model_id"] == DEFAULT_QWEN2_AUDIO_MODEL_ID


def test_factory_explicit_model_id_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_captioner_modules(monkeypatch, cuda_available=False)

    create_captioner_backend(backend="audio-flamingo-3", model_id="nvidia/music-flamingo-think-2601-hf")

    assert captured["model_id"] == "nvidia/music-flamingo-think-2601-hf"


def test_factory_unknown_backend_with_explicit_model_id_raises() -> None:
    with pytest.raises(CaptionerBackendUnavailable, match="Unsupported captioner backend"):
        create_captioner_backend(backend="not-a-backend", model_id="some/model")


def test_audio_flamingo_3_witness_goes_through_cross_examination(tmp_path: Path) -> None:
    dossier = analyze_waveform(np.zeros(16_000, dtype=np.float32), path=tmp_path / "clip.wav", sr=16_000)
    dossier = replace(
        dossier,
        whole_clip=WholeClip(
            reward=dossier.whole_clip.reward,
            description={**dossier.whole_clip.description, "pulse": "unstable"},
            features={**dossier.whole_clip.features, "pulse_instability": 0.9},
        ),
    )

    class FakeAF3Backend(FakeCaptionerBackend):
        name = "audio-flamingo-3"
        model_id = DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID

    enriched = attach_captioner_witness(
        dossier,
        captioner_backend=FakeAF3Backend("A steady beat carries a soaring melody in D minor."),
    )
    witness = enriched.captioner_witness

    assert witness.backend.name == "audio-flamingo-3"
    assert witness.backend.model_id == DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID
    assert any("steady beat" in warning for warning in witness.warnings)
    assert any("melody/harmony/chords" in warning for warning in witness.warnings)


@pytest.mark.parametrize(
    ("backend_cls", "default_model_id"),
    [
        (Qwen2AudioCaptionerBackend, DEFAULT_QWEN2_AUDIO_MODEL_ID),
        (AudioFlamingo3CaptionerBackend, DEFAULT_AUDIO_FLAMINGO_3_MODEL_ID),
    ],
)
def test_bnb_4bit_is_built_as_a_quantized_load(
    monkeypatch: pytest.MonkeyPatch, backend_cls: type, default_model_id: str
) -> None:
    captured = _install_fake_captioner_modules(monkeypatch, cuda_available=True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))

    backend = backend_cls(device="cuda", dtype="float16", quantization="bnb-4bit")

    kwargs = captured["model_kwargs"]
    bnb_kwargs = captured["bnb_kwargs"]
    assert isinstance(kwargs, dict)
    assert isinstance(bnb_kwargs, dict)
    assert backend.model_id == default_model_id
    assert backend.quantization == "bnb-4bit"
    assert kwargs["device_map"] == "auto"
    assert "quantization_config" in kwargs
    assert bnb_kwargs["load_in_4bit"] is True
    assert bnb_kwargs["bnb_4bit_quant_type"] == "nf4"
    assert bnb_kwargs["bnb_4bit_use_double_quant"] is True
    assert bnb_kwargs["bnb_4bit_compute_dtype"] == "torch.float16"


@pytest.mark.parametrize("backend_cls", [Qwen2AudioCaptionerBackend, AudioFlamingo3CaptionerBackend])
def test_auto_quantization_is_bnb_4bit_on_cuda(monkeypatch: pytest.MonkeyPatch, backend_cls: type) -> None:
    _install_fake_captioner_modules(monkeypatch, cuda_available=True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))

    backend = backend_cls(device="auto", dtype="auto", quantization="auto")

    assert backend.device == "cuda"
    assert backend.dtype == "float16"
    assert backend.quantization == "bnb-4bit"
