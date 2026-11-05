from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def _load_helper_module(monkeypatch):
    calls = []

    class FakeMusicGen:
        def __init__(self, name, compression_model, lm, max_duration=None):  # type: ignore[no-untyped-def]
            if max_duration <= 18.0:
                raise AssertionError("AudioCraft constructor default extend_stride would fail")
            self.name = name
            self.compression_model = compression_model
            self.lm = lm
            self.max_duration = max_duration
            self.sample_rate = 32000
            calls.append(max_duration)

        @classmethod
        def get_pretrained(cls, name, device=None):  # type: ignore[no-untyped-def]
            model = cls(name, object(), object(), max_duration=20.0)
            model.device = device
            return model

        def set_generation_params(self, **kwargs):  # type: ignore[no-untyped-def]
            self.generation_params = kwargs

        def generate_unconditional(self, num_samples):  # type: ignore[no-untyped-def]
            return np.zeros((num_samples, 1, 32), dtype=np.float32)

        def generate(self, prompts):  # type: ignore[no-untyped-def]
            raise AssertionError("text generation should not be called for micro-unconditional profile")

    audiocraft = types.ModuleType("audiocraft")
    audiocraft_data = types.ModuleType("audiocraft.data")
    audiocraft_audio = types.ModuleType("audiocraft.data.audio")
    audiocraft_audio.audio_read = lambda *args, **kwargs: None
    audiocraft_models = types.ModuleType("audiocraft.models")
    audiocraft_models.MusicGen = FakeMusicGen
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None)
    torch.manual_seed = lambda seed: None

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "audiocraft", audiocraft)
    monkeypatch.setitem(sys.modules, "audiocraft.data", audiocraft_data)
    monkeypatch.setitem(sys.modules, "audiocraft.data.audio", audiocraft_audio)
    monkeypatch.setitem(sys.modules, "audiocraft.models", audiocraft_models)

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_musicgen_generate.py"
    spec = importlib.util.spec_from_file_location("_test_run_musicgen_generate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def test_load_model_infers_xp_config_for_standalone_merged_checkpoint(monkeypatch, tmp_path: Path) -> None:
    module, _calls = _load_helper_module(monkeypatch)
    xp = tmp_path / ".dora" / "xps" / "abc123"
    xp.mkdir(parents=True)
    source_checkpoint = xp / "checkpoint_30.th"
    source_checkpoint.write_bytes(b"source")
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"profile": "micro-unconditional"}),
        encoding="utf-8",
    )
    merged = tmp_path / "artifacts" / "checkpoints" / "merged" / "blend"
    merged.mkdir(parents=True)
    checkpoint = merged / "checkpoint.th"
    checkpoint.write_bytes(b"merged")
    (merged / "merged_from.json").write_text(
        json.dumps({"sources": {"a": {"path": str(source_checkpoint)}}}),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    class FakeLm:
        def eval(self) -> None:
            seen["eval"] = True

    class FakeSolver:
        compression_model = object()
        model = FakeLm()
        cfg = {"dataset": {"segment_duration": 10}}
        stateful = types.SimpleNamespace(sources={})

        def load_state_dict(self, state):  # type: ignore[no-untyped-def]
            seen["state"] = state

    def fake_get_solver_from_sig(sig, **kwargs):  # type: ignore[no-untyped-def]
        seen["sig"] = sig
        return FakeSolver()

    train = types.ModuleType("audiocraft.train")
    train.get_solver_from_sig = fake_get_solver_from_sig
    checkpoint_utils = types.ModuleType("audiocraft.utils.checkpoint")
    checkpoint_utils.load_checkpoint = lambda path: {"model": {"weight": str(path)}}
    audiocraft_utils = types.ModuleType("audiocraft.utils")
    audiocraft_utils.checkpoint = checkpoint_utils
    sys.modules["audiocraft"].train = train
    sys.modules["audiocraft"].utils = audiocraft_utils
    monkeypatch.setitem(sys.modules, "audiocraft.train", train)
    monkeypatch.setitem(sys.modules, "audiocraft.utils", audiocraft_utils)
    monkeypatch.setitem(sys.modules, "audiocraft.utils.checkpoint", checkpoint_utils)

    args = types.SimpleNamespace(
        checkpoint_path=str(checkpoint),
        xp_path=None,
        profile=None,
        model_id="dummy/micro",
    )

    _model, source = module._load_model(args, "cpu")

    assert seen["sig"] == "abc123"
    assert source["kind"] == "checkpoint"
    assert source["profile"] == "micro-unconditional"
    assert source["xp_path"] == str(xp.resolve())
    assert source["checkpoint_path"] == str(checkpoint.resolve())
