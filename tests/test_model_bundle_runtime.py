"""Tests for ``macchina_ricordi.models.bundle_runtime``.

The runtime loader builds a top-level ``MusicGen`` from a portable bundle
without any Dora solver or XP. Torch/AudioCraft are mocked here (never claimed
to run real GPU inference); a separate CPU-torch validation env exercises the
real builder. The manifest/torch-free helpers are tested against a real bundle
written and read by the shared core without torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macchina_ricordi.models.bundle import write_bundle
from macchina_ricordi.models.bundle_runtime import (
    BundleRuntimeError,
    NotABundleError,
    bundle_dir_or_none,
    bundle_segment_duration,
    bundle_to_source_metadata,
    is_bundle_dir,
    load_bundle_model,
)

BASE_MODEL_CONFIG = {
    "lm_model": "transformer_lm",
    "transformer_lm": {
        "n_q": 4,
        "card": 2048,
        "dim": 64,
        "num_heads": 4,
        "num_layers": 1,
        "hidden_scale": 4,
        "layer_scale": None,
    },
    "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
    "attribute_dropout": {},
    "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 3.0},
    "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
    "conditioners": {},
    "dataset": {"segment_duration": 8.0},
    "dtype": "float32",
}

BASE_AUDIO = {"sample_rate": 32000, "channels": 1, "frame_rate": 50.0}


def write_tiny_bundle(tmp_path: Path, **manifest_overrides) -> Path:
    out = tmp_path / "model.bundle"
    tensors = {
        name: np.random.randn(*shape).astype(np.float32)
        for name, shape in {
            "emb.0.weight": (400, 64),
            "transformer.layers.0.self_attn.mha.in_proj_weight": (256, 64),
            "lm_head.weight": (400, 64),
        }.items()
    }
    return write_bundle(
        out,
        tensors,
        model_config=BASE_MODEL_CONFIG,
        codec={"name": "facebook/encodec_32khz"},
        audio=BASE_AUDIO,
        segment_duration=8.0,
        name="tiny",
        provenance={"recipe": "reshape-nano", "profile": "reshape-nano"},
        **manifest_overrides,
    ).path


def _stub_model_module():
    """Real audiocraft/parts are heavy; substitute minimal stand-ins for the loader's expects.

    These are pure control-flow doubles, never run on a GPU — the loader only
    calls ``get_lm_model`` (once on the meta device to prove the weight keys +
    shapes, once on the real device), ``state_dict``, ``load_state_dict``,
    ``eval``, the codec static loader and the ``MusicGen`` constructor.
    """

    # The three weights a tiny fixture carries, with matching shapes. The loader
    # validates the meta state_dict against the safetensors header, so the fake
    # model must report exactly the bundle's keys and shapes.
    FAKE_STATE = {
        "emb.0.weight": np.empty((400, 64), dtype=np.float32),
        "transformer.layers.0.self_attn.mha.in_proj_weight": np.empty((256, 64), dtype=np.float32),
        "lm_head.weight": np.empty((400, 64), dtype=np.float32),
    }

    class FakeLm:
        def __init__(self):
            self.loaded = None
            self.strict = None
            self.assign = None
            self.eval_called = False

        def state_dict(self):
            return {name: tensor.copy() for name, tensor in FAKE_STATE.items()}

        def load_state_dict(self, tensors, strict=True, assign=False):
            self.loaded = tensors
            self.strict = strict
            self.assign = assign

        def eval(self):
            self.eval_called = True

    fake_lm = FakeLm()

    class FakeCodec:
        sample_rate = 32000
        channels = 1
        frame_rate = 50.0

        def eval(self):
            pass

    fake_codec = FakeCodec()

    class FakeMusicGen:
        def __init__(self, name, compression_model, lm, max_duration=0.0):
            self.name = name
            self.compression_model = compression_model
            self.lm = lm
            self.max_duration = max_duration
            self.constructor_max_duration = max_duration

        @property
        def sample_rate(self):
            return 32000

    builders = SimpleNamespace(get_lm_model=lambda cfg: fake_lm)
    CompressionSolver = SimpleNamespace(wrapped_model_from_checkpoint=lambda cfg, path, device: fake_codec)
    MusicGen = FakeMusicGen
    return builders, CompressionSolver, MusicGen, fake_lm


def _install_fake_audiocraft(monkeypatch: pytest.MonkeyPatch):
    import sys

    builders, CompressionSolver, MusicGen, fake_lm = _stub_model_module()
    audiocraft = SimpleNamespace()
    models = SimpleNamespace(builders=builders, MusicGen=MusicGen)
    solvers = SimpleNamespace(compression=SimpleNamespace(CompressionSolver=CompressionSolver))
    monkeypatch.setitem(sys.modules, "audiocraft", audiocraft)
    monkeypatch.setitem(sys.modules, "audiocraft.models", models)
    monkeypatch.setitem(sys.modules, "audiocraft.models.builders", builders)
    monkeypatch.setitem(sys.modules, "audiocraft.solvers", solvers)
    monkeypatch.setitem(sys.modules, "audiocraft.solvers.compression", solvers.compression)

    # omegaconf is a runtime-only dependency; the base test venv has torch-free items only.
    class _FakeOmegaConf:
        @staticmethod
        def create(value):
            return value

    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=_FakeOmegaConf()))
    return fake_lm


# ── torch-free helpers (real bundle, no torch) ────────────────────────────────


def test_is_bundle_dir_and_segment_and_metadata(tmp_path: Path) -> None:
    bundle_path = write_tiny_bundle(tmp_path)
    assert is_bundle_dir(bundle_path) is True
    assert bundle_dir_or_none(str(bundle_path) + "/") == bundle_path.resolve()
    assert bundle_segment_duration(bundle_path) == 8.0
    assert bundle_dir_or_none(tmp_path / "nope") is None
    assert is_bundle_dir(tmp_path / "nope") is False


def test_bundle_dir_or_none_rejects_plain_folder(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-bundle"
    plain.mkdir()
    assert bundle_dir_or_none(plain) is None
    assert bundle_dir_or_none(None) is None


def test_bundle_to_source_metadata_from_manifest(tmp_path: Path) -> None:
    bundle_path = write_tiny_bundle(tmp_path)
    from macchina_ricordi.models.bundle import read_bundle

    bundle = read_bundle(bundle_path)
    source = bundle_to_source_metadata(bundle, bundle_path)
    assert source["kind"] == "bundle"
    assert source["bundle_path"] == str(bundle_path.resolve())
    assert source["weights_state"] == "model"
    assert source["sample_rate"] == 32000
    assert source["frame_rate"] == 50.0
    assert source["segment_duration"] == 8.0
    assert source["profile"] == "reshape-nano"


# ── loader control-flow (mocked torch/audiocraft) ─────────────────────────────


def test_load_bundle_model_wires_the_whole_pipeline(monkeypatch, tmp_path: Path) -> None:
    fake_lm = _install_fake_audiocraft(monkeypatch)
    bundle_path = write_tiny_bundle(tmp_path)

    model, source = load_bundle_model(bundle_path, device="cpu")

    assert model.lm is fake_lm
    assert fake_lm.eval_called is True
    assert fake_lm.loaded is not None
    assert fake_lm.strict is True
    assert fake_lm.assign is True  # keep the bundled tensor's own dtype
    assert any(k.startswith("emb.") for k in fake_lm.loaded)
    assert source["kind"] == "bundle"
    assert source["weights_state"] == "model"
    assert model.max_duration == 8.0
    assert model.constructor_max_duration == 19.0  # AudioCraft constructor guard


def test_load_bundle_model_rejects_huge_arch_config(monkeypatch, tmp_path: Path) -> None:
    _install_fake_audiocraft(monkeypatch)
    bundle_path = write_tiny_bundle(tmp_path)
    config = json.loads((bundle_path / "config.json").read_text())
    config["model_config"]["transformer_lm"]["num_layers"] = 10**9  # hostile scale
    (bundle_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # The shared core rejects the oversized architecture at manifest-read time,
    # so no LM — not even a meta build — and no codec ever loads.
    with pytest.raises(ValueError, match="absurdly large architecture"):
        load_bundle_model(bundle_path)


def test_load_bundle_model_rejects_weight_shape_mismatch(monkeypatch, tmp_path: Path) -> None:
    fake_lm = _install_fake_audiocraft(monkeypatch)
    # Spy on the (fake) codec static loader: an invalid bundle must never reach it.
    import sys

    from audiocraft.solvers.compression import CompressionSolver

    codec_calls: list[object] = []
    monkeypatch.setattr(CompressionSolver, "wrapped_model_from_checkpoint", lambda *a, **k: codec_calls.append(a))
    # A manifest-valid bundle whose emb.0.weight shape disagrees with the
    # architecture the meta model reports (400, 64 vs 400, 65).
    out = tmp_path / "shape-mismatch"
    from macchina_ricordi.models.bundle import write_bundle

    write_bundle(
        out,
        {
            "emb.0.weight": np.zeros((400, 65), dtype=np.float32),
            "transformer.layers.0.self_attn.mha.in_proj_weight": np.zeros((256, 64), dtype=np.float32),
            "lm_head.weight": np.zeros((400, 64), dtype=np.float32),
        },
        model_config=BASE_MODEL_CONFIG,
        codec={"name": "facebook/encodec_32khz"},
        audio=BASE_AUDIO,
        segment_duration=8.0,
        name="shape-mismatch",
    )
    with pytest.raises(BundleRuntimeError, match="shape"):
        load_bundle_model(out)
    assert codec_calls == []  # the codec is never fetched for a bad bundle
    assert fake_lm.eval_called is False  # no real LM was built or eval'ed
    assert sys.modules["audiocraft.solvers.compression"] is not None  # sanity: fake env active


def test_load_bundle_model_raises_not_a_bundle(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-bundle"
    plain.mkdir()
    with pytest.raises(NotABundleError):
        load_bundle_model(plain)


def test_load_bundle_model_rejects_codec_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_fake_audiocraft(monkeypatch)
    bundle_path = write_tiny_bundle(tmp_path)
    (bundle_path / "config.json").write_text(
        json.dumps(
            {
                **json.loads((bundle_path / "config.json").read_text()),
                "audio": {"sample_rate": 16000, "channels": 1, "frame_rate": 50.0},
            }
        ),
        encoding="utf-8",
    )
    # The shared core refuses a non-standard codec manifest before the loader runs.
    with pytest.raises(ValueError):
        load_bundle_model(bundle_path)
