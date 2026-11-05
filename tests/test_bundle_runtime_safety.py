"""CPU-safety tests for ``macchina_ricordi.models.bundle_runtime.load_bundle_model``.

These run only where a real (torch + AudioCraft + omegaconf + cached codec)
runtime is installed: they exercise the *real* builder against an untrusted
bundle's declared architecture, proving that a valid bundle loads with exact
weight parity and preserved dtype, and that a wrong-shape or absurdly-large
bundle is rejected *before* any real model geometry and before the codec is
ever fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("audiocraft")


def tiny_config() -> dict:
    return {
        "lm_model": "transformer_lm",
        "transformer_lm": {
            "dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "n_q": 4,
            "card": 2048,
            "hidden_scale": 2,
            "causal": True,
            "custom": True,
            "memory_efficient": False,
            "dropout": 0.0,
        },
        "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
        "attribute_dropout": {},
        "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
        "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
        "conditioners": {},
        "dtype": "float32",
        "dataset": {"segment_duration": 1.0},
    }


AUDIO = {"sample_rate": 32000, "channels": 1, "frame_rate": 50.0}


@pytest.fixture(scope="module")
def real_model():
    from audiocraft.models.builders import get_lm_model
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    torch.manual_seed(7)
    return get_lm_model(OmegaConf.create({**tiny_config(), "device": "cpu"})).eval()


def _write(model_or_tensors, out: Path, *, name: str = "safety") -> None:
    from macchina_ricordi.models.bundle import write_bundle

    tensors = model_or_tensors if isinstance(model_or_tensors, dict) else model_or_tensors.state_dict()
    config = tiny_config()
    write_bundle(
        out,
        tensors,
        model_config=config,
        codec={"name": "facebook/encodec_32khz"},
        audio=AUDIO,
        segment_duration=1.0,
        name=name,
    )


def _codec_spy(monkeypatch):
    """Return a recorder for codec fetches; they must never happen for invalid bundles."""
    from audiocraft.solvers.compression import CompressionSolver

    calls: list[object] = []
    monkeypatch.setattr(CompressionSolver, "wrapped_model_from_checkpoint", lambda *a, **k: calls.append(a))
    return calls


def test_valid_bundle_loads_with_exact_parity(real_model, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    bundle = tmp_path / "valid"
    _write(real_model, bundle)
    loaded, source = load_bundle_model(bundle, device="cpu")
    original = real_model.state_dict()
    got = loaded.lm.state_dict()
    assert set(original) == set(got)
    for key in original:
        assert got[key].shape == original[key].shape, key
        assert torch.equal(got[key], original[key]), f"value drift on {key}"
        assert got[key].dtype == original[key].dtype, f"dtype drift on {key}"
    assert source["weights_state"] == "model"
    assert loaded.max_duration == 1.0


def test_bundle_dtype_is_preserved_by_strict_assign_load(real_model, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    bundle = tmp_path / "float16"
    half = {k: v.half() for k, v in real_model.state_dict().items()}
    _write(half, bundle)
    loaded, _ = load_bundle_model(bundle, device="cpu")
    got = loaded.lm.state_dict()
    assert all(got[k].dtype == torch.float16 for k in got), "assign must keep the bundled raw dtype"
    assert all(torch.equal(got[k], half[k]) for k in half), "float16 values must survive exactly"


def test_wrong_weight_shape_is_rejected_before_codec(real_model, tmp_path: Path, monkeypatch) -> None:
    from macchina_ricordi.models.bundle_runtime import BundleRuntimeError, load_bundle_model

    calls = _codec_spy(monkeypatch)
    bad = {k: v.clone() for k, v in real_model.state_dict().items()}
    # emb.0.weight is (2049, 16) for dim=16; resize one axis so it mismatches the architecture.
    bad["emb.0.weight"] = bad["emb.0.weight"].new_zeros(2049, 15)
    bundle = tmp_path / "badshape"
    _write(bad, bundle)
    with pytest.raises(BundleRuntimeError, match="emb.0.weight"):
        load_bundle_model(bundle, device="cpu")
    assert calls == [], "the codec must not be fetched for a wrong-shaped bundle"


def test_missing_weight_key_is_rejected_before_codec(real_model, tmp_path: Path, monkeypatch) -> None:
    from macchina_ricordi.models.bundle_runtime import BundleRuntimeError, load_bundle_model

    calls = _codec_spy(monkeypatch)
    keys = list(real_model.state_dict())
    truncated = {k: v for k, v in real_model.state_dict().items() if k != keys[-1]}
    bundle = tmp_path / "missing"
    _write(truncated, bundle)
    with pytest.raises(BundleRuntimeError, match="missing from the bundle"):
        load_bundle_model(bundle, device="cpu")
    assert calls == []


def test_huge_architecture_is_rejected_before_codec(real_model, tmp_path: Path, monkeypatch) -> None:
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    calls = _codec_spy(monkeypatch)
    bundle = tmp_path / "huge"
    _write(real_model, bundle)
    config = json.loads((bundle / "config.json").read_text())
    config["model_config"]["transformer_lm"]["num_layers"] = 10**9
    (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="absurdly large architecture"):
        load_bundle_model(bundle, device="cpu")
    assert calls == [], "the codec must not be fetched for an oversized architecture"


def test_huge_hidden_scale_is_rejected(real_model, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    bundle = tmp_path / "huge-hidden"
    _write(real_model, bundle)
    config = json.loads((bundle / "config.json").read_text())
    config["model_config"]["transformer_lm"]["hidden_scale"] = 10**12
    (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="absurdly large architecture"):
        load_bundle_model(bundle, device="cpu")


def test_unbounded_dim_over_bounds_is_rejected(real_model, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    bundle = tmp_path / "huge-dim"
    _write(real_model, bundle)
    config = json.loads((bundle / "config.json").read_text())
    config["model_config"]["transformer_lm"]["dim"] = 4096 * 4  # dim only stays bounded via the manifest
    (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="absurdly large architecture"):
        load_bundle_model(bundle, device="cpu")
