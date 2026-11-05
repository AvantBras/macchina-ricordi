from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.models.bundle import (
    BUNDLE_SCHEMA,
    ENCODEC_32KHZ_AUDIO,
    ENCODEC_32KHZ_CODEC,
    WEIGHTS_FILENAME,
    BundleError,
    export_checkpoint_bundle,
    is_bundle,
    load_bundle_tensors,
    read_bundle,
    select_checkpoint_weights,
    write_bundle,
)

try:  # torch is optional; real-tensor tests skip when it is absent.
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - torch is usually not in the base env
    HAS_TORCH = False

real_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (real tensor tests need it)")

# The bare builder config every base test writes. Shapes satisfy the v1 contract.
_TRANSFORMER = {
    "n_q": 4,
    "card": 2048,
    "dim": 32,
    "num_heads": 4,
    "hidden_scale": 4,
    "num_layers": 2,
}
MODEL_CONFIG = {
    "lm_model": "transformer_lm",
    "transformer_lm": dict(_TRANSFORMER),
    "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
    "attribute_dropout": {},
    "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
    "fuser": {"cross": [], "prepend": [], "sum": [], "input_interpolate": []},
    "conditioners": {},
    "dtype": "float32",
    "dataset": {"segment_duration": 10.0},
}


def sample_tensors() -> dict[str, np.ndarray]:
    return {
        "emb.0.weight": np.arange(9, dtype=np.float32).reshape(3, 3),
        "linears.0.weight": np.ones((2, 4), dtype=np.float32),
        "linears.0.bias": np.zeros(2, dtype=np.float32),
        "codes": np.array([0, 1, 2, 3], dtype=np.int64),
    }


def _as_numpy(value) -> np.ndarray:
    """A tensor (torch or numpy) seen as a numpy array, so assertions hold either way."""
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def write_sample(tmp_path: Path, **overrides) -> Path:
    out = tmp_path / "bundle"
    return write_bundle(
        out,
        sample_tensors(),
        model_config=MODEL_CONFIG,
        codec=ENCODEC_32KHZ_CODEC,
        audio=ENCODEC_32KHZ_AUDIO,
        segment_duration=10.0,
        name="demo",
        **overrides,
    )


# ── write / read round-trip (real numpy safetensors I/O) ────────────────


def test_write_read_roundtrip_preserves_values_and_dtypes(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    assert bundle.path.is_dir()
    assert (bundle.path / WEIGHTS_FILENAME).is_file()
    assert (bundle.path / "config.json").is_file()
    assert (bundle.path / "README.md").is_file()
    assert (bundle.path / "LICENSE").is_file()

    assert bundle.sha256 is not None

    reopened = read_bundle(bundle.path, verify=True)
    assert reopened.config["schema"] == BUNDLE_SCHEMA
    assert reopened.name == "demo"
    assert reopened.segment_duration == 10.0
    assert reopened.codec == "facebook/encodec_32khz"
    assert reopened.selection == "model"
    assert reopened.sha256 == bundle.sha256

    tensors = load_bundle_tensors(str(bundle.path), device="cpu")
    assert set(tensors) == set(sample_tensors())
    assert np.array_equal(_as_numpy(tensors["codes"]), sample_tensors()["codes"])
    assert _as_numpy(tensors["codes"]).dtype == np.int64
    np.testing.assert_array_equal(_as_numpy(tensors["emb.0.weight"]), sample_tensors()["emb.0.weight"])
    assert _as_numpy(tensors["emb.0.weight"]).dtype == np.float32


def test_load_via_manifest_file_path(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    via_file = read_bundle(bundle.path / "config.json", verify=True)
    assert via_file.path == bundle.path


def test_is_bundle(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    assert is_bundle(bundle.path)
    assert is_bundle(bundle.path / "config.json")
    assert not is_bundle(tmp_path)
    assert not is_bundle(bundle.path / WEIGHTS_FILENAME)


def test_license_and_readme_text_are_preserved_verbatim(tmp_path: Path) -> None:
    readme = "# custom readme\nkept attributions\n"
    license_text = "Copyright (c) Pierre-André Arcand\nMIT\n"
    out = write_bundle(
        tmp_path / "bundle",
        sample_tensors(),
        model_config=MODEL_CONFIG,
        codec=ENCODEC_32KHZ_CODEC,
        audio=ENCODEC_32KHZ_AUDIO,
        segment_duration=10.0,
        name="demo",
        license_text=license_text,
        readme_text=readme,
    )
    assert (out.path / "LICENSE").read_text(encoding="utf-8") == license_text
    assert (out.path / "README.md").read_text(encoding="utf-8") == readme


def test_manifest_json_fields_are_those_of_the_contract(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    payload = json.loads((bundle.path / "config.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "name",
        "model_config",
        "codec",
        "audio",
        "segment_duration",
        "weights",
        "provenance",
    }
    assert payload["audio"] == {"sample_rate": 32000, "channels": 1, "frame_rate": 50.0}
    assert payload["weights"]["file"] == "model.safetensors"
    # Only whitelisted builder keys survive, never datasource/training/paths.
    assert set(payload["model_config"]) <= {
        "lm_model",
        "transformer_lm",
        "codebooks_pattern",
        "attribute_dropout",
        "classifier_free_guidance",
        "fuser",
        "conditioners",
        "dtype",
        "dataset",
    }


# ── overwrite / staging safety ──────────────────────────────────────────


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    write_sample(tmp_path)
    with pytest.raises(FileExistsError):
        write_sample(tmp_path)
    # The untouched first bundle still verifies.
    assert read_bundle(tmp_path / "bundle", verify=True).sha256 is not None


def test_force_preserves_unrelated_files(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    unrelated = bundle.path / "keep-me.txt"
    unrelated.write_text("sentinel", encoding="utf-8")
    write_sample(tmp_path, force=True)
    assert unrelated.read_text(encoding="utf-8") == "sentinel"
    # The weights were replaced with a fresh (identical) file.
    assert read_bundle(bundle.path, verify=True).sha256 == bundle.sha256


def test_output_must_be_a_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "bundle"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        write_sample_blocker(tmp_path, blocker)


def write_sample_blocker(tmp_path: Path, out: Path) -> Path:
    return write_bundle(
        out,
        sample_tensors(),
        model_config=MODEL_CONFIG,
        codec=ENCODEC_32KHZ_CODEC,
        audio=ENCODEC_32KHZ_AUDIO,
        segment_duration=10.0,
        name="demo",
    )


# ── verification / security ─────────────────────────────────────────────


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    with open(bundle.path / WEIGHTS_FILENAME, "r+b") as handle:
        handle.seek(-1, 2)
        handle.write(bytes([handle.read(1)[0] ^ 0xFF]))
    with pytest.raises(BundleError, match="checksum mismatch"):
        read_bundle(bundle.path, verify=True)


def test_corrupted_safetensors_header_raises(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    path = bundle.path / WEIGHTS_FILENAME
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    mangled = raw[:8] + b"\xff" * 2 + raw[10:]
    path.write_bytes(mangled)
    assert struct.unpack("<Q", mangled[:8])[0] == length  # header length unchanged
    with pytest.raises(BundleError, match="malformed safetensors header"):
        read_bundle(bundle.path, verify=True)


def test_symlinked_weights_refused(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    (bundle.path / WEIGHTS_FILENAME).unlink()
    outside = tmp_path / "outside.safetensors"
    write_bundle(
        outside,
        sample_tensors(),
        model_config=MODEL_CONFIG,
        codec=ENCODEC_32KHZ_CODEC,
        audio=ENCODEC_32KHZ_AUDIO,
        segment_duration=10.0,
        name="outside",
    )
    (bundle.path / WEIGHTS_FILENAME).symlink_to(outside / WEIGHTS_FILENAME)
    with pytest.raises(BundleError, match="symlink"):
        read_bundle(bundle.path, verify=True)


def test_unsupported_schema_raises(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    manifest = bundle.path / "config.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = "macchina_ricordi.something_else.v9"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="unsupported bundle schema"):
        read_bundle(bundle.path, verify=True)


def test_malformed_manifest_raises(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    (bundle.path / "config.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(BundleError, match="not readable JSON"):
        read_bundle(bundle.path, verify=False)


def test_manifest_weights_file_cannot_traverse(tmp_path: Path) -> None:
    bundle = write_sample(tmp_path)
    manifest = bundle.path / "config.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["weights"]["file"] = "../evil.safetensors"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleError, match="weights.file"):
        read_bundle(bundle.path, verify=False)


# ── provenance scrubbing ────────────────────────────────────────────────


def test_provenance_scrubs_paths_and_logs_keeps_scalars(tmp_path: Path) -> None:
    bundle = write_bundle(
        tmp_path / "bundle",
        sample_tensors(),
        model_config=MODEL_CONFIG,
        codec=ENCODEC_32KHZ_CODEC,
        audio=ENCODEC_32KHZ_AUDIO,
        segment_duration=10.0,
        name="demo",
        provenance={
            "sig": "4ae63477",
            "epoch": 47,
            "recipe": "reshape-nano",
            "parents": ["abc12345", "def67890"],
            "log_path": "/home/me/.dora/xps/4ae63477/log.txt",
            "multi_line": "line one\nline two",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    prov = bundle.provenance
    assert prov["sig"] == "4ae63477" and prov["epoch"] == 47 and prov["recipe"] == "reshape-nano"
    assert prov["parents"] == ["abc12345", "def67890"]
    assert prov["created_at"] == "2026-01-01T00:00:00Z"
    assert "log_path" not in prov
    assert "multi_line" not in prov


# ── config validation ───────────────────────────────────────────────────


def _cfg(**changes) -> dict:
    cfg = json.loads(json.dumps(MODEL_CONFIG))
    cfg.update(changes)
    return cfg


@pytest.mark.parametrize(
    "mutator",
    [
        lambda c: c.update(lm_model="transformer_lm_magnet"),
        lambda c: c.update(conditioners={"description": {"model": "t5"}}),
        lambda c: c["transformer_lm"].update(dim=0),
        lambda c: c["transformer_lm"].update(card="2048"),
        lambda c: c.update(dtype="float64"),
        lambda c: c["dataset"].update(segment_duration=0),
    ],
)
def test_invalid_config_rejected(tmp_path: Path, mutator) -> None:
    cfg = _cfg()
    mutator(cfg)
    with pytest.raises(BundleError):
        write_bundle(
            tmp_path / "bundle",
            sample_tensors(),
            model_config=cfg,
            codec=ENCODEC_32KHZ_CODEC,
            audio=ENCODEC_32KHZ_AUDIO,
            segment_duration=10.0,
            name="demo",
        )


def test_unsupported_codec_rejected(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="unsupported codec"):
        write_bundle(
            tmp_path / "bundle",
            sample_tensors(),
            model_config=MODEL_CONFIG,
            codec={"name": "facebook/custom_codec"},
            audio=ENCODEC_32KHZ_AUDIO,
            segment_duration=10.0,
            name="demo",
        )


def test_audio_contract_enforced(tmp_path: Path) -> None:
    bad_audio = dict(ENCODEC_32KHZ_AUDIO, sample_rate=16000)
    with pytest.raises(BundleError, match="sample_rate"):
        write_bundle(
            tmp_path / "bundle",
            sample_tensors(),
            model_config=MODEL_CONFIG,
            codec=ENCODEC_32KHZ_CODEC,
            audio=bad_audio,
            segment_duration=10.0,
            name="demo",
        )


def test_segment_duration_mismatch_rejected(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["dataset"]["segment_duration"] = 30.0
    with pytest.raises(BundleError, match="does not match segment_duration"):
        write_bundle(
            tmp_path / "bundle",
            sample_tensors(),
            model_config=cfg,
            codec=ENCODEC_32KHZ_CODEC,
            audio=ENCODEC_32KHZ_AUDIO,
            segment_duration=10.0,
            name="demo",
        )


# ── checkpoint weight selection ─────────────────────────────────────────


def _weights_dict() -> dict:
    # AudioCraft model state_dicts are FLAT: dotted keys, tensor leaves.
    return {
        "emb.0.weight": np.ones((3, 3), dtype=np.float32),
        "transformer.layers.0.self_attn.in_proj_weight": np.ones((2, 2), dtype=np.float32),
    }


def test_select_model_from_wrapped_state() -> None:
    state = {
        "model": _weights_dict(),
        "best_state": {"model": {"b": np.zeros((2, 2), dtype=np.float32)}},
        "ema": {"model": {"e": np.ones(2, dtype=np.float32)}},
    }
    assert select_checkpoint_weights(state, "model") is state["model"]


def test_select_model_from_bare_state_dict() -> None:
    weights = _weights_dict()
    assert select_checkpoint_weights(weights, "model") is weights


def test_select_best_from_best_state_model() -> None:
    state = {"model": {}, "best_state": {"model": {"b": np.ones(1, dtype=np.float32)}}}
    assert select_checkpoint_weights(state, "best")["b"] is not None


def test_select_best_from_fsdp_best_state_model() -> None:
    state = {"model": {}, "fsdp_best_state": {"model": {"b": np.ones(1, dtype=np.float32)}}}
    assert "b" in select_checkpoint_weights(state, "best")


def test_select_ema_from_ema_model() -> None:
    state = {"ema": {"model": {"e": np.ones(2, dtype=np.float32)}}}
    assert select_checkpoint_weights(state, "ema")["e"].size == 2


def test_select_ema_from_ema_state_model() -> None:
    state = {"ema": {"state": {"model": {"e": np.ones(2, dtype=np.float32)}}}}
    assert "e" in select_checkpoint_weights(state, "ema")


def test_missing_best_never_falls_back(tmp_path: Path) -> None:
    state = {"model": _weights_dict()}
    with pytest.raises(BundleError, match="best-state"):
        select_checkpoint_weights(state, "best")
    with pytest.raises(BundleError, match="EMA"):
        select_checkpoint_weights(state, "ema")


def test_unsupported_weights_state_rejected() -> None:
    with pytest.raises(BundleError, match="weights_state"):
        select_checkpoint_weights({"model": _weights_dict()}, "slerp")


# ── export (real file writing; checkpoint read injected for base) ───────


def test_export_with_injected_loader_writes_whitelisted_bundle(tmp_path: Path) -> None:
    weights = _weights_dict()
    state = {"model": weights, "ema": {"model": {"e": np.ones(2, dtype=np.float32)}}}
    checkpoint = tmp_path / "checkpoint.th"
    checkpoint.write_bytes(b"not-a-real-pickle")  # never opened: a loader is injected

    def fake_loader(path: Path) -> dict:
        assert path == checkpoint.resolve()
        return state

    bundle = export_checkpoint_bundle(
        checkpoint,
        dict(MODEL_CONFIG),  # a config dict stands in for the resolved Hydra source
        tmp_path / "out",
        name="demo",
        provenance={"sig": "4ae63477", "epoch": 47},
        loader=fake_loader,
    )
    payload = json.loads((bundle.path / "config.json").read_text(encoding="utf-8"))
    assert payload["provenance"] == {"sig": "4ae63477", "epoch": 47}
    assert set(payload["model_config"]) <= {
        "lm_model",
        "transformer_lm",
        "codebooks_pattern",
        "attribute_dropout",
        "classifier_free_guidance",
        "fuser",
        "conditioners",
        "dtype",
        "dataset",
    }
    reopened = read_bundle(bundle.path, verify=True)
    assert reopened.sha256 == bundle.sha256


# ── real torch tests (skip unless torch + safetensors are present) ──────


@real_torch
def test_real_torch_export_and_roundtrip(tmp_path: Path) -> None:
    import safetensors.torch  # noqa: F401  (guards that the torch plugin exists)

    from macchina_ricordi.models.bundle import load_bundle_tensors

    torch.manual_seed(0)
    model_w = {
        "emb.0.weight": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        "linears.0.weight": torch.ones(2, 4, dtype=torch.float64),
        "num_batches_tracked": torch.tensor([7], dtype=torch.int64),
    }
    best_w = {"emb.0.weight": torch.zeros(3, 3, dtype=torch.float32)}
    ema_w = {"emb.0.weight": torch.full((3, 3), 2.0, dtype=torch.float32)}
    state = {
        "model": model_w,
        "best_state": {"model": best_w},
        "ema": {"model": ema_w},
        "optimizer": {"ignore": 1},  # a stateful non-weight key at the root
    }
    checkpoint = tmp_path / "checkpoint.th"
    torch.save(state, checkpoint)

    bundle = export_checkpoint_bundle(
        checkpoint,
        dict(MODEL_CONFIG),
        tmp_path / "out",
        name="demo",
        weights_state="best",
        provenance={"epoch": 47},
    )
    assert bundle.selection == "best"
    reopened = read_bundle(bundle.path, verify=True)
    assert reopened.sha256 == bundle.sha256

    got = load_bundle_tensors(str(bundle.path), device="cpu")
    assert set(k for k in got) == {"emb.0.weight"}
    assert got["emb.0.weight"].dtype == torch.float32
    assert torch.equal(got["emb.0.weight"], best_w["emb.0.weight"])


@real_torch
def test_real_torch_checkpoint_default_loader_selects_model(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model_w = {"emb.0.weight": torch.randn(3, 3)}
    checkpoint = tmp_path / "checkpoint.th"
    torch.save({"model": model_w}, checkpoint)

    bundle = export_checkpoint_bundle(checkpoint, dict(MODEL_CONFIG), tmp_path / "out", name="demo")
    got = load_bundle_tensors(str(bundle.path), device="cpu")
    assert "emb.0.weight" in got and got["emb.0.weight"].shape == (3, 3)


@real_torch
def test_real_audiocraft_builder_rebuilds_from_stored_config(tmp_path: Path) -> None:
    """The whitelisted builder config is enough for get_lm_model to rebuild the LM."""
    pytest.importorskip("audiocraft")

    from audiocraft.models import builders
    from omegaconf import OmegaConf

    from macchina_ricordi.models.bundle import export_checkpoint_bundle, load_bundle_tensors

    torch.manual_seed(0)
    # A real solver-style config: dataset.segment_duration is used even though
    # the model is unconditional.
    cfg = OmegaConf.create(
        {
            "lm_model": "transformer_lm",
            "transformer_lm": {
                "n_q": 4,
                "card": 2048,
                "dim": 32,
                "num_heads": 4,
                "hidden_scale": 4,
                "num_layers": 2,
                "norm": "layer_norm",
                "norm_first": True,
                "weight_init": "gaussian",
                "depthwise_init": "current",
                "causal": True,
                "custom": True,
                "cross_attention": False,
                "dropout": 0.0,
                "activation": "gelu",
            },
            "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
            "attribute_dropout": {},
            "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
            "fuser": {"cross": [], "prepend": [], "sum": [], "input_interpolate": []},
            "conditioners": {},
            "dtype": "float32",
            "device": "cpu",
            "dataset": {"segment_duration": 10.0},
        }
    )
    model = builders.get_lm_model(cfg)
    checkpoint = tmp_path / "checkpoint.th"
    torch.save({"model": model.state_dict()}, checkpoint)

    # The stored model_config must be exactly the whitelist (no datasource/paths).
    bundle = export_checkpoint_bundle(checkpoint, {k: v for k, v in cfg.items()}, tmp_path / "out", name="demo")
    stored = bundle.model_config
    assert "datasource" not in stored and "device" not in stored and "optim" not in stored

    # Rebuild from the stored config plus load-time device; same key set.
    rebuild = OmegaConf.create({**stored, "device": "cpu"})
    rebuilt = builders.get_lm_model(rebuild)
    assert set(rebuilt.state_dict()) == set(model.state_dict())

    got = load_bundle_tensors(str(bundle.path), device="cpu")
    assert set(got) == set(model.state_dict())
    for key in list(model.state_dict())[:4]:
        assert got[key].shape == tuple(model.state_dict()[key].shape)
