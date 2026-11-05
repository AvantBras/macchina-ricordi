"""Real CPU snapshot operations; skipped in the deliberately torch-free base environment."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest


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


@pytest.fixture
def snapshot(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("audiocraft")
    from audiocraft.models.builders import get_lm_model
    from omegaconf import OmegaConf

    from macchina_ricordi.models.bundle import write_bundle

    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = tiny_config()
    model = get_lm_model(OmegaConf.create({**config, "device": "cpu"})).eval()
    bundle = write_bundle(
        tmp_path / "source",
        model.state_dict(),
        model_config=config,
        codec={"name": "facebook/encodec_32khz"},
        audio={"sample_rate": 32000, "channels": 1, "frame_rate": 50.0},
        segment_duration=1.0,
        name="cpu-test-snapshot",
        provenance={"epoch": 7},
        license_text="Test fixture licence and attribution; preserve verbatim.\n",
    )
    return torch, model, bundle


def test_real_bundle_export_retains_exact_selected_snapshot(snapshot, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from macchina_ricordi.models.bundle import export_checkpoint_bundle, load_bundle_tensors

    torch, model, _ = snapshot
    weights = model.state_dict()
    other = {k: v + 1 if v.is_floating_point() else v.clone() for k, v in weights.items()}
    checkpoint = tmp_path / "trusted.th"
    torch.save({"model": weights, "best_state": {"model": other}, "optimizer": {"private": "not exported"}}, checkpoint)
    config = tiny_config()
    config.update(
        {
            "compression_model_checkpoint": "//pretrained/facebook/encodec_32khz",
            "sample_rate": 32000,
            "channels": 1,
            "datasource": {"train": "/private/not-exported"},
        }
    )
    config_path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    output = export_checkpoint_bundle(checkpoint, config_path, tmp_path / "exported", name="chosen-epoch")
    loaded = load_bundle_tensors(output.path)
    assert set(loaded) == set(weights)
    assert all(torch.equal(loaded[k], v) and loaded[k].dtype == v.dtype for k, v in weights.items())
    text = (output.path / "config.json").read_text()
    assert "private" not in text and "optimizer" not in text and "datasource" not in text
    assert output.config["weights"]["selection"] == "model"


def test_real_zero_perturbation_is_lossless_and_self_contained(snapshot, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle import load_bundle_tensors, read_bundle
    from macchina_ricordi.models.perturb import perturb_checkpoint_files

    torch, original, source = snapshot
    output = tmp_path / "child"
    perturb_checkpoint_files(
        source_checkpoint=source.path, output_checkpoint=output, sigma=0.0, layer_scope="all", seed=42, force=False
    )
    assert (output / "LICENSE").read_text() == (source.path / "LICENSE").read_text()
    shutil.rmtree(source.path)
    child = read_bundle(output)
    loaded = load_bundle_tensors(child.path)
    assert all(torch.equal(loaded[k], v) for k, v in original.state_dict().items())
    assert child.config["provenance"]["parent_sha256"] == source.config["weights"]["sha256"]


def test_real_perturbation_changes_weights_and_can_be_loaded(snapshot, tmp_path: Path) -> None:
    from macchina_ricordi.models.bundle import load_bundle_tensors, read_bundle
    from macchina_ricordi.models.perturb import perturb_checkpoint_files

    torch, model, source = snapshot
    output = tmp_path / "perturbed"
    perturb_checkpoint_files(
        source_checkpoint=source.path, output_checkpoint=output, sigma=0.01, layer_scope="all", seed=42, force=False
    )
    child = read_bundle(output)
    assert child.config["weights"]["sha256"] != source.config["weights"]["sha256"]
    model.load_state_dict(load_bundle_tensors(output), strict=True)
    with torch.no_grad():
        tokens = model.generate(num_samples=1, max_gen_len=4, use_sampling=False)
    assert tuple(tokens.shape) == (1, 4, 4)


def test_real_warm_start_adapter_uses_published_weights_and_fresh_optimizer(snapshot, tmp_path: Path) -> None:
    from audiocraft.models.builders import get_lm_model
    from omegaconf import OmegaConf

    from macchina_ricordi.models.checkpoints import fetch_init_checkpoint

    torch, original, source = snapshot
    resolved = fetch_init_checkpoint(str(source.path), repo_root=tmp_path / "checkout", musicgen_python=sys.executable)
    # The derived cache is tensor-only and can be read with PyTorch's restricted loader.
    state = torch.load(resolved.resolved_local_path, map_location="cpu", weights_only=True)
    assert set(state) == {"model", "best_state"}
    assert all(torch.equal(state["model"][k], v) for k, v in original.state_dict().items())
    assert all(torch.equal(state["best_state"]["model"][k], v) for k, v in original.state_dict().items())
    fresh = get_lm_model(OmegaConf.create({**tiny_config(), "device": "cpu"}))
    fresh.load_state_dict(state["best_state"]["model"], strict=True)
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.001)
    assert not optimizer.state
    before = copy.deepcopy(fresh.state_dict())
    # A synthetic differentiable step checks trainability, not music-training quality.
    loss = sum(p.square().mean() for p in fresh.parameters())
    loss.backward()
    optimizer.step()
    assert optimizer.state
    assert any(not torch.equal(before[k], v) for k, v in fresh.state_dict().items())
    assert resolved.origin == "bundle" and resolved.bundle_config is not None
    assert not (tmp_path / "checkout" / "datasets").exists()


def test_portable_config_is_not_a_full_training_history(snapshot) -> None:
    _, _, source = snapshot
    payload = json.loads((source.path / "config.json").read_text())
    assert "datasource" not in payload["model_config"]
    assert payload["model_config"]["dataset"] == {"segment_duration": 1.0}
    assert set(p.name for p in source.path.iterdir()) == {"model.safetensors", "config.json", "README.md", "LICENSE"}
