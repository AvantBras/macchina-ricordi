from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from macchina_ricordi.models.bundle import BundleError, export_checkpoint_bundle, read_bundle, write_bundle


def config() -> dict:
    return {
        "lm_model": "transformer_lm",
        "transformer_lm": {"dim": 16, "num_heads": 2, "num_layers": 1, "n_q": 4, "card": 2048, "hidden_scale": 2},
        "dtype": "float32",
        "conditioners": None,
        "attribute_dropout": {},
        "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
        "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
        "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
        "dataset": {"segment_duration": 1.0},
    }


def write(path: Path, tensors: dict | None = None, **kwargs):
    return write_bundle(
        path,
        tensors if tensors is not None else {"w": np.ones((2, 3), dtype=np.float32)},
        model_config=config(),
        codec={"name": "facebook/encodec_32khz"},
        audio={"sample_rate": 32000, "channels": 1, "frame_rate": 50.0},
        segment_duration=1.0,
        name="regression",
        **kwargs,
    )


def test_numpy_scalar_buffer_keeps_its_shape(tmp_path: Path) -> None:
    from safetensors import safe_open

    bundle = write(tmp_path / "model", {"counter": np.array(7, dtype=np.int64)})
    with safe_open(str(bundle.path / "model.safetensors"), framework="numpy") as handle:
        value = handle.get_tensor("counter")
    assert value.shape == () and value.dtype == np.int64 and value.item() == 7
    assert bundle.config["model_config"]["conditioners"] is None


def test_force_cannot_write_through_a_managed_directory_symlink(tmp_path: Path) -> None:
    bundle = write(tmp_path / "model")
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_text("untouched")
    (bundle.path / "model.safetensors").unlink()
    (bundle.path / "model.safetensors").symlink_to(external, target_is_directory=True)
    with pytest.raises(BundleError, match="unsafe"):
        write(bundle.path, force=True)
    assert sorted(p.name for p in external.iterdir()) == ["sentinel"]
    assert (external / "sentinel").read_text() == "untouched"


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c["model_config"]["transformer_lm"].update(activation="${oc.env:HOME}"),
        lambda c: c["audio"].update(frame_rate=float("nan")),
        lambda c: c["model_config"].update(datasource={"train": "/not-allowed"}),
    ],
)
def test_unresolved_or_invalid_public_config_is_rejected(tmp_path: Path, change) -> None:
    bundle = write(tmp_path / "model")
    payload = copy.deepcopy(bundle.config)
    change(payload)
    (bundle.path / "config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_bundle(bundle.path)


def test_custom_codec_cannot_be_silently_relabelled_as_encodec(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.th"
    checkpoint.write_bytes(b"not loaded: codec must be checked first")
    cfg = config()
    cfg["compression_model_checkpoint"] = "/local/custom-codec.th"
    with pytest.raises(BundleError, match="codec"):
        export_checkpoint_bundle(checkpoint, cfg, tmp_path / "output", name="custom")
    assert not (tmp_path / "output").exists()


def test_bfloat16_and_float16_tensors_are_preserved(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from macchina_ricordi.models.bundle import load_bundle_tensors

    original = {
        "bf16": torch.tensor([0.5, 1.25], dtype=torch.bfloat16),
        "fp16": torch.tensor([2.0, -3.5], dtype=torch.float16),
        "counter": torch.tensor(7, dtype=torch.int64),
    }
    bundle = write(tmp_path / "model", original)
    loaded = load_bundle_tensors(bundle.path)
    for name, tensor in original.items():
        assert loaded[name].dtype == tensor.dtype
        assert loaded[name].shape == tensor.shape
        assert torch.equal(loaded[name], tensor)


def test_bundle_cli_runner_does_not_precreate_its_output(tmp_path: Path) -> None:
    from macchina_ricordi.models.export import BundleExportPlan, run_bundle_export

    output = tmp_path / "snapshots" / "regression"
    plan = BundleExportPlan(
        name="regression",
        weights_state="model",
        bundle_dir=str(output),
        manifest_path=str(output / "config.json"),
        weights_path=str(output / "model.safetensors"),
        provenance={},
        checkpoint="source.th",
        helper_command=("unused-test-runner",),
        missing=(),
        force=False,
    )

    def runner(*args, **kwargs):
        from subprocess import CompletedProcess

        assert not output.exists()
        write(output)
        return CompletedProcess(args[0], 0)

    result = run_bundle_export(plan, repo_root=tmp_path, run=runner)
    assert result["status"] == "completed"
    read_bundle(output)


def test_box_plan_passes_a_bundle_directory_not_a_dora_signature(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from macchina_ricordi.models.export import plan_box_export
    from macchina_ricordi.models.record import ModelNotFound

    bundle = write(tmp_path / "models" / "snapshot")
    monkeypatch.setattr(
        "macchina_ricordi.models.export.load_musicgen_project_config",
        lambda: SimpleNamespace(repo_root=tmp_path, musicgen_python=Path(sys.executable)),
    )
    plan = plan_box_export(str(bundle.path), pack=False, repo_root=tmp_path)
    assert str(bundle.path) in plan.export_command
    assert plan.card["kind"] == "bundle"
    with pytest.raises(ModelNotFound, match="one selected snapshot"):
        plan_box_export(str(bundle.path), weights_state="ema", pack=False, repo_root=tmp_path)
