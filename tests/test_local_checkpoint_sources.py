from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.models.checkpoints import (
    discover_xp_folder,
    fetch_init_checkpoint,
    parse_init_checkpoint_uri,
    validate_bundle_training_compatibility,
)


@pytest.mark.parametrize("uri", ["hf://author/model", "https://example.org/model.th", "file:///weights.th"])
def test_remote_checkpoint_protocol_is_retired(uri: str) -> None:
    with pytest.raises(ValueError, match="Download and extract"):
        parse_init_checkpoint_uri(uri)


def test_local_checkpoint_requires_no_dataset_or_hf(monkeypatch, tmp_path: Path) -> None:
    import sys

    # Any accidental access to HF would fail rather than silently succeeding.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    checkpoint = tmp_path / "checkpoint.th"
    checkpoint.write_bytes(b"local trusted checkpoint fixture")
    source = fetch_init_checkpoint(str(checkpoint))
    assert source.origin == "local"
    assert source.resolved_local_path == checkpoint.resolve()
    assert source.resume_metadata_summary is None


def test_missing_local_source_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        fetch_init_checkpoint(str(tmp_path / "missing.th"))


def test_local_sidecar_discovery_rejects_ambiguity(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_xp_folder(dora_dir=tmp_path)
    for name in ("aaaa1111", "bbbb2222"):
        p = tmp_path / "xps" / name / "checkpoint.th"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"test")
    with pytest.raises(RuntimeError, match="Multiple"):
        discover_xp_folder(dora_dir=tmp_path)
    assert discover_xp_folder(tmp_path / "xps" / "aaaa1111") == tmp_path / "xps" / "aaaa1111"


def test_bundle_target_shape_and_codec_are_checked() -> None:
    config = {
        "codec": {"name": "facebook/encodec_32khz"},
        "model_config": {"transformer_lm": {"dim": 192, "num_layers": 10}},
    }
    codec = "//pretrained/facebook/encodec_32khz"
    validate_bundle_training_compatibility(config, ["transformer_lm.dim=192", "optim.lr=0.001"], codec)
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_bundle_training_compatibility(config, ["transformer_lm.dim=512"], codec)
    with pytest.raises(ValueError, match="codec mismatch"):
        validate_bundle_training_compatibility(config, [], "/local/custom-codec.th")


def test_retired_cli_routes_reject_without_execution() -> None:
    from macchina_ricordi.cli import build_parser, translate_legacy

    parser = build_parser()
    for argv in (["model", "push"], ["model", "pull"], ["musicgen", "hub", "upload"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(translate_legacy(argv)[0])
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["train", "run", "--dataset", "x", "--init-checkpoint-token", "removed-option"])
    assert exc.value.code == 2
