from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.models.checkpoints import InitCheckpointSource
from macchina_ricordi.musicgen.templates import DATASET_CONFIG_NAME
from macchina_ricordi.training.plan import plan_musicgen_training


@pytest.mark.parametrize("prefix", ["", "++"])
@pytest.mark.parametrize("codec_override", [None, "//pretrained/facebook/encodec_32khz", "/custom/codec.th"])
def test_bundle_training_plan_uses_the_standard_codec_default(
    tmp_path: Path, monkeypatch, codec_override, prefix
) -> None:
    # A planner/control-flow regression; real bundle conversion is tested separately on CPU.
    workspace = tmp_path / "workspace"
    audiocraft = tmp_path / "audiocraft"
    python = tmp_path / "env/python.exe"
    python.parent.mkdir()
    python.write_text("")
    python.with_name("dora.exe").write_text("")
    (audiocraft / "config/dset/audio").mkdir(parents=True)
    for split in ("train", "valid"):
        manifest = workspace / "egs" / DATASET_CONFIG_NAME / split / "data.jsonl.gz"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
    (workspace / "logs").mkdir()
    (workspace / "logs/dataset_fingerprint.json").write_text(
        '{"profile":"micro-unconditional","dataset_name":"macchina_ricordi_micro_uncond",'
        '"train_aggregate_sha256":"fixture-train","valid_aggregate_sha256":"fixture-valid"}'
    )
    checkpoint = tmp_path / "derived/checkpoint.th"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"planner fixture, not loaded")
    source = InitCheckpointSource(
        "bundle",
        str(tmp_path / "published-bundle"),
        checkpoint,
        bundle_config={"codec": {"name": "facebook/encodec_32khz"}, "model_config": {"transformer_lm": {}}},
    )
    monkeypatch.setattr("macchina_ricordi.models.checkpoints.fetch_init_checkpoint", lambda *args, **kwargs: source)
    kwargs = dict(
        audiocraft_dir=str(audiocraft),
        profile_name="micro-unconditional",
        workspace_dir=workspace,
        musicgen_python=str(python),
        dry_run=True,
        init_from_checkpoint=source.user_input,
        overrides=[] if codec_override is None else [f"{prefix}compression_model_checkpoint={codec_override}"],
    )
    if codec_override == "/custom/codec.th":
        with pytest.raises(ValueError, match="codec mismatch"):
            plan_musicgen_training(**kwargs)
    else:
        plan = plan_musicgen_training(**kwargs)
        assert f"continue_from={checkpoint}" in plan.command
        assert plan.init_from_checkpoint["origin"] == "bundle"
