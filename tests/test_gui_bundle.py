"""Portable bundle support in the GUI data layer (summaries + model detail).

The catalogue lists a bundle like any model (``kind='bundle'``), and its model page must show the real
``models/<name>`` folder and its single snapshot instead of a fabricated ``.dora/xps``/merge-perturb path,
because a bundle is not a Dora record and has no ``.th`` checkpoints (see ``models.bundle`` and ``record``).
"""

from __future__ import annotations

from pathlib import Path

from macchina_ricordi.gui.data import GuiIndex
from macchina_ricordi.models.bundle import write_bundle


def _bundle_config() -> dict:
    """The minimal whitelisted builder config v1 accepts: an unconditional transformer_lm on EnCodec 32 kHz."""
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


def _with_bundle(repo: Path) -> Path:
    import numpy as np

    bundle = write_bundle(
        repo / "models" / "mycool",
        {"weight": np.ones((2, 3), dtype=np.float32)},
        model_config=_bundle_config(),
        codec={"name": "facebook/encodec_32khz"},
        audio={"sample_rate": 32000, "channels": 1, "frame_rate": 50.0},
        segment_duration=1.0,
        name="mycool",
    )
    return bundle.path


def test_bundle_appears_in_summaries_with_its_folder(tmp_path: Path) -> None:
    _with_bundle(tmp_path)

    index = GuiIndex(tmp_path)
    bundles = [s for s in index.summaries() if s.kind == "bundle"]
    assert len(bundles) == 1
    (bundle,) = bundles
    assert bundle.sig.startswith("bundle-")
    assert bundle.folder == "models/mycool"  # the real path, not a .dora/xps one
    assert bundle.checkpoint_count == 1
    assert bundle.epochs_done == 0  # a bundle has no history.json, so no trained epochs
    assert bundle.origin == "bundle"


def test_bundle_detail_lists_one_snapshot_and_no_dora_sections(tmp_path: Path) -> None:
    _with_bundle(tmp_path)

    index = GuiIndex(tmp_path)
    sig = index.summaries()[0].sig
    detail = index.xp_detail(sig)
    assert detail is not None
    assert detail["summary"]["kind"] == "bundle"
    assert detail["summary"]["folder"] == "models/mycool"
    # one portable snapshot, no .dora-only sections
    assert [c["name"] for c in detail["checkpoints"]] == ["model.safetensors"]
    assert detail["checkpoints"][0]["modified"] > 0
    assert detail["sample_epochs"] == []
    assert detail["history"] == {}
    assert detail["overrides"] == {}
    assert detail["override_diff"] is None
    assert detail["metadata"] is None


def test_bundle_detail_is_not_served_under_an_xp_sig_in_xp_folder(tmp_path: Path) -> None:
    """xp_folder() stays .dora/xps-only: a bundle sig must resolve through the bundle branch, not a fake folder."""
    _with_bundle(tmp_path)

    index = GuiIndex(tmp_path)
    sig = index.summaries()[0].sig
    assert index.xp_folder(sig) is None
    assert index.samples(sig, "1") is None
    # a genuinely unknown sig gets None from the detail too
    assert index.xp_detail("ffffffff") is None
