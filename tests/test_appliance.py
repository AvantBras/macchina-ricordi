"""The player has to stay honest without a model in the repo, or torch installed.

A fake backend returns random logits but is strict about what the loop feeds it —
offsets, batch, horizon — so the properties under test are the ones the player is
responsible for: sequence layout, emission accounting, seams, the watchdog, and
the box format.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from appliance.box import BOX_FORMAT, read_box, read_spec, write_box
from appliance.spec import ModelSpec


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        dim=16,
        num_heads=2,
        num_layers=2,
        n_q=4,
        card=32,
        hidden_scale=2.0,
        layer_scale=1e-4,
        delays=[0, 1, 2, 3],
        frame_rate=10.0,
        sample_rate=80,
        channels=1,
        segment_duration=4.0,
        name="stub",
    )


# ---- spec and box --------------------------------------------------------


def test_box_round_trip(tmp_path, spec):
    path = write_box(tmp_path / "t.nrbox", spec=spec, lm_onnx=b"lm-graph", decoder_onnx=b"decoder-graph")
    box = read_box(path)
    assert box.spec == spec and box.lm == b"lm-graph" and box.decoder == b"decoder-graph"
    assert read_spec(path) == spec
    with zipfile.ZipFile(path) as z:
        assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_STORED}
    assert not list(tmp_path.glob("*.partial"))


def test_box_rejects_foreign_files(tmp_path):
    not_zip = tmp_path / "a.nrbox"
    not_zip.write_bytes(b"hello")
    no_spec = tmp_path / "b.nrbox"
    with zipfile.ZipFile(no_spec, "w") as z:
        z.writestr("lm.onnx", b"")
    wrong = tmp_path / "c.nrbox"
    with zipfile.ZipFile(wrong, "w") as z:
        z.writestr("spec.json", json.dumps({"format": "something/else", "spec": {}}))
    for path in (not_zip, no_spec, wrong):
        with pytest.raises(ValueError, match=BOX_FORMAT.split("/")[0]):
            read_box(path)
