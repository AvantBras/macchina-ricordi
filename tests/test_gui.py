from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from macchina_ricordi.cli import build_parser
from macchina_ricordi.gui.data import GuiIndex
from macchina_ricordi.gui.server import serve


def _write_xp(
    root: Path,
    sig: str,
    *,
    label: str,
    valid_ce: list[float],
    init: dict | None = None,
    epochs: int = 10,
) -> Path:
    folder = root / ".dora" / "xps" / sig
    folder.mkdir(parents=True)
    metadata = {
        "xp_sig": sig,
        "profile": "reshape-nano",
        "effective_label": label,
        "workspace_dir": str(root / ".workspace_test"),
        "dataset_fingerprint": {"dataset_name": "ds", "train_count": 12},
        "command": ["dora", "run", f"optim.epochs={epochs}", "optim.lr=1e-4", f"label={label}"],
    }
    if init is not None:
        metadata["init_from_checkpoint"] = init
    (folder / "macchina_ricordi_train_metadata.json").write_text(json.dumps(metadata))
    history = [{"train": {"ce": ce + 0.1}, "valid": {"ce": ce}} for ce in valid_ce]
    (folder / "history.json").write_text(json.dumps(history))
    samples = folder / "samples" / str(len(valid_ce))
    samples.mkdir(parents=True)
    (samples / "a.wav").write_bytes(b"RIFF" + bytes(96))
    (folder / f"checkpoint_{len(valid_ce)}.th").write_bytes(b"x")
    return folder


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _write_xp(tmp_path, "aaaa1111", label="root", valid_ce=[6.0, 5.5, 5.7])
    _write_xp(
        tmp_path,
        "bbbb2222",
        label="child",
        valid_ce=[5.4, 5.2],
        init={"resolved_local_path": str(tmp_path / ".dora" / "xps" / "aaaa1111" / "checkpoint_2.th")},
        epochs=20,
    )
    (tmp_path / "secret.th").write_bytes(b"nope")
    return tmp_path


class TestGuiIndex:
    def test_resolve_file_refuses_escapes_and_checkpoints(self, repo: Path, tmp_path_factory) -> None:
        index = GuiIndex(repo)
        assert index.resolve_file(".dora/xps/aaaa1111/samples/3/a.wav") is not None
        assert index.resolve_file("../outside.wav") is None
        assert index.resolve_file(".dora/xps/aaaa1111/checkpoint_3.th") is None
        assert index.resolve_file("secret.th") is None
        outside = tmp_path_factory.mktemp("outside") / "x.wav"
        outside.write_bytes(b"RIFF")
        (repo / "link.wav").symlink_to(outside)
        assert index.resolve_file("link.wav") is None


def test_the_gui_binds_to_this_machine_only() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["gui", "--host", "0.0.0.0"])


def test_the_gui_lists_a_portable_bundle_with_a_kind(tmp_path: Path) -> None:
    import numpy as np

    from macchina_ricordi.models.bundle import write_bundle

    write_bundle(
        tmp_path / "models" / "drummer",
        {"w": np.zeros((2,), dtype="float32")},
        model_config={
            "lm_model": "transformer_lm",
            "transformer_lm": {"dim": 192, "num_heads": 3, "num_layers": 10, "hidden_scale": 4, "n_q": 4, "card": 2048},
            "dtype": "float32",
            "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
            "attribute_dropout": {},
            "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
            "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
            "conditioners": {},
        },
        codec={"name": "facebook/encodec_32khz"},
        audio={"sample_rate": 32000, "channels": 1, "frame_rate": 50.0},
        segment_duration=9.0,
        name="drummer",
        provenance={"epoch": 3},
    )
    summaries = GuiIndex(tmp_path).summaries()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.kind == "bundle" and summary.sig.startswith("bundle-")
    assert summary.label == "drummer" and summary.name == "drummer"


class TestServer:
    @pytest.fixture()
    def base_url(self, repo: Path):
        server = serve(GuiIndex(repo), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()

    def test_a_foreign_host_name_is_refused(self, base_url: str) -> None:
        # DNS rebinding: a page on another domain that resolves to this machine sends its own Host name
        request = urllib.request.Request(f"{base_url}/api/xps", headers={"Host": "evil.example"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 403
