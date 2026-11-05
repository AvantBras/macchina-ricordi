from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from macchina_ricordi.gui.data import GuiIndex
from macchina_ricordi.gui.derive import DeriveManager, DeriveRequest, DeriveRequestError, read_model_shape
from macchina_ricordi.gui.server import serve

# Stands in for `python -m macchina_ricordi.cli`: writes the output checkpoint (argv[5] after
# `model merge`, for a merge; argv[4] after `model perturb`) and exits 0.
FAKE_CLI = (
    sys.executable,
    "-c",
    "import pathlib, sys; out = pathlib.Path(sys.argv[5] if sys.argv[2] == 'merge' else sys.argv[4]); "
    "out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(b'w'); print('checkpoint:', out)",
)

CONFIG = """solver: musicgen
conditioners: {conditioners}
transformer_lm:
  dim: {dim}
  num_heads: {heads}
  num_layers: 10
  hidden_scale: 4
  n_q: 4
  card: 2048
  weight_init:
    dim: 9999
  bias_ff: false
  bias_attn: false
sample_rate: 32000
"""


def _xp(repo: Path, sig: str, *, dim: int = 192, heads: int = 3, conditioners: str = "null") -> Path:
    folder = repo / ".dora" / "xps" / sig
    (folder / ".hydra").mkdir(parents=True)
    (folder / ".hydra" / "config.yaml").write_text(CONFIG.format(dim=dim, heads=heads, conditioners=conditioners))
    checkpoint = folder / "checkpoint_10.th"
    checkpoint.write_bytes(b"x" * 2048)
    return checkpoint


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    a = _xp(tmp_path, "aaaa1111")
    b = _xp(tmp_path, "bbbb2222")
    _xp(tmp_path, "cccc3333", dim=384)
    _xp(tmp_path, "dddd4444", heads=6)
    merged = tmp_path / "artifacts" / "checkpoints" / "merged" / "ab_soup"
    merged.mkdir(parents=True)
    (merged / "checkpoint.th").write_bytes(b"m" * 1024)
    sidecar = {
        "created_at": "2026-09-01T10:00:00+00:00",
        "method": "slerp",
        "alpha": 0.3,
        "sources": {"a": {"path": str(a)}, "b": {"path": str(b)}},
        # written elsewhere, then moved here: children are found by the hash
        "output": {"path": "/elsewhere/ab_soup/checkpoint.th", "sha256": "f00d"},
        "warnings": [],
    }
    (merged / "merged_from.json").write_text(json.dumps(sidecar))
    child = tmp_path / ".dora" / "xps" / "eeee5555"
    child.mkdir()
    init = {
        "resolved_local_path": "/elsewhere/ab_soup/checkpoint.th",
        "source_resume_metadata": {"merged_from": sidecar},
    }
    (child / "macchina_ricordi_train_metadata.json").write_text(json.dumps({"init_from_checkpoint": init}))
    return tmp_path


def _manager(repo: Path, **kwargs) -> DeriveManager:
    kwargs.setdefault("cli_prefix", FAKE_CLI)
    kwargs.setdefault("mem_available", lambda: 2**40)
    return DeriveManager(repo, **kwargs)


def _merge(**payload) -> DeriveRequest:
    base = {
        "kind": "merge",
        "sources": [".dora/xps/aaaa1111/checkpoint_10.th", ".dora/xps/bbbb2222/checkpoint_10.th"],
        "name": "fresh_soup",
        "alpha": 0.25,
        "method": "slerp",
    }
    return DeriveRequest.from_payload({**base, **payload})


def _wait(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached in time")


def test_model_shape_reads_the_first_level_of_transformer_lm(repo: Path) -> None:
    shape = read_model_shape(repo / ".dora" / "xps" / "aaaa1111" / ".hydra" / "config.yaml")
    assert shape == {
        "dim": "192",  # not weight_init.dim
        "num_layers": "10",
        "num_heads": "3",
        "hidden_scale": "4",
        "n_q": "4",
        "card": "2048",
        "bias_ff": "false",
        "bias_attn": "false",
        "conditioned": False,
    }


def test_derived_checkpoints_are_listed_with_their_sources_and_children(repo: Path) -> None:
    [row] = _manager(repo).checkpoints()
    assert row["kind"] == "merge" and row["name"] == "ab_soup" and row["exists"]
    assert row["path"] == "artifacts/checkpoints/merged/ab_soup/checkpoint.th"
    assert [s["sig"] for s in row["sources"]] == ["aaaa1111", "bbbb2222"]
    assert row["shape"] == "dim 192 · 10 layers · 3 heads"  # from its first source's XP
    assert row["children"] == ["eeee5555"]
    assert (row["method"], row["alpha"]) == ("slerp", 0.3)


def test_merge_preview_checks_shapes_and_names(repo: Path) -> None:
    manager = _manager(repo)
    preview = manager.preview(_merge())
    assert preview["blockers"] == [] and preview["warnings"] == [] and preview["key"]
    assert preview["output"] == "artifacts/checkpoints/merged/fresh_soup/checkpoint.th"
    assert "model merge .dora/xps/aaaa1111/checkpoint_10.th .dora/xps/bbbb2222/checkpoint_10.th" in (preview["command"])
    assert preview["command"].endswith("--alpha 0.25 --method slerp")

    wider = manager.preview(
        _merge(sources=[".dora/xps/aaaa1111/checkpoint_10.th", ".dora/xps/cccc3333/checkpoint_10.th"])
    )
    assert wider["key"] is None and "not the same shape (dim)" in wider["blockers"][0]
    heads = manager.preview(
        _merge(sources=[".dora/xps/aaaa1111/checkpoint_10.th", ".dora/xps/dddd4444/checkpoint_10.th"])
    )
    assert heads["key"] and "heads" in heads["warnings"][0]
    # a derived file's shape comes from its sources
    soup = manager.preview(
        _merge(sources=["artifacts/checkpoints/merged/ab_soup/checkpoint.th", ".dora/xps/aaaa1111/checkpoint_10.th"])
    )
    assert soup["key"] and soup["warnings"] == []
    taken = manager.preview(_merge(name="ab_soup"))
    assert taken["key"] is None and "exists" in taken["blockers"][0]
    same = manager.preview(_merge(sources=[".dora/xps/aaaa1111/checkpoint_10.th"] * 2))
    assert "two different" in same["blockers"][0]


def test_preview_warns_about_memory_and_blocks_without_the_runtime(repo: Path) -> None:
    tight = _manager(repo, mem_available=lambda: 1000).preview(_merge())
    assert tight["key"] and "into memory" in tight["warnings"][0]
    missing = _manager(repo, musicgen_check=lambda: "no torch here").preview(_merge())
    assert missing["key"] is None and missing["blockers"] == ["no torch here"]


def test_requests_are_validated() -> None:
    for payload, message in (
        ({"kind": "melt"}, "merge or perturb"),
        ({"kind": "merge", "sources": ["a.th"]}, "two checkpoints"),
        ({"kind": "merge", "sources": ["a.th", "b.th"], "alpha": 1.5, "name": "x"}, "alpha"),
        ({"kind": "perturb", "sources": ["a.th"], "sigma": 0, "name": "x"}, "sigma"),
        ({"kind": "perturb", "sources": ["a.th"], "layer_scope": "contains:", "name": "x"}, "layer scope"),
        ({"kind": "perturb", "sources": ["a.th"], "name": "../up"}, "name"),
    ):
        with pytest.raises(DeriveRequestError, match=message):
            DeriveRequest.from_payload(payload)
    perturb = DeriveRequest.from_payload(
        {"kind": "perturb", "sources": ["a.th"], "name": "x", "sigma": 0.01, "layer_scope": "contains:layers.0."}
    )
    assert (perturb.layer_scope, perturb.alpha, perturb.method) == ("contains:layers.0.", 0.5, "linear")


def test_launch_runs_detached_and_reports_the_output(repo: Path) -> None:
    manager = _manager(repo)
    request = DeriveRequest.from_payload(
        {"kind": "perturb", "sources": [".dora/xps/aaaa1111/checkpoint_10.th"], "name": "shaken", "sigma": 0.02}
    )
    with pytest.raises(DeriveRequestError, match="preview again"):
        manager.launch(request, "nope")
    preview = manager.preview(request)
    assert preview["command"].endswith("--sigma 0.02 --layer-scope all --seed 0")
    job = manager.launch(request, preview["key"])
    assert job["title"] == "perturb shaken (sigma 0.02, all)"
    _wait(lambda: manager.job(job["id"])["state"] == "completed")
    done = manager.job(job["id"])
    assert done["ready"] and done["output"] == "artifacts/checkpoints/perturbed/shaken/checkpoint.th"
    assert manager.jobs()[0]["id"] == job["id"]
    # the output folder now exists, so the same name is refused
    assert manager.preview(request)["key"] is None


def test_http_routes(repo: Path) -> None:
    server = serve(GuiIndex(repo), port=0, derive=_manager(repo))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/derive") as response:
            overview = json.load(response)
        assert [c["name"] for c in overview["checkpoints"]] == ["ab_soup"] and overview["jobs"] == []
        with urllib.request.urlopen(
            f"{base}/api/derive/checkpoint?path=.dora/xps/cccc3333/checkpoint_10.th"
        ) as response:
            assert json.load(response)["shape_label"] == "dim 384 · 10 layers · 3 heads"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base}/api/derive/checkpoint?path=../outside.th")
        assert excinfo.value.code == 404
        bad = urllib.request.Request(
            f"{base}/api/derive/preview",
            data=json.dumps({"request": {"kind": "merge", "sources": ["x.th"]}}).encode(),
            headers={"Content-Type": "application/json", "X-Macchina-Ricordi": "1"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(bad)
        assert excinfo.value.code == 422
    finally:
        server.shutdown()
        server.server_close()
