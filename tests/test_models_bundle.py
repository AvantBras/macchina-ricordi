"""The portable model bundle: write/read/verify, discovery, references, zoo, export planning, CLI routing.

These are real-I/O tests: a bundle is built with ``bundle.write_bundle`` (numpy + safetensors, no torch) and read
back from disk. The export *planner* is read-only and needs no torch; running the helper (torch) is not exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import macchina_ricordi.models.export as export
from macchina_ricordi.models.bundle import (
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
    BundleError,
    is_bundle,
    read_bundle,
    write_bundle,
)
from macchina_ricordi.models.record import ModelNotFound, resolve_checkpoint_path, resolve_model
from macchina_ricordi.models.zoo import build_zoo

MODEL_CONFIG = {
    "lm_model": "transformer_lm",
    "transformer_lm": {"dim": 192, "num_heads": 3, "num_layers": 10, "hidden_scale": 4, "n_q": 4, "card": 2048},
    "dtype": "float32",
    "codebooks_pattern": {"modeling": "delay", "delay": {"delays": [0, 1, 2, 3]}},
    "attribute_dropout": {},
    "classifier_free_guidance": {"training_dropout": 0.0, "inference_coef": 1.0},
    "fuser": {"sum": [], "cross": [], "prepend": [], "input_interpolate": []},
    "conditioners": {},
}
CODEC = {"name": "facebook/encodec_32khz"}
AUDIO = {"sample_rate": 32000, "channels": 1, "frame_rate": 50.0}


def make_bundle(
    root: Path, name: str = "drummer", *, segment_duration: float = 10.0, weights_state: str = "model"
) -> Path:
    out = root / "models" / name
    write_bundle(
        out,
        {"w": np.zeros((3, 3), dtype="float32"), "emb": np.ones((8,), dtype="float32")},
        model_config=MODEL_CONFIG,
        codec=CODEC,
        audio=AUDIO,
        segment_duration=segment_duration,
        name=name,
        provenance={"sig": "abcd1234", "epoch": 9, "recipe": "reshape-nano", "dataset": ".ws"},
        weights_state=weights_state,
    )
    return out


def _xp(repo: Path, sig: str = "abcd1234") -> Path:
    folder = repo / ".dora" / "xps" / sig
    (folder / ".hydra").mkdir(parents=True)
    (folder / ".hydra" / "config.yaml").write_text(
        "datasource:\n  train: /data/.ws_disco/egs/disco/train\n", encoding="utf-8"
    )
    (folder / "history.json").write_text(json.dumps([{"valid": {"ce": 5.0}}]), encoding="utf-8")
    (folder / "checkpoint_1.th").write_bytes(b"x")
    (folder / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"effective_label": "disco", "recipe": "reshape-nano"}), encoding="utf-8"
    )
    return folder


# ── core read/verify ──


def test_write_and_read_bundle_roundtrip(tmp_path: Path) -> None:
    out = make_bundle(tmp_path)
    assert is_bundle(out)
    bundle = read_bundle(out, verify=True)
    assert bundle.name == "drummer"
    assert bundle.config["schema"] == "macchina_ricordi.model_bundle.v1"
    assert bundle.segment_duration == 10.0
    assert bundle.codec == "facebook/encodec_32khz"
    assert bundle.selection == "model"
    assert bundle.weights_file == WEIGHTS_FILENAME
    assert (out / MANIFEST_FILENAME).is_file() and (out / WEIGHTS_FILENAME).is_file()


def test_read_bundle_detects_a_bad_hash(tmp_path: Path) -> None:
    out = make_bundle(tmp_path)
    data = bytearray((out / WEIGHTS_FILENAME).read_bytes())
    data[-1] ^= 0xFF  # corrupt one data byte at the tail; the safetensors header stays valid
    (out / WEIGHTS_FILENAME).write_bytes(bytes(data))
    with pytest.raises(BundleError, match="checksum|sha256"):
        read_bundle(out, verify=True)


def test_is_bundle_false_without_a_manifest(tmp_path: Path) -> None:
    plain = tmp_path / "models" / "notabundle"
    plain.mkdir(parents=True)
    (plain / WEIGHTS_FILENAME).write_bytes(b"x")
    assert not is_bundle(plain)


# ── records and references ──


def test_bundle_resolves_to_a_synthetic_model(tmp_path: Path) -> None:
    make_bundle(tmp_path)
    ref = resolve_model("drummer", repo_root=tmp_path)
    assert ref.model is not None and ref.model.kind == "bundle"
    model = ref.model
    assert model.sig.startswith("bundle-") and len(model.sig) == len("bundle-") + 12
    assert model.name == "drummer"
    assert model.segment_duration == 10.0
    assert model.shape["dim"] == 192
    assert model.dataset == ".ws"
    assert ref.checkpoint == model.folder / WEIGHTS_FILENAME
    assert ref.epoch == 9
    assert model.selection == "model"


def test_bundle_resolves_by_path_and_refuses_a_selector(tmp_path: Path) -> None:
    out = make_bundle(tmp_path)
    ref = resolve_model(str(out), repo_root=tmp_path)
    assert ref.model.kind == "bundle"
    with pytest.raises(ModelNotFound, match="selected snapshot"):
        resolve_model("drummer@best", repo_root=tmp_path)


def test_bundle_id_matching_does_not_collide_with_xp_sigs(tmp_path: Path) -> None:
    make_bundle(tmp_path)
    with pytest.raises(ModelNotFound):
        resolve_model("abcd1234", repo_root=tmp_path)  # a bundle id is never an 8-hex XP name


def test_resolve_checkpoint_path_returns_the_bundle_directory(tmp_path: Path) -> None:
    make_bundle(tmp_path)
    # a named reference (the loading handle is the whole bundle directory, not the safetensors inside it):
    assert resolve_checkpoint_path("drummer", repo_root=tmp_path) == str(tmp_path / "models" / "drummer")
    # a direct existing bundle directory also bypasses the CLI resolve to its folder:
    assert resolve_checkpoint_path(str(tmp_path / "models" / "drummer"), repo_root=tmp_path) == str(
        tmp_path / "models" / "drummer"
    )


def test_the_zoo_lists_bundles_alongside_xps(tmp_path: Path) -> None:
    _xp(tmp_path)
    make_bundle(tmp_path)
    zoo = build_zoo(tmp_path)
    by_kind = {m.sig: m.kind for m in zoo.models}
    assert by_kind["abcd1234"] == "xp"
    assert any(kind == "bundle" for kind in by_kind.values())
    bundle_model = next(m for m in zoo.models if m.kind == "bundle")
    assert bundle_model.name == "drummer"  # no edge, no Dora provenance required


# ── export planning (no torch) ──


def _plan_bundle(repo: Path, monkeypatch: pytest.MonkeyPatch, reference: str = "disco@best"):
    (repo / ".venv-musicgen" / "bin").mkdir(parents=True)
    (repo / ".venv-musicgen" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "export_musicgen_bundle.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        export,
        "load_musicgen_project_config",
        lambda: SimpleNamespace(repo_root=repo, musicgen_python=repo / ".venv-musicgen" / "bin" / "python"),
    )
    monkeypatch.setattr("macchina_ricordi.models.record.get_repo_root", lambda explicit_root=None: repo)
    return export.plan_bundle_export(reference, out_dir="models", weights_state="best", repo_root=repo)


def test_plan_bundle_export_is_read_only_and_names_the_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _xp(tmp_path)
    plan = _plan_bundle(tmp_path, monkeypatch)
    assert plan.weights_state == "best"
    assert plan.bundle_dir.endswith("models/disco")  # default name is the model's display name
    assert plan.manifest_path.endswith("config.json")
    assert plan.weights_path.endswith("model.safetensors")
    assert plan.provenance["sig"] == "abcd1234" and plan.provenance["epoch"] == 1
    assert plan.helper_command[:4] == (
        str(tmp_path / ".venv-musicgen" / "bin" / "python"),
        "-m",
        "scripts.export_musicgen_bundle",
        str(tmp_path / ".dora" / "xps" / "abcd1234" / "checkpoint_1.th"),
    )
    assert plan.missing == ()  # both the runtime python and the helper script exist here
    assert not (tmp_path / "models").exists()  # the planner never writes
    json.dumps(plan.to_dict())


def test_plan_bundle_export_refuses_a_bundle_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = make_bundle(tmp_path)
    with pytest.raises(ModelNotFound, match="nothing to lift"):
        _plan_bundle(tmp_path, monkeypatch, reference=str(out))


def test_box_export_carries_the_weights_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _xp(tmp_path)
    (tmp_path / ".venv-musicgen" / "bin").mkdir(parents=True)
    (tmp_path / ".venv-musicgen" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        export,
        "load_musicgen_project_config",
        lambda: SimpleNamespace(repo_root=tmp_path, musicgen_python=tmp_path / ".venv-musicgen" / "bin" / "python"),
    )
    monkeypatch.setattr("macchina_ricordi.models.record.get_repo_root", lambda explicit_root=None: tmp_path)
    plan = export.plan_box_export("disco@best", weights_state="ema", repo_root=tmp_path)
    assert "--weights-state" in plan.export_command
    assert "ema" in plan.export_command


def test_export_parser_routes_to_bundle(tmp_path: Path) -> None:
    from macchina_ricordi.cli import build_parser

    parse = build_parser().parse_args
    args = parse(["model", "export", "midwife", "--bundle", "--weights-state", "best", "--force"])
    assert args.bundle is True and args.box is False
    assert args.weights_state == "best" and args.force is True and args.out_dir is None
    args2 = parse(["model", "export", "midwife"])
    assert args2.bundle is False and args2.weights_state == "model"
