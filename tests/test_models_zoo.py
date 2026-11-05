from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.models.catalog import build_catalog
from macchina_ricordi.models.lineage import build_lineage_graph_for_sig
from macchina_ricordi.models.provenance import resolve_checkpoint, resolve_init
from macchina_ricordi.models.record import load_model
from macchina_ricordi.models.zoo import build_zoo


def _xp(repo: Path, sig: str, *, init: dict | None = None, label: str | None = None) -> Path:
    folder = repo / ".dora" / "xps" / sig
    folder.mkdir(parents=True)
    (folder / "checkpoint.th").write_bytes(b"x")
    metadata: dict = {"xp_sig": sig, "effective_label": label or f"{sig}_run", "command": ["dora", "run"]}
    if init is not None:
        metadata["init_from_checkpoint"] = init
    (folder / "macchina_ricordi_train_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


def _merge(a: Path, b: Path, output: Path, *, created_at: str = "2026-06-01T00:00:00Z", **extra) -> dict:
    return {
        "method": "slerp",
        "alpha": 0.5,
        "created_at": created_at,
        "sources": {"a": {"path": str(a)}, "b": {"path": str(b)}},
        "output": {"path": str(output), "sha256": "feed"},
        **extra,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    alpha = _xp(tmp_path, "aaaa1111")
    beta = _xp(tmp_path, "bbbb2222")
    gamma = _xp(tmp_path, "cccc3333")
    _xp(tmp_path, "dddd4444", init={"resolved_local_path": str(alpha / "checkpoint.th")})
    merged = tmp_path / "artifacts" / "checkpoints" / "merged" / "blend" / "checkpoint.th"
    merged.parent.mkdir(parents=True)
    merged.write_bytes(b"m")
    merge = _merge(alpha / "checkpoint.th", beta / "checkpoint.th", merged)
    # a perturbation of the merge output inherits the merge record; the perturbation is the direct parent
    perturb = {
        "sigma": 0.01,
        "layer_scope": "all",
        "seed": 3,
        "created_at": "2026-07-01T00:00:00Z",
        "sources": {"source": {"path": str(merged)}},
        "source_resume_metadata": {"merged_from": merge},
        "output": {"path": str(tmp_path / "artifacts" / "checkpoints" / "perturbed" / "p" / "checkpoint.th")},
    }
    _xp(
        tmp_path,
        "eeee5555",
        init={
            "resolved_local_path": perturb["output"]["path"],
            "source_resume_metadata": {"merged_from": merge, "perturbed_from": perturb},
        },
    )
    # warm-started from the merge file, recorded before resume metadata existed: only the sidecar on disk says
    (merged.parent / "merged_from.json").write_text(json.dumps(merge), encoding="utf-8")
    _xp(tmp_path, "ffff6666", init={"resolved_local_path": str(merged)})
    _xp(tmp_path, "abab7777", init={"resolved_local_path": str(gamma / "checkpoint_4.th")})

    session = tmp_path / "artifacts" / "sessions" / "2026-10-06" / "duet_20261006T090656Z"
    session.mkdir(parents=True)
    (session / "session.json").write_text(
        json.dumps(
            {
                "id": "duet_20261006T090656Z",
                "kind": "duet",
                "created_at": "2026-10-06T09:06:56Z",
                "models": [str(alpha), str(beta)],
                "take_count": 2,
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "research" / "training"
    ledger.mkdir(parents=True)
    (ledger / "experiment_ledger.toml").write_text(
        'schema = "macchina_ricordi.experiment_ledger.v1"\n\n[[experiments]]\nid = "blend"\ntitle = "Blend"\n'
        'status = "running"\nxp_sigs = ["eeee5555", "aaaa1111"]\n',
        encoding="utf-8",
    )
    return tmp_path


def test_record_and_lineage_agree_when_a_perturbation_inherits_a_merge(repo: Path) -> None:
    model = load_model(repo / ".dora" / "xps" / "eeee5555")
    assert (model.parents, model.origin) == (("aaaa1111", "bbbb2222"), "perturb")
    graph = build_lineage_graph_for_sig("eeee5555", dora_dir=repo / ".dora", repo_root=repo)
    warm = next(edge for edge in graph.edges if edge.from_node_id == "xp:eeee5555")
    assert warm.to_node_id.startswith("perturb:")
    below = next(edge for edge in graph.edges if edge.from_node_id == warm.to_node_id)
    assert below.kind == "perturb_source" and below.to_node_id.startswith("merge:")
    assert {"xp:aaaa1111", "xp:bbbb2222"} <= {node.id for node in graph.nodes}


def test_a_sidecar_next_to_the_checkpoint_names_the_parents(repo: Path) -> None:
    model = load_model(repo / ".dora" / "xps" / "ffff6666")
    assert (model.parents, model.origin) == (("aaaa1111", "bbbb2222"), "merge")
    assert model.init is not None and model.init.sidecar is not None
    graph = build_lineage_graph_for_sig("ffff6666", dora_dir=repo / ".dora", repo_root=repo)
    assert any(node.kind == "merge" for node in graph.nodes)


def test_init_provenance_names_parents_and_operation(repo: Path) -> None:
    metadata = json.loads((repo / ".dora" / "xps" / "eeee5555" / "macchina_ricordi_train_metadata.json").read_text())
    init = resolve_init(metadata)
    assert (init.parent_sigs(), init.operation) == (("aaaa1111", "bbbb2222"), "perturb")
    assert resolve_init({}) is None
    assert resolve_checkpoint("facebook/musicgen-small").kind == "unknown"


def test_a_sidecar_that_names_its_own_output_does_not_loop(tmp_path: Path) -> None:
    folder = tmp_path / "loop"
    folder.mkdir()
    (folder / "merged_from.json").write_text(
        json.dumps(_merge(folder / "a.th", folder / "b.th", folder / "b.th")), encoding="utf-8"
    )
    origin = resolve_checkpoint(str(folder / "b.th"))
    assert origin.kind == "merge"
    a, b = (source.origin for source in origin.sources)
    # the sidecar describes b.th only: a.th, its neighbour, is not explained by it
    assert (a.kind, a.cut) == ("unknown", None)
    assert (b.kind, b.cut) == ("merge", "cycle")
    assert origin.parent_sigs() == ()


def test_derived_checkpoints_are_listed_once_with_their_parents(repo: Path) -> None:
    zoo = build_zoo(repo)
    [blend] = zoo.derived
    assert blend.kind == "merge" and blend.checkpoint.name == "checkpoint.th"
    assert blend.parent_sigs == ("aaaa1111", "bbbb2222")
    assert [role for role, _ in blend.sources] == ["a", "b"]
    assert zoo.warm_started_from(blend.checkpoint) == ("ffff6666",)
    catalog = build_catalog(repo, include_lineage=False)
    assert [(m.method, m.source_a_sig) for m in catalog.merges] == [("slerp", "aaaa1111")]


def test_zoo_family_sessions_and_experiments(repo: Path) -> None:
    zoo = build_zoo(repo)
    assert zoo.parents("dddd4444") == ("aaaa1111",)
    assert zoo.children("aaaa1111") == ("dddd4444", "eeee5555", "ffff6666")
    assert zoo.children("cccc3333") == ("abab7777",)
    assert set(zoo.descendants("aaaa1111")) == {"dddd4444", "eeee5555", "ffff6666"}
    assert zoo.ancestors("eeee5555") == ("aaaa1111", "bbbb2222")
    assert [s.kind for s in zoo.sessions_for("bbbb2222")] == ["duet"]
    assert [e.experiment_id for e in zoo.experiments_for("aaaa1111")] == ["blend"]
    family = zoo.family("aaaa1111")
    assert family["sessions"][0]["kind"] == "duet" and family["experiments"] == ["blend"]
    merged = repo / "artifacts" / "checkpoints" / "merged" / "blend" / "checkpoint.th"
    assert zoo.warm_started_from(merged) == ("ffff6666",)
    assert build_zoo(repo, include_sessions=False).sessions == ()


def test_catalog_renders_the_zoo(repo: Path) -> None:
    catalog = build_catalog(repo, include_lineage=False)
    models = {model.sig: model for model in catalog.models}
    assert models["eeee5555"].parents == ("aaaa1111", "bbbb2222")
    assert models["eeee5555"].perturbed_from is not None
    assert models["eeee5555"].merged_from is not None
    assert [e.experiment_id for e in catalog.experiments] == ["blend"]


def test_the_zoo_also_lists_a_portable_bundle(repo: Path) -> None:
    import numpy as np

    from macchina_ricordi.models.bundle import write_bundle

    bundle = repo / "models" / "drummer"
    write_bundle(
        bundle,
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
        segment_duration=10.0,
        name="drummer",
        provenance={"sig": "aaaa1111", "epoch": 3},
    )
    zoo = build_zoo(repo)
    bundle_model = next(m for m in zoo.models if m.kind == "bundle")
    assert bundle_model.sig.startswith("bundle-") and bundle_model.name == "drummer"
    assert bundle_model.recipe is None and bundle_model.dataset is None  # no Dora registry required
    assert zoo.parents(bundle_model.sig) == ()  # a bundle has no Dora lineage edge
    assert "drummer" in {m.name for m in zoo.models}
