from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.models.lineage import (
    build_lineage_graph_for_sig,
    format_lineage_text,
    lineage_graph_to_dict,
    lineage_graph_to_json,
)


def _write_xp(
    dora: Path,
    sig: str,
    *,
    label: str | None = None,
    command: list[str] | None = None,
    init_from_checkpoint: dict | None = None,
) -> Path:
    xp = dora / "xps" / sig
    xp.mkdir(parents=True)
    (xp / "checkpoint.th").write_bytes(b"checkpoint")
    metadata = {
        "xp_sig": sig,
        "xp_folder": str(xp),
        "effective_label": label,
        "workspace_dir": str(dora.parent / ".ws"),
        "dataset_name": "macchina_ricordi_micro_uncond",
        "dataset_fingerprint_digest": f"{sig}digest",
        "dataset_fingerprint": {
            "dataset_name": "macchina_ricordi_micro_uncond",
            "train_count": 12,
        },
        "command": command or ["dora", "run", "optim.epochs=5"],
    }
    if init_from_checkpoint is not None:
        metadata["init_from_checkpoint"] = init_from_checkpoint
    (xp / "macchina_ricordi_train_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return xp


def _node(graph, node_id: str):
    return next(node for node in graph.nodes if node.id == node_id)


def test_single_xp_lineage_has_root_only(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    _write_xp(dora, "abc123", label="solo")

    graph = build_lineage_graph_for_sig("abc123", dora_dir=dora, repo_root=tmp_path)

    assert graph.schema_version == 1
    assert graph.root_node_id == "xp:abc123"
    assert len(graph.nodes) == 1
    assert graph.edges == ()
    root = _node(graph, "xp:abc123")
    assert root.kind == "xp"
    assert root.label == "solo"
    assert root.epochs == 5
    assert "solo" in format_lineage_text(graph)


def test_warm_start_chain_and_override_diff(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    parent = _write_xp(
        dora,
        "parent1",
        label="parent",
        command=["dora", "run", "optim.epochs=4", "optim.lr=0.1", "label=old"],
    )
    _write_xp(
        dora,
        "child1",
        label="child",
        command=[
            "dora",
            "run",
            "optim.epochs=8",
            "optim.lr=0.2",
            "label=new",
            f"continue_from={parent / 'checkpoint.th'}",
        ],
        init_from_checkpoint={
            "resolved_local_path": str(parent / "checkpoint.th"),
            "user_input": str(parent / "checkpoint.th"),
            "source_resume_metadata": None,
        },
    )

    graph = build_lineage_graph_for_sig("child1", dora_dir=dora, repo_root=tmp_path)

    assert {node.id for node in graph.nodes} == {"xp:child1", "xp:parent1"}
    assert graph.edges[0].kind == "warm_start"
    assert graph.edges[0].from_node_id == "xp:child1"
    assert graph.edges[0].to_node_id == "xp:parent1"
    child = _node(graph, "xp:child1")
    assert child.override_diff is not None
    assert child.override_diff.changed["optim.lr"] == {"from": "0.1", "to": "0.2"}
    assert "continue_from" not in child.override_diff.added
    assert "continue_from" not in child.override_diff.changed
    text = format_lineage_text(graph)
    assert "warm-start" in text
    assert "diff:" in text


def test_merge_sidecar_lineage_resolves_two_source_checkpoints(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    source_a = _write_xp(dora, "sourcea", label="a")
    source_b = _write_xp(dora, "sourceb", label="b")
    merged = tmp_path / "merged_checkpoints" / "blend"
    merged.mkdir(parents=True)
    (merged / "checkpoint.th").write_bytes(b"merged")
    sidecar = {
        "method": "slerp",
        "alpha": 0.75,
        "sources": {
            "a": {"path": str(source_a / "checkpoint_150.th")},
            "b": {"path": str(source_b / "checkpoint_50.th")},
        },
        "output": {"path": str(merged / "checkpoint.th")},
    }
    (merged / "merged_from.json").write_text(json.dumps(sidecar), encoding="utf-8")
    _write_xp(
        dora,
        "child1",
        init_from_checkpoint={
            "resolved_local_path": str(merged / "checkpoint.th"),
            "source_resume_metadata": {"merged_from": sidecar},
        },
    )

    graph = build_lineage_graph_for_sig("child1", dora_dir=dora, repo_root=tmp_path)

    merge_nodes = [node for node in graph.nodes if node.kind == "merge"]
    assert len(merge_nodes) == 1
    assert merge_nodes[0].operation == {"method": "slerp", "alpha": 0.75, "created_at": None}
    source_edges = [edge for edge in graph.edges if edge.kind == "merge_source"]
    assert {edge.role for edge in source_edges} == {"a", "b"}
    assert {edge.summary["checkpoint_epoch"] for edge in source_edges} == {"150", "50"}
    assert {"xp:sourcea", "xp:sourceb"}.issubset({node.id for node in graph.nodes})
    assert "merge slerp" in format_lineage_text(graph)


def test_perturb_sidecar_lineage_resolves_source_checkpoint(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    source = _write_xp(dora, "source1", label="source")
    perturbed = tmp_path / "perturbed_checkpoints" / "damaged"
    perturbed.mkdir(parents=True)
    (perturbed / "checkpoint.th").write_bytes(b"perturbed")
    sidecar = {
        "sigma": 0.02,
        "layer_scope": "lm-head",
        "seed": 7,
        "sources": {"source": {"path": str(source / "checkpoint_20.th")}},
        "output": {"path": str(perturbed / "checkpoint.th")},
    }
    _write_xp(
        dora,
        "child1",
        init_from_checkpoint={
            "resolved_local_path": str(perturbed / "checkpoint.th"),
            "source_resume_metadata": {"perturbed_from": sidecar},
        },
    )

    graph = build_lineage_graph_for_sig("child1", dora_dir=dora, repo_root=tmp_path)

    perturb = next(node for node in graph.nodes if node.kind == "perturb")
    assert perturb.operation == {"sigma": 0.02, "layer_scope": "lm-head", "seed": 7, "created_at": None}
    assert any(edge.kind == "perturb_source" and edge.to_node_id == "xp:source1" for edge in graph.edges)
    assert "perturb sigma=0.02 scope=lm-head seed=7" in format_lineage_text(graph)


def test_unknown_parent_path_warns_without_crashing(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    _write_xp(
        dora,
        "child1",
        init_from_checkpoint={
            "resolved_local_path": str(tmp_path / "external" / "checkpoint.th"),
            "source_resume_metadata": None,
        },
    )

    graph = build_lineage_graph_for_sig("child1", dora_dir=dora, repo_root=tmp_path)

    assert any(node.kind == "unknown" for node in graph.nodes)
    assert graph.warnings
    assert "Could not resolve checkpoint ancestor" in graph.warnings[0]


def test_cycle_detection_records_warning(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    xp_a = dora / "xps" / "aaaaaaa"
    xp_b = dora / "xps" / "bbbbbbb"
    _write_xp(
        dora,
        "aaaaaaa",
        init_from_checkpoint={
            "resolved_local_path": str(xp_b / "checkpoint.th"),
            "source_resume_metadata": None,
        },
    )
    _write_xp(
        dora,
        "bbbbbbb",
        init_from_checkpoint={
            "resolved_local_path": str(xp_a / "checkpoint.th"),
            "source_resume_metadata": None,
        },
    )

    graph = build_lineage_graph_for_sig("aaaaaaa", dora_dir=dora, repo_root=tmp_path)

    assert any("Cycle detected" in warning for warning in graph.warnings)
    assert any(edge.from_node_id == "xp:bbbbbbb" and edge.to_node_id == "xp:aaaaaaa" for edge in graph.edges)


def test_depth_limit_records_warning_and_keeps_node(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    parent = _write_xp(dora, "parent1")
    _write_xp(
        dora,
        "child1",
        init_from_checkpoint={
            "resolved_local_path": str(parent / "checkpoint.th"),
            "source_resume_metadata": None,
        },
    )

    graph = build_lineage_graph_for_sig("child1", dora_dir=dora, repo_root=tmp_path, max_depth=0)

    assert "xp:parent1" in {node.id for node in graph.nodes}
    assert any("Depth limit reached" in warning for warning in graph.warnings)


def test_json_serialization_contains_schema_and_operations(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    _write_xp(dora, "abc123")

    graph = build_lineage_graph_for_sig("abc123", dora_dir=dora, repo_root=tmp_path)

    payload = json.loads(lineage_graph_to_json(graph))
    assert payload["schema_version"] == 1
    assert payload["root_node_id"] == "xp:abc123"
    assert lineage_graph_to_dict(graph)["schema_version"] == 1


def test_missing_root_metadata_suggests_backfill(tmp_path: Path) -> None:
    dora = tmp_path / ".dora"
    (dora / "xps" / "abc123").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="tidy backfill"):
        build_lineage_graph_for_sig("abc123", dora_dir=dora, repo_root=tmp_path)
