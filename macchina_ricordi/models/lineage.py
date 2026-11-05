from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.shared import read_json
from ..training.plan import TRAIN_METADATA_FILENAME
from .provenance import Origin, operation_summary, resolve_init

LINEAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OverrideDiff:
    added: dict[str, str]
    removed: dict[str, str]
    changed: dict[str, dict[str, str]]


@dataclass(frozen=True)
class LineageNode:
    id: str
    kind: str
    label: str | None = None
    path: str | None = None
    sig: str | None = None
    epoch: str | None = None
    checkpoint_path: str | None = None
    workspace: str | None = None
    dataset_name: str | None = None
    train_count: int | None = None
    fingerprint_digest: str | None = None
    epochs: int | None = None
    operation: dict[str, object] | None = None
    override_diff: OverrideDiff | None = None


@dataclass(frozen=True)
class LineageEdge:
    from_node_id: str
    to_node_id: str
    kind: str
    role: str | None = None
    summary: dict[str, object] | None = None


@dataclass(frozen=True)
class LineageGraph:
    schema_version: int
    generated_at: str
    root_node_id: str
    nodes: tuple[LineageNode, ...]
    edges: tuple[LineageEdge, ...]
    warnings: tuple[str, ...] = ()


def lineage_graph_to_dict(graph: LineageGraph) -> dict[str, object]:
    return asdict(graph)


def lineage_graph_to_json(graph: LineageGraph) -> str:
    return json.dumps(lineage_graph_to_dict(graph), indent=2, default=str)


def build_lineage_graph_for_sig(
    sig: str,
    *,
    dora_dir: Path,
    repo_root: Path,
    max_depth: int = 20,
) -> LineageGraph:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    builder = _LineageBuilder(dora_dir=dora_dir, repo_root=repo_root, max_depth=max_depth)
    root_node_id = builder.add_xp(sig, depth=0, stack=())
    return LineageGraph(
        schema_version=LINEAGE_SCHEMA_VERSION,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        root_node_id=root_node_id,
        nodes=tuple(builder.nodes.values()),
        edges=tuple(builder.edges),
        warnings=tuple(builder.warnings),
    )


def format_lineage_text(graph: LineageGraph) -> str:
    nodes = {node.id: node for node in graph.nodes}
    children = _edges_by_from_node(graph.edges)
    lines: list[str] = []

    def walk(node_id: str, prefix: str = "", edge: LineageEdge | None = None, seen: set[str] | None = None) -> None:
        seen = set() if seen is None else seen
        node = nodes.get(node_id)
        if node is None:
            lines.append(f"{prefix}? missing node {node_id}")
            return
        if edge is None:
            lines.append(_format_node_label(node))
        else:
            lines.append(f"{prefix}{_edge_connector(edge)} {_format_edge_label(edge)} -> {_format_node_label(node)}")
        if node.override_diff is not None and _has_diff(node.override_diff):
            lines.append(f"{prefix}  diff: {_format_override_diff(node.override_diff)}")
        if node_id in seen:
            lines.append(f"{prefix}  cycle: already visited {node_id}")
            return
        next_seen = seen | {node_id}
        node_edges = children.get(node_id, ())
        for index, child_edge in enumerate(node_edges):
            is_last = index == len(node_edges) - 1
            next_prefix = prefix + ("   " if edge is not None and is_last else "  ")
            walk(child_edge.to_node_id, prefix=next_prefix, edge=child_edge, seen=next_seen)

    walk(graph.root_node_id)
    if graph.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in graph.warnings)
    return "\n".join(lines)


def _edges_by_from_node(edges: tuple[LineageEdge, ...]) -> dict[str, tuple[LineageEdge, ...]]:
    grouped: dict[str, list[LineageEdge]] = {}
    for edge in edges:
        grouped.setdefault(edge.from_node_id, []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _edge_connector(edge: LineageEdge) -> str:
    if edge.role in {"b", "source_b"}:
        return "+-"
    return "`-"


def _format_node_label(node: LineageNode) -> str:
    if node.kind == "xp":
        name = node.label or node.sig or "unknown-xp"
        pieces = [f"xp {name}"]
        if node.sig and node.label:
            pieces.append(f"sig={node.sig}")
        if node.epochs is not None:
            pieces.append(f"ep={node.epochs}")
        if node.workspace:
            pieces.append(f"ws={node.workspace}")
        if node.fingerprint_digest:
            pieces.append(f"data={node.fingerprint_digest}")
        return pieces[0] if len(pieces) == 1 else f"{pieces[0]} ({', '.join(pieces[1:])})"
    if node.kind == "merge":
        op = node.operation or {}
        return f"merge {op.get('method', '?')} alpha={op.get('alpha', '?')}"
    if node.kind == "perturb":
        op = node.operation or {}
        return f"perturb sigma={op.get('sigma', '?')} scope={op.get('layer_scope', '?')} seed={op.get('seed', '?')}"
    if node.kind == "checkpoint":
        checkpoint = node.checkpoint_path or node.path or node.id
        epoch = f" epoch={node.epoch}" if node.epoch else ""
        return f"checkpoint {checkpoint}{epoch}"
    return f"unknown {node.label or node.path or node.id}"


def _format_edge_label(edge: LineageEdge) -> str:
    summary = edge.summary or {}
    if edge.kind == "warm_start":
        epoch = summary.get("checkpoint_epoch")
        return f"warm-start checkpoint_{epoch}" if epoch else "warm-start"
    if edge.kind == "merge_source":
        return f"merge source {edge.role or '?'}"
    if edge.kind == "perturb_source":
        return "perturb source"
    return edge.kind.replace("_", " ")


def _format_override_diff(diff: OverrideDiff) -> str:
    parts: list[str] = []
    if diff.added:
        parts.append("added " + ", ".join(f"{key}={value}" for key, value in sorted(diff.added.items())))
    if diff.removed:
        parts.append("removed " + ", ".join(sorted(diff.removed)))
    if diff.changed:
        parts.append(
            "changed "
            + ", ".join(
                f"{key}:{value.get('from', '')}->{value.get('to', '')}" for key, value in sorted(diff.changed.items())
            )
        )
    return "; ".join(parts)


def _has_diff(diff: OverrideDiff) -> bool:
    return bool(diff.added or diff.removed or diff.changed)


class _LineageBuilder:
    """Walks the provenance trees (``provenance.resolve_init``) of an XP and of every XP ancestor it meets."""

    def __init__(self, *, dora_dir: Path, repo_root: Path, max_depth: int) -> None:
        self.dora_dir = dora_dir.expanduser().resolve()
        self.repo_root = repo_root.expanduser().resolve()
        self.max_depth = max_depth
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []
        self.warnings: list[str] = []
        self._expanded_xps: set[str] = set()
        self._expanded_operations: set[str] = set()

    def add_xp(self, sig: str, *, depth: int, stack: tuple[str, ...]) -> str:
        node_id = f"xp:{sig}"
        if node_id in stack:
            self.warnings.append(f"Cycle detected at XP {sig}; lineage expansion stopped for that branch.")
            return node_id

        xp_folder = self.dora_dir / "xps" / sig
        metadata = _load_train_metadata(xp_folder)
        if metadata is None:
            if depth == 0:
                if not xp_folder.is_dir():
                    raise FileNotFoundError(f"XP folder does not exist: {xp_folder}")
                raise FileNotFoundError(
                    f"Missing {TRAIN_METADATA_FILENAME} for XP {sig}. Run `tidy backfill --commit` first."
                )
            self.nodes.setdefault(
                node_id,
                LineageNode(id=node_id, kind="unknown", label=sig, path=str(xp_folder), sig=sig),
            )
            self.warnings.append(f"Could not load train metadata for ancestor XP {sig}: {xp_folder}")
            return node_id

        node = _xp_node_from_metadata(node_id, xp_folder, metadata)
        self.nodes[node_id] = node
        if depth > self.max_depth:
            self.warnings.append(f"Depth limit reached at XP {sig}; pass --max-depth to inspect deeper ancestry.")
            return node_id
        if node_id in self._expanded_xps:
            return node_id
        self._expanded_xps.add(node_id)

        origin = resolve_init(metadata, xps_dir=self.dora_dir / "xps")
        if origin is None:
            return node_id
        parent_id = self._add_origin(origin, depth=depth + 1, stack=(*stack, node_id))
        if parent_id is not None:
            parent_node = self.nodes.get(parent_id)
            self.nodes[node_id] = _with_override_diff(
                node,
                _override_diff(metadata, _load_metadata_for_node(parent_node)),
            )
            self._add_edge(
                LineageEdge(
                    from_node_id=node_id,
                    to_node_id=parent_id,
                    kind="warm_start",
                    summary=_checkpoint_summary(origin.checkpoint),
                )
            )
        return node_id

    def _add_origin(self, origin: Origin, *, depth: int, stack: tuple[str, ...]) -> str | None:
        if origin.kind in ("merge", "perturb"):
            return self._add_operation(origin, depth=depth, stack=stack)
        checkpoint_path = origin.checkpoint
        if checkpoint_path is None:
            return None
        if origin.kind == "xp" and origin.sig and _load_train_metadata(self.dora_dir / "xps" / origin.sig):
            return self.add_xp(origin.sig, depth=depth, stack=stack)
        node_id = _unknown_node_id(checkpoint_path)
        parsed = _parse_checkpoint_path(checkpoint_path)
        self.nodes.setdefault(
            node_id,
            LineageNode(
                id=node_id,
                kind="unknown",
                label=parsed[0] if parsed else None,
                path=checkpoint_path,
                checkpoint_path=checkpoint_path,
                epoch=parsed[1] if parsed else None,
            ),
        )
        self.warnings.append(f"Could not resolve checkpoint ancestor to an XP with metadata: {checkpoint_path}")
        return node_id

    def _add_operation(self, origin: Origin, *, depth: int, stack: tuple[str, ...]) -> str:
        payload = dict(origin.payload or {})
        sidecar_path = Path(origin.sidecar) if origin.sidecar else None
        node_id = _sidecar_node_id(origin.kind, payload, sidecar_path, origin.checkpoint)
        self.nodes.setdefault(
            node_id,
            LineageNode(
                id=node_id,
                kind=origin.kind,
                path=origin.sidecar or origin.checkpoint,
                checkpoint_path=_output_path(payload) or origin.checkpoint,
                operation=operation_summary(origin),
            ),
        )
        if node_id in self._expanded_operations:
            return node_id
        self._expanded_operations.add(node_id)

        if depth > self.max_depth:
            self.warnings.append(f"Depth limit reached while expanding {origin.kind} sidecar.")
            return node_id

        edge_kind = "merge_source" if origin.kind == "merge" else "perturb_source"
        for source in origin.sources:
            parent_id = self._add_origin(source.origin, depth=depth + 1, stack=stack)
            if parent_id is not None:
                self._add_edge(
                    LineageEdge(
                        from_node_id=node_id,
                        to_node_id=parent_id,
                        kind=edge_kind,
                        role=source.role,
                        summary=_checkpoint_summary(source.checkpoint or None),
                    )
                )
        return node_id

    def _add_edge(self, edge: LineageEdge) -> None:
        if edge not in self.edges:
            self.edges.append(edge)


def _xp_node_from_metadata(node_id: str, xp_folder: Path, metadata: dict[str, Any]) -> LineageNode:
    fingerprint = metadata.get("dataset_fingerprint")
    train_count = None
    dataset_name = None
    if isinstance(fingerprint, dict):
        train_count = _int_or_none(fingerprint.get("train_count"))
        dataset_name = _string_or_none(fingerprint.get("dataset_name"))
    workspace_dir = _string_or_none(metadata.get("workspace_dir"))
    checkpoint_path = xp_folder / "checkpoint.th"
    return LineageNode(
        id=node_id,
        kind="xp",
        label=_string_or_none(metadata.get("effective_label")),
        path=str(xp_folder),
        sig=_string_or_none(metadata.get("xp_sig")) or xp_folder.name,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        workspace=Path(workspace_dir).name if workspace_dir else None,
        dataset_name=dataset_name or _string_or_none(metadata.get("dataset_name")),
        train_count=train_count,
        fingerprint_digest=_string_or_none(metadata.get("dataset_fingerprint_digest")),
        epochs=_extract_epochs(metadata.get("command")),
    )


def _load_metadata_for_node(node: LineageNode | None) -> dict[str, Any] | None:
    if node is None or node.kind != "xp" or node.path is None:
        return None
    return _load_train_metadata(Path(node.path))


def _load_train_metadata(xp_folder: Path) -> dict[str, Any] | None:
    path = xp_folder / TRAIN_METADATA_FILENAME
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _with_override_diff(node: LineageNode, diff: OverrideDiff | None) -> LineageNode:
    if diff is None or not _has_diff(diff):
        return node
    return LineageNode(
        id=node.id,
        kind=node.kind,
        label=node.label,
        path=node.path,
        sig=node.sig,
        epoch=node.epoch,
        checkpoint_path=node.checkpoint_path,
        workspace=node.workspace,
        dataset_name=node.dataset_name,
        train_count=node.train_count,
        fingerprint_digest=node.fingerprint_digest,
        epochs=node.epochs,
        operation=node.operation,
        override_diff=diff,
    )


def _override_diff(child_metadata: dict[str, Any], parent_metadata: dict[str, Any] | None) -> OverrideDiff | None:
    if parent_metadata is None:
        return None
    child = _parse_overrides(child_metadata.get("command"))
    parent = _parse_overrides(parent_metadata.get("command"))
    child.pop("continue_from", None)
    parent.pop("continue_from", None)
    added = {key: value for key, value in child.items() if key not in parent}
    removed = {key: value for key, value in parent.items() if key not in child}
    changed = {
        key: {"from": parent[key], "to": child[key]}
        for key in child.keys() & parent.keys()
        if child[key] != parent[key]
    }
    return OverrideDiff(added=added, removed=removed, changed=changed)


def _parse_overrides(command: Any) -> dict[str, str]:
    if not isinstance(command, list):
        return {}
    overrides: dict[str, str] = {}
    for item in command:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key:
            overrides[key] = value
    return overrides


def _extract_epochs(command: Any) -> int | None:
    overrides = _parse_overrides(command)
    return _int_or_none(overrides.get("optim.epochs"))


def _parse_checkpoint_path(checkpoint_path: str | None) -> tuple[str, str | None] | None:
    if not checkpoint_path or checkpoint_path.startswith("//"):
        return None
    path = Path(checkpoint_path)
    name = path.name
    if name == "checkpoint.th":
        return (path.parent.name, None) if path.parent.name else None
    if name.startswith("checkpoint_") and name.endswith(".th"):
        return (path.parent.name, name[len("checkpoint_") : -len(".th")]) if path.parent.name else None
    return None


def _checkpoint_summary(checkpoint_path: str | None) -> dict[str, object]:
    parsed = _parse_checkpoint_path(checkpoint_path)
    if parsed is None:
        return {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
    sig, epoch = parsed
    payload: dict[str, object] = {"checkpoint_path": checkpoint_path, "checkpoint_sig": sig}
    if epoch is not None:
        payload["checkpoint_epoch"] = epoch
    return payload


def _output_path(payload: dict[str, Any]) -> str | None:
    output = payload.get("output")
    if isinstance(output, dict):
        return _string_or_none(output.get("path"))
    return None


def _sidecar_node_id(kind: str, payload: dict[str, Any], sidecar_path: Path | None, checkpoint_path: str | None) -> str:
    if sidecar_path is not None:
        return f"{kind}:{sidecar_path.expanduser().resolve()}"
    output_path = _output_path(payload) or checkpoint_path
    if output_path:
        return f"{kind}:{output_path}"
    return f"{kind}:{id(payload)}"


def _unknown_node_id(path: str) -> str:
    return f"unknown:{path}"


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
