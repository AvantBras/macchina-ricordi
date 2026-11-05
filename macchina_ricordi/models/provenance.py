"""Where a model's starting weights came from: one reading, shared by the model record, lineage and the zoo.

A training that warm-starts records ``init_from_checkpoint`` in its metadata: the checkpoint path, and the
sidecars (``merged_from``, ``perturbed_from``) that sat next to that checkpoint when training read it. From that
record, ``resolve_init`` builds an ``Origin`` tree. Its leaves are XP checkpoints or files nothing explains, and
its inner nodes are the merges and perturbations in between.

Rules, in order, for one checkpoint:

1. the sidecar snapshot embedded in the record (when it holds both a merge and a perturbation, the newest
   operation is the direct one: a perturbation of a merge-derived checkpoint inherits the merge record);
2. a ``merged_from.json`` / ``perturbed_from.json`` next to the checkpoint on disk, when it names that file as
   its output (a folder keeps one sidecar, for the last output written there);
3. a checkpoint inside an XP folder (``.dora/xps/<sig>/``);
4. anything else is unknown (a Hugging Face id, a file whose sidecar is gone).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..musicgen.shared import read_json

SIDECARS = (("merged_from", "merge", "merged_from.json"), ("perturbed_from", "perturb", "perturbed_from.json"))
MAX_DEPTH = 32


@dataclass(frozen=True)
class OriginSource:
    role: str
    """``a`` or ``b`` for a merge, ``source`` for a perturbation."""
    checkpoint: str
    origin: Origin


@dataclass(frozen=True)
class Origin:
    kind: str
    """``xp``, ``merge``, ``perturb`` or ``unknown``."""
    checkpoint: str | None
    """The checkpoint file this node stands for."""
    sig: str | None = None
    epoch: str | None = None
    """The epoch in a ``checkpoint_<n>.th`` name."""
    sidecar: str | None = None
    """The sidecar file the operation was read from, when it was read from disk rather than from a record."""
    payload: Mapping[str, Any] | None = field(default=None, repr=False)
    sources: tuple[OriginSource, ...] = ()
    cut: str | None = None
    """``cycle`` or ``depth`` when resolution stopped at this node."""

    @property
    def operation(self) -> str:
        """How the weights were made: ``merge``, ``perturb``, or ``warm-start`` for a plain checkpoint."""
        return self.kind if self.kind in ("merge", "perturb") else "warm-start"

    def parent_sigs(self) -> tuple[str, ...]:
        """The nearest trained models: XPs reached by looking through merges and perturbations."""
        if self.kind == "xp":
            return (self.sig,) if self.sig else ()
        found: list[str] = []
        for source in self.sources:
            for sig in source.origin.parent_sigs():
                if sig not in found:
                    found.append(sig)
        return tuple(found)

    def operations(self) -> tuple[Origin, ...]:
        """Every merge and perturbation in the tree, nearest first."""
        found: list[Origin] = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if node.kind in ("merge", "perturb"):
                found.append(node)
            queue.extend(source.origin for source in node.sources)
        return tuple(found)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "checkpoint": self.checkpoint}
        for key in ("sig", "epoch", "sidecar", "cut"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        if self.payload is not None:
            data["operation"] = operation_summary(self)
        if self.sources:
            data["sources"] = [
                {"role": s.role, "checkpoint": s.checkpoint, "origin": s.origin.to_dict()} for s in self.sources
            ]
        return data


def init_checkpoint(metadata: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None] | None:
    """The init checkpoint a training recorded, and the sidecar snapshot that came with it; ``None`` from scratch."""
    init = metadata.get("init_from_checkpoint")
    if not isinstance(init, Mapping):
        return None
    source = init.get("resolved_local_path") or init.get("user_input")
    embedded = init.get("source_resume_metadata")
    return (
        str(source) if isinstance(source, str) and source else None,
        embedded if isinstance(embedded, Mapping) else None,
    )


def resolve_init(metadata: Mapping[str, Any], *, xps_dir: Path | None = None) -> Origin | None:
    """The origin of a model's starting weights, from its
    ``macchina_ricordi_train_metadata.json``; ``None`` from scratch.
    """
    init = init_checkpoint(metadata)
    if init is None:
        return None
    checkpoint, embedded = init
    if checkpoint is None and not embedded:
        return None
    return resolve_checkpoint(checkpoint, embedded, xps_dir=xps_dir)


def resolve_checkpoint(
    checkpoint: str | None,
    embedded: Mapping[str, Any] | None = None,
    *,
    xps_dir: Path | None = None,
) -> Origin:
    return _Resolver(xps_dir).resolve(checkpoint, embedded, depth=0, seen=frozenset())


def newest_operation(embedded: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any]] | None:
    """``(kind, payload)`` of the most recent operation in a sidecar snapshot."""
    if not isinstance(embedded, Mapping):
        return None
    found = [
        (str(payload.get("created_at") or ""), kind, payload)
        for key, kind, _ in SIDECARS
        if isinstance(payload := embedded.get(key), Mapping)
    ]
    if not found:
        return None
    _, kind, payload = max(found, key=lambda item: item[0])
    return kind, payload


def operation_summary(origin: Origin) -> dict[str, Any]:
    payload = origin.payload or {}
    if origin.kind == "merge":
        keys = ("method", "alpha", "created_at")
    elif origin.kind == "perturb":
        keys = ("sigma", "layer_scope", "seed", "created_at")
    else:
        return {}
    return {key: payload.get(key) for key in keys}


def operation_output(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    output = payload.get("output") if isinstance(payload, Mapping) else None
    return output if isinstance(output, Mapping) else {}


def xp_sig_from_path(path: object) -> str | None:
    if not isinstance(path, str | Path):
        return None
    parts = Path(path).parts
    for index in range(len(parts) - 2):
        if parts[index] == ".dora" and parts[index + 1] == "xps":
            return parts[index + 2]
    return None


def checkpoint_epoch(path: str | None) -> str | None:
    if not path:
        return None
    name = Path(path).name
    if name.startswith("checkpoint_") and name.endswith(".th"):
        return name[len("checkpoint_") : -len(".th")] or None
    return None


class _Resolver:
    def __init__(self, xps_dir: Path | None) -> None:
        self.xps_dir = xps_dir.expanduser().resolve() if xps_dir is not None else None

    def resolve(
        self,
        checkpoint: str | None,
        embedded: Mapping[str, Any] | None,
        *,
        depth: int,
        seen: frozenset[str],
    ) -> Origin:
        operation = newest_operation(embedded)
        if operation is not None:
            kind, payload = operation
            return self._operation(kind, payload, checkpoint, None, depth=depth, seen=seen)
        if checkpoint and not checkpoint.startswith("//"):
            for _, kind, filename in SIDECARS:
                sidecar = Path(checkpoint).expanduser().parent / filename
                payload = read_json(sidecar)
                if isinstance(payload, dict) and _describes(payload, checkpoint):
                    return self._operation(kind, payload, checkpoint, sidecar, depth=depth, seen=seen)
        sig = self._xp_sig(checkpoint)
        if sig is not None:
            return Origin("xp", checkpoint, sig=sig, epoch=checkpoint_epoch(checkpoint))
        return Origin("unknown", checkpoint, epoch=checkpoint_epoch(checkpoint))

    def _operation(
        self,
        kind: str,
        payload: Mapping[str, Any],
        checkpoint: str | None,
        sidecar: Path | None,
        *,
        depth: int,
        seen: frozenset[str],
    ) -> Origin:
        sidecar_text = str(sidecar.resolve()) if sidecar is not None else None
        key = sidecar_text or f"{kind}:{operation_output(payload).get('path') or checkpoint}"
        if key in seen:
            return Origin(kind, checkpoint, sidecar=sidecar_text, payload=payload, cut="cycle")
        if depth >= MAX_DEPTH:
            return Origin(kind, checkpoint, sidecar=sidecar_text, payload=payload, cut="depth")
        seen = seen | {key}
        sources = payload.get("sources")
        sources = sources if isinstance(sources, Mapping) else {}
        resolved: list[OriginSource] = []
        if kind == "merge":
            for role in ("a", "b"):
                path = _source_path(sources.get(role))
                if path is not None:
                    origin = self.resolve(path, None, depth=depth + 1, seen=seen)
                    resolved.append(OriginSource(role, path, origin))
        else:
            path = _source_path(sources.get("source"))
            nested = payload.get("source_resume_metadata")
            if path is not None or newest_operation(nested) is not None:
                origin = self.resolve(path, nested if isinstance(nested, Mapping) else None, depth=depth + 1, seen=seen)
                resolved.append(OriginSource("source", path or "", origin))
        return Origin(kind, checkpoint, sidecar=sidecar_text, payload=payload, sources=tuple(resolved))

    def _xp_sig(self, checkpoint: str | None) -> str | None:
        if not checkpoint or checkpoint.startswith("//"):
            return None
        sig = xp_sig_from_path(checkpoint)
        if sig is not None:
            return sig
        if self.xps_dir is not None:
            folder = Path(checkpoint).expanduser().parent
            try:
                if folder.resolve().parent == self.xps_dir:
                    return folder.name
            except OSError:
                return None
        return None


def _describes(payload: Mapping[str, Any], checkpoint: str) -> bool:
    """A folder keeps one sidecar per kind, for the last output written there; it describes only that file."""
    output = operation_output(payload).get("path")
    return not output or Path(str(output)).name == Path(checkpoint).name


def _source_path(source: Any) -> str | None:
    path = source.get("path") if isinstance(source, Mapping) else None
    return path if isinstance(path, str) and path else None
