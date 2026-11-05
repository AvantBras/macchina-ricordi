"""The zoo: every model, who it came from, what was played with it, and which experiments name it.

One graph, built once from what is on disk:

- models are the records ``record.list_models`` computes (one per ``.dora/xps/<sig>/``);
- an edge joins a model to each parent its provenance names (``provenance.resolve_init``), through merges and
  perturbations, so ``model show``, ``model lineage``, the catalog and the GUI agree on parents;
- sessions are everything ``play`` made (``sessions.store.list_sessions``), of every kind;
- experiments are the ledger's entries (``research/training/experiment_ledger.toml``).

Nothing here writes. Readers render the graph: the catalog's text, the GUI's covers, index and model page,
the derive page's "children" column, and ``model list`` / ``model show``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..musicgen.shared import read_json
from .ledger import ExperimentLedgerEntry, load_experiment_ledger
from .provenance import SIDECARS, Origin, operation_output, resolve_checkpoint
from .record import Model, list_bundle_models, list_models


@dataclass(frozen=True)
class ZooEdge:
    child: str
    parent: str
    operation: str
    """``warm-start``, ``merge`` or ``perturb``: what stood between the parent's weights and the child's start."""
    checkpoint: str | None
    """The file the child started from (the parent's checkpoint, or a merge or perturbation output)."""


DERIVED_ROOTS = ("artifacts/checkpoints", "artifacts/sessions", "merged_checkpoints", "perturbed_checkpoints")
"""Where merge and perturbation outputs live, besides each legacy workspace's own ``merged_checkpoints``."""


@dataclass(frozen=True)
class DerivedCheckpoint:
    """A merge or perturbation output: the file, the sidecar describing it, and where its weights came from."""

    kind: str
    """``merge`` or ``perturb``."""
    checkpoint: Path
    """The output the sidecar names, in the sidecar's folder (so a moved folder still finds it)."""
    sidecar: Path
    payload: dict[str, Any] = field(repr=False)
    origin: Origin = field(repr=False)
    unrecorded: tuple[Path, ...] = ()
    """Other checkpoints in the same folder: one folder keeps one sidecar, for the last output written there."""

    @property
    def sources(self) -> tuple[tuple[str, str], ...]:
        """``(role, path)`` for each source the sidecar names."""
        return tuple((source.role, source.checkpoint) for source in self.origin.sources if source.checkpoint)

    @property
    def parent_sigs(self) -> tuple[str, ...]:
        return self.origin.parent_sigs()


def scan_derived(repo_root: str | Path) -> tuple[DerivedCheckpoint, ...]:
    """Every merge and perturbation sidecar under the derived roots, oldest first."""
    root = Path(repo_root)
    roots = [root / name for name in DERIVED_ROOTS]
    roots += [path for pattern in (".*/merged_checkpoints", ".*/perturbed_checkpoints") for path in root.glob(pattern)]
    kinds = {filename: kind for _, kind, filename in SIDECARS}
    found: dict[Path, DerivedCheckpoint] = {}
    for base in roots:
        if not base.is_dir():
            continue
        for filename, kind in kinds.items():
            for sidecar in base.rglob(filename):
                resolved = sidecar.resolve()
                if resolved in found:
                    continue
                payload = read_json(sidecar)
                if not isinstance(payload, dict):
                    continue
                name = Path(str(operation_output(payload).get("path") or "checkpoint.th")).name
                checkpoint = sidecar.parent / name
                found[resolved] = DerivedCheckpoint(
                    kind=kind,
                    checkpoint=checkpoint,
                    sidecar=sidecar,
                    payload=payload,
                    origin=resolve_checkpoint(str(checkpoint)),
                    unrecorded=tuple(sorted(p for p in sidecar.parent.glob("*.th") if p.name != checkpoint.name)),
                )
    return tuple(sorted(found.values(), key=lambda d: (str(d.payload.get("created_at") or ""), str(d.sidecar))))


@dataclass(frozen=True)
class Zoo:
    repo_root: Path
    models: tuple[Model, ...]
    edges: tuple[ZooEdge, ...]
    sessions: tuple[Any, ...] = ()
    """``sessions.record.Session`` objects, newest first; empty unless built with ``include_sessions``."""
    sessions_loaded: bool = False
    experiments: tuple[ExperimentLedgerEntry, ...] = ()
    derived: tuple[DerivedCheckpoint, ...] = ()
    """Merge and perturbation outputs (``scan_derived``); empty unless built with ``include_derived``."""
    ledger_path: str | None = None
    ledger_warnings: tuple[str, ...] = ()
    generated_at: str = ""
    _by_sig: dict[str, Model] = field(default_factory=dict, init=False, repr=False, compare=False)
    _parents: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _children: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _sessions: dict[str, tuple[Any, ...]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _experiments: dict[str, tuple[ExperimentLedgerEntry, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        parents: dict[str, list[str]] = {}
        children: dict[str, list[str]] = {}
        for edge in self.edges:
            parents.setdefault(edge.child, []).append(edge.parent)
            children.setdefault(edge.parent, []).append(edge.child)
        sessions: dict[str, list[Any]] = {}
        for session in self.sessions:
            for sig in session.sigs:
                sessions.setdefault(sig, []).append(session)
        experiments: dict[str, list[ExperimentLedgerEntry]] = {}
        for entry in self.experiments:
            for sig in dict.fromkeys(entry.xp_sigs):
                experiments.setdefault(sig, []).append(entry)
        object.__setattr__(self, "_by_sig", {model.sig: model for model in self.models})
        object.__setattr__(self, "_parents", {sig: tuple(dict.fromkeys(v)) for sig, v in parents.items()})
        object.__setattr__(self, "_children", {sig: tuple(dict.fromkeys(v)) for sig, v in children.items()})
        object.__setattr__(self, "_sessions", {sig: tuple(v) for sig, v in sessions.items()})
        object.__setattr__(self, "_experiments", {sig: tuple(v) for sig, v in experiments.items()})

    # ── models and their family ──

    def model(self, sig: str) -> Model | None:
        return self._by_sig.get(sig)

    def parents(self, sig: str) -> tuple[str, ...]:
        """Direct parents, in the order the provenance names them; a parent may no longer be on disk."""
        return self._parents.get(sig, ())

    def children(self, sig: str) -> tuple[str, ...]:
        return tuple(sorted(self._children.get(sig, ())))

    def ancestors(self, sig: str) -> tuple[str, ...]:
        return _walk(sig, self.parents)

    def descendants(self, sig: str) -> tuple[str, ...]:
        return _walk(sig, self.children)

    def display_name(self, sig: str | None) -> str | None:
        """A model's name, else its Dora label, else the sig; ``sig-epoch`` checkpoint references work too."""
        if not sig:
            return None
        model = self._by_sig.get(sig) or self._by_sig.get(sig.split("-", 1)[0])
        return model.display_name if model is not None else sig

    def labels(self) -> dict[str, str]:
        """Each sig's display name, for the views that print a name instead of a sig."""
        return {model.sig: model.display_name for model in self.models if model.display_name != model.sig}

    def warm_started_from(self, checkpoint: str | Path, *, sha256: str | None = None) -> tuple[str, ...]:
        """Models that started from this file, matched by its path, or by its hash when the file has moved."""
        target = str(checkpoint)
        found = []
        for model in self.models:
            init = model.init
            if init is None or not init.checkpoint:
                continue
            if init.checkpoint == target:
                found.append(model.sig)
            elif sha256 and init.payload is not None:
                output = operation_output(init.payload)
                if output.get("sha256") == sha256 and Path(str(output.get("path"))).name == Path(init.checkpoint).name:
                    found.append(model.sig)
        return tuple(found)

    # ── what points at a model ──

    def sessions_for(self, sig: str) -> tuple[Any, ...]:
        return self._sessions.get(sig, ())

    def experiments_for(self, sig: str) -> tuple[ExperimentLedgerEntry, ...]:
        return self._experiments.get(sig, ())

    def family(self, sig: str) -> dict[str, Any]:
        """Parents, children and what points at one model, for ``model show`` and the GUI."""
        return {
            "parents": list(self.parents(sig)),
            "children": list(self.children(sig)),
            "ancestors": list(self.ancestors(sig)),
            "descendants": list(self.descendants(sig)),
            "sessions": [session_row(session) for session in self.sessions_for(sig)],
            "experiments": [entry.experiment_id for entry in self.experiments_for(sig)],
        }


def build_zoo(
    repo_root: str | Path | None = None,
    *,
    include_sessions: bool = True,
    include_experiments: bool = True,
    include_derived: bool = True,
    include_bundles: bool = True,
) -> Zoo:
    """Read the zoo. Sessions are the slow part (hundreds of folders); leave them out when only models matter."""
    from ..musicgen.config import get_repo_root

    root = Path(repo_root or get_repo_root()).expanduser().resolve()
    # without an explicit root, models come from AUDIOCRAFT_DORA_DIR or the configured Dora dir, as everywhere
    models = tuple(list_models(repo_root))
    bundles = tuple(list_bundle_models(repo_root)) if include_bundles else ()
    models = models + bundles
    edges = tuple(
        ZooEdge(model.sig, parent, model.origin or "warm-start", model.init.checkpoint if model.init else None)
        for model in models
        for parent in model.parents
    )
    sessions: tuple[Any, ...] = ()
    if include_sessions:
        from ..sessions.store import list_sessions

        sessions = tuple(list_sessions(root))
    experiments: tuple[ExperimentLedgerEntry, ...] = ()
    ledger_path = None
    ledger_warnings: tuple[str, ...] = ()
    if include_experiments:
        ledger = load_experiment_ledger(root)
        experiments, ledger_path, ledger_warnings = ledger.entries, ledger.path, ledger.warnings
    return Zoo(
        repo_root=root,
        models=models,
        edges=edges,
        sessions=sessions,
        sessions_loaded=include_sessions,
        experiments=experiments,
        derived=scan_derived(root) if include_derived else (),
        ledger_path=ledger_path,
        ledger_warnings=ledger_warnings,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def session_row(session: Any) -> dict[str, Any]:
    return {
        "id": session.id,
        "kind": session.kind,
        "created_at": session.created_at,
        "label": session.label,
        "take_count": session.take_count,
        "path": str(session.path),
    }


def _walk(start: str, step: Any) -> tuple[str, ...]:
    seen: list[str] = []
    queue: list[str] = list(step(start))
    while queue:
        sig = queue.pop(0)
        if sig in seen or sig == start:
            continue
        seen.append(sig)
        queue.extend(_unseen(step(sig), seen))
    return tuple(seen)


def _unseen(items: Iterable[str], seen: list[str]) -> list[str]:
    return [item for item in items if item not in seen]
