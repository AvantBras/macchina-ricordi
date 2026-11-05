from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from textwrap import fill
from typing import Any

from ..listening.reports import read_listening_report
from ..musicgen.shared import read_json
from .ledger import ExperimentLedgerEntry
from .lineage import LineageGraph, build_lineage_graph_for_sig
from .provenance import operation_output
from .record import TRAIN_METADATA_FILENAME, Model, is_xp_folder, load_model, load_registry
from .zoo import DerivedCheckpoint, Zoo, build_zoo, scan_derived


@dataclass(frozen=True)
class CatalogModelEntry:
    sig: str
    xp_folder: str
    label: str | None
    workspace: str
    dataset_name: str | None
    train_count: int | None
    fingerprint_digest: str | None
    epochs: int | None
    checkpoint_path: str | None
    checkpoint_size_mb: float | None
    continue_from: str | None
    merged_from: CatalogMergeEntry | None
    written_at: str | None
    running: bool = False
    perturbed_from: CatalogPerturbEntry | None = None
    lineage: LineageGraph | None = None
    name: str | None = None
    parents: tuple[str, ...] = ()
    origin: str | None = None


@dataclass(frozen=True)
class CatalogMergeEntry:
    method: str
    alpha: float
    source_a_sig: str | None
    source_b_sig: str | None
    source_a_path: str | None
    source_b_path: str | None
    output_size_mb: float | None
    created_at: str | None
    source_a_label: str | None = None
    source_b_label: str | None = None
    path: str | None = None
    """The sidecar's folder; None for the merge a model started from."""


@dataclass(frozen=True)
class CatalogPerturbEntry:
    sigma: float
    layer_scope: str
    seed: int | None
    source_sig: str | None
    source_path: str | None
    output_size_mb: float | None
    created_at: str | None
    source_label: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class CatalogGenerationEntry:
    run_id: str
    metadata_path: str
    mode: str
    prefix: str
    model_label: str | None
    model_sig: str | None
    model_workspace: str | None
    xp_path: str | None
    checkpoint_path: str | None
    duration_seconds: float
    num_samples: int
    prompt_audio: str | None
    temperature: float | None
    top_k: int | None
    seed: int | None
    written_at: str | None
    listening_summary: str | None = None
    listening_score: float | None = None
    listening_labels: str | None = None
    listening_report_paths: tuple[str, ...] = ()
    listening_markdown_paths: tuple[str, ...] = ()
    listening_error_count: int = 0


@dataclass(frozen=True)
class CatalogSweepEntry:
    sweep_id: str
    path: str
    index_path: str
    plan_path: str | None
    report_path: str | None
    html_path: str | None
    mode: str | None
    profile: str | None
    model_id: str | None
    model_sig: str | None
    checkpoint_path: str | None
    combo_count: int
    row_count: int
    sample_count: int
    scored_count: int
    listening_count: int
    created_at: str | None
    model_label: str | None = None


@dataclass(frozen=True)
class CatalogStreamEntry:
    run_id: str
    path: str
    manifest_path: str | None
    performance_path: str | None
    started_at: str | None
    finished_at: str | None
    stopped_by: str | None
    seed_mode: str | None
    model_xp: str | None
    model_sigs: tuple[str, ...]
    rounds_written: int
    scored_rounds: int
    novelty_rounds: int
    has_manifest: bool
    model_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogSessionEntry:
    session_id: str
    kind: str
    path: str
    created_at: str | None
    model_sigs: tuple[str, ...]
    take_count: int
    parent: str | None = None
    model_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogPortraitEntry:
    title: str
    path: str
    generated_at: str | None
    model_sigs: tuple[str, ...]
    model_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogExperimentEntry(ExperimentLedgerEntry):
    model_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Catalog:
    models: tuple[CatalogModelEntry, ...]
    merges: tuple[CatalogMergeEntry, ...]
    generations: tuple[CatalogGenerationEntry, ...]
    generated_at: str
    perturbations: tuple[CatalogPerturbEntry, ...] = ()
    sweeps: tuple[CatalogSweepEntry, ...] = ()
    streams: tuple[CatalogStreamEntry, ...] = ()
    portraits: tuple[CatalogPortraitEntry, ...] = ()
    experiments: tuple[CatalogExperimentEntry, ...] = ()
    ledger_path: str | None = None
    ledger_warnings: tuple[str, ...] = ()
    other_sessions: tuple[CatalogSessionEntry, ...] = ()


def _parse_sig_from_path(checkpoint_path: str | None) -> str | None:
    """Extract a human-readable sig+epoch from a checkpoint path.

    e.g. .../e16bf504/checkpoint_150.th -> "e16bf504-150"
    """
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    epoch = ""
    name = path.name
    if name.startswith("checkpoint_") and name.endswith(".th"):
        epoch = "-" + name[len("checkpoint_") : -len(".th")]
    xp_dir = path.parent.name
    return f"{xp_dir}{epoch}" if xp_dir else None


def _extract_epochs(command: list | None) -> int | None:
    if not command:
        return None
    epochs = None
    for part in command:
        if isinstance(part, str) and part.startswith("optim.epochs="):
            try:
                epochs = int(part.split("=", 1)[1])
            except ValueError:
                pass
    return epochs


def _resolve_checkpoint(xp_folder: Path) -> str | None:
    checkpoint = xp_folder / "checkpoint.th"
    if checkpoint.exists():
        return str(checkpoint)
    return None


def _output_size_mb(payload: dict) -> float | None:
    size = operation_output(payload).get("size_bytes")
    return round(float(size) / 1e6, 1) if isinstance(size, int | float) else None


def _merge_entry(mf: dict, path: str | None = None) -> CatalogMergeEntry:
    sources = mf.get("sources", {})
    source_a = sources.get("a", {}) if isinstance(sources, dict) else {}
    source_b = sources.get("b", {}) if isinstance(sources, dict) else {}
    source_a_path = source_a.get("path") if isinstance(source_a, dict) else None
    source_b_path = source_b.get("path") if isinstance(source_b, dict) else None
    return CatalogMergeEntry(
        method=str(mf.get("method", "?")),
        alpha=float(mf.get("alpha", 0.5)),
        source_a_path=source_a_path,
        source_b_path=source_b_path,
        source_a_sig=_parse_sig_from_path(source_a_path),
        source_b_sig=_parse_sig_from_path(source_b_path),
        output_size_mb=_output_size_mb(mf),
        created_at=mf.get("created_at"),
        path=path,
    )


def _perturb_entry(pf: dict, path: str | None = None) -> CatalogPerturbEntry:
    sources = pf.get("sources", {})
    source = sources.get("source", {}) if isinstance(sources, dict) else {}
    source_path = source.get("path") if isinstance(source, dict) else None
    return CatalogPerturbEntry(
        sigma=float(pf.get("sigma", 0.0)),
        layer_scope=str(pf.get("layer_scope", "?")),
        seed=int(pf["seed"]) if pf.get("seed") is not None else None,
        source_path=source_path,
        source_sig=_parse_sig_from_path(source_path),
        output_size_mb=_output_size_mb(pf),
        created_at=pf.get("created_at"),
        path=path,
    )


def _catalog_model(model: Model, metadata: dict, *, running: bool) -> CatalogModelEntry:
    fingerprint = metadata.get("dataset_fingerprint")
    train_count = None
    dataset_name = None
    if isinstance(fingerprint, dict):
        tc = fingerprint.get("train_count")
        train_count = int(tc) if tc is not None else None
        dataset_name = fingerprint.get("dataset_name")

    checkpoint_path = _resolve_checkpoint(model.folder)
    checkpoint_mb = None
    if checkpoint_path:
        try:
            checkpoint_mb = round(Path(checkpoint_path).stat().st_size / 1e6, 1)
        except OSError:
            pass

    merged_from = None
    perturbed_from = None
    for operation in model.init.operations() if model.init is not None else ():
        if operation.kind == "merge" and merged_from is None:
            merged_from = _merge_entry(dict(operation.payload or {}))
        if operation.kind == "perturb" and perturbed_from is None:
            perturbed_from = _perturb_entry(dict(operation.payload or {}))

    return CatalogModelEntry(
        sig=model.sig,
        xp_folder=str(model.folder),
        label=model.label,
        workspace=model.dataset or "unknown",
        dataset_name=dataset_name,
        train_count=train_count,
        fingerprint_digest=metadata.get("dataset_fingerprint_digest"),
        epochs=model.epochs_planned,
        checkpoint_path=checkpoint_path,
        checkpoint_size_mb=checkpoint_mb,
        continue_from=model.init.checkpoint if model.init is not None else None,
        merged_from=merged_from,
        written_at=metadata.get("written_at"),
        running=running,
        perturbed_from=perturbed_from,
        name=model.name,
        parents=model.parents,
        origin=model.origin,
    )


def gather_models(dora_dir: Path, *, models: tuple[Model, ...] | None = None) -> list[CatalogModelEntry]:
    """One catalog row per model record (``record.load_model``), or per record of an already built zoo."""
    if models is None:
        xps_dir = dora_dir / "xps"
        if not xps_dir.is_dir():
            return []
        registry = load_registry(dora_dir.parent)
        models = tuple(load_model(path, registry=registry) for path in sorted(xps_dir.iterdir()) if is_xp_folder(path))

    from ..training.jobs import training_sigs

    running = training_sigs()
    entries: list[CatalogModelEntry] = []
    for model in models:
        metadata = read_json(model.folder / TRAIN_METADATA_FILENAME) or {}
        entries.append(_catalog_model(model, metadata, running=model.sig in running))
    return entries


def _derived(root: Path, derived: Iterable[DerivedCheckpoint] | None) -> tuple[DerivedCheckpoint, ...]:
    return tuple(derived) if derived is not None else scan_derived(root)


def gather_merges(root: Path, *, derived: Iterable[DerivedCheckpoint] | None = None) -> list[CatalogMergeEntry]:
    """The zoo's merge outputs (``zoo.scan_derived``), one row per sidecar."""
    return [_merge_entry(d.payload, str(d.sidecar.parent)) for d in _derived(root, derived) if d.kind == "merge"]


def gather_perturbations(
    root: Path, *, derived: Iterable[DerivedCheckpoint] | None = None
) -> list[CatalogPerturbEntry]:
    """The zoo's perturbation outputs (``zoo.scan_derived``), one row per sidecar."""
    return [_perturb_entry(d.payload, str(d.sidecar.parent)) for d in _derived(root, derived) if d.kind == "perturb"]


GENERATION_KINDS = ("generate", "continue")


def _sessions(root: Path, sessions: Iterable[Any] | None) -> list[Any]:
    if sessions is not None:
        return list(sessions)
    from ..sessions.store import list_sessions

    return list_sessions(root)


def _repo_path(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None and path.exists() else None


def gather_generations(root: Path, *, sessions: Iterable[Any] | None = None) -> list[CatalogGenerationEntry]:
    """One row per generate or continue session (``sessions.read``), with its takes' listening reports."""
    from ..sessions.read import facts_of

    entries: list[CatalogGenerationEntry] = []
    for session in _sessions(root, sessions):
        if session.kind not in GENERATION_KINDS:
            continue
        facts = facts_of(session)
        run_id = facts.get("run_id")
        if not run_id:
            continue
        metadata_path = session.folder / session.native if session.path.is_dir() and session.native else session.path
        takes = session.takes()
        listening = _takes_listening(takes)
        xp_path = facts.get("xp_path")
        controls = session.controls
        entries.append(
            CatalogGenerationEntry(
                run_id=str(run_id),
                metadata_path=str(metadata_path),
                mode=str(controls.get("mode", "?")),
                prefix=str(facts.get("prefix") or session.id),
                model_label=facts.get("model_label"),
                model_sig=Path(xp_path).name if isinstance(xp_path, str) else None,
                model_workspace=facts.get("model_workspace"),
                xp_path=str(Path(xp_path).resolve()) if isinstance(xp_path, str) else None,
                checkpoint_path=facts.get("checkpoint_path"),
                duration_seconds=float(controls.get("duration_seconds", 0) or 0),
                num_samples=int(facts.get("num_samples") or session.take_count),
                prompt_audio=controls.get("prompt_audio_path"),
                temperature=controls.get("temperature"),
                top_k=controls.get("top_k"),
                seed=controls.get("seed"),
                written_at=_run_id_to_iso(str(run_id)),
                **listening,
            )
        )
    return entries


def _takes_listening(takes: Iterable[Any]) -> dict[str, Any]:
    """The listening reports beside a session's takes, summarised (``listening.reports``)."""
    summaries: list[str] = []
    scores: list[float] = []
    labels: list[str] = []
    report_paths: list[str] = []
    markdown_paths: list[str] = []
    errors = 0
    for take in takes:
        report = read_listening_report(take.audio)
        if report is None:
            continue
        if report.error:
            errors += 1
            continue
        report_paths.append(str(report.report_path))
        if report.markdown_path is not None:
            markdown_paths.append(str(report.markdown_path))
        if report.summary:
            summaries.append(report.summary)
        if report.score is not None:
            scores.append(report.score)
        labels.extend(report.labels)
    summary = None
    if summaries:
        summary = summaries[0] if len(summaries) == 1 else f"{len(summaries)} reports. First: {summaries[0]}"
    return {
        "listening_summary": summary,
        "listening_score": sum(scores) / len(scores) if scores else None,
        "listening_labels": ", ".join(dict.fromkeys(labels)) or None,
        "listening_report_paths": tuple(report_paths),
        "listening_markdown_paths": tuple(markdown_paths),
        "listening_error_count": errors,
    }


def gather_sweeps(root: Path, *, sessions: Iterable[Any] | None = None) -> list[CatalogSweepEntry]:
    """One row per sweep session, counted from its takes."""
    from ..sessions.read import facts_of

    entries: list[CatalogSweepEntry] = []
    for session in _sessions(root, sessions):
        if session.kind != "sweep":
            continue
        folder = session.folder
        facts = facts_of(session)
        takes = session.takes()
        entries.append(
            CatalogSweepEntry(
                sweep_id=str(facts.get("run_id") or session.id),
                path=str(folder.resolve()),
                index_path=str((folder / "index.csv").resolve()),
                plan_path=_repo_path(folder / "sweep_plan.json"),
                report_path=_repo_path(folder / "sweep_report.json"),
                html_path=_repo_path(folder / "index.html"),
                mode=facts.get("mode"),
                profile=facts.get("profile"),
                model_id=facts.get("model_id"),
                model_sig=session.sigs[0] if session.sigs else None,
                checkpoint_path=facts.get("checkpoint_path"),
                combo_count=int(facts.get("combo_count") or 0),
                row_count=session.take_count,
                sample_count=sum(1 for take in takes if take.audio is not None and not take.error),
                scored_count=sum(1 for take in takes if "score" in take.scores),
                listening_count=int(facts.get("listening_rows") or 0),
                created_at=session.created_at,
            )
        )
    return entries


def gather_streams(root: Path, *, sessions: Iterable[Any] | None = None) -> list[CatalogStreamEntry]:
    """One row per stream session, its rounds counted from its takes."""
    from ..sessions.read import facts_of

    entries: list[CatalogStreamEntry] = []
    for session in _sessions(root, sessions):
        if session.kind != "stream":
            continue
        folder = session.folder
        facts = facts_of(session)
        takes = session.takes()
        sigs = set(session.sigs) | {take.sig for take in takes if take.sig}
        backend_sig = next(iter(session.sigs), None)
        entries.append(
            CatalogStreamEntry(
                run_id=str(facts.get("run_id") or session.id),
                path=str(folder.resolve()),
                manifest_path=_repo_path(folder / "manifest.json"),
                performance_path=_repo_path(folder / "performance.jsonl"),
                started_at=session.created_at,
                finished_at=session.finished_at,
                stopped_by=facts.get("stopped_by"),
                seed_mode=facts.get("seed_mode"),
                model_xp=backend_sig,
                model_sigs=tuple(sorted(sigs)),
                rounds_written=int(facts.get("rounds_written") or session.take_count),
                scored_rounds=sum(1 for take in takes if "onset_density" in take.scores),
                novelty_rounds=sum(1 for take in takes if take.scores.get("novelty_whitened") is not None),
                has_manifest=(folder / "manifest.json").exists(),
            )
        )
    return entries


def gather_other_sessions(root: Path, *, sessions: Iterable[Any] | None = None) -> list[CatalogSessionEntry]:
    """Duets, variations and any later kind: the sessions the generation, sweep and stream rows don't cover."""
    covered = {*GENERATION_KINDS, "sweep", "stream"}
    return [
        CatalogSessionEntry(
            session_id=session.id,
            kind=session.kind,
            path=str(session.folder.resolve()),
            created_at=session.created_at,
            model_sigs=session.sigs,
            take_count=session.take_count,
            parent=session.parent,
        )
        for session in _sessions(root, sessions)
        if session.kind not in covered
    ]


_PORTRAIT_SIG_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")


def gather_portraits(root: Path) -> list[CatalogPortraitEntry]:
    """Index maintained model portraits without parsing their prose."""
    portraits_root = root / "research" / "portraits"
    if not portraits_root.is_dir():
        return []
    entries: list[CatalogPortraitEntry] = []
    for path in sorted(portraits_root.glob("*.md")):
        if path.name == "README.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        heading = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        generated_at = None
        match = re.search(r"^Generated:\s*(.+)$", text, flags=re.MULTILINE)
        if match:
            generated_at = match.group(1).strip()
        model_sigs = tuple(
            dict.fromkeys(
                sig for sig in _PORTRAIT_SIG_RE.findall(text) if not (sig.isdigit() and sig[:2] in {"19", "20"})
            )
        )
        entries.append(
            CatalogPortraitEntry(
                title=heading,
                path=str(path.resolve()),
                generated_at=generated_at,
                model_sigs=model_sigs,
            )
        )
    return entries


def _label_for_sig(sig: str | None, labels: dict[str, str]) -> str | None:
    """Resolve an XP/checkpoint reference to its human label when known."""
    if not sig:
        return None
    if sig in labels:
        return labels[sig]
    # Checkpoint references are rendered as ``xp_sig-epoch`` in some views.
    base_sig = sig.split("-", 1)[0]
    return labels.get(base_sig)


def _display_model_ref(sig: str | None, labels: dict[str, str]) -> str | None:
    return _label_for_sig(sig, labels) or sig


def _run_id_to_iso(run_id: str) -> str | None:
    """Convert YYYYmmddTHHMMSSZ to ISO format."""
    try:
        dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def build_catalog(repo_root: Path | None = None, *, include_lineage: bool = True, zoo: Zoo | None = None) -> Catalog:
    """The catalog renders the zoo (``zoo.build_zoo``): its models, parents and experiments, plus the loose
    merge and perturbation files and a summary per generation run, sweep, stream and portrait."""
    if zoo is None:
        if repo_root is None:
            from ..musicgen.config import load_musicgen_project_config

            repo_root = load_musicgen_project_config().repo_root
        zoo = build_zoo(repo_root, include_sessions=False)
    repo_root = zoo.repo_root

    dora_dir = repo_root / ".dora"
    models = tuple(gather_models(dora_dir, models=zoo.models))
    model_labels = zoo.labels()
    lineage_models: list[CatalogModelEntry] = []
    for model in models:
        graph = None
        if include_lineage:
            try:
                graph = build_lineage_graph_for_sig(model.sig, dora_dir=dora_dir, repo_root=repo_root)
            except (FileNotFoundError, ValueError):
                graph = None
        merged_from = model.merged_from
        if merged_from is not None:
            merged_from = replace(
                merged_from,
                source_a_label=_label_for_sig(merged_from.source_a_sig, model_labels),
                source_b_label=_label_for_sig(merged_from.source_b_sig, model_labels),
            )
        perturbed_from = model.perturbed_from
        if perturbed_from is not None:
            perturbed_from = replace(
                perturbed_from,
                source_label=_label_for_sig(perturbed_from.source_sig, model_labels),
            )
        lineage_models.append(replace(model, lineage=graph, merged_from=merged_from, perturbed_from=perturbed_from))
    models = tuple(lineage_models)

    derived = zoo.derived if zoo.derived else None
    merges = tuple(gather_merges(repo_root, derived=derived))
    merges = tuple(
        replace(
            merge,
            source_a_label=_label_for_sig(merge.source_a_sig, model_labels),
            source_b_label=_label_for_sig(merge.source_b_sig, model_labels),
        )
        for merge in merges
    )
    perturbations = tuple(gather_perturbations(repo_root, derived=derived))
    perturbations = tuple(
        replace(perturbation, source_label=_label_for_sig(perturbation.source_sig, model_labels))
        for perturbation in perturbations
    )
    sessions = list(zoo.sessions) if zoo.sessions_loaded else _sessions(repo_root, None)
    generations = tuple(gather_generations(repo_root, sessions=sessions))
    generations = tuple(
        replace(generation, model_label=generation.model_label or _label_for_sig(generation.model_sig, model_labels))
        for generation in generations
    )
    sweeps = tuple(gather_sweeps(repo_root, sessions=sessions))
    sweeps = tuple(replace(sweep, model_label=_label_for_sig(sweep.model_sig, model_labels)) for sweep in sweeps)
    streams = tuple(gather_streams(repo_root, sessions=sessions))
    streams = tuple(
        replace(
            stream,
            model_labels=tuple(_display_model_ref(sig, model_labels) or "?" for sig in stream.model_sigs),
        )
        for stream in streams
    )
    other_sessions = tuple(
        replace(entry, model_labels=tuple(_display_model_ref(sig, model_labels) or "?" for sig in entry.model_sigs))
        for entry in gather_other_sessions(repo_root, sessions=sessions)
    )
    portraits = tuple(gather_portraits(root=repo_root))
    portraits = tuple(
        replace(
            portrait,
            model_labels=tuple(_display_model_ref(sig, model_labels) or "?" for sig in portrait.model_sigs),
        )
        for portrait in portraits
    )
    experiments = tuple(CatalogExperimentEntry(**vars(entry)) for entry in zoo.experiments)
    experiments = tuple(
        replace(
            experiment,
            model_labels=tuple(_display_model_ref(sig, model_labels) or "?" for sig in experiment.xp_sigs),
        )
        for experiment in experiments
    )

    return Catalog(
        models=models,
        merges=merges,
        generations=generations,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        perturbations=perturbations,
        sweeps=sweeps,
        streams=streams,
        portraits=portraits,
        experiments=experiments,
        other_sessions=other_sessions,
        ledger_path=zoo.ledger_path,
        ledger_warnings=zoo.ledger_warnings,
    )


def format_catalog_text(catalog: Catalog, dataset_filter: str | None = None, *, include_listening: bool = False) -> str:
    lines: list[str] = []
    model_labels = {model.sig: model.name or model.label for model in catalog.models if model.name or model.label}

    def model_ref(sig: str | None, label: str | None = None) -> str:
        return label or _display_model_ref(sig, model_labels) or "?"

    def wrapped_detail(label: str, value: str | None) -> None:
        if not value:
            return
        wrapped = fill(value, width=104, subsequent_indent="  ")
        lines.append(f"  {label}: {wrapped}")

    # Models
    models = catalog.models
    if dataset_filter:
        dataset_query = dataset_filter.lower()
        models = tuple(
            m for m in models if dataset_query in m.workspace.lower() or (m.label and dataset_query in m.label.lower())
        )

    if models:
        lines.append("=== Models ===")
        header = f"{'model':<38} {'sig':<10} {'status':<8} {'ws':<28} {'ep':<5} {'#':<5}"
        lines.append(header)
        lines.append("-" * len(header))
        for m in models:
            status = "RUNNING" if m.running else ""
            label = (m.name or m.label or "")[:37]
            ws = m.workspace[:27]
            ep = str(m.epochs) if m.epochs is not None else "?"
            tc = str(m.train_count) if m.train_count is not None else "?"
            lines.append(f"{(label or m.sig)[:37]:<38} {m.sig:<10} {status:<8} {ws:<28} {ep:<5} {tc:<5}")
        lines.append("")

    # Merges
    if catalog.merges:
        lines.append("=== Merged Checkpoints ===")
        header = f"{'method':<8} {'alpha':<7} {'source_a':<22} {'source_b':<22} {'size':<8}"
        lines.append(header)
        lines.append("-" * len(header))
        for m in catalog.merges:
            mb = f"{m.output_size_mb:.0f}M" if m.output_size_mb else "?"
            lines.append(
                f"{m.method:<8} {m.alpha:<7.1f} {model_ref(m.source_a_sig, m.source_a_label)[:21]:<22} "
                f"{model_ref(m.source_b_sig, m.source_b_label)[:21]:<22} {mb:<8}"
            )
        lines.append("")

    if catalog.perturbations:
        lines.append("=== Perturbed Checkpoints ===")
        header = f"{'scope':<22} {'sigma':<10} {'seed':<8} {'source':<22} {'size':<8}"
        lines.append(header)
        lines.append("-" * len(header))
        for p in catalog.perturbations:
            mb = f"{p.output_size_mb:.0f}M" if p.output_size_mb else "?"
            seed = str(p.seed) if p.seed is not None else "?"
            lines.append(
                f"{p.layer_scope:<22} {p.sigma:<10.4g} {seed:<8} "
                f"{model_ref(p.source_sig, p.source_label)[:21]:<22} {mb:<8}"
            )
        lines.append("")

    # Generations
    if catalog.generations:
        lines.append("=== Generation Runs ===")
        if include_listening:
            header = (
                f"{'run_id':<17} {'mode':<14} {'prefix':<35} {'model':<25} "
                f"{'dur':<6} {'n':<4} {'listen':<40} {'lscore':<8}"
            )
        else:
            header = f"{'run_id':<17} {'mode':<14} {'prefix':<35} {'model':<25} {'dur':<6} {'n':<4}"
        lines.append(header)
        lines.append("-" * len(header))
        for g in catalog.generations:
            generation_model_ref = model_ref(g.model_sig, g.model_label) or g.model_workspace or "?"
            row = (
                f"{g.run_id:<17} {g.mode:<14} {g.prefix[:34]:<35} {str(generation_model_ref)[:24]:<25} "
                f"{g.duration_seconds:<6.0f} {g.num_samples:<4}"
            )
            if include_listening:
                listening_text = g.listening_summary or g.listening_labels or ""
                if not listening_text and g.listening_error_count:
                    listening_text = f"{g.listening_error_count} listening errors"
                listening_score = "" if g.listening_score is None else f"{g.listening_score:.3f}"
                row = f"{row} {listening_text[:39]:<40} {listening_score:<8}"
            lines.append(row)
        lines.append("")

    if catalog.experiments:
        lines.append("=== Experiments (ledger) ===")
        header = f"{'id':<28} {'status':<12} {'title':<42} {'xps':<18} {'sweeps':<8} {'streams':<8}"
        lines.append(header)
        lines.append("-" * len(header))
        for experiment in catalog.experiments:
            lines.append(
                f"{experiment.experiment_id[:27]:<28} {experiment.status:<12} {experiment.title[:41]:<42} "
                f"{','.join(experiment.model_labels or experiment.xp_sigs)[:17]:<18} {len(experiment.sweep_ids):<8} "
                f"{len(experiment.stream_ids):<8}"
            )
        lines.append("")
        for experiment in catalog.experiments:
            lines.append(f"[{experiment.status}] {experiment.title} ({experiment.experiment_id})")
            wrapped_detail("hypothesis", experiment.hypothesis)
            wrapped_detail("verdict", experiment.verdict)
            wrapped_detail("next", experiment.next_action)
            if experiment.annotations:
                annotation_text = ", ".join(f"{key}={value}" for key, value in sorted(experiment.annotations.items()))
                wrapped_detail("annotations", annotation_text)
            lines.append("")

    if catalog.sweeps:
        lines.append("=== Sweeps ===")
        header = (
            f"{'sweep_id':<36} {'mode':<14} {'profile':<20} {'model':<12} {'combos':<8} {'samples':<8} {'scored':<8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for sweep in catalog.sweeps:
            lines.append(
                f"{sweep.sweep_id[:35]:<36} {(sweep.mode or '')[:13]:<14} {(sweep.profile or '')[:19]:<20} "
                f"{model_ref(sweep.model_sig, sweep.model_label)[:11]:<12} {sweep.combo_count:<8} "
                f"{sweep.sample_count:<8} {sweep.scored_count:<8}"
            )
        lines.append("")

    if catalog.streams:
        lines.append("=== Streams ===")
        header = f"{'run_id':<18} {'started':<22} {'model':<12} {'rounds':<8} {'scored':<8} {'novelty':<8}"
        lines.append(header)
        lines.append("-" * len(header))
        for stream in catalog.streams:
            lines.append(
                f"{stream.run_id[:17]:<18} {(stream.started_at or '')[:21]:<22} "
                f"{','.join(stream.model_labels or stream.model_sigs)[:11]:<12} "
                f"{stream.rounds_written:<8} {stream.scored_rounds:<8} {stream.novelty_rounds:<8}"
            )
        lines.append("")

    if catalog.other_sessions:
        lines.append("=== Other sessions (duets, variations) ===")
        header = f"{'session':<40} {'kind':<11} {'model':<24} {'takes':<6}"
        lines.append(header)
        lines.append("-" * len(header))
        for entry in catalog.other_sessions:
            lines.append(
                f"{entry.session_id[:39]:<40} {entry.kind[:10]:<11} "
                f"{','.join(entry.model_labels or entry.model_sigs)[:23]:<24} {entry.take_count:<6}"
            )
        lines.append("")

    if catalog.portraits:
        lines.append("=== Portraits ===")
        header = f"{'title':<58} {'generated':<16} {'models':<24}"
        lines.append(header)
        lines.append("-" * len(header))
        for portrait in catalog.portraits:
            lines.append(
                f"{portrait.title[:57]:<58} {(portrait.generated_at or '')[:15]:<16} "
                f"{','.join(portrait.model_labels or portrait.model_sigs)[:23]:<24}"
            )
        lines.append("")

    if catalog.ledger_warnings:
        lines.append("=== Ledger Warnings ===")
        lines.extend(f"- {warning}" for warning in catalog.ledger_warnings)
        lines.append("")

    if not lines:
        lines.append("(no artifacts found)")
    lines.append(f"generated_at: {catalog.generated_at}")
    return "\n".join(lines)
