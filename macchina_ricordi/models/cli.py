from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..cli.common import MacchinaRicordiHelpFormatter, clip
from ..musicgen.runtime import load_musicgen_project_config
from .catalog import build_catalog, format_catalog_text
from .lineage import build_lineage_graph_for_sig, format_lineage_text, lineage_graph_to_json
from .record import (
    Model,
    ModelNotFound,
    RegistryError,
    dora_dir_or_default,
    resolve_model,
    set_model_name,
    xp_folder_for_sig,
)
from .zoo import build_zoo

REFERENCE_HELP = (
    "a name, sig (or unique prefix), Dora label, bundle name/path, or XP/checkpoint path, optionally with "
    "@best, @last, @final or @<epoch>"  # a bundle has one snapshot, so it takes no @selector
)


def cmd_model_list(args: argparse.Namespace) -> int:
    zoo = build_zoo(include_sessions=False, include_experiments=False, include_derived=False)
    models = list(zoo.models)
    if args.dataset:
        models = [m for m in models if m.dataset and args.dataset.lower() in m.dataset.lower()]
    if args.recipe:
        models = [m for m in models if m.recipe == args.recipe]
    if args.named:
        models = [m for m in models if m.name]
    if args.json:
        print(json.dumps([{**m.to_dict(), "children": list(zoo.children(m.sig))} for m in models], indent=2))
        return 0
    rows = [
        (
            ("bundle " if m.kind == "bundle" else "") + m.sig,
            m.name or "",
            clip(m.label or "", 34),
            m.recipe or "",
            clip(m.dataset or "", 30),
            (f"{m.epochs_done}/{m.epochs_planned or '?'}" if m.kind == "xp" else "—"),
            m.crops,
            (
                f"{m.best_valid_ce:.3f}@{m.best_epoch}"
                if m.best_valid_ce is not None
                else (f"@{m.selection_text}" if m.kind == "bundle" else "")
            ),
            str(len(m.checkpoints)),
        )
        for m in models
    ]
    header = ("sig", "name", "label", "recipe", "dataset", "epochs", "crops", "best ce", "ckpts")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    n_bundles = sum(1 for m in models if m.kind == "bundle")
    suffix = f" ({n_bundles} bundle)" if n_bundles else ""
    print(f"\n{len(rows)} models{suffix}")
    return 0


def _crops_text(model: Model) -> str:
    pool = model.pool
    if pool is None:
        return "fresh every step"
    if not pool.get("crops"):
        return f"from a token pool whose record is missing ({pool['path']})"
    return f"from a token pool: {pool['crops']:,} crops, each used {pool['repeats']:g} times ({pool['path']})"


def cmd_model_show(args: argparse.Namespace) -> int:
    try:
        ref = resolve_model(args.ref)
    except ModelNotFound as exc:
        return _fail(exc)
    if ref.model is None:
        payload = {"checkpoint": str(ref.checkpoint), "model": None}
        print(json.dumps(payload, indent=2) if args.json else f"checkpoint: {ref.checkpoint}\n(not inside an XP)")
        return 0
    family = build_zoo().family(ref.model.sig)
    if args.json:
        payload = {**ref.model.to_dict(), **family}
        if ref.checkpoint is not None:
            payload["selected_checkpoint"] = {"path": str(ref.checkpoint), "epoch": ref.epoch}
        print(json.dumps(payload, indent=2))
        return 0
    print(format_model(ref.model, family))
    if ref.checkpoint is not None:
        print(f"selected:     {ref.checkpoint}" + (f" (epoch {ref.epoch})" if ref.epoch is not None else ""))
    return 0


def cmd_model_name(args: argparse.Namespace) -> int:
    if (args.name is None) == (not args.clear):
        return _fail("give a name, or --clear to remove the current one")
    try:
        ref = resolve_model(args.ref)
        if ref.model is None:
            raise ModelNotFound(f"{args.ref!r} is a checkpoint file, not a model")
        entry = set_model_name(ref.model.sig, None if args.clear else args.name)
    except (ModelNotFound, RegistryError) as exc:
        return _fail(exc)
    if entry.name:
        print(f"{ref.model.sig} is now {entry.name!r} (musicgen/models.toml)")
    else:
        print(f"{ref.model.sig} has no name now (musicgen/models.toml)")
    return 0


def cmd_model_export(args: argparse.Namespace) -> int:
    from .export import DEFAULT_BOX_DIR, DEFAULT_BUNDLE_DIR

    out_dir = args.out_dir or (DEFAULT_BUNDLE_DIR if args.bundle else DEFAULT_BOX_DIR)
    if args.bundle:
        from .export import plan_bundle_export, run_bundle_export

        try:
            plan = plan_bundle_export(
                args.ref,
                out_dir=out_dir,
                name=args.name,
                weights_state=args.weights_state,
                force=args.force,
            )
        except ModelNotFound as exc:
            return _fail(exc)
        if args.dry_run:
            if args.json:
                print(json.dumps(plan.to_dict(), indent=2))
                return 0
            print(f"bundle:   {plan.bundle_dir}")
            print(f"model:    {plan.name} ({plan.checkpoint})")
            print(f"state:    {plan.weights_state}")
            print("command:  " + " ".join(plan.helper_command))
            for missing in plan.missing:
                print(f"missing:  {missing}")
            print("\nprovenance:  " + json.dumps(plan.provenance, indent=2))
            return 0
        try:
            status = run_bundle_export(plan)
        except FileNotFoundError as exc:
            return _fail(exc)
        if args.json:
            print(json.dumps(status, indent=2))
        elif status["status"] == "completed":
            print(f"wrote {plan.bundle_dir}  ({status['weights_path']})")
        else:
            print(
                f"export {status.get('failed_step')} failed: {status.get('error') or status.get('returncode')}",
                file=sys.stderr,
            )
        return 0 if status["status"] == "completed" else 1

    from .export import plan_box_export, run_box_export

    try:
        plan = plan_box_export(
            args.ref,
            out_dir=out_dir,
            name=args.name,
            note=args.note,
            pack=not args.no_pack,
            probe=not args.no_probe,
            weights_state=args.weights_state,
        )
    except ModelNotFound as exc:
        return _fail(exc)
    if args.dry_run:
        if args.json:
            print(json.dumps(plan.to_dict(), indent=2))
            return 0
        print(f"box:      {plan.box_path}" + ("  (not packed: --no-pack)" if plan.pack_command is None else ""))
        print(f"model:    {plan.card['name']} ({plan.sig}), {Path(plan.checkpoint).name}")
        print(f"state:    {args.weights_state}")
        print(f"card:     {plan.card_path}")
        print("export:   " + " ".join(plan.export_command))
        if plan.pack_command:
            print("pack:     " + " ".join(plan.pack_command))
        for missing in plan.missing:
            print(f"missing:  {missing}")
        print("\n" + json.dumps(plan.card, indent=2))
        return 0
    try:
        status = run_box_export(plan)
    except FileNotFoundError as exc:
        return _fail(exc)
    if args.json:
        print(json.dumps(status, indent=2))
    elif status["status"] == "completed":
        print(f"wrote {plan.box_path if plan.pack_command else plan.source_path}  (status: {plan.status_path})")
    else:
        print(f"{status.get('failed_step')} failed; see {plan.status_path}", file=sys.stderr)
    return 0 if status["status"] == "completed" else 1


def format_model(model: Model, family: dict | None = None) -> str:
    shape = model.shape
    shape_text = (
        f"dim {shape.get('dim')} · {shape.get('num_layers')} layers · {shape.get('num_heads')} heads"
        f" · n_q {shape.get('n_q')}"
        if shape
        else "unknown"
    )
    best = f"{model.best_valid_ce:.4f} at epoch {model.best_epoch}" if model.best_valid_ce is not None else "none"
    checkpoints = ", ".join(
        "checkpoint.th" + (f" (epoch {c.epoch})" if c.epoch else "") if c.final else str(c.epoch)
        for c in model.checkpoints
    )
    parents = ", ".join(model.parents) if model.parents else "none"
    lines = [
        f"sig:          {model.sig}",
        f"kind:         {model.kind}" + (f"  ({model.selection} snapshot)" if model.selection else ""),
        f"name:         {model.name or f'(none — `model name {model.sig} <name>`)'}",
        f"title:        {model.title or ''}",
        f"label:        {model.label or ''}",
        f"folder:       {model.folder}",
        f"recipe:       {model.recipe or 'unknown'}",
        f"dataset:      {model.dataset or 'unknown'}",
        f"shape:        {shape_text}" + (" · text-conditioned" if model.conditioned else ""),
        f"segment:      {model.segment_duration:g} s" if model.segment_duration else "segment:      unknown",
        f"codec:        {model.codec or 'unknown'}",
    ]
    if model.kind == "bundle":
        lines += [
            f"weights-file: {Path(model.checkpoints[0].path).name}",
            f"crops:        {_crops_text(model)}",
            f"checkpoints:  one selected snapshot ({model.selection or 'model'})",
        ]
        return "\n".join(lines)
    lines += [
        f"crops:        {_crops_text(model)}",
        f"epochs:       {model.epochs_done} done / {model.epochs_planned or '?'} planned",
        f"best valid:   {best}",
        f"checkpoints:  {checkpoints or 'none on disk'}",
        f"started from: {parents}" + (f" ({model.origin})" if model.origin else ""),
    ]
    if family is not None:
        kinds: dict[str, int] = {}
        for session in family["sessions"]:
            kinds[session["kind"]] = kinds.get(session["kind"], 0) + 1
        more = len(family["descendants"]) > len(family["children"])
        lines += [
            f"children:     {', '.join(family['children']) or 'none'}"
            + (f" ({len(family['descendants'])} descendants)" if more else ""),
            "sessions:     " + (", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items())) or "none"),
        ]
        if family["experiments"]:
            lines.append(f"experiments:  {', '.join(family['experiments'])}")
    if model.measured:
        lines.append("measured:     " + ", ".join(f"{k} {v}" for k, v in model.measured.items()))
    if model.notes:
        lines.append(f"notes:        {model.notes}")
    lines += [f"warning:      {warning}" for warning in model.warnings]
    return "\n".join(lines)


def register_model_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    model = subparsers.add_parser(
        "model",
        help="List, inspect, name and export models (trained XPs and portable bundles)",
        description=(
            "A model is a trained Dora XP or a portable bundle (models/<name>, see model export --bundle). "
            "Commands that take a model accept one reference: "
            f"{REFERENCE_HELP}. Names live in musicgen/models.toml."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli model list
  python -m macchina_ricordi.cli model list --dataset black_zone --json
  python -m macchina_ricordi.cli model show 4ae63477
  python -m macchina_ricordi.cli model name 4ae63477 midwife
  python -m macchina_ricordi.cli model show midwife@best
  python -m macchina_ricordi.cli model export midwife --bundle --dry-run
  python -m macchina_ricordi.cli play generate midwife@best""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    model_subparsers = model.add_subparsers(dest="model_command", required=True)

    listing = model_subparsers.add_parser(
        "list", help="Every model, one line each", formatter_class=MacchinaRicordiHelpFormatter
    )
    listing.add_argument("--dataset", default=None, help="Only models trained on a dataset containing this text")
    listing.add_argument("--recipe", default=None, help="Only models trained with this recipe (profile)")
    listing.add_argument("--named", action="store_true", help="Only models that have a name")
    listing.add_argument("--json", action="store_true", help="Machine-readable output: one object per model")
    listing.set_defaults(func=cmd_model_list)

    show = model_subparsers.add_parser(
        "show", help="Everything known about one model", formatter_class=MacchinaRicordiHelpFormatter
    )
    show.add_argument("ref", help=f"The model: {REFERENCE_HELP}")
    show.add_argument("--json", action="store_true", help="Machine-readable output")
    show.set_defaults(func=cmd_model_show)

    name = model_subparsers.add_parser(
        "name",
        help="Give a model a short name usable anywhere a model is expected",
        description="Writes musicgen/models.toml (Git-tracked). Names are lowercase and start with a letter.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    name.add_argument("ref", help=f"The model: {REFERENCE_HELP}")
    name.add_argument("name", nargs="?", default=None, help="The new name, e.g. midwife or deep-staircase")
    name.add_argument("--clear", action="store_true", help="Remove the model's name")
    name.set_defaults(func=cmd_model_name)

    export = model_subparsers.add_parser(
        "export",
        help="Export a model as a .nrbox for the appliance, or a portable safetensors model bundle",
        description=(
            "Two portable forms. --box resolves the model, writes its record as the card, then runs appliance.export "
            "in .venv-musicgen (refused if the checkpoint's architecture is not the record's) and appliance.pack in "
            ".venv-appliance. --bundle writes a torch-free safetensors + JSON bundle under models/<name>/ via "
            "scripts/export_musicgen_bundle.py in the AudioCraft runtime. --weights-state picks which snapshot: "
            "model (raw, newest), best (lowest valid ce) or ema. Both run on the CPU and take minutes; --dry-run "
            "prints the plan and writes nothing."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli model export midwife@best --box --dry-run
  python -m macchina_ricordi.cli model export midwife --bundle --dry-run
  python -m macchina_ricordi.cli model export 31f7067b --bundle --out-dir models --weights-state ema --force""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    export.add_argument("ref", help=f"The model: {REFERENCE_HELP} (snapshot picked by --weights-state)")
    export.add_argument("--bundle", action="store_true", help="Export a portable safetensors+JSON model bundle")
    export.add_argument("--box", action="store_true", help="A .nrbox for the appliance (the default export form)")
    export.add_argument(
        "--weights-state",
        choices=("model", "ema", "best"),
        default="model",
        help="Which training snapshot to export (default: model, the newest raw checkpoint)",
    )
    export.add_argument(
        "--out-dir",
        default=None,
        help="Where the bundle/box goes (bundle default: models; box default: boxes, source in <dir>/src)",
    )
    export.add_argument("--name", default=None, help="Bundle/box name (default: the model's name and epoch)")
    export.add_argument("--note", default="", help="A line shown with the model in the appliance (box only)")
    export.add_argument("--force", action="store_true", help="Overwrite an existing bundle (bundle only)")
    export.add_argument("--no-pack", action="store_true", help="Stop after the .nrsrc source (box only)")
    export.add_argument("--no-probe", action="store_true", help="Pack without probing for the silent attractor (box)")
    export.add_argument("--dry-run", action="store_true", help="Print the plan and the manifest/card; run nothing")
    export.add_argument("--json", action="store_true", help="Print the plan (and outcome) as JSON")
    export.set_defaults(func=cmd_model_export)

    merge = model_subparsers.add_parser(
        "merge",
        help="Interpolate two local MusicGen checkpoint files into a new checkpoint",
        description="Merge two local checkpoint files by interpolating compatible tensors.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    merge.add_argument("checkpoint_a", help="First source: a checkpoint path or a model reference such as midwife@best")
    merge.add_argument("checkpoint_b", help="Second source: a checkpoint path or a model reference")
    merge.add_argument("output_checkpoint", help="Destination checkpoint.th path")
    merge.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation amount: 0 keeps checkpoint A, 1 keeps checkpoint B (default: 0.5)",
    )
    merge.add_argument(
        "--method",
        choices=("linear", "slerp"),
        default="linear",
        help="Interpolation method for floating tensors (default: linear)",
    )
    merge.add_argument(
        "--musicgen-python",
        default=None,
        help=(
            "Python executable whose environment contains torch; defaults to the repo-managed .venv-musicgen runtime"
        ),
    )
    merge.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination checkpoint if it already exists",
    )
    merge.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the merge helper command without writing a checkpoint",
    )
    merge.set_defaults(func=cmd_model_merge)

    perturb = model_subparsers.add_parser(
        "perturb",
        help="Add controlled Gaussian noise to a local MusicGen checkpoint file",
        description="Create a perturbed checkpoint by adding controlled Gaussian noise to selected tensors.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    perturb.add_argument(
        "source_checkpoint",
        help="Source: a checkpoint path, a model reference (midwife@best), or a model bundle directory (models/<name>)",
    )
    perturb.add_argument("output_checkpoint", help="Destination checkpoint.th path")
    perturb.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Gaussian noise scale to add to selected floating tensors",
    )
    perturb.add_argument(
        "--layer-scope",
        default="all",
        help=(
            "Layer subset to perturb: all, attention, attention-projections, lm-head, "
            "or contains:<checkpoint-key-fragment> (default: all)"
        ),
    )
    perturb.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible noise (default: 0)",
    )
    perturb.add_argument(
        "--musicgen-python",
        default=None,
        help=(
            "Python executable whose environment contains torch; defaults to the repo-managed .venv-musicgen runtime"
        ),
    )
    perturb.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination checkpoint if it already exists",
    )
    perturb.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the perturb helper command without writing a checkpoint",
    )
    perturb.set_defaults(func=cmd_model_perturb)

    score_curve = model_subparsers.add_parser(
        "curve",
        help="Render a per-epoch arrhythmicity score curve from sidecar_cpu/scores.jsonl",
        description="Summarize sidecar CPU scores over training epochs as a table and sparkline.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_curve.add_argument("sig", help="Dora experiment signature whose scores.jsonl to read")
    score_curve.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the AudioCraft dora directory (defaults to AUDIOCRAFT_DORA_DIR or the repo-managed .dora)",
    )
    score_curve.add_argument(
        "--scores-path",
        default=None,
        help=("Override the scores JSONL path (default: <dora-dir>/xps/<sig>/sidecar_cpu/scores.jsonl)"),
    )
    score_curve.add_argument(
        "--metric",
        default="score",
        help="RewardComponents field name to chart (default: score)",
    )
    score_curve.add_argument(
        "--aggregate",
        choices=("mean", "median", "p90", "max", "min"),
        default="median",
        help="Per-epoch aggregation applied to the metric (default: median)",
    )
    score_curve.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary instead of the table + sparkline",
    )
    score_curve.add_argument(
        "--no-sparkline",
        action="store_true",
        help="Suppress the trailing Unicode sparkline in text output",
    )
    score_curve.set_defaults(func=cmd_model_curve)

    lineage = model_subparsers.add_parser(
        "lineage",
        help="Trace the warm-start, merge, and perturb ancestry for an XP signature",
        description="Trace warm-start, merge, and perturb ancestry for one Dora XP signature.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    lineage.add_argument("sig", help="The model whose ancestry should be traced: a sig, name or label")
    lineage.add_argument(
        "--dora-dir",
        default=None,
        help="Path to the .dora directory (defaults to the repo-managed .dora)",
    )
    lineage.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum ancestry depth to expand (default: 20)",
    )
    lineage.add_argument(
        "--json",
        action="store_true",
        help="Emit the lineage graph as machine-readable JSON",
    )
    lineage.set_defaults(func=cmd_model_lineage)

    catalog_parser = model_subparsers.add_parser(
        "catalog",
        help="List trained models, merges, and generation runs (the GUI is the browsing view)",
        description="Browse local MusicGen models, checkpoint derivatives, and generation runs.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    catalog_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    catalog_parser.add_argument("--dataset", default=None, help="Filter by workspace name (substring match)")
    catalog_parser.add_argument("--listen", action="store_true", help="Display existing listening sidecar summaries")
    catalog_parser.set_defaults(func=cmd_model_catalog)


def _fail(message: object) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def run_merge(
    *,
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    output_checkpoint: str | Path,
    method: str,
    alpha: float,
    musicgen_python: str | None,
    force: bool,
    dry_run: bool,
) -> int:
    from ..models.merge import execute_checkpoint_merge, plan_checkpoint_merge
    from ..musicgen.shared import format_command

    plan = plan_checkpoint_merge(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        output_checkpoint=output_checkpoint,
        method=method,
        alpha=alpha,
        musicgen_python=musicgen_python,
        force=force,
        dry_run=dry_run,
    )
    print(f"checkpoint_a: {plan.checkpoint_a}")
    print(f"checkpoint_b: {plan.checkpoint_b}")
    print(f"output_checkpoint: {plan.output_checkpoint}")
    print(f"merged_from: {plan.sidecar_path}")
    print(f"method: {plan.method}")
    print(f"alpha: {plan.alpha}")
    print(f"musicgen_python: {plan.musicgen_python}")
    print(f"command: {format_command(plan.command)}")

    result = execute_checkpoint_merge(plan)
    if result.dry_run:
        print("dry_run: command validated but checkpoint was not written")
    else:
        print(f"checkpoint: {result.output_checkpoint}")
        print(f"merged_from: {result.sidecar_path}")
    return 0


def run_perturb(
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    sigma: float,
    layer_scope: str,
    seed: int,
    musicgen_python: str | None,
    force: bool,
    dry_run: bool,
) -> int:
    from ..models.perturb import execute_checkpoint_perturb, plan_checkpoint_perturb
    from ..musicgen.shared import format_command

    plan = plan_checkpoint_perturb(
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        sigma=sigma,
        layer_scope=layer_scope,
        seed=seed,
        musicgen_python=musicgen_python,
        force=force,
        dry_run=dry_run,
    )
    print(f"source_checkpoint: {plan.source_checkpoint}")
    print(f"output_checkpoint: {plan.output_checkpoint}")
    print(f"perturbed_from: {plan.sidecar_path}")
    print(f"sigma: {plan.sigma}")
    print(f"layer_scope: {plan.layer_scope}")
    print(f"seed: {plan.seed}")
    print(f"musicgen_python: {plan.musicgen_python}")
    print(f"command: {format_command(plan.command)}")

    result = execute_checkpoint_perturb(plan)
    if result.dry_run:
        print("dry_run: command validated but checkpoint was not written")
    else:
        print(f"checkpoint: {result.output_checkpoint}")
        print(f"perturbed_from: {result.sidecar_path}")
    return 0


def resolve_sig_argument(args: argparse.Namespace) -> None:
    """Let ``sig`` positionals take any model reference; a sig with no folder (a deleted XP) passes through."""
    from ..models.record import ModelNotFound, resolve_sig

    try:
        args.sig = resolve_sig(args.sig)
    except ModelNotFound as exc:
        if not re.fullmatch(r"[0-9a-f]{8}", str(args.sig)):
            raise SystemExit(f"error: {exc}") from None


def resolve_checkpoint_arguments(args: argparse.Namespace, *names: str) -> None:
    """Let checkpoint positionals take a model reference (``midwife@best``); files pass through."""
    from ..models.record import ModelNotFound, resolve_checkpoint_path

    for name in names:
        try:
            setattr(args, name, resolve_checkpoint_path(getattr(args, name)))
        except ModelNotFound as exc:
            raise SystemExit(f"error: {name}: {exc}") from None


def cmd_model_merge(args: argparse.Namespace) -> int:
    resolve_checkpoint_arguments(args, "checkpoint_a", "checkpoint_b")
    return run_merge(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        output_checkpoint=args.output_checkpoint,
        method=args.method,
        alpha=args.alpha,
        musicgen_python=args.musicgen_python,
        force=args.force,
        dry_run=args.dry_run,
    )


def _perturb_source(text: str) -> str:
    """A perturb source: a checkpoint path, a model reference, or a bundle directory (which stays a directory).

    A bundle has no torch checkpoint to read tensors from, but the perturb helper accepts the bundle directory
    as its source, so the directory is preserved instead of resolving to the safetensors file inside it.
    """
    from pathlib import Path

    if Path(text).expanduser().is_file():
        return text
    try:
        ref = resolve_model(text)
    except ModelNotFound:
        return text
    if ref.model is not None and ref.model.kind == "bundle":
        return str(ref.model.folder)
    if ref.checkpoint is not None:
        return str(ref.checkpoint)
    return text


def cmd_model_perturb(args: argparse.Namespace) -> int:
    args.source_checkpoint = _perturb_source(args.source_checkpoint)
    return run_perturb(
        source_checkpoint=args.source_checkpoint,
        output_checkpoint=args.output_checkpoint,
        sigma=args.sigma,
        layer_scope=args.layer_scope,
        seed=args.seed,
        musicgen_python=args.musicgen_python,
        force=args.force,
        dry_run=args.dry_run,
    )


def cmd_model_curve(args: argparse.Namespace) -> int:
    resolve_sig_argument(args)
    from ..models.curve import (
        aggregate_by_epoch,
        format_score_curve_json,
        format_score_curve_table,
        load_score_rows,
        summarize_score_rows,
    )

    if args.scores_path:
        scores_path = Path(args.scores_path).expanduser().resolve()
    else:
        xp_folder = xp_folder_for_sig(
            args.sig,
            dora_dir=dora_dir_or_default(args.dora_dir) if args.dora_dir else None,
        )
        scores_path = xp_folder / "sidecar_cpu" / "scores.jsonl"

    rows = load_score_rows(scores_path, metric=args.metric)
    n_total, n_failed = summarize_score_rows(rows)
    points = aggregate_by_epoch(rows, aggregate=args.aggregate)

    if args.json:
        print(
            format_score_curve_json(
                sig=args.sig,
                metric=args.metric,
                aggregate=args.aggregate,
                points=points,
                n_total=n_total,
                n_failed=n_failed,
            )
        )
    else:
        print(f"scores_path: {scores_path}")
        print(
            format_score_curve_table(
                sig=args.sig,
                metric=args.metric,
                aggregate=args.aggregate,
                points=points,
                n_total=n_total,
                n_failed=n_failed,
                include_sparkline=not args.no_sparkline,
            )
        )
    return 0


def cmd_model_catalog(args: argparse.Namespace) -> int:
    catalog = build_catalog()

    if args.json:
        import json as _json

        def _convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                from dataclasses import asdict

                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, tuple):
                return [_convert(v) for v in obj]
            return obj

        print(_json.dumps(_convert(catalog), indent=2, default=str))
        return 0

    print(format_catalog_text(catalog, dataset_filter=args.dataset, include_listening=args.listen))
    return 0


def cmd_model_lineage(args: argparse.Namespace) -> int:
    resolve_sig_argument(args)
    config = load_musicgen_project_config()
    dora_dir = Path(args.dora_dir).expanduser().resolve() if args.dora_dir else config.dora_dir
    try:
        graph = build_lineage_graph_for_sig(
            args.sig,
            dora_dir=dora_dir,
            repo_root=config.repo_root,
            max_depth=args.max_depth,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(lineage_graph_to_json(graph))
    else:
        print(format_lineage_text(graph))
    return 0


def apply_model_reference(
    args: argparse.Namespace,
    *,
    xp_attr: str = "xp_path",
    checkpoint_attr: str | None = "checkpoint",
    checkpoint_path_attr: str | None = "checkpoint_path",
    positional_attr: str | None = None,
) -> None:
    """Turn a model reference (``midwife@best``, a sig, a label...) into the folder/checkpoint flags a command reads.

    An existing XP folder passes through untouched, so every older invocation keeps working.
    """
    from pathlib import Path

    from ..models.record import ModelNotFound, resolve_model, split_selector

    positional = getattr(args, positional_attr, None) if positional_attr else None
    flagged = getattr(args, xp_attr, None)
    if positional and flagged:
        raise SystemExit("error: give the model either as an argument or with --xp-path, not both")
    text = positional or flagged
    if not text:
        return
    base = split_selector(text)[0]
    if Path(text).expanduser().is_dir() or ("/" in base and not Path(base).expanduser().exists()):
        # an existing folder, or a path the command itself will report as missing, exactly as before
        setattr(args, xp_attr, text)
        return
    try:
        ref = resolve_model(text)
    except ModelNotFound as exc:
        raise SystemExit(f"error: {exc}") from None
    if ref.model is not None and ref.model.kind == "bundle":
        # a portable bundle: route its directory into the existing xp_path flag so the shared runtime load detects
        # it by path; no step registers it as a Dora XP, and it holds one snapshot so there is no checkpoint flag
        setattr(args, xp_attr, str(ref.model.folder))
        return
    if ref.model is None:
        if checkpoint_path_attr is None:
            raise SystemExit(f"error: {text!r} is a checkpoint file outside .dora/xps; this command needs a model")
        setattr(args, xp_attr, None)
        setattr(args, checkpoint_path_attr, str(ref.checkpoint))
        return
    setattr(args, xp_attr, str(ref.model.folder))
    if ref.checkpoint is None:
        return
    if checkpoint_path_attr is not None:
        setattr(args, checkpoint_path_attr, str(ref.checkpoint))
    elif checkpoint_attr is not None:
        setattr(args, checkpoint_attr, ref.checkpoint_selector)
    else:
        raise SystemExit(f"error: {text!r}: this command takes a whole model; drop the @selector")


def model_folder_or_text(text: str) -> str:
    from pathlib import Path

    from ..models.record import ModelNotFound, resolve_model

    if Path(text).expanduser().is_dir():
        return text
    try:
        ref = resolve_model(text)
    except ModelNotFound:
        return text
    return str(ref.model.folder) if ref.model is not None else text
