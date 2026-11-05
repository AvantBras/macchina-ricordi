"""`data`: build, list, show, audit and weight datasets."""

from __future__ import annotations

import argparse
import json
from typing import Any

from ..audio_files import DEFAULT_EXTENSIONS
from ..augmentation.core import DEFAULT_SPEED_MODE, SPEED_MODES
from ..cli.common import MacchinaRicordiHelpFormatter
from ..musicgen.config import (
    musicgen_default_split_seed,
    musicgen_default_valid_fraction,
)


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from ..data.prepare import prepare_musicgen_dataset
    from ..musicgen.shared import format_command

    prepared = prepare_musicgen_dataset(
        input_dir=args.input_dir,
        workspace_dir=args.workspace_dir,
        musicgen_python=args.musicgen_python,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        extensions=args.extensions,
        max_silence_fraction=args.max_silence_fraction,
        silence_frame_db=args.silence_frame_db,
        speed_ratios=args.speed_ratios,
        speed_mode=args.speed_mode,
        overwrite=args.overwrite,
    )

    print(f"workspace_dir: {prepared.workspace_dir}")
    print(f"train_count: {prepared.train_count}")
    if prepared.train_speed_variant_count:
        print(f"train_speed_variants: {prepared.train_speed_variant_count} ({args.speed_mode})")
    print(f"valid_count: {prepared.valid_count}")
    print(f"train_manifest: {prepared.train_manifest}")
    print(f"valid_manifest: {prepared.valid_manifest}")
    print(f"dataset_fingerprint: {prepared.dataset_fingerprint_path}")
    for index, command in enumerate(prepared.commands, start=1):
        print(f"manifest_command_{index}: {format_command(command)}")
    return 0


def cmd_data_rebalance(args: argparse.Namespace) -> int:
    import sys
    from pathlib import Path

    from ..data.prepare import resolve_workspace_dir, workspace_dataset_name
    from ..data.weights import (
        clear_manifest_weights,
        load_rebalance_plan,
        plan_rebalance,
        save_rebalance_plan,
        save_weights_overlay,
    )
    from ..musicgen.config import get_repo_root

    workspace = resolve_workspace_dir(args.workspace_dir)
    dataset_name = args.dataset_name or workspace_dataset_name(workspace)

    if args.reset:
        manifest = workspace / "egs" / dataset_name / "train" / "data.jsonl.gz"
        clear_manifest_weights(manifest)
        print(f"weights cleared: {manifest}")
        return 0
    if not (args.save_as or args.dry_run or args.plan_out):
        print("error: say where the weights go: --save-as NAME (or --dry-run / --plan-out PATH)", file=sys.stderr)
        return 1

    if args.apply_plan:
        plan = load_rebalance_plan(Path(args.apply_plan))
    else:
        plan = plan_rebalance(
            workspace_dir=workspace,
            dataset_name=dataset_name,
            gens_dirs=[Path(d) for d in args.gens_dir] if args.gens_dir else None,
            k=args.k,
            alpha=args.alpha,
            cap=args.cap,
            seed=args.seed,
            progress=(
                (lambda stage, done, total: print(f"progress: {stage} {done}/{total}", flush=True))
                if args.plan_out
                else None
            ),
        )
        print(f"mode: {plan.mode} (k={plan.k}, alpha={plan.alpha}, cap={plan.cap})")
        _print_cluster_table(plan.cluster_table)

    if args.plan_out:
        save_rebalance_plan(plan, Path(args.plan_out))
        print(f"plan: {args.plan_out} (weights_digest {plan.weights_digest}); save it with --apply-plan --save-as NAME")
    if args.save_as:
        try:
            overlay = save_weights_overlay(plan, name=args.save_as, repo_root=get_repo_root())
        except (ValueError, FileExistsError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"weights: {overlay.name} ({overlay.weights_digest}) -> {overlay.folder}")
        print(f"train with: train run --workspace-dir {args.workspace_dir} --weights {overlay.name}")
    elif args.dry_run:
        print("dry-run: nothing written")
    return 0


def cmd_data_mix(args: argparse.Namespace) -> int:
    import sys

    from ..data.prepare import resolve_workspace_dir, workspace_dataset_name
    from ..data.weights import plan_folder_mix, save_weights_overlay
    from ..musicgen.config import get_repo_root

    workspace = resolve_workspace_dir(args.workspace_dir)
    dataset_name = args.dataset_name or workspace_dataset_name(workspace)
    if not (args.save_as or args.dry_run):
        print("error: say where the mix goes: --save-as NAME (or --dry-run)", file=sys.stderr)
        return 1
    try:
        plan = plan_folder_mix(workspace / "egs" / dataset_name / "train" / "data.jsonl.gz", _parse_mix(args.shares))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_folder_table(plan.cluster_table)
    if not args.save_as:
        print("dry-run: nothing written")
        return 0
    try:
        overlay = save_weights_overlay(plan, name=args.save_as, repo_root=get_repo_root())
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"weights: {overlay.name} ({overlay.weights_digest}) -> {overlay.folder}")
    print(f"train with: train run --workspace-dir {args.workspace_dir} --weights {overlay.name}")
    return 0


def _parse_mix(items: list[str]) -> dict[str, float]:
    mix: dict[str, float] = {}
    for item in items:
        folder, separator, share = item.partition("=")
        if not separator or not folder:
            raise ValueError(f"{item!r}: write FOLDER=SHARE, for example album_a=3")
        if folder in mix:
            raise ValueError(f"{folder!r} is given twice")
        try:
            mix[folder] = float(share)
        except ValueError:
            raise ValueError(f"{item!r}: the share must be a number") from None
    return mix


def _print_folder_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'folder':<24} {'segs':>6} {'share':>7} {'weight':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['folder']:<24} {row['segments']:>6} {row['share']:>7.1%} {row['weight']:>7.3f}")


def _print_cluster_table(cluster_table: list[dict[str, Any]]) -> None:
    has_gens = any("gen_share" in row for row in cluster_table)
    gens_head = f" {'gens':>7}" if has_gens else ""
    header = f"{'cluster':>7} {'segs':>6} {'share':>7}{gens_head} {'weight':>7}  top groups"
    print(header)
    print("-" * len(header))
    for row in cluster_table:
        gens_col = f" {row['gen_share']:>7.1%}" if has_gens else ""
        print(
            f"{row['cluster']:>7} {row['segments']:>6} {row['share']:>7.1%}"
            f"{gens_col} {row['weight']:>7.3f}  {', '.join(row['top_groups'])}"
        )


def cmd_data_weights(args: argparse.Namespace) -> int:
    import sys

    from ..data.prepare import resolve_workspace_dir
    from ..data.weights import FOLDER_MIX_MODE, delete_weights_overlay, list_weights_overlays
    from ..musicgen.config import get_repo_root

    workspace = resolve_workspace_dir(args.workspace_dir)
    if args.weights_command == "delete":
        try:
            folder = delete_weights_overlay(get_repo_root(), workspace.name, args.name)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"deleted: {folder}")
        return 0
    overlays = list_weights_overlays(get_repo_root(), workspace.name)
    if args.json:
        print(json.dumps([{**o.summary(), "stale": o.is_stale()} for o in overlays], indent=2))
        return 0
    if not overlays:
        print(f"no saved weights for {workspace.name} (make some with `data rebalance --save-as NAME` or `data mix`)")
        return 0
    for o in overlays:
        stale = "  STALE: the workspace manifest changed since" if o.is_stale() else ""
        settings = (
            " ".join(f"{folder}={share:g}" for folder, share in o.mix.items())
            if o.mode == FOLDER_MIX_MODE
            else f"k={o.k:<3} alpha={o.alpha:<5g} cap={o.cap:<4g}"
        )
        print(f"{o.name:<24} {o.mode:<13} {settings} _rw{o.weights_digest[:8]}  {o.written_at[:16]}{stale}")
    return 0


def cmd_data_corpus(args: argparse.Namespace) -> int:
    import sys
    from pathlib import Path

    from ..data.corpus import CorpusError, CorpusSpec, build_corpus, plan_corpus
    from ..musicgen.config import get_repo_root

    repo_root = get_repo_root()
    spec = CorpusSpec(
        name=args.name,
        sources=tuple(Path(source).expanduser().resolve() for source in args.sources),
        slice_seconds=args.slice,
        hop_seconds=args.hop,
        min_clip_seconds=args.min_clip,
        sample_rate=args.sample_rate,
        channels=args.channels,
        exclude=frozenset(Path(path).expanduser().resolve() for path in args.exclude),
        skip_duplicates=args.skip_duplicates,
    )
    default_dir = repo_root / "training_datasets" / args.name
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_dir
    try:
        plan = plan_corpus(spec, output_dir)
    except CorpusError as exc:
        print(f"corpus_error: {exc}", file=sys.stderr)
        return 1

    _print_corpus_plan(plan, skip_duplicates=spec.skip_duplicates)
    if args.dry_run:
        print(f"dry_run: nothing written (would write {plan.clips_dir})")
        return 0
    try:
        result = build_corpus(
            plan, workers=args.workers, progress=lambda done, total: print(f"progress: {done}/{total}", flush=True)
        )
    except CorpusError as exc:
        print(f"corpus_error: {exc}", file=sys.stderr)
        return 1
    print(f"written: {result.written}")
    for path, error in result.failed:
        print(f"failed: {path}: {error}")
    print(f"manifest: {result.manifest_path}")
    print(f"next: python -m macchina_ricordi.cli data prepare {plan.clips_dir} --workspace-dir .workspace_{args.name}")
    print("(or build gathering and preparing in one step: `data build`)")
    return 1 if result.failed and not result.written else 0


def _print_corpus_plan(plan: Any, *, skip_duplicates: bool) -> None:
    print(f"{'source':<48} {'files':>6} {'hours':>7} {'clips':>7} {'clip h':>7} {'skipped':>8}")
    for group, counts in plan.per_source.items():
        print(
            f"{group[:48]:<48} {int(counts['files']):>6} {counts['source_seconds'] / 3600:>7.2f} "
            f"{int(counts['clips']):>7} {counts['seconds'] / 3600:>7.2f} {int(counts['skipped']):>8}"
        )
    print(f"clips: {len(plan.clips)} ({plan.total_seconds / 3600:.2f} h)")
    for path, reason in plan.skipped[:20]:
        print(f"skipped: {path}: {reason}")
    if len(plan.skipped) > 20:
        print(f"skipped: … and {len(plan.skipped) - 20} more")
    if plan.excluded:
        print(f"excluded: {len(plan.excluded)} files")
    if plan.duplicates and not skip_duplicates:
        for path, keeper in plan.duplicates[:20]:
            print(f"probable duplicate: {path} (same recording as {keeper}; --skip-duplicates keeps one)")


def cmd_data_build(args: argparse.Namespace) -> int:
    import sys
    from pathlib import Path

    from ..data.corpus import CorpusError, CorpusSpec, plan_corpus
    from ..data.datasets import DatasetError, build_dataset, plan_dataset_folder

    try:
        folder = plan_dataset_folder(args.name)
        spec = CorpusSpec(
            name=args.name,
            sources=tuple(Path(source).expanduser().resolve() for source in args.sources),
            slice_seconds=args.slice,
            hop_seconds=args.hop,
            min_clip_seconds=args.min_clip,
            sample_rate=args.sample_rate,
            channels=args.channels,
            exclude=frozenset(Path(path).expanduser().resolve() for path in args.exclude),
            skip_duplicates=args.skip_duplicates,
        )
        _print_corpus_plan(plan_corpus(spec, folder), skip_duplicates=args.skip_duplicates)
        if args.dry_run:
            print(f"dry_run: nothing written (would build {folder})")
            return 0
        dataset = build_dataset(
            args.name,
            args.sources,
            slice_seconds=args.slice,
            hop_seconds=args.hop,
            min_clip_seconds=args.min_clip,
            sample_rate=args.sample_rate,
            channels=args.channels,
            exclude=args.exclude,
            skip_duplicates=args.skip_duplicates,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
            max_silence_fraction=args.max_silence_fraction,
            speed_ratios=args.speed_ratios,
            speed_mode=args.speed_mode,
            musicgen_python=args.musicgen_python,
            workers=args.workers,
            progress=lambda done, total: print(f"progress: {done}/{total}", flush=True),
        )
    except (CorpusError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"dataset: {dataset.name} -> {dataset.path}")
    print(f"train_clips: {dataset.train_clips}  valid_clips: {dataset.valid_clips}")
    print(f"next: python -m macchina_ricordi.cli train run --recipe <recipe> --dataset {dataset.name} --dry-run")
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..data.datasets import list_datasets

    datasets = list_datasets()
    if args.json:
        print(json.dumps([dataset.to_dict() for dataset in datasets], indent=2))
        return 0
    header = f"{'dataset':<40} {'train':>6} {'valid':>6} {'hours':>6} {'slice s':>8}  sources"
    print(header)
    print("-" * len(header))
    for d in datasets:
        hours = f"{d.train_seconds / 3600:.2f}" if d.train_seconds else "?"
        slice_s = f"{d.slice_length_s:g}" if d.slice_length_s else "?"
        sources = ", ".join(Path(source).name for source in d.sources[:3]) + (" …" if len(d.sources) > 3 else "")
        name = d.name + (" (legacy)" if d.legacy else "")
        print(f"{name[:40]:<40} {d.train_clips or '?':>6} {d.valid_clips or '?':>6} {hours:>6} {slice_s:>8}  {sources}")
    print(f"\n{len(datasets)} datasets ({sum(1 for d in datasets if d.legacy)} legacy workspaces)")
    return 0


def cmd_data_show(args: argparse.Namespace) -> int:
    import sys

    from ..data.datasets import DatasetError, open_dataset
    from ..data.weights import list_weights_overlays
    from ..models.record import list_models
    from ..musicgen.config import get_repo_root

    try:
        dataset = open_dataset(args.name)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    models = [m for m in list_models() if m.dataset == dataset.name]
    weights = list_weights_overlays(get_repo_root(), dataset.name)
    if args.json:
        payload = dataset.to_dict()
        payload["models"] = [m.sig for m in models]
        payload["weights"] = [{**w.summary(), "stale": w.is_stale()} for w in weights]
        print(json.dumps(payload, indent=2))
        return 0
    for key, value in dataset.to_dict().items():
        if key != "sources":
            print(f"{key + ':':<20}{value}")
    for source in dataset.sources:
        print(f"{'source:':<20}{source}")
    print(f"{'models:':<20}{', '.join(m.name or m.sig for m in models) or 'none'}")
    print(f"{'weights:':<20}{', '.join(w.name for w in weights) or 'none'}")
    return 0


def cmd_data_audit(args: argparse.Namespace) -> int:
    from ..data.prepare import audit_workspace_slice_lengths

    rows = audit_workspace_slice_lengths(
        root=args.root,
        workspace_dir=args.workspace_dir,
    )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = f"{'workspace':<52} {'n':>5} {'min':>6} {'med':>6} {'p90':>6} {'slice':>6} {'short':>6} {'safe_D':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        stats = row["stats"]
        name = str(row["workspace"])
        if stats is None:
            print(f"{name:<52} {'(no readable manifest)':>40}")
            continue
        print(
            f"{name:<52} {stats['count']:>5} {stats['min']:>6.1f} "
            f"{stats['median']:>6.1f} {stats['p90']:>6.1f} "
            f"{stats['slice_length_s']:>6.1f} {stats['n_shorter_than_half']:>6} "
            f"{row['safe_segment_duration']:>6}s"
        )
    return 0


def register_data_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    data = subparsers.add_parser(
        "data",
        help="Datasets: build, list, show, audit, and weight them",
        description="A dataset is a named set of clips split into train and valid, built once from sources, never "
        "rewritten (docs/glossary.md). `corpus` and `prepare` are the two older, lower-level steps `build` does in "
        "one.",
        epilog="""examples:
  python -m macchina_ricordi.cli data build noisy_10s examples/sample_data/<album> --slice 10 --dry-run
  python -m macchina_ricordi.cli data list --json
  python -m macchina_ricordi.cli data rebalance --dataset noisy_10s --save-as flat_a05
  python -m macchina_ricordi.cli data mix --dataset noisy_10s album_a=3 album_b=1 --save-as a3_b1""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    data_build = data_subparsers.add_parser(
        "build",
        help="Build a new dataset in one step: gather clips from sources, split them, write manifests",
        description=(
            "Gather the audio under one or more source folders (whole files, hard-linked, or --slice into FLAC "
            "slices), split it into train and valid with hard links, and write AudioCraft manifests, all into "
            "datasets/<name>/ with a dataset.json describing the build. A dataset is never rebuilt: a different "
            "cut is a new dataset under a new name."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli data build bear_bones "examples/sample_data/Split" \\
      "examples/sample_data/[2017] Feng Shen" --dry-run
  python -m macchina_ricordi.cli data build drones_10s "examples/sample_data/Barn Owl" --slice 10 --skip-duplicates
  python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset drones_10s --dry-run""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_build.add_argument("name", help="Dataset name: datasets/<name>")
    data_build.add_argument("sources", nargs="+", help="Source folders; each becomes a group named after the folder")
    data_build.add_argument("--slice", type=float, default=None, help="Cut files into slices of this many seconds")
    data_build.add_argument("--hop", type=float, default=None, help="Seconds between slice starts (default: --slice)")
    data_build.add_argument("--min-clip", type=float, default=None, help="Skip files and tails shorter than this")
    data_build.add_argument("--sample-rate", type=int, default=32000, help="Slice sample rate")
    data_build.add_argument("--channels", type=int, choices=(1, 2), default=1, help="Slice channels")
    data_build.add_argument("--exclude", action="append", default=[], metavar="FILE", help="Leave this file out")
    data_build.add_argument("--skip-duplicates", action="store_true", help="Keep one copy of each probable duplicate")
    data_build.add_argument("--valid-fraction", type=float, default=None, help="Share of clips held out (default 0.1)")
    data_build.add_argument("--seed", type=int, default=None, help="Split seed (default 42)")
    data_build.add_argument(
        "--max-silence-fraction", type=float, default=None, help="Drop clips more silent than this (default 0.85)"
    )
    data_build.add_argument("--speed-ratios", default=None, help="Sped-up/slowed train copies, e.g. 0.9,1.1")
    data_build.add_argument("--speed-mode", choices=SPEED_MODES, default=DEFAULT_SPEED_MODE, help="Speed variant mode")
    data_build.add_argument("--musicgen-python", default=None, help="Python with audiocraft (default: .venv-musicgen)")
    data_build.add_argument("--workers", type=int, default=None, help="Files decoded in parallel")
    data_build.add_argument("--dry-run", action="store_true", help="Print the per-source table and write nothing")
    data_build.set_defaults(func=cmd_data_build)

    data_list = data_subparsers.add_parser(
        "list", help="Every dataset: built ones and legacy workspaces", formatter_class=MacchinaRicordiHelpFormatter
    )
    data_list.add_argument("--json", action="store_true", help="Machine-readable output")
    data_list.set_defaults(func=cmd_data_list)

    data_show = data_subparsers.add_parser(
        "show",
        help="One dataset: how it was built, the models trained on it, its weights",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_show.add_argument("name", help="Dataset name (`data list`) or folder")
    data_show.add_argument("--json", action="store_true", help="Machine-readable output")
    data_show.set_defaults(func=cmd_data_show)

    data_audit = data_subparsers.add_parser(
        "audit",
        help="Report clip-length stats and the safe segment_duration per workspace",
        description=(
            "Read each workspace's prepared manifest and report its clip-length "
            "profile, so you can confirm dataset.segment_duration (D) does not exceed "
            "the slice length (S). Scans all .workspace*/.macchina_ricordi* dirs by default."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_audit.add_argument(
        "--dataset",
        "--workspace-dir",
        dest="workspace_dir",
        default=None,
        help="Audit a single workspace directory instead of scanning the repo root",
    )
    data_audit.add_argument(
        "--root",
        default=None,
        help="Directory to scan for workspaces (defaults to the repo root)",
    )
    data_audit.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON rows instead of the formatted table",
    )
    data_audit.set_defaults(func=cmd_data_audit)

    data_rebalance = data_subparsers.add_parser(
        "rebalance",
        help="Compute cluster-based sampling weights and save them as a named overlay",
        description=(
            "Cluster the train segments by timbre and compute per-segment sampling weights. "
            "Without --gens-dir the cluster prior is flattened (inverse-share); with one or more "
            "--gens-dir the clusters a model under-produces are upweighted (coverage mode). "
            "--save-as NAME keeps them as an overlay under artifacts/weights/<workspace>/NAME/ (a weighted "
            "copy of the train manifest; the workspace itself is never written). Train with them via "
            "`train run --weights NAME`: the auto-label gains a _rw<digest> suffix, so a fresh Dora signature."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_x --dry-run
  python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_x --save-as flat_a05
  python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_x --gens-dir artifacts/sweeps/s1 \\
      --alpha 0.5 --save-as cov_r1
  python -m macchina_ricordi.cli train run --workspace-dir .workspace_x --recipe reshape-nano --weights cov_r1""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_rebalance.add_argument("--profile", "--recipe", dest="profile", default=None, help=argparse.SUPPRESS)
    data_rebalance.add_argument(
        "--dataset",
        "--workspace-dir",
        dest="workspace_dir",
        required=True,
        help="The dataset to weight: a name (`data list`) or a folder",
    )
    data_rebalance.add_argument(
        "--dataset-name",
        default=None,
        help="The workspace's egs/<name> to weight, when it holds more than one (default: the only one)",
    )
    data_rebalance.add_argument(
        "--gens-dir",
        action="append",
        default=None,
        help="Directory of generated WAVs to measure model coverage against (repeatable)",
    )
    data_rebalance.add_argument("--k", type=int, default=10, help="Number of clusters")
    data_rebalance.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Reweighting exponent; defaults to 0.5 for both modes. Higher "
            "corrects faster but overshoots (validated: 1.0 in one +15ep shot "
            "suppressed previously-covered clusters); iterate instead"
        ),
    )
    data_rebalance.add_argument(
        "--cap",
        type=float,
        default=4.0,
        help="Max per-cluster weight ratio; weights are clipped to [1/cap, cap]",
    )
    data_rebalance.add_argument("--seed", type=int, default=0, help="Clustering seed")
    data_rebalance.add_argument("--save-as", default=None, help="Save the weights as this named overlay")
    data_rebalance.add_argument("--dry-run", action="store_true", help="Print the cluster/weight table only")
    data_rebalance.add_argument(
        "--plan-out",
        default=None,
        help="Save the computed weights to this file, to look at and save later with --apply-plan",
    )
    data_rebalance.add_argument(
        "--apply-plan",
        default=None,
        help="Use weights saved by --plan-out instead of computing them (with --save-as)",
    )
    data_rebalance.add_argument(
        "--reset",
        action="store_true",
        help="Clear weights written into the workspace itself before overlays existed",
    )
    data_rebalance.set_defaults(func=cmd_data_rebalance)

    data_mix = data_subparsers.add_parser(
        "mix",
        help="Weight a dataset's train split by source folder: the share of training draws each folder gets",
        description=(
            "Give each source folder a share of the train split's draws, as relative numbers (album_a=3 album_b=1 "
            "is three quarters and a quarter). A clip's chance of being drawn is its weight, so each clip gets "
            "share / clips in its folder and the folder's share comes out exact. --save-as NAME keeps it as a "
            "weights overlay, like `data rebalance --save-as`; train with it via `train run --weights NAME`. Every "
            "folder with train clips must be listed. Exact under the default per-file sampling; with sampling in "
            "proportion to length, each clip's weight also scales with its length."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli data mix --dataset noisy_10s album_a=3 album_b=1 --dry-run
  python -m macchina_ricordi.cli data mix --dataset noisy_10s album_a=3 album_b=1 --save-as a3_b1
  python -m macchina_ricordi.cli train run --dataset noisy_10s --recipe reshape-nano --weights a3_b1""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_mix.add_argument("shares", nargs="+", metavar="FOLDER=SHARE", help="A source folder and its relative share")
    data_mix.add_argument(
        "--dataset",
        "--workspace-dir",
        dest="workspace_dir",
        required=True,
        help="The dataset to weight: a name (`data list`) or a folder",
    )
    data_mix.add_argument(
        "--dataset-name",
        default=None,
        help="The workspace's egs/<name> to weight, when it holds more than one (default: the only one)",
    )
    data_mix.add_argument("--save-as", default=None, help="Save the mix as this named overlay")
    data_mix.add_argument("--dry-run", action="store_true", help="Print the folder table only")
    data_mix.set_defaults(func=cmd_data_mix)

    data_weights = data_subparsers.add_parser(
        "weights",
        help="List or delete the weights overlays saved for a workspace",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    weights_subparsers = data_weights.add_subparsers(dest="weights_command", required=True)
    weights_list = weights_subparsers.add_parser("list", help="The overlays saved for a workspace")
    weights_list.add_argument("--dataset", "--workspace-dir", dest="workspace_dir", required=True, help="The dataset")
    weights_list.add_argument("--json", action="store_true", help="Machine-readable output")
    weights_list.set_defaults(func=cmd_data_weights)
    weights_delete = weights_subparsers.add_parser("delete", help="Delete one overlay")
    weights_delete.add_argument("name", help="The overlay's name")
    weights_delete.add_argument("--dataset", "--workspace-dir", dest="workspace_dir", required=True, help="The dataset")
    weights_delete.set_defaults(func=cmd_data_weights)

    data_prepare = data_subparsers.add_parser(
        "prepare",
        help="Split a clip folder into a new workspace and write manifests (lower level than `data build`)",
        description=(
            "Convert a curated clip folder into a repo-local AudioCraft training workspace. `data build` does this "
            "and the gathering of clips in one step, into datasets/<name>; use prepare for a clip folder you made "
            "yourself. It refuses a workspace that is already prepared unless --overwrite."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_prepare.add_argument("input_dir", help="Directory tree containing curated clip files")
    # Accepted and ignored: profiles used to name the workspace and dataset; recipes do not.
    data_prepare.add_argument("--profile", "--recipe", dest="profile", default=None, help=argparse.SUPPRESS)
    data_prepare.add_argument(
        "--workspace-dir",
        required=True,
        help="New workspace folder to prepare into, e.g. .workspace_<name>. Never an existing one: prepare wipes it",
    )
    data_prepare.add_argument(
        "--musicgen-python",
        default=None,
        help="Python executable that has audiocraft installed; defaults to the repo-managed .venv-musicgen runtime",
    )
    data_prepare.add_argument(
        "--valid-fraction",
        type=float,
        default=musicgen_default_valid_fraction(),
        help="Fraction of clips reserved for validation",
    )
    data_prepare.add_argument(
        "--seed",
        type=int,
        default=musicgen_default_split_seed(),
        help="Deterministic split seed",
    )
    data_prepare.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated audio extensions to include",
    )
    data_prepare.add_argument(
        "--max-silence-fraction",
        type=float,
        default=0.85,
        help="Drop input clips whose silence fraction exceeds this; set to 1.0 to disable",
    )
    data_prepare.add_argument(
        "--silence-frame-db",
        type=float,
        default=-35.0,
        help="Peak-relative dB threshold for classifying a frame as silent during silence filtering",
    )
    data_prepare.add_argument(
        "--speed-ratios",
        default=None,
        help=(
            "Comma-separated speed ratios, e.g. 0.9,1.1: adds a FLAC copy of each train clip per ratio "
            "(1 is ignored, validation is never augmented). Ratios > 1 shorten clips; check `data audit`"
        ),
    )
    data_prepare.add_argument(
        "--speed-mode",
        choices=SPEED_MODES,
        default=DEFAULT_SPEED_MODE,
        help="varispeed: pitch follows speed, artefact-free; stretch: pitch kept, transients smeared",
    )
    data_prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="Prepare an already prepared workspace again: wipes its data/, egs/ and logs/ (run records included)",
    )
    data_prepare.set_defaults(func=cmd_data_prepare)

    data_corpus = data_subparsers.add_parser(
        "corpus",
        help="Assemble a clip folder from source folders, whole files or fixed-length slices",
        description=(
            "Collect the audio under one or more source folders into <output-dir>/clips, ready for "
            "`data prepare`. Clips are named <source>__<file>[__<index>] so they stay attributable. "
            "Whole files are hard-linked (or copied); with --slice, files are cut into fixed-length FLAC "
            "slices (32 kHz mono by default). corpus.json records the recipe and every clip's origin."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli data corpus drones_10s "examples/sample_data/Barn Owl" \\
      examples/sample_data/emptyset --slice 10 --dry-run
  python -m macchina_ricordi.cli data corpus noisy_whole examples/sample_data/midwife
  python -m macchina_ricordi.cli data corpus stott "examples/sample_data/2014 - Faith in Strangers" \
      "examples/sample_data/Andy Stott - Faith in Strangers (2014) - FLAC" --skip-duplicates \
      --exclude "examples/sample_data/2014 - Faith in Strangers/09 - Missing.mp3" --dry-run""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    data_corpus.add_argument("name", help="Corpus name; also the default output folder training_datasets/<name>")
    data_corpus.add_argument("sources", nargs="+", help="Source folders; each becomes a group named after the folder")
    data_corpus.add_argument("--output-dir", default=None, help="Where to write clips/ and corpus.json")
    data_corpus.add_argument("--slice", type=float, default=None, help="Cut files into slices of this many seconds")
    data_corpus.add_argument("--hop", type=float, default=None, help="Seconds between slice starts (default: --slice)")
    data_corpus.add_argument(
        "--min-clip",
        type=float,
        default=None,
        help="Skip files and tails shorter than this (default: the slice length, so every slice is full)",
    )
    data_corpus.add_argument("--sample-rate", type=int, default=32000, help="Slice sample rate (default: 32000)")
    data_corpus.add_argument("--channels", type=int, choices=(1, 2), default=1, help="Slice channels (default: 1)")
    data_corpus.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="FILE",
        help="Leave this file out (repeatable); it must be inside one of the sources",
    )
    data_corpus.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Keep one copy of each probable duplicate recording (same title and length, matching audio): "
        "lossless over lossy, then the larger file. Without it they are only listed",
    )
    data_corpus.add_argument("--workers", type=int, default=None, help="Files decoded in parallel")
    data_corpus.add_argument("--dry-run", action="store_true", help="Print the per-source table and write nothing")
    data_corpus.set_defaults(func=cmd_data_corpus)
