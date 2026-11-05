from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files, parse_extensions
from ..cli.common import (
    MacchinaRicordiHelpFormatter,
    _get_tqdm,
    add_shared_scoring_args,
    format_metric_value,
    write_csv,
)


def _score_arrhythmic_percussive(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .reward import score_arrhythmic_percussive

    return score_arrhythmic_percussive(*args, **kwargs)


def _reward_components_cls():
    from .reward import RewardComponents

    return RewardComponents


def _build_calibration_summary(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .calibration import build_calibration_summary

    return build_calibration_summary(*args, **kwargs)


def score_files(
    files: list[Path],
    *,
    label: str | None = None,
    quiet: bool = False,
    sr: int = 32_000,
    hop_length: int = 512,
    res_type: str = "soxr_hq",
    tempo_penalty_method: str = "tempogram",
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0
    progress_desc = f"Scoring ({label})" if label else "Scoring"
    reward_components_cls = _reward_components_cls()
    tqdm = _get_tqdm()

    for path in tqdm(
        files,
        total=len(files),
        desc=progress_desc,
        unit="file",
        disable=quiet,
    ):
        row: dict[str, Any] = {
            "path": str(path),
            **reward_components_cls().to_dict(),
            "error": "",
        }
        if label is not None:
            row["label"] = label

        try:
            components = _score_arrhythmic_percussive(
                path,
                sr=sr,
                hop_length=hop_length,
                res_type=res_type,
                tempo_penalty_method=tempo_penalty_method,
            )
            row.update(components.to_dict())
        except Exception as exc:  # pylint: disable=broad-except
            failures += 1
            row["error"] = str(exc)

        rows.append(row)

    return rows, failures


def cmd_score_file(args: argparse.Namespace) -> int:
    components = _score_arrhythmic_percussive(
        args.audio_path,
        sr=args.sr,
        hop_length=args.hop_length,
        res_type=args.res_type,
        tempo_penalty_method=args.tempo_method,
    )
    payload = {
        "path": str(Path(args.audio_path).expanduser().resolve()),
        **components.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"path: {payload['path']}")
        for key, value in components.to_dict().items():
            print(f"{key}: {format_metric_value(value)}")
    return 0


def cmd_score_dir(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list(
        iter_audio_files(
            input_dir,
            recursive=args.recursive,
            extensions=parse_extensions(args.extensions),
        )
    )
    rows, failures = score_files(
        files,
        quiet=args.quiet,
        sr=args.sr,
        hop_length=args.hop_length,
        res_type=args.res_type,
        tempo_penalty_method=args.tempo_method,
    )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)

    output_path = Path(args.output).expanduser().resolve()
    reward_components_cls = _reward_components_cls()
    fieldnames = ["path", *reward_components_cls.fieldnames(), "error"]
    write_csv(output_path, fieldnames, rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Successes: {len(rows) - failures} | Failures: {failures}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    steady_dir = Path(args.steady_dir).expanduser().resolve()
    arrhythmic_dir = Path(args.arrhythmic_dir).expanduser().resolve()
    if not steady_dir.exists():
        raise FileNotFoundError(f"Steady directory does not exist: {steady_dir}")
    if not arrhythmic_dir.exists():
        raise FileNotFoundError(f"Arrhythmic directory does not exist: {arrhythmic_dir}")

    extensions = parse_extensions(args.extensions)
    steady_files = list(iter_audio_files(steady_dir, recursive=args.recursive, extensions=extensions))
    arrhythmic_files = list(iter_audio_files(arrhythmic_dir, recursive=args.recursive, extensions=extensions))

    steady_rows, steady_failures = score_files(
        steady_files,
        label="steady",
        quiet=args.quiet,
        sr=args.sr,
        hop_length=args.hop_length,
        res_type=args.res_type,
        tempo_penalty_method=args.tempo_method,
    )
    arrhythmic_rows, arrhythmic_failures = score_files(
        arrhythmic_files,
        label="arrhythmic",
        quiet=args.quiet,
        sr=args.sr,
        hop_length=args.hop_length,
        res_type=args.res_type,
        tempo_penalty_method=args.tempo_method,
    )

    combined_rows = arrhythmic_rows + steady_rows
    combined_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    arrhythmic_scores = [float(row["score"]) for row in arrhythmic_rows if not row["error"]]
    steady_scores = [float(row["score"]) for row in steady_rows if not row["error"]]

    summary = _build_calibration_summary(
        arrhythmic_scores=arrhythmic_scores,
        steady_scores=steady_scores,
    )
    summary["arrhythmic"].update(
        {
            "files_total": len(arrhythmic_rows),
            "files_failed": arrhythmic_failures,
        }
    )
    summary["steady"].update(
        {
            "files_total": len(steady_rows),
            "files_failed": steady_failures,
        }
    )

    output_path = Path(args.output).expanduser().resolve()
    reward_components_cls = _reward_components_cls()
    fieldnames = ["label", "path", *reward_components_cls.fieldnames(), "error"]
    write_csv(output_path, fieldnames, combined_rows)

    print("Calibration summary")
    print(f"arrhythmic files: {summary['arrhythmic']['files_total']} (failed: {summary['arrhythmic']['files_failed']})")
    print(f"steady files: {summary['steady']['files_total']} (failed: {summary['steady']['files_failed']})")
    print(f"mean score (arrhythmic vs steady): {summary['arrhythmic']['mean']} vs {summary['steady']['mean']}")
    print(f"ranking accuracy: {summary['separation']['ranking_accuracy']}")
    print(f"cohens_d: {summary['separation']['cohens_d']}")
    print(f"best_threshold: {summary['separation']['best_threshold']}")
    print(f"wrote calibration rows: {output_path}")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"wrote summary json: {summary_path}")

    return 0


def cmd_score_sim(args: argparse.Namespace) -> int:
    from .similarity import compute_similarity

    path_a = Path(args.path_a).expanduser().resolve()
    path_b = Path(args.path_b).expanduser().resolve()
    components = compute_similarity(path_a, path_b, sr=args.sr, res_type=args.res_type)

    if args.json:
        import json

        payload: dict[str, Any] = {
            "path_a": str(path_a),
            "path_b": str(path_b),
            **components.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"path_a: {path_a}")
        print(f"path_b: {path_b}")
        print(f"mfcc_cosine:   {components.mfcc_cosine:.4f}")
        print(f"spectral_cosine: {components.spectral_cosine:.4f}")
        print(f"rhythm_cosine: {components.rhythm_cosine:.4f}")
        print(f"combined:      {components.combined:.4f}")
    return 0


def cmd_score_sim_calibrate(args: argparse.Namespace) -> int:
    import json

    from .similarity import calibrate_from_dir, load_calibration

    result = calibrate_from_dir(args.directory, sr=args.sr, max_files=args.max_files)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(f"Calibrated from {result['num_files']} files across 3 feature groups.")
    print(f"Saved to: {output_path}")
    if args.activate:
        load_calibration(result)
        print("Calibration active — subsequent 'score sim' calls will use z-score normalisation.")
    else:
        print("Tip: pass --activate to load this calibration for the current session.")
    return 0


def cmd_score_structure(args: argparse.Namespace) -> int:
    from .structure import compute_structure

    components = compute_structure(
        args.audio_path,
        sr=args.sr,
        res_type=args.res_type,
        window_count=args.windows,
    )
    payload = {
        "path": str(Path(args.audio_path).expanduser().resolve()),
        **components.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"path: {payload['path']}")
        for key, value in components.to_dict().items():
            print(f"{key}: {format_metric_value(value)}")
    return 0


def cmd_score_rescore(args: argparse.Namespace) -> int:
    from ..sessions.record import SessionNotFound
    from ..sessions.store import resolve_session_folders
    from .batch import (
        iter_sweep_items,
        rank_groups,
        rescore_paths,
        summarize_by_group,
        sweep_base_metrics,
        write_rows_csv,
    )

    try:
        args.sweep_dirs = [str(folder) for folder in resolve_session_folders(args.sweep_dirs)]
    except SessionNotFound as exc:
        print(f"error: {exc}")
        return 1
    items = iter_sweep_items(args.sweep_dirs, group_column=args.group_column)
    if not items:
        print("No playable takes found in the given sessions.")
        return 1

    total = len(items)
    state = {"done": 0}

    def _progress(done: int, _total: int) -> None:
        state["done"] = done
        if not args.quiet and (done % 10 == 0 or done == total):
            print(f"  scored {done}/{total}", flush=True)

    if not args.quiet:
        print(f"Re-scoring {total} clips from {len(list(args.sweep_dirs))} sessions ...")
    rows = rescore_paths(
        items,
        jobs=args.jobs,
        progress=_progress,
        base_metrics=sweep_base_metrics(args.sweep_dirs, group_column=args.group_column),
    )
    dict_rows = [row.to_dict() for row in rows]

    output_path = write_rows_csv(args.output, dict_rows)
    summary = summarize_by_group(dict_rows, metrics=RESCORE_SUMMARY_METRICS)

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nScored {len(rows)} clips → {output_path}\n")
        header = f"{'group':<40} {'n':>4} {'sil':>4} " + " ".join(
            f"{metric.replace('structure_', '').replace('arc_', ''):>10}" for metric in RESCORE_SUMMARY_METRICS
        )
        print(header)
        print("-" * len(header))
        for group, entry in summary.items():
            cells = " ".join(
                f"{entry[metric]:>10.3f}" if entry.get(metric) is not None else f"{'-':>10}"
                for metric in RESCORE_SUMMARY_METRICS
            )
            print(f"{group[:40]:<40} {entry['usable']:>4} {entry['dropped_silent']:>4} {cells}")
        ranked = rank_groups(summary, args.rank_by)
        if ranked:
            print(f"\nranked by {args.rank_by}:")
            for position, (group, value) in enumerate(ranked, start=1):
                print(f"  {position}. {group}  {value:.3f}")
    return 0


# `arc_coherence_gated` and `arc_tv4` are still scored and still land in the CSVs;
# they are off both summary lists because each is a restatement of
# `arc_coherence_score` (98% identical, ρ = 0.9994), so listing them alongside it
# triples the weight of one axis in anything that reads these.
RESCORE_SUMMARY_METRICS = (
    "structure_score",
    "dev_top2",
    "dev_harmonic",
    "dev_timbre",
    "dev_dynamics",
    "dev_rhythm",
    "ssm_novelty",
    "ssm_return",
    "ssm_mean",
    "arc_coherence_score",
)


def cmd_score_similar(args: argparse.Namespace) -> int:
    from ..listening.archive import (
        archive_extensions,
        build_similar_report,
        load_archive_index,
        render_similar_markdown,
    )
    from ..listening.captioner import CaptionerBackendUnavailable
    from ..listening.cli import _analysis_kwargs, _captioner_kwargs, _semantic_kwargs
    from ..listening.semantic import SemanticBackendUnavailable

    index = load_archive_index(args.index) if args.index else None
    try:
        report = build_similar_report(
            args.audio_path,
            args.root,
            index=index,
            neighbors=args.neighbors,
            recursive=args.recursive,
            extensions=archive_extensions(args.extensions),
            read_only=args.read_only,
            refresh=args.refresh,
            **_semantic_kwargs(args),
            **_captioner_kwargs(args),
            **_analysis_kwargs(args),
        )
    except SemanticBackendUnavailable as exc:
        raise SystemExit(f"semantic probes unavailable: {exc}") from exc
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner witness unavailable: {exc}") from exc
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_similar_markdown(report), end="")
    return 0


def cmd_score_novelty(args: argparse.Namespace) -> int:
    from ..listening.archive import (
        archive_extensions,
        build_archive_index,
        build_novelty_report,
        load_archive_index,
        render_novelty_markdown,
    )
    from ..listening.captioner import CaptionerBackendUnavailable
    from ..listening.cli import _analysis_kwargs, _captioner_kwargs, _semantic_kwargs
    from ..listening.semantic import SemanticBackendUnavailable

    try:
        index = (
            load_archive_index(args.index)
            if args.index
            else build_archive_index(
                args.root,
                recursive=args.recursive,
                extensions=archive_extensions(args.extensions),
                neighbors=max(args.limit, 1),
                read_only=args.read_only,
                refresh=args.refresh,
                **_semantic_kwargs(args),
                **_captioner_kwargs(args),
                **_analysis_kwargs(args),
            )
        )
    except SemanticBackendUnavailable as exc:
        raise SystemExit(f"semantic probes unavailable: {exc}") from exc
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner witness unavailable: {exc}") from exc
    report = build_novelty_report(index, limit=args.limit)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_novelty_markdown(report), end="")
    return 0


def register_scoring_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from ..listening.cli import _add_archive_build_args

    score = subparsers.add_parser(
        "score",
        help="Numbers about audio: reward, structure, similarity, novelty, calibration",
        description=(
            "Numbers out: the arrhythmic/percussive reward, arc and structure axes, similarity between files or "
            "within an archive, novelty, and calibration. Text descriptions are `listen`; verdicts are `judge`."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli score file sample.wav
  python -m macchina_ricordi.cli score dir clips --recursive          # artifacts/scores/scores.csv
  python -m macchina_ricordi.cli score sim before.wav after.wav --json
  python -m macchina_ricordi.cli score similar query.wav artifacts/sessions""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_subparsers = score.add_subparsers(dest="score_command", required=True)

    score_file = score_subparsers.add_parser(
        "file",
        help="Score one audio file",
        description="Score one file with the arrhythmic/percussive reward components.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_file.add_argument("audio_path", help="Path to audio file")
    score_file.add_argument("--json", action="store_true", help="Output as JSON")
    add_shared_scoring_args(score_file)
    score_file.set_defaults(func=cmd_score_file)

    score_dir = score_subparsers.add_parser(
        "dir",
        help="Score all files in a directory",
        description="Score a directory of audio files and write a CSV table.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_dir.add_argument("input_dir", help="Directory containing audio files")
    score_dir.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list, e.g. wav,flac,mp3",
    )
    add_shared_scoring_args(score_dir)
    score_dir.add_argument("--recursive", action="store_true", help="Include subdirectories")
    score_dir.add_argument("--output", default="artifacts/scores/scores.csv", help="Output CSV path")
    score_dir.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    score_dir.set_defaults(func=cmd_score_dir)

    calibrate = score_subparsers.add_parser(
        "calibrate",
        help="Compare steady and arrhythmic folders to validate score separation",
        description="Compare steady-tempo and arrhythmic reference folders to inspect reward separation.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    calibrate.add_argument("steady_dir", help="Directory with steady-tempo reference clips")
    calibrate.add_argument(
        "arrhythmic_dir",
        help="Directory with arrhythmic-percussive reference clips",
    )
    calibrate.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list, e.g. wav,flac,mp3",
    )
    calibrate.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    calibrate.add_argument("--hop-length", type=int, default=512, help="Hop length for onset analysis")
    calibrate.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    calibrate.add_argument(
        "--tempo-method",
        choices=("autocorr", "tempogram"),
        default="autocorr",
        help="Tempo clarity penalty backend",
    )
    calibrate.add_argument("--recursive", action="store_true", help="Include subdirectories")
    calibrate.add_argument(
        "--output", default="artifacts/scores/calibration_scores.csv", help="Per-file calibration output CSV"
    )
    calibrate.add_argument("--summary-json", default="", help="Optional summary JSON output path")
    calibrate.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    calibrate.set_defaults(func=cmd_calibrate)

    score_sim = score_subparsers.add_parser(
        "sim",
        help="Compute audio similarity between two files",
        description="Compare two audio files with deterministic MFCC, spectral, and rhythm similarity.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_sim.add_argument("path_a", help="First audio file")
    score_sim.add_argument("path_b", help="Second audio file")
    score_sim.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    score_sim.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    score_sim.add_argument("--json", action="store_true", help="Output as JSON")
    score_sim.set_defaults(func=cmd_score_sim)

    sim_calibrate = score_subparsers.add_parser(
        "sim-calibrate",
        help="Build z-score calibration from a directory of audio files",
        description="Build similarity calibration statistics from a recursive audio directory.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    sim_calibrate.add_argument("directory", help="Directory of audio files (recursive)")
    sim_calibrate.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    sim_calibrate.add_argument("--max-files", type=int, default=200, help="Max files to sample")
    sim_calibrate.add_argument(
        "--output", default="artifacts/scores/similarity_calibration.json", help="Output JSON path"
    )
    sim_calibrate.add_argument("--activate", action="store_true", help="Load calibration into the current session")
    sim_calibrate.set_defaults(func=cmd_score_sim_calibrate)

    score_similar = score_subparsers.add_parser(
        "similar",
        help="Find sounds similar to an audio file",
        description="Find nearest archive entries for one query audio file.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_similar.add_argument("audio_path", help="Target audio path")
    score_similar.add_argument("root", help="Audio archive root")
    score_similar.add_argument("--index", default="", help="Existing archive index JSON")
    score_similar.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    score_similar.add_argument("--neighbors", type=int, default=5, help="Number of neighbors to show")
    _add_archive_build_args(score_similar)
    score_similar.set_defaults(func=cmd_score_similar)

    score_novelty = score_subparsers.add_parser(
        "novelty",
        help="List the most novel sounds in an archive",
        description="Rank archive entries by nearest-neighbor novelty.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_novelty.add_argument("root", help="Audio archive root")
    score_novelty.add_argument("--index", default="", help="Existing archive index JSON")
    score_novelty.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    score_novelty.add_argument("--limit", type=int, default=20, help="Number of entries to show")
    _add_archive_build_args(score_novelty)
    score_novelty.set_defaults(func=cmd_score_novelty)

    score_structure = score_subparsers.add_parser(
        "structure",
        help="Multi-axis development + section-structure metrics for one file",
        description=(
            "Score development on four axes (harmonic, timbral, dynamic, rhythmic) plus "
            "self-similarity structure (novelty, return). Complements 'score file', which "
            "only sees arrhythmic/percussive qualities, and the chroma-only arc metrics."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_structure.add_argument("audio_path", help="Path to audio file")
    score_structure.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    score_structure.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    score_structure.add_argument("--windows", type=int, default=8, help="Overlapping analysis windows")
    score_structure.add_argument("--json", action="store_true", help="Output as JSON")
    score_structure.set_defaults(func=cmd_score_structure)

    score_rescore = score_subparsers.add_parser(
        "rescore",
        help="Re-score existing sweeps on the arc + structure axes and summarise per group",
        description=(
            "Run the arc and structure metrics over every sample in one or more sweep "
            "directories, write a combined CSV, and print per-group means over clips that "
            "pass the silence gate."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli score rescore artifacts/sweeps/uncond_deep50_* artifacts/sweeps/uncond_deep25hz_*
  python -m macchina_ricordi.cli score rescore artifacts/sweeps/uncond_* --output /tmp/rescore.csv --jobs 8""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_rescore.add_argument(
        "sweep_dirs",
        nargs="+",
        metavar="sessions",
        help="Sessions to re-score: folders or ids (`session list`); sweeps, streams (every round) or any other kind",
    )
    score_rescore.add_argument("--output", default="artifacts/scores/rescore.csv", help="Per-clip output CSV")
    score_rescore.add_argument("--summary-json", default="", help="Optional per-group summary JSON path")
    score_rescore.add_argument("--group-column", default="sweep_id", help="index.csv column to group by")
    score_rescore.add_argument("--rank-by", default="structure_score", help="Metric used for the ranking print")
    score_rescore.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = auto, 1 = inline)")
    score_rescore.add_argument("--json", action="store_true", help="Print the summary as JSON")
    score_rescore.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    score_rescore.set_defaults(func=cmd_score_rescore)
