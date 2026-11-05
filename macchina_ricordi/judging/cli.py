"""The ``judge`` group: verdicts about takes (blind A/B, alarm labels, metric agreement, the captioner witness)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files
from ..cli.common import MacchinaRicordiHelpFormatter, write_csv
from .label import DEFAULT_LABEL_PATH, alarm_ids
from .preference import DEFAULT_PREFERENCE_PATH as DEFAULT_PREFERENCE_STORE
from .questions import DEFAULT_QUESTION, PAIRWISE, PAIRWISE_CONTEXT, get_question, question_ids
from .store import DEFAULT_IMPRESSION_PATH, DEFAULT_WITNESS_PATH, VERDICT_KINDS


def _resolve_session_sources(args: argparse.Namespace) -> bool:
    """Turn session ids among ``args.sources`` into their folders; print why and return False if one is unknown."""
    from ..sessions.record import SessionNotFound
    from ..sessions.store import resolve_session_folders

    try:
        args.sources = [str(folder) for folder in resolve_session_folders(args.sources)]
    except SessionNotFound as exc:
        print(f"error: {exc}")
        return False
    return True


def _testing(value: str) -> str:
    from .ab import parse_testing

    try:
        return parse_testing(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def cmd_judge_ab(args: argparse.Namespace) -> int:
    from contextlib import nullcontext

    from ..instrument.stream.tui import TerminalSession
    from .ab import load_candidates, run_ab_session, select_pairs, testing_group_by, varied_knob
    from .ab_context import ContextAudioError, ContextAudioPreparer
    from .preference import judged_pair_keys, load_preferences, new_session_id
    from .questions import PAIRWISE, PAIRWISE_CONTEXT, get_question

    try:
        question = get_question(args.question)
    except KeyError as exc:
        print(exc.args[0])
        return 1
    if question.type not in (PAIRWISE, PAIRWISE_CONTEXT):
        print(f"'{question.id}' is a {question.type} question and cannot be asked by this pairwise session.")
        return 1
    if not _resolve_session_sources(args):
        return 1
    contextual = question.type == PAIRWISE_CONTEXT
    if contextual:
        non_streams = [
            str(source) for source in args.sources if not (Path(source).expanduser() / "performance.jsonl").exists()
        ]
        if non_streams:
            print(
                "'newer' requires stream-session sources containing performance.jsonl; "
                f"other sessions or invalid sources are not supported: {', '.join(non_streams)}"
            )
            return 1

    disagree = [metric.strip() for metric in args.disagree.split(",") if metric.strip()]
    if disagree and len(disagree) != 2:
        print(f"--disagree takes exactly two metrics, got {len(disagree)}.")
        return 1
    if disagree and args.target_metric:
        print("--target-metric and --disagree are two different targetings; pass one.")
        return 1
    within = args.testing != "groups"
    knob = varied_knob(args.testing)
    group_by = testing_group_by(args.testing, args.group_by)
    if within and (args.group_by != "model_xp" or args.group_column != "sweep_id" or args.same_group_ok):
        print("--group-by, --group-column and --same-group-ok choose groups: pass them with --testing groups.")
        return 1

    candidates = load_candidates(
        args.sources,
        group_column=args.group_column,
        group_by=group_by,
    )
    if len(candidates) < 2:
        print("Need at least two playable clips across the given sweeps / stream sessions.")
        return 1

    existing = load_preferences(args.store)
    already = judged_pair_keys(existing, question=question.id)
    pairs = select_pairs(
        candidates,
        count=args.pairs,
        seed=args.seed,
        cross_group=not (within or args.same_group_ok),
        within_group=within,
        vary=knob,
        exclude_keys=already,
        target_metric=args.target_metric or None,
        disagree=disagree or None,
        target_quantile=args.target_quantile,
        context_window=args.context_window if contextual else None,
        allow_same_group_fallback=not contextual,
    )
    if not pairs:
        if contextual:
            print(
                "No eligible shared-context pairs remain. Add stream sessions or use a wider "
                "--context-window (maximum 3); pairs must be in one session, 1–3 playable "
                "positions apart, share at least five previous-eight rounds, and satisfy the targeting."
            )
        elif args.target_metric or disagree:
            print(
                "No pairs matched the targeting — the pool has no unjudged pairs that "
                "separate on it. Widen --target-quantile, add sources, or drop the targeting."
            )
        elif knob == "model":
            print("No new pairs: no two unjudged clips from different models share every knob.")
        elif knob:
            print(f"No new pairs: no two unjudged clips share a model and every knob but {knob}, which must differ.")
        elif within:
            print("No new pairs: no two unjudged clips share a model and knobs. Add sources with longer runs.")
        else:
            print("No new pairs to judge — every eligible pair is already in the store.")
        return 0

    groups = sorted({candidate.group for candidate in candidates})
    terminal = TerminalSession()
    requested_tui = getattr(args, "tui", None)
    use_tui = terminal.is_interactive() if requested_tui is None or requested_tui else False
    sidecar = {
        "sources": [str(source) for source in args.sources],
        "seed": args.seed,
        "testing": args.testing,
        "group_by": group_by,
        "group_column": args.group_column,
        "cross_group": not (within or args.same_group_ok),
        "target_metric": args.target_metric or None,
        "disagree": disagree or None,
        "target_quantile": args.target_quantile,
        "candidates": len(candidates),
        "groups": groups,
        "context_window": args.context_window if contextual else None,
    }
    audio_context = ContextAudioPreparer() if contextual else nullcontext()
    try:
        with audio_context as preparer:
            runtime_pairs = preparer.prepare(pairs) if contextual else pairs
            if use_tui:
                from .ab import run_ab_tui

                print("Blind A/B session — nothing identifies a clip until the reveal. Press ? for keys.\n")
                run_ab_tui(
                    runtime_pairs,
                    store_path=args.store,
                    question=question.id,
                    listener=args.listener,
                    seconds=args.seconds if args.seconds > 0 else None,
                    session_id=new_session_id(),
                    context={"source": "score ab", "testing": args.testing},
                    sidecar=sidecar,
                    metric_lookup={candidate.path: candidate.metrics for candidate in candidates},
                    terminal=terminal,
                )
            else:
                _run_line_ab_session(
                    args,
                    runtime_pairs,
                    candidates=candidates,
                    groups=groups,
                    already=already,
                    question=question,
                    disagree=disagree,
                    sidecar=sidecar,
                    run_ab_session=run_ab_session,
                    new_session_id=new_session_id,
                )
    except ContextAudioError as exc:
        print(f"Shared-context audio preparation failed: {exc}")
        return 1
    return 0


def _run_line_ab_session(
    args: argparse.Namespace,
    pairs: Any,
    *,
    candidates: Any,
    groups: list[str],
    already: set[str],
    question: Any,
    disagree: list[str],
    sidecar: dict[str, Any],
    run_ab_session: Any,
    new_session_id: Any,
) -> None:
    grouping = "groups" if args.testing == "groups" else "conditions (model + knobs)"
    print(f"{len(candidates)} clips across {len(groups)} {grouping}; {len(already)} '{question.id}' pairs judged.")
    if args.target_metric:
        print(f"Targeted on {args.target_metric}: pairs in the top {1 - args.target_quantile:.0%} of |Δ|.")
    if disagree:
        print(f"Targeted on {disagree[0]} vs {disagree[1]} disagreement.")
    if args.target_metric or disagree:
        print("  Caveat: high-contrast pairs usually differ on other axes too — evidence, not isolation.")
    print("Listening blind: which group each clip came from is hidden until you are done.\n")

    run_ab_session(
        pairs,
        store_path=args.store,
        question=question.id,
        listener=args.listener,
        seconds=args.seconds if args.seconds > 0 else None,
        session_id=new_session_id(),
        context={"source": "score ab", "testing": args.testing},
        sidecar=sidecar,
    )


def cmd_judge_label(args: argparse.Namespace) -> int:
    from contextlib import nullcontext

    from ..instrument.stream.tui import TerminalSession
    from .ab_context import ContextAudioError, FreshAudioPreparer
    from .label import (
        get_alarm,
        labeled_identities,
        load_label_candidates,
        load_labels,
        select_label_candidates,
        with_playback_path,
    )
    from .label_session import run_label_session
    from .preference import new_session_id

    alarm = get_alarm(args.alarm)
    if not _resolve_session_sources(args):
        return 1
    non_streams = [
        str(source) for source in args.sources if not (Path(source).expanduser() / "performance.jsonl").exists()
    ]
    if alarm.stream_only and non_streams:
        print(
            f"'{alarm.id}' requires stream-session sources containing performance.jsonl; "
            f"other sessions or invalid sources are not supported: {', '.join(non_streams)}"
        )
        return 1
    candidates = load_label_candidates(args.sources, group_column=args.group_column)
    existing = load_labels(args.store)
    try:
        selections = select_label_candidates(
            candidates,
            alarm_id=alarm.id,
            count=args.clips,
            seed=args.seed,
            exclude=labeled_identities(existing),
        )
    except ValueError as exc:
        print(f"Cannot build '{alarm.id}' labeling session: {exc}")
        return 1

    terminal = TerminalSession()
    requested_tui = getattr(args, "tui", None)
    use_tui = terminal.is_interactive() if requested_tui is None or requested_tui else False
    unique_strata: dict[int, dict[str, Any]] = {}
    for selection in selections:
        unique_strata.setdefault(
            selection.stratum_index,
            {
                "index": selection.stratum_index,
                "label": selection.stratum_label,
                "population": selection.stratum_population,
                "offered": selection.stratum_offered,
                "inclusion_probability": selection.inclusion_probability,
            },
        )
    sidecar = {
        "sources": [str(source) for source in args.sources],
        "seed": args.seed,
        "requested_clips": args.clips,
        "eligible_candidates": sum(item["population"] for item in unique_strata.values()),
        "sampling_frame_id": selections[0].sampling_frame_id,
        "strata": list(unique_strata.values()),
    }

    needs_temporary = any(selection.candidate.source_type == "stream" for selection in selections)
    audio_context = FreshAudioPreparer() if needs_temporary else nullcontext()
    try:
        with audio_context as preparer:
            runtime = []
            for selection in selections:
                if selection.candidate.source_type == "stream":
                    playback_path = preparer.prepare_fresh_wav(
                        selection.candidate.path,
                        selection.candidate.prompt_duration_seconds,
                    )
                    runtime.append(with_playback_path(selection, playback_path))
                else:
                    runtime.append(selection)
            run_id = new_session_id()
            if use_tui:
                from .label_tui import run_label_tui

                print("Blind alarm labeling — predictor values stay hidden until the reveal. Press ? for keys.\n")
                run_label_tui(
                    runtime,
                    store_path=args.store,
                    listener=args.listener,
                    seconds=args.seconds if args.seconds > 0 else None,
                    session_id=run_id,
                    sidecar=sidecar,
                    terminal=terminal,
                )
            else:
                print(
                    f"{len(runtime)} clips for '{alarm.id}', balanced across "
                    f"{len(alarm.strata)} strata. Predictor values are hidden until completion.\n"
                )
                run_label_session(
                    runtime,
                    store_path=args.store,
                    listener=args.listener,
                    seconds=args.seconds if args.seconds > 0 else None,
                    session_id=run_id,
                    sidecar=sidecar,
                )
    except ContextAudioError as exc:
        print(f"Fresh-audio preparation failed: {exc}")
        return 1
    return 0


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def cmd_judge_labels(args: argparse.Namespace) -> int:
    from .label import analyze_detection, load_labels

    if args.min_labels < 1:
        print("--min-labels must be at least 1.")
        return 1
    records = load_labels(args.store)
    if not records:
        print(f"No alarm labels recorded in {args.store}. Run 'judge label' first.")
        return 1
    requested = (
        [args.alarm]
        if args.alarm
        else [alarm for alarm in alarm_ids() if any(record.alarm_id == alarm for record in records)]
    )
    reports = [
        analyze_detection(
            records,
            alarm_id=alarm,
            min_labels=args.min_labels,
            session_id=args.session_id or None,
        )
        for alarm in requested
    ]
    reports = [report for report in reports if report.labels]
    if not reports:
        qualifier = f" in session {args.session_id}" if args.session_id else ""
        print(f"No matching labels{qualifier} in {args.store}.")
        return 1
    payload = {"schema": "macchina_ricordi.judging.label.v1", "reports": [report.to_dict() for report in reports]}
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for report in reports:
        print(f"{report.alarm_id}: {report.labels} labels ({report.decided} decided, {report.unsure} unsure)")
        for frame in report.frames:
            recommendation = (
                f"recommended {frame.recommended_threshold:.6g}"
                if frame.recommended_threshold is not None
                else "recommendation suppressed"
            )
            print(
                f"  session {frame.session_id} · frame {frame.sampling_frame_id} · "
                f"{frame.decided} decided · unsure {frame.unsure_weight_share:.1%} weighted · {recommendation}"
            )
            print("    threshold   TP  FP  TN  FN   precision recall   F1  reference")
            for point in frame.points:
                print(
                    f"    {point.threshold:>9.5g} {point.raw_tp:>4} {point.raw_fp:>3} "
                    f"{point.raw_tn:>3} {point.raw_fn:>3}   "
                    f"{_format_optional(point.weighted_precision):>9} "
                    f"{_format_optional(point.weighted_recall):>6} "
                    f"{_format_optional(point.weighted_f1):>6}  "
                    f"{'yes' if point.is_reference else ''}"
                )
        if len(report.frames) > 1 and report.raw_aggregate is not None:
            print("  aggregate: raw labeled sample only; no combined corpus weighting or threshold recommendation")
            print("    threshold   TP  FP  TN  FN   precision recall   F1  reference")
            for point in report.raw_aggregate.points:
                print(
                    f"    {point.threshold:>9.5g} {point.raw_tp:>4} {point.raw_fp:>3} "
                    f"{point.raw_tn:>3} {point.raw_fn:>3}   "
                    f"{_format_optional(point.raw_precision):>9} "
                    f"{_format_optional(point.raw_recall):>6} "
                    f"{_format_optional(point.raw_f1):>6}  "
                    f"{'yes' if point.is_reference else ''}"
                )
        print()
    if args.output:
        print(f"Saved → {args.output}")
    return 0


def cmd_judge_prefs(args: argparse.Namespace) -> int:
    from ..scoring.batch import read_metric_lookup
    from .preference import (
        filter_by_question,
        fit_preference_weights,
        load_preferences,
        metric_agreement,
        recorded_questions,
        save_preference_model,
        structure_weights_from_model,
    )

    records = load_preferences(args.store)
    if not records:
        print(f"No preferences recorded yet in {args.store}. Run 'judge ab' first.")
        return 1

    if args.question:
        records = filter_by_question(records, args.question)
        if not records:
            print(f"No '{args.question}' judgements in {args.store}.")
            return 1

    decided = [record for record in records if record.winner in ("a", "b")]
    counts = ", ".join(
        f"{question}: {len(filter_by_question(records, question))}" for question in recorded_questions(records)
    )
    print(f"{len(records)} judgements ({len(decided)} decided, {len(records) - len(decided)} ties) — {counts}\n")

    if not args.metrics_csv:
        print("Pass --metrics-csv (output of 'score rescore') to analyse metric agreement.")
        return 0

    lookup = read_metric_lookup(args.metrics_csv)
    metrics = args.metrics.split(",") if args.metrics else list(PREFERENCE_METRICS)

    # One table per question: an attribute verdict is evidence about its own family only,
    # so pooling them into a single agreement column would report a number about nothing.
    agreements: list[Any] = []
    for question in recorded_questions(records):
        # The default list is the composite's; an attribute question also gets its own
        # family, which is the whole reason it was asked. Metrics absent from the scored
        # CSV drop out on their own.
        family = [metric for metric in get_question(question).metrics if metric not in metrics]
        rows = metric_agreement(records, lookup, metrics=metrics + family, min_pairs=args.min_pairs, question=question)
        agreements.extend(rows)
        if args.json or not rows:
            continue
        print(f"{question} — {len(filter_by_question(decided, question))} decided")
        print(f"{'metric':<28} {'pairs':>6} {'agree':>7} {'z':>7}  verdict")
        print("-" * 62)
        for item in rows:
            print(f"{item.metric:<28} {item.pairs:>6} {item.agreement:>7.3f} {item.z_score:>7.2f}  {item.verdict}")
        print()

    if args.json:
        payload: dict[str, Any] = {
            "judgements": len(records),
            "decided": len(decided),
            "agreement": [item.to_dict() for item in agreements],
        }
    else:
        payload = {}

    # Taste is defined by `better` alone; attribute questions validate axes.
    preference_records = filter_by_question(records, "better") if args.fit else []
    if args.fit and not preference_records:
        print("No 'better' judgements to fit a composite from.")

    if preference_records:
        model = fit_preference_weights(preference_records, lookup, metrics, l2=args.l2)
        weights = structure_weights_from_model(model)
        if args.json:
            payload["model"] = model.to_dict()
            payload["structure_weights"] = weights
        else:
            print(f"\nfitted composite ({model.pairs} usable pairs, accuracy {model.accuracy:.3f}):")
            for metric, weight in zip(model.metrics, model.weights, strict=True):
                print(f"  {metric:<28} {weight:+.3f}")
            if model.dropped_metrics:
                print(f"  (dropped, too few scored pairs: {', '.join(model.dropped_metrics)})")
            print("\nstructure_score weights:")
            for key, value in weights.items():
                print(f"  {key:<28} {value:.3f}")
        if args.output:
            saved = save_preference_model(model, path=args.output, structure_weights=weights)
            if not args.json:
                print(f"\nSaved → {saved}")

    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_judge_witness(args: argparse.Namespace) -> int:
    from .store import WitnessRecord, append_witness
    from .witness import JudgeComponents, create_judge_backend, judge_audio

    paths: list[Path] = []
    for raw in args.paths:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            paths.extend(sorted(iter_audio_files(candidate, recursive=True, extensions=DEFAULT_EXTENSIONS)))
        elif candidate.exists():
            paths.append(candidate)
    if not paths:
        print("No audio files found.")
        return 1
    if args.limit > 0:
        paths = paths[: args.limit]

    try:
        backend = create_judge_backend(
            backend=args.backend,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            quantization=args.quantization,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Captioner backend unavailable: {exc}")
        return 1

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        components = judge_audio(
            path,
            paraphrases=args.paraphrases,
            captioner_backend=backend,
            max_new_tokens=args.max_new_tokens,
            max_audio_seconds=args.max_audio_seconds,
        )
        rows.append({"path": str(path), **components.to_row()})
        if args.record:
            append_witness(
                WitnessRecord(
                    path=str(path),
                    backend=args.backend,
                    model_id=args.model_id,
                    score=components.judge_score,
                    ratings={
                        "shape": components.judge_shape,
                        "coherence": components.judge_coherence,
                        "intent": components.judge_intent,
                        "engagement": components.judge_engagement,
                    },
                    answered=components.judge_answered,
                    asked=components.judge_asked,
                    paraphrases=args.paraphrases,
                    error=components.judge_error,
                    answers=components.answers,
                ),
                path=args.store,
            )
        if not args.quiet:
            print(
                f"[{index}/{len(paths)}] {path.name}  score={components.judge_score:.3f} "
                f"shape={components.judge_shape:.2f} coh={components.judge_coherence:.2f} "
                f"intent={components.judge_intent:.2f} eng={components.judge_engagement:.2f}"
                + (f"  ERROR {components.judge_error}" if components.judge_error else ""),
                flush=True,
            )

    if args.record and not args.json:
        print(f"\nRecorded {len(rows)} ratings in {args.store}")
    if args.output:
        fieldnames = ["path", *JudgeComponents.fieldnames()]
        write_csv(Path(args.output).expanduser(), fieldnames, rows)
        print(f"\nJudged {len(rows)} files → {args.output}")
    if args.json:
        print(json.dumps(rows, indent=2))
    return 0


PREFERENCE_METRICS = (
    "structure_score",
    "dev_top2",
    "dev_harmonic",
    "dev_timbre",
    "dev_dynamics",
    "dev_rhythm",
    "ssm_novelty",
    "ssm_return",
    "ssm_mean",
    "boundaries_per_minute",
    "arc_coherence_score",
    "timbral_coherence",
    "score",
)


def cmd_judge_log(args: argparse.Namespace) -> int:
    from .store import load_verdicts, verdict_counts

    verdicts = load_verdicts(
        preferences_path=args.preferences,
        labels_path=args.labels,
        witness_path=args.witness,
        impressions_path=args.impressions,
        kinds=args.kind or VERDICT_KINDS,
    )
    if args.take:
        verdicts = [v for v in verdicts if any(args.take in path for path in v.paths)]
    if args.session:
        verdicts = [v for v in verdicts if v.session_id == args.session]
    counts = verdict_counts(verdicts)
    shown = verdicts[-args.limit :] if args.limit > 0 else verdicts
    if args.json:
        print(json.dumps({"counts": counts, "verdicts": [v.to_dict() for v in shown]}, indent=2))
        return 0
    if not verdicts:
        print("no verdicts yet: `judge ab`, `judge label`, `judge witness` and the GUI's model page record them")
        return 0
    for verdict in shown:
        takes = "  vs  ".join(Path(path).name for path in verdict.paths) or verdict.detail.get("note", "")[:60]
        when = verdict.created_at[:19]
        print(f"{when:<19}  {verdict.kind:<10}  {verdict.question:<22}  {verdict.answer:<6}  {takes}")
    print()
    for kind, questions in counts.items():
        listed = ", ".join(f"{question} {n}" for question, n in sorted(questions.items()))
        print(f"{kind}: {sum(questions.values())} ({listed})")
    if len(shown) < len(verdicts):
        print(f"showing the newest {len(shown)} of {len(verdicts)}; --limit 0 shows all")
    return 0


def register_judge_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    judge = subparsers.add_parser(
        "judge",
        help="Verdicts: blind A/B, labels, preferences, a captioner witness",
        description="Judgements about takes: your blind A/B verdicts and labels, what they say about the metrics, "
        "and a captioner's holistic ratings.",
        epilog="""examples:
  python -m macchina_ricordi.cli judge ab artifacts/streams/<a>/rounds artifacts/streams/<b>/rounds
  python -m macchina_ricordi.cli judge prefs --metrics-csv artifacts/scores/rescore.csv
  python -m macchina_ricordi.cli judge witness clip.wav --json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    judge_subparsers = judge.add_subparsers(dest="judge_command", required=True)

    score_ab = judge_subparsers.add_parser(
        "ab",
        help="Blind A/B listening session that records which clip you preferred",
        description=(
            "Play pairs of clips blind and record the verdict. Sources are sessions of any kind "
            "(sweeps, streams, generations, duets, variations), by folder or id; they are read but "
            "never written. By default (--testing metric) both clips of a pair come from one model at the "
            "same knobs, so a verdict says whether a metric follows your ear and nothing else differs; "
            "--testing groups draws pairs across groups (models, sessions, buckets) instead. Sides are "
            "shuffled, and pairs already judged for the same question are skipped. The resulting store "
            "is what 'judge prefs' fits the metrics against."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --target-metric arc_coherence_score
  python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --disagree arc_coherence_score,score
  python -m macchina_ricordi.cli judge ab artifacts/sweeps/uncond_* --question harder --seconds 8 --seed 7
  python -m macchina_ricordi.cli judge ab
  artifacts/sweeps/patient_staircase_ep50/sweep_* --testing parameter:temperature
  python -m macchina_ricordi.cli judge ab artifacts/sweeps/patient_staircase_ep*/sweep_* --testing model
  python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --question newer --pairs 40 \\
      --context-window 3 --disagree novelty_raw,novelty_whitened
  python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --testing groups --group-by session \\
      --question more_developed""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_ab.add_argument(
        "sources", nargs="+", help="Sessions to draw clips from: folders or ids (`session list`), any kind"
    )
    score_ab.add_argument("--pairs", type=int, default=20, help="How many pairs to present")
    score_ab.add_argument("--seed", type=int, default=0, help="Pair selection seed")
    score_ab.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Seconds to play per A/B clip (0 = whole clip; a context reel always plays in full)",
    )
    score_ab.add_argument("--store", default=str(DEFAULT_PREFERENCE_STORE), help="Preference JSONL store")
    score_ab.add_argument("--listener", default="me", help="Who is listening")
    score_ab.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help=f"What to ask ({', '.join((*question_ids(type=PAIRWISE), *question_ids(type=PAIRWISE_CONTEXT)))})",
    )
    score_ab.add_argument(
        "--context-window",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Maximum generated-position gap for 'newer' pairs (used only by 'newer')",
    )
    score_ab.add_argument(
        "--testing",
        type=_testing,
        default="metric",
        help="metric: both clips from one model at the same knobs; parameter:<knob> (e.g. parameter:temperature): "
        "the same, but that knob differs; model: the same knobs, different models; groups: pairs across --group-by "
        "groups",
    )
    score_ab.add_argument("--group-column", default="sweep_id", help="index.csv column that defines a group (sweeps)")
    score_ab.add_argument(
        "--group-by",
        default="model_xp",
        help="With --testing groups, what defines a group for stream rounds: model_xp, session, or bucket:METRIC",
    )
    score_ab.add_argument(
        "--target-metric",
        default="",
        help="Prefer pairs this metric strongly separates, so each verdict is informative about it",
    )
    score_ab.add_argument(
        "--disagree",
        default="",
        help="Two comma-separated metrics; prefer pairs where they rank the sides oppositely",
    )
    score_ab.add_argument(
        "--target-quantile",
        type=float,
        default=0.5,
        help="Targeting cut: keep pairs above this quantile of the margin (0.5 = top half)",
    )
    score_ab.add_argument(
        "--same-group-ok",
        action="store_true",
        help="With --testing groups, allow pairs from the same group (default: cross-group only)",
    )
    tui_mode = score_ab.add_mutually_exclusive_group()
    tui_mode.add_argument(
        "--tui",
        dest="tui",
        action="store_true",
        help="Use the single-keystroke terminal interface when a TTY is available",
    )
    tui_mode.add_argument(
        "--line",
        dest="tui",
        action="store_false",
        help="Force the line-based interface",
    )
    score_ab.set_defaults(tui=None)
    score_ab.set_defaults(func=cmd_judge_ab)

    score_label = judge_subparsers.add_parser(
        "label",
        help="Blind single-clip labeling for threshold alarms",
        description=(
            "Sample one alarm's range in three strata, play each clip blind, and append "
            "yes/no/unsure answers to the separate label store."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge label artifacts/streams/2026* --alarm timbral_coherence
  python -m macchina_ricordi.cli judge label artifacts/streams/2026* --alarm horizon --clips 30 --line""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_label.add_argument(
        "sources",
        nargs="+",
        help="Sessions to draw clips from: folders or ids (`session list`), any kind; stream-only alarms need streams",
    )
    score_label.add_argument("--alarm", choices=alarm_ids(), default="timbral_coherence", help="Alarm to validate")
    score_label.add_argument("--clips", type=int, default=30, help="How many clips to label (minimum 3)")
    score_label.add_argument("--seed", type=int, default=0, help="Stratified sampling seed")
    score_label.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Seconds of fresh audio to play (0 = whole fresh continuation)",
    )
    score_label.add_argument("--store", default=str(DEFAULT_LABEL_PATH), help="Label JSONL store")
    score_label.add_argument("--listener", default="me", help="Who is listening")
    score_label.add_argument("--group-column", default="sweep_id", help="index.csv group column for sweep sources")
    label_tui_mode = score_label.add_mutually_exclusive_group()
    label_tui_mode.add_argument("--tui", dest="tui", action="store_true", help="Use the single-keystroke TUI")
    label_tui_mode.add_argument("--line", dest="tui", action="store_false", help="Force line-based input")
    score_label.set_defaults(tui=None, func=cmd_judge_label)

    score_labels = judge_subparsers.add_parser(
        "labels",
        help="Report precision/recall and fit alarm thresholds from labels",
        description=(
            "Sweep an alarm threshold over yes/no labels, showing raw confusion counts and "
            "sampling-weighted precision, recall, and F1 per labeling session."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge labels --alarm timbral_coherence
  python -m macchina_ricordi.cli judge labels --alarm silence_gate --json --output /tmp/silence-labels.json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_labels.add_argument("--store", default=str(DEFAULT_LABEL_PATH), help="Label JSONL store")
    score_labels.add_argument("--alarm", choices=alarm_ids(), default="", help="One alarm (default: all recorded)")
    score_labels.add_argument("--session-id", default="", help="Restrict analysis to one labeling session")
    score_labels.add_argument("--min-labels", type=int, default=8, help="Minimum decided labels for a recommendation")
    score_labels.add_argument("--json", action="store_true", help="Print the complete report as JSON")
    score_labels.add_argument("--output", default="", help="Optional JSON report path")
    score_labels.set_defaults(func=cmd_judge_labels)

    score_prefs = judge_subparsers.add_parser(
        "prefs",
        help="Report which metrics agree with your blind A/B judgements, and fit weights",
        description=(
            "Compare recorded blind preferences against metric values. Reports per-metric "
            "agreement (0.5 = coin flip, below 0.5 = inverted) and optionally fits a "
            "logistic composite whose weights come from the ear rather than a guess."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge prefs --metrics-csv artifacts/scores/rescore.csv
  python -m macchina_ricordi.cli judge prefs --metrics-csv artifacts/scores/rescore.csv --fit --output prefs.json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_prefs.add_argument("--store", default=str(DEFAULT_PREFERENCE_STORE), help="Preference JSONL store")
    score_prefs.add_argument("--metrics-csv", default="", help="Scored CSV from 'score rescore'")
    score_prefs.add_argument("--metrics", default="", help="Comma-separated metric subset")
    score_prefs.add_argument("--question", default="", help="Analyse one question only (default: all recorded)")
    score_prefs.add_argument("--min-pairs", type=int, default=8, help="Pairs below this are reported as insufficient")
    score_prefs.add_argument("--fit", action="store_true", help="Fit composite weights from the preferences")
    score_prefs.add_argument("--l2", type=float, default=1.0, help="Ridge strength for the fit")
    score_prefs.add_argument("--output", default="", help="Write the fitted model + weights to JSON")
    score_prefs.add_argument("--json", action="store_true", help="Output as JSON")
    score_prefs.set_defaults(func=cmd_judge_prefs)

    score_judge = judge_subparsers.add_parser(
        "witness",
        help="Rate clips with a captioner witness (semantic axis, independent of DSP)",
        description=(
            "Ask a captioner witness to rate shape, coherence, intent and engagement from "
            "1 to 5 per clip. Questions are holistic by design: this project has "
            "established that the witness's claims about timing and order are unreliable."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge witness artifacts/sweeps/uncond_deep50_*/combo_000 --output judge.csv
  python -m macchina_ricordi.cli judge witness clip.wav --paraphrases 2 --json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    score_judge.add_argument("paths", nargs="+", help="Audio files or directories")
    score_judge.add_argument("--backend", default="audio-flamingo-3", help="Captioner backend")
    score_judge.add_argument("--model-id", default="", help="Override the backend model id")
    score_judge.add_argument("--device", default="auto", help="Device for the captioner")
    score_judge.add_argument("--dtype", default="auto", help="Captioner dtype")
    score_judge.add_argument("--quantization", default="auto", help="Captioner quantization")
    score_judge.add_argument("--paraphrases", type=int, default=1, help="Phrasings per axis (1 or 2)")
    score_judge.add_argument("--max-new-tokens", type=int, default=24, help="Generation budget per answer")
    score_judge.add_argument("--max-audio-seconds", type=float, default=120.0, help="Audio truncation")
    score_judge.add_argument("--limit", type=int, default=0, help="Judge at most N files (0 = all)")
    score_judge.add_argument("--output", default="", help="Output CSV path")
    score_judge.add_argument("--json", action="store_true", help="Output as JSON")
    score_judge.add_argument("--quiet", action="store_true", help="Suppress per-file prints")
    score_judge.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append each rating to the witness log beside the other verdicts (default: on)",
    )
    score_judge.add_argument("--store", default=str(DEFAULT_WITNESS_PATH), help="Witness log (JSONL)")
    score_judge.set_defaults(func=cmd_judge_witness)

    log = judge_subparsers.add_parser(
        "log",
        help="Every verdict, of every kind: A/B, labels, witness ratings and model impressions",
        description=(
            "Read the four verdict logs in artifacts/preferences/ (A/B preferences, alarm labels, witness ratings, "
            "model impressions) "
            "as one list, oldest first, with counts per kind and question."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli judge log
  python -m macchina_ricordi.cli judge log --kind ab --take stream_20261005 --limit 20
  python -m macchina_ricordi.cli judge log --json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    log.add_argument("--kind", choices=VERDICT_KINDS, action="append", help="Only this kind (repeatable)")
    log.add_argument("--take", default="", help="Only verdicts on a clip whose path contains this")
    log.add_argument("--session", default="", help="Only verdicts from this judging session id")
    log.add_argument("--limit", type=int, default=50, help="Show the newest N (0 = all)")
    log.add_argument("--preferences", default=str(DEFAULT_PREFERENCE_STORE), help="A/B preference log")
    log.add_argument("--labels", default=str(DEFAULT_LABEL_PATH), help="Alarm label log")
    log.add_argument("--witness", default=str(DEFAULT_WITNESS_PATH), help="Witness rating log")
    log.add_argument("--impressions", default=str(DEFAULT_IMPRESSION_PATH), help="Model impression log")
    log.add_argument("--json", action="store_true", help="Output as JSON")
    log.set_defaults(func=cmd_judge_log)
