from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..augmentation.core import DEFAULT_OUTPUT_FORMAT, OUTPUT_FORMATS, write_audio
from ..cli.common import MacchinaRicordiHelpFormatter
from .core import (
    ANALYSIS_SR,
    DEFAULT_BAR_COUNTS,
    DEFAULT_BEATS_PER_BAR,
    DEFAULT_FADE_MS,
    DEFAULT_MIN_BREATH_SECONDS,
    DEFAULT_PRE_ROLL_SECONDS,
    DEFAULT_TARGET_SECONDS,
    DEFAULT_VOICE_THRESHOLD_DB,
    LAYERS,
    analyze_loops,
    candidate_label,
    parse_bar_counts,
    render_loop,
    render_stuck_voice,
    repeats_for,
    shortlist_candidates,
)
from .stems import DEFAULT_DEMUCS_MODEL, DEMUCS_DEVICES, separate_vocals

REPORT_FILENAME = "loop_report.json"
CANDIDATES_DIRNAME = "candidates"
STEMS_DIRNAME = "stems"
DEFAULT_TOP = 8
DEFAULT_OUTPUT_SR = 44_100
_QUIET_VOCALS_DB = -50.0


def _load(path: str | Path, *, sr: int, mono: bool) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(str(path), sr=sr, mono=mono)
    return waveform


def _format_time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)}:{rest:05.2f}"


def _remove_previous_auditions(report_path: Path) -> None:
    """Delete only the audition files a previous run listed, so a re-run never leaves stale candidates."""
    if not report_path.is_file():
        return
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for entry in previous.get("candidates", []):
        audition = entry.get("audition_path")
        if audition:
            Path(audition).unlink(missing_ok=True)


def cmd_loop_find(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if args.top < 1:
        raise ValueError(f"--top must be >= 1, got {args.top}")
    if not 0.0 <= args.min_voice <= 1.0:
        raise ValueError(f"--min-voice must be between 0 and 1, got {args.min_voice}")
    bar_counts = parse_bar_counts(args.bars)
    output_dir = Path(args.output_dir).expanduser().resolve()
    stems_dir = Path(args.stems_dir).expanduser().resolve() if args.stems_dir else output_dir / STEMS_DIRNAME

    stems = separate_vocals(
        audio_path,
        stems_dir,
        python=args.demucs_python,
        model=args.demucs_model,
        device=args.device,
        force=args.force_separate,
    )
    if stems.reused:
        print(f"Reusing stems in {stems_dir}")

    analysis = analyze_loops(
        _load(stems.vocals, sr=ANALYSIS_SR, mono=True),
        _load(stems.instrumental, sr=ANALYSIS_SR, mono=True),
        ANALYSIS_SR,
        bpm=args.bpm,
        beats_per_bar=args.beats_per_bar,
        bar_counts=bar_counts,
        threshold_db=args.voice_threshold_db,
        min_breath_seconds=args.min_breath_ms / 1000.0,
        pre_roll_seconds=args.pre_roll_ms / 1000.0,
    )
    shortlist = shortlist_candidates(
        analysis.candidates,
        top=args.top,
        min_spacing_seconds=analysis.bar_seconds,
        min_voice_fraction=args.min_voice,
        max_gap_seconds=args.max_gap_ms / 1000.0 if args.max_gap_ms is not None else None,
        allow_overlap=args.overlap,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    _remove_previous_auditions(report_path)
    audition_paths: list[str | None] = [None] * len(shortlist)
    if args.render and shortlist:
        mix = _load(audio_path, sr=args.output_sr, mono=args.mono)
        for index, candidate in enumerate(shortlist):
            name = f"{index + 1:02d}_{candidate_label(candidate.start_seconds, candidate.bars)}.{args.format}"
            path = output_dir / CANDIDATES_DIRNAME / name
            looped = render_loop(
                mix,
                args.output_sr,
                start_seconds=candidate.start_seconds,
                length_seconds=candidate.length_seconds,
                repeats=repeats_for(args.seconds, candidate.length_seconds),
                fade_ms=args.fade_ms,
            )
            write_audio(path, looped, args.output_sr)
            audition_paths[index] = str(path)

    report = {
        "source_path": str(audio_path),
        "stems_dir": str(stems_dir),
        "stems": stems.to_dict(),
        "demucs_model": args.demucs_model,
        "tempo_bpm": round(analysis.tempo_bpm, 4),
        "bpm_source": analysis.bpm_source,
        "beats_per_bar": analysis.beats_per_bar,
        "bar_seconds": round(analysis.bar_seconds, 6),
        "duration_seconds": round(analysis.duration_seconds, 3),
        "vocal_reference_db": round(analysis.activity.reference_db, 2),
        "vocal_onset_count": len(analysis.onsets),
        "candidate_count": len(analysis.candidates),
        "settings": {
            "bar_counts": list(bar_counts),
            "voice_threshold_db": args.voice_threshold_db,
            "min_breath_ms": args.min_breath_ms,
            "pre_roll_ms": args.pre_roll_ms,
            "min_voice": args.min_voice,
            "max_gap_ms": args.max_gap_ms,
            "overlap": args.overlap,
            "fade_ms": args.fade_ms,
            "target_seconds": args.seconds,
            "output_sr": args.output_sr,
            "mono": args.mono,
            "format": args.format,
        },
        "candidates": [
            {
                "rank": index + 1,
                **candidate.to_dict(),
                "end_seconds": round(candidate.start_seconds + candidate.length_seconds, 4),
                "repeats": repeats_for(args.seconds, candidate.length_seconds),
                "audition_path": audition_paths[index],
            }
            for index, candidate in enumerate(shortlist)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Tempo {analysis.tempo_bpm:.2f} BPM ({analysis.bpm_source}), bar {analysis.bar_seconds:.4f}s; "
        f"{len(analysis.onsets)} vocal phrase onsets -> {len(analysis.candidates)} candidates"
    )
    if analysis.activity.reference_db < _QUIET_VOCALS_DB:
        print(
            f"Warning: the vocal stem is nearly silent ({analysis.activity.reference_db:.0f} dBFS); is there a voice?"
        )
    if not shortlist:
        print("No loop candidates. Try a lower --min-breath-ms or --min-voice, or a higher --voice-threshold-db.")
    else:
        print(
            f"{'#':>3}  {'start':>8}  {'bars':>4}  {'voice':>5}  {'voice@end':>9}  {'breath':>7}  {'gap':>5}  audition"
        )
        for index, candidate in enumerate(shortlist):
            audition = audition_paths[index]
            shown = Path(audition).relative_to(output_dir).as_posix() if audition else "-"
            print(
                f"{index + 1:>3}  {_format_time(candidate.start_seconds):>8}  {candidate.bars:>4}  "
                f"{candidate.voice_fraction:>5.0%}  {candidate.end_voice_db:>6.0f} dB  "
                f"{candidate.breath_seconds:>6.2f}s  {candidate.longest_gap_seconds:>4.1f}s  {shown}"
            )
    print(f"Report: {report_path}")
    return 0


def cmd_loop_render(args: argparse.Namespace) -> int:
    report_path = Path(args.report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.suffix.lower().lstrip(".") not in OUTPUT_FORMATS:
        raise ValueError(f"Output must end in one of {OUTPUT_FORMATS}, got {output_path.name!r}")
    bar_seconds = float(report["bar_seconds"])

    if args.start is not None:
        if args.bars is None:
            raise ValueError("--start needs --bars")
        pick, start, bars = None, float(args.start), int(args.bars)
    else:
        candidates = report.get("candidates", [])
        pick = args.pick if args.pick is not None else 1
        if not 1 <= pick <= len(candidates):
            raise ValueError(f"--pick must be between 1 and {len(candidates)}, got {pick}")
        chosen = candidates[pick - 1]
        start = float(chosen["start_seconds"])
        bars = int(args.bars if args.bars is not None else chosen["bars"])
    if bars <= 0:
        raise ValueError(f"--bars must be > 0, got {bars}")
    length = bars * bar_seconds
    sr = args.output_sr

    repeats: int | None
    if args.layer == "stuck":
        if args.seconds is not None or args.repeats is not None:
            raise ValueError("The stuck layer runs to the end of the song; drop --seconds/--repeats")
        repeats = None
        audio = render_stuck_voice(
            _load(report["stems"]["vocals"], sr=sr, mono=args.mono),
            _load(report["stems"]["instrumental"], sr=sr, mono=args.mono),
            sr,
            start_seconds=start,
            length_seconds=length,
            fade_ms=args.fade_ms,
        )
    else:
        source = {
            "mix": report["source_path"],
            "vocals": report["stems"]["vocals"],
            "instrumental": report["stems"]["instrumental"],
        }[args.layer]
        target = args.seconds if args.seconds is not None else DEFAULT_TARGET_SECONDS
        repeats = args.repeats if args.repeats is not None else repeats_for(target, length)
        audio = render_loop(
            _load(source, sr=sr, mono=args.mono),
            sr,
            start_seconds=start,
            length_seconds=length,
            repeats=repeats,
            fade_ms=args.fade_ms,
        )

    write_audio(output_path, audio, sr)
    duration = audio.shape[-1] / sr
    sidecar_path = output_path.with_name(f"{output_path.stem}_report.json")
    sidecar = {
        "loop_report": str(report_path),
        "source_path": report["source_path"],
        "output_path": str(output_path),
        "layer": args.layer,
        "pick": pick,
        "start_seconds": start,
        "bars": bars,
        "bar_seconds": bar_seconds,
        "length_seconds": round(length, 6),
        "repeats": repeats,
        "fade_ms": args.fade_ms,
        "output_sr": sr,
        "mono": args.mono,
        "duration_seconds": round(duration, 3),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    loops = f"{repeats} x {length:.4f}s" if repeats is not None else f"voice looping every {length:.4f}s"
    print(f"Wrote {output_path} ({args.layer}, {loops}, {duration:.1f}s)")
    print(f"Report: {sidecar_path}")
    return 0


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fade-ms", type=float, default=DEFAULT_FADE_MS, help="Crossfade at each loop seam")
    parser.add_argument("--output-sr", type=int, default=DEFAULT_OUTPUT_SR, help="Output sample rate")
    parser.add_argument(
        "--mono",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downmix to mono (source channels are kept by default)",
    )


def register_loop_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    loop = subparsers.add_parser(
        "loop",
        help="Find and render voice-phrase loops in a song",
        description=(
            "Find sung phrases that loop cleanly (Demucs vocal split, bar grid from the drums, cuts in the "
            "singer's breaths), then render the one you like."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli loop find song.mp3 artifacts/loops/song
  python -m macchina_ricordi.cli loop render artifacts/loops/song/loop_report.json song_loop.flac --pick 4 --seconds 115
  python -m macchina_ricordi.cli loop render artifacts/loops/song/loop_report.json
  song_stuck.flac --pick 4 --layer stuck""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    loop_subparsers = loop.add_subparsers(dest="loop_command", required=True)

    find = loop_subparsers.add_parser(
        "find",
        help="Rank candidate voice loops and render them for listening",
        description=(
            "Split the song with Demucs (cached in <output_dir>/stems), measure the bar length from the "
            "instrumental, and propose loops that start just before a sung phrase and last a whole number of "
            "bars. Writes candidates/<rank>_<start>_<bars>bar.<format> and loop_report.json. The ranking is a "
            "heuristic shortlist: pick by ear."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    find.add_argument("audio_path", help="Song to loop")
    find.add_argument("output_dir", help="Directory for stems, auditions, and loop_report.json")
    find.add_argument("--bars", default=",".join(map(str, DEFAULT_BAR_COUNTS)), help="Loop lengths to try, in bars")
    find.add_argument("--top", type=int, default=DEFAULT_TOP, help="How many candidates to keep and render")
    find.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_TARGET_SECONDS,
        help="Approximate audition length; each loop is repeated a whole number of times",
    )
    find.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render audition files (--no-render writes only the report)",
    )
    find.add_argument("--bpm", type=float, default=None, help="Skip tempo detection and use this tempo")
    find.add_argument("--beats-per-bar", type=int, default=DEFAULT_BEATS_PER_BAR, help="Beats in one bar")
    find.add_argument(
        "--voice-threshold-db",
        type=float,
        default=DEFAULT_VOICE_THRESHOLD_DB,
        help="Voice counts as present within this many dB of its loud level; raise it for quiet singing",
    )
    find.add_argument(
        "--min-breath-ms",
        type=float,
        default=DEFAULT_MIN_BREATH_SECONDS * 1000.0,
        help="Shortest vocal gap that counts as the start of a new phrase",
    )
    find.add_argument(
        "--pre-roll-ms",
        type=float,
        default=DEFAULT_PRE_ROLL_SECONDS * 1000.0,
        help="Start the loop this long before the phrase, so the first word keeps its attack",
    )
    find.add_argument(
        "--min-voice",
        type=float,
        default=0.0,
        help="Keep only loops where the voice is present at least this fraction of the time (0-1)",
    )
    find.add_argument(
        "--max-gap-ms",
        type=float,
        default=None,
        help="Keep only loops whose longest stretch without voice is at most this long (drops instrumental breaks)",
    )
    find.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow shortlisted loops to share audio (--no-overlap keeps distinct sections only)",
    )
    find.add_argument("--format", choices=OUTPUT_FORMATS, default=DEFAULT_OUTPUT_FORMAT, help="Audition file format")
    _add_output_args(find)
    find.add_argument("--stems-dir", default=None, help="Where stems are cached (default: <output_dir>/stems)")
    find.add_argument("--force-separate", action="store_true", help="Re-run Demucs even if stems exist")
    find.add_argument("--demucs-model", default=DEFAULT_DEMUCS_MODEL, help="Demucs model name")
    find.add_argument(
        "--device",
        choices=DEMUCS_DEVICES,
        default="cpu",
        help="Demucs device; CPU takes about a minute per song and leaves the GPU to training",
    )
    find.add_argument(
        "--demucs-python",
        default=None,
        help="Python with demucs installed; defaults to the repo-managed .venv-musicgen runtime",
    )
    find.set_defaults(func=cmd_loop_find)

    render = loop_subparsers.add_parser(
        "render",
        help="Render one loop from a loop_report.json",
        description=(
            "Render a candidate (--pick, default 1) or your own start (--start with --bars) using the song's "
            "measured bar length. Layers: mix (the song), vocals (voice alone), instrumental (band alone), "
            "stuck (the song plays normally, then the voice loops while the band carries on to the end). "
            "Writes <output>_report.json next to the audio."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    render.add_argument("report", help="loop_report.json written by `loop find`")
    render.add_argument("output_path", help="Output .flac or .wav")
    which = render.add_mutually_exclusive_group()
    which.add_argument("--pick", type=int, default=None, help="Candidate rank from the report (default: 1)")
    which.add_argument("--start", type=float, default=None, help="Loop start in seconds (needs --bars)")
    render.add_argument("--bars", type=int, default=None, help="Loop length in bars (overrides the candidate's)")
    render.add_argument("--layer", choices=LAYERS, default="mix", help="What gets looped")
    length = render.add_mutually_exclusive_group()
    length.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=f"Approximate output length (default: {DEFAULT_TARGET_SECONDS:g}); not for --layer stuck",
    )
    length.add_argument("--repeats", type=int, default=None, help="Exact number of repeats; not for --layer stuck")
    _add_output_args(render)
    render.set_defaults(func=cmd_loop_render)
