from __future__ import annotations

import argparse
from pathlib import Path

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files, parse_extensions
from ..cli.common import MacchinaRicordiHelpFormatter, _get_tqdm, write_csv


def _export_track_segments(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .core import export_track_segments

    return export_track_segments(*args, **kwargs)


def _detect_track_segments(*args, **kwargs):  # type: ignore[no-untyped-def]
    from .core import detect_track_segments

    return detect_track_segments(*args, **kwargs)


def _segment_single_file(
    audio_path: Path,
    *,
    output_dir: Path,
    analysis_sr: int,
    frame_length: int,
    hop_length: int,
    min_silence_seconds: float,
    silence_threshold_db: float,
    min_track_seconds: float,
    output_sr: int,
    mono: bool,
    res_type: str,
    max_silence_fraction: float,
    silence_frame_db: float,
) -> tuple[list[dict[str, object]], int]:
    detected = _detect_track_segments(
        audio_path,
        analysis_sr=analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        min_silence_seconds=min_silence_seconds,
        silence_threshold_db=silence_threshold_db,
        min_track_seconds=min_track_seconds,
        res_type=res_type,
    )
    rows = _export_track_segments(
        audio_path,
        output_dir,
        analysis_sr=analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        min_silence_seconds=min_silence_seconds,
        silence_threshold_db=silence_threshold_db,
        min_track_seconds=min_track_seconds,
        output_sr=output_sr,
        mono=mono,
        res_type=res_type,
        max_silence_fraction=max_silence_fraction,
        silence_frame_db=silence_frame_db,
    )
    dropped = max(0, len(detected) - len(rows))
    return rows, dropped


def _segment_manifest_fieldnames() -> list[str]:
    return [
        "source_path",
        "segment_path",
        "track_index",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
    ]


def cmd_segment_file(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    album_dir = output_dir / audio_path.stem
    rows, dropped = _segment_single_file(
        audio_path,
        output_dir=album_dir,
        analysis_sr=args.analysis_sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        min_silence_seconds=args.min_silence_seconds,
        silence_threshold_db=args.silence_threshold_db,
        min_track_seconds=args.min_track_seconds,
        output_sr=args.output_sr,
        mono=args.mono,
        res_type=args.res_type,
        max_silence_fraction=args.max_silence_fraction,
        silence_frame_db=args.silence_frame_db,
    )

    manifest_path = album_dir / "segments.csv"
    write_csv(manifest_path, _segment_manifest_fieldnames(), rows)
    print(f"Wrote {len(rows)} segments (dropped {dropped} silent) to {album_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


def cmd_segment_dir(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    files = list(
        iter_audio_files(
            input_dir,
            recursive=args.recursive,
            extensions=parse_extensions(args.extensions),
        )
    )

    rows: list[dict[str, object]] = []
    total_dropped = 0
    tqdm = _get_tqdm()
    for audio_path in tqdm(
        files,
        total=len(files),
        desc="Segmenting",
        unit="file",
        disable=args.quiet,
    ):
        relative_parent = audio_path.parent.relative_to(input_dir)
        album_dir = output_dir / relative_parent / audio_path.stem
        file_rows, file_dropped = _segment_single_file(
            audio_path,
            output_dir=album_dir,
            analysis_sr=args.analysis_sr,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            min_silence_seconds=args.min_silence_seconds,
            silence_threshold_db=args.silence_threshold_db,
            min_track_seconds=args.min_track_seconds,
            output_sr=args.output_sr,
            mono=args.mono,
            res_type=args.res_type,
            max_silence_fraction=args.max_silence_fraction,
            silence_frame_db=args.silence_frame_db,
        )
        rows.extend(file_rows)
        total_dropped += file_dropped

    manifest_path = output_dir / "segments.csv"
    write_csv(manifest_path, _segment_manifest_fieldnames(), rows)
    print(f"Wrote {len(rows)} segments (dropped {total_dropped} silent) from {len(files)} source files")
    print(f"Manifest: {manifest_path}")
    return 0


def register_segmentation_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    segment = subparsers.add_parser(
        "segment",
        help="Split long recordings into scorer-friendly track files",
        description="Split long recordings at quiet gaps and export scorer-friendly track WAVs.",
        epilog="""examples:
  python -m macchina_ricordi.cli segment file album.wav library/album
  python -m macchina_ricordi.cli segment dir raw_recordings library/recordings --recursive""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    segment_subparsers = segment.add_subparsers(dest="segment_command", required=True)

    segment_file = segment_subparsers.add_parser(
        "file",
        help="Split a long album file into scorer-friendly track WAVs",
        description="Split one long source file into separate WAV tracks using quiet-gap detection.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    segment_file.add_argument("audio_path", help="Path to a long source audio file")
    segment_file.add_argument("output_dir", help="Where the segments go: they are sources, so a folder under library/")
    segment_file.add_argument("--analysis-sr", type=int, default=8_000, help="Sample rate used for gap detection")
    segment_file.add_argument("--frame-length", type=int, default=4_096, help="RMS frame size for gap detection")
    segment_file.add_argument("--hop-length", type=int, default=1_024, help="Hop length for gap detection")
    segment_file.add_argument(
        "--min-silence-seconds",
        type=float,
        default=1.5,
        help="Minimum quiet gap duration required to create a split",
    )
    segment_file.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-32.0,
        help="Relative RMS threshold for quiet-gap detection",
    )
    segment_file.add_argument(
        "--min-track-seconds",
        type=float,
        default=30.0,
        help="Minimum duration for an exported track",
    )
    segment_file.add_argument("--output-sr", type=int, default=32_000, help="Export sample rate for track WAVs")
    segment_file.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="Export mono WAVs")
    segment_file.add_argument("--res-type", default="soxr_hq", help="librosa resampler type")
    segment_file.add_argument(
        "--max-silence-fraction",
        type=float,
        default=0.85,
        help="Drop segments whose silence fraction exceeds this; set to 1.0 to disable",
    )
    segment_file.add_argument(
        "--silence-frame-db",
        type=float,
        default=-35.0,
        help="Peak-relative dB threshold for classifying a frame as silent",
    )
    segment_file.set_defaults(func=cmd_segment_file)

    segment_dir = segment_subparsers.add_parser(
        "dir",
        help="Split every long file in a directory into scorer-friendly track WAVs",
        description="Split each supported audio file in a directory into separate WAV tracks.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    segment_dir.add_argument("input_dir", help="Directory containing long source files")
    segment_dir.add_argument("output_dir", help="Where the segments go: they are sources, so a folder under library/")
    segment_dir.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list, e.g. wav,flac,mp3",
    )
    segment_dir.add_argument("--analysis-sr", type=int, default=8_000, help="Sample rate used for gap detection")
    segment_dir.add_argument("--frame-length", type=int, default=4_096, help="RMS frame size for gap detection")
    segment_dir.add_argument("--hop-length", type=int, default=1_024, help="Hop length for gap detection")
    segment_dir.add_argument(
        "--min-silence-seconds",
        type=float,
        default=1.5,
        help="Minimum quiet gap duration required to create a split",
    )
    segment_dir.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-32.0,
        help="Relative RMS threshold for quiet-gap detection",
    )
    segment_dir.add_argument(
        "--min-track-seconds",
        type=float,
        default=30.0,
        help="Minimum duration for an exported track",
    )
    segment_dir.add_argument("--output-sr", type=int, default=32_000, help="Export sample rate for track WAVs")
    segment_dir.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="Export mono WAVs")
    segment_dir.add_argument("--res-type", default="soxr_hq", help="librosa resampler type")
    segment_dir.add_argument(
        "--max-silence-fraction",
        type=float,
        default=0.85,
        help="Drop segments whose silence fraction exceeds this; set to 1.0 to disable",
    )
    segment_dir.add_argument(
        "--silence-frame-db",
        type=float,
        default=-35.0,
        help="Peak-relative dB threshold for classifying a frame as silent",
    )
    segment_dir.add_argument("--recursive", action="store_true", help="Include subdirectories")
    segment_dir.add_argument("--quiet", action="store_true", help="Suppress progress prints")
    segment_dir.set_defaults(func=cmd_segment_dir)
