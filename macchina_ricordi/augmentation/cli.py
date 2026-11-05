from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files, parse_extensions
from ..cli.common import MacchinaRicordiHelpFormatter, _get_tqdm
from .core import (
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SPEED_MODE,
    OUTPUT_FORMATS,
    SPEED_MODES,
    export_speed_variants,
    parse_speed_ratios,
)

REPORT_FILENAME = "augment_report.json"


def cmd_augment_speed(args: argparse.Namespace) -> int:
    ratios = parse_speed_ratios(args.ratios)
    if not ratios:
        raise ValueError("--ratios must list at least one speed ratio, e.g. 0.8,1,1.25")

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if input_path.is_file():
        jobs = [(input_path, output_dir)]
    elif input_path.is_dir():
        files = sorted(
            iter_audio_files(input_path, recursive=args.recursive, extensions=parse_extensions(args.extensions))
        )
        jobs = [(path, output_dir / path.parent.relative_to(input_path)) for path in files]
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    variants = []
    tqdm = _get_tqdm()
    for audio_path, destination in tqdm(jobs, total=len(jobs), desc="Augmenting", unit="file", disable=args.quiet):
        variants.extend(
            export_speed_variants(
                audio_path,
                destination,
                ratios,
                mode=args.mode,
                output_format=args.format,
                output_sr=args.output_sr,
                mono=args.mono,
            )
        )

    report_path = output_dir / REPORT_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input_path": str(input_path),
        "ratios": list(ratios),
        "mode": args.mode,
        "format": args.format,
        "output_sr": args.output_sr,
        "mono": args.mono,
        "source_count": len(jobs),
        "variant_count": len(variants),
        "variants": [variant.to_dict() for variant in variants],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(variants)} variants from {len(jobs)} source files ({args.mode}) to {output_dir}")
    print(f"Report: {report_path}")
    return 0


def register_augmentation_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    augment = subparsers.add_parser(
        "augment",
        help="Create transformed copies of audio clips to enlarge a dataset",
        description="Create transformed copies of audio clips to enlarge a dataset.",
        epilog="""examples:
  python -m macchina_ricordi.cli augment speed clip.wav augmented --ratios 0.8,1,1.25
  python -m macchina_ricordi.cli augment speed curated_clips augmented --ratios 0.9,1,1.1 --recursive""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    augment_subparsers = augment.add_subparsers(dest="augment_command", required=True)

    speed = augment_subparsers.add_parser(
        "speed",
        help="Write sped-up / slowed-down copies of a file or directory",
        description=(
            "Write one <stem>__speed<ratio> file per ratio (1 = original speed, 2 = twice as fast). "
            "For training data, prefer `data prepare --speed-ratios`: it augments the train "
            "split only, so no variant of a validation clip leaks into train."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    speed.add_argument("input_path", help="Audio file or directory of audio files")
    speed.add_argument("output_dir", help="Directory where variants should be written")
    speed.add_argument("--ratios", required=True, help="Comma-separated speed ratios, e.g. 0.8,1,1.25")
    speed.add_argument(
        "--mode",
        choices=SPEED_MODES,
        default=DEFAULT_SPEED_MODE,
        help="varispeed: pitch follows speed, artefact-free; stretch: pitch kept, transients smeared",
    )
    speed.add_argument("--format", choices=OUTPUT_FORMATS, default=DEFAULT_OUTPUT_FORMAT, help="Output file format")
    speed.add_argument(
        "--output-sr",
        type=int,
        default=None,
        help="Output sample rate; defaults to each source file's native rate",
    )
    speed.add_argument(
        "--mono",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Downmix to mono (source channels are kept by default)",
    )
    speed.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated list, e.g. wav,flac,mp3 (directory input only)",
    )
    speed.add_argument("--recursive", action="store_true", help="Include subdirectories (directory input only)")
    speed.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    speed.set_defaults(func=cmd_augment_speed)
