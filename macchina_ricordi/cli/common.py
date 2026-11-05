from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


def _get_tqdm():
    from tqdm.auto import tqdm

    return tqdm


class MacchinaRicordiHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults while preserving hand-written examples."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        help_text = action.help
        if help_text is None:
            return None
        if "%(default)" in help_text or "(default:" in help_text:
            return help_text
        if action.default is argparse.SUPPRESS or action.default in (None, "", False):
            return help_text
        if isinstance(action.default, dict | list | set | tuple) and not action.default:
            return help_text
        return f"{help_text} (default: %(default)s)"


def clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def format_metric_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def add_shared_scoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sr", type=int, default=32_000, help="Target sample rate")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for onset analysis")
    parser.add_argument("--res-type", default="soxr_hq", help="librosa resampler type")
    parser.add_argument(
        "--tempo-method",
        choices=("autocorr", "tempogram"),
        default="tempogram",
        help="Tempo clarity penalty backend",
    )
