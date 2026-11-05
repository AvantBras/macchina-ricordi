#!/usr/bin/env python3
"""Export a trusted legacy MusicGen ``.th`` checkpoint as a portable model bundle.

Runs inside the AudioCraft runtime (torch present). The parent CLI agent
invokes this helper; it exists so a checkpoint the project owns can become a
``macchina_ricordi.model_bundle.v1`` bundle that any safetensors reader can
open without torch.

Example::

    python -m scripts.export_musicgen_bundle checkpoint.th .hydra/config.yaml out/bundle \\
        --name my-model --weights-state best --sig 4ae63477 --epoch 47 --force

The ``--weights-state`` picks which snapshot to lift: ``model`` (raw, default),
``ema`` or ``best``. A requested ``ema``/``best`` that the checkpoint does not
carry fails the export rather than silently substituting another state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from macchina_ricordi.models.bundle import (
    WEIGHTS_STATES,
    BundleError,
    export_checkpoint_bundle,
    load_bundle_tensors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trusted legacy MusicGen .th checkpoint as a portable model bundle."
    )
    parser.add_argument("checkpoint", help="Source checkpoint.th (or checkpoint_N.th) path")
    parser.add_argument(
        "config", help="Resolved Hydra builder source: an .hydra/config.yaml path or a YAML config path"
    )
    parser.add_argument("output", help="Destination bundle directory")
    parser.add_argument("--name", required=True, help="Bundle name (goes in the manifest)")
    parser.add_argument(
        "--weights-state",
        choices=WEIGHTS_STATES,
        default="model",
        help="Which snapshot to export: model (raw), ema, or best",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing bundle directory")
    # Optional provenance entries (the safe, portable ones): never absolute paths.
    parser.add_argument("--sig", help="Source model signature (provenance)")
    parser.add_argument("--epoch", type=int, help="Source checkpoint epoch (provenance)")
    parser.add_argument("--recipe", help="Recipe/training profile name (provenance)")
    parser.add_argument("--parents", help="Comma-separated parent signatures (provenance)")
    parser.add_argument(
        "--license",
        help="Explicit licence text for the bundle's LICENSE (otherwise an honest default note)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read the bundle back and report the weight checksum and tensor count",
    )
    return parser


def _provenance_from_args(args: argparse.Namespace) -> dict:
    provenance: dict = {}
    for key in ("sig", "epoch", "recipe"):
        value = getattr(args, key)
        if value is not None:
            provenance[key] = int(value) if key == "epoch" else str(value)
    if args.parents:
        provenance["parents"] = [p.strip() for p in args.parents.split(",") if p.strip()]
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    try:  # only reached when run outside the AudioCraft runtime
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"this helper needs torch (run inside the AudioCraft runtime): {exc}", file=sys.stderr)
        return 2

    args = build_parser().parse_args(argv)
    provenance = _provenance_from_args(args)
    try:
        bundle = export_checkpoint_bundle(
            Path(args.checkpoint),
            Path(args.config),
            Path(args.output),
            name=args.name,
            weights_state=args.weights_state,
            provenance=provenance,
            force=args.force,
            license_text=args.license,
        )
    except (BundleError, FileExistsError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = {
        "name": bundle.name,
        "path": str(bundle.path.resolve()),
        "schema": bundle.config.get("schema"),
        "segment_duration": bundle.segment_duration,
        "codec": bundle.codec,
        "weights": {
            "file": bundle.weights_file,
            "sha256": bundle.sha256,
            "selection": bundle.selection,
        },
    }
    if args.verify:
        tensors = load_bundle_tensors(str(bundle.path), device="cpu")
        report["tensors"] = len(tensors)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
