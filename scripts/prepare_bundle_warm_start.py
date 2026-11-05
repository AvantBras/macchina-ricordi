"""Prepare a local, weights-only AudioCraft adapter for a portable warm-start.

This is derived runtime cache, not the public release format. It contains no
optimizer or training history. Both roots intentionally refer to the same selected
weights: AudioCraft warm-starts another XP from best_state, not model.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def prepare(bundle_path: str | Path, destination: str | Path) -> Path:
    import torch

    from macchina_ricordi.models.bundle import load_bundle_tensors, read_bundle

    bundle = read_bundle(bundle_path)
    weights = load_bundle_tensors(bundle.path, device="cpu")
    target = Path(destination)
    if target.is_symlink() or target.parent.is_symlink():
        raise ValueError("Refusing a symlink warm-start cache destination.")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".th", dir=target.parent)
    os.close(fd)
    try:
        torch.save({"model": weights, "best_state": {"model": weights}}, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("destination")
    args = parser.parse_args()
    try:
        print(prepare(args.bundle, args.destination))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
