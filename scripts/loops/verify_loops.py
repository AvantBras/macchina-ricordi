#!/usr/bin/env python3
"""Check rendered loops: format, length, beat drift across repeats, and where clipping comes from.

    .venv/bin/python scripts/loops/verify_loops.py examples/sample_data/eurodance_voices_loops
    .venv/bin/python scripts/loops/verify_loops.py loop_a.flac loop_b.flac --no-drift

Reads each file's `<stem>_report.json` sidecar (written by `loop render`). Per loop:
- format and duration against repeats x loop length;
- beat drift: the beat phase of every repeat against one continuous grid. A whole-bar loop reads ~0 ms
  everywhere. One repeat near half a beat is the reading flipping to the off-beat (hi-hats) and counts as a
  blip, not drift; real drift grows repeat after repeat. Skipped for the vocals and stuck layers;
- clipping: samples at full scale, and for seam clipping whether the source is itself at full scale there.
  Loud masters decode from MP3 slightly above 1.0, so clipping inside the loop body comes from the source;
  seam clipping where the source is below full scale means the render made it.
Exit code 1 if any loop fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from common import SR, beat_deviation_ms

SEAM_MS = 12.0
DRIFT_OK_MS = 5.0


def sidecar_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}_report.json")


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".flac", ".wav"}))
        else:
            files.append(path)
    return files


def check(path: Path, sidecar: dict, *, drift: bool, sources: dict) -> tuple[bool, str]:
    import librosa
    import soundfile

    audio, sr = soundfile.read(str(path), always_2d=True)
    peak_per_frame = np.abs(audio).max(axis=1)
    mono = audio.mean(axis=1)
    problems, notes = [], []

    span = int(round(sidecar["length_seconds"] * sr))
    repeats = sidecar.get("repeats")
    duration = audio.shape[0] / sr
    if repeats and abs(audio.shape[0] - span * repeats) > 1:
        problems.append(f"length {duration:.2f}s != {repeats} x {sidecar['length_seconds']:.4f}s")

    if drift and repeats and repeats >= 3 and sidecar.get("layer") in ("mix", "instrumental"):
        beats_per_bar = 4
        report_path = Path(sidecar.get("loop_report", ""))
        if report_path.is_file():
            beats_per_bar = json.loads(report_path.read_text(encoding="utf-8")).get("beats_per_bar", 4)
        beat = sidecar["bar_seconds"] / beats_per_bar
        deviation = np.abs(beat_deviation_ms(librosa.resample(mono, orig_sr=sr, target_sr=SR), SR, beat, repeats))
        large = deviation > 0.4 * beat * 1000.0
        calm = deviation <= DRIFT_OK_MS
        neighbours_calm = np.array(
            [(i == 0 or calm[i - 1]) and (i == repeats - 1 or calm[i + 1]) for i in range(repeats)]
        )
        blips = large & neighbours_calm
        steady = deviation[~blips]
        drift_ms = float(np.median(steady)) if steady.size else float("nan")
        notes.append(f"drift {drift_ms:.1f} ms" + (f" ({int(blips.sum())} off-beat blip)" if blips.any() else ""))
        if not steady.size or drift_ms > DRIFT_OK_MS or blips.sum() > max(1, repeats // 8):
            problems.append(f"beat drift {deviation.round().astype(int).tolist()} ms")

    clipped = np.flatnonzero(peak_per_frame >= 0.999)
    if clipped.size:
        position = clipped % span
        at_seam = clipped[np.minimum(position, span - position) < int(SEAM_MS / 1000.0 * sr)]
        note = f"clip {clipped.size}"
        if at_seam.size and sidecar.get("layer") == "mix":
            key = (sidecar["source_path"], sr, bool(sidecar.get("mono")))
            if key not in sources:
                decoded = librosa.load(sidecar["source_path"], sr=sr, mono=bool(sidecar.get("mono")))[0]
                sources[key] = np.atleast_2d(np.abs(decoded)).max(axis=0)
            source_peak = sources[key]
            begin = int(round(sidecar["start_seconds"] * sr))
            width = int(SEAM_MS / 1000.0 * sr)
            hot = max(
                source_peak[begin : begin + width].max(initial=0.0),
                source_peak[begin + span - width : begin + span + width].max(initial=0.0),
            )
            if hot < 0.999:
                problems.append(f"{at_seam.size} clipped samples at seams where the source is below full scale")
            else:
                note += " (seams: source at full scale there too)"
        elif not at_seam.size:
            note += " (inside the loop body: from the source)"
        notes.append(note)

    channels = "mono" if audio.shape[1] == 1 else f"{audio.shape[1]} ch"
    loops = f"{repeats} x {sidecar['length_seconds']:.3f}s" if repeats else sidecar.get("layer", "")
    summary = f"{sr} Hz {channels} {duration:6.1f}s  {loops}  " + "  ".join(notes)
    return not problems, summary + ("" if not problems else "\n       " + "; ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Loop files and/or folders of loops")
    parser.add_argument("--no-drift", action="store_true", help="Skip the beat-drift check (the slow part)")
    args = parser.parse_args()

    sources: dict = {}
    failures = checked = missing = 0
    for path in collect(args.paths):
        if not path.is_file():
            print(f"MISSING {path}")
            missing += 1
            continue
        sidecar_path = sidecar_for(path)
        if not sidecar_path.is_file():
            print(f"SKIP {path.name}: no {sidecar_path.name}")
            continue
        ok, summary = check(
            path, json.loads(sidecar_path.read_text(encoding="utf-8")), drift=not args.no_drift, sources=sources
        )
        checked += 1
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {path.name}\n       {summary}")
    print(f"\n{checked - failures} ok, {failures} failed" + (f", {missing} missing" if missing else ""))
    sys.exit(1 if failures or missing else 0)


if __name__ == "__main__":
    main()
