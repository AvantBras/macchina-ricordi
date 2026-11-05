#!/usr/bin/env python3
"""Compare candidate loop starts for one loop length, and find where a sung line comes back.

    .venv/bin/python scripts/loops/compare_starts.py REPORT --bars 8 --starts 48.66,49.73,49.98
    .venv/bin/python scripts/loops/compare_starts.py REPORT --bars 8 --reference 42.563 --starts 57.11

Use it when the user asks for a loop "starting around X s where the voice restarts" and no clean breath
shows up there (see vocal_map.py --around X). Two kinds of evidence:

- --reference S: where does the line that starts at S come back? Lines in dance music recur at whole
  2/4/8-bar offsets; a recurrence close to X is the restart the user hears. Recurrences at whole-bar offsets
  are added to the comparison as "ref+N bars" starts. (Matches that start in silence are still valid: the
  melody lines up, the singer just skips a pick-up note.)
- For every start, the seam: voice level just before the start and at the loop end (below about -20 dB
  means the cut falls in a breath), voice coverage, longest gap, and how alike the audio is after the end vs
  after the start, and before the end vs before the start (1.0 = the jump back is inaudible).
"""

from __future__ import annotations

import argparse

import numpy as np
from common import SR, features, fmt_time, load_mono, load_report, similarity, vocal_activity, window

from macchina_ricordi.looping import VocalActivity

SEAM_SECONDS = 1.0
LINE_SECONDS = 3.0


def recurrences(
    vocal_feats: np.ndarray, fps: float, activity: VocalActivity, reference: float, bar: float, top: int = 8
) -> list[tuple]:
    """Best matches of the reference line elsewhere; silent stretches are skipped (silence looks alike)."""
    template = window(vocal_feats, fps, reference, LINE_SECONDS)
    last = vocal_feats.shape[1] / fps - LINE_SECONDS
    scored = [(similarity(template, window(vocal_feats, fps, t, LINE_SECONDS)), t) for t in np.arange(0, last, 0.01)]
    scored.sort(reverse=True)
    picked: list[tuple] = []
    for score, t in scored:
        if abs(t - reference) < 0.4 or any(abs(t - other) < 0.4 for _, other in picked):
            continue
        if activity.voiced_fraction(t, t + LINE_SECONDS) < 0.5:
            continue
        picked.append((score, t))
        if len(picked) == top:
            break
    return [(score, t, (t - reference) / bar) for score, t in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="loop_report.json written by `loop find`")
    parser.add_argument("--bars", type=int, required=True, help="Loop length in bars")
    parser.add_argument("--starts", default="", help="Comma-separated candidate starts in seconds")
    parser.add_argument("--reference", type=float, default=None, help="Start (s) of a line to look for elsewhere")
    args = parser.parse_args()

    report = load_report(args.report)
    bar = float(report["bar_seconds"])
    length = args.bars * bar
    vocals = load_mono(report["stems"]["vocals"])
    activity = vocal_activity(report, vocals)
    vocal_feats, fps = features(vocals)
    mix_feats, _ = features(load_mono(report["source_path"]))

    starts = [(float(s), "given") for s in args.starts.split(",") if s.strip()]
    if args.reference is not None:
        starts.insert(0, (args.reference, "reference"))
        print(f"where the line at {fmt_time(args.reference)} comes back ({LINE_SECONDS:g} s of voice compared):")
        for score, t, offset in recurrences(vocal_feats, fps, activity, args.reference, bar):
            whole = round(offset)
            on_grid = abs(offset - whole) < 0.1 and whole != 0
            note = f"  -> ref{whole:+d} bars = {args.reference + whole * bar:.3f}s" if on_grid else ""
            print(f"  {fmt_time(t):>8}  similarity {score:.2f}  {offset:+6.2f} bars{note}")
            if on_grid and all(abs(args.reference + whole * bar - s) > 0.05 for s, _ in starts):
                starts.append((round(args.reference + whole * bar, 4), f"ref{whole:+d} bars"))
        print()

    duration = vocals.shape[-1] / SR
    print(f"{args.bars}-bar loops ({length:.4f}s). Voice levels in dB below the loud voice; seam 1.0 = inaudible.")
    print(
        f"{'start':>9} {'end':>9} {'origin':>12} {'before':>7} {'at end':>7} {'voice':>6} {'gap':>5}"
        f"  {'voice seam':>12}  {'mix seam':>12}"
    )
    for start, origin in starts:
        end = start + length
        if start < SEAM_SECONDS or end + SEAM_SECONDS > duration:
            print(f"{start:9.3f}  (too close to the song's edges for a {args.bars}-bar loop)")
            continue
        seams = []
        for feats in (vocal_feats, mix_feats):
            after = similarity(window(feats, fps, end, SEAM_SECONDS), window(feats, fps, start, SEAM_SECONDS))
            before = similarity(
                window(feats, fps, end - SEAM_SECONDS, SEAM_SECONDS),
                window(feats, fps, start - SEAM_SECONDS, SEAM_SECONDS),
            )
            seams.append(f"{after:.2f} / {before:.2f}")
        print(
            f"{start:9.3f} {end:9.3f} {origin:>12} {activity.level_db(start - 0.1, start):7.1f}"
            f" {activity.level_db(end - 0.12, end + 0.01):7.1f} {activity.voiced_fraction(start, end):6.0%}"
            f" {activity.longest_gap_seconds(start, end):4.1f}s  {seams[0]:>12}  {seams[1]:>12}"
        )
    print("\nrender one:")
    print(
        f"  .venv/bin/python -m macchina_ricordi.cli loop render {args.report} "
        f"out.flac --start <start> --bars {args.bars}"
    )


if __name__ == "__main__":
    main()
