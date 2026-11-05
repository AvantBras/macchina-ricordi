#!/usr/bin/env python3
"""Text map of where the voice is in a song, from a `loop find` report.

    .venv/bin/python scripts/loops/vocal_map.py artifacts/loops/song/loop_report.json
    .venv/bin/python scripts/loops/vocal_map.py REPORT --anchor 42.563            # bar 0 = a known loop start
    .venv/bin/python scripts/loops/vocal_map.py REPORT --around 49 --window 1.5   # zoom on one moment

Map: one row per two bars, one character per eighth note. '#' loud voice, '+' voice, '.' faint (breath or
bleed), ' ' silent. The report gives the bar length but not the bar phase, so pass --anchor (a loop or
phrase start) to line rows up with the music. Phrase starts are listed with their offset in bars from the
anchor: sung lines usually recur at whole 2/4/8-bar offsets, which is how to find where the voice "restarts".

--around zooms in: voice level every 40 ms and syllable onsets with the dip before each. A syllable that
follows a deep dip is a clean place to start a loop; with no dip, the voice runs straight through.
"""

from __future__ import annotations

import argparse

import numpy as np
from common import SR, fmt_time, load_mono, load_report, vocal_activity

from macchina_ricordi.looping import find_vocal_onsets


def _cell(level: float, threshold: float) -> str:
    if level >= -12.0:
        return "#"
    if level >= -threshold:
        return "+"
    if level >= -40.0:
        return "."
    return " "


def print_map(report: dict, vocals: np.ndarray, anchor: float, phrase_breath: float) -> None:
    activity = vocal_activity(report, vocals)
    bar = float(report["bar_seconds"])
    duration = vocals.shape[-1] / SR
    row = 2.0 * bar
    first = anchor - np.floor(anchor / row) * row
    print(f"{report['source_path']}")
    print(f"tempo {report['tempo_bpm']:.2f} BPM, bar {bar:.4f}s, bar 0 at {fmt_time(anchor)}")
    for row_start in np.arange(first - (row if first > 0 else 0.0), duration, row):
        cells = []
        for eighth in range(16):
            t = row_start + eighth * bar / 8.0
            level = activity.level_db(t, t + bar / 8.0) if 0.0 <= t < duration else -99.0
            cells.append(_cell(level, activity.threshold_db))
        bar_index = int(round((row_start - anchor) / bar))
        print(f"{fmt_time(row_start):>8} bar {bar_index:+4d} |{''.join(cells[:8])}|{''.join(cells[8:])}|")

    onsets = find_vocal_onsets(activity, min_breath_seconds=phrase_breath)
    print(f"\nphrase starts (voice back after >= {phrase_breath:.2f} s), bars from anchor:")
    entries = [
        f"{fmt_time(o.time_seconds)} ({(o.time_seconds - anchor) / bar:+.2f} bars, breath {o.breath_seconds:.2f}s)"
        for o in onsets
    ]
    for index in range(0, len(entries), 3):
        print("  " + "   ".join(entries[index : index + 3]))


def print_zoom(report: dict, vocals: np.ndarray, anchor: float, around: float, half_window: float) -> None:
    import librosa

    activity = vocal_activity(report, vocals)
    bar = float(report["bar_seconds"])
    beat = bar / report.get("beats_per_bar", 4)
    print(
        f"voice level around {fmt_time(around)} "
        f"(dB below the song's loud voice; active above -{activity.threshold_db:g})"
    )
    for t in np.arange(around - half_window, around + half_window, 0.04):
        level = activity.level_db(t, t + 0.04)
        print(f"  {fmt_time(t):>8} {level:6.1f} " + "#" * max(0, int((level + 60.0) / 1.5)))

    hop = 110
    strength = librosa.onset.onset_strength(y=vocals, sr=SR, hop_length=hop)
    rms_db = 20.0 * np.log10(librosa.feature.rms(y=vocals, frame_length=441, hop_length=hop)[0] + 1e-9)
    reference = activity.reference_db
    peaks = librosa.util.peak_pick(
        strength,
        pre_max=5,
        post_max=5,
        pre_avg=10,
        post_avg=10,
        delta=float(np.percentile(strength, 90)) * 0.5,
        wait=8,
    )
    print("\nsyllable onsets (dip before = quietest point in the 80 ms before; deeper = cleaner cut):")
    for peak in peaks:
        t = peak * hop / SR
        if abs(t - around) > half_window:
            continue
        dip = rms_db[max(0, peak - 8) : max(1, peak - 1)].min() - reference
        print(
            f"  {fmt_time(t):>8}  dip before {dip:6.1f} dB  strength {strength[peak]:4.1f}  "
            f"{(t - anchor) / beat:+7.2f} beats / {(t - anchor) / bar:+6.2f} bars from anchor"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="loop_report.json written by `loop find`")
    parser.add_argument("--anchor", type=float, default=0.0, help="Time (s) that counts as bar 0")
    parser.add_argument("--around", type=float, default=None, help="Zoom on this time (s) instead of the map")
    parser.add_argument("--window", type=float, default=1.5, help="Zoom half-width in seconds")
    parser.add_argument(
        "--phrase-breath",
        type=float,
        default=0.25,
        help="Shortest breath (s) listed as a phrase start; `loop find` itself uses 0.08 (gaps between words)",
    )
    args = parser.parse_args()

    report = load_report(args.report)
    vocals = load_mono(report["stems"]["vocals"])
    if args.around is None:
        print_map(report, vocals, args.anchor, args.phrase_breath)
    else:
        print_zoom(report, vocals, args.anchor, args.around, args.window)


if __name__ == "__main__":
    main()
