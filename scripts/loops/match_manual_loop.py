#!/usr/bin/env python3
"""Work out what a hand-made loop is (e.g. an Audacity edit) and how to make it bar-exact.

    .venv/bin/python scripts/loops/match_manual_loop.py LOOP_FILE SOURCE_AUDIO
    .venv/bin/python scripts/loops/match_manual_loop.py LOOP_FILE SOURCE_AUDIO --report REPORT   # + bar-exact fix

1. Loop length: the loop file's own autocorrelation. Exact copies correlate ~1.0, so the shortest lag with a
   near-maximal peak is the period (sample-accurate).
2. Where it comes from: cross-correlation of the first repeat against the source, then every repeat is
   checked against the same place (a hand edit sometimes drifts or mixes sections).
3. With a `loop find` report for the same song: the length in bars, the timing error at each seam (a few tens
   of ms is audible as a stumble), whether the start clips the first word, and the `loop render` command for a
   bar-exact version that starts just before the phrase.
"""

from __future__ import annotations

import argparse

import numpy as np
from common import SR, fmt_time, load_mono, load_report, vocal_activity

from macchina_ricordi.looping import find_vocal_onsets
from macchina_ricordi.looping.core import DEFAULT_PRE_ROLL_SECONDS


def loop_period(loop: np.ndarray, min_seconds: float = 2.0) -> tuple[float, float]:
    """Return (period seconds, normalised correlation at that lag)."""
    size = loop.shape[0]
    spectrum = np.fft.rfft(loop, 2 * size)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    norm = ac / (ac[0] * (size - np.arange(size)) / size + 1e-12)
    lo, hi = int(min_seconds * SR), size // 2
    if hi <= lo:
        raise SystemExit("Loop file is too short to hold two repeats of at least 2 s")
    best = lo + int(np.argmax(norm[lo:hi]))
    first = lo + int(np.flatnonzero(norm[lo:hi] >= 0.9 * norm[best])[0])
    peak = first + int(np.argmax(norm[first : first + int(0.01 * SR)]))
    left, centre, right = norm[peak - 1 : peak + 2]
    curvature = left - 2.0 * centre + right
    offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    return (peak + offset) / SR, float(norm[peak])


def locate(source: np.ndarray, query: np.ndarray) -> tuple[float, float]:
    """Best normalised cross-correlation position of ``query`` inside ``source`` (seconds, score)."""
    from scipy.signal import fftconvolve

    corr = fftconvolve(source, query[::-1], mode="valid")
    energy = np.sqrt(np.maximum(fftconvolve(source**2, np.ones(query.shape[0]), mode="valid"), 0.0))
    score = corr / (energy * np.linalg.norm(query) + 1e-9)
    index = int(np.argmax(score))
    return index / SR, float(score[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("loop_file", help="The hand-made loop")
    parser.add_argument("source", help="The song it was cut from")
    parser.add_argument("--report", default=None, help="loop_report.json of the same song, for the bar-exact fix")
    args = parser.parse_args()

    import librosa

    loop = load_mono(args.loop_file)
    source = load_mono(args.source)
    channels = librosa.load(args.loop_file, sr=None, mono=False, duration=1.0)[0].ndim
    period, period_score = loop_period(loop)
    repeats = int(round(loop.shape[0] / SR / period))
    probe = slice(int(1.0 * SR), int(3.0 * SR))
    found, score = locate(source, loop[probe])
    start = found - 1.0

    print(f"loop file: {loop.shape[0] / SR:.2f}s, {'mono' if channels == 1 else 'stereo'}")
    print(f"period {period:.4f}s (self-correlation {period_score:.3f}), {repeats} repeats")
    print(f"source section: {fmt_time(start)} -> {fmt_time(start + period)} ({start:.3f}s, match {score:.3f})")
    worst = (1.0, 0.0)
    for index in range(1, repeats):
        base = int((index * period + 1.0) * SR)
        query = loop[base : base + probe.stop - probe.start]
        if query.shape[0] < probe.stop - probe.start:
            break
        window = source[int((start + 0.95) * SR) : int((start + 3.05) * SR)]
        where, match = locate(window, query)
        shift_ms = (where - 0.05) * 1000.0
        if match < worst[0]:
            worst = (match, shift_ms)
    print(f"every repeat maps back to the same section: worst match {worst[0]:.3f} (shift {worst[1]:+.1f} ms)")
    if worst[0] < 0.8:
        print("  warning: some repeats do not match the first one; this is not a single section repeated")

    if args.report is None:
        print("\npass --report <loop_report.json> (from `loop find` on the source) for the bar-exact fix")
        return

    report = load_report(args.report)
    bar = float(report["bar_seconds"])
    bars = period / bar
    whole = max(1, int(round(bars)))
    error_ms = (period - whole * bar) * 1000.0
    print(f"\nbar {bar:.4f}s ({report['tempo_bpm']:.2f} BPM): the loop is {bars:.3f} bars")
    print(
        f"  {error_ms:+.1f} ms per seam vs {whole} bars -> the beat is {abs(error_ms) * (repeats - 1):.0f} ms "
        f"{'late' if error_ms > 0 else 'early'} against a steady grid by the last repeat"
    )

    activity = vocal_activity(report, load_mono(report["stems"]["vocals"]))
    print(
        f"  voice just before the start {activity.level_db(start - 0.1, start):+.1f} dB, at the start "
        f"{activity.level_db(start, start + 0.05):+.1f} dB, at the loop end "
        f"{activity.level_db(start + period - 0.12, start + period + 0.01):+.1f} dB (below -20 = in a breath)"
    )
    nearby = [o for o in find_vocal_onsets(activity) if start - 0.5 < o.time_seconds < start + 0.3]
    suggested = start
    if nearby:
        onset = min(nearby, key=lambda o: abs(o.time_seconds - start))
        suggested = onset.time_seconds - min(DEFAULT_PRE_ROLL_SECONDS, onset.breath_seconds / 2.0)
        lag_ms = (start - onset.time_seconds) * 1000.0
        where = "into" if lag_ms > 0 else "before"
        print(f"  the phrase starts at {onset.time_seconds:.3f}s: the cut is {abs(lag_ms):.0f} ms {where} it")
    for candidate in report.get("candidates", []):
        if candidate["bars"] == whole and abs(candidate["start_seconds"] - suggested) < 0.1:
            print(f"  `loop find` already shortlisted it: rank {candidate['rank']}")

    mono = " --mono" if channels == 1 else ""
    print("\nbar-exact version:")
    print(
        f"  .venv/bin/python -m macchina_ricordi.cli loop render {args.report} fixed.flac "
        f"--start {suggested:.4f} --bars {whole} --repeats {repeats}{mono}"
    )


if __name__ == "__main__":
    main()
