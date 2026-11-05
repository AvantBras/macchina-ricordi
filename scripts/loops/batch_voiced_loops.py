#!/usr/bin/env python3
"""Voiced-only loops (whole bars, no overlap) for every track in a folder, via the `loop` CLI.

    nice -n 10 .venv/bin/python scripts/loops/batch_voiced_loops.py \\
        examples/sample_data/eurodance_voices examples/sample_data/eurodance_voices_loops
    ... --dry-run            # analyse + CLAP scores, render nothing
    ... --bars 8 --seconds 60 --no-mono --no-clap

Per track:
1. `loop find --bars 4,8 --min-voice 0.8 --max-gap-ms 1800 --no-overlap --top 50 --no-render` into
   <analysis-root>/<track stem>/ (Demucs runs on first use, ~0.25x real time on CPU; stems are cached).
2. CLAP check on each candidate's vocal stem: P(singing voice) from three 6 s windows (msclap random-crops
   anything longer than 7 s, so windows keep it deterministic). Demucs' "vocals" can be a synth lead on
   trance tracks (Mauro Picotto); those score low. Loops below --clap-min are dropped but listed.
3. `loop render --pick N --seconds S [--mono]` into OUT_DIR as `<track stem>__<0m42s56_8bar>.flac` + sidecar.

Writes OUT_DIR/loops_manifest.json (kept loops, dropped loops with their scores, settings). A re-run first
deletes the files the previous manifest listed, so stale loops never linger. Thresholds were calibrated on
Hit My Heart (sung loops: 87-98% voice, gaps <= 1.6 s; breaks >= 2.2 s; CLAP 0.90-0.99) - not ear-validated
beyond that, so tell the user what was dropped and why.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from common import SR, load_mono

from macchina_ricordi.audio_files import DEFAULT_EXTENSIONS, parse_extensions
from macchina_ricordi.looping import candidate_label

MANIFEST = "loops_manifest.json"
CLI = [sys.executable, "-m", "macchina_ricordi.cli", "loop"]
VOICE_PROMPTS = ["a woman singing", "a man singing", "a singing voice", "vocals"]
OTHER_PROMPTS = ["a synthesizer melody", "electronic instrumental music", "a piano melody", "drums and bass"]


class ClapVoice:
    def __init__(self, device: str) -> None:
        import torch

        from macchina_ricordi.listening.semantic import MsClapSemanticProbeBackend

        torch.set_num_threads(8)
        self._backend = MsClapSemanticProbeBackend(device=device)
        self._clip = Path(tempfile.mkdtemp()) / "window.wav"

    def p_voice(self, vocals: np.ndarray, start: float, length: float) -> float:
        import soundfile

        probabilities = []
        for window in (start + 0.5, start + length / 2.0 - 3.0, start + length - 6.5):
            soundfile.write(self._clip, vocals[int(max(window, 0.0) * SR) : int((max(window, 0.0) + 6.0) * SR)], SR)
            scores = np.asarray(self._backend.score_audio_text(self._clip, VOICE_PROMPTS + OTHER_PROMPTS), dtype=float)
            weights = np.exp(scores - scores.max())
            probabilities.append(float(weights[: len(VOICE_PROMPTS)].sum() / weights.sum()))
        return float(np.mean(probabilities))


def remove_previous(out_dir: Path) -> None:
    manifest_path = out_dir / MANIFEST
    if not manifest_path.is_file():
        return
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")).get("loops", []):
        audio = out_dir / entry["file"]
        audio.unlink(missing_ok=True)
        audio.with_name(f"{audio.stem}_report.json").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", help="Folder of songs")
    parser.add_argument("output_dir", help="Folder for the loops (created)")
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Default: artifacts/loops/<source_dir name>. Each run rewrites the loop_report.json files there, "
        "which earlier manifests point to: use a separate root for experiments",
    )
    parser.add_argument("--bars", default="4,8", help="Loop lengths in bars")
    parser.add_argument("--min-voice", type=float, default=0.8, help="Voice present this fraction of the loop")
    parser.add_argument("--max-gap-ms", type=float, default=1800.0, help="Longest stretch without voice")
    parser.add_argument("--top", type=int, default=50, help="Max loops per track before the CLAP check")
    parser.add_argument("--seconds", type=float, default=115.0, help="Approximate length of each rendered loop")
    parser.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="Downmix to mono")
    parser.add_argument("--clap", action=argparse.BooleanOptionalAction, default=True, help="Run the CLAP check")
    parser.add_argument("--clap-min", type=float, default=0.8, help="Minimum P(singing voice) to keep a loop")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Demucs and CLAP device")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS), help="Audio file extensions")
    parser.add_argument("--dry-run", action="store_true", help="Analyse and score only; render nothing")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        parser.error(f"source folder does not exist: {source_dir}")
    out_dir = Path(args.output_dir)
    analysis_root = Path(args.analysis_root or f"artifacts/loops/{source_dir.name}")
    extensions = {f".{ext}" for ext in parse_extensions(args.extensions)}
    tracks = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    select = ["--bars", args.bars, "--min-voice", str(args.min_voice), "--max-gap-ms", str(args.max_gap_ms)]
    select += ["--no-overlap", "--top", str(args.top), "--no-render", "--device", args.device]
    render = ["--seconds", str(args.seconds)] + (["--mono"] if args.mono else [])
    clap = ClapVoice(args.device) if args.clap else None

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        remove_previous(out_dir)
    kept, dropped, failed = [], [], []
    for track in tracks:
        analysis_dir = analysis_root / track.stem
        found = subprocess.run(CLI + ["find", str(track), str(analysis_dir)] + select, capture_output=True, text=True)
        if found.returncode != 0:
            failed.append({"track": track.name, "error": found.stderr.strip().splitlines()[-1:]})
            print(f"{track.stem[:48]:48s} FAILED: {' '.join(failed[-1]['error'])}", flush=True)
            continue
        report_path = analysis_dir / "loop_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        vocals = load_mono(report["stems"]["vocals"]) if clap else None
        track_kept = 0
        for candidate in report["candidates"]:
            entry = {
                "track": track.name,
                "start_seconds": candidate["start_seconds"],
                "bars": candidate["bars"],
                "length_seconds": candidate["length_seconds"],
                "tempo_bpm": report["tempo_bpm"],
                "voice_fraction": candidate["voice_fraction"],
                "longest_gap_seconds": candidate["longest_gap_seconds"],
                "clap_p_voice": None,
                "loop_report": str(report_path),
                "pick": candidate["rank"],
            }
            if clap is not None:
                entry["clap_p_voice"] = round(
                    clap.p_voice(vocals, candidate["start_seconds"], candidate["length_seconds"]), 3
                )
                if entry["clap_p_voice"] < args.clap_min:
                    dropped.append(entry)
                    continue
            entry["file"] = f"{track.stem}__{candidate_label(candidate['start_seconds'], candidate['bars'])}.flac"
            if not args.dry_run:
                command = CLI + ["render", str(report_path), str(out_dir / entry["file"]), "--pick", str(entry["pick"])]
                subprocess.run(command + render, check=True, capture_output=True)
            kept.append(entry)
            track_kept += 1
        print(
            f"{track.stem[:48]:48s} {report['tempo_bpm']:7.2f} BPM  "
            f"keep {track_kept:2d} / {len(report['candidates']):2d}",
            flush=True,
        )

    if args.dry_run:
        print("\n(dry run) would keep:")
        for entry in kept:
            clap_note = "" if entry["clap_p_voice"] is None else f"  clap {entry['clap_p_voice']:.2f}"
            print(
                f"  {entry['file']}  voice {entry['voice_fraction']:.0%}  "
                f"gap {entry['longest_gap_seconds']:.1f}s{clap_note}"
            )
    else:
        manifest = {
            "source_dir": str(source_dir),
            "analysis_root": str(analysis_root),
            "selection": {"loop_find": " ".join(select), "loop_render": " ".join(render)},
            "clap": {"min_p_voice": args.clap_min, "voice_prompts": VOICE_PROMPTS, "other_prompts": OTHER_PROMPTS}
            if clap
            else None,
            "loop_count": len(kept),
            "loops": kept,
            "dropped_by_clap": dropped,
            "failed_tracks": failed,
        }
        (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if dropped:
        print("\ndropped by the CLAP voice check (listen before trusting it):")
        for entry in dropped:
            print(
                f"  {entry['track'][:40]:40s} {entry['start_seconds']:7.2f}s {entry['bars']} bars  "
                f"clap {entry['clap_p_voice']:.2f}"
            )
    print(
        f"\n{len(kept)} loops {'would be ' if args.dry_run else ''}kept, "
        f"{len(dropped)} dropped, {len(failed)} tracks failed"
    )


if __name__ == "__main__":
    main()
