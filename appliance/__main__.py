"""``python -m appliance`` — play a box, render one to a file, or open the panel.

With no arguments it opens the panel, which is what double-clicking a packaged
build does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from macchina_ricordi.instrument import controls

from .engine import Engine
from .loader import describe, find_boxes, load_player, read_any_spec, recommended_knobs
from .player import Knobs
from .wav import write_wav


def _knobs(args, spec) -> Knobs:
    base = Knobs(
        temperature=1.0 if args.temperature is None else args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tau_seconds=args.tau,
        round_seconds=args.round_seconds,
        tilt_low_db=args.tilt_low,
        tilt_mid_db=args.tilt_mid,
        tilt_high_db=args.tilt_high,
        prompt_rate=args.tape_speed,
        recap_every=args.recall_every,
    )
    return recommended_knobs(spec, base) if args.temperature is None else base.clamped()


def _help(control, extra: str = "") -> str:
    """A flag's help from the shared vocabulary: what it is called on the panel, what it does, its range."""
    tail = f", {extra}" if extra else ""
    return f"{control.label.lower()}: {control.help} ({control.range_text(appliance=True)}{tail})"


def _add_common(ap) -> None:
    ap.add_argument("--temperature", type=float, default=None, help="default: whatever the box recommends")
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="about 30%% quicker, uses about 850 MB more memory")
    ap.add_argument(
        "--tau", "--context-seconds", dest="tau", type=float, default=None, help=_help(controls.TAU, "default 5")
    )
    ap.add_argument("--round-seconds", type=float, default=None, help=_help(controls.ROUND, "default 4"))
    for c, flag_default in ((controls.TILT_LOW, 0.0), (controls.TILT_MID, 0.0), (controls.TILT_HIGH, 0.0)):
        ap.add_argument(c.flag, type=float, default=flag_default, help=_help(c))
    ap.add_argument("--tape-speed", type=float, default=1.0, help=_help(controls.TAPE_SPEED))
    ap.add_argument("--recall-every", type=int, default=0, help="blend a pinned moment back in every N rounds")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gain", type=float, default=0.9)


def _open(args):
    return load_player(
        args.box,
        threads=args.threads,
        fast=args.fast,
        seed=args.seed,
    )


def cmd_play(args) -> int:
    player, spec = _open(args)
    knobs = _knobs(args, spec)
    engine = Engine(player, knobs=knobs, gain=args.gain)
    memory, stride = (frames / spec.frame_rate for frames in player.frames_for(knobs))
    print(
        f"{spec.name} — {spec.sample_rate} Hz, memory {memory:.1f}s, stride {stride:.1f}s, "
        f"{player.latency_seconds * 1000:.0f} ms held back, temperature {knobs.temperature:g}"
    )
    engine.on_round(
        lambda info: print(
            f"  round {engine.health.rounds:>4}  {info['xrt']:.2f} xRT  "
            f"buffer {engine.health.buffered_seconds:4.1f}s" + ("  UNDERRUN" if engine.health.underruns else ""),
            flush=True,
        )
    )
    print("playing — Ctrl-C to stop")
    engine.play(device=args.device, seconds=args.seconds)
    print("\n" + json.dumps(engine.health.snapshot(), indent=2))
    return 0


def cmd_render(args) -> int:
    player, spec = _open(args)
    engine = Engine(player, knobs=_knobs(args, spec), gain=args.gain)
    audio = engine.render(args.seconds)
    out = write_wav(args.out, audio, spec.sample_rate)
    print(f"wrote {out} ({len(audio) / spec.sample_rate:.1f}s)")
    print(json.dumps(engine.health.snapshot(), indent=2))
    return 0


def cmd_list(args) -> int:
    boxes = find_boxes(*args.roots) if args.roots else find_boxes()
    if not boxes:
        print("no .nrbox files found")
        return 1
    for path in boxes:
        d = describe(read_any_spec(path), path)
        temperature = d["tuning"].get("temperature", "—")
        print(f"{d['name']:<24} {d['size_mb']:>6} MB  {d['dim']}d/{d['layers']}L  T={temperature}  {d['file']}")
    return 0


def cmd_panel(args) -> int:
    from .panel import serve

    return serve(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="appliance", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="play a box through the speakers")
    p.add_argument("box", type=Path)
    p.add_argument("--device", default=None)
    p.add_argument("--seconds", type=float, default=None)
    _add_common(p)
    p.set_defaults(func=cmd_play)

    r = sub.add_parser("render", help="write audio to a file, no audio device needed")
    r.add_argument("box", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--seconds", type=float, default=30.0)
    _add_common(r)
    r.set_defaults(func=cmd_render)

    ls = sub.add_parser("list", help="list boxes")
    ls.add_argument("roots", nargs="*", type=Path)
    ls.set_defaults(func=cmd_list)

    pa = sub.add_parser("panel", help="open the control panel in a browser")
    pa.add_argument("--boxes", nargs="*", type=Path, default=None)
    pa.add_argument("--port", type=int, default=8731)
    pa.add_argument("--device", default=None)
    pa.add_argument("--no-browser", action="store_true")
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--fast", action="store_true", help="about 30%% quicker, uses about 850 MB more memory")
    pa.set_defaults(func=cmd_panel)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv or ["panel"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
