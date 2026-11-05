from __future__ import annotations

import argparse
import json
import sys

from ..cli.common import MacchinaRicordiHelpFormatter, clip
from ..models.record import ModelNotFound, model_display_names, resolve_sig
from .record import KINDS, Session, SessionNotFound, Take
from .store import find_session, list_sessions

SESSION_REFERENCE_HELP = "a session folder, its id (stream_20261005T141908Z) or any unique part of it (20261005T1419)"


def _fail(exc: Exception) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


def _models_text(session: Session, names: dict[str, str]) -> str:
    sigs = session.sigs
    if sigs:
        return ", ".join(names.get(sig, sig) for sig in sigs)
    return ", ".join(model.rsplit("/", 1)[-1] for model in session.models)


def cmd_session_list(args: argparse.Namespace) -> int:
    sig = None
    if args.model:
        try:
            sig = resolve_sig(args.model)
        except ModelNotFound as exc:
            return _fail(exc)
    sessions = list_sessions(kind=args.kind, sig=sig)
    shown = sessions if args.all else sessions[: args.limit]
    if args.json:
        print(json.dumps([session.summary() for session in shown], indent=2))
        return 0
    names = model_display_names()
    rows = [
        (
            session.id,
            session.kind,
            (session.created_at or "")[:16].replace("T", " "),
            clip(_models_text(session, names), 40),
            str(session.take_count),
        )
        for session in shown
    ]
    header = ("session", "kind", "created (UTC)", "models", "takes")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    hidden = len(sessions) - len(shown)
    print(f"\n{len(sessions)} sessions" + (f" ({hidden} older not shown: --all)" if hidden else ""))
    return 0


def _take_line(take: Take) -> str:
    bits = [f"#{take.index}", take.label]
    if take.role != "take":
        bits.append(f"[{take.role}]")
    if take.seed is not None:
        bits.append(f"seed {take.seed}")
    for key in ("temperature", "top_k"):
        if key in take.controls:
            bits.append(f"{key} {take.controls[key]}")
    if take.error:
        bits.append(f"error: {take.error}")
    elif take.audio is not None and not take.audio.exists():
        bits.append("(audio missing)")
    return "  ".join(str(bit) for bit in bits if bit != "")


def cmd_session_show(args: argparse.Namespace) -> int:
    try:
        session = find_session(args.ref)
    except SessionNotFound as exc:
        return _fail(exc)
    takes = session.takes() if args.takes or args.json else ()
    if args.json:
        payload = session.summary()
        payload["takes"] = [take.to_json(session.folder) for take in takes]
        print(json.dumps(payload, indent=2))
        return 0
    names = model_display_names()
    print(f"session:   {session.id}")
    print(f"kind:      {session.kind}")
    print(f"folder:    {session.path}")
    print(f"created:   {session.created_at or '?'}")
    print(f"finished:  {session.finished_at or ('(live or interrupted)' if session.kind == 'stream' else '?')}")
    print(f"models:    {_models_text(session, names) or '?'}")
    if session.parent:
        print(f"from:      {session.parent}")
    controls = {key: value for key, value in session.controls.items() if key != "grid" and value not in (None, "")}
    if controls:
        print("controls:  " + ", ".join(f"{key}={value}" for key, value in list(controls.items())[:12]))
    grid = session.controls.get("grid")
    if grid:
        print("grid:      " + ", ".join(f"{key}={values}" for key, values in grid.items()))
    print(f"takes:     {session.take_count}" + ("" if args.takes else f"  (--takes lists them; one is {session.id}#0)"))
    if not session.sealed:
        print(f"record:    read from {session.native or 'its own files'} (made before sessions had session.json)")
    for take in takes:
        print("  " + _take_line(take))
    return 0


def register_session_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    session = subparsers.add_parser(
        "session",
        help="List and inspect sessions: everything play made, as takes",
        description=(
            "Every output of `play` is a session of takes: generate and continue make one batch, a sweep one take "
            "per grid point, a stream one take per round, a duet a mix and two voices, variations N re-performances "
            "of a parent take. Takes are numbered from 0; <session>#<n> names one. See docs/sessions.md."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli session list
  python -m macchina_ricordi.cli session list --kind stream --model deep
  python -m macchina_ricordi.cli session show 20261005T141908Z --takes
  python -m macchina_ricordi.cli judge ab stream_20261005T141908Z sweep_20261006T090000Z
  python -m macchina_ricordi.cli play replay generate_20261006T084943Z""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    session_subparsers = session.add_subparsers(dest="session_command", required=True)

    listing = session_subparsers.add_parser(
        "list", help="Sessions, newest first", formatter_class=MacchinaRicordiHelpFormatter
    )
    listing.add_argument("--kind", choices=KINDS, default=None, help="Only sessions of this kind")
    listing.add_argument("--model", default=None, help="Only sessions that used this model (any model reference)")
    listing.add_argument("--limit", type=int, default=30, help="How many to show (default: 30)")
    listing.add_argument("--all", action="store_true", help="Show every session")
    listing.add_argument("--json", action="store_true", help="Machine-readable output: one object per session")
    listing.set_defaults(func=cmd_session_list)

    show = session_subparsers.add_parser(
        "show", help="One session: kind, models, controls and takes", formatter_class=MacchinaRicordiHelpFormatter
    )
    show.add_argument("ref", help=f"The session: {SESSION_REFERENCE_HELP}")
    show.add_argument("--takes", action="store_true", help="List every take")
    show.add_argument("--json", action="store_true", help="Machine-readable output, takes included")
    show.set_defaults(func=cmd_session_show)
