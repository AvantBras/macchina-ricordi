from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

from macchina_ricordi.cli.common import MacchinaRicordiHelpFormatter
from macchina_ricordi.judging.preference import DEFAULT_PREFERENCE_PATH
from macchina_ricordi.musicgen.config import get_repo_root

from .ab import AbManager
from .data import GuiIndex
from .server import serve


def cmd_gui(args: argparse.Namespace) -> int:
    repo_root = get_repo_root(args.repo_root)
    # A/B verdicts store clip paths relative to the working directory, as `judge ab` does when
    # run from the repo root; both front ends must agree for already-judged pairs to be skipped.
    os.chdir(repo_root)
    index = GuiIndex(repo_root)
    index.refresh()
    ab = AbManager(repo_root, store_path=Path(args.store).expanduser())
    server = serve(index, host=args.host, port=args.port, ab=ab)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"macchina_ricordi gui: {len(index.summaries())} XPs, serving {url} (Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ab.close_all()
    return 0


def register_gui_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    gui = subparsers.add_parser(
        "gui",
        help="Browse XPs, edit the ledger, run blind A/B sessions, launch training and tidy .dora in a local web page",
        description=(
            "Serve a local web GUI over .dora/xps, generation runs and the experiment ledger. It writes only "
            "when asked: experiment ledger edits, blind A/B verdicts, training launches (which run the "
            "`train run` CLI detached, so they outlive the GUI), datasets (`data corpus`, `data prepare` "
            "into new workspaces, rebalance weights), merged and perturbed checkpoints (`model merge`, "
            "`model perturb`) and housekeeping (`tidy clean`, `tidy prune` and `tidy archive`), each "
            "previewed first."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli gui
  python -m macchina_ricordi.cli gui --port 9000 --open""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    gui.add_argument(
        "--host",
        default="127.0.0.1",
        choices=("127.0.0.1", "localhost"),
        help="Interface to bind; the page serves this machine only (default: 127.0.0.1)",
    )
    gui.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    gui.add_argument("--open", action="store_true", help="Open the page in the default browser")
    gui.add_argument("--repo-root", default=None, help="Repository root (default: this checkout)")
    gui.add_argument(
        "--store",
        default=str(DEFAULT_PREFERENCE_PATH),
        help="Preference JSONL store for blind A/B verdicts, relative to the repo root (same default as judge ab)",
    )
    gui.set_defaults(func=cmd_gui)
