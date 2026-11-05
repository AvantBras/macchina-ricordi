"""`runtime`: set up and check the AudioCraft/MusicGen runtime."""

from __future__ import annotations

import argparse
import json

from ..cli.common import MacchinaRicordiHelpFormatter


def cmd_runtime_setup(args: argparse.Namespace) -> int:
    from .setup import setup_repo_managed_musicgen_runtime

    result = setup_repo_managed_musicgen_runtime()
    print(f"musicgen_python: {result.musicgen_python}")
    print(f"audiocraft_dir: {result.audiocraft_dir}")
    print(f"audiocraft_commit: {result.audiocraft_commit_sha}")
    return 0


def cmd_runtime_check(args: argparse.Namespace) -> int:
    from .setup import run_musicgen_preflight

    report = run_musicgen_preflight(
        explicit_python=args.musicgen_python,
        explicit_audiocraft_dir=args.audiocraft_dir,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"musicgen_python: {report.musicgen_python}")
        if report.audiocraft_dir is not None:
            print(f"audiocraft_dir: {report.audiocraft_dir}")
        for check in report.checks:
            print(f"{check.status}: {check.name}: {check.message}")
        print(f"ready: {report.is_ready}")
    return 0 if report.is_ready else 1


def register_runtime_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    runtime = subparsers.add_parser(
        "runtime",
        help="Manage the repo-scoped MusicGen runtime and checkout",
        description="Set up or inspect the repo-managed AudioCraft/MusicGen runtime.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    runtime_subparsers = runtime.add_subparsers(dest="runtime_command", required=True)

    runtime_setup = runtime_subparsers.add_parser(
        "setup",
        help="Create or refresh the repo-managed MusicGen runtime",
        description="Create or refresh the .venv-musicgen runtime and pinned AudioCraft checkout.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    runtime_setup.set_defaults(func=cmd_runtime_setup)

    runtime_check = runtime_subparsers.add_parser(
        "check",
        help="Check that the MusicGen runtime and AudioCraft checkout are ready",
        description="Run a preflight check for AudioCraft, Dora, and the MusicGen runtime.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    runtime_check.add_argument(
        "--audiocraft-dir",
        default=None,
        help="Path to an AudioCraft checkout; defaults to the repo-managed .audiocraft checkout",
    )
    runtime_check.add_argument(
        "--musicgen-python",
        default=None,
        help=(
            "Python executable whose environment contains audiocraft and dora; "
            "defaults to the repo-managed .venv-musicgen runtime"
        ),
    )
    runtime_check.add_argument(
        "--json",
        action="store_true",
        help="Emit the preflight report as JSON",
    )
    runtime_check.set_defaults(func=cmd_runtime_check)

