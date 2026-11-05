from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from .common import MacchinaRicordiHelpFormatter

# The nouns in docs/glossary.md that read or write the MusicGen workspace: they all need the
# checkout's musicgen/pyproject.toml. Audio and analysis nouns (score, listen, segment, loop,
# augment, judge) never need a checkout and are deliberately absent, so they are never blocked.
WORKSPACE_COMMAND_NOUNS: frozenset[str] = frozenset(
    {"data", "train", "model", "play", "tidy", "runtime", "gui", "session"}
)

# The CLI follows the nouns (docs/glossary.md). Commands moved there in October 2026; their old spellings keep
# working through this table, so scripts, notes and agent habits do not break. Most specific first.
LEGACY_COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("musicgen", "runtime"), ("runtime",)),
    (("musicgen", "data"), ("data",)),
    (("musicgen", "train", "sidecar-cpu"), ("train", "sidecar")),
    (("musicgen", "train"), ("train",)),
    (("musicgen", "generate", "unconditional"), ("play", "generate")),
    (("musicgen", "generate", "continue"), ("play", "continue")),
    (("musicgen", "generate", "sweep"), ("play", "sweep")),
    (("musicgen", "stream"), ("play", "stream")),
    (("musicgen", "monitor"), ("play", "monitor")),
    (("musicgen", "duet"), ("play", "duet")),
    (("musicgen", "variations"), ("play", "variations")),
    (("musicgen", "admin", "resume-xp"), ("train", "resume")),
    (("musicgen", "admin", "handoff-resume"), ("train", "handoff")),
    (("musicgen", "admin", "plateau-watch"), ("train", "watch")),
    (("musicgen", "admin", "replay"), ("play", "replay")),
    (("musicgen", "admin", "merge"), ("model", "merge")),
    (("musicgen", "admin", "perturb"), ("model", "perturb")),
    (("musicgen", "admin", "score-curve"), ("model", "curve")),
    (("musicgen", "admin", "lineage"), ("model", "lineage")),
    (("musicgen", "admin", "catalog"), ("model", "catalog")),
    (("musicgen", "admin", "prune"), ("tidy", "prune")),
    (("musicgen", "admin", "clean-checkpoints"), ("tidy", "clean")),
    (("musicgen", "admin", "archive-checkpoints"), ("tidy", "archive")),
    (("musicgen", "admin", "backfill-metadata"), ("tidy", "backfill")),
    (("score", "ab"), ("judge", "ab")),
    (("score", "label"), ("judge", "label")),
    (("score", "labels"), ("judge", "labels")),
    (("score", "prefs"), ("judge", "prefs")),
    (("score", "judge"), ("judge", "witness")),
    (("score", "describe-calibrate"), ("listen", "describe-calibrate")),
    (("score", "describe-dir"), ("listen", "describe-dir")),
    (("score", "describe"), ("listen", "describe")),
    (("score", "diff"), ("listen", "diff")),
    (("score", "compare"), ("listen", "compare")),
    (("listen", "similar"), ("score", "similar")),
    (("listen", "novelty"), ("score", "novelty")),
)


def translate_legacy(argv: Sequence[str]) -> tuple[list[str], str | None]:
    """An old spelling (``musicgen admin resume-xp``) as today's (``train resume``), with a note saying so."""
    for old, new in LEGACY_COMMANDS:
        if tuple(argv[: len(old)]) == old:
            return [*new, *argv[len(old) :]], f"note: `{' '.join(old)}` is now `{' '.join(new)}`"
    return list(argv), None


def build_parser() -> argparse.ArgumentParser:
    from ..augmentation.cli import register_augmentation_group
    from ..data.cli import register_data_group
    from ..gui.cli import register_gui_group
    from ..instrument.cli import register_play_group
    from ..judging.cli import register_judge_group
    from ..listening.cli import register_listening_group
    from ..looping.cli import register_loop_group
    from ..models.cli import register_model_group
    from ..musicgen.cli import register_runtime_group
    from ..scoring.cli import register_scoring_group
    from ..segmentation.cli import register_segmentation_group
    from ..sessions.cli import register_session_group
    from ..tidy.cli import register_tidy_group
    from ..training.cli import register_train_group

    parser = argparse.ArgumentParser(
        description=(
            "macchina_ricordi, by noun: data (datasets) → train (a recipe on a dataset) → model → play → session → "
            "listen, score and judge. Plus the tools around them: segment, augment, loop, tidy, runtime, gui."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli data build my_dataset examples/sample_data/<album> --slice 10
  python -m macchina_ricordi.cli train queue add --dataset my_dataset --recipe reshape-nano --epochs 30 --start
  python -m macchina_ricordi.cli model list
  python -m macchina_ricordi.cli play generate <model>@best --listen
  python -m macchina_ricordi.cli score file sample.wav
  python -m macchina_ricordi.cli gui --open""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_data_group(subparsers)
    register_train_group(subparsers)
    register_model_group(subparsers)
    register_play_group(subparsers)
    register_session_group(subparsers)
    register_listening_group(subparsers)
    register_scoring_group(subparsers)
    register_judge_group(subparsers)
    register_segmentation_group(subparsers)
    register_augmentation_group(subparsers)
    register_loop_group(subparsers)
    register_tidy_group(subparsers)
    register_runtime_group(subparsers)
    register_gui_group(subparsers)

    return parser


def workspace_guard_error(args: argparse.Namespace) -> str | None:
    """Actionable error for a workspace command with no valid checkout, else ``None``.

    Runs after argparse has parsed (so top-level and sub-``--help`` are already answered) but
    before the command body, so a workspace command without the checkout can never touch the
    package install (e.g. write into site-packages) or, in ``gui``'s case, start a server. An
    explicit root (``gui --repo-root <dir>``) is honoured: it passes when it is a valid checkout.
    Audio/analysis nouns never reach here and are never blocked.
    """
    from ..musicgen.config import find_workspace_root, get_repo_root, workspace_guidance

    noun = getattr(args, "command", None)
    if noun not in WORKSPACE_COMMAND_NOUNS:
        return None
    explicit_root = getattr(args, "repo_root", None)
    if find_workspace_root(explicit_root) is not None:
        return None
    return workspace_guidance(get_repo_root(explicit_root))


def main(argv: list[str] | None = None) -> int:
    from ..musicgen.config import WorkspaceMissingError

    args_in, note = translate_legacy(sys.argv[1:] if argv is None else argv)
    if note:
        print(note, file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args(args_in)
    func: Callable[[argparse.Namespace], int] = args.func
    guard_error = workspace_guard_error(args)
    if guard_error is not None:
        # A workspace command with no checkout: say how to get one instead of a traceback
        # (the body would raise, or worse, write into the bare install). Runs after argparse,
        # so --help on any noun already worked.
        print(f"error: {guard_error}", file=sys.stderr)
        return 2
    try:
        return int(func(args))
    except WorkspaceMissingError as exc:
        # A command needs the source checkout and none is present: say how to get one
        # instead of dumping a traceback (audio/analysis commands never reach this).
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
