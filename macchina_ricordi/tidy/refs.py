"""Where XPs and their checkpoints are referenced from, so cleanup can leave alone what something points at.

A reference is a mention of ``xps/<sig>/checkpoint….th`` (that checkpoint) or ``xps/<sig>`` (the XP)
in the project's text records: generation runs, sweeps, stream sessions, merges and perturbations,
child XPs' metadata, workspace logs, research notes. Markdown and TOML files (notes, the experiment
ledger) also refer to an XP by its bare sig. Mentions inside an XP's own folder do not count for that
XP: its own reports and sidecars list its checkpoints.

A plain text scan, not a model of every artifact format: whatever records a path is found, including
formats written after this module.
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = (".json", ".jsonl", ".toml", ".csv", ".md", ".txt", ".yaml", ".yml")
BARE_SIG_SUFFIXES = (".md", ".toml")
MAX_FILE_BYTES = 64 * 1024 * 1024
SCANNED_REPO_DIRS = ("artifacts", "research", "docs")
# prepared datasets: large, and never a record of a model
WORKSPACE_SKIPPED_DIRS = frozenset({"data", "egs", "clips"})
# cleanup's own reports quote the references that protected a checkpoint; they are not uses of it
IGNORED_FILE_NAMES = frozenset({"checkpoint_cleanup_report.json"})

_PATH_REF = re.compile(rb"xps/([0-9A-Za-z_]+)(?:/(checkpoint[^\"'\s/\\,;)\]]*\.th))?")
_BARE_SIG = re.compile(rb"(?<![0-9A-Za-z_])([0-9a-f]{8})(?![0-9A-Za-z_])")


@dataclass(frozen=True)
class CheckpointReferences:
    checkpoints: dict[tuple[str, str], tuple[str, ...]]
    """(sig, checkpoint file name) -> the files that mention it, repo-relative."""
    xps: dict[str, tuple[str, ...]]
    """sig -> the files that mention the XP at all (a path into it, or its bare sig in notes)."""

    def for_checkpoint(self, sig: str, name: str) -> tuple[str, ...]:
        return self.checkpoints.get((sig, name), ())

    def for_xp(self, sig: str) -> tuple[str, ...]:
        return self.xps.get(sig, ())


def scan_checkpoint_references(repo_root: Path, dora_dir: Path | None = None) -> CheckpointReferences:
    repo_root = repo_root.resolve()
    xps_root = (dora_dir or repo_root / ".dora").resolve() / "xps"
    known_sigs = {path.name for path in xps_root.iterdir() if path.is_dir()} if xps_root.is_dir() else set()
    checkpoints: dict[tuple[str, str], set[str]] = {}
    xps: dict[str, set[str]] = {}
    for path, own_sig in _text_files(repo_root, xps_root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        shown = _display_path(path, repo_root)
        for match in _PATH_REF.finditer(data):
            sig = match.group(1).decode("ascii")
            if sig == own_sig:
                continue
            xps.setdefault(sig, set()).add(shown)
            if match.group(2):
                checkpoints.setdefault((sig, match.group(2).decode("ascii")), set()).add(shown)
        if path.suffix in BARE_SIG_SUFFIXES:
            for match in _BARE_SIG.finditer(data):
                sig = match.group(1).decode("ascii")
                if sig in known_sigs and sig != own_sig:
                    xps.setdefault(sig, set()).add(shown)
    return CheckpointReferences(
        checkpoints={key: tuple(sorted(files)) for key, files in checkpoints.items()},
        xps={sig: tuple(sorted(files)) for sig, files in xps.items()},
    )


def _text_files(repo_root: Path, xps_root: Path) -> Iterator[tuple[Path, str | None]]:
    for name in SCANNED_REPO_DIRS:
        yield from ((path, None) for path in _walk(repo_root / name))
    from ..data.datasets import dataset_dirs

    for workspace in dataset_dirs(repo_root):
        yield from ((path, None) for path in _walk(workspace, skip_top=WORKSPACE_SKIPPED_DIRS))
    if xps_root.is_dir():
        for xp_folder in sorted(xps_root.iterdir()):
            if xp_folder.is_dir():
                yield from ((path, xp_folder.name) for path in _walk(xp_folder))


def _walk(root: Path, *, skip_top: Collection[str] = ()) -> Iterator[Path]:
    if not root.is_dir():
        return
    for directory, subdirectories, files in os.walk(root):
        if skip_top and Path(directory) == root:
            subdirectories[:] = [name for name in subdirectories if name not in skip_top]
        for name in files:
            if name.endswith(TEXT_SUFFIXES) and name not in IGNORED_FILE_NAMES:
                yield Path(directory) / name


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


__all__ = ["CheckpointReferences", "scan_checkpoint_references"]
