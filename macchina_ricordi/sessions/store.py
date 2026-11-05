"""Where sessions live, finding one by reference, and sealing a finished one.

New sessions go to ``artifacts/sessions/<YYYY-MM-DD>/<kind>_<run id>[_<label>]/``. The folders under
``artifacts/sessions/`` that are not dates are the hand-kept listening notebook and are not sessions. Sessions made
before October 2026 stay where they were made (``artifacts/runs``, ``sweeps``, ``streams``, ``duets``,
``variations``) and are listed with the rest.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from ..instrument.run_paths import LEGACY_GENERATED_AUDIO_DIR, iter_sweep_index_paths
from ..musicgen.config import get_repo_root
from ..musicgen.shared import write_text_atomic
from .read import describe_native, load_session
from .record import SESSION_FILENAME, TAKES_FILENAME, Session, SessionNotFound, Take

SESSIONS_DIR = Path("artifacts/sessions")
LEGACY_HOMES = {
    "generate": Path("artifacts/runs"),
    "sweep": Path("artifacts/sweeps"),
    "stream": Path("artifacts/streams"),
    "duet": Path("artifacts/duets"),
    "variations": Path("artifacts/variations"),
}
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
TAKE_SEPARATOR = "#"


def repo_root_or_default(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root).expanduser().resolve() if repo_root is not None else get_repo_root()


def sessions_root(repo_root: str | Path | None = None) -> Path:
    return repo_root_or_default(repo_root) / SESSIONS_DIR


def utc_run_id(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")


def session_folder_name(kind: str, run_id: str, label: str | None = None) -> str:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", label.strip()).strip("_") if label else ""
    return f"{kind}_{run_id}" + (f"_{slug}" if slug else "")


def new_session_dir(
    kind: str,
    *,
    run_id: str | None = None,
    label: str | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """A fresh, not yet created folder for a new session: ``artifacts/sessions/<day>/<kind>_<run id>[_<label>]``."""
    run_id = run_id or utc_run_id()
    try:
        day = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").date().isoformat()
    except ValueError:
        day = datetime.now(UTC).date().isoformat()
    parent = sessions_root(repo_root) / day
    name = session_folder_name(kind, run_id, label)
    candidate = parent / name
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{name}-{suffix}"
        suffix += 1
    return candidate


def seal_session(folder: str | Path, *, label: str | None = None, parent: str | None = None) -> Session:
    """Write ``session.json`` (and ``takes.jsonl``, except for a stream) from the producer's own files.

    Producers call this when they finish; calling it again after more takes or scores land is safe.
    """
    folder = Path(folder).expanduser().resolve()
    native = describe_native(folder)
    takes = native.takes()
    batch = native.kind != "stream"
    session = Session(
        id=folder.name,
        kind=native.kind,
        path=folder,
        created_at=native.created_at,
        finished_at=native.finished_at or (datetime.now(UTC).isoformat() if batch else None),
        label=label if label is not None else native.label,
        models=native.models,
        engine=native.engine,
        controls=native.controls,
        parent=parent if parent is not None else native.parent,
        take_count=len(takes),
        takes_log=TAKES_FILENAME if batch else native.takes_log,
        native=native.native,
        sealed=True,
        facts=native.facts,
    )
    if batch:
        write_text_atomic(
            folder / TAKES_FILENAME,
            "".join(json.dumps(take.to_json(folder)) + "\n" for take in takes),
        )
    write_text_atomic(folder / SESSION_FILENAME, json.dumps(session.to_json(), indent=2) + "\n")
    return session


# ── listing ──


def iter_session_paths(repo_root: str | Path | None = None) -> Iterator[Path]:
    """Every session folder (or legacy metadata file), new home first, each once."""
    root = repo_root_or_default(repo_root)
    seen: set[Path] = set()

    def fresh(path: Path) -> bool:
        resolved = path.resolve()
        if resolved in seen:
            return False
        seen.add(resolved)
        return True

    home = root / SESSIONS_DIR
    if home.is_dir():
        for day in sorted((path for path in home.iterdir() if _DAY.fullmatch(path.name)), reverse=True):
            for folder in sorted((path for path in day.iterdir() if path.is_dir()), reverse=True):
                if fresh(folder):
                    yield folder
    runs = root / LEGACY_HOMES["generate"]
    if runs.is_dir():
        for metadata in sorted(runs.glob("**/run.json")):
            if fresh(metadata.parent):
                yield metadata.parent
    flat = root / LEGACY_GENERATED_AUDIO_DIR
    if flat.is_dir():
        for metadata in sorted(flat.glob("*.json")):
            if fresh(metadata):
                yield metadata
    for index in iter_sweep_index_paths(root):
        if fresh(index.parent):
            yield index.parent
    for kind in ("stream", "duet", "variations"):
        legacy = root / LEGACY_HOMES[kind]
        if legacy.is_dir():
            for folder in sorted(path for path in legacy.iterdir() if path.is_dir()):
                if fresh(folder):
                    yield folder


def list_sessions(
    repo_root: str | Path | None = None,
    *,
    kind: str | None = None,
    sig: str | None = None,
) -> list[Session]:
    """Every readable session, newest first."""
    sessions = []
    for path in iter_session_paths(repo_root):
        try:
            session = load_session(path)
        except (SessionNotFound, OSError, ValueError):
            continue
        if kind is not None and session.kind != kind:
            continue
        if sig is not None and sig not in session.sigs:
            continue
        sessions.append(session)
    sessions.sort(key=lambda session: session.created_at or "", reverse=True)
    return sessions


def stream_folders(repo_root: str | Path | None = None) -> list[Path]:
    """Every stream folder, in both homes, oldest first, whether or not it got as far as a first round."""
    root = repo_root_or_default(repo_root)
    found: list[Path] = []
    legacy = root / LEGACY_HOMES["stream"]
    if legacy.is_dir():
        found.extend(path for path in legacy.iterdir() if path.is_dir())
    home = root / SESSIONS_DIR
    if home.is_dir():
        for day in home.iterdir():
            if _DAY.fullmatch(day.name):
                found.extend(path for path in day.glob("stream_*") if path.is_dir())
    return sorted(found, key=_stream_sort_key)


def stream_session_dirs(repo_root: str | Path | None = None) -> list[Path]:
    """Every stream session folder holding a performance log, in both homes, oldest first."""
    return [path for path in stream_folders(repo_root) if (path / "performance.jsonl").is_file()]


def latest_stream_dir(repo_root: str | Path | None = None) -> Path | None:
    dirs = stream_session_dirs(repo_root)
    return dirs[-1] if dirs else None


def _stream_sort_key(path: Path) -> str:
    return path.name.removeprefix("stream_")


# ── references ──


def find_session(ref: str | Path, *, repo_root: str | Path | None = None) -> Session:
    """A session by folder, id, run id or a unique part of its id (``stream_20261005T1419``, ``20261005T141908Z``)."""
    root = repo_root_or_default(repo_root)
    text = str(ref).strip()
    for candidate in (Path(text).expanduser(), root / text):
        if candidate.exists():
            return load_session(candidate)
    sessions = list_sessions(root)
    exact = [session for session in sessions if session.id == text]
    if len(exact) == 1:
        return exact[0]
    partial = [session for session in sessions if text in session.id]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(session.id for session in partial[:6])
        more = f" and {len(partial) - 6} more" if len(partial) > 6 else ""
        raise SessionNotFound(f"{text!r} matches several sessions: {names}{more}")
    close = difflib.get_close_matches(text, [session.id for session in sessions], n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(close)}?" if close else " `session list` shows every session."
    raise SessionNotFound(f"no session called {text!r}.{hint}")


def find_take(ref: str | Path, *, repo_root: str | Path | None = None) -> tuple[Session, Take]:
    """``<session>#<n>`` or a take's audio file → (session, take)."""
    text = str(ref).strip()
    if TAKE_SEPARATOR in text:
        session_ref, _, number = text.rpartition(TAKE_SEPARATOR)
        if not number.isdigit():
            raise SessionNotFound(f"{text!r}: a take reference is <session>#<take number>")
        session = find_session(session_ref, repo_root=repo_root)
        return session, session.take(int(number))
    audio = Path(text).expanduser().resolve()
    if audio.is_file():
        for folder in audio.parents:
            try:
                session = load_session(folder)
            except SessionNotFound:
                continue
            for take in session.takes():
                if take.audio is not None and take.audio.resolve() == audio:
                    return session, take
            break
    raise SessionNotFound(f"{text!r} is not a take: use <session>#<take number>, or a take's audio file")


def resolve_session_folders(sources: Iterable[str | Path]) -> list[Path]:
    """Each source as a folder: a path as given, else a session id (``stream_20261005T141908Z``, ``20261005T1419``).

    A session found by id comes back relative to the working directory when it is under it, the same path
    ``judge ab artifacts/streams/<id>`` gives: the preference store keys pairs by clip path, so one clip must
    not be spelled two ways.
    """
    cwd = Path.cwd().resolve()
    folders: list[Path] = []
    for source in sources:
        path = Path(source).expanduser()
        if path.exists():
            folders.append(path)
            continue
        folder = find_session(str(source)).folder.resolve()
        folders.append(folder.relative_to(cwd) if folder.is_relative_to(cwd) else folder)
    return folders
