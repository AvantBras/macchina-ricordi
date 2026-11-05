"""Where things live on the listener's machine, on Linux, macOS and Windows.

Two kinds of place. The person's own folder — models they add, recordings they
make — has to be somewhere they can find without a terminal. The app's folders
are wherever the build put its files, which on macOS is less obvious than it
sounds.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

APP_NAME = "macchina-ricordi"


def user_dir() -> Path:
    """One folder a person can find. On macOS that is inside Music, which Finder
    shows in its sidebar; elsewhere it is the home folder."""
    override = os.environ.get("MACCHINA_RICORDI_HOME")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Music" / APP_NAME
    return Path.home() / APP_NAME


def models_dir() -> Path:
    return user_dir() / "models"


def recordings_dir() -> Path:
    return user_dir() / "recordings"


def log_path() -> Path:
    return user_dir() / f"{APP_NAME}.log"


def app_dirs() -> list[Path]:
    """Folders belonging to the running build, empty when running from source.

    A folder build keeps its executable at the top of the folder. A macOS bundle
    keeps it at ``X.app/Contents/MacOS``: models built into the bundle are in
    ``Contents/Resources/boxes``, and the folder it was unpacked into is above the
    ``.app``. That last one is unreliable — Gatekeeper runs a quarantined app from
    a randomised read-only copy (App Translocation), which contains the bundle and
    nothing beside it. Models built in, or added through the panel, survive that;
    models dropped next to the app may not.
    """
    if not getattr(sys, "frozen", False):
        return []
    executable = Path(sys.executable).resolve()
    bundle = next((p for p in executable.parents if p.suffix == ".app"), None)
    if bundle is not None:
        dirs = [bundle / "Contents" / "Resources" / "boxes", bundle.parent / "boxes", bundle.parent]
    else:
        dirs = [executable.parent / "boxes", executable.parent]
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        dirs.append(Path(bundled) / "boxes")
    return dirs
