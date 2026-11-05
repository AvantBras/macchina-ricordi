"""Where the built app starts. With no arguments it opens the control panel."""

from __future__ import annotations

import ctypes.util
import os
import platform
import sys
import time
from pathlib import Path


def _use_bundled_portaudio() -> None:
    """Point sounddevice at the PortAudio shipped inside the app.

    On Linux, sounddevice finds PortAudio through ``ctypes.util.find_library``,
    which searches the system and never the app's own folder — so a machine
    without ``libportaudio2`` installed would fail to open the speakers even
    though the library is sitting right beside the program.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or not bundle or platform.system() != "Linux":
        return
    bundled = Path(bundle) / "libportaudio.so.2"
    if not bundled.exists():
        return
    system_lookup = ctypes.util.find_library

    def find_library(name: str):
        return str(bundled) if "portaudio" in name else system_lookup(name)

    ctypes.util.find_library = find_library


def _log_when_launched_from_the_desktop() -> None:
    """Send output to a log file when there is no terminal to send it to.

    Launched from Finder, a macOS app's output goes nowhere (and a windowed build
    on Windows has no stdout at all). A log in the person's own folder is then the
    only record of what went wrong — which matters when the person is a friend on
    another continent. ``MACCHINA_RICORDI_CONSOLE=1`` keeps output on the terminal.
    """
    if not getattr(sys, "frozen", False) or os.environ.get("MACCHINA_RICORDI_CONSOLE") == "1":
        return
    interactive = sys.stdout is not None and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if interactive:
        return
    from appliance.places import log_path

    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - lives as long as the app
    sys.stdout = sys.stderr = stream
    print(f"--- started {time.strftime('%Y-%m-%d %H:%M:%S')} on {platform.platform()} ---", flush=True)


_use_bundled_portaudio()
_log_when_launched_from_the_desktop()

from appliance.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
