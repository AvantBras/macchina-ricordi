"""Build the player as something someone can double-click, for this operating system.

    uv venv --python-preference only-managed --python 3.11 .venv-appliance-build
    uv pip install --python .venv-appliance-build/bin/python onnxruntime numpy sounddevice pyinstaller
    PYTHONPATH=. .venv-appliance-build/bin/python packaging/appliance/build.py --boxes boxes/*.nrbox

Linux gives a folder with a ``macchina-ricordi`` program in it. macOS gives
``macchina-ricordi.app``, zipped with its README. The macOS build has to be made on a Mac
— PyInstaller cannot cross-compile — which is what the ``appliance-macos``
workflow is for.

Build from an environment **without torch**. PyInstaller ships whatever it can
import, and an environment that has torch will quietly bundle 700 MB of it —
which is the whole thing this build exists to avoid. The script refuses if torch
is importable.

The interpreter must have a shared ``libpython`` for PyInstaller to embed.
Debian's system Python does not; uv's managed builds and python.org's do.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = "macchina-ricordi"
VERSION = "0.2.0"
BUNDLE_ID = "local.macchina-ricordi.player"
BUNDLED_DISTRIBUTIONS = ("onnxruntime", "numpy", "cffi", "sounddevice")

EXCLUDES = [
    "torch",
    "torchaudio",
    "onnx",
    "onnxscript",
    "transformers",
    "soundfile",
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
    "sympy",
    "onnxruntime.transformers",
    "onnxruntime.quantization",
    "onnxruntime.tools",
]

README_LINUX = """macchina ricordi — a model that plays itself

1. Run "macchina-ricordi" (double-click it, or type ./macchina-ricordi in a terminal here).
2. A page opens in your web browser. Press Play.

Sound comes out of this computer's speakers, not the browser. Press Quit on the
page to stop it.

Models: press "Add a model" on the page and choose a .nrbox file. They are kept
in {models}. Models in the "boxes" folder next to the program show up too.

Recordings are saved in {recordings}.
"""

README_MAC = """macchina ricordi — a model that plays itself

Needs an Apple Silicon Mac (M1 or later) with macOS {min_macos} or newer.


THE FIRST TIME ONLY

This app is not signed by Apple, so macOS will refuse to open it at first.

1. Double-click "macchina-ricordi". macOS says it cannot verify the app. Click Done.
2. Open System Settings > Privacy & Security. Scroll down: next to the message
   about "macchina-ricordi", click "Open Anyway", then confirm with your password.
   (On macOS 14 or older, you can instead right-click the app, choose Open,
   then Open again.)

From then on it opens with a normal double-click.

If macOS says the app "is damaged", open Terminal and run this line, dragging
the app onto the Terminal window where it says APP:

    xattr -dr com.apple.quarantine APP


USING IT

- Double-click "macchina-ricordi". A page opens in your web browser. Press Play.
- Sound comes from the Mac's speakers. The app has no window and no Dock icon:
  the browser page is its control panel. If you close the tab, double-click
  the app again to get the page back.
- To stop it completely, press Quit on the page.
{models}
- Recordings are saved in Music > macchina-ricordi > recordings.
- If something goes wrong, the file Music > macchina-ricordi > macchina-ricordi.log says what.
"""


def _portaudio_binary() -> list[str]:
    """Linux wheels of sounddevice do not contain PortAudio; the bundle has to."""
    if platform.system() != "Linux":
        return []
    listing = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    for line in listing.splitlines():
        arch_ok = ("x86-64" in line) if platform.machine() == "x86_64" else ("aarch64" in line)
        if "libportaudio.so.2" in line and "=>" in line and arch_ok:
            path = line.split("=>", 1)[1].strip()
            return ["--add-binary", f"{path}:."]
    raise SystemExit("libportaudio.so.2 not found; install it (apt install libportaudio2) and rebuild")


def minimum_macos() -> str:
    """The newest macOS any bundled wheel was built for is the oldest the app runs on.

    numpy's Apple Silicon wheels, for one, are built against Accelerate and ask for
    macOS 14 — so this is read off the wheels rather than assumed.
    """
    newest = (11, 0)
    for name in BUNDLED_DISTRIBUTIONS:
        try:
            wheel = importlib.metadata.distribution(name).read_text("WHEEL") or ""
        except importlib.metadata.PackageNotFoundError:
            continue
        for major, minor in re.findall(r"macosx_(\d+)_(\d+)_", wheel):
            newest = max(newest, (int(major), int(minor)))
    return f"{newest[0]}.{newest[1]}"


def _models_paragraph(boxes: list[Path]) -> str:
    more = """press "Add a model" and choose a .nrbox file. They are kept in
  Music > macchina-ricordi > models."""
    if not boxes:
        return "- Models: " + more
    names = ", ".join(box.stem for box in boxes)
    return f"- It comes with a model already inside ({names}): just press Play.\n- More models: " + more


def _du(path: Path) -> int:
    # PyInstaller links some libraries under two names; count each file once.
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def _finish_macos(dist: Path, boxes: list[Path]) -> tuple[Path, Path]:
    """Turn PyInstaller's bundle into one that behaves like a Mac app, then sign and zip it."""
    app = dist / f"{NAME}.app"
    min_macos = minimum_macos()
    info_path = app / "Contents" / "Info.plist"
    info = plistlib.loads(info_path.read_bytes())
    info.update(
        {
            "CFBundleDisplayName": "macchina ricordi",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": min_macos,
            "LSApplicationCategoryType": "public.app-category.music",
            "NSHighResolutionCapable": True,
            # No window and no event loop: with a Dock icon, macOS would report the
            # app as not responding and its Quit menu would do nothing. The browser
            # page is the interface, and it has its own Quit.
            "LSUIElement": True,
        }
    )
    info_path.write_bytes(plistlib.dumps(info))
    if boxes:
        inside = app / "Contents" / "Resources" / "boxes"
        inside.mkdir(parents=True, exist_ok=True)
        for box in boxes:
            shutil.copy2(box, inside / box.name)
    # Editing the bundle invalidated PyInstaller's signature. Apple Silicon will not
    # run unsigned code at all, so it is re-signed ad hoc: enough to run, not enough
    # to satisfy Gatekeeper without the one-time "Open Anyway".
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)

    folder = dist / f"{NAME}-macos-{platform.machine().lower()}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    # ditto, not shutil: a Mac bundle is full of symlinks and extended attributes
    # that an ordinary copy or zip silently flattens, breaking the signature.
    subprocess.run(["ditto", str(app), str(folder / app.name)], check=True)
    (folder / "README.txt").write_text(
        README_MAC.format(min_macos=min_macos, models=_models_paragraph(boxes)), encoding="utf-8"
    )
    archive = folder.with_suffix(".zip")
    archive.unlink(missing_ok=True)
    subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(folder), str(archive)], check=True)
    print(f"  minimum macOS: {min_macos}")
    return app, archive


def _finish_folder(dist: Path, boxes: list[Path], system: str) -> tuple[Path, Path | None]:
    from appliance.places import models_dir, recordings_dir

    app = dist / NAME
    (app / "README.txt").write_text(
        README_LINUX.format(models=models_dir(), recordings=recordings_dir()), encoding="utf-8"
    )
    boxes_dir = app / "boxes"
    boxes_dir.mkdir(exist_ok=True)
    for box in boxes:
        shutil.copy2(box, boxes_dir / box.name)
    base = dist / f"{NAME}-{system}-{platform.machine().lower()}"
    fmt = "xztar" if system == "linux" else "zip"
    archive = Path(shutil.make_archive(str(base), fmt, root_dir=dist, base_dir=NAME))
    return app, archive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", nargs="*", type=Path, default=[], help=".nrbox files to build into the app")
    ap.add_argument("--dist", type=Path, default=ROOT / "dist" / "appliance")
    ap.add_argument("--work", type=Path, default=ROOT / "build" / "appliance")
    ap.add_argument("--windowed", action="store_true", help="no console window (Windows; always on for macOS)")
    args = ap.parse_args()

    if importlib.util.find_spec("torch") is not None:
        raise SystemExit("torch is importable in this environment; build from one without it (see the docstring)")
    import PyInstaller.__main__

    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system(), "unknown")
    command = [
        str(HERE / "entry.py"),
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        NAME,
        "--distpath",
        str(args.dist),
        "--workpath",
        str(args.work),
        "--specpath",
        str(args.work),
        "--paths",
        str(ROOT),
        "--hidden-import",
        "appliance.panel",
        "--hidden-import",
        "appliance.runtime_onnx",
        *[flag for module in EXCLUDES for flag in ("--exclude-module", module)],
        *_portaudio_binary(),
    ]
    if system == "macos":
        command += ["--windowed", "--osx-bundle-identifier", BUNDLE_ID]
    elif args.windowed:
        command.append("--windowed")
    PyInstaller.__main__.run(command)

    if system == "macos":
        app, archive = _finish_macos(args.dist, args.boxes)
    else:
        app, archive = _finish_folder(args.dist, args.boxes, system)
    print(f"\napp: {app}")
    print(f"  on disk: {_du(app) / 1e6:6.1f} MB  ({len(args.boxes)} model(s) inside)")
    if archive:
        print(f"  to send: {archive.stat().st_size / 1e6:6.1f} MB  ({archive.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
