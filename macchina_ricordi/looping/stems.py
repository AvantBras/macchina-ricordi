from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..musicgen.runtime import require_musicgen_python

DEFAULT_DEMUCS_MODEL = "htdemucs"
DEMUCS_DEVICES = ("cpu", "cuda")
VOCALS_FILENAME = "vocals.wav"
INSTRUMENTAL_FILENAME = "no_vocals.wav"
STATUS_FILENAME = "stems_status.json"
_SCRATCH_DIRNAME = "_demucs"


@dataclass(frozen=True)
class StemPaths:
    vocals: str
    instrumental: str
    reused: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def build_demucs_command(python: str, audio_path: Path, output_dir: Path, *, model: str, device: str) -> list[str]:
    return [
        python,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-d",
        device,
        "-o",
        str(output_dir),
        "--filename",
        "{stem}.{ext}",
        str(audio_path),
    ]


def _check_stems_source(status_path: Path, source: Path) -> None:
    if not status_path.is_file():
        return
    recorded = json.loads(status_path.read_text(encoding="utf-8")).get("source_path")
    if recorded and recorded != str(source):
        raise ValueError(
            f"{status_path.parent} holds stems for {recorded}, not {source}. "
            "Pass --force-separate or choose another --stems-dir."
        )


def separate_vocals(
    audio_path: str | Path,
    stems_dir: str | Path,
    *,
    python: str | None = None,
    model: str = DEFAULT_DEMUCS_MODEL,
    device: str = "cpu",
    force: bool = False,
) -> StemPaths:
    """Split ``audio_path`` into ``vocals.wav`` + ``no_vocals.wav`` inside ``stems_dir`` with Demucs.

    Existing stems are reused unless ``force`` is set; hand-made stems (no status file) are trusted.
    """
    if device not in DEMUCS_DEVICES:
        raise ValueError(f"Unknown device {device!r}; expected one of {DEMUCS_DEVICES}")
    source = Path(audio_path).expanduser().resolve()
    destination = Path(stems_dir).expanduser().resolve()
    vocals = destination / VOCALS_FILENAME
    instrumental = destination / INSTRUMENTAL_FILENAME
    status_path = destination / STATUS_FILENAME

    if not force and vocals.is_file() and instrumental.is_file():
        _check_stems_source(status_path, source)
        return StemPaths(vocals=str(vocals), instrumental=str(instrumental), reused=True)

    # Demucs lives in the AudioCraft runtime (.venv-musicgen), not in the main env.
    interpreter = require_musicgen_python(python, flag="--demucs-python")
    scratch = destination / _SCRATCH_DIRNAME
    command = build_demucs_command(interpreter, source, scratch, model=model, device=device)
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    subprocess.run(command, check=True, env=env)
    elapsed = time.monotonic() - started

    produced = scratch / model
    for name, target in ((VOCALS_FILENAME, vocals), (INSTRUMENTAL_FILENAME, instrumental)):
        if not (produced / name).is_file():
            raise FileNotFoundError(f"Demucs finished but {produced / name} is missing")
        (produced / name).replace(target)
    shutil.rmtree(scratch, ignore_errors=True)

    status = {
        "source_path": str(source),
        "model": model,
        "device": device,
        "python": interpreter,
        "elapsed_seconds": round(elapsed, 2),
        "vocals": str(vocals),
        "instrumental": str(instrumental),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return StemPaths(vocals=str(vocals), instrumental=str(instrumental), reused=False)
