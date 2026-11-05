from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_EXTENSIONS = ("wav", "flac", "mp3", "ogg", "m4a", "aiff", "aif")


def parse_extensions(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_EXTENSIONS
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    parsed = tuple(part.strip().lower().lstrip(".") for part in parts if part.strip())
    return parsed or DEFAULT_EXTENSIONS


def iter_audio_files(
    root: Path,
    *,
    recursive: bool,
    extensions: tuple[str, ...],
) -> Iterable[Path]:
    ext_set = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in ext_set:
            yield path
