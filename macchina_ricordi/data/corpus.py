"""Assemble a clip folder for ``data prepare`` from source folders.

Each source file becomes one clip (``whole``) or fixed-length slices. Clips are named
``<source>__<file>`` or ``<source>__<file>__<index>``: the part before the first ``__`` is
the source folder, which is how ``data rebalance`` and the GUI group clips.

Whole files are hard-linked when possible (copied otherwise) and keep their format. Slices
are decoded with ffmpeg and written as FLAC, 32 kHz mono by default (what the MusicGen
models here train on), so any input format works.

``corpus.json`` next to ``clips/`` records the recipe (sources, slicing), the counts per
source, what was skipped and why, and every clip's origin (file, start, duration).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..musicgen.shared import utc_now_iso, write_json, write_text_atomic

CORPUS_SCHEMA = "macchina_ricordi.corpus.v1"
CORPUS_MANIFEST = "corpus.json"
CLIPS_DIRNAME = "clips"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus", ".aac", ".wma")
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_CHANNELS = 1
# "__" separates the folder from the file in clip names, so a folder name must never contain it
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+|_{2,}")


class CorpusError(ValueError):
    pass


def sanitize(name: str) -> str:
    return _UNSAFE.sub("_", name).strip("_.") or "x"


@dataclass(frozen=True)
class SourceFile:
    path: Path
    group: str
    duration: float | None
    size_bytes: int


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    sources: tuple[Path, ...]
    slice_seconds: float | None = None
    """None keeps whole files."""
    hop_seconds: float | None = None
    """Distance between slice starts; defaults to ``slice_seconds`` (no overlap)."""
    min_clip_seconds: float | None = None
    """Shortest clip kept: a file or tail shorter than this is skipped. Defaults to the slice length."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    exclude: frozenset[Path] = frozenset()
    """Files left out, by resolved path; each must be inside one of ``sources``."""
    skip_duplicates: bool = False
    """Leave out all but one copy of each probable duplicate recording (see ``find_duplicate_recordings``)."""

    def validate(self) -> None:
        if not self.name or sanitize(self.name) != self.name:
            raise CorpusError("the corpus name may only use letters, digits, '.', '_' and '-'")
        if not self.sources:
            raise CorpusError("pick at least one source folder")
        if self.slice_seconds is not None and not 0.5 <= self.slice_seconds <= 600:
            raise CorpusError("slices must be between 0.5 and 600 seconds")
        if self.hop_seconds is not None and (self.slice_seconds is None or not 0.1 <= self.hop_seconds <= 600):
            raise CorpusError("a hop only applies to slices, and must be between 0.1 and 600 seconds")
        if self.min_clip_seconds is not None and self.min_clip_seconds < 0:
            raise CorpusError("the minimum clip length cannot be negative")

    @property
    def hop(self) -> float | None:
        return self.hop_seconds or self.slice_seconds

    @property
    def min_clip(self) -> float:
        if self.min_clip_seconds is not None:
            return self.min_clip_seconds
        return self.slice_seconds or 0.0


@dataclass(frozen=True)
class PlannedClip:
    name: str
    source: Path
    group: str
    start: float
    duration: float | None


@dataclass(frozen=True)
class CorpusPlan:
    spec: CorpusSpec
    output_dir: Path
    clips: tuple[PlannedClip, ...]
    skipped: tuple[tuple[Path, str], ...]
    per_source: dict[str, dict[str, float]] = field(default_factory=dict)
    excluded: tuple[Path, ...] = ()
    duplicates: tuple[tuple[Path, Path], ...] = ()
    """(file, the copy of the same recording that is kept), for each probable duplicate."""

    @property
    def clips_dir(self) -> Path:
        return self.output_dir / CLIPS_DIRNAME

    @property
    def total_seconds(self) -> float:
        return sum(clip.duration or 0.0 for clip in self.clips)


@dataclass(frozen=True)
class CorpusBuildResult:
    plan: CorpusPlan
    written: int
    failed: tuple[tuple[str, str], ...]
    manifest_path: Path


# ── durations and tags ────────────────────────────────────────────────

TAG_KEYS = ("artist", "album_artist", "album", "date", "genre", "title")


@dataclass(frozen=True)
class AudioProbe:
    duration: float | None
    tags: dict[str, str] = field(default_factory=dict)
    """Lowercased keys from ``TAG_KEYS``, when the file carries them."""

    @property
    def artist(self) -> str | None:
        return self.tags.get("album_artist") or self.tags.get("artist")


def probe_audio(path: Path) -> AudioProbe:
    """Duration and tags from the header (libsndfile), or ffprobe for what it cannot read (m4a, aac)."""
    try:
        import soundfile

        with soundfile.SoundFile(str(path)) as handle:
            duration = handle.frames / handle.samplerate if handle.samplerate else None
            tags = handle.copy_metadata()
        return AudioProbe(duration, _clean_tags(tags))
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "ffprobe",
                *("-v", "error", "-of", "json"),
                *("-show_entries", f"format=duration:format_tags={','.join(TAG_KEYS)}"),
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(result.stdout).get("format") or {}
        return AudioProbe(float(payload["duration"]), _clean_tags(payload.get("tags") or {}))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return AudioProbe(None)


def _clean_tags(raw: dict[str, Any]) -> dict[str, str]:
    tags = {}
    for key, value in raw.items():
        key = str(key).lower()
        text = str(value).strip() if value is not None else ""
        if key in TAG_KEYS and text:
            tags[key] = text
    return tags


def audio_duration(path: Path) -> float | None:
    return probe_audio(path).duration


class AudioProbeCache:
    """Probes (and loudness envelopes) keyed on (path, size, mtime), so a library is read once per change.

    ``load``/``save`` keep them across processes: probing m4a files through ffprobe is what
    makes a cold library scan slow.
    """

    SCHEMA = "macchina_ricordi.audio_probe_cache.v1"

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, float, AudioProbe]] = {}
        self._envelopes: dict[str, tuple[int, float, list[float] | None]] = {}
        self._dirty = False
        self._lock = threading.Lock()

    def probe(self, path: Path) -> AudioProbe | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(path)
        cached = self._entries.get(key)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime):
            return cached[2]
        probe = probe_audio(path)
        self._entries[key] = (stat.st_size, stat.st_mtime, probe)
        self._dirty = True
        return probe

    def get(self, path: Path) -> float | None:
        probe = self.probe(path)
        return probe.duration if probe else None

    def envelope(self, path: Path) -> list[float] | None:
        probe = self.probe(path)
        if probe is None or probe.duration is None:
            return None
        stat = path.stat()
        key = str(path)
        cached = self._envelopes.get(key)
        if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime):
            return cached[2]
        envelope = loudness_envelope(path, probe.duration)
        self._envelopes[key] = (stat.st_size, stat.st_mtime, envelope)
        self._dirty = True
        return envelope

    def load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("schema") != self.SCHEMA:
            return
        for key, entry in (payload.get("entries") or {}).items():
            try:
                size, mtime, duration, tags = entry
                self._entries.setdefault(key, (int(size), float(mtime), AudioProbe(duration, dict(tags))))
            except (TypeError, ValueError):
                continue
        for key, entry in (payload.get("envelopes") or {}).items():
            try:
                size, mtime, envelope = entry
                self._envelopes.setdefault(key, (int(size), float(mtime), envelope))
            except (TypeError, ValueError):
                continue

    def save(self, path: Path) -> None:
        with self._lock:
            if not self._dirty:
                return
            entries = {
                key: [size, mtime, probe.duration, probe.tags]
                for key, (size, mtime, probe) in list(self._entries.items())
                if Path(key).exists()
            }
            envelopes = {key: list(entry) for key, entry in list(self._envelopes.items()) if key in entries}
            write_text_atomic(path, json.dumps({"schema": self.SCHEMA, "entries": entries, "envelopes": envelopes}))
            self._dirty = False


# ── probable duplicates ───────────────────────────────────────────────
#
# The same recording often sits in two folders in different formats (an album ripped as mp3
# and as FLAC), where neither a hard link nor the size can tell. Candidates share a title
# (the tag, else the file name without track numbers) and a length within a couple of
# seconds; they count as one recording when their loudness envelopes, 10 s from 30% in,
# line up. Measured on the library: the same track in two formats scores ~1.0, different
# recordings with the same name and length (generic `set_01.wav` clips) at most ~0.6.

DUPLICATE_LENGTH_TOLERANCE = 2.0
DUPLICATE_MIN_SIMILARITY = 0.9
ENVELOPE_SECONDS = 10.0
ENVELOPE_HOP = 0.05
ENVELOPE_RATE = 8000
# which copy of a duplicate is kept: lossless first, then the larger file
_FORMAT_RANK = {".flac": 0, ".wav": 0, ".aiff": 0, ".aif": 0, ".m4a": 1, ".ogg": 1, ".opus": 1, ".aac": 1, ".mp3": 2}


def _normalized(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"^[\d\s._-]+", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def title_key(path: Path, probe: AudioProbe) -> str:
    return _normalized(probe.tags.get("title") or path.stem)


def loudness_envelope(path: Path, duration: float) -> list[float] | None:
    """Log RMS in 50 ms frames over 10 s from 30% into the file, or None if it cannot be decoded."""
    import numpy as np

    try:
        decoded = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{duration * 0.3:.3f}", "-t", f"{ENVELOPE_SECONDS:g}",
                "-i", str(path), "-ac", "1", "-ar", str(ENVELOPE_RATE), "-f", "s16le", "-",
            ],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    samples = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(ENVELOPE_RATE * ENVELOPE_HOP)
    frames = len(samples) // hop
    if frames < 20:
        return None
    rms = np.sqrt((samples[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1) + 1e-8)
    return [round(float(value), 3) for value in np.log(rms)]


def envelope_similarity(first: Sequence[float], second: Sequence[float], max_lag_seconds: float = 2.0) -> float:
    """Best correlation of two envelopes over shifts up to ``max_lag_seconds`` (leading silence, seek offsets)."""
    import numpy as np

    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    best = -1.0
    max_lag = int(max_lag_seconds / ENVELOPE_HOP)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        length = min(len(x), len(y))
        x, y = x[:length], y[:length]
        if length < 50 or x.std() < 1e-3 or y.std() < 1e-3:
            continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best


def _artists_agree(first: AudioProbe, second: AudioProbe) -> bool:
    a, b = _normalized(first.artist or ""), _normalized(second.artist or "")
    return not a or not b or a in b or b in a


def find_duplicate_recordings(paths: Sequence[Path], cache: AudioProbeCache) -> list[list[Path]]:
    """Groups of files that are probably the same recording, each group in ``paths`` order."""
    candidates: dict[str, list[tuple[Path, AudioProbe]]] = {}
    for path in paths:
        probe = cache.probe(path)
        if probe is None or probe.duration is None:
            continue
        key = title_key(path, probe)
        if len(key) >= 3:
            candidates.setdefault(key, []).append((path, probe))

    parent: dict[Path, Path] = {}

    def root(path: Path) -> Path:
        while parent.get(path, path) != path:
            path = parent[path]
        return path

    for group in candidates.values():
        for i, (first, first_probe) in enumerate(group):
            for second, second_probe in group[i + 1 :]:
                assert first_probe.duration is not None and second_probe.duration is not None
                if root(first) == root(second):
                    continue
                if abs(first_probe.duration - second_probe.duration) > DUPLICATE_LENGTH_TOLERANCE:
                    continue
                if not _artists_agree(first_probe, second_probe):
                    continue
                if _same_bytes(first, second) or _envelopes_match(first, second, cache):
                    parent[root(second)] = root(first)
    order = {path: index for index, path in enumerate(paths)}
    groups: dict[Path, list[Path]] = {}
    for path in parent:
        groups.setdefault(root(path), []).append(path)
    for top in list(groups):
        if top not in groups[top]:
            groups[top].append(top)
    return sorted((sorted(g, key=order.__getitem__) for g in groups.values()), key=lambda g: order[g[0]])


def _same_bytes(first: Path, second: Path) -> bool:
    a, b = first.stat(), second.stat()
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino) or (a.st_size == b.st_size and first.name == second.name)


def _envelopes_match(first: Path, second: Path, cache: AudioProbeCache) -> bool:
    a, b = cache.envelope(first), cache.envelope(second)
    return a is not None and b is not None and envelope_similarity(a, b) >= DUPLICATE_MIN_SIMILARITY


def preferred_copy(paths: Sequence[Path]) -> Path:
    """The copy of a recording worth keeping: lossless over lossy, then the larger file."""
    return min(paths, key=lambda p: (_FORMAT_RANK.get(p.suffix.lower(), 3), -p.stat().st_size))


# ── planning ──────────────────────────────────────────────────────────


def scan_source(folder: Path, durations: AudioProbeCache | None = None) -> list[SourceFile]:
    """Every audio file under ``folder``, sorted, grouped under the folder's sanitized name."""
    cache = durations or AudioProbeCache()
    group = sanitize(folder.name)
    files = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and not path.name.startswith("."):
            files.append(SourceFile(path, group, cache.get(path), path.stat().st_size))
    return files


def slice_starts(duration: float, slice_seconds: float, hop_seconds: float, min_clip_seconds: float) -> list[float]:
    """Start times of the slices cut from a file: full slices, then a tail if it is at least ``min_clip_seconds``."""
    starts: list[float] = []
    start = 0.0
    while start + slice_seconds <= duration + 1e-6:
        starts.append(round(start, 6))
        start += hop_seconds
    tail = duration - start
    if tail >= max(min_clip_seconds, 1e-3) and tail < slice_seconds and (not starts or start > starts[-1]):
        starts.append(round(start, 6))
    return starts


def plan_corpus(
    spec: CorpusSpec,
    output_dir: Path,
    *,
    durations: AudioProbeCache | None = None,
) -> CorpusPlan:
    spec.validate()
    cache = durations or AudioProbeCache()
    missing = [str(source) for source in spec.sources if not source.is_dir()]
    if missing:
        raise CorpusError(f"not a folder: {', '.join(missing)}")
    groups = [sanitize(source.name) for source in spec.sources]
    if len(set(groups)) != len(groups):
        raise CorpusError("two source folders have the same name; clips would not be attributable")
    stray = sorted(str(p) for p in spec.exclude if not any(p.is_relative_to(source) for source in spec.sources))
    if stray:
        raise CorpusError(f"excluded files outside the sources: {', '.join(stray[:3])}")

    skipped: list[tuple[Path, str]] = []
    excluded: list[Path] = []
    per_source: dict[str, dict[str, float]] = {}
    seen_inodes: dict[tuple[int, int], Path] = {}
    seen_content: dict[tuple[int, str], Path] = {}
    kept: list[tuple[Path, SourceFile]] = []
    for source in spec.sources:
        counts = per_source.setdefault(
            sanitize(source.name),
            {"files": 0, "clips": 0, "seconds": 0.0, "source_seconds": 0.0, "skipped": 0, "excluded": 0},
        )
        for item in scan_source(source, cache):
            if item.path in spec.exclude:
                excluded.append(item.path)
                counts["excluded"] += 1
                continue
            stat = item.path.stat()
            twin = seen_inodes.get((stat.st_dev, stat.st_ino)) or seen_content.get(
                (item.size_bytes, item.path.name.lower())
            )
            if twin is not None:
                skipped.append((item.path, f"same file as {twin}"))
                counts["skipped"] += 1
                continue
            seen_inodes[(stat.st_dev, stat.st_ino)] = item.path
            seen_content[(item.size_bytes, item.path.name.lower())] = item.path
            counts["files"] += 1
            if item.duration is None:
                skipped.append((item.path, "unreadable (no duration)"))
                counts["skipped"] += 1
                continue
            kept.append((source, item))

    duplicates: list[tuple[Path, Path]] = []
    for group in find_duplicate_recordings([item.path for _, item in kept], cache):
        keeper = preferred_copy(group)
        duplicates += [(path, keeper) for path in group if path != keeper]
    left_out = {path: keeper for path, keeper in duplicates} if spec.skip_duplicates else {}

    clips: list[PlannedClip] = []
    names: set[str] = set()
    for source, item in kept:
        assert item.duration is not None
        counts = per_source[sanitize(source.name)]
        if item.path in left_out:
            skipped.append((item.path, f"probably the same recording as {left_out[item.path]}"))
            counts["skipped"] += 1
            continue
        counts["source_seconds"] += item.duration
        stem = sanitize(item.path.relative_to(source).with_suffix("").as_posix().replace("/", "_"))
        if spec.slice_seconds is None:
            if item.duration < spec.min_clip:
                skipped.append((item.path, f"shorter than {spec.min_clip:g} s"))
                counts["skipped"] += 1
                continue
            name = _unique(f"{item.group}__{stem}{item.path.suffix.lower()}", names)
            clips.append(PlannedClip(name, item.path, item.group, 0.0, item.duration))
            counts["clips"] += 1
            counts["seconds"] += item.duration
            continue
        assert spec.hop is not None
        starts = slice_starts(item.duration, spec.slice_seconds, spec.hop, spec.min_clip)
        if not starts:
            skipped.append((item.path, f"shorter than {spec.min_clip:g} s"))
            counts["skipped"] += 1
            continue
        width = max(3, len(str(len(starts))))
        for index, start in enumerate(starts):
            length = min(spec.slice_seconds, item.duration - start)
            name = _unique(f"{item.group}__{stem}__{index:0{width}d}.flac", names)
            clips.append(PlannedClip(name, item.path, item.group, start, round(length, 6)))
            counts["clips"] += 1
            counts["seconds"] += length
    return CorpusPlan(
        spec=spec,
        output_dir=output_dir,
        clips=tuple(clips),
        skipped=tuple(skipped),
        per_source=per_source,
        excluded=tuple(excluded),
        duplicates=tuple(duplicates),
    )


def _unique(name: str, taken: set[str]) -> str:
    candidate, stem, suffix = name, *os.path.splitext(name)
    counter = 2
    while candidate.lower() in taken:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    taken.add(candidate.lower())
    return candidate


def plan_digest(plan: CorpusPlan) -> str:
    payload = [(clip.name, str(clip.source), clip.start, clip.duration) for clip in plan.clips]
    return hashlib.sha256(json.dumps([str(plan.output_dir), payload]).encode()).hexdigest()[:20]


# ── building ──────────────────────────────────────────────────────────


def build_corpus(
    plan: CorpusPlan,
    *,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CorpusBuildResult:
    """Write the clips and ``corpus.json``. Refuses an output folder that already holds clips."""
    clips_dir = plan.clips_dir
    if clips_dir.exists() and any(clips_dir.iterdir()):
        raise CorpusError(f"{clips_dir} already holds clips; choose another name or remove it")
    clips_dir.mkdir(parents=True, exist_ok=True)

    by_source: dict[Path, list[PlannedClip]] = {}
    for clip in plan.clips:
        by_source.setdefault(clip.source, []).append(clip)
    failed: list[tuple[str, str]] = []
    written = 0
    done = 0

    def run(source: Path, clips: list[PlannedClip]) -> tuple[int, list[tuple[str, str]]]:
        try:
            if plan.spec.slice_seconds is None:
                _link_or_copy(source, clips_dir / clips[0].name)
                return 1, []
            return _write_slices(source, clips, clips_dir, plan.spec), []
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            return 0, [(str(source), str(exc))]

    with ThreadPoolExecutor(max_workers=workers or max(1, min(8, (os.cpu_count() or 2) // 2))) as pool:
        for count, errors in pool.map(lambda item: run(*item), by_source.items()):
            written += count
            failed.extend(errors)
            done += 1
            if progress is not None:
                progress(done, len(by_source))

    manifest_path = plan.output_dir / CORPUS_MANIFEST
    write_json(manifest_path, corpus_manifest(plan, written=written, failed=failed))
    return CorpusBuildResult(plan=plan, written=written, failed=tuple(failed), manifest_path=manifest_path)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_slices(source: Path, clips: Sequence[PlannedClip], clips_dir: Path, spec: CorpusSpec) -> int:
    import numpy as np
    import soundfile

    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(source),
            "-f", "s16le", "-acodec", "pcm_s16le", "-ac", str(spec.channels), "-ar", str(spec.sample_rate), "-",
        ],
        capture_output=True,
        check=True,
    )  # fmt: skip
    samples = np.frombuffer(decoded.stdout, dtype=np.int16).reshape(-1, spec.channels)
    if not len(samples):
        raise RuntimeError("ffmpeg decoded no audio")
    written = 0
    for clip in clips:
        first = int(round(clip.start * spec.sample_rate))
        last = first + int(round((clip.duration or 0.0) * spec.sample_rate))
        chunk = samples[first : min(last, len(samples))]
        if len(chunk) == 0:
            continue
        soundfile.write(str(clips_dir / clip.name), chunk, spec.sample_rate, format="FLAC", subtype="PCM_16")
        written += 1
    return written


def corpus_manifest(plan: CorpusPlan, *, written: int, failed: Sequence[tuple[str, str]]) -> dict[str, Any]:
    spec = plan.spec
    return {
        "schema": CORPUS_SCHEMA,
        "name": spec.name,
        "created_at": utc_now_iso(),
        "sources": [str(source) for source in spec.sources],
        "slicing": {
            "mode": "whole" if spec.slice_seconds is None else "slices",
            "slice_seconds": spec.slice_seconds,
            "hop_seconds": spec.hop,
            "min_clip_seconds": spec.min_clip,
            "sample_rate": spec.sample_rate if spec.slice_seconds is not None else None,
            "channels": spec.channels if spec.slice_seconds is not None else None,
        },
        "total_clips": len(plan.clips),
        "written_clips": written,
        "total_seconds": round(plan.total_seconds, 3),
        "per_source": {group: {k: round(v, 3) for k, v in counts.items()} for group, counts in plan.per_source.items()},
        "skipped": [{"path": str(path), "reason": reason} for path, reason in plan.skipped],
        "excluded": [str(path) for path in plan.excluded],
        "skip_duplicates": spec.skip_duplicates,
        "duplicates": [{"path": str(path), "kept": str(keeper)} for path, keeper in plan.duplicates],
        "failed": [{"path": path, "error": error} for path, error in failed],
        "clips": [
            {"name": clip.name, "source": str(clip.source), "start": clip.start, "duration": clip.duration}
            for clip in plan.clips
        ],
    }


def read_corpus_manifest(folder: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((folder / CORPUS_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("schema") == CORPUS_SCHEMA else None


__all__ = [
    "CorpusBuildResult",
    "CorpusError",
    "CorpusPlan",
    "CorpusSpec",
    "AudioProbe",
    "AudioProbeCache",
    "PlannedClip",
    "audio_duration",
    "build_corpus",
    "envelope_similarity",
    "find_duplicate_recordings",
    "loudness_envelope",
    "preferred_copy",
    "plan_corpus",
    "plan_digest",
    "probe_audio",
    "read_corpus_manifest",
    "scan_source",
    "slice_starts",
]
