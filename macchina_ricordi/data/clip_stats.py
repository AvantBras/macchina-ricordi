"""Clip-duration statistics and segment-duration validation.

These helpers exist to prevent (and surface) a silent training-data hazard: when a
workspace's audio clip length ``S`` is shorter than the profile's
``dataset.segment_duration`` ``D``, the AudioCraft loader collapses its random seek
window to 0 (disabling crop augmentation) and zero-pads every example up to ``D``.
The model then learns "music, then silence" as structure.

The single source of truth for ``S`` is the prepared manifest
(``egs/<dataset>/<split>/data.jsonl.gz``), which already records a per-file
``duration``. Everything here reads from that and is intentionally pure and
dependency-light so it can be unit-tested and reused at prepare time, at train time,
and by the ``data audit`` command.
"""

from __future__ import annotations

import gzip
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "read_manifest_durations",
    "compute_clip_stats",
    "clip_stats_for_workspace",
    "pad_share",
    "extract_override_float",
    "validate_segment_duration",
    "SegmentDurationMismatch",
    "EMPTY_CLIP_MAX_SECONDS",
    "SHORT_FRAGMENT_PAD_WARN_SHARE",
]

# Clips at or below this duration are treated as empty/degenerate fragments.
EMPTY_CLIP_MAX_SECONDS = 0.05
# Above this share of padded clips (with median >= D) we warn about the tail.
SHORT_FRAGMENT_PAD_WARN_SHARE = 0.05


class SegmentDurationMismatch(RuntimeError):
    """Raised when ``segment_duration`` exceeds the dataset's typical clip length.

    Carries enough context for callers (and tests) to inspect the offending values
    and to surface the recommended fix.
    """

    def __init__(
        self,
        message: str,
        *,
        segment_duration: float,
        stats: dict,
        recommended_segment_duration: int,
    ) -> None:
        super().__init__(message)
        self.segment_duration = segment_duration
        self.stats = stats
        self.recommended_segment_duration = recommended_segment_duration


def read_manifest_durations(manifest_path: str | Path) -> list[float] | None:
    """Return per-clip durations from a gzip JSONL manifest, or ``None`` on failure.

    Never raises: a missing, unreadable, non-gzip, or duration-less manifest yields
    ``None`` so callers can fail-open (skip validation) rather than crash.
    """
    path = Path(manifest_path)
    if not path.is_file():
        return None
    durations: list[float] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = payload.get("duration") if isinstance(payload, dict) else None
                if isinstance(value, (int, float)):
                    durations.append(float(value))
    except (OSError, EOFError, gzip.BadGzipFile):
        return None
    return durations or None


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    index = int(round(fraction * (len(sorted_values) - 1)))
    index = max(0, min(len(sorted_values) - 1, index))
    return float(sorted_values[index])


def compute_clip_stats(durations: Sequence[float] | None) -> dict | None:
    """Summarise clip durations into a stats dict, or ``None`` if there are none.

    ``slice_length_s`` uses the 90th percentile, which equals the slice length for
    cleanly pre-sliced datasets while staying robust to a short-fragment tail.
    """
    if not durations:
        return None
    values = [float(d) for d in durations]
    ordered = sorted(values)
    count = len(ordered)
    p90 = _percentile(ordered, 0.90)
    half = 0.5 * p90
    return {
        "count": count,
        "min": round(ordered[0], 3),
        "p05": round(_percentile(ordered, 0.05), 3),
        "median": round(statistics.median(ordered), 3),
        "p90": round(p90, 3),
        "max": round(ordered[-1], 3),
        "slice_length_s": round(p90, 3),
        "n_shorter_than_half": sum(1 for d in ordered if d < half),
        "n_empty": sum(1 for d in ordered if d <= EMPTY_CLIP_MAX_SECONDS),
    }


def clip_stats_for_workspace(manifest_path: str | Path) -> dict | None:
    """Compute clip stats directly from a manifest path (convenience wrapper)."""
    return compute_clip_stats(read_manifest_durations(manifest_path))


def pad_share(durations: Sequence[float] | None, segment_duration: float) -> float:
    """Fraction of clips shorter than ``segment_duration`` (i.e. that get padded).

    Operates on raw durations because the summary stats deliberately do not retain
    every value.
    """
    if not durations:
        return 0.0
    return sum(1 for d in durations if d < segment_duration) / len(durations)


def extract_override_float(overrides: Sequence[str], key: str) -> float | None:
    """Return the value of the last ``key=...`` override token as a float.

    The *last* match wins so that a user ``--override`` overrides the profile's
    declared value (which is how AudioCraft/Dora resolve duplicate keys).
    """
    found: float | None = None
    prefix = f"{key}="
    for override in overrides:
        if override.startswith(prefix):
            raw = override[len(prefix) :]
            try:
                found = float(raw)
            except ValueError:
                continue
    return found


def validate_segment_duration(
    stats: dict | None,
    segment_duration: float | None,
    *,
    durations: Sequence[float] | None = None,
    min_segment_ratio: float = 0.8,
    allow_padding: bool = False,
) -> list[str]:
    """Validate ``segment_duration`` against the dataset's clip-length profile.

    Returns a list of human-readable warnings (possibly empty). Raises
    ``SegmentDurationMismatch`` when the typical clip falls into the loader's
    degenerate regime, unless ``allow_padding`` is set (then it becomes a warning).

    The degenerate regime mirrors the AudioCraft loader exactly: its seek window is
    ``max_seek = max(0, clip_duration - segment_duration * min_segment_ratio)``, so a
    clip collapses to ``max_seek == 0`` (read from offset 0, then silence-padded, with
    no random-crop augmentation) precisely when
    ``clip_duration <= segment_duration * min_segment_ratio``. The error therefore
    triggers when the *median* clip meets that condition, which is why
    ``min_segment_ratio`` scales the threshold rather than being ignored. Clips that
    are merely shorter than ``segment_duration`` (padded but still seekable) produce a
    softer warning.

    Fails open: if stats or ``segment_duration`` are unavailable, returns ``[]``.
    """
    if stats is None or segment_duration is None:
        return []

    median = float(stats.get("median") or 0.0)
    slice_length = float(stats.get("slice_length_s") or 0.0)
    recommended = max(1, int(slice_length))

    # Threshold below which the loader's seek window collapses to 0 for the median
    # clip. Scales with min_segment_ratio (the loader's own tolerance knob).
    collapse_threshold = segment_duration * min_segment_ratio

    if median <= collapse_threshold:
        share = pad_share(durations, segment_duration) if durations is not None else 1.0
        message = (
            f"segment_duration={segment_duration:g}s puts the median clip "
            f"({median:g}s, slice≈{slice_length:g}s) at or below the loader's seek "
            f"threshold segment_duration*min_segment_ratio="
            f"{collapse_threshold:g}s (min_segment_ratio={min_segment_ratio:g}): "
            f"~{share:.0%} of training examples are silence-padded and random-crop "
            f"augmentation is disabled (max_seek collapses to 0). Re-slice the source "
            f"audio to ≥{segment_duration:g}s, or set "
            f"dataset.segment_duration={recommended}. Pass --allow-segment-padding to "
            f"train anyway."
        )
        if allow_padding:
            return [f"segment padding allowed: {message}"]
        raise SegmentDurationMismatch(
            message,
            segment_duration=float(segment_duration),
            stats=stats,
            recommended_segment_duration=recommended,
        )

    if durations is not None:
        share = pad_share(durations, segment_duration)
        if share > SHORT_FRAGMENT_PAD_WARN_SHARE:
            return [
                f"{share:.0%} of clips are shorter than "
                f"segment_duration={segment_duration:g}s and will be silence-padded "
                f"(median clip is {median:g}s, above the "
                f"segment_duration*min_segment_ratio={collapse_threshold:g}s seek "
                f"threshold, so the bulk still gets crop augmentation; consider "
                f"filtering the short-clip tail)."
            ]
    return []
