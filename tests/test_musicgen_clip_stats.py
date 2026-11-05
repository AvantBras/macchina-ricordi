from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from macchina_ricordi.data.clip_stats import (
    SegmentDurationMismatch,
    compute_clip_stats,
    read_manifest_durations,
    validate_segment_duration,
)


def _write_manifest(path: Path, durations: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index, duration in enumerate(durations):
            handle.write(json.dumps({"path": f"clip_{index}.wav", "duration": duration}) + "\n")
    return path


def test_read_manifest_durations_roundtrip(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "data.jsonl.gz", [10.0, 9.9, 10.1])
    assert read_manifest_durations(manifest) == [10.0, 9.9, 10.1]


# --- validate_segment_duration -------------------------------------------------


def test_validate_raises_when_d_exceeds_slice() -> None:
    durations = [10.0] * 50
    stats = compute_clip_stats(durations)
    with pytest.raises(SegmentDurationMismatch) as excinfo:
        validate_segment_duration(stats, 15.0, durations=durations)
    assert excinfo.value.recommended_segment_duration == 10
    assert "segment_duration=10" in str(excinfo.value)


@pytest.mark.parametrize(
    "durations",
    [
        [20.0] * 50,  # slices longer than D
        [300.0, 280.0, 420.0, 175.0],  # raw tracks, which are cut later
    ],
)
def test_validate_passes_when_nothing_will_be_padded(durations: list[float]) -> None:
    stats = compute_clip_stats(durations)
    assert validate_segment_duration(stats, 15.0, durations=durations) == []


def test_validate_allow_padding_downgrades_to_warning() -> None:
    durations = [10.0] * 50
    stats = compute_clip_stats(durations)
    warnings = validate_segment_duration(stats, 15.0, durations=durations, allow_padding=True)
    assert len(warnings) == 1
    assert "segment padding allowed" in warnings[0]
