from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile

from macchina_ricordi.data.corpus import (
    CorpusError,
    CorpusSpec,
    build_corpus,
    plan_corpus,
    sanitize,
    slice_starts,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _wav(path: Path, seconds: float, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    soundfile.write(str(path), (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate)
    return path


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    _wav(root / "Album One" / "01 intro.wav", 25.0)
    _wav(root / "Album One" / "disc 2" / "02 long.wav", 12.0)
    _wav(root / "Album One" / "03 tiny.wav", 3.0)
    _wav(root / "second" / "a.wav", 10.0)
    return root


@pytest.mark.parametrize(
    ("duration", "slice_seconds", "hop", "minimum", "expected"),
    [
        (25.0, 10.0, 10.0, 10.0, [0.0, 10.0]),
        (25.0, 10.0, 10.0, 5.0, [0.0, 10.0, 20.0]),
        (25.0, 10.0, 5.0, 10.0, [0.0, 5.0, 10.0, 15.0]),
        (10.0, 10.0, 10.0, 10.0, [0.0]),
        (4.0, 10.0, 10.0, 10.0, []),
        (4.0, 10.0, 10.0, 2.0, [0.0]),
    ],
)
def test_slice_starts(duration, slice_seconds, hop, minimum, expected) -> None:
    assert slice_starts(duration, slice_seconds, hop, minimum) == expected


@needs_ffmpeg
def test_slices_are_32k_mono_flac_of_the_right_length(library: Path, tmp_path: Path) -> None:
    spec = CorpusSpec(name="demo", sources=(library / "Album One",), slice_seconds=10.0, min_clip_seconds=4.0)
    result = build_corpus(plan_corpus(spec, tmp_path / "out"))
    assert not result.failed
    lengths = {}
    for path in sorted((tmp_path / "out" / "clips").iterdir()):
        info = soundfile.info(str(path))
        assert (info.samplerate, info.channels, info.format) == (32_000, 1, "FLAC")
        lengths[path.name] = round(info.duration, 2)
    assert lengths == {
        "Album_One__01_intro__000.flac": 10.0,
        "Album_One__01_intro__001.flac": 10.0,
        "Album_One__01_intro__002.flac": 5.0,
        "Album_One__disc_2_02_long__000.flac": 10.0,
    }


def test_excluded_files_are_left_out(library: Path, tmp_path: Path) -> None:
    left_out = (library / "Album One" / "01 intro.wav").resolve()
    spec = CorpusSpec(name="demo", sources=(library / "Album One",), exclude=frozenset({left_out}))
    plan = plan_corpus(spec, tmp_path / "out")
    assert left_out not in {clip.source for clip in plan.clips}
    assert plan.excluded == (left_out,) and plan.per_source["Album_One"]["excluded"] == 1
    stray = CorpusSpec(name="demo", sources=(library / "second",), exclude=frozenset({left_out}))
    with pytest.raises(CorpusError, match="outside the sources"):
        plan_corpus(stray, tmp_path / "out")


def test_folder_names_cannot_hold_the_clip_separator() -> None:
    assert sanitize("my__album") == "my_album"
    assert sanitize("a___b") == "a_b"
