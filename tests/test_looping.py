from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

import macchina_ricordi.looping.cli as cli_looping
import macchina_ricordi.looping.stems as looping_stems
from macchina_ricordi.cli import build_parser
from macchina_ricordi.looping import (
    parse_bar_counts,
)

SR = 22_050


def _click_track(bpm: float, seconds: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    out = np.zeros(int(seconds * sr), dtype=np.float32)
    burst = np.random.default_rng(0).standard_normal(int(0.01 * sr)) * np.exp(-np.arange(int(0.01 * sr)) / 40.0)
    for beat_time in np.arange(0.0, seconds - 0.02, 60.0 / bpm):
        start = int(round(beat_time * sr))
        out[start : start + burst.shape[0]] += amplitude * burst.astype(np.float32)
    return out


def _voice(spans: list[tuple[float, float]], seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    out = np.zeros_like(t, dtype=np.float32)
    for start, end in spans:
        mask = (t >= start) & (t < end)
        out[mask] = 0.3 * np.sin(2.0 * np.pi * 440.0 * t[mask])
    return out


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "many", " , "])
def test_parse_bar_counts_rejects_invalid(value: str) -> None:
    with pytest.raises(ValueError):
        parse_bar_counts(value)


def _write_stems(stems_dir: Path, seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    band = _click_track(120.0, seconds)
    vocals = _voice([(2.0, 3.6), (6.0, 7.6), (10.0, 11.6), (14.0, 15.6)], seconds)
    stems_dir.mkdir(parents=True, exist_ok=True)
    soundfile.write(stems_dir / looping_stems.VOCALS_FILENAME, vocals, SR)
    soundfile.write(stems_dir / looping_stems.INSTRUMENTAL_FILENAME, band, SR)
    return vocals, band


def test_loop_find_and_render_end_to_end(tmp_path: Path) -> None:
    vocals, band = _write_stems(tmp_path / "stems")
    song = tmp_path / "song.wav"
    soundfile.write(song, vocals + band, SR)
    out = tmp_path / "loops"
    common = ["--output-sr", str(SR)]

    args = build_parser().parse_args(
        ["loop", "find", str(song), str(out), "--stems-dir", str(tmp_path / "stems"), "--top", "3", "--seconds", "8"]
        + ["--no-overlap", "--min-voice", "0.3", "--max-gap-ms", "2500"]
        + common
    )
    assert args.func is cli_looping.cmd_loop_find
    assert args.func(args) == 0

    report_path = out / cli_looping.REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tempo_bpm"] == pytest.approx(120.0, abs=0.1)
    assert report["bar_seconds"] == pytest.approx(2.0, abs=0.005)
    assert report["vocal_onset_count"] == 4
    assert [c["rank"] for c in report["candidates"]] == [1, 2, 3]
    spans = sorted((c["start_seconds"], c["end_seconds"]) for c in report["candidates"])
    assert all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False))
    assert all(c["voice_fraction"] >= 0.3 and c["longest_gap_seconds"] <= 2.5 for c in report["candidates"])
    assert report["settings"]["overlap"] is False
    first = report["candidates"][0]
    assert Path(first["audition_path"]).is_file()
    assert soundfile.info(first["audition_path"]).duration == pytest.approx(
        first["repeats"] * first["length_seconds"], abs=0.01
    )

    rerun = build_parser().parse_args(
        ["loop", "find", str(song), str(out), "--stems-dir", str(tmp_path / "stems"), "--top", "1"] + common
    )
    rerun.func(rerun)
    assert len(list((out / cli_looping.CANDIDATES_DIRNAME).iterdir())) == 1

    rendered = tmp_path / "voice.flac"
    render = build_parser().parse_args(
        ["loop", "render", str(report_path), str(rendered), "--start", "1.97", "--bars", "1", "--repeats", "3"]
        + ["--layer", "vocals"]
        + common
    )
    assert render.func(render) == 0
    assert soundfile.info(rendered).duration == pytest.approx(3 * report["bar_seconds"], abs=0.01)
    sidecar = json.loads((tmp_path / "voice_report.json").read_text(encoding="utf-8"))
    assert (sidecar["layer"], sidecar["bars"], sidecar["repeats"], sidecar["pick"]) == ("vocals", 1, 3, None)

    stuck = tmp_path / "stuck.wav"
    render = build_parser().parse_args(["loop", "render", str(report_path), str(stuck), "--layer", "stuck"] + common)
    assert render.func(render) == 0
    assert soundfile.info(stuck).duration == pytest.approx(20.0, abs=0.01)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--start", "2.0"], "needs --bars"),
        (["--pick", "9"], "between 1 and"),
        (["--pick", "1", "--bars", "0"], "must be > 0"),
        (["--layer", "stuck", "--repeats", "2"], "end of the song"),
    ],
)
def test_loop_render_rejects_bad_requests(tmp_path: Path, extra: list[str], message: str) -> None:
    report_path = tmp_path / "loop_report.json"
    report_path.write_text(
        json.dumps(
            {"bar_seconds": 2.0, "source_path": "x", "stems": {}, "candidates": [{"start_seconds": 1.0, "bars": 1}]}
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(["loop", "render", str(report_path), str(tmp_path / "o.flac")] + extra)

    with pytest.raises(ValueError, match=message):
        args.func(args)
