from __future__ import annotations

import json
from pathlib import Path

from macchina_ricordi.models.specs import (
    MIN_PROMPT_HEADROOM_SECONDS,
    clamp_prompt_seconds,
    resolve_segment_duration,
)


def _xp(tmp_path: Path, sig: str, segment_duration: float | None) -> Path:
    folder = tmp_path / "xps" / sig
    folder.mkdir(parents=True)
    argv = ["solver=musicgen/musicgen_base_32khz", "optim.epochs=5"]
    if segment_duration is not None:
        argv.append(f"dataset.segment_duration={segment_duration:g}")
    (folder / ".argv.json").write_text(json.dumps(argv), encoding="utf-8")
    return folder


def test_resolve_segment_duration_prefers_the_xp_argv(tmp_path: Path) -> None:
    # A folder named like a known sig but trained at a different length: the
    # folder wins, because it is ground truth and the table is only a fallback.
    folder = _xp(tmp_path, "73e1a314", 7.5)
    assert resolve_segment_duration(folder) == 7.5


def test_clamp_prompt_seconds_keeps_headroom() -> None:
    assert clamp_prompt_seconds(11.0, segment_duration=10.0) == 10.0 - MIN_PROMPT_HEADROOM_SECONDS
    assert clamp_prompt_seconds(5.0, segment_duration=10.0) == 5.0
    # Unknown horizon and degenerate horizons leave the request alone.
    assert clamp_prompt_seconds(11.0, segment_duration=None) == 11.0
    assert clamp_prompt_seconds(11.0, segment_duration=0.5) == 11.0
