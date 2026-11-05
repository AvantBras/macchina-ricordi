from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.cli import build_parser, main


def test_score_ab_parser_exposes_newer_and_context_window() -> None:
    parser = build_parser()
    args = parser.parse_args(["judge", "ab", "stream", "--question", "newer", "--context-window", "2"])
    help_text = parser.format_help()

    assert args.question == "newer"
    assert args.context_window == 2
    assert "score" in help_text


def test_score_ab_parser_rejects_invalid_context_window() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["judge", "ab", "stream", "--question", "newer", "--context-window", "4"])


def test_newer_rejects_sweep_and_mixed_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stream = tmp_path / "stream"
    sweep = tmp_path / "sweep"
    stream.mkdir()
    sweep.mkdir()
    (stream / "performance.jsonl").write_text("")
    (sweep / "index.csv").write_text("path,sweep_id\n")

    assert main(["judge", "ab", str(sweep), "--question", "newer"]) == 1
    assert main(["judge", "ab", str(stream), str(sweep), "--question", "newer"]) == 1
    assert "requires stream-session sources" in capsys.readouterr().out
