from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.cli import build_parser, main
from macchina_ricordi.judging.label import LabelRecord, append_label


def test_label_cli_exposes_collection_and_analysis_commands() -> None:
    parser = build_parser()
    collect = parser.parse_args(["judge", "label", "stream", "--alarm", "horizon", "--clips", "30", "--line"])
    report = parser.parse_args(["judge", "labels", "--alarm", "timbral_coherence", "--json"])
    assert collect.alarm == "horizon" and collect.clips == 30 and collect.tui is False
    assert report.alarm == "timbral_coherence" and report.json is True


def test_label_cli_rejects_unknown_alarm_and_too_few_clips() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["judge", "label", "stream", "--alarm", "unknown"])


def test_label_cli_rejects_too_few_clips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "stream"
    source.mkdir()
    (source / "performance.jsonl").write_text("")
    assert main(["judge", "label", str(source), "--clips", "2"]) == 1
    assert "clips must be at least 3" in capsys.readouterr().out


def test_metadata_alarm_rejects_sweep_or_mixed_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stream = tmp_path / "stream"
    sweep = tmp_path / "sweep"
    stream.mkdir()
    sweep.mkdir()
    (stream / "performance.jsonl").write_text("")
    (sweep / "index.csv").write_text("path,sweep_id\n")
    assert main(["judge", "label", str(sweep), "--alarm", "horizon"]) == 1
    assert main(["judge", "label", str(stream), str(sweep), "--alarm", "temperature_floor"]) == 1
    assert "requires stream-session sources" in capsys.readouterr().out


def test_labels_cli_reports_missing_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["judge", "labels", "--store", str(tmp_path / "missing.jsonl")]) == 1
    assert "Run 'judge label' first" in capsys.readouterr().out


def test_labels_cli_writes_json_report_and_validates_minimum(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = tmp_path / "labels.jsonl"
    output = tmp_path / "report.json"
    for index, (value, answer) in enumerate(((0.9, "yes"), (0.99, "no"))):
        append_label(
            LabelRecord(
                path=f"clip_{index}.wav",
                alarm_id="timbral_coherence",
                question="broke",
                answer=answer,
                session_id="session",
                context={
                    "audit_values": {"timbral_coherence": value},
                    "sampling_frame_id": "frame",
                    "inclusion_probability": 0.5,
                },
            ),
            path=store,
        )
    assert (
        main(
            [
                "judge",
                "labels",
                "--store",
                str(store),
                "--alarm",
                "timbral_coherence",
                "--json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert '"alarm_id": "timbral_coherence"' in capsys.readouterr().out
    assert output.exists()
    assert main(["judge", "labels", "--store", str(store), "--min-labels", "0"]) == 1
    assert "must be at least 1" in capsys.readouterr().out
