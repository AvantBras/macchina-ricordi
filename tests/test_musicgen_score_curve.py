from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.models.curve import (
    EpochAggregate,
    ScoreRow,
    aggregate_by_epoch,
    format_score_curve_json,
    format_score_curve_table,
    load_score_rows,
    render_sparkline,
    summarize_score_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_load_score_rows_skips_invalid_lines(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "sample_id": "a", "score": 0.4}),
                "not-json",
                json.dumps({"epoch": "bad", "sample_id": "z", "score": 0.0}),
                json.dumps({"epoch": 1, "sample_id": "b", "score": 0.6, "score_error": "boom"}),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_score_rows(scores_path)

    assert len(rows) == 2
    assert rows[0].sample_id == "a"
    assert rows[0].metric_value == pytest.approx(0.4)
    assert rows[1].score_error == "boom"


def test_load_score_rows_missing_file_returns_empty(tmp_path: Path) -> None:
    rows = load_score_rows(tmp_path / "missing.jsonl")
    assert rows == []


def test_aggregate_by_epoch_skips_failed_rows() -> None:
    rows = [
        ScoreRow(epoch=1, sample_id="a", wav_path="", metric_value=0.2, score_error=None),
        ScoreRow(epoch=1, sample_id="b", wav_path="", metric_value=0.4, score_error=None),
        ScoreRow(epoch=1, sample_id="c", wav_path="", metric_value=None, score_error="boom"),
        ScoreRow(epoch=2, sample_id="d", wav_path="", metric_value=0.6, score_error=None),
        ScoreRow(epoch=2, sample_id="e", wav_path="", metric_value=0.9, score_error=None),
    ]

    points = aggregate_by_epoch(rows, aggregate="median")
    assert [point.epoch for point in points] == [1, 2]
    assert points[0].n == 2
    assert points[0].value == pytest.approx(0.3)
    assert points[1].value == pytest.approx(0.75)


def test_aggregate_by_epoch_supports_percentile() -> None:
    rows = [
        ScoreRow(epoch=7, sample_id=f"s{i}", wav_path="", metric_value=v, score_error=None)
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ]
    points = aggregate_by_epoch(rows, aggregate="p90")
    assert points[0].value == pytest.approx(0.91)


def test_aggregate_by_epoch_rejects_unknown_aggregate() -> None:
    with pytest.raises(ValueError):
        aggregate_by_epoch([], aggregate="bogus")


def test_render_sparkline_handles_edge_cases() -> None:
    assert render_sparkline([]) == ""
    assert render_sparkline([0.5]) == "▁"
    assert render_sparkline([0.5, 0.5, 0.5]) == "▁▁▁"
    rising = render_sparkline([0.1, 0.5, 1.0])
    assert rising[0] == "▁"
    assert rising[-1] == "█"
    monotone = render_sparkline([0.1, 0.2, 0.3, 0.4])
    bars = "▁▂▃▄▅▆▇█"
    indices = [bars.index(ch) for ch in monotone]
    assert indices == sorted(indices)


def test_format_score_curve_table_with_sparkline() -> None:
    points = [
        EpochAggregate(epoch=3, n=4, value=0.612, min_value=0.401, max_value=0.788),
        EpochAggregate(epoch=4, n=4, value=0.674, min_value=0.502, max_value=0.812),
    ]
    text = format_score_curve_table(
        sig="abc",
        metric="score",
        aggregate="median",
        points=points,
        n_total=8,
        n_failed=0,
    )
    assert "sig: abc" in text
    assert "epoch" in text
    assert "0.6120" in text
    assert text.splitlines()[-1] in ("▁█", "▁▇", "▁█")


def test_format_score_curve_table_no_points_message() -> None:
    text = format_score_curve_table(sig="abc", metric="score", aggregate="median", points=[], n_total=0, n_failed=0)
    assert "no scored samples yet" in text


def test_format_score_curve_json_schema() -> None:
    points = [EpochAggregate(epoch=1, n=2, value=0.5, min_value=0.4, max_value=0.6)]
    payload = json.loads(
        format_score_curve_json(
            sig="abc",
            metric="score",
            aggregate="mean",
            points=points,
            n_total=2,
            n_failed=0,
        )
    )
    assert payload["sig"] == "abc"
    assert payload["aggregate"] == "mean"
    assert payload["points"] == [{"epoch": 1, "n": 2, "value": 0.5, "min": 0.4, "max": 0.6}]


def test_summarize_score_rows_counts_failures() -> None:
    rows = [
        ScoreRow(epoch=1, sample_id="a", wav_path="", metric_value=0.5, score_error=None),
        ScoreRow(epoch=1, sample_id="b", wav_path="", metric_value=None, score_error="boom"),
    ]
    total, failed = summarize_score_rows(rows)
    assert (total, failed) == (2, 1)


def test_load_score_rows_picks_alternate_metric(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        scores_path,
        [
            {
                "epoch": 1,
                "sample_id": "a",
                "score": 0.5,
                "onset_density": 4.2,
                "score_error": None,
            },
        ],
    )

    rows = load_score_rows(scores_path, metric="onset_density")
    assert len(rows) == 1
    assert rows[0].metric_value == pytest.approx(4.2)
