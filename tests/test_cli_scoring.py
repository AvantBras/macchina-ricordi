from __future__ import annotations

import argparse
import json
from pathlib import Path

import macchina_ricordi.listening.cli as cli_listening
import macchina_ricordi.scoring.cli as cli_scoring
from macchina_ricordi.scoring.reward import RewardComponents


def _make_reward_components(*, score: float, tempo_instability: float = 0.5) -> RewardComponents:
    return RewardComponents(
        score=score,
        tempo_instability=tempo_instability,
        onset_density=1.5,
        tempo_penalty=0.25,
        duration_seconds=2.0,
        onset_count=3,
    )


def test_score_files_records_errors_without_stopping(monkeypatch) -> None:
    def mock_tqdm(iterable: list[Path], **kwargs: object) -> list[Path]:
        return iterable

    def mock_score_function(path: Path, **kwargs: object) -> RewardComponents:
        if Path(path).name == "bad.wav":
            raise RuntimeError("decode failed")
        return _make_reward_components(score=2.0)

    monkeypatch.setattr(cli_scoring, "_get_tqdm", lambda: mock_tqdm)
    monkeypatch.setattr(cli_scoring, "_score_arrhythmic_percussive", mock_score_function)

    rows, failures = cli_scoring.score_files([Path("good.wav"), Path("bad.wav")], label="steady")

    assert failures == 1
    assert rows[0]["label"] == "steady"
    assert rows[0]["error"] == ""
    assert rows[1]["score"] == 0.0
    assert rows[1]["error"] == "decode failed"


def test_cmd_score_file_prints_json_output(monkeypatch, capsys, tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(
        cli_scoring,
        "_score_arrhythmic_percussive",
        lambda *args, **kwargs: _make_reward_components(score=3.5),
    )

    cli_scoring.cmd_score_file(
        argparse.Namespace(
            audio_path=str(audio_path),
            sr=32_000,
            hop_length=512,
            res_type="soxr_hq",
            tempo_method="autocorr",
            json=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(audio_path.resolve())
    assert payload["score"] == 3.5


def test_cmd_score_dir_sorts_rows_and_writes_csv(monkeypatch, capsys, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_path = tmp_path / "scores.csv"
    input_dir.mkdir()
    written: dict[str, object] = {}

    monkeypatch.setattr(
        cli_scoring, "iter_audio_files", lambda *args, **kwargs: [input_dir / "b.wav", input_dir / "a.wav"]
    )
    monkeypatch.setattr(
        cli_scoring,
        "score_files",
        lambda *args, **kwargs: (
            [
                {"path": "b.wav", **_make_reward_components(score=1.0).to_dict(), "error": ""},
                {"path": "a.wav", **_make_reward_components(score=2.0).to_dict(), "error": ""},
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        cli_scoring,
        "write_csv",
        lambda path, fieldnames, rows: written.update(path=path, fieldnames=fieldnames, rows=rows),
    )

    exit_code = cli_scoring.cmd_score_dir(
        argparse.Namespace(
            input_dir=str(input_dir),
            recursive=True,
            extensions="wav",
            quiet=True,
            sr=16_000,
            hop_length=256,
            res_type="soxr_qq",
            tempo_method="autocorr",
            output=str(output_path),
        )
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert written["path"] == output_path.resolve()
    assert [row["path"] for row in written["rows"]] == ["a.wav", "b.wav"]
    assert "Wrote 2 rows" in stdout
    assert "Successes: 1 | Failures: 1" in stdout


def test_cmd_calibrate_writes_summary_json_and_combined_rows(monkeypatch, capsys, tmp_path: Path) -> None:
    steady_dir = tmp_path / "steady"
    arrhythmic_dir = tmp_path / "arrhythmic"
    output_path = tmp_path / "calibration.csv"
    summary_path = tmp_path / "summary" / "summary.json"
    steady_dir.mkdir()
    arrhythmic_dir.mkdir()

    written: dict[str, object] = {}
    score_calls: list[str] = []

    monkeypatch.setattr(cli_listening, "iter_audio_files", lambda root, **kwargs: [Path(root) / "clip.wav"])

    def mock_score_files(
        files: list[Path],
        *,
        label: str | None = None,
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], int]:
        score_calls.append(label or "")
        if label == "arrhythmic":
            return (
                [{"label": label, "path": "arr.wav", **_make_reward_components(score=2.0).to_dict(), "error": ""}],
                0,
            )
        return (
            [{"label": label, "path": "steady.wav", **_make_reward_components(score=1.0).to_dict(), "error": ""}],
            1,
        )

    monkeypatch.setattr(cli_scoring, "score_files", mock_score_files)
    monkeypatch.setattr(
        cli_scoring,
        "_build_calibration_summary",
        lambda **kwargs: {
            "arrhythmic": {"mean": 2.0},
            "steady": {"mean": 1.0},
            "separation": {"ranking_accuracy": 1.0, "cohens_d": 2.5, "best_threshold": {"threshold": 1.5}},
        },
    )
    monkeypatch.setattr(
        cli_scoring,
        "write_csv",
        lambda path, fieldnames, rows: written.update(path=path, fieldnames=fieldnames, rows=rows),
    )

    exit_code = cli_scoring.cmd_calibrate(
        argparse.Namespace(
            steady_dir=str(steady_dir),
            arrhythmic_dir=str(arrhythmic_dir),
            recursive=False,
            extensions="wav",
            quiet=True,
            sr=16_000,
            hop_length=512,
            res_type="soxr_qq",
            tempo_method="autocorr",
            output=str(output_path),
            summary_json=str(summary_path),
        )
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert score_calls == ["steady", "arrhythmic"]
    assert written["path"] == output_path.resolve()
    assert [row["path"] for row in written["rows"]] == ["arr.wav", "steady.wav"]
    assert payload["arrhythmic"]["files_total"] == 1
    assert payload["steady"]["files_failed"] == 1
    assert "Calibration summary" in stdout
    assert f"wrote summary json: {summary_path.resolve()}" in stdout
