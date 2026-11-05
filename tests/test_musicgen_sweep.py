from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from macchina_ricordi.instrument.sweep import (
    SWEEP_INDEX_FIELDNAMES,
    build_sweep_combinations,
    listen_sweep_rows,
    parse_bool_grid,
    parse_float_grid,
    run_musicgen_sweep,
    score_sweep_rows,
)
from macchina_ricordi.musicgen.runtime import MusicGenProjectConfig


@dataclass(frozen=True)
class _FakeRewardComponents:
    score: float = 0.0
    tempo_instability: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def test_parse_grid_values() -> None:
    assert parse_float_grid("0.8, 1.2", default=1.0, name="temperature_values") == (0.8, 1.2)
    assert parse_float_grid(None, default=1.0, name="temperature_values") == (1.0,)
    assert parse_bool_grid("true,false,1,0", default=True, name="use_sampling_values") == (
        True,
        False,
        True,
        False,
    )


def test_build_sweep_combinations_expands_in_deterministic_order() -> None:
    combos = build_sweep_combinations(
        mode="continue",
        duration_seconds=20.0,
        prompt_duration_seconds=5.0,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
        cfg_coef=3.0,
        extend_stride=12.0,
        seed=None,
        use_sampling=True,
        temperature_values="1.0,1.5",
        top_k_values="80,250",
        seed_values="10,11",
        max_combinations=8,
    )

    assert [combo["combo_id"] for combo in combos] == [f"combo_{index:03d}" for index in range(8)]
    assert combos[0]["temperature"] == 1.0
    assert combos[0]["top_k"] == 80
    assert combos[0]["seed"] == 10
    assert combos[1]["temperature"] == 1.0
    assert combos[1]["top_k"] == 80
    assert combos[1]["seed"] == 11
    assert combos[-1]["temperature"] == 1.5
    assert combos[-1]["top_k"] == 250
    assert combos[-1]["seed"] == 11


def test_build_sweep_combinations_rejects_invalid_prompt_duration() -> None:
    with pytest.raises(ValueError, match="shorter"):
        build_sweep_combinations(
            mode="continue",
            duration_seconds=10.0,
            prompt_duration_seconds=5.0,
            temperature=1.0,
            top_k=250,
            top_p=0.0,
            cfg_coef=3.0,
            extend_stride=12.0,
            seed=None,
            use_sampling=True,
            prompt_duration_seconds_values="10",
        )


def test_build_sweep_combinations_enforces_max_combinations() -> None:
    with pytest.raises(ValueError, match="above --max-combinations=2"):
        build_sweep_combinations(
            mode="unconditional",
            duration_seconds=10.0,
            prompt_duration_seconds=None,
            temperature=1.0,
            top_k=250,
            top_p=0.0,
            cfg_coef=3.0,
            extend_stride=12.0,
            seed=None,
            use_sampling=True,
            temperature_values="1.0,1.5",
            top_k_values="80,250",
            max_combinations=2,
        )


def test_run_musicgen_sweep_dry_run_writes_plan(tmp_path: Path) -> None:
    prompt_audio = tmp_path / "seed.wav"
    prompt_audio.write_bytes(b"not really audio")

    run = run_musicgen_sweep(
        mode="continue",
        model_id=None,
        profile_name="micro-unconditional",
        device=None,
        xp_path=None,
        checkpoint="latest",
        checkpoint_path=None,
        prompt_audio_path=prompt_audio,
        prompt_start_seconds=0.0,
        prompt_duration_seconds=2.0,
        num_samples=2,
        duration_seconds=10.0,
        seed=None,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
        cfg_coef=3.0,
        use_sampling=True,
        extend_stride=12.0,
        output_dir=tmp_path / "sweeps",
        output_prefix="demo",
        temperature_values="1.0,1.5",
        max_combinations=4,
        dry_run=True,
        listen=True,
        now=datetime(2026, 4, 26, tzinfo=UTC),
    )

    plan = json.loads(run.plan_path.read_text(encoding="utf-8"))
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert run.sweep_id == "demo_20260426T000000Z"
    assert len(plan["combinations"]) == 2
    assert plan["prompt_audio_path"] == str(prompt_audio.resolve())
    assert report["status"] == "dry_run"
    assert report["listen_enabled"] is True
    assert report["listening_reports"] == 0
    assert report["listening_failed"] == 0
    assert run.index_csv_path.exists()
    assert run.index_html_path is not None
    assert run.index_html_path.exists()


def test_run_musicgen_sweep_calls_helper_once(monkeypatch, tmp_path: Path) -> None:
    musicgen_python = tmp_path / "python"
    musicgen_python.write_text("", encoding="utf-8")
    helper_script = tmp_path / "scripts" / "run_musicgen_sweep.py"
    helper_script.parent.mkdir()
    helper_script.write_text("", encoding="utf-8")
    config = MusicGenProjectConfig(
        repo_root=tmp_path,
        project_dir=tmp_path / "musicgen",
        env_dir=tmp_path / ".venv-musicgen",
        dora_dir=tmp_path / ".dora",
        musicgen_python=musicgen_python,
        audiocraft_dir=tmp_path / ".audiocraft",
        python_request="3.11",
        audiocraft_repo_url="https://github.com/aaronabebe/audiocraft",
        audiocraft_commit_sha="commit",
        default_recipe="micro-unconditional",
        default_valid_fraction=0.1,
        default_split_seed=42,
        recipes={},
        codecs={},
    )
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    checkpoint = xp_path / "checkpoint_5.th"
    checkpoint.write_text("", encoding="utf-8")
    prompt_audio = tmp_path / "seed.wav"
    prompt_audio.write_bytes(b"not really audio")
    commands: list[list[str]] = []

    def fake_run(command, check, capture_output, text, env):  # type: ignore[no-untyped-def]
        commands.append(list(command))
        plan_path = Path(command[command.index("--plan-json") + 1])
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        index_path = plan_path.parent / "index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEEP_INDEX_FIELDNAMES)
            writer.writeheader()
            writer.writerow(
                {
                    **dict.fromkeys(SWEEP_INDEX_FIELDNAMES, ""),
                    "sweep_id": plan["sweep_id"],
                    "combo_id": "combo_000",
                    "combo_index": "0",
                    "sample_index": "0",
                    "path": str(plan_path.parent / "combo_000" / "sample_000.wav"),
                    "mode": "continuation",
                    "temperature": "1.0",
                }
            )
        (plan_path.parent / "sweep_report.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sweep_id": plan["sweep_id"],
                    "status": "completed",
                    "combo_count": len(plan["combinations"]),
                    "combo_failures": 0,
                    "samples_written": 1,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("macchina_ricordi.instrument.sweep.load_musicgen_project_config", lambda: config)
    monkeypatch.setattr("macchina_ricordi.musicgen.runtime.load_musicgen_project_config", lambda: config)
    monkeypatch.setattr("macchina_ricordi.instrument.sweep.musicgen_script_env", lambda: {})
    monkeypatch.setattr("macchina_ricordi.instrument.sweep.subprocess.run", fake_run)
    monkeypatch.setattr(
        "macchina_ricordi.instrument.sweep.listen_sweep_rows",
        lambda rows: [
            {
                **row,
                "listening_report_path": str(tmp_path / "sample_listening_report.json"),
                "listening_markdown_path": str(tmp_path / "sample_listening.md"),
                "listening_summary": "A bright sparse fragment.",
                "listening_score": "1.25",
                "listening_labels": "bright, sparse",
                "listening_error": "",
            }
            for row in rows
        ],
    )

    run = run_musicgen_sweep(
        mode="continue",
        model_id="dummy/micro",
        profile_name="micro-unconditional",
        device="cpu",
        xp_path=xp_path,
        checkpoint="latest",
        checkpoint_path=None,
        prompt_audio_path=prompt_audio,
        prompt_start_seconds=0.0,
        prompt_duration_seconds=2.0,
        num_samples=1,
        duration_seconds=10.0,
        seed=None,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
        cfg_coef=3.0,
        use_sampling=True,
        extend_stride=12.0,
        output_dir=tmp_path / "sweeps",
        output_prefix="demo",
        max_combinations=4,
        listen=True,
        now=datetime(2026, 4, 26, tzinfo=UTC),
    )

    plan = json.loads(run.plan_path.read_text(encoding="utf-8"))
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    with run.index_csv_path.open("r", newline="", encoding="utf-8") as handle:
        index_rows = list(csv.DictReader(handle))
    assert len(commands) == 1
    assert commands[0][0] == str(musicgen_python)
    assert commands[0][1] == str(helper_script)
    assert plan["source"]["checkpoint_path"] == str(checkpoint.resolve())
    assert report["listen_enabled"] is True
    assert report["listening_reports"] == 1
    assert report["listening_failed"] == 0
    assert index_rows[0]["listening_summary"] == "A bright sparse fragment."
    assert run.sample_count == 1


def test_score_sweep_rows_adds_optional_scores(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    rows = score_sweep_rows(
        [{"path": str(audio_path), "error": ""}],
        scorer=lambda path: _FakeRewardComponents(score=2.5, tempo_instability=1.2),
        reward_components_cls=_FakeRewardComponents,
    )

    assert rows[0]["score"] == "2.5"
    assert rows[0]["tempo_instability"] == "1.2"
    assert rows[0]["score_error"] == ""


def test_listen_sweep_rows_adds_sidecar_fields(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    report_path = tmp_path / "sample_listening_report.json"
    markdown_path = tmp_path / "sample_listening.md"
    audio_path.write_bytes(b"fake")

    rows = listen_sweep_rows(
        [{"path": str(audio_path), "error": ""}],
        listener=lambda path: SimpleNamespace(
            report_path=report_path,
            markdown_path=markdown_path,
            summary="A bright sparse fragment.",
            score=1.25,
            labels="bright, sparse",
            error="",
        ),
    )

    assert rows[0]["listening_report_path"] == str(report_path)
    assert rows[0]["listening_markdown_path"] == str(markdown_path)
    assert rows[0]["listening_summary"] == "A bright sparse fragment."
    assert rows[0]["listening_score"] == "1.25"
    assert rows[0]["listening_labels"] == "bright, sparse"
    assert rows[0]["listening_error"] == ""


def test_listen_sweep_rows_skips_empty_and_failed_rows(tmp_path: Path) -> None:
    calls: list[Path] = []

    def listener(path: Path) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(report_path="", markdown_path="", summary="", score=None, labels="", error="")

    rows = listen_sweep_rows(
        [
            {"path": "", "error": ""},
            {"path": str(tmp_path / "failed.wav"), "error": "backend failed"},
        ],
        listener=listener,
    )

    assert calls == []
    assert rows[0]["listening_report_path"] == ""
    assert rows[1]["listening_error"] == ""


def test_listen_sweep_rows_records_listener_exceptions(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    def listener(path: Path) -> SimpleNamespace:
        raise RuntimeError("analysis exploded")

    rows = listen_sweep_rows([{"path": str(audio_path), "error": ""}], listener=listener)

    assert rows[0]["listening_report_path"] == ""
    assert rows[0]["listening_error"] == "analysis exploded"
