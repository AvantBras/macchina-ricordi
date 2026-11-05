from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from macchina_ricordi.musicgen.config import musicgen_script_env
from macchina_ricordi.musicgen.runtime import load_musicgen_project_config
from macchina_ricordi.training.sidecar import (
    MusicGenCheckpointEpoch,
    build_musicgen_sidecar_runner_command,
    discover_generated_samples,
    get_last_processed_epoch,
    list_epoch_checkpoints,
    load_existing_scored_sample_ids,
    load_musicgen_cpu_sidecar_state,
    mark_stage_epoch_processed,
    plan_musicgen_cpu_sidecar,
    resolve_sidecar_xp_path,
    resolve_terminal_epoch_from_hydra_overrides,
    run_musicgen_cpu_sidecar,
    save_musicgen_cpu_sidecar_state,
    score_samples_for_epoch,
    select_latest_checkpoint_for_sidecar,
)


def _write_musicgen_project(repo_root: Path) -> None:
    musicgen_dir = repo_root / "musicgen"
    musicgen_dir.mkdir(parents=True)
    (musicgen_dir / "pyproject.toml").write_text(
        """
[project]
name = "macchina_ricordi_musicgen_runtime"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = []

[tool.uv]
package = false

[tool.macchina_ricordi.runtime]
project_environment = ".venv-musicgen"
checkout_dir = ".audiocraft"
dora_dir = ".dora"
python_request = "3.11"

[tool.macchina_ricordi.audiocraft]
repo_url = "https://github.com/aaronabebe/audiocraft"
commit_sha = "1ca32f28d2d6c223b4e2335042ac558c308f2ea2"

[tool.macchina_ricordi.defaults]
recipe = "micro-unconditional"
valid_fraction = 0.1
split_seed = 42

[tool.macchina_ricordi.recipes.base]
abstract = true
overrides = ["solver=musicgen/musicgen_base_32khz"]

[tool.macchina_ricordi.recipes."micro-unconditional"]
extends = "base"
overrides = ["dataset.segment_duration=15"]

[tool.macchina_ricordi.recipes."reshape-nano"]
extends = "base"
overrides = ["transformer_lm.dim=192"]
""".strip(),
        encoding="utf-8",
    )


def _write_hydra_overrides(xp_path: Path, overrides: list[str]) -> None:
    hydra_dir = xp_path / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "overrides.yaml").write_text(
        "\n".join(f"- {override}" for override in overrides) + "\n",
        encoding="utf-8",
    )


def test_list_epoch_checkpoints_ignores_non_epoch_files(tmp_path: Path) -> None:
    xp_path = tmp_path / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_2.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_10.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_best.th").write_text("", encoding="utf-8")

    checkpoints = list_epoch_checkpoints(xp_path)

    assert [checkpoint.epoch for checkpoint in checkpoints] == [2, 10]


def test_select_latest_checkpoint_for_sidecar_uses_stage_watermarks(tmp_path: Path) -> None:
    xp_path = tmp_path / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    for epoch in (1, 2, 3):
        (xp_path / f"checkpoint_{epoch}.th").write_text("", encoding="utf-8")

    checkpoints = list_epoch_checkpoints(xp_path)
    state = {
        "version": 1,
        "last_processed_epoch": {"evaluate": 2, "generate": 2},
    }

    candidate = select_latest_checkpoint_for_sidecar(
        checkpoints,
        state=state,
        stages=("evaluate", "generate"),
    )

    assert candidate is not None
    assert candidate.epoch == 3


def test_sidecar_state_persists_stage_watermarks(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = load_musicgen_cpu_sidecar_state(state_path)

    mark_stage_epoch_processed(state, stage="evaluate", epoch=4)
    save_musicgen_cpu_sidecar_state(state_path, state)
    reloaded = load_musicgen_cpu_sidecar_state(state_path)

    assert get_last_processed_epoch(reloaded, "evaluate") == 4
    assert get_last_processed_epoch(reloaded, "generate") == 0


def test_resolve_terminal_epoch_from_hydra_overrides_uses_last_override(tmp_path: Path) -> None:
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    _write_hydra_overrides(
        xp_path,
        [
            "optim.epochs=5",
            "dataset.batch_size=1",
            "optim.epochs=10",
        ],
    )

    terminal_epoch = resolve_terminal_epoch_from_hydra_overrides(xp_path)

    assert terminal_epoch == 10


def test_plan_musicgen_cpu_sidecar_uses_repo_defaults(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path)

    assert plan.xp_sig == "abcd1234"
    assert plan.musicgen_python == str(python_exe)
    assert plan.repo_root == tmp_path
    assert plan.dora_dir == tmp_path / ".dora"
    assert plan.stages == ("evaluate", "generate")
    assert plan.terminal_epoch is None


def test_plan_musicgen_cpu_sidecar_keeps_explicit_symlinked_python(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    real_python = tmp_path / "python-real"
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    real_python.write_text("", encoding="utf-8")
    try:
        python_exe.symlink_to(real_python)
    except OSError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"symlinks are unavailable: {exc}")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, musicgen_python=str(python_exe), once=True)
    command = build_musicgen_sidecar_runner_command(
        plan=plan,
        checkpoint=MusicGenCheckpointEpoch(epoch=1, path=xp_path / "checkpoint_1.th"),
        stage="evaluate",
    )

    assert plan.musicgen_python == str(python_exe.absolute())
    assert command[0] == str(python_exe.absolute())


def test_resolve_sidecar_xp_path_accepts_live_xp_dir_without_checkpoint_file(tmp_path: Path) -> None:
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)

    resolved = resolve_sidecar_xp_path(xp_path)

    assert resolved == xp_path.resolve()


def test_build_musicgen_sidecar_runner_command_and_env(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_3.th").write_text("", encoding="utf-8")
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, batch_size=2)
    checkpoint = list_epoch_checkpoints(xp_path)[0]
    command = build_musicgen_sidecar_runner_command(plan=plan, checkpoint=checkpoint, stage="evaluate")
    env = musicgen_script_env(dora_dir=plan.dora_dir, repo_root=plan.repo_root)

    assert command[:3] == [str(python_exe), "-m", "macchina_ricordi.training.sidecar_runner"]
    assert "--batch-size" in command
    assert env["AUDIOCRAFT_DORA_DIR"] == str(tmp_path / ".dora")
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert str(tmp_path) in env["PYTHONPATH"]


def test_run_musicgen_cpu_sidecar_once_processes_newest_checkpoint(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    for epoch in (1, 3):
        (xp_path / f"checkpoint_{epoch}.th").write_text("", encoding="utf-8")
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    seen_commands: list[list[str]] = []

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        seen_commands.append(list(command))
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="evaluate")
    returncode = run_musicgen_cpu_sidecar(plan)

    assert returncode == 0
    assert len(seen_commands) == 1
    assert str(xp_path / "checkpoint_3.th") in seen_commands[0]

    state_payload = json.loads(plan.state_path.read_text(encoding="utf-8"))
    assert state_payload["last_processed_epoch"]["evaluate"] == 3


def test_run_musicgen_cpu_sidecar_skips_terminal_generate_in_both_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    for epoch in (9, 10):
        (xp_path / f"checkpoint_{epoch}.th").write_text("", encoding="utf-8")
    _write_hydra_overrides(xp_path, ["optim.epochs=10"])
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    seen_commands: list[list[str]] = []

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        seen_commands.append(list(command))
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="both")
    assert plan.terminal_epoch == 10
    returncode = run_musicgen_cpu_sidecar(plan)

    assert returncode == 0
    assert len(seen_commands) == 1
    assert "--stage" in seen_commands[0]
    assert seen_commands[0][seen_commands[0].index("--stage") + 1] == "evaluate"

    state_payload = json.loads(plan.state_path.read_text(encoding="utf-8"))
    assert state_payload["last_processed_epoch"]["evaluate"] == 10
    assert state_payload["last_processed_epoch"]["generate"] == 10

    events = [json.loads(line) for line in plan.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip_events = [event for event in events if event["event"] == "stage_skipped_terminal_generate"]
    assert len(skip_events) == 1
    assert skip_events[0]["epoch"] == 10
    assert skip_events[0]["terminal_epoch"] == 10


def test_run_musicgen_cpu_sidecar_skips_terminal_generate_in_generate_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_10.th").write_text("", encoding="utf-8")
    _write_hydra_overrides(xp_path, ["optim.epochs=10"])
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    seen_commands: list[list[str]] = []

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        seen_commands.append(list(command))
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="generate")
    assert plan.terminal_epoch == 10
    returncode = run_musicgen_cpu_sidecar(plan)

    assert returncode == 0
    assert seen_commands == []
    state_payload = json.loads(plan.state_path.read_text(encoding="utf-8"))
    assert state_payload["last_processed_epoch"]["generate"] == 10

    events = [json.loads(line) for line in plan.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip_events = [event for event in events if event["event"] == "stage_skipped_terminal_generate"]
    assert len(skip_events) == 1
    assert skip_events[0]["epoch"] == 10


def test_run_musicgen_cpu_sidecar_uses_legacy_generate_behavior_when_terminal_epoch_unknown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_10.th").write_text("", encoding="utf-8")
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    seen_commands: list[list[str]] = []

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        seen_commands.append(list(command))
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="generate")
    assert plan.terminal_epoch is None
    returncode = run_musicgen_cpu_sidecar(plan)

    assert returncode == 0
    assert len(seen_commands) == 1
    assert "--stage" in seen_commands[0]
    assert seen_commands[0][seen_commands[0].index("--stage") + 1] == "generate"
    assert str(xp_path / "checkpoint_10.th") in seen_commands[0]


class _FakeRewardComponents:
    def __init__(self, **fields: float) -> None:
        self._fields = {
            "score": 0.0,
            "tempo_instability": 0.0,
            "onset_density": 0.0,
        }
        self._fields.update(fields)

    def to_dict(self) -> dict[str, float]:
        return dict(self._fields)


def _make_scorer(score_map: dict[str, float]):
    def _scorer(path):  # type: ignore[no-untyped-def]
        sample_id = Path(path).stem
        if sample_id not in score_map:
            raise ValueError(f"unknown sample: {sample_id}")
        return _FakeRewardComponents(score=score_map[sample_id])

    return _scorer


def _populate_generated_samples(xp_path: Path, epoch: int, sample_ids: list[str]) -> list[Path]:
    epoch_dir = xp_path / "samples" / str(epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sid in sample_ids:
        wav_path = epoch_dir / f"{sid}.wav"
        wav_path.write_bytes(b"fake-wav")
        paths.append(wav_path)
    return paths


def _make_plan(monkeypatch, tmp_path: Path, score_samples: bool = True):
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, score_samples=score_samples)
    return plan, xp_path


def test_discover_generated_samples_filters_reserved_dirs(tmp_path: Path) -> None:
    xp_path = tmp_path / "xp"
    xp_path.mkdir()
    epoch_dir = xp_path / "samples" / "5"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "alpha.wav").write_bytes(b"")
    (epoch_dir / "beta.wav").write_bytes(b"")
    (epoch_dir / "alpha.json").write_text("{}", encoding="utf-8")
    nested = epoch_dir / "prompt"
    nested.mkdir()
    (nested / "should_be_ignored.wav").write_bytes(b"")

    samples = discover_generated_samples(xp_path, 5)

    assert sorted(p.name for p in samples) == ["alpha.wav", "beta.wav"]


def test_score_samples_for_epoch_writes_jsonl_and_event(monkeypatch, tmp_path: Path) -> None:
    plan, xp_path = _make_plan(monkeypatch, tmp_path)
    _populate_generated_samples(xp_path, epoch=2, sample_ids=["alpha", "beta"])
    score_map = {"alpha": 0.4, "beta": 0.8}
    plan.artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary = score_samples_for_epoch(
        plan=plan,
        epoch=2,
        scorer=_make_scorer(score_map),
        reward_components_cls=_FakeRewardComponents,
    )

    assert summary["n_total"] == 2
    assert summary["n_failed"] == 0
    rows = [json.loads(line) for line in plan.scores_path.read_text().splitlines() if line.strip()]
    assert {row["sample_id"] for row in rows} == {"alpha", "beta"}
    assert all(row["epoch"] == 2 for row in rows)
    assert {row["score"] for row in rows} == {0.4, 0.8}
    assert all(row["score_error"] is None for row in rows)

    events = [json.loads(line) for line in plan.events_path.read_text().splitlines() if line.strip()]
    scored_events = [event for event in events if event["event"] == "samples_scored"]
    assert len(scored_events) == 1
    assert scored_events[0]["epoch"] == 2
    assert scored_events[0]["n_total"] == 2
    assert scored_events[0]["mean_score"] == pytest.approx(0.6)


def test_score_samples_for_epoch_records_per_sample_failures(monkeypatch, tmp_path: Path) -> None:
    plan, xp_path = _make_plan(monkeypatch, tmp_path)
    _populate_generated_samples(xp_path, epoch=1, sample_ids=["alpha", "broken"])
    plan.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def scorer(path):  # type: ignore[no-untyped-def]
        if Path(path).stem == "broken":
            raise RuntimeError("scoring kaput")
        return _FakeRewardComponents(score=0.5)

    summary = score_samples_for_epoch(
        plan=plan,
        epoch=1,
        scorer=scorer,
        reward_components_cls=_FakeRewardComponents,
    )

    assert summary["n_total"] == 2
    assert summary["n_failed"] == 1
    rows = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in plan.scores_path.read_text().splitlines()
        if line.strip()
    }
    assert rows["broken"]["score_error"] == "scoring kaput"
    assert rows["alpha"]["score_error"] is None

    events = [json.loads(line) for line in plan.events_path.read_text().splitlines() if line.strip()]
    failed = [event for event in events if event["event"] == "sample_score_failed"]
    assert len(failed) == 1
    assert failed[0]["sample_id"] == "broken"


def test_score_samples_for_epoch_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    plan, xp_path = _make_plan(monkeypatch, tmp_path)
    _populate_generated_samples(xp_path, epoch=4, sample_ids=["alpha", "beta"])
    plan.artifacts_dir.mkdir(parents=True, exist_ok=True)
    scorer = _make_scorer({"alpha": 0.3, "beta": 0.7})

    score_samples_for_epoch(plan=plan, epoch=4, scorer=scorer, reward_components_cls=_FakeRewardComponents)
    rows_first = [json.loads(line) for line in plan.scores_path.read_text().splitlines() if line.strip()]
    summary_second = score_samples_for_epoch(
        plan=plan, epoch=4, scorer=scorer, reward_components_cls=_FakeRewardComponents
    )
    rows_second = [json.loads(line) for line in plan.scores_path.read_text().splitlines() if line.strip()]

    assert len(rows_first) == 2
    assert rows_first == rows_second
    assert summary_second["n_total"] == 0
    assert summary_second["n_skipped"] == 2

    already = load_existing_scored_sample_ids(plan.scores_path, 4)
    assert already == {"alpha", "beta"}


def test_run_musicgen_cpu_sidecar_invokes_scoring_for_generate_stage(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_5.th").write_text("", encoding="utf-8")
    _populate_generated_samples(xp_path, epoch=5, sample_ids=["alpha"])
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    captured: list[dict] = []

    def fake_score_samples_for_epoch(*, plan, epoch):  # type: ignore[no-untyped-def]
        captured.append({"epoch": epoch, "scores_path": str(plan.scores_path)})
        return {"epoch": epoch, "n_total": 1, "n_failed": 0}

    monkeypatch.setattr("macchina_ricordi.training.sidecar.score_samples_for_epoch", fake_score_samples_for_epoch)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="generate")
    returncode = run_musicgen_cpu_sidecar(plan)
    assert returncode == 0
    assert captured == [{"epoch": 5, "scores_path": str(plan.scores_path)}]


def test_run_musicgen_cpu_sidecar_skips_scoring_when_disabled(monkeypatch, tmp_path: Path) -> None:
    _write_musicgen_project(tmp_path)
    xp_path = tmp_path / ".dora" / "xps" / "abcd1234"
    xp_path.mkdir(parents=True)
    (xp_path / "checkpoint.th").write_text("", encoding="utf-8")
    (xp_path / "checkpoint_2.th").write_text("", encoding="utf-8")
    _populate_generated_samples(xp_path, epoch=2, sample_ids=["alpha"])
    python_exe = tmp_path / ".venv-musicgen" / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.load_musicgen_project_config",
        lambda: load_musicgen_project_config(tmp_path),
    )
    monkeypatch.setattr(
        "macchina_ricordi.training.sidecar.require_musicgen_python", lambda explicit=None: str(python_exe)
    )

    def fake_run(command, check, stdout, stderr, env, cwd):  # type: ignore[no-untyped-def]
        stdout.write("ok\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("macchina_ricordi.training.sidecar.subprocess.run", fake_run)

    def fake_score_samples_for_epoch(*, plan, epoch):  # type: ignore[no-untyped-def]
        raise AssertionError("scoring must not run when score_samples=False")

    monkeypatch.setattr("macchina_ricordi.training.sidecar.score_samples_for_epoch", fake_score_samples_for_epoch)

    plan = plan_musicgen_cpu_sidecar(xp_path=xp_path, once=True, stage_mode="generate", score_samples=False)
    returncode = run_musicgen_cpu_sidecar(plan)
    assert returncode == 0
    assert not plan.scores_path.exists()
