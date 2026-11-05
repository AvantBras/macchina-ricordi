from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import macchina_ricordi.models.export as export
from appliance.spec import ModelSpec, check_card
from macchina_ricordi.models.export import plan_box_export, run_box_export


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    xp = tmp_path / ".dora" / "xps" / "abcd1234"
    (xp / ".hydra").mkdir(parents=True)
    (xp / ".hydra" / "config.yaml").write_text(
        "datasource:\n  train: /data/.workspace_disco/egs/disco/train\n"
        "dataset:\n  segment_duration: 10\n"
        "transformer_lm:\n  dim: 192\n  num_heads: 3\n  num_layers: 10\n  hidden_scale: 4\n  n_q: 4\n  card: 2048\n",
        encoding="utf-8",
    )
    (xp / "history.json").write_text(json.dumps([{"valid": {"ce": 3.0}}, {"valid": {"ce": 2.5}}]), encoding="utf-8")
    (xp / "checkpoint_2.th").write_bytes(b"x")
    (xp / "macchina_ricordi_train_metadata.json").write_text(
        json.dumps({"effective_label": "nano_disco", "recipe": "reshape-nano"}), encoding="utf-8"
    )
    (tmp_path / "musicgen").mkdir()
    (tmp_path / "musicgen" / "models.toml").write_text(
        'schema = "macchina_ricordi.models.v1"\n\n[models.abcd1234]\nname = "disco"\ntitle = "nano disco"\n'
        "measured = { clean_share = 0.9 }\n",
        encoding="utf-8",
    )
    musicgen_python = tmp_path / ".venv-musicgen" / "bin" / "python"
    monkeypatch.setattr(
        export,
        "load_musicgen_project_config",
        lambda: SimpleNamespace(repo_root=tmp_path, musicgen_python=musicgen_python),
    )
    monkeypatch.setattr("macchina_ricordi.models.record.get_repo_root", lambda explicit_root=None: tmp_path)
    return tmp_path


def test_the_plan_reads_the_model_record(repo: Path) -> None:
    plan = plan_box_export("disco@best", repo_root=repo)
    assert (plan.sig, plan.name, plan.epoch) == ("abcd1234", "disco_ep2", 2)
    assert plan.card["dataset"] == ".workspace_disco" and plan.card["recipe"] == "reshape-nano"
    assert plan.card["shape"]["dim"] == 192 and plan.card["note"] == "nano disco"
    assert plan.card["valid_ce"] == 2.5 and plan.card["measured"] == {"clean_share": 0.9}
    assert plan.export_command[1:4] == ("-m", "appliance.export", "abcd1234")
    assert "--card" in plan.export_command and plan.box_path.endswith("boxes/disco_ep2.nrbox")
    assert ("--weights-state", "model") == (plan.export_command[-2], plan.export_command[-1])
    assert plan.pack_command is not None and plan.pack_command[-2:] == ("--out", plan.box_path)
    assert len(plan.missing) == 2
    assert plan_box_export("disco", repo_root=repo, pack=False, name="x").pack_command is None


def test_run_writes_the_card_and_a_status_sidecar(repo: Path) -> None:
    plan = plan_box_export("disco@best", repo_root=repo)
    with pytest.raises(FileNotFoundError, match="missing"):
        run_box_export(plan, repo_root=repo)

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]["AUDIOCRAFT_DORA_DIR"]))
        return subprocess.CompletedProcess(command, 0 if len(calls) == 1 else 3)

    status = run_box_export(plan.__class__(**{**plan.to_dict(), "missing": ()}), repo_root=repo, run=fake_run)
    assert [c[0][2] for c in calls] == ["appliance.export", "appliance.pack"]
    assert calls[0][1] == str(repo / ".dora")
    assert (status["status"], status["failed_step"]) == ("failed", "pack")
    assert json.loads(Path(plan.card_path).read_text())["sig"] == "abcd1234"
    assert json.loads(Path(plan.status_path).read_text())["status"] == "failed"


def _spec(**overrides) -> ModelSpec:
    values = dict(
        dim=192,
        num_heads=3,
        num_layers=10,
        n_q=4,
        card=2048,
        hidden_scale=4.0,
        layer_scale=None,
        delays=[0, 1, 2, 3],
        frame_rate=50.0,
        sample_rate=32000,
        channels=1,
        segment_duration=10.0,
    )
    return ModelSpec(**(values | overrides))


def test_the_box_refuses_a_checkpoint_that_is_not_the_record() -> None:
    card = {"sig": "abcd1234", "shape": {"dim": 192, "num_layers": 10}, "segment_duration": 10.0}
    check_card(_spec(), card)
    with pytest.raises(ValueError, match="num_layers: record 10, weights 12"):
        check_card(_spec(num_layers=12), card)
    spec = ModelSpec.from_dict(_spec(model=card).to_dict())
    assert spec.model["sig"] == "abcd1234" and spec.card == 2048
    assert ModelSpec.from_dict({k: v for k, v in _spec().to_dict().items() if k != "model"}).model == {}
