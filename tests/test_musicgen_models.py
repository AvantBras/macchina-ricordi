from __future__ import annotations

import json
from pathlib import Path

import pytest

from macchina_ricordi.cli import main
from macchina_ricordi.models import record as models
from macchina_ricordi.models.record import (
    list_models,
    load_model,
    resolve_model,
)

HEADER = "# hand-written header\n# second line\n"


def _xp(
    root: Path,
    sig: str,
    *,
    label: str,
    workspace: str,
    valid_ce: list[float],
    epoch_checkpoints: tuple[int, ...] = (),
    final: bool = True,
    metadata: dict | None = None,
) -> Path:
    folder = root / ".dora" / "xps" / sig
    (folder / ".hydra").mkdir(parents=True)
    (folder / ".argv.json").write_text(json.dumps(["solver=musicgen/musicgen_base_32khz", f"label={label}"]))
    train = root / workspace / "egs" / "macchina_ricordi_micro_uncond" / "train"
    (folder / ".hydra" / "config.yaml").write_text(
        f"""label: {label}
compression_model_checkpoint: //pretrained/facebook/encodec_32khz
conditioners: {{}}
datasource:
  train: {train}
  valid: {train.parent / "valid"}
dataset:
  segment_duration: 10
optim:
  epochs: 3
transformer_lm:
  dim: 192
  num_heads: 3
  num_layers: 10
  hidden_scale: 4
  n_q: 4
  card: 2048
  dropout: 0.0
"""
    )
    history = [{"train": {"ce": ce + 0.5}, "valid": {"ce": ce}} for ce in valid_ce]
    (folder / "history.json").write_text(json.dumps(history))
    for epoch in epoch_checkpoints:
        (folder / f"checkpoint_{epoch}.th").write_bytes(b"x")
    if final:
        (folder / "checkpoint.th").write_bytes(b"x")
    payload = {"effective_label": label, "profile": "reshape-nano", "workspace_dir": str(root / workspace)}
    payload.update(metadata or {})
    (folder / "macchina_ricordi_train_metadata.json").write_text(json.dumps(payload))
    return folder


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(models, "get_repo_root", lambda explicit_root=None: tmp_path)
    monkeypatch.setenv("AUDIOCRAFT_DORA_DIR", str(tmp_path / ".dora"))
    alpha = _xp(
        tmp_path,
        "aaaa1111",
        label="alpha_run",
        workspace=".workspace_alpha",
        valid_ce=[3.0, 2.5, 2.7],
        epoch_checkpoints=(1, 2),
    )
    _xp(
        tmp_path,
        "bbbb2222",
        label="beta_run",
        workspace=".workspace_beta",
        valid_ce=[4.0, 3.9],
        metadata={"init_from_checkpoint": {"resolved_local_path": str(alpha / "checkpoint_2.th")}},
    )
    (tmp_path / "musicgen").mkdir()
    (tmp_path / "musicgen" / "models.toml").write_text(
        HEADER
        + '\nschema = "macchina_ricordi.models.v1"\n\n[models.aaaa1111]\nname = "alpha"\ntitle = "the first one"\n'
        "measured = { clean_share = 0.9, top_k_cliff = 25 }\n"
    )
    loose = tmp_path / "artifacts" / "checkpoints" / "merged" / "blend"
    loose.mkdir(parents=True)
    (loose / "checkpoint.th").write_bytes(b"x")
    return tmp_path


def test_load_model_reads_what_training_wrote(repo: Path) -> None:
    alpha = load_model(repo / ".dora" / "xps" / "aaaa1111")
    assert (alpha.name, alpha.title, alpha.label, alpha.recipe) == (
        "alpha",
        "the first one",
        "alpha_run",
        "reshape-nano",
    )
    assert alpha.dataset == ".workspace_alpha"
    assert alpha.shape == {"dim": 192, "num_heads": 3, "num_layers": 10, "hidden_scale": 4, "n_q": 4, "card": 2048}
    assert (alpha.segment_duration, alpha.epochs_planned, alpha.epochs_done) == (10.0, 3, 3)
    assert (alpha.best_epoch, alpha.best_valid_ce) == (2, 2.5)
    assert [(c.path.name, c.epoch, c.final) for c in alpha.checkpoints] == [
        ("checkpoint_1.th", 1, False),
        ("checkpoint_2.th", 2, False),
        ("checkpoint.th", 3, True),
    ]
    assert alpha.measured == {"clean_share": 0.9, "top_k_cliff": 25}
    assert alpha.warnings == ()
    beta = load_model(repo / ".dora" / "xps" / "bbbb2222")
    assert (beta.parents, beta.origin) == (("aaaa1111",), "warm-start")


@pytest.mark.parametrize(
    ("text", "sig"),
    [
        ("alpha", "aaaa1111"),
        ("ALPHA", "aaaa1111"),
        ("aaaa1111", "aaaa1111"),
        ("bbbb", "bbbb2222"),
        ("beta_run", "bbbb2222"),
        ("beta-run", "bbbb2222"),
        ("Beta Run", "bbbb2222"),
    ],
)
def test_resolve_by_name_sig_prefix_and_label(repo: Path, text: str, sig: str) -> None:
    ref = resolve_model(text)
    assert ref.model is not None and ref.model.sig == sig
    assert ref.checkpoint is None


def test_model_cli_list_show_and_name(repo: Path, capsys) -> None:
    assert main(["model", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [entry["sig"] for entry in listed] == ["aaaa1111", "bbbb2222"]
    assert listed[0]["name"] == "alpha"

    assert main(["model", "list", "--dataset", "beta"]) == 0
    out = capsys.readouterr().out
    assert "bbbb2222" in out and "aaaa1111" not in out

    assert main(["model", "show", "alpha@best", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["selected_checkpoint"]["epoch"] == 2

    assert main(["model", "name", "beta_run", "beta"]) == 0
    assert "is now 'beta'" in capsys.readouterr().out
    assert [m.name for m in list_models()] == ["alpha", "beta"]

    assert main(["model", "show", "nope"]) == 1
    assert "no model called 'nope'" in capsys.readouterr().err
