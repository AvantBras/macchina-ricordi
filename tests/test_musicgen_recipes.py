from __future__ import annotations

from pathlib import Path

import pytest

from macchina_ricordi.musicgen.config import load_musicgen_project_config

HEAD = """
[tool.macchina_ricordi.runtime]
project_environment = ".venv-musicgen"

[tool.macchina_ricordi.audiocraft]
repo_url = "https://example.invalid/audiocraft"
commit_sha = "0"
"""


def _config(tmp_path: Path, body: str):
    (tmp_path / "musicgen").mkdir(exist_ok=True)
    (tmp_path / "musicgen" / "pyproject.toml").write_text(HEAD + body)
    return load_musicgen_project_config(tmp_path)


def test_recipes_compose_through_extends_and_codecs(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        """
[tool.macchina_ricordi.defaults]
recipe = "small"

[tool.macchina_ricordi.codecs]
stock = "//pretrained/facebook/encodec_32khz"
mine = "/codecs/mine.th"

[tool.macchina_ricordi.recipes.base]
abstract = true
codec = "stock"
overrides = ["solver=musicgen/musicgen_base_32khz", "dataset.batch_size=3", "optim.epochs=5"]

[tool.macchina_ricordi.recipes.small]
extends = "base"
overrides = ["transformer_lm.dim=192", "dataset.batch_size=4"]

[tool.macchina_ricordi.recipes.small-mine]
extends = "small"
codec = "mine"
overrides = ["+dataset.train.sample_on_duration=true"]
""",
    )
    assert config.recipe_choices == ("small", "small-mine")
    assert config.default_recipe == "small"
    small = config.recipes["small"]
    assert small.overrides == (
        "compression_model_checkpoint=//pretrained/facebook/encodec_32khz",
        "solver=musicgen/musicgen_base_32khz",
        "dataset.batch_size=4",
        "optim.epochs=5",
        "transformer_lm.dim=192",
    )
    assert (small.extends, small.value("dataset.batch_size"), small.value("optim.lr")) == ("base", "4", None)
    mine = config.recipes["small-mine"]
    assert mine.value("compression_model_checkpoint") == "/codecs/mine.th"
    assert mine.value("dataset.train.sample_on_duration") == "true"
    assert len(mine.overrides) == len(small.overrides) + 1


@pytest.mark.parametrize(
    ("name", "dim", "heads", "layers", "hidden_scale"),
    [
        ("reshape-pico", "128", "2", "8", None),
        ("reshape-pico-wide", "192", "3", "2", "2"),
        ("reshape-pico-deep", "96", "2", "19", None),
        ("reshape-pico-slim", "128", "2", "12", "2"),
    ],
)
def test_pico_recipes_inherit_nano_training_defaults(name, dim, heads, layers, hidden_scale):
    config = load_musicgen_project_config(Path(__file__).resolve().parents[1])
    recipe = config.recipes[name]
    nano = config.recipes["reshape-nano"]
    assert name in config.recipe_choices
    assert "pico-family" not in config.recipe_choices
    assert recipe.extends == "pico-family"
    assert recipe.value("transformer_lm.dim") == dim
    assert recipe.value("transformer_lm.num_heads") == heads
    assert recipe.value("transformer_lm.num_layers") == layers
    assert recipe.value("transformer_lm.hidden_scale") == hidden_scale
    assert recipe.value("dataset.batch_size") == "4"
    assert recipe.value("checkpoint.keep_last") == "20"
    assert recipe.value("transformer_lm.checkpointing") == "none"
    for key in ("optim.lr", "optim.epochs", "transformer_lm.layer_scale", "compression_model_checkpoint"):
        assert recipe.value(key) == nano.value(key)


def test_the_repo_recipes_never_name_data() -> None:
    config = load_musicgen_project_config()
    assert config.default_recipe in config.recipes
    for recipe in config.recipes.values():
        assert recipe.value("solver") == "musicgen/musicgen_base_32khz", recipe.name
        assert recipe.value("dset") is None, recipe.name
        assert recipe.value("compression_model_checkpoint"), recipe.name
        assert recipe.value("dataset.segment_duration"), recipe.name
