from __future__ import annotations

import json
from pathlib import Path

# The single name new workspaces are prepared under (``egs/<name>/``), which is also the AudioCraft dataset
# config every training run points through (``config/dset/audio/<name>.yaml``). Which workspace that file
# names is decided at each launch or resume, never by a recipe.
DATASET_CONFIG_NAME = "macchina_ricordi_micro_uncond"


def _yaml_quote_path(path: Path) -> str:
    return json.dumps(path.resolve().as_posix())


def render_dataset_config(
    *,
    train_dir: Path,
    valid_dir: Path,
    evaluate_dir: Path | None = None,
    generate_dir: Path | None = None,
    max_sample_rate: int = 32_000,
    max_channels: int = 1,
) -> str:
    eval_dir = evaluate_dir or valid_dir
    gen_dir = generate_dir or valid_dir
    return "\n".join(
        [
            "# @package __global__",
            "",
            "datasource:",
            f"  max_sample_rate: {int(max_sample_rate)}",
            f"  max_channels: {int(max_channels)}",
            f"  train: {_yaml_quote_path(train_dir)}",
            f"  valid: {_yaml_quote_path(valid_dir)}",
            f"  evaluate: {_yaml_quote_path(eval_dir)}",
            f"  generate: {_yaml_quote_path(gen_dir)}",
            "",
        ]
    )


def render_and_write_dataset_config(
    *,
    audiocraft_dir: Path,
    workspace_dir: Path,
    dataset_name: str = DATASET_CONFIG_NAME,
    train_dir: Path | None = None,
) -> Path:
    """Point the shared dataset config at a workspace; ``train_dir`` swaps in a weights overlay's manifest."""
    egs_dir = workspace_dir / "egs" / dataset_name
    config_path = audiocraft_dir / "config" / "dset" / "audio" / f"{dataset_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        render_dataset_config(
            train_dir=train_dir or egs_dir / "train",
            valid_dir=egs_dir / "valid",
        ),
        encoding="utf-8",
    )
    return config_path
