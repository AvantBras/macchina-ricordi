"""Point the shared AudioCraft dataset config back at the data an XP trained on.

Training writes its datasource paths into ``<audiocraft>/config/dset/audio/<dataset_name>.yaml``, and
every profile with the same ``dataset_name`` shares that one file. Dora rebuilds an XP's config from it
whenever the XP is loaded by signature, so without this a resume or a CPU evaluate reads whichever
workspace was trained last. The XP's own ``.hydra/config.yaml`` holds the datasource it was created
with; that is what gets written back, every time, not only when the file is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .shared import utc_now_iso, write_json
from .templates import render_dataset_config

STATUS_FILENAME = "dataset_config_status.json"


class XpDatasourceMissing(FileNotFoundError):
    pass


@dataclass(frozen=True)
class XpDatasetConfig:
    status: str
    """``restored`` or ``skipped`` (the XP does not train on an ``audio/`` dset)."""
    config_path: Path | None
    train_dir: Path | None
    valid_dir: Path | None
    previous_train_dir: str | None
    """What the shared config pointed at before; differs from ``train_dir`` when the restore mattered."""

    @property
    def changed(self) -> bool:
        return self.train_dir is not None and self.previous_train_dir != str(self.train_dir)


def dset_override_from_argv(xp_folder: Path) -> str | None:
    argv_path = xp_folder / ".argv.json"
    if not argv_path.exists():
        return None
    payload = json.loads(argv_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"XP argv file is invalid: {argv_path}")
    for item in payload:
        if isinstance(item, str) and item.startswith("dset="):
            return item.split("=", 1)[1]
    return None


def load_saved_datasource(xp_folder: Path) -> dict[str, object]:
    hydra_config = xp_folder / ".hydra" / "config.yaml"
    if not hydra_config.exists():
        raise FileNotFoundError(f"XP Hydra config is missing: {hydra_config}")

    import yaml

    payload = yaml.safe_load(hydra_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasource"), dict):
        raise RuntimeError(f"XP Hydra config has no datasource block: {hydra_config}")
    return payload["datasource"]


def _configured_train_dir(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    import yaml

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    datasource = payload.get("datasource") if isinstance(payload, dict) else None
    train = datasource.get("train") if isinstance(datasource, dict) else None
    return str(train) if train is not None else None


def restore_xp_dataset_config(xp_folder: Path, *, audiocraft_dir: Path, purpose: str) -> XpDatasetConfig:
    """Rewrite the XP's dset config from its saved datasource and record it in ``dataset_config_status.json``.

    Raises ``XpDatasourceMissing`` when the XP's train or valid data no longer exists: loading the XP would
    otherwise run on another workspace's clips without saying so.
    """
    if not xp_folder.is_dir():
        raise FileNotFoundError(f"XP folder does not exist: {xp_folder}")
    status_path = xp_folder / STATUS_FILENAME
    base_status = {"purpose": purpose, "checked_at": utc_now_iso(), "xp_folder": str(xp_folder)}

    dset_override = dset_override_from_argv(xp_folder)
    if dset_override is None or not dset_override.startswith("audio/"):
        write_json(status_path, {**base_status, "status": "skipped", "reason": "XP argv has no audio dset override"})
        return XpDatasetConfig("skipped", None, None, None, None)

    config_path = audiocraft_dir / "config" / "dset" / "audio" / f"{dset_override.split('/', 1)[1]}.yaml"
    previous_train_dir = _configured_train_dir(config_path)
    datasource = load_saved_datasource(xp_folder)
    try:
        train_dir = Path(str(datasource["train"]))
        valid_dir = Path(str(datasource["valid"]))
    except KeyError as exc:
        raise RuntimeError(f"XP datasource is missing required key: {exc.args[0]}") from exc

    missing = [str(path) for path in (train_dir, valid_dir) if not path.exists()]
    if missing:
        write_json(
            status_path,
            {
                **base_status,
                "status": "failed",
                "reason": "XP training data no longer exists",
                "missing": missing,
                "dataset_config_path": str(config_path),
                "previous_train_dir": previous_train_dir,
            },
        )
        raise XpDatasourceMissing(
            f"XP {xp_folder.name} was trained on data that no longer exists: {', '.join(missing)}. "
            f"Refusing to {purpose} it on whatever {config_path.name} points at now "
            f"({previous_train_dir or 'nothing'})."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        render_dataset_config(
            train_dir=train_dir,
            valid_dir=valid_dir,
            evaluate_dir=Path(str(datasource.get("evaluate", valid_dir))),
            generate_dir=Path(str(datasource.get("generate", valid_dir))),
            max_sample_rate=int(datasource.get("max_sample_rate", 32_000)),
            max_channels=int(datasource.get("max_channels", 1)),
        ),
        encoding="utf-8",
    )
    result = XpDatasetConfig("restored", config_path, train_dir, valid_dir, previous_train_dir)
    write_json(
        status_path,
        {
            **base_status,
            "status": "restored",
            "dset_override": dset_override,
            "dataset_config_path": str(config_path),
            "train_dir": str(train_dir),
            "valid_dir": str(valid_dir),
            "previous_train_dir": previous_train_dir,
            "changed": result.changed,
        },
    )
    return result
