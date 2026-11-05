from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..musicgen.config import default_recipe_name

TRAIN_METADATA_FILENAME = "macchina_ricordi_train_metadata.json"


def infer_xp_profile(xp_path: str | Path | None) -> str | None:
    if xp_path is None:
        return None
    xp = Path(xp_path).expanduser().resolve()
    for metadata_path in (xp / TRAIN_METADATA_FILENAME, xp / "metadata.json"):
        payload = _read_json_object(metadata_path)
        if payload is None:
            continue
        profile = payload.get("recipe") or payload.get("profile")
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
        fingerprint = payload.get("dataset_fingerprint")
        if isinstance(fingerprint, dict):
            profile = fingerprint.get("profile")
            if isinstance(profile, str) and profile.strip():
                return profile.strip()
    return None


def resolve_generation_xp_path(
    *,
    xp_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> Path | None:
    if xp_path is not None:
        resolved = Path(xp_path).expanduser().resolve()
        # A bundle is not the model's original training XP; never report it as
        # the XP driving generation (its metadata is self-contained).
        if _is_bundle_dir(resolved):
            return None
        return resolved
    if checkpoint_path is None:
        return None
    if _is_bundle_dir(checkpoint_path):
        return None
    return infer_checkpoint_xp_path(checkpoint_path)


def resolve_generation_profile(
    *,
    profile_name: str | None,
    xp_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> str:
    explicit = profile_name.strip() if profile_name else ""
    if explicit:
        return explicit
    resolved_xp = resolve_generation_xp_path(xp_path=xp_path, checkpoint_path=checkpoint_path)
    inferred = infer_xp_profile(resolved_xp)
    return inferred or default_recipe_name()


def _is_bundle_dir(path: str | Path | None) -> bool:
    if path in (None, ""):
        return False
    try:
        from ..models.bundle_runtime import is_bundle_dir

        return is_bundle_dir(path)
    except Exception:  # noqa: BLE001 - a path that cannot be probed is not a bundle here
        return False


def infer_checkpoint_xp_path(checkpoint_path: str | Path) -> Path | None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    parent = checkpoint.parent
    if _looks_like_xp_folder(parent):
        return parent

    for sidecar_name in ("merged_from.json", "perturbed_from.json"):
        payload = _read_json_object(parent / sidecar_name)
        if payload is None:
            continue
        for source_path in _iter_source_paths(payload):
            source = Path(source_path).expanduser()
            if not source.is_absolute():
                source = parent / source
            source = source.resolve()
            candidate = source.parent if source.name.startswith("checkpoint") else source
            if _looks_like_xp_folder(candidate):
                return candidate
    return None


def _looks_like_xp_folder(path: Path) -> bool:
    return (
        (path / TRAIN_METADATA_FILENAME).exists()
        or (path / "metadata.json").exists()
        or (path / ".hydra" / "config.yaml").exists()
    )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_source_paths(value: object) -> tuple[str, ...]:
    paths: list[str] = []

    def walk(item: object) -> None:
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str) and path.strip():
                paths.append(path)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return tuple(paths)
