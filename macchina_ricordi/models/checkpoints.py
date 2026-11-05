"""Local warm-start sources; downloading and publishing are deliberately external.

Portable bundles are converted to a small, derived AudioCraft checkpoint in the
local cache. The original bundle stays untouched and no training state is copied.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..musicgen.config import get_repo_root, load_musicgen_project_config, musicgen_script_env
from ..musicgen.runtime import require_musicgen_python


@dataclass(frozen=True)
class InitCheckpointSource:
    origin: str
    user_input: str
    resolved_local_path: Path
    resume_metadata_summary: dict[str, Any] | None = None
    bundle_config: dict[str, Any] | None = None


def resolve_dora_dir() -> Path:
    if value := os.environ.get("AUDIOCRAFT_DORA_DIR"):
        return Path(value).expanduser()
    return load_musicgen_project_config().dora_dir


def discover_xp_folder(xp_path: str | Path | None = None, *, dora_dir: Path | None = None) -> Path:
    """Resolve one existing local XP for sidecar evaluation, without any transfer protocol."""
    if xp_path is not None:
        path = Path(xp_path).expanduser().resolve()
        if not (path / "checkpoint.th").is_file():
            raise FileNotFoundError(f"No checkpoint.th found in {path}")
        return path
    base = (dora_dir or resolve_dora_dir()) / "xps"
    candidates = list(base.glob("*/checkpoint.th"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.th files under {base}; pass --xp-path explicitly.")
    if len(candidates) != 1:
        raise RuntimeError("Multiple XP folders found; pass --xp-path explicitly.")
    return candidates[0].parent.resolve()


def parse_init_checkpoint_uri(value: str, *, profile_name: str | None = None) -> tuple[str, dict[str, str]]:
    """Validate a local source, rejecting the retired remote-checkpoint protocol."""
    if "://" in value:
        raise ValueError(
            "Remote checkpoint URIs are no longer supported. Download and extract a model bundle, "
            "then pass its local directory with --from."
        )
    if not value.strip():
        raise ValueError("A local checkpoint file or model bundle directory is required.")
    return "local", {"path": value}


def fetch_init_checkpoint(
    value: str,
    *,
    profile_name: str | None = None,
    repo_root: Path | None = None,
    musicgen_python: str | None = None,
) -> InitCheckpointSource:
    """Resolve local weights for a fresh run, never a resume of the source run."""
    from .bundle import is_bundle, read_bundle

    _, parsed = parse_init_checkpoint_uri(value, profile_name=profile_name)
    path = Path(parsed["path"]).expanduser().resolve()
    if path.is_dir():
        if not is_bundle(path):
            raise ValueError(f"Not a portable model bundle: {path}. Pass a bundle directory or a checkpoint file.")
        bundle = read_bundle(path)
        canonical = json.dumps(bundle.config, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        root = Path(repo_root) if repo_root is not None else get_repo_root()
        destination = root / ".checkpoint_cache" / "bundles" / digest / "checkpoint.th"
        interpreter = require_musicgen_python(musicgen_python)
        helper = Path(__file__).resolve().parents[2] / "scripts" / "prepare_bundle_warm_start.py"
        if not helper.is_file():
            raise FileNotFoundError(f"Missing model bundle warm-start helper: {helper}")
        completed = subprocess.run(
            [interpreter, str(helper), str(path), str(destination)],
            env=musicgen_script_env(),
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"Cannot prepare bundle warm-start weights: {completed.stderr.strip()}")
        if not destination.is_file():
            raise RuntimeError("Bundle warm-start helper did not produce its checkpoint.")
        summary = {
            "bundle": {
                "name": bundle.config["name"],
                "weights": bundle.config["weights"],
                "provenance": bundle.config.get("provenance", {}),
            }
        }
        return InitCheckpointSource("bundle", value, destination, summary, bundle.config)
    if not path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {path}")
    if path.suffix == ".safetensors":
        raise ValueError("Pass the entire portable bundle directory, not model.safetensors alone.")
    return InitCheckpointSource("local", value, path, _summarize_local_metadata(path.parent))


def _summarize_local_metadata(folder: Path) -> dict[str, Any] | None:
    """Keep legacy local provenance usable without the retired transfer protocol."""
    for root in (folder / "resume-metadata", folder):
        fingerprint = root / "dataset_fingerprint.json"
        if fingerprint.is_file():
            try:
                payload = json.loads(fingerprint.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict):
                return {"dataset_fingerprint": payload}
    return None


def validate_bundle_training_compatibility(config: dict[str, Any], overrides: list[str], codec: str) -> None:
    """Catch explicit shape/codec mismatches before launching the training process.

    The runtime's strict state load remains the final check for resolved defaults.
    A fresh training schedule and different source audio are intentionally allowed.
    """
    expected_codec = "//pretrained/" + config["codec"]["name"]
    if codec != expected_codec:
        raise ValueError(f"Bundle codec mismatch: source uses {expected_codec!r}, target recipe uses {codec!r}.")
    keys = {
        "dim",
        "num_heads",
        "num_layers",
        "n_q",
        "card",
        "hidden_scale",
        "layer_scale",
        "positional_embedding",
        "max_period",
        "cross_attention",
        "norm_first",
        "bias_proj",
        "bias_ff",
    }
    actual = config["model_config"]["transformer_lm"]
    effective: dict[str, str] = {}
    for override in overrides:
        key, sep, raw = override.lstrip("+").partition("=")
        if sep:
            effective[key] = raw
    for key in keys:
        raw = effective.get("transformer_lm." + key)
        if raw is None or key not in actual:
            continue
        try:
            target = json.loads(raw)
        except ValueError:
            target = raw
        if target != actual[key]:
            raise ValueError(
                f"Bundle architecture mismatch for transformer_lm.{key}: source={actual[key]!r}, "
                f"target={target!r}. Select a compatible recipe/override; warm-start does not reshape weights."
            )
