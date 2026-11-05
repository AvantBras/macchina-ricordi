"""Plan a training: resolve the recipe, dataset and warm start into a Dora command and its signature."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..data.clip_stats import (
    compute_clip_stats,
    extract_override_float,
    read_manifest_durations,
    validate_segment_duration,
)
from ..data.prepare import (
    _dataset_fingerprint_digest,
    _dataset_fingerprint_summary,
    _load_json_file,
    _load_or_build_dataset_fingerprint,
    resolve_workspace_dir,
    workspace_dataset_name,
)
from ..data.weights import load_weights_overlay, read_rebalance_sidecar, rebalance_label_suffix
from ..models.record import sanitize_label
from ..musicgen.config import (
    FALLBACK_DEFAULT_RECIPES,
    default_recipe_name,
    get_recipe,
    get_repo_root,
)
from ..musicgen.runtime import (
    build_musicgen_subprocess_env,
    load_musicgen_project_config,
    require_musicgen_python,
    resolve_audiocraft_dir,
)
from ..musicgen.shared import resolve_dora_command
from ..musicgen.templates import DATASET_CONFIG_NAME, render_and_write_dataset_config

DEFAULT_TRAINING_PROFILE = FALLBACK_DEFAULT_RECIPES[0]
TRAIN_METADATA_FILENAME = "macchina_ricordi_train_metadata.json"
_FALLBACK_DORA_EXCLUDES = (
    "device",
    "wandb.*",
    "tensorboard.*",
    "logging.*",
    "dataset.num_workers",
    "eval.num_workers",
    "special.*",
    "metrics.visqol.bin",
    "metrics.fad.bin",
    "execute_only",
    "execute_best",
    "generate.every",
    "optim.eager_sync",
    "profiler.*",
    "deadlock.*",
    "efficient_attention_backend",
    "num_threads",
    "mp_start_method",
    "dora.*",
    "slurm.*",
)


@dataclass(frozen=True)
class MusicGenTrainingPlan:
    workspace_dir: Path
    audiocraft_dir: Path
    dataset_config_path: Path
    command: list[str]
    dry_run: bool
    init_from_checkpoint: dict[str, object] | None = None
    xp_sig: str | None = None
    xp_folder: Path | None = None
    effective_label: str | None = None
    auto_dataset_label: bool = False
    dataset_fingerprint_digest: str | None = None
    dataset_fingerprint_summary: dict[str, object] | None = None
    train_metadata_path: Path | None = None
    train_metadata: dict[str, object] | None = None
    segment_warnings: tuple[str, ...] = ()
    plan_record: dict[str, object] | None = None
    """What ``train_plan.json`` holds; written into the job's folder when the training starts."""


@dataclass(frozen=True)
class MusicGenXpReference:
    sig: str
    folder: Path


def build_training_command(
    *,
    musicgen_python: str,
    profile_name: str | None = None,
    overrides: Sequence[str] | None = None,
    init_from_checkpoint_path: str | Path | None = None,
    dataset_name: str = DATASET_CONFIG_NAME,
) -> list[str]:
    # Runtime default: honour the checkout's [tool.macchina_ricordi.defaults] recipe when not given.
    if profile_name is None:
        profile_name = default_recipe_name()
    final_overrides = [*get_recipe(profile_name).overrides, f"dset=audio/{dataset_name}"]
    if overrides:
        final_overrides.extend(overrides)
    if init_from_checkpoint_path is not None:
        final_overrides = _replace_continue_from_override(
            final_overrides,
            str(init_from_checkpoint_path),
        )
    return [*resolve_dora_command(musicgen_python), "run", *final_overrides]


def _replace_continue_from_override(
    overrides: Sequence[str],
    checkpoint_path: str,
) -> list[str]:
    filtered = [override for override in overrides if not override.startswith("continue_from=")]
    filtered.append(f"continue_from={checkpoint_path}")
    return filtered


def _has_user_continue_from_override(user_overrides: Sequence[str] | None) -> bool:
    if not user_overrides:
        return False
    return any(override.startswith("continue_from=") for override in user_overrides)


def _label_override_value(overrides: Sequence[str] | None) -> str | None:
    if not overrides:
        return None
    for override in reversed(overrides):
        if override.startswith("label="):
            return override.split("=", 1)[1]
    return None


def _workspace_slug(workspace_dir: Path) -> str:
    return sanitize_label(workspace_dir.name.lstrip(".") or "workspace")[:48]


def _build_auto_dataset_label(*, workspace_dir: Path, fingerprint_digest: str) -> str:
    return f"nr_{_workspace_slug(workspace_dir)}_{fingerprint_digest}"


def _resolve_training_overrides(
    *,
    overrides: Sequence[str] | None,
    run_label: str | None,
    auto_dataset_label: str | None,
    no_auto_dataset_label: bool,
) -> tuple[list[str], str | None, bool]:
    resolved = list(overrides or [])
    user_label = _label_override_value(resolved)
    if run_label is not None and user_label is not None:
        raise RuntimeError("Cannot combine --run-label with --override label=... . Pick exactly one label source.")

    if run_label is not None:
        label = sanitize_label(run_label)
        resolved.append(f"label={label}")
        return resolved, label, False

    if user_label is not None:
        return resolved, user_label, False

    if no_auto_dataset_label:
        return resolved, None, False

    if auto_dataset_label is None:
        raise RuntimeError("Automatic dataset label requested but no dataset label could be derived.")
    resolved.append(f"label={auto_dataset_label}")
    return resolved, auto_dataset_label, True


def _command_dora_argv(command: Sequence[str]) -> list[str]:
    try:
        run_index = list(command).index("run")
    except ValueError as exc:
        raise RuntimeError("Training command does not contain Dora `run`.") from exc
    return list(command[run_index + 1 :])


def _is_executable_python(path: str) -> bool:
    python_path = Path(path)
    return python_path.exists() and python_path.is_file() and os.access(python_path, os.X_OK)


def resolve_training_xp_reference(
    *,
    command: Sequence[str],
    musicgen_python: str,
    audiocraft_dir: Path,
) -> MusicGenXpReference:
    argv = _command_dora_argv(command)
    if _is_executable_python(musicgen_python):
        script = (
            "import json, os, sys\n"
            "audiocraft_dir = os.environ['MACCHINA_RICORDI_AUDIOCRAFT_DIR']\n"
            "sys.path.insert(0, audiocraft_dir)\n"
            "from audiocraft import train\n"
            "argv = json.loads(os.environ['MACCHINA_RICORDI_DORA_ARGV'])\n"
            "xp = train.main.get_xp(argv)\n"
            "print(json.dumps({'sig': xp.sig, 'folder': str(xp.folder)}))\n"
        )
        env = _build_training_reference_env(
            argv=argv,
            audiocraft_dir=audiocraft_dir,
        )
        try:
            completed = subprocess.run(
                [musicgen_python, "-c", script],
                cwd=audiocraft_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError("Failed to resolve the Dora XP signature before launch.") from exc
        return MusicGenXpReference(
            sig=str(payload["sig"]),
            folder=Path(str(payload["folder"])).expanduser().resolve(),
        )

    return _fallback_training_xp_reference(argv)


def _build_training_reference_env(*, argv: Sequence[str], audiocraft_dir: Path) -> dict[str, str]:
    env = build_musicgen_subprocess_env()
    env["MACCHINA_RICORDI_DORA_ARGV"] = json.dumps(list(argv))
    env["MACCHINA_RICORDI_AUDIOCRAFT_DIR"] = str(audiocraft_dir)
    env["PYTHONPATH"] = (
        str(audiocraft_dir) if not env.get("PYTHONPATH") else f"{audiocraft_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    return env


def _fallback_training_xp_reference(argv: Sequence[str]) -> MusicGenXpReference:
    simplified = _simplify_dora_argv(argv)
    delta: list[tuple[str, str]] = []
    for override in simplified:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        key = key.strip()
        if not _fallback_dora_excluded(key):
            delta.append((key, value))
    sig_payload = json.dumps(sorted(delta)).encode("utf-8")
    sig = hashlib.sha1(sig_payload).hexdigest()[:8]
    dora_dir = Path(os.environ.get("AUDIOCRAFT_DORA_DIR", load_musicgen_project_config().dora_dir)).resolve()
    return MusicGenXpReference(sig=sig, folder=dora_dir / "xps" / sig)


def _simplify_dora_argv(argv: Sequence[str]) -> list[str]:
    simplified: list[str] = []
    seen: set[str] = set()
    for override in reversed(list(argv)):
        if "=" not in override:
            simplified.append(override)
            continue
        key = override.split("=", 1)[0].strip()
        if key in seen:
            continue
        seen.add(key)
        simplified.append(override)
    return list(reversed(simplified))


def _fallback_dora_excluded(key: str) -> bool:
    return any(fnmatch.fnmatch(key, pattern) for pattern in _FALLBACK_DORA_EXCLUDES)


def _validate_existing_xp_metadata(
    *,
    xp_reference: MusicGenXpReference,
    expected_metadata: dict[str, object],
    auto_dataset_label: bool,
) -> Path:
    metadata_path = xp_reference.folder / TRAIN_METADATA_FILENAME
    if not xp_reference.folder.exists():
        return metadata_path

    if metadata_path.exists():
        existing = _load_json_file(metadata_path)
        mismatches = [
            field
            for field in ("profile", "dataset_name", "workspace_dir", "dataset_fingerprint_digest")
            if existing.get(field) != expected_metadata.get(field)
        ]
        # Records from before overlays carry no weights_digest; an unweighted run matches them.
        if existing.get("weights_digest") != expected_metadata.get("weights_digest") and (
            "weights_digest" in existing or expected_metadata.get("weights_digest") is not None
        ):
            mismatches.append("weights_digest")
        if mismatches:
            raise RuntimeError(
                "Refusing to reuse existing Dora XP "
                f"{xp_reference.sig}: {TRAIN_METADATA_FILENAME} differs for {', '.join(mismatches)}. "
                "Use a unique --run-label for a new run, or inspect the existing XP before resuming."
            )
        return metadata_path

    if auto_dataset_label:
        return metadata_path

    raise RuntimeError(
        "Refusing to reuse existing Dora XP "
        f"{xp_reference.sig} because it has no {TRAIN_METADATA_FILENAME}. "
        "Use a unique --run-label for a new run, or pass --no-auto-dataset-label only when you intentionally "
        "want legacy Dora signature behavior."
    )


def plan_musicgen_training(
    *,
    audiocraft_dir: str | None,
    profile_name: str | None = None,
    workspace_dir: str | Path | None = None,
    musicgen_python: str | None = None,
    overrides: Sequence[str] | None = None,
    run_label: str | None = None,
    no_auto_dataset_label: bool = False,
    dry_run: bool = False,
    init_from_checkpoint: str | None = None,
    allow_segment_padding: bool = False,
    weights: str | None = None,
    repo_root: str | Path | None = None,
) -> MusicGenTrainingPlan:
    """Resolve everything a training launch needs. Writes no record; ``run_musicgen_training`` writes the run's."""
    # Runtime default: honour the checkout's configured default recipe when not given.
    if profile_name is None:
        profile_name = default_recipe_name()
    recipe = get_recipe(profile_name)
    workspace = resolve_workspace_dir(workspace_dir)
    dataset_name = workspace_dataset_name(workspace)
    train_manifest = workspace / "egs" / dataset_name / "train" / "data.jsonl.gz"

    if init_from_checkpoint is not None and _has_user_continue_from_override(overrides):
        raise RuntimeError(
            "Cannot combine --init-from-checkpoint with a user-supplied "
            "--override continue_from=... . Pick exactly one of them."
        )

    rebalance_sidecar = read_rebalance_sidecar(train_manifest.parent)
    overlay = None
    if weights is not None:
        if rebalance_sidecar is not None:
            raise RuntimeError(
                f"{workspace.name} has weights written into its own manifest (from before overlays). Clear them "
                f"with `data rebalance --reset --workspace-dir {workspace.name}`, or train without --weights."
            )
        overlay = load_weights_overlay(Path(repo_root) if repo_root else get_repo_root(), workspace.name, weights)
        if overlay.dataset_name != dataset_name or overlay.is_stale():
            raise RuntimeError(
                f"weights {weights!r} were computed for a manifest that has changed since; compute them again"
            )

    resolved_audiocraft = resolve_audiocraft_dir(audiocraft_dir)
    resolved_python = require_musicgen_python(musicgen_python)
    config_path = render_and_write_dataset_config(
        audiocraft_dir=resolved_audiocraft,
        workspace_dir=workspace,
        dataset_name=dataset_name,
        train_dir=overlay.train_dir if overlay is not None else None,
    )
    dataset_fingerprint = _load_or_build_dataset_fingerprint(
        workspace_dir=workspace,
        dataset_name=dataset_name,
    )
    fingerprint_summary = _dataset_fingerprint_summary(dataset_fingerprint)
    fingerprint_digest = _dataset_fingerprint_digest(dataset_fingerprint)
    auto_label = (
        None
        if no_auto_dataset_label
        else _build_auto_dataset_label(
            workspace_dir=workspace,
            fingerprint_digest=fingerprint_digest,
        )
    )
    if auto_label is not None:
        # A rebalanced manifest (per-segment sampling weights) is a different
        # training distribution over the same audio files, so it must yield a
        # different Dora signature than the unweighted run.
        suffix = overlay.label_suffix if overlay is not None else rebalance_label_suffix(rebalance_sidecar)
        auto_label = f"{auto_label}{suffix}"
    effective_overrides, effective_label, auto_label_used = _resolve_training_overrides(
        overrides=overrides,
        run_label=run_label,
        auto_dataset_label=auto_label,
        no_auto_dataset_label=no_auto_dataset_label,
    )
    if rebalance_sidecar is not None or overlay is not None:
        # The base musicgen solver disables weight sampling ("uniform all the
        # way"), so a rebalanced manifest needs this re-enabled for the train
        # split. The `+` prefix is required: the per-split section does not
        # declare the key, so Hydra strict mode only accepts an append.
        effective_overrides = [
            "+dataset.train.sample_on_weight=true",
            *effective_overrides,
        ]

    # Guard the data<->model boundary: the effective segment_duration (D) must not
    # exceed the dataset's typical clip length (S), otherwise every example is
    # silence-padded and crop augmentation is disabled. Profile overrides come
    # first, then user --override, so the last value wins (matches build_training_command).
    combined_overrides = list(recipe.overrides) + list(effective_overrides)
    segment_duration = extract_override_float(combined_overrides, "dataset.segment_duration")
    min_segment_ratio = extract_override_float(combined_overrides, "dataset.min_segment_ratio") or 0.8
    clip_durations = read_manifest_durations(train_manifest)
    clip_stats = compute_clip_stats(clip_durations)
    segment_warnings = validate_segment_duration(
        clip_stats,
        segment_duration,
        durations=clip_durations,
        min_segment_ratio=min_segment_ratio,
        allow_padding=allow_segment_padding,
    )

    init_metadata: dict[str, object] | None = None
    init_checkpoint_path: Path | None = None
    if init_from_checkpoint is not None:
        # Imported lazily to avoid circular imports.
        from ..models.checkpoints import fetch_init_checkpoint, validate_bundle_training_compatibility

        source = fetch_init_checkpoint(
            init_from_checkpoint,
            profile_name=profile_name,
            repo_root=Path(repo_root) if repo_root is not None else None,
            musicgen_python=resolved_python,
        )
        if source.bundle_config is not None:
            codec = next(
                (
                    item.split("=", 1)[1]
                    for item in reversed(combined_overrides)
                    if item.lstrip("+").startswith("compression_model_checkpoint=")
                ),
                # The pinned AudioCraft musicgen_base_32khz default; recipes need not repeat it.
                "//pretrained/facebook/encodec_32khz",
            )
            validate_bundle_training_compatibility(source.bundle_config, combined_overrides, codec)
        init_checkpoint_path = source.resolved_local_path
        init_metadata = _build_init_checkpoint_metadata(source=source, workspace_dir=workspace)

    command = build_training_command(
        musicgen_python=resolved_python,
        profile_name=profile_name,
        overrides=effective_overrides,
        init_from_checkpoint_path=init_checkpoint_path,
        dataset_name=dataset_name,
    )
    xp_reference = resolve_training_xp_reference(
        command=command,
        musicgen_python=resolved_python,
        audiocraft_dir=resolved_audiocraft,
    )
    train_metadata: dict[str, object] = {
        "recipe": recipe.name,
        "profile": recipe.name,
        "dataset_name": dataset_name,
        "workspace_dir": str(workspace),
        "dataset_fingerprint_digest": fingerprint_digest,
        "dataset_fingerprint": fingerprint_summary,
        "command": command,
        "effective_label": effective_label,
        "auto_dataset_label": auto_label_used,
        "xp_sig": xp_reference.sig,
        "xp_folder": str(xp_reference.folder),
        "segment_duration": segment_duration,
        "clip_slice_length_s": (clip_stats or {}).get("slice_length_s"),
        "segment_padding_allowed": allow_segment_padding,
        "segment_warnings": segment_warnings,
    }
    if overlay is not None:
        train_metadata["weights"] = overlay.summary()
        train_metadata["weights_digest"] = overlay.weights_digest
    if rebalance_sidecar is not None:
        train_metadata["rebalance"] = {
            key: rebalance_sidecar.get(key) for key in ("mode", "k", "alpha", "cap", "weights_digest", "written_at")
        }
    if init_metadata is not None:
        train_metadata["init_from_checkpoint"] = init_metadata
    metadata_path = _validate_existing_xp_metadata(
        xp_reference=xp_reference,
        expected_metadata=train_metadata,
        auto_dataset_label=auto_label_used,
    )
    train_plan_payload: dict[str, object] = {
        "recipe": recipe.name,
        "profile": recipe.name,
        "dataset_name": dataset_name,
        "workspace_dir": str(workspace),
        "audiocraft_dir": str(resolved_audiocraft),
        "dataset_config_path": str(config_path),
        "command": command,
        "dry_run": dry_run,
        "xp_sig": xp_reference.sig,
        "xp_folder": str(xp_reference.folder),
        "effective_label": effective_label,
        "auto_dataset_label": auto_label_used,
        "dataset_fingerprint_digest": fingerprint_digest,
        "dataset_fingerprint": fingerprint_summary,
        "train_metadata_path": str(metadata_path),
        "segment_duration": segment_duration,
        "clip_slice_length_s": (clip_stats or {}).get("slice_length_s"),
        "segment_padding_allowed": allow_segment_padding,
        "segment_warnings": segment_warnings,
    }
    if init_metadata is not None:
        train_plan_payload["init_from_checkpoint"] = init_metadata
    if overlay is not None:
        train_plan_payload["weights"] = overlay.summary()
    plan = MusicGenTrainingPlan(
        workspace_dir=workspace,
        audiocraft_dir=resolved_audiocraft,
        dataset_config_path=config_path,
        command=command,
        dry_run=dry_run,
        init_from_checkpoint=init_metadata,
        xp_sig=xp_reference.sig,
        xp_folder=xp_reference.folder,
        effective_label=effective_label,
        auto_dataset_label=auto_label_used,
        dataset_fingerprint_digest=fingerprint_digest,
        dataset_fingerprint_summary=fingerprint_summary,
        train_metadata_path=metadata_path,
        train_metadata=train_metadata,
        segment_warnings=tuple(segment_warnings),
        plan_record=train_plan_payload,
    )
    return plan


def _build_init_checkpoint_metadata(
    *,
    source,  # type: ignore[no-untyped-def]
    workspace_dir: Path,
) -> dict[str, object]:
    warnings: list[str] = []
    fingerprint_comparison: bool | None = None
    current_fingerprint_path = workspace_dir / "logs" / "dataset_fingerprint.json"
    current_fingerprint: dict[str, object] | None = None
    if current_fingerprint_path.exists():
        try:
            current_fingerprint = _load_json_file(current_fingerprint_path)
        except (OSError, ValueError):
            current_fingerprint = None

    source_fingerprint: dict[str, object] | None = None
    if source.resume_metadata_summary is not None:
        source_fingerprint = source.resume_metadata_summary.get("dataset_fingerprint")

    if isinstance(source_fingerprint, dict) and isinstance(current_fingerprint, dict):
        fingerprint_comparison = source_fingerprint.get("train_aggregate_sha256") == current_fingerprint.get(
            "train_aggregate_sha256"
        ) and source_fingerprint.get("valid_aggregate_sha256") == current_fingerprint.get("valid_aggregate_sha256")
        if not fingerprint_comparison:
            warnings.append(
                "Source checkpoint was trained on a different dataset than the current "
                "workspace (dataset fingerprints differ). This is expected for artistic "
                "warm-starts; the new run will still load source weights as initialization."
            )
    elif source.resume_metadata_summary is None:
        warnings.append(
            "Source checkpoint has no accompanying resume-metadata; dataset compatibility "
            "cannot be inspected. Proceeding without comparison."
        )
    elif source.origin != "bundle" and not isinstance(source_fingerprint, dict):
        warnings.append(
            "Source checkpoint resume-metadata does not include a dataset fingerprint "
            "(e.g., merged or perturbed checkpoints may carry only lineage sidecars); dataset compatibility "
            "cannot be inspected. Proceeding without comparison."
        )

    metadata: dict[str, object] = {
        "origin": source.origin,
        "user_input": source.user_input,
        "resolved_local_path": str(source.resolved_local_path),
        "source_resume_metadata": source.resume_metadata_summary,
        "dataset_fingerprint_matches_current_workspace": fingerprint_comparison,
        "warnings": warnings,
    }
    return metadata
