"""Prepare a dataset for AudioCraft: split clips into train and valid, write manifests, fingerprint the split."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..audio_files import iter_audio_files, parse_extensions
from ..augmentation import DEFAULT_SPEED_MODE, SPEED_MODES, export_speed_variants, parse_speed_ratios
from ..musicgen.config import (
    FALLBACK_DEFAULT_SPLIT_SEED,
    FALLBACK_DEFAULT_VALID_FRACTION,
    get_repo_root,
    musicgen_default_split_seed,
    musicgen_default_valid_fraction,
)
from ..musicgen.runtime import (
    require_musicgen_python,
)
from ..musicgen.shared import sha256_file, write_json
from ..musicgen.templates import DATASET_CONFIG_NAME
from ..segmentation import compute_silence_fraction
from .clip_stats import (
    clip_stats_for_workspace,
)

DATASET_FINGERPRINT_SCHEMA_VERSION = 1
# Published defaults for the pure split and its help text. The workspace's own configured values
# (musicgen/pyproject.toml [tool.macchina_ricordi.defaults]) are looked up at call time by the
# runtime functions that use them, so a custom checkout is honoured even after this module was
# imported — importing never reads the workspace config.
DEFAULT_VALID_FRACTION = FALLBACK_DEFAULT_VALID_FRACTION
DEFAULT_SPLIT_SEED = FALLBACK_DEFAULT_SPLIT_SEED
DEFAULT_MAX_SILENCE_FRACTION = 0.85
DEFAULT_SILENCE_FRAME_DB = -35.0
_SILENCE_ANALYSIS_SR = 8_000
_SILENCE_FRAME_LENGTH = 4_096
_SILENCE_HOP_LENGTH = 1_024


@dataclass(frozen=True)
class PreparedMusicGenDataset:
    workspace_dir: Path
    train_dir: Path
    valid_dir: Path
    train_manifest: Path
    valid_manifest: Path
    train_count: int
    valid_count: int
    commands: list[list[str]]
    dataset_fingerprint_path: Path
    train_speed_variant_count: int = 0


def resolve_workspace_dir(workspace_dir: str | Path | None) -> Path:
    """A dataset's folder from its path or its name (``bear_bones`` is ``datasets/bear_bones``)."""
    if workspace_dir is None or str(workspace_dir).strip() == "":
        raise ValueError("Pass the dataset to use (--dataset / --workspace-dir): a recipe does not name one.")
    from ..data.datasets import find_dataset_dir

    found = find_dataset_dir(workspace_dir)
    return found if found is not None else Path(workspace_dir).expanduser().resolve()


def workspace_dataset_name(workspace_dir: Path) -> str:
    """The ``egs/<name>/`` a workspace was prepared under; new workspaces all use ``DATASET_CONFIG_NAME``."""
    egs = workspace_dir / "egs"
    names = sorted(
        path.name
        for path in egs.glob("*")
        if (path / "train" / "data.jsonl.gz").exists() and (path / "valid" / "data.jsonl.gz").exists()
    )
    if not names:
        raise FileNotFoundError(
            f"No prepared manifests under {egs}. Run `data prepare <clips> --workspace-dir {workspace_dir}`."
        )
    if len(names) > 1:
        if DATASET_CONFIG_NAME in names:
            return DATASET_CONFIG_NAME
        raise RuntimeError(f"{workspace_dir} holds several prepared datasets ({', '.join(names)}); keep one.")
    return names[0]


def split_audio_files(
    files: Sequence[Path],
    *,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[Path], list[Path]]:
    items = list(files)
    if len(items) < 2:
        raise ValueError("Need at least 2 audio files to create train/valid splits.")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1.")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * valid_fraction))))
    valid_set = {path.resolve() for path in shuffled[:valid_count]}

    train_files = [path for path in items if path.resolve() not in valid_set]
    valid_files = [path for path in items if path.resolve() in valid_set]
    return train_files, valid_files


def build_manifest_command(
    *,
    musicgen_python: str,
    source_dir: Path,
    manifest_path: Path,
) -> list[str]:
    return [
        musicgen_python,
        "-m",
        "audiocraft.data.audio_dataset",
        str(source_dir),
        str(manifest_path),
    ]


def _load_or_build_dataset_fingerprint(
    *,
    workspace_dir: Path,
    dataset_name: str,
) -> dict[str, object]:
    fingerprint_path = workspace_dir / "logs" / "dataset_fingerprint.json"
    if fingerprint_path.exists():
        return _load_json_file(fingerprint_path)

    train_dir = workspace_dir / "data" / "train"
    valid_dir = workspace_dir / "data" / "valid"
    if train_dir.exists() and valid_dir.exists():
        return build_dataset_fingerprint(
            train_dir=train_dir,
            valid_dir=valid_dir,
            dataset_name=dataset_name,
        )

    raise FileNotFoundError(f"Dataset fingerprint not found: {fingerprint_path}. Run `data prepare` first.")


def _dataset_fingerprint_summary(fingerprint: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": fingerprint.get("schema_version"),
        "profile": fingerprint.get("profile"),
        "dataset_name": fingerprint.get("dataset_name"),
        "train_aggregate_sha256": fingerprint.get("train_aggregate_sha256"),
        "valid_aggregate_sha256": fingerprint.get("valid_aggregate_sha256"),
        "train_count": len(fingerprint.get("train_files") or []),
        "valid_count": len(fingerprint.get("valid_files") or []),
    }


def _dataset_fingerprint_digest(fingerprint: dict[str, object]) -> str:
    identity = {
        "profile": fingerprint.get("profile"),
        "dataset_name": fingerprint.get("dataset_name"),
        "train_aggregate_sha256": fingerprint.get("train_aggregate_sha256"),
        "valid_aggregate_sha256": fingerprint.get("valid_aggregate_sha256"),
    }
    # Fingerprints written before recipes existed carry the profile used to prepare the workspace; it stays in
    # their identity so existing auto-labels, and the Dora signatures built on them, do not change.
    missing = [key for key, value in identity.items() if not value and key != "profile"]
    if missing:
        raise RuntimeError(f"Dataset fingerprint is missing required field(s): {', '.join(missing)}.")
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_dataset_fingerprint(
    *,
    train_dir: Path,
    valid_dir: Path,
    dataset_name: str,
    profile_name: str | None = None,
    clip_duration_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    train_files, train_aggregate_sha256 = _fingerprint_prepared_split(train_dir)
    valid_files, valid_aggregate_sha256 = _fingerprint_prepared_split(valid_dir)
    # ``clip_duration_stats`` is additive and is intentionally NOT part of the
    # identity hashed by ``_dataset_fingerprint_digest``, so recording it does not
    # change Dora signatures, auto labels, or checkpoint provenance digests.
    return {
        "schema_version": DATASET_FINGERPRINT_SCHEMA_VERSION,
        "profile": profile_name,
        "dataset_name": dataset_name,
        "train_files": train_files,
        "valid_files": valid_files,
        "train_aggregate_sha256": train_aggregate_sha256,
        "valid_aggregate_sha256": valid_aggregate_sha256,
        "clip_duration_stats": clip_duration_stats,
    }


def find_workspace_manifest(workspace_dir: Path) -> Path | None:
    """Return a workspace's manifest, preferring the train split."""
    egs_dir = workspace_dir / "egs"
    if not egs_dir.is_dir():
        return None
    candidates = sorted(egs_dir.rglob("data.jsonl.gz"))
    if not candidates:
        return None
    for path in candidates:
        if path.parent.name == "train":
            return path
    return candidates[0]


def audit_workspace_slice_lengths(
    *,
    root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """Compute clip-length stats for one or all workspaces under ``root``.

    Reads the existing prepared manifest (the source of truth) so it works
    retroactively over the whole back-catalog without re-preparing anything.
    Rows are sorted by slice length, with unreadable workspaces last.
    """
    if workspace_dir is not None:
        workspaces = [Path(workspace_dir).expanduser().resolve()]
    else:
        base = Path(root).expanduser().resolve() if root is not None else get_repo_root()
        from ..data.datasets import dataset_dirs

        workspaces = dataset_dirs(base)

    rows: list[dict[str, object]] = []
    for workspace in workspaces:
        manifest = find_workspace_manifest(workspace)
        stats = clip_stats_for_workspace(manifest) if manifest is not None else None
        rows.append(
            {
                "workspace": workspace.name,
                "manifest": str(manifest) if manifest is not None else None,
                "stats": stats,
                "safe_segment_duration": (int(stats["slice_length_s"]) if stats is not None else None),
            }
        )
    rows.sort(
        key=lambda row: (
            row["stats"] is None,
            (row["stats"] or {}).get("slice_length_s", 0.0),
        )
    )
    return rows


def _drop_degenerate_manifest_rows(manifest_path: Path) -> int:
    """Remove manifest rows with non-finite or non-positive durations.

    Returns the number of rows dropped; leaves the file untouched (0) when it
    is missing or unreadable (e.g. mocked manifest commands in tests).
    """
    import gzip
    import math

    try:
        with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, EOFError, json.JSONDecodeError):
        return 0

    def healthy(row: dict[str, object]) -> bool:
        duration = row.get("duration")
        return isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0

    kept = [row for row in rows if healthy(row)]
    dropped = len(rows) - len(kept)
    if dropped:
        with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row) + "\n")
    return dropped


def prepare_musicgen_dataset(
    *,
    input_dir: str | Path,
    workspace_dir: str | Path | None = None,
    musicgen_python: str | None = None,
    valid_fraction: float | None = None,
    seed: int | None = None,
    extensions: str | Sequence[str] | None = None,
    max_silence_fraction: float = DEFAULT_MAX_SILENCE_FRACTION,
    silence_frame_db: float = DEFAULT_SILENCE_FRAME_DB,
    speed_ratios: str | Sequence[float] | None = None,
    speed_mode: str = DEFAULT_SPEED_MODE,
    link: bool = False,
    overwrite: bool = False,
) -> PreparedMusicGenDataset:
    """Split ``input_dir`` into ``workspace_dir`` and write AudioCraft manifests.

    ``link`` hard-links the split files instead of copying them (``data build`` does, since its clips sit next
    to the split). A workspace that already holds manifests is refused unless ``overwrite``: preparing wipes
    ``data/``, ``egs/`` and ``logs/``, and ``logs/`` also holds the records of the runs trained on it.
    """
    # Runtime default: honour the checkout's [tool.macchina_ricordi.defaults] when the caller did
    # not pick a value, so a custom checkout applies even after this module was imported.
    if valid_fraction is None:
        valid_fraction = musicgen_default_valid_fraction()
    if seed is None:
        seed = musicgen_default_split_seed()
    input_root = Path(input_dir).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    # Validate before the existing workspace data below gets wiped.
    parsed_speed_ratios = tuple(ratio for ratio in parse_speed_ratios(speed_ratios) if ratio != 1.0)
    if speed_mode not in SPEED_MODES:
        raise ValueError(f"Unknown speed mode {speed_mode!r}; expected one of {SPEED_MODES}")

    workspace = resolve_workspace_dir(workspace_dir)
    if not overwrite and any((workspace / "egs").glob("*/train/data.jsonl.gz")):
        raise FileExistsError(
            f"{workspace} is already prepared; preparing it again would wipe its data, manifests and run records. "
            "Prepare into a new folder (or `data build`), or pass --overwrite if you mean it."
        )
    dataset_name = DATASET_CONFIG_NAME
    musicgen_python = require_musicgen_python(musicgen_python)
    parsed_extensions = parse_extensions(extensions)
    audio_files = sorted(iter_audio_files(input_root, recursive=True, extensions=parsed_extensions))
    train_files, valid_files = split_audio_files(
        audio_files,
        valid_fraction=valid_fraction,
        seed=seed,
    )

    data_dir = workspace / "data"
    egs_dir = workspace / "egs" / dataset_name
    logs_dir = workspace / "logs"
    train_out = data_dir / "train"
    valid_out = data_dir / "valid"

    for path in (train_out, valid_out, egs_dir, logs_dir):
        if path.exists():
            shutil.rmtree(path)

    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)
    (egs_dir / "train").mkdir(parents=True, exist_ok=True)
    (egs_dir / "valid").mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_dropped = _copy_split_files(
        split_files=train_files,
        input_root=input_root,
        output_root=train_out,
        max_silence_fraction=max_silence_fraction,
        silence_frame_db=silence_frame_db,
        link=link,
    )
    valid_dropped = _copy_split_files(
        split_files=valid_files,
        input_root=input_root,
        output_root=valid_out,
        max_silence_fraction=max_silence_fraction,
        silence_frame_db=silence_frame_db,
        link=link,
    )
    # Train only, after the split: augmenting before it would put speed variants
    # of validation clips into train.
    train_speed_variants = _add_speed_variants(
        root=train_out,
        extensions=parsed_extensions,
        ratios=parsed_speed_ratios,
        mode=speed_mode,
    )

    train_manifest = egs_dir / "train" / "data.jsonl.gz"
    valid_manifest = egs_dir / "valid" / "data.jsonl.gz"
    commands = [
        build_manifest_command(
            musicgen_python=musicgen_python,
            source_dir=train_out,
            manifest_path=train_manifest,
        ),
        build_manifest_command(
            musicgen_python=musicgen_python,
            source_dir=valid_out,
            manifest_path=valid_manifest,
        ),
    ]

    for command in commands:
        subprocess.run(command, check=True)

    # Degenerate slicing remainders can land in the manifest with zero or even
    # negative durations (mp3 header artifacts). They poison duration-weighted
    # sampling (`torch.multinomial` rejects non-finite/negative probabilities)
    # in solvers that keep `sample_on_duration=true`, e.g. compression training.
    manifest_dropped = {
        "train": _drop_degenerate_manifest_rows(train_manifest),
        "valid": _drop_degenerate_manifest_rows(valid_manifest),
    }

    # The manifests now carry per-clip durations; summarise them so the workspace
    # *declares* its slice length (S). This is the source of truth the train-time
    # guard validates segment_duration (D) against. May be None if the manifest is
    # unreadable (e.g. mocked subprocess in tests) — callers fail open on None.
    clip_duration_stats = clip_stats_for_workspace(train_manifest)

    prepared = PreparedMusicGenDataset(
        workspace_dir=workspace,
        train_dir=train_out,
        valid_dir=valid_out,
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        train_count=len(train_files) - train_dropped,
        valid_count=len(valid_files) - valid_dropped,
        commands=commands,
        dataset_fingerprint_path=logs_dir / "dataset_fingerprint.json",
        train_speed_variant_count=train_speed_variants,
    )
    write_json(
        logs_dir / "prepare_dataset.json",
        {
            "input_dir": str(input_root),
            "dataset_name": dataset_name,
            "workspace_dir": str(workspace),
            "train_count": prepared.train_count,
            "valid_count": prepared.valid_count,
            "train_dropped_silent": train_dropped,
            "valid_dropped_silent": valid_dropped,
            "speed_ratios": list(parsed_speed_ratios),
            "speed_mode": speed_mode,
            "train_speed_variants": train_speed_variants,
            "max_silence_fraction": max_silence_fraction,
            "silence_frame_db": silence_frame_db,
            "valid_fraction": valid_fraction,
            "seed": seed,
            "commands": commands,
            "clip_duration_stats": clip_duration_stats,
            "manifest_dropped_degenerate": manifest_dropped,
        },
    )
    write_json(
        prepared.dataset_fingerprint_path,
        build_dataset_fingerprint(
            train_dir=train_out,
            valid_dir=valid_out,
            dataset_name=dataset_name,
            clip_duration_stats=clip_duration_stats,
        ),
    )
    return prepared


def _file_is_mostly_silent(
    source_path: Path,
    *,
    max_silence_fraction: float,
    silence_frame_db: float,
) -> bool:
    if max_silence_fraction >= 1.0:
        return False
    try:
        import librosa

        waveform, _ = librosa.load(source_path, sr=_SILENCE_ANALYSIS_SR, mono=True)
    except Exception:
        return False
    fraction = compute_silence_fraction(
        waveform,
        frame_length=_SILENCE_FRAME_LENGTH,
        hop_length=_SILENCE_HOP_LENGTH,
        frame_threshold_db=silence_frame_db,
    )
    return fraction > max_silence_fraction


def _copy_split_files(
    *,
    split_files: Sequence[Path],
    input_root: Path,
    output_root: Path,
    max_silence_fraction: float = DEFAULT_MAX_SILENCE_FRACTION,
    silence_frame_db: float = DEFAULT_SILENCE_FRAME_DB,
    link: bool = False,
) -> int:
    dropped = 0
    for source_path in split_files:
        if _file_is_mostly_silent(
            source_path,
            max_silence_fraction=max_silence_fraction,
            silence_frame_db=silence_frame_db,
        ):
            dropped += 1
            continue

        relative_path = source_path.relative_to(input_root)
        destination_path = output_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if link:
            try:
                os.link(source_path, destination_path)
                continue
            except OSError:
                pass  # another filesystem: copy instead
        shutil.copy2(source_path, destination_path)
    return dropped


def _add_speed_variants(
    *,
    root: Path,
    extensions: tuple[str, ...],
    ratios: Sequence[float],
    mode: str,
) -> int:
    if not ratios:
        return 0
    sources = sorted(iter_audio_files(root, recursive=True, extensions=extensions))
    written = 0
    for source_path in sources:
        written += len(export_speed_variants(source_path, source_path.parent, ratios, mode=mode, skip_identity=True))
    return written


def _fingerprint_prepared_split(root: Path) -> tuple[list[dict[str, object]], str]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Prepared dataset directory does not exist: {root}")

    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    entries: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for file_path in files:
        relative_path = file_path.relative_to(root).as_posix()
        size_bytes = file_path.stat().st_size
        sha256 = sha256_file(file_path)
        entries.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size_bytes).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(sha256.encode("utf-8"))
        aggregate.update(b"\n")
    return entries, aggregate.hexdigest()


def _load_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
