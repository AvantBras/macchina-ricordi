"""Cluster-based dataset rebalancing for MusicGen training manifests.

Unconditional models trained on the mega dataset collapse entire acoustic
modes ("wells"): clusters covering ~20% of the data receive ~0% of
generations while the dominant percussive cluster is over-produced 2-3x
(measured 2026-06-08 on uncond sweeps of mega-mid/deep vs mega_10s).

AudioCraft's ``AudioDataset`` draws each file with probability proportional to
its ``weight`` when ``sample_on_weight`` is on (times its duration when
``sample_on_duration`` is on too). MusicGen's base config turns both off;
``train run --weights <name>`` turns weight sampling back on for the train split
and points it at an overlay: a weighted copy of the workspace's train manifest
under ``artifacts/weights/<workspace>/<name>/``. The workspace itself is never
written, so one workspace can carry several weightings side by side.

Two weighting modes:

- **inverse-share** (no gens needed): flatten the cluster prior,
  ``w_c ∝ (mean_share / share_c) ** alpha``.
- **coverage** (gens-informed): upweight what the model under-produces,
  ``w_c ∝ (data_share_c / gen_share_c) ** alpha`` with Laplace smoothing on
  the generation histogram.

Both are clipped to ``[1/cap, cap]`` and normalised so the mean weight over
segments is 1.0 (epoch scale is preserved). The overlay's ``overlay.json``
records the full decision; a run trained with it gets the auto-label suffix
``_rw<digest>``, so a fresh Dora signature instead of colliding with the
unweighted XP. Workspaces weighted in place before overlays existed (a
``rebalance.json`` next to the manifest) still train as they did, and
``data rebalance --reset`` clears them.

A folder mix (``plan_folder_mix``) fills an overlay without clustering: each source folder gets
the share of draws asked for, and its weights are not clipped, since the share is the point.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..musicgen.shared import sha256_file, utc_now_iso, write_text_atomic

REBALANCE_SIDECAR_FILENAME = "rebalance.json"
DEFAULT_K = 10
DEFAULT_PCA_DIMS = 24
DEFAULT_ALPHA = 0.5
# Validated 2026-06-08 (deep@90 +15ep, XP ae9b572d): alpha=1.0 reopened the
# collapsed clusters (0% -> 10% gen mass) but overshot — previously-covered
# clusters got suppressed and TV-to-data worsened 0.31 -> 0.42. Coverage
# repair should iterate gently (re-derive from a fresh sweep each round).
DEFAULT_COVERAGE_ALPHA = 0.5
DEFAULT_CAP = 4.0
FEATURE_SR = 16_000
_MIN_FEATURE_SECONDS = 1.0


# ── feature extraction ────────────────────────────────────────────────


def _feature_one(path: str) -> np.ndarray | None:
    """Timbral feature vector for one audio file (run in a worker process)."""
    import librosa

    from ..scoring.similarity import _mfcc_stats_vector, _spectral_stats_vector

    try:
        waveform, _ = librosa.load(path, sr=FEATURE_SR, mono=True)
        if waveform.size < int(_MIN_FEATURE_SECONDS * FEATURE_SR):
            return None
        return np.concatenate(
            [
                _mfcc_stats_vector(waveform, sr=FEATURE_SR),
                _spectral_stats_vector(waveform, sr=FEATURE_SR),
            ]
        ).astype(np.float32)
    except Exception:
        return None


def extract_segment_features(
    paths: Sequence[str],
    *,
    max_workers: int | None = None,
    feature_fn: Callable[[str], np.ndarray | None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract feature vectors for ``paths``; silently drops unreadable/short files.

    Returns ``(features, kept_paths)`` with one row per kept path. ``progress(done, total)``
    is called as files are read.
    """
    fn = feature_fn or _feature_one
    feats: list[np.ndarray] = []
    kept: list[str] = []
    results: list[np.ndarray | None] = []
    if feature_fn is not None:
        for path in paths:
            results.append(fn(path))
            if progress is not None:
                progress(len(results), len(paths))
    else:
        workers = max_workers or max(1, (os.cpu_count() or 4) - 4)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(fn, paths, chunksize=8):
                results.append(result)
                if progress is not None and (len(results) % 32 == 0 or len(results) == len(paths)):
                    progress(len(results), len(paths))
    for path, feat in zip(paths, results, strict=True):
        if feat is not None:
            feats.append(feat)
            kept.append(path)
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(feats), kept


# ── clustering ────────────────────────────────────────────────────────


@dataclass
class ClusterModel:
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray  # (pca_dims, n_features)
    centroids: np.ndarray  # (k, pca_dims)

    @property
    def k(self) -> int:
        return int(self.centroids.shape[0])

    def transform(self, features: np.ndarray) -> np.ndarray:
        z = (features.astype(np.float64) - self.mean) / self.std
        return z @ self.components.T

    def predict(self, features: np.ndarray) -> np.ndarray:
        projected = self.transform(features)
        dists = np.linalg.norm(projected[:, None, :] - self.centroids[None, :, :], axis=2)
        return dists.argmin(axis=1)


def fit_clusters(
    features: np.ndarray,
    *,
    k: int = DEFAULT_K,
    pca_dims: int = DEFAULT_PCA_DIMS,
    seed: int = 0,
) -> tuple[ClusterModel, np.ndarray]:
    """Z-score + PCA + KMeans. Returns the fitted model and train labels."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = features.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    z = (X - mean) / std
    dims = min(pca_dims, z.shape[0], z.shape[1])
    pca = PCA(n_components=dims, random_state=seed).fit(z)
    projected = pca.transform(z)
    kmeans = KMeans(n_clusters=min(k, len(X)), n_init=10, random_state=seed).fit(projected)
    model = ClusterModel(
        mean=mean,
        std=std,
        components=pca.components_,
        centroids=kmeans.cluster_centers_,
    )
    return model, kmeans.labels_.astype(int)


# ── weighting ─────────────────────────────────────────────────────────


def _clip_and_normalise(raw: np.ndarray, labels: np.ndarray, *, cap: float) -> np.ndarray:
    weights = np.clip(raw, 1.0 / cap, cap)
    # normalise so the mean weight over *segments* (not clusters) is 1.0
    per_segment = weights[labels]
    weights = weights / per_segment.mean()
    return weights


def inverse_share_weights(
    labels: np.ndarray,
    k: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    cap: float = DEFAULT_CAP,
) -> np.ndarray:
    """Per-cluster weights flattening the cluster prior toward uniform."""
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    shares = counts / counts.sum()
    mean_share = 1.0 / k
    with np.errstate(divide="ignore"):
        raw = np.where(shares > 0, (mean_share / shares) ** alpha, cap)
    return _clip_and_normalise(raw, labels, cap=cap)


def coverage_weights(
    labels_data: np.ndarray,
    labels_gens: np.ndarray,
    k: int,
    *,
    alpha: float = DEFAULT_COVERAGE_ALPHA,
    cap: float = DEFAULT_CAP,
) -> np.ndarray:
    """Per-cluster weights upweighting clusters the model under-produces.

    The generation histogram is Laplace-smoothed (add-one) so a cluster with
    zero generations gets a large-but-finite ratio instead of infinity.
    """
    data_counts = np.bincount(labels_data, minlength=k).astype(np.float64)
    data_shares = data_counts / data_counts.sum()
    gen_counts = np.bincount(labels_gens, minlength=k).astype(np.float64)
    gen_shares = (gen_counts + 1.0) / (gen_counts.sum() + k)
    raw = (data_shares / gen_shares) ** alpha
    return _clip_and_normalise(raw, labels_data, cap=cap)


# ── manifest I/O ──────────────────────────────────────────────────────


def read_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_manifest_rows(manifest_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def clear_manifest_weights(manifest_path: Path) -> None:
    """Reset all weights to ``None`` and remove the rebalance sidecar."""
    rows = read_manifest_rows(manifest_path)
    for row in rows:
        row["weight"] = None
    _write_manifest_rows(manifest_path, rows)
    sidecar = manifest_path.parent / REBALANCE_SIDECAR_FILENAME
    sidecar.unlink(missing_ok=True)


def weights_digest(weights_by_path: Mapping[str, float]) -> str:
    """Stable 8-hex digest of the path->weight assignment."""
    digest = hashlib.sha256()
    for path in sorted(weights_by_path):
        digest.update(f"{path}\t{weights_by_path[path]:.6f}\n".encode())
    return digest.hexdigest()[:8]


def read_rebalance_sidecar(train_dir: Path) -> dict[str, Any] | None:
    sidecar = train_dir / REBALANCE_SIDECAR_FILENAME
    if not sidecar.exists():
        return None
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def rebalance_label_suffix(sidecar: Mapping[str, Any] | None) -> str:
    """Auto-dataset-label suffix for a rebalanced manifest (`''` when absent)."""
    if not sidecar:
        return ""
    digest = str(sidecar.get("weights_digest") or "")[:8]
    return f"_rw{digest}" if digest else ""


# ── high-level workflow ───────────────────────────────────────────────


def _cluster_examples(model: ClusterModel, features: np.ndarray, labels: np.ndarray, paths: Sequence[str], n: int):
    """Per cluster, the ``n`` segments nearest its centroid: what the cluster sounds like."""
    projected = model.transform(features)
    examples: list[list[str]] = []
    for c in range(model.k):
        members = np.flatnonzero(labels == c)
        dists = np.linalg.norm(projected[members] - model.centroids[c], axis=1)
        examples.append([paths[i] for i in members[np.argsort(dists)[:n]]])
    return examples


def _cluster_table(
    labels: np.ndarray,
    cluster_weights_arr: np.ndarray,
    paths: Sequence[str],
    gen_hist: np.ndarray | None,
    examples: Sequence[Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    k = len(cluster_weights_arr)
    counts = np.bincount(labels, minlength=k)
    total = max(int(counts.sum()), 1)
    table: list[dict[str, Any]] = []
    for c in range(k):
        member_names = [Path(paths[i]).name for i in np.flatnonzero(labels == c)]
        groups: dict[str, int] = {}
        for name in member_names:
            group = name.split("__")[0]
            groups[group] = groups.get(group, 0) + 1
        top_groups = sorted(groups.items(), key=lambda kv: -kv[1])[:3]
        entry: dict[str, Any] = {
            "cluster": c,
            "segments": int(counts[c]),
            "share": round(float(counts[c]) / total, 4),
            "weight": round(float(cluster_weights_arr[c]), 4),
            "top_groups": [f"{g}:{n}" for g, n in top_groups],
        }
        if gen_hist is not None:
            entry["gen_share"] = round(float(gen_hist[c]), 4)
        if examples is not None:
            entry["examples"] = list(examples[c])
        table.append(entry)
    return table


@dataclass
class RebalancePlan:
    """Computed weights, not yet written. ``apply_rebalance`` writes them."""

    manifest_path: Path
    mode: str
    k: int | None
    alpha: float | None
    cap: float | None
    seed: int
    gens_dirs: list[str]
    weights_by_path: dict[str, float]
    weights_digest: str
    cluster_table: list[dict[str, Any]]
    manifest_signature: tuple[int, float]
    segments: int = 0
    readable_segments: int = 0
    generations: int = 0
    mix: dict[str, float] = field(default_factory=dict)

    @property
    def train_dir(self) -> Path:
        return self.manifest_path.parent


def _manifest_signature(manifest_path: Path) -> tuple[int, float]:
    stat = manifest_path.stat()
    return stat.st_size, stat.st_mtime


def plan_rebalance(
    *,
    workspace_dir: Path,
    dataset_name: str,
    gens_dirs: Sequence[Path] | None = None,
    k: int = DEFAULT_K,
    alpha: float | None = None,
    cap: float = DEFAULT_CAP,
    seed: int = 0,
    max_workers: int | None = None,
    feature_fn: Callable[[str], np.ndarray | None] | None = None,
    examples_per_cluster: int = 4,
    progress: Callable[[str, int, int], None] | None = None,
) -> RebalancePlan:
    """Cluster the train manifest and compute per-segment sampling weights, writing nothing.

    With ``gens_dirs`` the coverage mode is used (upweight clusters the model
    under-produces, judged from the generated WAVs found there); otherwise the
    inverse-share mode flattens the cluster prior. ``progress(stage, done, total)``
    reports feature extraction, stage ``"data"`` then ``"generations"``.
    """
    train_dir = workspace_dir / "egs" / dataset_name / "train"
    manifest_path = train_dir / "data.jsonl.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {manifest_path}")
    signature = _manifest_signature(manifest_path)

    def stage(name: str) -> Callable[[int, int], None] | None:
        return None if progress is None else (lambda done, total: progress(name, done, total))

    rows = read_manifest_rows(manifest_path)
    paths = [str(row["path"]) for row in rows]
    features, kept_paths = extract_segment_features(
        paths, max_workers=max_workers, feature_fn=feature_fn, progress=stage("data")
    )
    if len(kept_paths) < k:
        raise RuntimeError(
            f"Only {len(kept_paths)} readable segments for k={k} clusters; lower --k or check the manifest."
        )
    model, labels = fit_clusters(features, k=k, seed=seed)
    effective_k = model.k

    gen_hist: np.ndarray | None = None
    generations = 0
    if gens_dirs:
        gen_paths = sorted(str(p) for d in gens_dirs for p in Path(d).rglob("*.wav"))
        if not gen_paths:
            raise RuntimeError(f"No .wav files found under {[str(d) for d in gens_dirs]}")
        gen_features, _ = extract_segment_features(
            gen_paths, max_workers=max_workers, feature_fn=feature_fn, progress=stage("generations")
        )
        if len(gen_features) == 0:
            raise RuntimeError("None of the generation WAVs were readable/long enough.")
        generations = len(gen_features)
        labels_gens = model.predict(gen_features)
        mode = "coverage"
        effective_alpha = DEFAULT_COVERAGE_ALPHA if alpha is None else alpha
        cluster_w = coverage_weights(labels, labels_gens, effective_k, alpha=effective_alpha, cap=cap)
        gen_hist = np.bincount(labels_gens, minlength=effective_k).astype(np.float64)
        gen_hist /= max(gen_hist.sum(), 1.0)
    else:
        mode = "inverse-share"
        effective_alpha = DEFAULT_ALPHA if alpha is None else alpha
        cluster_w = inverse_share_weights(labels, effective_k, alpha=effective_alpha, cap=cap)

    weights_by_path = {path: float(cluster_w[label]) for path, label in zip(kept_paths, labels, strict=True)}
    examples = _cluster_examples(model, features, labels, kept_paths, examples_per_cluster)
    return RebalancePlan(
        manifest_path=manifest_path,
        mode=mode,
        k=effective_k,
        alpha=effective_alpha,
        cap=cap,
        seed=seed,
        gens_dirs=[str(d) for d in (gens_dirs or [])],
        weights_by_path=weights_by_path,
        weights_digest=weights_digest(weights_by_path),
        cluster_table=_cluster_table(labels, cluster_w, kept_paths, gen_hist, examples),
        manifest_signature=signature,
        segments=len(paths),
        readable_segments=len(kept_paths),
        generations=generations,
    )


# ── folder mix ────────────────────────────────────────────────────────

FOLDER_MIX_MODE = "folder-mix"


def _source_folder(path: str) -> str:
    # shortcut: folder read from the clip name, as the cluster table does; upgrade if names stop carrying it
    name = Path(path).name
    if "__" not in name:
        raise ValueError(f"{path}: no 'folder__' prefix in the clip name, so its source folder is unknown")
    return name.split("__", 1)[0]


def plan_folder_mix(manifest_path: Path, mix: Mapping[str, float]) -> RebalancePlan:
    """Per-clip weights giving each source folder the share of training draws that ``mix`` asks for.

    A clip's draw chance is its weight (duration sampling is off unless asked for), so a folder's share is
    the sum of its clips' weights. Each clip gets ``segments * share / clips_in_folder``, which averages 1
    over the train split like the cluster weights do. Nothing is clipped: the share is what was asked for.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {manifest_path}")
    if not mix or any(not math.isfinite(share) or share < 0 for share in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("each folder needs a share of 0 or more, and at least one share must be above 0")
    rows = read_manifest_rows(manifest_path)
    paths = [str(row["path"]) for row in rows]
    folders = [_source_folder(path) for path in paths]
    counts = Counter(folders)
    unlisted = sorted(set(counts) - set(mix))
    if unlisted:
        raise ValueError(f"folders in the train split but not in the mix: {', '.join(unlisted)}")
    empty = sorted(folder for folder in mix if not counts[folder])
    if empty:
        raise ValueError(f"no train clips from: {', '.join(empty)}")
    total = sum(mix.values())
    per_clip = {folder: len(rows) * (share / total) / counts[folder] for folder, share in mix.items()}
    weights_by_path = {path: per_clip[folder] for path, folder in zip(paths, folders, strict=True)}
    table = [
        {
            "folder": folder,
            "segments": counts[folder],
            "share": round(share / total, 4),
            "weight": round(per_clip[folder], 4),
        }
        for folder, share in sorted(mix.items())
    ]
    return RebalancePlan(
        manifest_path=manifest_path,
        mode=FOLDER_MIX_MODE,
        k=None,
        alpha=None,
        cap=None,
        seed=0,
        gens_dirs=[],
        weights_by_path=weights_by_path,
        weights_digest=weights_digest(weights_by_path),
        cluster_table=table,
        manifest_signature=_manifest_signature(manifest_path),
        segments=len(rows),
        readable_segments=len(rows),
        mix=dict(mix),
    )


REBALANCE_PLAN_SCHEMA = "macchina_ricordi.rebalance_plan.v1"


def save_rebalance_plan(plan: RebalancePlan, path: Path) -> Path:
    """Write computed weights to a file, to look at or apply later (``--apply-plan``)."""
    payload = {
        "schema": REBALANCE_PLAN_SCHEMA,
        "written_at": utc_now_iso(),
        "manifest_path": str(plan.manifest_path),
        "mode": plan.mode,
        "k": plan.k,
        "alpha": plan.alpha,
        "cap": plan.cap,
        "seed": plan.seed,
        "gens_dirs": plan.gens_dirs,
        "weights_digest": plan.weights_digest,
        "cluster_table": plan.cluster_table,
        "manifest_signature": list(plan.manifest_signature),
        "segments": plan.segments,
        "readable_segments": plan.readable_segments,
        "generations": plan.generations,
        "weights_by_path": plan.weights_by_path,
    }
    write_text_atomic(path, json.dumps(payload) + "\n")
    return path


def load_rebalance_plan(path: Path) -> RebalancePlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REBALANCE_PLAN_SCHEMA:
        raise ValueError(f"{path} is not a rebalance plan")
    size, mtime = payload["manifest_signature"]
    return RebalancePlan(
        manifest_path=Path(payload["manifest_path"]),
        mode=payload["mode"],
        k=int(payload["k"]),
        alpha=float(payload["alpha"]),
        cap=float(payload["cap"]),
        seed=int(payload["seed"]),
        gens_dirs=list(payload.get("gens_dirs") or []),
        weights_by_path={str(k): float(v) for k, v in payload["weights_by_path"].items()},
        weights_digest=payload["weights_digest"],
        cluster_table=list(payload["cluster_table"]),
        manifest_signature=(int(size), float(mtime)),
        segments=int(payload.get("segments") or 0),
        readable_segments=int(payload.get("readable_segments") or 0),
        generations=int(payload.get("generations") or 0),
    )


# ── overlays ──────────────────────────────────────────────────────────

WEIGHTS_ROOT = Path("artifacts") / "weights"
OVERLAY_FILENAME = "overlay.json"
OVERLAY_SCHEMA = "macchina_ricordi.weights_overlay.v1"
_OVERLAY_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def _check_overlay_name(name: str) -> None:
    if not _OVERLAY_NAME.match(name):
        raise ValueError(f"invalid weights name {name!r}: lowercase letters, digits, '.', '_' or '-'")


@dataclass(frozen=True)
class WeightsOverlay:
    """Named sampling weights for one workspace's train split, kept outside the workspace."""

    name: str
    folder: Path
    workspace: str
    dataset_name: str
    source_manifest: Path
    source_sha256: str
    weights_digest: str
    mode: str
    k: int | None
    alpha: float | None
    cap: float | None
    seed: int
    gens_dirs: tuple[str, ...]
    written_at: str
    cluster_table: tuple[dict[str, Any], ...] = ()
    mix: dict[str, float] = field(default_factory=dict)

    @property
    def train_dir(self) -> Path:
        return self.folder / "train"

    @property
    def label_suffix(self) -> str:
        return f"_rw{self.weights_digest[:8]}"

    def is_stale(self) -> bool:
        """True when the workspace manifest these weights were computed for has changed since."""
        return not self.source_manifest.exists() or sha256_file(self.source_manifest) != self.source_sha256

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "folder": str(self.folder),
            "weights_digest": self.weights_digest,
            "mode": self.mode,
            "k": self.k,
            "alpha": self.alpha,
            "cap": self.cap,
            "gens_dirs": list(self.gens_dirs),
            "mix": dict(self.mix),
            "written_at": self.written_at,
        }


def weights_root(repo_root: Path, workspace_name: str) -> Path:
    return Path(repo_root) / WEIGHTS_ROOT / workspace_name


def save_weights_overlay(plan: RebalancePlan, *, name: str, repo_root: Path) -> WeightsOverlay:
    """Write a plan as a named overlay; refused if the name is taken or the manifest changed since planning."""
    _check_overlay_name(name)
    if _manifest_signature(plan.manifest_path) != plan.manifest_signature:
        raise RuntimeError(f"{plan.manifest_path} changed since the weights were computed; compute them again")
    train_dir = plan.manifest_path.parent
    dataset_name = train_dir.parent.name
    workspace = train_dir.parents[2]
    folder = weights_root(repo_root, workspace.name) / name
    if folder.exists():
        raise FileExistsError(f"weights {name!r} already exist for {workspace.name}; pick another name or delete them")
    rows = read_manifest_rows(plan.manifest_path)
    for row in rows:
        weight = plan.weights_by_path.get(str(row.get("path")))
        row["weight"] = round(float(weight), 6) if weight is not None else None
    (folder / "train").mkdir(parents=True)
    _write_manifest_rows(folder / "train" / "data.jsonl.gz", rows)
    overlay = WeightsOverlay(
        name=name,
        folder=folder.resolve(),
        workspace=workspace.name,
        dataset_name=dataset_name,
        source_manifest=plan.manifest_path.resolve(),
        source_sha256=sha256_file(plan.manifest_path),
        weights_digest=plan.weights_digest,
        mode=plan.mode,
        k=plan.k,
        alpha=plan.alpha,
        cap=plan.cap,
        seed=plan.seed,
        gens_dirs=tuple(plan.gens_dirs),
        written_at=utc_now_iso(),
        cluster_table=tuple(plan.cluster_table),
        mix=dict(plan.mix),
    )
    payload = {
        "schema": OVERLAY_SCHEMA,
        **overlay.summary(),
        "workspace": overlay.workspace,
        "dataset_name": dataset_name,
        "source_manifest": str(overlay.source_manifest),
        "source_sha256": overlay.source_sha256,
        "seed": plan.seed,
        "segments": plan.segments,
        "readable_segments": plan.readable_segments,
        "generations": plan.generations,
        "cluster_table": plan.cluster_table,
    }
    payload.pop("folder")
    (folder / OVERLAY_FILENAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return overlay


def load_weights_overlay(repo_root: Path, workspace_name: str, name: str) -> WeightsOverlay:
    _check_overlay_name(name)
    folder = weights_root(repo_root, workspace_name) / name
    path = folder / OVERLAY_FILENAME
    if not path.is_file():
        available = ", ".join(o.name for o in list_weights_overlays(repo_root, workspace_name)) or "none"
        raise FileNotFoundError(f"no weights {name!r} for {workspace_name} (saved: {available})")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != OVERLAY_SCHEMA:
        raise ValueError(f"{path} is not a weights overlay")
    return WeightsOverlay(
        name=name,
        folder=folder.resolve(),
        workspace=str(payload["workspace"]),
        dataset_name=str(payload["dataset_name"]),
        source_manifest=Path(payload["source_manifest"]),
        source_sha256=str(payload["source_sha256"]),
        weights_digest=str(payload["weights_digest"]),
        mode=str(payload["mode"]),
        k=payload.get("k"),
        alpha=payload.get("alpha"),
        cap=payload.get("cap"),
        seed=int(payload.get("seed") or 0),
        gens_dirs=tuple(payload.get("gens_dirs") or ()),
        written_at=str(payload.get("written_at") or ""),
        cluster_table=tuple(payload.get("cluster_table") or ()),
        mix=dict(payload.get("mix") or {}),
    )


def list_weights_overlays(repo_root: Path, workspace_name: str) -> list[WeightsOverlay]:
    root = weights_root(repo_root, workspace_name)
    if not root.is_dir():
        return []
    overlays = []
    for folder in sorted(root.iterdir()):
        if (folder / OVERLAY_FILENAME).is_file():
            try:
                overlays.append(load_weights_overlay(repo_root, workspace_name, folder.name))
            except (OSError, ValueError, KeyError):
                continue
    return overlays


def delete_weights_overlay(repo_root: Path, workspace_name: str, name: str) -> Path:
    overlay = load_weights_overlay(repo_root, workspace_name, name)
    shutil.rmtree(overlay.folder)
    return overlay.folder
