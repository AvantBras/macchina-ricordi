from __future__ import annotations

# Perceptual audio description via rule-based lexical mapping.
#
# Thresholds
# ----------
# The module ships with default thresholds tuned for general audio
# (speech, music, environmental sounds).  For a specific corpus (e.g.
# percussive free-jazz segments), thresholds should be calibrated:
#
#     auto_calibrate("library/segmented_tracks/JDR-001")
#     # Or via CLI: macchina_ricordi listen describe-calibrate <dir>
#
# After calibration, thresholds are saved to
# ~/.macchina_ricordi/describe_thresholds.json.
#
# Script usage (outside the CLI) must load thresholds explicitly:
#
#     from macchina_ricordi.listening.describe import load_default_thresholds
#     load_default_thresholds()
#
# The CLI commands (listen describe, diff, compare, describe-dir) all
# call load_default_thresholds() automatically.  Direct use of
# describe_audio or describe_waveform does not — the caller must
# load explicitly if calibration is desired.
from collections import namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..scoring.features import AudioFeatureSummary, summarize_waveform

# ── dimension definition ──────────────────────────────────────────────

_Dimension = namedtuple("_Dimension", ["name", "labels", "thresholds", "extractor"])


def _extract_loudness(features: AudioFeatureSummary) -> float:
    return features.rms_mean


def _extract_brightness(features: AudioFeatureSummary) -> float:
    return features.spectral_centroid_mean


def _extract_attack(features: AudioFeatureSummary) -> float:
    return features.attack_db


def _extract_noisiness(features: AudioFeatureSummary) -> float:
    return features.spectral_flatness_mean


def _extract_density(features: AudioFeatureSummary) -> float:
    return features.onset_density


def _extract_pulse(features: AudioFeatureSummary) -> float:
    return features.pulse_instability


def _extract_resonance(features: AudioFeatureSummary) -> float:
    return features.resonance_ratio


_DIMENSIONS: list[_Dimension] = [
    _Dimension(
        "loudness",
        ["whisper", "hushed", "moderate", "present", "forceful", "thunderous"],
        [0.01, 0.05, 0.12, 0.25, 0.50],
        _extract_loudness,
    ),
    _Dimension(
        "brightness",
        ["hollow", "dark", "warm", "mid-bright", "bright", "piercing", "glassy"],
        [300.0, 800.0, 1500.0, 3000.0, 5000.0],
        _extract_brightness,
    ),
    _Dimension(
        "attack",
        ["pillowy", "soft-edged", "round", "crisp", "sharp", "needle-like"],
        [1.5, 2.0, 2.8, 4.0, 7.0],
        _extract_attack,
    ),
    _Dimension(
        "noisiness",
        ["pure-tonal", "clean", "textured", "noisy", "rough", "chaotic"],
        [0.05, 0.12, 0.25, 0.45, 0.70],
        _extract_noisiness,
    ),
    _Dimension(
        "density", ["sparse", "airy", "moderate", "dense", "saturated"], [1.0, 3.0, 6.0, 12.0], _extract_density
    ),
    _Dimension(
        "pulse",
        ["still", "drifting", "limping", "unstable", "walking", "machine-like"],
        [0.05, 0.15, 0.30, 0.55, 0.85],
        _extract_pulse,
    ),
    _Dimension(
        "resonance",
        ["anechoic", "dry", "room", "resonant", "ringing", "metallic"],
        [0.05, 0.15, 0.30, 0.50, 0.75],
        _extract_resonance,
    ),
]

_DEFAULT_THRESHOLDS: dict[str, list[float]] = {dim.name: list(dim.thresholds) for dim in _DIMENSIONS}


def calibrate_thresholds_from_dir(
    directory: str | Path,
    *,
    sr: int = 16_000,
    max_files: int = 100,
) -> dict[str, list[float]]:
    """Compute quantile-based thresholds from a corpus of audio files.

    Returns a dict mapping dimension name → list of thresholds suitable
    for replacing the defaults.  Each threshold list has length ``n_labels - 1``
    and is derived from equal-probability quantiles of the corpus distribution.
    """
    from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files

    directory = Path(directory).expanduser().resolve()
    files = list(iter_audio_files(directory, recursive=True, extensions=DEFAULT_EXTENSIONS))
    if max_files and len(files) > max_files:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(files), size=max_files, replace=False)
        files = [files[i] for i in indices]

    accum: dict[str, list[float]] = {d.name: [] for d in _DIMENSIONS}
    for path in files:
        try:
            waveform, _ = librosa.load(path, sr=sr, mono=True)
            if waveform.size < sr:
                continue
            features = summarize_waveform(waveform, sr=sr)
            for dim in _DIMENSIONS:
                accum[dim.name].append(dim.extractor(features))
        except Exception:
            continue

    thresholds: dict[str, list[float]] = {}
    for dim in _DIMENSIONS:
        vals = sorted(accum[dim.name])
        n = len(vals)
        n_labels = len(dim.labels)
        if n < n_labels:
            thresholds[dim.name] = list(dim.thresholds)
            continue
        thresholds[dim.name] = [vals[int(n * (i + 1) / n_labels)] for i in range(n_labels - 1)]
    return thresholds


def apply_thresholds(thresholds: dict[str, list[float]]) -> None:
    """Replace the threshold tables of all dimensions in-place.

    Call after ``calibrate_thresholds_from_dir`` to install quantile-based
    thresholds for the current session.
    """
    for i, dim in enumerate(_DIMENSIONS):
        if dim.name in thresholds:
            _DIMENSIONS[i] = dim._replace(thresholds=list(thresholds[dim.name]))


def save_thresholds(path: str | Path) -> None:
    """Persist the current threshold tables to a JSON file."""
    import json

    data = {dim.name: list(dim.thresholds) for dim in _DIMENSIONS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def load_thresholds(path: str | Path) -> bool:
    """Load threshold tables from a JSON file. Returns True on success."""
    import json

    try:
        data = json.loads(Path(path).read_text())
        return _try_apply_thresholds(data)
    except (OSError, json.JSONDecodeError):
        return False


def _try_apply_thresholds(data: dict[str, Any]) -> bool:
    """Apply threshold data if it looks valid. Returns True on success."""
    if not isinstance(data, dict):
        return False

    parsed: dict[str, list[float]] = {}
    for dim in _DIMENSIONS:
        values = data.get(dim.name)
        if not isinstance(values, list) or len(values) != len(dim.labels) - 1:
            return False
        try:
            parsed[dim.name] = [float(value) for value in values]
        except (TypeError, ValueError):
            return False

    apply_thresholds(parsed)
    return True


def thresholds_loaded() -> bool:
    """Check whether non-default thresholds are active.

    Heuristic: if any dimension's thresholds differ from the module defaults,
    calibration has been applied.
    """
    return any(list(dim.thresholds) != _DEFAULT_THRESHOLDS[dim.name] for dim in _DIMENSIONS)


_AUTO_CALIB_PATH = Path.home() / ".macchina_ricordi" / "describe_thresholds.json"


def load_default_thresholds() -> bool:
    """Load thresholds from the default per-user location when available."""
    if thresholds_loaded():
        return True
    if not _AUTO_CALIB_PATH.exists():
        return False
    return load_thresholds(_AUTO_CALIB_PATH)


def auto_calibrate(directory: str | Path, *, sr: int = 16_000, max_files: int = 100) -> dict[str, list[float]]:
    """Calibrate, save to default location, and apply. One-shot convenience."""
    th = calibrate_thresholds_from_dir(directory, sr=sr, max_files=max_files)
    apply_thresholds(th)
    save_thresholds(_AUTO_CALIB_PATH)
    return th


def _lexicalize(value: float, dim: _Dimension) -> str:
    """Map a continuous feature value to the highest-index label whose threshold is <= value."""
    for i, threshold in enumerate(dim.thresholds):
        if value < threshold:
            return dim.labels[i]
    return dim.labels[-1]


# ── public dataclass ──────────────────────────────────────────────────


@dataclass(frozen=True)
class AudioDescription:
    loudness: str
    brightness: str
    attack: str
    noisiness: str
    density: str
    pulse: str
    resonance: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_sentence(self) -> str:
        """Render as a readable English sentence."""
        return (
            f"A {self.loudness}, {self.brightness} sound with "
            f"{self.attack} attack, {self.noisiness} texture, "
            f"{self.density} events, {self.pulse} pulse, "
            f"and {self.resonance} resonance."
        )

    @classmethod
    def fieldnames(cls) -> list[str]:
        return ["loudness", "brightness", "attack", "noisiness", "density", "pulse", "resonance"]


# ── public API ────────────────────────────────────────────────────────


def describe_waveform(waveform: np.ndarray, *, sr: int = 16_000) -> AudioDescription:
    """Extract perceptual labels from a 1-D mono waveform."""
    features = summarize_waveform(waveform, sr=sr)
    return describe_feature_summary(features)


def describe_feature_summary(features: AudioFeatureSummary) -> AudioDescription:
    """Map an already-computed feature summary to perceptual labels."""
    if features.rms_mean < 1e-6:
        return AudioDescription(
            loudness="whisper",
            brightness="hollow",
            attack="pillowy",
            noisiness="pure-tonal",
            density="sparse",
            pulse="still",
            resonance="anechoic",
        )
    labels: dict[str, str] = {}
    for dim in _DIMENSIONS:
        value = dim.extractor(features)
        labels[dim.name] = _lexicalize(value, dim)
    return AudioDescription(**labels)


def describe_audio(
    path: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    res_type: str = "soxr_qq",
) -> AudioDescription:
    """Load an audio file and produce a perceptual description."""
    waveform, loaded_sr = librosa.load(path, sr=sr, mono=mono, res_type=res_type)
    return describe_waveform(waveform, sr=int(loaded_sr))


def diff_waveform(
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    *,
    sr: int = 16_000,
) -> str:
    """Describe what changed between two waveforms in natural language."""
    desc_a = describe_waveform(waveform_a, sr=sr)
    desc_b = describe_waveform(waveform_b, sr=sr)

    changes: list[tuple[str, str, str]] = []  # (dim_name, label_a, label_b)
    for dim in _DIMENSIONS:
        label_a = getattr(desc_a, dim.name)
        label_b = getattr(desc_b, dim.name)
        if label_a != label_b:
            changes.append((dim.name, label_a, label_b))

    if not changes:
        return "No perceptual shift detected — the two sounds are perceptually identical."

    clauses: list[str] = []
    for name, lab_a, lab_b in changes:
        clauses.append(f"{name} shifted from {lab_a} to {lab_b}")

    if len(clauses) == 1:
        return clauses[0].capitalize() + "."

    return (
        " — ".join(clauses[:1])
        + ". "
        + ", ".join(c[0].lower() + c[1:] for c in clauses[1:-1])
        + (", and " + clauses[-1][0].lower() + clauses[-1][1:] if len(clauses) > 1 else "")
        + "."
    )


def diff_audio(
    path_a: str | Path,
    path_b: str | Path,
    *,
    sr: int = 16_000,
    mono: bool = True,
    res_type: str = "soxr_qq",
) -> str:
    """Load two audio files and describe what changed between them."""
    waveform_a, loaded_sr_a = librosa.load(path_a, sr=sr, mono=mono, res_type=res_type)
    waveform_b, loaded_sr_b = librosa.load(path_b, sr=sr, mono=mono, res_type=res_type)
    return diff_waveform(waveform_a, waveform_b, sr=int(max(loaded_sr_a, loaded_sr_b)))
