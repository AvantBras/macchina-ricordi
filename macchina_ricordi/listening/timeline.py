from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from .schema import TimelineRegion, TimelineWindow

_REGION_FEATURE_FIELDS = (
    "rms_mean",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "spectral_flux_mean",
    "onset_density",
    "pulse_instability",
    "silence_fraction",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
)


def _level(value: float, thresholds: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for threshold, label in zip(thresholds, labels, strict=False):
        if value < threshold:
            return label
    return labels[-1]


def label_window(features: dict[str, float]) -> str:
    if features["rms_mean"] < 1e-6 or features["silence_fraction"] > 0.85:
        return "near-silent"

    density = _level(
        features["onset_density"],
        (1.0, 3.0, 6.0, 12.0),
        ("sparse", "airy", "moderate", "dense", "saturated"),
    )
    brightness = _level(
        features["spectral_centroid_mean"],
        (600.0, 1600.0, 3200.0, 5200.0),
        ("dark", "warm", "mid-bright", "bright", "piercing"),
    )
    texture = _level(
        features["spectral_flatness_mean"],
        (0.08, 0.22, 0.45, 0.70),
        ("clean", "textured", "rough", "noisy", "chaotic"),
    )
    pulse = _level(
        features["pulse_instability"],
        (0.05, 0.25, 0.55, 0.95),
        ("still", "drifting", "unstable", "restless", "chaotic"),
    )

    if features["high_band_ratio"] >= 0.55:
        band = "high-band"
    elif features["low_band_ratio"] >= 0.55:
        band = "low-band"
    else:
        band = "broad-band"

    return f"{density} {brightness} {texture} {band} texture, {pulse} pulse"


def _average_features(windows: Iterable[TimelineWindow]) -> dict[str, float]:
    window_list = list(windows)
    if not window_list:
        return {}
    keys = set().union(*(window.features for window in window_list))
    return {key: sum(window.features.get(key, 0.0) for window in window_list) / len(window_list) for key in keys}


def _evidence_level(value: float, thresholds: tuple[float, ...]) -> str:
    return _level(value, thresholds, ("very_low", "low", "medium", "high", "very_high"))


def _band_focus(features: dict[str, float]) -> str:
    if features.get("high_band_ratio", 0.0) >= 0.55:
        return "high"
    if features.get("low_band_ratio", 0.0) >= 0.55:
        return "low"
    if features.get("mid_band_ratio", 0.0) >= 0.55:
        return "mid"
    return "broad"


def region_evidence(windows: Iterable[TimelineWindow]) -> dict[str, str]:
    features = _average_features(windows)
    if not features:
        return {}
    return {
        "onset_density": _evidence_level(features.get("onset_density", 0.0), (1.0, 3.0, 6.0, 12.0)),
        "brightness": _evidence_level(features.get("spectral_centroid_mean", 0.0), (600.0, 1600.0, 3200.0, 5200.0)),
        "silence_fraction": _evidence_level(features.get("silence_fraction", 0.0), (0.05, 0.25, 0.55, 0.85)),
        "spectral_flatness": _evidence_level(features.get("spectral_flatness_mean", 0.0), (0.08, 0.22, 0.45, 0.70)),
        "spectral_flux": _evidence_level(features.get("spectral_flux_mean", 0.0), (0.5, 2.0, 5.0, 10.0)),
        "pulse_instability": _evidence_level(features.get("pulse_instability", 0.0), (0.05, 0.25, 0.55, 0.95)),
        "band_focus": _band_focus(features),
    }


def label_region(windows: Iterable[TimelineWindow]) -> str:
    features = _average_features(windows)
    if not features:
        return "unknown region"
    if features.get("rms_mean", 0.0) < 1e-6 or features.get("silence_fraction", 0.0) > 0.85:
        return "near-silence with residue" if features.get("onset_density", 0.0) > 0.0 else "near-silence"

    density = _level(
        features.get("onset_density", 0.0),
        (1.0, 3.0, 6.0, 12.0),
        ("sparse", "airy", "moderate", "dense", "saturated"),
    )
    brightness = _level(
        features.get("spectral_centroid_mean", 0.0),
        (600.0, 1600.0, 3200.0, 5200.0),
        ("dark", "warm", "mid-bright", "bright", "piercing"),
    )
    texture = _level(
        features.get("spectral_flatness_mean", 0.0),
        (0.08, 0.22, 0.45, 0.70),
        ("clean", "textured", "rough", "noisy", "chaotic"),
    )
    pulse = _level(
        features.get("pulse_instability", 0.0),
        (0.05, 0.25, 0.55, 0.95),
        ("still", "drifting", "unstable", "restless", "chaotic"),
    )
    focus = _band_focus(features)

    if features.get("silence_fraction", 0.0) > 0.55:
        return f"gapped {brightness} residue"
    if density in {"sparse", "airy"} and features.get("onset_density", 0.0) > 0.0:
        return f"{density} {brightness} impacts"
    if density in {"dense", "saturated"} and focus == "high":
        return f"{density} high-band swarm"
    if texture in {"rough", "noisy", "chaotic"}:
        return f"{density} {texture} noise field"
    if focus == "low":
        return f"{density} low {brightness} texture with {pulse} pulse"
    return f"{density} {brightness} {texture} {focus}-band texture"


def normalize_window_features(windows: list[TimelineWindow]) -> list[dict[str, float]]:
    if not windows:
        return []
    normalized = [{field: 0.0 for field in _REGION_FEATURE_FIELDS} for _ in windows]
    for field in _REGION_FEATURE_FIELDS:
        values = [window.features.get(field, 0.0) for window in windows]
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-12:
            continue
        for index, value in enumerate(values):
            normalized[index][field] = (value - lo) / (hi - lo)
    return normalized


def compute_novelty(normalized_rows: list[dict[str, float]]) -> list[float]:
    novelty: list[float] = []
    for left, right in zip(normalized_rows, normalized_rows[1:], strict=False):
        novelty.append(
            sum(abs(right[field] - left[field]) for field in _REGION_FEATURE_FIELDS) / len(_REGION_FEATURE_FIELDS)
        )
    return novelty


def detect_change_points(novelty: list[float]) -> list[int]:
    if len(novelty) < 2:
        return []
    mean = sum(novelty) / len(novelty)
    variance = sum((value - mean) ** 2 for value in novelty) / len(novelty)
    threshold = mean + 0.75 * sqrt(variance)
    candidates = [index + 1 for index, value in enumerate(novelty) if value >= threshold and value > 1e-9]
    if candidates:
        return candidates
    strongest = max(novelty)
    if strongest >= 0.35:
        return [novelty.index(strongest) + 1]
    return []


def _region_duration(windows: list[TimelineWindow], start: int, end: int) -> float:
    return max(0.0, windows[end - 1].end_seconds - windows[start].start_seconds)


def _merge_short_segments(
    segments: list[tuple[int, int]],
    windows: list[TimelineWindow],
    novelty: list[float],
) -> list[tuple[int, int]]:
    if len(segments) <= 1:
        return segments

    window_span = max(0.0, windows[0].end_seconds - windows[0].start_seconds)
    hop_span = window_span
    if len(windows) > 1:
        hop_span = max(0.0, windows[1].start_seconds - windows[0].start_seconds)
    min_duration = max(window_span, 2.0 * hop_span)

    merged = list(segments)
    index = 0
    while index < len(merged) and len(merged) > 1:
        start, end = merged[index]
        too_few_windows = end - start < 2
        too_short = _region_duration(windows, start, end) < min_duration
        if not too_few_windows and not too_short:
            index += 1
            continue

        left_novelty = novelty[start - 1] if index > 0 and start > 0 else float("inf")
        right_novelty = novelty[end - 1] if index < len(merged) - 1 and end - 1 < len(novelty) else float("inf")
        if left_novelty <= right_novelty and index > 0:
            previous_start, _ = merged[index - 1]
            merged[index - 1] = (previous_start, end)
            del merged[index]
            index = max(0, index - 1)
        else:
            _, next_end = merged[index + 1]
            merged[index] = (start, next_end)
            del merged[index + 1]
    return merged


def build_timeline_regions(windows: list[TimelineWindow]) -> list[TimelineRegion]:
    if not windows:
        return []
    normalized = normalize_window_features(windows)
    novelty = compute_novelty(normalized)
    change_points = detect_change_points(novelty)
    boundaries = sorted(set([0, *change_points, len(windows)]))
    segments = [(start, end) for start, end in zip(boundaries, boundaries[1:], strict=False) if start < end]
    segments = _merge_short_segments(segments, windows, novelty)

    regions: list[TimelineRegion] = []
    for start, end in segments:
        region_windows = windows[start:end]
        regions.append(
            TimelineRegion(
                start_seconds=region_windows[0].start_seconds,
                end_seconds=region_windows[-1].end_seconds,
                label=label_region(region_windows),
                evidence=region_evidence(region_windows),
                window_indices=list(range(start, end)),
            )
        )
    return regions


def summarize_time_contour(
    windows: Iterable[TimelineWindow],
    regions: Iterable[TimelineRegion] | None = None,
) -> str:
    region_list = list(regions or [])
    if region_list:
        if len(region_list) == 1:
            return f"The detected form is one region: {region_list[0].label}."
        first_region = region_list[0]
        last_region = region_list[-1]
        longest = max(region_list, key=lambda item: item.end_seconds - item.start_seconds)
        return (
            f"The detected form has {len(region_list)} regions, opening as {first_region.label} "
            f"and ending as {last_region.label}. The longest region is "
            f"{longest.start_seconds:.1f}-{longest.end_seconds:.1f}s."
        )

    window_list = list(windows)
    if not window_list:
        return "No timeline windows were available."
    if len(window_list) == 1:
        only = window_list[0]
        return f"The whole clip fits in one window: {only.label}."

    first = window_list[0]
    last = window_list[-1]
    highest_density = max(window_list, key=lambda item: item.features["onset_density"])
    brightest = max(window_list, key=lambda item: item.features["spectral_centroid_mean"])
    quietest = max(window_list, key=lambda item: item.features["silence_fraction"])

    clauses = [f"It opens as {first.label} and ends as {last.label}."]
    if highest_density.features["onset_density"] > 0.0:
        clauses.append(f"The densest window is {highest_density.start_seconds:.1f}-{highest_density.end_seconds:.1f}s.")
    if brightest is not highest_density:
        clauses.append(f"The brightest window is {brightest.start_seconds:.1f}-{brightest.end_seconds:.1f}s.")
    if quietest.features["silence_fraction"] > 0.25:
        clauses.append(f"The most open window is {quietest.start_seconds:.1f}-{quietest.end_seconds:.1f}s.")
    return " ".join(clauses)
