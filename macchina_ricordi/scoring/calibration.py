from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ranking_accuracy(arrhythmic_scores: Sequence[float], steady_scores: Sequence[float]) -> float | None:
    arr = np.asarray(arrhythmic_scores, dtype=np.float64)
    steady = np.asarray(steady_scores, dtype=np.float64)
    if arr.size == 0 or steady.size == 0:
        return None

    steady_sorted = np.sort(steady)
    lower = np.searchsorted(steady_sorted, arr, side="left")
    upper = np.searchsorted(steady_sorted, arr, side="right")

    wins = float(np.sum(lower))
    ties = float(np.sum(upper - lower))
    total = float(arr.size * steady.size)
    return (wins + 0.5 * ties) / total


def best_threshold(arrhythmic_scores: Sequence[float], steady_scores: Sequence[float]) -> dict[str, float | int] | None:
    arr = np.asarray(arrhythmic_scores, dtype=np.float64)
    steady = np.asarray(steady_scores, dtype=np.float64)
    if arr.size == 0 or steady.size == 0:
        return None

    all_scores = np.unique(np.concatenate([arr, steady]))
    best: dict[str, float | int] | None = None

    for threshold in all_scores:
        tp = int(np.sum(arr >= threshold))
        fn = int(arr.size - tp)
        tn = int(np.sum(steady < threshold))
        fp = int(steady.size - tn)

        tpr = tp / arr.size
        tnr = tn / steady.size
        balanced_accuracy = 0.5 * (tpr + tnr)

        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": float(tpr),
            "tnr": float(tnr),
        }
        if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
            best = candidate

    return best


def cohens_d(arrhythmic_scores: Sequence[float], steady_scores: Sequence[float]) -> float | None:
    arr = np.asarray(arrhythmic_scores, dtype=np.float64)
    steady = np.asarray(steady_scores, dtype=np.float64)
    if arr.size < 2 or steady.size < 2:
        return None

    arr_std = float(np.std(arr, ddof=1))
    steady_std = float(np.std(steady, ddof=1))
    pooled_num = ((arr.size - 1) * (arr_std**2)) + ((steady.size - 1) * (steady_std**2))
    pooled_den = float(arr.size + steady.size - 2)
    if pooled_den <= 0:
        return None

    pooled_std = math.sqrt(pooled_num / pooled_den)
    if pooled_std <= 1e-12:
        return None

    return float((np.mean(arr) - np.mean(steady)) / pooled_std)


def build_calibration_summary(
    *,
    arrhythmic_scores: Sequence[float],
    steady_scores: Sequence[float],
) -> dict[str, Any]:
    arr_desc = describe(arrhythmic_scores)
    steady_desc = describe(steady_scores)

    arr_mean = arr_desc["mean"]
    steady_mean = steady_desc["mean"]
    delta_mean = None
    if arr_mean is not None and steady_mean is not None:
        delta_mean = float(arr_mean - steady_mean)

    return {
        "arrhythmic": arr_desc,
        "steady": steady_desc,
        "separation": {
            "delta_mean": delta_mean,
            "ranking_accuracy": ranking_accuracy(arrhythmic_scores, steady_scores),
            "cohens_d": cohens_d(arrhythmic_scores, steady_scores),
            "best_threshold": best_threshold(arrhythmic_scores, steady_scores),
        },
    }
