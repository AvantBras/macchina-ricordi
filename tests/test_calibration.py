from __future__ import annotations

from macchina_ricordi.scoring.calibration import build_calibration_summary, ranking_accuracy


def test_ranking_accuracy_perfect_separation() -> None:
    arr = [0.9, 1.2, 1.8]
    steady = [0.1, 0.2, 0.3]
    result = ranking_accuracy(arr, steady)
    assert result == 1.0


def test_build_calibration_summary_outputs_threshold() -> None:
    arr = [1.0, 1.2, 1.4, 1.6]
    steady = [0.2, 0.3, 0.4, 0.5]
    summary = build_calibration_summary(arrhythmic_scores=arr, steady_scores=steady)

    assert summary["separation"]["delta_mean"] is not None
    assert summary["separation"]["ranking_accuracy"] == 1.0
    assert summary["separation"]["best_threshold"] is not None
    assert summary["separation"]["best_threshold"]["balanced_accuracy"] == 1.0
