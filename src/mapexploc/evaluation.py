"""Evaluation utilities for predictions and explanations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.calibration import calibration_curve


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Compute the Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.abs(prob_true - prob_pred).mean())


def aopc(scores: Sequence[float]) -> float:
    """Area over the Perturbation Curve (AOPC)."""
    scores = np.asarray(scores)
    return float(scores.mean())


def insertion_deletion(reference: Sequence[float], perturbed: Sequence[float]) -> float:
    """Faithfulness metric comparing reference and perturbed outputs."""
    reference = np.asarray(reference)
    perturbed = np.asarray(perturbed)
    return float(np.abs(reference - perturbed).mean())


__all__ = ["expected_calibration_error", "aopc", "insertion_deletion"]
