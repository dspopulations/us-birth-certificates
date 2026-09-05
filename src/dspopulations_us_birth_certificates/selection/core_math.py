"""Small, independently testable calculations used by the DSP models."""

from __future__ import annotations

import numpy as np


def calibrated_age_intercept(target_logit, offset, weights):
    """Solve the weighted logistic margin with safeguarded Newton steps.

    The root lies between target_logit - max(offset) and
    target_logit - min(offset). Bracketing prevents Newton overshoot. The
    calculation stays in PyTensor so gradients and the supported samplers use
    the same equation. Even at a zero residual, take a Newton step so its
    derivative includes the implicit dependence on offsets and weights.
    """
    import pytensor
    import pytensor.tensor as pt

    target = pt.sigmoid(target_logit)
    lower = target_logit - pt.max(offset)
    upper = target_logit - pt.min(offset)
    initial = (lower + upper) / 2

    def step(lo, hi, current, target, offset, weights):
        value = pt.sigmoid(current[:, None] + offset[None, :])
        residual = (weights * value).sum(axis=1) - target
        derivative = (weights * value * (1 - value)).sum(axis=1)
        lo = pt.switch(residual < 0, current, lo)
        hi = pt.switch(residual > 0, current, hi)
        proposed = current - residual / pt.maximum(derivative, 1e-15)
        safe = (proposed >= lo) & (proposed <= hi)
        return lo, hi, pt.switch(safe, proposed, (lo + hi) / 2)

    results = pytensor.scan(
        step,
        outputs_info=[lower, upper, initial],
        non_sequences=[target, offset, weights],
        n_steps=64,
        strict=True,
        return_updates=False,
    )
    return results[2][-1]


def window_design(midpoints: np.ndarray, half_width: int, weights=None) -> np.ndarray:
    """Map padded annual prevalence to weighted window prevalence."""
    width = 2 * half_width + 1
    if weights is None:
        weights = np.ones((len(midpoints), width))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(midpoints), width):
        raise ValueError(
            "window weights must have one row per window and column per year"
        )
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("window weights must be finite and positive")
    design = np.zeros((len(midpoints), int(np.max(midpoints)) + width))
    for row, start in enumerate(midpoints):
        design[row, int(start) : int(start) + width] = weights[row] / weights[row].sum()
    return design


def window_error_correlation(
    design: np.ndarray, overlap_share: float = 1.0
) -> np.ndarray:
    """Correlation of window averages of shared, equal-variance annual errors.

    This is an explicit working covariance, not an estimated source covariance.
    overlap_share=0 reproduces independent residuals; 1 uses full overlap.
    Both choices preserve the user-specified marginal observation SD.
    """
    if not np.isfinite(overlap_share) or not 0 <= overlap_share <= 1:
        raise ValueError("anchor overlap share must lie in [0, 1]")
    gram = design @ design.T
    scales = np.sqrt(np.diag(gram))
    correlation = gram / np.outer(scales, scales)
    return overlap_share * correlation + (1 - overlap_share) * np.eye(len(design))
