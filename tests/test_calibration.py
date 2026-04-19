"""Tests for ``explain.calibration`` helpers.

Covers edge cases lifted from the notebook's inline implementations:
ties in ``y_score``, ``k`` larger than ``n``, ``y_true`` all zeros (recall
undefined), and alignment of ``tail_calibration_table`` between predicted
and observed rates on a perfectly calibrated synthetic stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from dspopulations_us_birth_certificates.explain.calibration import (
    precision_recall_at_k,
    tail_calibration_table,
)


def test_precision_recall_at_k_breaks_ties_stably() -> None:
    # Five scores, all tied. Positives are at indices 0 and 3.
    # Stable mergesort preserves the original order within the tie group,
    # so the top-K picks rows 0, 1, 2, ... and at K=3 we include one positive.
    y_true = np.array([1, 0, 0, 1, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    df = precision_recall_at_k(y_true, y_score, ks=[1, 3, 5])

    assert df["K"].tolist() == [1, 3, 5]
    # Top-1 is row 0 → 1 tp
    assert df.loc[df["K"] == 1, "tp"].iloc[0] == 1
    # Top-3 is rows 0,1,2 → still 1 tp
    assert df.loc[df["K"] == 3, "tp"].iloc[0] == 1
    # Top-5 = everyone → 2 tps
    assert df.loc[df["K"] == 5, "tp"].iloc[0] == 2


def test_precision_recall_at_k_clamps_k_to_n() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2])

    df = precision_recall_at_k(y_true, y_score, ks=[100])

    assert df["K"].iloc[0] == 4  # clamped from 100 to n=4
    assert df["tp"].iloc[0] == 2
    assert df["precision_at_k"].iloc[0] == pytest.approx(0.5)
    assert df["recall_at_k"].iloc[0] == pytest.approx(1.0)


def test_precision_recall_at_k_rejects_empty_positives() -> None:
    with pytest.raises(ValueError, match="no positives"):
        precision_recall_at_k(
            np.zeros(10, dtype=int),
            np.linspace(0, 1, 10),
            ks=[5],
        )


def test_precision_recall_at_k_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        precision_recall_at_k(
            np.array([1, 0, 1]),
            np.array([0.1, 0.2]),
            ks=[1],
        )


def test_precision_recall_at_k_perfect_ranking() -> None:
    # Positives are the top 10 scores; recall@10 should be 1.0.
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.ones(10), np.zeros(90)]).astype(int)
    y_score = np.concatenate([rng.uniform(0.9, 1.0, 10), rng.uniform(0.0, 0.1, 90)])

    df = precision_recall_at_k(y_true, y_score, ks=[10, 50, 100])

    assert df.loc[df["K"] == 10, "precision_at_k"].iloc[0] == pytest.approx(1.0)
    assert df.loc[df["K"] == 10, "recall_at_k"].iloc[0] == pytest.approx(1.0)
    assert df.loc[df["K"] == 100, "recall_at_k"].iloc[0] == pytest.approx(1.0)


def test_tail_calibration_rows_match_requested_fracs() -> None:
    n = 10_000
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)

    fracs = (1e-1, 1e-2, 1e-3)
    df = tail_calibration_table(y, p, fracs=fracs)

    assert df["top_frac"].tolist() == list(fracs)
    assert df["k"].tolist() == [1000, 100, 10]
    # tp + fp == k for every row
    assert (df["tp"] + df["fp"] == df["k"]).all()


def test_tail_calibration_on_perfectly_calibrated_stream() -> None:
    # When y ~ Bernoulli(p) and p is well-dispersed, the mean predicted rate
    # in the top fraction should track the observed rate closely.
    rng = np.random.default_rng(42)
    n = 200_000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)

    df = tail_calibration_table(y, p, fracs=(1e-1,))

    # With 20,000 samples in the top-decile bucket, the empirical pred and
    # observed rates should agree to within 2 percentage points.
    assert abs(df["pred_minus_obs"].iloc[0]) < 0.02


def test_tail_calibration_handles_zero_observed() -> None:
    # All negatives: obs_rate = 0 → ratio_pred_to_obs must be NaN, not inf or a ZeroDivisionError.
    y = np.zeros(100, dtype=int)
    p = np.linspace(0.0, 0.5, 100)

    df = tail_calibration_table(y, p, fracs=(1e-1,))

    assert df["obs_rate"].iloc[0] == 0.0
    assert np.isnan(df["ratio_pred_to_obs"].iloc[0])
