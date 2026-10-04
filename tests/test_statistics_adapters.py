# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility and corrected edge cases in the shared statistics adapters."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dse_research_utils.ml import feature_dependence

from dspopulations_us_birth_certificates import stats_utils
from dspopulations_us_birth_certificates.selection import core_reporting, priors, render


def test_descriptive_table_keeps_existing_rows_and_group_layout():
    expected = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "range",
        "range_std",
        "coef_var",
        "entropy",
        "skew",
        "kurtosis",
        "shapiro_stat",
        "shapiro_pvalue",
        "shapiro_normality",
    ]
    frame = pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10, "x": np.arange(20)})
    summary = stats_utils.describe_all(frame, alpha=0.05)
    assert summary.index.tolist() == expected
    assert summary.columns.tolist() == ["x"]
    assert summary.loc["mean", "x"] == frame.x.mean()
    grouped = stats_utils.describe_all_grouped(frame.groupby("group"), alpha=0.05)
    for key in ("a", "b"):
        assert grouped.loc[key].index.tolist() == expected
        assert grouped.loc[(key, "count"), "x"] == 10


@pytest.mark.parametrize("as_frame", [False, True])
def test_spearman_two_features_and_missing_pairs(as_frame):
    values = np.array([[1, 5], [2, np.nan], [3, 3], [4, 2], [5, 1]])
    data = pd.DataFrame(values) if as_frame else values
    distance, correlation = stats_utils.spearman_distance_matrix(data)
    np.testing.assert_allclose(correlation, [[1, -1], [-1, 1]])
    np.testing.assert_allclose(distance, np.zeros((2, 2)))


@pytest.mark.parametrize("as_frame", [False, True])
def test_zero_mutual_information_has_finite_distances(monkeypatch, as_frame):
    monkeypatch.setattr(
        feature_dependence, "mutual_info_regression", lambda *a, **kw: np.zeros(2)
    )
    values = np.arange(20).reshape(10, 2)
    data = pd.DataFrame(values) if as_frame else values
    np.testing.assert_array_equal(
        stats_utils.mutual_info_dissimilarity(data), [[0, 1], [1, 0]]
    )


def test_prior_transforms_retain_lists_and_avoid_overflow():
    np.testing.assert_allclose(
        priors.logit([0.25, 0.5, 0.75]), [-np.log(3), 0, np.log(3)]
    )
    with np.errstate(over="raise"):
        np.testing.assert_array_equal(priors.inv_logit([-1000, 0, 1000]), [0, 0.5, 1])


@pytest.mark.parametrize("writer", ["render", "core"])
def test_required_svg_failure_propagates_and_leaves_figure_open(
    tmp_path, monkeypatch, writer
):
    fig, _ = plt.subplots()
    savefig = fig.savefig

    def fail_svg(path, *args, **kwargs):
        if str(path).endswith(".svg"):
            raise RuntimeError("required SVG failed")
        return savefig(path, *args, **kwargs)

    monkeypatch.setattr(fig, "savefig", fail_svg)
    try:
        with pytest.raises(RuntimeError, match="required SVG failed"):
            if writer == "render":
                render._save_figure(
                    fig, tmp_path / "plots", "check", data=pd.DataFrame({"x": [1]})
                )
            else:
                core_reporting._save_figure(fig, tmp_path, "check")
        assert (tmp_path / "plots/check.png").is_file()
        assert not (tmp_path / "plots/check.csv").exists()
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)
