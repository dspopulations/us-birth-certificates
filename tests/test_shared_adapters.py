"""Behaviour of the local adapters onto ``dse-research-utils`` 0.14.0 helpers.

These cover the contracts the project keeps for itself on top of the shared
functions — coverage restriction, NaN policy, cluster identifiers, donor-plan
reproducibility, score direction, manifest fields and file permissions — plus
the three shared contracts the upgrade deliberately adopts.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score, roc_auc_score

from dspopulations_us_birth_certificates import (
    feature_groups,
    file_io,
    intervals,
    manifest,
    ml_utils,
    plot_utils,
)
from dspopulations_us_birth_certificates.models import base_pipeline

# --------------------------------------------------------------- intervals


def _reference_eti(draws, prob, axis, nan):
    lo_q, hi_q = intervals.eti_quantiles(prob)
    quantile = np.nanquantile if nan else np.quantile
    return quantile(draws, lo_q, axis=axis), quantile(draws, hi_q, axis=axis)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, (0, 1), (1, 2), (0, 2), None])
def test_equal_tail_interval_matches_quantiles_over_axis_permutations(axis) -> None:
    draws = np.random.default_rng(11).normal(size=(4, 60, 3))

    lo, hi = intervals.equal_tail_interval(draws, axis=axis)
    ref_lo, ref_hi = _reference_eti(draws, intervals.DEFAULT_ETI_PROB, axis, nan=False)

    np.testing.assert_array_equal(lo, ref_lo)
    np.testing.assert_array_equal(hi, ref_hi)
    assert np.shape(lo) == np.shape(ref_lo)


def test_equal_tail_interval_full_reduction_returns_scalars() -> None:
    draws = np.random.default_rng(12).normal(size=(3, 40))

    lo, hi = intervals.equal_tail_interval(draws)

    assert isinstance(lo, np.floating) and isinstance(hi, np.floating)
    assert np.ndim(lo) == 0 and np.ndim(hi) == 0


def test_equal_tail_interval_keeps_the_project_coverage_restriction() -> None:
    draws = np.arange(100.0)

    # The shared helper accepts prob=1.0; this project never has.
    for prob in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            intervals.equal_tail_interval(draws, prob=prob)


def test_equal_tail_interval_nan_policy_is_propagate_by_default() -> None:
    draws = np.array([[0.0, 1.0], [2.0, np.nan], [4.0, 5.0], [6.0, 7.0]])

    lo, hi = intervals.equal_tail_interval(draws, axis=0)
    assert np.isnan(lo[1]) and np.isnan(hi[1])
    assert not np.isnan(lo[0]) and not np.isnan(hi[0])

    lo_omit, hi_omit = intervals.equal_tail_interval(draws, axis=0, nan=True)
    ref_lo, ref_hi = _reference_eti(draws, intervals.DEFAULT_ETI_PROB, 0, nan=True)
    np.testing.assert_array_equal(lo_omit, ref_lo)
    np.testing.assert_array_equal(hi_omit, ref_hi)


def test_equal_tail_interval_empty_and_all_missing_slices_are_nan() -> None:
    # Adopted from the shared contract: these return NaN bounds rather than
    # raising (empty) or warning (all-NaN). The reduced shape is unchanged.
    lo, hi = intervals.equal_tail_interval(np.empty((0,)))
    assert np.isnan(lo) and np.isnan(hi)

    all_nan = np.full((5, 2), np.nan)
    lo, hi = intervals.equal_tail_interval(all_nan, axis=0, nan=True)
    assert lo.shape == (2,)
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_equal_tail_interval_retains_infinities() -> None:
    # Infinities are retained under both NaN settings, and NumPy's linear
    # interpolation over them is unchanged — including where interpolating
    # between an infinity and a finite value yields NaN.
    draws = np.array([-np.inf, 0.0, 1.0, 2.0, np.inf])

    for nan in (False, True):
        lo, hi = intervals.equal_tail_interval(draws, prob=0.6, nan=nan)
        with np.errstate(invalid="ignore"):
            ref_lo, ref_hi = _reference_eti(draws, 0.6, None, nan=nan)
        np.testing.assert_array_equal(lo, ref_lo)
        np.testing.assert_array_equal(hi, ref_hi)

    with_nan = np.array([-np.inf, 0.0, np.nan, 2.0, np.inf])
    lo, hi = intervals.equal_tail_interval(with_nan, prob=0.6)
    assert np.isnan(lo) and np.isnan(hi)


def test_equal_tail_interval_converts_float32_to_float64() -> None:
    draws = np.linspace(0.0, 1.0, 501, dtype=np.float32)

    lo, hi = intervals.equal_tail_interval(draws)

    assert lo.dtype == np.float64 and hi.dtype == np.float64


def test_posterior_mean_eti_summarises_with_the_mean() -> None:
    # A skewed sample so mean and median differ; the summary must be the mean.
    draws = np.concatenate([np.zeros(90), np.full(10, 10.0)])

    summary = intervals.posterior_mean_eti(draws)

    assert summary["mean"] == pytest.approx(float(np.mean(draws)))
    assert summary["mean"] != pytest.approx(float(np.median(draws)))
    lo, hi = intervals.equal_tail_interval(draws)
    assert (summary["lo"], summary["hi"]) == (float(lo), float(hi))


def test_posterior_mean_eti_missing_value_policy() -> None:
    draws = np.array([0.0, 1.0, np.nan, 3.0, 4.0])

    propagated = intervals.posterior_mean_eti(draws)
    assert np.isnan(propagated["mean"])
    assert np.isnan(propagated["lo"]) and np.isnan(propagated["hi"])

    omitted = intervals.posterior_mean_eti(draws, nan=True)
    assert omitted["mean"] == pytest.approx(2.0)
    assert not np.isnan(omitted["lo"]) and not np.isnan(omitted["hi"])


# ----------------------------------------------------------- feature groups

_FEATURES = ["mage_c", "fagecomb", "ca_cchd", "ab_nicu", "sex"]
_DISTANCE = np.array(
    [
        [0.00, 0.10, 0.90, 0.90, 0.80],
        [0.10, 0.00, 0.90, 0.90, 0.80],
        [0.90, 0.90, 0.00, 0.20, 0.70],
        [0.90, 0.90, 0.20, 0.00, 0.70],
        [0.80, 0.80, 0.70, 0.70, 0.00],
    ]
)


def test_feature_groups_keep_cluster_identifiers_and_feature_order() -> None:
    linkage = hierarchy.linkage(squareform(_DISTANCE), method="average")

    groups = feature_groups.feature_groups_from_linkage(_FEATURES, linkage)

    # Identifiers number the groups by their first feature in ``_FEATURES``,
    # and members stay in that same order.
    assert groups == {
        "cluster_01": ["mage_c", "fagecomb"],
        "cluster_02": ["ca_cchd", "ab_nicu"],
        "cluster_03": ["sex"],
    }


def test_feature_groups_threshold_and_prefix() -> None:
    linkage = hierarchy.linkage(squareform(_DISTANCE), method="average")

    every_feature_alone = feature_groups.feature_groups_from_linkage(
        _FEATURES, linkage, distance_threshold=0.01, prefix="grp"
    )
    assert list(every_feature_alone) == [f"grp_0{i}" for i in range(1, 6)]
    assert [cols[0] for cols in every_feature_alone.values()] == _FEATURES

    one_group = feature_groups.feature_groups_from_linkage(
        _FEATURES, linkage, distance_threshold=0.95
    )
    assert one_group == {"cluster_01": _FEATURES}


def test_feature_groups_degenerate_feature_sets() -> None:
    empty = np.empty((0, 4))

    assert feature_groups.feature_groups_from_linkage([], empty) == {}
    assert feature_groups.feature_groups_from_linkage(["mage_c"], empty) == {
        "cluster_01": ["mage_c"]
    }


def test_distance_corr_linkage_matches_the_scipy_reference() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    frame["b"] = frame["a"] * 0.9 + rng.normal(scale=0.05, size=80)

    distance, corr, linkage = base_pipeline._distance_corr_linkage(frame)
    expected = hierarchy.linkage(squareform(distance, checks=True), method="average")

    np.testing.assert_array_equal(linkage, expected)
    assert corr.shape == (4, 4)
    assert base_pipeline._distance_corr_linkage(frame[["a"]])[2].shape == (0, 4)


# ------------------------------------------------- grouped permutation scoring


class _ProbabilityEstimator:
    """Deterministic classifier that reads two of the four columns."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logit = (
            1.4 * np.asarray(X["signal"], dtype=float)
            + 0.8 * np.asarray(X["grade"].cat.codes, dtype=float)
            - 0.5
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _explanation_sample(seed: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 150
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "grade": pd.Categorical(
                rng.choice(["low", "mid", "high"], size=n),
                categories=["low", "mid", "high"],
                ordered=True,
            ),
            "count": pd.array(rng.integers(0, 5, size=n), dtype="Int64"),
        },
        # Duplicate labels: the explanation sample is drawn positionally from
        # the validation frame, so its index is not unique.
        index=pd.Index(rng.integers(0, 40, size=n), name="row"),
    )
    p = _ProbabilityEstimator().predict_proba(X)[:, 1]
    y = pd.Series((rng.random(n) < p).astype(int), index=X.index)
    return X, y


_GROUPS = {
    "cluster_01": ["signal", "grade"],
    "cluster_02": ["noise"],
    "cluster_03": ["count"],
}


def test_group_permutation_importance_is_reproducible_for_a_seed() -> None:
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()

    first = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=4, random_state=0
    )
    again = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=4, random_state=0
    )
    other_seed = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=4, random_state=1
    )

    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(
        first["importance_mean"].to_numpy(), other_seed["importance_mean"].to_numpy()
    )


def test_group_permutation_importance_score_direction_and_population() -> None:
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()

    out = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=4, random_state=0
    )

    # The baseline is the scorer applied to the whole explanation sample —
    # every row of X, using the positive-class probability column.
    expected_baseline = average_precision_score(y, estimator.predict_proba(X)[:, 1])
    assert out["baseline_score"].to_numpy() == pytest.approx(expected_baseline)

    ranked = out.set_index("group")
    # Positive importance means permutation made average precision worse.
    assert ranked.loc["cluster_01", "importance_mean"] > 0.05
    assert abs(ranked.loc["cluster_02", "importance_mean"]) < 0.02
    assert out["rank"].to_list() == [1, 2, 3]
    assert out["importance_mean"].is_monotonic_decreasing


def test_group_permutation_importance_honours_a_custom_scorer() -> None:
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()

    out = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, scorer=roc_auc_score, n_repeats=2, random_state=0
    )

    assert out["baseline_score"].to_numpy() == pytest.approx(
        roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    )


def test_group_permutation_importance_standard_deviation_convention() -> None:
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()

    single = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=1, random_state=0
    )
    assert (single["importance_std"] == 0.0).all()

    # Only the group the estimator actually reads varies across repeats;
    # permuting an unused column leaves the score — and so the spread — at 0.
    repeated = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=5, random_state=0
    ).set_index("group")
    assert repeated.loc["cluster_01", "importance_std"] > 0.0
    assert repeated.loc["cluster_02", "importance_std"] == 0.0
    assert repeated.loc["cluster_03", "importance_std"] == 0.0


def test_group_permutation_importance_reproduces_the_draw_sequence() -> None:
    """Mean, ddof=1 spread and the per-repeat deltas, recomputed by hand.

    The donor plan is drawn group by group and then repeat by repeat from
    ``np.random.default_rng(random_state)``, so the whole result can be
    rebuilt outside the helper.
    """
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()
    n_repeats = 4
    groups = {"cluster_01": ["signal", "grade"], "cluster_02": ["noise"]}

    out = ml_utils.group_permutation_importance(
        estimator, X, y, groups, n_repeats=n_repeats, random_state=0
    ).set_index("group")

    rng = np.random.default_rng(0)
    baseline = average_precision_score(y, estimator.predict_proba(X)[:, 1])
    for group, columns in groups.items():
        deltas = []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            permuted = X.copy()
            for column in columns:
                # Take through the column's own array so categorical and
                # nullable dtypes survive, as the shared evaluator does.
                permuted[column] = X[column].array.take(order)
            score = average_precision_score(y, estimator.predict_proba(permuted)[:, 1])
            deltas.append(baseline - score)

        assert out.loc[group, "baseline_score"] == pytest.approx(baseline)
        assert out.loc[group, "importance_mean"] == pytest.approx(np.mean(deltas))
        assert out.loc[group, "importance_std"] == pytest.approx(np.std(deltas, ddof=1))
        if np.std(deltas) > 0:
            # ddof=1, not the population standard deviation.
            assert out.loc[group, "importance_std"] != pytest.approx(np.std(deltas))


def test_group_permutation_importance_drops_absent_features_without_draws() -> None:
    X, y = _explanation_sample()
    estimator = _ProbabilityEstimator()

    plain = ml_utils.group_permutation_importance(
        estimator, X, y, _GROUPS, n_repeats=3, random_state=0
    )
    # A group whose features are all absent is skipped before any random
    # number is drawn for it, so the surviving groups are unchanged.
    with_absent = ml_utils.group_permutation_importance(
        estimator,
        X,
        y,
        {"cluster_00": ["not_a_column"], **_GROUPS},
        n_repeats=3,
        random_state=0,
    )
    pd.testing.assert_frame_equal(plain, with_absent)

    partial = ml_utils.group_permutation_importance(
        estimator,
        X,
        y,
        {"cluster_01": ["signal", "grade", "not_a_column"]},
        n_repeats=3,
        random_state=0,
    )
    assert partial.loc[0, "features"] == ["signal", "grade"]
    assert partial.loc[0, "n_features"] == 2


def test_group_permutation_importance_empty_groups_returns_the_schema() -> None:
    X, y = _explanation_sample()

    out = ml_utils.group_permutation_importance(_ProbabilityEstimator(), X, y, {})

    assert out.empty
    assert out.columns.to_list() == [
        "rank",
        "group",
        "n_features",
        "features",
        "baseline_score",
        "importance_mean",
        "importance_std",
    ]


def test_group_permutation_importance_preserves_column_metadata() -> None:
    X, y = _explanation_sample()
    seen: list[pd.DataFrame] = []

    class _Recording(_ProbabilityEstimator):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            seen.append(X)
            return super().predict_proba(X)

    ml_utils.group_permutation_importance(
        _Recording(), X, y, _GROUPS, n_repeats=1, random_state=0
    )

    assert seen, "the estimator was never called"
    for frame in seen:
        assert frame.dtypes.equals(X.dtypes)
        assert frame["grade"].cat.categories.to_list() == ["low", "mid", "high"]
        assert frame["grade"].cat.ordered
        assert frame.index.equals(X.index)
    # The input frame itself is untouched.
    pd.testing.assert_frame_equal(X, _explanation_sample()[0])


def test_group_permutation_importance_permutes_within_the_column() -> None:
    X, y = _explanation_sample()
    seen: list[pd.DataFrame] = []

    class _Recording(_ProbabilityEstimator):
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            seen.append(X.copy())
            return super().predict_proba(X)

    ml_utils.group_permutation_importance(
        _Recording(), X, y, {"cluster_01": ["grade"]}, n_repeats=1, random_state=0
    )

    baseline, permuted = seen[0], seen[-1]
    assert (
        not permuted["grade"].to_numpy().tolist()
        == baseline["grade"].to_numpy().tolist()
    )
    assert permuted["grade"].value_counts().equals(baseline["grade"].value_counts())


# ------------------------------------------------------- provenance and files


def test_git_info_reports_this_repository() -> None:
    info = manifest._git_info()

    assert set(info) == {"sha", "branch", "dirty"}
    # The tests run from a checkout, so the SHA is known and the flag is set.
    assert isinstance(info["sha"], str) and len(info["sha"]) in (40, 64)
    assert isinstance(info["dirty"], bool)


def test_git_info_falls_back_to_none_outside_a_repository(tmp_path: Path) -> None:
    assert manifest._git_info(tmp_path / "does-not-exist") == {
        "sha": None,
        "branch": None,
        "dirty": None,
    }


def test_package_versions_covers_every_tracked_distribution() -> None:
    versions = manifest._package_versions()

    assert tuple(versions) == manifest._TRACKED_PACKAGES
    assert versions["numpy"] is not None
    assert versions["dspopulations-us-birth-certificates"] is not None


def test_write_text_atomically_keeps_default_permissions(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    written = file_io.write_text_atomically(tmp_path / "out" / "written.json", "{}")

    assert written.read_text() == "{}"
    assert stat.S_IMODE(written.stat().st_mode) == stat.S_IMODE(
        reference.stat().st_mode
    )
    assert sorted(p.name for p in written.parent.iterdir()) == ["written.json"]


def test_write_atomically_leaves_the_previous_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "keep.json"
    target.write_text('{"kept": true}')

    def _boom(temporary: Path) -> None:
        temporary.write_text("half")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        file_io.write_atomically(target, _boom)

    assert json.loads(target.read_text()) == {"kept": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.json"]


def test_selection_artefacts_are_replaced_atomically(tmp_path: Path) -> None:
    from dspopulations_us_birth_certificates.selection import config as selection_config
    from dspopulations_us_birth_certificates.selection import io as selection_io

    class _Config:
        def to_dict(self) -> dict[str, str]:
            return {"model_id": "adapter_smoke"}

    context = selection_config.FitContext(
        config=_Config(),
        run_config=selection_config.selection_run_config(
            selection_config.preset_names()[0]
        ),
        output_dir=tmp_path,
    )
    run_dir = tmp_path / "run"

    selection_io.save_artefacts(context, run_dir)

    written = sorted(p.name for p in run_dir.iterdir())
    assert written == ["config.json", "manifest.json", "run_config.json"]
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8")) == {
        "model_id": "adapter_smoke"
    }

    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    for name in written:
        assert stat.S_IMODE((run_dir / name).stat().st_mode) == stat.S_IMODE(
            reference.stat().st_mode
        )

    # The manifest is written last, so it hashes the configs beside it.
    manifest_payload = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest_payload["artefact_sha256"]) == {
        "config.json",
        "run_config.json",
    }
    assert manifest_payload["packages"]["pymc"] is not None
    assert all(
        len(digest) == 64 for digest in manifest_payload["artefact_sha256"].values()
    )
    if Path("uv.lock").is_file():
        assert len(manifest_payload["lockfile_sha256"]) == 64


def test_save_summary_keeps_the_index(tmp_path: Path) -> None:
    from dspopulations_us_birth_certificates.selection import io as selection_io

    summary = pd.DataFrame(
        {"mean": [1.0, 2.0], "sd": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="var")
    )

    selection_io.save_summary(summary, tmp_path)

    text = (tmp_path / "summary.csv").read_text()
    assert text.splitlines()[0] == "var,mean,sd"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["summary.csv"]


# ------------------------------------------------------------------- figures


def test_save_fig_writes_the_stem_trio_and_leaves_the_figure_open(
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    # The directory does not exist yet: the shared saver creates it.
    plot_utils.save_fig(fig, str(tmp_path / "plots"), "demo", data=data)

    out = tmp_path / "plots"
    assert sorted(p.name for p in out.iterdir()) == ["demo.csv", "demo.png", "demo.svg"]
    assert out.joinpath("demo.csv").read_text().splitlines()[0] == "x,y"
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_save_fig_without_data_writes_no_csv(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    plot_utils.save_fig(fig, str(tmp_path), "nodata", dpi=72)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["nodata.png", "nodata.svg"]
    plt.close(fig)
