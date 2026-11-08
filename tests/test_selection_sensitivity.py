"""Check isolated prior changes and exact aggregation of binomial cells."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from dspopulations_us_birth_certificates.selection.sensitivity import (
    SCENARIOS,
    combine_likelihood_cells,
    scenario_priors,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_changes_only_its_named_prior_family(scenario):
    baseline = asdict(scenario_priors("baseline"))
    actual = asdict(scenario_priors(scenario))
    changed = {
        key for key in baseline if not np.array_equal(actual[key], baseline[key])
    }
    expected = {
        "baseline": set(),
        "baseline_rate_low": {"theta_lb_logit"},
        "baseline_rate_high": {"theta_lb_logit"},
        "baseline_rate_wide": {"theta_lb_sigma"},
        "recording_wide": {"s_race_year_sigma", "s_edu_sigma"},
        "termination_wide": {
            "eta_term_sigma",
            "eta_term_race_sigma",
            "eta_term_edu_sigma",
            "eta_term_age_sigma",
            "eta_term_year_sigma",
        },
        "false_positive_zero": {"false_positive_rate"},
        "false_positive_double": {"false_positive_rate"},
    }
    assert changed == expected[scenario]
    # A new call cannot inherit modifications made to an earlier scenario.
    assert (
        asdict(scenario_priors("baseline"))["false_positive_rate"]
        == baseline["false_positive_rate"]
    )


def test_combining_clinical_strata_preserves_likelihood_ratios():
    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 0, 0],
            "age_idx": [0, 0, 1, 1],
            "race_idx": [0, 0, 0, 0],
            "edu_idx": [0, 0, 0, 0],
            "payer_idx": [0, 0, 0, 0],
            "cchd": [0, 1, 0, 1],
            "N_cell": [1000, 300, 800, 400],
            "R_cell": [2, 3, 1, 2],
        }
    )
    combined = combine_likelihood_cells(cells)
    assert len(combined) == 2
    assert combined.N_cell.tolist() == [1300, 1200]
    assert combined.R_cell.tolist() == [5, 3]
    differences = []
    for probability in ([0.001, 0.003], [0.002, 0.005], [0.02, 0.01]):
        original = binom.logpmf(
            cells.R_cell, cells.N_cell, np.repeat(probability, 2)
        ).sum()
        collapsed = binom.logpmf(combined.R_cell, combined.N_cell, probability).sum()
        differences.append(original - collapsed)
    np.testing.assert_allclose(differences, differences[0], rtol=0, atol=1e-11)


def test_full_model_log_density_changes_only_by_a_constant_after_combining():
    from dspopulations_us_birth_certificates.selection import build_model

    cells = pd.DataFrame(
        {
            "year_idx": [0, 0, 1, 1],
            "age_idx": [0, 0, 1, 1],
            "race_idx": [0, 0, 0, 0],
            "edu_idx": [0, 0, 0, 0],
            "payer_idx": [0, 0, 0, 0],
            "cchd": [0, 1, 0, 1],
            "nicu": [0, 1, 0, 1],
            "aven": [0, 1, 0, 1],
            "preterm": [0, 1, 0, 1],
            "N_cell": [1000, 300, 800, 400],
            "R_cell": [2, 3, 1, 2],
        }
    )
    original = build_model(cells, scenario_priors("baseline"), spec="full", n_year=2)
    combined = build_model(
        combine_likelihood_cells(cells),
        scenario_priors("baseline"),
        spec="full",
        n_year=2,
    )
    original_logp, combined_logp = original.compile_logp(), combined.compile_logp()
    point = original.initial_point()
    changed = {key: value.copy() for key, value in point.items()}
    changed["theta_lb_age"] += 0.1
    changed["eta_term_int"] -= 0.3
    assert original_logp(point) - combined_logp(point) == pytest.approx(
        original_logp(changed) - combined_logp(changed),
        abs=1e-8,
    )


def test_summaries_keep_joint_draws_and_predictive_noise():
    import xarray as xr

    from dspopulations_us_birth_certificates.selection.sensitivity import (
        predictive_summary,
        total_summary,
    )

    cells = pd.DataFrame(
        {
            "year_idx": [0, 1],
            "race_idx": [0, 0],
            "age_idx": [0, 1],
            "N_cell": [100, 100],
            "R_cell": [2, 9],
        }
    )
    expected = np.array([[6.0, 12.0, 18.0, 24.0]])
    natural = np.array([[[10.0, 20.0], [10.0, 20.0], [20.0, 40.0], [20.0, 40.0]]])
    replicated = np.array([[[1, 2], [2, 3], [3, 4], [4, 5]]])
    idata = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "expected_ds_count_total": (("chain", "draw"), expected),
                    "natural_ds_count_year": (("chain", "draw", "year"), natural),
                }
            ),
            "posterior_predictive": xr.Dataset(
                {
                    "R_obs": (("chain", "draw", "cell"), replicated),
                }
            ),
        }
    )
    total = total_summary(idata, cells, "baseline", 0.001).iloc[0]
    assert total.reduction_mean == pytest.approx(
        (1 - expected / natural.sum(axis=-1)).mean()
    )
    assert total.expected_ds_lo89 == pytest.approx(np.quantile(expected, 0.055))
    assert total.recorded_fraction_adjusted_mean == pytest.approx(
        ((11 - 0.001 * (200 - expected)) / expected).mean()
    )
    checks = predictive_summary(idata, cells, "baseline")
    race = checks.loc[checks.stratum == "race_idx"].iloc[0]
    assert race.predicted_mean == pytest.approx(replicated.sum(axis=-1).mean())
    assert race.predicted_hi89 == pytest.approx(
        np.quantile(replicated.sum(axis=-1), 0.945)
    )
    assert not race.observed_in_interval
