"""Parameter-recovery validation for the three-stage selection model.

Simulates cells from a known ground truth, fits the full spec, and
checks that the posterior 95% credible interval covers the true value
for the main identifiable parameter families at the rate a
well-specified model should.

Why this test exists
--------------------
If the model cannot recover parameters it generated itself, nothing
downstream (real-data fits, sensitivity variants, Quarto report) is
trustworthy. This is the single most load-bearing test in the suite
and should be run end-to-end before starting Phase 4 on real data.

Why this test is marked ``slow``
--------------------------------
Meaningful coverage assessment needs at least a few hundred draws per
chain to get stable quantile estimates. Locally this test takes a few
minutes; CI does not install pymc (see plan §3.2) so running it there
would add no value. Invoke with ``pytest -m slow`` to opt in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz")

from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    TrueParams,
    build_model,
    simulate_cells,
    variant_C_default,
)

# Test-wide constants. A 9-year window matches the real-data fit
# configuration in Phase 4.
N_YEAR = 9
SEED = 42


def _pick_sampler() -> str:
    """Prefer nutpie (≥10× faster than the pymc default); fall back gracefully."""
    try:
        import nutpie  # noqa: F401

        return "nutpie"
    except ImportError:
        return "pymc"


@pytest.fixture(scope="module")
def recovery_fit() -> tuple[TrueParams, object]:
    """Simulate from a known truth and fit the full spec.

    One fit, reused across the coverage assertions so we pay the MCMC
    cost once per pytest run. Sized to complete in a few minutes on
    nutpie and still give stable quantile estimates for the coverage
    statistic.
    """
    import pymc as pm

    priors = variant_C_default()
    truth = TrueParams.from_priors(priors, n_year=N_YEAR, seed=SEED)
    cells = simulate_cells(
        truth,
        n_cells_per_month=15,
        n_year=N_YEAR,
        n_cells_mean=1500,
        seed=SEED,
    )
    model = build_model(cells, priors, spec="full", n_year=N_YEAR)
    with model:
        idata = pm.sample(
            draws=400,
            tune=400,
            chains=2,
            target_accept=0.9,
            random_seed=SEED,
            progressbar=False,
            nuts_sampler=_pick_sampler(),
        )
    return truth, idata


def _coverage_95(
    idata, name: str, true_values: np.ndarray
) -> float:
    """Fraction of true values falling inside the posterior 95% CI."""
    post = idata.posterior[name]
    lo = post.quantile(0.025, dim=("chain", "draw")).values
    hi = post.quantile(0.975, dim=("chain", "draw")).values
    true_values = np.asarray(true_values)
    # Broadcasting handles both scalar and array-valued parameters.
    covered = (true_values >= lo) & (true_values <= hi)
    return float(np.mean(covered))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("param", "truth_attr", "min_coverage"),
    [
        # Array parameters: 70% threshold absorbs the finite-sample
        # variance of the coverage statistic on small arrays. 95% CIs
        # over N∈{6, 7, 9} hit discrete coverage levels (7/9 = 78%,
        # 5/6 = 83%, 6/7 = 86%); requiring 80% would make the test fail
        # about 10% of the time on a correctly-specified model through
        # sampling noise alone. 70% still catches genuine miscalibration
        # (5/9 = 56%) loudly.
        ("theta_lb_age", "theta_lb_age_logit", 0.7),
        ("eta_term_race", "eta_term_race", 0.7),
        ("eta_term_year", "eta_term_year", 0.7),
        ("s_race", "s_race", 0.7),
    ],
)
def test_parameter_recovery_95_ci_coverage(
    recovery_fit, param: str, truth_attr: str, min_coverage: float
) -> None:
    truth, idata = recovery_fit
    true_values = getattr(truth, truth_attr)
    coverage = _coverage_95(idata, param, true_values)
    assert coverage >= min_coverage, (
        f"{param}: only {coverage:.0%} of true values fall inside the "
        f"posterior 95% CI (need >= {min_coverage:.0%})"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("param", "truth_attr", "tol"),
    [
        # Scalar intercepts are checked by posterior-mean offset in logit
        # units, not by CI coverage (a single point can't support the
        # coverage statistic). 0.5 on logit is a loose band covering
        # typical MCMC variance under these sample sizes.
        ("eta_term_int", "eta_term_int", 0.5),
        ("s_int", "s_int", 0.5),
        ("s_preterm", "s_preterm", 0.5),
        ("s_cchd", "s_cchd", 0.5),
    ],
)
def test_scalar_posterior_mean_within_tolerance(
    recovery_fit, param: str, truth_attr: str, tol: float
) -> None:
    truth, idata = recovery_fit
    true_value = float(getattr(truth, truth_attr))
    mean = float(idata.posterior[param].mean().item())
    assert abs(mean - true_value) <= tol, (
        f"{param}: posterior mean {mean:+.3f} vs truth {true_value:+.3f} "
        f"exceeds tolerance {tol}"
    )


@pytest.mark.slow
def test_sampler_converged(recovery_fit) -> None:
    """Hard-fail on R-hat blow-up so broken fits don't masquerade as bad recovery."""
    import arviz as az

    _, idata = recovery_fit
    summary = az.summary(
        idata,
        var_names=["theta_lb_age", "eta_term_race", "eta_term_year", "s_race"],
    )
    rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
    max_rhat = float(summary[rhat_col].max())
    assert max_rhat < 1.05, (
        f"max R-hat={max_rhat:.3f} — sampler did not converge; "
        f"coverage assertions are unreliable."
    )
