"""Concrete Bayesian model definitions.

Subclasses of ``BayesModelDefinition`` build PyMC models from an
aggregated cell frame. Import this module to auto-register the models
into ``bayes.registry.MODELS``.

M1 (time + age) is deliberately the simplest useful model: it fits a
smooth trend over continuous time ``t = year + (dob_mm - 1) / 12`` and
a smooth maternal-age effect, for both the ``recorded`` and
``recorded_plus_predicted`` outcomes so their posteriors can be compared
side-by-side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from dspopulations_us_birth_certificates.bayes.hsgp import make_hsgp_component
from dspopulations_us_birth_certificates.bayes.registry import BayesModelDefinition

if TYPE_CHECKING:
    import pandas as pd
    import pymc as pm


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _add_time_and_age_coords(cells: pd.DataFrame) -> pd.DataFrame:
    """Attach continuous-time ``t`` and alias ``age`` coord columns.

    ``t = year + (dob_mm - 1) / 12`` gives the trend smooth monthly
    resolution in units of years; ``age`` is an alias for ``mage_c`` so
    the smooth variable name (``f_age``) lines up with a coord column
    in ``cells`` — required by the CLI's per-coord plot loop.
    """
    required = ("year", "dob_mm", "mage_c")
    missing = [c for c in required if c not in cells.columns]
    if missing:
        raise ValueError(
            f"Expected columns {required} on cells to derive 't' / 'age'; "
            f"missing {missing!r} (got {list(cells.columns)!r})"
        )
    cells = cells.copy()
    year = cells["year"].to_numpy(dtype=np.float64)
    month = cells["dob_mm"].to_numpy(dtype=np.float64)
    cells["t"] = year + (month - 1.0) / 12.0
    cells["age"] = cells["mage_c"].to_numpy(dtype=np.float64)
    return cells


class M1YearAge(BayesModelDefinition):
    """M1 — binomial cell model with HSGP smooths on time and maternal age.

    Likelihood::

        y_cell[i] ~ Binomial(n_cell[i], p[i])
        logit(p[i]) = alpha + f_t(t[i]) + f_age(age[i])

    where ``t = year + (dob_mm - 1) / 12`` is a continuous-time coord in
    units of years, giving the trend smooth monthly resolution rather
    than annual steps.

    Priors::

        alpha        ~ Normal(logit(5e-4), 1.0)
        f_t          ~ HSGP(ExpQuad)  — m=24, c=1.5
        f_age        ~ HSGP(ExpQuad)  — m=12, c=1.5

    Intercept prior centred on the ~5/10,000 population DS live-birth rate
    so the prior predictive covers plausible cell counts without pushing
    mass toward implausible probabilities. The time-axis basis count is
    bumped relative to the age smooth because ``t`` covers ~108 monthly
    points across 2016–2024.
    """

    model_id: ClassVar[str] = "m1-year-age"
    dims: ClassVar[tuple[str, ...]] = ("year", "dob_mm", "mage_c")
    smooth_coords: ClassVar[tuple[str, ...]] = ("t", "age")
    year_range: ClassVar[tuple[int, int]] = (2016, 2024)
    priors: ClassVar[dict] = {
        "alpha_mu": _logit(5e-4),
        "alpha_sigma": 1.0,
        "t_m": 24,
        "t_c": 1.5,
        "age_m": 12,
        "age_c": 1.5,
    }
    notes: ClassVar[str] = (
        "Time (HSGP on year + month/12) + maternal age (HSGP) on "
        "year×month×age cells. Seed for M2 (ethnicity) and M3 (education)."
    )

    @classmethod
    def prepare_cells(cls, cells: pd.DataFrame) -> pd.DataFrame:
        return _add_time_and_age_coords(cells)

    @classmethod
    def build(cls, cells: pd.DataFrame) -> pm.Model:
        import pymc as pm

        if "t" not in cells.columns or "age" not in cells.columns:
            cells = cls.prepare_cells(cells)
        t = cells["t"].to_numpy(dtype=np.float64)
        age = cells["age"].to_numpy(dtype=np.float64)
        n_cell = cells["n_cell"].to_numpy(dtype=np.int64)
        y_cell = cells["y_cell"].to_numpy(dtype=np.int64)

        coords = {
            "cell": np.arange(len(cells)),
        }

        with pm.Model(coords=coords) as model:
            pm.Data("t", t, dims="cell")
            pm.Data("age", age, dims="cell")
            n_data = pm.Data("n_cell", n_cell, dims="cell")

            alpha = pm.Normal(
                "alpha",
                mu=cls.priors["alpha_mu"],
                sigma=cls.priors["alpha_sigma"],
            )

            t_component = make_hsgp_component(
                t,
                name="t",
                m=cls.priors["t_m"],
                c=cls.priors["t_c"],
            )
            age_component = make_hsgp_component(
                age,
                name="age",
                m=cls.priors["age_m"],
                c=cls.priors["age_c"],
            )

            eta = alpha + t_component.f + age_component.f
            p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="cell")

            pm.Binomial(
                "y_obs",
                n=n_data,
                p=p,
                observed=y_cell,
                dims="cell",
            )

        return model
