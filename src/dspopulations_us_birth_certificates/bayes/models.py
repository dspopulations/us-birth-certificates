"""Concrete Bayesian model definitions.

Subclasses of ``BayesModelDefinition`` build PyMC models from an
aggregated cell frame. Import this module to auto-register the models
into ``bayes.registry.MODELS``.

M1 (year + age) is deliberately the simplest useful model: it extends
``notebooks/notes-maternal-age-model-1.py`` by letting both year and age
carry smooth HSGP effects, and by fitting identically-shaped models for
the ``recorded`` and ``recorded_plus_predicted`` outcomes so their
posteriors can be compared side-by-side.
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


class M1YearAge(BayesModelDefinition):
    """M1 — binomial cell model with HSGP smooths on year and maternal age.

    Likelihood::

        y_cell[i] ~ Binomial(n_cell[i], p[i])
        logit(p[i]) = alpha + f_year(year[i]) + f_age(age[i])

    Priors::

        alpha        ~ Normal(logit(5e-4), 1.0)
        f_year       ~ HSGP(ExpQuad)  — m=12, c=1.5
        f_age        ~ HSGP(ExpQuad)  — m=12, c=1.5

    Intercept prior centred on the ~5/10,000 population DS live-birth rate
    so the prior predictive covers plausible cell counts without pushing
    mass toward implausible probabilities.
    """

    model_id: ClassVar[str] = "m1-year-age"
    dims: ClassVar[tuple[str, ...]] = ("year", "mage_c")
    year_range: ClassVar[tuple[int, int]] = (2016, 2024)
    priors: ClassVar[dict] = {
        "alpha_mu": _logit(5e-4),
        "alpha_sigma": 1.0,
        "year_m": 12,
        "year_c": 1.5,
        "age_m": 12,
        "age_c": 1.5,
    }
    notes: ClassVar[str] = (
        "Year (HSGP) + maternal age (HSGP) on year×age cells. "
        "First trend model — seed for M2 (ethnicity) and M3 (education)."
    )

    @classmethod
    def build(cls, cells: pd.DataFrame) -> pm.Model:
        import pymc as pm

        if "year" not in cells.columns or "mage_c" not in cells.columns:
            raise ValueError(
                "M1YearAge expects columns 'year' and 'mage_c' in cells; "
                f"got {list(cells.columns)!r}"
            )
        year = cells["year"].to_numpy(dtype=np.float64)
        age = cells["mage_c"].to_numpy(dtype=np.float64)
        n_cell = cells["n_cell"].to_numpy(dtype=np.int64)
        y_cell = cells["y_cell"].to_numpy(dtype=np.int64)

        coords = {
            "cell": np.arange(len(cells)),
        }

        with pm.Model(coords=coords) as model:
            pm.Data("year", year, dims="cell")
            pm.Data("age", age, dims="cell")
            n_data = pm.Data("n_cell", n_cell, dims="cell")

            alpha = pm.Normal(
                "alpha",
                mu=cls.priors["alpha_mu"],
                sigma=cls.priors["alpha_sigma"],
            )

            year_component = make_hsgp_component(
                year,
                name="year",
                m=cls.priors["year_m"],
                c=cls.priors["year_c"],
            )
            age_component = make_hsgp_component(
                age,
                name="age",
                m=cls.priors["age_m"],
                c=cls.priors["age_c"],
            )

            eta = alpha + year_component.f + age_component.f
            p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="cell")

            pm.Binomial(
                "y_obs",
                n=n_data,
                p=p,
                observed=y_cell,
                dims="cell",
            )

        return model
