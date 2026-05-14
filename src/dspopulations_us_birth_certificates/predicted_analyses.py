"""Analyses comparing recorded, predicted-missing, and recorded+predicted DS births.

This module assumes that the LightGBM prediction pipeline has already
populated ``us_births.p_ds_lb_pred_01`` and the derived
``us_births.ds_pred_missing`` flag. See ``scripts/fit_model.py`` —
specifically ``write_predictions_to_duckdb`` — for the flagging logic.

Three populations are compared throughout:

- ``predicted``  — ``ds_pred_missing = TRUE`` (likely-missing cases)
- ``recorded``   — ``down_ind = 1`` (birth-certificate-recorded cases)
- ``rprime``     — ``down_ind = 1 OR ds_pred_missing`` (union = R')

Each comparison runs against a :class:`CategoryGrouping` that names the
variable, its SQL expression, its human-readable label map, and an
optional not-null filter. :data:`CATEGORY_GROUPINGS` holds the registry
of groupings the analyse_predicted script iterates over.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot import styles
from matplotlib.figure import Figure

from dspopulations_us_birth_certificates.plot_utils import _save_fig

DOCS_TEMPLATE_ROOT = Path("docs/analysis")

# Default DuckDB column names populated by the usbc10 family. Callers can
# override to target a different model variant's columns (e.g.
# ``ds_pred_missing_02`` + ``p_ds_lb_pred_02`` for usbc11).
DEFAULT_FLAG_COLUMN = "ds_pred_missing"
DEFAULT_PREDICTIONS_COLUMN = "p_ds_lb_pred_01"

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str, kind: str) -> None:
    """Column names are interpolated into SQL, so reject exotic chars."""
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Invalid {kind} column name: {name!r}")


# ---------------------------------------------------------------------------
# Category groupings
# ---------------------------------------------------------------------------

# 2014+ NCHS coding for the raw `mracehisp` column. The same integer
# codes follow a different ORRACEM-style scheme pre-2014, so this label
# set is only correct for rows from the 2014 certificate revision
# onward. The analyse_predicted CLI defaults to year >= 2016 (matching
# the model training window), so the regimes never collide here — see
# notes/202604201112-mracehisp-error.md.
MRACEHISP_LABELS: dict[int, str] = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "Non-Hispanic AIAN",
    4: "Non-Hispanic Asian",
    5: "Non-Hispanic NHOPI",
    6: "Non-Hispanic more than one race",
    7: "Hispanic",
    8: "Origin unknown or not stated",
}

# fracehisp shares codes 1–8 with mracehisp from 2014 on and adds a 9
# bucket for records where paternal ethnicity is unclassified (commonly
# when the father is not on the birth certificate). Same regime caveat
# as MRACEHISP_LABELS.
FRACEHISP_LABELS: dict[int, str] = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "Non-Hispanic AIAN",
    4: "Non-Hispanic Asian",
    5: "Non-Hispanic NHOPI",
    6: "Non-Hispanic more than one race",
    7: "Hispanic",
    8: "Origin unknown or not stated",
    9: "Unknown or not classified",
}

MAGE_C_LABELS: dict[int, str] = {
    1: "Under 20 years",
    2: "20–24 years",
    3: "25–29 years",
    4: "30–34 years",
    5: "35–39 years",
    6: "40–44 years",
    7: "45–49 years",
    8: "50 years and over",
}

MEDUC_LABELS: dict[int, str] = {
    1: "8th grade or less",
    2: "9th through 12th grade with no diploma",
    3: "High school graduate or GED completed",
    4: "Some college credit, but not a degree",
    5: "Associate degree",
    6: "Bachelor's degree",
    7: "Master's degree",
    8: "Doctorate or professional degree",
    9: "Unknown",
}

PAY_REC_LABELS: dict[int, str] = {
    1: "Medicaid",
    2: "Private insurance",
    3: "Self-pay",
    4: "Other",
    9: "Unknown",
}

# Yes / No / Unknown coding shared by ab_nicu, ab_aven1, ca_cchd.
YNU_LABELS: dict[int, str] = {
    0: "No",
    1: "Yes",
    2: "Unknown",
}

GESTREC10_LABELS: dict[int, str] = {
    1: "Under 20 weeks",
    2: "20–27 weeks",
    3: "28–31 weeks",
    4: "32–33 weeks",
    5: "34–36 weeks",
    6: "37–38 weeks",
    7: "39 weeks",
    8: "40 weeks",
    9: "41 weeks",
    10: "42 weeks and over",
    99: "Unknown",
}

# Birth weight banded into clinical buckets. Code 7 captures the NCHS
# "Not stated" sentinel (dbwt = 9999) so the report shows how much is
# unstated rather than silently dropping it.
DBWT_LABELS: dict[int, str] = {
    1: "Under 1500 g (VLBW)",
    2: "1500–2499 g (LBW)",
    3: "2500–2999 g",
    4: "3000–3499 g",
    5: "3500–3999 g",
    6: "4000 g and over",
    7: "Not stated",
}

# Maternal weight gain banded. Code 6 captures the NCHS "Not stated"
# sentinel (wtgain = 99).
WTGAIN_LABELS: dict[int, str] = {
    1: "Less than 15 lbs",
    2: "15–24 lbs",
    3: "25–34 lbs",
    4: "35–44 lbs",
    5: "45 lbs and over",
    6: "Not stated",
}

DMETH_REC_LABELS: dict[int, str] = {
    1: "Vaginal",
    2: "Cesarean",
    9: "Unknown",
}

PRECARE_LABELS: dict[int, str] = {
    0: "No prenatal care",
    1: "1st trimester (months 1–3)",
    2: "2nd trimester (months 4–6)",
    3: "3rd trimester (months 7–10)",
    4: "Not stated",
}

# ca_disor carries the C/P/N/U schema common to NCHS congenital-anomaly
# indicators. C = Confirmed, P = Pending (review), N = None reported,
# U = Unknown or not stated. Mapped to 0/1/2/3 integer codes here.
CA_DISOR_LABELS: dict[int, str] = {
    0: "No",
    1: "Confirmed",
    2: "Pending",
    3: "Unknown",
}

ME_PRES_LABELS: dict[int, str] = {
    1: "Cephalic",
    2: "Breech",
    3: "Other",
    9: "Unknown",
}

# Father's education uses the same NCHS recode schema as maternal.
FEDUC_LABELS: dict[int, str] = MEDUC_LABELS

FAGECOMB_LABELS: dict[int, str] = {
    1: "Under 20 years",
    2: "20–24 years",
    3: "25–29 years",
    4: "30–34 years",
    5: "35–39 years",
    6: "40–44 years",
    7: "45–49 years",
    8: "50 years and over",
    9: "Not stated",
}

BFACIL3_LABELS: dict[int, str] = {
    1: "Hospital",
    2: "Not hospital",
    3: "Unknown",
}

# Fertility-enhancing drugs (rf_fedrg) uses Y/N/U plus an X code for
# "Not applicable" (only queried when rf_inftr = Y).
RF_FEDRG_LABELS: dict[int, str] = {
    0: "No",
    1: "Yes",
    2: "Unknown",
    3: "Not applicable",
}

APGAR5_LABELS: dict[int, str] = {
    1: "0–3 (severely depressed)",
    2: "4–6 (moderately depressed)",
    3: "7–10 (normal)",
    4: "Not stated",
}

SEX_LABELS: dict[int, str] = {
    0: "Female",
    1: "Male",
}

# Maternal pre-pregnancy BMI banded into WHO clinical categories, with
# a final "Not stated" bucket for the NCHS sentinel (≈ 99.875).
BMI_LABELS: dict[int, str] = {
    1: "Underweight (< 18.5)",
    2: "Normal (18.5–24.9)",
    3: "Overweight (25–29.9)",
    4: "Obese I (30–34.9)",
    5: "Obese II (35–39.9)",
    6: "Obese III (≥ 40)",
    7: "Not stated",
}


@dataclass(frozen=True)
class CategoryGrouping:
    """A single categorical comparison plotted and tabulated in the report.

    ``variable`` is the short id used as an output-file prefix
    (``<variable>_recorded_vs_predicted.png``, ``<variable>_summary.csv``).
    ``group_sql`` is the SQL expression that maps a row to an integer
    label-code; for a simple GROUP BY column the expression is just the
    column name. ``labels`` maps those codes to human-readable strings
    and also fixes the row order used everywhere downstream.
    ``legend_title`` is what the plot legend shows; ``not_null_filter``
    is an optional SQL predicate applied before grouping.
    ``colormap`` is the matplotlib colormap name used for the stacked
    segments — default ``tab10`` works up to 10 categories; use a
    continuous map (e.g. ``viridis``) for ordinal variables or when
    there are more than 10 levels.
    """

    variable: str
    title: str
    legend_title: str
    group_sql: str
    labels: dict[int, str]
    not_null_filter: str | None = None
    colormap: str = "tab10"


CATEGORY_GROUPINGS: dict[str, CategoryGrouping] = {
    "mage_c": CategoryGrouping(
        variable="mage_c",
        title="Maternal age",
        legend_title="mage_c",
        group_sql=(
            "CASE "
            "WHEN mage_c < 20 THEN 1 "
            "WHEN mage_c BETWEEN 20 AND 24 THEN 2 "
            "WHEN mage_c BETWEEN 25 AND 29 THEN 3 "
            "WHEN mage_c BETWEEN 30 AND 34 THEN 4 "
            "WHEN mage_c BETWEEN 35 AND 39 THEN 5 "
            "WHEN mage_c BETWEEN 40 AND 44 THEN 6 "
            "WHEN mage_c BETWEEN 45 AND 49 THEN 7 "
            "WHEN mage_c >= 50 THEN 8 "
            "END"
        ),
        labels=MAGE_C_LABELS,
        not_null_filter="mage_c IS NOT NULL",
    ),
    "mracehisp": CategoryGrouping(
        variable="mracehisp",
        title="Maternal race / Hispanic origin",
        legend_title="mracehisp",
        group_sql="mracehisp",
        labels=MRACEHISP_LABELS,
        not_null_filter="mracehisp IS NOT NULL",
    ),
    "meduc": CategoryGrouping(
        variable="meduc",
        title="Maternal education",
        legend_title="meduc",
        group_sql="meduc",
        labels=MEDUC_LABELS,
        not_null_filter="meduc IS NOT NULL",
    ),
    "pay_rec": CategoryGrouping(
        variable="pay_rec",
        title="Principal source of payment",
        legend_title="pay_rec",
        group_sql="pay_rec",
        labels=PAY_REC_LABELS,
        not_null_filter="pay_rec IS NOT NULL",
    ),
    "ab_nicu": CategoryGrouping(
        variable="ab_nicu",
        title="Admission to NICU",
        legend_title="ab_nicu",
        group_sql=(
            "CASE WHEN ab_nicu = 'N' THEN 0 "
            "WHEN ab_nicu = 'Y' THEN 1 "
            "WHEN ab_nicu = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ab_nicu IS NOT NULL",
    ),
    "dbwt": CategoryGrouping(
        variable="dbwt",
        title="Birth weight",
        legend_title="dbwt",
        group_sql=(
            "CASE "
            "WHEN dbwt = 9999 THEN 7 "
            "WHEN dbwt < 1500 THEN 1 "
            "WHEN dbwt BETWEEN 1500 AND 2499 THEN 2 "
            "WHEN dbwt BETWEEN 2500 AND 2999 THEN 3 "
            "WHEN dbwt BETWEEN 3000 AND 3499 THEN 4 "
            "WHEN dbwt BETWEEN 3500 AND 3999 THEN 5 "
            "WHEN dbwt >= 4000 AND dbwt < 9999 THEN 6 "
            "END"
        ),
        labels=DBWT_LABELS,
        not_null_filter="dbwt IS NOT NULL",
        colormap="viridis",
    ),
    "gestrec10": CategoryGrouping(
        variable="gestrec10",
        title="Gestational age at delivery",
        legend_title="gestrec10",
        group_sql="gestrec10",
        labels=GESTREC10_LABELS,
        not_null_filter="gestrec10 IS NOT NULL",
        colormap="viridis",
    ),
    "ab_aven1": CategoryGrouping(
        variable="ab_aven1",
        title="Assisted ventilation immediately following delivery",
        legend_title="ab_aven1",
        group_sql=(
            "CASE WHEN ab_aven1 = 'N' THEN 0 "
            "WHEN ab_aven1 = 'Y' THEN 1 "
            "WHEN ab_aven1 = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ab_aven1 IS NOT NULL",
    ),
    "wtgain": CategoryGrouping(
        variable="wtgain",
        title="Maternal weight gain during pregnancy",
        legend_title="wtgain",
        group_sql=(
            "CASE "
            "WHEN wtgain = 99 THEN 6 "
            "WHEN wtgain < 15 THEN 1 "
            "WHEN wtgain BETWEEN 15 AND 24 THEN 2 "
            "WHEN wtgain BETWEEN 25 AND 34 THEN 3 "
            "WHEN wtgain BETWEEN 35 AND 44 THEN 4 "
            "WHEN wtgain BETWEEN 45 AND 98 THEN 5 "
            "END"
        ),
        labels=WTGAIN_LABELS,
        not_null_filter="wtgain IS NOT NULL",
        colormap="viridis",
    ),
    "ca_cchd": CategoryGrouping(
        variable="ca_cchd",
        title="Cyanotic congenital heart disease",
        legend_title="ca_cchd",
        group_sql=(
            "CASE WHEN ca_cchd = 'N' THEN 0 "
            "WHEN ca_cchd = 'Y' THEN 1 "
            "WHEN ca_cchd = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ca_cchd IS NOT NULL",
    ),
    "dmeth_rec": CategoryGrouping(
        variable="dmeth_rec",
        title="Delivery method",
        legend_title="dmeth_rec",
        group_sql="dmeth_rec",
        labels=DMETH_REC_LABELS,
        not_null_filter="dmeth_rec IS NOT NULL",
    ),
    "precare": CategoryGrouping(
        variable="precare",
        title="Month prenatal care began",
        legend_title="precare",
        group_sql=(
            "CASE "
            "WHEN precare = 99 THEN 4 "
            "WHEN precare = 0 THEN 0 "
            "WHEN precare BETWEEN 1 AND 3 THEN 1 "
            "WHEN precare BETWEEN 4 AND 6 THEN 2 "
            "WHEN precare BETWEEN 7 AND 10 THEN 3 "
            "END"
        ),
        labels=PRECARE_LABELS,
        not_null_filter="precare IS NOT NULL",
        colormap="viridis",
    ),
    "ca_disor": CategoryGrouping(
        variable="ca_disor",
        title="Chromosomal disorder flag",
        legend_title="ca_disor",
        group_sql=(
            "CASE WHEN ca_disor = 'N' THEN 0 "
            "WHEN ca_disor = 'C' THEN 1 "
            "WHEN ca_disor = 'P' THEN 2 "
            "WHEN ca_disor = 'U' THEN 3 END"
        ),
        labels=CA_DISOR_LABELS,
        not_null_filter="ca_disor IS NOT NULL",
    ),
    "ab_aven6": CategoryGrouping(
        variable="ab_aven6",
        title="Assisted ventilation for more than 6 hours",
        legend_title="ab_aven6",
        group_sql=(
            "CASE WHEN ab_aven6 = 'N' THEN 0 "
            "WHEN ab_aven6 = 'Y' THEN 1 "
            "WHEN ab_aven6 = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ab_aven6 IS NOT NULL",
    ),
    "rf_ghype": CategoryGrouping(
        variable="rf_ghype",
        title="Gestational hypertension",
        legend_title="rf_ghype",
        group_sql=(
            "CASE WHEN rf_ghype = 'N' THEN 0 "
            "WHEN rf_ghype = 'Y' THEN 1 "
            "WHEN rf_ghype = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="rf_ghype IS NOT NULL",
    ),
    "bmi": CategoryGrouping(
        variable="bmi",
        title="Maternal pre-pregnancy BMI",
        legend_title="bmi",
        group_sql=(
            "CASE "
            "WHEN bmi >= 69.9 THEN 7 "
            "WHEN bmi < 18.5 THEN 1 "
            "WHEN bmi < 25 THEN 2 "
            "WHEN bmi < 30 THEN 3 "
            "WHEN bmi < 35 THEN 4 "
            "WHEN bmi < 40 THEN 5 "
            "WHEN bmi < 69.9 THEN 6 "
            "END"
        ),
        labels=BMI_LABELS,
        not_null_filter="bmi IS NOT NULL",
        colormap="viridis",
    ),
    "fracehisp": CategoryGrouping(
        variable="fracehisp",
        title="Paternal race / Hispanic origin",
        legend_title="fracehisp",
        group_sql="fracehisp",
        labels=FRACEHISP_LABELS,
        not_null_filter="fracehisp IS NOT NULL",
    ),
    "me_pres": CategoryGrouping(
        variable="me_pres",
        title="Fetal presentation at delivery",
        legend_title="me_pres",
        group_sql="me_pres",
        labels=ME_PRES_LABELS,
        not_null_filter="me_pres IS NOT NULL",
    ),
    "feduc": CategoryGrouping(
        variable="feduc",
        title="Paternal education",
        legend_title="feduc",
        group_sql="feduc",
        labels=FEDUC_LABELS,
        not_null_filter="feduc IS NOT NULL",
    ),
    "ab_anti": CategoryGrouping(
        variable="ab_anti",
        title="Antibiotics received by the newborn",
        legend_title="ab_anti",
        group_sql=(
            "CASE WHEN ab_anti = 'N' THEN 0 "
            "WHEN ab_anti = 'Y' THEN 1 "
            "WHEN ab_anti = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ab_anti IS NOT NULL",
    ),
    "fagecomb": CategoryGrouping(
        variable="fagecomb",
        title="Paternal age",
        legend_title="fagecomb",
        group_sql=(
            "CASE "
            "WHEN fagecomb = 99 THEN 9 "
            "WHEN fagecomb < 20 THEN 1 "
            "WHEN fagecomb BETWEEN 20 AND 24 THEN 2 "
            "WHEN fagecomb BETWEEN 25 AND 29 THEN 3 "
            "WHEN fagecomb BETWEEN 30 AND 34 THEN 4 "
            "WHEN fagecomb BETWEEN 35 AND 39 THEN 5 "
            "WHEN fagecomb BETWEEN 40 AND 44 THEN 6 "
            "WHEN fagecomb BETWEEN 45 AND 49 THEN 7 "
            "WHEN fagecomb BETWEEN 50 AND 98 THEN 8 "
            "END"
        ),
        labels=FAGECOMB_LABELS,
        not_null_filter="fagecomb IS NOT NULL",
        colormap="viridis",
    ),
    "bfacil3": CategoryGrouping(
        variable="bfacil3",
        title="Birth facility type",
        legend_title="bfacil3",
        group_sql="bfacil3",
        labels=BFACIL3_LABELS,
        not_null_filter="bfacil3 IS NOT NULL",
    ),
    "ld_augm": CategoryGrouping(
        variable="ld_augm",
        title="Augmentation of labour",
        legend_title="ld_augm",
        group_sql=(
            "CASE WHEN ld_augm = 'N' THEN 0 "
            "WHEN ld_augm = 'Y' THEN 1 "
            "WHEN ld_augm = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ld_augm IS NOT NULL",
    ),
    "rf_inftr": CategoryGrouping(
        variable="rf_inftr",
        title="Infertility treatment used",
        legend_title="rf_inftr",
        group_sql=(
            "CASE WHEN rf_inftr = 'N' THEN 0 "
            "WHEN rf_inftr = 'Y' THEN 1 "
            "WHEN rf_inftr = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="rf_inftr IS NOT NULL",
    ),
    "rf_fedrg": CategoryGrouping(
        variable="rf_fedrg",
        title="Fertility-enhancing drugs",
        legend_title="rf_fedrg",
        group_sql=(
            "CASE WHEN rf_fedrg = 'N' THEN 0 "
            "WHEN rf_fedrg = 'Y' THEN 1 "
            "WHEN rf_fedrg = 'U' THEN 2 "
            "WHEN rf_fedrg = 'X' THEN 3 END"
        ),
        labels=RF_FEDRG_LABELS,
        not_null_filter="rf_fedrg IS NOT NULL",
    ),
    "rf_phype": CategoryGrouping(
        variable="rf_phype",
        title="Pre-pregnancy hypertension",
        legend_title="rf_phype",
        group_sql=(
            "CASE WHEN rf_phype = 'N' THEN 0 "
            "WHEN rf_phype = 'Y' THEN 1 "
            "WHEN rf_phype = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="rf_phype IS NOT NULL",
    ),
    "apgar5": CategoryGrouping(
        variable="apgar5",
        title="Apgar score at 5 minutes",
        legend_title="apgar5",
        group_sql=(
            "CASE "
            "WHEN apgar5 = 99 THEN 4 "
            "WHEN apgar5 BETWEEN 0 AND 3 THEN 1 "
            "WHEN apgar5 BETWEEN 4 AND 6 THEN 2 "
            "WHEN apgar5 BETWEEN 7 AND 10 THEN 3 "
            "END"
        ),
        labels=APGAR5_LABELS,
        not_null_filter="apgar5 IS NOT NULL",
        colormap="viridis",
    ),
    "ld_indl": CategoryGrouping(
        variable="ld_indl",
        title="Induction of labour",
        legend_title="ld_indl",
        group_sql=(
            "CASE WHEN ld_indl = 'N' THEN 0 "
            "WHEN ld_indl = 'Y' THEN 1 "
            "WHEN ld_indl = 'U' THEN 2 END"
        ),
        labels=YNU_LABELS,
        not_null_filter="ld_indl IS NOT NULL",
    ),
    "sex": CategoryGrouping(
        variable="sex",
        title="Sex of baby",
        legend_title="sex",
        group_sql=(
            "CASE WHEN sex = 'F' THEN 0 WHEN sex = 'M' THEN 1 END"
        ),
        labels=SEX_LABELS,
        not_null_filter="sex IS NOT NULL",
    ),
}


# ---------------------------------------------------------------------------
# Population columns (one per stacked bar)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationColumn:
    """One column in the three-column comparison plot.

    The actual SQL predicate that selects each population is built
    inline by :func:`load_category_counts` from ``flag_column``, so
    this struct only carries the display metadata.
    """

    key: str
    title: str


# Column order runs left → right on the plot. The user's requested layout
# is: predicted on the left, recorded in the centre, recorded+predicted
# on the right.
POPULATION_COLUMNS: tuple[PopulationColumn, ...] = (
    PopulationColumn("predicted", "Predicted missing"),
    PopulationColumn("recorded", "Recorded"),
    PopulationColumn("rprime", "Recorded + predicted"),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_category_counts(
    grouping: CategoryGrouping,
    db_path: str | Path,
    start_year: int | None = None,
    end_year: int | None = None,
    restrict_to_prediction_coverage: bool = True,
    flag_column: str = DEFAULT_FLAG_COLUMN,
    predictions_column: str = DEFAULT_PREDICTIONS_COLUMN,
) -> pd.DataFrame:
    """Count births by ``grouping`` for each of the three populations.

    Returns a DataFrame with one row per code in ``grouping.labels``
    (even if the count is zero) and integer columns ``predicted``,
    ``recorded``, ``rprime``. A ``code`` column holds the integer code
    and a ``label`` column the human-readable name.

    By default the query is restricted to rows with a non-null
    ``predictions_column``, so ``recorded`` and ``predicted`` describe
    the same underlying year range (the model's training years). Pass
    ``restrict_to_prediction_coverage=False`` to count all years.

    ``flag_column`` names the BOOLEAN column that marks predicted-
    missing rows (default ``ds_pred_missing`` for usbc10). Pair it with
    the matching ``predictions_column`` (default ``p_ds_lb_pred_01``).
    Point both at a different model's columns (e.g. ``ds_pred_missing_02``
    + ``p_ds_lb_pred_02`` for usbc11) to compare predicted pools from
    different model variants side by side.
    """
    _check_ident(flag_column, "flag")
    _check_ident(predictions_column, "predictions")

    where: list[str] = []
    params: list[int] = []
    if grouping.not_null_filter:
        where.append(grouping.not_null_filter)
    if restrict_to_prediction_coverage:
        where.append(f"{predictions_column} IS NOT NULL")
    if start_year is not None:
        where.append("year >= ?")
        params.append(start_year)
    if end_year is not None:
        where.append("year <= ?")
        params.append(end_year)
    where_sql = " AND ".join(where) if where else "1 = 1"

    sql = f"""
        SELECT ({grouping.group_sql}) AS code,
               SUM(CASE WHEN {flag_column} THEN 1 ELSE 0 END)         AS predicted,
               SUM(CASE WHEN down_ind = 1 THEN 1 ELSE 0 END)          AS recorded,
               SUM(CASE WHEN down_ind = 1 OR {flag_column} THEN 1
                        ELSE 0 END)                                   AS rprime
        FROM us_births
        WHERE {where_sql}
        GROUP BY code
        ORDER BY code
    """

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    df = df.dropna(subset=["code"])
    df["code"] = df["code"].astype(int)
    df = df.set_index("code")
    df = df.reindex(list(grouping.labels.keys()), fill_value=0)
    df["label"] = [grouping.labels[c] for c in df.index]
    df = df.reset_index()
    return df[["code", "label", "predicted", "recorded", "rprime"]]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stack_bottoms_and_tops(
    proportions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cum = np.concatenate([[0.0], np.cumsum(proportions)])
    return cum[:-1], cum[1:]


def _pick_category_colours(colormap: str, n_cats: int) -> list:
    """Return ``n_cats`` colours sampled from ``colormap``.

    Discrete palettes (``tab10``, ``Set1``, etc. — small ``cmap.N``)
    cycle their entries by index. Continuous palettes (``viridis``,
    ``plasma``, etc. — ``cmap.N == 256``) are sampled evenly across the
    full range so the ordering of categories reads as ordinal. Note
    that matplotlib 3.x represents continuous palettes like viridis as
    ``ListedColormap`` with 256 entries, so ``isinstance`` can't be
    used — discriminate on ``cmap.N``.
    """
    cmap = plt.get_cmap(colormap)
    if cmap.N <= 32:
        return [cmap(i % cmap.N) for i in range(n_cats)]
    return [cmap(i / max(n_cats - 1, 1)) for i in range(n_cats)]


def plot_stacked_proportions(
    counts: pd.DataFrame,
    *,
    title: str,
    legend_title: str,
    colormap: str = "tab10",
    save: bool = False,
    output_dir: str = ".",
    file_name: str = "recorded_vs_predicted",
) -> Figure:
    """Three stacked-proportion columns linked by per-category bands.

    Layout (left → right): predicted / recorded / recorded+predicted.
    Each category's top and bottom edges are joined across columns with
    a lightly-shaded band so shifts in share stand out.

    ``counts`` must be a DataFrame with columns ``code``, ``label``,
    ``predicted``, ``recorded``, and ``rprime`` — the schema produced
    by :func:`load_category_counts`.
    """
    columns = POPULATION_COLUMNS
    n_cats = len(counts)

    proportions: dict[str, np.ndarray] = {}
    totals: dict[str, int] = {}
    for col in columns:
        total = int(counts[col.key].sum())
        totals[col.key] = total
        if total == 0:
            proportions[col.key] = np.zeros(n_cats)
        else:
            proportions[col.key] = counts[col.key].to_numpy() / total

    bottoms: dict[str, np.ndarray] = {}
    tops: dict[str, np.ndarray] = {}
    for col in columns:
        b, t = _stack_bottoms_and_tops(proportions[col.key])
        bottoms[col.key] = b
        tops[col.key] = t

    x_positions = np.arange(len(columns), dtype=float)
    bar_width = 0.45

    colours = _pick_category_colours(colormap, n_cats)

    fig, ax = plt.subplots(figsize=styles.FIGSIZE_XL)

    # Connecting bands: drawn behind the bars so their edges meet the bar
    # faces cleanly.
    for i in range(len(columns) - 1):
        a, b = columns[i].key, columns[i + 1].key
        xa = x_positions[i] + bar_width / 2
        xb = x_positions[i + 1] - bar_width / 2
        for cat_idx in range(n_cats):
            ax.fill(
                [xa, xb, xb, xa],
                [
                    tops[a][cat_idx],
                    tops[b][cat_idx],
                    bottoms[b][cat_idx],
                    bottoms[a][cat_idx],
                ],
                color=colours[cat_idx],
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                [xa, xb],
                [tops[a][cat_idx], tops[b][cat_idx]],
                color=colours[cat_idx],
                linewidth=0.75,
                alpha=0.7,
                zorder=2,
            )

    legend_handles = []
    for cat_idx, cat_label in enumerate(counts["label"]):
        patch = None
        for i, col in enumerate(columns):
            height = proportions[col.key][cat_idx]
            if height <= 0:
                continue
            rect = ax.bar(
                x_positions[i],
                height,
                bottom=bottoms[col.key][cat_idx],
                width=bar_width,
                color=colours[cat_idx],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            if patch is None:
                patch = rect[0]
        if patch is not None:
            patch.set_label(cat_label)
            legend_handles.append(patch)

    # Column totals below each bar.
    for i, col in enumerate(columns):
        ax.text(
            x_positions[i],
            -0.04,
            f"n = {totals[col.key]:,}",
            ha="center",
            va="top",
            fontsize=styles.FONT_SIZE_DEFAULT - 1,
            color=styles.TEXT_COLOUR,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([c.title for c in columns])
    ax.set_ylim(0, 1)
    ax.set_xlim(x_positions[0] - 0.6, x_positions[-1] + 0.6)
    ax.set_ylabel("Share of births")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title=legend_title,
    )

    if save:
        data = _tidy_plot_data(counts, proportions, bottoms, tops)
        _save_fig(fig, output_dir, file_name, data=data)

    return fig


def _tidy_plot_data(
    counts: pd.DataFrame,
    proportions: dict[str, np.ndarray],
    bottoms: dict[str, np.ndarray],
    tops: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Long-format CSV of what actually got drawn."""
    rows = []
    for col in POPULATION_COLUMNS:
        for cat_idx, (code, label) in enumerate(
            zip(counts["code"], counts["label"], strict=True)
        ):
            rows.append(
                {
                    "column": col.key,
                    "column_title": col.title,
                    "code": int(code),
                    "label": label,
                    "count": int(counts[col.key].iloc[cat_idx]),
                    "proportion": float(proportions[col.key][cat_idx]),
                    "stack_bottom": float(bottoms[col.key][cat_idx]),
                    "stack_top": float(tops[col.key][cat_idx]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def category_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Wide summary table: counts + proportions side-by-side per population.

    Columns: ``code``, ``label``, ``predicted_n``, ``predicted_pct``,
    ``recorded_n``, ``recorded_pct``, ``rprime_n``, ``rprime_pct``. Also
    appends a ``Total`` row. Proportions are expressed as percentages to
    two decimals.
    """
    rows = counts[["code", "label"]].copy()
    for col in POPULATION_COLUMNS:
        total = int(counts[col.key].sum())
        rows[f"{col.key}_n"] = counts[col.key].astype(int)
        rows[f"{col.key}_pct"] = (
            (counts[col.key] / total * 100).round(2) if total else 0.0
        )

    total_row = {"code": "", "label": "Total"}
    for col in POPULATION_COLUMNS:
        total_row[f"{col.key}_n"] = int(counts[col.key].sum())
        total_row[f"{col.key}_pct"] = 100.0 if int(counts[col.key].sum()) else 0.0
    return pd.concat([rows, pd.DataFrame([total_row])], ignore_index=True)


# ---------------------------------------------------------------------------
# Artefact persistence + template handling
# ---------------------------------------------------------------------------

def save_config(output_dir: Path, config: dict) -> None:
    """Write ``config.json`` to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def save_category_summary(
    counts: pd.DataFrame,
    output_dir: Path,
    *,
    variable: str,
) -> None:
    """Write ``<variable>_summary.csv`` to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    category_summary(counts).to_csv(
        output_dir / f"{variable}_summary.csv", index=False
    )


def copy_analysis_template(
    output_dir: Path,
    *,
    template_name: str = "predicted",
    docs_root: Path = DOCS_TEMPLATE_ROOT,
) -> Path | None:
    """Copy ``docs/analysis/<template_name>.qmd`` next to the artefacts.

    Returns the destination path, or ``None`` if no template exists.
    """
    src = docs_root / f"{template_name}.qmd"
    if not src.exists():
        return None
    dst = output_dir / "index.qmd"
    shutil.copy(src, dst)
    return dst


def render_quarto(qmd_path: Path) -> None:
    """Invoke ``quarto render`` on a QMD file."""
    subprocess.run(["quarto", "render", str(qmd_path)], check=True)


# ---------------------------------------------------------------------------
# Cross-run comparison (usbc10 vs usbc11 etc.)
# ---------------------------------------------------------------------------

def stage_compare_artefacts(
    *,
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    left_label: str,
    right_label: str,
) -> dict:
    """Copy `<var>_summary.csv` from two source runs into ``output_dir``.

    Both runs must have been produced by ``scripts/analyse_predicted.py``
    — i.e. contain a ``config.json`` plus one
    ``<variable>_summary.csv`` per registered grouping. Files from
    ``left_dir`` are suffixed ``_left`` in the output, files from
    ``right_dir`` are suffixed ``_right``; the Quarto compare template
    keys off those suffixes.

    Returns the compare config (also written as
    ``compare_config.json`` inside ``output_dir``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    left_config: dict = {}
    right_config: dict = {}
    try:
        left_config = json.loads((left_dir / "config.json").read_text())
    except FileNotFoundError:
        pass
    try:
        right_config = json.loads((right_dir / "config.json").read_text())
    except FileNotFoundError:
        pass

    for variable in CATEGORY_GROUPINGS:
        for side, src_dir in (("left", left_dir), ("right", right_dir)):
            src = src_dir / f"{variable}_summary.csv"
            if not src.exists():
                continue
            shutil.copy(src, output_dir / f"{variable}_summary_{side}.csv")

    compare_config = {
        "left_label": left_label,
        "right_label": right_label,
        "left_source": str(left_dir),
        "right_source": str(right_dir),
        "left_config": left_config,
        "right_config": right_config,
    }
    (output_dir / "compare_config.json").write_text(
        json.dumps(compare_config, indent=2, default=str), encoding="utf-8"
    )
    return compare_config
