"""Descriptive analyses of *recorded* Down-syndrome (DS) live births, 1989-2024.

Backs the report template at ``docs/analysis/descriptive.md`` and the rendered
``docs/analysis/descriptive.qmd``. Every figure/table is produced from read-only
DuckDB queries against ``data/us_births.db`` (table ``us_births``); a recorded DS
birth is ``down_ind = 1`` (``ca_down_c in {'C','P'}``). Artefacts (PNG/SVG/CSV)
land in a run directory so the Quarto template can render against them, mirroring
``predicted_analyses`` / ``scripts/analyse_predicted.py``.

Colours/fonts come from the ``dse_research_utils`` default plotting style applied
by ``init_script()`` (the caller) — figures use the default property cycle rather
than hand-picked colours. Section letters match ``descriptive.md``: A counts &
trends, B recorded-vs-expected, C maternal characteristics, D pregnancy/infant,
E co-occurring conditions.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from dse_research_utils.plot import styles

from dspopulations_us_birth_certificates.plot_utils import save_fig

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

TRANSITION = (2003, 2013)  # 2003-certificate gradual-rollout band

MRACEHISP5: dict[int, str] = {
    1: "NH White",
    2: "NH Black",
    3: "NH AIAN",
    4: "NH Asian/PI",
    5: "Hispanic",
}

PAY_REC_LABELS: dict[int, str] = {
    1: "Medicaid",
    2: "Private",
    3: "Self-pay",
    4: "Other",
}

# Co-occurring congenital-anomaly checkboxes (Y/N/U/blank), 2014+.
COOCCUR_YN: dict[str, str] = {
    "ca_cchd": "Cyanotic congenital heart disease",
    "ca_cdh": "Diaphragmatic hernia",
    "ca_omph": "Omphalocele",
    "ca_gast": "Gastroschisis",
    "ca_limb": "Limb reduction defect",
    "ca_cleft": "Cleft lip +/- palate",
    "ca_clpal": "Cleft palate alone",
    "ca_anen": "Anencephaly",
    "ca_mnsb": "Meningomyelocele / spina bifida",
    "ca_hypo": "Hypospadias",
}


def _q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def _transition_band(ax: plt.Axes, label: str | None = "2003-cert transition") -> None:
    lo, hi = TRANSITION
    ax.axvspan(lo - 0.5, hi + 0.5, color="#d8f0ff", alpha=0.45, zorder=0, label=label)


def _pct_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))


def _cycle_colours(n: int) -> list:
    """First ``n`` colours of the active style's property cycle (dse default)."""
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colours:
        return [None] * n
    return [colours[i % len(colours)] for i in range(n)]


def _save(fig: plt.Figure, out: Path, name: str, data: pd.DataFrame) -> None:
    save_fig(fig, str(out), name, data=data)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section A — recorded numbers and trends
# ---------------------------------------------------------------------------

def section_a_counts(con: duckdb.DuckDBPyConnection, out: Path) -> dict:
    df = _q(
        con,
        """
        SELECT year,
               COUNT(*) FILTER (WHERE ca_down_c = 'C')   AS confirmed,
               COUNT(*) FILTER (WHERE ca_down_c = 'P')   AS pending,
               COUNT(*) FILTER (WHERE ca_down_c = 'N')   AS no,
               COUNT(*) FILTER (WHERE ca_down_c = 'U')   AS unknown,
               COUNT(*) FILTER (WHERE ca_down_c IS NULL) AS not_on_cert,
               COUNT(*)                                  AS births,
               SUM(down_ind)                             AS recorded_ds,
               1e4 * SUM(down_ind) / COUNT(*)            AS recorded_per_10k
        FROM us_births WHERE year IS NOT NULL
        GROUP BY year ORDER BY year
        """,
    )
    df.to_csv(out / "recorded_by_status.csv", index=False)

    # A2 — confirmed vs pending stacked bar + confirmed-only line (band kept).
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    _transition_band(ax)
    ax.bar(df["year"], df["confirmed"], alpha=0.5, label="Confirmed (C)")
    ax.bar(df["year"], df["pending"], bottom=df["confirmed"], alpha=0.5, label="Pending (P, 2004+)")
    ax.plot(df["year"], df["confirmed"], linewidth=1.2, label="Confirmed only (era-robust)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recorded DS live births")
    ax.set_title("Recorded Down syndrome live births by confirmation status, 1989–2024")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out, "recorded_confirmed_pending", df[["year", "confirmed", "pending"]])

    # A3 — recorded rate per 10,000 (no band).
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(df["year"], df["recorded_per_10k"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Recorded DS per 10,000 live births")
    ax.set_title("Recorded DS rate per 10,000 live births (recording completeness, not prevalence)")
    _save(fig, out, "recorded_rate_per_10k", df[["year", "recorded_per_10k"]])

    # A4 — unknown / not-on-certificate trend (no band).
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(df["year"], df["unknown"], marker="o", label="Unknown (U)")
    ax.plot(df["year"], df["not_on_cert"], marker="s", label="Not on certificate (NULL)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of births")
    ax.set_title("DS item recorded as Unknown or absent from the certificate")
    ax.legend(frameon=False)
    _save(fig, out, "ds_status_unknown", df[["year", "unknown", "not_on_cert"]])

    return {
        "total_births": int(df["births"].sum()),
        "total_recorded": int(df["recorded_ds"].sum()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
    }


# ---------------------------------------------------------------------------
# Section B — recorded vs surveillance-expected / recording rate
# ---------------------------------------------------------------------------

def section_b_recording_rate(con: duckdb.DuckDBPyConnection, out: Path) -> dict:
    df = _q(
        con,
        """
        SELECT year,
               SUM(down_ind)         AS recorded,
               SUM(p_ds_lb_nt)       AS expected_no_term,
               SUM(p_ds_lb_nt_reduc) AS expected_after_term,
               SUM(p_ds_lb_wt)       AS expected_surveillance
        FROM us_births WHERE year IS NOT NULL
        GROUP BY year ORDER BY year
        """,
    )
    df["rate_surveillance"] = df["recorded"] / df["expected_surveillance"]
    df["rate_after_term"] = df["recorded"] / df["expected_after_term"]
    df.to_csv(out / "recorded_vs_expected.csv", index=False)

    # B1 — recorded vs expected counts (transition band kept).
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    _transition_band(ax)
    ax.bar(df["year"], df["recorded"], alpha=0.5, label="Recorded", zorder=3)
    ax.plot(df["year"], df["expected_surveillance"], linewidth=1.5,
            label="Expected (surveillance prevalence)")
    ax.plot(df["year"], df["expected_after_term"], linewidth=1.5, linestyle="--",
            label="Expected (age-risk, after terminations)")
    ax.plot(df["year"], df["expected_no_term"], linewidth=1.2, linestyle=":",
            label="Expected (age-risk, no terminations)")
    ax.set_xlabel("Year")
    ax.set_ylabel("DS live births")
    ax.set_ylim(bottom=0)
    ax.set_title("Recorded vs surveillance-expected Down syndrome live births")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    _save(
        fig, out, "recorded_vs_expected",
        df[["year", "recorded", "expected_after_term", "expected_surveillance"]],
    )

    # B2 — recording rate over time (reference band kept).
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.axhspan(0.36, 0.43, color="#d8f0ff", alpha=0.5, label="~36–43% band")
    # The two expected denominators nearly coincide (a useful cross-check); small
    # markers keep both legible where they diverge (chiefly the carried-forward tail).
    ax.plot(df["year"], df["rate_after_term"], marker="s", markersize=4, linewidth=1.5,
            label="vs age-risk (after terminations)")
    ax.plot(df["year"], df["rate_surveillance"], marker="o", markersize=4, linewidth=1.5,
            label="vs surveillance prevalence")
    _pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Recording rate (recorded / expected)")
    ax.set_title("Down syndrome recording rate, 1989–2024")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out, "recording_rate_by_year", df[["year", "rate_surveillance", "rate_after_term"]])

    # B4 — recording rate by race / Hispanic origin over time (age-risk expected).
    race = _q(
        con,
        """
        SELECT year, mracehisp_c AS code,
               SUM(down_ind)         AS recorded,
               SUM(p_ds_lb_nt_reduc) AS expected
        FROM us_births
        WHERE year IS NOT NULL AND mracehisp_c BETWEEN 1 AND 5
        GROUP BY year, mracehisp_c
        """,
    )
    race["rate"] = race["recorded"] / race["expected"]
    rwide = (
        race.pivot(index="year", columns="code", values="rate")
        .reindex(columns=list(MRACEHISP5))
        .sort_index()
    )
    # 3-year centred rolling mean to tame small-group (esp. AIAN) year-to-year
    # noise; the raw per-year rates are saved to the companion CSV.
    rsmooth = rwide.rolling(3, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    for code in MRACEHISP5:
        ax.plot(rsmooth.index, rsmooth[code], label=MRACEHISP5[code])
    _pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Recording rate (recorded / age-risk expected)")
    ax.set_title("Recording rate by maternal race / Hispanic origin (3-year rolling mean)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, out, "recording_rate_by_race", rwide.reset_index())

    overall = float(df["recorded"].sum() / df["expected_surveillance"].sum())
    return {"overall_recording_rate": round(overall, 4)}


# ---------------------------------------------------------------------------
# Section C — maternal characteristics
# ---------------------------------------------------------------------------

def section_c_maternal(con: duckdb.DuckDBPyConnection, out: Path) -> dict:
    # C1a — recorded DS prevalence by single-year maternal age.
    age = _q(
        con,
        """
        SELECT mage_c AS age,
               SUM(down_ind)                 AS recorded,
               COUNT(*)                      AS births,
               1e4 * SUM(down_ind) / COUNT(*) AS recorded_per_10k
        FROM us_births
        WHERE mage_c BETWEEN 12 AND 54
        GROUP BY mage_c ORDER BY mage_c
        """,
    )
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(age["age"], age["recorded_per_10k"], marker="o")
    ax.set_xlabel("Maternal age (single years)")
    ax.set_ylabel("Recorded DS per 10,000 births")
    ax.set_title("Recorded Down syndrome rate by maternal age, 1989–2024")
    _save(fig, out, "recorded_ds_by_mage", age[["age", "recorded", "births", "recorded_per_10k"]])

    # C1b — mean AND median maternal age, recorded DS vs all births, by year (no band).
    mean_age = _q(
        con,
        """
        SELECT year,
               AVG(mage_c)    FILTER (WHERE down_ind = 1) AS mean_recorded_ds,
               MEDIAN(mage_c) FILTER (WHERE down_ind = 1) AS median_recorded_ds,
               AVG(mage_c)                                AS mean_all,
               MEDIAN(mage_c)                             AS median_all
        FROM us_births WHERE year IS NOT NULL AND mage_c IS NOT NULL
        GROUP BY year ORDER BY year
        """,
    )
    cds, call = _cycle_colours(2)
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(mean_age["year"], mean_age["mean_recorded_ds"], marker="o", color=cds,
            label="Recorded DS — mean")
    ax.plot(mean_age["year"], mean_age["median_recorded_ds"], linestyle="--", color=cds,
            label="Recorded DS — median")
    ax.plot(mean_age["year"], mean_age["mean_all"], color=call, label="All births — mean")
    ax.plot(mean_age["year"], mean_age["median_all"], linestyle="--", color=call,
            label="All births — median")
    ax.set_xlabel("Year")
    ax.set_ylabel("Maternal age (years)")
    ax.set_title("Mean and median maternal age: recorded DS births vs all births")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    _save(fig, out, "mean_mage_by_year", mean_age)

    # C2 — race/Hispanic shares of recorded DS over time (5-category, default palette).
    race = _q(
        con,
        """
        SELECT year, mracehisp_c AS code, COUNT(*) AS n
        FROM us_births
        WHERE down_ind = 1 AND year IS NOT NULL AND mracehisp_c BETWEEN 1 AND 5
        GROUP BY year, mracehisp_c
        """,
    )
    wide = race.pivot(index="year", columns="code", values="n").fillna(0).sort_index()
    wide = wide.reindex(columns=list(MRACEHISP5), fill_value=0)
    shares = wide.div(wide.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_LG)
    ax.stackplot(
        shares.index,
        *[shares[c].to_numpy() for c in MRACEHISP5],
        labels=[MRACEHISP5[c] for c in MRACEHISP5],
        alpha=0.5,
    )
    _pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of recorded DS births")
    ax.set_title("Recorded DS births by maternal race / Hispanic origin (5-category)")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    _save(fig, out, "recorded_ds_by_race", shares.reset_index())

    # C3 — maternal education over time (two non-poolable panels: years of
    # schooling pre-2003 vs attainment level 2014+), each collapsed to four
    # roughly-parallel bands for a readable stacked-share time series. The two
    # schemes are NOT equivalent across the 2003 certificate change (see
    # descriptive.md); the differing band labels flag that.
    edu_years = _q(
        con,
        """
        SELECT year,
               CASE WHEN meduc6 IN (1, 2) THEN 1 WHEN meduc6 = 3 THEN 2
                    WHEN meduc6 = 4 THEN 3 WHEN meduc6 = 5 THEN 4 END AS bucket,
               COUNT(*) AS n
        FROM us_births
        WHERE down_ind = 1 AND year BETWEEN 1989 AND 2002 AND meduc6 BETWEEN 1 AND 5
        GROUP BY year, bucket
        """,
    )
    edu_attain = _q(
        con,
        """
        SELECT year,
               CASE WHEN meduc IN (1, 2) THEN 1 WHEN meduc = 3 THEN 2
                    WHEN meduc IN (4, 5) THEN 3 WHEN meduc IN (6, 7, 8) THEN 4 END AS bucket,
               COUNT(*) AS n
        FROM us_births
        WHERE down_ind = 1 AND year >= 2014 AND meduc BETWEEN 1 AND 8
        GROUP BY year, bucket
        """,
    )
    years_lbl = {1: "< 12 years", 2: "12 years", 3: "13–15 years", 4: "16+ years"}
    attain_lbl = {1: "< High school", 2: "HS / GED", 3: "Some college", 4: "Bachelor's+"}

    def _edu_shares(frame: pd.DataFrame, keys: dict) -> pd.DataFrame:
        wide = (
            frame.pivot(index="year", columns="bucket", values="n")
            .reindex(columns=list(keys))
            .fillna(0)
            .sort_index()
        )
        return wide.div(wide.sum(axis=1), axis=0)

    s1 = _edu_shares(edu_years, years_lbl)
    s2 = _edu_shares(edu_attain, attain_lbl)
    fig, axes = plt.subplots(1, 2, figsize=styles.FIGSIZE_LG)
    for ax, shares, labels, title in (
        (axes[0], s1, years_lbl, "1989–2002 (years of schooling)"),
        (axes[1], s2, attain_lbl, "2014–2024 (attainment level)"),
    ):
        ax.stackplot(
            shares.index,
            *[shares[c].to_numpy() for c in labels],
            labels=list(labels.values()),
            alpha=0.5,
        )
        _pct_axis(ax)
        ax.set_ylim(0, 1)
        ax.set_xlim(shares.index.min(), shares.index.max())
        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    axes[0].set_ylabel("Share of recorded DS births")
    fig.suptitle("Maternal education of recorded DS births over time (two non-poolable schemes)")
    _save(fig, out, "recorded_ds_by_education", s2.rename(columns=attain_lbl).reset_index())

    # C4 — marital status (coalesced dmar/mar), share married by year (no band).
    marital = _q(
        con,
        """
        WITH m AS (
            SELECT year, COALESCE(TRY_CAST(dmar AS INTEGER), mar) AS code
            FROM us_births WHERE down_ind = 1 AND year IS NOT NULL
        )
        SELECT year,
               COUNT(*) FILTER (WHERE code = 1)        AS married,
               COUNT(*) FILTER (WHERE code IN (2, 3))  AS unmarried
        FROM m GROUP BY year ORDER BY year
        """,
    )
    marital["pct_married"] = marital["married"] / (marital["married"] + marital["unmarried"])
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(marital["year"], marital["pct_married"], marker="o")
    _pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share married")
    ax.set_title("Marital status of recorded DS mothers: share married")
    _save(fig, out, "recorded_ds_by_marital", marital)

    # C6 — payer mix among recorded DS over time, 2014-2024 (stacked-share time series).
    payer = _q(
        con,
        """
        SELECT year, pay_rec AS code, COUNT(*) AS n
        FROM us_births
        WHERE down_ind = 1 AND year >= 2014 AND pay_rec IN (1, 2, 3, 4)
        GROUP BY year, pay_rec
        """,
    )
    pwide = payer.pivot(index="year", columns="code", values="n").fillna(0).sort_index()
    pwide = pwide.reindex(columns=list(PAY_REC_LABELS), fill_value=0)
    pshares = pwide.div(pwide.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.stackplot(
        pshares.index,
        *[pshares[c].to_numpy() for c in PAY_REC_LABELS],
        labels=[PAY_REC_LABELS[c] for c in PAY_REC_LABELS],
        alpha=0.5,
    )
    _pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_xlim(pshares.index.min(), pshares.index.max())
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of recorded DS births")
    ax.set_title("Principal source of payment, recorded DS births (2014–2024)")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    _save(fig, out, "recorded_ds_by_payer", pshares.reset_index())

    return {}


# ---------------------------------------------------------------------------
# Section D — pregnancy and infant characteristics
# ---------------------------------------------------------------------------

def _grouped_dist(
    con: duckdb.DuckDBPyConnection, band_sql: str, where: str, order_labels: list[str]
) -> pd.DataFrame:
    df = _q(
        con,
        f"""
        SELECT {band_sql} AS band,
               COUNT(*) FILTER (WHERE down_ind = 1) AS ds,
               COUNT(*)                             AS all_n
        FROM us_births WHERE {where}
        GROUP BY band ORDER BY band
        """,
    )
    df = df.dropna(subset=["band"])
    df["band"] = df["band"].astype(int)
    df["ds_share"] = df["ds"] / df["ds"].sum()
    df["all_share"] = df["all_n"] / df["all_n"].sum()
    df["label"] = [order_labels[b - 1] for b in df["band"]]
    return df


def section_d_infant(con: duckdb.DuckDBPyConnection, out: Path) -> dict:
    # D2a — birthweight distribution, recorded DS vs all (2003+).
    bw_labels = ["<1500 g", "1500–2499", "2500–2999", "3000–3499", "3500–3999", "4000+ g"]
    bw = _grouped_dist(
        con,
        (
            "CASE WHEN dbwt < 1500 THEN 1 WHEN dbwt < 2500 THEN 2 "
            "WHEN dbwt < 3000 THEN 3 WHEN dbwt < 3500 THEN 4 "
            "WHEN dbwt < 4000 THEN 5 ELSE 6 END"
        ),
        "year >= 2003 AND dbwt BETWEEN 227 AND 8165",
        bw_labels,
    )
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(len(bw))
    ax.bar(x - 0.2, bw["ds_share"], width=0.4, alpha=0.5, label="Recorded DS")
    ax.bar(x + 0.2, bw["all_share"], width=0.4, alpha=0.5, label="All births")
    ax.set_xticks(x)
    ax.set_xticklabels(bw["label"], rotation=30)
    _pct_axis(ax)
    ax.set_ylabel("Share of births")
    ax.set_title("Birth weight: recorded DS vs all births (2003–2024)")
    ax.legend(frameon=False)
    _save(fig, out, "recorded_ds_birthweight", bw[["band", "label", "ds", "all_n", "ds_share", "all_share"]])

    # D2b — gestational age (gestrec10), recorded DS vs all (2003+).
    gest_labels = [
        "<20 wk", "20–27", "28–31", "32–33", "34–36",
        "37–38", "39", "40", "41", "42+ wk",
    ]
    gest = _grouped_dist(
        con,
        "CASE WHEN gestrec10 BETWEEN 1 AND 10 THEN gestrec10 END",
        "year >= 2003 AND gestrec10 BETWEEN 1 AND 10",
        gest_labels,
    )
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    x = np.arange(len(gest))
    ax.bar(x - 0.2, gest["ds_share"], width=0.4, alpha=0.5, label="Recorded DS")
    ax.bar(x + 0.2, gest["all_share"], width=0.4, alpha=0.5, label="All births")
    ax.set_xticks(x)
    ax.set_xticklabels(gest["label"], rotation=30)
    _pct_axis(ax)
    ax.set_ylabel("Share of births")
    ax.set_title("Gestational age: recorded DS vs all births (2003–2024)")
    ax.legend(frameon=False)
    _save(fig, out, "recorded_ds_gestation", gest[["band", "label", "ds", "all_n", "ds_share", "all_share"]])

    # D1a — plurality, recorded DS vs all births (typically-developing comparison).
    plur = _q(
        con,
        """
        SELECT CASE WHEN dplural = 1 THEN 'Singleton' ELSE 'Multiple' END AS plurality,
               COUNT(*) FILTER (WHERE down_ind = 1) AS ds_n,
               COUNT(*)                             AS all_n
        FROM us_births WHERE dplural IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """,
    )
    plur["ds_share"] = plur["ds_n"] / plur["ds_n"].sum()
    plur["all_share"] = plur["all_n"] / plur["all_n"].sum()
    plur.to_csv(out / "recorded_ds_plurality.csv", index=False)

    # D1b — infant sex (% male) by year, recorded DS vs all births (TD), 2003+.
    sex = _q(
        con,
        """
        SELECT year,
               COUNT(*) FILTER (WHERE down_ind = 1 AND sex = 'M')         AS ds_male,
               COUNT(*) FILTER (WHERE down_ind = 1 AND sex IN ('M', 'F')) AS ds_total,
               COUNT(*) FILTER (WHERE sex = 'M')                          AS all_male,
               COUNT(*) FILTER (WHERE sex IN ('M', 'F'))                  AS all_total
        FROM us_births WHERE year >= 2003
        GROUP BY year ORDER BY year
        """,
    )
    sex["ds_pct_male"] = sex["ds_male"] / sex["ds_total"]
    sex["all_pct_male"] = sex["all_male"] / sex["all_total"]
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.axhline(0.5, color=styles.LINE_COLOUR, linewidth=0.8, linestyle="--", label="50%")
    ax.plot(sex["year"], sex["ds_pct_male"], marker="o", label="Recorded DS")
    ax.plot(sex["year"], sex["all_pct_male"], marker="s", label="All births (TD)")
    _pct_axis(ax)
    ax.set_ylim(0.4, 0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share male")
    ax.set_title("Infant sex: recorded DS vs all births, share male (2003–2024)")
    ax.legend(frameon=False)
    _save(fig, out, "recorded_ds_sex", sex)

    return {}


# ---------------------------------------------------------------------------
# Section E — co-occurring conditions and newborn morbidity (2014+)
# ---------------------------------------------------------------------------

def section_e_cooccurring(con: duckdb.DuckDBPyConnection, out: Path) -> dict:
    # E1 — co-occurring anomalies among recorded DS, % Yes over Y/N base.
    yn_exprs = [
        f"100.0 * COUNT(*) FILTER (WHERE {col} = 'Y') "
        f"/ NULLIF(COUNT(*) FILTER (WHERE {col} IN ('Y','N')), 0) AS {col}"
        for col in COOCCUR_YN
    ]
    disor_expr = (
        "100.0 * COUNT(*) FILTER (WHERE ca_disor = 'C') "
        "/ NULLIF(COUNT(*) FILTER (WHERE ca_disor IN ('C','P','N')), 0) AS ca_disor"
    )
    row = _q(
        con,
        f"""
        SELECT {", ".join([disor_expr, *yn_exprs])}
        FROM us_births WHERE down_ind = 1 AND year >= 2014
        """,
    ).iloc[0]
    labels = {"ca_disor": "Suspected chromosomal disorder", **COOCCUR_YN}
    table = (
        pd.DataFrame({"item": list(labels), "condition": list(labels.values())})
        .assign(pct=lambda d: d["item"].map(lambda c: float(row[c])))
        .sort_values("pct", ascending=True)
        .reset_index(drop=True)
    )
    table.to_csv(out / "recorded_ds_co_occurring.csv", index=False)
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.barh(table["condition"], table["pct"], alpha=0.5)
    ax.set_xlabel("% of recorded DS cases (Y/N base)")
    ax.set_title("Co-occurring conditions among recorded DS births (2014–2024)")
    _save(fig, out, "recorded_ds_co_occurring", table)

    # E2 — newborn morbidity / intervention over time (2014-2024 time series).
    morb = _q(
        con,
        """
        SELECT year,
            100.0 * COUNT(*) FILTER (WHERE ab_nicu = 'Y')
                  / NULLIF(COUNT(*) FILTER (WHERE ab_nicu IN ('Y','N')), 0)   AS nicu,
            100.0 * COUNT(*) FILTER (WHERE ab_aven1 = 'Y')
                  / NULLIF(COUNT(*) FILTER (WHERE ab_aven1 IN ('Y','N')), 0)  AS assisted_ventilation,
            100.0 * COUNT(*) FILTER (WHERE apgar5 BETWEEN 0 AND 6)
                  / NULLIF(COUNT(*) FILTER (WHERE apgar5 BETWEEN 0 AND 10), 0) AS low_apgar5
        FROM us_births WHERE down_ind = 1 AND year >= 2014
        GROUP BY year ORDER BY year
        """,
    )
    fig, ax = plt.subplots(figsize=styles.FIGSIZE_MD)
    ax.plot(morb["year"], morb["nicu"], marker="o", label="NICU admission")
    ax.plot(morb["year"], morb["assisted_ventilation"], marker="s", label="Assisted ventilation")
    ax.plot(morb["year"], morb["low_apgar5"], marker="^", label="5-min APGAR < 7")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of recorded DS newborns")
    ax.set_ylim(bottom=0)
    ax.set_title("Newborn morbidity among recorded DS births (2014–2024)")
    ax.legend(frameon=False)
    _save(fig, out, "recorded_ds_morbidity", morb)

    return {}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all(db_path: str | Path, out: Path) -> dict:
    """Run every section against ``db_path``, writing artefacts to ``out``.

    Returns a summary dict suitable for ``config.json``.
    """
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        summary: dict = {"duckdb_path": str(db_path)}
        summary.update(section_a_counts(con, out))
        summary.update(section_b_recording_rate(con, out))
        section_c_maternal(con, out)
        section_d_infant(con, out)
        section_e_cooccurring(con, out)
    finally:
        con.close()
    return summary
