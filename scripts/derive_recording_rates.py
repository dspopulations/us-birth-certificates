"""Derive the birth-certificate recording-rate anchor s(race, year) for 2016-2024 by
working back from de Graaf surveillance prevalence.

Chain (per race x year):

    true(race, year) = prevalence(race, year) / 1e4 * births(race, year)
    s(race, year)    = recorded_DS(race, year) / true(race, year)

Prevalence is observed (de Graaf) only for 2000-2014, 2016, 2018. The within-window
gaps (2017; and 2019-2024 entirely) are imputed by *indirect standardisation*:

    prevalence = exp_prev(age structure, KNOWN every year) * surv_ratio(net survival)
    exp_prev   = sum_age share(age | race, year) * Morris theta_LB(age)
    surv_ratio = de Graaf prevalence / exp_prev          (smooth; observed years only)

We characterise surv_ratio's per-race trajectory on the observed history, validate the
extrapolation rule with a hold-out backtest (fit <=2010, predict 2011-2014), extrapolate
the 2019-2024 tail, interpolate 2017, reconstruct prevalence with the KNOWN age structure,
and divide the REAL recorded counts (available every year) by the reconstructed true count.

Because recorded counts are real for every study year, tail uncertainty in s comes only
from the imputed prevalence (the survival-ratio extrapolation) -- modelled as a logit-scale
prior sigma that widens with the extrapolation horizon. AIAN is unreliable (tiny counts;
survival ratio exceeds 1 historically) so it is held flat at a robust recent level with a
deliberately wide sigma and no trend.

Outputs (DUA-safe aggregates):
    data/reference/recording_rates_by_race_year.csv   -- the s surface + prior sigma
    notes/figures/recording_rates_anchor.(png/svg/csv) -- survival-ratio fan + s surface

Usage:
    python scripts/derive_recording_rates.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import os  # noqa: E402

import duckdb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from dse_research_utils.plot import styles  # noqa: E402

from dspopulations_us_birth_certificates.plot_utils import save_fig  # noqa: E402
from dspopulations_us_birth_certificates.selection.data import (  # noqa: E402
    RACE_MAP,
    RACE_UNKNOWN_IDX,
    case_from_map,
)
from dspopulations_us_birth_certificates.selection.priors import (  # noqa: E402
    MORRIS_THETA_LB_PER_1000,
    N_RACE,
    RACE_LEVELS,
    inv_logit,
    logit,
)

# Shared with selection.data.prepare_cells's race_idx CASE — same RACE_MAP,
# so a future coding change only needs to happen in one place.
RACE_IDX_CASE = case_from_map("mracehisp_c", RACE_MAP, default=RACE_UNKNOWN_IDX)

DB = "data/us_births.db"
CSV = "data/reference/ds_prevalence_ethnicity_2000_2023.csv"
OUT_CSV = "data/reference/recording_rates_by_race_year.csv"
ANCHOR_MODULE = "src/dspopulations_us_birth_certificates/selection/recording_anchor.py"
OUTPUT_DIR = "notes/figures"

UNKNOWN_S = 0.40  # idx-5 Unknown AND idx-6 NH Multi-race have no de Graaf anchor -> weak neutral fallback
UNKNOWN_SIGMA = 0.50

STUDY_YEARS = list(range(2016, 2025))  # model window
NAMED = list(range(5))  # White, Black, AIAN, Asian/PI, Hispanic
AIAN = 2
STABLE = [0, 1, 3, 4]  # well-behaved survival-ratio trajectories
MORRIS_PER_10K = MORRIS_THETA_LB_PER_1000 * 10.0

ETH_TO_RACE = {
    "Non-Hispanic White": 0,
    "Non-Hispanic Black": 1,
    "American Indian or Alaska Native": 2,
    "Non-Hispanic Asian or Pacific Islander": 3,
    "Hispanic": 4,
}

# Prior-sigma construction (logit scale). Relative SE of s combines the recorded-count
# sampling error (1/sqrt(R), tiny for big groups) with prevalence uncertainty that grows
# from observed -> interpolated -> extrapolated-by-horizon. Tunable; documented in the note.
REL_PREV_OBSERVED = 0.07
REL_PREV_INTERP = 0.12
REL_PREV_EXTRAP_BASE = 0.10
REL_PREV_EXTRAP_PER_YEAR = 0.035  # added per year beyond 2018
SIGMA_LOGIT_FLOOR = 0.05
SIGMA_LOGIT_FLOOR_AIAN = 0.50


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


def _load(con: duckdb.DuckDBPyConnection):
    age = con.execute(
        f"""
        SELECT CAST(year AS INTEGER) AS year,
          {RACE_IDX_CASE} AS race_idx,
          CASE WHEN mage_c < 20 THEN 0 WHEN mage_c < 25 THEN 1 WHEN mage_c < 30 THEN 2
               WHEN mage_c < 35 THEN 3 WHEN mage_c < 40 THEN 4 WHEN mage_c < 45 THEN 5
               ELSE 6 END AS age_idx,
          COUNT(*) AS n
        FROM us_births WHERE year BETWEEN 2000 AND 2024 AND mage_c IS NOT NULL
        GROUP BY 1, 2, 3
        """
    ).df()
    rec = con.execute(
        f"""
        SELECT CAST(year AS INTEGER) AS year,
          {RACE_IDX_CASE} AS race_idx,
          COUNT(*) AS N, SUM(CAST(down_ind AS INTEGER)) AS R
        FROM us_births WHERE year BETWEEN 2016 AND 2024 AND mage_c IS NOT NULL AND down_ind IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    prev = pd.read_csv(CSV)
    prev["race_idx"] = prev["ethnicity"].map(ETH_TO_RACE)
    prev = prev.dropna(subset=["race_idx"]).astype({"race_idx": int})
    prev = prev[["year", "race_idx", "prevalence"]].dropna(subset=["prevalence"])
    return age, rec, prev


def _exp_prev(age: pd.DataFrame) -> pd.DataFrame:
    """Age-structure-only expected prevalence per 10k, every race x year."""
    piv = (
        age[age["race_idx"].isin(NAMED)]
        .pivot_table(index=["year", "race_idx"], columns="age_idx", values="n", fill_value=0)
        .reindex(columns=range(7), fill_value=0)
    )
    shares = piv.div(piv.sum(axis=1), axis=0)
    out = pd.Series((shares.to_numpy() * MORRIS_PER_10K).sum(axis=1), index=piv.index, name="exp_prev")
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Extrapolation: backtest then apply                                          #
# --------------------------------------------------------------------------- #


def _backtest(sr: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Hold out 2011-2014, fit on <=2010, compare constant vs linear (stable races).

    Returns the winning rule name and a per-race RMSE table.
    """
    rows = []
    for r in STABLE:
        d = sr[(sr["race_idx"] == r) & sr["surv_ratio"].notna()].sort_values("year")
        train = d[d["year"] <= 2010]
        test = d[(d["year"] >= 2011) & (d["year"] <= 2014)]
        if len(train) < 4 or test.empty:
            continue
        const = train["surv_ratio"].iloc[-1]  # last fitted value (2010)
        b, a = np.polyfit(train["year"], train["surv_ratio"], 1)
        lin = a + b * test["year"].to_numpy()
        rows.append({
            "race": RACE_LEVELS[r],
            "rmse_const": float(np.sqrt(np.mean((test["surv_ratio"] - const) ** 2))),
            "rmse_linear": float(np.sqrt(np.mean((test["surv_ratio"] - lin) ** 2))),
        })
    bt = pd.DataFrame(rows)
    rule = "const" if bt["rmse_const"].mean() <= bt["rmse_linear"].mean() else "linear"
    return rule, bt


def _extrapolate(sr: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Fill surv_ratio for every study year: observed kept, 2017 interpolated,
    2019-2024 extrapolated by `rule` (stable races) / held flat (AIAN)."""
    out = []
    for r in NAMED:
        d = sr[(sr["race_idx"] == r) & sr["surv_ratio"].notna()].sort_values("year")
        obs = dict(zip(d["year"], d["surv_ratio"], strict=True))
        recent = d[d["year"] >= 2008]
        if r == AIAN:
            level = d[d["year"].isin([2016, 2018])]["surv_ratio"].mean()
            predict = lambda y, level=level: level  # noqa: E731  flat, no trend
        elif rule == "linear":
            b, a = np.polyfit(recent["year"], recent["surv_ratio"], 1)
            predict = lambda y, a=a, b=b: a + b * y  # noqa: E731
        else:
            level = obs[2018]
            predict = lambda y, level=level: level  # noqa: E731
        for y in STUDY_YEARS:
            if y in obs:
                val, src = obs[y], "observed"
            elif y == 2017:  # interior gap -> interpolate 2016/2018
                val, src = 0.5 * (obs[2016] + obs[2018]), "interpolated"
            else:  # 2019-2024
                val, src = float(predict(y)), "extrapolated"
            out.append({"year": y, "race_idx": r, "surv_ratio_used": val, "source": src})
    return pd.DataFrame(out)


def _rel_prev(source: str, year: int) -> float:
    """Relative prevalence uncertainty: tight observed -> wider interpolated ->
    extrapolated growing with the horizon beyond 2018."""
    if source == "observed":
        return REL_PREV_OBSERVED
    if source == "interpolated":
        return REL_PREV_INTERP
    return REL_PREV_EXTRAP_BASE + REL_PREV_EXTRAP_PER_YEAR * (year - 2018)


def _sigma_logit(s: float, R: float, source: str, year: int, race_idx: int) -> float:
    rel_prev = _rel_prev(source, year)
    rel_R = 1.0 / np.sqrt(max(R, 1.0))
    se_s = s * np.sqrt(rel_R**2 + rel_prev**2)
    sig = se_s / max(s * (1.0 - s), 1e-3)
    floor = SIGMA_LOGIT_FLOOR_AIAN if race_idx == AIAN else SIGMA_LOGIT_FLOOR
    return float(max(sig, floor))


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #


def _write_anchor_module(surf: pd.DataFrame, years: list[int]) -> None:
    """Emit the committed anchor module imported by the selection model.

    Shape [N_RACE, n_year]: rows = race idx 0..N_RACE-1, columns = study years. The five
    de Graaf groups (idx 0-4) are anchored from surveillance; idx 5 (Unknown) and idx 6
    (NH Multi-race) have no de Graaf category, so both carry the weak fallback. Emits four
    surfaces:

      * S_RACE_YEAR_LOGIT / _SIGMA   -- recording-rate s prior (idx 5, 6 = weak fallback)
      * PREV_RACE_YEAR / _SIGMA      -- de Graaf true-prevalence margin target (per 10k)
        used by the FULL-MARGIN anchor; idx 5/6 have no surveillance target -> NaN.
    """
    n_year = len(years)
    logit_mat = np.full((N_RACE, n_year), float(logit(UNKNOWN_S)))
    sigma_mat = np.full((N_RACE, n_year), float(UNKNOWN_SIGMA))
    prev_mat = np.full((N_RACE, n_year), np.nan)
    prevsig_mat = np.full((N_RACE, n_year), np.nan)
    for r in range(5):
        d = surf[surf["race_idx"] == r].set_index("year")
        for j, y in enumerate(years):
            logit_mat[r, j] = float(d.loc[y, "s_logit"])
            sigma_mat[r, j] = float(d.loc[y, "s_logit_sigma"])
            prev_mat[r, j] = float(d.loc[y, "prev_used"])
            prevsig_mat[r, j] = float(d.loc[y, "prev_sigma"])

    def _v(v: float) -> str:
        return "  np.nan" if np.isnan(v) else f"{v:8.4f}"

    def _fmt(mat: np.ndarray) -> str:
        lines = []
        for r in range(N_RACE):
            label = RACE_LEVELS[r] if r < len(RACE_LEVELS) else "Unknown"
            vals = ", ".join(_v(v) for v in mat[r])
            lines.append(f"    [{vals}],  # {label}")
        return "[\n" + "\n".join(lines) + "\n]"

    header = (
        '"""GENERATED by scripts/derive_recording_rates.py -- do not edit by hand.\n\n'
        "De Graaf surveillance anchors for the selection model, derived by working back\n"
        "from estimated DS prevalence (data/reference/ds_prevalence_ethnicity_2000_2023.csv)\n"
        "through livebirth counts:\n"
        "    true = prevalence/1e4 * births;  s = recorded_DS / true.\n\n"
        "S_RACE_YEAR_*  -- recording-rate s(race, year) prior (logit mean/sigma; idx 5\n"
        f"  Unknown and idx 6 NH Multi-race = weak fallback s={UNKNOWN_S}).\n"
        "PREV_RACE_YEAR / _SIGMA -- de Graaf TRUE prevalence per 10k (mean/sigma) used as\n"
        "  the full-margin target that ties the model's N-weighted marginal p_ds_lb per\n"
        "  race x year to surveillance; idx 5 Unknown and idx 6 Multi-race have no target\n"
        "  (NaN -> not anchored).\n\n"
        f"Rows = race idx 0..{N_RACE - 1}; columns = years {years[0]}-{years[-1]}. 2019-2024 prevalence\n"
        "is imputed (survival ratio held flat; see the script), so sigma widens across the\n"
        "tail. Regenerate after a data refresh or when surveillance years fill in.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import numpy as np\n\n"
        f"ANCHOR_YEARS = {tuple(years)!r}\n\n"
    )
    body = (
        "S_RACE_YEAR_LOGIT = np.array(" + _fmt(logit_mat) + ")\n\n"
        "S_RACE_YEAR_SIGMA = np.array(" + _fmt(sigma_mat) + ")\n\n"
        "PREV_RACE_YEAR = np.array(" + _fmt(prev_mat) + ")\n\n"
        "PREV_RACE_YEAR_SIGMA = np.array(" + _fmt(prevsig_mat) + ")\n"
    )
    with open(ANCHOR_MODULE, "w", encoding="utf-8") as fh:
        fh.write(header + body)
    print(f"wrote {ANCHOR_MODULE}")


def _figure(sr: pd.DataFrame, surf: pd.DataFrame) -> plt.Figure:
    colours = [styles.COLOUR_BLUE, styles.COLOUR_ORANGE, styles.COLOUR_GREEN,
               styles.COLOUR_RED, styles.COLOUR_PURPLE]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(styles.FIGSIZE_LG[0] * 1.5, styles.FIGSIZE_LG[1]))

    # Left: survival-ratio history + imputed study-year trajectory.
    for r in NAMED:
        c = colours[r]
        hist = sr[(sr["race_idx"] == r) & sr["surv_ratio"].notna()].sort_values("year")
        axl.plot(hist["year"], hist["surv_ratio"], "o", ms=4, color=c, label=RACE_LEVELS[r])
        used = surf[surf["race_idx"] == r].sort_values("year")
        axl.plot(used["year"], used["surv_ratio_used"], "-", color=c, lw=1.5)
        ext = used[used["source"] == "extrapolated"]
        axl.plot(ext["year"], ext["surv_ratio_used"], "x", ms=5, color=c)
    axl.axvspan(2018.5, 2024.5, color=styles.TEXT_COLOUR, alpha=0.08)
    axl.set(xlabel="year", ylabel="survival ratio = de Graaf prevalence / age-expected",
            title="Net survival by ethnicity: history + imputed tail")
    axl.legend(fontsize=7)
    axl.grid(alpha=0.3)

    # Right: recording-rate anchor s(race, year), with prior-sigma band.
    for r in NAMED:
        c = colours[r]
        d = surf[surf["race_idx"] == r].sort_values("year")
        axr.plot(d["year"], d["s"], "-", color=c, lw=1.5, label=RACE_LEVELS[r])
        lo = inv_logit(logit(d["s"]) - d["s_logit_sigma"])
        hi = inv_logit(logit(d["s"]) + d["s_logit_sigma"])
        axr.fill_between(d["year"], lo, hi, color=c, alpha=0.12)
        obs = d[d["source"] == "observed"]
        axr.plot(obs["year"], obs["s"], "o", ms=6, color=c)
    axr.axhline(0.40, ls=":", color=styles.TEXT_COLOUR, lw=1, label="current pin (0.40)")
    axr.set(xlabel="year", ylabel="recording rate s = recorded / true",
            title="Recording-rate anchor s(race, year) +/- prior sigma")
    axr.legend(fontsize=7)
    axr.grid(alpha=0.3)
    fig.suptitle("Birth-certificate recording rate from de Graaf surveillance (s-anchor)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> int:
    setup.init_script()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 200)

    con = duckdb.connect(DB, read_only=True)
    age, rec, prev = _load(con)
    con.close()

    exp_prev = _exp_prev(age)
    sr = exp_prev.merge(prev, on=["year", "race_idx"], how="left")
    sr["surv_ratio"] = sr["prevalence"] / sr["exp_prev"]

    rule, bt = _backtest(sr)
    print("=== Extrapolation backtest (fit <=2010, predict 2011-2014; stable races) ===")
    print(bt.round(4).to_string(index=False))
    print(f"chosen rule: {rule}  (lower mean hold-out RMSE)\n")

    surf = _extrapolate(sr, rule)
    surf = surf.merge(exp_prev, on=["year", "race_idx"]).merge(
        rec[["year", "race_idx", "N", "R"]], on=["year", "race_idx"], how="left"
    )
    surf["prev_used"] = surf["exp_prev"] * surf["surv_ratio_used"]
    surf["true"] = surf["prev_used"] / 1e4 * surf["N"]
    surf["s"] = surf["R"] / surf["true"]
    surf["s_logit"] = logit(surf["s"].clip(1e-4, 0.999))
    surf["s_logit_sigma"] = [
        _sigma_logit(s, R, src, y, r)
        for s, R, src, y, r in zip(
            surf["s"], surf["R"], surf["source"], surf["year"], surf["race_idx"], strict=True
        )
    ]
    # de Graaf true-prevalence margin sigma (per 10k), for the full-margin anchor.
    surf["prev_sigma"] = [
        _rel_prev(src, y) * pv
        for src, y, pv in zip(surf["source"], surf["year"], surf["prev_used"], strict=True)
    ]
    surf["race"] = surf["race_idx"].map(dict(enumerate(RACE_LEVELS)))

    cols = ["year", "race_idx", "race", "source", "N", "R", "exp_prev",
            "surv_ratio_used", "prev_used", "prev_sigma", "true", "s", "s_logit", "s_logit_sigma"]
    surf = surf[cols].sort_values(["race_idx", "year"]).reset_index(drop=True)
    surf.to_csv(OUT_CSV, index=False)
    _write_anchor_module(surf, STUDY_YEARS)

    print("=== Recording-rate anchor s(race, year), 2016-2024 ===")
    show = surf.copy()
    show["s"] = show["s"].round(3)
    show["s_logit_sigma"] = show["s_logit_sigma"].round(3)
    show["surv_ratio_used"] = show["surv_ratio_used"].round(3)
    show["true"] = show["true"].round(0)
    print(show[["year", "race", "source", "R", "true", "s", "s_logit_sigma"]].to_string(index=False))

    tot_true = surf["true"].sum()
    tot_R = surf["R"].sum()
    print(f"\nNamed-race totals 2016-2024:  recorded R = {tot_R:,.0f}   "
          f"de-Graaf-anchored true = {tot_true:,.0f}   overall s = {tot_R / tot_true:.3f}")
    pooled = surf.groupby("race", sort=False).apply(
        lambda d: pd.Series({"s_pooled": d["R"].sum() / d["true"].sum(),
                             "true": d["true"].sum()}), include_groups=False
    )
    print("\nPooled s by ethnicity (2016-2024, anchored):")
    print(pooled.round({"s_pooled": 3, "true": 0}).to_string())

    fig = _figure(sr, surf)
    save_fig(fig, OUTPUT_DIR, "recording_rates_anchor", data=surf)
    plt.close(fig)
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote recording_rates_anchor to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
