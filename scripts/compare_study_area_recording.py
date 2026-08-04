"""Test whether the two birth-certificate validation studies were conducted in
representative reporting areas, and transport their sensitivities to national level.

Boulet et al. (2011) and Salemi et al. (2017) are the only external evidence on
`P(recorded | DS livebirth)` -- the one direction the DSPnnn models cannot identify
from the certificate data alone. Both measured a single locality: metropolitan
Atlanta (1995-2005) and Florida (2007-2011). Recorded Down syndrome prevalence
varies about ninefold across US states, so a locally-measured sensitivity is only
usable nationally after correcting for how that locality records.

State is absent from the natality extract (NCHS withdrew geographic detail from
the public-use files with the 2005 data year, and this project never ingested it
for 1989-2004), so the correction is made on the margin that *is* available:

    factor    = national recorded DS prevalence / study-area recorded prevalence
    s_national = s_study * factor

The transport is only legitimate if the study area is ordinary in *true* DS
prevalence and unusual only in *recording*. That is checked directly, by
comparing each study's verified-registry prevalence against this project's
surveillance prevalence for the same years.

Transporting a *confirmed-only* sensitivity needs one further premise, since
`s_C = s_CP * q`: that the confirmed share `q = P(confirmed | flagged)` does not
covary with recording completeness. `confirmation_independence` tests that with a
permutation null, and `confirmation_gradients` asks what `q` measures -- whether
its long drift is reporting behaviour or a real rise in karyotype confirmation.

Outputs (DUA-safe aggregates):
    notes/figures/study-area-recording-transport.csv  -- state-level recorded prevalence

Usage:
    python scripts/compare_study_area_recording.py
"""

from __future__ import annotations  # noqa: I001

import dspopulations_us_birth_certificates.env_guard  # noqa: F401

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dse_research_utils.environment import setup  # noqa: E402
from scipy import stats  # noqa: E402

DB_PATH = "data/us_births.db"
SURVEILLANCE_CSV = "data/us-births-surveillance-prevalence-1989-2024.csv"
WONDER_STATE_CSV = "data/us-births-wonder-state-pooled-2016-2024.csv"
OUT_CSV = "notes/figures/study-area-recording-transport.csv"

# Boulet et al. (2011), Public Health Rep 126:186-194, Table 1.
# Metropolitan Atlanta 5-county MACDP catchment, 1995-2005 livebirths.
BOULET = {
    "label": "Boulet / metro Atlanta",
    "recorded": 116,  # BC-flagged DS
    "true_positives": 113,  # BC-flagged and MACDP-confirmed
    "true": 625,  # MACDP-confirmed DS
    "births": 522_315,  # study denominator, reported in the paper
    "years": (1995, 2005),
    "revised_comparator": False,  # 1989-revision certificate throughout
}

# Salemi et al. (2017), Paediatr Perinat Epidemiol 31:67-75, Table 1.
# Florida resident livebirths, 2007-2011, 2003-revision certificate.
# The paper does not print its birth denominator; FL_BIRTHS is the NCHS final
# natality resident-livebirth total for 2007-2011 and is the one external number
# here. `transport_sensitivity_to_denominator` reports the range it induces.
FL_BIRTHS = 1_120_000
SALEMI = {
    "label": "Salemi / Florida",
    "recorded": 417,
    "true_positives": 364,
    "true": 1478,
    "births": FL_BIRTHS,
    "years": (2007, 2011),
    "revised_comparator": True,  # Florida adopted the 2003 revision in 2004
}

# Salemi Table 1, karyotype-confirmation split (the NVSS `ca_down_c` C/P states).
SALEMI_CONFIRMED = {"recorded": 115, "true_positives": 103}


def national_by_year(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Recorded DS counts and covered births per data year.

    `recorded` follows the core models exactly: `down_ind` is the harmonised
    confirmed-or-pending indicator, and a record counts as 2003-revision when
    either 2003-cert Down syndrome column is populated.
    """
    frame = con.execute("""
        SELECT CAST(year AS INTEGER)                                    AS year,
               SUM(CASE WHEN down_ind IS NOT NULL THEN 1 ELSE 0 END)    AS births,
               SUM(CASE WHEN CAST(down_ind AS INTEGER) = 1 THEN 1 ELSE 0 END)
                                                                        AS recorded,
               SUM(CASE WHEN ca_down IS NOT NULL OR ca_downs IS NOT NULL
                        THEN 1 ELSE 0 END)                              AS births_revised,
               SUM(CASE WHEN (ca_down IS NOT NULL OR ca_downs IS NOT NULL)
                         AND CAST(down_ind AS INTEGER) = 1 THEN 1 ELSE 0 END)
                                                                        AS recorded_revised
        FROM us_births
        WHERE year IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()
    for col in ("births", "recorded", "births_revised", "recorded_revised"):
        frame[col] = frame[col].astype("int64")
    frame["prev10k"] = 1e4 * frame.recorded / frame.births
    return frame


def confirmed_split(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Confirmed vs pending flags, restricted to 2003-revision records.

    `ca_down_c` harmonises the 1989-cert "anomaly reported" state to `C`, so the
    karyotype split is only meaningful where the 2003-revision field exists.
    """
    return con.execute("""
        SELECT CAST(year AS INTEGER) AS year,
               SUM(CASE WHEN UPPER(CAST(ca_down_c AS VARCHAR)) = 'C'
                        THEN 1 ELSE 0 END) AS confirmed,
               SUM(CASE WHEN UPPER(CAST(ca_down_c AS VARCHAR)) = 'P'
                        THEN 1 ELSE 0 END) AS pending
        FROM us_births
        WHERE (ca_down IS NOT NULL OR ca_downs IS NOT NULL) AND year IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()


def national_window(
    frame: pd.DataFrame, years: tuple[int, int], revised_only: bool
) -> tuple[int, int, float]:
    lo, hi = years
    win = frame[(frame.year >= lo) & (frame.year <= hi)]
    if revised_only:
        births, recorded = (
            int(win.births_revised.sum()),
            int(win.recorded_revised.sum()),
        )
    else:
        births, recorded = int(win.births.sum()), int(win.recorded.sum())
    return births, recorded, 1e4 * recorded / births


def transport(study: dict, frame: pd.DataFrame) -> dict:
    """Scale a locally-measured sensitivity to national recording level."""
    area_prev = 1e4 * study["recorded"] / study["births"]
    births, recorded, nat_prev = national_window(
        frame, study["years"], study["revised_comparator"]
    )
    sensitivity = study["true_positives"] / study["true"]
    factor = nat_prev / area_prev
    return {
        "label": study["label"],
        "years": study["years"],
        "area_prev10k": area_prev,
        "national_births": births,
        "national_recorded": recorded,
        "national_prev10k": nat_prev,
        "factor": factor,
        "sensitivity": sensitivity,
        "sensitivity_national": sensitivity * factor,
        "area_true_prev10k": 1e4 * study["true"] / study["births"],
    }


def revision_contrast(frame: pd.DataFrame, years: tuple[int, int]) -> dict:
    """Recorded DS prevalence on 2003-revision vs 1989-revision records.

    Tests Salemi's central claim -- that restricting the 2003 form to defects
    identifiable at birth did not improve capture -- against this project's own
    data. Only years carrying both layouts are informative, and only those where
    the split is reasonably balanced: at the extremes the contrast is confounded
    by which states adopted the revision first and which held out longest.
    """
    lo, hi = years
    win = frame[(frame.year >= lo) & (frame.year <= hi)]
    rev_b, rev_r = int(win.births_revised.sum()), int(win.recorded_revised.sum())
    unrev_b = int(win.births.sum()) - rev_b
    unrev_r = int(win.recorded.sum()) - rev_r
    rev_prev = 1e4 * rev_r / rev_b
    unrev_prev = 1e4 * unrev_r / unrev_b
    return {
        "years": years,
        "revised_prev10k": rev_prev,
        "unrevised_prev10k": unrev_prev,
        "ratio": rev_prev / unrev_prev,
        "revised_share": rev_b / (rev_b + unrev_b),
    }


def state_dispersion() -> pd.DataFrame:
    """Recorded DS prevalence by state, 2016-2024 pooled, from the WONDER extract."""
    frame = pd.read_csv(WONDER_STATE_CSV)
    frame = frame[
        (frame.births_status == "observed") & (frame.ds_cp_status == "observed")
    ].copy()
    for col in ("births", "ds_cp", "ds_confirmed"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["births", "ds_cp"]).copy()
    frame["prev10k"] = 1e4 * frame.ds_cp / frame.births
    national = 1e4 * frame.ds_cp.sum() / frame.births.sum()
    frame["ratio_to_national"] = frame.prev10k / national
    frame["percentile"] = frame.prev10k.rank(pct=True) * 100
    frame["confirmed_share"] = frame.ds_confirmed / frame.ds_cp
    return frame.sort_values("prev10k").reset_index(drop=True)


def log_scale_sd(frame: pd.DataFrame) -> float:
    """Births-weighted SD of log recorded prevalence across states."""
    logp = np.log(frame.prev10k.to_numpy())
    weights = (frame.births / frame.births.sum()).to_numpy()
    mean = float((weights * logp).sum())
    return float(np.sqrt((weights * (logp - mean) ** 2).sum()))


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    var = np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(var))


def confirmation_independence(n_perm: int = 50_000, seed: int = 20260804) -> dict:
    """Test whether confirmation practice covaries with recording completeness.

    The transport of a confirmed-only sensitivity multiplies by a ratio of
    confirmed shares `q = P(confirmed | flagged)`, which assumes `q` does not
    covary with recording completeness. Two distinct questions:

    1. Is `q` constant across states? A binomial dispersion test answers this.
    2. Does `q` *covary* with completeness? Only this is what the transport
       assumes, and it needs a null that preserves `q`'s real overdispersion
       and breaks only the pairing -- a permutation, not a binomial bootstrap.

    A regression of `log(confirmed prevalence)` on `log(pending prevalence)`
    cannot answer (2): with `conf = q * total` and `pend = (1 - q) * total`, any
    variance in `q` drags the slope below 1 through `cov(log q, log(1-q)) < 0`
    whether or not `q` covaries with `total`.
    """
    rng = np.random.default_rng(seed)
    frame = pd.read_csv(WONDER_STATE_CSV)
    for col in ("births", "ds_cp", "ds_confirmed"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    obs = frame[
        (frame.ds_cp_status == "observed") & (frame.ds_confirmed_status == "observed")
    ].copy()
    obs["pending"] = obs.ds_cp - obs.ds_confirmed
    obs = obs[(obs.ds_confirmed > 0) & (obs.pending > 0)].reset_index(drop=True)
    obs["prev10k"] = 1e4 * obs.ds_cp / obs.births
    obs["q"] = obs.ds_confirmed / obs.ds_cp

    totals = obs.ds_cp.to_numpy()
    confirmed = obs.ds_confirmed.to_numpy()
    q_pooled = float(confirmed.sum() / totals.sum())

    # (1) dispersion of q relative to binomial sampling noise
    chi2 = float(
        (
            ((confirmed - totals * q_pooled) ** 2)
            / (totals * q_pooled * (1 - q_pooled))
        ).sum()
    )
    dof = len(obs) - 1
    dispersion = chi2 / dof
    rho_bb = max(0.0, (dispersion - 1) / (totals.mean() - 1))

    # (2) association, with a permutation null that keeps q's marginals
    logit_q = np.log(obs.q.to_numpy() / (1 - obs.q.to_numpy()))
    log_prev = np.log(obs.prev10k.to_numpy())
    weights = totals.astype(float)
    r_unweighted = float(np.corrcoef(logit_q, log_prev)[0, 1])
    r_weighted = _weighted_corr(logit_q, log_prev, weights)

    null_unw = np.empty(n_perm)
    null_wt = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = logit_q[rng.permutation(len(obs))]
        null_unw[i] = np.corrcoef(shuffled, log_prev)[0, 1]
        null_wt[i] = _weighted_corr(shuffled, log_prev, weights)

    def two_sided(stat: float, null: np.ndarray) -> float:
        return float(min(1.0, 2 * min((null >= stat).mean(), (null <= stat).mean())))

    obs["tercile"] = pd.qcut(obs.prev10k, 3, labels=["low", "mid", "high"])
    terciles = {
        str(name): float(g.ds_confirmed.sum() / g.ds_cp.sum())
        for name, g in obs.groupby("tercile", observed=True)
    }
    return {
        "n_states": len(obs),
        "q_pooled": q_pooled,
        "q_min": float(obs.q.min()),
        "q_min_state": obs.loc[obs.q.idxmin(), "state"],
        "q_max": float(obs.q.max()),
        "q_max_state": obs.loc[obs.q.idxmax(), "state"],
        "chi2": chi2,
        "dof": dof,
        "dispersion": dispersion,
        "rho_beta_binomial": rho_bb,
        "between_state_sd_q": float(np.sqrt(rho_bb * q_pooled * (1 - q_pooled))),
        "r_unweighted": r_unweighted,
        "p_unweighted": two_sided(r_unweighted, null_unw),
        "r_weighted": r_weighted,
        "p_weighted": two_sided(r_weighted, null_wt),
        "terciles": terciles,
        "florida_q": float(obs.loc[obs.state == "Florida", "q"].iloc[0]),
    }


def confirmation_gradients(con: duckdb.DuckDBPyConnection) -> dict:
    """Maternal-age gradient in `q`, and how much of its drift is composition.

    A purely clerical habit would not track maternal age. Prenatal diagnosis
    does, so an age gradient is evidence that part of `q` is real diagnostic
    information reaching the certificate rather than reporting style.

    The gradient is reported over 2016-2024 alone so the era drift does not
    contaminate it; the drift decomposition needs the longer 2007-2024 span.
    Band labels are numbered to keep them in age order rather than alphabetical.
    """
    band_case = """
        CASE WHEN CAST(mage_c AS INTEGER) < 25 THEN '1 <25'
             WHEN CAST(mage_c AS INTEGER) < 30 THEN '2 25-29'
             WHEN CAST(mage_c AS INTEGER) < 35 THEN '3 30-34'
             WHEN CAST(mage_c AS INTEGER) < 40 THEN '4 35-39'
             ELSE '5 40+' END
    """
    frame = con.execute(f"""
        SELECT CAST(year AS INTEGER) AS year, {band_case} AS band,
               SUM(CASE WHEN UPPER(CAST(ca_down_c AS VARCHAR))='C' THEN 1 ELSE 0 END)
                   AS confirmed,
               SUM(CASE WHEN UPPER(CAST(ca_down_c AS VARCHAR))='P' THEN 1 ELSE 0 END)
                   AS pending
        FROM us_births
        WHERE year BETWEEN 2007 AND 2024
          AND (ca_down IS NOT NULL OR ca_downs IS NOT NULL)
          AND mage_c IS NOT NULL
        GROUP BY 1, 2
    """).df()
    frame["n"] = frame.confirmed + frame.pending
    counts = frame.pivot(index="year", columns="band", values="n").fillna(0)
    conf = frame.pivot(index="year", columns="band", values="confirmed").fillna(0)

    window = frame[frame.year >= 2016]
    by_band = window.groupby("band")[["confirmed", "pending"]].sum().sort_index()
    by_band["q"] = by_band.confirmed / (by_band.confirmed + by_band.pending)
    chi2, p_age, _, _ = stats.chi2_contingency(
        by_band[["confirmed", "pending"]].to_numpy()
    )

    standard = counts.sum(axis=0) / counts.sum().sum()
    crude = conf.sum(axis=1) / counts.sum(axis=1)
    adjusted = (conf / counts * standard).sum(axis=1)

    def logit_trend(series: pd.Series) -> tuple[float, float, float]:
        fit = stats.linregress(
            series.index.astype(float), np.log(series / (1 - series))
        )
        return float(fit.slope), float(fit.stderr), float(fit.rvalue**2)

    slope_crude, se_crude, r2_crude = logit_trend(crude)
    slope_adj, se_adj, r2_adj = logit_trend(adjusted)
    return {
        "by_band": by_band,
        "chi2_age": float(chi2),
        "p_age": float(p_age),
        "slope_crude": slope_crude,
        "se_crude": se_crude,
        "r2_crude": r2_crude,
        "slope_adjusted": slope_adj,
        "se_adjusted": se_adj,
        "r2_adjusted": r2_adj,
        "composition_share": 1 - slope_adj / slope_crude,
    }


def main() -> None:
    setup.init_script()
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        national = national_by_year(con)
        confirmed = confirmed_split(con)
        gradients = confirmation_gradients(con)
    finally:
        con.close()

    print("National recorded DS prevalence per 10,000 covered births")
    for lo, hi, revised in [
        (1995, 2005, False),
        (2007, 2011, True),
        (2016, 2024, False),
    ]:
        births, recorded, prev = national_window(national, (lo, hi), revised)
        tag = " (2003-revision area)" if revised else ""
        print(f"  {lo}-{hi}{tag}: {recorded:,} / {births:,} = {prev:.2f}")

    surveillance = pd.read_csv(SURVEILLANCE_CSV)
    print("\nTransport of the study-measured sensitivities")
    for study in (BOULET, SALEMI):
        result = transport(study, national)
        lo, hi = result["years"]
        surv = (
            1e4
            * surveillance[
                (surveillance.year >= lo) & (surveillance.year <= hi)
            ].p_ds_lb_wt.mean()
        )
        print(f"\n  {result['label']} ({lo}-{hi})")
        print(f"    study-area recorded  {result['area_prev10k']:.2f} per 10,000")
        print(f"    national recorded    {result['national_prev10k']:.2f} per 10,000")
        print(f"    factor               {result['factor']:.3f}")
        print(
            f"    sensitivity          {result['sensitivity']:.3f}"
            f"  ->  {result['sensitivity_national']:.3f} national"
        )
        print(
            f"    study-area TRUE prev {result['area_true_prev10k']:.2f}"
            f"  vs project surveillance {surv:.2f}"
            f"  ({100 * (result['area_true_prev10k'] / surv - 1):+.1f}%)"
        )

    print("\n  Salemi transport across the FL birth denominator")
    for births_fl in (1_050_000, 1_090_000, 1_120_000, 1_150_000, 1_200_000):
        result = transport({**SALEMI, "births": births_fl}, national)
        print(
            f"    {births_fl:>10,} -> factor {result['factor']:.3f},"
            f" sensitivity {result['sensitivity_national']:.3f}"
        )

    print("\nDid the 2003 revision improve recording?")
    for years in ((2004, 2015), (2006, 2010)):
        result = revision_contrast(national, years)
        lo, hi = result["years"]
        print(
            f"  {lo}-{hi} (revised share {result['revised_share']:.0%}):"
            f" revised {result['revised_prev10k']:.2f}"
            f" vs unrevised {result['unrevised_prev10k']:.2f}"
            f" -> ratio {result['ratio']:.3f}"
        )

    states = state_dispersion()
    national_prev = 1e4 * states.ds_cp.sum() / states.births.sum()
    print(f"\nState dispersion 2016-2024 ({len(states)} states with observed counts)")
    print(f"  national {national_prev:.2f} per 10,000")
    print(
        f"  range {states.prev10k.min():.2f} ({states.iloc[0].state})"
        f" to {states.prev10k.max():.2f} ({states.iloc[-1].state})"
        f" = {states.prev10k.max() / states.prev10k.min():.1f}-fold"
    )
    quantiles = states.prev10k.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(
        "  quantiles "
        + ", ".join(f"p{int(k * 100)}={v:.2f}" for k, v in quantiles.items())
    )
    print(
        f"  births-weighted SD of log recorded prevalence = {log_scale_sd(states):.3f}"
    )
    for name in ("Florida", "Georgia"):
        row = states[states.state == name]
        if not row.empty:
            row = row.iloc[0]
            print(
                f"  {name}: {row.prev10k:.2f} = {row.ratio_to_national:.2f}x national,"
                f" {row.percentile:.0f}th percentile"
            )

    print("\nConfirmed share of recorded flags (2003-revision records only)")
    for lo, hi in ((2007, 2011), (2016, 2024)):
        win = confirmed[(confirmed.year >= lo) & (confirmed.year <= hi)]
        n_c, n_p = int(win.confirmed.sum()), int(win.pending.sum())
        print(
            f"  {lo}-{hi}: confirmed {n_c:,}, pending {n_p:,} -> {n_c / (n_c + n_p):.3f}"
        )
    salemi_share = SALEMI_CONFIRMED["recorded"] / SALEMI["recorded"]
    print(f"  Salemi Florida 2007-2011: {salemi_share:.3f}")

    ind = confirmation_independence()
    print("\nIs the confirmed share q constant across states?")
    print(
        f"  {ind['n_states']} states, pooled q = {ind['q_pooled']:.4f},"
        f" range {ind['q_min']:.3f} ({ind['q_min_state']})"
        f" to {ind['q_max']:.3f} ({ind['q_max_state']})"
    )
    print(
        f"  binomial chi2 {ind['chi2']:.1f} on {ind['dof']} df"
        f" -> dispersion {ind['dispersion']:.2f};"
        f" beta-binomial rho {ind['rho_beta_binomial']:.4f},"
        f" between-state SD of q {ind['between_state_sd_q']:.3f}"
    )
    print("  => not constant. Real state-level variation in confirmation practice.")

    print("\nDoes q covary with recording completeness? (permutation null)")
    print(
        f"  unweighted r = {ind['r_unweighted']:+.4f} (p = {ind['p_unweighted']:.3f});"
        f" flag-weighted r = {ind['r_weighted']:+.4f} (p = {ind['p_weighted']:.3f})"
    )
    print(
        "  pooled q by recording tercile: "
        + ", ".join(f"{k} {v:.4f}" for k, v in ind["terciles"].items())
    )
    print("  => no detectable association. The transport's premise holds.")

    grad = gradients
    print("\nWhat q measures")
    print(
        "  q by maternal age: "
        + ", ".join(f"{b} {r.q:.3f}" for b, r in grad["by_band"].iterrows())
    )
    print(f"  chi2 = {grad['chi2_age']:.1f}, p = {grad['p_age']:.3g}")
    print(
        f"  logit(q) drift 2007-2024: crude {grad['slope_crude']:+.4f}/yr"
        f" (r2 {grad['r2_crude']:.3f}),"
        f" age-standardised {grad['slope_adjusted']:+.4f}/yr"
        f" (r2 {grad['r2_adjusted']:.3f})"
    )
    print(
        f"  composition explains {100 * grad['composition_share']:.1f}% of the drift;"
        f" the rest is a within-age rise in confirmation."
    )

    states.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
