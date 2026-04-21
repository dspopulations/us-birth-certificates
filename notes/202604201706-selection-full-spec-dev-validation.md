# Selection model — dev-profile full-spec validation

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from an
> autonomous session on 2026-04-20. Figures and claims below reflect a
> single dev-profile fit; a human reviewer should verify before citing.
> A reporting-profile replication is required before any substantive
> interpretation.

## Context

End-to-end verification of the Phase 4/5 selection-model pipeline: one
full-spec fit on real data (variant C, dev profile) to confirm that the
identifiability pair-plot and year-effect trajectory — the two
diagnostics that `theta_only` cannot produce — render correctly with
non-degenerate posterior content.

This is **not** a reporting-quality run. Its job is to validate the
plumbing, not to support inference.

> [!NOTE]
> This note was written against an earlier model specification that
> carried a heteroscedastic pre/post-2022 sigma on `eta_term_year`.
> That asymmetric year treatment has since been removed in favour of a
> single homoscedastic year sigma. The year-effect findings below
> reflect the old specification and should not be read as current
> results.

## Setup

```
python scripts/fit_selection_model.py \
    --variant C --spec full --profile dev \
    --output-dir output/selection/C/full/dev_validation
```

- `variant_C_default()` priors (Morris / Natoli / Kuppermann / Boulet
  default sigmas).
- `spec="full"` — θ_LB + η_detect · η_term + s with per-cell clinical
  markers on s.
- `dev` profile — 400 tune + 400 draws × 2 chains, `target_accept=0.9`,
  `nuts_sampler="nutpie"`.
- Real-data cells: 60,057 aggregated from 33,498,266 livebirths
  (17,776 recorded DS) across 2016–2024.

Wall-clock: ~44 min (30 min nutpie compile + NUTS; 14 min `az.summary`
over per-cell deterministics).

## Convergence

| quantity | value | target |
|---|---:|---|
| max R̂ across **79 named RVs** | 1.02 | < 1.01 |
| min ESS bulk across RVs | 100 | ≥ 400 |
| min ESS tail across RVs | 188 | ≥ 400 |

Dev profile does not clear the reporting-quality gates, as expected. A
reporting-profile rerun (1500 × 4, `target_accept=0.95`) is required
before quoting point estimates.

## Identifiability

All six race panels of the `eta_term_race` × `s_race` pair-plot are
well below the |r| = 0.7 prior-driven threshold:

| race | \|r\| | interpretation |
|---|---:|---|
| NH White | 0.02 | data-informed |
| NH Black | 0.06 | data-informed |
| NH AIAN/NHOPI/Other | 0.04 | data-informed |
| NH Asian | 0.13 | data-informed |
| Hispanic | 0.08 | data-informed |
| Unknown | 0.15 | data-informed |

The clinical-marker → s channel (CCHD / NICU / Aven / Preterm) plus the
year-level signal appear to identify the decomposition even without
state-level contrast. This is a **positive** finding for the no-region
model (plan §2.2).

## Headline race effects (logit scale, NH White = 0)

`eta_term_race` (termination):

| race | mean | 94% HDI |
|---|---:|---|
| NH White | −0.26 | [−0.61, +0.06] |
| NH Black | −0.81 | [−1.18, −0.46] |
| NH AIAN/NHOPI/Other | −0.33 | [−0.69, +0.05] |
| NH Asian | +0.00 | [−0.36, +0.42] |
| Hispanic | −0.47 | [−0.84, −0.14] |
| Unknown | −0.02 | [−0.36, +0.34] |

`s_race` (BC sensitivity):

| race | mean | 94% HDI |
|---|---:|---|
| NH White | −0.22 | [−0.41, −0.05] |
| NH Black | −0.98 | [−1.18, −0.79] |
| NH AIAN/NHOPI/Other | +0.13 | [−0.17, +0.39] |
| NH Asian | −1.10 | [−1.33, −0.90] |
| Hispanic | −0.22 | [−0.43, −0.05] |
| Unknown | −0.41 | [−0.66, −0.19] |

Qualitatively consistent with the priors' direction (Kuppermann on
termination for NH Black / Hispanic, Boulet on sensitivity for NH Black
/ "other non-Hispanic") but posterior magnitudes are larger than the
prior means in several places — the data is pulling on the decomposition.

## Operational finding — idata.nc size

`idata.nc` is **4.2 GB** for this dev-profile run. Per-cell
deterministics (`theta_lb`, `eta_detect`, `eta_term`, `eta`, `s`,
`p_ds_lb`, `p_recorded` — all `dims="cell"` at 60,057 cells) drive this.

A reporting-profile run (1500 × 4 = 6000 posterior draws, vs 800 here)
would extrapolate to **~30 GB**. This is impractical on any machine
with spinning disk or modest SSD, and kills interactive Quarto rendering.

**Recommendation (follow-up):** before running the reporting sweep,
demote most per-cell values from `pm.Deterministic` to inline tensor
expressions. Keep only `p_ds_lb` and `p_recorded` (used by multiple
diagnostics across the rendering loop). That cuts the saved InferenceData
by roughly 5/7 — ≈ 9 GB at reporting. Still big but tractable.

## What this validates

- ✅ End-to-end fit pipeline (`scripts/fit_selection_model.py`) works on
  real data with full spec.
- ✅ All six diagnostics render (identifiability + year trajectory,
  which were not reachable under `theta_only`).
- ✅ Deduplicated render pipeline (commit `2a9da20`) correctly reuses
  the fit-CLI-written `summary.csv` on post-hoc re-rendering.
- ✅ Quarto template at `docs/models/selection/index.qmd` is copied
  into the run directory; manual `quarto render` should work.
- ✅ Identifiability diagnostic produces non-degenerate |r| values;
  decomposition is data-informed even without region.

## What still needs doing

- **Reporting-profile run** (plan Task 4.3) for publishable numbers on
  convergence and race effects. Expect 4–8 h per variant on the dev
  profile scale (longer at reporting without the idata trim).
- **idata size trim** before the reporting sweep.
- **Variant sweep A/B/C** (plan Task 4.3–4.4) to run the sensitivity
  comparison in `scripts/compare_selection_variants.py`.
- **State-level data** if and when it becomes available — a state-level
  contrast would let the model separate detection-year from
  termination-year drift.

## Artefacts

Under `output/selection/C/full/dev_validation/` (gitignored):

```
idata.nc                 4.2 GB
summary.csv              27 MB  (az.summary on 180k+ variables)
cells.parquet            244 KB
config.json, run_config.json, index.qmd
plots/                   18 files — 6 diagnostics × (png, svg) +
                         5 CSV companions
tables/                  5 CSV tables
```

The `identifiability.csv` and (under the old specification) a
year-trajectory table drove the analysis above. Newer runs have
`eta_term_year_trajectory.csv` in place of the old artefact.
