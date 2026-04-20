# Selection model — reporting-profile variant sweep: findings

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from an
> autonomous session on 2026-04-20. Figures below reflect the four
> reporting-profile fits completed at 18:27–21:42 on 2026-04-20. A
> human reviewer should verify the interpretation and decide on the
> next design iteration before these numbers are cited anywhere.

## TL;DR

The plumbing works. The posteriors don't.

Four variants (A/B/C/D) × `--profile reporting` × `--spec full` ran
cleanly in ~3.5 h total on nutpie. All four produced the complete
artefact set (idata.nc, summary.csv, six diagnostic plots + tables,
rendered Quarto HTML). Convergence ranged from clean (A) to marginal
(B). The cross-variant identifiability diagnostic reports all
race-level panels as **data-informed** (|r| ≤ 0.3 everywhere).

**But the posterior intercepts are wildly inconsistent with the
priors.** θ_LB sits 10–15σ above Morris, `s_int` sits 6–10σ below
Boulet, and `eta_detect_int` sits ~6σ below its prior. The model is
making the data fit by pulling the slowest-σ prior (Morris, σ=0.10)
around; with 33.5 M rows, even σ=0.10 is not tight in any useful
sense. The implied "4% birth-certificate sensitivity" is not credible.

Design invariant #1 (Morris priors stay tight) is empirically
**broken** under the current model+priors combination. Before any of
these posteriors support inference, the model needs redesign.

## Fit log

| variant | output dir | runtime |
|---|---|---:|
| A | `output/selection/A/full/20260420-182719/` | ~52 min |
| B | `output/selection/B/full/20260420-191926/` | ~49 min |
| C | `output/selection/C/full/20260420-200804/` | ~47 min |
| D | `output/selection/D/full/20260420-205506/` | ~47 min |

Total wall-clock ≈ 3.5 h. `idata.nc` ~1 GB per variant (as expected
after the Task 4.2a Deterministic trim). All four HTML reports
rendered successfully via the `--render` flag.

## Convergence

| variant | n RVs | max R̂ | min ESS bulk | min ESS tail | gates |
|---|---:|---:|---:|---:|---|
| A | 79 | 1.000 | 806 | 1559 | **PASS** |
| B | 79 | 1.030 | 201 | 298 | **FAIL** (R̂ > 1.01, ESS < 400) |
| C | 79 | 1.020 | 491 | 973 | **FAIL** (R̂ > 1.01) |
| D | 79 | 1.010 | 374 | 635 | **MARGINAL** (R̂ = 1.01, ESS just below 400) |

Variant B's min ESS 201 is the standout — likely a handful of highly
correlated RVs the sampler couldn't separate. A cleaner rerun at
`--target-accept 0.98` might get B to gate-passing, but given the
structural findings below, further B-specific tuning isn't the
priority.

## Identifiability pair-plot (race panels)

All six race panels are below the |r| = 0.7 "prior-driven" threshold
in every variant:

|   | A | B | C | D |
|---|---:|---:|---:|---:|
| NH White | 0.160 | 0.019 | 0.068 | 0.130 |
| NH Black | 0.202 | 0.036 | 0.081 | 0.126 |
| NH AIAN/NHOPI/Other | 0.103 | 0.012 | 0.085 | 0.185 |
| NH Asian | 0.280 | 0.055 | 0.147 | 0.278 |
| Hispanic | 0.164 | 0.018 | 0.049 | 0.149 |
| Unknown | 0.230 | 0.047 | 0.128 | 0.226 |

Read naively this says the decomposition is data-informed. Read more
carefully alongside the intercept-level findings below, it says the
diagnostic **doesn't look at the right place**: low |r| at the race
level coexists with massive prior violations at the intercept level,
because the model is making the data fit by moving intercepts far from
their anchors rather than by trading race-level effects between η_term
and s.

## Race effects shift with variant

`eta_term_race` posterior means:

|   | A (tight s) | B (tight η_term) | C (default) | D (Dobbs-only) |
|---|---:|---:|---:|---:|
| NH White | −0.74 | −0.07 | −0.27 | −1.50 |
| NH Black | −1.04 | −0.73 | −0.80 | −1.65 |
| NH AIAN/NHOPI/Other | −0.56 | −0.30 | −0.32 | −0.82 |
| NH Asian | +0.53 | −0.12 | +0.01 | +0.84 |
| Hispanic | −0.57 | −0.42 | −0.46 | −0.63 |
| Unknown | −0.21 | −0.01 | −0.02 | −1.06 |

`s_race` posterior means:

|   | A | B | C | D |
|---|---:|---:|---:|---:|
| NH White | −0.03 | −0.75 | −0.23 | −0.29 |
| NH Black | −0.76 | −1.53 | −0.99 | −1.01 |
| NH AIAN/NHOPI/Other | +0.05 | −0.30 | +0.13 | +0.12 |
| NH Asian | −0.66 | −1.71 | −1.11 | −0.97 |
| Hispanic | +0.01 | −0.76 | −0.23 | −0.23 |
| Unknown | −0.13 | −0.98 | −0.41 | −0.43 |

Variant A (tight priors on s) pushes demographic variation into
η_term; Variant B (tight priors on η_term) pushes it into s. The
between-variant spread is substantial — this is the sensitivity
analysis working as intended, but the spread itself says the data
cannot pin down the decomposition without external information.

## Intercept-level prior violations

This is the key finding. Prior means/sigmas vs posterior means, on the
logit scale, across all four variants:

| RV | prior μ | prior σ | A | B | C | D | max |z| |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eta_detect_int` | +0.85 | 0.30 | −0.53 | −0.93 | −0.84 | −0.25 | **5.9** |
| `eta_term_int` | +0.71 | 0.25 | +0.30 | +0.17 | +0.21 | +0.40 | **2.1** |
| `s_int` | −0.41 | 0.30 | **−3.40** | −2.23 | −3.08 | −3.02 | **10.0** |

The `theta_lb_age` posterior means per age band are **1–1.5 logit
units above Morris** — with a σ=0.10 prior, that's a 10–15σ shift.

Translating to probability scale for variant C:
- `eta_detect`: prior 0.70 → posterior 0.30 (halved)
- `eta_term`: prior 0.67 → posterior 0.55 (modestly down)
- `s`: prior 0.40 → **posterior 0.044** (9× lower)
- `theta_lb(30–34)`: Morris 1.48/1000 → **posterior 5.52/1000** (3.7× higher)

Combined, these imply posterior:
- True DS livebirths (total, 2016–2024): **~231k** across all variants.
- Recorded: **17,776** (data).
- Implied **93–96% of true DS livebirths missed** per race.
- Implied birth-certificate sensitivity: **4–7%**.

Neither number is compatible with the literature (Boulet ≈ 40%
sensitivity; published under-ascertainment ≈ 60%, i.e. ~40% recorded).

## CCHD consistency check is conceptually broken

Posterior CCHD prevalence among true DS livebirths: **0.1%** across
all four variants. EUROCAT target: ~22.5%.

**This is by construction**, not a fit failure. The model's design
invariant #2 says clinical features (CCHD, NICU, Aven, Preterm) enter
only `s`. The consequence: per cell, `theta_LB × eta` depends only on
maternal factors (age, race, education, payer, year). CCHD is
statistically independent of "being a DS livebirth" in the model,
conditional on maternal factors. So posterior CCHD prevalence in
model-true DS livebirths just returns the **underlying CCHD base rate
in the population**, which is ~0.4% of livebirths.

The `cchd_consistency_check` diagnostic was comparing the model's
inescapable answer ("CCHD ≈ base rate") against epidemiological
reality ("22.5% of DS livebirths have CCHD because DS causes CCHD").
The model has **no mechanism** to produce 22.5% as long as clinical
markers enter only `s`.

Either invariant #2 is wrong (CCHD should be allowed to influence
whether a cell contains DS, not only whether DS is recorded), or the
diagnostic should be removed and replaced with something the model
can actually test.

## Why this happened

Two reinforcing causes.

1. **Massive likelihood vs the declared priors.** With 33.5 M rows and
   17,776 recorded DS, Fisher information at the intercepts is
   ~10⁵ on logit, dwarfing prior precision (1/σ²) of 100 for Morris
   and ~10 for the η/s intercepts. Prior → data ratio ≈ 10⁻³.
   "Tight" σ=0.10 priors behave like data points in this size regime.

2. **Identification at the intercept level between θ_LB and s.**
   The likelihood `p_recorded = θ_LB · η · s + (1 − θ_LB·η) · f` can
   be matched to the observed rate by many combinations: high θ_LB /
   low s vs Morris θ_LB / Boulet s. The identifiability pair-plot
   checks race-level residuals only, so this intercept-scale
   indeterminacy is invisible to it. The posterior lands on
   "θ_LB 3–4× Morris, s 4%" because the tightest prior (Morris,
   σ=0.10) is still an order of magnitude looser than the data signal
   — the model takes any movement room it can get.

## What to do next (decisions for a human)

These are for discussion, not action:

1. **Reinterpret σ=0.10 as "tight"**. It isn't, at this data scale. A
   defensible Morris prior at N=33M rows is probably σ=0.01 or even
   σ=0.001 on logit, or a non-Gaussian prior that genuinely forbids
   movement (e.g. a pinned deterministic at Morris with residual
   Normal on log-multiplicative deviations).

2. **Reconsider invariant #2** (clinical features enter only s).
   Epidemiologically, CCHD is caused by DS ~50% of the time; the
   model's "CCHD independent of DS-status" assumption is the opposite
   direction. This might be worth relaxing in a v2 that explicitly
   models DS-conditional clinical features.

3. **Add an intercept-level identifiability diagnostic.** The current
   pair-plot looks at race panels. Add a joint scatter of
   `theta_lb_age[*]` vs `s_int` across posterior draws so the
   intercept-scale collinearity that happened here would have been
   visible diagnostically, not just in post-hoc sigma counts.

4. **Ground-truth check with surveillance.** Published DS surveillance
   in the US says ~1 DS livebirth per 700–800 births (~1.25e-3 to
   1.4e-3), which is higher than raw recorded (5.3e-4) by ~2.5×, not
   13×. Any v2 model should calibrate its total DS livebirth estimate
   against this published number — 231k over 2016–2024 (~26k/year,
   ~6e-4/birth) is incompatible with ~42k/year that surveillance
   implies.

5. **Consider constraining `f` upward.** Ohio/NY's 7.8e-5 FP rate is
   small, but if FP is actually larger (say 2e-4, matching a 35%
   false-positive fraction on observed 5.3e-4), the non-DS branch of
   the likelihood explains more of the observed rate and less needs
   to come through `θ_LB · η · s`. A prior-sensitivity run at, say,
   `f = 2e-4` would quantify this.

## Practical finding for the PR

The PR's infrastructure goal (build + fit + diagnose four variants,
produce a comparison table + forest plot, render Quarto) is **met**.
The scientific result is that the model, as specified, is not
appropriate for inference on this data scale. The variant sweep
correctly flagged this — that is what sensitivity analysis is for —
but the mechanism it surfaced is at the intercept level rather than
the race level the diagnostic was designed to catch.

`scripts/compare_selection_variants.py` produced
`output/selection/_compare_reporting_20260420/comparison.csv` and
`comparison_forest.png`; those figures show the between-variant
spread cleanly. They are not suitable for a publication figure
pending the v2 redesign.

## Artefacts

```
output/selection/
├── A/full/20260420-182719/    idata.nc, summary.csv, index.html, plots/, tables/
├── B/full/20260420-191926/    (same)
├── C/full/20260420-200804/    (same)
├── D/full/20260420-205506/    (same)
├── _compare_reporting_20260420/
│       comparison.csv, comparison_forest.{png,svg}
└── _run_logs/
        batch_20260420-182718.log
        20260420-182719_{A,B,C,D}.log
```

All artefacts under `output/` are gitignored per NCHS DUA
considerations; this note is the canonical record.
