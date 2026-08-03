> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP004 coherent surveillance-calibration sensitivity

**Date:** 2026-08-03

**Status:** Implemented and fitted. The scenario grid and decision rules below
were fixed before the posterior fits were run.

## Question

The existing `DSP004` surveillance sensitivity widens each year's combined-
reduction prior independently. Independent errors can partly average out when
annual totals are combined. That analysis therefore does not test a calibration
error shared across surveillance years.

This follow-up asks how the population-level `DSP004` total and its recording
decomposition change when annual surveillance-prior errors are correlated or
when the complete prior trajectory is shifted on the logit scale. These are
controlled sensitivity scenarios, not additional fitted model identities and
not attempts to let the certificate data estimate surveillance bias.

## Fixed analysis scope

Every scenario retains the preferred simple-resolution `DSP004` structure and
the same analytic cohort, maternal-age curve, yearly combined-reduction
anchors, constant recording sensitivity, and sampling/reporting rules. The
following inputs are fixed throughout the primary grid:

- false-positive probability `f=7.8e-5` per non-DS birth;
- observed-year reduction-prior logit SD `0.20`;
- extrapolated-year reduction-prior logit SD `0.45`;
- the current 2020 start of the extrapolated period; and
- the confirmed-or-pending recorded-case definition.

Holding these choices fixed isolates the two new calibration controls. It does
not make the fixed values known or costless.

## Calibration parameterisation

Let $\mu_y$ be the existing surveillance-derived logit reduction centre and
$\sigma_y$ its existing logit SD. The sensitivity prior is

$$
\operatorname{logit}(\rho_y)
= \mu_y + \delta + \varepsilon_y,
\qquad
\boldsymbol{\varepsilon}\sim
\mathcal{N}(\boldsymbol{0},\boldsymbol{\Sigma}_\lambda),
$$

with

$$
(\Sigma_\lambda)_{ij}
= \sigma_i\sigma_j\left[(1-\lambda)\mathbb{1}(i=j)+\lambda\right].
$$

Thus each diagonal remains $\sigma_y^2$: changing $\lambda$ changes only the
correlation of annual logit-scale errors, not any year's marginal prior
variance. The implementation can equivalently use a Cholesky factor
$L_\lambda L_\lambda^\mathsf{T}=\Sigma_\lambda$ and independent standard-normal
$z$, setting $\varepsilon=L_\lambda z$. An equivalent shared-factor expression
is

$$
\varepsilon_y=\sigma_y\left(\sqrt{\lambda}\,b
+\sqrt{1-\lambda}\,e_y\right),
$$

where $b$ and the $e_y$ values are independent standard-normal errors. The
Cholesky form is the implementation used by the fitting code.

The controls are exposed as:

- CLI `--reduction-error-correlation`, stored as
  `reduction_error_correlation` ($\lambda$); and
- CLI `--reduction-calibration-shift-logit`, stored as
  `reduction_calibration_shift_logit` ($\delta$).

At $\lambda=0$, yearly errors are independent, reproducing the current prior.
Positive $\lambda$ makes errors co-move. The fixed $\delta$ translates every
annual logit prior centre by the same amount; it is not estimated from the
certificate likelihood.

## Pre-specified primary scenarios

The seven primary scenarios vary one axis at a time. All use `f=7.8e-5` and
reduction-prior logit SDs `0.20 / 0.45`.

| Scenario | $\lambda$ | $\delta$ | Purpose |
| --- | ---: | ---: | --- |
| Baseline | `0` | `0` | Existing independent, unshifted DSP004 prior |
| Moderate correlation | `0.5` | `0` | Errors share half their standardised variance |
| Strong correlation | `0.9` | `0` | Errors strongly co-move while retaining marginal variances |
| Negative shift, larger | `0` | `-0.4` | Directional common-level stress |
| Negative shift, smaller | `0` | `-0.2` | Directional common-level stress |
| Positive shift, smaller | `0` | `+0.2` | Directional common-level stress |
| Positive shift, larger | `0` | `+0.4` | Directional common-level stress |

The four $\delta$ values are deliberately symmetric stress values. They are
**not externally validated bounds**, do not define a probability distribution
for calibration bias, and must not be presented as a confidence or credible
range.

## Sampling and posterior-quality gates

Every primary and conditional joint-corner fit will use four chains, 3,000
tuning iterations and 3,000 retained draws per chain, `target_accept=0.95`, and
the `nutpie` sampler. Posterior-predictive generation and reporting settings
will be matched across scenarios.

A scenario is eligible for interpretation only if it has:

- zero divergences;
- maximum unrounded Rhat below `1.01` across the monitored variables;
- minimum bulk effective sample size of at least 400; and
- minimum tail effective sample size of at least 400.

The Monte Carlo precision of the decision contrasts will also be checked. If a
materiality comparison lies within two combined Monte Carlo standard errors
(MCSEs) of its pre-specified threshold, increase the retained draws before
classifying it as material or immaterial. Here the combined MCSE is the MCSE of
the scenario-minus-baseline contrast, using independent-run propagation where
needed; this rule is not a reason to move the materiality threshold.

## Estimands and pre-specified materiality rules

Each primary scenario is compared with the baseline using posterior summaries
calculated in the same way and at the same interval probability.

The aggregate estimand is the total number of true DS livebirths over the
analysis period. An aggregate change is material if either:

1. the absolute change in its posterior mean is at least 5% of the baseline
   posterior mean; or
2. its 89% equal-tailed interval (ETI) width increases by at least 25% relative
   to the baseline width.

Two decomposition estimands are also pre-specified:

- the posterior mean of constant recording sensitivity $s$, material if its
  absolute change from baseline is at least `0.05`; and
- model-implied missed true cases, calculated draw by draw as the true total
  minus true-positive recorded cases. For constant-$s$ `DSP004`, this is
  $M^{(d)}=T^{(d)}[1-s^{(d)}]$. Its change is material if the absolute change
  in posterior mean is at least 10% of the baseline posterior mean.

The missed-case estimand is an aggregate model-implied count. It does not label
which births were missed, and it must not be calculated as the true total minus
all observed DS flags because the observed flags include the fixed
false-positive branch.

The decomposition rules diagnose whether an apparently stable total conceals a
materially different allocation between true-positive recording and missed
true cases. They do not by themselves trigger the joint-corner runs below.

## Conditional joint-corner rule

If either aggregate materiality rule is triggered by any primary scenario, fit
two additional joint corners:

| Joint corner | $\lambda$ | $\delta$ |
| --- | ---: | ---: |
| Strong correlation, larger negative shift | `0.9` | `-0.4` |
| Strong correlation, larger positive shift | `0.9` | `+0.4` |

For the seven separate-axis scenarios, define a descriptive envelope using the
minimum and maximum total posterior means and the outermost lower and upper 89%
ETI endpoints. Check whether either joint corner's corresponding total summary
falls outside that envelope. Report the same comparison for the two
decomposition posterior means. This check tests whether combining strong
correlation with a large level shift produces behaviour not exposed by either
axis alone.

The scenario envelope is **not a posterior interval**: it has no posterior
coverage probability. Draws must not be pooled across scenarios, and the
scenarios must not be model-averaged without a separately justified probability
model and weights.

## Interpretation limits

- The exercise concerns aggregate population accounting only. It cannot
  identify individual missed cases or individual false flags.
- $\lambda$ is a deliberately simple equicorrelation stress. It does not claim
  to recover the actual covariance structure of surveillance error.
- $\delta$ tests a coherent level displacement and leaves the logit-scale
  differences between yearly prior centres unchanged. Neither $\lambda$ nor
  $\delta$ introduces or tests an alternative extrapolated-tail slope.
- Consequently, stability across this grid would support robustness only to
  the pre-specified correlation and level stresses. It would **not establish
  trend robustness**, validate the 2020-2024 extrapolation, or answer how a
  different extrapolated-tail slope changes the result.
- The certificate likelihood cannot cleanly distinguish surveillance
  calibration, combined reduction, and recording sensitivity. These controls
  remain externally chosen scenarios rather than data-estimated corrections.

## Reporting decision

Report every fitted scenario conditionally, including convergence diagnostics,
the aggregate materiality comparisons, and both decomposition checks. If no
aggregate rule triggers, retain the baseline as the conditional headline and
describe the limits of the tested grid. If an aggregate rule triggers, show the
separate-axis envelope and the conditional joint-corner check, but do not turn
their union into a wider posterior interval or an averaged headline estimate.

## Implementation and prior checks

The fitted implementation uses a Cholesky-whitened annual error vector. An
exact covariance test verifies that its diagonal is the existing
$\sigma_y^2$ and that every off-diagonal standardised correlation is
$\lambda$. The `lambda=0` branch retains the original independent Normal model
graph. The reporting tables retain the unshifted surveillance value separately
from the shifted logit-normal prior centre.

A seeded (`20260803`) 200,000-draw prior simulation confirms the intended
aggregate effect for the unshifted correlation scenarios:

| $\lambda$ | Prior mean total | Prior 89% interval | Interval width |
| ---: | ---: | ---: | ---: |
| `0` | 45,361 | 42,032-48,570 | 6,539 |
| `0.5` | 45,357 | 38,282-52,045 | 13,763 |
| `0.9` | 45,349 | 36,228-53,790 | 17,562 |

The near-identical prior means and increasing interval widths show that
correlation changes joint uncertainty without changing the annual marginal
centres or SDs. These simulated intervals describe the model prior, not
external confidence limits for surveillance calibration.

A fresh unshifted, independent baseline reproduces the earlier `DSP004` fit.
Its total differs from the earlier posterior mean by 26 births and its mean
$s$ differs by `0.00018`. Previously calculated combined Monte Carlo standard
errors (MCSEs) were 69 births and `0.00055`, respectively, but those values use
independent-run propagation. Because the posterior fits reused random seed 47,
that propagation is descriptive rather than a defensible formal test of the
between-run differences.

## Posterior quality

All nine required fits used the pre-specified 4 chains, 3,000 tuning iterations
and 3,000 retained draws per chain. Every fit had zero divergences, maximum
unrounded Rhat below `1.006`, and minimum bulk/tail effective sample size above
1,000. Those original fits all reused random seed 47, so a blanket between-run
two-combined-MCSE claim based on independent-error propagation is not
justified. The seven primary-scenario materiality conclusions and the main
positive-corner interaction are nevertheless far enough from their decision
thresholds to remain unchanged under a conservative dependence bound.

The original negative joint-corner result was borderline under that bound and
was therefore not classified from the shared-seed run. A refined rerun used
independent seed 83, four chains, 3,000 tuning iterations and 6,000 retained
draws per chain. It had zero divergences, maximum unrounded Rhat below `1.001`, and
minimum effective sample size 5,185. The exact mean-based MCSE check is reported
with the joint-corner results below.

## Primary aggregate results

| Scenario | $\lambda$ | $\delta$ | Model-implied true DS livebirths, mean (89% ETI) | Mean change | ETI-width change | Aggregate rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | `0` | `0` | 44,255 (41,934-46,565) | reference | reference | No |
| Moderate correlation | `0.5` | `0` | 42,469 (39,211-45,808) | -4.04% | +42.43% | **Yes: width** |
| Strong correlation | `0.9` | `0` | 40,883 (38,859-42,921) | -7.62% | -12.28% | **Yes: mean** |
| Negative shift, larger | `0` | `-0.4` | 50,190 (48,178-52,177) | +13.41% | -13.67% | **Yes: mean** |
| Negative shift, smaller | `0` | `-0.2` | 47,390 (45,108-49,512) | +7.09% | -4.92% | **Yes: mean** |
| Positive shift, smaller | `0` | `+0.2` | 41,056 (38,661-43,480) | -7.23% | +4.05% | **Yes: mean** |
| Positive shift, larger | `0` | `+0.4` | 37,610 (35,023-40,262) | -15.02% | +13.13% | **Yes: mean** |

The separate-axis primary envelope spans posterior means of 37,610-50,190 and
outer 89% ETI endpoints of 35,023-52,177. This is a sensitivity envelope, not a
credible interval.

The correlation result needs careful interpretation. Before conditioning on
the certificate likelihood and the $s$ prior, stronger correlation widens the
aggregate prior as intended. In the posterior, `lambda=0.5` widens the ETI but
`lambda=0.9` shifts the mean downward and narrows it. At strong correlation the
annual errors move mainly through one common trajectory direction. The
certificate likelihood and the $s$ prior then constrain that lower-dimensional
direction along the existing reduction-recording ridge. The narrower posterior
is therefore specific to this conditional model geometry; it is not evidence
that strongly correlated surveillance error is more precise or that
`lambda=0.9` is preferred.

## Recording and missed-case decomposition

| Scenario | Mean $s$ | Change in mean $s$ | Model-implied missed true cases, mean (89% ETI) | Missed-count change | Decomposition rule |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.340 | reference | 29,216 (26,885-31,519) | reference | No |
| Moderate correlation | 0.355 | +0.015 | 27,433 (24,153-30,771) | -6.10% | No |
| Strong correlation | 0.368 | +0.028 | 25,844 (23,797-27,871) | -11.54% | **Yes: missed count** |
| Negative shift, larger | 0.300 | -0.040 | 35,151 (33,122-37,146) | +20.31% | **Yes: missed count** |
| Negative shift, smaller | 0.318 | -0.023 | 32,352 (30,066-34,482) | +10.73% | **Yes: missed count** |
| Positive shift, smaller | 0.367 | +0.027 | 26,019 (23,605-28,454) | -10.94% | **Yes: missed count** |
| Positive shift, larger | 0.401 | +0.060 | 22,572 (19,997-25,233) | -22.74% | **Yes: $s$ and missed count** |

Across the seven primary scenarios, the decomposition envelope spans
`0.300-0.401` for mean $s$ and `22,572-35,151` for the posterior mean of
model-implied missed true cases. These are descriptive scenario envelopes, not
posterior intervals.

Across the seven primary scenarios, the posterior correlation between the
derived standardised trajectory-error index and $s$ ranges from 0.929 to 0.979
(0.929 to 0.980 on the logit scale). This near-ridge is the expected
identification warning. The index is a summary of the scenario's sampled prior
errors, not an estimate of surveillance bias learned from certificates.

## Conditional joint corners

Because the aggregate rules triggered, the two pre-specified joint corners were
fitted:

| Joint corner | True total, mean (89% ETI) | Mean change | Mean $s$ | Missed true cases, mean (89% ETI) |
| --- | ---: | ---: | ---: | ---: |
| $\lambda=0.9,\delta=-0.4$ refined | 46,336.589 (44,331.066-48,337.931) | +4.704% | 0.325 | 31,298 (29,276-33,312) |
| $\lambda=0.9,\delta=+0.4$ | 34,806 (32,727-36,924) | -21.35% | 0.433 | 19,769 (17,674-21,876) |

The refined negative-shift corner remains inside the separate-axis total
envelope. Relative to the baseline total mean of `44,254.871` (mean MCSE
`41.528`), its mean is `2,081.718` births higher, or `+4.704%`, with mean MCSE
`16.367`. The distance from that difference to the 5% threshold is `131.025`
births. For the independent-seed comparison, the pre-specified two-combined-
MCSE band is

$$
2\sqrt{41.528^2+16.367^2}=89.274\ \text{births}.
$$

Because `131.025 > 89.274`, the comparison is no longer inside the protocol's
borderline band. The refined negative corner is classified as immaterial under
the aggregate mean rule; its 89% ETI width also does not trigger the width-
increase rule. Its refined mean $s=0.325$ is inside the primary `0.300-0.401`
envelope. Its model-implied missed-count mean is 31,298 (89% ETI
29,276-33,312), a `+7.123%` change from baseline and inside the primary
`22,572-35,151` posterior-mean envelope. The refined negative corner therefore
remains inside both primary decomposition envelopes and triggers neither
decomposition rule.

The positive-shift corner lies outside all three primary envelopes. Its total
mean is 2,804 births below the primary minimum, and its lower ETI endpoint is
2,296 below the primary lower endpoint. Its mean $s=0.433$ is above the primary
maximum of `0.401`, while its missed-count mean of 19,769 is below the primary
minimum of 22,572. Its mean $s$ is 0.092 above baseline and its missed-count
mean is 32.34% lower. Correlation and a positive level shift therefore interact
in a way that the separate-axis scenarios do not cover; this conclusion is not
affected by the refined negative-corner result.

Including the corners, the descriptive envelope spans posterior means of
34,806-50,190 and outer ETI endpoints of 32,727-52,177. It remains a scenario
envelope with no posterior coverage probability.

## Remaining maternal-age limitation

Broad-age posterior-predictive coverage remains `1/7`. The coherent
calibration scenarios vary the surveillance-prior correlation and level; they
do not resolve how the remaining discrepancy is allocated across maternal
ages. This phase therefore does not establish a maternal-age-specific
reduction or recording mechanism. The next model-adequacy gate is the mirrored
age-on-recording diagnostic, fitted under the same age grid and comparison
rules, to test whether assigning the residual age pattern to recording gives a
materially different aggregate or decomposition without claiming that either
age allocation is identified from certificates alone.

## Decision

1. The estimated 2016-2024 total is materially sensitive to coherent
   surveillance calibration. Do not present the baseline 89% ETI as if it
   incorporates shared surveillance-source uncertainty.
2. Retain `DSP004` as the preferred simple accounting structure and its
   unshifted independent fit as a clearly labelled reference, but report totals
   conditionally by calibration scenario. Do not pool draws or model-average
   without defensible scenario probabilities.
3. Report recording sensitivity and missed true cases by scenario. A stable
   false-positive sensitivity result does not make this decomposition stable to
   surveillance calibration.
4. Do not select a correlation or shift using in-sample PPCs or posterior
   concentration. The certificate data cannot validate these external
   calibration assumptions.
5. Do not make a recent-trend robustness claim from this analysis. No scenario
   changes the extrapolated-tail slope.
6. Treat the unresolved broad-age fit as the next model-adequacy gate. Run the
   mirrored age-on-recording diagnostic before making a maternal-age allocation
   claim.
7. Preserve the population-level interpretation: these are aggregate
   ascertainment corrections and cannot identify which individual unflagged
   births were missed cases.

## Reproducible artefacts

The nine originally required fitted run directories use the prefix:

```text
output/selection_core_reduction/DSP004/20260803-calibration-
```

The independent-seed refined negative corner is:

```text
output/selection_core_reduction/DSP004/
  20260803-calibration-corner-corr-090-shift-m040-refined-reporting
```

The primary and joint-corner comparisons are under:

```text
output/selection_core_reduction/comparisons/DSP004-coherent-calibration/
  20260803-primary-reporting
  20260803-with-joint-corners-reporting
```

The full `20260803-with-joint-corners-reporting` comparison was regenerated
with the refined independent-seed negative-corner run and supersedes the
shared-seed corner comparison.
