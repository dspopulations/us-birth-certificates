> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Nested classifier evaluation and selection-model sensitivities

Run on 8 November 2026 for [issue 67](https://github.com/dspopulations/us-birth-certificates/issues/67).

## Classifier evaluation design

The 2016–2025 data had already informed the existing classifiers. A new random split cannot make those past choices independent of the test records. This evaluation therefore starts again with all 48 original USBC10 candidate features. It does not reuse saved tuned parameters or the later pruned feature lists. It evaluates that fresh procedure on 2025 records only. It does not retrospectively validate the four published M1 models or estimate their performance across 2016–2025.

Each run uses three stratified outer folds. Stratification gives each fold a similar share of recorded positives. Each outer training set has a fresh five-trial Optuna search using two inner folds. Inner validation average precision selects parameters. Inner early stopping selects boosting lengths, with a 500-round limit and patience of 30 rounds. The outer fit uses the median stopping length from the selected trial, then scores its test rows once. Category vocabularies also come only from training rows. No outer test score changes the search or stopping decision.

This separation follows the [scikit-learn nested evaluation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) and its [guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). The feature definitions, LightGBM model family and search ranges still reflect prior project work. The estimate is conditional on those fixed choices, the small search budget and the 2025 cohort. It is not a test on a wholly new external population. If later changes respond to these scores, these scores no longer provide an independent evaluation of the changed procedure.

The seed is 67. The confirmed-plus-pending run treats both statuses as positive. The confirmed-only run excludes pending records. Both retain the existing policy of treating unknown status as negative. The command saves aggregate metrics and trial histories. It does not write scores or selection flags to the database.

Average precision summarises precision across recall levels. Precision is the share of selected records with a positive label; recall is the share of all positive labels selected. ROC AUC measures how often a positive record ranks above a negative record. These metrics concern the recorded checkbox. They cannot identify true diagnoses among births without a recorded diagnosis. Random folds within 2025 also do not measure performance in a later year. Fold ranges below describe variation between these splits; they are not confidence intervals.

## Classifier results

| Recorded label | Records | Positives | Positive fraction | Pooled average precision | Pooled ROC AUC | Fold average precision range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Confirmed or pending | 3,619,990 | 1,797 | 0.0004964 | 0.01895 | 0.8760 | 0.01572–0.02156 |
| Confirmed, pending excluded | 3,619,027 | 834 | 0.0002304 | 0.007836 | 0.8867 | 0.007138–0.01280 |

Pooled metrics use the combined test predictions, with one prediction per record. They differ from the mean of fold metrics because scores from separately fitted models are ranked together. Both labels have average precision above their positive fraction, but both values remain below 0.02. ROC AUC alone does not show this limit with rare labels. The two label definitions have different positive fractions and cohorts, so their average precision values do not establish that one definition is preferable.

The [pooled metrics](data/20261108-model-review/classifier-metrics.csv), [fold metrics](data/20261108-model-review/classifier-folds.csv) and [settings and selected parameters](data/20261108-model-review/classifier-provenance.json) retain the exact values. The same directory contains all six five-trial search histories. The database SHA-256 still matches the published 6 November database, `76c155a9c5ace6c7fcce4cb62bafa847e434d4f36aca1546432edcae353b8e01`.

## Selection sensitivity design

All cases use the full three-stage selection model, with the same 2016–2024 recorded-count likelihood. The prepared cohort contains 33,498,320 births and 17,776 recorded cases. The existing preparation excludes missing maternal age or Down syndrome status and unknown clinical flags used to form its cells. These totals therefore describe the eligible cohort, not all national births.

The study combines 66,205 original strata into 9,667 groups with identical model probabilities. Summing their birth and recorded counts changes the binomial likelihood only by a constant that does not depend on model parameters. A regression test checks this against the full model log density. This reduces fit cost without changing the posterior. The original cells stay in the ignored local run directory. Clinical-stratum checks are not included.

The baseline uses the current variant C priors. Each other case changes only the named family. These are specified stress tests, not uncertainty bounds established by external evidence.

| Scenario | Change from baseline |
| --- | --- |
| Baseline | Current variant C priors; Morris logit standard deviation 0.001; false-positive probability 0.000078. |
| Baseline rate low | Multiply each age-specific baseline probability by 0.8. |
| Baseline rate high | Multiply each age-specific baseline probability by 1.2. |
| Baseline rate wide | Increase the baseline logit standard deviation from 0.001 to 0.2. |
| Recording wide | Double race-by-year and education recording prior standard deviations. |
| Termination wide | Double all termination prior standard deviations. |
| False positive zero | Set the false-positive probability to zero. |
| False positive double | Set the false-positive probability to 0.000156. |

Every case starts with seed 67, four chains, 1,500 tuning iterations and 1,500 retained draws per chain, with target acceptance 0.95. Validation checks every free parameter and annual and total expected counts. Thresholds are R-hat below 1.01, effective sample size at least 400, no divergences, finite diagnostics and energy diagnostic BFMI at least 0.3. These checks assess sampling, not the scientific assumptions.

Intervals are 89% equal-tail posterior intervals within a scenario. A range across scenarios is not a credible interval. Replicated-count checks include binomial outcome variation and compare recorded counts by year, race and age. The recording-anchor table compares prior and posterior logit means and standard deviations. Its anchor reuses recorded counts, so agreement is not independent validation. See the [anchor-dependence note](20260707-s-anchor-and-identifiability-diagnostic.md).

## Selection results

The table reports expected Down syndrome livebirths in the eligible cohort. Prenatal reduction is the modelled share removed from its baseline expected livebirth count. The adjusted recorded share subtracts the assumed false-positive contribution from recorded counts, then divides by the expected count within each draw. It is a model-dependent quantity, not an observed recording rate.

| Scenario | Expected livebirths, mean [89% interval] | Prenatal reduction, mean [89% interval] | Adjusted recorded share, mean |
| --- | ---: | ---: | ---: |
| Baseline | 36,906 [35,365–38,502] | 49.1% [46.9–51.2%] | 41.1% |
| Baseline rate low | 35,287 [33,885–36,697] | 39.1% [36.7–41.5%] | 43.0% |
| Baseline rate high | 38,004 [36,289–39,774] | 56.3% [54.3–58.3%] | 39.9% |
| Baseline rate wide | 37,898 [36,226–39,650] | 54.7% [50.7–58.7%] | 40.1% |
| Recording wide | 35,354 [32,822–38,002] | 51.2% [47.6–54.7%] | 43.0% |
| Termination wide | 37,882 [36,340–39,436] | 47.7% [45.6–49.8%] | 40.1% |
| False positive zero | 42,875 [41,488–44,256] | 40.8% [38.9–42.7%] | 41.5% |
| False positive double | 29,698 [28,489–30,999] | 59.0% [57.2–60.7%] | 42.3% |

The false-positive assumption changes the expected total from about 29,700 to 42,900 across the specified extremes. That shift exceeds the uncertainty within each case. A precise total that ignores this assumption is not supported by these fits.

Moving the baseline prevalence down or up by 20% changes the mean expected count by about -4.4% or +3.0% from baseline. The inferred prenatal reduction moves more, from 39.1% to 56.3%. The recorded likelihood can fit these different decompositions. The widening of the baseline prior also moves its posterior baseline count and prenatal reduction. The decomposition therefore depends on both prior centres and prior widths.

Doubling the recording prior widths increases the expected-count interval width from about 3,136 to 5,180 births. Its posterior mean also shifts. Narrow uncertainty under the baseline prior should not be presented as uncertainty over all recording assumptions.

Doubling the termination prior widths moves the expected total up by about 2.6% and the mean prenatal reduction down from 49.1% to 47.7%. This change is smaller than the effects of the false-positive and baseline-rate stress tests. It does not show that termination is independently identified. Other prior centres, structures or recording assumptions could give different results.

All eight selected fits pass the stated numerical checks. The high-baseline case initially failed with maximum R-hat 1.0119. Its repeat used 3,000 tuning and 3,000 retained draws per chain and passed with maximum R-hat 1.0025. The wider termination case initially had maximum R-hat 1.0154 and minimum effective sample size 245. A first repeat with 3,000 tuning and 4,000 retained draws still had maximum R-hat 1.0106. Its final repeat used 4,000 tuning and 10,000 retained draws per chain and passed with maximum R-hat 1.0018 and minimum effective sample size 1,819. No selected fit has divergences. Earlier failures remain in the provenance; the thresholds and priors were not relaxed.

The [total estimates](data/20261108-model-review/selection/totals.csv) and [annual estimates](data/20261108-model-review/selection/annual.csv) retain unrounded values. The [numerical checks](data/20261108-model-review/selection/validation.csv) and [run provenance](data/20261108-model-review/selection/provenance.json) retain sampling settings and failed earlier checks.

## Predictive and anchor checks

All nine annual recorded counts lie within their 89% replicated-count intervals in all eight cases. That does not imply adequate fit to all groups. The baseline covers six of seven age groups. For ages 20–24, the observed count is 1,638, below the interval of 1,657–1,836. The low-baseline case covers only four age groups; wider recording covers five; doubled false positives cover five and also miss one race group. The wider baseline and wider termination cases cover all seven age groups. These are descriptive checks on data used for fitting, not independent coverage tests.

The [replicated-count table](data/20261108-model-review/selection/predictive_checks.csv) exposes every year, race and age comparison. It uses simulated recorded outcomes, including binomial variation. The older saved plots labelled PPC use expected recorded counts, so their intervals answer a different question.

The [recording-anchor table](data/20261108-model-review/selection/recording_anchor.csv) retains all 63 race-by-year prior/posterior comparisons per scenario. In the baseline, the median posterior standard deviation is 0.60 times the prior standard deviation. The largest absolute mean shift is 1.40 prior standard deviations. Such narrowing shows the effect of this likelihood and prior together. It does not establish independent evidence for recording sensitivity because the anchor already uses recorded counts.

## Reproduction

Run from the repository root with the locked environment. The two classifier commands used the following shared options, with `--label confirmed` added for the second run.

```bash
uv run python scripts/evaluate_classifier.py --start-year 2025 --end-year 2025 --outer-folds 3 --inner-folds 2 --n-trials 5 --num-boost-round 500 --early-stopping-rounds 30 --num-threads 4 --output-dir output/reviews/issue-67/classifier-2025
uv run python scripts/run_selection_sensitivity.py --output-dir output/reviews/issue-67/selection-sensitivity
uv run python scripts/run_selection_sensitivity.py --scenarios baseline_rate_high --draws 3000 --tune 3000 --output-dir output/reviews/issue-67/selection-sensitivity-extended
uv run python scripts/run_selection_sensitivity.py --scenarios termination_wide --draws 4000 --tune 3000 --output-dir output/reviews/issue-67/selection-sensitivity-termination-extended
uv run python scripts/run_selection_sensitivity.py --scenarios termination_wide --draws 10000 --tune 4000 --output-dir output/reviews/issue-67/selection-sensitivity-termination-final
uv run python scripts/summarise_selection_sensitivity.py output/reviews/issue-67/selection-sensitivity --replacement-study-dir output/reviews/issue-67/selection-sensitivity-extended output/reviews/issue-67/selection-sensitivity-termination-extended output/reviews/issue-67/selection-sensitivity-termination-final --output-dir notes/data/20261108-model-review/selection
```

The exact output directory must be new. Full posterior draws, local data and figures remain in ignored run directories. Only aggregate evidence is tracked with this note. Saved selection manifests record package versions, source hashes, input hashes, sampler choice and validation status. Classifier settings record features, seed, label policy, package versions and the ordered cohort identifier hash.

The recorded base revision is `40b47749d1fa8b45fa84b1dfa026c4340baa4c90`. The runs used the then-uncommitted implementation included with this evidence, rather than that base revision alone.

## Regression coverage

The new fold tests check that outer test rows never enter inner fitting, tuning or early stopping. They also check category vocabularies and reject reused tuned parameters.

The year-trend test saves synthetic draws with a non-zero year-by-age detection interaction. It checks annual expected counts and reduction intervals against those draws. Both variant paths must reject a deliberately inconsistent saved probability before drawing figures.

The summary-figure test renders all four PNG, SVG and CSV outputs with all seven race groups. It checks race order and aggregate rates. It also checks that baseline probabilities and reduction ratios are computed within draws before averaging. Figure titles now describe conditional estimates instead of asserting fixed empirical conclusions. The C/B display compares posterior means under different priors; it does not define an uncertainty bound.

The locked environment sync, Ruff and spellcheck passed. All 538 standard tests and all 12 slow tests passed. The slow checks assess numerical behaviour on simulated data; they do not establish the assumptions used for population estimates.
