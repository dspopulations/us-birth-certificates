> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Recording-anchor dependence and diagnostic limits

**Original date:** 7 July 2026. Historical research note, revised for clarity in October 2026.

This reviewed the recording-anchor generator, selection diagnostics, simulation tests and year-trend reconstruction. The missing year-by-age term in trend reconstruction was fixed in [PR #68](https://github.com/dspopulations/us-birth-certificates/pull/68). The methodological limits below remain relevant.

## How the anchor is formed

`scripts/derive_recording_rates.py` combines birth counts by age, race and year with the Morris expected rate and de Graaf prevalence. It works with a retained-fraction ratio, interpolates missing years and extrapolates the tail. It then reconstructs true counts and divides recorded counts by them to derive recording priors.

The prior surface uses the same recorded counts as the likelihood and the same natural-rate benchmark as the model. It is not an independent measurement of recording. Unknown and multi-race have weak fallback priors because the source has no matching categories.

A pre-2015 holdout favoured a constant extrapolation. It cannot establish the trajectory in later years. The [workbook audit](20260803-degraaf-surveillance-workbook-extraction.md) further showed that the source rows represent overlapping five-year windows.

## Posterior correlation does not test identification

The diagnostic computes correlations between race-level termination and recording summaries. High correlation can reveal a trade-off. Low correlation does not prove that the data separate the quantities. Tight priors, nonlinear dependence, aggregation and other model terms can all affect this summary.

The earlier note argued that a nearly constant recording parameter must have correlation near zero. That argument was incorrect: correlation divides covariance by both standard deviations. Small variance alone does not determine correlation.

Report the correlations as correlations. Compare prior and posterior distributions, vary the restrictions, inspect the quantities that actually enter the recorded-count probability, and check sensitivity of the scientific estimates. A prior-to-posterior SD ratio measures dispersion change; it is not a percentage of information supplied by the data.

## What recovery tests establish

Simulation under the fitted model can reveal implementation or sampling errors. It cannot test whether that model describes the real population. A parameter drawn and fitted under the same tight prior can have good interval coverage with little learning from the simulated data.

Recovery checks therefore need interpretation alongside prior strength, misspecification checks and independent source comparisons. The [September core-model audit](20260905-dsp-code-review-fixes.md) added mode-specific checks and a small simulation pilot. That pilot was not a full calibration study.
