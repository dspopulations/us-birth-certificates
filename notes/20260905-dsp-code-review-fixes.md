> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP statistical-code review fixes

This change addresses the DSP001–DSP010 code review. It changes the anchored
model as well as its reporting. Existing production fits have not been refitted
and must not be presented as results from the revised model.

## Model and calculation changes

- DSP003 now solves its weighted logistic margin with a bracketed Newton
  method. Tests check values against an independent scalar root finder and
  gradients against the implicit derivative. The reproduced wide-age-prior
  failure now preserves the national reduction margin.
- Surveillance windows now have a joint observation likelihood. Its default
  correlation comes from the overlap of annual window weights. This is a
  working covariance, not an estimate of the source programmes' error
  covariance. Shared biases may extend beyond the overlap of years.
- The CLI now weights pooled surveillance prevalence by annual birth counts,
  including every year in the window. Missing weights stop the run. Equal-year
  weights, partial or zero error correlation, and a non-overlapping subset are
  explicit sensitivity options.
- The starting log-prevalence prior now has a fixed median of 0.0013 and
  log-scale SD of 0.25. Its median is a saved setting. Previously its centre
  came from the first surveillance observation. Separating the prior from that
  observation also permits joint simulation checks with the same generating
  and fitting model. This working prior still needs sensitivity analysis.
- Prior prediction for the probability-barrier model uses rejection sampling.
  Forward proposals have acceptance probability equal to the exponential of
  the non-positive barrier. The resulting draws include the penalty used in
  posterior fitting. Unknown penalties fail explicitly instead of silently
  producing a different prior. Acceptance and proposal counts are saved.

## Validation and reporting changes

- Builders reject invalid counts and indices before integer conversion. The
  checks also cover probabilities, age ordering, calendar alignment, positive
  denominators, duplicate anchor windows and panel-condition names.
- Numerical validation checks every free parameter and the main reported
  quantities. Partial missing diagnostics fail. Exemptions apply only to known
  deterministic constants. The gate includes both effective sample sizes,
  R-hat, divergences, energy diagnostics, the DSP003 margin and probability
  clipping. Maximum-tree-depth hits are retained as warnings.
- Runs save validation.json. Failed numerical validation returns exit code 2
  and retains the raw artefacts. The report marks the fit as unvalidated.
  Automatic discovery skips a saved non-passing status. Legacy runs without a
  status remain available unless the caller requires validated runs.
- Report priors come from the actual model's draws. Anchored models no longer
  load an unused reduction CSV in the fitting CLI. Legacy anchored prior draws
  without the penalty-aware sampler marker are not trusted. Missing prior
  information remains unavailable instead of being replaced by another model.
- Annual sensitivity priors include drift, certificate version and panel
  terms. A logit-normal prior's median is no longer labelled as its mean.
- The report names the count estimand as expected burden. The old true_count
  names remain compatibility aliases. They are not draws of the unknown
  realised count among the observed births.
- The panel documentation states that loading one means equal changes in log
  odds. A fixed walk scale still applies shrinkage. The shared prevalence
  trend can be updated by the joint likelihood but is not identified by the
  controls alone. Omitting within-condition uncertainty can inflate the
  descriptive Q and I-squared statistics.

## Code structure and reproducibility

The model's anchor, age reduction, panel and recording components are separate
functions. Small numerical and input-validation helpers have direct tests.
CoreFitSpecification supplies the settings used to build, save and report a
fit. It replaces the separate CLI and saved-configuration assembly paths.

Each saved fit has a manifest with the code revision, working-tree state,
source hashes, input and configuration hashes, package versions, lockfile hash,
actual sampler and prior-sampling method. Aggregate panel inputs and anchor
weights are saved in the configuration. No raw birth records are added.

Earlier model-development notes retain their historical results. The concise
current assumptions belong in the component code and report template, rather
than in comments that combine old fit results with current definitions.

## Checks and limits

Fast regression tests exercise invalid input, numerical margins and gradients,
overlap covariance, prior sampling, diagnostic failures, prior-to-report
calculations for DSP006–DSP010, and failed-fit CLI output for anchored models.
Slow tests fit synthetic data for DSP003, DSP008, DSP009 and DSP010.

The new simulation command generates all observation channels jointly and
saves ranks, interval coverage and numerical health. An initial pilot used two
synthetic data sets per model, seeds 47 and 49, four chains, 1,000 warmup and
1,000 retained draws per chain. All eight fits passed numerical validation with
no divergences. Across those fits, the largest R-hat was 1.0095, the smallest
effective sample size was above 847 and the smallest per-chain BFMI was above
0.87. These are synthetic-data regression checks, not production estimates or
evidence that interval coverage is calibrated.

Run the checks from the repository root:

```bash
uv run --locked pytest
uv run --locked pytest -m slow
uv run ruff check src tests scripts
npm run spellcheck
uv run --locked python scripts/validate_dsp_simulation.py --replicates 2 --output-dir /path/to/empty/pilot
```

For a larger simulation experiment, increase --replicates and inspect ranks
and coverage separately for each model and quantity. Do not count correlated
years or cells as independent simulation repetitions. Account for finite
simulation error and dependence among posterior draws. A large run does not
automatically establish calibration simply because its numerical checks pass.

## Required next analysis

Refit the production data before comparing revised estimates. Compare the
overlap covariance with independent errors and a non-overlapping subset. Vary
the observation scale, starting prevalence prior, recording drift and panel
exclusion assumptions. Retain comparisons with the simpler models.

The code does not resolve missing external information. Stronger claims about
the prevalence-recording split still require external validation, sensitivity
analysis, or both. A realised-count analysis would require an additional
conditional latent-count calculation, rather than relabelling expected counts.
