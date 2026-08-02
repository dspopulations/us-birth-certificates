> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP002 `s_year` extension

**Date:** 2026-08-02
**Status:** Implemented and fitted as the first direct extension of the core
age-reduction-recording model.

## Decision

Use stable `DSPnnn` identifiers for the aggregate Bayesian accounting models,
following the vocabulary-growth convention of a declarative model registry.
The numbers record historical order, not preference.

- `DSP001`: core model with constant certificate recording sensitivity `s`.
- `DSP002`: first extension with year-specific `s_year`.

`DSP002` is intentionally narrow. It changes only the recording-sensitivity
structure so it can be compared directly against `DSP001` before adding race,
education, payer, age modifiers, or a detection/termination decomposition.

## Model Change

The baseline observation model is:

```text
p_recorded[y,a] =
  theta_age[a] * (1 - rho_year[y]) * s
  + (1 - theta_age[a] * (1 - rho_year[y])) * f
```

`DSP002` replaces the scalar recording sensitivity in the year-specific part of
that expression:

```text
s_year_offset_raw[y] ~ Normal(0, recording_s_year_sigma)
s_year_offset[y]     = s_year_offset_raw[y] - mean(s_year_offset_raw)
s_year[y]            = inv_logit(recording_s_logit + s_year_offset[y])
```

The offsets are centred on the logit scale. This preserves a global
`recording_s` centre comparable with `DSP001` and avoids letting the global
level and year offsets duplicate each other.

## Fit Commands

Reporting-quality fits were run with matched sampler settings:

```bash
PYTENSOR_FLAGS=base_compiledir=/private/tmp/pytensor-codex \
MPLCONFIGDIR=/private/tmp/mpl-codex \
conda run -n dspop-us-birth-certificates \
python scripts/fit_core_reduction_model.py DSP001 \
  --profile reporting \
  --draws 3000 \
  --tune 3000 \
  --chains 4 \
  --prior-predictive-samples 1000 \
  --nuts-sampler pymc
```

```bash
PYTENSOR_FLAGS=base_compiledir=/private/tmp/pytensor-codex \
MPLCONFIGDIR=/private/tmp/mpl-codex \
conda run -n dspop-us-birth-certificates \
python scripts/fit_core_reduction_model.py DSP002 \
  --profile reporting \
  --draws 3000 \
  --tune 3000 \
  --chains 4 \
  --prior-predictive-samples 1000 \
  --nuts-sampler pymc
```

Artefacts:

- `DSP001`: `output/selection_core_reduction/DSP001/20260802-152403`
- `DSP002`: `output/selection_core_reduction/DSP002/20260802-152422`
- Direct comparison:
  `output/selection_core_reduction/comparisons/DSP001-vs-DSP002/20260802-152550`

The copied Quarto reports rendered successfully to `index.html` for both model
fit directories after running `quarto render` outside the filesystem sandbox.

## Diagnostics

Both matched reporting fits passed the current convergence gate.

| Model | Max Rhat | Min ESS |
| --- | ---: | ---: |
| `DSP001` | 1.0000 | 1519 |
| `DSP002` | 1.0000 | 7055 |

## Comparison Findings

Headline comparison, 2016-2024:

| Metric | `DSP001` | `DSP002` | Difference |
| --- | ---: | ---: | ---: |
| True DS livebirths, posterior mean | 43,828 | 44,535 | +706 |
| True DS livebirths, 89% ETI | 41,444-46,130 | 41,376-47,617 |  |
| Aggregate reduction, posterior mean | 0.396 | 0.386 | -0.010 |
| Global recording sensitivity, posterior mean | 0.344 | 0.341 | -0.002 |

The central accounting story is stable. Allowing `s_year` raises the posterior
mean true-livebirth total by about 706 over nine years, but the 89% ETIs overlap
strongly. This looks like a sensitivity check, not a materially different
headline.

The `DSP002` year-specific recording sensitivity posterior is:

| Year | Posterior mean | 89% ETI |
| --- | ---: | ---: |
| 2016 | 0.363 | 0.325-0.409 |
| 2017 | 0.341 | 0.305-0.383 |
| 2018 | 0.363 | 0.323-0.410 |
| 2019 | 0.348 | 0.309-0.394 |
| 2020 | 0.346 | 0.280-0.435 |
| 2021 | 0.335 | 0.267-0.425 |
| 2022 | 0.326 | 0.257-0.414 |
| 2023 | 0.328 | 0.260-0.418 |
| 2024 | 0.331 | 0.259-0.424 |

The early-year posterior means are somewhat higher and the post-2021 means are
somewhat lower than the constant-`s` estimate, but the recent-year intervals are
wide because `s_year` competes with the extrapolated `rho_year` prior.

## Interpretation

`DSP002` is useful as a first extension because it checks whether the aggregate
story depends on forcing recording sensitivity to be constant. The current
answer is: not much. That supports keeping `DSP001` as the primary simple model
and presenting `DSP002` as a sensitivity analysis unless a stronger external
recording-rate anchor is added.

This comparison does not identify recording drift from certificates alone. The
birth-certificate likelihood still mainly sees:

```text
theta_age * (1 - rho_year) * s_year
```

so any publication wording should say that `DSP002` tests robustness to
year-varying recording assumptions. It should not claim that the data have
separated true recording changes from surveillance-informed reduction changes.
