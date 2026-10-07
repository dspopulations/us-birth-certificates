> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Descriptive report methods

Generate the recorded-birth report from the repository root:

```bash
uv run python scripts/analyse_descriptive.py --render
```

The script queries `data/us_births.db` read-only, writes tables, figures and `config.json` to `output/analyse_descriptive/<timestamp>/`, and copies `descriptive.qmd` into that directory. Quarto renders the copy when `--render` is supplied. The calculations and queries are in `src/dspopulations_us_birth_certificates/descriptive_analyses.py`.

## Population and denominators

Recorded births have `down_ind = 1`, normally confirmed or pending Down syndrome (`ca_down_c` C or P). The database retains NULL for unknown or absent status. The report includes those births in all-birth denominators but not in the recorded numerator. Time series exclude missing year. This differs from Bayesian cell cohorts that exclude unknown status.

A confirmed-only series avoids including a pending category that did not exist in the older certificate. It does not remove other changes in reporting. The revised certificate phased in through 2015 and reached full national coverage in 2016. The report's shaded 2003–2013 band marks the earlier transition years, not the complete phase-in.

## Sections and benchmarks

| Section | Content | Coverage limits |
| --- | --- | --- |
| A | Status counts and recorded rates | Pending begins in 2004; unknown and absent status remain visible |
| B | Recorded counts against expected-count benchmarks | Surveillance carry-forward after 2018; reduction tail extrapolated; age/group benchmarks have shorter coverage |
| C | Maternal age, race/origin, education, marital status and payer | Race changes across eras; education schemes are separate; revised items have partial early coverage |
| D | Birthweight, gestation, sex and plurality | Earlier source fields are not all imported; state the available years and known-value denominator |
| E | Co-occurring conditions and newborn morbidity | Checkbox ascertainment and selection into recorded DS cases affect the proportions |

The annual recording comparison is recorded flags divided by a modelled expected count. It is a benchmark ratio, not a direct measurement of sensitivity from linked true cases. False-positive flags and source-population differences can affect it.

`p_ds_lb_wt` holds 2018 prevalence fixed through 2024. `p_ds_lb_nt_reduc` uses an age-risk curve and a reduction series with an extrapolated tail. The maternal-age benchmark stops at 2018. The ethnicity benchmark covers 2000–2018 with blanks for 2015 and 2017 and no multi-race category. Do not use `ds_case_weight` as an independent benchmark or use the unpopulated placeholder probability columns. See [data preparation](../data-preparation.md).

## Interpretation

The report describes recorded cases. Their characteristics need not represent all Down syndrome births. A condition percentage among recorded cases is not necessarily a lower bound for all cases; selection, false positives and different denominators can move it either way.

The cyanotic congenital heart disease checkbox is narrower than all congenital heart defects. A clinical estimate for all heart defects cannot serve as a like-for-like benchmark for that checkbox.

The extract contains live births only, no state identifier, no linked later diagnoses or outcomes, and no Down syndrome subtype. It cannot establish pregnancy prevalence, resolve pending diagnoses or distinguish recording changes from true prevalence changes by itself.

Classifier-cohort reports describe an additional selected population. They do not extend this descriptive report to all missed cases. Population estimates belong in the Bayesian model reports and retain their external-data assumptions.
