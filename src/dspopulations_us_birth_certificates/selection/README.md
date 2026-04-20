# `dspopulations_us_birth_certificates.selection`

Three-stage Bayesian selection model decomposing observed DS recording on
U.S. birth certificates (2016–2024) into:

```
P(R=1 | X) = θ_LB(age) · η(X) · s(X) + (1 − θ_LB·η) · f
```

- `θ_LB` — baseline DS livebirth rate in absence of screening (Morris
  2002 / de Graaf 2015). Stage 1.
- `η = 1 − η_detect · η_term` — screening/termination pass-through
  (Kuppermann / Natoli / Chaiken). Stage 2.
- `s` — birth-certificate sensitivity given a DS livebirth (Boulet 2011
  / Salemi 2017). Stage 3.
- `f` — false-positive rate, fixed at 7.8e-5 (Ohio/NY validation).

The design is documented at `plans/docs/bayesian_selection_model.md`;
the implementation plan and status are at
`plans/20260420-selection-model.md`.

## Public API

```python
from dspopulations_us_birth_certificates.selection import (
    # Priors + variants
    ModelPriors, VARIANTS,
    variant_A_tight_s, variant_B_tight_eta_term,
    variant_C_default, variant_D_dobbs_only,
    # Model
    build_model, SPECS,
    # Data
    prepare_cells, summarise_cells,
    DEFAULT_DB_PATH, DEFAULT_POST_DOBBS_YEAR, DEFAULT_YEAR_RANGE,
    # Simulation (for parameter-recovery validation)
    TrueParams, simulate_cells,
    # Config + run profile
    SelectionModelConfig, selection_run_config, preset_names,
    # Diagnostics module (figures + tidy-DataFrame companions)
    diagnostics,
)
```

## Typical flow

```
scripts/fit_selection_model.py \
    --variant C --spec full --profile reporting --render
        │
        ├── selection.prepare_cells(con)          # DuckDB → cell frame
        ├── selection.build_model(cells, priors)  # PyMC full spec
        ├── bayes.sampling.sample(...)            # NUTS + posterior predictive
        ├── bayes.io.save_artefacts(...)          # idata.nc, cells.parquet, ...
        ├── bayes.io.copy_docs_template(...)      # docs/models/selection/index.qmd → run dir
        ├── selection.render.render_all(...)      # six diagnostic plots + tables
        └── bayes.io.render_quarto(...)           # index.qmd → index.html
```

Output layout:

```
output/selection/<variant>/<spec>/<timestamp>/
├── idata.nc            # posterior InferenceData
├── cells.parquet       # exact input frame
├── config.json         # SelectionModelConfig snapshot
├── run_config.json     # BayesRunConfig (profile + overrides)
├── summary.csv         # az.summary on posterior
├── index.qmd           # copied Quarto template
├── index.html          # rendered (if --render)
├── plots/              # identifiability, dobbs_year_trajectory,
│                       # cchd_consistency, age_curve,
│                       # decomposition_by_race, ppc_{year,race,age}_idx
└── tables/             # CSV companions for the non-PPC plots
```

## Run profiles

- **`dev`** — 400 tune + 400 draws × 2 chains, target_accept=0.9, nutpie.
  Inner-loop iteration.
- **`reporting`** — 1500 tune + 1500 draws × 4 chains, target_accept=0.95,
  nutpie. Publication-quality posteriors; ≥ 1 h wall-clock per variant
  at full spec.

## Variants

- **A (tight s)** — tight priors on `s` race/edu effects, loose on
  `eta_term`. If race/edu decomposition loads onto `s` under A but onto
  `eta_term` under B, the data alone cannot separate them.
- **B (tight eta_term)** — the mirror image.
- **C (default)** — both informative. The main specification.
- **D (Dobbs-only)** — race/edu priors on `eta_term` shrunk to zero
  with wide sigma; termination effects must be identified through the
  pre-vs-post-2022 national year shift alone. Agreement with C is
  evidence of data-driven identification rather than prior-driven
  decomposition. (Weaker test than the original design called for
  because `us_births.db` has no state-level column — see
  `plans/20260420-selection-model.md` §2.2.)

## Important invariants

See `plans/20260420-selection-model.md` §12 for the full list. Summary:

1. Morris priors stay tight (σ=0.10 on logit).
2. Clinical features (CCHD, NICU, Aven, Preterm) enter only `s`, never `η`.
3. False-positive rate is fixed, not estimated.
4. Reference levels: Race = NH White, Education = Some college, Payer = Private.
5. Year coding is `year − year_start` (0-based).
6. Stage 1 is θ_LB (baseline livebirth rate), not θ (conception rate).
7. Region is intentionally absent (no state-level column in the DB).

Changes to any of these should be discussed before implementation —
they have specific identifiability consequences.
