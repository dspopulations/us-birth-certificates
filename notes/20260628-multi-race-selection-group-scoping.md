# Scoping: promoting multi-race to its own selection-model group

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

## 1. Summary & recommendation

Promoting the multi-race population (`mracehisp_c=6`, currently buried in the Unknown cell) to its own selection-model group is worth doing and is statistically viable. Over the 2016-2024 training window the group has 771,018 births and 383 recorded Down syndrome (DS) cases — comfortably more than the existing NH AIAN group (246,101 births / 184 DS) that already gets its own `race_idx`, with 32-51 recorded DS cases in *every* model year and all seven age bins populated. The recommended approach is a **separate group with the model's normal partial-pooling regime** (informative race priors, no flat prior), **not** partial pooling under a forced hierarchy and **not** a 2014+-only restriction (the model already starts at 2016, so the harmonisation onset is inside the window). The one genuine limitation is the Stage-3 recording anchor: de Graaf surveillance has no multi-race category, so the multi-race `s(race, year)` prior must be a constructed weak fallback (mirroring how Unknown is treated today) rather than a measured surveillance gradient — this affects only the prior, not whether the likelihood can identify the group.

## 2. How race is used now

The race vocabulary is `RACE_LEVELS` (`priors.py:65-72`), an ordered six-element list — idx0 NH White, 1 NH Black, 2 NH AIAN, 3 NH Asian/Pacific Islander, 4 Hispanic, 5 Unknown — with `N_RACE = len(RACE_LEVELS) = 6` (`priors.py:84`). Order is positionally load-bearing: every race-indexed prior array is aligned to it and must agree with `data.RACE_MAP`. Three prior stages are race-indexed:

- **Detection** (Stage 2): `ETA_DETECT_RACE` shape `[N_RACE]` (`priors.py:148-157`), reference NH White = 0.00, shared scalar `ETA_DETECT_RACE_SIGMA = 0.20`; consumed at `model.py:173-178` as `eta_detect_race ~ Normal(dims="race")`.
- **Termination given diagnosis** (Stage 2): `ETA_TERM_RACE` shape `[N_RACE]` (`priors.py:238-247`), shared scalar `ETA_TERM_RACE_SIGMA = 0.20`; used as `mu` for *both* `eta_race` (`model.py:131-136`) and `eta_term_race` (`model.py:206-211`).
- **Recording sensitivity** `s(race, year)` (Stage 3): `S_RACE_YEAR_LOGIT`/`S_RACE_YEAR_SIGMA` shape `[N_RACE, n_year] = [6, 9]` (`recording_anchor.py:25-41`), generated from de Graaf surveillance by `scripts/derive_recording_rates.py`; rows 0-4 are *measured*, row 5 (Unknown) is a flat weak fallback `logit(0.4) = -0.4055`. Consumed at `model.py:250-255` as `s_race_year ~ Normal(dims=("race","year"))`. The companion full-margin anchor target `PREV_RACE_YEAR`/`PREV_RACE_YEAR_SIGMA` `[6, 9]` (`recording_anchor.py:43-59`) is passed separately to `build_model`; row 5 is all `np.nan` (unanchored), so the margin loop at `model.py:285-314` skips it.

Crucially, `mracehisp_c` *already codes* 6 = 'NH more than one race', but `data.py:60-66` routes both code 6 and `NULL` through the SQL `ELSE` to `RACE_UNKNOWN_IDX = 5` (`data.py:168-173`). So multi-race births are currently folded into Unknown and silently inherit its weak `s = 0.40` fallback with no surveillance margin. The data-side comment at `data.py:62-64` explicitly flags promoting it as a planned follow-up.

## 3. The anchor problem

This is the crux. **De Graaf does not supply a multi-race anchor.** The surveillance source mapped by `ETH_TO_RACE` (`scripts/derive_recording_rates.py:74-80`) has exactly five ethnicity strings — Non-Hispanic White, Non-Hispanic Black, American Indian or Alaska Native, Non-Hispanic Asian or Pacific Islander, Hispanic — with no 'two or more races' / 'more than one race' category. `NAMED = range(5)` is hard-coded as the anchored set; the sixth index is filled with the synthetic `UNKNOWN_S = 0.40` / `UNKNOWN_SIGMA = 0.50` fallback, not derived. A seventh (multi-race) group therefore cannot be anchored directly from surveillance.

Three options:

- **Option 1 (recommended) — own row, weak/wide prior, `PREV = NaN`.** Give multi-race its own `s(race, year)` row using the same fallback the Unknown row uses (`s_logit ≈ logit(0.40-0.45)`, `sigma ≈ 0.45-0.55`) and a `NaN` `PREV` row so the full-margin term (which already skips non-finite rows via the `np.isfinite` check at `model.py:285-314`) leaves it unanchored. Because the recorded Binomial `R_obs` is observed for this group every year with large `N`, the recording level is still informed bottom-up; the wide prior just avoids importing a level that surveillance cannot justify. This mirrors the existing Unknown treatment and is the most honest move.
- **Option 2 — pooled-total anchor as the prior *mean*.** Seed the multi-race row's prior mean from a births-weighted `s_total(year)` pooled across the five named de Graaf groups (and optionally the all-race pooled prevalence for the margin), with a `sigma` between the named-group and Unknown-fallback widths. The generator already holds per-cell births and recorded counts to build this aggregate. Defensible if the recorded multi-race count justifies a sharper centre, but assumes multi-race recording behaves like the population average.
- **Option 3 — borrow a donor/composite group.** Least defensible without external evidence on the multi-race component mix; avoid NH AIAN specifically as a donor (already flagged unreliable: survival ratio > 1, tiny counts).

**Recommendation:** Option 1 as the default, optionally seeding the prior mean from Option 2's pooled level if the recorded count warrants it. Keep multi-race **out** of the full-margin surveillance anchor (`PREV = NaN`) so it cannot distort the surveillance-tied total.

## 4. Data viability

Over 2016-2024 (the model window), multi-race is clearly estimable and more so than the existing NH AIAN group:

- **Overall:** multi-race 771,018 births / 383 recorded DS; NH AIAN (yardstick) 246,101 / 184; true-Unknown (`NULL`) 306,717 / 187. Multi-race has >3x the births and >2x the DS of the group that already gets its own index, and dominates the current pooled Unknown cell (771k vs 307k true `NULL`s).
- **Per year:** multi-race DS = 44, 45, 48, 34, 43, 47, 32, 39, 51 (range 32-51, mean ~43); births rise 80,847 → 91,453. Every model year exceeds NH AIAN's per-year DS (14-28, mean ~20), so the per-year `s(race, year)` path is estimable.
- **By age bin (7 bins):** multi-race DS = 17, 44, 64, 76, 107, 67, 8 — all seven populated, exceeding NH AIAN in every bin. No empty age margin.
- **Sparsity (age × educ, 7×6 = 42 margins):** multi-race fills 30/42 (median 4 DS per occupied margin) vs NH AIAN 26/42 (median 3). The 12 empty multi-race margins are demographically implausible corners (very young × high education, most of the 45+ row) that are empty for NH AIAN too — not multi-race-specific sparsity. At full model-cell granularity all race groups are sparse and rely on informative priors + partial pooling, which multi-race supports at least as well as NH AIAN.

**Implication:** the likelihood can identify the group; no 2014+-only restriction is needed within the 2016-2024 window. The only estimability caveat is the prior side (the unanchored `s`), not the data.

## 5. Change list by file

**Ordering decision (governs everything below):** append multi-race at idx 5 and move Unknown to idx 6, *or* append multi-race at idx 6 and keep Unknown at idx 5. Both keep the five published de Graaf anchor rows (0-4) stable. The findings present both; the multi-race=5 / Unknown=6 layout is the data-side recommendation (`data-viability`, `data-and-rates`) because it groups the two unanchored fallback rows (5 and 6) adjacently, but it shifts `RACE_UNKNOWN_IDX` from 5→6 and so touches more hard-coded "idx 5 = Unknown" references. **Pick one and apply it consistently across priors, anchor, data, and tests** — a mismatch silently attaches the wrong prior to the wrong group with no error. The change must be atomic across priors + anchor + data or the build fails fast (shape mismatch / `race_idx out of range`).

**`selection/priors.py`** *(genuine modelling choices on the new offsets)*
- `RACE_LEVELS` (`:65-72`): add a 7th label (e.g. `'Multi-race'` / `'NH 2+ races'`); `N_RACE` → 7 automatically (`:84`). *Mechanical, but the index position is a real decision.*
- `ETA_DETECT_RACE` (`:148-157`): extend to length 7. **Modelling choice** — detection/screening-access offset for multi-race; defensible default 0.00 (neutral, like Unknown). Scalar sigma unchanged.
- `ETA_TERM_RACE` (`:238-247`): extend to length 7. **Modelling choice** — termination-given-diagnosis offset; default 0.00. Scalar sigma unchanged. (Used as `mu` for both `eta_race` and `eta_term_race`.)
- `ModelPriors` dataclass (`:333-397`): *no edit* — `default_factory` lambdas copy the module arrays, so extending the arrays propagates. **Verify** all four race arrays have first-dim length 7 after the change.
- Variants A/B/C/D (`:405-453`): *no edit* — all use element-wise scaling or `np.full_like` / `np.zeros(N_EDU)`; re-run the variant smoke checks only.

**`scripts/derive_recording_rates.py`** *(the durable anchor fix lives here, not in the generated module)*
- `ETH_TO_RACE` / `NAMED = range(5)` (`:74-80`, `:64-70`): *no change to the named set* — multi-race is unanchored. Optionally add a `MULTI_S` constant (or reuse `UNKNOWN_S`/`UNKNOWN_SIGMA`). *Mechanical.*
- Two `_load` CASE expressions (`:97-120`): add a `WHEN 6 THEN 5` arm (per chosen layout) so multi-race is split out of the `ELSE` in the age-structure and recorded-count frames. *Mechanical.*
- `_write_anchor_module` (`:222-282`): widen every `np.full((6, n_year), ...)` to height 7 (prefer importing `N_RACE` from priors over a literal); the named-fill `range(5)` loop is unchanged; the format `range(6)` loop → 7. Write the multi-race row as the weak fallback (`logit(0.40)`/`sigma 0.50`, `PREV = NaN`). Update the generated header/docstring text describing row layout. **Modelling choice** baked into the fallback values; *the widening itself is mechanical.*

**`selection/recording_anchor.py`** *(GENERATED — do not hand-edit)*
- Regenerate via `python scripts/derive_recording_rates.py`. Confirm `S_RACE_YEAR_LOGIT`/`_SIGMA` and `PREV_RACE_YEAR`/`_SIGMA` become shape `[7, 9]`, with the multi-race row a weak fallback and its `PREV` row `NaN`. *Mechanical once the generator is right.*

**`selection/data.py`** *(mandatory cross-file edit — the data half of the promotion)*
- `RACE_MAP` (`:65`): add the new key (`6:5` or `6:6` per layout). *Mechanical.*
- `RACE_UNKNOWN_IDX` (`:66`): set to the chosen Unknown index (6 under the recommended layout; unchanged under the append-at-6 layout). *Mechanical.*
- `race_case` SQL (`:168-173`): add a `WHEN 6 THEN …` arm so code 6 gets its own index and only `NULL`/other falls to `RACE_UNKNOWN_IDX`. *Mechanical.*
- Module docstring (`:14-21`) and `RACE_MAP` comments (`:60-66`): drop the "6 and NULL share the Unknown cell" note; document the new idx. *Mechanical.*
- `_sanity_check` (`:330-334`): *no edit* — bounds `race_idx` by `N_RACE` automatically.

**`selection/model.py`** *(no edits required)*
- Race dimension is sized solely from `coords["race"] = np.arange(N_RACE)` (`:99`); every race RV uses `dims="race"`/`("race","year")` with the data-driven `race_idx`; the `prev_margin` loop iterates `range(target_mat.shape[0])` (`:285-314`). All auto-resize once `N_RACE`, the prior arrays, and the anchor rows are length 7. *No change.*

**`selection/simulate.py`, `diagnostics.py`, `config.py`, `sampling.py`, `io.py`, `__init__.py`** *(no edits required)*
- `simulate.py` draws `race_idx = rng.integers(0, N_RACE)` and sizes truth arrays `(N_RACE,)` from the priors — correct once the prior arrays are length 7. `diagnostics.py` derives `n_race` from posterior shape and labels via `RACE_LEVELS[:n_race]`, so the 7th label is picked up automatically (it falls back to `idx_6` only if `RACE_LEVELS` lacks the label). *Optional polish:* `posterior_predictive_by_stratum('race_idx')` labels bars with the raw integer, so the new group reads as `5`/`6` unless a label map is added. The others have no race assumptions.

**Tests** *(fast-suite edits are required to keep CI green; slow re-tuning is opt-in)*
- `tests/test_selection_data.py:319-326` (`test_code_maps_are_complete`): rewrite the hard-coded `RACE_MAP` keys/values to the new mapping. *Mechanical but required.*
- `tests/test_selection_data.py:73` (`tiny_db` fixture): add `6` to the `for race in (1,2,3,4,5,None)` tuple so the new arm is exercised; add an assertion that `mracehisp_c=6` and `NULL` map to distinct indices. *Mechanical.*
- `tests/test_selection_model_compile.py:92,96,110`: change `n_race = 6` → 7; move the `NaN` row to the new Unknown index; keep the anchored-group assertion at `(5 * N_YEAR,)` if multi-race stays unanchored (recommended) — only become `6 * N_YEAR` if a finite multi-race target is supplied. **Decision-dependent.**
- `tests/test_selection_priors.py:31-41` (`test_factor_level_lengths_match_arrays`): *no literal edit* — but this is the canonical length-lock guard; it fails until every race array is extended to 7.
- `tests/test_selection_parameter_recovery.py:101-176` (`@pytest.mark.slow`): re-check the `min_coverage = 0.7` threshold (discrete coverage levels shift for a length-7 axis, e.g. 5/7 = 71% vs 4/7 = 57%) and the R-hat < 1.05 gate, which a sparse 7th level can stress. **May need re-tuning of `min_coverage` / draws.**
- `test_selection_simulate.py`, `test_selection_diagnostics.py`, `test_render_selection_diagnostics.py`: *no edit* — `N_RACE`/shape-parameterised; just ensure `RACE_LEVELS` has the 7th label.

**Committed reference artefacts** *(regenerate, do not hand-edit)*
- `data/reference/recording_rates_by_race_year.csv` and `notes/figures/recording_rates_anchor.csv` encode the old 6-row layout and are regenerated by the rates script. Grep `docs/` and `notes/` for `idx 5`, `idx-5`, `race_idx == 5`, and `Unknown` and update prose (esp. under the recommended Unknown 5→6 shift).

## 6. Risks & open decisions

- **Multi-race prior/anchor choice (open decision).** The new `ETA_DETECT_RACE` and `ETA_TERM_RACE` offsets and the `s(race, year)` recording row are *substantive prior assumptions, not fills*. There is little external evidence on screening uptake / termination / recording by multi-race status; a neutral 0.00 (matching Unknown) is the most defensible default but biases multi-race toward the NH White reference. Flag these as unpinned placeholders, not literature-derived.
- **2014+ restriction (decision: not needed).** `mracehisp_c=6` exists only from 2014, but the selection window starts at 2016, so the group is fully populated in-window and no restriction is required. Caveat: `derive_recording_rates.py._load` reads 2000-2024 for the age structure, so pre-2014 years have zero multi-race births — harmless because the named-anchor set excludes the group, but confirm no divide-by-zero / empty-group assumption trips when a race index has zero births in a year.
- **Index-order fragility.** Inserting multi-race anywhere other than the agreed slot silently re-indexes every existing offset/anchor row and breaks the de Graaf alignment *without raising an error*. The two unanchored rows (multi-race + Unknown) must carry `NaN` `PREV`; verify after regeneration that both `PREV` rows are `NaN`, else the margin loop will try to anchor an unanchored group.
- **Unknown-cell composition shift.** Splitting multi-race out shrinks Unknown to `NULL`-only (307k / 187 DS) — genuinely missing-race. Unknown's existing weak-neutral priors no longer describe the same population; re-confirm they remain appropriate and that no code assumes Unknown is the largest/least-informative cell.
- **Headline-total movement.** The DS total is sensitive to where `s` is pinned (notes record ~38-46k swings on `s` alone). Multi-race currently rides the 0.40 fallback inside Unknown; giving it its own (still-weak) row will move the total. Re-run the dev preset before/after to quantify; treat as cosmetic only if the shift is small.
- **Re-fit + re-tune cost.** Adding a sparse 7th level can lower ESS / raise R-hat on `s_race_year[5]` and `eta_term_race[5]`; the slow recovery tests (`draws=600/tune=600`) may need more draws or an index-specific threshold to avoid spurious failures.
- **Backward-compat of saved fits.** Existing `output/selection/*/idata.nc` and `summary.csv` have a length-6 race axis. `selection_coefficients.py` (reshapes to `len(RACE_LEVELS)`) and `compare_selection_variants.py` (`RACE_LEVELS[idx]`) will mis-map or raise on stale 6-level artefacts. There is no version guard tying `idata` race-length to `RACE_LEVELS` — **all reporting fits (variants A/B/C[/D]) must be regenerated** after the change.
- **Margin asymmetry (document).** With `PREV = NaN`, multi-race births are excluded from the surveillance-tied total while their recorded DS still count in `R_obs` — the same asymmetry Unknown has today. Document it so the de Graaf-vs-model reconciliation (the 41k vs 46k discussion) is not misread.

## 7. Effort estimate

Small-to-moderate, concentrated in three source files plus the generated artefact and tests; **`model.py` needs zero edits**.

- **Source edits:** `priors.py` (≈3 lines: one label + two array entries), `data.py` (≈4-5 lines: `RACE_MAP`, `RACE_UNKNOWN_IDX`, SQL arm, comments), `derive_recording_rates.py` (≈6-10 lines: two CASE arms + matrix-height widening + header text). Genuine modelling content is just the four new offset/fallback values.
- **Regeneration:** one run of `python scripts/derive_recording_rates.py` to rebuild `recording_anchor.py` and the two reference CSVs; verify shapes `[7, 9]` and `NaN` `PREV` rows.
- **Tests:** ~3 fast-suite test edits (`test_selection_data.py` ×2, `test_selection_model_compile.py`); the length-lock and simulate/diagnostics tests pass for free once arrays are length 7. Slow recovery tests (`pytest -m slow`) likely need light re-tuning of `min_coverage` / draws — budget one or two iterations to settle the R-hat and coverage gates on the sparse new level.
- **Fits:** one parameter-recovery + small `--profile dev` end-to-end run to confirm shapes line up, then a full re-fit of all reporting variants (the dominant wall-clock cost). Stale 6-level fits cannot be reused.

Files touched (absolute): `V:\dev\dspopulations\us-birth-certificates\src\dspopulations_us_birth_certificates\selection\priors.py`, `…\selection\data.py`, `…\selection\recording_anchor.py` (generated), `…\selection\model.py` (verify-only), `V:\dev\dspopulations\us-birth-certificates\scripts\derive_recording_rates.py`, `…\tests\test_selection_data.py`, `…\tests\test_selection_model_compile.py`, `…\tests\test_selection_priors.py`, `…\tests\test_selection_parameter_recovery.py`.