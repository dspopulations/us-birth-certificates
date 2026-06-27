# Data-preparation adjustments: validation against NCHS user guides

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Validation of the variable-harmonisation and type-clamp adjustments in `scripts/duckdb_prepare.py`, `scripts/prepare_parquet.py`, `src/dspopulations_us_birth_certificates/variables.py`, and the data-preparation docs against the NCHS natality user guides (1989-standard `Nat20xxdoc`, 2003-revision `UserGuide20xx`). Every flagged item was adversarially re-checked against the guide text. Most fixes below have been **applied**; the remainder are decisions left to the maintainer.

## Summary

| Outcome (per adjudicated item, 6 clusters) | Count |
|---|---|
| Validated correct | 30 |
| Questionable (survived adjudication) | 13 |
| Incorrect (confirmed) | 2 |
| Undocumented choice | 3 |
| of which: documentation gaps / inaccuracies | 13 |

The numeric/behavioural mappings that actually fire in production are overwhelmingly correct. The residual problems were one genuine data-loss bug (`DPLURAL`), one latent (unreachable) bug (`MBRACE` two-digit scheme), several comment/docstring inaccuracies, and a provenance gap on the externally-sourced model parameters.

## Confirmed issues and resolution

### 1. `mrace_c` rule mis-stated in `previous/.../data-preparation.md` (HIGH, doc) — **FIXED**
`previous/.../data-preparation.md` stated "if mrace15, use **mrace6**, 4-14:4" — impossible (MRACE6 has only codes 1-6) and contradicting the code, which keys off **MRACE15**. Evidence: MRACE6 codes 1-6 (`UserGuide2014.txt` 518-524); MRACE15 01-15 including 15 "More than one race" (`UserGuide2014.txt` 526-542). Fix: rewrote the rule to MRACE15 with `15 -> missing` and the era-aware MBRACE mapping.

### 2. `DPLURAL (1,4)` clamp nulls valid quintuplet-plus births (MEDIUM, code) — **FIXED**
DPLURAL code 5 ("Quintuplet or higher") is valid 1989-2019; the `(1,4)` clamp NULLed every `DPLURAL=5` record. Evidence: code 5 present `Nat2000doc.txt` 2224-2225, `UserGuide2019-508.txt` 2731-2732; collapsed to "4 Quadruplet or higher" only from 2020 (`UserGuide2020.txt` 2775-2778). Fix: `prepare_parquet.py` clamp -> `(1,5)` (valid pre-2020, never produced 2020+, loses no data). Low volume (tens/yr).

### 3. `MBRACE` branch wrong for the 2003-2013 two-digit scheme (MEDIUM, code / latent) — **FIXED**
The `IN(1,2,3,4)` test was correct only for the 2014-2019 one-digit scheme; for the 2003-2013 two-digit scheme it dropped A/PI codes 05-14 and bridged-multiple 21-24 to NULL. **Unreachable in production** (MRACEREC precedes MBRACE 2003-2013; MRACE15 precedes it 2014-2019; only PR 2014-2019 reaches it, where it is one-digit), but `prepare_parquet.py` admits the two-digit codes, so it could corrupt if reached. Evidence: two-digit scheme `Nat2003doc.txt` 1078-1096, `UserGuide2009.txt` 1239-1258; one-digit `UserGuide2014.txt` 544-556. Fix: made the branch era-aware (`1,21->1; 2,22->2; 3,23->3; 4,24,5-14->4`; verified in DuckDB) and corrected the `MBRACE` docstring to 2003-2019 with the scheme-change note.

### 4. `MHISPX` / `FHISPX` docstrings claim "(2014+)" but field exists only from 2018 (MEDIUM, doc) — **FIXED**
2014-2017 file positions 112-114 / 159 are FILLER; MHISPX/FHISPX (with code 5 = Dominican) appear from 2018. Evidence: `UserGuide2014.txt` 569 (FILLER), `UserGuide2018-508.txt` 1067-1083 and Technical Notes 4426-4429. Fix: docstrings -> "(2018+)".

### 5. `ca_down_c` code comment contradicts behaviour for `downs = 8` (MEDIUM, doc) — **FIXED**
No `WHEN downs = 8` arm, so `downs=8` -> NULL, but the inline comment said "treated as unknown". Evidence: 1989-cert anomaly header 1/2/8/9 (`Nat2002doc.txt` 3835-3838). Fix: reworded the comment to "falls through to NULL, distinct from 9 -> 'U'". The NULL semantics (8 = structurally absent) are correct; `docs/data-preparation.md` was already right.

### 6. `mracehisp_c` reconstructed (not the NCHS field) + `MRACEHISP` docstring incomplete (MEDIUM, doc) — **FIXED**
Reconstruction is the correct choice (raw `MRACEHISP` is dual-coded across eras and absent pre-2003) but was undocumented; the imported `MRACEHISP` docstring covered only the 2014+ scheme. Evidence: old scheme `UserGuide2005.txt` 736-745; new scheme `UserGuide2014.txt` 582-589. Fix: documented the reconstruction in the `MRACEHISP_C` docstring and `data-preparation.md`, and expanded the `MRACEHISP` docstring with both era schemes and the naming clash.

### 7. Multi-race / unknown -> NULL in `mrace_c` is silent (MEDIUM, doc) — **FIXED (documented)**
Collapsing all multi-race births to NULL silently drops a growing share of 2014+ records. Fix: documented the deliberate NULL outcome in the `MRACE_C` docstring and `data-preparation.md`. **Open decision:** whether to instead add an explicit "More than one race" category to `mrace_c` (a modelling change that would ripple into `mracehisp_c`/`ds_case_weight`).

### 8. Lookup-series provenance undocumented; two series carried past their evidence horizon (MEDIUM, doc) — **FIXED (documented)**
`p_ds_lb_wt` carries 2018 forward to 2024 (seven identical literals); reduction rates extrapolated 2020-2024; age/ethnicity CSVs stop at 2018. Math correct; provenance missing. Fix: added a "Provenance of model parameters" section to `data-preparation.md`. **Open (optional):** replace repeated literals with a named constant + comment; move the inline prevalence table to a versioned CSV; cite Morris et al. and each series' source/vintage.

### 9. Lower-severity tidy-ups
- `p_ds_lb_wt_mage_reduc` `_mage` **misnomer** (multiplies `p_ds_lb_nt`, not `p_ds_lb_wt_mage`) — behaviour already documented; only the name misleads. **Open (optional)** rename.
- Three **placeholder columns** (`p_ds_lb_nt_mage`, `p_ds_lb_wt_ethn`, `p_ds_lb_nt_ethn`) ADDed but never populated — **FIXED**: docstrings now say "Placeholder - not yet populated".
- `UCA_DOWNS` docstring omitted code 8 (2003 file only) — **FIXED**.
- Maternal-age "Under 15" -> 14 via +13 — **FIXED**: caveat added to `MAGE36` docstring and `data-preparation.md` (immaterial; lowest analytic boundary is age 20).
- `MAGER9 (1,14)` and `DOWNS (0,255)` loose clamps — keep all valid codes; **open (cosmetic)** tighten to `(1,9)`.

## Verified-correct (no change needed)

- **DS indicator:** DOWNS 1/2/8/9 definitions; `ca_down_c` 1->C/2->N/9->U and the UCA_DOWNS/CA_DOWN/CA_DOWNS mappings and era claims; COALESCE pass-through; `down_ind` treating Pending ('P') as positive (matches NCHS practice, `UserGuide2014` PAGE 47) and 'U' -> NULL.
- **Maternal race:** the MRACE15, MRACEREC-US, and 1989-cert MRACE branches; the `IS NOT NULL` precedence MRACE15 > MRACEREC > MBRACE > MRACE; Asian (04-10)+NHOPI (11-14) -> 4.
- **Maternal Hispanic:** all `mhisp_c` mappings via MHISP_R/MHISPX/UMHISP/ORRACEM (including ORRACEM 6-8 -> 0 and MHISPX 4-6 -> 4); `mracehisp_c` core mapping.
- **Maternal age:** `COALESCE(mager, dmage, mage36+13, mager41+13)` over disjoint eras; MAGE36/MAGER41 share the 01-41 coding; 2003-via-MAGER41 recovery.
- **Year & clamps:** `year = COALESCE(dob_yy, datayear)`; clamps `MAGER (12,50)`, `MAGE36/MAGER41 (1,41)`, `UCA_DOWNS (1,9)`, `DBWT (0,9999)`, `APGAR5 (0,99)`, `PRECARE (0,99)` keep all codes + sentinels.
- **Probabilities & weights:** Morris constants reproduce the published curve (~1/1477 at 20, ~1/86 at 40); the age-split, reduction, and weight arithmetic are internally consistent.

## Outstanding decisions

1. **Re-run the pipeline?** The `DPLURAL (1,5)` fix changes output (recovers `DPLURAL=5`, tens of records/yr); the `MBRACE` fix is unreachable (no output change). The current `us_births.db` is otherwise valid. Re-run (~1 hr) only if you want `DPLURAL=5` materialised.
2. **Add a "More than one race" category to `mrace_c`?** (modelling change) vs keep the documented NULL exclusion.
3. **Optional refactors:** centralise the Morris formula (call `chance.py` from SQL); rename `p_ds_lb_wt_mage_reduc`; move the inline prevalence table + carry-forward to a versioned CSV/named constant; tighten the `MAGER9`/`DOWNS` clamps; add a prep-time count asserting the `MBRACE` two-digit branch is never reached.

Source guides for all coding claims: `data/*.pdf` (extracted text was used for grep-based validation).
