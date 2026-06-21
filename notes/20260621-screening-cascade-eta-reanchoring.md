# Re-anchoring the Stage-2 (η) screening/termination cascade

**Date:** 2026-06-21
**Status:** Literature assessment — provisional. Awaiting FB's recent termination-rate
papers (the "~90%" claim) before finalising `priors.py` constants.
**Scope:** Stage 2 of the selection model only (`eta_detect`, `eta_term`). Stage 1
(θ_LB) and Stage 3 (s) untouched here.

> Sourcing note: quantitative anchors below are from PubMed-indexed articles (DOIs
> given inline). Programme-level uptake/coverage figures are from grey literature
> (ACOG bulletins, lab-coverage reports) and are flagged as such — they shape the
> *trajectory* but should not be treated as peer-reviewed point estimates.

---

## 1. Why this came up

The committed model writes the screening/termination pass-through as

```
η = 1 − η_detect · η_term
```

with `ETA_DETECT_BASELINE = 0.70` and `ETA_TERM_BASELINE = 0.67`. The 0.67 is
Natoli 2012 — a pre-2012, US, *pooled* termination-given-diagnosis rate. Two
problems:

1. **It conflates two cascade steps.** `η_term` should be termination given a
   *confirmed diagnosis*; the step from *positive screen* → *confirmed diagnosis*
   (governed by the screen's PPV and the woman's follow-through) is a separate,
   strongly **time-varying** quantity that lives in `η_detect`.
2. **The serum→NIPS transition (2013–2024) moved the first step, not the second.**
   The "% true positives given a positive screen" (PPV) rose by more than an order
   of magnitude as cfDNA replaced serum screening. That is the engine driving η's
   time trend across our 2016–2024 window.

Only the **product** `η_detect · η_term` enters η, so for the model we need the
*combined* time path of `P(a true DS pregnancy is electively terminated)` — we do
not have to split detection from termination internally.

---

## 2. Strand 1 — Screen performance: serum vs NIPS (trisomy 21)

According to PubMed:

| Quantity (T21) | Serum screening | cfDNA / NIPS |
|---|---|---|
| Detection rate (sensitivity) | combined FTS ~85%; quad ~70–81% at 5% FPR (standard screening lit; see §6) | **99.2%** (95% CI 98.5–99.6) — Gil 2015; **99.4%** — Mackie 2017; **99.3%** general-population — Iwarsson 2017 |
| False-positive rate | ~5% (by design of the risk cut-off) | **0.09%** — Gil 2015 |
| **PPV (true positives / screen positives)** | **~0.78%** head-to-head — Xiao 2022; generally low single digits | **81.5%** head-to-head — Xiao 2022; **85.7–90.8%** SNP-NIPT high-risk — Verma 2018 / Eiben 2015 |

- **Gil et al. 2015**, *Ultrasound Obstet Gynecol* — updated cfDNA meta-analysis,
  37 studies. T21 DR 99.2%, FPR 0.09%. [DOI](https://doi.org/10.1002/uog.14791)
- **Mackie et al. 2017**, *BJOG* — bivariate meta-analysis, 148 344 T21 tests:
  sensitivity 0.994, specificity 0.999; **"no evidence that population risk had any
  effect"** on sensitivity/specificity; cfDNA is a *screening* (not diagnostic)
  test for aneuploidy. [DOI](https://doi.org/10.1111/1471-0528.14050)
- **Iwarsson et al. 2017**, *Acta Obstet Gynecol Scand* — GRADE meta-analysis;
  **general** pregnant population T21 sensitivity 0.993, specificity 0.999; still
  advises confirming a positive by invasive testing if termination is considered.
  [DOI](https://doi.org/10.1111/aogs.13047)
- **Xiao et al. 2022**, *PLoS One* — same-population head-to-head (Zhuhai, 17 363
  women, 2018–19): serum screening **PPV 0.78%, DR 36.4%**; NIPT **PPV 81.5%, DR
  100%**. The single cleanest serum-vs-NIPS contrast.
  [DOI](https://doi.org/10.1371/journal.pone.0266718)
- **Suo et al. 2023**, *J Obstet Gynaecol* — NIPT PPV is **prior-risk-dependent**:
  14.3% / 64.3% / 86.4% for pre-test low / intermediate / high risk.
  [DOI](https://doi.org/10.1080/01443615.2023.2288226)
- **Yaron et al. 2015**, *Obstet Gynecol* — PPV is governed by prevalence; average-
  risk PPV < high-risk PPV for the same test. [DOI](https://doi.org/10.1097/AOG.0000000000001091)

**Takeaway:** PPV rose from ~1–5% (serum) to ~80–90% (NIPS, higher-prior) — but NIPS
PPV is itself prevalence-dependent and is lower (~50–80%) in younger/average-risk
women. Sensitivity rose from ~85% to ~99%; FPR fell ~50× to ~0.1%.

---

## 3. Strand 2 — The serum→NIPS transition over 2016–2024 (uptake)

cfDNA NIPT became commercially available in 2011 (high-risk), gained average-risk
validation ~2014–15, and **ACOG recommended it for all pregnant patients regardless
of risk in September 2020** (Practice Bulletin 226). Grey-literature programme data:
US average-risk NIPT commercial coverage reached ~77% of covered lives by ~2022;
state-level testing rates span ~30–70%; patient awareness rose ~30%→70% over five
years. (Sources: ACOG 2020; Natera/GrandView coverage reports — grey literature.)

A 2019 review (**Mikhaylova et al.**, *F1000Research*) confirms the qualitative arc:
cfDNA NIPT spread rapidly and globally post-2011, ACMG accepts it as able to replace
conventional screening, **but a positive must be confirmed invasively**.
[DOI](https://doi.org/10.12688/f1000research.16837.1)

**Shape for the model:** effective detection rises across 2016–2024, steepening after
~2018 and plateauing ~2022–24 as NIPS becomes the dominant first-tier screen. The
*level* is capped by uptake (not universal) and follow-through, so effective
detection of true DS sits well below the ~99% screen sensitivity.

---

## 4. Strand 3 — Termination given a confirmed diagnosis (the contested knob)

According to PubMed:

- **Natoli et al. 2012**, *Prenat Diagn* — the US systematic review (1995–2011, 24
  studies): **weighted mean 67%** (range 61–93%) among population-based studies; 85%
  (60–90%) hospital-based; 50% (0–100%) anomaly-based. Explicitly: termination rates
  **"have decreased in recent years"**, are **lower than** earlier international
  reviews, and **vary with maternal age, gestational age, and race/ethnicity** — "a
  summary termination rate may not be applicable to the entire US population."
  [DOI](https://doi.org/10.1002/pd.2910)
- **No post-Natoli US systematic review exists.** A recency-sorted search (129 hits,
  2017→2025) returned only non-US single-centre / anomaly-specific studies
  (Denmark, China, Singapore, France) — none supersedes Natoli for the US.
- NIPS-era European data argue termination-given-diagnosis is **not** rising with
  NIPS, and may fall as information-only testing grows:
  - **Lund et al. 2021**, *Acta Obstet Gynecol Scand* — Danish public NIPT: **45% of
    true-positive results led to live birth by choice** (11 DS liveborn by choice).
    [DOI](https://doi.org/10.1111/aogs.14052)
  - **Miltoft et al. 2018**, *Acta Obstet Gynecol Scand* — offering cfDNA raises
    follow-up uptake **"without a corresponding rise in the termination rate"**;
    cfDNA-only acceptors were far more likely to continue an affected pregnancy
    (30% vs 3.6%). [DOI](https://doi.org/10.1111/aogs.13297)

**Takeaway:** the ~90% figure is characteristic of **European registries (EUROCAT)
and US hospital-based series**, not US population-based data. The US population-based
anchor remains ~67%, heterogeneous by age/race/region, and if anything trending
down. NIPS changed *who gets detected*, not *the decision once diagnosed*.

---

## 5. Implications for the model

**Structural finding: the time-varying engine is `η_detect`, not `η_term`.**

1. **Keep the year trend on `η_detect`; re-anchor its *shape* to the serum→NIPS
   transition** (rising, steepening post-2018, plateauing 2022–24). The current
   offsets `[-0.25,-0.15,-0.05,0.05,0.15,0.20,0.25,0.28,0.30]` around a 0.70
   baseline give effective detection ~64%→76% — a defensible shape, but revisit the
   endpoints against an explicit cascade calc:
   `effective detection ≈ uptake(~0.7–0.9) × sensitivity(serum 0.85 → NIPS 0.99) ×
   follow-through(~0.85–0.95)`, i.e. ~0.55–0.65 early → ~0.75–0.85 late.

2. **Do *not* raise `ETA_TERM_BASELINE` to ~0.90.** That imports a European/hospital
   level into a US population model. Keep ~0.67 (Natoli US population-based), **widen
   the sigma** to reflect the contested level + real heterogeneity, retain the
   race/edu offsets (NH Black −0.70, Hispanic −0.40), and **let the data identify the
   level** rather than pinning it. A mild *downward* drift is as defensible as a flat
   prior; an upward drift is not well supported for the US.

3. **This is the identification lever.** Recording sensitivity `s` has no reason to
   track NIPS penetration; the termination cascade does. So with the `η_detect`
   time-*shape* anchored externally and `s` pinned from validation (Boulet/Salemi),
   the **year dimension** separates η from s — the residual time trend in recorded
   DS (net of the rising-maternal-age push that θ_LB(age) already absorbs) reads the
   η decline. Re-anchoring η must therefore land **before** the A/B/C pin-s sweep.

---

## 6. Open items / to confirm

- **FB's "~90%" papers** — anchor the termination discussion to those specific
  sources; classify each as US vs European and population- vs hospital-based.
- **Canonical serum detection rates** (combined FTS ~85% @ 5% FPR; quad ~70–81%) are
  stated here from standard screening literature (FASTER / SURUSS lineage) — pin an
  explicit citation before committing to `priors.py`.
- **Decision:** pin `η_term` at a US level with wide sigma, *or* leave it fully
  data-identified under the pin-s strategy. Recommend the latter.
- Consider whether `η_detect` deserves a **race/payer × year interaction** (NIPS
  uptake diffused unevenly — later/lower among Medicaid and some minority groups),
  which would matter for the demographic decomposition even if not for the total.

---

## 7. Citations (PubMed)

All via PubMed. Gil 2015 [DOI](https://doi.org/10.1002/uog.14791); Mackie 2017
[DOI](https://doi.org/10.1111/1471-0528.14050); Iwarsson 2017
[DOI](https://doi.org/10.1111/aogs.13047); Xiao 2022
[DOI](https://doi.org/10.1371/journal.pone.0266718); Suo 2023
[DOI](https://doi.org/10.1080/01443615.2023.2288226); Yaron 2015
[DOI](https://doi.org/10.1097/AOG.0000000000001091); Natoli 2012
[DOI](https://doi.org/10.1002/pd.2910); Lund 2021
[DOI](https://doi.org/10.1111/aogs.14052); Miltoft 2018
[DOI](https://doi.org/10.1111/aogs.13297); Verma 2018
[DOI](https://doi.org/10.1007/s13224-017-1061-9); Eiben 2015
[DOI](https://doi.org/10.1055/s-0035-1555765); Mikhaylova 2019
[DOI](https://doi.org/10.12688/f1000research.16837.1).
