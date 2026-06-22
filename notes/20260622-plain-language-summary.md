# Estimating Down syndrome births from US birth certificates — a plain-language summary

**Date:** 2026-06-22
**Audience:** readers with some statistics background (frequentist) but not
necessarily Bayesian modelling. Jargon is defined as it appears.
**Status:** preliminary. Numbers are from fast ("dev") fits; a higher-quality
("reporting") run is pending. All figures should be treated as provisional.

---

## 1. The question

How many babies are actually born with Down syndrome (DS) in the United States?
We have US birth certificates for 2016–2024 — about **33.5 million births**, of
which **17,776 were recorded as having Down syndrome**. The catch: birth
certificates are known to **miss** many true cases (validation studies that
checked certificates against medical records find only roughly 40% of true DS
babies get the box ticked). So the recorded count is an undercount, and we want
the **true** number — recorded **plus** missed.

You cannot get that by counting. You have to **model** it: write down the chain of
events that turns a real DS conception into (or fails to turn it into) a ticked box
on a certificate, attach a probability to each step, and use the observed counts to
estimate those probabilities — and hence the missing cases.

## 2. The model: a chain from conception to checkbox

We split the journey into three stages, each a probability:

1. **θ (theta) — the natural rate.** Out of all births to mothers of a given age,
   what fraction would be DS livebirths *if no-one screened or terminated*? This is
   biology, and it rises steeply with maternal age (about 1 in 1,500 under age 20,
   about 1 in 33 at age 45+). It's well measured by earlier studies (Morris et
   al.), so we treat it as known.
2. **η (eta) — surviving to birth.** Not every DS pregnancy is born: many are
   detected by prenatal screening and the pregnancy is electively terminated. η is
   the fraction that *survive to a livebirth*. (So `1 − η` is the termination
   "reduction".)
3. **s — getting recorded.** Given a DS baby *is* born, what's the chance the
   certificate records it? Validation studies put this around 0.40 — i.e. ~60% are
   missed.

Put together, the chance a given birth shows up as a *recorded* DS case is roughly

```
recorded rate ≈ θ × η × s     (plus a tiny rate of false positives)
```

and the **true** number of DS livebirths is `θ × η` summed over all births.

## 3. A note for the frequentist reader: what "Bayesian" adds

In a frequentist analysis you'd maximise a **likelihood** — the probability of the
data given the parameters — and report point estimates with confidence intervals.
We use the same likelihood (here, a binomial count of recorded DS in each
demographic "cell"), but we also attach a **prior** to each parameter: a
distribution that encodes what earlier studies already tell us (e.g. "s is about
0.40"). Multiplying prior × likelihood and renormalising gives the **posterior** —
the updated distribution of each parameter given both the data *and* the outside
knowledge. A **95% credible interval** is then read directly as "95% probability
the parameter lies here" (conditional on the model and priors).

Why bother? Because — as the next section shows — the data alone *cannot* pin down
all the parameters. The external knowledge in the priors is doing essential work.
That is a strength (it lets us use everything we know) and a liability (the answer
leans on assumptions), and being explicit about it is the whole point.

## 4. The hard part: things that only ever appear multiplied together

Look again at `recorded rate ≈ θ × η × s`. The data only ever see the **product**.
That means the data cannot, by themselves, tell apart:

- *"few DS babies, but well recorded"* (low η, high s), from
- *"many DS babies, but poorly recorded"* (high η, low s).

Both give the same recorded counts. This is called **non-identifiability**. The
closest frequentist analogue is **perfect multicollinearity** in a regression: when
two predictors move together, you can estimate their combined effect but not their
separate coefficients. Here, η and s (and, it turns out, θ) are tangled the same
way. The data fix the product; *something else* has to fix the split.

That "something else" is the priors — i.e. external numbers we pin down. The total
DS estimate therefore depends not just on the data but on **which external numbers
we trust and how hard we pin them**. Keep this in mind; it's the main caveat.

## 5. What went wrong the first time (and what it taught us)

Our first runs produced a nonsense answer: about **226,000** true DS livebirths
over 2016–2024 — roughly **five times** the figure independent surveillance implies
(~45,000). Worse, three model variants that should have disagreed if the answer
were prior-driven all gave the *same* inflated number, which initially looked
reassuring but was actually all three sharing one fault.

The fault: the "natural rate" θ — which we *thought* was nailed down by biology —
quietly drifted to **3–4× its known value**. We had given it a "tight" prior
(a standard deviation of 0.10 on the log-odds scale), but with **33.5 million**
data points the likelihood is so overwhelmingly strong that a 0.10 prior behaves
like a loose suggestion, not a constraint. The data dragged θ 11–15 standard
deviations away from its prior — something that should be astronomically unlikely,
and only happens because the prior was, in effect, far weaker than it looked.

*Why* did θ drift? Because the recorded-DS pattern across maternal age didn't match
what the model could otherwise produce. Older mothers screen and terminate far
more, so their *recorded* DS rate, relative to the natural rate, is much lower than
younger mothers'. The model had no way to express "termination rises steeply with
age" (that age effect was missing from η), so the only knob it could turn to fit
the age pattern was θ itself — and it turned it, breaking the biology.

**Lesson:** at this data scale, "informative prior" is not the same as "fixed
number". Anything you mean to hold fixed has to be pinned *hard*.

## 6. What we changed

Four changes, all aimed at putting each effect where it belongs and pinning the
genuinely-external quantities so the data can't overwrite them:

1. **Pinned θ hard** (standard deviation 0.10 → 0.001). θ is now held at the
   biological values; it can no longer absorb other effects.
2. **Put maternal age where the biology says it acts.** Age now enters three
   places: the natural rate θ (conception — already there), **screening access**
   (older mothers reach screening more), and **termination choice** (it varies with
   age). The steep age pattern now lands in η — the screening/termination stage —
   instead of corrupting θ.
3. **Pinned the recording rate s hard too** (it was escaping the same way θ did),
   at the validation value ~0.40.
4. **Dropped "clinical flags" from the recording stage.** We had let conditions
   like congenital heart defect (CCHD) and NICU admission influence the *recording*
   probability. They misbehaved badly — the model decided NICU babies are recorded
   almost perfectly — because those conditions are actually signs the baby *is* DS
   (CCHD is caused by DS about half the time), not signs of better paperwork. The
   model wasn't allowed to say "more true DS here", so it said "better recording
   here" instead. Removing them stops the distortion. (They're kept for a separate
   analysis of co-occurring conditions, where they belong as *outcomes* of DS.)

We also refreshed the screening numbers for the modern era: prenatal screening
shifted from older blood tests to **cell-free DNA (NIPT)** between roughly 2013 and
2022, which sharply raised detection (and so termination) — a literature review
documents this and feeds the age/year structure of η.

## 7. The result

With those fixes the model behaves sensibly and the three variants now **agree**
(they no longer share a hidden fault):

| quantity | estimate (2016–2024) | plain meaning |
|---|---|---|
| **True DS livebirths** | **~39,000–40,000** | recorded + missed |
| Recorded (data) | 17,776 (~15,200 after removing estimated false positives) | what the certificates caught |
| **Implied missed** | **~24,000 (~61%)** | true babies the certificates didn't record |
| Recording rate `s` | ~0.39 | fraction of true DS recorded |
| Termination reduction `1 − η` | ~46%, rising 7% (under 20) → 64% (45+) | share of DS pregnancies electively terminated |

Read it as a funnel: absent screening, roughly **72,000** DS babies would have been
born; about **46%** of those pregnancies were electively terminated, leaving
**~39,000** born; of those, only **~39%** were recorded, leaving **~24,000 missed**.
The model also reproduces the recorded counts age-band by age-band (a
"posterior-predictive check") to within about 5%.

## 8. Limitations and criticisms

This is where honesty matters most. The model works, but the result should not be
oversold.

- **It is a reconstruction, not a measurement.** Because of the
  non-identifiability (Section 4), the birth-certificate data *cannot* pin the
  total on their own. We resolved the ambiguity by **assumption** — pinning the
  recording rate `s` and the natural rate θ from outside studies. Change the
  assumption and the headline moves: pinning `s = 0.40` gives ~39k; `s = 0.34`
  gives ~46k. **The total is roughly inversely proportional to the assumed
  recording rate.** So the honest one-line statement is: *"if certificates record
  ~40% of DS births, then there were ~39,000."* The "if" is load-bearing.

- **The external numbers may not fit this population.** The recording-rate
  validation studies are older and from a few states; the screening/termination
  literature is extrapolated to the national 2016–2024 population. If those don't
  generalise, the estimate is biased in ways the credible intervals do **not**
  capture (the intervals reflect sampling noise, not the risk that a pinned
  assumption is simply wrong).

- **"Fits the data" ≠ "correct".** The model reproduces the recorded counts by
  construction — that's what fitting does. We have **no independent gold standard**
  (e.g. a linked birth-defects registry) to check the *missed* cases against,
  because that data isn't accessible. The age-band check is reassuring but weak: it
  only confirms internal consistency, not external truth.

- **We can't separate two of the age effects.** The data identify only the
  *combined* effect of age on η. How much of the age gradient is "older mothers
  access screening more" versus "older mothers choose termination more" is
  **prior-driven** — a modelling choice, not a finding. Treat any such split as
  illustrative.

- **Recording is surely not one flat number.** We pinned `s` at a single level
  (modulated only mildly by race/education). In reality it varies by hospital,
  state, year and case mix. Collapsing that is a simplification that could bias
  subgroup comparisons.

- **The co-occurring-conditions model is unresolved.** The current model treats DS
  as statistically independent of conditions like CCHD, which is biologically
  false. We sidestepped it for the headline total, but it limits the planned
  analysis of conditions that co-occur with DS.

- **There's a real gap to independent estimates.** Our ~39–40k (at `s = 0.40`) sits
  below surveillance-based figures (de Graaf and colleagues: ~48k). Either our
  termination estimate is a little high, or true recording is below the validation
  studies. That gap is unresolved — it is itself a caveat, and a question worth
  chasing, not something we should quietly tune away.

- **Provisional.** These are fast-fit numbers; the publication-quality run is still
  pending, and the project's own plan flags all data and models as preliminary.

## 9. Bottom line

We built a transparent, internally-consistent framework that turns 17,776 recorded
DS births into an estimate of **~39,000–46,000 true DS livebirths** for 2016–2024,
depending on the assumed recording rate — implying that **roughly 55–65% of cases
are missing from birth certificates**, with a clear maternal-age gradient in
elective termination.

Its real value is not the single number but that **every assumption is explicit and
you can see how the answer depends on it**. The estimate is only as good as the
external screening and recording figures it leans on; it is a literature-anchored
reconstruction, not a direct count, and the honest reading is conditional: *"given
what published studies say about screening and recording, this is the implied
number."* Strengthening it means better external anchors — ideally a linked
registry — not more clever modelling of the same birth-certificate data, which has
already told us everything it structurally can.
