# Plans

> [!WARNING]
> This is work in progress. All data and models are preliminary.

This study aims to provide updated estimates of the numbers of babies born with Down syndrome in the U.S. and to explore factors influencing those births and how they may be changing over time.

Recorded births of babies with Down syndrome in US birth certificate data are systematically **under-reported**, and the under-reporting is **not missing at random** — it varies with clinical severity, socioeconomic status, and race/ethnicity (Boulet 2011; Salemi 2017). This has a direct methodological consequence that shapes the whole study: because recording depends on the same characteristics we want to measure, a classifier trained on the recorded flag learns the *recording process*, not Down syndrome status, and **cannot identify which individual unrecorded births were missed cases without inheriting that bias** — clinically subtle cases are, by construction, indistinguishable from ordinary births in the certificate data. This is a positive–unlabelled learning problem under a biased (Selected-At-Random) labelling mechanism (Bekker and Davis, 2020).

We therefore estimate what *is* recoverable — the **number** of missed cases and its distribution across characteristics (an aggregate, or "class prior") — rather than attempting to label individual missed cases. The correction is made at the population level by a structural Bayesian **selection model** that anchors the recording rate to external validation and surveillance data. The machine-learning strand is retained for what it can legitimately do: characterising the recording process itself, recovering the narrow subset of clerical/communications omissions, and serving as an independent cross-check on the total.

More specifically, we aim to:

1. **Document** the numbers and characteristics of babies with Down syndrome **recorded** in birth certificate data from 1989 to 2024, and compare against surveillance-based estimates.

2. **Characterise the recording process.** Identify the factors that predict whether a Down syndrome livebirth is *recorded* on the birth certificate (2016–2024) — i.e. model the recording propensity `P(recorded | characteristics)` — by training, pruning, and evaluating a classifier and examining its predictors. This is explicitly a model of *recording*, not of true Down syndrome status.

3. **Estimate the total (recorded + missed) at the population level.** Estimate the total number of Down syndrome livebirths and the *number* of missed cases for 2016–2024 using a structural Bayesian selection model that corrects for under-ascertainment by anchoring the recording rate to external evidence. Individual missed cases are **not** identified — under the biased recording mechanism this is not achievable without bias — so the target is the missed *count* and its distribution, reported with explicit dependence on the recording assumption.

4. **Characterise the full population, including co-occurring conditions.** Document the numbers and characteristics of the full (recorded + missed) Down syndrome population for 2016–2024, estimating co-occurring-condition rates by **stratified class-prior estimation** (weighting each stratum by its recording rate) rather than by summing individually-predicted cases. Compare against surveillance and against previous co-occurrence estimates, and document how a naïve "predicted-missing" cohort *inverts* those rates.

5. **Model factors over time.** Develop a statistical model to explore and estimate factors influencing live births of babies with Down syndrome and how they may be evolving over time.
