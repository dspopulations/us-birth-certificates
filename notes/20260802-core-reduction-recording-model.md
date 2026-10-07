> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# The initial core reduction and recording model

**Original date:** 2 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP001` reduced the selection model to counterfactual live-birth prevalence, combined prenatal reduction and certificate recording. It omitted demographic, clinical and separate screening/termination terms.

```text
p_true[y,a] = theta_lb[a] * (1 - rho[y])
p_recorded[y,a] = p_true[y,a] * s + (1 - p_true[y,a]) * f
R[y,a] ~ Binomial(N[y,a], p_recorded[y,a])
```

`rho` is the reduction relative to the Morris counterfactual live-birth rate. It is not the probability of fetal loss from conception. `s` is sensitivity among true live-born cases. `f` is a probability among births without Down syndrome.

The priors constrain the Morris rate and the annual reduction. The recorded counts mainly constrain their product with recording. Thus the model estimates a total conditional on the external reduction inputs, rather than learning that total from certificates alone.

The original seven age bands average risk within bands. Later exact-age models test that approximation. `DSP002` tests annual recording; `DSP003` adds an age-reduction term. The [current inventory](../docs/models/README.md) describes all ten models and their validation requirements.

The model reports expected true and missed counts for each parameter draw. Those intervals do not add realised-count variation for unobserved individual cases. Keep that distinction when describing a total.

The initial plan and command list have been replaced by the [workflow guide](../docs/modelling-workflow.md). This note preserves the model's motivation, rather than prescribing the preferred current model.
