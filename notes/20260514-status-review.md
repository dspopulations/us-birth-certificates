> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude, Opus 4.7) on
> 2026-05-14 after the bug-sweep PR (#38) was opened on
> `dev/frank/project-review`. Inventory, timeline, and pipeline structure
> are pulled directly from the repo and PR history; the interpretation
> of findings and the prioritised next steps are AI-synthesised from the
> existing `notes/` corpus and should be confirmed by a human reviewer
> before any external citation or planning use.

# May review of study scope and unresolved assumptions

**Original date:** 14 May 2026. Historical research note, revised for clarity in October 2026.

This was a project-status snapshot. Its task lists, old run totals and setup instructions have been replaced by the [project plan](../plans/readme.md), [model inventory](../docs/models/README.md) and [workflow guide](../docs/modelling-workflow.md).

The review identified a lasting distinction between three tasks. Descriptive analyses measure recorded births. Classifiers rank births by similarity to the recorded label. Selection models estimate unrecorded totals under assumptions about natural prevalence, prenatal reduction and recording.

None of these tasks validates individual missed cases without an external reference. In particular, a quota based on an assumed recording rate sets a cohort size; it cannot provide independent evidence for that recording rate.

Two factual problems in the original review have been removed. Its surveillance total and annual conversion were inconsistent. It also attributed a recording sensitivity near 0.40 to Boulet. The [August source review](20260804-salemi-boulet-study-area-transport.md) corrects that attribution and states the assumptions needed to compare local studies with national data.

Later work addressed the Apgar filter, race/Hispanic-origin harmonisation, the surveillance extraction and the core-model numerical checks. Their remaining scientific limits are documented in the current guides and dated audit notes. Completed work is no longer listed here as pending.
