"""Fit the core age-reduction-recording Bayesian model."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import duckdb
import numpy as np

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output
from dspopulations_us_birth_certificates.selection import (
    FitContext,
    copy_docs_template,
    diagnostics,
    render_report,
    sample,
    save_artefacts,
    save_summary,
    selection_run_config,
)
from dspopulations_us_birth_certificates.selection.anomaly_panel import (
    DEFAULT_ANOMALY_CONDITIONS_CSV,
    DEFAULT_PANEL_CONDITION_TREND_SIGMA,
    DEFAULT_PANEL_FACTOR_SIGMA,
    DEFAULT_PANEL_FROM_YEAR,
    DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA,
    DEFAULT_PANEL_LOADING_SIGMA,
    DEFAULT_PANEL_PREVALENCE_TREND_SIGMA,
    AnomalyPanelConditions,
    prepare_anomaly_panel,
)
from dspopulations_us_birth_certificates.selection.core_models import (
    core_model_names,
    get_core_model_definition,
)
from dspopulations_us_birth_certificates.selection.core_reduction import (
    CORE_REDUCTION_MODEL_ID,
    DEFAULT_ANCHOR_CSV,
    DEFAULT_ANCHOR_LEVEL_SIGMA,
    DEFAULT_ANCHOR_OBS_SIGMA,
    DEFAULT_ANCHOR_OBS_SIGMA_FIXED,
    DEFAULT_ANCHOR_TREND_SIGMA,
    DEFAULT_ANCHOR_WINDOW_HALF_WIDTH,
    DEFAULT_EXTRAPOLATED_REDUCTION_START,
    DEFAULT_RECORDING_S_DRIFT_SIGMA,
    DEFAULT_REDUCTION_AGE_STEP_SIGMA,
    DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT,
    DEFAULT_REDUCTION_CSV,
    DEFAULT_REDUCTION_ERROR_CORRELATION,
    CoreReductionModelConfig,
    CoreReductionPriors,
    SurveillanceAnchor,
    build_core_reduction_model,
    core_year_summary,
    prepare_core_age_year_cells,
)
from dspopulations_us_birth_certificates.selection.core_reporting import (
    render_core_all,
)
from dspopulations_us_birth_certificates.selection.priors import FALSE_POSITIVE_RATE


def _parse_years(raw: str | None, fallback: tuple[int, int]) -> tuple[int, int]:
    if raw is None:
        return fallback
    try:
        lo, hi = raw.split("-", 1)
        return int(lo), int(hi)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--years expects 'YYYY-YYYY', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit the core age-reduction-recording Bayesian model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "model",
        nargs="?",
        default="DSP001",
        help=f"Core model ID. Valid models: {', '.join(core_model_names())}.",
    )
    p.add_argument("--profile", default="dev", choices=("dev", "reporting"))
    p.add_argument(
        "--years",
        default=None,
        help="Year range as 'YYYY-YYYY'. Defaults to 2016-2024.",
    )
    p.add_argument("--random-seed", type=int, default=47)
    p.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path("data/us_births.db"),
        help="Path to the DuckDB database.",
    )
    p.add_argument(
        "--reduction-csv",
        type=Path,
        default=DEFAULT_REDUCTION_CSV,
        help="CSV containing year,reduction prior means.",
    )
    p.add_argument(
        "--extrapolated-start",
        type=int,
        default=DEFAULT_EXTRAPOLATED_REDUCTION_START,
        help="First year treated as extrapolated for wider reduction uncertainty.",
    )
    p.add_argument("--observed-reduction-sigma", type=float, default=0.20)
    p.add_argument("--extrapolated-reduction-sigma", type=float, default=0.45)
    p.add_argument(
        "--reduction-error-correlation",
        type=float,
        default=DEFAULT_REDUCTION_ERROR_CORRELATION,
        help=(
            "Correlation between yearly logit-scale reduction-prior errors. "
            "Marginal yearly variances are preserved."
        ),
    )
    p.add_argument(
        "--reduction-calibration-shift-logit",
        type=float,
        default=DEFAULT_REDUCTION_CALIBRATION_SHIFT_LOGIT,
        help="Fixed logit-scale shift applied to the complete reduction trajectory.",
    )
    p.add_argument("--recording-s-mean", type=float, default=0.5)
    p.add_argument("--recording-s-sigma", type=float, default=1.0)
    p.add_argument(
        "--false-positive-rate",
        type=float,
        default=FALSE_POSITIVE_RATE,
        help="Fixed probability that a non-DS birth is recorded as DS.",
    )
    p.add_argument(
        "--recording-s-year-sigma",
        type=float,
        default=0.35,
        help=(
            "Logit-scale prior SD for centred year offsets in s_year models. "
            "Ignored by constant-s models."
        ),
    )
    p.add_argument(
        "--recording-s-drift-sigma",
        type=float,
        default=DEFAULT_RECORDING_S_DRIFT_SIGMA,
        help=(
            "Per-year logit-scale SD of the random walk on recording sensitivity "
            "over the years no surveillance window covers. Used only by models "
            "with recording_drift='post_anchor'. Set 0 for the all-prevalence "
            "corner, which reproduces the undrifted parent model exactly."
        ),
    )
    p.add_argument(
        "--anchor-forecast-flat",
        action="store_true",
        help=(
            "Hold latent prevalence at its last anchored value instead of "
            "forecasting it, so the whole post-window decline in the recorded "
            "rate is absorbed by recording. This is the all-recording corner "
            "opposite --recording-s-drift-sigma 0."
        ),
    )
    p.add_argument(
        "--reduction-age-step-sigma",
        type=float,
        default=DEFAULT_REDUCTION_AGE_STEP_SIGMA,
        help=(
            "Logit-scale prior SD for adjacent-age RW1 increments in "
            "age-specific reduction models."
        ),
    )
    p.add_argument(
        "--anchor-csv",
        type=Path,
        default=DEFAULT_ANCHOR_CSV,
        help=(
            "Pooled surveillance anchor CSV, from "
            "scripts/extract_degraaf_surveillance.py. Used only by anchored models."
        ),
    )
    p.add_argument(
        "--anchor-half-width",
        type=int,
        default=DEFAULT_ANCHOR_WINDOW_HALF_WIDTH,
        help="Half-width of the surveillance averaging window, in years.",
    )
    p.add_argument(
        "--anchor-level-sigma",
        type=float,
        default=DEFAULT_ANCHOR_LEVEL_SIGMA,
        help=(
            "Half-normal prior scale for the year-to-year SD of latent log "
            "prevalence. Measured at 1.5-1.8%% in the model-family review."
        ),
    )
    p.add_argument(
        "--anchor-trend-sigma",
        type=float,
        default=DEFAULT_ANCHOR_TREND_SIGMA,
        help="Half-normal prior scale for drift in the latent trend slope.",
    )
    p.add_argument(
        "--anchor-obs-sigma",
        type=float,
        default=DEFAULT_ANCHOR_OBS_SIGMA,
        help=(
            "Half-normal prior scale for the surveillance observation SD. Used "
            "only with --anchor-obs-sigma-estimated, since the SD is fixed by "
            "default."
        ),
    )
    p.add_argument(
        "--anchor-obs-sigma-fixed",
        type=float,
        default=DEFAULT_ANCHOR_OBS_SIGMA_FIXED,
        help=(
            "Surveillance observation SD, held fixed. This is the sensitivity axis "
            "for surveillance accuracy: report across 0.05 / 0.10 / 0.20 and say "
            "which value was chosen."
        ),
    )
    p.add_argument(
        "--anchor-obs-sigma-estimated",
        action="store_true",
        help=(
            "Estimate the surveillance observation SD instead of fixing it. NOT "
            "recommended: the estimate measures only whether the windows agree "
            "with a smooth path, not whether surveillance is accurate, and a free "
            "SD admits a degenerate mode in which the anchor switches off. Audit "
            "any such fit with scripts/audit_anchored_chain_health.py."
        ),
    )
    p.add_argument(
        "--panel-conditions-csv",
        type=Path,
        default=DEFAULT_ANOMALY_CONDITIONS_CSV,
        help=(
            "Curated congenital-anomaly checkbox table naming the control "
            "conditions and the reason each one is or is not admissible. Used "
            "only by anomaly-panel models."
        ),
    )
    p.add_argument(
        "--panel-conditions",
        default=None,
        help=(
            "Comma-separated override of the control set, e.g. "
            "'ca_hypo,ca_clpal'. Any curated condition may be named, including "
            "an excluded one, so a sensitivity fit can add one back "
            "deliberately. Default: every condition marked 'control'."
        ),
    )
    p.add_argument(
        "--panel-from-year",
        type=int,
        default=DEFAULT_PANEL_FROM_YEAR,
        help=(
            f"First panel year. Cannot precede {DEFAULT_PANEL_FROM_YEAR}, when "
            "revised-certificate coverage reaches 100%%; earlier years would read "
            "state composition as recording behaviour."
        ),
    )
    p.add_argument(
        "--panel-prevalence-trend-sigma",
        type=float,
        default=DEFAULT_PANEL_PREVALENCE_TREND_SIGMA,
        help=(
            "Prior SD on a true-prevalence trend shared by every control, log "
            "per year. This is the assumption the panel cannot check internally. "
            "Set 0 to assert the exclusion restriction exactly; raise it to move "
            "back towards DSP009."
        ),
    )
    p.add_argument(
        "--panel-loading-sigma",
        type=float,
        default=DEFAULT_PANEL_LOADING_SIGMA,
        help=(
            "Prior SD on the Down syndrome loading on the item-wide recording "
            "factor, centred on 1."
        ),
    )
    p.add_argument(
        "--panel-loading-fixed",
        type=float,
        default=None,
        help=(
            "Pin the Down syndrome loading instead of estimating it. "
            "--panel-loading-fixed 1.0 is the strict shared-factor corner: it "
            "asserts DS recording moved exactly with the item, which the panel's "
            "own internal disagreement argues against. A reportable corner, not "
            "a default."
        ),
    )
    p.add_argument(
        "--panel-factor-sigma",
        type=float,
        default=DEFAULT_PANEL_FACTOR_SIGMA,
        help=(
            "Fixed per-year SD of the common log-rate walk innovation. Fixed "
            "rather than estimated on purpose: an estimated scale shrinks the "
            "common change, which is the quantity being measured."
        ),
    )
    p.add_argument(
        "--panel-condition-trend-sigma",
        type=float,
        default=DEFAULT_PANEL_CONDITION_TREND_SIGMA,
        help=(
            "Half-normal prior scale for the SD of the per-condition trend "
            "deviations, which is where the controls' disagreement lives."
        ),
    )
    p.add_argument(
        "--panel-idiosyncratic-sigma",
        type=float,
        default=DEFAULT_PANEL_IDIOSYNCRATIC_SIGMA,
        help=("Half-normal prior scale for the year-by-condition idiosyncratic term."),
    )
    p.add_argument(
        "--confirmed-only",
        action="store_true",
        help=(
            "Count only confirmed certificate DS flags as recorded cases; pending "
            "flags remain in the birth denominator but count as non-cases."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: output/selection_core_reduction/"
            "<DSPnnn>/<timestamp>)."
        ),
    )
    p.add_argument("--prior-only", action="store_true")
    p.add_argument(
        "--render",
        action="store_true",
        help="Copy and render the checked-in Quarto report template.",
    )
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=None)
    p.add_argument("--target-accept", type=float, default=None)
    p.add_argument("--prior-predictive-samples", type=int, default=None)
    p.add_argument(
        "--nuts-sampler",
        default=None,
        help="Override sampler backend in the selected profile, e.g. nutpie or pymc.",
    )
    ns = p.parse_args(argv)
    try:
        ns.model_definition = get_core_model_definition(ns.model)
    except ValueError as exc:
        p.error(str(exc))
    ns.year_range = _parse_years(ns.years, (2016, 2024))
    if ns.anchor_obs_sigma_estimated:
        if ns.anchor_obs_sigma_fixed != DEFAULT_ANCHOR_OBS_SIGMA_FIXED:
            p.error(
                "--anchor-obs-sigma-estimated and --anchor-obs-sigma-fixed are "
                "mutually exclusive; pick one"
            )
        # Downstream the two controls are a single value: None means estimate.
        ns.anchor_obs_sigma_fixed = None
    if ns.anchor_forecast_flat and ns.model_definition.reduction_model != "anchor":
        p.error(
            "--anchor-forecast-flat applies only to surveillance-anchored models; "
            f"{ns.model_definition.model_id} sets its level from the reduction CSV"
        )
    ns.panel_condition_names = (
        None
        if ns.panel_conditions is None
        else [name.strip() for name in ns.panel_conditions.split(",") if name.strip()]
    )
    if ns.model_definition.recording_panel != "anomaly":
        for flag, value in (
            ("--panel-conditions", ns.panel_condition_names),
            ("--panel-loading-fixed", ns.panel_loading_fixed),
        ):
            if value is not None:
                p.error(
                    f"{flag} applies only to anomaly-panel models; "
                    f"{ns.model_definition.model_id} has no panel channel"
                )
    ns.recorded_definition = (
        "confirmed_only" if ns.confirmed_only else "confirmed_or_pending"
    )
    ns.output_dir = ns.output_dir or (
        Path("output")
        / CORE_REDUCTION_MODEL_ID
        / ns.model_definition.model_id
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    ns.overrides = {
        k: v
        for k, v in {
            "draws": ns.draws,
            "tune": ns.tune,
            "chains": ns.chains,
            "target_accept": ns.target_accept,
            "prior_predictive_samples": ns.prior_predictive_samples,
            "nuts_sampler": ns.nuts_sampler,
        }.items()
        if v is not None
    }
    return ns


def _build_run_config(ns: argparse.Namespace):
    base = selection_run_config(ns.profile, random_seed=ns.random_seed)
    if not ns.overrides:
        return base
    return replace(base, **ns.overrides)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    setup.init_script()
    np.random.seed(ns.random_seed)
    ns.output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.banner(
        "fit_core_reduction_model",
        (
            f"model={ns.model_definition.model_id}  "
            f"years={ns.year_range[0]}-{ns.year_range[1]}  profile={ns.profile}"
        ),
    )
    cli_output.info(ns.model_definition.title)

    cli_output.section("Environment")
    package_metadata.report_package_versions(list(PACKAGE_LIST))

    cli_output.section("Run configuration")
    run_config = _build_run_config(ns)
    cli_output.info(
        f"draws=[bold]{run_config.draws}[/bold], "
        f"tune=[bold]{run_config.tune}[/bold], "
        f"chains=[bold]{run_config.chains}[/bold], "
        f"target_accept=[bold]{run_config.target_accept}[/bold], "
        f"sampler=[bold]{run_config.nuts_sampler}[/bold]"
    )
    cli_output.info(f"output_dir=[blue]{ns.output_dir}[/blue]")

    cli_output.section("Load age-year cells")
    con = duckdb.connect(str(ns.duckdb_path), read_only=True)
    try:
        cells = prepare_core_age_year_cells(
            con,
            year_range=ns.year_range,
            age_model=ns.model_definition.age_model,
            recorded_definition=ns.recorded_definition,
            split_revision=ns.model_definition.recording_model == "revision",
        )
        # Read the panel while the connection is still open. It is reported and
        # validated further down, next to the anchor it is tested against.
        panel = (
            prepare_anomaly_panel(
                con,
                year_range=ns.year_range,
                panel_from_year=ns.panel_from_year,
                conditions=AnomalyPanelConditions.from_csv(ns.panel_conditions_csv),
                condition_names=ns.panel_condition_names,
            )
            if ns.model_definition.recording_panel == "anomaly"
            else None
        )
    finally:
        con.close()
    n_total = int(cells["N_cell"].sum())
    r_total = int(cells["R_cell"].sum())
    cli_output.print_kv(
        "Cells",
        [
            ("n_cells", len(cells)),
            ("n_total (livebirths)", f"{n_total:,}"),
            ("r_total (recorded DS)", f"{r_total:,}"),
            ("recorded_rate", f"{r_total / n_total:.2e}"),
            ("age_model", ns.model_definition.age_model),
            ("recorded_definition", ns.recorded_definition),
            ("year_range", cells.attrs.get("year_range")),
        ],
    )

    cli_output.section("Build model")
    priors = CoreReductionPriors.from_reduction_csv(
        year_range=ns.year_range,
        path=ns.reduction_csv,
        observed_logit_sigma=ns.observed_reduction_sigma,
        extrapolated_logit_sigma=ns.extrapolated_reduction_sigma,
        reduction_error_correlation=ns.reduction_error_correlation,
        reduction_calibration_shift_logit=ns.reduction_calibration_shift_logit,
        extrapolated_start=ns.extrapolated_start,
        recording_s_mean=ns.recording_s_mean,
        recording_s_sigma=ns.recording_s_sigma,
        recording_s_year_sigma=ns.recording_s_year_sigma,
        recording_s_drift_sigma=ns.recording_s_drift_sigma,
        reduction_age_step_sigma=ns.reduction_age_step_sigma,
        false_positive_rate=ns.false_positive_rate,
    )
    cli_output.info(
        "reduction prior: "
        f"observed sigma={ns.observed_reduction_sigma}, "
        f"extrapolated sigma={ns.extrapolated_reduction_sigma} "
        f"from {ns.extrapolated_start}"
    )
    cli_output.info(
        "reduction calibration: "
        f"year-error correlation={ns.reduction_error_correlation}, "
        f"fixed logit shift={ns.reduction_calibration_shift_logit:+.3g}"
    )
    if ns.model_definition.recording_model == "year":
        cli_output.info(
            "recording sensitivity: partially pooled s_year offsets "
            f"(sigma={ns.recording_s_year_sigma})"
        )
    elif ns.model_definition.recording_model == "revision":
        cli_output.info(
            "recording sensitivity: separate levels for revised and unrevised "
            "certificates (recording_s is the revised level)"
        )
    else:
        cli_output.info("recording sensitivity: constant s")
    if ns.model_definition.recording_drift == "post_anchor":
        if ns.recording_s_drift_sigma > 0.0:
            cli_output.info(
                "recording drift: random walk on logit s over the years no "
                f"surveillance window covers (sigma={ns.recording_s_drift_sigma}). "
                "The split of the post-window decline between prevalence and "
                "recording is set by this prior, NOT identified by the data - "
                "report both corners alongside it"
            )
        else:
            cli_output.warning(
                "recording drift: switched off (sigma=0). This is the "
                "all-prevalence corner and reproduces the undrifted parent model "
                "exactly."
            )
    if ns.anchor_forecast_flat:
        cli_output.info(
            "anchor forecast: latent prevalence held flat past the last window, "
            "so the whole post-window decline is attributed to recording "
            "(all-recording corner)"
        )
    if ns.model_definition.reduction_model == "year_age":
        cli_output.info(
            "combined reduction: exact-age Morris curve with centred RW1 age "
            f"effect (step sigma={ns.reduction_age_step_sigma})"
        )
    elif ns.model_definition.age_model == "single_year":
        cli_output.info(
            "combined reduction: common across maternal age within each year; "
            "Morris curve evaluated at represented age codes"
        )
    cli_output.info(f"false-positive rate: fixed at {ns.false_positive_rate:.3g}")
    anchor = None
    if ns.model_definition.reduction_model == "anchor":
        anchor = SurveillanceAnchor.from_csv(
            year_range=ns.year_range,
            path=ns.anchor_csv,
            half_width=ns.anchor_half_width,
        )
        details = anchor.to_dict()
        cli_output.print_kv(
            "Surveillance anchor",
            [
                ("source", anchor.source),
                ("windows", details["n_windows"]),
                ("mid_years", f"{anchor.mid_years[0]}-{anchor.mid_years[-1]}"),
                (
                    "effective independent",
                    f"{details['effective_independent_windows']:.1f}",
                ),
                ("window width", 2 * anchor.half_width + 1),
                ("level sigma prior", ns.anchor_level_sigma),
                ("trend sigma prior", ns.anchor_trend_sigma),
                (
                    "obs sigma",
                    (
                        f"estimated (prior scale {ns.anchor_obs_sigma})"
                        if ns.anchor_obs_sigma_fixed is None
                        else f"fixed at {ns.anchor_obs_sigma_fixed}"
                    ),
                ),
            ],
        )
        cli_output.info(
            "level: latent log prevalence with a local linear trend, observed "
            "through centred window means; the reduction-rate CSV prior is "
            "loaded for comparison only and does NOT enter the likelihood"
        )
        if ns.anchor_obs_sigma_fixed is None:
            cli_output.warning(
                "surveillance observation SD is ESTIMATED. It measures only "
                "whether the windows agree with a smooth path, not whether "
                "surveillance is accurate, so the resulting interval overstates "
                "precision - and a free SD admits a degenerate mode in which the "
                "anchor switches off. Audit this fit with "
                "scripts/audit_anchored_chain_health.py before citing it."
            )
        else:
            cli_output.info(
                "surveillance observation SD fixed at "
                f"{ns.anchor_obs_sigma_fixed}: this is an assumption about "
                "surveillance accuracy, not an estimate. Report across the "
                "0.05 / 0.10 / 0.20 axis and say which value was chosen"
            )
    if panel is not None:
        assert anchor is not None
        panel_details = panel.to_dict()
        heterogeneity = panel_details["heterogeneity"]
        overlap = panel.anchored_overlap_years(
            int(anchor.mid_year_idx.max()) + anchor.half_width
        )
        cli_output.print_kv(
            "Anomaly panel",
            [
                ("curation", panel.conditions_source),
                ("controls", ", ".join(panel.condition)),
                ("years", f"{panel.years[0]}-{panel.years[-1]}"),
                ("reference year", panel.years[panel.reference_year_idx]),
                (
                    "anchored overlap",
                    f"{len(overlap)} years"
                    + (f" ({overlap[0]}-{overlap[-1]})" if overlap else ""),
                ),
                (
                    "flags/year",
                    f"{sum(panel_details['flags_per_year']) / panel.n_year:.0f}",
                ),
                (
                    "loading",
                    (
                        f"fixed at {ns.panel_loading_fixed}"
                        if ns.panel_loading_fixed is not None
                        else f"estimated (prior 1 +- {ns.panel_loading_sigma})"
                    ),
                ),
                (
                    "common prevalence trend",
                    (
                        f"prior 0 +- {ns.panel_prevalence_trend_sigma} log/year"
                        if ns.panel_prevalence_trend_sigma > 0.0
                        else "asserted zero (exclusion restriction exact)"
                    ),
                ),
            ],
        )
        if not overlap:
            cli_output.warning(
                "no panel year is covered by a surveillance window, so the Down "
                "syndrome loading is determined by its prior alone and the panel "
                "cannot be tested against the anchor at all"
            )
        if heterogeneity.get("available"):
            # The panel's credibility is its controls' agreement. Print it every
            # run, because a shared factor the controls contradict is exactly the
            # thing a reader would otherwise take on trust.
            cli_output.info(
                "panel heterogeneity: "
                f"I^2={heterogeneity['i_squared'] * 100:.0f}%, "
                f"Q={heterogeneity['q_statistic']:.1f} on "
                f"{heterogeneity['degrees_of_freedom']} df, "
                f"between-condition SD={heterogeneity['tau'] * 100:.1f}%"
            )
            change = (np.exp(heterogeneity["random_effect_log_change"]) - 1.0) * 100.0
            cli_output.info(
                "panel common change "
                f"{heterogeneity['early_years'][0]}-"
                f"{heterogeneity['early_years'][-1]} to "
                f"{heterogeneity['late_years'][0]}-"
                f"{heterogeneity['late_years'][-1]}: {change:+.2f}% "
                f"(random-effects SE {heterogeneity['random_effect_se'] * 100:.2f}%)"
            )
            if heterogeneity["i_squared"] > 0.5:
                cli_output.warning(
                    "the control conditions disagree markedly about the common "
                    "recording change, so the shared factor is a weighted "
                    "compromise rather than a measured quantity. Do not read the "
                    "panel as identifying the split - it narrows the range the "
                    "prior alone would give, and no more"
                )
        if ns.panel_loading_fixed is not None:
            cli_output.warning(
                f"Down syndrome loading PINNED at {ns.panel_loading_fixed}. This "
                "asserts the shared-factor restriction rather than testing it; "
                "report it as a corner beside the estimated fit"
            )
        if ns.panel_prevalence_trend_sigma <= 0.0:
            cli_output.warning(
                "common prevalence trend asserted to be exactly zero, so the "
                "whole common movement of the controls is attributed to recording"
            )

    model = build_core_reduction_model(
        cells,
        priors,
        n_year=cells.attrs["n_year"],
        recording_model=ns.model_definition.recording_model,
        reduction_model=ns.model_definition.reduction_model,
        recording_drift=ns.model_definition.recording_drift,
        recording_panel=ns.model_definition.recording_panel,
        anchor=anchor,
        anchor_level_sigma=ns.anchor_level_sigma,
        anchor_trend_sigma=ns.anchor_trend_sigma,
        anchor_obs_sigma=ns.anchor_obs_sigma,
        anchor_obs_sigma_fixed=ns.anchor_obs_sigma_fixed,
        anchor_forecast_flat=ns.anchor_forecast_flat,
        panel=panel,
        panel_prevalence_trend_sigma=ns.panel_prevalence_trend_sigma,
        panel_loading_sigma=ns.panel_loading_sigma,
        panel_loading_fixed=ns.panel_loading_fixed,
        panel_factor_sigma=ns.panel_factor_sigma,
        panel_condition_trend_sigma=ns.panel_condition_trend_sigma,
        panel_idiosyncratic_sigma=ns.panel_idiosyncratic_sigma,
    )

    cli_output.section("Sample")
    idata = sample(model, config=run_config, prior_only=ns.prior_only)

    cli_output.section("Save artefacts")
    model_config = CoreReductionModelConfig.from_priors(
        year_range=ns.year_range,
        priors_obj=priors,
        model_definition=ns.model_definition,
        recorded_definition=ns.recorded_definition,
        notes=f"profile={ns.profile}",
        anchor=anchor,
        anchor_hyperpriors={
            "level_sigma": ns.anchor_level_sigma,
            "trend_sigma": ns.anchor_trend_sigma,
            "obs_sigma": ns.anchor_obs_sigma,
            "obs_sigma_fixed": ns.anchor_obs_sigma_fixed,
            "forecast_flat": ns.anchor_forecast_flat,
        },
        panel=panel,
        panel_hyperpriors={
            "panel_prevalence_trend_sigma": ns.panel_prevalence_trend_sigma,
            "panel_loading_sigma": ns.panel_loading_sigma,
            "panel_loading_fixed": ns.panel_loading_fixed,
            "panel_factor_sigma": ns.panel_factor_sigma,
            "panel_condition_trend_sigma": ns.panel_condition_trend_sigma,
            "panel_idiosyncratic_sigma": ns.panel_idiosyncratic_sigma,
        },
    )
    context = FitContext(
        config=model_config,
        run_config=run_config,
        output_dir=ns.output_dir,
        cells=cells,
        model=model,
        idata=idata,
    )
    save_artefacts(context, ns.output_dir)
    cli_output.success(f"idata.nc, cells.parquet, config.json -> {ns.output_dir}")
    qmd_path = copy_docs_template(ns.model_definition.template_id, ns.output_dir)
    if qmd_path is not None:
        cli_output.success(f"index.qmd -> {qmd_path}")
    else:
        cli_output.warning(
            "No Quarto template at "
            f"docs/models/{ns.model_definition.template_id}/index.qmd."
        )

    if ns.prior_only:
        cli_output.info("Prior-only run - skipping posterior summary / report render.")
        return 0

    cli_output.section("Posterior summary")
    summary_vars = [
        "rho_year",
        "eta_year",
        "recording_s",
        "recording_s_year",
        "true_count_year",
        "true_count_total",
    ]
    if "recording_s_year_offset" in idata.posterior:
        summary_vars.insert(4, "recording_s_year_offset")
    if "recording_s_unrevised" in idata.posterior:
        summary_vars[3:3] = ["recording_s_unrevised", "recording_s_unrevised_offset"]
    if "recording_s_drift_logit" in idata.posterior:
        # The innovations are the actual free random variables, so the convergence
        # gate has to see them and not only their cumulated transform.
        summary_vars[3:3] = [
            "recording_s_drift_ratio",
            "recording_s_drift_logit",
            "recording_s_drift_innovation_raw",
        ]
    if "panel_recording_log_factor" in idata.posterior:
        # As with the drift, the walk innovations are the free variables and the
        # cumulated factor is a transform, so the gate must see both. The
        # anchored-era coordinates of recording_s_panel_logit are identically zero
        # and return NaN diagnostics by construction, so it is deliberately absent.
        summary_vars[3:3] = [
            "recording_s_panel_ratio",
            "panel_loading_ds",
            "panel_prevalence_trend",
            "panel_recording_factor_ratio",
            "panel_common_change_ratio",
            "panel_recording_log_factor",
            "panel_common_innovation_raw",
            "panel_condition_trend",
            "panel_condition_trend_raw",
            "panel_condition_trend_scale",
            "panel_condition_trend_mean",
            "panel_condition_logit_level",
            "panel_idiosyncratic_scale",
            "panel_idiosyncratic_raw",
        ]
    if "prevalence_year" in idata.posterior:
        summary_vars[0:0] = [
            "prevalence_year",
            "anchor_window_prevalence",
            "anchor_obs_sigma",
            "anchor_level_sigma",
            "anchor_trend_sigma",
        ]
    if "rho_age_offset" in idata.posterior:
        summary_vars.insert(2, "rho_age_offset")
    if "rho_logit_year_raw" in idata.posterior:
        summary_vars.insert(2, "rho_logit_year_raw")
    summary = diagnostics.summary_table(
        idata,
        var_names=tuple(summary_vars),
    )
    save_summary(summary, ns.output_dir)
    year_summary = core_year_summary(idata, cells)
    year_summary.to_csv(ns.output_dir / "year_summary.csv", index=False)
    health = diagnostics.convergence_health(summary)
    cli_output.info(
        f"max Rhat=[bold]{health['max_rhat']:.4f}[/bold] "
        f"(target <{health['rhat_threshold']}), "
        f"min ESS=[bold]{health['min_ess']:.0f}[/bold] "
        f"(target >={health['ess_threshold']:.0f})"
    )
    if health["all_ok"]:
        cli_output.success("Convergence checks passed.")
    else:
        cli_output.warning("Convergence checks did not all pass - inspect summary.csv.")

    cli_output.section("Report outputs")
    render_core_all(
        idata,
        cells,
        ns.output_dir,
        priors_config=priors.to_dict(),
        year_range=ns.year_range,
        recording_model=ns.model_definition.recording_model,
    )
    cli_output.success(f"plots/ and tables/ -> {ns.output_dir}")
    render_report(qmd_path, do_render=ns.render)

    cli_output.section("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
