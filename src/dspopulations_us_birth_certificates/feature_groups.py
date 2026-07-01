"""Feature-cluster helpers for grouped importance diagnostics.

The grouped-importance table is a diagnostic map for downstream modelling:
it ranks correlated dimensions of ``P(recorded DS | X)``, not causal effects
on true Down-syndrome livebirth rates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

DEFAULT_GROUP_DISTANCE_THRESHOLD = 0.30

_AGE_FEATURES = {"mage_c", "fagecomb"}
_TIME_FEATURES = {"year", "dob_mm", "dob_wk", "dob_tt_pm"}
_SOCIODEMOGRAPHIC_FEATURES = {
    "mracehisp",
    "meduc",
    "fracehisp",
    "feduc",
    "pay_rec",
    "wic",
}
_FACILITY_WORKFLOW_FEATURES = {"bfacil3", "precare", "ld_indl", "ld_augm"}
_DELIVERY_FEATURES = {"me_pres", "dmeth_rec"}
_PREGNANCY_RISK_FEATURES = {
    "bmi",
    "wtgain",
    "rf_pdiab",
    "rf_gdiab",
    "rf_phype",
    "rf_ghype",
    "rf_ehype",
    "rf_ppterm",
    "rf_inftr",
    "rf_fedrg",
    "rf_artec",
}
_CLINICAL_SEVERITY_FEATURES = {
    "dbwt",
    "gestrec10",
    "apgar5",
    "apgar10",
    "ab_aven1",
    "ab_aven6",
    "ab_nicu",
    "ab_surf",
    "ab_anti",
    "ab_seiz",
    "ca_cchd",
    "ca_anen",
    "ca_mnsb",
    "ca_cdh",
    "ca_omph",
    "ca_gast",
    "ca_limb",
    "ca_cleft",
    "ca_clpal",
    "ca_hypo",
}
_CERTIFICATE_PROXY_FEATURES = {"ca_disor"}
_INFANT_DEMOGRAPHIC_FEATURES = {"sex"}

_DIMENSION_METADATA: dict[str, tuple[str, str]] = {
    "age": (
        "theta_lb / eta_detect / eta_term",
        "age combines biological risk with screening and termination gradients",
    ),
    "time": (
        "eta_detect / eta_term / s",
        "calendar ranking does not by itself identify a rate trend",
    ),
    "sociodemographic": (
        "eta_detect / eta_term / s",
        "demographic recording-vs-termination split is not identified by ranking",
    ),
    "facility_workflow": (
        "s",
        "facility and workflow variables are plausible recording-sensitivity drivers",
    ),
    "delivery": (
        "s / clinical workflow",
        "delivery variables can mix clinical severity with certificate workflow",
    ),
    "pregnancy_risk": (
        "theta/eta or confounded clinical pathway",
        "risk-factor association may be mediated by age, care access, or severity",
    ),
    "clinical_severity": (
        "s / DS co-occurrence readout",
        "clinical severity can reflect true co-occurrence and recording propensity",
    ),
    "certificate_proxy": (
        "s / certificate workflow",
        "near-proxy for certificate-recorded anomaly; avoid true-rate interpretation",
    ),
    "infant_demographic": (
        "theta/eta/s unclear",
        "infant demographic association requires external interpretation",
    ),
    "data_cluster": (
        "unspecified",
        "data-derived cluster without a project-specific stage hint",
    ),
}


def feature_groups_from_linkage(
    feature_names: Sequence[str],
    linkage_matrix: np.ndarray,
    *,
    distance_threshold: float = DEFAULT_GROUP_DISTANCE_THRESHOLD,
    prefix: str = "cluster",
) -> dict[str, list[str]]:
    """Cut a feature dendrogram into ordered correlation groups.

    ``distance_threshold=0.30`` corresponds to distance-correlation greater
    than about 0.70, matching the existing feature-pruning rationale.
    """
    names = list(feature_names)
    if not names:
        return {}
    if len(names) == 1:
        return {f"{prefix}_01": names}

    labels = hierarchy.fcluster(
        linkage_matrix,
        t=distance_threshold,
        criterion="distance",
    )
    by_label: dict[int, list[str]] = defaultdict(list)
    for name, label in zip(names, labels, strict=True):
        by_label[int(label)].append(name)

    index = {name: i for i, name in enumerate(names)}
    ordered_groups = sorted(
        by_label.values(),
        key=lambda cols: min(index[c] for c in cols),
    )
    return {
        f"{prefix}_{i:02d}": sorted(cols, key=index.__getitem__)
        for i, cols in enumerate(ordered_groups, start=1)
    }


def annotate_feature_group(features: Iterable[str]) -> dict[str, str]:
    """Return cautious project-specific interpretation hints for a group."""
    feature_set = set(features)
    dimensions: list[str] = []
    for name, members in (
        ("certificate_proxy", _CERTIFICATE_PROXY_FEATURES),
        ("clinical_severity", _CLINICAL_SEVERITY_FEATURES),
        ("age", _AGE_FEATURES),
        ("time", _TIME_FEATURES),
        ("sociodemographic", _SOCIODEMOGRAPHIC_FEATURES),
        ("facility_workflow", _FACILITY_WORKFLOW_FEATURES),
        ("delivery", _DELIVERY_FEATURES),
        ("pregnancy_risk", _PREGNANCY_RISK_FEATURES),
        ("infant_demographic", _INFANT_DEMOGRAPHIC_FEATURES),
    ):
        if feature_set & members:
            dimensions.append(name)

    if not dimensions:
        dimensions = ["data_cluster"]

    stages = [_DIMENSION_METADATA[d][0] for d in dimensions]
    flags = [_DIMENSION_METADATA[d][1] for d in dimensions]
    return {
        "dimension_hint": " + ".join(dimensions),
        "candidate_stage": " + ".join(dict.fromkeys(stages)),
        "interpretation_flag": " | ".join(dict.fromkeys(flags)),
    }


def annotate_grouped_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Add dimension/stage interpretation hints to grouped importance output."""
    if df.empty:
        return df.assign(
            dimension_hint=pd.Series(dtype="object"),
            candidate_stage=pd.Series(dtype="object"),
            interpretation_flag=pd.Series(dtype="object"),
        )

    annotations = pd.DataFrame(
        [annotate_feature_group(features) for features in df["features"]]
    )
    return pd.concat([df.reset_index(drop=True), annotations], axis=1)


def grouped_importance_to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a CSV-safe version of a grouped-importance frame."""
    out = df.copy()
    if "features" in out:
        out["features"] = out["features"].map(lambda xs: json.dumps(list(xs)))

    preferred = [
        "rank",
        "group",
        "dimension_hint",
        "candidate_stage",
        "interpretation_flag",
        "n_features",
        "features",
        "baseline_score",
        "importance_mean",
        "importance_std",
    ]
    ordered = [c for c in preferred if c in out.columns]
    ordered.extend(c for c in out.columns if c not in ordered)
    return out[ordered]


def feature_groups_summary_frame(groups: dict[str, list[str]]) -> pd.DataFrame:
    """Serialisable summary of dendrogram-derived groups before permutation."""
    rows: list[dict[str, Any]] = []
    for rank, (group, features) in enumerate(groups.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "group": group,
                "n_features": len(features),
                "features": json.dumps(features),
                **annotate_feature_group(features),
            }
        )
    return pd.DataFrame(rows)
