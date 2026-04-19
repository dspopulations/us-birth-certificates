"""Tests for ``manifest.write_manifest`` and ``data_fingerprint``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates import manifest
from dspopulations_us_birth_certificates.models import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)


def _stub_context(output_dir: Path) -> ModelFitContext:
    config = ModelConfig(
        model_id="manifest_smoke",
        variant_of=None,
        target_var="y",
        numeric_features=("a",),
        categorical_features=("b",),
        base_params={"objective": "binary"},
        params={"learning_rate": 0.05},
        train_config={"training_split": 0.8},
        year_range=(2020, 2024),
        include_unknown=True,
        selection_history=(),
        shap_scatter_specs=(),
    )
    return ModelFitContext(
        config=config,
        run_config=RunConfig.from_name("dev", random_seed=123),
        output_dir=output_dir,
        X_train=pd.DataFrame({"a": np.arange(80, dtype=float), "b": np.zeros(80)}),
        y_train=pd.Series(np.concatenate([np.ones(10), np.zeros(70)])),
        X_valid=pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.zeros(20)}),
        y_valid=pd.Series(np.concatenate([np.ones(3), np.zeros(17)])),
        metrics={"average_precision": 0.42},
    )


def test_data_fingerprint_counts_and_schema_hash() -> None:
    X_train = pd.DataFrame({"a": np.arange(80, dtype=float), "b": np.zeros(80)})
    X_valid = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.zeros(20)})
    y_train = pd.Series(np.concatenate([np.ones(10), np.zeros(70)]))
    y_valid = pd.Series(np.concatenate([np.ones(3), np.zeros(17)]))

    fp = manifest.data_fingerprint(X_train, X_valid, y_train, y_valid)
    assert fp["n_train"] == 80
    assert fp["n_valid"] == 20
    assert fp["n_positive_train"] == 10
    assert fp["n_positive_valid"] == 3
    assert isinstance(fp["schema"], str)
    assert len(fp["schema"]) == 16


def test_data_fingerprint_schema_changes_with_dtype() -> None:
    # Same column names, different dtype → different hash.
    X1 = pd.DataFrame({"a": np.arange(10, dtype=float)})
    X2 = pd.DataFrame({"a": np.arange(10, dtype="int64")})

    fp1 = manifest.data_fingerprint(X1, None, None, None)
    fp2 = manifest.data_fingerprint(X2, None, None, None)
    assert fp1["schema"] != fp2["schema"]


def test_data_fingerprint_schema_stable_across_runs() -> None:
    X1 = pd.DataFrame({"a": np.arange(10, dtype=float), "b": np.zeros(10)})
    X2 = pd.DataFrame({"a": np.arange(50, dtype=float), "b": np.ones(50)})
    # Different row counts, same schema → identical hash.
    fp1 = manifest.data_fingerprint(X1, None, None, None)
    fp2 = manifest.data_fingerprint(X2, None, None, None)
    assert fp1["schema"] == fp2["schema"]


def test_write_manifest_produces_expected_shape(tmp_path: Path) -> None:
    ctx = _stub_context(tmp_path)
    path = manifest.write_manifest(ctx, tmp_path)
    assert path == tmp_path / "manifest.json"
    assert path.is_file()

    data = json.loads(path.read_text())
    assert data["config"]["model_id"] == "manifest_smoke"
    assert data["run_config"]["name"] == "dev"
    assert data["run_config"]["random_seed"] == 123
    assert data["data_fingerprint"]["n_train"] == 80
    assert data["data_fingerprint"]["n_positive_train"] == 10
    assert data["metrics"]["average_precision"] == 0.42
    # git info is present even if the repo isn't a git clone (values are None)
    assert "git" in data
    assert "environment" in data
    assert "packages" in data["environment"]
    # Tracked packages are listed whether or not each is installed
    assert "lightgbm" in data["environment"]["packages"]
    assert "scikit-learn" in data["environment"]["packages"]


def test_write_manifest_extra_field(tmp_path: Path) -> None:
    ctx = _stub_context(tmp_path)
    manifest.write_manifest(ctx, tmp_path, extra={"note": "hello"})
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["extra"] == {"note": "hello"}


def test_write_manifest_roundtrips_through_json(tmp_path: Path) -> None:
    ctx = _stub_context(tmp_path)
    manifest.write_manifest(ctx, tmp_path)
    # If any field couldn't serialise, write would raise; load to double-check.
    data = json.loads((tmp_path / "manifest.json").read_text())
    # Tuples from ModelConfig.to_dict come back as lists.
    assert data["config"]["numeric_features"] == ["a"]
    assert data["config"]["year_range"] == [2020, 2024]


def test_manifest_skipped_without_crash_when_no_data(tmp_path: Path) -> None:
    # No X_train / X_valid populated → fingerprint returns empty but write still works.
    config = ModelConfig(
        model_id="empty",
        variant_of=None,
        target_var="y",
        numeric_features=(),
        categorical_features=(),
        base_params={},
        params={},
        train_config={},
        year_range=(2020, 2024),
        include_unknown=False,
        selection_history=(),
        shap_scatter_specs=(),
    )
    ctx = ModelFitContext(
        config=config,
        run_config=RunConfig.from_name("dev"),
        output_dir=tmp_path,
    )
    manifest.write_manifest(ctx, tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["data_fingerprint"] == {}
