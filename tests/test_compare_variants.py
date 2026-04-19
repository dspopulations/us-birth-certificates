"""Tests for ``scripts/compare_variants.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_variants  # noqa: E402


def _write_fake_run(
    run_dir: Path,
    model_id: str,
    ap: float,
    n_features: int = 30,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "average_precision": ap,
                "roc_auc": 0.9,
                "log_loss": 0.01,
                "brier_score": 0.001,
                "mean_predicted_prob": 0.002,
                "best_iteration": 123,
                "n_valid": 10_000,
                "n_positive_valid": 20,
            }
        )
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "year_range": [2016, 2024],
                "numeric_features": ["a"] * (n_features // 4),
                "categorical_features": ["b"] * (n_features - n_features // 4),
            }
        )
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git": {"sha": "abcdef1234567890", "dirty": False},
            }
        )
    )


def test_compare_runs_produces_expected_columns(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_fake_run(run_a, "usbc10_m0", ap=0.40, n_features=48)
    _write_fake_run(run_b, "usbc10_m1", ap=0.42, n_features=30)

    df = compare_variants.compare_runs([run_a, run_b])

    assert len(df) == 2
    assert list(df["model_id"]) == ["usbc10_m0", "usbc10_m1"]
    assert list(df["n_features"]) == [48, 30]
    assert list(df["average_precision"]) == [0.40, 0.42]
    assert list(df["git_sha"]) == ["abcdef12", "abcdef12"]


def test_compare_runs_tolerates_missing_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"average_precision": 0.5}))
    (run / "config.json").write_text(json.dumps({"model_id": "bare"}))
    # No manifest.json

    df = compare_variants.compare_runs([run])
    assert df.iloc[0]["model_id"] == "bare"
    assert df.iloc[0]["average_precision"] == 0.5


def test_latest_run_for_model_picks_newest(tmp_path: Path) -> None:
    import os

    root = tmp_path / "models"
    dir_a = root / "usbc10_m0-dev" / "20260101-120000"
    dir_b = root / "usbc10_m0-dev" / "20260101-130000"
    dir_c = root / "usbc10_m0-reporting" / "20260101-110000"
    # Set mtimes explicitly so the test is deterministic across filesystems
    # whose timestamp resolution would otherwise collapse time.sleep(0.01)
    # intervals into identical mtimes.
    for d, mtime in ((dir_a, 1_000.0), (dir_b, 2_000.0), (dir_c, 3_000.0)):
        d.mkdir(parents=True)
        os.utime(d, (mtime, mtime))

    picked = compare_variants._latest_run_for_model(root, "usbc10_m0")
    # dir_c has the newest mtime, so it's picked regardless of the
    # timestamp embedded in its path.
    assert picked == dir_c


def test_latest_run_for_model_returns_none_when_absent(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    assert compare_variants._latest_run_for_model(root, "missing") is None


def test_main_reads_run_dirs_and_writes_csv(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_fake_run(run, "usbc10_m0", ap=0.33)
    out = tmp_path / "comparison.csv"
    rc = compare_variants.main([str(run), "--output", str(out)])
    assert rc == 0
    assert out.is_file()


def test_main_errors_on_empty_input(tmp_path: Path) -> None:
    # No run dirs and no --by-model-id → usage error path.
    rc = compare_variants.main(["--output", str(tmp_path / "out.csv")])
    assert rc == 1


def test_main_with_by_model_id_missing_raises(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    rc = compare_variants.main(
        [
            "--by-model-id",
            "nonexistent",
            "--models-root",
            str(models_root),
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )
    assert rc == 1


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test from a temp cwd so default output paths don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
