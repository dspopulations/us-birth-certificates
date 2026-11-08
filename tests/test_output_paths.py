"""Output relocation keeps run layouts, explicit paths and discovery aligned."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dspopulations_us_birth_certificates.paths import resolve_output_root
from dspopulations_us_birth_certificates.selection import latest_fit_dir

from .conftest import SYNTHETIC_CATEGORICAL, SYNTHETIC_NUMERIC

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_output_root(monkeypatch):
    monkeypatch.delenv("DSPOP_OUTPUT_ROOT", raising=False)


@pytest.fixture(scope="module")
def scripts():
    modules = {}
    for name in (
        "fit_model",
        "fit_selection_model",
        "fit_core_reduction_model",
        "tune_model",
        "run_all_selection_variants",
        "compare_selection_variants",
        "compare_variants",
    ):
        spec = importlib.util.spec_from_file_location(
            f"output_paths_{name}", REPO_ROOT / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_root_precedence_and_call_time_environment(monkeypatch, tmp_path):
    assert resolve_output_root() == Path("output")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", "")
    assert resolve_output_root() == Path("output")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(tmp_path / "scratch"))
    assert resolve_output_root() == tmp_path / "scratch"
    assert resolve_output_root("explicit") == Path("explicit")
    assert resolve_output_root(Path(".")) == Path(".")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", "~/scratch")
    assert resolve_output_root() == Path.home() / "scratch"
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize(
    "script,args,parent",
    [
        ("fit_model", [], "fit_model"),
        ("fit_model", ["--profile", "dev"], "fit_model_dev"),
        (
            "fit_selection_model",
            ["--variant", "B", "--spec", "full"],
            "selection/B/full",
        ),
        ("fit_core_reduction_model", ["DSP004"], "selection_core_reduction/DSP004"),
    ],
)
def test_fit_cli_layout_and_precedence(
    scripts, monkeypatch, tmp_path, script, args, parent
):
    parse = scripts[script].parse_args
    default = parse(args).output_dir
    assert default.parent == Path("output") / parent
    assert re.fullmatch(r"\d{8}-\d{6}", default.name)

    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(tmp_path / "environment"))
    assert parse(args).output_dir.parent == tmp_path / "environment" / parent
    explicit = args + ["--output-root", str(tmp_path / "cli")]
    assert parse(explicit).output_dir.parent == tmp_path / "cli" / parent
    exact = parse(explicit + ["--output-dir", "exact-run"]).output_dir
    assert exact == Path("exact-run")


def test_existing_tuning_and_models_roots_keep_their_meaning(
    scripts, monkeypatch, tmp_path
):
    tune = scripts["tune_model"].parse_args
    compare = scripts["compare_variants"].parse_args
    assert tune(["usbc10_m1"]).output_root == Path("output/tuning")
    assert compare([]).models_root == Path("output/models")
    assert compare([]).output == Path("comparison.csv")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(tmp_path))
    assert tune(["usbc10_m1"]).output_root == tmp_path / "tuning"
    assert compare([]).models_root == tmp_path / "models"
    assert tune(["usbc10_m1", "--output-root", "studies"]).output_root == Path(
        "studies"
    )
    assert compare(["--output-root", "scratch"]).models_root == Path("scratch/models")
    assert compare(
        ["--output-root", "scratch", "--models-root", "fits"]
    ).models_root == Path("fits")


def _completed_fit(root, variant="C", spec="full"):
    run = root / variant / spec / "20260101-120000"
    run.mkdir(parents=True)
    (run / "idata.nc").touch()
    return run


def test_selection_discovery_honours_environment_and_explicit_root(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    local = _completed_fit(Path("output/selection"))
    assert latest_fit_dir("C") == local
    scratch = tmp_path / "scratch [1]"
    remote = _completed_fit(scratch / "selection")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(scratch))
    assert latest_fit_dir("C") == remote
    assert latest_fit_dir("C", root="output/selection") == local
    # Environment discovery retains validation filtering.
    (remote / "validation.json").write_text('{"status": "failed"}')
    with pytest.raises(FileNotFoundError, match=re.escape(str(scratch / "selection"))):
        latest_fit_dir("C")


def test_selection_comparison_uses_resolved_root(scripts, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    local = _completed_fit(Path("output/selection"))
    parse = scripts["compare_selection_variants"]._parse_args
    default = parse([])
    assert default.fit_dirs == {"C": local}
    assert default.output_dir.parent == Path("output/selection")
    scratch = tmp_path / "scratch"
    remote = _completed_fit(scratch / "selection")
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(scratch))
    resolved = parse([])
    assert resolved.fit_dirs == {"C": remote}
    assert resolved.output_dir.parent == scratch / "selection"
    explicit_root = tmp_path / "cli"
    explicit_run = _completed_fit(explicit_root / "selection")
    assert parse(["--output-root", str(explicit_root)]).fit_dirs == {"C": explicit_run}
    exact = parse(["--root", "output/selection", "--output-dir", "comparison"])
    assert exact.fit_dirs == {"C": local}
    assert exact.output_dir == Path("comparison")


def test_classifier_discovery_uses_environment(scripts, monkeypatch, tmp_path):
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(tmp_path / "scratch"))
    run = tmp_path / "scratch/models/usbc10_m1-dev/20260101-120000"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"average_precision": 0.5}')
    output = tmp_path / "comparison.csv"
    assert (
        scripts["compare_variants"].main(
            [
                "--by-model-id",
                "usbc10_m1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert str(run) in output.read_text()


def test_batch_runner_relocates_logs_children_and_resume(
    scripts, monkeypatch, tmp_path
):
    runner = scripts["run_all_selection_variants"]
    monkeypatch.chdir(tmp_path)
    assert runner._parse_args([]).output_root == Path("output")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", str(tmp_path / "environment"))
    assert runner._parse_args([]).output_root == tmp_path / "environment"
    _completed_fit(scratch / "selection", variant="A")
    # A local C fit must not cause a skip under the chosen root.
    _completed_fit(tmp_path / "output/selection", variant="C")
    calls = []

    def run_child(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output-dir") + 1])
        assert out.parent == scratch / "selection/C/full"
        (out / "idata.nc").touch()
        kwargs["stdout"].write("child output\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.setup, "init_script", lambda: None)
    monkeypatch.setattr(runner.subprocess, "run", run_child)
    args = ["--variants", "A", "C", "--skip-existing", "--output-root", str(scratch)]
    assert runner.main(args) == 0
    assert len(calls) == 1
    logs = list((scratch / "selection/_run_logs").glob("*_C.log"))
    assert len(logs) == 1
    assert "child output" in logs[0].read_text()
    assert runner.main(args) == 0
    assert len(calls) == 1
    assert not (tmp_path / "environment").exists()


def test_fit_command_writes_artefacts_and_absolute_path(
    scripts,
    monkeypatch,
    tmp_path,
    synthetic_predictors_frame,
):
    from dspopulations_us_birth_certificates import reporting

    fit = scripts["fit_model"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", "scratch")
    monkeypatch.setattr(fit.setup, "init_script", lambda: None)
    monkeypatch.setattr(fit.package_metadata, "report_package_versions", lambda _: None)
    monkeypatch.setattr(fit, "DEFAULT_NUMERIC", SYNTHETIC_NUMERIC)
    monkeypatch.setattr(fit, "DEFAULT_CATEGORICAL", SYNTHETIC_CATEGORICAL)
    monkeypatch.setattr(
        fit.data_utils,
        "load_predictors_data",
        lambda **_: synthetic_predictors_frame.copy(),
    )
    monkeypatch.setattr(
        fit, "load_prior_best_params", lambda _: {"min_data_in_leaf": 20}
    )
    monkeypatch.setattr(
        reporting, "DEFAULT_TEMPLATE", REPO_ROOT / "docs/models/usbc10/index.qmd"
    )
    assert (
        fit.main(
            [
                "--profile",
                "dev",
                "--no-optimize",
                "--num-boost-round",
                "5",
                "--early-stopping-rounds",
                "2",
                "--num-threads",
                "1",
                "--no-plots",
                "--no-permutation",
                "--no-shap",
            ]
        )
        == 0
    )
    runs = list((tmp_path / "scratch/fit_model_dev").iterdir())
    assert len(runs) == 1
    run = runs[0]
    for name in ("model.txt", "metrics.json", "manifest.json", "index.qmd"):
        assert (run / name).is_file()
    saved = json.loads((run / "run_config.json").read_text())
    assert saved["output_dir"] == str(run.resolve())
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_tuning_api_records_resolved_output(
    monkeypatch,
    tmp_path,
    synthetic_predictors_frame,
    explicit,
):
    from dataclasses import replace

    from dspopulations_us_birth_certificates import tuning
    from dspopulations_us_birth_certificates.models import MODELS, RunConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSPOP_OUTPUT_ROOT", "scratch")
    monkeypatch.setattr(tuning.joblib, "cpu_count", lambda **_: 1)
    monkeypatch.setattr(
        tuning,
        "_load_xy",
        lambda *_, **__: (
            synthetic_predictors_frame[["num_age"]],
            synthetic_predictors_frame["ca_down_c_p_n"],
            [],
        ),
    )
    config = replace(
        RunConfig.from_name("dev"), num_boost_round=2, early_stopping_rounds=1
    )
    root = Path("explicit-tuning") if explicit else None
    tuning.run_optuna_study(MODELS["usbc10_m1"], config, n_trials=1, output_root=root)
    expected_root = tmp_path / ("explicit-tuning" if explicit else "scratch/tuning")
    run = expected_root / "usbc10_m1"
    for name in ("best_params.json", "trials.csv", "study.pkl"):
        assert (run / name).is_file()
    saved = json.loads((run / "run_config.json").read_text())
    assert saved["output_root"] == str(expected_root.resolve())
    assert saved["output_dir"] == str(run.resolve())
    assert saved["n_trials"] == 1
    assert saved["model_id"] == "usbc10_m1"
    assert not (tmp_path / "output").exists()
