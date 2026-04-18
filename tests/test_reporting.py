"""Tests for ``reporting.copy_template`` and ``render_quarto_report``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dspopulations_us_birth_certificates import reporting
from dspopulations_us_birth_certificates.models import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)


def _stub_context(output_dir: Path) -> ModelFitContext:
    config = ModelConfig(
        model_id="report_smoke",
        variant_of=None,
        target_var="y",
        numeric_features=(),
        categorical_features=(),
        base_params={},
        params={},
        train_config={},
        year_range=(2020, 2024),
        include_unknown=True,
        selection_history=(),
        shap_scatter_specs=(),
    )
    return ModelFitContext(
        config=config,
        run_config=RunConfig.from_name("dev"),
        output_dir=output_dir,
    )


def test_copy_template_writes_index_qmd(tmp_path: Path) -> None:
    template = tmp_path / "tpl.qmd"
    template.write_text("---\ntitle: Test\n---\n\nHello.")

    ctx = _stub_context(tmp_path / "run")
    ctx.output_dir.mkdir()

    dst = reporting.copy_template(ctx, template)
    assert dst == ctx.output_dir / "index.qmd"
    assert dst.read_text() == "---\ntitle: Test\n---\n\nHello."


def test_copy_template_missing_template_raises(tmp_path: Path) -> None:
    ctx = _stub_context(tmp_path / "run")
    ctx.output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="template not found"):
        reporting.copy_template(ctx, tmp_path / "does_not_exist.qmd")


def test_render_quarto_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Quarto source not found"):
        reporting.render_quarto_report(tmp_path / "absent.qmd")


def test_render_quarto_report_invokes_quarto_cli(tmp_path: Path) -> None:
    qmd = tmp_path / "index.qmd"
    qmd.write_text("---\ntitle: Test\n---\n\nHello.")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        # Simulate quarto producing index.html
        (tmp_path / "index.html").write_text("<html></html>")
        out = reporting.render_quarto_report(qmd, output_format="html")

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "quarto"
    assert "render" in args
    assert str(qmd) in args
    assert out == tmp_path / "index.html"


def test_render_quarto_report_missing_cli_is_friendly(tmp_path: Path) -> None:
    qmd = tmp_path / "index.qmd"
    qmd.write_text("# hi")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError, match="binary not found"):
            reporting.render_quarto_report(qmd)


def test_pipeline_report_copies_but_does_not_render_by_default(
    tmp_path: Path,
) -> None:
    """pipeline.report(render=False) copies the template without shelling out."""
    from dspopulations_us_birth_certificates.models.lgbm_pipeline import (
        LGBMClassifierPipeline,
    )

    # Minimal real template on disk so the copy step has something to read.
    template = tmp_path / "index.qmd"
    template.write_text("---\ntitle: Pipeline report test\n---\nHello.")

    config = ModelConfig(
        model_id="m",
        variant_of=None,
        target_var="y",
        numeric_features=(),
        categorical_features=(),
        base_params={},
        params={},
        train_config={},
        year_range=(2020, 2024),
        include_unknown=True,
        selection_history=(),
        shap_scatter_specs=(),
    )
    run_dir = tmp_path / "run"
    pipeline = LGBMClassifierPipeline(
        config=config, run_config=RunConfig.from_name("dev"), output_dir=run_dir
    )
    with patch("subprocess.run") as mock_run:
        pipeline.report(render=False, template_path=template)

    mock_run.assert_not_called()
    assert (run_dir / "index.qmd").is_file()


def test_pipeline_report_logs_when_template_absent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from dspopulations_us_birth_certificates.models.lgbm_pipeline import (
        LGBMClassifierPipeline,
    )

    config = ModelConfig(
        model_id="m",
        variant_of=None,
        target_var="y",
        numeric_features=(),
        categorical_features=(),
        base_params={},
        params={},
        train_config={},
        year_range=(2020, 2024),
        include_unknown=True,
        selection_history=(),
        shap_scatter_specs=(),
    )
    pipeline = LGBMClassifierPipeline(
        config=config,
        run_config=RunConfig.from_name("dev"),
        output_dir=tmp_path / "run",
    )
    with caplog.at_level("WARNING"):
        pipeline.report(render=False, template_path=tmp_path / "absent.qmd")
    assert "Skipping report" in caplog.text
