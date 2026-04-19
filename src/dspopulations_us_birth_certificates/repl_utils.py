"""Compatibility shim delegating to ``dse_research_utils``.

Environment and package-version reporting now come from the shared
``dse_research_utils`` library (see sibling DSE research repos). This
module keeps two things for existing notebooks:

- ``RANDOM_SEED`` — the project-wide default seed used across notebooks.
- ``print_environment_info()`` — calls ``setup.init_workbook()`` (style +
  environment summary) and ``report_package_versions()`` with the
  project's ``PACKAGE_LIST``.

New code should import ``dse_research_utils.environment.setup`` and
``dse_research_utils.metadata.packages`` directly.
"""

from __future__ import annotations

import dse_research_utils.environment.setup as _setup
import dse_research_utils.metadata.packages as _package_metadata

from dspopulations_us_birth_certificates import PACKAGE_LIST

RANDOM_SEED = 202512


def print_environment_info() -> None:
    """Initialise the workbook environment and report package versions."""
    _setup.init_workbook()
    _package_metadata.report_package_versions(list(PACKAGE_LIST))
