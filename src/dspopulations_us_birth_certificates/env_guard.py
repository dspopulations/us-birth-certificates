"""Tame MKL/OpenMP threading before numpy is imported.

Importing this module (for its side effect) sets ``MKL_NUM_THREADS`` /
``OMP_NUM_THREADS`` / ``MKL_THREADING_LAYER`` via ``os.environ.setdefault``.
On this project's Windows/conda environment, ``pm.sample`` / nutpie and
other numpy/numba paths crash inside MKL's threadpool
(``OSError [WinError 0xc06d007f]``) unless MKL threading is tamed first.
``setdefault`` keeps any caller override and is a harmless thread cap on
other machines.

This must be imported before numpy (directly or transitively via pandas /
xarray / matplotlib) is imported, so it belongs at the very top of a
script's import block, ahead of those libraries:

    import dspopulations_us_birth_certificates.env_guard  # noqa: F401

    import numpy as np  # noqa: E402
    ...

``dspopulations_us_birth_certificates/__init__.py`` does not itself import
numpy, so importing this submodule does not risk pulling numpy in first.
"""

from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
