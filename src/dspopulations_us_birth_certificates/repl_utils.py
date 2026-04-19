import datetime
import platform
import sys
from typing import Any

import arviz as az
import joblib
import lightgbm
import numpy as np
import pandas as pd
import psutil
import pymc as pm
import pytensor
import scipy as sp
import sklearn as skl
import torch

RANDOM_SEED = 202512

def get_environment_info() -> dict[str, str | bool | Any]:
    mem = psutil.virtual_memory()
    return {
        "date": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "platform_version": platform.version(),
        "cpu": platform.processor(),
        "cores": joblib.cpu_count(),
        "physical_cores": joblib.cpu_count(only_physical_cores=True),
        "ram": f"{mem.total // (1024 ** 3)} GB",
        "ram_available": f"{mem.available // (1024 ** 3)} GB",
        "cuda": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.is_available() and torch.cuda.get_device_name(0),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": sp.__version__,
        "sklearn": skl.__version__,
        "lightgbm": lightgbm.__version__,
        "pytorch": torch.__version__,
        "pymc": pm.__version__,
        "pytensor": pytensor.__version__,
        "arviz": az.__version__,
    }

def print_environment_info():
    """Render ``get_environment_info()`` as a rich two-column table."""
    from dspopulations_us_birth_certificates import cli_output

    cli_output.print_environment_table(get_environment_info())
