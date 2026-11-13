"""
rstools: High-performance astronomical detector processing tools.
Developed by Bernard J. Rauscher, Rauscher Scientific LLC.
"""

from ._backend import HAS_GPU, get_backend, to_cpu, to_gpu
from .polynomials import Legendre
from .robust_savgol import robust_savgol
from .rowcor import rowcor
from .stats import mad, rmean
from .stretch_time_axis import stretch_time_axis
from .xslice import xslice

__version__ = "1.0.0"

__all__ = [
    # Core Algorithms
    "mad",
    "rmean",
    "rowcor",
    "Legendre",
    "robust_savgol",
    "stretch_time_axis",
    "xslice",
    # Device / Backend Utilities
    "HAS_GPU",
    "get_backend",
    "to_cpu",
    "to_gpu",
]

