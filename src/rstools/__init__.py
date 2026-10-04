"""
rstools: High-performance astronomical detector processing tools.
Developed by Bernard J. Rauscher, Rauscher Scientific LLC.
"""

from rstools.stats import mad, rmean
from rstools.rowcor import rowcor
from rstools.polynomials import Legendre

__version__ = "1.0.0"

__all__ = [
    "mad",
    "rmean",
    "rowcor",
    "Legendre",
]
