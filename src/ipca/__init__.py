"""Public API for the ipca package.

The package exposes utilities for preparing instruments and estimating
Instrumented Principal Component Analysis models.
"""

from .instruments import Instruments
from .ipca import ipca

IPCA = ipca

__all__ = ["Instruments", "IPCA", "ipca"]
__version__ = "0.1.0"