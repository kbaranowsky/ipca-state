"""Public API for SC-IPCA.

The package exposes data-preparation utilities and an Instrumented Principal
Component Analysis estimator.
"""

from .instruments import Instruments
from .ipca import ipca as IPCA
from .ipca import ipca

__all__ = ["Instruments", "IPCA", "ipca"]
__version__ = "0.1.0"
