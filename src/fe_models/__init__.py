"""Standard ferroelectric switching and hysteresis models for HfO2-type thin films.

Submodules
----------
landau       Landau-Devonshire free energy, static S-curve, Landau-Khalatnikov dynamics
preisach     Scalar Preisach model, major/minor loops, FORC distribution
kinetics     KAI, NLS (log-normal / Lorentzian), Merz law, and fitting helpers
measurement  PUND and triangular-wave measurements, leakage compensation, loop parameters
units        SI constants and uC/cm^2, MV/cm conversions

Everything works in SI units. Default parameters are illustrative
HfO2-like values, not fitted to any device.
"""

from . import kinetics, landau, measurement, preisach, units
from .preisach import PreisachModel

__all__ = ["kinetics", "landau", "measurement", "preisach", "units", "PreisachModel"]
__version__ = "0.1.0"
