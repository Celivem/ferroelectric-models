"""Units and physical constants.

All functions in :mod:`fe_models` work in SI units internally:

==============  ===========  =========================================
quantity        SI unit      common thin-film unit
==============  ===========  =========================================
polarization    C/m^2        uC/cm^2   (1 uC/cm^2 = 1e-2 C/m^2)
electric field  V/m          MV/cm     (1 MV/cm   = 1e8 V/m)
thickness       m            nm        (1 nm      = 1e-9 m)
area            m^2          um^2      (1 um^2    = 1e-12 m^2)
energy density  J/m^3        MJ/m^3    (1 MJ/m^3  = 1e6 J/m^3)
==============  ===========  =========================================

The helpers below convert between the two columns so that inputs and plots
can use the units people actually quote for HfO2-based films.
"""

from __future__ import annotations

import numpy as np

#: Vacuum permittivity in F/m (CODATA 2018 value).
EPS0 = 8.8541878128e-12

#: 1 uC/cm^2 expressed in C/m^2.
UC_PER_CM2 = 1e-2
#: 1 MV/cm expressed in V/m.
MV_PER_CM = 1e8
#: 1 nm in m.
NM = 1e-9
#: 1 um in m.
UM = 1e-6


def to_uC_cm2(P):
    """Convert polarization from C/m^2 to uC/cm^2."""
    return np.asarray(P) / UC_PER_CM2


def from_uC_cm2(P):
    """Convert polarization from uC/cm^2 to C/m^2."""
    return np.asarray(P) * UC_PER_CM2


def to_MV_cm(E):
    """Convert electric field from V/m to MV/cm."""
    return np.asarray(E) / MV_PER_CM


def from_MV_cm(E):
    """Convert electric field from MV/cm to V/m."""
    return np.asarray(E) * MV_PER_CM


#: Illustrative HfO2-like defaults. They are round numbers in the range
#: typically quoted for doped HfO2 / HZO thin films and are NOT fitted to any
#: particular device.
HFO2_ILLUSTRATIVE = {
    "Pr": 20 * UC_PER_CM2,      # remanent polarization, C/m^2
    "Ec": 1.0 * MV_PER_CM,      # coercive field, V/m
    "thickness": 10 * NM,       # film thickness, m
    "eps_r": 30.0,              # background relative permittivity
    "area": (100 * UM) ** 2,    # capacitor area, m^2
}
