"""Triangular-wave P-E loop with leakage, and two compensation routes
(docs/leakage_compensation.png). All device values are illustrative.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models import measurement as ms
from fe_models.preisach import PreisachModel
from fe_models.units import to_MV_cm, to_uC_cm2

DOCS = Path(__file__).resolve().parents[1] / "docs"

area, thickness, eps_r, Vmax = 1e-8, 10e-9, 30.0, 3.0


def model_in_remanent_state():
    m = PreisachModel(n_grid=300)
    m.apply([-m.E_max, 0.0])
    return m


def measure(f, leakage=None, n=1201):
    t = np.linspace(0, 1 / f, n)
    V = ms.triangle_wave(t, Vmax, f)
    I = ms.capacitor_current(t, V, area=area, thickness=thickness, eps_r=eps_r,
                             polarization=model_in_remanent_state().apply, leakage=leakage)
    return t, V, I


R = 0.5e6                                   # ohmic leakage, Ohm
f1, f2 = 1e3, 3e3
t, V, I_true = measure(f1)
_, _, I_ohm = measure(f1, R)
E = V / thickness

P_true = ms.polarization_from_current(t, I_true, area)
P_raw = ms.polarization_from_current(t, I_ohm, area)
R_est = ms.estimate_leakage_resistance(t, V, I_ohm)
P_ohm = ms.polarization_from_current(t, ms.compensate_leakage_ohmic(V, I_ohm, R_est), area)


def leak_nl(v):                             # nonlinear (non-ohmic) leakage law
    return 2e-7 * np.sinh(v / 0.8)


_, _, I1 = measure(f1, leak_nl)
t2, _, I2 = measure(f2, leak_nl)
I1c, _, I_leak = ms.compensate_leakage_two_frequency(I1, f1, I2, f2)
P_nl_raw = ms.polarization_from_current(t, I1, area)
P_nl_comp = ms.polarization_from_current(t, I1c, area)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
ax = axes[0]
ax.plot(to_MV_cm(E), I_true * 1e6, color="0.4", lw=1.4, label="leak-free")
ax.plot(to_MV_cm(E), I_ohm * 1e6, color="C3", lw=1.4, label=f"with ohmic leakage, R = {R / 1e6:g} MΩ")
ax.set_xlabel(r"Electric field $E$ (MV/cm)")
ax.set_ylabel("Current (µA)")
ax.set_title(f"(a) Terminal current, {f1 / 1e3:g} kHz triangle")
ax.legend(frameon=False, loc="upper left", fontsize=9)

ax = axes[1]
ax.plot(to_MV_cm(E), to_uC_cm2(P_raw), color="C3", lw=1.6, label="raw (integrated current)")
ax.plot(to_MV_cm(E), to_uC_cm2(P_true), color="0.3", lw=3.0, alpha=0.35, label="leak-free")
ax.plot(to_MV_cm(E), to_uC_cm2(P_ohm), color="C0", lw=1.2, ls="--",
        label=f"ohmic correction, R$_{{est}}$ = {R_est / 1e6:.3f} MΩ")
ax.set_xlabel(r"Electric field $E$ (MV/cm)")
ax.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax.set_title("(b) Ohmic leakage: estimate R at the voltage tips")
ax.legend(frameon=False, loc="upper left", fontsize=9)

ax = axes[2]
ax.plot(to_MV_cm(E), to_uC_cm2(P_nl_raw), color="C3", lw=1.6, label=f"raw, {f1 / 1e3:g} kHz")
ax.plot(to_MV_cm(E), to_uC_cm2(P_true), color="0.3", lw=3.0, alpha=0.35, label="leak-free")
ax.plot(to_MV_cm(E), to_uC_cm2(P_nl_comp), color="C2", lw=1.2, ls="--",
        label=f"two-frequency correction ({f1 / 1e3:g} + {f2 / 1e3:g} kHz)")
ax.set_xlabel(r"Electric field $E$ (MV/cm)")
ax.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax.set_title(r"(c) Nonlinear leakage $I \propto \sinh(V/V_0)$")
ax.legend(frameon=False, loc="upper left", fontsize=9)

for ax in axes[1:]:
    ax.set_ylim(-33, 45)
for ax in axes:
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)

fig.savefig(DOCS / "leakage_compensation.png", dpi=150)
print("saved", DOCS / "leakage_compensation.png", R_est)
