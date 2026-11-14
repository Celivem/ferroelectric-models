"""Landau-Khalatnikov loops at several sweep frequencies (docs/lk_loops.png)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models import landau
from fe_models.measurement import extract_loop_parameters
from fe_models.units import to_MV_cm, to_uC_cm2

DOCS = Path(__file__).resolve().parents[1] / "docs"

Pr, Ec = 0.20, 1.0e8
alpha, beta = landau.coefficients_from_pr_ec(Pr, Ec)
tau = 10e-9                                   # illustrative LK relaxation time
rho = landau.rho_from_time_constant(tau, alpha, beta)
E_amp = 2.0 * Ec

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.25, 1]})

E_s, P_s = landau.static_loop(E_amp, alpha, beta, n=1601)
ax1.plot(to_MV_cm(E_s), to_uC_cm2(P_s), color="0.55", lw=1.5, ls="--", label="static (Landau)")
freqs = [1e4, 1e5, 3e5, 1e6]
colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(freqs)))
for f, c in zip(freqs, colors):
    _, E, P = landau.lk_loop(f, E_amp, alpha, beta, rho, n_points=3000)
    ax1.plot(to_MV_cm(E), to_uC_cm2(P), color=c, lw=1.8, label=(f"{f / 1e6:g} MHz" if f >= 1e6 else f"{f / 1e3:g} kHz"))
ax1.set_xlabel(r"Electric field $E$ (MV/cm)")
ax1.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax1.set_title(rf"(a) LK loops, $\tau$ = {tau * 1e9:g} ns, triangular drive")
ax1.axhline(0, color="0.8", lw=0.6)
ax1.axvline(0, color="0.8", lw=0.6)
ax1.legend(frameon=False, loc="center", fontsize=9)

fs = np.logspace(2, 6, 17)
ecs = []
for f in fs:
    _, E, P = landau.lk_loop(f, E_amp, alpha, beta, rho, n_points=3000)
    ecs.append(extract_loop_parameters(E, P)["Ec"])
ax2.semilogx(fs, to_MV_cm(np.array(ecs)), "o-", color="C0", ms=4)
ax2.axhline(to_MV_cm(Ec), color="0.55", ls="--", lw=1.2, label=r"static $E_c$")
ax2.set_xlabel("Sweep frequency (Hz)")
ax2.set_ylabel(r"Dynamic coercive field $E_c$ (MV/cm)")
ax2.set_title(r"(b) $E_c$ grows with sweep rate")
ax2.legend(frameon=False, loc="upper left")
ax2.grid(True, which="both", color="0.92")

fig.savefig(DOCS / "lk_loops.png", dpi=150)
print("saved", DOCS / "lk_loops.png")
