"""Landau free-energy landscape and the static S-curve (docs/landau_energy.png)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models import landau
from fe_models.units import to_MV_cm, to_uC_cm2

DOCS = Path(__file__).resolve().parents[1] / "docs"

Pr, Ec = 0.20, 1.0e8                       # illustrative: 20 uC/cm^2, 1 MV/cm
alpha, beta = landau.coefficients_from_pr_ec(Pr, Ec)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

# (a) F(P) for several applied fields
P = np.linspace(-0.3, 0.3, 801)
colors = plt.cm.viridis(np.linspace(0.0, 0.85, 4))
labels = [r"$E = 0$", r"$E = 0.5\,E_c$", r"$E = E_c$", r"$E = 1.5\,E_c$"]
for frac, c, lab in zip((0.0, 0.5, 1.0, 1.5), colors, labels):
    F = landau.free_energy(P, alpha, beta, E=frac * Ec)
    ax1.plot(to_uC_cm2(P), F / 1e6, color=c, lw=1.8, label=lab)
ax1.axhline(0, color="0.7", lw=0.6)
ax1.set_xlabel(r"Polarization $P$ (µC/cm$^2$)")
ax1.set_ylabel(r"Free-energy density $F$ (MJ/m$^3$)")
ax1.set_title(r"(a) $F = \alpha P^2 + \beta P^4 - EP$")
ax1.set_ylim(-40, 25)
ax1.legend(frameon=False, loc="lower left")

# (b) S-curve with the negative-capacitance branch and the static loop jumps
Ps, Es, stable = landau.s_curve(alpha, beta, P_max=0.3, n=2001)
x, y = to_MV_cm(Es), to_uC_cm2(Ps)
ax2.plot(np.where(stable & (Ps < 0), x, np.nan), y, color="C0", lw=2, label="stable")
ax2.plot(np.where(stable & (Ps > 0), x, np.nan), y, color="C0", lw=2)
ax2.plot(np.where(~stable, x, np.nan), y, color="C3", lw=2, ls="--", label=r"unstable ($dE/dP<0$, NC)")
pc = Pr / np.sqrt(3)
Pjump = landau.static_branches(np.array([Ec * (1 + 1e-9)]), alpha, beta)[1][0]
for s in (1, -1):
    ax2.annotate("", xy=(s * 1.0, s * to_uC_cm2(Pjump)), xytext=(s * 1.0, -s * to_uC_cm2(pc)),
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2, ls=":"))
ax2.plot([0, 0], [to_uC_cm2(Pr), -to_uC_cm2(Pr)], "o", color="k", ms=5)
ax2.plot([1.0, -1.0], [-to_uC_cm2(pc), to_uC_cm2(pc)], "s", color="k", ms=5)
ax2.text(0.08, to_uC_cm2(Pr) + 1.0, r"$+P_r$", fontsize=10)
ax2.text(0.08, -to_uC_cm2(Pr) - 3.0, r"$-P_r$", fontsize=10)
ax2.text(1.06, -to_uC_cm2(pc) - 1.0, r"$+E_c$", fontsize=10)
ax2.text(-1.35, to_uC_cm2(pc) + 0.2, r"$-E_c$", fontsize=10)
ax2.axhline(0, color="0.7", lw=0.6)
ax2.axvline(0, color="0.7", lw=0.6)
ax2.set_xlim(-2.5, 2.5)
ax2.set_xlabel(r"Electric field $E$ (MV/cm)")
ax2.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax2.set_title(r"(b) Static S-curve $E = 2\alpha P + 4\beta P^3$")
ax2.legend(frameon=False, loc="upper left")

fig.savefig(DOCS / "landau_energy.png", dpi=150)
print("saved", DOCS / "landau_energy.png")
