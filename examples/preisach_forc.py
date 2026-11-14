"""Preisach major / minor loops and a FORC diagram (docs/preisach_forc.png)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models.preisach import PreisachModel, forc_distribution, simulate_forcs
from fe_models.units import to_MV_cm, to_uC_cm2

DOCS = Path(__file__).resolve().parents[1] / "docs"

model = PreisachModel(Ps=0.20, Ec=1.0e8, sigma_c=0.25e8, sigma_b=0.15e8, bias=0.1e8, n_grid=120)
E, up, dn = model.major_loop()
x = lambda e: to_MV_cm(e)
y = lambda p: to_uC_cm2(p)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

# (a) major loop and first-order reversal curves
ax = axes[0]
Er_show = np.linspace(-1.5e8, -0.5e8, 7)
Er_c, Ef, P = simulate_forcs(model, E_reversal=Er_show, E_field=E)
colors = plt.cm.viridis(np.linspace(0, 0.9, Er_show.size))
for i, c in enumerate(colors):
    ok = ~np.isnan(P[i])
    ax.plot(x(Ef[ok]), y(P[i, ok]), color=c, lw=1.4)
    ax.plot(x(Er_show[i]), y(P[i, ok][0]), "o", color=c, ms=4)
ax.plot(x(E), y(up), color="k", lw=1.8, label="major loop")
ax.plot(x(E), y(dn), color="k", lw=1.8)
ax.set_title("(a) Major loop and FORCs (colour = reversal field)")
ax.set_xlabel(r"Electric field $E$ (MV/cm)")
ax.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax.set_xlim(-3, 3)
ax.legend(frameon=False, loc="upper left")

# (b) nested minor loops: return-point memory
ax = axes[1]
ax.plot(x(E), y(up), color="0.7", lw=1.4)
ax.plot(x(E), y(dn), color="0.7", lw=1.4)
m = model.copy().reset(1)
# outer loop between -1.1 and +1.3 MV/cm; an inner loop (+1.3 -> -0.7 -> +1.3) is inserted
legs = [(3e8, -1.1e8), (-1.1e8, 1.3e8), (1.3e8, -0.7e8), (-0.7e8, 1.3e8), (1.3e8, -1.1e8)]
styles = ["0.35", "C0", "C1", "C1", "C0"]
ends = []
for (a, b), c in zip(legs, styles):
    path = np.linspace(a, b, 400)
    p = m.apply(path)
    ax.plot(x(path), y(p), color=c, lw=1.8)
    ends.append((b, p[-1]))
ax.plot(x([ends[0][0], ends[1][0]]), y([ends[0][1], ends[1][1]]), "o", color="C0", ms=11, mfc="none", mew=1.8,
        label="outer loop turning points")
ax.plot(x([ends[2][0]]), y([ends[2][1]]), "s", color="C1", ms=6, label="inner loop turning point")
ax.plot(x([ends[4][0]]), y([ends[4][1]]), "x", color="k", ms=9, mew=1.5,
        label="path returns here after the inner loop")
ax.set_title("(b) Nested minor loops (return-point memory)")
ax.set_xlabel(r"Electric field $E$ (MV/cm)")
ax.set_ylabel(r"Polarization $P$ (µC/cm$^2$)")
ax.set_xlim(-3, 3)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)

# (c) FORC distribution in (Ec, Eb) coordinates
ax = axes[2]
Er, Eg, Pg = simulate_forcs(model)
rho = forc_distribution(Er, Eg, Pg)
EE, RR = np.meshgrid(Eg, Er)
Ec_h = 0.5 * (EE - RR)
Eb_h = 0.5 * (EE + RR)
rho_plot = rho / np.nanmax(rho)
pc = ax.pcolormesh(x(Ec_h), x(Eb_h), np.ma.masked_invalid(rho_plot), shading="gouraud", cmap="magma_r",
                   vmin=0, vmax=1)
ax.set_xlim(0, 2.0)
ax.set_ylim(-1.0, 1.0)
ax.axhline(x(0.1e8), color="C0", lw=0.8, ls=":")
ax.set_xlabel(r"Coercive coordinate $(E - E_r)/2$ (MV/cm)")
ax.set_ylabel(r"Bias coordinate $(E + E_r)/2$ (MV/cm)")
ax.set_title(r"(c) FORC distribution $-\frac{1}{2}\,\partial^2 P/\partial E_r\partial E$")
cb = fig.colorbar(pc, ax=ax, shrink=0.9)
cb.set_label("normalized density")
ax.text(1.95, x(0.1e8) + 0.04, "input bias 0.1 MV/cm", ha="right", va="bottom", fontsize=8, color="C0")

fig.savefig(DOCS / "preisach_forc.png", dpi=150)
print("saved", DOCS / "preisach_forc.png")
