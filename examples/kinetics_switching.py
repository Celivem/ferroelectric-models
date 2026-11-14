"""KAI vs NLS switching curves, Merz-law field dependence and fits (docs/kinetics.png).

The "data" are synthetic: model curves plus Gaussian noise with a fixed seed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models import kinetics as k
from fe_models.units import to_MV_cm

DOCS = Path(__file__).resolve().parents[1] / "docs"
rng = np.random.default_rng(2026)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

# (a) model comparison at one field, same characteristic time 1 us
ax = axes[0]
t = np.logspace(-10, -2, 600)
ax.semilogx(t, k.kai_fraction(t, 1e-6, 1.0), color="C0", lw=1.8, label="KAI, $n=1$")
ax.semilogx(t, k.kai_fraction(t, 1e-6, 2.0), color="C0", lw=1.8, ls="--", label="KAI, $n=2$")
ax.semilogx(t, k.nls_fraction(t, -6, 0.5, n=2, kind="lognormal"), color="C2", lw=1.8,
            label="NLS log-normal, $w=0.5$")
ax.semilogx(t, k.nls_fraction(t, -6, 0.5, n=2, kind="lorentzian"), color="C3", lw=1.8,
            label="NLS Lorentzian, $w=0.5$")
ax.axvline(1e-6, color="0.8", lw=0.8)
ax.set_xlabel("Time after field step $t$ (s)")
ax.set_ylabel(r"Switched fraction $\Delta P / 2P_s$")
ax.set_title(r"(a) Constant-field switching, $t_0 = 10^{-6}$ s")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_ylim(-0.02, 1.02)

# (b) NLS with Merz-law centre time at several fields; noisy synthetic data and fits
ax = axes[1]
t_inf, Ea, m_true, width = 1e-9, 8e8, 1.0, 0.4
fields = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0]) * 1e8
t_data = np.logspace(-10, -3, 40)
colors = plt.cm.viridis(np.linspace(0, 0.9, fields.size))
t1_fit = []
for E, c in zip(fields, colors):
    f_true = k.nls_fraction_field(t_data, E, t_inf, Ea, m_true, width=width)
    f_noisy = f_true + rng.normal(0, 0.015, t_data.size)
    r = k.fit_nls(t_data, f_noisy, n=2.0, kind="lorentzian")
    t1_fit.append(10 ** r["log10_t1"])
    ax.semilogx(t_data, f_noisy, "o", color=c, ms=3.5, alpha=0.8)
    ax.semilogx(t, k.nls_fraction(t, r["log10_t1"], r["width"]), color=c, lw=1.5,
                label=f"{to_MV_cm(E):.2f} MV/cm")
ax.set_xlabel("Time after field step $t$ (s)")
ax.set_ylabel(r"Switched fraction $\Delta P / 2P_s$")
ax.set_title("(b) Lorentzian NLS fits to synthetic data")
ax.legend(frameon=False, loc="upper left", fontsize=8, title="field", title_fontsize=8)
ax.set_xlim(1e-10, 1e-2)
ax.set_ylim(-0.06, 1.06)

# (c) Merz-law fit of the fitted centre times
ax = axes[2]
t1_fit = np.array(t1_fit)
rm = k.fit_merz(fields, t1_fit)
E_line = np.linspace(0.9e8, 3.2e8, 200)
ax.semilogy(1 / to_MV_cm(fields), t1_fit, "o", color="C0", ms=6, label="from NLS fits")
ax.semilogy(1 / to_MV_cm(E_line), k.merz_time(E_line, rm["t_inf"], rm["Ea"], rm["m"]), color="C0", lw=1.5,
            label=(rf"Merz fit: $t_\infty$={rm['t_inf']:.2g} s, $E_a$={to_MV_cm(rm['Ea']):.2f} MV/cm, "
                   rf"$m$={rm['m']:.2f}"))
ax.semilogy(1 / to_MV_cm(E_line), k.merz_time(E_line, t_inf, Ea, m_true), color="0.5", lw=1.0, ls="--",
            label=rf"true: $t_\infty$=1e-09 s, $E_a$=8 MV/cm, $m$=1")
ax.set_xlabel(r"Inverse field $1/E$ (cm/MV)")
ax.set_ylabel(r"Centre switching time $t_1$ (s)")
ax.set_title(r"(c) Merz law $t_1 = t_\infty \exp[(E_a/E)^m]$")
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.grid(True, which="both", color="0.93")

fig.savefig(DOCS / "kinetics.png", dpi=150)
print("saved", DOCS / "kinetics.png")
print("Merz fit:", rm["t_inf"], rm["Ea"], rm["m"])
