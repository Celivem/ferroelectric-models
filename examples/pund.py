"""PUND pulse train and the simulated capacitor current (docs/pund.png).

Device: 10 nm film, eps_r = 30, 100 um x 100 um, switchable part from a
Preisach model, plus a 20 kOhm ohmic leakage path. All values illustrative.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fe_models import measurement as ms
from fe_models.preisach import PreisachModel
from fe_models.units import to_uC_cm2

DOCS = Path(__file__).resolve().parents[1] / "docs"

area, thickness, eps_r = 1e-8, 10e-9, 30.0
t, V, win = ms.pund_waveform(amplitude=3.0, rise=1e-6, width=1e-6, delay=1e-6, n_per_segment=300)
model = PreisachModel(n_grid=120)
I = ms.capacitor_current(t, V, area=area, thickness=thickness, eps_r=eps_r,
                         polarization=model.apply, leakage=20e3)
res = ms.analyze_pund(t, I, win, area=area)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, constrained_layout=True,
                               gridspec_kw={"height_ratios": [1, 1.6]})
us = t * 1e6
ax1.plot(us, V, color="k", lw=1.5)
for name, (a, top, b) in win.items():
    ax1.text(1e6 * (a + b) / 2, 3.35 if V[np.searchsorted(t, (a + top) / 2)] > 0 else -3.9,
             name, ha="center", fontsize=10)
ax1.set_ylabel("Voltage (V)")
ax1.set_ylim(-4.6, 4.3)
ax1.set_title("PUND: preset, then Positive, Up, Negative, Down pulses (3 V on 10 nm = 3 MV/cm)")

ax2.plot(us, I * 1e3, color="C0", lw=1.4, label="total current")
# overlay U on P and D on N to show the switching current (difference)
for sw, ns, col in (("P", "U", "C3"), ("N", "D", "C2")):
    a_sw, top_sw, _ = win[sw]
    a_ns, top_ns, _ = win[ns]
    m_sw = (t >= a_sw) & (t <= top_sw)
    m_ns = (t >= a_ns) & (t <= top_ns)
    n = min(m_sw.sum(), m_ns.sum())
    tt = t[m_sw][:n]
    ax2.fill_between(tt * 1e6, I[m_sw][:n] * 1e3, I[m_ns][:n] * 1e3, color=col, alpha=0.35,
                     label=f"switching current ({sw} - {ns})")
ax2.axhline(0, color="0.7", lw=0.6)
ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Current (mA)")
ax2.legend(frameon=False, loc="upper right", fontsize=9)
txt = (f"(P* - U*)/A = {to_uC_cm2(res['dP_pos']):.1f} µC/cm²\n"
       f"(D* - N*)/A = {to_uC_cm2(res['dP_neg']):.1f} µC/cm²\n"
       f"Pr = {to_uC_cm2(res['Pr']):.1f} µC/cm²  (leakage cancels in the difference)")
ax2.text(0.01, 0.04, txt, transform=ax2.transAxes, fontsize=9, va="bottom",
         bbox=dict(boxstyle="round", fc="white", ec="0.8"))

fig.savefig(DOCS / "pund.png", dpi=150)
print("saved", DOCS / "pund.png", res["Pr"])
