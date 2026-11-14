"""Figures of the compiled Verilog-A model (ngspice) overlaid on the Python reference.

Writes docs/spice_pv_loops.png, docs/spice_pund.png, docs/spice_switching_map.png
and docs/spice_small_signal.png.  Needs ngspice and the compiled model (see
spice/README.md); runs the decks through spice/compare_ngspice.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))
import compare_ngspice as cn  # noqa: E402
from fe_models import nls_multidomain as nm  # noqa: E402

DOCS = ROOT / "docs"
P = cn.P
UC = 1e-2

names = ["pund", "loop_1k", "loop_10k", "loop_100k", "minor", "switching", "small_signal", "slow_accumulation"]
res = {r["name"]: r for r in cn.run(names, verbose=False)}
st = {k: v["store"] for k, v in res.items()}

# ---------------------------------------------------------------- P-V loops
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for k, (name, lab) in enumerate((("loop_1k", "1 kHz"), ("loop_10k", "10 kHz"), ("loop_100k", "100 kHz"))):
    s = st[name]
    t_ng, v_ng, q_ng = s["t_ng"], s["v_ng"], s["q_ng"]
    m = t_ng >= 0.25 / {"loop_1k": 1e3, "loop_10k": 1e4, "loop_100k": 1e5}[name]
    a1.plot(v_ng[m], q_ng[m], color=f"C{k}", lw=1.6, label=f"ngspice, {lab}")
    mp = s["t_py"] >= 0.25 / {"loop_1k": 1e3, "loop_10k": 1e4, "loop_100k": 1e5}[name]
    a1.plot(s["v_py"][mp][::12], s["q_py"][mp][::12], "o", ms=1.8, color="k", alpha=0.45)
a1.plot([], [], "o", ms=3, color="k", alpha=0.6, label="Python reference")
a1.set_xlabel("Voltage (V)")
a1.set_ylabel(r"Terminal charge density $Q/A$ ($\mu$C/cm$^2$)")
a1.set_title("Triangular P-V loops, $\\pm$3 V (10 nm film)")
a1.legend(fontsize=8, frameon=False, loc="upper left")
s = st["minor"]
a2.plot(s["v_ng"], s["q_ng"], color="C2", lw=1.4, label="ngspice")
a2.plot(s["v_py"][::10], s["q_py"][::10], "o", ms=1.8, color="k", alpha=0.45, label="Python reference")
a2.set_xlabel("Voltage (V)")
a2.set_title("Minor loops of growing amplitude (1.2 to 3.0 V)")
a2.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(DOCS / "spice_pv_loops.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- PUND
s = st["pund"]
inst = cn.deck_pund().instances[0]
te = np.linspace(0, inst.t[-1], 40001)
r = cn._sim(inst, t_eval=te)
_, keep = np.unique(r["t"], return_index=True)  # segment boundaries appear twice
r = {k: r[k][keep] for k in ("t", "qd", "i_leak")}
i_py = P["area"] * np.gradient(r["qd"], r["t"]) + r["i_leak"]
fig = plt.figure(figsize=(13, 5.8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 2])
b1 = fig.add_subplot(gs[0, 0])
b2 = fig.add_subplot(gs[1, 0], sharex=b1)
ins = fig.add_subplot(gs[:, 1])
b1.plot(s["t_ng"] * 1e6, s["v_ng"], color="k", lw=1.2)
for lab, tc in (("preset", 6.1), ("P", 26.3), ("U", 46.5), ("N", 66.7), ("D", 86.9)):
    b1.text(tc, 3.4 if lab in "PU" else -4.3, lab, ha="center", fontsize=10)
b1.set_ylim(-5, 4.5)
b1.set_ylabel("V (V)")
b1.set_title("PUND, 3 V / 10 us pulses (preset, P, U, N, D)")
b2.plot(s["t_ng"] * 1e6, s["i_ng"] * 1e6, color="C0", lw=1.6, label="ngspice")
b2.plot(r["t"][::40] * 1e6, i_py[::40] * 1e6, "k.", ms=2.5, alpha=0.7, label="Python reference")
b2.set_xlabel("Time (us)")
b2.set_ylabel(r"Current ($\mu$A), area 100 $\mu$m$^2$")
b2.legend(fontsize=8, frameon=False)
chk = {c["quantity"]: c for c in res["pund"]["checks"]}
pu = chk["P - U switched charge [uC/cm2]"]["ngspice"]
b2.text(0.99, 0.05, f"P - U (ngspice) = {pu:.2f} uC/cm$^2$, 2 Ps = {2*P['ps']/UC:.1f}", transform=b2.transAxes,
        ha="right", fontsize=9, bbox=dict(fc="white", ec="0.8"))
# inset: P and U pulses overlaid from their rising edges (switching current = P - U)
for lab, t0p, col in (("P", 21.2e-6, "C3"), ("U", 41.4e-6, "C2")):
    m = (s["t_ng"] >= t0p - 0.2e-6) & (s["t_ng"] <= t0p + 2.5e-6)
    ins.plot((s["t_ng"][m] - t0p) * 1e6, s["i_ng"][m] * 1e6, color=col, lw=1.3, label=f"{lab}, ngspice")
    mp = (r["t"] >= t0p - 0.2e-6) & (r["t"] <= t0p + 2.5e-6)
    ins.plot((r["t"][mp][::3] - t0p) * 1e6, i_py[mp][::3] * 1e6, "k.", ms=2)
ins.set_xlim(-0.2, 2.5)
ins.set_ylim(-5, 100)
ins.set_xlabel("Time from the rising edge (us)")
ins.set_ylabel(r"Current ($\mu$A)")
ins.set_title("P and U pulses overlaid (dots: Python)")
ins.legend(fontsize=8, frameon=False)
fig.savefig(DOCS / "spice_pund.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- switching map
s = st["switching"]
amps, widths = s["amps"], s["widths"]
fig, (c1, c2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
wf = np.logspace(-8, -4, 120)
for k, a in enumerate(amps):
    col = f"C{k}"
    c1.semilogx(wf, [nm.switched_fraction(P, a, w + cn.SW_EDGE) for w in wf], color=col, lw=1, alpha=0.7)
    c1.semilogx(widths, s["f_py"][k], "o", mfc="white", mec=col, ms=7)
    c1.semilogx(widths, s["f_ng"][k], "x", color=col, ms=6, label=f"{a} V")
c1.plot([], [], "k-", lw=1, label="closed form (single-pulse NLS)")
c1.plot([], [], "o", mfc="white", mec="k", label="Python ODE")
c1.plot([], [], "x", color="k", label="ngspice")
c1.set_xlabel("Pulse width (s)")
c1.set_ylabel("Switched fraction")
c1.set_title("Switching from the saturated state")
c1.legend(fontsize=7.5, frameon=False, loc="upper left")
va = np.linspace(0.6, 3.5, 80)
wm = np.logspace(-8, -3, 90)
F = np.array([[nm.switched_fraction(P, a, w) for w in wm] for a in va])
im = c2.pcolormesh(wm, va, F, shading="auto", cmap="viridis", vmin=0, vmax=1)
cs = c2.contour(wm, va, F, levels=[0.1, 0.5, 0.9], colors="w", linewidths=0.8)
c2.clabel(cs, fmt="%.1f", fontsize=8)
for k, a in enumerate(amps):
    c2.scatter(widths, np.full(len(widths), a), c=s["f_ng"][k], cmap="viridis", vmin=0, vmax=1, edgecolors="r", s=40)
c2.set_xscale("log")
c2.set_xlabel("Pulse width (s)")
c2.set_ylabel("Pulse amplitude (V)")
c2.set_title("Switching-kinetics map (closed form); dots: ngspice")
fig.colorbar(im, ax=c2, label="switched fraction")
fig.savefig(DOCS / "spice_switching_map.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- small-signal stability
fig, (d1, d2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
s = st["small_signal"]
period = s["period"]


def x_at_cycle_ends(ser, n_cycles):
    """Mean state read from v(qm) at the 0 V instants that end each cycle (q_rev = 0 there)."""
    n = np.arange(0, n_cycles + 1)
    return n, np.interp(cn.SS_START + n * period, ser["t"], ser["x"])


for k, ser in s["series"].items():
    n, x = x_at_cycle_ends(ser, s["n_cycles"])
    d1.plot(n, (x - x[0]) * 1e12, lw=1.2, label=ser["label"].split(",")[0])
d1.set_ylabel(r"$\bar x(n)-\bar x(0)$  ($\times 10^{-12}$)")
d1.set_xlabel("Cycle number")
d1.set_ylim(-1, 1)
d1.set_title(r"$\pm$0.3 V, 1 MHz, 1000 cycles (ngspice): no drift", fontsize=10)
d1.legend(fontsize=8, frameon=False)
s = st["slow_accumulation"]
ser = s["series"][1]
n, x = x_at_cycle_ends(ser, s["n_cycles"])
d2.plot(n, (x - x[0]) * 1e6, color="C3", lw=1.4, label="ngspice")
tt = np.linspace(0, period, 801)
vv = np.interp(tt, [0, period / 4, 3 * period / 4, period], [0, -ser["amp"], ser["amp"], 0])
nn = np.unique(np.round(np.logspace(0, np.log10(s["n_cycles"]), 25)))
xs = [nm.repeat_waveform(P, tt, vv, n_, ser["s_start"]) - ser["x_py_start"] for n_ in nn]
d2.plot(nn, np.array(xs) * 1e6, "ko", ms=4, mfc="white", label="Python (closed form per cycle)")
d2.set_ylabel(r"$\bar x(n)-\bar x(0)$  ($\times 10^{-6}$)")
d2.set_xlabel("Cycle number")
d2.set_title(r"$\pm$0.8 V from a partially switched state: genuine slow switching", fontsize=10)
d2.legend(fontsize=8, frameon=False)
fig.savefig(DOCS / "spice_small_signal.png", dpi=150)
plt.close(fig)
print("saved spice_*.png")
