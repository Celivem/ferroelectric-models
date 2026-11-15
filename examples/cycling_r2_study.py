"""Second round of cycling mechanisms (E18-E20 and the loop read-out) against published data.

Writes the figures docs/research/r2_*.png and docs/research/cycling_r2.json.
Mechanisms, candidate list and reasoning: docs/research/cycling-r2.md.

Inputs: digitized values in data/literature/ (each file has the citation, the
figure, the device and "digitized from figure, approx." or "values stated in
the text" in its header).  No unpublished data are used.

1. Coercive field and 2Pr during wake-up vs temperature, Fengler et al. AEM 2017
   Fig. 7a + 7c, read out the way the paper does (10 kHz triangular P-V loop,
   zero crossings of the total charge).  Hypotheses for the temperature effect:
   P_s(T) vs Arrhenius wake-up (vs both).
2. Pulse width / duty cycle of wake-up, Lenox et al. pssa 2023 Fig. 5a, 5b and
   text: dose laws N w^m against the depinning / re-pinning balance (E18).
3. Leakage onset and failure vs pulse width (Lenox 2023 text): field-time dose
   with and without the build-up time of E19; leakage growth (E20).
4. Recovery after rest, Pesic et al. AFM 2016 Fig. 7b (a.u.): what the data can say.

Run: PYTHONPATH=src python examples/cycling_r2_study.py   (a few minutes once the part-1 fits exist; run
examples/cycling_r2_fengler_fits.py A B C D E first, ideally one process per letter, about 30-60 min each)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fe_models import nls_multidomain as nm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "literature"
OUTDIR = ROOT / "docs" / "research"
OUT: dict = {}
T0 = time.time()


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


# =============================================================== 1. Fengler 2017: 2Pr and 2Ec vs T
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycling_r2_fengler_fits as ff  # noqa: E402

TEMPS, d7a, d7c, TFE, AMP, CYC, NS = ff.TEMPS, ff.D7A, ff.D7C, ff.TFE, ff.AMP, ff.CYC, ff.NS
f_card, f_model = ff.card, ff.model
# the fits themselves run in cycling_r2_fengler_fits.py (one process per hypothesis); missing ones are run here
fits = {}
for letter in ff.HYP:
    try:
        res = ff.load(letter)
    except FileNotFoundError:
        res = ff.fit(letter)
    fits[letter] = np.array(res["theta"])
    OUT.setdefault("fengler2017_wakeup_2Pr_2Ec", {})[res["hypothesis"]] = {k: v for k, v in res.items()
                                                                          if k not in ("hypothesis", "theta", "ln wk_nw")}
    log(f"1. {res['hypothesis']}: RMS 2Pr {res['RMS 2Pr [uC/cm2] (35 points)']:.2f}, "
        f"2Ec {res['RMS 2Ec [MV/cm] (18 points)']:.3f}")
OUT["fengler2017_wakeup_2Pr_2Ec"]["read-out"] = (
    "10 kHz triangle +-2.3 MV/cm (paper, Fig. 2 and Fig. 4 text), zero crossings of the total charge P + dielectric + "
    "reversible, loop centred; cycling 2.3 MV/cm 100 kHz rectangular; read at the cycling temperature (assumed)")
# figure: the best fit with the physical sign of E_c(T) (A0: it picks an Arrhenius wake-up) against the best
# unconstrained fit (C: P_s(T) and an intrinsic E_c that rises with T)
thB, thC = fits["A0"], fits["C"]

# round-one read-out on the same card: the coercive width without the dielectric charge
ec_old = {tc: f_model(thB, tc, readout="mid")[1] for tc in (25, 125)}
OUT["fengler2017_wakeup_2Pr_2Ec"]["round-one read-out on fit A0 (mid crossing of the switching state) [MV/cm]"] = {
    f"{tc} C": list(map(float, v)) for tc, v in ec_old.items()}

# uncompensated leakage as the source of the temperature effect: how large would it have to be?
lk = {}
for g in (0.0, 30.0, 100.0, 300.0):
    p_ = f_card(thB, g_oh=g, phi_oh=0.3)
    L = [nm.loop_figures(p_, AMP, 1e4, nd=16, temp=273.15 + tc, history=nm.History(nw=1e9), npts=300, leak=True)
         for tc in (25, 125)]
    lk[f"g_oh = {g:g} A/m2/V, 0.3 eV"] = {"2Pr 25 C": (L[0]["pr_p"] - L[0]["pr_m"]) * 100,
                                         "2Pr 125 C": (L[1]["pr_p"] - L[1]["pr_m"]) * 100,
                                         "2Ec 25 C": (L[0]["vc_p"] - L[0]["vc_m"]) / TFE / 1e8,
                                         "2Ec 125 C": (L[1]["vc_p"] - L[1]["vc_m"]) / TFE / 1e8}
OUT["fengler2017_wakeup_2Pr_2Ec"]["uncompensated ohmic leakage, woken film, fit A0 card"] = lk

fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
nn = np.logspace(0.7, 5.2, 24)
for k, tc in enumerate(TEMPS):
    c = f"C{k}"
    pr, ec = f_model(thB, tc, nn)
    prc, ecc = f_model(thC, tc, nn)
    ax[0, 0].semilogx(d7a[tc][:, 0], d7a[tc][:, 1], "o", color=c, mec="k", ms=5, label=f"{tc} C")
    ax[0, 0].semilogx(nn, pr, "-", color=c)
    ax[0, 0].semilogx(nn, prc, ":", color=c)
    ax[0, 1].semilogx(d7c[tc][:, 0], d7c[tc][:, 1], "o", color=c, mec="k", ms=5)
    ax[0, 1].semilogx(nn, ec, "-", color=c)
    ax[0, 1].semilogx(nn, ecc, ":", color=c)
for tc, ls in ((25, "--"), (125, "-.")):
    ax[0, 1].semilogx(NS, ec_old[tc], ls, color="0.4", lw=1)
ax[0, 1].plot([], [], "--", color="0.4", label="round-one read-out (switching state only), 25/125 C")
ax[0, 0].plot([], [], "k-", label="A0: Arrhenius wake-up, E_c(T) <= 0")
ax[0, 0].plot([], [], "k:", label="C: P_s(T) and E_c rising with T")
ax[0, 0].set(xlabel="Cycles (2.3 MV/cm, 100 kHz)", ylabel=r"$2P_r$ ($\mu$C/cm$^2$)",
             title="2Pr, Fengler AEM 2017 Fig. 7a (digitized)")
ax[0, 0].legend(fontsize=7.5, frameon=False, ncol=2)
ax[0, 1].plot([], [], "k-", label="A0 (same cards as left)")
ax[0, 1].plot([], [], "k:", label="C")
ax[0, 1].set(xlabel="Cycles", ylabel=r"$2E_c$ (MV/cm)", title="2Ec from the same read-out loops, Fig. 7c (digitized)")
ax[0, 1].legend(fontsize=7.5, frameon=False)
pB = f_card(thB)
for nn_, c in ((10, "C3"), (100, "C1"), (1e4, "C2"), (1e9, "k")):
    h = nm.cycle_history(pB, *CYC, [nn_], nd=16, sub=40)
    L = nm.loop_figures(pB, AMP, 1e4, nd=16, history=nm.History(nw=h["a_w"][0] * pB["wk_nw"]), npts=400)
    ax[1, 0].plot(L["v"] / TFE / 1e8, L["q"] * 100, color=c, label=f"{nn_:g} cycles")
    ax[1, 0].plot([L["vc_m"] / TFE / 1e8, L["vc_p"] / TFE / 1e8], [0, 0], "o", color=c, ms=5)
ax[1, 0].axhline(0, color="0.7", lw=0.6)
ax[1, 0].set(xlabel="E (MV/cm)", ylabel=r"Q/A, centred ($\mu$C/cm$^2$)",
             title="Model read-out loops at 25 C (fit A0): the pinched loop opens")
ax[1, 0].legend(fontsize=8, frameon=False)
# pinched-loop mechanism: switching charge and total charge on the rising branch
L0 = nm.loop_figures(pB, AMP, 1e4, nd=16, history=nm.History(nw=0.0), npts=400)
qd_ = L0["q"] * 100
ax[1, 1].plot(L0["v"] / TFE / 1e8, qd_, "k-", label="total charge (what is measured)")
c_d = pB["c_d"] * L0["v"] * 100
ax[1, 1].plot(L0["v"] / TFE / 1e8, qd_ - c_d, "C0--", label="minus the linear dielectric part")
ax[1, 1].axhline(0, color="0.7", lw=0.6)
ax[1, 1].set(xlabel="E (MV/cm)", ylabel=r"Q/A ($\mu$C/cm$^2$)",
             title="Pristine loop (fit A0): crossings sit inside the pinch")
ax[1, 1].legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "r2_wakeup_coercive.png", dpi=150)
plt.close(fig)
log("1. figure done")

# =============================================================== 2. Lenox 2023: pulse width and duty cycle
rl = rows("lenox2023_pssa_fig5a_pulse_width.csv")
d5a = np.array([(float(r["cycles_nominal"]), float(r["series"].split("us")[0]), float(r["Pr_norm_to_pristine"])) for r in rl])
d6 = np.array([(1e6, w, 1 + x) for w, x in zip([2, 5, 10, 20, 50], [0.07, 0.15, 0.18, 0.25, 0.30])])   # text values
d5b = np.array([(float(r["cycles_digitized"]), 2.0, float(r["Pr_norm_to_pristine"]))
                for r in rows("lenox2023_pssa_fig5b_2us_1e6_1e8.csv") if r["flag"] != "outlier_in_original"])
DL = np.vstack([d5a, d6, d5b])
PER = 100e-6
EDGE = 20e-9


def dose_law(th, n, w, m):
    a, lnk, ka = th
    return 1 + a * (1 - np.exp(-((n * (w / 2.0) ** m) / np.exp(lnk)) ** ka))


def e18_closed(th, n, w, nq=8):
    """Pr/Pr0 = (1 - f_q (1 - Z)) / (1 - f_q), Z from the E18 states under the Lenox train (2 MV/cm, n_q = 2).

    ``rho`` is the re-pinning per period in units of 1/tau (rest channel wk_rho, or wk_tr f for the per-reversal
    channel: every Lenox pulse switches fully, so the two channels give the same curves).
    """
    lnfq, lnrho, lnt1, lnt2 = th
    fq = 1 / (1 + np.exp(-lnfq))
    rho = np.exp(lnrho)
    phi = 4.0   # (2 MV/cm / 1 MV/cm)^2
    e = np.linspace(lnt1, lnt2, nq + 1)
    tau = np.exp(0.5 * (e[:-1] + e[1:]))
    duty = 2 * (w * 1e-6 + EDGE) / PER
    zss = phi * duty / (phi * duty + rho)
    z = zss[:, None] * (1 - np.exp(-(n * PER * (phi * duty + rho))[:, None] / tau[None, :]))
    return (1 - fq * (1 - z.mean(1))) / (1 - fq)


def multistart(fun, x0s, lo, hi):
    best = None
    for x0 in x0s:
        r = least_squares(fun, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return best


res2 = {}
for label, dat in (("Fig. 5a only (9 points)", d5a), ("Fig. 5a + text at 1e6 + Fig. 5b (2 us to 1e8), 23 points", DL)):
    entry = {}
    for m, lab in ((0.0, "m = 0 (switching count)"), (1.0, "m = 1 (time under field)"),
                   (2.07, "m = beta = 2.07 (rare switching of pinned domains)")):
        r = multistart(lambda th: dose_law(th, dat[:, 0], dat[:, 1], m) - dat[:, 2],
                       [[0.4, lk, ka] for lk in np.linspace(0, 30, 11) for ka in (0.1, 0.3, 0.6)],
                       [0, -20, 0.02], [5, 60, 3])
        entry[lab] = {"RMS": rms(r.fun), "amplitude": float(r.x[0]), "kappa": float(r.x[2])}
    r = multistart(lambda th: dose_law(th[:3], dat[:, 0], dat[:, 1], th[3]) - dat[:, 2],
                   [[0.4, lk, 0.3, m0] for lk in np.linspace(0, 40, 9) for m0 in (1, 2, 3)],
                   [0, -20, 0.02, -1], [5, 80, 3, 8])
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun**2) / max(len(r.fun) - 4, 1)
    entry["m free"] = {"m": float(r.x[3]), "m std": float(np.sqrt(cov[3, 3])), "RMS": rms(r.fun)}
    rq = multistart(lambda th: e18_closed(th, dat[:, 0], dat[:, 1]) - dat[:, 2],
                    [[-0.5, np.log(0.6), l1, l1 + dl] for l1 in np.arange(-8, 4, 1.0) for dl in (2, 5, 9)],
                    [-5, -8, -20, -20], [3, 5, 15, 25])
    rq0 = multistart(lambda th: e18_closed([th[0], -30.0, th[1], th[2]], dat[:, 0], dat[:, 1]) - dat[:, 2],
                     [[-0.5, l1, l1 + dl] for l1 in np.arange(-8, 4, 1.0) for dl in (2, 5, 9)],
                     [-5, -20, -20], [3, 15, 25])
    fq = 1 / (1 + np.exp(-rq.x[0]))
    entry["E18 depinning + re-pinning, 4 parameters"] = {
        "RMS": rms(rq.fun), "wk_fq": float(fq), "re-pinning: wk_rho (rest channel, n_q = 2)": float(np.exp(rq.x[1])),
        "or wk_tr = wk_rho / 10 kHz [s] (per-reversal channel)": float(np.exp(rq.x[1]) / 1e4),
        "rho_eff at 2 MV/cm = wk_rho / 4": float(np.exp(rq.x[1]) / 4),
        "wk_q1 [s at 1 MV/cm]": float(np.exp(rq.x[2])), "wk_q2 [s at 1 MV/cm]": float(np.exp(rq.x[3]))}
    entry["E18 without re-pinning (rho = 0), 3 parameters"] = {"RMS": rms(rq0.fun)}
    res2[label] = entry
    res2[label]["_x"] = rq.x
OUT["lenox2023_pulse_width"] = {k: {kk: vv for kk, vv in v.items() if kk != "_x"} for k, v in res2.items()}
OUT["lenox2023_pulse_width"]["digitizing uncertainty"] = "+-0.015 (Fig. 5a/5b); text values exact but leakage-affected for 20/50 us"
xq = res2["Fig. 5a + text at 1e6 + Fig. 5b (2 us to 1e8), 23 points"]["_x"]
fq = float(1 / (1 + np.exp(-xq[0])))
# the per-reversal re-pinning channel (wake-up persists at rest, Sec. 1.2 of the notes); the rest channel gives the
# same curves under this waveform
pL = nm.generic_card(wk_fq=fq, wk_tr=float(np.exp(xq[1])) / 1e4, wk_q1=float(np.exp(xq[2])), wk_q2=float(np.exp(xq[3])),
                     wk_nq=2.0)
pL_rest = dict(pL, wk_tr=0.0, wk_rho=float(np.exp(xq[1])))


def lenox_period(w):
    w = w * 1e-6
    return ([0, EDGE, EDGE + w, 2 * EDGE + w, PER / 2, PER / 2 + EDGE, PER / 2 + EDGE + w, PER / 2 + 2 * EDGE + w, PER],
            [0, 2.0, 2.0, 0, 0, -2.0, -2.0, 0, 0])


# the full model (cycle-averaged E18 inside the domain model) against the closed form
chk = {}
nn = np.logspace(1, 8, 22)
model_curves = {}
for w in (2, 5, 10, 20, 50):
    r = nm.cycle_history(pL, *lenox_period(w), nn, nd=16, sub=80)
    f0 = float(nm.pinned_fraction(pL, 0.0, 0.0)[0])
    model_curves[w] = (1 - r["f"]) / (1 - f0)
    chk[f"{w} us"] = float(np.max(np.abs(model_curves[w] - e18_closed(xq, nn, np.full_like(nn, w)))))
OUT["lenox2023_pulse_width"]["full model (cycle_history) vs closed form, max |difference|"] = chk
# the full model at the data points (per-reversal channel; the closed-form parameters are not refitted)
res_full = []
for w in (2, 5, 10, 20, 50):
    m = DL[:, 1] == w
    r = nm.cycle_history(pL, *lenox_period(w), DL[m, 0], nd=16, sub=80)
    f0 = float(nm.pinned_fraction(pL, 0.0, 0.0)[0])
    order = np.argsort(DL[m, 0])
    res_full += list((1 - r["f"]) / (1 - f0) - DL[m, 2][order])
OUT["lenox2023_pulse_width"]["full model (per-reversal channel) at the 23 data points, RMS"] = rms(res_full)
OUT["lenox2023_pulse_width"]["plateau Z vs pulse width"] = {
    f"{w} us": float(nm.depin_plateau(pL, 2 * (w * 1e-6 + EDGE) / PER, 2.0e8 * pL["tfe"], freq=1e4)) for w in (2, 5, 10, 20, 50)}
# rest after wake-up: the two re-pinning channels differ only here (50 us, 1e8 cycles, then 0 V at 25 C)
zq = r["zq"][-1]
rest = {"per-reversal channel (wk_tr)": {}, "rest channel (wk_rho)": {}}
for tr in (1.0, 60.0, 3600.0, 86400.0):
    for lab, pp in (("per-reversal channel (wk_tr)", pL), ("rest channel (wk_rho)", pL_rest)):
        z2 = nm.depin_step(pp, zq, 0.0, tr, nm.depin_taus(pp))
        rest[lab][f"{tr:g} s"] = float((1 - fq * (1 - z2.mean())) / (1 - fq))
OUT["lenox2023_pulse_width"]["prediction: Pr/Pr0 after resting at 0 V (50 us, 1e8 cycles)"] = rest

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
nl = np.logspace(1, 8.2, 60)
for k, w in enumerate((2, 5, 10, 20, 50)):
    c = f"C{k}"
    for dat, mk in ((d5a, "o"), (d6, "s"), (d5b, "^")):
        m = dat[:, 1] == w
        ax[0].semilogx(dat[m, 0], dat[m, 2], mk, color=c, mec="k", ms=5)
    ax[0].semilogx(nl, e18_closed(xq, nl, np.full_like(nl, w)), "-", color=c, label=f"{w} us")
    ax[0].semilogx(nn, model_curves[w], "x", color=c, ms=4)
ax[0].plot([], [], "ko", label="Fig. 5a"), ax[0].plot([], [], "ks", label="text, 1e6")
ax[0].plot([], [], "k^", label="Fig. 5b (2 us)"), ax[0].plot([], [], "kx", label="full model (cycle_history)")
e = res2["Fig. 5a + text at 1e6 + Fig. 5b (2 us to 1e8), 23 points"]
ax[0].set(xlabel="Cycles (+-2 MV/cm, 10 kHz)", ylabel="Pr / Pr(pristine)", ylim=(0.98, 1.36),
          title=f"E18 depinning + re-pinning, RMS {e['E18 depinning + re-pinning, 4 parameters']['RMS']:.3f}")
ax[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
mfree = e["m free"]
for k, w in enumerate((2, 5, 10, 20, 50)):
    c = f"C{k}"
    for dat, mk in ((d5a, "o"), (d6, "s"), (d5b, "^")):
        m = dat[:, 1] == w
        ax[1].semilogx(dat[m, 0], dat[m, 2], mk, color=c, mec="k", ms=5)
r = multistart(lambda th: dose_law(th[:3], DL[:, 0], DL[:, 1], th[3]) - DL[:, 2],
               [[0.4, lk, 0.3, m0] for lk in np.linspace(0, 40, 9) for m0 in (1, 2, 3)], [0, -20, 0.02, -1], [5, 80, 3, 8])
for k, w in enumerate((2, 5, 10, 20, 50)):
    ax[1].semilogx(nl, dose_law(r.x[:3], nl, np.full_like(nl, w), r.x[3]), "-", color=f"C{k}")
ax[1].set(xlabel="Cycles", ylabel="Pr / Pr(pristine)", ylim=(0.98, 1.36),
          title=f"Single dose law N w^m (m = {mfree['m']:.1f} free), RMS {mfree['RMS']:.3f}")
fig.savefig(OUTDIR / "r2_pulse_width.png", dpi=150)
plt.close(fig)
log("2. Lenox wake-up done")

# =============================================================== 3. Lenox 2023: leakage onset and failure
rt = rows("lenox2023_pssa_text_leakage_failure.csv")
onset = {float(r["pulse_width_us"]): float(r["cycles"]) for r in rt if r["quantity"] == "leakage_onset"}
fail = {float(r["pulse_width_us"]): float(r["cycles"]) for r in rt if r["quantity"] == "failure"}
incr = [(float(r["pulse_width_us"]), float(r["cycles"]), float(r["value_pct"])) for r in rt
        if r["quantity"] == "leakage_increase"]
WS = np.array(sorted(onset))


def dose_per_cycle(w, tl, nf):
    """Field-time stress per cycle in seconds at 2 MV/cm (fat_tf = 1 s, field factor normalized to 2 MV/cm)."""
    p = nm.generic_card(fat_d=0.1, fat_tf=1.0, fat_nf=nf, fat_tl=tl)
    return nm.stress_per_period(p, *lenox_period(w)) / 2.0**nf


leak_out = {"plain time under field (no build-up)": {}}
d_plain = np.array([onset[w] * dose_per_cycle(w, 0.0, 1.0) for w in WS])
leak_out["plain time under field (no build-up)"] = {
    "onset dose [s] (2, 5, 10, 20, 50 us)": list(map(float, d_plain)),
    "spread (max/min)": float(d_plain.max() / d_plain.min()), "log std": float(np.std(np.log(d_plain))),
    "failure dose [s]": {f"{w:g} us": float(fail[w] * dose_per_cycle(w, 0.0, 1.0)) for w in fail}}
best_tl = {}
for nf in (1.0, 2.0, 4.0):
    grid = np.exp(np.linspace(np.log(1e-8), np.log(3e-5), 60))
    sp = [np.std(np.log([onset[w] * dose_per_cycle(w, tl, nf) for w in WS])) for tl in grid]
    tl = float(grid[int(np.argmin(sp))])
    best_tl[nf] = tl
    dd = np.array([onset[w] * dose_per_cycle(w, tl, nf) for w in WS])
    leak_out[f"with E19 build-up, n_F = {nf:g}"] = {
        "t_L [us]": tl * 1e6, "onset dose [s]": list(map(float, dd)), "spread (max/min)": float(dd.max() / dd.min()),
        "log std": float(min(sp)), "failure dose [s]": {f"{w:g} us": float(fail[w] * dose_per_cycle(w, tl, nf)) for w in fail}}
# E20 growth: ratio - 1 = k (dose)^m_L with the n_F = 4 build-up
tl4 = best_tl[4.0]
X = np.array([[np.log(n * dose_per_cycle(w, tl4, 4.0)), np.log(pct / 100)] for w, n, pct in incr])
A = np.vstack([X[:, 0], np.ones(len(X))]).T
coef, *_ = np.linalg.lstsq(A, X[:, 1], rcond=None)
pred = A @ coef
leak_out["E20 growth fit, ratio - 1 = k_L (dose)^m_L (4 points, n_F = 4 build-up)"] = {
    "m_L": float(coef[0]), "residual factor (rms of ln)": float(np.exp(rms(pred - X[:, 1]))),
    "points (w us, cycles, %)": incr}
OUT["lenox2023_leakage_failure"] = leak_out

# Pesic 2016: leakage (Fig. 1c) and Pr fatigue (Fig. 1a) of the same device -> E20 against the round-one fatigue age
rc = rows("pesic2016_afm_fig1c_leakage_cycles.csv")
jl = {}
for r in rc:
    jl.setdefault(float(r["cycles_nominal"]), []).append(float(r["J_A_cm2"]))
jl = {k: np.array(v) for k, v in jl.items()}
jratio = {n: float(np.exp(np.mean(np.log(jl[n] / jl[1e3])))) for n in (1e4, 1e5, 1e6)}
jspread = {n: [float(np.min(jl[n] / jl[1e3])), float(np.max(jl[n] / jl[1e3]))] for n in (1e4, 1e5, 1e6)}
r1 = json.loads((OUTDIR / "fatigue_imprint_comparison.json").read_text(encoding="utf-8"))["wakeup_fatigue_pesic2016"]["fit (7 parameters)"]
pP = nm.generic_card(tfe=10e-9, ps=r1["ps [C/m^2]"], wk_fp=r1["wk_fp"], wk_nw=r1["wk_nw [cycles]"], wk_kw=r1["wk_kw"],
                     fat_d=r1["fat_d"], fat_tf=r1["fat_tf [s at 1 MV/cm]"], fat_nf=4.0, fat_kf=r1["fat_kf"])
hP = nm.cycle_history(pP, [0, 10e-9, 50e-6, 50e-6 + 20e-9, 1e-4 - 10e-9, 1e-4], [0, 4, 4, -4, -4, 0],
                      [1e3, 1e4, 1e5, 1e6], nd=16, sub=40)
aF = dict(zip([1e3, 1e4, 1e5, 1e6], hP["a_f"]))
# The '0-1 k' curve is one curve for 0 and 1000 cycles (leakage constant during wake-up, Pesic2016 text), so it is
# J_0.  Fit J(N)/J_0 = 1 + k a_F(N)^m on the three later curves and check what it predicts at 1e3 cycles.
def e20_res(th):
    k, m = np.exp(th[0]), th[1]
    return [np.log(1 + k * aF[n] ** m) - np.log(jratio[n]) for n in (1e4, 1e5, 1e6)]


fe20 = least_squares(e20_res, [0.0, 0.5], bounds=([-10, 0.05], [10, 3]))
k20, m20 = float(np.exp(fe20.x[0])), float(fe20.x[1])
OUT["pesic2016_leakage_E20"] = {
    "data": "Fig. 1c (digitized) J(N)/J(0-1k), geometric mean over 0.2-3 MV/cm; Fig. 1a fatigue fitted in round one",
    "J ratio (geometric mean)": {f"{n:g}": v for n, v in jratio.items()},
    "J ratio range over the 7 fields": {f"{n:g}": v for n, v in jspread.items()},
    "a_F from the round-one Pesic fit": {f"{n:g}": float(v) for n, v in aF.items()},
    "D from the round-one Pesic fit": {f"{n:g}": float(v) for n, v in zip([1e3, 1e4, 1e5, 1e6], hP["D"])},
    "fit fat_lk, fat_lm (2 parameters, 3 ratios; J_0 = the 0-1k curve)": {
        "fat_lk": k20, "fat_lm": m20, "RMS of ln ratio": rms(fe20.fun),
        "predicted J(1e3)/J_0 (data: 1, same curve as 0 cycles)": float(1 + k20 * aF[1e3] ** m20)},
}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
ax[0].loglog(WS, d_plain, "ko-", label="time under field N 2w")
for nf, c in ((2.0, "C0"), (4.0, "C3")):
    dd = [onset[w] * dose_per_cycle(w, best_tl[nf], nf) for w in WS]
    ax[0].loglog(WS, dd, "s-", color=c, label=f"E19 build-up, n_F = {nf:g}, t_L = {best_tl[nf] * 1e6:.1f} us")
ax[0].set(xlabel="Pulse width (us)", ylabel="Dose at +50 % leakage (s at 2 MV/cm)",
          title="Leakage onset, Lenox 2023 (text values)")
ax[0].set_xticks(WS)
ax[0].set_xticklabels([f"{w:g}" for w in WS])
ax[0].minorticks_off()
ax[0].set_yticks([40, 60, 100, 200, 400])
ax[0].set_yticklabels(["40", "60", "100", "200", "400"])
ax[0].legend(fontsize=8, frameon=False)
for w, mk in ((2, "o"), (20, "s"), (50, "^")):
    pts = [(n, pct) for ww, n, pct in incr if ww == w]
    ax[1].loglog([n * dose_per_cycle(w, tl4, 4.0) for n, _ in pts], [pct / 100 for _, pct in pts], mk, mec="k",
                 label=f"{w} us")
xx = np.logspace(1.5, 2.8, 20)
ax[1].loglog(xx, np.exp(coef[1]) * xx ** coef[0], "k-", lw=1, label=f"k dose^{coef[0]:.1f} (residual x{np.exp(rms(pred - X[:, 1])):.1f})")
ax[1].set(xlabel="Dose with build-up, n_F = 4 (s at 2 MV/cm)", ylabel="Leakage increase J/J0 - 1",
          title="Leakage growth, Lenox 2023 (text values)")
ax[1].minorticks_off()
ax[1].set_xticks([40, 100, 200, 400])
ax[1].set_xticklabels(["40", "100", "200", "400"])
ax[1].legend(fontsize=8, frameon=False)
for n, c in ((1e3, "C0"), (1e4, "C3"), (1e5, "C2"), (1e6, "k")):
    ee = [float(r["E_MV_cm"]) for r in rc if float(r["cycles_nominal"]) == n]
    ax[2].semilogy(ee, jl[n], "o-", color=c, ms=4, label=("0-1k" if n == 1e3 else f"{n:g}") + " cycles")
    if n > 1e3:
        ax[2].semilogy(ee, jl[1e3] * (1 + k20 * aF[n] ** m20), "--", color=c, lw=1)
ax[2].plot([], [], "k--", lw=1, label=f"E20: J0 (1 + {k20:.1f} a_F^{m20:.2f})")
ax[2].set(xlabel="E (MV/cm)", ylabel="J (A/cm$^2$)", title="Leakage vs cycles, Pesic 2016 Fig. 1c (digitized)")
ax[2].legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "r2_leakage.png", dpi=150)
plt.close(fig)
log("3. leakage done")

# =============================================================== 4. Pesic 2016 Fig. 7b: recovery after rest (a.u.)
rp = rows("pesic2016_afm_fig7b_recovery_au.csv")
ser = {}
for r in rp:
    ser.setdefault(r["series"], []).append((float(r["cycles_nominal"]), float(r["y_ticks"])))
ser = {k: np.array(v) for k, v in ser.items()}
first = ser["1st_375K_pristine"]


def fall_cycles(s, dy=1.0):
    """Cycle count (log-interpolated) at which a series first falls dy ticks below its own maximum."""
    i = int(np.argmax(s[:, 1]))
    y0 = s[i, 1]
    for j in range(i + 1, len(s)):
        if s[j, 1] <= y0 - dy:
            a, b = s[j - 1], s[j]
            f = (a[1] - (y0 - dy)) / (a[1] - b[1])
            return float(10 ** (np.log10(max(a[0], 1)) + f * (np.log10(b[0]) - np.log10(max(a[0], 1)))))
    return float("nan")

drop = first[:, 1].max() - first[-1, 1]
rec = {}
for k, wt in (("2nd_after_1s", 1.0), ("2nd_after_1min", 60.0), ("2nd_after_30min", 1800.0)):
    s = ser[k]
    rec[k] = {"wait [s]": wt, "start (N = 1) - end of 1st run [ticks]": float(s[0, 1] - first[-1, 1]),
              "plateau max - end of 1st run [ticks]": float(s[:, 1].max() - first[-1, 1]),
              "plateau recovery / 1st-run fatigue drop": float((s[:, 1].max() - first[-1, 1]) / drop),
              "cycles to fall 1 tick below own max": fall_cycles(s)}
OUT["pesic2016_recovery"] = {"1st-run fatigue drop (max - last) [ticks]": float(drop), "second runs": rec,
                             "1st run: cycles to fall 1 tick below max": fall_cycles(first)}
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
for (k, s), c in zip(ser.items(), ("C3", "C5", "C2", "k")):
    ax.semilogx(np.maximum(s[:, 0], 1), s[:, 1], "o-", color=c, ms=4, label=k.replace("_", " "))
ax.axhline(0, color="C0", ls=":", lw=1)
ax.set(xlabel="Cycles (375 K)", ylabel="2Pr (y-tick units, a.u.)", title="Recovery after rest, Pesic 2016 Fig. 7b (digitized)")
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "r2_recovery.png", dpi=150)
plt.close(fig)
log("4. Pesic recovery done")

(OUTDIR / "cycling_r2.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
print(json.dumps(OUT, indent=1, default=float))
log("total")
