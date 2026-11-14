"""Wake-up, fatigue and imprint of the NLS model (E13-E17) against published data.

Writes the figures docs/research/fi_*.png and docs/research/fatigue_imprint_comparison.json.

Inputs: digitized values in data/literature/ (each file has the citation, the
figure, the device and "digitized from figure, approx." in its header), and the
read-endurance prediction of examples/literature_comparison.py stored in
docs/literature_comparison.json.  Every fitted parameter is listed in the JSON
with the number of points it was fitted to.  No measured raw data were used.

1. Wake-up vs temperature, Fengler et al. AEM 2017 Fig. 7a/7c (TiN/7 nm HZO/TiN).
2. Wake-up and fatigue at one condition, Pesic et al. AFM 2016 Fig. 1a (Sr:HfO2).
3. Two cycling fields read at the cycling field, Park et al. JMCC 2020 Fig. 5e (TDMA-HZO):
   pinning only vs pinning plus activation-field increase.
4. Pulse-width dependence of wake-up, Lenox et al. pssa 2023 Fig. 5a: which driver?
5. Read endurance after 1e8 reads, Mukherjee et al. IEDM 2023 Fig. 7: read-stress fatigue or imprint.
6. Imprint vs storage time, Buragohain et al. ACS AMI 2019 Fig. 2c (La:HfO2) and vs
   temperature, Fengler et al. JAP 2018 Fig. 3a (HZO).
7. Cycle-averaged integration against direct simulation (numerical check).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares

from fe_models import nls_multidomain as nm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "literature"
OUTDIR = ROOT / "docs" / "research"
OUTDIR.mkdir(parents=True, exist_ok=True)
UC = 1e-2
OUT: dict = {}
T0 = time.time()


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def square_wave(v, period, edge=10e-9):
    """Bipolar rectangular period 0 -> +v -> -v -> 0 (half period each)."""
    return [0, edge, period / 2, period / 2 + 2 * edge, period - edge, period], [0, v, v, -v, -v, 0]


# ------------------------------------------------------------- 1. wake-up vs temperature
r7a = rows("fengler2017_aem_fig7a_wakeup_2pr_T.csv")
r7c = rows("fengler2017_aem_fig7c_wakeup_2ec_T.csv")
TEMPS = [25, 50, 75, 100, 125]
d7a = {t: np.array([(float(r["cycles_nominal"]), float(r["2Pr_uC_cm2"])) for r in r7a if r["series"] == f"{t}C"])
       for t in TEMPS}
d7c = {t: np.array([(float(r["cycles_nominal"]), float(r["2Ec_MV_cm"])) for r in r7c if r["series"] == f"{t}C"])
       for t in TEMPS}
TFE_F = 7e-9
CYC_F = square_wave(2.3e8 * TFE_F, 1e-5)          # 2.3 MV/cm, 100 kHz rectangular (paper)
N7 = [10, 100, 1000, 5000, 1e4, 5e4, 1e5]


def wake_card(th, **kw):
    ps, tcps, fp, lnw, kw_, ea = th
    return nm.generic_card(tfe=TFE_F, ps=ps, tc_ps=tcps, wk_fp=fp, wk_nw=float(np.exp(lnw)), wk_kw=kw_, wk_ea=ea, **kw)


def wake_model(th, t_c, ns=N7):
    return nm.cycle_history(wake_card(th), *CYC_F, ns, nd=16, temp=273.15 + t_c, sub=40)["dP"] / UC


def wake_resid(th):
    return np.concatenate([wake_model(th, t) - d7a[t][:, 1] for t in TEMPS])


lo, hi = [0.1, -2e-3, 0.0, 0.0, 0.05, 0.0], [0.4, 5e-3, 0.95, 25.0, 2.0, 1.5]
xs = [0.02, 1e-3, 0.1, 1.0, 0.1, 0.1]
fit_a = least_squares(wake_resid, [0.19, 1e-3, 0.45, np.log(300), 0.35, 0.3], bounds=(lo, hi), x_scale=xs)
fit_b = least_squares(lambda th: wake_resid([th[0], 0.0, *th[1:]]), [0.19, 0.45, np.log(3e3), 0.25, 0.3],
                      bounds=(lo[:1] + lo[2:], hi[:1] + hi[2:]), x_scale=xs[:1] + xs[2:])
th_b = [fit_b.x[0], 0.0, *fit_b.x[1:]]


def names6(th):
    return {"ps [C/m^2]": th[0], "tc_ps [1/K]": th[1], "wk_fp": th[2], "wk_nw [cycles]": float(np.exp(th[3])),
            "wk_kw": th[4], "wk_ea [eV]": th[5]}


OUT["wakeup_temperature_fengler2017"] = {
    "data": "Fig. 7a, 2Pr at 7 cycle counts x 5 temperatures (35 points), digitizing uncertainty +-0.3 uC/cm2",
    "fit A (6 parameters: ps, tc_ps, wk_fp, wk_nw, wk_kw, wk_ea)": {**names6(fit_a.x), "RMS [uC/cm2]": rms(fit_a.fun)},
    "fit B (tc_ps = 0, Arrhenius only; 5 parameters)": {**names6(th_b), "RMS [uC/cm2]": rms(fit_b.fun)},
}

# coercive field during wake-up (Fig. 7c): the split +-V_sp and the activation-field scale, with fit A
TRI_F = ([0, 25e-6, 75e-6, 100e-6], [0, 2.3e8 * TFE_F, -2.3e8 * TFE_F, 0])  # read-out loop (assumed 10 kHz, +-2.3 MV/cm)


def ec_model(th_ec, t_c, ns, th=fit_a.x):
    vsp, mea = th_ec
    p = wake_card(th, wk_vsp=vsp, mul_ea=mea)
    hist = nm.cycle_history(p, *CYC_F, ns, nd=16, temp=273.15 + t_c, sub=40)
    out = []
    for aw in hist["a_w"]:
        r = nm.cycle_history(p, *TRI_F, [0], nd=16, temp=273.15 + t_c, sub=200, s_start=-1.0,
                             history=nm.History(nw=aw * p["wk_nw"]))
        out.append((r["vc_p"][0] - r["vc_m"][0]) / TFE_F / 1e8)
    return np.array(out)


def ec_resid(th_ec):
    return np.concatenate([ec_model(th_ec, t, d7c[t][:, 0]) - d7c[t][:, 1] for t in TEMPS])


fit_ec = least_squares(ec_resid, [0.1, 0.55], bounds=([0.0, 0.3], [1.0, 2.0]), x_scale=[0.05, 0.05], diff_step=3e-2)
fit_ec0 = least_squares(lambda th: ec_resid([0.0, th[0]]), [0.55], bounds=([0.3], [2.0]), diff_step=3e-2)
OUT["wakeup_temperature_fengler2017"]["2Ec, Fig. 7c (18 points, +-0.005 MV/cm)"] = {
    "read-out assumption": "10 kHz triangle at +-2.3 MV/cm (frequency from the paper's Fig. 2, amplitude assumed)",
    "with split: wk_vsp [V], mul_ea": list(map(float, fit_ec.x)), "RMS [MV/cm]": rms(fit_ec.fun),
    "without split (wk_vsp = 0): mul_ea": float(fit_ec0.x[0]), "RMS without split [MV/cm]": rms(fit_ec0.fun),
    "data range [MV/cm]": [float(min(d[:, 1].min() for d in d7c.values())), float(max(d[:, 1].max() for d in d7c.values()))],
}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
nn = np.logspace(0.5, 5.3, 30)
for k, t in enumerate(TEMPS):
    c = f"C{k}"
    ax[0].semilogx(d7a[t][:, 0], d7a[t][:, 1], "o", color=c, mec="k", ms=5, label=f"{t} C")
    ax[0].semilogx(nn, wake_model(fit_a.x, t, nn), "-", color=c)
    ax[0].semilogx(nn, wake_model(th_b, t, nn), ":", color=c)
    ax[1].semilogx(d7c[t][:, 0], d7c[t][:, 1], "o", color=c, mec="k", ms=5)
    ne = np.logspace(1.5, 4.3, 10)
    ax[1].semilogx(ne, ec_model(fit_ec.x, t, ne), "-", color=c)
    ax[1].semilogx(ne, ec_model([0.0, fit_ec0.x[0]], t, ne), ":", color=c)
ax[0].plot([], [], "k-", label=f"fit A: P_s(T) + wake-up, RMS {rms(fit_a.fun):.2f}")
ax[0].plot([], [], "k:", label=f"fit B: Arrhenius wake-up only, RMS {rms(fit_b.fun):.2f}")
ax[0].set(xlabel="Cycles (2.3 MV/cm, 100 kHz)", ylabel=r"$2P_r$ ($\mu$C/cm$^2$)",
          title="Wake-up vs temperature: Fengler AEM 2017 Fig. 7a (digitized)")
ax[0].legend(fontsize=7.5, frameon=False, ncol=2)
ne = np.logspace(1.5, 4.3, 10)
ax[1].semilogx(ne, ec_model([0.4, fit_ec.x[1]], 25, ne), "k--", lw=1)
ax[1].plot([], [], "k--", lw=1, label="split fixed at $V_{sp}$ = 0.4 V (25 C): pristine loop is wider")
ax[1].plot([], [], "k-", label=f"with split $\\pm V_{{sp}}$, RMS {rms(fit_ec.fun):.3f} MV/cm")
ax[1].plot([], [], "k:", label=f"without split, RMS {rms(fit_ec0.fun):.3f} MV/cm")
ax[1].set(xlabel="Cycles", ylabel=r"$2E_c$ (MV/cm)", title="Coercive field during wake-up: Fig. 7c (digitized)")
ax[1].legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "fi_wakeup_temperature.png", dpi=150)
plt.close(fig)
print(f"[{time.time()-T0:.0f} s] 1. wake-up vs T done")

# ------------------------------------------------------------- 2. Pesic 2016: wake-up and fatigue
rp = rows("pesic2016_afm_fig1a_pr_cycles.csv")
npr = sorted({float(r["cycles_nominal"]) for r in rp})
d_pe = np.array([(n, np.mean([abs(float(r["Pr_uC_cm2"])) for r in rp if float(r["cycles_nominal"]) == n]) * 2)
                 for n in npr])
d_pe[0, 0] = 0.0  # the pristine point is plotted at 0.1 cycles
TFE_P = 10e-9
CYC_P = square_wave(4.0, 1e-4)                   # 4 V, 10 kHz (waveform shape not stated: rectangular assumed)


def pe_card(th, **kw):
    ps, fp, lnw, kw_, fd, lntf, kf = th
    return nm.generic_card(tfe=TFE_P, ps=ps, wk_fp=fp, wk_nw=float(np.exp(lnw)), wk_kw=kw_, fat_d=fd,
                           fat_tf=float(np.exp(lntf)), fat_nf=4.0, fat_kf=kf, **kw)


def pe_model(th, ns):
    return nm.cycle_history(pe_card(th), *CYC_P, ns, nd=16, sub=40)["dP"] / UC


fit_pe = least_squares(lambda th: pe_model(th, d_pe[:, 0]) - d_pe[:, 1], [0.2, 0.5, np.log(10), 0.4, 0.3, 0.0, 0.5],
                       bounds=([0.1, 0, -5, 0.15, 0, -15, 0.1], [0.4, 0.95, 15, 2, 0.6, 20, 1.0]),
                       x_scale=[0.02, 0.1, 1, 0.1, 0.1, 1, 0.1])
pe = pe_card(fit_pe.x)
OUT["wakeup_fatigue_pesic2016"] = {
    "data": "Fig. 1a, 2Pr = |Pr+| + |Pr-| at 11 cycle counts (pristine included), +-0.4 uC/cm2",
    "fit (7 parameters)": {"ps [C/m^2]": fit_pe.x[0], "wk_fp": fit_pe.x[1], "wk_nw [cycles]": pe["wk_nw"],
                           "wk_kw": fit_pe.x[3], "fat_d": fit_pe.x[4], "fat_tf [s at 1 MV/cm]": pe["fat_tf"],
                           "fat_kf": fit_pe.x[6], "fat_nf (fixed)": 4.0},
    "RMS [uC/cm2]": rms(fit_pe.fun),
    "note": "one cycling condition: the field exponent and the choice of driver are not identifiable from it; "
            "wk_kw bounded below by 0.15 (without the bound the fit runs to 0.05 to follow the first-cycle jump); "
            "fat_kf bounded above by 1 (a distribution of rates; free, it runs to a cliff-like onset)",
}

# ------------------------------------------------------------- 3. Park 2020: two fields, read at the cycling field
rk = rows("park2020_jmcc_fig5e_tdma_pr_field.csv")


def park_series(field):
    pts = {}
    for r in rk:
        if r["series"].startswith(field):
            pts.setdefault(round(float(r["cycles_digitized"]), -1 if float(r["cycles_digitized"]) > 10 else 0), []).append(
                abs(float(r["Pr_uC_cm2"])))
    d = np.array(sorted((n, 2 * np.mean(v)) for n, v in pts.items() if len(v) == 2))
    d[d[:, 0] < 10, 0] = 0.0
    return d


d_pk = {2.5: park_series("2.5MVcm"), 3.0: park_series("3.0MVcm")}
TFE_K = 10e-9   # thickness not given in the review: 10 nm assumed


def pk_card(th, ke):
    ps, mea, fp, lnw, fd, lntf, nf = th[:7]
    return nm.generic_card(tfe=TFE_K, ps=ps, mul_ea=mea, wk_fp=fp, wk_nw=float(np.exp(lnw)), wk_kw=0.5, fat_d=fd,
                           fat_tf=float(np.exp(lntf)), fat_nf=nf, fat_kf=0.5, fat_ke=ke)


def pk_model(th, ke, field, ns):
    return nm.cycle_history(pk_card(th, ke), *square_wave(field * 1e8 * TFE_K, 1e-5), ns, nd=16, sub=40)["dP"] / UC


def pk_resid(th, ke=None):
    ke = th[7] if ke is None else ke
    return np.concatenate([pk_model(th, ke, f, d[:, 0]) - d[:, 1] for f, d in d_pk.items()])


x0 = [0.16, 1.2, 0.15, np.log(1e3), 0.2, 0.0, 4.0]
blo, bhi = [0.08, 0.5, 0.0, 0.0, 0.0, -15, 0.0], [0.3, 2.5, 0.9, 20, 0.9, 25, 12]
fit_pk0 = least_squares(lambda th: pk_resid(th, 0.0), x0, bounds=(blo, bhi), x_scale=[0.01, 0.05, 0.05, 1, 0.05, 1, 0.5])
fit_pk1 = least_squares(pk_resid, x0 + [1.0], bounds=(blo + [0.0], bhi + [20.0]),
                        x_scale=[0.01, 0.05, 0.05, 1, 0.05, 1, 0.5, 0.2])
n_pts = sum(len(d) for d in d_pk.values())


def pk_names(th):
    return {"ps": th[0], "mul_ea": th[1], "wk_fp": th[2], "wk_nw": float(np.exp(th[3])), "fat_d": th[4],
            "fat_tf [s]": float(np.exp(th[5])), "fat_nf": th[6], **({"fat_ke": th[7]} if len(th) > 7 else {})}


OUT["two_fields_park2020"] = {
    "data": f"Fig. 5e TDMA-HZO, 2Pr read at the cycling field, 2.5 and 3.0 MV/cm ({n_pts} points; the curves are "
            "traces of markers and connecting lines); thickness 10 nm assumed; wk_kw = fat_kf = 0.5 fixed",
    "pinning only (fat_ke = 0), 7 parameters": {**pk_names(fit_pk0.x), "RMS [uC/cm2]": rms(fit_pk0.fun)},
    "pinning + activation-field increase, 8 parameters": {**pk_names(fit_pk1.x), "RMS [uC/cm2]": rms(fit_pk1.fun)},
}
print(f"[{time.time()-T0:.0f} s] 2-3 done")

# ------------------------------------------------------------- 4. Lenox 2023: pulse width
rl = rows("lenox2023_pssa_fig5a_pulse_width.csv")
d_le = np.array([(float(r["cycles_nominal"]), float(r["series"].split("us")[0]), float(r["Pr_norm_to_pristine"]))
                 for r in rl])


def lenox_law(X, amp, lnk, kappa, m):
    n, w = X
    return 1.0 + amp * (1.0 - np.exp(-((n * (w / 2.0) ** m) / np.exp(lnk)) ** kappa))


popt, pcov = curve_fit(lenox_law, d_le[:, :2].T, d_le[:, 2], p0=[0.4, np.log(1e6), 0.3, 1.0],
                       bounds=([0, -10, 0.05, -2], [3, 40, 2, 6]), maxfev=20000)
res_m = {}
for m in (0.0, 1.0):
    pf, _ = curve_fit(lambda X, a, lk, ka: lenox_law(X, a, lk, ka, m), d_le[:, :2].T, d_le[:, 2],
                      p0=[0.4, np.log(1e6), 0.3], bounds=([0, -10, 0.05], [3, 40, 2]), maxfev=20000)
    res_m[m] = rms(lenox_law(d_le[:, :2].T, *pf, m) - d_le[:, 2])
OUT["pulse_width_lenox2023"] = {
    "data": "Fig. 5a, Pr normalized to pristine at 1e4 and 1e5 cycles for 2-50 us pulses (9 points)",
    "law": "1 + A {1 - exp[-(N (w/2 us)^m / N_0)^kappa]}; m = 0: switching-count driver (this model), "
           "m = 1: field-time driver",
    "free m": {"m": float(popt[3]), "m std": float(np.sqrt(pcov[3, 3])), "kappa": float(popt[2]),
               "RMS": rms(lenox_law(d_le[:, :2].T, *popt) - d_le[:, 2])},
    "RMS with m = 0": res_m[0.0], "RMS with m = 1": res_m[1.0],
}

# ------------------------------------------------------------- 5. Mukherjee 2023: read endurance after 1e8 reads
# The no-history prediction of literature_comparison.py follows the -0.1 V curve but overestimates the tail
# switching at -0.3 and -0.5 V, so the late drop is analysed relative to each curve's own value at 1e6 reads.
lc = json.loads((ROOT / "docs" / "literature_comparison.json").read_text(encoding="utf-8"))
base = lc["read_endurance"]["S4 tail (Alessandri 2019)"]
cvfit = lc["cv"]["Mukherjee IEDM 2023 fit (eps_inf, d_eps, d0 [MV/cm], w [MV/cm])"]
TFE_M, PERIOD_R, T_ON, A_M = 9e-9, 1.5e-6, 1e-6, 60e-6 * 80e-6   # as in literature_comparison.py
reads = []
for v in ("0.1", "0.3", "0.5"):
    b = base[f"-{v} V"]
    for n, q in b["data_pC"].items():
        reads.append((float(v), float(n), q, b["model_pC"][n]))
reads = np.array(reads)
ref = {v: reads[(reads[:, 0] == v) & (reads[:, 1] == 1e6), 2][0] for v in (0.1, 0.3, 0.5)}
late = reads[:, 1] >= 1e6
rd = reads[late]
ratio = rd[:, 2] / np.array([ref[v] for v in rd[:, 0]])

# (a) read-stress fatigue: pinned fraction loses switching and reversible response, both states together
def fat_ratio(th, v, n):
    lnd, lntf, kappa, nf = th
    d = lambda nn: 1.0 / (1.0 + np.exp(-lnd)) * (1.0 - np.exp(-(nn * T_ON * (v / TFE_M / 1e8) ** nf / np.exp(lntf)) ** kappa))
    return (1.0 - d(n)) / (1.0 - d(1e6))


fit_rf = least_squares(lambda th: fat_ratio(th, rd[:, 0], rd[:, 1]) - ratio, [-1.5, np.log(1e3), 0.3, 1.0],
                       bounds=([-8, -30, 0.05, 0], [3, 60, 3, 10]), x_scale=[0.5, 1, 0.05, 0.2])
fit_rf4 = least_squares(lambda th: fat_ratio([th[0], th[1], th[2], 4.0], rd[:, 0], rd[:, 1]) - ratio,
                        [-1.5, np.log(1e3), 0.3], bounds=([-8, -30, 0.05], [3, 60, 3]), x_scale=[0.5, 1, 0.05])

# (b) imprint during the read sequence: each stored state moves its C-V branches away from the read voltage
q1 = nm.generic_card(tfe=TFE_M, ps=0.165, c_d=cvfit[0] * nm.EPS0 / TFE_M, dc=cvfit[1] * nm.EPS0 / TFE_M,
                     vs=cvfit[2] * 1e8 * TFE_M, vw=cvfit[3] * 1e8 * TFE_M, vi=0.87e8 * TFE_M, vimp=0.45e8 * TFE_M)


def cmw_rev(v, dv):
    pp = dict(q1, imp_v=dv)
    qp = nm.term2_aged(pp, -v, 1.0, -v, q1["t0"], 1.0)
    qm = nm.term2_aged(pp, -v, -1.0, -v, q1["t0"], -1.0)
    return abs(qp - qm)


def imp_ratio(th, v, n):
    vi, lnt1 = th
    p = nm.generic_card(imp_v=1.0, imp_t1=float(np.exp(lnt1)), imp_t2=1e9)
    dv = lambda nn: vi * nm.imprint_response(p, nn * PERIOD_R)
    return np.array([cmw_rev(a, dv(b)) / cmw_rev(a, dv(1e6)) for a, b in zip(np.atleast_1d(v), np.atleast_1d(n))])


fit_ri = least_squares(lambda th: imp_ratio(th, rd[:, 0], rd[:, 1]) - ratio, [3.0, np.log(100.0)],
                       bounds=([0.0, -10], [40, 25]), x_scale=[0.5, 1])
p_ri = nm.generic_card(imp_v=fit_ri.x[0], imp_t1=float(np.exp(fit_ri.x[1])), imp_t2=1e9)
OUT["read_endurance_mukherjee2023"] = {
    "approach": "late drop relative to each curve's value at 1e6 reads (7 ratio points at 1e8, 1e10, 1e11 reads plus "
                "the 1e6 anchors); read period 1.5 us, read pulse 1 us (as in literature_comparison.py)",
    "data drop 1e6 -> last point": {f"-{v} V": float(ratio[(rd[:, 0] == v)][-1]) for v in (0.1, 0.3, 0.5)},
    "(a) read-stress fatigue, 4 parameters (D_max, t_F, kappa_F, n_F)": {
        "fat_d": float(1 / (1 + np.exp(-fit_rf.x[0]))), "fat_tf [s at 1 MV/cm]": float(np.exp(fit_rf.x[1])),
        "fat_kf": float(fit_rf.x[2]), "fat_nf": float(fit_rf.x[3]), "n_F * kappa_F": float(fit_rf.x[2] * fit_rf.x[3]),
        "RMS of the ratio": rms(fit_rf.fun)},
    "(a') same with n_F = 4 fixed (field-driven injection)": {"RMS of the ratio": rms(fit_rf4.fun)},
    "(b) imprint growing during the sequence, 2 parameters (V_I, tau_I1; tau_I2 = 1e9 s)": {
        "imp_v [V]": float(fit_ri.x[0]), "imp_t1 [s]": float(np.exp(fit_ri.x[1])),
        "V0 per e-fold [V]": float(fit_ri.x[0] / np.log(1e9 / np.exp(fit_ri.x[1]))),
        "shift at 1e11 reads (1.5e5 s) [V]": float(fit_ri.x[0] * nm.imprint_response(p_ri, 1e11 * PERIOD_R)),
        "RMS of the ratio": rms(fit_ri.fun)},
}
fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
nn = np.logspace(6, 11, 40)
for k, v in enumerate((0.1, 0.3, 0.5)):
    m = reads[:, 0] == v
    ax.semilogx(reads[m, 1], reads[m, 2] / ref[v], "o", color=f"C{k}", mec="k", label=f"data, -{v} V")
    ax.semilogx(nn, fat_ratio(fit_rf.x, v, nn), "-", color=f"C{k}")
    ax.semilogx(nn, imp_ratio(fit_ri.x, np.full_like(nn, v), nn), "--", color=f"C{k}")
    ax.semilogx(nn, fat_ratio([fit_rf4.x[0], fit_rf4.x[1], fit_rf4.x[2], 4.0], v, nn), ":", color=f"C{k}")
ax.plot([], [], "k-", label=f"read-stress pinning, n_F = {fit_rf.x[3]:.2f} (RMS {rms(fit_rf.fun):.3f})")
ax.plot([], [], "k:", label=f"read-stress pinning, n_F = 4 (RMS {rms(fit_rf4.fun):.3f})")
ax.plot([], [], "k--", label=f"imprint during the sequence (RMS {rms(fit_ri.fun):.3f})")
ax.set(xlabel="Number of reads", ylabel=r"CMW$_Q$ / CMW$_Q$(10$^6$ reads)", ylim=(0.6, 1.4),
       title="Read endurance after 1e8 reads, Mukherjee IEDM 2023 Fig. 7 (digitized)")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
fig.savefig(OUTDIR / "fi_read_endurance.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------- 6. imprint vs time and temperature
rb = rows("buragohain2019_acsami_fig2c_imprint.csv")
d_bu = {s: np.array([(float(r["t_delay_s"]), float(r["Vshift_V"])) for r in rb if r["set_pulse"] == s])
        for s in ("-6V_5ms", "+6V_5ms")}
T_SET = 5e-3


def bu_shift(th, t, sign, tau2=1e9):
    vb, vi, lnt1 = th
    p = nm.generic_card(imp_v=vi, imp_t1=float(np.exp(lnt1)), imp_t2=tau2)
    return vb + sign * vi * nm.imprint_response(p, t + T_SET)


def bu_resid(th, tau2=1e9):
    return np.concatenate([bu_shift(th, d_bu[s][:, 0], sg, tau2) - d_bu[s][:, 1]
                           for s, sg in (("-6V_5ms", 1.0), ("+6V_5ms", -1.0))])


fit_bu = least_squares(bu_resid, [0.05, 4.0, np.log(0.1)], bounds=([-0.5, 0.01, -25], [0.5, 50, 15]))
v0_model = fit_bu.x[1] / np.log(1e9 / np.exp(fit_bu.x[2]))
bu_tau = {}
for tau2 in (1e6, 1e12):
    f2 = least_squares(lambda th: bu_resid(th, tau2), fit_bu.x, bounds=([-0.5, 0.01, -25], [0.5, 80, 15]))
    bu_tau[f"{tau2:g}"] = {"RMS [V]": rms(f2.fun), "imp_v [V]": float(f2.x[1]), "imp_t1 [s]": float(np.exp(f2.x[2])),
                           "V0 per e-fold [V]": float(f2.x[1] / np.log(tau2 / np.exp(f2.x[2])))}


def authors(t, vb, v0, t0, sign):
    return vb + sign * v0 * np.log1p(t / t0)


fa = least_squares(lambda th: np.concatenate([authors(d_bu[s][:, 0], th[0], th[1], np.exp(th[2]), sg) - d_bu[s][:, 1]
                                              for s, sg in (("-6V_5ms", 1.0), ("+6V_5ms", -1.0))]),
                   [0.05, 0.13, np.log(0.1)])
OUT["imprint_buragohain2019"] = {
    "data": "Fig. 2c, V_shift vs delay after a 5 ms, +-6 V set pulse, both polarities (18 points, +-0.02 V)",
    "model": "V_b + -V_I sigma_I(t + 5 ms), sigma_I from 12 log-spaced states, imp_t2 = 1e9 s (beyond the data)",
    "fit (V_b, imp_v, imp_t1), 3 parameters": {"V_b [V]": float(fit_bu.x[0]), "imp_v [V]": float(fit_bu.x[1]),
                                               "imp_t1 [s]": float(np.exp(fit_bu.x[2])),
                                               "V0 = imp_v / ln(t2/t1) [V]": float(v0_model), "RMS [V]": rms(fit_bu.fun)},
    "sensitivity to imp_t2": bu_tau,
    "authors' form V_b + -V0 ln(1 + t/tau0), 3 parameters": {"V0 [V]": float(fa.x[1]), "tau0 [s]": float(np.exp(fa.x[2])),
                                                           "RMS [V]": rms(fa.fun)},
}

rf = rows("fengler2018_jap_fig3a_imprint_T.csv")
d_fe = np.array([(float(r["T_C"]) + 273.15, float(r["t_s"]), float(r["dV_shift_V"])) for r in rf])


def fe_shift(th, temp, t):
    vi, lnt1, ea, lnref = th
    p = nm.generic_card(imp_v=vi, imp_t1=float(np.exp(lnt1)), imp_t2=1e9, imp_ea=ea)
    out = np.empty(len(t))
    for k, (tk, tt) in enumerate(zip(temp, t)):
        out[k] = -vi * (nm.imprint_response(p, tt, tk) - nm.imprint_response(p, np.exp(lnref), tk))
    return out


def fe_resid(th):
    return fe_shift(th, d_fe[:, 0], d_fe[:, 1]) - d_fe[:, 2]


fit_fe = least_squares(fe_resid, [1.0, np.log(10.0), 0.3, np.log(1.0)], bounds=([0.01, -20, 0.0, -10], [20, 20, 2.0, 4]),
                       x_scale=[0.1, 1, 0.05, 1])
fit_fe0 = least_squares(lambda th: fe_resid([th[0], th[1], 0.0, th[2]]), [1.0, np.log(10.0), np.log(1.0)],
                        bounds=([0.01, -20, -10], [20, 20, 4]))
OUT["imprint_temperature_fengler2018"] = {
    "data": "Fig. 3a, imprint shift relative to the loop right after the reset cycling, 1-60 min, 25-125 C (19 points, "
            "+-0.002 V)",
    "model": "-V_I [sigma_I(t) - sigma_I(t_ref)], 12 states, imp_t2 = 1e9 s, Arrhenius time constants",
    "fit (imp_v, imp_t1, imp_ea, t_ref), 4 parameters": {
        "imp_v [V]": float(fit_fe.x[0]), "imp_t1 at 25 C [s]": float(np.exp(fit_fe.x[1])), "imp_ea [eV]": float(fit_fe.x[2]),
        "t_ref [s]": float(np.exp(fit_fe.x[3])), "V0 per e-fold [V]": float(fit_fe.x[0] / np.log(1e9 / np.exp(fit_fe.x[1]))),
        "RMS [V]": rms(fit_fe.fun)},
    "same with imp_ea = 0 (3 parameters)": {"RMS [V]": rms(fit_fe0.fun)},
    "paper": "activation energy of the 10-min shift 0.099 +- 0.010 eV (0.079 without 25 C)",
}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
tt = np.logspace(-5.5, 5.2, 200)
for s, sg, c in (("-6V_5ms", 1.0, "k"), ("+6V_5ms", -1.0, "C3")):
    ax[0].semilogx(d_bu[s][:, 0], d_bu[s][:, 1], "s" if sg > 0 else "o", color=c, mec="k",
                   label=f"data, {s.replace('_', ' ')} set")
    ax[0].semilogx(tt, bu_shift(fit_bu.x, tt, sg), "-", color=c)
    ax[0].semilogx(tt, authors(tt, fa.x[0], fa.x[1], np.exp(fa.x[2]), sg), ":", color="0.5")
ax[0].plot([], [], "k-", label=f"model: 12 log-spaced trap states, RMS {rms(fit_bu.fun)*1e3:.0f} mV")
ax[0].plot([], [], ":", color="0.5", label=f"$V_0\\ln(1+t/\\tau_0)$, RMS {rms(fa.fun)*1e3:.0f} mV")
ax[0].set(xlabel="Delay after the set pulse (s)", ylabel=r"$V_{shift}$ (V)",
          title="Imprint vs time, La:HfO$_2$ 20 nm (Buragohain 2019, Fig. 2c)")
ax[0].legend(fontsize=7.5, frameon=False, loc="upper left")
tl = np.logspace(1.5, 3.7, 50)
for k, tc in enumerate(TEMPS):
    m = np.isclose(d_fe[:, 0], tc + 273.15)
    ax[1].semilogx(d_fe[m, 1] / 60, d_fe[m, 2], "o", color=f"C{k}", mec="k", label=f"{tc} C")
    ax[1].semilogx(tl / 60, fe_shift(fit_fe.x, np.full_like(tl, tc + 273.15), tl), "-", color=f"C{k}")
ax[1].plot([], [], "k-", label=f"model, $E_I$ = {fit_fe.x[2]:.2f} eV, RMS {rms(fit_fe.fun)*1e3:.1f} mV")
ax[1].set(xlabel="Delay (min)", ylabel="Imprint shift (V)",
          title="Imprint vs temperature, HZO 10 nm (Fengler 2018, Fig. 3a)")
ax[1].legend(fontsize=7.5, frameon=False, ncol=2)
fig.savefig(OUTDIR / "fi_imprint.png", dpi=150)
plt.close(fig)
print(f"[{time.time()-T0:.0f} s] 4-6 done")

# ------------------------------------------------------------- figure: Pesic, Park, Lenox
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
nn = np.unique(np.concatenate([[0], np.logspace(0, 6.3, 40)]))
ax[0].semilogx(np.maximum(d_pe[:, 0], 0.1), d_pe[:, 1], "o", mec="k", label="data (pristine plotted at 0.1)")
ax[0].semilogx(np.maximum(nn, 0.1), pe_model(fit_pe.x, nn), "-", label=f"model, 7 parameters, RMS {rms(fit_pe.fun):.2f}")
ax[0].set(xlabel="Cycles (4 V, 10 kHz)", ylabel=r"$2P_r$ ($\mu$C/cm$^2$)",
          title="Sr:HfO$_2$ 10 nm (Pesic 2016, Fig. 1a)")
ax[0].legend(fontsize=8, frameon=False)
nn = np.unique(np.concatenate([[0], np.logspace(0, 9.2, 40)]))
for k, (fld, d) in enumerate(d_pk.items()):
    nk = nn if fld < 2.9 else nn[nn <= 1.5e7]   # the 3.0 MV/cm device broke down at 1.5e7 cycles
    ax[1].semilogx(np.maximum(d[:, 0], 0.1), d[:, 1], "o", color=f"C{k}", mec="k", ms=4, label=f"data, {fld} MV/cm")
    ax[1].semilogx(np.maximum(nk, 0.1), pk_model(fit_pk1.x, fit_pk1.x[7], fld, nk), "-", color=f"C{k}")
    ax[1].semilogx(np.maximum(nk, 0.1), pk_model(fit_pk0.x, 0.0, fld, nk), ":", color=f"C{k}")
ax[1].plot([], [], "k-", label=f"pinning + Ec increase, RMS {rms(fit_pk1.fun):.2f}")
ax[1].plot([], [], "k:", label=f"pinning only, RMS {rms(fit_pk0.fun):.2f}")
ax[1].axvline(1.5e7, color="C1", lw=0.8, ls="--")
ax[1].text(1.7e7, 29.6, "3.0 MV/cm\nbreakdown", fontsize=7, color="C1")
ax[1].set(xlabel="Cycles (100 kHz, read at the cycling field)", ylabel=r"$2P_r$ ($\mu$C/cm$^2$)",
          title="TDMA-HZO, two fields (Park 2020, Fig. 5e)")
ax[1].legend(fontsize=8, frameon=False, loc="lower left")
for k, w in enumerate((2, 5, 10, 20, 50)):
    m = d_le[:, 1] == w
    ax[2].semilogx(d_le[m, 0], d_le[m, 2], "o", color=f"C{k}", mec="k", label=f"{w} us")
    n4 = np.logspace(3, 6, 30)
    ax[2].semilogx(n4, lenox_law((n4, np.full_like(n4, w)), *popt), "-", color=f"C{k}")
ax[2].set(xlabel="Cycles (+-2 MV/cm, 10 kHz)", ylabel="Pr / Pr(pristine)",
          title=f"Pulse width (Lenox 2023): driver N w$^m$, m = {popt[3]:.1f}")
ax[2].legend(fontsize=8, frameon=False, title="pulse width", title_fontsize=8)
fig.savefig(OUTDIR / "fi_wakeup_fatigue.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------- 7. cycle-averaged vs direct
pc = nm.generic_card(wk_fp=0.3, wk_vsp=0.3, wk_nw=4.0, fat_d=0.3, fat_tf=1e-3, fat_kf=0.7, fat_ke=0.2, imp_v=0.3,
                     imp_t1=1e-6, imp_t2=1e-2)
q, ncyc = 2.5e-6, 12
t, v = [0.0], [0.0]
for a in [3.0, -3.0] * ncyc + [0.0]:
    t.append(t[-1] + abs(a - v[-1]) / 3.0 * q)
    v.append(a)
vf, br = nm.pwl(t, v)
rd = nm.simulate(pc, vf, t[-1], breaks=br, rtol=1e-7, atol=1e-10, max_step=q / 100)
ca = nm.cycle_history(pc, [0, q, 3 * q, 4 * q], [0, 3, -3, 0], np.arange(ncyc), nd=16, sub=400)
ends = np.arange(ncyc) * 4 * q
dpd = [pc["ps"] * (1 - rd["f"][(rd["t"] >= e) & (rd["t"] <= e + 4 * q)].mean())
       * np.ptp(rd["x"][(rd["t"] >= e) & (rd["t"] <= e + 4 * q)]) / UC for e in ends]
OUT["cycle_average_check"] = {k: float(np.max(np.abs(np.interp(ends, rd["t"], rd[k]) - ca[k]))) for k in ("W", "D", "s_i")}
OUT["cycle_average_check"]["max |dP| difference, periods >= 1 [uC/cm2]"] = float(np.max(np.abs(np.array(dpd)[1:] - ca["dP"][1:] / UC)))
t1 = time.time()
long = nm.cycle_history(pc, [0, q, 3 * q, 4 * q], [0, 3, -3, 0], [1e12], nd=16, sub=100)
OUT["cycle_average_check"]["seconds for 1e12 cycles"] = round(time.time() - t1, 2)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
ax[0].plot(rd["t"] / (4 * q), rd["x_t"], color="0.6", lw=0.6, label="direct: total state $x_t$")
for k in ("W", "D", "s_i"):
    ax[0].plot(rd["t"] / (4 * q), rd[k], label=f"direct: {k}")
    ax[0].plot(np.arange(ncyc), ca[k], "o", mfc="none", color="k")
ax[0].plot([], [], "ko", mfc="none", label="cycle-averaged")
ax[0].set(xlabel="Cycles (+-3 V, 100 kHz)", title="Fast illustrative card: direct ODE vs cycle-averaged")
ax[0].legend(fontsize=7.5, frameon=False, ncol=2)
ax[1].plot(np.arange(ncyc), dpd, "-", label="direct")
ax[1].plot(np.arange(ncyc), ca["dP"] / UC, "o", mfc="none", color="k", label="cycle-averaged")
ax[1].set(xlabel="Cycle", ylabel=r"switched charge per half cycle ($\mu$C/cm$^2$)")
ax[1].legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "fi_cycle_average_check.png", dpi=150)
plt.close(fig)

(OUTDIR / "fatigue_imprint_comparison.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
print(json.dumps(OUT, indent=1, default=float))
print(f"total {time.time()-T0:.0f} s")
