"""Why does the model over-predict the first reads of Mukherjee IEDM 2023, Fig. 7?

Research study behind docs/research/tail-fit.md (and .zh-TW.md).  Writes
docs/research/tail_fit_*.png and docs/research/tail_fit.json; slow steps are
cached in build/tail_fit_cache.json (delete it to recompute everything,
about 15 min).

Data (all taken from papers; see the CSV headers in data/literature/):

* Alessandri 2018 / 2019 switching: curves regenerated from the published NLS fits.
* Mukherjee IEDM 2023: C-V points (Fig. 5a), the two switching-current peaks of
  the 50 us triangular P-V loop (+1.35 V, -0.54 V, read from Fig. 3d), read
  endurance CMW_Q(n) at -0.1, -0.3, -0.5 V (Fig. 7; points with n <= 1e8 are
  fitted, the drop after 1e8 is outside the model and left out).
* Lancaster 2022 back-switching (12 points).

Steps: (1) first-read decomposition, (2) loop calibration, (3) alternatives
(tail shape, kinetics, h rule, stored-state shift, slow reversible part),
(4) joint fits of C-V + read endurance, (5) identifiability (Fisher matrices
per dataset), (6) figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fe_models import nls_multidomain as nm
from fe_models import read_disturb as rd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "literature"
OUTDIR = ROOT / "docs" / "research"
CACHE = ROOT / "build" / "tail_fit_cache.json"
EPS0, MV, UC = nm.EPS0, 1e8, 1e-2
T9 = 9e-9
A_FE = 60e-6 * 80e-6
V_PEAK = 3.0 * MV * T9          # +-3 MV/cm sweep of the P-V loop
PERIOD = 50e-6                  # 50 us triangle
VC_POS, VC_NEG = 1.35, -0.54    # switching-current peaks of the loop (digitized)
T_OFF = 0.5e-6                  # off time between reads: not given in the paper (assumption)
CAL = dict(nd=128, nstep=3000)  # loop calibration resolution (peak moves < 1 mV from nd = 64 to 256)
SIG = dict(eps=1.0, q=0.5, frac=0.01, vc=0.02, ploss=0.1)   # noise scales used to weight residuals


# ------------------------------------------------------------- cache
def _load_cache():
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_CACHE = _load_cache()


def cached(tag, fn, *key):
    k = tag + ":" + hashlib.sha1(json.dumps(key, sort_keys=True, default=float).encode()).hexdigest()[:16]
    if k not in _CACHE:
        _CACHE[k] = fn()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(_CACHE, default=float), encoding="utf-8")
    return _CACHE[k]


# ------------------------------------------------------------- data
def read_csv(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def params(name):
    return {r["parameter"]: float(r["value"]) for r in read_csv(name)}


READS_ALL = {}
for r in read_csv("mukherjee2023_iedm_read_endurance.csv"):
    READS_ALL.setdefault(float(r["V_R"]), []).append((float(r["n_reads"]), float(r["CMW_Q_pC"])))
READS = {v: [(n, q) for n, q in d if n <= 1e8] for v, d in READS_ALL.items()}
VR = sorted(READS)                       # -0.5, -0.3, -0.1
CV = read_csv("mukherjee2023_iedm_cv.csv")
CV_PTS = []
for r in CV:
    for br, x in (("up", 1.0), ("down", -1.0)):
        if r["branch"] in (br, "both"):
            CV_PTS.append((float(r["E_MV_cm"]), x, float(r["eps_r"])))

s2 = params("alessandri2018_edl_nls_fit.csv")
s4 = params("alessandri2019_ted_nls_fit.csv")
SRC = {"A2019": dict(t_nm=s4["thickness"], tau=s4["tau_inf"], alpha=s4["alpha"], beta=s4["beta"],
                     g=(s4["g_a"], s4["g_b"], s4["g_p"], s4["g_q"]), tmax=1e-3),
       "A2018": dict(t_nm=s2["thickness"], tau=s2["tau_inf"], alpha=s2["alpha"], beta=s2["beta"],
                     g=(s2["gb2_a"], s2["Ea"] / s2["gb2_b"], s2["gb2_q"], s2["gb2_p"]), tmax=7.6e-3)}
VOLTS_SW = (0.8, 1.0, 1.2, 1.5, 2.0)


def published_fraction(a, b, p, q, tau, alpha, beta, e_ext, t):
    y = np.logspace(np.log10(b) - 1.5, np.log10(b) + 1.5, 6001)
    pdf = np.gradient(nm.gb2_cdf(y, a, b, p, q), y)
    tt = tau * np.exp(np.minimum((y / e_ext) ** alpha, 600.0))
    return float(np.trapezoid(pdf * (1.0 - np.exp(-((t / tt) ** beta))), y))


def sw_times(src):
    return np.logspace(np.log10(200e-9), np.log10(src["tmax"]), 14)


def sw_data(src):
    ts = sw_times(src)
    return np.array([[published_fraction(*src["g"], src["tau"], src["alpha"], src["beta"],
                                         v / (src["t_nm"] * 1e-7) * 1e-6, t) for t in ts] for v in VOLTS_SW])


def sw_model(p, src, nd=401):
    q = dict(p, tfe=src["t_nm"] * 1e-9, vimp=0.0)
    dom = nm.make_domains(q, nd)
    ts = sw_times(src)
    out = []
    for v in VOLTS_SW:
        rp, _ = nm.rates(q, dom, v, 0.0, q["t0"])
        out.append([float(dom.w @ (1.0 - np.exp(-((rp * t) ** q["beta"])))) for t in ts])
    return np.array(out)


# Dagum refits of the two published distributions (same procedure as examples/literature_comparison.py)
REFIT = {"A2019": [1.774926538483093, 8.684209794145604, 1.1358190641689732, 4.1097816869244],
         "A2018": [1.8586616097619353, 9.885967858896969, 8.810311904255387, 3.7294328528809495]}


def kinetics_card(name, base):
    b, a, pt, al = REFIT[name]
    src = SRC[name]
    return dict(base, emp=b * MV, emn=b * MV, ap=a, an=a, pt=pt, alpha=al, tau0=src["tau"], beta=src["beta"])


# ------------------------------------------------------------- Mukherjee device
def lorentz(d, e_inf, de, d0, w):
    return e_inf + de / (1.0 + ((d - d0) / w) ** 2)


def cv_only_fit():
    """The 4-parameter Lorentzian fit of examples/literature_comparison.py (crossing fixed at 0.87 MV/cm)."""
    pts = []
    for r in CV:
        e, br, eps = float(r["E_MV_cm"]), r["branch"], float(r["eps_r"])
        if br in ("up", "both"):
            pts.append((0.87 - e, eps))
        if br == "down":
            pts.append((e - 0.87, eps))
    pts = np.array(pts)
    sol = least_squares(lambda t: lorentz(pts[:, 0], *t) - pts[:, 1], [28, 15, 1.5, 1.0],
                        bounds=([5, 0, 0, 0.1], [60, 60, 6, 6]))
    return [float(x) for x in sol.x]


CVX = cv_only_fit()


def rev_card(base, cvx, vi_mvcm=0.87):
    e_inf, de, d0, w = cvx
    return dict(base, c_d=e_inf * EPS0 / T9, dc=de * EPS0 / T9, vs=d0 * MV * T9, vw=w * MV * T9, vi=vi_mvcm * MV * T9)


BASE = rev_card(nm.generic_card(tfe=T9, ps=0.165, vimp=0.45 * MV * T9, area=A_FE), CVX)


def eps_model(p, e, x):
    v = e * MV * p["tfe"]
    return (nm.q_rev(p, v + 1e-4, x) - nm.q_rev(p, v - 1e-4, x)) / 2e-4 * p["tfe"] / EPS0


def cv_resid(p):
    return np.array([eps_model(p, e, x) - y for e, x, y in CV_PTS])


def calibrate(p):
    """Scale b- and b+ so that the loop's switching-current peaks sit at -0.54 V and +1.35 V."""
    def run():
        q, sn = rd.calibrate_to_loop(p, VC_NEG, V_PEAK, PERIOD, -1, **CAL)
        q, sp = rd.calibrate_to_loop(q, VC_POS, V_PEAK, PERIOD, +1, **CAL)
        return [sn, sp]
    key = {k: p[k] for k in ("emn", "emp", "an", "ap", "pt", "alpha", "tau0", "beta", "vimp", "tau_h", "tfe")}
    sn, sp = cached("cal", run, key)
    return dict(p, emn=p["emn"] * sn, emp=p["emp"] * sp), (sn, sp)


def reads_model(p, delta=0.0, gate_centre=None, t_off=T_OFF, shift_branches=False, all_points=False):
    src = READS_ALL if all_points else READS
    return {v: rd.read_sequence(p, v, [n for n, _ in src[v]], t_off=t_off, stored_shift=delta,
                                gate_centre=gate_centre, shift_branches=shift_branches)["cmw"] * 1e12 for v in VR}


def reads_resid(p, **kw):
    m = reads_model(p, **kw)
    return np.concatenate([m[v] - np.array([q for _, q in READS[v]]) for v in VR])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


OUT = {"assumptions": {"t_off_s": T_OFF, "read pulse": "trapezoid 500/500/500 ns, charge of rise + plateau",
                       "loop": "50 us triangle, +-2.7 V, peaks of the switching current at -0.54 V and +1.35 V",
                       "fitted read points": "n <= 1e8 (18 points)", "noise scales": SIG}}

# ------------------------------------------------------------- (1)+(2) first read and calibration
P_S4, SC_S4 = calibrate(kinetics_card("A2019", BASE))
P_S2, SC_S2 = calibrate(kinetics_card("A2018", BASE))
OUT["calibration"] = {"A2019 kinetics: scale of b- and b+": SC_S4, "A2018 kinetics: scale of b- and b+": SC_S2,
                      "b- = b+ [MV/cm] (A2019 kinetics)": [P_S4["emn"] / MV, P_S4["emp"] / MV]}


def first_read(p, delta=0.0):
    out = {}
    for v in VR:
        r = rd.read_sequence(p, v, [1], stored_shift=delta)
        rev = abs(nm.q_rev(p, v, 1.0) - nm.q_rev(p, v, -1.0)) * p["area"] * 1e12
        out[v] = {"reversible_pC": rev, "switching_pC": float(abs(r["q_sw_plus"][0]) * 1e12),
                  "total_pC": float(r["cmw"][0] * 1e12), "switched_fraction": float(r["f_plus"][0]),
                  "data_pC": READS[v][0][1]}
    return out


# the existing comparison calibrated b with a constant-voltage approximation (switched_fraction at
# V_imp + 0.945 V for 12.5 us = 0.5); keep it for continuity with docs/literature_comparison.json
P_CONSTV = dict(kinetics_card("A2019", BASE), emn=1.775e8 * 0.7972750015027764, emp=1.775e8 * 0.7972750015027764)
OUT["first_read"] = {"A2019, loop-calibrated": first_read(P_S4), "A2018, loop-calibrated": first_read(P_S2),
                     "A2019, constant-voltage calibration of literature_comparison.py": first_read(P_CONSTV)}

# ------------------------------------------------------------- (3) alternatives with delta = 0
SHAPES = [(8.684, 1.1358), (16, 1.1358), (32, 1.1358), (64, 1.1358), (8.684, 4), (16, 4), (32, 4), (8.684, 16),
          (16, 16), (64, 16)]


def with_shape(p, a, pt):
    """Change the Dagum shape of both directions, keeping each median."""
    q = dict(p)
    for b, k in (("emn", "an"), ("emp", "ap")):
        med = p[b] * (2 ** (1 / p["pt"]) - 1) ** (-1 / p[k])
        q[k] = a
        q[b] = med / (2 ** (1 / pt) - 1) ** (-1 / a)
    q["pt"] = pt
    return q


def alt_eval(p, **kw):
    m = reads_model(p, **kw)
    res = np.concatenate([m[v] - np.array([q for _, q in READS[v]]) for v in VR])
    return {"rms_pC": rms(res), "model_pC": {str(v): [float(x) for x in m[v]] for v in VR}}


shape_scan = []
for a, pt in SHAPES:
    pc, sc = calibrate(with_shape(kinetics_card("A2019", BASE), a, pt))
    r = cached("shape", lambda: alt_eval(pc), [a, pt, sc])
    shape_scan.append(dict(a=a, p=pt, tail_exponent=a * pt, scale=sc, **r))
KIN = [(4.11, 3.87e-7), (8, 3.87e-7), (4.11, 1e-9), (8, 1e-9), (16, 1e-9), (4.11, 1e-13), (8, 1e-13), (16, 1e-13),
       (30, 1e-13)]
kin_scan = []
for al, t0 in KIN:
    pc, sc = calibrate(dict(kinetics_card("A2019", BASE), alpha=al, tau0=t0))
    r = cached("kin", lambda: alt_eval(pc), [al, t0, sc])
    kin_scan.append(dict(alpha=al, tau0=t0, scale=sc, **r))
hrule = []
for gc, toff in ((None, 0.5e-6), (0.0, 0.5e-6), (0.0, 5e-6), (0.0, 50e-6), (None, 50e-6)):
    r = cached("hrule", lambda: alt_eval(P_S4, gate_centre=gc, t_off=toff), [gc, toff, SC_S4])
    hrule.append(dict(gate_centre="V_imp" if gc is None else gc, t_off=toff, **r))
OUT["alternatives_delta0"] = {"tail shape (A2019 kinetics)": shape_scan, "kinetics (A2019 shape)": kin_scan,
                              "h relaxation rule (A2019)": hrule}


# ------------------------------------------------------------- (4) joint fits
def joint_fit(pk, free_delta=True, free_rev=True, slow_rev=False, shift_branches=False):
    """Weighted least squares of C-V (sigma 1 eps) + read endurance n <= 1e8 (sigma 0.5 pC)."""
    names, th0, lb, ub = [], [], [], []
    if free_rev:
        names += ["eps_inf", "ln d_eps", "d0 [MV/cm]", "ln w [MV/cm]"]
        th0 += [CVX[0], np.log(CVX[1]), CVX[2], np.log(CVX[3])]
        lb += [5, 0.0, 0.1, np.log(0.1)]
        ub += [60, np.log(80), 5, np.log(6)]
    if free_delta:
        names.append("delta [V]")
        th0.append(0.4)
        lb.append(0.0)
        ub.append(1.0)
    if slow_rev:
        names += ["f_rev", "log10 tau_r [s]"]
        th0 += [0.8, -6.0]
        lb += [0.05, -8.0]
        ub += [1.0, -4.3]

    def unpack(th):
        th = list(th)
        q = dict(pk)
        if free_rev:
            q = rev_card(q, [th[0], np.exp(th[1]), th[2], np.exp(th[3])])
            th = th[4:]
        d = th.pop(0) if free_delta else 0.0
        if slow_rev:
            q["frev"], q["tau_r"] = th[0], 10 ** th[1]
        return q, d

    def resid(th):
        q, d = unpack(th)
        return np.concatenate([cv_resid(q) / SIG["eps"], reads_resid(q, delta=d, shift_branches=shift_branches) / SIG["q"]])

    def run():
        sol = least_squares(resid, th0, bounds=(lb, ub), diff_step=1e-3, max_nfev=150, x_scale="jac")
        return [float(x) for x in sol.x]

    th = cached("joint", run, [names, {k: pk[k] for k in ("emn", "emp", "an", "pt", "alpha", "tau0", "beta")},
                               shift_branches])
    q, d = unpack(th)
    m = reads_model(q, delta=d, shift_branches=shift_branches, all_points=True)
    cvr = cv_resid(q)
    rr = reads_resid(q, delta=d, shift_branches=shift_branches)
    return {"parameters": dict(zip(names, th)), "rms_eps": rms(cvr), "rms_pC": rms(rr), "delta_V": d,
            "rms_pC_by_voltage": {str(v): rms(rr[6 * i:6 * i + 6]) for i, v in enumerate(VR)},
            "model_pC_all_n": {str(v): [float(x) for x in m[v]] for v in VR},
            "cv_residuals": [float(x) for x in cvr], "card": q}


FITS = {
    "M0 A2019, C-V-only reversible part, no shift": dict(pk=P_S4, free_delta=False, free_rev=False),
    "M1 A2019, reversible part refit, no shift": dict(pk=P_S4, free_delta=False, free_rev=True),
    "M2 A2019, C-V-only reversible part, shift fitted": dict(pk=P_S4, free_delta=True, free_rev=False),
    "M3 A2019, reversible part refit + shift": dict(pk=P_S4, free_delta=True, free_rev=True),
    "M4 A2018, reversible part refit + shift": dict(pk=P_S2, free_delta=True, free_rev=True),
    "M5 A2019, M3 + slow reversible fraction": dict(pk=P_S4, free_delta=True, free_rev=True, slow_rev=True),
    "M6 A2019, M3 with the branches moving with the shift": dict(pk=P_S4, free_delta=True, free_rev=True,
                                                                shift_branches=True),
}
fits = {}
for name, kw in FITS.items():
    if not kw.get("free_delta") and not kw.get("free_rev"):
        q = kw["pk"]
        rr = reads_resid(q)
        m = reads_model(q, all_points=True)
        fits[name] = {"parameters": {}, "rms_eps": rms(cv_resid(q)), "rms_pC": rms(rr), "delta_V": 0.0,
                      "rms_pC_by_voltage": {str(v): rms(rr[6 * i:6 * i + 6]) for i, v in enumerate(VR)},
                      "model_pC_all_n": {str(v): [float(x) for x in m[v]] for v in VR},
                      "cv_residuals": [float(x) for x in cv_resid(q)], "card": q}
    else:
        fits[name] = joint_fit(**kw)
    print(f"{name}: RMS eps {fits[name]['rms_eps']:.2f}, RMS Q {fits[name]['rms_pC']:.2f} pC, "
          f"delta {fits[name]['delta_V']:.3f} V", flush=True)
BEST = "M3 A2019, reversible part refit + shift"
P_BEST = fits[BEST]["card"]
D_BEST = fits[BEST]["delta_V"]
OUT["joint_fits"] = {k: {kk: vv for kk, vv in v.items() if kk != "card"} for k, v in fits.items()}

# charge budget: the smallest switched charge a model with CMW_rev ~ (1 + x)/2 needs for the early decline
d05 = dict(READS[-0.5])
r_full = d05[1.0]
f12 = (d05[1.0] - d05[12.0]) / r_full
OUT["charge_budget_-0.5V"] = {
    "CMW drop from read 1 to read 12 [pC]": d05[1.0] - d05[12.0],
    "state loss needed if the drop is lost polarization (CMW_rev proportional to (1 + x)/2)": f12,
    "switched charge that loss would add, spread over reads 2-12 [pC per read]":
        f12 * 2 * BASE["ps"] * A_FE * 1e12 / 11}

# ------------------------------------------------------------- (5) identifiability
PRIOR = 3.0  # weak prior sigma on ln(theta): a marginal sigma near 3 means "not constrained"


def marginal(J):
    F = J.T @ J + np.eye(J.shape[1]) / PRIOR**2
    return np.sqrt(np.diag(np.linalg.inv(F)))


def jac(fun, th, names, step=0.02):
    r0 = fun(th)
    cols = []
    for i in range(len(th)):
        t = dict(th)
        t[names[i]] = th[names[i]] + step
        cols.append((fun(t) - r0) / step)
    return np.array(cols).T


ID_NAMES = ["tau0", "alpha", "beta", "b", "a", "p", "vimp", "delta", "tau_h", "t_off", "dc", "vs", "vw", "c_d",
            "vdep", "tau_s"]


def apply(p, th):
    """th = {name: ln(value / reference)} -> card (and delta, t_off)."""
    q = dict(p)
    f = {k: np.exp(v) for k, v in th.items()}
    q["tau0"] *= f.get("tau0", 1)
    q["alpha"] *= f.get("alpha", 1)
    q["beta"] = 1 + (q["beta"] - 1) * f.get("beta", 1)
    q["emn"] *= f.get("b", 1)
    q["emp"] *= f.get("b", 1)
    q["an"] *= f.get("a", 1)
    q["ap"] *= f.get("a", 1)
    q["pt"] *= f.get("p", 1)
    q["vimp"] *= f.get("vimp", 1)
    q["tau_h"] *= f.get("tau_h", 1)
    for k in ("dc", "vs", "vw", "c_d", "vdep", "tau_s"):
        q[k] *= f.get(k, 1)
    return q, D_BEST * f.get("delta", 1), T_OFF * f.get("t_off", 1)


def id_switching(name):
    pk = kinetics_card(name, nm.generic_card())
    data = sw_data(SRC[name])
    names = ["tau0", "alpha", "beta", "b", "a", "p"]

    def fun(th):
        q, _, _ = apply(pk, th)
        return ((sw_model(q, SRC[name]) - data) / SIG["frac"]).ravel()

    return names, jac(fun, {k: 0.0 for k in names}, names)


def id_cv():
    names = ["dc", "vs", "vw", "c_d"]
    return names, jac(lambda th: cv_resid(apply(P_BEST, th)[0]) / SIG["eps"], {k: 0.0 for k in names}, names)


def id_loop():
    names = ["tau0", "alpha", "beta", "b", "a", "p", "vimp", "tau_h"]

    def fun(th):
        q, _, _ = apply(P_BEST, th)
        return np.array([rd.loop_peak(q, V_PEAK, PERIOD, -1, total=False, **CAL) - VC_NEG,
                         rd.loop_peak(q, V_PEAK, PERIOD, +1, total=False, **CAL) - VC_POS]) / SIG["vc"]

    return names, jac(fun, {k: 0.0 for k in names}, names)


def id_reads():
    names = ["tau0", "alpha", "beta", "b", "a", "p", "vimp", "delta", "tau_h", "t_off", "dc", "vs", "vw"]

    def fun(th):
        q, d, toff = apply(P_BEST, th)
        return reads_resid(q, delta=d, t_off=toff) / SIG["q"]

    return names, jac(fun, {k: 0.0 for k in names}, names)


BS = {}
for r in read_csv("lancaster2022_backswitching.csv"):
    BS.setdefault(float(r["t_write_s"]), {})[float(r["t_delay_s"])] = float(r["P_loss_uC_cm2"])
DELAYS = (3e-7, 1e-5, 1e-3, 1.0)
G = nm.generic_card()
BASE11 = nm.generic_card(tfe=11e-9, vimp=0.0, c_d=G["c_d"] * 10 / 11, dc=G["dc"] * 10 / 11, vdep=1.0348362180687762,
                         tau_s=0.001739158219653844)


def ploss(q, t_write, nd=16):
    """Back-switched polarization after a 3.5 V write (same scheme as examples/literature_comparison.py)."""
    dom = nm.make_domains(q, nd)
    s, h, sig, prev = np.full(nd, -1.0), np.zeros(nd), 0.0, 0.0
    for t in np.logspace(np.log10(t_write * 1e-5), np.log10(t_write), 40):
        dt = t - prev
        s, h = nm.step(q, dom, s, h, 3.5, dt, None, sig)
        x = float(dom.w @ s)
        sig = x + (sig - x) * np.exp(-dt / q["tau_s"])
        prev = t
    x0, out, prev, want = float(dom.w @ s), {}, 0.0, list(DELAYS)
    for t in np.logspace(-9, 0, 160):
        dt = t - prev
        s, h = nm.step(q, dom, s, h, 0.0, dt, None, sig)
        x = float(dom.w @ s)
        sig = x + (sig - x) * np.exp(-dt / q["tau_s"])
        prev = t
        while want and t >= want[0] * (1 - 1e-9):
            out[want.pop(0)] = (x0 - x) * q["ps"] / UC
    return np.array([out[d] for d in DELAYS])


def id_lancaster():
    names = ["tau0", "alpha", "beta", "b", "a", "p", "tau_h", "vdep", "tau_s"]

    def fun(th):
        q, _, _ = apply(BASE11, th)
        return np.concatenate([ploss(q, tw) - np.array([BS[tw][d] for d in DELAYS]) for tw in sorted(BS)]) / SIG["ploss"]

    return names, jac(fun, {k: 0.0 for k in names}, names)


def identifiability():
    blocks = {"Alessandri 2019 switching": id_switching("A2019"), "Alessandri 2018 switching": id_switching("A2018"),
              "Mukherjee C-V": id_cv(), "Mukherjee loop peaks": id_loop(), "Mukherjee read endurance": id_reads(),
              "Lancaster back-switching": id_lancaster()}
    res = {}
    for k, (names, J) in blocks.items():
        res[k] = {"names": names, "J": J.tolist()}
    return res


ID = cached("ident", identifiability, [BEST, D_BEST, SIG, PRIOR])
ident = {}
for k, v in ID.items():
    J = np.array(v["J"])
    ident[k] = {n: {"sensitivity": float(np.linalg.norm(J[:, i])), "marginal_sigma_ln": float(m)}
                for i, (n, m) in enumerate(zip(v["names"], marginal(J)))}
def combine(keys):
    """Stack the Jacobians of several datasets on the union of their parameters."""
    union = sorted({n for k in keys for n in ID[k]["names"]}, key=ID_NAMES.index)
    rows = []
    for k in keys:
        J = np.array(ID[k]["J"])
        full = np.zeros((J.shape[0], len(union)))
        for i, n in enumerate(ID[k]["names"]):
            full[:, union.index(n)] = J[:, i]
        rows.append(full)
    J = np.vstack(rows)
    return union, J, {n: {"sensitivity": float(np.linalg.norm(J[:, i])), "marginal_sigma_ln": float(m)}
                      for i, (n, m) in enumerate(zip(union, marginal(J)))}


MUK = ("Mukherjee C-V", "Mukherjee loop peaks", "Mukherjee read endurance")
union, Jm, ident["Mukherjee all three"] = combine(MUK)
# the same with the kinetics (tau0, alpha, beta, Dagum shape) assumed to be those of the Alessandri 2019 device
_, _, ident["Mukherjee all three + Alessandri 2019 kinetics"] = combine(MUK + ("Alessandri 2019 switching",))
# the best-determined and worst-determined directions of the Mukherjee block
w, V = np.linalg.eigh(Jm.T @ Jm)
OUT["identifiability"] = {"prior_sigma_ln": PRIOR, "datasets": ident,
                          "Mukherjee eigen-directions (weakest first)": [
                              {"sigma": float(1 / np.sqrt(max(ev, 1e-30))),
                               "direction": {n: round(float(c), 3) for n, c in zip(union, V[:, j]) if abs(c) > 0.2}}
                              for j, ev in enumerate(w)]}


# profile of tau0: refit (b, a, p, alpha) of each published switching fit with tau0 held fixed
def tau0_profile(name):
    src = SRC[name]
    data = sw_data(src)
    out, x0 = [], None
    for tau0 in (src["tau"], 1e-7, 1e-8, 1e-9, 1e-11, 1e-13):
        base = nm.generic_card(tau0=tau0, beta=src["beta"])

        def unpack(th):
            b, a, pt, al = np.exp(th)
            return dict(base, emp=b * MV, emn=b * MV, ap=a, an=a, pt=pt, alpha=al)

        starts = ([x0] if x0 is not None else []) + [np.log([src["g"][1] * f, src["g"][0], 1.0, src["alpha"] * m])
                                                    for f in (1.0, 2.5) for m in (1.0, 2.0)]
        best = None
        for st in starts:
            sol = least_squares(lambda th: (sw_model(unpack(th), src) - data).ravel(), st, diff_step=1e-3, max_nfev=200)
            if best is None or sol.cost < best.cost:
                best = sol
        x0 = best.x
        b, a, pt, al = np.exp(best.x)
        out.append({"tau0": tau0, "b_MV_cm": b, "a": a, "p": pt, "alpha": al, "rms_fraction": rms(best.fun)})
    return out


OUT["tau0_profile"] = {k: cached("tau0prof", lambda: tau0_profile(k), [k]) for k in ("A2019", "A2018")}

# ------------------------------------------------------------- (6) figures
OUTDIR.mkdir(parents=True, exist_ok=True)
COL = {-0.1: "C2", -0.3: "C1", -0.5: "C3"}

# Fig. 1: first read decomposition + the field-axis picture
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
cases = [("A2019, constant-V\ncalibration (repo)", OUT["first_read"]["A2019, constant-voltage calibration of literature_comparison.py"]),
         ("A2019, loop\ncalibration", OUT["first_read"]["A2019, loop-calibrated"]),
         ("A2018, loop\ncalibration", OUT["first_read"]["A2018, loop-calibrated"])]
xs = np.arange(len(VR))
wbar = 0.2
for i, (lab, fr) in enumerate(cases):
    rev = [fr[v]["reversible_pC"] for v in VR]
    swc = [fr[v]["switching_pC"] for v in VR]
    a1.bar(xs + (i - 1) * wbar, rev, wbar, color="0.75", edgecolor="k", lw=0.5)
    a1.bar(xs + (i - 1) * wbar, swc, wbar, bottom=rev, color=f"C{i}", edgecolor="k", lw=0.5, label=lab)
a1.plot(xs + 1.6 * wbar, [READS[v][0][1] for v in VR], "kD", ms=7, label="data, read 1")
a1.set_yscale("log")
a1.set_xticks(xs, [f"{v} V" for v in VR])
a1.set_ylabel(r"CMW$_Q$ of the first read (pC)")
a1.set_title("First read: reversible part (grey) + switching part (colour)", fontsize=10)
a1.legend(fontsize=7.5, frameon=False, loc="upper right")
# field axis: fraction switched by one read vs read voltage, and the loop branch
vv = np.linspace(-0.05, -0.8, 31)
for lab, p, ls in (("A2019", P_S4, "-"), ("A2018", P_S2, "--")):
    f1 = [rd.read_sequence(p, v, [1], nd=128)["f_plus"][0] for v in vv]
    a2.semilogy(-vv, np.maximum(f1, 1e-7), ls, color="C0", label=f"one 500/500/500 ns read, {lab}")
    vb, xb, _ = rd.loop_branch(p, V_PEAK, PERIOD, s_start=1.0, nd=128, nstep=3000)
    m = (vb < 0) & (vb > -0.85)
    a2.semilogy(-vb[m], np.maximum(0.5 * (1 - xb[m]), 1e-7), ls, color="C3", label=f"50 us loop branch, {lab}")
    f1d = [rd.read_sequence(p, v, [1], nd=128, stored_shift=D_BEST)["f_plus"][0] for v in vv]
    a2.semilogy(-vv, np.maximum(f1d, 1e-7), ls, color="C2", label=f"one read, stored state +{D_BEST:.2f} V, {lab}")
a2.axvline(-VC_NEG, color="C3", lw=0.8, ls=":")
a2.text(-VC_NEG + 0.01, 2e-7, "loop peak\n-0.54 V", color="C3", fontsize=8)
for v in VR:
    a2.axvline(-v, color=COL[v], lw=0.8, alpha=0.6)
a2.axhspan(1e-7, 1e-3, color="0.9", zorder=0)
a2.text(0.06, 2e-4, "allowed by read 1 of Fig. 7", fontsize=8, color="0.3")
a2.set_ylim(1e-7, 1.2)
a2.set_xlim(0.05, 0.8)
a2.set_xlabel("|read voltage| (V), negative polarity")
a2.set_ylabel("fraction switched away from the + state")
a2.set_title("Same distribution, two time scales", fontsize=10)
a2.legend(fontsize=7, frameon=False, loc="lower right")
fig.savefig(OUTDIR / "tail_fit_first_read.png", dpi=150)
plt.close(fig)

# Fig. 2: alternatives with no shift
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
for ax, (title, scan, lab) in zip(axs, (
        ("Tail shape (A2019 kinetics)", shape_scan, lambda r: f"a={r['a']:.3g}, p={r['p']:.3g}"),
        ("Kinetics (A2019 shape)", kin_scan, lambda r: f"alpha={r['alpha']:g}, tau0={r['tau0']:.0e}"),
        ("h relaxation rule (A2019)", hrule, lambda r: f"gate {r['gate_centre']}, t_off={r['t_off']*1e6:g} us"))):
    best = sorted(scan, key=lambda r: r["rms_pC"])[:3]
    for v in VR:
        n = [x for x, _ in READS[v]]
        ax.semilogx(n, [q for _, q in READS[v]], "o", color=COL[v], mec="k", zorder=5)
    for k, r in enumerate(best):
        for v in VR:
            n = [x for x, _ in READS[v]]
            ax.semilogx(n, r["model_pC"][str(v)], ("-", "--", ":")[k], color=COL[v], lw=1.2)
        ax.plot([], [], ("-", "--", ":")[k], color="k", label=f"{lab(r)}: RMS {r['rms_pC']:.1f} pC")
    ax.set_ylim(0, 45)
    ax.set_title(title + " - three best", fontsize=10)
    ax.set_xlabel("Number of reads")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
axs[0].set_ylabel(r"CMW$_Q$ (pC)")
fig.savefig(OUTDIR / "tail_fit_alternatives.png", dpi=150)
plt.close(fig)

# Fig. 3: joint fits
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
ax = axs[0]
for v in VR:
    n = [x for x, _ in READS_ALL[v]]
    q = [y for _, y in READS_ALL[v]]
    ax.semilogx(n, q, "o", color=COL[v], mec="k", label=f"data {v} V", zorder=5)
    ax.semilogx(n[len(READS[v]):], q[len(READS[v]):], "o", color="white", mec=COL[v], zorder=6)
    for name, ls in ((BEST, "-"), ("M2 A2019, C-V-only reversible part, shift fitted", "--"), ("M0 A2019, C-V-only reversible part, no shift", ":")):
        ax.semilogx(n, fits[name]["model_pC_all_n"][str(v)], ls, color=COL[v], lw=1.3)
for name, ls in ((BEST, "-"), ("M2 A2019, C-V-only reversible part, shift fitted", "--"), ("M0 A2019, C-V-only reversible part, no shift", ":")):
    f = fits[name]
    ax.plot([], [], ls, color="k", label=f"{name[:2]}: shift {f['delta_V']:.2f} V, RMS {f['rms_pC']:.2f} pC / {f['rms_eps']:.1f} eps")
ax.set_ylim(0, 40)
ax.set_xlabel("Number of reads")
ax.set_ylabel(r"CMW$_Q$ (pC)")
ax.set_title("Joint fit of C-V + read endurance (open: not fitted)", fontsize=10)
ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=1)
ax = axs[1]
e = np.linspace(-3.0, 3.0, 241)
for name, ls in (("M0 A2019, C-V-only reversible part, no shift", ":"), (BEST, "-")):
    q = fits[name]["card"]
    ax.plot(e, [eps_model(q, x, 1.0) for x in e], ls, color="C0")
    ax.plot(e, [eps_model(q, x, -1.0) for x in e], ls, color="C3")
    ax.plot([], [], ls, color="k", label=("C-V-only fit" if name.startswith("M0") else "joint fit M3") +
            f": RMS {fits[name]['rms_eps']:.2f} eps")
for r in CV:
    ax.plot(float(r["E_MV_cm"]), float(r["eps_r"]), "o", color={"up": "C0", "down": "C3", "both": "k"}[r["branch"]], mec="k")
for v in VR:
    ax.axvline(v / T9 / MV, color=COL[v], lw=0.7, alpha=0.6)
ax.set_xlabel("Field (MV/cm)")
ax.set_ylabel(r"$\varepsilon_r$ (small signal)")
ax.set_title("C-V branches (+ state blue, - state red); vertical lines: read fields", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "tail_fit_joint.png", dpi=150)
plt.close(fig)

# Fig. 4: residuals of the best fit
fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
mb = fits[BEST]["model_pC_all_n"]
for v in VR:
    n = np.array([x for x, _ in READS_ALL[v]])
    q = np.array([y for _, y in READS_ALL[v]])
    ax.semilogx(n, q - np.array(mb[str(v)]), "o-", color=COL[v], mec="k", label=f"{v} V")
ax.axhline(0, color="k", lw=0.7)
ax.axvline(1e8, color="0.5", ls=":", lw=0.8)
ax.text(1.3e8, 0.5, "not fitted", fontsize=8, color="0.4")
ax.set_xlabel("Number of reads")
ax.set_ylabel("data - model (pC)")
ax.set_title("Residuals of M3 (A2019 kinetics, stored-state shift)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUTDIR / "tail_fit_residuals.png", dpi=150)
plt.close(fig)

# Fig. 5: identifiability map
cols = ["Alessandri 2019 switching", "Alessandri 2018 switching", "Mukherjee C-V", "Mukherjee loop peaks",
        "Mukherjee read endurance", "Mukherjee all three", "Mukherjee all three + Alessandri 2019 kinetics",
        "Lancaster back-switching"]
mat = np.full((len(ID_NAMES), len(cols)), np.nan)
for j, c in enumerate(cols):
    for n, d in ident[c].items():
        mat[ID_NAMES.index(n), j] = d["marginal_sigma_ln"]
fig, ax = plt.subplots(figsize=(9.6, 6.4), constrained_layout=True)
im = ax.imshow(np.log10(mat), cmap="viridis_r", vmin=-2.5, vmax=np.log10(PRIOR), aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if np.isfinite(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] < 0.05 else "black")
ax.set_xticks(range(len(cols)), [textwrap.fill(c, 13) for c in cols], fontsize=7.5)
ax.set_yticks(range(len(ID_NAMES)), ID_NAMES, fontsize=8)
ax.set_title(r"Marginal uncertainty of ln(parameter) per dataset (weak prior $\sigma$ = 3; blank: no effect)",
             fontsize=9.5)
fig.colorbar(im, ax=ax, label=r"log$_{10}$ $\sigma$(ln $\theta$)")
fig.savefig(OUTDIR / "tail_fit_identifiability.png", dpi=150)
plt.close(fig)

OUT["best"] = {"model": BEST, "shift_V": D_BEST, "shift_MV_cm": D_BEST / T9 / MV,
               "card": {k: P_BEST[k] for k in ("tfe", "ps", "emn", "emp", "an", "ap", "pt", "alpha", "beta", "tau0",
                                               "vimp", "c_d", "dc", "vs", "vw", "vi", "frev", "tau_r", "tau_h")}}
(OUTDIR / "tail_fit.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: OUT[k] for k in ("calibration", "first_read", "charge_budget_-0.5V")}, indent=1, default=float))
for k, v in OUT["joint_fits"].items():
    print(k, {kk: v[kk] for kk in ("rms_eps", "rms_pC", "delta_V", "parameters")})
for k, v in OUT["alternatives_delta0"].items():
    print(k, [(round(r["rms_pC"], 2), {kk: r[kk] for kk in r if kk not in ("model_pC", "rms_pC")}) for r in sorted(v, key=lambda r: r["rms_pC"])[:3]])

# ------------------------------------------------------------- (7) device card
CARD_NOTES = {
    "tfe": "La:HZO thickness of the stack (data/literature/mukherjee2023_iedm_cv.csv)",
    "ps": "remanent polarization used for this device in examples/literature_comparison.py (P-V loop, digitized, approx.)",
    "vimp": "centre of the loop's switching-current peaks +1.35 V and -0.54 V (digitized, approx.; "
            "examples/literature_comparison.py)",
    "ap": "Dagum refit of [Alessandri2019] (the generic card's distribution, not fitted to this device)",
    "an": "Dagum refit of [Alessandri2019] (not fitted to this device)",
    "pt": "Dagum refit of [Alessandri2019] (not fitted to this device); tail exponent a*p = 9.9",
    "alpha": "Dagum refit of [Alessandri2019] (not fitted to this device)",
    "emp": "FITTED: b+ scaled so that the + branch peak of the 50 us loop is at +1.35 V (switching current, "
           "examples/tail_fit_study.py)",
    "emn": "FITTED: b- scaled so that the - branch peak of the 50 us loop is at -0.54 V (same procedure)",
    "c_d": "FITTED: Lorentzian branch fit to the C-V of Fig. 5a [Mukherjee2023IEDM] (7 points, 4 parameters)",
    "dc": "FITTED: same C-V fit",
    "vs": "FITTED: same C-V fit",
    "vw": "FITTED: same C-V fit",
    "vi": "crossing point E_CROSS = 0.87 MV/cm of Fig. 5a (digitized)",
}


def write_card(p, path):
    lines = ["* Device card for fecap_nls: TiN/MoOx/La:HZO 9 nm/Nb2O5/TiN capacitor of Mukherjee et al.,",
             "* IEDM 2023 (doi:10.1109/IEDM45741.2023.10413879).  Generated by examples/tail_fit_study.py.",
             "* Fitted: the reversible branches to the C-V of Fig. 5a, and the activation-field scales b+ / b-",
             "* to the two switching-current peaks of the 50 us P-V loop.  Kinetics (tau0, alpha, beta) and the",
             "* Dagum shape are the Alessandri 2019 values of the generic card (not fitted to this device).",
             "* Not reproduced: the read endurance of Fig. 7 at -0.3 / -0.5 V.  With this card the first reads",
             "* switch far too much; the data need the stored state to be "
             f"{fits['M2 A2019, C-V-only reversible part, shift fitted']['delta_V']:.2f}-{D_BEST:.2f} V harder to "
             "switch than in the loop",
             "* (a stabilizing imprint; docs/research/tail-fit.md).  That shift is not part of this card.",
             f"* Instance parameters: area = {A_FE:g} m^2 (60 x 80 um^2), s0 = +1 or -1.",
             ".model fecap_mukherjee2023 fecap_nls"]
    g = nm.generic_card()
    for k, prm in nm.GENERIC.items():
        if k in ("area", "s0", "nd") or k not in p:
            continue
        if k in CARD_NOTES or abs(p[k] - g[k]) > 1e-12 * max(1.0, abs(g[k])):
            lines.append(f"+ {prm.va_name}={p[k]:.6g}")
            lines.append(f"* ^ [{prm.unit}] {prm.meaning}. Source: {CARD_NOTES.get(k, prm.source)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


write_card(P_S4, ROOT / "spice" / "params" / "mukherjee2023.inc")
