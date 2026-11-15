"""Read disturb and imprint dynamics, round 2 (docs/research/read-imprint-r2.md and .zh-TW.md).

Writes docs/research/rir2_*.png and docs/research/read_imprint_r2.json; slow steps are cached in
build/rir2_cache.json (delete it to recompute everything).

Questions (numbering as in the notes):

1. Can the constant stored-state shift Delta = 0.41-0.44 V of the tail-fit study be replaced by imprint
   dynamics (write-time part + log-time growth)?  How much shift can build up before the first read?
2. What is the 4.6 pC drop of CMW_Q in the first 12 reads at -0.5 V?
3. Why is the pulse CMW_Q linear in V_R while the 10 kHz C-V branches are curved?
4. Is the decline after 1e8 reads read-stress pinning or imprint growing during the reads?
5. Asymmetric branch shifts of the imprint (E17a, imp_kw).

Data (public papers only; see the CSV headers in data/literature/): Mukherjee IEDM 2023 Fig. 5a (C-V),
Fig. 6f (retention), Fig. 7b-e (read endurance, |Q+| and |Q-| separately); Buragohain 2019 Figs. 2a-c;
Fengler 2018 Fig. 3a; Alcala 2022 Fig. 5b.  The device card and the stored-state shift of the first round
are read from docs/research/tail_fit.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
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
CACHE = ROOT / "build" / "rir2_cache.json"
EPS0, MV = nm.EPS0, 1e8
T9 = 9e-9
A_FE = 60e-6 * 80e-6
T_OFF = 0.5e-6                    # off time between reads (not given in the paper; as in the tail-fit study)
T_LOOP = 21e-6                    # age of the + state at the - branch of the 50 us loop (+1.35 V -> +2.7 V -> -0.54 V)
VR = (-0.5, -0.3, -0.1)
COL = {-0.1: "C2", -0.3: "C1", -0.5: "C3"}
OUT: dict = {}


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


def read_csv(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# ------------------------------------------------------------- data
READS = {}
for r in read_csv("mukherjee2023_iedm_read_endurance.csv"):
    READS.setdefault(float(r["V_R"]), []).append((float(r["n_reads"]), float(r["CMW_Q_pC"])))
QPM = {}
for r in read_csv("mukherjee2023_iedm_fig7_qplus_qminus.csv"):
    QPM.setdefault(float(r["V_R"]), []).append((float(r["n_reads"]), float(r["Q_plus_pC"]), float(r["Q_minus_pC"])))
RET = np.array([[float(r["t_s"]), float(r["Q_plus_pC"]), float(r["Q_minus_pC"])]
                for r in read_csv("mukherjee2023_iedm_fig6f_retention.csv")])
CV = read_csv("mukherjee2023_iedm_cv.csv")
BUR_VC = read_csv("buragohain2019_acsami_fig2ab_branch_vc.csv")
BUR_SHIFT = read_csv("buragohain2019_acsami_fig2c_imprint.csv")

TF = json.loads((OUTDIR / "tail_fit.json").read_text(encoding="utf-8"))
P_M3 = nm.generic_card()
P_M3.update(TF["best"]["card"])
P_M3["area"] = A_FE
DELTA_M3 = TF["best"]["shift_V"]
DELTA_M2 = TF["joint_fits"]["M2 A2019, C-V-only reversible part, shift fitted"]["delta_V"]


def lorentz(d, e_inf, de, d0, w):
    return e_inf + de / (1.0 + ((d - d0) / w) ** 2)


def cv_only_fit():
    """The 4-parameter Lorentzian fit of the C-V (crossing fixed at 0.87 MV/cm), as in the tail-fit study."""
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


def rev_card(base, cvx, vi_mvcm=0.87):
    e_inf, de, d0, w = cvx
    return dict(base, c_d=e_inf * EPS0 / T9, dc=de * EPS0 / T9, vs=d0 * MV * T9, vw=w * MV * T9, vi=vi_mvcm * MV * T9)


CVX = cv_only_fit()
P_M2 = rev_card(P_M3, CVX)          # same switching card, C-V-only reversible branches
EPS_UNIT = A_FE * EPS0 / T9 * 1e12  # pC per volt per unit of eps_r (4.72 pC/V)


def eps_model(p, e, x):
    v = e * MV * p["tfe"]
    return (nm.q_rev(p, v + 1e-4, x) - nm.q_rev(p, v - 1e-4, x)) / 2e-4 * p["tfe"] / EPS0


def q_rev_state(p, v, s0, shift=0.0):
    """|reversible read charge| [pC] of state s0 at V, with that state's branches moved away by ``shift`` [V]."""
    q = dict(p, vi=p["vi"] - s0 * shift / p["rho"])
    return abs(nm.q_rev(q, v, s0)) * p["area"] * 1e12


# =================================================================== Q3: C-V curvature vs pulse linearity
def q3():
    out = {}
    ratio_data, ratio_cv, ratio_m3, per_v = {}, {}, {}, {}
    for v in VR:
        rows = {n: (a, b) for n, a, b in QPM[v]}
        ref = rows[12.0] if v == -0.5 else rows[1.0]     # -0.5 V: read 1 contains switching charge (Q2)
        ratio_data[v] = ref[0] / ref[1]
        ratio_cv[v] = q_rev_state(P_M2, v, 1) / q_rev_state(P_M2, v, -1)
        ratio_m3[v] = q_rev_state(P_M3, v, 1) / q_rev_state(P_M3, v, -1)
        per_v[v] = {"Q+/|V_R| data [pC/V]": ref[0] / abs(v), "Q-/|V_R| data [pC/V]": ref[1] / abs(v),
                    "secant eps+ (data, nominal V_R)": ref[0] / abs(v) / EPS_UNIT,
                    "secant eps- (data, nominal V_R)": ref[1] / abs(v) / EPS_UNIT,
                    "secant eps+ (C-V fit)": q_rev_state(P_M2, v, 1) / abs(v) / EPS_UNIT,
                    "secant eps- (C-V fit)": q_rev_state(P_M2, v, -1) / abs(v) / EPS_UNIT}
    out["Q+/Q- data (read 1; -0.5 V: read 12)"] = ratio_data
    out["Q+/Q- from the C-V branches (secant)"] = ratio_cv
    out["Q+/Q- from the M3 branches (refit to reads)"] = ratio_m3
    out["eps+(0)/eps-(0) C-V (Fig. 5d means 39.9 / 31.2)"] = 39.9 / 31.2
    out["per volt"] = per_v
    # (b) "slow configuration" form q2 = g(y) + g'(y)(V - y): after many reads y -> duty * V_R
    duty = 1.0e-6 / (1.5e-6 + T_OFF)
    pred = {}
    for v in VR:
        y = duty * v
        ep = eps_model(P_M2, y / T9 / MV, 1.0)
        em = eps_model(P_M2, y / T9 / MV, -1.0)
        e0p, e0m = eps_model(P_M2, 0.0, 1.0), eps_model(P_M2, 0.0, -1.0)
        pred[v] = {"Q+ change [%]": 100 * (ep / e0p - 1), "Q- change [%]": 100 * (em / e0m - 1)}
    out["slow-configuration form: change of the read charge once y follows the read train"] = pred
    # C-V time scale: switching of the stabilized + state during a slow DC down-sweep (13 ms per point)
    out["C-V sweep switching onset"] = cv_sweep_switching()
    return out


def cv_sweep_switching(delta=None, dwell=13e-3, step_mvcm=0.125):
    """Switched fraction of the stored + state along a DC staircase from 0 toward -1.5 MV/cm (closed-form steps).

    The + state is stabilized by ``delta`` (the stored-state shift of the tail-fit M2 card): in the C-V sweep it is
    about as old as in the reads (tenths of a second).  Returns the fields where 1 %, 10 % and 50 % have switched,
    together with the same for one 1.5 us read pulse train of 1 read (for comparison).
    """
    delta = DELTA_M2 if delta is None else delta
    p = dict(P_M2, vimp=P_M2["vimp"] - delta)
    dom = nm.make_domains(p, 256)
    s, h = np.ones(256), np.zeros(256)
    es = np.arange(0.0, -1.5001, -step_mvcm)
    f = []
    for e in es:
        s, h = nm.step(p, dom, s, h, e * MV * T9, dwell)
        f.append(0.5 * (1.0 - float(dom.w @ s)))
    f = np.array(f)

    def cross(level):
        i = int(np.argmax(f >= level))
        return float(np.interp(level, f[i - 1:i + 1], es[i - 1:i + 1])) if f.max() >= level and i > 0 else None

    return {"fields_MV_cm": es.tolist(), "switched": f.tolist(), "E at 1 %": cross(0.01), "E at 10 %": cross(0.1),
            "E at 50 %": cross(0.5), "dwell_s": dwell, "delta_V": delta}


OUT["Q3"] = q3()


# =================================================================== Q4: retention -> imprint rate of this device
def q4_retention():
    """Branch shift implied by the Fig. 6f retention decline (C-V branches move away from the read voltage)."""
    t, qp, qm = RET[:, 0], RET[:, 1], RET[:, 2]
    lt = np.log10(t)
    sp, ip = np.polyfit(lt, qp, 1)
    sm, im_ = np.polyfit(lt, qm, 1)
    k_v = 0.9  # read amplitude / nominal (Fig. 6c: -100 mV pulses reach about -90 mV; Fig. 6d uses about 83 mV)
    out = {"slope Q+ [pC/decade]": sp, "slope Q- [pC/decade]": sm}
    for name, p in (("C-V branches", P_M2), ("M3 branches", P_M3)):
        v = -0.1 * k_v
        dqp = (q_rev_state(p, v, 1, 0.05) - q_rev_state(p, v, 1, -0.05)) / 0.1
        dqm = (q_rev_state(p, v, -1, 0.05) - q_rev_state(p, v, -1, -0.05)) / 0.1
        out[name] = {"dQ+/dshift [pC/V]": dqp, "dQ-/dshift [pC/V]": dqm,
                     "escape-branch shift rate from Q+ [V/decade]": sp / dqp,
                     "escape-branch shift rate from Q- [V/decade]": sm / dqm,
                     "V0_esc from Q+ [V/e-fold]": sp / dqp / np.log(10),
                     "V0_esc from Q- [V/e-fold]": sm / dqm / np.log(10)}
    return out


OUT["Q4 retention"] = q4_retention()


# =================================================================== Q5: branch asymmetry, Buragohain 2019 Figs. 2a,b
def q5_branch_fit():
    """Fit Vc-(t), Vc+(t) of both stored states with E17 (+ E17a): log-spaced states between tau_1 and 1e8 s."""
    rows = []
    for r in BUR_VC:
        s0 = 1.0 if "Pdown" in r["panel"] else -1.0          # Pdown is written by +6 V (our + state)
        rows.append((s0, float(r["t_delay_s"]), float(r["Vc_minus_V"]), float(r["Vc_plus_V"])))
    rows = np.array(rows)

    def model(th, kw_free=True, pts=None):
        vi_, lt1, kw = th[0], th[1], (th[2] if kw_free else 0.0)
        base = th[3:] if kw_free else th[2:]
        p = nm.generic_card(imp_v=vi_, imp_t1=10 ** lt1, imp_t2=1e8, imp_kw=kw)
        out = []
        for s0, t, _, _ in (rows if pts is None else pts):
            si = float(rd.stored_imprint(p, s0, t, t_write=5e-3, v_write=0.0))
            si0 = float(rd.stored_imprint(p, s0, 1e-5, t_write=5e-3, v_write=0.0))
            dc = -vi_ * (si - si0)                                    # centre shift relative to t = 10 us
            dw = float(nm.imprint_widening(p, si) - nm.imprint_widening(p, si0))
            k = 0 if s0 > 0 else 2
            out += [base[k] + dc - dw, base[k + 1] + dc + dw]       # Vc-, Vc+
        return np.array(out)

    data = rows[:, 2:].ravel()
    TD = np.logspace(-5, 5, 61)
    res = {}
    for name, kw_free in (("E17 + E17a (kappa free)", True), ("E17 only (kappa = 0)", False)):
        th0 = [3.0, -1.0] + ([0.2] if kw_free else []) + [-2.6, 2.65, -2.5, 2.7]
        lb = [0.1, -6, ] + ([-0.5] if kw_free else []) + [-4] * 4
        ub = [30, 2] + ([1.0] if kw_free else []) + [4] * 4
        sol = least_squares(lambda th: model(th, kw_free) - data, th0, bounds=(lb, ub))
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * (sol.fun @ sol.fun) / max(1, len(data) - len(th0))
        th = sol.x
        p = nm.generic_card(imp_v=th[0], imp_t1=10 ** th[1], imp_t2=1e8)
        res[name] = {"V_I [V]": th[0], "log10 tau_1 [s]": th[1], "kappa": th[2] if kw_free else 0.0,
                     "sd kappa": float(np.sqrt(cov[2, 2])) if kw_free else 0.0,
                     "V0 centre [V/e-fold]": th[0] / np.log(1e8 / 10 ** th[1]),
                     "rms_V": rms(sol.fun), "n_points": len(data), "n_parameters": len(th0),
                     "model": model(th, kw_free).tolist(), "theta": th.tolist(),
                     "dense": {"t": TD.tolist(), "Pdown": model(th, kw_free, [(1.0, t, 0, 0) for t in TD]).tolist(),
                               "Pup": model(th, kw_free, [(-1.0, t, 0, 0) for t in TD]).tolist()}}
    res["data"] = rows.tolist()
    return res


OUT["Q5 Buragohain branches"] = q5_branch_fit()


# =================================================================== Q1: how much shift before the first read
def delay_needed(v0_esc, tau0, delta, t_loop=T_LOOP):
    """Delay d with V0 [ln(1 + d/tau0) - ln(1 + t_loop/tau0)] = delta (single log law, no saturation)."""
    g = delta / v0_esc + np.log1p(t_loop / tau0)
    return tau0 * np.expm1(g)


def q1_table(targets):
    kap = OUT["Q5 Buragohain branches"]["E17 + E17a (kappa free)"]["kappa"]
    ret = OUT["Q4 retention"]["C-V branches"]
    rows = []
    lit = [("Fengler 2018, TiN/HZO 10 nm, 25 C (centre)", 0.0205 * 0.9, 1.1),
           ("Fengler 2018, TiN/HZO 10 nm, 125 C (centre)", 0.0272 * 0.9, 0.13),
           ("Alcala 2022, TiN/Si:HfO2 10 nm, 85 C, SS (centre)", 0.051 * 0.9, None),
           ("Buragohain 2019, IrOx/La:HfO2 20 nm, RT (centre, Pup)", 0.136 * 9 / 20, 0.18),
           ("Buragohain 2019, IrOx/La:HfO2 20 nm, RT (centre, Pdown)", 0.122 * 9 / 20, 0.068)]
    for name, v0c, tau0 in lit:
        v0e = v0c * (1.0 + kap)
        for t0 in ([tau0] if tau0 else []) + [1e-6]:
            for target in targets:
                rows.append({"source": name, "V0_esc [V/e-fold] at 9 nm": v0e, "tau0 [s]": t0, "Delta [V]": target,
                             "delay needed [s]": float(delay_needed(v0e, t0, target))})
    for k in ("V0_esc from Q+ [V/e-fold]", "V0_esc from Q- [V/e-fold]"):
        for target in targets:
            rows.append({"source": f"Mukherjee 2023 Fig. 6f retention, {k[7:]} (C-V branches)",
                         "V0_esc [V/e-fold] at 9 nm": ret[k], "tau0 [s]": 1e-6, "Delta [V]": target,
                         "delay needed [s]": float(delay_needed(ret[k], 1e-6, target))})
    return {"kappa used": kap, "loop age t_L [s]": T_LOOP, "rows": rows}



# =================================================================== Q1 + Q2: fits of |Q+| and |Q-| (Fig. 7b-d)
SIG_Q = {-0.1: 0.15, -0.3: 0.4, -0.5: 1.5}       # digitizing / scatter scale of each panel [pC]
KAPPA = OUT["Q5 Buragohain branches"]["E17 + E17a (kappa free)"]["kappa"]
T_CV = 0.3                                        # age of a state in the C-V sweep at the field of interest [s]


def series(v, nmax=1e8):
    return [(n, a, b) for n, a, b in QPM[v] if n <= nmax]


def loop_sigma(p, period=50e-6, t_after=T_LOOP):
    """Mean imprint of the + state at the - branch of a periodic +-loop: x_t a +-1 square wave (no acceleration)."""
    tau = nm.imprint_taus(p)
    half = 0.5 * period
    s_sw = -np.tanh(half / (2.0 * tau))              # periodic value just before the switch to +
    return float(np.mean(1.0 - (1.0 - s_sw) * np.exp(-t_after / tau)))


def model_const(theta_k, delta, nmax=1e8, p=None):
    """|Q+|, |Q-| [pC] with a constant stored-state shift; read amplitude factors theta_k (one per V_R)."""
    p = P_M2 if p is None else p
    out = {}
    for v, k in zip(VR, theta_k):
        ns = [n for n, _, _ in series(v, nmax)]
        r = rd.read_sequence(p, k * v, ns, t_off=T_OFF, stored_shift=delta)
        out[v] = (np.abs(r["q_plus"]) * 1e12, np.abs(r["q_minus"]) * 1e12, r)
    return out


def dyn_card(v_i, lt1, p=None, kappa=None):
    p = P_M2 if p is None else p
    return dict(p, imp_v=v_i, imp_t1=10 ** lt1, imp_t2=1e8, imp_kw=KAPPA if kappa is None else kappa)


def model_dyn(theta_k, v_i, lt1, ld, nmax=1e8, p=None, kappa=None, nd=256, fast=False):
    """|Q+|, |Q-| [pC] with the dynamic stored-state imprint (E17, E17a): delay 10^ld s before read 1."""
    q = dyn_card(v_i, lt1, p, kappa)
    sl = loop_sigma(q)
    scv = float(rd.stored_imprint(q, 1.0, T_CV, t_write=1e-6, v_write=0.0))
    out = {}
    for v, k in zip(VR, theta_k):
        ns = [n for n, _, _ in series(v, nmax)]
        r = rd.read_sequence_dynamic(q, k * v, ns, 10 ** ld, t_off=T_OFF, nd=nd, sigma_loop=sl, sigma_cv=scv,
                                     t_write=1e-6, v_write=0.0, k_exact=40 if fast else 200,
                                     n_grid=120 if fast else 300)
        out[v] = (np.abs(r["q_plus"]) * 1e12, np.abs(r["q_minus"]) * 1e12, r)
    return out


def resid_of(m, nmax=1e8):
    res = []
    for v in VR:
        d = np.array(series(v, nmax))
        res += [(m[v][0] - d[:, 1]) / SIG_Q[v], (m[v][1] - d[:, 2]) / SIG_Q[v]]
    return np.concatenate(res)


def summarize(m, theta, names, nmax=1e8):
    r = resid_of(m, nmax)
    cmw = []
    for v in VR:
        d = np.array(series(v, nmax))
        cmw.append((m[v][0] - m[v][1]) - (d[:, 1] - d[:, 2]))
    return {"parameters": dict(zip(names, [float(x) for x in theta])), "chi2_per_point": float(np.mean(r**2)),
            "rms_Qplus_pC": {str(v): rms(m[v][0] - np.array(series(v, nmax))[:, 1]) for v in VR},
            "rms_Qminus_pC": {str(v): rms(m[v][1] - np.array(series(v, nmax))[:, 2]) for v in VR},
            "rms_CMW_pC": rms(np.concatenate(cmw)),
            "model": {str(v): {"n": [n for n, _, _ in series(v, nmax)], "Q_plus": m[v][0].tolist(),
                               "Q_minus": m[v][1].tolist(), "Q_sw_plus": (np.abs(m[v][2]["q_sw_plus"]) * 1e12).tolist(),
                               "f_plus": m[v][2]["f_plus"].tolist(),
                               "s_i_plus": m[v][2].get("s_i_plus", np.zeros(1)).tolist()} for v in VR}}


def fit_const(free_k=True):
    names = (["k(-0.5)", "k(-0.3)", "k(-0.1)"] if free_k else []) + ["Delta [V]"]
    th0 = ([0.85, 0.9, 0.88] if free_k else []) + [0.4]
    lb = ([0.6] * 3 if free_k else []) + [0.0]
    ub = ([1.1] * 3 if free_k else []) + [1.0]

    def unpack(th):
        return (th[:3] if free_k else [1.0, 1.0, 1.0]), th[-1]

    def run():
        sol = least_squares(lambda th: resid_of(model_const(*unpack(th))), th0, bounds=(lb, ub), diff_step=1e-3,
                            x_scale="jac", max_nfev=80)
        return [float(x) for x in sol.x]

    th = cached("fit_const", run, [names, CVX, DELTA_M2])
    return summarize(model_const(*unpack(th)), th, names), th


def fit_dyn(nmax=1e8, kappa=None, tag="dyn"):
    names = ["k(-0.5)", "k(-0.3)", "k(-0.1)", "V_I [V]", "log10 tau_I1 [s]", "log10 delay [s]"]
    th0 = [0.8, 0.9, 0.86, 0.6, -6.0, -1.5]
    lb = [0.6, 0.6, 0.6, 0.05, -8.0, -5.0]
    ub = [1.1, 1.1, 1.1, 5.0, 1.0, 3.0]

    def run():
        best = None
        for st in (th0, [0.8, 0.9, 0.86, 0.6, -2.0, 0.5]):
            sol = least_squares(lambda th: resid_of(model_dyn(th[:3], *th[3:], nmax=nmax, kappa=kappa, nd=128, fast=True), nmax),
                                st, bounds=(lb, ub), diff_step=1e-3, x_scale="jac", max_nfev=60)
            if best is None or sol.cost < best.cost:
                best = sol
        return [float(x) for x in best.x]

    th = cached("fit_" + tag, run, [names, CVX, nmax, kappa, KAPPA, T_CV, T_LOOP])
    m = model_dyn(th[:3], *th[3:], nmax=1e11, kappa=kappa)
    s = summarize(model_dyn(th[:3], *th[3:], nmax=nmax, kappa=kappa), th, names, nmax)
    q = dyn_card(th[3], th[4], kappa=kappa)
    s["V0_centre [V/e-fold]"] = th[3] / np.log(1e8 / 10 ** th[4])
    s["V0_escape [V/e-fold]"] = s["V0_centre [V/e-fold]"] * (1.0 + q["imp_kw"])
    s["sigma_loop"] = loop_sigma(q)
    s["escape shift at read 1 relative to the loop [V]"] = float(
        th[3] * (1 + q["imp_kw"]) * (rd.stored_imprint(q, 1.0, 10 ** th[5], 1e-6, 0.0) - loop_sigma(q)))
    s["model_all_n"] = {str(v): {"n": [n for n, _, _ in series(v, 1e11)], "Q_plus": m[v][0].tolist(),
                                 "Q_minus": m[v][1].tolist(), "s_i_plus": m[v][2]["s_i_plus"].tolist()} for v in VR}
    return s, th

if __name__ == "__main__" and "--quick" in __import__("sys").argv:
    print(json.dumps({k: OUT[k] for k in OUT}, indent=1, default=float)[:20000])
    raise SystemExit

# R0: nominal read amplitude, constant shift refit; R1: read amplitude factors + constant shift (the tail-fit
# stand-in); R2: amplitude factors + dynamic imprint, fitted to n <= 1e8; R3: the same fitted to all n.
R0, TH_R0 = fit_const(free_k=False)
R1, TH_R1 = fit_const(free_k=True)
OUT["Q1 delay table"] = q1_table((TH_R1[3], DELTA_M3))
print("R0", {k: R0[k] for k in ("parameters", "chi2_per_point", "rms_CMW_pC")}, flush=True)
print("R1", {k: R1[k] for k in ("parameters", "chi2_per_point", "rms_CMW_pC")}, flush=True)
R2, TH_R2 = fit_dyn(nmax=1e8, tag="dyn8")
print("R2", {k: R2[k] for k in R2 if not k.startswith("model")}, flush=True)
R3, TH_R3 = fit_dyn(nmax=1e11, tag="dyn11")
print("R3", {k: R3[k] for k in R3 if not k.startswith("model")}, flush=True)
FITS = {"R0 nominal amplitude, constant shift": R0, "R1 amplitude factors, constant shift": R1,
        "R2 amplitude factors, dynamic imprint (fit n <= 1e8)": R2,
        "R3 amplitude factors, dynamic imprint (fit all n)": R3}


# =================================================================== Q2: the early drop at -0.5 V
def q2():
    r_state = dict((n, a / b) for n, a, b in QPM[-0.3])[1.0]       # |Q+|/|Q-| without switching (Q3)
    rows = []
    for n, a, b in QPM[-0.5]:
        rows.append({"n": n, "Q+ data": a, "Q- data": b, "excess Q+ - r Q- [pC]": a - r_state * b})
    out = {"state ratio r used (-0.3 V, read 1)": r_state, "excess": rows}
    for name in ("R1 amplitude factors, constant shift", "R2 amplitude factors, dynamic imprint (fit n <= 1e8)",
                 "R3 amplitude factors, dynamic imprint (fit all n)"):
        m = FITS[name]["model"]["-0.5"]
        out[name] = {"n": m["n"], "switching charge per read [pC]": m["Q_sw_plus"], "switched fraction": m["f_plus"]}
    # soft-region bound: a domain at its switching field gives at most (1 - 1/sqrt(3)) P of reversible
    # displacement (Landau double well), against 2 P when it switches
    drop = dict((n, a) for n, a, _ in QPM[-0.5])
    d12 = drop[1.0] - drop[12.0]
    ratio = (1.0 - 1.0 / np.sqrt(3.0)) / 2.0
    out["soft-region bound"] = {"Q+ drop read 1 -> 12 [pC]": d12, "max reversible / switching charge per domain": ratio,
                                "switching charge needed in reads 1-12 at least [pC]": d12 / ratio,
                                "excess of read 1 in the data [pC]": rows[0]["excess Q+ - r Q- [pC]"]}
    return out


OUT["Q2"] = q2()


# =================================================================== Q4: retention of the fitted imprint, and ratios
def retention_model(th, t, kappa=None):
    """|Q+|, |Q-| [pC] of a single -0.1 V read after storage t (no reads in between) for fit parameters th."""
    q = dyn_card(th[3], th[4], kappa=kappa)
    scv = float(rd.stored_imprint(q, 1.0, T_CV, 1e-6, 0.0))
    v = th[2] * -0.1
    out = []
    for tt in t:
        row = []
        for s0 in (1.0, -1.0):
            si = float(rd.stored_imprint(q, s0, tt, 1e-6, 0.0))
            c = rd.imprint_card(q, si, loop_sigma(q), s0, scv)
            row.append(abs(nm.q_rev(c, v, s0)) * c["area"] * 1e12)
        out.append(row)
    return np.array(out)


def q4():
    out = {}
    tt = np.logspace(0, 5, 11)
    for name, th in (("R2", TH_R2), ("R3", TH_R3)):
        m = retention_model(th, tt)
        sp = np.polyfit(np.log10(tt), m[:, 0], 1)[0]
        sm = np.polyfit(np.log10(tt), m[:, 1], 1)[0]
        out[f"{name}: retention slopes predicted [pC/decade] (Q+, Q-)"] = [float(sp), float(sm)]
        out[f"{name}: retention curve"] = {"t": tt.tolist(), "Q+": m[:, 0].tolist(), "Q-": m[:, 1].tolist()}
    out["Fig. 6f slopes [pC/decade] (Q+, Q-)"] = [OUT["Q4 retention"]["slope Q+ [pC/decade]"],
                                                  OUT["Q4 retention"]["slope Q- [pC/decade]"]]
    # late decline: dQ+/dQ- in the data vs imprint (branch shift) and pinning (term (2) lost) predictions
    rat = {}
    for v, k in zip(VR, TH_R1[:3]):
        if v == -0.5:
            continue
        d = {n: (a, b) for n, a, b in QPM[v]}
        n_hi = 1e11 if 1e11 in d else 1e10
        dqp, dqm = d[1e8][0] - d[n_hi][0], d[1e8][1] - d[n_hi][1]
        vv = k * v
        imp = (q_rev_state(P_M2, vv, 1, 0.05) - q_rev_state(P_M2, vv, 1, -0.05)) / \
              (q_rev_state(P_M2, vv, -1, 0.05) - q_rev_state(P_M2, vv, -1, -0.05))

        def t2(s):
            return abs(nm.q_rev(P_M2, vv, s) - P_M2["c_d"] * vv) * A_FE * 1e12

        rat[str(v)] = {"data dQ+/dQ- (1e8 -> end)": dqp / dqm, "dQ+ [pC]": dqp, "dQ- [pC]": dqm,
                       "imprint (C-V branches move away)": imp, "pinning (term (2) lost)": t2(1.0) / t2(-1.0)}
    rp, rm = OUT["Q4 retention"]["slope Q+ [pC/decade]"], OUT["Q4 retention"]["slope Q- [pC/decade]"]
    rat["retention Fig. 6f (-0.1 V, time only)"] = {"data dQ+/dQ-": rp / rm}
    out["late decline ratio test"] = rat
    return out


OUT["Q4"] = q4()
OUT["fits"] = FITS
OUT["assumptions"] = {"t_off_s": T_OFF, "loop age of the + state at the - branch [s]": T_LOOP,
                      "age of a C-V state [s]": T_CV, "write pulse": "1 us, its imprint time not accelerated (V_a off)",
                      "kappa (from Buragohain 2019 Figs. 2a,b)": KAPPA, "tau_I2 [s]": 1e8,
                      "weights": SIG_Q, "x_t": "stored state (switched fraction neglected in the imprint drive)"}
(OUTDIR / "read_imprint_r2.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")

# =================================================================== figures
OUTDIR.mkdir(parents=True, exist_ok=True)

# Fig. 1: Q3, per-volt charge of both states and the C-V time-scale switching onset
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
vv = np.array(VR)
for s0, key, c, lab in ((1, 0, "C0", "+ state"), (-1, 1, "C3", "- state")):
    dat = [({n: (a, b) for n, a, b in QPM[v]}[12.0 if v == -0.5 else 1.0][key]) / abs(v) / EPS_UNIT for v in VR]
    a1.plot(-vv, dat, "o", color=c, mec="k", ms=7, label=f"{lab}: pulse data (read 1; -0.5 V: read 12)")
    xs = np.linspace(0.02, 0.6, 60)
    a1.plot(xs, [q_rev_state(P_M2, -x, s0) / x / EPS_UNIT for x in xs], "-", color=c, lw=1.2,
            label=f"{lab}: C-V branches, nominal V_R")
    a1.plot(-vv, [q_rev_state(P_M2, k * v, s0) / abs(v) / EPS_UNIT for v, k in zip(VR, TH_R1[:3])],
            "x", color=c, ms=9, mew=2, label=f"{lab}: C-V branches at the fitted amplitude k V_R")
a1.set_xlabel("|V_R| nominal (V)")
a1.set_ylabel(r"$|Q|\,t_{FE}/(A\varepsilon_0|V_R|)$  (secant $\varepsilon_r$)")
a1.set_title("Read charge per volt: the deficit is common to both states", fontsize=10)
a1.legend(fontsize=7, frameon=False, loc="lower left")
a1.set_ylim(18, 46)
cvs = OUT["Q3"]["C-V sweep switching onset"]
e = -np.array(cvs["fields_MV_cm"])
a2.semilogy(e, np.maximum(cvs["switched"], 1e-6), "o-", color="C0", ms=3,
            label=f"+ state stabilized by {cvs['delta_V']:.2f} V, 13 ms per 0.125 MV/cm step")
a2.axvline(0.48, color="C0", ls=":", lw=1)
a2.text(0.5, 1e-5, r"C-V peak of $\varepsilon_+$" + "\n(-0.48 MV/cm)", fontsize=8, color="C0")
for v in VR:
    a2.axvline(-v / T9 / MV, color=COL[v], lw=0.7, alpha=0.6)
a2.set_xlabel("|E| of the DC bias (MV/cm), negative polarity")
a2.set_ylabel("fraction switched during the down-sweep")
a2.set_ylim(1e-6, 1.2)
a2.set_title("C-V time scale: the stored state starts to switch at the C-V peak", fontsize=10)
a2.legend(fontsize=7.5, frameon=False, loc="center right")
fig.savefig(OUTDIR / "rir2_cv_vs_pulse.png", dpi=150)
plt.close(fig)

# Fig. 2: Q5, branch coercive voltages of Buragohain 2019
fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
q5r = OUT["Q5 Buragohain branches"]
dat = np.array(q5r["data"])
for name, ls in (("E17 + E17a (kappa free)", "-"), ("E17 only (kappa = 0)", ":")):
    for key, c in (("Pdown", "C3"), ("Pup", "C0")):
        mod = np.array(q5r[name]["dense"][key]).reshape(-1, 2)
        ax.semilogx(q5r[name]["dense"]["t"], mod[:, 0], ls, color=c, lw=1.2)
        ax.semilogx(q5r[name]["dense"]["t"], mod[:, 1], ls, color=c, lw=1.2)
    ax.plot([], [], ls, color="k", label=f"{name}: RMS {q5r[name]['rms_V'] * 1e3:.0f} mV")
for s0, c, lab in ((1.0, "C3", "Pdown (+6 V set)"), (-1.0, "C0", "Pup (-6 V set)")):
    m = dat[:, 0] == s0
    ax.semilogx(dat[m, 1], dat[m, 2], "v", color=c, mec="k", label=f"{lab}: Vc-")
    ax.semilogx(dat[m, 1], dat[m, 3], "^", color=c, mec="k", label=f"{lab}: Vc+")
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlabel("delay after the set pulse (s)")
ax.set_ylabel("coercive voltage (V)")
ax.set_title(f"Buragohain 2019 Figs. 2a,b: the escape branch moves more (kappa = {KAPPA:.2f})", fontsize=10)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="center left")
fig.savefig(OUTDIR / "rir2_branch_asymmetry.png", dpi=150)
plt.close(fig)

# Fig. 3: Q1, shift built between the loop (age 21 us) and the first read
fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
dd = np.logspace(-5, 5, 200)
cases = [("Fengler 2018 HZO/TiN 25 C", 0.0205 * 0.9 * (1 + KAPPA), 1e-6, "C7"),
         ("Alcala 2022 Si:HfO2 85 C (SS)", 0.051 * 0.9 * (1 + KAPPA), 1e-6, "C2"),
         ("Buragohain 2019 La:HfO2 RT", 0.129 * 0.45 * (1 + KAPPA), 1e-6, "C1"),
         ("Buragohain 2019, its own tau0 = 0.1 s", 0.129 * 0.45 * (1 + KAPPA), 0.1, "C1"),
         ("this device, Fig. 6f via |Q+|", OUT["Q4 retention"]["C-V branches"]["V0_esc from Q+ [V/e-fold]"], 1e-6, "C0"),
         ("this device, Fig. 6f via |Q-|", OUT["Q4 retention"]["C-V branches"]["V0_esc from Q- [V/e-fold]"], 1e-6, "C9")]
for lab, v0, t0, c in cases:
    ax.semilogx(dd, v0 * (np.log1p(dd / t0) - np.log1p(T_LOOP / t0)), "--" if t0 > 1e-3 else "-", color=c,
                label=f"{lab}: {v0 * 1e3:.0f} mV/e-fold" + (f", tau0 {t0:g} s" if t0 > 1e-3 else ""))
for name, key in (("R2", "R2 amplitude factors, dynamic imprint (fit n <= 1e8)"),
                  ("R3", "R3 amplitude factors, dynamic imprint (fit all n)")):
    f = FITS[key]
    ax.plot(10 ** f["parameters"]["log10 delay [s]"], f["escape shift at read 1 relative to the loop [V]"], "*",
            ms=13, mec="k", label=f"{name} fit: delay and shift at read 1")
ax.axhspan(TH_R1[3] - 0.005, DELTA_M3 + 0.005, color="0.85", zorder=0)
ax.text(2e-5, DELTA_M3 + 0.03, f"needed at read 1: {TH_R1[3]:.2f} V (R1) to {DELTA_M3:.2f} V (tail-fit M3)",
        fontsize=8)
ax.set_ylim(0, 0.9)
ax.set_xlabel("delay between write and first read (s)")
ax.set_ylabel("escape-field shift relative to the loop (V)")
ax.set_title("Storage-time imprint built before the first read (scaled to 9 nm)", fontsize=10)
ax.legend(fontsize=6.8, frameon=False, loc="upper left")
fig.savefig(OUTDIR / "rir2_delay.png", dpi=150)
plt.close(fig)

# Fig. 4: fits of |Q+| and |Q-|
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
for ax, v in zip(axs, VR):
    d = np.array(QPM[v])
    ax.semilogx(d[:, 0], d[:, 1], "o", color="C0", mec="k", label="|Q+| data")
    ax.semilogx(d[:, 0], d[:, 2], "s", color="C3", mec="k", label="|Q-| data")
    for name, ls, short in (("R1 amplitude factors, constant shift", ":", "R1 constant shift"),
                            ("R2 amplitude factors, dynamic imprint (fit n <= 1e8)", "--", "R2 dynamic, fit n <= 1e8"),
                            ("R3 amplitude factors, dynamic imprint (fit all n)", "-", "R3 dynamic, fit all n")):
        f = FITS[name]
        mm = f.get("model_all_n", f["model"])[str(v)]
        ax.semilogx(mm["n"], mm["Q_plus"], ls, color="C0", lw=1.3)
        ax.semilogx(mm["n"], mm["Q_minus"], ls, color="C3", lw=1.3)
        ax.plot([], [], ls, color="k", label=short)
    ax.axvline(1e8, color="0.6", ls=":", lw=0.8)
    ax.set_title(f"V_R = {v} V (nominal)", fontsize=10)
    ax.set_xlabel("read number n")
axs[0].set_ylabel("|Q| (pC)")
axs[0].legend(fontsize=7.5, frameon=False, loc="center left")
fig.savefig(OUTDIR / "rir2_fits.png", dpi=150)
plt.close(fig)

# Fig. 5: Q2 excess charge of the + state at -0.5 V, and Q4 retention on one time axis
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
ex = OUT["Q2"]["excess"]
a1.semilogx([r["n"] for r in ex], [r["excess Q+ - r Q- [pC]"] for r in ex], "o-", color="k", mec="k",
            label=r"data: $|Q_+| - r|Q_-|$, $r$ = 1.28 (state ratio at -0.3 V)")
for name, c in (("R1 amplitude factors, constant shift", "C1"),
                ("R3 amplitude factors, dynamic imprint (fit all n)", "C2")):
    m = OUT["Q2"][name]
    a1.semilogx(m["n"], m["switching charge per read [pC]"], "s--", color=c, ms=4,
                label=name.split(",")[0] + ": switching charge of read n")
a1.axhline(0, color="0.6", lw=0.6)
a1.set_xlabel("read number n")
a1.set_ylabel("charge (pC)")
a1.set_title("-0.5 V: read 1 carries switching charge, later reads lose response", fontsize=10)
a1.legend(fontsize=7, frameon=False)
a2.semilogx(RET[:, 0], RET[:, 1], "o", color="C0", mec="k", label="|Q+| Fig. 6f (retention, -0.1 V)")
a2.semilogx(RET[:, 0], RET[:, 2], "s", color="C3", mec="k", label="|Q-| Fig. 6f")
for name, ls in (("R2", "--"), ("R3", "-")):
    rc = OUT["Q4"][f"{name}: retention curve"]
    a2.semilogx(rc["t"], rc["Q+"], ls, color="C0")
    a2.semilogx(rc["t"], rc["Q-"], ls, color="C3")
    a2.plot([], [], ls, color="k", label=f"{name}: imprint fitted to the reads, storage only")
d01 = np.array(QPM[-0.1])
t_el = 10 ** TH_R3[5] + (d01[:, 0] - 1) * (1.5e-6 + T_OFF)
a2.semilogx(t_el, d01[:, 1], "^", color="C0", mfc="none", label="|Q+| Fig. 7b at the elapsed time (R3 delay, 2 us/read)")
a2.semilogx(t_el, d01[:, 2], "v", color="C3", mfc="none", label="|Q-| Fig. 7b at the elapsed time")
a2.set_xlim(1e-3, 1e6)
a2.set_xlabel("time since write (s)")
a2.set_ylabel("|Q| at -0.1 V (pC)")
a2.set_title("Retention (Fig. 6f) and read endurance (Fig. 7b) on one time axis", fontsize=10)
a2.legend(fontsize=6.6, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.47))
fig.savefig(OUTDIR / "rir2_early_drop_retention.png", dpi=150)
plt.close(fig)
print(json.dumps({k: OUT[k] for k in ("Q2", "Q4")}, indent=1, default=float)[:6000])


# =================================================================== Q1b: write-time (field-accelerated) imprint
def loop_sigma_accel(p, period=50e-6, vpk=2.7, v_up=1.35, v_dn=-0.54, nper=30, m=4000):
    """Mean imprint of the + state when the - branch switches, in a periodic +-2.7 V, 50 us triangle (E17 + E17b).

    x_t flips at the measured switching-current peaks (+1.35 V, -0.54 V); every imprint rate carries
    cosh(V / V_a).  Integrated exactly per sub-step for ``nper`` periods (periodic state reached).
    """
    tau = nm.imprint_taus(p)
    sj, x, rec = np.zeros_like(tau), -1.0, None
    t = np.linspace(0.0, period, m + 1)

    def vt(tt):
        ph = (tt / period) % 1.0
        return vpk * (4 * ph if ph < 0.25 else 2 - 4 * ph if ph < 0.75 else 4 * ph - 4)

    for k in range(nper):
        for a, b in zip(t[:-1], t[1:]):
            v = vt(0.5 * (a + b))
            if x < 0 and v >= v_up:
                x = 1.0
            if x > 0 and v <= v_dn:
                if k == nper - 1 and rec is None:
                    rec = float(np.mean(sj))
                x = -1.0
            sj = x + (sj - x) * np.exp(-(b - a) * float(nm.imprint_accel(p, v)) / tau)
    return rec


def q1_write_time():
    rows = []
    for va in (0.0, 0.5, 0.3, 0.2):
        p = nm.generic_card(imp_v=1.0, imp_t1=1e-7, imp_t2=1e8, imp_va=va)
        sl = loop_sigma_accel(p)
        for d in (1e-3, 1e-1, 10.0):
            ss = float(rd.stored_imprint(p, 1.0, d, t_write=1e-6, v_write=2.7))
            rows.append({"V_a [V]": va, "sigma loop (- branch)": sl, "delay [s]": d, "sigma stored": ss,
                         "difference (per volt of V_I)": ss - sl})
    return {"spectrum": "tau 1e-7 ... 1e8 s, 12 states; write 1 us at 2.7 V; loop 50 us, +-2.7 V", "rows": rows}


OUT["Q1 write-time imprint"] = cached("q1wt", q1_write_time, ["v1"])
(OUTDIR / "read_imprint_r2.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
