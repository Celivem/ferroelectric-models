"""Compare the multi-domain NLS model with values taken from published papers.

Writes docs/literature_switching.png, docs/literature_cv.png,
docs/literature_read_endurance.png, docs/literature_backswitching.png,
docs/literature_series_read.png and docs/literature_comparison.json.

Inputs are the CSV files in data/literature/ (each has a header with the full
citation, the figure, and whether the values are a published fit formula or
were digitized from a figure).  No raw measurement files were available, so
"data" here always means numbers taken from the papers.

What was fitted and what was not
--------------------------------
* Switching (Alessandri 2018 / 2019): the Dagum parameters of the generic card
  were fitted to curves computed from the published GB2 fit of Alessandri 2019;
  this script refits both papers the same way and reports the residual.
* C-V (Mukherjee EDL 2023, IEDM 2023): the reversible-branch parameters are fits
  to the digitized points (3 points + width for Dev A; 4 parameters to 7 points
  for the IEDM device).
* Read endurance (Mukherjee IEDM 2023, Fig. 7): no switching parameter is fitted
  to these points.  Only the activation-field scale is set so that the model's
  dynamic coercive voltage matches the device's P-V loop; the tail shape comes
  from Alessandri 2019 (or 2018).  A third curve uses an illustrative 2 ns
  attempt time to show how strongly sub-coercive reads depend on tau0.
* Back-switching (Lancaster 2022): V_dep and tau_s are fitted to the digitized
  P_loss values (two parameters, twelve points).
* Series-capacitor reads (Duan IEDM 2025, Fig. 9): qualitative; nothing fitted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, least_squares

from fe_models import nls_multidomain as nm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "literature"
DOCS = ROOT / "docs"
UC = 1e-2
MV = 1e8
EPS0 = nm.EPS0
OUT = {}


def read_csv(name):
    with open(DATA / name, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    return rows


def params(name):
    return {r["parameter"]: float(r["value"]) for r in read_csv(name)}


# ------------------------------------------------- 1. switching kinetics
def published_fraction(a, b, p, q, tau, alpha, beta, e_ext, t):
    """NLS of the papers: int {1 - exp[-(t/tau(E'))^beta]} g(E') dE' with g = GB2(a, b, p, q) [MV/cm]."""
    y = np.logspace(np.log10(b) - 1.5, np.log10(b) + 1.5, 6001)
    pdf = np.gradient(nm.gb2_cdf(y, a, b, p, q), y)
    tt = tau * np.exp(np.minimum((y / e_ext) ** alpha, 600.0))
    return float(np.trapezoid(pdf * (1.0 - np.exp(-((t / tt) ** beta))), y))


s2 = params("alessandri2018_edl_nls_fit.csv")
s4 = params("alessandri2019_ted_nls_fit.csv")
# Alessandri 2018 gives f(eta); E' = Ea/eta is then GB2(a, Ea/b, q, p)
S2 = dict(name="Alessandri 2018 (EDL)", t_nm=s2["thickness"], tau=s2["tau_inf"], alpha=s2["alpha"], beta=s2["beta"],
          g=(s2["gb2_a"], s2["Ea"] / s2["gb2_b"], s2["gb2_q"], s2["gb2_p"]), tmax=7.6e-3)
S4 = dict(name="Alessandri 2019 (TED)", t_nm=s4["thickness"], tau=s4["tau_inf"], alpha=s4["alpha"], beta=s4["beta"],
          g=(s4["g_a"], s4["g_b"], s4["g_p"], s4["g_q"]), tmax=1e-3)
VOLTS = (0.8, 1.0, 1.2, 1.5, 2.0)


def window(src, n=14):
    return np.logspace(np.log10(200e-9), np.log10(src["tmax"]), n)


def published_curves(src, ts):
    return np.array([[published_fraction(*src["g"], src["tau"], src["alpha"], src["beta"], v / (src["t_nm"] * 1e-7) * 1e-6, t)
                      for t in ts] for v in VOLTS])


def model_curves(p, src, ts, nd):
    q = dict(p, tfe=src["t_nm"] * 1e-9, vimp=0.0)  # the published formulas use E = V / t_fe
    return np.array([[nm.switched_fraction(q, v, t, nd=nd) for t in ts] for v in VOLTS])


def fit_dagum(src, ts, data):
    base = nm.generic_card(tau0=src["tau"], beta=src["beta"])

    def unpack(th):
        b, a, pt, al = np.exp(th)
        return dict(base, emp=b * MV, emn=b * MV, ap=a, an=a, pt=pt, alpha=al)

    def resid(th):
        return (model_curves(unpack(th), src, ts, 801) - data).ravel()

    a0, b0 = src["g"][0], src["g"][1]
    sol = least_squares(resid, np.log([b0, a0, 1.0, src["alpha"]]), diff_step=1e-3)
    p = unpack(sol.x)
    return p, float(np.sqrt(np.mean(sol.fun ** 2)))


sw = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, sharey=True)
generic = nm.generic_card()
for ax, src in zip(axes, (S4, S2)):
    ts = window(src)
    tf = np.logspace(np.log10(200e-9), np.log10(src["tmax"]), 60)
    data = published_curves(src, ts)
    p_fit, rms_fit = fit_dagum(src, ts, data)
    gen16 = model_curves(generic, src, ts, 16)
    gen_c = model_curves(generic, src, ts, 2000)
    fit16 = model_curves(p_fit, src, ts, 16)
    rec = {"Dagum refit: b [MV/cm], a, p, alpha": [p_fit["emp"] / MV, p_fit["ap"], p_fit["pt"], p_fit["alpha"]],
           "Dagum refit RMS (fraction, continuous)": rms_fit,
           "refit with N_d = 16, RMS": float(np.sqrt(np.mean((fit16 - data) ** 2))),
           "generic card (continuous), RMS": float(np.sqrt(np.mean((gen_c - data) ** 2))),
           "generic card (N_d = 16), RMS": float(np.sqrt(np.mean((gen16 - data) ** 2)))}
    sw[src["name"]] = rec
    pub_f = published_curves(src, tf)
    gen_f = model_curves(generic, src, tf, 16)
    for k, v in enumerate(VOLTS):
        col = plt.cm.viridis(k / (len(VOLTS) - 1) * 0.9)
        ax.semilogx(tf, pub_f[k], color=col, lw=2.2, alpha=0.45)
        ax.semilogx(tf, gen_f[k], color=col, lw=1.2, ls="--")
        ax.plot(ts, fit16[k], "o", ms=3.5, color=col, label=f"{v} V")
    ax.set_title(f"{src['name']}, {src['t_nm']:g} nm")
    ax.set_xlabel("Pulse width (s)")
    ax.text(0.02, 0.97, f"refit (dots, N_d = 16): RMS {rec['refit with N_d = 16, RMS']:.4f}\n"
                        f"generic card (dashed, N_d = 16): RMS {rec['generic card (N_d = 16), RMS']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="0.8"))
axes[0].set_ylabel("Switched fraction")
axes[1].legend(title="pulse amplitude", fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.suptitle("Switching kinetics: published NLS fits (thick lines) vs. the multi-domain model", fontsize=11)
fig.savefig(DOCS / "literature_switching.png", dpi=150)
plt.close(fig)
OUT["switching"] = sw


# ------------------------------------------------- 2. C-V branches
def lorentz(d, e_inf, de, d0, w):
    return e_inf + de / (1.0 + ((d - d0) / w) ** 2)


def distance(rows, e_cross):
    """Distance toward the switching side of each branch: up softens toward -E, down toward +E."""
    out = []
    for r in rows:
        e, br, eps = float(r["E_MV_cm"]), r["branch"], float(r["eps_r"])
        if br in ("up", "both"):
            out.append((e_cross - e, eps))
        if br == "down":
            out.append((e - e_cross, eps))
    return np.array(out)


q1_rows = read_csv("mukherjee2023_iedm_cv.csv")
q5_rows = read_csv("mukherjee2023_edl_cv_devA.csv")
pts1 = distance(q1_rows, 0.87)
sol1 = least_squares(lambda t: lorentz(pts1[:, 0], *t) - pts1[:, 1], [28, 15, 1.5, 1.0], bounds=([5, 0, 0, 0.1], [60, 60, 6, 6]))
pts5 = distance(q5_rows, 0.0)
w1 = sol1.x[3]
sol5 = least_squares(lambda t: lorentz(pts5[:, 0], t[0], t[1], t[2], w1) - pts5[:, 1], [25, 15, 1.0],
                     bounds=([5, 0, 0], [60, 60, 6]))
OUT["cv"] = {"Mukherjee IEDM 2023 fit (eps_inf, d_eps, d0 [MV/cm], w [MV/cm])": list(map(float, sol1.x)),
             "Mukherjee IEDM 2023 RMS eps_r (7 points, 4 parameters)": float(np.sqrt(np.mean(sol1.fun ** 2))),
             "Mukherjee EDL 2023 Dev A fit (eps_inf, d_eps, d0) with w from IEDM": list(map(float, sol5.x)),
             "Mukherjee EDL 2023 Dev A RMS eps_r (5 points incl. mirror images, 3 parameters)": float(np.sqrt(np.mean(sol5.fun ** 2))),
             "generic card c_d, dcap, vs, vw (10 nm)": [generic["c_d"], generic["dc"], generic["vs"], generic["vw"]],
             "same numbers from this refit": [sol5.x[0] * EPS0 / 1e-8, sol5.x[1] * EPS0 / 1e-8, sol5.x[2] * MV * 1e-8, w1 * MV * 1e-8]}


def eps_branches(p, e):
    v = e * MV * p["tfe"]
    c = lambda x: (nm.q_rev(p, v + 1e-4, x) - nm.q_rev(p, v - 1e-4, x)) / 2e-4
    return c(1.0) * p["tfe"] / EPS0, c(-1.0) * p["tfe"] / EPS0


t9 = 9e-9
q1_card = nm.generic_card(tfe=t9, ps=0.165, c_d=sol1.x[0] * EPS0 / t9, dc=sol1.x[1] * EPS0 / t9,
                          vs=sol1.x[2] * MV * t9, vw=sol1.x[3] * MV * t9, vi=0.87 * MV * t9, vimp=0.45 * MV * t9)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, card, rows, title, ylim in ((axes[0], generic, q5_rows, "Mukherjee EDL 2023, Dev A (TiN/TiN), generic card", (20, 50)),
                                    (axes[1], q1_card, q1_rows, "Mukherjee IEDM 2023 (asymmetric stack), own fit", (20, 50))):
    e = np.linspace(-3.2, 3.2, 321)
    up, dn = np.array([eps_branches(card, x) for x in e]).T
    ax.plot(e, up, color="C0", label="model, up-polarized domains (x = +1)")
    ax.plot(e, dn, color="C3", label="model, down-polarized domains (x = -1)")
    for r in rows:
        col = {"up": "C0", "down": "C3", "both": "k"}[r["branch"]]
        ax.plot(float(r["E_MV_cm"]), float(r["eps_r"]), "o", color=col, mec="k", ms=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Field (MV/cm)")
    ax.set_ylim(*ylim)
axes[0].set_ylabel(r"Small-signal $\varepsilon_r$")
axes[0].legend(fontsize=8, frameon=False, loc="lower center")
axes[1].text(0.02, 0.04, "dots: digitized from the papers (approx.)\nmodel branches continue past the point where the\n"
             "state would switch; a measured C-V follows each\nbranch only up to its switching field",
             transform=axes[1].transAxes, fontsize=8, va="bottom")
fig.savefig(DOCS / "literature_cv.png", dpi=150)
plt.close(fig)

# ------------------------------------------------- 3. read endurance (Mukherjee IEDM 2023, Fig. 7)
A_FE = 60e-6 * 80e-6
end_rows = read_csv("mukherjee2023_iedm_read_endurance.csv")
tails = {"S4 tail (Alessandri 2019)": dict(q1_card),
         "S2 tail (Alessandri 2018, Dagum refit)": dict(q1_card, emp=sw[S2["name"]]["Dagum refit: b [MV/cm], a, p, alpha"][0] * MV,
                                                        ap=sw[S2["name"]]["Dagum refit: b [MV/cm], a, p, alpha"][1],
                                                        pt=sw[S2["name"]]["Dagum refit: b [MV/cm], a, p, alpha"][2],
                                                        alpha=sw[S2["name"]]["Dagum refit: b [MV/cm], a, p, alpha"][3],
                                                        tau0=S2["tau"], beta=S2["beta"]),
         "S4 tail, tau0 = 2 ns (illustrative)": dict(q1_card, tau0=2e-9)}
for k in tails:
    tails[k]["emn"], tails[k]["an"] = tails[k]["emp"], tails[k]["ap"]
VC_HALF, T_QUARTER = 0.945, 12.5e-6  # dynamic Vc half width of the 50 us P-V loop (digitized: +1.35 V, -0.54 V)
T_ON, T_OFF = 1e-6, 0.5e-6  # 500/500/500 ns trapezoid -> 1 us effective; off time not given (assumed)


def cmw(p, v, n):
    """|Q+ - Q-| of the n-th read at -v [pC]: reversible charge plus the switching charge of that read."""
    f_n = nm.train_fraction(p, -v, T_ON, T_OFF, n, nd=256)
    f_p = nm.train_fraction(p, -v, T_ON, T_OFF, n - 1, nd=256) if n > 1 else 0.0
    x_n, x_p = 1 - 2 * f_n, 1 - 2 * f_p
    q_plus = nm.q_rev(p, -v, x_n) + p["ps"] * (x_n - x_p)
    q_minus = nm.q_rev(p, -v, -1.0)
    return abs(q_plus - q_minus) * A_FE * 1e12


endur = {}
fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
for name, p in tails.items():
    sc = brentq(lambda f: nm.switched_fraction(dict(p, emp=p["emp"] * f, emn=p["emn"] * f), p["vimp"] + VC_HALF,
                                               T_QUARTER, nd=256) - 0.5, 0.3, 3.0)
    p["emp"] *= sc
    p["emn"] *= sc
    endur[name] = {"activation-field scale": sc}
ns = np.unique(np.round(np.logspace(0, 11, 45)))
styles = {"S4 tail (Alessandri 2019)": "-", "S2 tail (Alessandri 2018, Dagum refit)": "--",
          "S4 tail, tau0 = 2 ns (illustrative)": ":"}
for k, v in enumerate((0.1, 0.3, 0.5)):
    col = f"C{k}"
    d = [(float(r["n_reads"]), float(r["CMW_Q_pC"])) for r in end_rows if abs(float(r["V_R"]) + v) < 1e-9]
    dn, dq = np.array(d).T
    ax.semilogx(dn, dq, "o", color=col, mec="k", label=f"data, -{v} V")
    for name, p in tails.items():
        ax.semilogx(ns, [cmw(p, v, n) for n in ns], styles[name], color=col, lw=1.3)
        mod = np.array([cmw(p, v, n) for n in dn])
        m = dn <= 1e8
        endur[name][f"-{v} V"] = {"model_pC": dict(zip(map(float, dn), map(float, mod))),
                                  "data_pC": dict(zip(map(float, dn), map(float, dq))),
                                  "RMS_pC (n <= 1e8)": float(np.sqrt(np.mean((mod[m] - dq[m]) ** 2))),
                                  "RMS_pC (all)": float(np.sqrt(np.mean((mod - dq) ** 2)))}
ax.plot([], [], "k-", label="model, S4 tail, tau0 = 387 ns")
ax.plot([], [], "k--", label="model, S2 tail, tau0 = 236 ns")
ax.plot([], [], "k:", label="model, S4 tail, tau0 = 2 ns")
ax.set_xlabel("Number of reads")
ax.set_ylabel(r"CMW$_Q$ (pC)")
ax.set_title("Read endurance, Mukherjee IEDM 2023 Fig. 7 (digitized) vs. model", fontsize=10)
ax.set_ylim(0, 40)
ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper right")
fig.savefig(DOCS / "literature_read_endurance.png", dpi=150)
plt.close(fig)
OUT["read_endurance"] = endur

# ------------------------------------------------- 4. back-switching (Lancaster 2022)
bs_rows = read_csv("lancaster2022_backswitching.csv")
DELAYS = (3e-7, 1e-5, 1e-3, 1.0)
BS = {}
for r in bs_rows:
    BS.setdefault(float(r["t_write_s"]), {})[float(r["t_delay_s"])] = float(r["P_loss_uC_cm2"])
base11 = nm.generic_card(tfe=11e-9, vimp=0.0, c_d=generic["c_d"] * 10 / 11, dc=generic["dc"] * 10 / 11)


def ploss(vdep, tau_s, t_write, delays=DELAYS, nd=16):
    """Back-switched polarization [uC/cm^2] after a 3.5 V write of length t_write; sigma starts at 0."""
    q = dict(base11, vdep=vdep)
    dom = nm.make_domains(q, nd)
    s, h, sig, prev = np.full(nd, -1.0), np.zeros(nd), 0.0, 0.0
    for t in np.logspace(np.log10(t_write * 1e-5), np.log10(t_write), 40):
        dt = t - prev
        s, h = nm.step(q, dom, s, h, 3.5, dt, None, sig)
        x = float(dom.w @ s)
        sig = x + (sig - x) * np.exp(-dt / tau_s)
        prev = t
    x0, out, prev, want = float(dom.w @ s), {}, 0.0, list(delays)
    for t in np.logspace(-9, np.log10(max(delays)), 160):
        dt = t - prev
        s, h = nm.step(q, dom, s, h, 0.0, dt, None, sig)
        x = float(dom.w @ s)
        sig = x + (sig - x) * np.exp(-dt / tau_s)
        prev = t
        while want and t >= want[0] * (1 - 1e-9):
            out[want.pop(0)] = (x0 - x) * q["ps"] / UC
    return [out[d] for d in delays]


def cost(th):
    vd, ts = np.exp(th)
    return np.concatenate([np.array(ploss(vd, ts, tw)) - np.array([BS[tw][d] for d in DELAYS]) for tw in BS])


best = min(((float(np.sum(cost(np.log([a, b])) ** 2)), a, b) for a in (0.5, 0.7, 0.9) for b in (1e-4, 1e-3, 1e-2)))
sol = least_squares(cost, np.log(best[1:]), diff_step=0.02, max_nfev=60)
vd, ts_fit = np.exp(sol.x)
OUT["backswitching"] = {"fitted vdep [V]": float(vd), "fitted tau_s [s]": float(ts_fit),
                        "RMS [uC/cm2] (12 points, 2 parameters)": float(np.sqrt(np.mean(sol.fun ** 2))),
                        "constant vdep = 0.8 V, no screening: P_loss at the four delays (100 us write)":
                            ploss(0.8, 1e12, 100e-6)}
fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
dd = np.logspace(-7, 0, 40)
for k, tw in enumerate(sorted(BS)):
    col = f"C{k}"
    ax.semilogx(list(BS[tw]), list(BS[tw].values()), "o", color=col, mec="k", label=f"data, {tw*1e6:g} us write")
    ax.semilogx(dd, ploss(vd, ts_fit, tw, tuple(dd)), "-", color=col)
ax.semilogx(dd, ploss(0.8, 1e12, 100e-6, tuple(dd)), ":", color="0.4", label="constant V_dep = 0.8 V, no screening (100 us)")
ax.set_xlabel("Delay after the write (s)")
ax.set_ylabel(r"Back-switched polarization $P_{loss}$ ($\mu$C/cm$^2$)")
ax.set_title(f"Lancaster 2022 (digitized) vs. model with screening: V_dep = {vd:.2f} V, tau_s = {ts_fit*1e3:.1f} ms (fitted)",
             fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.savefig(DOCS / "literature_backswitching.png", dpi=150)
plt.close(fig)


# ------------------------------------------------- 5. series-capacitor reads (Duan IEDM 2025, Fig. 9)
def series_reads(p, nd, v_read, t_on, gap, c_s, reset, kmax, sample, dx_tol=0.01):
    """Capacitor in series with a linear capacitor c_s per FeCap area (floating node between them).

    Each read applies v_read for t_on; with reset=True the floating node is
    discharged to 0 V after every read, otherwise it floats during the gap.
    Returns the remaining state difference (x_up - x_down)/2 at the sampled reads.
    """
    dom = nm.make_domains(p, nd)
    n, ps = len(dom.w), p["ps"]

    def solve_v(vfl, x, q0, v):
        for _ in range(30):
            f0 = c_s * (vfl - v) - nm.q_term(p, v, x) - q0
            df = -c_s - (nm.q_term(p, v + 1e-3, x) - nm.q_term(p, v - 1e-3, x)) / 2e-3
            dv = -f0 / df
            v += dv
            if abs(dv) < 1e-6:
                break
        return v

    def march(s, h, vfl, q0, duration):
        t, dt, v = 0.0, duration / 4096, 0.0
        while t < duration * (1 - 1e-12):
            dt = min(dt, duration - t)
            x = float(dom.w @ s)
            v = solve_v(vfl, x, q0, v)
            s2, h2 = nm.step(p, dom, s, h, v, dt)
            if abs(float(dom.w @ s2) - x) > dx_tol and dt > duration * 1e-8:
                dt *= 0.5
                continue
            s, h, t, dt = s2, h2, t + dt, dt * 2.0
        return s, h

    res = {}
    for s0 in (1.0, -1.0):
        s, h = np.full(n, s0), np.zeros(n)
        q0 = -ps * float(dom.w @ s)
        rec = {}
        for k in range(1, kmax + 1):
            s, h = march(s, h, v_read, q0, t_on)
            if reset:
                q0 = -ps * float(dom.w @ s)
                s, h = nm.step(p, dom, s, h, 0.0, gap)
            else:
                s, h = march(s, h, 0.0, q0, gap)
            if k in sample:
                rec[k] = float(dom.w @ s)
        res[s0] = rec
    return {k: 0.5 * (res[1.0][k] - res[-1.0][k]) for k in sample}


AR = 0.83
c_s = 20 * EPS0 / 10e-9 / AR  # 10 nm HfO2 gate dielectric, eps_r = 20 assumed; per FeCap area
samples = [1, 2, 5, 10, 20, 50, 100, 200, 300]
ser = {}
for mode, reset in (("discharge", True), ("wait", False)):
    ser[mode] = series_reads(generic, 16, 2.5, 10e-6, 1e-6, c_s, reset, max(samples), samples)
OUT["series_reads"] = {"conditions": "2.5 V, 10 us reads (Fig. 5f conditions), 1 us between reads, AR = 0.83, "
                                     "gate eps_r = 20 assumed, generic card",
                       "state difference (x_up - x_down)/2": ser}
q2 = read_csv("duan2025_iedm_fig9.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for mode, mk in (("wait", "o"), ("discharge", "s")):
    rr = [r for r in q2 if r["mode"] == mode]
    nn = [float(r["n_reads"]) for r in rr]
    a1.loglog(nn, [float(r["I_state0_A"]) for r in rr], mk + "-", color="C0" if mode == "wait" else "C3", label=f"{mode}, state '0'")
    a1.loglog(nn, [float(r["I_state1_A"]) for r in rr], mk + ":", color="C0" if mode == "wait" else "C3", mfc="white",
              label=f"{mode}, state '1'")
    a2.semilogx(samples, [ser[mode][k] for k in samples], mk + "-", color="C0" if mode == "wait" else "C3", label=mode)
a1.set_xlabel("Number of reads")
a1.set_ylabel("Read current (A)")
a1.set_title("Duan IEDM 2025 Fig. 9 (digitized)", fontsize=10)
a1.legend(fontsize=8, frameon=False)
a2.set_xlabel("Number of reads")
a2.set_ylabel(r"Model: remaining state difference $(\bar x_\uparrow-\bar x_\downarrow)/2$")
a2.set_ylim(0, 1.05)
a2.set_title("Model: FeCap + series gate capacitor, 2.5 V / 10 us reads", fontsize=10)
a2.legend(fontsize=8, frameon=False)
fig.savefig(DOCS / "literature_series_read.png", dpi=150)
plt.close(fig)

(DOCS / "literature_comparison.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in OUT.items() if k != "read_endurance"}, indent=1, default=float))
for name, d in endur.items():
    print(name, {k: (v["RMS_pC (n <= 1e8)"], v["RMS_pC (all)"]) if isinstance(v, dict) else v for k, v in d.items()})
