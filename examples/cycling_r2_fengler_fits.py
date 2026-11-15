"""Joint fit of 2Pr and 2Ec during wake-up, Fengler et al. AEM 2017 Fig. 7a + 7c (part 1 of cycling_r2_study.py).

The read-out follows the paper: a 10 kHz triangular P-V loop at +-2.3 MV/cm after
cycling with 2.3 MV/cm, 100 kHz rectangular pulses; 2Pr and 2Ec are taken from
the total charge (:func:`fe_models.nls_multidomain.loop_figures`).  Hypotheses
for the temperature effect differ in which parameters are free (``HYP``).

Usage::

    PYTHONPATH=src python examples/cycling_r2_fengler_fits.py A B C D E   # one process per letter is fastest

Each letter writes docs/research/r2_fengler_fit_<letter>.json; cycling_r2_study.py reads them.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fe_models import nls_multidomain as nm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "literature"
OUTDIR = ROOT / "docs" / "research"


def _rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


TEMPS = [25, 50, 75, 100, 125]
D7A = {t: np.array([(float(r["cycles_nominal"]), float(r["2Pr_uC_cm2"])) for r in _rows("fengler2017_aem_fig7a_wakeup_2pr_T.csv")
                    if r["series"] == f"{t}C"]) for t in TEMPS}
D7C = {t: np.array([(float(r["cycles_nominal"]), float(r["2Ec_MV_cm"])) for r in _rows("fengler2017_aem_fig7c_wakeup_2ec_T.csv")
                    if r["series"] == f"{t}C"]) for t in TEMPS}
TFE = 7e-9
AMP = 2.3e8 * TFE
CYC = ([0, 10e-9, 5e-6, 5e-6 + 20e-9, 1e-5 - 10e-9, 1e-5], [0, AMP, AMP, -AMP, -AMP, 0])   # 2.3 MV/cm, 100 kHz
NS = [10, 100, 1000, 5000, 1e4, 5e4, 1e5]
SPR, SEC = 0.5, 0.03          # residual weights: uC/cm2 (2Pr), MV/cm (2Ec; read-out assumptions dominate the error)
NAMES = ["ps", "mul_ea", "wk_vsp", "wk_fp", "ln wk_nw", "wk_kw", "wk_ea", "tc_ps", "tc_ea"]
LO = [0.1, 0.3, 0.0, 0.0, -3, 0.05, 0.0, -3e-3, -5e-3]
HI = [0.4, 1.5, 2.0, 0.95, 20, 2.0, 1.5, 5e-3, 5e-3]
HYP = {
    "A": ("A: split + pinned, Arrhenius wake-up, P_s(T), E_c(T)", {}),
    "B": ("B: Arrhenius wake-up, no P_s(T) (tc_ps = 0)", {7: 0.0}),
    "C": ("C: P_s(T), no Arrhenius wake-up (wk_ea = 0)", {6: 0.0}),
    "D": ("D: no split (wk_vsp = 0)", {2: 0.0}),
    "E": ("E: split only, no pinned fraction (wk_fp = 0)", {3: 0.0}),
    # thermally activated switching lowers the coercive field at higher T: tc_ea <= 0
    "A0": ("A0: as A with tc_ea <= 0", {}),
    "B0": ("B0: as B with tc_ea <= 0", {7: 0.0}),
    "C0": ("C0: as C with tc_ea <= 0", {6: 0.0}),
    # uncompensated leakage in the read-out instead of P_s(T) and E_c(T): theta[7] = log10 g_oh [A/m2/V] at 25 C,
    # theta[8] = its activation energy [eV]; the loop integrates the leakage current (loop_figures(leak=True))
    "F": ("F: Arrhenius wake-up + uncompensated ohmic leakage (no P_s(T), no E_c(T))", {}),
    "F0": ("F0: uncompensated ohmic leakage only (no Arrhenius wake-up, no P_s(T), no E_c(T))", {6: 0.0}),
}
CAP_TC_EA = ("A0", "B0", "C0")
LEAK = ("F", "F0")
LEAK_LO, LEAK_HI = [-4.0, 0.0], [4.0, 1.5]
X0 = [[0.2, 0.6, 0.6, 0.1, np.log(300), 0.3, 0.4, 0.0, 0.0], [0.2, 0.7, 0.9, 0.0, np.log(30), 0.5, 0.5, 1e-3, -1e-3]]


def card(th, leak_mode=False, **kw):
    ps, mea, vsp, fp, lnw, kw_, ea, tcps, tcea = th
    if leak_mode:   # theta[7], theta[8] are the leakage parameters
        kw = dict(kw, g_oh=float(10 ** tcps), phi_oh=float(tcea))
        tcps, tcea = 0.0, 0.0
    # eps_r = 32 (paper, at the highest field); reversible branches of the generic card scaled to 7 nm
    return nm.generic_card(tfe=TFE, c_d=32 * nm.EPS0 / TFE, dc=0.03023 * 10 / 7, vw=2.236 * 0.7, vs=3.006 * 0.7, ps=ps,
                           mul_ea=mea, wk_vsp=vsp, wk_fp=fp, wk_nw=float(np.exp(lnw)), wk_kw=kw_, wk_ea=ea, tc_ps=tcps,
                           tc_ea=tcea, **kw)


def model(th, tc, ns=NS, readout="total", leak=False, leak_mode=False, **kw):
    """2Pr [uC/cm2] and 2Ec [MV/cm] from the 10 kHz read-out after n cycles at tc [C]."""
    p = card(th, leak_mode=leak_mode, **kw)
    leak = leak or leak_mode
    temp = 273.15 + tc
    h = nm.cycle_history(p, *CYC, ns, nd=16, temp=temp, sub=40)
    pr, ec = [], []
    for aw in h["a_w"]:
        hist = nm.History(nw=aw * p["wk_nw"])
        if readout == "total":
            L = nm.loop_figures(p, AMP, 1e4, nd=16, temp=temp, history=hist, npts=300, leak=leak)
            pr.append((L["pr_p"] - L["pr_m"]) * 100)
            ec.append((L["vc_p"] - L["vc_m"]) / TFE / 1e8)
        else:   # round-one read-out: mid crossing of the mean switching state, no dielectric charge
            r = nm.cycle_history(p, [0, 25e-6, 75e-6, 100e-6], [0, AMP, -AMP, 0], [0], nd=16, temp=temp, sub=200,
                                 history=hist)
            pr.append(np.nan)
            ec.append((r["vc_p"][0] - r["vc_m"][0]) / TFE / 1e8)
    return np.array(pr), np.array(ec)


def resid(th, leak_mode=False):
    out = []
    for tc in TEMPS:
        pr, ec = model(th, tc, leak_mode=leak_mode)
        out.append((pr - D7A[tc][:, 1]) / SPR)
        m = np.isin(NS, D7C[tc][:, 0])
        out.append((ec[m] - D7C[tc][:, 1]) / SEC)
    return np.concatenate(out)


def split_rms(th, leak_mode=False):
    pr_r, ec_r = [], []
    for tc in TEMPS:
        pr, ec = model(th, tc, leak_mode=leak_mode)
        pr_r += list(pr - D7A[tc][:, 1])
        m = np.isin(NS, D7C[tc][:, 0])
        ec_r += list(ec[m] - D7C[tc][:, 1])
    return float(np.sqrt(np.mean(np.square(pr_r)))), float(np.sqrt(np.mean(np.square(ec_r))))


def fit(letter, max_nfev=250):
    label, fixed = HYP[letter]
    free = [i for i in range(9) if i not in fixed]
    hi, lo = list(HI), list(LO)
    if letter in CAP_TC_EA:
        hi[8] = 0.0
    lm = letter in LEAK
    if lm:
        lo[7:9], hi[7:9] = LEAK_LO, LEAK_HI
    best = None
    t0 = time.time()
    for x0 in X0:
        full = np.array(x0, float)
        if lm:
            full[7:9] = [1.0, 0.4]
        for i, v in fixed.items():
            full[i] = v

        def r(xf, full=full):
            f = full.copy()
            f[free] = xf
            return resid(f, lm)

        res = least_squares(r, np.clip(full[free], [lo[i] for i in free], [hi[i] - 1e-12 for i in free]),
                            bounds=([lo[i] for i in free], [hi[i] for i in free]), diff_step=1e-3, max_nfev=max_nfev,
                            x_scale="jac")
        print(f"[{letter}] start done, cost {res.cost:.2f}, nfev {res.nfev}, {time.time() - t0:.0f} s", flush=True)
        if best is None or res.cost < best[0].cost:
            f = full.copy()
            f[free] = res.x
            best = (res, f)
    th = best[1]
    r_pr, r_ec = split_rms(th, lm)
    names = NAMES[:7] + (["log10 g_oh [A/m2/V] at 25 C", "phi_oh [eV]"] if lm else NAMES[7:])
    out = {"hypothesis": label, "theta": list(map(float, th)), "leak_mode": lm,
           **{n: float(v) for n, v in zip(names, th) if not n.startswith("ln")},
           "wk_nw": float(np.exp(th[4])), "free parameters": len(free),
           "RMS 2Pr [uC/cm2] (35 points)": r_pr, "RMS 2Ec [MV/cm] (18 points)": r_ec,
           "weighted cost": float(best[0].cost)}
    (OUTDIR / f"r2_fengler_fit_{letter}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[{letter}] {label}: RMS 2Pr {r_pr:.3f}, 2Ec {r_ec:.4f}", flush=True)
    return out


def load(letter):
    return json.loads((OUTDIR / f"r2_fengler_fit_{letter}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    for letter in sys.argv[1:] or list(HYP):
        fit(letter)
