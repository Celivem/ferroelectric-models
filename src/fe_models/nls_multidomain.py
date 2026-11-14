r"""Dynamic multi-domain NLS model of a ferroelectric capacitor (Python reference).

This is the reference implementation of the Verilog-A model
``spice/fecap_nls.va``; both implement the same equations (E1-E12 below), and
``spice/compare_ngspice.py`` checks the compiled model against this module.

Picture
-------
The film is split into ``nd`` classes of domains ordered by how hard they are
to switch (quantiles of an activation-field distribution).  Class ``k`` has a
weight ``w_k`` (``sum w_k = 1``), a switching state ``s_k`` in ``[-1, +1]``
and, for a Weibull exponent ``beta > 1``, a signed nucleation-progress
variable ``h_k``.  This is a mean-field, deterministic form of the dynamic NLS
models of Alessandri et al. (IEEE TED 66, 3527, 2019) and Deng et al.
(VLSI 2020): it describes the expectation value for a large number of grains.

Equations (SI units: V, m, s, C/m^2, F/m^2, A/m^2)
-------------------------------------------------
* (E5)  mean state and polarization: :math:`\bar x=\sum_k w_k s_k`,
  :math:`P = P_s m_P \bar x`.
* (E1)  drive of class k: :math:`u_k=\rho V-V_{dep}(\bar x-\sigma)-V_{imp}-\delta_k`.
* (E2)  one-directional Merz-type rates
  :math:`R_k^\pm=\tau_0^{-1}\exp[-(E_k^\pm t_{fe}/|u_k|)^\alpha]` for
  :math:`\pm u_k>0`, zero otherwise.
* (E3)  :math:`ds_k/dt=r_k^+(1-s_k)-r_k^-(1+s_k)`,
  :math:`r_k^\pm=\beta R_k^\pm (h_k^\pm)^{\beta-1}`.
* (E3b) :math:`dh_k/dt=R_k^+[1-(h_k^+/h_m)^4]-R_k^-[1-(h_k^-/h_m)^4]
  -h_k/\tau_h^{eff}+\tau_x^{-1}[G(u_k-u_x)h_k^- - G(-u_k-u_x)h_k^+]`
  with gated relaxation :math:`\tau_h^{eff}=\tau_h[1+((V-V_{imp})/V_g)^4]`.
* (E4)  activation fields from the Dagum quantile
  :math:`E(q)=b[q^{-1/p}-1]^{-1/a}`, sampled on a logit grid.
* (E6, E7) reversible charge: linear dielectric, two state-dependent arctan
  branches (split into an instantaneous fraction ``frev`` and a low-pass part
  driven by :math:`\tau_r dy/dt = V-y`), domain-wall and slow terms.
* (E8)  terminal current :math:`I=A\,d(P+q_{rev})/dt + A J_L(V,T)`.
* (E9)  leakage: ohmic + Poole-Frenkel + Schottky + trap-assisted tunnelling
  (effective form), one parameter set per polarity.
* (E12) screening state :math:`\tau_s d\sigma/dt=\bar x-\sigma` (back-switching).

Under a constant drive and from a saturated state with ``h = 0``,
:math:`h_k=R_k t` and (E3) reduces to the single-pulse NLS form
:math:`f(t)=\sum_k w_k\{1-\exp[-(R_k t)^\beta]\}` (see
:func:`switched_fraction`).

Solvers
-------
* :func:`simulate` -- arbitrary ``V(t)``, implicit Radau integration of E3-E12.
* :func:`step` -- closed-form update for a constant voltage over ``dt``.
* :func:`switched_fraction`, :func:`train_fraction` -- closed forms for one
  rectangular pulse and for ``n`` identical unipolar pulses.
* :func:`bipolar_pairs` -- closed form for ``k`` pairs of (+V, 0, -V, 0) pulses.
* :func:`repeat_waveform` -- state after ``n`` repetitions of one period.
* :func:`march` -- fixed-grid theta-method (trapezoid / backward Euler), to
  check time-step convergence the way a circuit simulator would integrate.
* :func:`rhs` -- right-hand side of the state equations.

The default parameters (:data:`GENERIC`, :func:`generic_card`) are a generic,
literature-based card; they are not fitted to any particular device.  See
``spice/README.md`` for the source of every value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

Q_E = 1.602176634e-19
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12

X_MAX = 60.0     #: rates below exp(-60)/tau0 are set to exactly 0
H_MAX = 5.0      #: saturation h_m of the progress variable (exp(-h_m^beta) < 1e-10)
H_CLIP = 6.0     #: cap of the (h/h_m)^4 saturation argument (only reached during iterations)
H_S = 1e-6       #: h^(beta-1) is made linear below this value (finite slope at h = 0)
TAU_X = 1e-10    #: time constant of the reversal reset of h [s]
U_XM = 0.1       #: reversal threshold: |u| > |V_imp| + U_XM counts as a reversed drive [V]
U_XW = 0.01      #: smoothing width of the reversal gate [V]
V_GATE = 0.1     #: h relaxes only when |V - V_imp| is below about this voltage [V]
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True)
class Param:
    """One entry of a parameter card."""

    value: float
    unit: str
    va_name: str
    symbol: str
    meaning: str
    source: str


# Reference keys used in the ``source`` fields; full citations in spice/README.md.
GENERIC: dict[str, Param] = {
    "area": Param(1e-10, "m^2", "area", "A", "capacitor area",
                  "illustrative (10 um x 10 um)"),
    "tfe": Param(1e-8, "m", "tfe", "t_fe", "ferroelectric thickness",
                 "illustrative; 8-10 nm is typical of the cited HZO films [Alessandri2018, Alessandri2019]"),
    "t0": Param(298.15, "K", "t0", "T_0", "reference temperature", "25 C (convention)"),
    "s0": Param(-1.0, "1", "s0", "s_0", "initial state of every class (instance parameter)", "instance setting"),
    "ps": Param(0.229, "C/m^2", "ps", "P_s", "saturation polarization",
                "P_R = 22.9 uC/cm^2, Table I of [Alessandri2019] (W/HZO 8.3 nm/W)"),
    "emp": Param(1.775e8, "V/m", "emp", "b^+", "Dagum scale of the activation field (+ direction)",
                 "Dagum fit to the GB2 g(E'a) of [Alessandri2019] Table I (a = 12.1, b = 1.79 MV/cm, p = 0.691, q = 0.633)"),
    "emn": Param(1.775e8, "V/m", "emn", "b^-", "Dagum scale of the activation field (- direction)",
                 "same as emp ([Alessandri2019] does not separate polarities)"),
    "ap": Param(8.68, "1", "ap", "a^+", "Dagum shape (core width), + direction", "same Dagum fit as emp"),
    "an": Param(8.68, "1", "an", "a^-", "Dagum shape (core width), - direction", "same Dagum fit as emp"),
    "pt": Param(1.136, "1", "pt", "p", "Dagum shape; low-field tail G ~ (E/b)^(a p)",
                "same Dagum fit as emp; tail exponent a*p = 9.9 (the [Alessandri2018] fit gives about 87)"),
    "alpha": Param(4.11, "1", "alpha", "alpha", "Merz-type exponent",
                   "[Alessandri2019] Table I; 3.7-5.5 in [Alessandri2018, AlessandriIEDM2018]"),
    "beta": Param(2.07, "1", "beta", "beta", "Weibull exponent (nucleation incubation)",
                  "[Alessandri2019] Table I; 2.06 in [Alessandri2018]; 2 in [Jo2007]"),
    "tau0": Param(3.87e-7, "s", "tau0", "tau_0", "attempt time (tau_inf of the Merz-type law)",
                  "[Alessandri2019] Table I (387 ns); 236 ns in [Alessandri2018]; strongly device dependent"),
    "tau_h": Param(3.7e-6, "s", "tau_h", "tau_h", "relaxation time of h without drive",
                   "geometric mean of 1.7 us and 8.3 us, the single-exponential times implied by gamma = 0.55 "
                   "(1 us off) and 0.3 (10 us off) in [Alessandri2019] Eq. 14"),
    "vimp": Param(-0.08, "V", "vimp", "V_imp", "built-in (imprint) voltage; loop centre",
                  "V_OS = 80 mV in [Alessandri2019] (V_FE = V_A + V_OS)"),
    "rho": Param(1.0, "1", "rho", "rho", "voltage division C_i/(C_i+C_f) by an interface layer",
                 "1 = no interface layer (limit of the charge balance in [Tian2025])"),
    "vdep": Param(0.0, "V", "vdep", "V_dep", "depolarization voltage (back-switching)", "off by default"),
    "tau_s": Param(1e-4, "s", "tau_s", "tau_s", "screening time constant",
                   "illustrative; inactive while vdep = 0"),
    "sb": Param(0.0, "V", "sb", "s_b", "spread of per-class offsets delta_k", "off by default"),
    "c_d": Param(0.02055, "F/m^2", "c_d", "C_d", "(1) linear dielectric capacitance (eps_r about 23 at 10 nm)",
                 "fit of one Lorentzian branch to C-V values read from Fig. 2 (Dev A, TiN/TiN) of "
                 "[Mukherjee2023EDL]; width from [Mukherjee2023IEDM] Fig. 5a (digitized, approx.)"),
    "dc": Param(0.03023, "F/m^2", "dcap", "Delta C", "(2) peak height of each reversible branch",
                "same fit as c_d"),
    "frev": Param(1.0, "1", "frev", "f_rev", "instantaneous fraction of term (2)",
                  "placeholder; the time scale is unresolved in the literature ([Baigol2026] vs [Mukherjee2023IEDM])"),
    "c_w": Param(0.0, "F/m^2", "c_w", "C_w", "(3) reversible domain-wall capacitance, weighted (1 - x^2)",
                 "off by default"),
    "vw": Param(2.236, "V", "vw", "V_w", "half width of each branch",
                "2.24 MV/cm x 10 nm, from a fit to [Mukherjee2023IEDM] Fig. 5a (digitized, approx.)"),
    "vs": Param(3.006, "V", "vs", "V_s", "distance of each branch peak from V_i",
                "same fit as c_d (3.0 MV/cm x 10 nm)"),
    "vi": Param(0.0, "V", "vi", "V_i", "crossing voltage of the two branches",
                "about 0 for symmetric electrodes ([Mukherjee2023EDL] Dev A)"),
    "c_r": Param(0.0, "F/m^2", "c_r", "C_r", "state-independent slow capacitance", "off by default"),
    "tau_r": Param(1e-7, "s", "tau_r", "tau_r", "low-pass time constant of y",
                   "inactive while frev = 1 and c_w = c_r = 0; order of magnitude from the >2 MHz roll-off "
                   "of the small-signal window in [Baigol2026] (inference)"),
    "g_oh": Param(1e-4, "A/m^2/V", "g_oh", "G_ohm", "ohmic leakage conductance per area",
                  "low-voltage end of the 0-2 V leakage range in [Fehlings2025b] Fig. 6a (digitized, approx.)"),
    "phi_oh": Param(0.0, "eV", "phi_oh", "phi_ohm", "activation energy of G_ohm", "placeholder (no source found)"),
    "spf_p": Param(4.8e-12, "S/m", "spf_p", "sigma_PF^+", "Poole-Frenkel prefactor, V > 0",
                   "scaled so that J(2 V) = 0.1 A/m^2, the upper end of [Fehlings2025b] Fig. 6a (digitized, approx.)"),
    "spf_n": Param(4.8e-12, "S/m", "spf_n", "sigma_PF^-", "Poole-Frenkel prefactor, V < 0", "same as spf_p"),
    "phit_p": Param(0.68, "eV", "phit_p", "phi_t^+", "Poole-Frenkel trap depth, V > 0",
                    "phi_tr,fe = 0.68 eV, Table 1 of [Fehlings2024] (model fit, W/HZO 9.8 nm/W)"),
    "phit_n": Param(0.68, "eV", "phit_n", "phi_t^-", "Poole-Frenkel trap depth, V < 0", "same as phit_p"),
    "eps_op": Param(70.0, "1", "eps_op", "eps_op", "permittivity in the barrier lowering",
                    "eps_fe = 70 as used in the PF term of [Fehlings2024]"),
    "js_p": Param(0.0, "A/m^2", "js_p", "J_S^+", "Schottky prefactor at T_0, V > 0", "off by default"),
    "js_n": Param(0.0, "A/m^2", "js_n", "J_S^-", "Schottky prefactor at T_0, V < 0", "off by default"),
    "phib_p": Param(0.48, "eV", "phib_p", "phi_b^+", "Schottky barrier, V > 0", "illustrative; inactive while js_p = 0"),
    "phib_n": Param(0.48, "eV", "phib_n", "phi_b^-", "Schottky barrier, V < 0", "illustrative; inactive while js_n = 0"),
    "jt_p": Param(0.0, "A/m^2", "jt_p", "J_T^+", "trap-assisted tunnelling prefactor, V > 0", "off by default"),
    "jt_n": Param(0.0, "A/m^2", "jt_n", "J_T^-", "trap-assisted tunnelling prefactor, V < 0", "off by default"),
    "et_p": Param(1e9, "V/m", "et_p", "E_T^+", "TAT characteristic field, V > 0", "illustrative; inactive while jt_p = 0"),
    "et_n": Param(1e9, "V/m", "et_n", "E_T^-", "TAT characteristic field, V < 0", "illustrative; inactive while jt_n = 0"),
    "e0_leak": Param(1e6, "V/m", "e0_leak", "E_0", "field regularization near 0 V",
                     "numerical (10 mV across 10 nm), not a physical quantity"),
    "tc_ea": Param(0.0, "1/K", "tc_ea", "tc_E", "linear temperature coefficient of the activation field",
                   "placeholder (no quantitative source found)"),
    "tc_ps": Param(0.0, "1/K", "tc_ps", "tc_P", "linear temperature coefficient of P_s", "placeholder"),
    "tc_tau": Param(0.0, "1/K", "tc_tau", "tc_tau", "linear temperature coefficient of tau_0", "placeholder"),
    "tc_c": Param(0.0, "1/K", "tc_c", "tc_C", "linear temperature coefficient of terms (1)+(2)", "placeholder"),
    "mul_ps": Param(1.0, "1", "mul_ps", "m_P", "multiplier on P_s (variation / fatigue hook)", "hook"),
    "mul_ea": Param(1.0, "1", "mul_ea", "m_E", "multiplier on the activation field (variation hook)", "hook"),
    "nd": Param(16, "1", "FE_ND", "N_d", "number of domain classes (compile-time macro in Verilog-A)", "numerical"),
    "y_max": Param(9.0, "1", "y_max", "y_m", "depth of the logit quantile grid (smallest q about 1.2e-4)", "numerical"),
    "grid_a": Param(1.5, "1", "grid_a", "a_g", "sinh stretching of the quantile grid", "numerical"),
}


def generic_card(**overrides) -> dict:
    """Return the generic parameter card as a plain ``{name: value}`` dict."""
    p = {k: v.value for k, v in GENERIC.items()}
    unknown = set(overrides) - set(p)
    if unknown:
        raise KeyError(f"unknown parameters: {sorted(unknown)}")
    p.update(overrides)
    return p


def spice_model_card(model_name: str = "fecap_generic", module: str = "fecap_nls") -> str:
    """Return the ngspice ``.model`` card for :data:`GENERIC` with one source comment per value.

    ``area`` and ``s0`` are instance parameters and ``nd`` is a compile-time
    macro, so they are listed as comments only.
    """
    lines = [
        "* Generic, literature-based parameter card for fecap_nls (not fitted to any device).",
        "* Generated by fe_models.nls_multidomain.spice_model_card(); keep in sync with GENERIC.",
        "* Instance parameters (put them on the N line): area, s0, sg0.",
        f"*   area default {GENERIC['area'].value:g} m^2 ({GENERIC['area'].source})",
        f"* FE_ND = {GENERIC['nd'].value} domain classes is a compile-time macro (see build_osdi.py).",
        f".model {model_name} {module}",
    ]
    for name, prm in GENERIC.items():
        if name in ("area", "s0", "nd"):
            continue
        lines.append(f"+ {prm.va_name}={prm.value:.6g}")
        lines.append(f"* ^ [{prm.unit}] {prm.meaning}. Source: {prm.source}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- domains
def dagum_quantile(q, b, a, pt):
    """Inverse of the Dagum CDF :math:`G(E)=[1+(E/b)^{-a}]^{-p}` (GB2 with q = 1)."""
    q = np.clip(np.asarray(q, dtype=float), 1e-300, 1.0 - 1e-15)
    return b * (q ** (-1.0 / pt) - 1.0) ** (-1.0 / a)


def dagum_cdf(E, b, a, pt):
    """Dagum CDF :math:`G(E)=[1+(E/b)^{-a}]^{-p}`."""
    E = np.asarray(E, dtype=float)
    return (1.0 + (E / b) ** (-a)) ** (-pt)


def gb2_cdf(E, a, b, p, q):
    """CDF of the generalized beta distribution of the second kind GB2(a, b, p, q)."""
    from scipy.special import betainc

    z = (np.asarray(E, dtype=float) / b) ** a
    return betainc(p, q, z / (1.0 + z))


def quantile_grid(nd, y_max=9.0, a=1.5):
    """Mid-point quantiles and weights of the logit grid.

    The cell edges are :math:`y_j=y_m\\sinh(a\\xi_j)/\\sinh a` in
    :math:`y=\\ln[q/(1-q)]` with :math:`\\xi_j` uniform in [-1, 1]; the outer
    edges are moved to q = 0 and q = 1 so the weights sum to one.  Compared with
    equal-probability cells this resolves the low-field tail of the distribution.
    """
    xi = np.linspace(-1.0, 1.0, nd + 1)
    y_edge = y_max * np.sinh(a * xi) / np.sinh(a)
    q_edge = 1.0 / (1.0 + np.exp(-y_edge))
    q_edge[0], q_edge[-1] = 0.0, 1.0
    w = np.diff(q_edge)
    y_mid = 0.5 * (y_edge[:-1] + y_edge[1:])
    q_mid = 1.0 / (1.0 + np.exp(-y_mid))
    return q_mid, w


@dataclass
class Domains:
    q: np.ndarray      #: mid-point quantile of each class
    w: np.ndarray      #: weight of each class
    va_p: np.ndarray   #: activation voltage toward + [V]
    va_n: np.ndarray   #: activation voltage toward - [V]
    delta: np.ndarray  #: per-class offset [V]


def make_domains(p: dict, nd: int | None = None, temp: float | None = None) -> Domains:
    """Sample the activation-field distribution into ``nd`` classes (E4)."""
    nd = int(nd if nd is not None else p["nd"])
    temp = p["t0"] if temp is None else temp
    q, w = quantile_grid(nd, p.get("y_max", 9.0), p.get("grid_a", 1.5))
    te = (1.0 + p["tc_ea"] * (temp - p["t0"])) * p["mul_ea"] * p["tfe"]
    va_p = dagum_quantile(q, p["emp"], p["ap"], p["pt"]) * te
    va_n = dagum_quantile(q, p["emn"], p["an"], p["pt"]) * te
    sb = p.get("sb", 0.0)
    if sb > 0:
        phi = np.clip((np.arange(1, nd + 1) * 0.6180339887498949) % 1.0, 1e-3, 1 - 1e-3)
        delta = sb * np.log(phi / (1.0 - phi))
    else:
        delta = np.zeros(nd)
    return Domains(q, w, va_p, va_n, delta)


# ------------------------------------------------------------------ rates
def rate(u, va, tau0, alpha):
    """One-directional Merz-type rate (E2): exp[-(va/u)^alpha]/tau0 for u > 0, else 0."""
    u = np.asarray(u, dtype=float)
    r = np.zeros(np.broadcast(u, va).shape)
    z = np.divide(u, va, out=np.zeros_like(r), where=u > 0)
    ok = z > X_MAX ** (-1.0 / alpha)
    r[ok] = np.exp(-(z[ok] ** (-alpha))) / tau0
    return r


def rates(p, dom: Domains, v, x, temp):
    """(R+, R-) of every class at voltage ``v``; ``x`` is :math:`\\bar x-\\sigma` (used only if vdep != 0)."""
    vf = p["rho"] * v - p["vdep"] * x
    u = vf - p["vimp"] - dom.delta
    tau0 = p["tau0"] * (1.0 + p["tc_tau"] * (temp - p["t0"]))
    return rate(u, dom.va_p, tau0, p["alpha"]), rate(-u, dom.va_n, tau0, p["alpha"])


def ps_of_t(p, temp):
    return p["ps"] * p["mul_ps"] * (1.0 + p["tc_ps"] * (temp - p["t0"]))


def tau_h_eff(p, v):
    """Gated relaxation time of h: :math:`\\tau_h[1+((V-V_{imp})/V_g)^4]`."""
    th = p["tau_h"]
    if th <= 0:
        raise ValueError("tau_h must be > 0; use a very large value or inf for no relaxation")
    if not np.isfinite(th):
        return np.inf
    return th * (1.0 + ((v - p["vimp"]) / V_GATE) ** 4)


def relax_factor(p, dt, v=0.0):
    th = tau_h_eff(p, v)
    return np.exp(-dt / th) if np.isfinite(th) else 1.0


def rev_gates(p, u):
    """Reversal gates (G+, G-) of E3b; they depend on the field direction only, not on the rate."""
    ux = abs(p["vimp"]) + U_XM
    return 0.5 * (1.0 + np.tanh((u - ux) / U_XW)), 0.5 * (1.0 + np.tanh((-u - ux) / U_XW))


def h_pow(ha, beta):
    """:math:`h^{\\beta-1}`, made linear below H_S so its slope at h = 0 stays finite for beta < 2."""
    w = np.maximum(ha, H_S)
    return ha / w * w ** (beta - 1.0)


# -------------------------------------------------------- reversible charge
def branches(p, v):
    """The two reversible branches of term (2): (g_up, g_dn), both zero at V = 0.

    g_up (up-polarized domains) peaks at V_i - V_s, i.e. it softens toward
    negative voltage where those domains approach switching; g_dn peaks at
    V_i + V_s.
    """
    vw, vs, vi = p["vw"], p["vs"], p["vi"]
    up = p["dc"] * vw * (np.arctan((v - vi + vs) / vw) - np.arctan((-vi + vs) / vw))
    dn = p["dc"] * vw * (np.arctan((v - vi - vs) / vw) - np.arctan((-vi - vs) / vw))
    return up, dn


def q_rev(p, v, x, y=None, temp=None):
    """Reversible charge density (E6) [C/m^2].

    ``q_rev = C_d V + f_rev [gbar(V) + x g_d(V)] + (1 - f_rev) [gbar(y) + x g_d(y)]
    + [C_w (1 - x^2) + C_r] y``, with ``gbar = (g_up + g_dn)/2`` and
    ``g_d = (g_up - g_dn)/2``.  It vanishes at V = y = 0, so the charge left at
    0 V is the polarization alone.  ``y`` defaults to ``v`` (low-pass settled).
    """
    temp = p["t0"] if temp is None else temp
    k = 1.0 + p["tc_c"] * (temp - p["t0"])
    y = v if y is None else y
    up, dn = branches(p, v)
    gbar, gd = 0.5 * (up + dn), 0.5 * (up - dn)
    frev = p["frev"]
    if frev < 1.0:
        # the whole term (2) is split, so each part has positive capacitance (passivity)
        upy, dny = branches(p, y)
        gbar = frev * gbar + (1.0 - frev) * 0.5 * (upy + dny)
        gd = frev * gd + (1.0 - frev) * 0.5 * (upy - dny)
    slow = (p["c_w"] * (1.0 - x * x) + p["c_r"]) * (y if (p["c_w"] or p["c_r"]) else 0.0)
    return k * (p["c_d"] * v + gbar + x * gd) + slow


def q_term(p, v, x, y=None, temp=None):
    """Terminal charge density P + q_rev [C/m^2] (without the leakage integral)."""
    temp = p["t0"] if temp is None else temp
    return ps_of_t(p, temp) * x + q_rev(p, v, x, y, temp)


# ----------------------------------------------------------------- leakage
def j_leak(p, v, temp=None):
    """Leakage current density (E9) [A/m^2]: ohmic + Poole-Frenkel + Schottky + TAT."""
    temp = p["t0"] if temp is None else temp
    v = np.asarray(v, dtype=float)
    pos = v >= 0

    def pol(a, b):
        return np.where(pos, p[a], p[b])

    vth = K_B * temp / Q_E
    vth0 = K_B * p["t0"] / Q_E
    e = v / p["tfe"]
    e_s = np.sqrt(e**2 + p["e0_leak"] ** 2)
    root = np.sqrt(e_s) - np.sqrt(p["e0_leak"])
    beta_s = np.sqrt(Q_E / (4 * np.pi * EPS0 * p["eps_op"]))

    def arr(phi):
        return np.exp(np.asarray(phi) * (1.0 / vth0 - 1.0 / vth))

    s = np.tanh(v / vth)
    j = p["g_oh"] * v * arr(p["phi_oh"])
    j = j + s * pol("spf_p", "spf_n") * e_s * arr(pol("phit_p", "phit_n")) * np.exp(
        np.minimum(2 * beta_s * root / vth, 80.0))
    j = j + s * pol("js_p", "js_n") * (temp / p["t0"]) ** 2 * arr(pol("phib_p", "phib_n")) * (
        np.exp(np.minimum(beta_s * root / vth, 80.0)) - 1.0)
    j = j + s * pol("jt_p", "jt_n") * np.exp(-pol("et_p", "et_n") / e_s)
    return j


# ----------------------------------------------------------- closed forms
def step(p, dom: Domains, s, h, v, dt, temp=None, sigma=0.0):
    """Closed-form update of (s, h) for a constant voltage ``v`` held for ``dt``.

    With vdep != 0 the depolarization term uses :math:`\\bar x-\\sigma` at the
    start of the interval (operator splitting; keep dt small).  A reversed drive
    resets h before the interval, as the tau_x term of E3b does.
    """
    temp = p["t0"] if temp is None else temp
    x = float(dom.w @ s) - sigma if p["vdep"] != 0 else 0.0
    rp, rn = rates(p, dom, v, x, temp)
    r = rp - rn
    beta = p["beta"]
    target = np.where(r > 0, 1.0, np.where(r < 0, -1.0, 0.0))
    ra = np.abs(r)
    if beta == 1.0:
        lam = ra * dt
        h_new = h
    else:
        gu = p["rho"] * v - (p["vdep"] * x if p["vdep"] != 0 else 0.0) - p["vimp"] - dom.delta
        ux = abs(p["vimp"]) + U_XM
        h = np.where((gu * h < 0) & (np.abs(gu) > ux), 0.0, h)
        ha = np.abs(h)
        g = relax_factor(p, dt, v)
        if g == 1.0:
            ha_new = ha + ra * dt
            lam = ha_new**beta - ha**beta
        elif dt < 1e-3 * tau_h_eff(p, v):
            ha_new = ha * g + ra * dt
            lam = ha_new**beta - (ha * g) ** beta
        else:
            th = tau_h_eff(p, v)
            ha_new = ha * g + ra * th * (1.0 - g)
            # Lambda = int beta R h(t)^(beta-1) dt with h(t) the relaxing solution;
            # Gauss quadrature on [0, min(dt, tau)] and the rest
            lam = np.zeros_like(ha)
            active = ra > 0
            if active.any():
                t_split = min(dt, th)
                for a, b in ((0.0, t_split), (t_split, dt)):
                    if b <= a:
                        continue
                    for ni, wi in zip(_GL_NODES, _GL_WEIGHTS):
                        ti = 0.5 * (b - a) * ni + 0.5 * (b + a)
                        e = np.exp(-ti / th)
                        h_i = ha[active] * e + ra[active] * th * (1.0 - e)
                        lam[active] += 0.5 * (b - a) * wi * beta * ra[active] * h_i ** (beta - 1.0)
        h_new = np.where(target != 0, target * ha_new, np.sign(h) * ha_new)
    s_new = np.where(target != 0, target + (s - target) * np.exp(-lam), s)
    return s_new, h_new


def train_lambda(p, r, t_on, t_off, n):
    """Accumulated hazard of ``n`` identical unipolar pulses; ``r`` is |R| per class."""
    beta = p["beta"]
    a = r * t_on
    n = float(n)
    if beta == 1.0:
        return a * n
    g = relax_factor(p, t_off)
    if g >= 1.0 or (1.0 - g) * n < 1e-3:
        return (a * n) ** beta
    if g == 0.0:
        return n * a**beta
    n0 = int(min(n, np.ceil(30.0 / (1.0 - g))))
    k = np.arange(1, n0 + 1, dtype=float)
    h_end = np.outer(a, (1.0 - g**k) / (1.0 - g))
    h_start = g * np.hstack([np.zeros((len(a), 1)), h_end[:, :-1]])
    lam = np.sum(h_end**beta - h_start**beta, axis=1)
    if n > n0:
        h_ss = a / (1.0 - g)
        lam = lam + (n - n0) * (h_ss**beta - (g * h_ss) ** beta)
    return lam


def switched_fraction(p, v, t_on, temp=None, nd=None):
    """Fraction switched by one rectangular pulse from the opposite saturated state (h = 0).

    This is the single-pulse NLS form :math:`\\sum_k w_k\\{1-\\exp[-(R_k t)^\\beta]\\}`.
    """
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    rp, rn = rates(p, dom, v, 0.0, temp)
    lam = (np.abs(rp - rn) * t_on) ** p["beta"]
    return float(dom.w @ (1.0 - np.exp(-lam)))


def train_fraction(p, v, t_on, t_off, n, temp=None, nd=None):
    """Fraction switched by ``n`` identical rectangular pulses from the opposite saturated state."""
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    rp, rn = rates(p, dom, v, 0.0, temp)
    lam = train_lambda(p, np.abs(rp - rn), t_on, t_off, n)
    return float(dom.w @ (1.0 - np.exp(-lam)))


def _pair_lambdas(p, dom, v, t_on, gap, temp, h_rule):
    beta = p["beta"]
    rp1, rn1 = rates(p, dom, +v, 0.0, temp)
    rp2, rn2 = rates(p, dom, -v, 0.0, temp)
    rup = rp1 + rp2
    rdn = rn1 + rn2
    ux = abs(p["vimp"]) + U_XM
    u1, u2 = p["rho"] * v - p["vimp"] - dom.delta, -p["rho"] * v - p["vimp"] - dom.delta
    rev = (u1 > ux) & (-u2 > ux)

    def per_pair():
        if beta == 1.0:
            return rup * t_on, rdn * t_on
        return (rp1 * t_on) ** beta + (rp2 * t_on) ** beta, (rn1 * t_on) ** beta + (rn2 * t_on) ** beta

    if h_rule == "keep" and beta != 1.0:
        return lambda k: (train_lambda(p, rup, t_on, 2 * gap + t_on, k),
                          train_lambda(p, rdn, t_on, 2 * gap + t_on, k))
    lp, ln = per_pair()
    if beta == 1.0 or rev.all():
        return lambda k: (k * lp, k * ln)

    def mixed(k):
        tp = train_lambda(p, rup, t_on, 2 * gap + t_on, k)
        tn = train_lambda(p, rdn, t_on, 2 * gap + t_on, k)
        return np.where(rev, k * lp, tp), np.where(rev, k * ln, tn)

    return mixed


def bipolar_pairs(p, v, t_on, gap, k, temp=None, nd=None, h_rule="reset"):
    """Remaining fraction of the two-state difference after ``k`` (+V, 0, -V, 0) pairs.

    Returns :math:`\\sum_k w_k\\exp(-\\Lambda^+-\\Lambda^-)` (1 = nothing lost).
    ``h_rule="reset"`` is the model's rule (h reset on reversal); ``"keep"``
    lets both directions accumulate independently (the extreme case discussed
    in Alessandri et al. 2019).  Closed form, vdep = 0.
    """
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    lp, ln = _pair_lambdas(p, dom, v, t_on, gap, temp, h_rule)(float(k))
    return float(dom.w @ np.exp(-(lp + ln)))


def pund(p, v, tw, temp=None, nd=None, t_gap=1e-6):
    """Switched charge density [C/m^2] of the four PUND pulses (closed form, polarization only)."""
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    ps = ps_of_t(p, temp)
    s = np.full(len(dom.w), -1.0)
    h = np.zeros(len(dom.w))
    res = {}
    for name, vv in (("P", v), ("U", v), ("N", -v), ("D", -v)):
        s2, h = step(p, dom, s, h, vv, tw, temp)
        res[name] = float(ps * dom.w @ (s2 - s))
        s = s2
        s, h = step(p, dom, s, h, 0.0, t_gap, temp)
    return res


def repeat_waveform(p, t, v, n_rep, s0, nd=None, temp=None, carry_h=False):
    """Mean state after ``n_rep`` repetitions of one period ``(t, v)`` of a waveform.

    Within one period every s_k is affine in its start value (s' = A s + B), so
    ``n_rep`` periods have a closed form.  ``carry_h=False`` starts h at 0 in
    every period (exact for bipolar waveforms, where each reversal resets h, or
    for off times much longer than tau_h; a lower bound otherwise).
    ``carry_h=True`` uses the periodic steady state of h (an upper bound for
    unipolar trains with off times shorter than tau_h).
    """
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    n = len(dom.w)
    t, v = np.asarray(t, float), np.asarray(v, float)

    def one(s, h0):
        ss, hh = np.array(s, float), np.array(h0, float)
        for a, b, va, vb in zip(t[:-1], t[1:], v[:-1], v[1:]):
            ss, hh = step(p, dom, ss, hh, 0.5 * (va + vb), b - a, temp)
        return ss, hh

    h0 = np.zeros(n)
    if carry_h:
        for _ in range(500):
            _, h1 = one(np.zeros(n), h0)
            if np.max(np.abs(h1 - h0)) < 1e-9 * (1.0 + np.max(np.abs(h1))):
                h0 = h1
                break
            h0 = h1
    sp, _ = one(np.ones(n), h0)
    sn, _ = one(-np.ones(n), h0)
    a = np.clip(0.5 * (sp - sn), 0.0, 1.0)
    b = 0.5 * (sp + sn)
    s0 = np.broadcast_to(np.asarray(s0, float), (n,))
    n_rep = float(n_rep)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(1.0 - a) > 1e-14, (1.0 - a**n_rep) / (1.0 - a), n_rep)
    s_end = s0 * a**n_rep + b * ratio
    return float(dom.w @ s_end)


# ------------------------------------------------------------ ODE solvers
def rhs(p, dom: Domains, s, h, y, sig, v, temp):
    """Right-hand side of E3, E3b, E7, E12: returns (ds/dt, dh/dt, dy/dt, dsigma/dt)."""
    n = len(dom.w)
    x = float(dom.w @ s)
    rp, rn = rates(p, dom, v, x - sig, temp)
    th = p["tau_h"]
    relax = 0.0 if not np.isfinite(th) else 1.0 / tau_h_eff(p, v)
    beta = p["beta"]
    if beta == 1.0:
        hp, hn, dh = rp, rn, np.zeros(n)
    else:
        hpos, hneg = np.maximum(h, 0.0), np.maximum(-h, 0.0)
        hp = beta * rp * h_pow(hpos, beta)
        hn = beta * rn * h_pow(hneg, beta)
        gp, gn = rev_gates(p, p["rho"] * v - p["vdep"] * (x - sig) - p["vimp"] - dom.delta)
        cp, cn = np.minimum(hpos, H_CLIP), np.minimum(hneg, H_CLIP)
        dh = (rp * (1.0 - (cp / H_MAX) ** 4) - rn * (1.0 - (cn / H_MAX) ** 4)
              - relax * h + (gp * hneg - gn * hpos) / TAU_X)
    return hp * (1 - s) - hn * (1 + s), dh, (v - y) / p["tau_r"], (x - sig) / p["tau_s"]


def simulate(p, v_of_t, t_end, s0=None, nd=None, temp=None, t_eval=None, breaks=None,
             rtol=1e-7, atol=1e-9, sigma0=None, max_step=np.inf):
    """Integrate the model for an arbitrary drive ``v_of_t(t)`` [V] from 0 to ``t_end`` [s].

    ``breaks`` are times where ``v_of_t`` has kinks (the integration restarts
    there).  ``s0`` is a scalar or one value per class (default ``p["s0"]``);
    h starts at 0, y at v(0) and sigma at ``sigma0`` (default 0).

    Returns a dict with ``t``, ``v``, ``s``, ``h``, ``sig``, ``x``, ``y``,
    ``qd`` (terminal charge density, C/m^2), ``q`` (= qd * area, C),
    ``i_leak`` (A) and ``dom``.
    """
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    n = len(dom.w)
    s0 = np.full(n, float(p["s0"])) if s0 is None else np.broadcast_to(np.asarray(s0, float), (n,)).copy()
    sig0 = 0.0 if sigma0 is None else sigma0

    def f(t, z):
        ds, dh, dy, dg = rhs(p, dom, z[:n], z[n:2 * n], z[2 * n], z[2 * n + 1], float(v_of_t(t)), temp)
        return np.concatenate([ds, dh, [dy, dg]])

    edges = np.unique(np.concatenate([[0.0, t_end], np.asarray(breaks if breaks is not None else [], float)]))
    edges = edges[(edges >= 0) & (edges <= t_end)]
    ts, zs = [], []
    z = np.concatenate([s0, np.zeros(n), [float(v_of_t(0.0)), sig0]])
    for a, b in zip(edges[:-1], edges[1:]):
        te = None
        if t_eval is not None:
            te = np.asarray(t_eval)[(np.asarray(t_eval) >= a) & (np.asarray(t_eval) <= b)]
            te = np.unique(np.concatenate([[a], te, [b]]))
        sol = solve_ivp(f, (a, b), z, method="Radau", rtol=rtol, atol=atol, t_eval=te, max_step=max_step)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts.append(sol.t)
        zs.append(sol.y)
        z = sol.y[:, -1]
    t = np.concatenate(ts)
    zz = np.concatenate(zs, axis=1)
    s, y = zz[:n], zz[2 * n]
    v = np.array([float(v_of_t(a)) for a in t])
    x = dom.w @ s
    qd = np.array([q_term(p, a, b, c, temp) for a, b, c in zip(v, x, y)])
    return {"t": t, "v": v, "s": s, "h": zz[n:2 * n], "sig": zz[2 * n + 1], "x": x, "y": y,
            "qd": qd, "q": p["area"] * qd, "i_leak": p["area"] * j_leak(p, v, temp), "dom": dom}


def march(p, dom: Domains, s, h, times, volts, temp=None, theta=0.5, cache=None):
    """Integrate E3 and E3b on a fixed time grid with the theta method (vdep = 0).

    ``theta = 0.5`` is the trapezoidal rule, ``1`` backward Euler; one backward
    Euler step is taken whenever the voltage changes, which damps the fast
    reversal mode of E3b.  ``s`` may be (nd,) or (m, nd).  Used to check that a
    result converges with the time step, as genuine accumulation must and
    numerical drift does not.
    """
    temp = p["t0"] if temp is None else temp
    cache = {} if cache is None else cache
    beta, th = p["beta"], p["tau_h"]

    def rr(v):
        key = float(v)
        got = cache.get(key)
        if got is None:
            rp, rn = rates(p, dom, key, 0.0, temp)
            rel = 0.0 if not np.isfinite(th) else 1.0 / tau_h_eff(p, key)
            gp, gn = rev_gates(p, p["rho"] * key - p["vimp"] - dom.delta)
            got = cache[key] = (rp, rn, rel, gp / TAU_X, gn / TAU_X)
        return got

    def fh(hh, rp, rn, rel, gp, gn):
        hpos, hneg = np.maximum(hh, 0.0), np.maximum(-hh, 0.0)
        cp, cn = np.minimum(hpos, H_CLIP), np.minimum(hneg, H_CLIP)
        val = rp * (1.0 - (cp / H_MAX) ** 4) - rn * (1.0 - (cn / H_MAX) ** 4) - rel * hh + gp * hneg - gn * hpos
        der = (-4.0 * rp * (hpos < H_CLIP) * cp**3 / H_MAX**4 - 4.0 * rn * (hneg < H_CLIP) * cn**3 / H_MAX**4
               - rel - gp * (hh < 0) - gn * (hh > 0))
        return val, der

    def haz(hh, rp, rn):
        if beta == 1.0:
            return rp, rn
        return beta * rp * h_pow(np.maximum(hh, 0.0), beta), beta * rn * h_pow(np.maximum(-hh, 0.0), beta)

    rp0, rn0, rel0, gp0, gn0 = rr(volts[0])
    theta_in = theta
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        rp1, rn1, rel1, gp1, gn1 = rr(volts[i])
        theta = 1.0 if volts[i] != volts[i - 1] or i == 1 else theta_in
        f0, _ = fh(h, rp0, rn0, rel0, gp0, gn0)
        guess = h
        for _ in range(8):
            f1, d1 = fh(guess, rp1, rn1, rel1, gp1, gn1)
            res = guess - h - dt * (theta * f1 + (1.0 - theta) * f0)
            guess = guess - res / (1.0 - dt * theta * d1)
            if np.max(np.abs(res)) < 1e-12:
                break
        hu0, hd0 = haz(h, rp0, rn0)
        hu1, hd1 = haz(guess, rp1, rn1)
        expl = hu0 * (1 - s) - hd0 * (1 + s)
        s = (s + dt * (1.0 - theta) * expl + dt * theta * (hu1 - hd1)) / (1.0 + dt * theta * (hu1 + hd1))
        h, rp0, rn0, rel0, gp0, gn0 = guess, rp1, rn1, rel1, gp1, gn1
    return s, h


# ----------------------------------------------------------- waveforms
def pwl(times, volts):
    """Piecewise-linear drive: returns ``(v_of_t, breaks)`` for :func:`simulate`."""
    ts, vs = np.asarray(times, float), np.asarray(volts, float)
    return (lambda a: float(np.interp(a, ts, vs))), ts
