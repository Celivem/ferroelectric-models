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
    # ---- cycling and storage history (E13-E17); every term is off by default.  See
    # docs/research/fatigue-imprint.md for the mechanisms and the literature behind each form.
    "wk_fp": Param(0.0, "1", "wk_fp", "f_p0", "pristine pinned (non-switching) fraction, released by wake-up",
                   "off by default; fit knob (wake-up data)"),
    "wk_vsp": Param(0.0, "V", "wk_vsp", "V_sp", "pristine internal-bias split, +-V_sp on alternating classes, removed by wake-up",
                    "off by default; fit knob (split switching-current peaks)"),
    "wk_nw": Param(1e3, "1", "wk_nw", "N_w", "wake-up constant in equivalent full switching cycles",
                   "illustrative; inactive while wk_fp = wk_vsp = 0"),
    "wk_kw": Param(1.0, "1", "wk_kw", "kappa_w", "stretch exponent of the wake-up law", "illustrative"),
    "wk_ea": Param(0.0, "eV", "wk_ea", "E_w", "activation energy of the wake-up rate", "off by default"),
    "fat_d": Param(0.0, "1", "fat_d", "D_max", "largest fraction pinned by fatigue",
                   "off by default; fit knob (fatigue data)"),
    "fat_tf": Param(1.0, "s", "fat_tf", "t_F", "field-time fatigue: stress time at 1 MV/cm for a_F = 1 (0 = off)",
                    "illustrative; inactive while fat_d = 0"),
    "fat_ns": Param(0.0, "1", "fat_ns", "N_F", "switching fatigue: full cycles switched at 1 MV/cm for a_F = 1 (0 = off)",
                    "off by default"),
    "fat_nf": Param(4.0, "1", "fat_nf", "n_F", "field exponent of both fatigue drivers",
                    "illustrative; inactive while fat_d = 0"),
    "fat_ea": Param(0.0, "eV", "fat_ea", "E_F", "activation energy of both fatigue drivers", "off by default"),
    "fat_kf": Param(1.0, "1", "fat_kf", "kappa_F", "stretch exponent of the fatigue law", "illustrative"),
    "fat_ke": Param(0.0, "1", "fat_ke", "k_E", "relative increase of the activation field per unit damage D",
                    "off by default"),
    "imp_v": Param(0.0, "V", "imp_v", "V_I", "imprint bias at full trapped-charge screening (sigma_I = +-1)",
                   "off by default; fit knob (imprint data)"),
    "imp_t1": Param(1e-7, "s", "imp_t1", "tau_I1", "shortest imprint (trapping) time constant at T_0",
                    "illustrative; inactive while imp_v = 0"),
    "imp_t2": Param(1e5, "s", "imp_t2", "tau_I2", "longest imprint (trapping) time constant at T_0",
                    "illustrative; inactive while imp_v = 0"),
    "imp_ea": Param(0.0, "eV", "imp_ea", "E_I", "activation energy of the imprint time constants", "off by default"),
    "imp_n": Param(12, "1", "FE_NI", "N_I", "number of imprint states (compile-time macro in Verilog-A)", "numerical"),
    "nd":Param(16, "1", "FE_ND", "N_d", "number of domain classes (compile-time macro in Verilog-A)", "numerical"),
    "y_max": Param(9.0, "1", "y_max", "y_m", "depth of the logit quantile grid (smallest q about 1.2e-4)", "numerical"),
    "grid_a": Param(1.5, "1", "grid_a", "a_g", "sinh stretching of the quantile grid", "numerical"),
    # ---- cycling, second round (E18-E20); every term is off by default.  See docs/research/cycling-r2.md.
    "wk_fq": Param(0.0, "1", "wk_fq", "f_q", "fraction pinned by mobile defects, depinned under field and re-pinned "
                   "at rest (wk_rho) or at reversals (wk_tr) (reversible wake-up pool, E18)",
                   "off by default; fit knob (pulse-width / duty-cycle data)"),
    "wk_rho": Param(0.0, "1", "wk_rho", "rho_q", "re-pinning at rest: rate / depinning rate at 1 MV/cm (E18)",
                    "off by default"),
    "wk_q1": Param(1e-2, "s", "wk_q1", "tau_q1", "shortest depinning time constant at 1 MV/cm and T_0 (E18)",
                   "illustrative; inactive while wk_fq = 0"),
    "wk_q2": Param(1e2, "s", "wk_q2", "tau_q2", "longest depinning time constant at 1 MV/cm and T_0 (E18)",
                   "illustrative; inactive while wk_fq = 0"),
    "wk_nq": Param(2.0, "1", "wk_nq", "n_q", "field exponent of the depinning drive (E18)",
                   "illustrative; inactive while wk_fq = 0"),
    "wk_ns": Param(8, "1", "FE_NQ", "N_Q", "number of depinning states (compile-time macro in Verilog-A)", "numerical"),
    "wk_tr": Param(0.0, "s", "wk_tr", "t_r", "re-pinning per full switching cycle, in seconds of depinning at "
                   "1 MV/cm (E18)", "off by default; fit knob"),
    "fat_tl": Param(0.0, "s", "fat_tl", "t_L", "build-up time of the field-time fatigue stress after a voltage change: "
                    "E_s^2 = max(E E_L, 0), E_L low-passed with t_L (E19; 0 = instantaneous)", "off by default"),
    "fat_lk": Param(0.0, "1", "fat_lk", "k_L", "leakage increase per unit fatigue age, J -> J (1 + k_L a_F^m_L) (E20)",
                    "off by default; fit knob (leakage-vs-cycling data)"),
    "fat_lm": Param(1.0, "1", "fat_lm", "m_L", "exponent of the fatigue-induced leakage increase (E20)",
                    "illustrative; inactive while fat_lk = 0"),
    # ---- imprint dynamics, round 2 (E17a, E17b); off by default.  See docs/research/read-imprint-r2.md.
    "imp_kw": Param(0.0, "1", "imp_kw", "kappa_I",
                    "branch asymmetry of the imprint: the switching field out of the imprinted state moves by "
                    "(1 + kappa_I) V_I |sigma_I|, the other one by (1 - kappa_I) V_I |sigma_I|",
                    "off by default; 0.2-0.3 from the branch coercive voltages of [Buragohain2019] Figs. 2a,b"),
    "imp_va": Param(0.0, "V", "imp_va", "V_a",
                    "field acceleration of the imprint states, rate x cosh(rho V / V_a) (0 = off)",
                    "off by default; write-time ('fluid') imprint of [Buragohain2019] Fig. 3b"),
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
        "* Instance parameters (put them on the N line): area, s0, sg0, nw0, af0, si0, zq0.",
        f"*   area default {GENERIC['area'].value:g} m^2 ({GENERIC['area'].source})",
        f"* FE_ND = {GENERIC['nd'].value} domain classes is a compile-time macro (see build_osdi.py).",
        f"* FE_NI = {GENERIC['imp_n'].value} imprint states is a compile-time macro.",
        f"* FE_NQ = {GENERIC['wk_ns'].value} depinning states is a compile-time macro.",
        f".model {model_name} {module}",
    ]
    for name, prm in GENERIC.items():
        if name in ("area", "s0", "nd", "imp_n", "wk_ns"):
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
    # "_imp_w" (set only by aged_card, E17a): both switching fields move away from the centre by this amount
    w = p.get("_imp_w", 0.0)
    return rate(u - w, dom.va_p, tau0, p["alpha"]), rate(-u - w, dom.va_n, tau0, p["alpha"])


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
        # Reversed drive below the reset threshold: h is not reset, so (as in E3b,
        # where the hazard uses h+ = max(h, 0) and h- = max(-h, 0)) the old progress
        # first counts down; the new direction has no hazard until h crosses zero.
        back = (target * h) < 0
        if back.any():
            if g == 1.0:
                hs = h + r * dt
            elif dt < 1e-3 * tau_h_eff(p, v):
                hs = h * g + r * dt
            else:
                hs = h * g + r * tau_h_eff(p, v) * (1.0 - g)
            prog = np.maximum(target * hs, 0.0)
            lam = np.where(back, prog**beta, lam)
            h_new = np.where(back, hs, h_new)
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


# ------------------------------------------------ cycling and storage history
# E13-E17: wake-up, fatigue and logarithmic-time imprint (all off by default).
#
# * Pinned pool (E13): a fraction f of the film does not switch.  Its
#   polarization is one state variable pi (in units of P_s) and its frozen
#   reversible displacement another, r (C/m^2).  Pinning takes the current mean
#   state and displacement with it, so it never moves charge; release returns
#   the pool's mean state to the switchable classes.
# * Wake-up (E14): f_w = f_p0 exp(-g(a_w)) and an internal-bias split
#   +-V_sp exp(-g(a_w)) on alternating classes; a_w counts equivalent full
#   switching cycles in units of N_w: da_w/dt = A_w(T) sum_k w_k |ds_k/dt|_sw / (4 N_w).
# * Fatigue (E15): D = D_max [1 - exp(-g(a_F))] with two optional drivers,
#   da_F/dt = A_F(T) (E_s / 1 MV/cm)^n_F [1/t_F + sum_k w_k |ds_k/dt|_sw / (4 N_F)],
#   E_s = sqrt((rho V / t_fe)^2 + E_fl^2): field-time stress and switching under field.
#   It pins switchable domains and raises their activation field by (1 + k_E D).
# * A_w, A_F = exp[-E_a/k_B (1/T - 1/T_0)] (1 at T_0).
# * f = 1 - (1 - f_w)(1 - D) (E16).
# * Imprint (E17): N_I trapped-charge states with log-spaced time constants,
#   tau_j dsigma_j/dt = x_t - sigma_j; the bias V_I mean(sigma_j) is subtracted
#   from V_imp, and the reversible branches move with it.
# * Imprint, round 2 (both optional, docs/research/read-imprint-r2.md):
#   E17a branch asymmetry: both switching fields (and the branch peaks) also move
#   outward by w = kappa_I V_I |sigma_I|, so the field that switches out of the
#   imprinted state moves (1 + kappa_I) times the centre shift, the other (1 - kappa_I);
#   E17b field acceleration: every imprint rate is multiplied by cosh(rho V / V_a)
#   (charge injection during write pulses and under bias builds imprint faster).
# g(a) = (a + a_e)^kappa - a_e^kappa (a_e = 1e-4 keeps dg/da finite at a = 0).
#
# Second round (E18-E20, docs/research/cycling-r2.md), also off by default:
# * Reversible depinning pool (E18): a fraction f_q of the film is pinned by mobile
#   defects.  N_Q states z_j (depinned fraction, log-spaced tau_j) follow
#   tau_j dz_j/dt = A_w(T) [phi(V) (1 - z_j) - (rho_q + t_r a(t)/4) z_j],  phi = (E_s / 1 MV/cm)^n_q (minus its
#   value at 0 V),
#   a(t) = sum_k w_k |ds_k/dt| (a full cycle switches 4): the field depins; re-pinning acts at
#   rest (rho_q) and/or at every reversal (t_r).  The pool's pinned part is
#   f_q (1 - Z), Z = mean z_j, and f = 1 - (1 - f_w)(1 - f_q (1 - Z))(1 - D).
#   Under a pulse train of duty D_c and f full cycles per second the plateau is
#   Z = phi D_c / (phi D_c + rho_q + t_r f).
# * Fatigue lag (E19): the field-time driver needs the present field and a slowly
#   established condition aligned with it: E_s^2 = max(E E_L, 0) + E_fl^2 with the
#   low-passed field E_L, t_L dV_L/dt = V - V_L.  The first ~t_L of every pulse
#   adds little stress and none is added at 0 V.
# * Leakage increase (E20): J -> J [1 + k_L a_F^m_L], a_F standing for the
#   generated defect density.
E_REF = 1e8      #: reference field of the fatigue stress rate (1 MV/cm) [V/m]
E_FLOOR = 1e3    #: field floor of the stress rate (keeps its derivative finite at 0 V) [V/m]
A_EPS = 1e-4     #: regularization of the stretched laws at a = 0
K_B_EV = K_B / Q_E


def aging_flags(p) -> tuple[bool, bool, bool]:
    """(wake-up on, fatigue on, imprint on) for a parameter card."""
    return (p.get("wk_fp", 0.0) > 0 or p.get("wk_vsp", 0.0) != 0,
            p.get("fat_d", 0.0) > 0 or p.get("fat_lk", 0.0) > 0,
            p.get("imp_v", 0.0) != 0)


def depin_on(p) -> bool:
    """Reversible depinning pool (E18) on."""
    return p.get("wk_fq", 0.0) > 0


def lag_on(p) -> bool:
    """Lagged field-time fatigue driver (E19) on (needs the fatigue age to be active)."""
    return aging_flags(p)[1] and p.get("fat_tl", 0.0) > 0 and p["fat_tf"] > 0


def aging_on(p) -> bool:
    return any(aging_flags(p)) or depin_on(p)


def _stretch(a, kappa):
    """g(a) = (a + a_e)^kappa - a_e^kappa and dg/da."""
    a = np.maximum(a, 0.0)
    return (a + A_EPS) ** kappa - A_EPS ** kappa, kappa * (a + A_EPS) ** (kappa - 1.0)


def wake_state(p, a_w):
    """Unwoken fraction 1 - W = exp(-g(a_w)) and its derivative d(1 - W)/da_w."""
    g, dg = _stretch(a_w, p["wk_kw"])
    u = np.exp(-g)
    return u, -u * dg


def fatigue_damage(p, a_f):
    """Damage D(a_F) = D_max [1 - exp(-g(a_F))] and dD/da_F."""
    g, dg = _stretch(a_f, p["fat_kf"])
    e = np.exp(-g)
    return p["fat_d"] * (1.0 - e), p["fat_d"] * e * dg


def pinned_fraction(p, a_w, a_f, zq=0.0):
    """(f, df/da_w, df/da_F, unwoken fraction 1 - W, D) of E16 (with the E18 pool at mean depinned state ``zq``)."""
    uw, duw = wake_state(p, a_w)
    D, dD = fatigue_damage(p, a_f)
    fw, dfw = p["wk_fp"] * uw, p["wk_fp"] * duw
    gq = 1.0 - p.get("wk_fq", 0.0) * (1.0 - zq)
    return 1.0 - (1.0 - fw) * gq * (1.0 - D), gq * (1.0 - D) * dfw, gq * (1.0 - fw) * dD, uw, D


def depin_dfdz(p, a_w, a_f):
    """df/dZ of E18 (negative: depinning releases domains)."""
    uw, _ = wake_state(p, a_w)
    D, _ = fatigue_damage(p, a_f)
    return -p.get("wk_fq", 0.0) * (1.0 - p["wk_fp"] * uw) * (1.0 - D)


def depin_taus(p, temp=None):
    """Depinning time constants tau_j at 1 MV/cm (E18): geometric mid-points of equal log bins, divided by A_w(T)."""
    n = int(p.get("wk_ns", 8))
    e = np.linspace(np.log(p["wk_q1"]), np.log(p["wk_q2"]), n + 1)
    return np.exp(0.5 * (e[:-1] + e[1:])) / arrhenius(p, p["wk_ea"], temp)


def depin_drive(p, v):
    """Depinning drive phi(V) = (E_s / 1 MV/cm)^n_q - (E_fl / 1 MV/cm)^n_q (E18; exactly 0 at 0 V)."""
    e = np.sqrt((p["rho"] * np.asarray(v, float) / p["tfe"]) ** 2 + E_FLOOR**2)
    return (e / E_REF) ** p["wk_nq"] - (E_FLOOR / E_REF) ** p["wk_nq"]


def depin_step(p, zq, v, dt, qtaus, act=0.0):
    """Exact update of the E18 states over ``dt`` at constant voltage ``v``: z -> z_ss + (z - z_ss) exp(-k dt).

    ``act`` is the switched amount sum_k w_k |ds_k| during ``dt`` (re-pinning per reversal, t_r).  The
    switching of a constant-voltage piece happens at its start, so that re-pinning is applied first and the
    field then acts for ``dt`` (the per-period map stays affine in z).
    """
    zq = np.asarray(zq, float)
    if act > 0 and p.get("wk_tr", 0.0) > 0:
        zq = zq * np.exp(-p["wk_tr"] * act / 4.0 / qtaus)
    phi = float(depin_drive(p, v))
    tot = phi + p["wk_rho"]
    if tot <= 0:
        return zq.copy()
    zs = phi / tot
    return zs + (zq - zs) * np.exp(-tot * dt / qtaus)


def depin_plateau(p, duty, v, freq=0.0):
    """Steady depinned fraction under a pulse train of amplitude ``v``, duty ``duty`` (time at |V| / period) and
    ``freq`` full switching cycles per second."""
    phi = float(depin_drive(p, v))
    return phi * duty / (phi * duty + p["wk_rho"] + p.get("wk_tr", 0.0) * freq)


def leak_factor(p, a_f):
    """Leakage multiplier 1 + k_L a_F^m_L (E20)."""
    k = p.get("fat_lk", 0.0)
    if k <= 0:
        return np.ones_like(np.asarray(a_f, float))
    return 1.0 + k * np.maximum(np.asarray(a_f, float), 0.0) ** p.get("fat_lm", 1.0)


def arrhenius(p, ea, temp):
    """Rate factor exp[-E_a/k_B (1/T - 1/T_0)] (1 at T_0)."""
    temp = p["t0"] if temp is None else temp
    return float(np.exp(-ea / K_B_EV * (1.0 / temp - 1.0 / p["t0"])))


def field_factor(p, v):
    """(E_s / 1 MV/cm)^n_F of both fatigue drivers (E15)."""
    e = np.sqrt((p["rho"] * np.asarray(v, float) / p["tfe"]) ** 2 + E_FLOOR**2)
    return (e / E_REF) ** p["fat_nf"]


def stress_rate(p, v, temp=None):
    """Field-time part of da_F/dt [1/s] at terminal voltage ``v`` (E15); 0 when t_F = 0."""
    if p["fat_tf"] <= 0:
        return np.zeros_like(np.asarray(v, float))
    return field_factor(p, v) / p["fat_tf"] * arrhenius(p, p["fat_ea"], temp)


def lag_stress_rate(p, v, vl, temp=None):
    """Field-time part of da_F/dt with the E19 build-up: (E_s / 1 MV/cm)^n_F / t_F, E_s^2 = max(E E_L, 0) + E_fl^2."""
    if p["fat_tf"] <= 0:
        return np.zeros_like(np.asarray(v, float))
    e2 = np.maximum(p["rho"] ** 2 * np.asarray(v, float) * np.asarray(vl, float) / p["tfe"] ** 2, 0.0)
    e = np.sqrt(e2 + E_FLOOR**2)
    return (e / E_REF) ** p["fat_nf"] / p["fat_tf"] * arrhenius(p, p["fat_ea"], temp)


def stress_per_period(p, t, v, sub=64, temp=None):
    """Integral of :func:`stress_rate` over one period of a piecewise-linear waveform (Gauss-Legendre)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    if lag_on(p):
        return lagged_stress_per_period(p, t, v, temp=temp)
    tot = 0.0
    xi, wi = np.polynomial.legendre.leggauss(sub)
    for a, b, va, vb in zip(t[:-1], t[1:], v[:-1], v[1:]):
        if b <= a:
            continue
        tt = 0.5 * (b - a) * xi + 0.5 * (b + a)
        vv = va + (vb - va) * (tt - a) / (b - a)
        tot += 0.5 * (b - a) * float(wi @ stress_rate(p, vv, temp))
    return tot


def _lag_pieces(p, t, v, sub=400):
    """Constant-voltage pieces of one period and the periodic lagged voltage at their starts (E19)."""
    dts, vms, _ = _refine_period(t, v, sub)
    return dts, vms, _lag_starts(p, dts, vms)


def _lag_starts(p, dts, vms):
    """Periodic steady state of the lagged voltage V_L at the start of every constant-voltage piece (E19)."""
    a = np.exp(-dts / p["fat_tl"])
    # affine map over one period y_end = A y_0 + B, fixed point y* = B / (1 - A)
    y, A = 0.0, 1.0
    for ai, vi in zip(a, vms):
        y = vi + (y - vi) * ai
        A *= ai
    y0 = y / (1.0 - A) if A < 1.0 else 0.0
    starts = np.empty(len(dts))
    y = y0
    for k, (ai, vi) in enumerate(zip(a, vms)):
        starts[k] = y
        y = vi + (y - vi) * ai
    return starts


def lagged_stress_per_period(p, t, v, temp=None, sub=400, gl=12):
    """Field-time fatigue stress of one period in the periodic steady state of the lagged voltage (E19)."""
    dts, vms, y0 = _lag_pieces(p, t, v, sub)
    xi, wi = np.polynomial.legendre.leggauss(gl)
    tot = 0.0
    tl = p["fat_tl"]
    for dt, vm, ys in zip(dts, vms, y0):
        # the transient lasts a few t_L: integrate it and the rest of the piece separately
        for a, b in ((0.0, min(dt, 2.0 * tl)), (min(dt, 2.0 * tl), min(dt, 10.0 * tl)), (min(dt, 10.0 * tl), dt)):
            if b <= a:
                continue
            tt = a + 0.5 * (b - a) * (xi + 1.0)
            yy = vm + (ys - vm) * np.exp(-tt / tl)
            tot += 0.5 * (b - a) * float(wi @ lag_stress_rate(p, vm, yy, temp))
    return tot


def split_signs(nd):
    """+1, -1, +1, ... : the two sub-populations of the pristine internal-bias split."""
    return np.where(np.arange(nd) % 2 == 0, 1.0, -1.0)


def imprint_taus(p, temp=None):
    """Time constants tau_j of the N_I imprint states: geometric mid-points of equal log bins, Arrhenius in T."""
    temp = p["t0"] if temp is None else temp
    n = int(p["imp_n"])
    e = np.linspace(np.log(p["imp_t1"]), np.log(p["imp_t2"]), n + 1)
    tau = np.exp(0.5 * (e[:-1] + e[1:]))
    return tau * np.exp(p["imp_ea"] / K_B_EV * (1.0 / temp - 1.0 / p["t0"]))


I_EPS = 1e-3     #: smoothing of |sigma_I| in the branch asymmetry (E17a): sqrt(s^2 + e^2) - e


def imprint_widening(p, s_i):
    """E17a: amount w = kappa_I V_I (sqrt(sigma_I^2 + e^2) - e) by which both switching fields move outward.

    With the centre shift -V_I sigma_I of E17, the switching field out of the
    imprinted state moves by (1 + kappa_I) V_I |sigma_I| and the other one by
    (1 - kappa_I) V_I |sigma_I|; 0 when imp_kw = 0.
    """
    kw = p.get("imp_kw", 0.0)
    if kw == 0.0:
        return 0.0 * np.asarray(s_i, float)
    s_i = np.asarray(s_i, float)
    return kw * p["imp_v"] * (np.sqrt(s_i * s_i + I_EPS * I_EPS) - I_EPS)


def imprint_accel(p, v):
    """E17b: field acceleration cosh(rho V / V_a) of every imprint rate (1 when imp_va = 0; argument capped at 80)."""
    va = p.get("imp_va", 0.0)
    v = np.asarray(v, float)
    if va <= 0.0:
        return np.ones_like(v)
    return np.cosh(np.minimum(np.abs(p["rho"] * v) / va, 80.0))


def imprint_response(p, t, temp=None):
    """sigma_I(t) after x_t steps from 0 to 1 at t = 0 with sigma_j(0) = 0: mean_j[1 - exp(-t/tau_j)]."""
    tau = imprint_taus(p, temp)
    t = np.asarray(t, float)
    return np.mean(1.0 - np.exp(-t[..., None] / tau), axis=-1)


@dataclass
class History:
    """Slow state of the cycling and storage history (defaults: pristine, never stored).

    ``nw``: equivalent full switching cycles (at T_0); ``af``: fatigue age a_F
    (dimensionless); ``si``: value of every imprint state sigma_j (scalar or
    one per state); ``pi``: pinned polarization in units of P_s (None: f(0)
    times the mean initial class state).
    """

    nw: float = 0.0
    af: float = 0.0
    si: float | np.ndarray = 0.0
    pi: float | None = None
    zq: float | np.ndarray = 0.0  #: depinned fraction of every E18 state (0 = pristine)


def aged_card(p, nd=None, temp=None, a_w=0.0, a_f=0.0, s_i=0.0, zq=0.0):
    """Card and domains with the slow states frozen at (a_w, a_F, sigma_I).

    Returns ``(p_eff, dom_eff, f)`` for the closed-form solvers such as
    :func:`step`: the activation field carries (1 + k_E D), V_imp carries
    -V_I sigma_I, the class offsets carry the wake-up split, and ``f`` is the
    pinned fraction (the switchable charge is P_s (1 - f) times the change of
    the mean class state).
    """
    temp = p["t0"] if temp is None else temp
    f, _, _, uw, D = pinned_fraction(p, a_w, a_f, float(np.mean(zq)))
    pe = dict(p, mul_ea=p["mul_ea"] * (1.0 + p["fat_ke"] * D), vimp=p["vimp"] - p["imp_v"] * float(np.mean(s_i)))
    if p.get("imp_kw", 0.0) != 0.0 and p["imp_v"] != 0.0:
        pe["_imp_w"] = float(imprint_widening(p, float(np.mean(s_i))))
    dom = make_domains(pe, nd, temp)
    dom.delta = dom.delta + split_signs(len(dom.w)) * p["wk_vsp"] * uw
    return pe, dom, float(f)


def term2_aged(p, v, x, y, temp, s_i):
    """Term (2) of the reversible charge, k_C [gbar + x g_d], with the branches moved by the imprint bias."""
    k = 1.0 + p["tc_c"] * (temp - p["t0"])
    pv = dict(p, vi=p["vi"] - p["imp_v"] * s_i / p["rho"])
    if p.get("imp_kw", 0.0) != 0.0:
        # E17a: the branch peaks sit at the switching fields, so they move outward with them
        pv["vs"] = p["vs"] + imprint_widening(p, s_i) / p["rho"]
    up, dn = branches(pv, v)
    gbar, gd = 0.5 * (up + dn), 0.5 * (up - dn)
    frev = p["frev"]
    if frev < 1.0:
        upy, dny = branches(pv, y)
        gbar = frev * gbar + (1.0 - frev) * 0.5 * (upy + dny)
        gd = frev * gd + (1.0 - frev) * 0.5 * (upy - dny)
    return k * (gbar + x * gd)


def q_rev_aged(p, v, x, y, temp, f, s_i, r=0.0):
    """Reversible charge with history: term (2) of the switchable part (1 - f) plus the frozen part r."""
    k = 1.0 + p["tc_c"] * (temp - p["t0"])
    slow = (p["c_w"] * (1.0 - x * x) + p["c_r"]) * (y if (p["c_w"] or p["c_r"]) else 0.0)
    return k * p["c_d"] * v + (1.0 - f) * term2_aged(p, v, x, y, temp, s_i) + r + slow


def q_term_aged(p, v, x, y, temp, f, pi, s_i, r=0.0):
    """Terminal charge density with history: P_s [(1 - f) x + pi] + q_rev [C/m^2]."""
    return ps_of_t(p, temp) * ((1.0 - f) * x + pi) + q_rev_aged(p, v, x, y, temp, f, s_i, r)


def n_depin(p) -> int:
    """Number of E18 states carried in the Python state vector (0 when the pool is off)."""
    return int(p.get("wk_ns", 8)) if depin_on(p) else 0


def rhs_aged(p, dom: Domains, z, v, temp, taus, signs, qtaus=None):
    """Right-hand side of E3-E20.

    z = [s (nd), h (nd), y, sigma, pi, r, a_w, a_F, sigma_j (N_I), z_j (N_Q, only with E18 on),
    V_L (only with E19 on)].  Returns the matching derivatives (the last two possibly empty).
    """
    n = len(dom.w)
    s, h = z[:n], z[n:2 * n]
    y, sig, pi, r, a_w, a_f = z[2 * n:2 * n + 6]
    ni = len(taus)
    sj = z[2 * n + 6:2 * n + 6 + ni]
    nq = n_depin(p)
    zq = z[2 * n + 6 + ni:2 * n + 6 + ni + nq]
    vl = z[2 * n + 6 + ni + nq:2 * n + 7 + ni + nq] if lag_on(p) else np.zeros(0)
    if nq and qtaus is None:
        qtaus = depin_taus(p, temp)
    zbar = float(np.mean(zq)) if nq else 0.0
    wake, fat, imp = aging_flags(p)
    f, dfw, dff, uw, D = pinned_fraction(p, a_w, a_f, zbar)
    xs = float(dom.w @ s)
    xt = (1.0 - f) * xs + pi
    s_i = float(np.mean(sj)) if imp else 0.0
    vimp = p["vimp"] - p["imp_v"] * s_i
    m_e = 1.0 + p["fat_ke"] * D
    delta = dom.delta + signs * p["wk_vsp"] * uw
    u = p["rho"] * v - p["vdep"] * (xt - sig) - vimp - delta
    tau0 = p["tau0"] * (1.0 + p["tc_tau"] * (temp - p["t0"]))
    w_i = float(imprint_widening(p, s_i)) if imp else 0.0
    rp, rn = rate(u - w_i, dom.va_p * m_e, tau0, p["alpha"]), rate(-u - w_i, dom.va_n * m_e, tau0, p["alpha"])
    th = p["tau_h"]
    relax = 0.0 if not np.isfinite(th) else 1.0 / (th * (1.0 + ((v - vimp) / V_GATE) ** 4))
    beta = p["beta"]
    if beta == 1.0:
        hp, hn, dh = rp, rn, np.zeros(n)
    else:
        hpos, hneg = np.maximum(h, 0.0), np.maximum(-h, 0.0)
        hp = beta * rp * h_pow(hpos, beta)
        hn = beta * rn * h_pow(hneg, beta)
        ux = abs(vimp) + U_XM
        gp, gn = 0.5 * (1.0 + np.tanh((u - ux) / U_XW)), 0.5 * (1.0 + np.tanh((-u - ux) / U_XW))
        cp, cn = np.minimum(hpos, H_CLIP), np.minimum(hneg, H_CLIP)
        dh = (rp * (1.0 - (cp / H_MAX) ** 4) - rn * (1.0 - (cn / H_MAX) ** 4)
              - relax * h + (gp * hneg - gn * hpos) / TAU_X)
    ds = hp * (1 - s) - hn * (1 + s)
    act = float(dom.w @ np.abs(ds))
    da_w = act / (4.0 * p["wk_nw"]) * arrhenius(p, p["wk_ea"], temp) if wake else 0.0
    da_f = 0.0
    if fat:
        da_f = float(lag_stress_rate(p, v, vl[0], temp) if vl.size else stress_rate(p, v, temp))
        if p["fat_ns"] > 0:
            da_f += float(field_factor(p, v)) * act / (4.0 * p["fat_ns"]) * arrhenius(p, p["fat_ea"], temp)
    fdot = dfw * da_w + dff * da_f
    dzq = np.zeros(0)
    if nq:
        phi = float(depin_drive(p, v))
        dzq = (phi * (1.0 - zq) - (p["wk_rho"] + p.get("wk_tr", 0.0) * act / 4.0) * zq) / qtaus
        fdot += depin_dfdz(p, a_w, a_f) * float(np.mean(dzq))
    dvl = (v - vl) / p["fat_tl"] if vl.size else np.zeros(0)
    fpin, frel = max(fdot, 0.0), max(-fdot, 0.0)
    ratio = min(max(pi / max(f, 1e-12), -1.0), 1.0)
    ds = ds + frel * (ratio - s) / (1.0 - f)
    dpi = fpin * xs - frel * ratio
    dr = (fpin * term2_aged(p, v, xs, y, temp, s_i) - frel * r / max(f, 1e-12)) if (fpin or frel) else 0.0
    dsj = (xt - sj) / taus * float(imprint_accel(p, v)) if imp else np.zeros_like(sj)
    if nq or vl.size:
        return ds, dh, (v - y) / p["tau_r"], (xt - sig) / p["tau_s"], dpi, dr, da_w, da_f, dsj, dzq, dvl
    return ds, dh, (v - y) / p["tau_r"], (xt - sig) / p["tau_s"], dpi, dr, da_w, da_f, dsj


def _simulate_aged(p, v_of_t, t_end, s0, dom, temp, t_eval, breaks, rtol, atol, sig0, max_step, history):
    n = len(dom.w)
    hist = history or History()
    taus = imprint_taus(p, temp)
    signs = split_signs(n)
    a_w0 = hist.nw / p["wk_nw"]
    a_f0 = hist.af
    nq = n_depin(p)
    qtaus = depin_taus(p, temp) if nq else None
    zq0 = np.broadcast_to(np.asarray(hist.zq, float), (nq,)).copy()
    f0 = float(pinned_fraction(p, a_w0, a_f0, float(np.mean(zq0)) if nq else 0.0)[0])
    pi0 = f0 * float(np.mean(s0)) if hist.pi is None else hist.pi
    sj0 = np.broadcast_to(np.asarray(hist.si, float), taus.shape).copy()
    v0 = float(v_of_t(0.0))
    r0 = f0 * float(term2_aged(p, v0, float(dom.w @ s0), v0, temp, float(np.mean(sj0)) if aging_flags(p)[2] else 0.0))
    vl0 = np.array([v0]) if lag_on(p) else np.zeros(0)

    def fz(t, z):
        out = rhs_aged(p, dom, z, float(v_of_t(t)), temp, taus, signs, qtaus)
        return np.concatenate([out[0], out[1], out[2:8], *out[8:]])

    edges = np.unique(np.concatenate([[0.0, t_end], np.asarray(breaks if breaks is not None else [], float)]))
    edges = edges[(edges >= 0) & (edges <= t_end)]
    ts, zs = [], []
    z = np.concatenate([s0, np.zeros(n), [v0, sig0, pi0, r0, a_w0, a_f0], sj0, zq0, vl0])
    for a, b in zip(edges[:-1], edges[1:]):
        te = None
        if t_eval is not None:
            te = np.asarray(t_eval)[(np.asarray(t_eval) >= a) & (np.asarray(t_eval) <= b)]
            te = np.unique(np.concatenate([[a], te, [b]]))
        sol = solve_ivp(fz, (a, b), z, method="Radau", rtol=rtol, atol=atol, t_eval=te, max_step=max_step)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts.append(sol.t)
        zs.append(sol.y)
        z = sol.y[:, -1]
    t = np.concatenate(ts)
    zz = np.concatenate(zs, axis=1)
    s, y = zz[:n], zz[2 * n]
    ni = len(taus)
    pi, r, a_w, a_f = zz[2 * n + 2], zz[2 * n + 3], zz[2 * n + 4], zz[2 * n + 5]
    sj = zz[2 * n + 6:2 * n + 6 + ni]
    zq = zz[2 * n + 6 + ni:2 * n + 6 + ni + nq]
    zbar = np.mean(zq, axis=0) if nq else np.zeros_like(t)
    s_i = np.mean(sj, axis=0) if aging_flags(p)[2] else np.zeros_like(t)
    f, _, _, uw, D = pinned_fraction(p, a_w, a_f, zbar)
    v = np.array([float(v_of_t(a)) for a in t])
    x = dom.w @ s
    qd = np.array([q_term_aged(p, a, b, c, temp, ff, pp, si, rr)
                   for a, b, c, ff, pp, si, rr in zip(v, x, y, f, pi, s_i, r)])
    out = {"t": t, "v": v, "s": s, "h": zz[n:2 * n], "sig": zz[2 * n + 1], "x": x, "y": y,
           "qd": qd, "q": p["area"] * qd, "i_leak": p["area"] * j_leak(p, v, temp) * leak_factor(p, a_f), "dom": dom,
           "pi": pi, "r": r, "a_w": a_w, "a_f": a_f, "sj": sj, "s_i": s_i, "f": f, "W": 1.0 - uw, "D": D,
           "x_t": (1.0 - f) * x + pi}
    if nq:
        out.update(zq=zq, Z=zbar)
    if lag_on(p):
        out["v_lag"] = zz[2 * n + 6 + ni + nq]
    return out


def _refine_period(t, v, sub):
    """Split one period of a PWL waveform into about ``sub`` constant-voltage pieces.

    Returns the durations, the mid-point voltages, and the voltages at the end
    of each piece (used to interpolate coercive voltages).
    """
    t, v = np.asarray(t, float), np.asarray(v, float)
    T = t[-1] - t[0]
    dts, vms, ves = [], [], []
    for a, b, va, vb in zip(t[:-1], t[1:], v[:-1], v[1:]):
        if b <= a:
            continue
        m = 1 if va == vb else max(1, int(np.ceil(sub * (b - a) / T)))
        e = np.linspace(0.0, 1.0, m + 1)
        mid = 0.5 * (e[:-1] + e[1:])
        dts += list(np.diff(e) * (b - a))
        vms += list(va + (vb - va) * mid)
        ves += list(va + (vb - va) * e[1:])
    return np.array(dts), np.array(vms), np.array(ves)


def _one_period(p, nd, temp, dts, vms, s, h, a_w, a_f, sj, pi, taus, zq=None, qtaus=None, vl_starts=None):
    """One period with the slow states frozen.

    Returns the end state, the switched amount ``act`` (and ``actf``, weighted
    by the fatigue field factor), the mean class state after every piece, f,
    the domains, the change of the pinned pool over the period, ``dpi``
    (pinning and release are taken piece by piece with the mean state of that
    piece, as in E13: they happen at different times of a period), and the E18
    states at the end of the period (exact piecewise update; None when off).
    ``vl_starts`` is the periodic lagged voltage of E19 at each piece start.
    """
    wake, fat, imp = aging_flags(p)
    s_i = float(np.mean(sj)) if imp else 0.0
    zbar = float(np.mean(zq)) if zq is not None else 0.0
    pe, dom, f = aged_card(p, nd, temp, a_w, a_f, s_i, zbar)
    _, dfw, dff, _, _ = pinned_fraction(p, a_w, a_f, zbar)
    dfz = depin_dfdz(p, a_w, a_f) if zq is not None else 0.0
    arr_w, arr_f = arrhenius(p, p["wk_ea"], temp), arrhenius(p, p["fat_ea"], temp)
    ratio = min(max(pi / max(f, 1e-12), -1.0), 1.0)
    act, actf, dpi, xs, sj = 0.0, 0.0, 0.0, [float(dom.w @ s)], np.array(sj, float)
    for k, (dt, vv) in enumerate(zip(dts, vms)):
        s2, h = step(pe, dom, s, h, vv, dt, temp)
        dsw = float(dom.w @ np.abs(s2 - s))
        ff = float(field_factor(p, vv))
        act += dsw
        actf += dsw * ff
        x0, x1 = float(dom.w @ s), float(dom.w @ s2)
        daw = dsw / (4.0 * p["wk_nw"]) * arr_w if wake else 0.0
        daf = 0.0
        if fat:
            if vl_starts is None:
                daf = float(stress_rate(p, vv, temp)) * dt
            else:
                # Simpson on the exponential approach of the lagged voltage (bookkeeping of pi only)
                ys = vv + (vl_starts[k] - vv) * np.exp(-np.array([0.0, 0.5, 1.0]) * dt / p["fat_tl"])
                daf = dt / 6.0 * float(np.array([1.0, 4.0, 1.0]) @ lag_stress_rate(p, vv, ys, temp))
            if p["fat_ns"] > 0:
                daf += ff * dsw / (4.0 * p["fat_ns"]) * arr_f
        df = dfw * daw + dff * daf
        if zq is not None:
            zq2 = depin_step(p, zq, vv, dt, qtaus, dsw)
            df += dfz * float(np.mean(zq2 - zq))
            zq = zq2
        dpi += max(df, 0.0) * 0.5 * (x0 + x1) - max(-df, 0.0) * ratio
        a = np.exp(-dt * float(imprint_accel(p, vv)) / taus)
        sj = sj * a + ((1.0 - f) * 0.5 * (x0 + x1) + pi) * (1.0 - a)
        s = s2
        xs.append(x1)
    return s, h, sj, act, actf, np.array(xs), f, dom, dpi, zq


def cycle_history(p, t, v, n_out, nd=None, temp=None, history=None, s_start=-1.0, sub=200, df_max=0.01,
                  dsi_max=0.02):
    """Slow states and loop figures after ``n`` repetitions of one period (t, v), for every n in ``n_out``.

    Two-time-scale (cycle-averaged) integration of E13-E17 for vdep = 0.  Each
    macro step simulates one period with the slow states frozen (closed-form
    :func:`step` on about ``sub`` pieces), takes the per-period increments
    -- a_w from the switched amount, a_F from :func:`stress_per_period` and the
    switched amount weighted by the field factor, the
    imprint states from their exact one-period affine map -- and repeats them
    for m periods, with m chosen so that f changes by at most ``df_max`` and the
    mean imprint state by at most ``dsi_max`` (beyond its change in one period)
    per macro step.  1e12 periods take seconds.  :func:`simulate` gives the same result for small n
    (checked in the tests).

    Returns a dict of arrays over ``n_out``: ``n``, ``W``, ``D``, ``f``,
    ``s_i``, ``dP`` (switched charge per half period, P_s (1 - f) (max x -
    min x), C/m^2), ``vc_p`` and ``vc_m`` (voltages at which the mean class
    state crosses its mid value while rising and falling; NaN if it does not
    switch), ``a_w`` and ``a_f`` (the ages, e.g. for :class:`History`), ``Z``
    (mean E18 depinned state, 0 when off), ``zq`` (the E18 states), ``leak``
    (leakage multiplier of E20) and ``state`` (final ``(s, h, a_w, a_F, sigma_j, pi)``).
    The E18 states use their exact one-period affine map, like the imprint
    states, and E19 the periodic steady state of the lagged voltage.
    """
    if p["vdep"] != 0:
        raise ValueError("cycle_history assumes vdep = 0")
    temp = p["t0"] if temp is None else temp
    nd = int(nd if nd is not None else p["nd"])
    hist = history or History()
    wake, fat, imp = aging_flags(p)
    taus = imprint_taus(p, temp)
    dts, vms, ves = _refine_period(t, v, sub)
    t_eff = float(np.sum(dts * imprint_accel(p, vms)))   # imprint time per period (E17b); = period when off
    vstart = np.concatenate([[float(np.asarray(v, float)[0])], ves[:-1]])
    da_ft = stress_per_period(p, t, v, temp=temp) if fat else 0.0
    vl_starts = _lag_starts(p, dts, vms) if lag_on(p) else None
    nq = n_depin(p)
    qtaus = depin_taus(p, temp) if nq else None
    zq = np.broadcast_to(np.asarray(hist.zq, float), (nq,)).copy() if nq else None
    lnaq0 = -float(np.sum((depin_drive(p, vms) + p["wk_rho"]) * dts)) if nq else None
    a_w, a_f = hist.nw / p["wk_nw"], hist.af
    arr_w, arr_f = arrhenius(p, p["wk_ea"], temp), arrhenius(p, p["fat_ea"], temp)
    s = np.full(nd, float(s_start))
    h = np.zeros(nd)
    sj = np.broadcast_to(np.asarray(hist.si, float), taus.shape).copy()

    def zmean(z):
        return float(np.mean(z)) if z is not None else 0.0

    f0 = float(pinned_fraction(p, a_w, a_f, zmean(zq))[0])
    pi = f0 * float(s_start) if hist.pi is None else hist.pi
    out = {k: [] for k in ("n", "W", "D", "f", "s_i", "dP", "vc_p", "vc_m", "a_w", "a_f", "Z", "leak")}
    zq_out = []
    n_out = np.sort(np.asarray(n_out, float))
    ps = ps_of_t(p, temp)
    n_done = 0.0

    def record(n):
        # one more period at the current slow state (not counted) gives the loop figures
        xs, f = _one_period(p, nd, temp, dts, vms, s, h, a_w, a_f, sj, pi, taus, zq, qtaus, vl_starts)[5:7]
        f_, _, _, uw, D = pinned_fraction(p, a_w, a_f, zmean(zq))
        out["n"].append(n)
        out["a_w"].append(a_w)
        out["a_f"].append(a_f)
        out["W"].append(1.0 - float(uw))
        out["D"].append(float(D))
        out["f"].append(float(f_))
        out["Z"].append(zmean(zq))
        out["leak"].append(float(leak_factor(p, a_f)))
        zq_out.append(None if zq is None else zq.copy())
        out["s_i"].append(float(np.mean(sj)) if imp else 0.0)
        out["dP"].append(ps * (1.0 - f) * (xs.max() - xs.min()))
        mid = 0.5 * (xs.max() + xs.min())
        vc = {}
        for sign, key in ((1, "vc_p"), (-1, "vc_m")):
            vc[key] = np.nan
            if xs.max() - xs.min() > 1e-3:
                d = sign * (xs - mid)
                idx = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
                if idx.size:
                    # crossing within piece i: interpolate between its start and end voltages
                    i = idx[0]
                    frac = -d[i] / (d[i + 1] - d[i])
                    vc[key] = float(vstart[i] + frac * (ves[i] - vstart[i]))
            out[key].append(vc[key])

    for n_target in n_out:
        while n_done < n_target:
            s_new, h_new, sj_new, act, actf, xs, f, dom, dpi, zq_new = _one_period(
                p, nd, temp, dts, vms, s, h, a_w, a_f, sj, pi, taus, zq, qtaus, vl_starts)
            dw = act / (4.0 * p["wk_nw"]) * arr_w if wake else 0.0
            da_f = da_ft + (actf / (4.0 * p["fat_ns"]) * arr_f if fat and p["fat_ns"] > 0 else 0.0)
            # how many periods may share these increments
            f1 = float(pinned_fraction(p, a_w + dw, a_f + da_f, zmean(zq_new))[0])
            dfdn = abs(f1 - f)
            lna = -t_eff / taus
            a_sj = np.exp(lna)
            b_sj = sj_new - a_sj * sj
            m = n_target - n_done
            if dfdn > 0:
                m = min(m, max(1.0, np.floor(df_max / dfdn)))
            if imp:
                # change of the mean imprint state over m periods of the affine map; the fast states
                # jump to their fixed point at once, so only the slow ones limit m
                gap = b_sj / (1.0 - a_sj) - sj

                def dsi(mm):
                    return float(np.mean(-np.expm1(mm * lna) * gap))

                lim = dsi_max + abs(dsi(1.0))
                if abs(dsi(m)) > lim:
                    lo_m, hi_m = 1.0, m
                    while hi_m - lo_m > max(1.0, 1e-3 * lo_m):
                        mid_m = np.sqrt(lo_m * hi_m)
                        if abs(dsi(mid_m)) > lim:
                            hi_m = mid_m
                        else:
                            lo_m = mid_m
                    m = max(1.0, np.floor(lo_m))
            if nq:
                # the same for the E18 states (a change of Z moves f by f_q dZ); the per-reversal re-pinning
                # adds t_r act/4 to the exponent of the one-period map
                lnaq = (lnaq0 - p.get("wk_tr", 0.0) * act / 4.0) / qtaus
                a_q = np.exp(lnaq)
                b_q = zq_new - a_q * zq
                with np.errstate(divide="ignore", invalid="ignore"):
                    gq = np.where(1.0 - a_q > 1e-15, b_q / (1.0 - a_q) - zq, 0.0)

                def dzm(mm):
                    return float(np.mean(np.where(1.0 - a_q > 1e-15, -np.expm1(mm * lnaq) * gq, mm * b_q)))

                lim = df_max / p["wk_fq"] + abs(dzm(1.0))
                if abs(dzm(m)) > lim:
                    lo_m, hi_m = 1.0, m
                    while hi_m - lo_m > max(1.0, 1e-3 * lo_m):
                        mid_m = np.sqrt(lo_m * hi_m)
                        if abs(dzm(mid_m)) > lim:
                            hi_m = mid_m
                        else:
                            lo_m = mid_m
                    m = max(1.0, np.floor(lo_m))
            if abs(dpi) > 0:
                m = min(m, max(1.0, np.floor(df_max / max(abs(dpi), 1e-300))))
            m = max(1.0, np.floor(m))
            if nq:
                am = a_q**m
                with np.errstate(divide="ignore", invalid="ignore"):
                    geo = np.where(1.0 - a_q > 1e-15, (1.0 - am) / (1.0 - a_q), m)
                zq = np.clip(am * zq + b_q * geo, 0.0, 1.0)
            # pinned pool: m times the change of one period, kept within |pi| <= f
            f_end = float(pinned_fraction(p, a_w + m * dw, a_f + m * da_f, zmean(zq))[0])
            pi = float(np.clip(pi + m * dpi, -f_end, f_end))
            a_w += m * dw
            a_f += m * da_f
            if imp:
                am = a_sj**m
                with np.errstate(divide="ignore", invalid="ignore"):
                    geo = np.where(1.0 - a_sj > 1e-15, (1.0 - am) / (1.0 - a_sj), m)
                sj = am * sj + b_sj * geo
            s, h = s_new, h_new
            n_done += m
        record(n_target)
    res = {k: np.array(v_) for k, v_ in out.items()}
    res["state"] = (s, h, a_w, a_f, sj, pi)
    if nq:
        res["zq"] = np.array(zq_out)
    return res


def loop_figures(p, amp, freq, nd=None, temp=None, history=None, npts=400, n_pre=1, leak=False):
    """P-V loop of a triangular read-out as a dynamic-hysteresis measurement sees it, slow states frozen.

    The film is preset to the negative state and driven by ``n_pre + 1``
    periods of a 0 -> +amp -> -amp -> 0 triangle at ``freq``; the last period
    is the loop.  The loop is the terminal charge density P + q_rev (the
    dielectric and reversible charge are part of what the integrated current
    measures), plus the integrated leakage charge when ``leak`` is True (an
    uncompensated measurement).  It is centred vertically, as a loop from
    current integration is.  The slow states (wake-up, fatigue, imprint, E18)
    are taken from ``history`` and frozen during the read-out.

    Returns ``pr_p``, ``pr_m`` (charge at V = 0 on the falling and rising
    branches, C/m^2), ``vc_p``, ``vc_m`` (zero crossings on the rising and
    falling branches, V; NaN if there is none), and the loop ``v``, ``q``.
    In a pinched loop the zero crossings sit where the first sub-population
    has switched, which is why the width 2E_c grows as the loop opens.
    """
    temp = p["t0"] if temp is None else temp
    hist = history or History()
    wake, fat, imp = aging_flags(p)
    a_w, a_f = hist.nw / p["wk_nw"], hist.af
    s_i = float(np.mean(hist.si)) if imp else 0.0
    zq = float(np.mean(hist.zq)) if depin_on(p) else 0.0
    pe, dom, f = aged_card(p, nd, temp, a_w, a_f, s_i, zq)
    T = 1.0 / freq
    tt = np.linspace(0.0, T, npts + 1)
    ph = tt * freq
    vv = amp * np.where(ph < 0.25, 4 * ph, np.where(ph < 0.75, 2 - 4 * ph, 4 * ph - 4))
    vm = 0.5 * (vv[:-1] + vv[1:])
    dt = np.diff(tt)
    s, h = np.full(len(dom.w), -1.0), np.zeros(len(dom.w))
    for _ in range(n_pre):
        for a, b in zip(dt, vm):
            s, h = step(pe, dom, s, h, b, a, temp)
    xs = [float(dom.w @ s)]
    for a, b in zip(dt, vm):
        s, h = step(pe, dom, s, h, b, a, temp)
        xs.append(float(dom.w @ s))
    xs = np.array(xs)
    ps = ps_of_t(p, temp)
    q = np.array([ps * (1.0 - f) * x + q_rev_aged(p, V, x, V, temp, f, s_i) for x, V in zip(xs, vv)])
    if leak:
        jl = j_leak(p, vm, temp) * leak_factor(p, a_f)
        q = q + np.concatenate([[0.0], np.cumsum(jl * dt)])
    q = q - 0.5 * (q.max() + q.min())
    i_top, i_bot = int(np.argmax(vv)), int(np.argmin(vv))
    fall_v, fall_q = vv[i_top:i_bot + 1], q[i_top:i_bot + 1]
    rise_v = np.concatenate([vv[i_bot:], vv[1:i_top + 1]])
    rise_q = np.concatenate([q[i_bot:], q[1:i_top + 1] + (q[-1] - q[0])])

    def cross(x, y):
        idx = np.nonzero((y[:-1] < 0) != (y[1:] < 0))[0]
        if not idx.size:
            return np.nan
        i = idx[0]
        return float(x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i]))

    return {"pr_p": cross(fall_q, fall_v), "pr_m": cross(rise_q, rise_v), "vc_p": cross(rise_v, rise_q),
            "vc_m": cross(fall_v, fall_q), "v": vv, "q": q, "f": f}


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
             rtol=1e-7, atol=1e-9, sigma0=None, max_step=np.inf, history=None):
    """Integrate the model for an arbitrary drive ``v_of_t(t)`` [V] from 0 to ``t_end`` [s].

    ``breaks`` are times where ``v_of_t`` has kinks (the integration restarts
    there).  ``s0`` is a scalar or one value per class (default ``p["s0"]``);
    h starts at 0, y at v(0) and sigma at ``sigma0`` (default 0).

    Returns a dict with ``t``, ``v``, ``s``, ``h``, ``sig``, ``x``, ``y``,
    ``qd`` (terminal charge density, C/m^2), ``q`` (= qd * area, C),
    ``i_leak`` (A) and ``dom``.

    When any history term is on (:func:`aging_on`), the state also carries the
    pinned polarization, the wake-up and fatigue ages and the imprint states
    (E13-E17), starting from ``history`` (a :class:`History`, default pristine);
    the result then also has ``pi``, ``r``, ``a_w``, ``a_f``, ``sj``, ``s_i``, ``f``,
    ``W``, ``D`` and ``x_t`` (total polarization state in units of P_s).
    """
    temp = p["t0"] if temp is None else temp
    dom = make_domains(p, nd, temp)
    n = len(dom.w)
    s0 = np.full(n, float(p["s0"])) if s0 is None else np.broadcast_to(np.asarray(s0, float), (n,)).copy()
    sig0 = 0.0 if sigma0 is None else sigma0
    if aging_on(p):
        return _simulate_aged(p, v_of_t, t_end, s0, dom, temp, t_eval, breaks, rtol, atol, sig0, max_step, history)

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
