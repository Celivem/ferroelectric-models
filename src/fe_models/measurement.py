r"""Synthetic electrical measurements on a ferroelectric capacitor.

This module produces and analyses the signals one gets from a
metal / ferroelectric / metal capacitor:

* drive waveforms: triangular wave and a PUND pulse train;
* the terminal current ``I = A dD/dt + I_leak(V)`` with
  ``D = eps0 eps_r E + P_sw(E history)`` and ``E = V / thickness``
  (uniform field, no interface layer, no series resistance);
* P-E loops from current integration, leakage compensation, and extraction
  of Pr, +/-Ec and imprint;
* a phenomenological wake-up / fatigue curve.

The switchable polarization ``P_sw`` comes from any callable that maps a
field history to polarization, for example
:meth:`fe_models.preisach.PreisachModel.apply`. Because the Preisach model is
rate independent, its current scales with ``dV/dt``; that property is what the
two-frequency leakage compensation below relies on.

All defaults are illustrative HfO2-like numbers (10 nm, eps_r = 30,
100 um x 100 um), not parameters of a measured device.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from .units import EPS0, HFO2_ILLUSTRATIVE

__all__ = [
    "triangle_wave",
    "pund_waveform",
    "capacitor_current",
    "polarization_from_current",
    "analyze_pund",
    "estimate_leakage_resistance",
    "compensate_leakage_ohmic",
    "compensate_leakage_two_frequency",
    "extract_loop_parameters",
    "wakeup_fatigue_form",
    "cycling_pr",
]


# --------------------------------------------------------------------------
# Waveforms
# --------------------------------------------------------------------------
def triangle_wave(t, amplitude, frequency):
    """Triangular wave that starts at 0, rises to ``+amplitude`` at T/4,
    falls to ``-amplitude`` at 3T/4 and returns to 0 at T.

    Works for scalar or array ``t``.
    """
    phase = np.mod(np.asarray(t, dtype=float) * frequency, 1.0)
    out = np.where(phase < 0.25, 4 * phase,
                   np.where(phase < 0.75, 2 - 4 * phase, 4 * phase - 4))
    out = amplitude * out
    return float(out) if np.ndim(out) == 0 else out


def pund_waveform(amplitude=3.0, rise=1e-6, width=1e-6, delay=1e-6, n_per_segment=200, preset=True):
    """Piecewise-linear PUND pulse train.

    Sequence: [preset (-)] -> P (+) -> U (+) -> N (-) -> D (-), each pulse a
    trapezoid (rise, plateau ``width``, fall = rise) separated by ``delay``
    at 0 V. The preset pulse puts the film in the negative remanent state.

    Returns
    -------
    t : ndarray
        Time (s). Every corner of the waveform is a sample point.
    V : ndarray
        Voltage (V).
    windows : dict
        ``windows[name] = (t_start, t_top_end, t_end)`` for name in
        ``"P", "U", "N", "D"`` (and ``"preset"``): start of the rise, end of
        the plateau, end of the fall.
    """
    corners_t = [0.0]
    corners_v = [0.0]
    windows = {}
    names = (["preset"] if preset else []) + ["P", "U", "N", "D"]
    signs = {"preset": -1, "P": 1, "U": 1, "N": -1, "D": -1}
    tc = 0.0
    for name in names:
        tc += delay
        corners_t.append(tc); corners_v.append(0.0)
        t0 = tc
        tc += rise
        corners_t.append(tc); corners_v.append(signs[name] * amplitude)
        tc += width
        corners_t.append(tc); corners_v.append(signs[name] * amplitude)
        t_top = tc
        tc += rise
        corners_t.append(tc); corners_v.append(0.0)
        windows[name] = (t0, t_top, tc)
    tc += delay
    corners_t.append(tc); corners_v.append(0.0)

    t_parts = [np.linspace(a, b, n_per_segment, endpoint=False) for a, b in zip(corners_t[:-1], corners_t[1:])]
    t = np.concatenate(t_parts + [np.array([corners_t[-1]])])
    V = np.interp(t, corners_t, corners_v)
    return t, V, windows


# --------------------------------------------------------------------------
# Device model
# --------------------------------------------------------------------------
def capacitor_current(t, V, *, area=HFO2_ILLUSTRATIVE["area"], thickness=HFO2_ILLUSTRATIVE["thickness"],
                      eps_r=HFO2_ILLUSTRATIVE["eps_r"], polarization=None, leakage=None):
    """Terminal current of a ferroelectric capacitor driven by ``V(t)``.

    ``I = A d/dt[eps0 eps_r E + P_sw] + I_leak(V)`` with ``E = V/thickness``.

    Parameters
    ----------
    t, V : array_like
        Time (s) and voltage (V) samples.
    polarization : callable or None
        ``polarization(E_array) -> P_sw array`` (C/m^2) evaluated along the
        whole field history (it may be stateful, e.g. ``PreisachModel.apply``).
        ``None`` means a purely linear dielectric.
    leakage : None, float or callable
        ``None``: no leakage. A float is an ohmic resistance in Ohm.
        A callable is ``I_leak(V)`` in A (any nonlinear law).

    Returns
    -------
    I : ndarray
        Current (A).
    """
    t = np.asarray(t, dtype=float)
    V = np.asarray(V, dtype=float)
    E = V / thickness
    D = EPS0 * eps_r * E
    if polarization is not None:
        D = D + np.asarray(polarization(E), dtype=float)
    I = area * np.gradient(D, t)
    if leakage is not None:
        I = I + (V / leakage if np.isscalar(leakage) else np.asarray(leakage(V), dtype=float))
    return I


def polarization_from_current(t, I, area=HFO2_ILLUSTRATIVE["area"], center=True):
    """Integrate current to polarization ``P = (1/A) int I dt`` (C/m^2).

    With ``center=True`` the loop is shifted so that ``max(P) = -min(P)``,
    as is done for measured loops where the initial state is unknown.
    """
    P = cumulative_trapezoid(np.asarray(I, dtype=float), np.asarray(t, dtype=float), initial=0.0) / area
    if center:
        P = P - 0.5 * (P.max() + P.min())
    return P


# --------------------------------------------------------------------------
# PUND
# --------------------------------------------------------------------------
def analyze_pund(t, I, windows, area=HFO2_ILLUSTRATIVE["area"]):
    """Integrate the PUND current pulse by pulse.

    For each pulse two charges are computed:

    * ``on``: rise + plateau (``P*``, ``U*``, ... ; includes the
      non-remanent, back-switching and dielectric parts);
    * ``full``: rise + plateau + fall (remanent change only, because the
      linear dielectric charge returns on the falling edge).

    The switched (remanent) polarization is taken from the "on" charges,
    ``dP = (Q_P - Q_U)/A`` and ``(Q_D - Q_N)/A`` for the negative side, which
    removes the dielectric and the leakage charge (identical in P and U), and
    ``Pr = dP/2``.

    Returns
    -------
    dict with keys ``Q_on``, ``Q_full`` (dicts of charges in C),
    ``dP_pos``, ``dP_neg`` (C/m^2, both positive for a normal loop),
    ``Pr`` (C/m^2, mean of the two sides divided by 2) and
    ``nonswitching`` (``Q_U/A``, C/m^2).
    """
    t = np.asarray(t, dtype=float)
    I = np.asarray(I, dtype=float)

    def integ(a, b):
        m = (t >= a) & (t <= b)
        return float(trapezoid(I[m], t[m]))

    Q_on = {k: integ(w[0], w[1]) for k, w in windows.items()}
    Q_full = {k: integ(w[0], w[2]) for k, w in windows.items()}
    dP_pos = (Q_on["P"] - Q_on["U"]) / area
    dP_neg = (Q_on["D"] - Q_on["N"]) / area
    return {
        "Q_on": Q_on,
        "Q_full": Q_full,
        "dP_pos": dP_pos,
        "dP_neg": dP_neg,
        "Pr": 0.25 * (dP_pos + dP_neg),
        "nonswitching": Q_on["U"] / area,
    }


# --------------------------------------------------------------------------
# Leakage compensation
# --------------------------------------------------------------------------
def estimate_leakage_resistance(t, V, I, skip=1):
    """Estimate an ohmic leakage resistance from a triangular-wave measurement.

    At a voltage extremum the displacement current reverses sign
    (``+C|dV/dt|`` just before, ``-C|dV/dt|`` just after) while the ohmic
    current ``V/R`` is continuous. Averaging the currents ``skip`` samples
    before and after each extremum therefore leaves ``V_tip/R``. This assumes
    the film is saturated at the tips (no switching current there) and a
    symmetric triangle.

    Returns
    -------
    R : float
        Estimated resistance (Ohm).
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)
    n = V.size
    conductances = []
    for i in range(skip, n - skip):
        if (V[i] >= V[i - 1] and V[i] > V[i + 1]) or (V[i] <= V[i - 1] and V[i] < V[i + 1]):
            if abs(V[i]) < 0.5 * np.max(np.abs(V)):
                continue
            I_avg = 0.5 * (I[i - skip] + I[i + skip])
            V_avg = 0.5 * (V[i - skip] + V[i + skip])
            conductances.append(I_avg / V_avg)
    if not conductances:
        raise ValueError("no voltage extrema found")
    G = float(np.mean(conductances))
    return np.inf if G <= 0 else 1.0 / G


def compensate_leakage_ohmic(V, I, R):
    """Subtract an ohmic leakage current ``V/R`` (A)."""
    return np.asarray(I, dtype=float) - np.asarray(V, dtype=float) / R


def compensate_leakage_two_frequency(I1, f1, I2, f2):
    """Remove a frequency-independent leakage current using two frequencies.

    ``I1`` and ``I2`` are currents measured with the same waveform shape at
    frequencies ``f1`` and ``f2`` and sampled at the same phase points (same
    voltages). If the displacement and switching currents scale with ``f``
    (rate-independent switching) and the leakage depends only on ``V``, then
    ``I_k = f_k g(V) + I_leak(V)`` and

    ``I_leak = (f2 I1 - f1 I2) / (f2 - f1)``.

    The leakage law need not be ohmic. The assumption that switching is rate
    independent is only approximate for real films, so ``f1`` and ``f2``
    should be close (a factor of 2 to 5). This idea is commonly called
    dynamic leakage current compensation.

    Returns
    -------
    I1_comp, I2_comp, I_leak : ndarray
    """
    I1 = np.asarray(I1, dtype=float)
    I2 = np.asarray(I2, dtype=float)
    if f1 == f2:
        raise ValueError("frequencies must differ")
    I_leak = (f2 * I1 - f1 * I2) / (f2 - f1)
    return I1 - I_leak, I2 - I_leak, I_leak


# --------------------------------------------------------------------------
# Loop parameters
# --------------------------------------------------------------------------
def extract_loop_parameters(E, P):
    """Extract Pr, +/-Ec and imprint from one closed P-E loop.

    ``E`` and ``P`` must be in time order and cover a full cycle. Branches
    are identified by the sign of ``dE`` between consecutive samples (the
    path is treated as closed). Crossings are linearly interpolated; if a
    crossing occurs more than once on a branch (noise), the crossings are
    averaged.

    Returns
    -------
    dict with
        ``Ec_plus`` (P crosses 0 upward on the ascending branch, V/m),
        ``Ec_minus`` (P crosses 0 downward on the descending branch, < 0),
        ``Ec = (Ec_plus - Ec_minus)/2``,
        ``imprint = (Ec_plus + Ec_minus)/2``,
        ``Pr_plus`` (P at E = 0 on the descending branch),
        ``Pr_minus`` (P at E = 0 on the ascending branch, < 0),
        ``Pr = (Pr_plus - Pr_minus)/2``,
        ``P_offset = (Pr_plus + Pr_minus)/2``.
    """
    E = np.asarray(E, dtype=float)
    P = np.asarray(P, dtype=float)
    E2 = np.append(E, E[0])
    P2 = np.append(P, P[0])
    e0, e1, p0, p1 = E2[:-1], E2[1:], P2[:-1], P2[1:]
    up = e1 > e0
    dn = e1 < e0

    def interp_zero(a0, a1, b0, b1, mask):
        # value of b where a crosses zero, for the masked sample pairs
        frac = a0[mask] / (a0[mask] - a1[mask])
        vals = b0[mask] + frac * (b1[mask] - b0[mask])
        if vals.size == 0:
            return np.nan
        return float(np.mean(vals))

    ec_p = interp_zero(p0, p1, e0, e1, up & (p0 < 0) & (p1 >= 0))
    ec_m = interp_zero(p0, p1, e0, e1, dn & (p0 > 0) & (p1 <= 0))
    pr_m = interp_zero(e0, e1, p0, p1, up & (e0 < 0) & (e1 >= 0))
    pr_p = interp_zero(e0, e1, p0, p1, dn & (e0 > 0) & (e1 <= 0))
    return {
        "Ec_plus": ec_p,
        "Ec_minus": ec_m,
        "Ec": 0.5 * (ec_p - ec_m),
        "imprint": 0.5 * (ec_p + ec_m),
        "Pr_plus": pr_p,
        "Pr_minus": pr_m,
        "Pr": 0.5 * (pr_p - pr_m),
        "P_offset": 0.5 * (pr_p + pr_m),
    }


# --------------------------------------------------------------------------
# Wake-up / fatigue (phenomenological)
# --------------------------------------------------------------------------
def wakeup_fatigue_form(N, Pr0=0.15, dPr_wake=0.05, N_wake=1e3, N_fatigue=1e8, k=1.0):
    """Phenomenological Pr(N): saturating wake-up times a power-law fatigue roll-off.

    ``Pr(N) = [Pr0 + dPr_wake (1 - exp(-N/N_wake))] / (1 + (N/N_fatigue)^k)``

    This is a curve shape, not a physical model: it encodes no mechanism
    (oxygen-vacancy redistribution, phase change, defect generation, ...).
    The defaults (Pr0 = 15 uC/cm^2, +5 uC/cm^2 wake-up) are illustrative.
    """
    N = np.asarray(N, dtype=float)
    return (Pr0 + dPr_wake * (1 - np.exp(-N / N_wake))) / (1 + (N / N_fatigue) ** k)


def cycling_pr(cycles, form=wakeup_fatigue_form, **params):
    """Evaluate a user-supplied phenomenological ``form(cycles, **params)`` for Pr vs cycles.

    Any callable can be passed; the default is :func:`wakeup_fatigue_form`.
    """
    return np.asarray(form(np.asarray(cycles, dtype=float), **params), dtype=float)
