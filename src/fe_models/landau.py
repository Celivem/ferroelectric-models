r"""Landau-Devonshire free energy and Landau-Khalatnikov dynamics.

Model
-----
For a uniaxial ferroelectric with polarization ``P`` (C/m^2) in an applied
field ``E`` (V/m) the Landau-Devonshire free-energy density (J/m^3) is

.. math::

    F(P) = \alpha P^2 + \beta P^4 + \gamma P^6 - E P .

Static equilibria satisfy :math:`\partial F/\partial P = 0`, i.e.

.. math::

    E(P) = 2\alpha P + 4\beta P^3 + 6\gamma P^5 ,

which is the S-shaped curve. The part with :math:`\partial E/\partial P < 0`
(equivalently :math:`\partial^2 F/\partial P^2 < 0`) is the unstable
"negative-capacitance" branch.

Relation between (alpha, beta) and (Pr, Ec) for the 2-4 model
-------------------------------------------------------------
With :math:`\gamma = 0`, :math:`\alpha < 0` and :math:`\beta > 0`:

* Remanent polarization (minimum of F at E = 0):
  :math:`2\alpha P + 4\beta P^3 = 0 \Rightarrow P_r = \sqrt{-\alpha/(2\beta)}`.
* Coercive field: the stable branch ends where :math:`dE/dP = 0`,
  :math:`2\alpha + 12\beta P^2 = 0 \Rightarrow P_c^2 = -\alpha/(6\beta) = P_r^2/3`.
  Inserting :math:`P = -P_c` into :math:`E(P) = P(2\alpha + 4\beta P^2)` gives
  :math:`E = -P_c(2\alpha - 2\alpha/3) = -\tfrac{4}{3}\alpha P_c`, so

  .. math::

      E_c = \frac{4}{3\sqrt{3}}\,(-\alpha)\,P_r .

* Inverting: :math:`\alpha = -\dfrac{3\sqrt{3}}{4}\dfrac{E_c}{P_r}`,
  :math:`\beta = \dfrac{3\sqrt{3}}{8}\dfrac{E_c}{P_r^3}`.

(With the other common convention :math:`F = \tfrac{a}{2}P^2 + \tfrac{b}{4}P^4`
the same physics reads :math:`P_r = \sqrt{-a/b}`,
:math:`E_c = \tfrac{2}{3\sqrt{3}}(-a)P_r`; the factor 2 is absorbed in ``a``.)

The intrinsic (Landau) coercive field is the field at which the metastable
minimum disappears. Real films switch by nucleation at much lower fields, so
in this module ``Ec`` is a fitting handle that sets the loop width, not a
prediction of a measured coercive field.

Dynamics
--------
Landau-Khalatnikov (LK) relaxation with a phenomenological viscosity
``rho`` (Ohm m):

.. math::

    \rho \frac{dP}{dt} = -\frac{\partial F}{\partial P}
                       = E(t) - 2\alpha P - 4\beta P^3 - 6\gamma P^5 .

Under a finite sweep rate the polarization lags behind the static curve, so
the dynamic coercive field exceeds the static one and grows with frequency.

References
----------
* A. F. Devonshire, Phil. Mag. 40, 1040 (1949).
* L. D. Landau and I. M. Khalatnikov, Dokl. Akad. Nauk SSSR 96, 469 (1954).

All default numbers are illustrative HfO2-like values, not fits to a device.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .measurement import triangle_wave
from .units import HFO2_ILLUSTRATIVE

__all__ = [
    "coefficients_from_pr_ec",
    "free_energy",
    "field_from_polarization",
    "stiffness",
    "remanent_polarization",
    "coercive_field",
    "s_curve",
    "static_branches",
    "static_loop",
    "lk_time_constant",
    "rho_from_time_constant",
    "lk_simulate",
    "lk_loop",
]


def coefficients_from_pr_ec(Pr=HFO2_ILLUSTRATIVE["Pr"], Ec=HFO2_ILLUSTRATIVE["Ec"]):
    """Return ``(alpha, beta)`` of the 2-4 model with the given Pr and Ec.

    Parameters
    ----------
    Pr : float
        Remanent polarization in C/m^2 (> 0).
    Ec : float
        Intrinsic (Landau) coercive field in V/m (> 0).

    Returns
    -------
    alpha : float
        Quadratic coefficient in m/F (negative).
    beta : float
        Quartic coefficient in m^5/(F C^2) (positive).

    Notes
    -----
    ``alpha = -3*sqrt(3)/4 * Ec/Pr`` and ``beta = 3*sqrt(3)/8 * Ec/Pr**3``;
    see the module docstring for the derivation.
    """
    if Pr <= 0 or Ec <= 0:
        raise ValueError("Pr and Ec must be positive")
    s3 = np.sqrt(3.0)
    alpha = -3.0 * s3 / 4.0 * Ec / Pr
    beta = 3.0 * s3 / 8.0 * Ec / Pr**3
    return alpha, beta


def free_energy(P, alpha, beta, gamma=0.0, E=0.0):
    """Landau free-energy density ``alpha P^2 + beta P^4 + gamma P^6 - E P`` (J/m^3)."""
    P = np.asarray(P, dtype=float)
    return alpha * P**2 + beta * P**4 + gamma * P**6 - E * P


def field_from_polarization(P, alpha, beta, gamma=0.0):
    """Equilibrium field ``E(P) = dF/dP`` at zero applied field (the S-curve), V/m."""
    P = np.asarray(P, dtype=float)
    return 2 * alpha * P + 4 * beta * P**3 + 6 * gamma * P**5


def stiffness(P, alpha, beta, gamma=0.0):
    """Second derivative ``d^2F/dP^2 = dE/dP`` (m/F). Negative on the NC branch."""
    P = np.asarray(P, dtype=float)
    return 2 * alpha + 12 * beta * P**2 + 30 * gamma * P**4


def remanent_polarization(alpha, beta, gamma=0.0):
    """Positive spontaneous polarization at E = 0 (C/m^2).

    Solves ``2 alpha + 4 beta x + 6 gamma x^2 = 0`` for ``x = P^2`` and keeps
    the largest root that is a local minimum of F. Returns 0 if the only
    minimum is ``P = 0``.
    """
    roots = np.roots([6 * gamma, 4 * beta, 2 * alpha]) if gamma != 0 else np.array([-alpha / (2 * beta)])
    candidates = []
    for x in np.atleast_1d(roots):
        if abs(np.imag(x)) < 1e-12 * max(1.0, abs(x)) and np.real(x) > 0:
            P = np.sqrt(np.real(x))
            if stiffness(P, alpha, beta, gamma) > 0:
                candidates.append(P)
    return max(candidates) if candidates else 0.0


def coercive_field(alpha, beta, gamma=0.0):
    """Intrinsic coercive field (V/m) from the free energy.

    The negative-P minimum of F disappears at the field where ``dE/dP = 0``
    on the negative side of the S-curve. With ``x = P^2`` the condition is
    ``2 alpha + 12 beta x + 30 gamma x^2 = 0``. Among the positive roots the
    coercive field is ``-min E(sqrt(x))`` (the deepest local minimum of the
    S-curve on the P > 0 side, mirrored by symmetry).

    Raises
    ------
    ValueError
        If the parameters give no single hysteresis loop (no positive root or
        a positive minimum of E, e.g. a double loop).
    """
    if gamma != 0:
        roots = np.roots([30 * gamma, 12 * beta, 2 * alpha])
    else:
        roots = np.array([-2 * alpha / (12 * beta)])
    xs = [np.real(x) for x in np.atleast_1d(roots) if abs(np.imag(x)) < 1e-12 * max(1.0, abs(x)) and np.real(x) > 0]
    if not xs:
        raise ValueError("no extremum of E(P): the free energy has no hysteresis")
    e_min = min(float(field_from_polarization(np.sqrt(x), alpha, beta, gamma)) for x in xs)
    if e_min >= 0:
        raise ValueError("S-curve minimum is not below zero: no single loop (double loop?)")
    return -e_min


def s_curve(alpha, beta, gamma=0.0, P_max=None, n=1001):
    """Full S-curve including the negative-capacitance branch.

    Returns
    -------
    P : ndarray
        Polarization grid from ``-P_max`` to ``P_max`` (C/m^2).
    E : ndarray
        Equilibrium field ``E(P)`` (V/m).
    stable : ndarray of bool
        True where ``d^2F/dP^2 > 0`` (locally stable), False on the NC branch.
    """
    if P_max is None:
        P_max = 1.6 * remanent_polarization(alpha, beta, gamma)
    P = np.linspace(-P_max, P_max, n)
    return P, field_from_polarization(P, alpha, beta, gamma), stiffness(P, alpha, beta, gamma) > 0


def _stable_roots(E, alpha, beta, gamma):
    coeffs = [6 * gamma, 0.0, 4 * beta, 0.0, 2 * alpha, -E] if gamma != 0 else [4 * beta, 0.0, 2 * alpha, -E]
    r = np.roots(coeffs)
    scale = max(1.0, np.max(np.abs(r)))
    real = np.real(r[np.abs(np.imag(r)) < 1e-9 * scale])
    return np.sort(real[stiffness(real, alpha, beta, gamma) > 0])


def static_branches(E, alpha, beta, gamma=0.0):
    """Lowest and highest locally stable polarization at each field.

    Returns ``(P_low, P_high)``; they coincide where only one minimum exists
    (``|E| > Ec``) and differ inside the bistable window.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    lo = np.empty_like(E)
    hi = np.empty_like(E)
    for i, e in enumerate(E):
        s = _stable_roots(e, alpha, beta, gamma)
        lo[i], hi[i] = s[0], s[-1]
    return lo, hi


def static_loop(E_max, alpha, beta, gamma=0.0, n=801):
    """Quasi-static (infinitely slow, noise-free) Landau hysteresis loop.

    The ascending branch follows the lowest stable minimum (it jumps up at
    +Ec when that minimum disappears); the descending branch follows the
    highest one.

    Returns
    -------
    E, P : ndarray
        Field (V/m) and polarization (C/m^2) along the closed path
        ``-E_max -> +E_max -> -E_max``.
    """
    E_up = np.linspace(-E_max, E_max, n)
    lo, _ = static_branches(E_up, alpha, beta, gamma)
    E_dn = E_up[::-1]
    _, hi = static_branches(E_dn, alpha, beta, gamma)
    return np.concatenate([E_up, E_dn]), np.concatenate([lo, hi])


def lk_time_constant(alpha, beta, rho, gamma=0.0):
    """Small-signal LK relaxation time ``rho / F''(Pr)`` near the remanent state (s).

    For the 2-4 model ``F''(Pr) = -4 alpha``.
    """
    Pr = remanent_polarization(alpha, beta, gamma)
    return rho / float(stiffness(Pr, alpha, beta, gamma))


def rho_from_time_constant(tau, alpha, beta, gamma=0.0):
    """Viscosity ``rho`` (Ohm m) that gives a small-signal relaxation time ``tau``."""
    Pr = remanent_polarization(alpha, beta, gamma)
    return tau * float(stiffness(Pr, alpha, beta, gamma))


def lk_simulate(t, E_func, alpha, beta, rho, gamma=0.0, P0=None, rtol=1e-8, atol=None, max_step=None):
    """Integrate ``rho dP/dt = E(t) - dF/dP`` with :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    t : array_like
        Output times (s), increasing.
    E_func : callable
        ``E_func(t) -> field in V/m``; any waveform.
    alpha, beta, gamma : float
        Landau coefficients (SI).
    rho : float
        LK viscosity (Ohm m).
    P0 : float, optional
        Initial polarization; default ``-Pr``.
    max_step : float, optional
        Largest solver step; default is the output spacing so that features
        of the drive waveform are not stepped over.

    Returns
    -------
    P : ndarray
        Polarization (C/m^2) at the times ``t``.
    """
    t = np.asarray(t, dtype=float)
    Pr = remanent_polarization(alpha, beta, gamma)
    if P0 is None:
        P0 = -Pr
    if atol is None:
        atol = 1e-9 * max(Pr, 1e-6)
    if max_step is None:
        max_step = float(np.min(np.diff(t))) if t.size > 1 else np.inf

    def rhs(tt, P):
        return (E_func(tt) - field_from_polarization(P, alpha, beta, gamma)) / rho

    def jac(tt, P):
        return np.array([[-float(stiffness(P[0], alpha, beta, gamma)) / rho]])

    sol = solve_ivp(rhs, (t[0], t[-1]), [P0], t_eval=t, method="LSODA", jac=jac,
                    rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def lk_loop(frequency, E_amp, alpha, beta, rho, gamma=0.0, n_points=2000, n_cycles=1):
    """Dynamic P-E loop under a triangular field of the given frequency (Hz).

    The field starts at 0 and rises first; the film starts in the ``-Pr``
    state, so the first cycle is already a full loop. Only the last cycle is
    returned.

    Returns
    -------
    t, E, P : ndarray
        Time (s), field (V/m) and polarization (C/m^2) over one period.
    """
    T = 1.0 / frequency
    t = np.linspace(0.0, n_cycles * T, n_cycles * n_points + 1)

    def E_func(tt):
        return triangle_wave(tt, E_amp, frequency)

    P = lk_simulate(t, E_func, alpha, beta, rho, gamma=gamma)
    sl = slice((n_cycles - 1) * n_points, None)
    return t[sl] - t[sl][0], E_func(t[sl]), P[sl]
