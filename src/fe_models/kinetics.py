r"""Polarization-switching kinetics under a constant applied field.

All models here describe the switched fraction

.. math::

    f(t) = \frac{\Delta P(t)}{2 P_s} \in [0, 1]

after a field step of fixed amplitude is applied to a fully poled film.

**Assumption: constant field.** KAI, NLS and the Merz law (and similar
step-response models) are derived for a field that is switched on at
``t = 0`` and then held constant while the film switches. Fitting them to data
taken under ramped or triangular fields (for example the switching current of
a P-E loop) is outside their stated assumptions, because the switching rate
then changes during the measurement.

KAI
---
Kolmogorov-Avrami-Ishibashi (homogeneous nucleation and unrestricted domain
growth in an infinite medium):

.. math:: f(t) = 1 - \exp[-(t/t_0)^n] .

In the Avrami picture ``n`` is set by the growth dimensionality and by
whether nuclei are pre-existing or form continuously. ``f(t_0) = 1 - 1/e``
for every ``n``.

NLS
---
Nucleation-limited switching: the film is an ensemble of independent regions,
each switching with its own characteristic time; switching is limited by
nucleation, not by domain-wall motion. With ``u = log10(t0)`` distributed as
``F(u)``,

.. math:: f(t) = \int \{1 - \exp[-(t/10^u)^n]\} F(u)\, du .

Two distributions are provided:

* ``"lorentzian"`` -- :math:`F(u) \propto w / [(u - u_1)^2 + w^2]`,
  the form used by Jo et al. for disordered thin films; and
* ``"lognormal"`` -- Gaussian in ``u`` (log-normal in ``t0``).

A broad ``F`` gives the slow, stretched switching seen in polycrystalline
thin films; ``w -> 0`` recovers KAI.

Merz law
--------
Field dependence of the characteristic switching time:

.. math:: t_0(E) = t_\infty \exp[(E_a/E)^m] .

``fit_merz`` fits ``ln t0`` versus ``E``. ``nls_fraction_field`` lets the NLS
centre time ``10^{u_1}`` follow this law at fixed width.

References
----------
* Y. Ishibashi and Y. Takagi, J. Phys. Soc. Jpn. 31, 506 (1971).
* A. K. Tagantsev et al., Phys. Rev. B 66, 214109 (2002).
* J. Y. Jo et al., Phys. Rev. Lett. 99, 267602 (2007).
* W. J. Merz, Phys. Rev. 95, 690 (1954).

Default parameters are illustrative, not fitted to any device.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "kai_fraction",
    "merz_time",
    "log_time_distribution",
    "nls_fraction",
    "nls_fraction_field",
    "fit_kai",
    "fit_nls",
    "fit_merz",
]


def kai_fraction(t, t0, n=2.0):
    """KAI switched fraction ``1 - exp[-(t/t0)^n]``.

    Parameters
    ----------
    t : array_like
        Time after the field step (s), ``t >= 0``.
    t0 : float
        Characteristic switching time (s).
    n : float
        Avrami exponent.
    """
    t = np.asarray(t, dtype=float)
    return 1.0 - np.exp(-np.power(np.clip(t, 0, None) / t0, n))


def merz_time(E, t_inf, Ea, m=1.0):
    """Merz-law switching time ``t_inf exp[(Ea/|E|)^m]`` (s) for field ``E`` (V/m)."""
    E = np.abs(np.asarray(E, dtype=float))
    return t_inf * np.exp(np.power(Ea / E, m))


def _span(width, kind):
    # half-range of the u-grid in units of decades
    return (100.0 if kind == "lorentzian" else 8.0) * width


def log_time_distribution(u, u1, width, kind="lorentzian"):
    """Distribution ``F(u)`` of ``u = log10(t0)``, normalized on the given grid.

    Returns weights ``F(u) du`` that sum to 1 on the (uniform) grid ``u``.
    """
    u = np.asarray(u, dtype=float)
    x = (u - u1) / width
    if kind == "lorentzian":
        F = 1.0 / (1.0 + x**2)
    elif kind == "lognormal":
        F = np.exp(-0.5 * x**2)
    else:
        raise ValueError("kind must be 'lorentzian' or 'lognormal'")
    return F / F.sum()


def nls_fraction(t, log10_t1, width, n=2.0, kind="lorentzian", n_grid=4001):
    """NLS switched fraction for a distribution of ``log10(t0)``.

    Parameters
    ----------
    t : array_like
        Time (s).
    log10_t1 : float
        Centre of the ``log10(t0)`` distribution.
    width : float
        Width in decades (HWHM for Lorentzian, standard deviation for
        log-normal). ``width = 0`` returns the KAI curve with ``t0 = 10**log10_t1``.
    n : float
        Exponent of each region's KAI kernel.
    kind : {"lorentzian", "lognormal"}
    n_grid : int
        Number of integration points in ``u``.
    """
    t = np.asarray(t, dtype=float)
    if width <= 0:
        return kai_fraction(t, 10.0**log10_t1, n)
    half = _span(width, kind)
    u = np.linspace(log10_t1 - half, log10_t1 + half, n_grid)
    w = log_time_distribution(u, log10_t1, width, kind)
    tt = np.clip(t, 0, None)[..., None]
    kernel = 1.0 - np.exp(-np.power(tt / 10.0**u, n))
    return kernel @ w


def nls_fraction_field(t, E, t_inf, Ea, m=1.0, width=0.5, n=2.0, kind="lorentzian"):
    """NLS switched fraction at constant field ``E`` with a Merz-law centre time.

    ``log10_t1 = log10(merz_time(E, t_inf, Ea, m))``; the width is held fixed.
    """
    u1 = float(np.log10(merz_time(E, t_inf, Ea, m)))
    return nls_fraction(t, u1, width, n=n, kind=kind)


def _check_fraction(frac):
    frac = np.asarray(frac, dtype=float)
    if np.any(~np.isfinite(frac)):
        raise ValueError("switched fraction contains non-finite values")
    return frac


def fit_kai(t, frac, p0=None):
    """Least-squares fit of :func:`kai_fraction` to switched-fraction data.

    The fit parameters are ``log10(t0)`` and ``n`` (fitting the logarithm keeps
    the problem well conditioned over many decades of time).

    Returns
    -------
    dict with ``t0``, ``n``, ``log10_t0`` and ``cov`` (covariance of
    ``(log10_t0, n)``).
    """
    t = np.asarray(t, dtype=float)
    frac = _check_fraction(frac)
    if p0 is None:
        i = int(np.argmin(np.abs(frac - (1 - np.exp(-1)))))
        p0 = (np.log10(t[i]), 2.0)

    def model(tt, lt0, n):
        return kai_fraction(tt, 10.0**lt0, n)

    popt, pcov = curve_fit(model, t, frac, p0=p0, bounds=([-np.inf, 0.1], [np.inf, 10.0]))
    return {"log10_t0": popt[0], "t0": 10.0**popt[0], "n": popt[1], "cov": pcov}


def fit_nls(t, frac, n=2.0, kind="lorentzian", p0=None):
    """Least-squares fit of :func:`nls_fraction` (centre and width; ``n`` fixed).

    Returns
    -------
    dict with ``log10_t1``, ``width`` and ``cov``.
    """
    t = np.asarray(t, dtype=float)
    frac = _check_fraction(frac)
    if p0 is None:
        i = int(np.argmin(np.abs(frac - 0.5)))
        p0 = (np.log10(t[i]), 0.5)

    def model(tt, lt1, w):
        return nls_fraction(tt, lt1, w, n=n, kind=kind, n_grid=2001)

    popt, pcov = curve_fit(model, t, frac, p0=p0, bounds=([-np.inf, 1e-3], [np.inf, 10.0]))
    return {"log10_t1": popt[0], "width": popt[1], "cov": pcov}


def fit_merz(E, t0, p0=None):
    """Fit the Merz law ``ln t0 = ln t_inf + (Ea/E)^m`` to switching times.

    Parameters
    ----------
    E : array_like
        Fields (V/m), positive.
    t0 : array_like
        Characteristic switching times (s).

    Returns
    -------
    dict with ``t_inf`` (s), ``Ea`` (V/m), ``m`` and ``cov`` (covariance of
    ``(ln t_inf, ln Ea, m)``).
    """
    E = np.abs(np.asarray(E, dtype=float))
    y = np.log(np.asarray(t0, dtype=float))
    if p0 is None:
        slope, intercept = np.polyfit(1.0 / E, y, 1)
        p0 = (intercept, np.log(max(slope, 1e-3 * np.min(E))), 1.0)

    def model(EE, ln_tinf, ln_Ea, m):
        return ln_tinf + np.power(np.exp(ln_Ea) / EE, m)

    popt, pcov = curve_fit(model, E, y, p0=p0, bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 10.0]))
    return {"t_inf": float(np.exp(popt[0])), "Ea": float(np.exp(popt[1])), "m": popt[2], "cov": pcov}
