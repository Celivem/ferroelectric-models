r"""Classical scalar Preisach model and FORC analysis.

Model
-----
The film is represented by a set of bistable "hysterons". A hysteron with
up-switching field ``a`` and down-switching field ``b`` (``a > b``) has state
``s = +1`` once ``E >= a``, ``s = -1`` once ``E <= b``, and keeps its state
in between. The polarization is

.. math::

    P(t) = P_s \sum_i w_i\, s_i(t), \qquad \sum_i w_i = 1 ,

so the loop saturates at ``+/-Ps``. It is convenient to describe the weight
distribution in the rotated coordinates

* hysteron coercive field ``Ec_h = (a - b)/2``,
* hysteron bias (interaction) field ``Eb_h = (a + b)/2``,

and to take a product distribution ``w ~ f_c(Ec_h) f_b(Eb_h)``. Gaussian and
Lorentzian shapes are provided; ``bias`` shifts the whole distribution along
the field axis and produces an imprinted loop.

The model is rate independent and history dependent; it obeys the wiping-out
(return-point memory) and congruency properties characteristic of classical
Preisach models.

Implementation: the Preisach half-plane ``a > b`` inside
``[-E_max, E_max]^2`` is discretized on an ``n_grid x n_grid`` mesh of cells
of size ``d``. To avoid a staircase P(E) (and spiky currents), each cell is
not a single hysteron but a short line of hysterons ``(a + x d, b + x d)``
with ``x`` uniform in ``[-1/2, 1/2]``. The cell's mean state ``s`` then obeys
the exact update

``s <- max(s, clip(2(E - a)/d, -1, 1))`` followed by
``s <- min(s, clip(2(E - b)/d, -1, 1))``,

so the model is still a classical Preisach model (with a continuous
distribution) and P(E) is piecewise linear. On the field values
``field_edges`` (the cell boundaries) every cell is fully up or down.

FORC
----
A first-order reversal curve starts from positive saturation, decreases the
field to a reversal field ``Er`` and then measures ``P(Er, E)`` while
increasing ``E`` back to saturation. The FORC distribution

.. math::

    \rho(E_r, E) = -\frac{1}{2}\,\frac{\partial^2 P}{\partial E_r\,\partial E}

equals ``Ps`` times the Preisach density at ``(a, b) = (E, Er)`` for this
model, so a FORC map recovers the hysteron distribution.

References
----------
* F. Preisach, Z. Phys. 94, 277 (1935).
* I. D. Mayergoyz, *Mathematical Models of Hysteresis* (Springer, 1991).
* C. R. Pike, A. P. Roberts and K. L. Verosub, J. Appl. Phys. 85, 6660 (1999)
  (FORC diagrams).

Default parameters are illustrative HfO2-like values (Ps = 20 uC/cm^2,
Ec = 1 MV/cm), not fitted to a device.
"""

from __future__ import annotations

import copy

import numpy as np

from .units import HFO2_ILLUSTRATIVE

__all__ = ["PreisachModel", "simulate_forcs", "forc_distribution"]


def _shape(x, center, width, kind):
    u = (x - center) / width
    if kind == "gaussian":
        return np.exp(-0.5 * u**2)
    if kind == "lorentzian":
        return 1.0 / (1.0 + u**2)
    raise ValueError("distribution must be 'gaussian' or 'lorentzian'")


class PreisachModel:
    """Discretized scalar Preisach model.

    Parameters
    ----------
    Ps : float
        Saturation polarization of the switchable part (C/m^2).
    Ec : float
        Centre of the hysteron coercive-field distribution (V/m).
    sigma_c : float
        Width of the coercive-field distribution (V/m): standard deviation
        for ``"gaussian"``, half width at half maximum for ``"lorentzian"``.
    sigma_b : float
        Width of the bias (interaction) field distribution (V/m).
    bias : float
        Centre of the bias distribution (V/m); nonzero gives imprint.
    E_max : float
        Half-size of the discretized Preisach plane (V/m). Fields beyond
        ``+/-E_max`` saturate the model.
    n_grid : int
        Number of cells per axis.
    distribution : {"gaussian", "lorentzian"}
        Shape used for both ``f_c`` and ``f_b``.
    state : {-1, +1}
        Initial saturated state.
    """

    def __init__(self, Ps=HFO2_ILLUSTRATIVE["Pr"], Ec=HFO2_ILLUSTRATIVE["Ec"], sigma_c=0.25e8, sigma_b=0.15e8,
                 bias=0.0, E_max=3e8, n_grid=100, distribution="gaussian", state=-1):
        self.Ps = float(Ps)
        self.E_max = float(E_max)
        self.n_grid = int(n_grid)
        self.field_edges = np.linspace(-E_max, E_max, n_grid + 1)
        self.cell = self.field_edges[1] - self.field_edges[0]
        centers = 0.5 * (self.field_edges[:-1] + self.field_edges[1:])
        A, B = np.meshgrid(centers, centers, indexing="ij")
        self._mask = A > B
        self.alpha = A[self._mask]
        self.beta = B[self._mask]
        ec_h = 0.5 * (self.alpha - self.beta)
        eb_h = 0.5 * (self.alpha + self.beta)
        w = _shape(ec_h, Ec, sigma_c, distribution) * _shape(eb_h, bias, sigma_b, distribution)
        self.weights = w / w.sum()
        self.reset(state)

    # -- state handling -----------------------------------------------------
    def reset(self, state=-1):
        """Put every hysteron in ``state`` (+1 or -1): positive or negative saturation."""
        if state not in (-1, 1):
            raise ValueError("state must be +1 or -1")
        self.s = np.full(self.alpha.size, float(state))
        return self

    def copy(self):
        """Independent copy, including the current hysteron states."""
        return copy.deepcopy(self)

    @property
    def polarization(self):
        """Current polarization (C/m^2)."""
        return self.Ps * float(self.weights @ self.s)

    # -- driving ------------------------------------------------------------
    def apply(self, E):
        """Drive the model through the field sequence ``E`` (V/m).

        The hysteron states are updated in place, so successive calls continue
        the same history. Returns the polarization after each field value
        (array of the same shape as ``E``; float for scalar input).
        """
        E_arr = np.atleast_1d(np.asarray(E, dtype=float))
        out = np.empty(E_arr.shape)
        s = self.s
        k_up = 2.0 / self.cell
        for k, e in enumerate(E_arr.ravel()):
            np.maximum(s, np.clip(k_up * (e - self.alpha), -1.0, 1.0), out=s)
            np.minimum(s, np.clip(k_up * (e - self.beta), -1.0, 1.0), out=s)
            out.flat[k] = self.Ps * float(self.weights @ s)
        return float(out[0]) if np.ndim(E) == 0 else out

    def major_loop(self, n=None):
        """Ascending branch from negative saturation and descending branch from positive saturation.

        Returns ``(E, P_ascending, P_descending)`` on the field grid
        ``field_edges`` (or ``n`` evenly spaced points). The model's own state
        is left unchanged.
        """
        E = self.field_edges if n is None else np.linspace(-self.E_max, self.E_max, n)
        up = self.copy().reset(-1).apply(E)
        dn = self.copy().reset(1).apply(E[::-1])[::-1]
        return E, up, dn

    def distribution_grid(self):
        """Hysteron weights on the ``(alpha, beta)`` mesh, NaN where ``alpha <= beta``.

        Returns ``(centers, W)`` with ``W[i, j]`` the weight of the cell at
        ``alpha = centers[i]``, ``beta = centers[j]``.
        """
        centers = 0.5 * (self.field_edges[:-1] + self.field_edges[1:])
        W = np.full((self.n_grid, self.n_grid), np.nan)
        W[self._mask] = self.weights
        return centers, W


def simulate_forcs(model, E_reversal=None, E_field=None):
    """Simulate first-order reversal curves.

    Parameters
    ----------
    model : PreisachModel
        Template model; it is copied, its state is not modified.
    E_reversal : array_like, optional
        Reversal fields (V/m). Default: ``model.field_edges``.
    E_field : array_like, optional
        Field grid of the increasing sweeps (V/m). Default: same as
        ``E_reversal``.

    Returns
    -------
    Er, E : ndarray
        Reversal and sweep field grids.
    P : ndarray, shape (len(Er), len(E))
        ``P[i, j] = P(Er[i], E[j])``; NaN where ``E[j] < Er[i]`` (not measured).
    """
    Er = model.field_edges if E_reversal is None else np.asarray(E_reversal, dtype=float)
    E = Er if E_field is None else np.asarray(E_field, dtype=float)
    P = np.full((Er.size, E.size), np.nan)
    for i, er in enumerate(Er):
        m = model.copy().reset(1)
        m.apply(er)
        j = E >= er
        P[i, j] = m.apply(E[j])
    return Er, E, P


def forc_distribution(Er, E, P):
    """FORC distribution ``rho = -1/2 d^2P/(dEr dE)`` (C/m^2 per (V/m)^2).

    Each FORC is extended to ``E < Er`` with its starting value (the usual
    convention, which makes the derivative zero there), the mixed derivative
    is taken with second-order central differences, and the unmeasured region
    ``E < Er`` is set to NaN in the output.

    Returns
    -------
    rho : ndarray, shape (len(Er), len(E))
    """
    Er = np.asarray(Er, dtype=float)
    E = np.asarray(E, dtype=float)
    P = np.array(P, dtype=float)
    filled = P.copy()
    for i in range(P.shape[0]):
        row = filled[i]
        ok = ~np.isnan(row)
        if ok.any():
            row[~ok] = row[ok][0]
    d1 = np.gradient(filled, E, axis=1)
    rho = -0.5 * np.gradient(d1, Er, axis=0)
    rho[np.isnan(P)] = np.nan
    return rho
