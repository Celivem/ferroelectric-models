r"""Repeated reads of a stored state (read disturb, CMW_Q) and loop coercive voltages.

Companion of :mod:`fe_models.nls_multidomain`; it uses the same equations
(E1-E7) and adds no physics of its own.  It answers two questions that the
literature comparison needs:

* :func:`loop_peak` -- at which voltage does the switching-current peak of a
  triangular P-V loop sit?  :func:`calibrate_to_loop` scales the activation
  field of one direction so that the model's peak matches a measured one.
* :func:`read_sequence` -- the charge read by the ``n``-th of ``N`` identical
  trapezoidal read pulses applied to a stored state, for ``n`` up to 1e11,
  and the memory window ``CMW_Q = |Q(+) - Q(-)|`` between the two stored
  states.

Read pulses
-----------
Each read is a trapezoid 0 -> ``v_read`` (rise ``t_r``), plateau ``t_pw``,
back to 0 (fall ``t_f``), followed by ``t_off`` at 0 V.  The charge of read
``n`` is integrated over the rise and the plateau (as in Mukherjee et al.,
IEDM 2023, Fig. 7):

.. math::  Q_n = P_s(\bar x_{n,\mathrm{pl}} - \bar x_{n,0}) + q_{rev}(V_R, \bar x_{n,\mathrm{pl}}, y)

where :math:`\bar x_{n,0}` is the mean state before the read and
:math:`\bar x_{n,\mathrm{pl}}` the state at the end of the plateau.  Only the
direction that would destroy the stored state is driven (``u`` has the sign
opposite to the stored state during the whole read and the off time), so every
class relaxes toward the opposite state with the accumulated hazard
:math:`\Lambda`, :math:`s = -s_0 + 2 s_0 e^{-\Lambda}`.

Hazard per read (E3, E3b with ``h`` starting from the value left by the
previous read): with :math:`a=\int R\,dt` over the pulse,
:math:`\lambda=(h+a)^\beta-h^\beta`.  Between reads ``h`` relaxes with the
gated time :math:`\tau_h[1+((0-V_g)/0.1\,\mathrm V)^4]` of the model, where the
gate centre ``V_g`` is the imprint voltage (the model's rule) unless
``gate_centre`` is given, and it keeps growing under the internal drive at 0 V.
The first ``k_exact`` reads are summed exactly; beyond that ``h`` follows its
closed-form geometric approach to the periodic steady state and the hazard is
summed with the Euler-Maclaurin rule on a logarithmic grid.  Relaxation of
``h`` during the edges of the pulse is neglected (the gate keeps it closed there
unless ``V_imp`` is within about 0.1 V of 0).

``stored_shift`` (default 0) is an optional, constant bias that stabilizes the
stored state: the imprint voltage seen by a state ``s_0`` becomes
:math:`V_{imp} - s_0\,\Delta`.  It is an analysis knob for "the stored state is
harder to switch than in the loop" and stands in for a time-dependent imprint
model; with ``shift_branches=True`` the reversible branches move with it.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import nls_multidomain as nm

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


# ----------------------------------------------------------------- loops
def loop_branch(p, v_peak, period, s_start=1.0, nd=256, nstep=4000):
    """One half period of a triangular loop from ``s_start * v_peak`` to ``-s_start * v_peak``.

    Starts from the saturated state ``s_start`` with h = 0 and integrates E3
    and E3b with :func:`nm.march` (backward Euler, ``nstep`` equal steps; first
    order, the switched fraction is within about 2e-3 of the Radau solution of
    :func:`nm.simulate` for 4000 steps).  :func:`nm.step` is not used here: when
    the drive reverses while ``|u|`` is still below the reset threshold it
    carries the magnitude of ``h`` over to the new direction, which E3/E3b do
    not.  Returns ``(v, x, qd)``: voltages at the end of each step, the mean
    state there and the terminal charge density P + q_rev [C/m^2].  Requires
    ``vdep = 0`` (as :func:`nm.march`).
    """
    if p["vdep"] != 0:
        raise ValueError("loop_branch needs vdep = 0")
    dom = nm.make_domains(p, nd)
    s = np.full(nd, float(s_start))
    h = np.zeros(nd)
    half = 0.5 * period
    ts = np.linspace(0.0, half, nstep + 1)
    v_edge = s_start * v_peak * (1.0 - 2.0 * ts / half)
    x = np.empty(nstep)
    cache = {}
    for i in range(nstep):
        s, h = nm.march(p, dom, s, h, ts[i:i + 2], v_edge[i:i + 2], theta=1.0, cache=cache)
        cache.pop(float(v_edge[i]), None)
        x[i] = dom.w @ s
    qd = np.array([nm.q_term(p, a, b) for a, b in zip(v_edge[1:], x)])
    return v_edge[1:], x, qd


def loop_peak(p, v_peak, period, direction=-1, nd=256, nstep=4000, total=True, smooth=0.03):
    """Voltage of the current peak of the branch that switches toward ``direction`` (+1 or -1).

    ``total=True`` uses the whole terminal current d(P + q_rev)/dt, as a
    measured loop does; ``total=False`` only the switching part P_s dx/dt.
    Each class adds its own bump to the current, so the current is first
    smoothed with a Gaussian of width ``smooth`` volts (about the class spacing
    near the peak for ``nd`` = 64-256); the maximum is then refined by a
    parabola.  This keeps the peak position a continuous function of the
    parameters, which the calibration needs.
    """
    v, x, qd = loop_branch(p, v_peak, period, s_start=-float(direction), nd=nd, nstep=nstep)
    q = qd if total else p["ps"] * x
    q0 = nm.q_term(p, -direction * v_peak, -float(direction)) if total else -direction * p["ps"]
    j = direction * np.diff(np.concatenate([[q0], q]))
    dv = abs(v[1] - v[0])
    if smooth > 0:
        half = int(np.ceil(4 * smooth / dv))
        ker = np.exp(-0.5 * (np.arange(-half, half + 1) * dv / smooth) ** 2)
        j = np.convolve(j, ker / ker.sum(), mode="same")
    k = int(np.argmax(j))
    if 0 < k < len(j) - 1:
        a, b, c = j[k - 1], j[k], j[k + 1]
        den = a - 2.0 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(v[k] + d * (v[1] - v[0]))
    return float(v[k])


def calibrate_to_loop(p, v_target, v_peak, period, direction=-1, nd=256, nstep=4000, total=False):
    """Scale the Dagum scale of one direction so that :func:`loop_peak` equals ``v_target``.

    ``direction=-1`` scales ``emn`` (the + -> - branch), ``+1`` scales ``emp``.
    Returns ``(card, scale)``.
    """
    key = "emn" if direction < 0 else "emp"

    def f(sc):
        return loop_peak(dict(p, **{key: p[key] * sc}), v_peak, period, direction, nd, nstep, total) - v_target

    # the peak moves monotonically away from V_imp as the scale grows; bracket from scale 1
    sgn = 1.0 if direction > 0 else -1.0
    lo = hi = 1.0
    f1 = f(1.0)
    if sgn * f1 < 0:          # peak not far enough out: increase the scale
        while sgn * f(hi) < 0:
            hi *= 1.25
        lo = hi / 1.25
    else:
        while sgn * f(lo) > 0:
            lo /= 1.25
        hi = lo * 1.25
    sc = brentq(f, lo, hi, xtol=1e-5)
    return dict(p, **{key: p[key] * sc}), float(sc)


# ----------------------------------------------------------------- reads
def _y_end(v_read, t_r, t_pw, tau_r):
    """Low-pass voltage y (E7) at the end of the plateau of a trapezoid, starting from y = 0."""
    if t_r > 0:
        y_r = v_read / t_r * (t_r - tau_r * (1.0 - np.exp(-t_r / tau_r)))
    else:
        y_r = 0.0
    return v_read + (y_r - v_read) * np.exp(-t_pw / tau_r)


def _state_card(p, s0, stored_shift, shift_branches):
    q = dict(p, vimp=p["vimp"] - s0 * stored_shift)
    if shift_branches:
        q["vi"] = p["vi"] - s0 * stored_shift / p["rho"]
    return q


def _hazard_terms(p, dom, s0, v_read, t_r, t_pw, t_f, t_off, gate_centre):
    """Integrated rates of the destroying direction: (a_rise+plateau, a_pulse, R at 0 V, g, h_inf)."""
    temp = p["t0"]

    def r_bad(v):
        rp, rn = nm.rates(p, dom, v, 0.0, temp)
        return rn if s0 > 0 else rp

    tt = 0.5 * (_GL_X + 1.0)
    a_rise = sum(w * 0.5 * t_r * r_bad(v_read * x) for w, x in zip(_GL_W, tt)) if t_r > 0 else 0.0
    a_fall = sum(w * 0.5 * t_f * r_bad(v_read * (1.0 - x)) for w, x in zip(_GL_W, tt)) if t_f > 0 else 0.0
    a_pl = r_bad(v_read) * t_pw
    r0 = r_bad(0.0)
    gc = p["vimp"] if gate_centre is None else gate_centre
    th = p["tau_h"] * (1.0 + ((0.0 - gc) / nm.V_GATE) ** 4) if np.isfinite(p["tau_h"]) else np.inf
    g = float(np.exp(-t_off / th)) if np.isfinite(th) else 1.0
    h_inf = r0 * th if np.isfinite(th) else np.zeros_like(r0)
    return a_rise + a_pl, a_rise + a_pl + a_fall, r0, g, h_inf, th


def _cumulative_hazard(p, dom, s0, v_read, t_r, t_pw, t_f, t_off, ns, gate_centre, k_exact):
    """Hazard accumulated before read n and at the end of its plateau, for every n in ns."""
    beta = p["beta"]
    a_rp, a, r0, g, h_inf, th = _hazard_terms(p, dom, s0, v_read, t_r, t_pw, t_f, t_off, gate_centre)
    nd = len(dom.w)
    if beta == 1.0:
        per = a + r0 * t_off
        return {int(n): ((n - 1) * per, (n - 1) * per + a_rp) for n in ns}

    tt = 0.5 * t_off * (_GL_X + 1.0)
    e_t = np.exp(-tt / th) if np.isfinite(th) else np.ones_like(tt)

    def parts(hs):
        """hs: (m, nd) h at the start of a read -> (lambda to plateau end, lambda of read + off, h next)."""
        he = hs + a
        l_rp = (hs + a_rp) ** beta - hs**beta
        l_read = he**beta - hs**beta
        hh = he[None] * e_t[:, None, None] + h_inf[None, None, :] * (1.0 - e_t[:, None, None])
        l_off = np.tensordot(0.5 * t_off * _GL_W, beta * r0 * hh ** (beta - 1.0), axes=(0, 0))
        return l_rp, l_read + l_off, he * g + h_inf * (1.0 - g)

    ns = np.asarray(sorted({int(n) for n in ns}), dtype=float)
    out = {}
    kmax = int(min(ns.max(), k_exact))
    hs = np.zeros((1, nd))
    cum = np.zeros(nd)
    want = set(int(n) for n in ns if n <= kmax)
    for k in range(1, kmax + 1):
        l_rp, l_tot, h_next = parts(hs)
        if k in want:
            out[k] = (cum.copy(), cum + l_rp[0])
        cum = cum + l_tot[0]
        hs = h_next
    big = ns[ns > kmax]
    if len(big):
        c0 = g * a + h_inf * (1.0 - g)
        h1 = hs[0]  # h at the start of read kmax + 1

        def h_of(k):
            k = np.asarray(k, float)[:, None]
            if g < 1.0:
                hss = c0 / (1.0 - g)
                return hss[None] + (h1 - hss)[None] * g ** (k - kmax - 1)
            return h1[None] + (k - kmax - 1) * (a + r0 * t_off)[None]

        k0 = kmax + 1.0
        grid = np.unique(np.concatenate([np.geomspace(k0, big.max(), 800), big, [k0]]))
        lt = parts(h_of(grid))[1]
        integ = np.vstack([np.zeros(nd), np.cumsum(0.5 * (lt[1:] + lt[:-1]) * np.diff(grid)[:, None], axis=0)])
        for n in big:
            j = int(np.searchsorted(grid, n))
            # sum_{k=k0}^{n-1} L(k) ~ int_{k0}^{n} L dk + [L(k0) - L(n)]/2  (Euler-Maclaurin)
            before = cum + integ[j] + 0.5 * (lt[0] - lt[j])
            l_rp, _, _ = parts(h_of([n]))
            out[int(n)] = (before, before + l_rp[0])
    return out


def read_sequence(p, v_read, ns, t_r=5e-7, t_pw=5e-7, t_f=5e-7, t_off=5e-7, nd=256, gate_centre=None,
                  stored_shift=0.0, shift_branches=False, k_exact=300):
    """Charge of read ``n`` (``n`` in ``ns``) of a train of identical reads on both stored states.

    Returns a dict of arrays over ``ns``: ``q_plus``, ``q_minus`` (charge of the
    read on the state stored as +1 / -1, C, signed), ``cmw`` = |q_plus - q_minus|
    [C], ``q_sw_plus`` / ``q_sw_minus`` (switching part of that read, C) and
    ``f_plus`` / ``f_minus`` (fraction of the film switched away from the stored
    state after the read, 0 = intact).  ``p["area"]`` sets the area.  Rise and
    plateau charge only; see the module docstring for the rules.
    """
    ns = np.atleast_1d(np.asarray(ns, dtype=float))
    if np.any(ns < 1) or np.any(ns != np.round(ns)):
        raise ValueError("read numbers must be integers >= 1")
    res = {}
    for s0, tag in ((1.0, "plus"), (-1.0, "minus")):
        q = _state_card(p, s0, stored_shift, shift_branches)
        dom = nm.make_domains(q, nd)
        cum = _cumulative_hazard(q, dom, s0, v_read, t_r, t_pw, t_f, t_off, ns, gate_centre, k_exact)
        y = _y_end(v_read, t_r, t_pw, q["tau_r"])
        qq, qs, ff = [], [], []
        for n in ns:
            c_b, c_p = cum[int(n)]
            x_b = s0 * float(dom.w @ (2.0 * np.exp(-c_b) - 1.0))
            x_p = s0 * float(dom.w @ (2.0 * np.exp(-c_p) - 1.0))
            q_sw = nm.ps_of_t(q, q["t0"]) * (x_p - x_b)
            qq.append((q_sw + nm.q_rev(q, v_read, x_p, y)) * q["area"])
            qs.append(q_sw * q["area"])
            ff.append(0.5 * (1.0 - s0 * x_p))
        res[f"q_{tag}"] = np.array(qq)
        res[f"q_sw_{tag}"] = np.array(qs)
        res[f"f_{tag}"] = np.array(ff)
    res["n"] = ns
    res["cmw"] = np.abs(res["q_plus"] - res["q_minus"])
    return res
