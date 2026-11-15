"""Tests for the round-2 imprint terms (E17a branch asymmetry, E17b field acceleration) and the read
sequence with a dynamic stored-state imprint (fe_models.read_disturb.read_sequence_dynamic).

Both terms are off in the generic card; each test below is built so that it fails if the term is dropped,
mis-signed or applied to the wrong branch.
"""

import re
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from fe_models import nls_multidomain as nm
from fe_models import read_disturb as rd

ROOT = Path(__file__).resolve().parents[1]
T9 = 9e-9


def half_field(p, dom, direction, t_on=1e-6):
    """Voltage at which one rectangular pulse of t_on switches half of the film in ``direction`` (+1 / -1)."""
    def f(v):
        rp, rn = nm.rates(p, dom, v, 0.0, p["t0"])
        r = rp if direction > 0 else rn
        return float(dom.w @ (1.0 - np.exp(-(r * t_on) ** p["beta"]))) - 0.5

    return brentq(f, 0.01, 8.0) if direction > 0 else brentq(f, -8.0, -0.01)


def card(**kw):
    base = dict(tfe=T9, ps=0.165, vimp=0.405, vi=0.78, vs=1.62, vw=2.0, area=4.8e-9, emn=1.4e8, emp=1.4e8)
    base.update(kw)
    return nm.generic_card(**base)


class TestDefaults(unittest.TestCase):
    def test_terms_are_off_in_the_generic_card(self):
        p = nm.generic_card()
        self.assertEqual(p["imp_kw"], 0.0)
        self.assertEqual(p["imp_va"], 0.0)
        self.assertEqual(float(nm.imprint_widening(dict(p, imp_v=0.5), 0.7)), 0.0)
        np.testing.assert_array_equal(nm.imprint_accel(p, np.array([-3.0, 0.0, 2.0])), 1.0)
        pe, _, _ = nm.aged_card(dict(p, imp_v=0.5), s_i=0.7)
        self.assertNotIn("_imp_w", pe)

    def test_imprint_without_new_terms_is_unchanged(self):
        """imp_kw = imp_va = 0 written explicitly gives bit-identical results to the card without them."""
        p = nm.generic_card(imp_v=0.3, imp_t1=1e-7, imp_t2=1e-2)
        q = dict(p)
        q.pop("imp_kw"), q.pop("imp_va")
        vf, br = nm.pwl([0, 1e-7, 2e-6, 2.1e-6, 30e-6, 30.1e-6, 35e-6], [0, 3, 3, 0, 0, -1.6, -1.6])
        a = nm.simulate(p, vf, 35e-6, s0=-1.0, breaks=br)
        b = nm.simulate(q, vf, 35e-6, s0=-1.0, breaks=br)
        np.testing.assert_array_equal(a["qd"], b["qd"])

    def test_verilog_a_uses_the_same_smoothing(self):
        va = (ROOT / "spice" / "fecap_nls.va").read_text(encoding="utf-8")
        self.assertIn("sqrt(sI * sI + 1.0e-6) - 1.0e-3", va)
        self.assertAlmostEqual(nm.I_EPS, 1e-3)
        self.assertIn("cosh(min(abs(rho * v) / imp_va, 80.0))", va)
        self.assertTrue(re.search(r"parameter real imp_kw = 0\.0;", va))


class TestBranchAsymmetry(unittest.TestCase):
    """E17a: the switching field out of the imprinted state moves (1 + kappa) V_I |sigma|, the other (1 - kappa)."""

    def test_escape_branch_moves_more(self):
        for kw in (0.0, 0.2, 0.5):
            p = nm.generic_card(imp_v=0.6, imp_kw=kw)
            p0, dom0, _ = nm.aged_card(p, s_i=0.0)
            for s in (0.8, -0.8):
                pe, dom, _ = nm.aged_card(p, s_i=s)
                dn = half_field(pe, dom, -1) - half_field(p0, dom0, -1)
                up = half_field(pe, dom, +1) - half_field(p0, dom0, +1)
                w = 0.6 * kw * (np.sqrt(s * s + 1e-6) - 1e-3)
                # centre shift -V_I s plus the outward move w of both fields
                self.assertAlmostEqual(dn, -0.6 * s - w, places=6)
                self.assertAlmostEqual(up, -0.6 * s + w, places=6)
                esc, ret = (abs(dn), abs(up)) if s > 0 else (abs(up), abs(dn))
                if kw > 0:
                    self.assertGreater(esc, ret)
                    self.assertAlmostEqual((esc - ret) / (esc + ret), kw, delta=2e-3)

    def test_ode_rates_match_the_aged_card(self):
        p = nm.generic_card(imp_v=0.5, imp_kw=0.3, imp_t1=1e-6, imp_t2=1e-3)
        dom = nm.make_domains(p)
        n = len(dom.w)
        taus = nm.imprint_taus(p)
        sj = np.full(len(taus), 0.6)
        z = np.concatenate([np.full(n, 1.0) * 0.2, np.full(n, 1e-3), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], sj])
        for v in (-1.2, 1.4):
            ds = nm.rhs_aged(p, dom, z, v, p["t0"], taus, nm.split_signs(n))[0]
            pe, dome, _ = nm.aged_card(p, s_i=0.6)
            rp, rn = nm.rates(pe, dome, v, 0.0, p["t0"])
            hp = p["beta"] * rp * nm.h_pow(np.maximum(z[n:2 * n], 0), p["beta"])
            hn = p["beta"] * rn * nm.h_pow(np.maximum(-z[n:2 * n], 0), p["beta"])
            np.testing.assert_allclose(ds, hp * (1 - z[:n]) - hn * (1 + z[:n]), rtol=1e-12, atol=1e-300)

    def test_branch_peaks_move_outward(self):
        p = nm.generic_card(imp_v=0.5, imp_kw=0.4, vw=0.5, vs=1.0)
        v = np.linspace(-3, 3, 6001)

        def peaks(s_i):
            up = nm.term2_aged(p, v, 1.0, v, p["t0"], s_i)
            dn = nm.term2_aged(p, v, -1.0, v, p["t0"], s_i)
            return v[np.argmax(np.gradient(up, v))], v[np.argmax(np.gradient(dn, v))]

        u0, d0 = peaks(0.0)
        u1, d1 = peaks(0.8)
        w = 0.4 * 0.5 * (np.sqrt(0.64 + 1e-6) - 1e-3)
        self.assertAlmostEqual(u1 - u0, -0.5 * 0.8 - w, delta=2e-3)   # escape-side peak of the + state
        self.assertAlmostEqual(d1 - d0, -0.5 * 0.8 + w, delta=2e-3)


class TestFieldAcceleration(unittest.TestCase):
    """E17b: every imprint rate carries cosh(rho V / V_a)."""

    def test_held_state_imprints_faster_under_bias(self):
        p = nm.generic_card(imp_v=0.2, imp_t1=1e-5, imp_t2=1e-1, imp_va=0.5)
        t_end = 2e-4
        for v in (0.0, 0.6):
            vf = lambda t, v=v: v  # noqa: E731
            r = nm.simulate(p, vf, t_end, s0=1.0, t_eval=[t_end], rtol=1e-9, atol=1e-12)
            want = float(np.mean(1.0 - np.exp(-t_end * np.cosh(v / 0.5) / nm.imprint_taus(p))))
            self.assertAlmostEqual(float(r["s_i"][-1]), want, delta=2e-4)

    def test_cycle_history_uses_the_accelerated_time(self):
        p = nm.generic_card(imp_v=0.2, imp_t1=1e-5, imp_t2=1e-2, imp_va=0.4)
        t = np.array([0.0, 2.5e-6, 7.5e-6, 10e-6])
        v = np.array([0.0, 0.5, -0.5, 0.0])          # well below the switching fields: x_t stays at -1
        out = nm.cycle_history(p, t, v, [50], history=nm.History(si=0.0))
        dts, vms, _ = nm._refine_period(t, v, 200)
        t_eff = 50 * float(np.sum(dts * np.cosh(np.abs(vms) / 0.4)))
        want = -float(np.mean(1.0 - np.exp(-t_eff / nm.imprint_taus(p))))
        self.assertAlmostEqual(float(out["s_i"][0]), want, delta=1e-3)
        plain = nm.cycle_history(dict(p, imp_va=0.0), t, v, [50], history=nm.History(si=0.0))
        self.assertGreater(abs(float(out["s_i"][0])) - abs(float(plain["s_i"][0])), 0.02)


class TestDynamicReads(unittest.TestCase):
    def test_reduces_to_read_sequence_without_imprint(self):
        p = card()
        ns = [1, 5, 300, 1e5, 1e8]
        a = rd.read_sequence(p, -0.6, ns)
        b = rd.read_sequence_dynamic(p, -0.6, ns, t_delay=1e-3)
        np.testing.assert_allclose(b["q_plus"], a["q_plus"], rtol=1e-6)
        np.testing.assert_allclose(b["q_minus"], a["q_minus"], rtol=1e-6)

    def test_saturated_imprint_equals_a_constant_shift(self):
        p = card()
        q = dict(p, imp_v=0.3, imp_t1=1e-12, imp_t2=1e-11)
        ns = [1, 12, 1e3, 1e7]
        a = rd.read_sequence(p, -0.6, ns, stored_shift=0.3)
        b = rd.read_sequence_dynamic(q, -0.6, ns, t_delay=1e-3, sigma_cv=1.0)
        np.testing.assert_allclose(b["q_plus"], a["q_plus"], rtol=1e-6)
        np.testing.assert_allclose(b["s_i_plus"], 1.0)

    def test_matches_step_integration_with_a_growing_imprint(self):
        """First 6 reads against nm.step with the card updated read by read (imprint grows between reads)."""
        p = dict(card(), imp_v=0.4, imp_t1=1e-7, imp_t2=1e2, imp_kw=0.2)
        t_delay, m = 2e-6, 200
        r = rd.read_sequence_dynamic(p, -0.65, range(1, 7), t_delay, k_exact=10)
        dom = nm.make_domains(p, 256)
        s, h = np.ones(256), np.zeros(256)
        T = 2e-6
        out = []
        for k in range(6):
            si = float(rd.stored_imprint(p, 1.0, t_delay + k * T, 1e-6, 2.7))
            q = rd.imprint_card(p, si, 0.0, 1.0)
            xb = dom.w @ s
            for i in range(m):
                s, h = nm.step(q, dom, s, h, -0.65 * (i + 0.5) / m, 5e-7 / m)
            s, h = nm.step(q, dom, s, h, -0.65, 5e-7)
            xp = dom.w @ s
            out.append((q["ps"] * (xp - xb) + nm.q_rev(q, -0.65, xp)) * q["area"])
            for i in range(m):
                s, h = nm.step(q, dom, s, h, -0.65 * (1 - (i + 0.5) / m), 5e-7 / m)
            s, h = nm.step(q, dom, s, h, 0.0, 5e-7)
        np.testing.assert_allclose(r["q_plus"], out, rtol=3e-3)
        self.assertGreater(r["s_i_plus"][-1], r["s_i_plus"][0])

    def test_growing_imprint_slows_the_disturb(self):
        p = dict(card(), imp_v=0.3, imp_t1=1e-6, imp_t2=1e6)
        ns = [1, 100, 1e4, 1e6, 1e8]
        grow = rd.read_sequence_dynamic(p, -0.65, ns, t_delay=1e-4)
        si1 = float(grow["s_i_plus"][0])
        frozen = rd.read_sequence(p, -0.65, ns, stored_shift=0.3 * si1)
        self.assertTrue(np.all(np.diff(grow["s_i_plus"]) > 0))
        self.assertLess(grow["f_plus"][-1], frozen["f_plus"][-1])
        self.assertAlmostEqual(grow["f_plus"][0], frozen["f_plus"][0], delta=1e-9)

    def test_stored_imprint_log_slope(self):
        p = nm.generic_card(imp_v=0.2, imp_t1=1e-6, imp_t2=1e6)
        t = np.array([1e-2, 1e0])
        s = rd.stored_imprint(p, 1.0, t, t_write=0.0, v_write=0.0)
        self.assertAlmostEqual((s[1] - s[0]) / np.log(100.0), 1.0 / np.log(1e12), delta=0.12 / np.log(1e12))
        s_m = rd.stored_imprint(p, -1.0, t, t_write=0.0, v_write=0.0)
        np.testing.assert_allclose(s_m, -s)


if __name__ == "__main__":
    unittest.main()
