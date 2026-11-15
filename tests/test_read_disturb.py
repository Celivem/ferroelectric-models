"""Tests for fe_models.read_disturb (repeated reads of a stored state, loop coercive voltages)."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp

from fe_models import nls_multidomain as nm
from fe_models import read_disturb as rd

T9 = 9e-9


def card(**kw):
    """A 9 nm card with a large imprint, like the device of the read-endurance comparison."""
    base = dict(tfe=T9, ps=0.165, vimp=0.405, vi=0.78, vs=1.62, vw=2.0, area=4.8e-9, emn=1.4e8, emp=1.4e8)
    base.update(kw)
    return nm.generic_card(**base)


def brute_force(p, v_read, n_reads, nd, t_r=5e-7, t_pw=5e-7, t_f=5e-7, t_off=5e-7, m=400):
    """Charge of each read on the + state with nm.step on a fine grid (reference for read_sequence)."""
    dom = nm.make_domains(p, nd)
    s, h = np.ones(nd), np.zeros(nd)
    out = []
    for _ in range(n_reads):
        xb = dom.w @ s
        for i in range(m):
            s, h = nm.step(p, dom, s, h, v_read * (i + 0.5) / m, t_r / m)
        s, h = nm.step(p, dom, s, h, v_read, t_pw)
        xp = dom.w @ s
        out.append((p["ps"] * (xp - xb) + nm.q_rev(p, v_read, xp)) * p["area"])
        for i in range(m):
            s, h = nm.step(p, dom, s, h, v_read * (1 - (i + 0.5) / m), t_f / m)
        s, h = nm.step(p, dom, s, h, 0.0, t_off)
    return np.array(out)


class TestReadSequence(unittest.TestCase):
    def test_matches_step_integration(self):
        p = card()
        ref = brute_force(p, -0.5, 8, nd=32)
        got = rd.read_sequence(p, -0.5, range(1, 9), nd=32)["q_plus"]
        np.testing.assert_allclose(got, ref, rtol=3e-3)
        # the case is not trivial: reads switch charge and h carries over (read 2 > read 1)
        self.assertGreater(abs(ref[0] - nm.q_rev(p, -0.5, 1.0) * p["area"]), 1e-12)
        self.assertGreater(abs(ref[1]), abs(ref[0]))

    def test_closed_form_continues_the_exact_sum(self):
        p = card()
        ns = [100, 250, 400, 3000, 1e5]
        a = rd.read_sequence(p, -0.4, ns, nd=64, k_exact=60)
        b = rd.read_sequence(p, -0.4, ns, nd=64, k_exact=3000)
        np.testing.assert_allclose(a["f_plus"][:4], b["f_plus"][:4], rtol=2e-3)
        self.assertTrue(np.all(np.diff(a["f_plus"]) >= 0))

    def test_without_switching_cmw_is_the_reversible_difference(self):
        p = card(mul_ea=6.0)
        r = rd.read_sequence(p, -0.3, [1, 10, 1e9], nd=32)
        expect = abs(nm.q_rev(p, -0.3, 1.0) - nm.q_rev(p, -0.3, -1.0)) * p["area"]
        np.testing.assert_allclose(r["cmw"], expect, rtol=1e-12)
        self.assertEqual(float(np.max(r["f_plus"])), 0.0)

    def test_negative_reads_do_not_disturb_the_minus_state(self):
        r = rd.read_sequence(card(), -0.5, [1, 1e6], nd=32)
        self.assertEqual(float(np.max(r["f_minus"])), 0.0)
        self.assertGreater(float(r["f_plus"][-1]), 0.1)

    def test_mirror_symmetry(self):
        p = card(vimp=0.0, vi=0.0)
        a = rd.read_sequence(p, -0.9, [1, 50, 1e5], nd=32)
        b = rd.read_sequence(p, +0.9, [1, 50, 1e5], nd=32)
        np.testing.assert_allclose(a["f_plus"], b["f_minus"], rtol=1e-12)
        self.assertGreater(float(a["f_plus"][-1]), 1e-3)

    def test_stored_shift_stabilizes_the_stored_state(self):
        f = [float(rd.read_sequence(card(), -0.5, [1000], nd=64, stored_shift=d)["f_plus"][0])
             for d in (0.0, 0.1, 0.2, 0.3)]
        self.assertTrue(all(a > b for a, b in zip(f, f[1:])), f)
        # shifting the branches as well changes only the reversible part
        a = rd.read_sequence(card(mul_ea=6.0), -0.5, [1], nd=32, stored_shift=0.2)
        b = rd.read_sequence(card(mul_ea=6.0), -0.5, [1], nd=32, stored_shift=0.2, shift_branches=True)
        self.assertNotAlmostEqual(float(a["cmw"][0]), float(b["cmw"][0]), places=15)

    def test_gate_rule_matters_only_after_the_first_read(self):
        p = card()
        a = rd.read_sequence(p, -0.4, [1, 1000], nd=64, t_off=2e-5)
        b = rd.read_sequence(p, -0.4, [1, 1000], nd=64, t_off=2e-5, gate_centre=0.0)
        self.assertAlmostEqual(float(a["f_plus"][0]), float(b["f_plus"][0]), places=14)
        # with the gate at V_imp, h does not relax at 0 V, so more is switched after many reads
        self.assertGreater(float(a["f_plus"][1]), 1.5 * float(b["f_plus"][1]))

    def test_beta_one_is_linear_accumulation(self):
        p = card(beta=1.0)
        n = np.array([1, 7, 400, 5e4])
        r = rd.read_sequence(p, -0.5, n, nd=16, k_exact=100)
        dom = nm.make_domains(p, 16)
        a_rp, a, r0, *_ = rd._hazard_terms(p, dom, 1.0, -0.5, 5e-7, 5e-7, 5e-7, 5e-7, None)
        lam = (n[:, None] - 1) * (a + r0 * 5e-7) + a_rp
        np.testing.assert_allclose(r["f_plus"], (1 - np.exp(-lam)) @ dom.w, rtol=1e-9)

    def test_low_pass_voltage_at_plateau_end(self):
        tau = 3e-7
        sol = solve_ivp(lambda t, y: [(-0.5 * min(t / 5e-7, 1.0) - y[0]) / tau], (0, 1e-6), [0.0],
                        rtol=1e-10, atol=1e-13, max_step=1e-9)
        self.assertAlmostEqual(rd._y_end(-0.5, 5e-7, 5e-7, tau), sol.y[0, -1], places=7)

    def test_rejects_non_integer_read_numbers(self):
        with self.assertRaises(ValueError):
            rd.read_sequence(card(), -0.3, [0.5])


class TestLoopPeak(unittest.TestCase):
    def test_calibration_hits_the_target(self):
        p = card()
        q, sc = rd.calibrate_to_loop(p, -0.54, 2.7, 50e-6, -1, nd=64)
        self.assertAlmostEqual(rd.loop_peak(q, 2.7, 50e-6, -1, nd=64, total=False), -0.54, delta=1e-3)
        self.assertNotAlmostEqual(sc, 1.0, places=2)
        q, sc = rd.calibrate_to_loop(p, 1.35, 2.7, 50e-6, +1, nd=64)
        self.assertAlmostEqual(rd.loop_peak(q, 2.7, 50e-6, +1, nd=64, total=False), 1.35, delta=1e-3)

    def test_peak_moves_out_with_the_activation_field(self):
        p = card()
        v = [rd.loop_peak(dict(p, emn=p["emn"] * f), 2.7, 50e-6, -1, nd=64) for f in (0.8, 1.0, 1.2)]
        self.assertTrue(v[0] > v[1] > v[2], v)

    def test_peak_is_centred_on_the_imprint_for_a_symmetric_card(self):
        p = card(vimp=0.3, vi=0.3)
        vp = rd.loop_peak(p, 2.7, 50e-6, +1, nd=64)
        vn = rd.loop_peak(p, 2.7, 50e-6, -1, nd=64)
        self.assertAlmostEqual(0.5 * (vp + vn), 0.3, delta=5e-3)


if __name__ == "__main__":
    unittest.main()
