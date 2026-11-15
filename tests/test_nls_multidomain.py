"""Tests for fe_models.nls_multidomain (the Python reference of spice/fecap_nls.va)."""

import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.integrate import quad, solve_ivp

from fe_models import nls_multidomain as nm

ROOT = Path(__file__).resolve().parents[1]
UC = 1e-2


def card(**kw):
    return nm.generic_card(**kw)


def pulse(v, width, edge=1e-9, pre=1e-9, post=1e-6):
    t = [0.0, pre, pre + edge, pre + edge + width, pre + 2 * edge + width, pre + 2 * edge + width + post]
    return nm.pwl(t, [0.0, 0.0, v, v, 0.0, 0.0]), t[-1]


def triangle(amps, quarter, start=0.0):
    t, v = [0.0], [start]
    vmax = max(abs(a) for a in amps)
    for a in amps:
        t.append(t[-1] + abs(a - v[-1]) / vmax * quarter)
        v.append(a)
    return nm.pwl(t, v), t[-1]


def cycle_work(p, v0, s0):
    """Electrical work per area of one 0 -> v0 -> 0 square cycle with switching frozen [J/m^2]."""
    (vf, br), t_end = pulse(v0, 2e-6, edge=1e-9, post=2e-6)
    r = nm.simulate(p, vf, t_end, s0=s0, breaks=br, rtol=1e-9)
    return float(np.trapezoid(r["v"], r["qd"]))


def small_signal_drift(p, amp=0.3, cycles=20):
    (vf, br), t_end = triangle([1.6, 0.0], 2e-6)
    s_part = nm.simulate(p, vf, t_end, s0=-1.0, breaks=br, rtol=1e-7)["s"][:, -1]
    (vf, br), t_end = triangle([amp, -amp] * cycles + [0.0], 0.25e-6)
    r = nm.simulate(p, vf, t_end, s0=s_part, breaks=br, rtol=1e-8)
    return abs(float(r["x"][-1] - r["x"][0])), float(r["x"][0])


class TestGridAndDistribution(unittest.TestCase):
    def test_weights_and_ordering(self):
        p = card()
        for nd in (4, 16, 64):
            dom = nm.make_domains(p, nd)
            self.assertAlmostEqual(dom.w.sum(), 1.0, places=12)
            self.assertTrue(np.all(dom.w > 0))
            self.assertTrue(np.all(np.diff(dom.va_p) > 0))

    def test_dagum_quantile_inverts_cdf(self):
        q = np.array([1e-4, 0.01, 0.3, 0.5, 0.9, 0.9999])
        E = nm.dagum_quantile(q, 1.775e8, 8.68, 1.136)
        np.testing.assert_allclose(nm.dagum_cdf(E, 1.775e8, 8.68, 1.136), q, rtol=1e-10)

    def test_dagum_is_gb2_with_q_equal_one(self):
        E = np.linspace(0.5e8, 4e8, 30)
        np.testing.assert_allclose(nm.gb2_cdf(E, 8.68, 1.775e8, 1.136, 1.0), nm.dagum_cdf(E, 1.775e8, 8.68, 1.136),
                                   rtol=1e-10)

    def test_activation_voltage_scales_with_thickness(self):
        d1, d2 = nm.make_domains(card(tfe=1e-8)), nm.make_domains(card(tfe=2e-8))
        np.testing.assert_allclose(d2.va_p, 2 * d1.va_p)


class TestRatesAndLimits(unittest.TestCase):
    def test_rate_is_one_directional(self):
        p = card()
        dom = nm.make_domains(p)
        rp, rn = nm.rates(p, dom, 2.0, 0.0, p["t0"])
        self.assertTrue(np.all(rp > 0) and np.all(rn == 0))
        rp, rn = nm.rates(p, dom, -2.0, 0.0, p["t0"])
        self.assertTrue(np.all(rn > 0) and np.all(rp == 0))
        rp, rn = nm.rates(p, dom, p["vimp"], 0.0, p["t0"])
        self.assertTrue(np.all(rp == 0) and np.all(rn == 0))

    def test_saturated_states_are_fixed_points(self):
        p = card()
        dom = nm.make_domains(p)
        n = len(dom.w)
        ds, dh, _, _ = nm.rhs(p, dom, np.ones(n), np.full(n, 2.0), 3.0, 0.0, 3.0, p["t0"])
        np.testing.assert_allclose(ds, 0.0, atol=1e-30)

    def test_states_stay_bounded(self):
        p = card()
        (vf, br), t_end = triangle([3.0, -3.0, 2.0, -1.5, 3.0, 0.0], 2e-6)
        r = nm.simulate(p, vf, t_end, s0=-1.0, breaks=br, rtol=1e-7)
        self.assertLessEqual(np.max(r["s"]), 1.0 + 1e-6)
        self.assertGreaterEqual(np.min(r["s"]), -1.0 - 1e-6)
        self.assertLessEqual(np.max(np.abs(r["h"])), nm.H_MAX + 1e-3)

    def test_reversible_charge_vanishes_at_zero(self):
        p = card(frev=0.5, c_w=0.01, c_r=0.003)
        for x in (-1.0, -0.2, 0.7, 1.0):
            self.assertAlmostEqual(nm.q_rev(p, 0.0, x, 0.0), 0.0, places=15)

    def test_leakage_odd_monotonic_and_calibrated(self):
        p = card()
        v = np.linspace(-3, 3, 241)
        j = nm.j_leak(p, v)
        self.assertEqual(float(nm.j_leak(p, 0.0)), 0.0)
        self.assertTrue(np.all(np.diff(j) > 0))
        np.testing.assert_allclose(j, -j[::-1], atol=1e-12 * np.max(np.abs(j)))
        self.assertTrue(0.05 < float(nm.j_leak(p, 2.0)) < 0.2)  # card set to J(2 V) ~ 0.1 A/m^2
        q = dict(p, spf_n=p["spf_n"] * 100)
        self.assertEqual(float(nm.j_leak(q, 1.5)), float(nm.j_leak(p, 1.5)))
        self.assertGreater(abs(float(nm.j_leak(q, -1.5))), 10 * abs(float(nm.j_leak(p, -1.5))))

    def test_leakage_activation_energy(self):
        pf = card(g_oh=0.0)
        t1, t2 = 298.15, 358.15
        j1, j2 = float(nm.j_leak(pf, 0.2, t1)), float(nm.j_leak(pf, 0.2, t2))
        ea = np.log(j2 / j1) * (nm.K_B / nm.Q_E) / (1 / t1 - 1 / t2)
        self.assertTrue(0.4 < ea < 0.68, ea)  # trap depth 0.68 eV minus the field lowering


class TestConstantFieldReduction(unittest.TestCase):
    def test_ode_matches_single_pulse_nls_form(self):
        """Under a constant field from saturation, h = R t and E3 gives sum w [1 - exp(-(R t)^beta)]."""
        p = card()
        for v, w in ((1.8, 1e-6), (2.5, 0.3e-6), (3.0, 3e-6)):
            (vf, br), t_end = pulse(v, w, edge=1e-12, post=1e-9)
            r = nm.simulate(p, vf, t_end, s0=-1.0, breaks=br, rtol=1e-9)
            f_ode = 0.5 * (r["x"][-1] + 1.0)
            f_cf = nm.switched_fraction(p, v, w)
            self.assertLess(abs(f_ode - f_cf), 2e-4, (v, w, f_ode, f_cf))

    def test_sampled_sum_converges_to_continuous_nls_integral(self):
        """sum_k w_k f_k -> int {1 - exp[-(t/tau(E))^beta]} dG(E) for a dense, deep grid."""
        p = card(y_max=30.0)
        v, t = 2.0, 1e-6
        a, b, pt = p["ap"], p["emp"], p["pt"]

        def integrand(E):
            dens = pt * a / b * (E / b) ** (-a - 1) * (1 + (E / b) ** (-a)) ** (-pt - 1)
            u = v - p["vimp"]
            rate = np.exp(-((E * p["tfe"] / u) ** p["alpha"])) / p["tau0"]
            return (1 - np.exp(-((rate * t) ** p["beta"]))) * dens

        ref = quad(integrand, 1e5, 1e10, points=[1e8, 2e8, 4e8], limit=400)[0]
        self.assertLess(abs(nm.switched_fraction(p, v, t, nd=2000) - ref), 1e-4)

    def test_beta_one_reduces_to_kai_per_class(self):
        p = card(beta=1.0)
        dom = nm.make_domains(p)
        rp, _ = nm.rates(p, dom, 2.0, 0.0, p["t0"])
        self.assertAlmostEqual(nm.switched_fraction(p, 2.0, 1e-6), float(dom.w @ (1 - np.exp(-rp * 1e-6))), places=12)


class TestClosedFormsAgainstODE(unittest.TestCase):
    def test_pulse_train(self):
        p = card()
        n_p, t_on, t_off, v = 10, 0.5e-6, 2e-6, 1.6
        t, vv = [0.0], [0.0]
        for _ in range(n_p):
            t0 = t[-1]
            t += [t0 + 1e-12, t0 + t_on, t0 + t_on + 1e-12, t0 + t_on + t_off]
            vv += [v, v, 0.0, 0.0]
        vf, br = nm.pwl(t, vv)
        r = nm.simulate(p, vf, t[-1], s0=-1.0, breaks=br, rtol=1e-9)
        ode = 0.5 * (r["x"][-1] + 1.0)
        closed = nm.train_fraction(p, v, t_on, t_off, n_p)
        self.assertLess(abs(ode - closed) / closed, 0.01, (ode, closed))

    def test_pund_closed_form(self):
        p = card()
        closed = nm.pund(p, 3.0, 10e-6)
        self.assertAlmostEqual(closed["P"] - closed["U"], 2 * p["ps"], delta=0.01 * 2 * p["ps"])
        self.assertAlmostEqual(closed["N"] - closed["D"], -2 * p["ps"], delta=0.01 * 2 * p["ps"])

    def test_monotonic_in_amplitude_width_and_count(self):
        p = card()
        byv = [nm.switched_fraction(p, v, 1e-6) for v in (1.0, 1.5, 2.0, 2.5, 3.0)]
        byt = [nm.switched_fraction(p, 2.0, t) for t in (1e-7, 1e-6, 1e-5)]
        byn = [nm.train_fraction(p, 1.2, 1e-7, 1e-6, n) for n in (1, 10, 100, 1000)]
        for seq in (byv, byt, byn):
            self.assertTrue(all(b > a for a, b in zip(seq, seq[1:])), seq)
        byoff = [nm.train_fraction(p, 1.4, 1e-7, t, 100) for t in (1e-7, 1e-6, 1e-5, 1e-4)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(byoff, byoff[1:])), byoff)

    def test_repeat_waveform_matches_ode_for_bipolar_cycles(self):
        p = card()
        period, amp, n = 1e-6, 0.9, 8
        tt = np.linspace(0, period, 801)
        vv = np.interp(tt, [0, period / 4, 3 * period / 4, period], [0, -amp, amp, 0])
        closed = nm.repeat_waveform(p, tt, vv, n, 1.0)
        (vf, br), t_end = triangle([-amp, amp, 0.0] * n, period / 4)
        r = nm.simulate(p, vf, t_end, s0=1.0, breaks=br, rtol=1e-9)
        self.assertLess(abs((1 - closed) - (1 - r["x"][-1])) / (1 - r["x"][-1]), 0.02)

    def test_fixed_grid_march_converges_with_step(self):
        p = card()
        dom = nm.make_domains(p)
        out = []
        for n_steps in (400, 1600):
            t = np.linspace(0, 1.2e-6, n_steps + 1)
            v = np.where(t < 1e-6, 1.8, 0.0)
            s, _ = nm.march(p, dom, -np.ones(len(dom.w)), np.zeros(len(dom.w)), t, v, theta=1.0)
            out.append(float(dom.w @ s))
        ref = 2 * nm.switched_fraction(p, 1.8, 1e-6) - 1
        self.assertLess(abs(out[1] - ref), abs(out[0] - ref) + 1e-9)
        self.assertLess(abs(out[1] - ref), 5e-3)


class TestNoArtificialAccumulation(unittest.TestCase):
    def test_small_symmetric_signal_from_partial_state(self):
        drift, x0 = small_signal_drift(card())
        self.assertTrue(-0.9 < x0 < 0.9)
        self.assertLess(drift, 1e-9)

    def test_millions_of_small_cycles(self):
        p = card()
        period = 1e-6
        tt = np.linspace(0, period, 401)
        for amp in (0.026, 0.3):
            vv = np.interp(tt, [0, period / 4, 3 * period / 4, period], [0, -amp, amp, 0])
            for s0 in (1.0, -1.0):
                self.assertEqual(nm.repeat_waveform(p, tt, vv, 1e6, s0), s0)

    def test_real_accumulation_converges_with_classes(self):
        """Near the tail the loss is genuine: it stays finite as nd grows and grows with the cycle count."""
        p = card()
        loss = {nd: 1 - nm.bipolar_pairs(p, 0.9, 40e-9, 100e-9, 1e4, nd=nd) for nd in (64, 256)}
        self.assertGreater(loss[256], 0.01)
        self.assertLess(abs(loss[64] - loss[256]) / loss[256], 0.05)
        self.assertGreater(1 - nm.bipolar_pairs(p, 0.9, 40e-9, 100e-9, 1e6), 1 - nm.bipolar_pairs(p, 0.9, 40e-9, 100e-9, 1e4))


class TestReversibleResponse(unittest.TestCase):
    def test_butterfly_branches(self):
        p = card()

        def c(v, x):
            return (nm.q_rev(p, v + 1e-4, x) - nm.q_rev(p, v - 1e-4, x)) / 2e-4

        vs = np.linspace(-1.2, 1.2, 121)
        up, dn = np.array([c(v, 1.0) for v in vs]), np.array([c(v, -1.0) for v in vs])
        np.testing.assert_allclose(up, dn[::-1], atol=1e-9)
        self.assertGreater(up[vs < 0].mean(), up[vs > 0].mean())
        half = np.array([c(v, 0.0) for v in vs])
        np.testing.assert_allclose(half, 0.5 * (up + dn), atol=1e-9)

    def test_per_class_sum_equals_mean_state_form(self):
        p = card()
        dom = nm.make_domains(p, 16)
        gup, gdn = nm.branches(p, 0.6)
        s = np.ones(16)
        s[3] = -1.0
        direct = p["c_d"] * 0.6 + float(np.sum(dom.w * ((1 + s) / 2 * gup + (1 - s) / 2 * gdn)))
        self.assertAlmostEqual(direct, nm.q_rev(p, 0.6, float(dom.w @ s)), places=14)

    def test_frev_split(self):
        p = card()
        q5 = dict(p, frev=0.5)
        d0 = nm.q_rev(p, 0.6, 1.0) - nm.q_rev(p, 0.6, -1.0)
        d_fast = nm.q_rev(q5, 0.6, 1.0, 0.0) - nm.q_rev(q5, 0.6, -1.0, 0.0)
        self.assertAlmostEqual(d_fast, 0.5 * d0, places=12)
        self.assertAlmostEqual(nm.q_rev(q5, 0.6, 1.0, 0.6), nm.q_rev(p, 0.6, 1.0, 0.6), places=14)

    def test_passivity_with_low_pass_split(self):
        for frev in (0.5, 0.2):
            p = card(frev=frev, emp=1e11, emn=1e11, g_oh=0.0, spf_p=0.0, spf_n=0.0)
            for v0 in (0.67, -0.9):
                for s0 in (1.0, -1.0):
                    self.assertGreater(cycle_work(p, v0, s0), 0.0)


class TestChargeConservation(unittest.TestCase):
    def test_series_resistor_circuit(self):
        """Source charge = change of terminal charge + leakage integral, with q's partial derivatives written out."""
        p = card(frev=0.5, c_w=0.005, c_r=0.003, tau_r=2e-8, g_oh=1e-2, area=1e-12)
        temp, dom, n = p["t0"], nm.make_domains(p, 16), 16
        area, rs = p["area"], 2e5
        vf, br = nm.pwl([0, 5e-9, 7e-9, 1.2e-6, 1.202e-6, 1.5e-6, 1.502e-6, 2.7e-6, 2.702e-6, 3e-6],
                        [0, 0, 2.8, 2.8, 0, 0, -2.4, -2.4, 0, 0])
        vw_, vs_, vi_, dc_, fr = p["vw"], p["vs"], p["vi"], p["dc"], p["frev"]

        def lor(w, sgn):
            return dc_ / (1.0 + ((w - vi_ + sgn * vs_) / vw_) ** 2)

        def gd(w):
            up, dn = nm.branches(p, w)
            return 0.5 * (up - dn)

        def f(t, z):
            s, h, y, sig, v = z[:n], z[n:2 * n], z[2 * n], z[2 * n + 1], z[2 * n + 2]
            ds, dh, dy, dg = nm.rhs(p, dom, s, h, y, sig, v, temp)
            x, dx = float(dom.w @ s), float(dom.w @ ds)
            dqv = p["c_d"] + fr * 0.5 * (lor(v, 1) + lor(v, -1) + x * (lor(v, 1) - lor(v, -1)))
            dqx = p["ps"] + fr * gd(v) + (1 - fr) * gd(y) - 2 * p["c_w"] * x * y
            dqy = (1 - fr) * 0.5 * (lor(y, 1) + lor(y, -1) + x * (lor(y, 1) - lor(y, -1))) + p["c_w"] * (1 - x * x) + p["c_r"]
            i_src = (vf(t) - v) / rs
            dv = (i_src / area - float(nm.j_leak(p, v, temp)) - dqx * dx - dqy * dy) / dqv
            return np.concatenate([ds, dh, [dy, dg, dv, i_src, area * float(nm.j_leak(p, v, temp))]])

        z = np.concatenate([-np.ones(n), np.zeros(n), [0.0, -1.0, 0.0, 0.0, 0.0]])
        q0 = nm.q_term(p, 0.0, -1.0, 0.0, temp)
        for a, b in zip(br[:-1], br[1:]):
            atol = np.concatenate([np.full(2 * n + 3, 1e-9), [1e-24, 1e-24]])
            sol = solve_ivp(f, (a, b), z, method="Radau", rtol=1e-8, atol=atol)
            self.assertTrue(sol.success, sol.message)
            z = sol.y[:, -1]
        x = float(dom.w @ z[:n])
        dq = area * (nm.q_term(p, z[2 * n + 2], x, z[2 * n], temp) - q0)
        err = abs(z[2 * n + 3] - dq - z[2 * n + 4]) / (2 * p["ps"] * area)
        self.assertGreater(abs(x + 1.0), 0.01)  # something switched
        self.assertLess(err, 1e-4)


class TestHistoricalBugsAreCaught(unittest.TestCase):
    """Each check above must fail for a bug of the kind found while the model was developed."""

    def test_state_dependent_only_low_pass_violates_passivity(self):
        def q_rev_bad(p, v, x, y=None, temp=None):
            y = v if y is None else y
            up, dn = nm.branches(p, v)
            upy, dny = nm.branches(p, y)
            fr = p["frev"]
            return p["c_d"] * v + 0.5 * (up + dn) + x * (fr * 0.5 * (up - dn) + (1 - fr) * 0.5 * (upy - dny))

        p = card(frev=0.2, emp=1e11, emn=1e11, g_oh=0.0, spf_p=0.0, spf_n=0.0)
        with mock.patch.object(nm, "q_rev", q_rev_bad):
            works = [cycle_work(p, v0, s0) for v0 in (0.67, -0.9) for s0 in (1.0, -1.0)]
        self.assertLess(min(works), 0.0)

    def test_path_driven_update_shows_artificial_drift(self):
        """An update that pulls the state toward zero whenever a voltage is applied, regardless of any threshold, drifts under small signals."""
        orig = nm.rhs

        def rhs_bad(p, dom, s, h, y, sig, v, temp):
            ds, dh, dy, dg = orig(p, dom, s, h, y, sig, v, temp)
            return ds - 1e5 * abs(v) * s * 1e-3, dh, dy, dg

        with mock.patch.object(nm, "rhs", rhs_bad):
            drift, _ = small_signal_drift(card(), cycles=5)
        self.assertGreater(drift, 1e-6)

    def test_rate_based_reversal_breaks_the_closed_form(self):
        """Deciding 'reversal' from the rate instead of the field direction leaves h un-reset in the slow tail.

        Bipolar pulse pairs (+-0.67 V, 40 ns, 100 ns gaps, 300 pairs) are integrated on a
        fixed 1 ns grid (backward Euler) and compared with the closed form of bipolar_pairs.
        """
        p = card()
        dom = nm.make_domains(p)
        t, v = [0.0], [0.0]
        for _ in range(300):
            for vv in (0.67, -0.67):
                t += list(t[-1] + np.arange(1e-9, 40.5e-9, 1e-9))
                v += [vv] * 40
                t += list(t[-1] + np.arange(10e-9, 100.5e-9, 10e-9))
                v += [0.0] * 10
        t, v = np.array(t), np.array(v)
        s0 = np.vstack([np.ones(16), -np.ones(16)])

        def loss():
            s, _ = nm.march(p, dom, s0, np.zeros(16), t, v, theta=1.0, cache={})
            return 1 - 0.5 * float(dom.w @ s[0] - dom.w @ s[1])

        closed = 1 - nm.bipolar_pairs(p, 0.67, 40e-9, 100e-9, 300)
        self.assertLess(abs(loss() - closed) / closed, 0.1)
        orig = nm.rev_gates

        def gates_by_rate(p_, u):
            gp, gn = orig(p_, u)
            rp, rn = nm.rates(p_, dom, u + p_["vimp"], 0.0, p_["t0"])
            return gp * (rp > 1e2), gn * (rn > 1e2)

        with mock.patch.object(nm, "rev_gates", gates_by_rate):
            bad = loss()
        self.assertGreater(bad / closed, 1.5)

    def test_step_does_not_carry_progress_across_a_sub_threshold_reversal(self):
        """A reversed drive below the reset threshold must not reuse the old direction's h.

        Write at 1.5 V for 20 us, then hold at -0.05 V (|u| = 0.35 V < u_x = 0.4 V) for 1 us.
        E3b (simulate) switches nothing; the unfixed closed form switched 7.8 %.
        """
        p = card(vimp=0.3, mul_ea=0.4)
        dom = nm.make_domains(p)
        n = len(dom.w)
        s, h = -np.ones(n), np.zeros(n)
        for _ in range(200):
            s, h = nm.step(p, dom, s, h, 1.5, 1e-7)
        x_w = float(dom.w @ s)
        for _ in range(100):
            s, h = nm.step(p, dom, s, h, -0.05, 1e-8)
        vf, br = nm.pwl([0, 20e-6, 20.001e-6, 21.001e-6], [1.5, 1.5, -0.05, -0.05])
        r = nm.simulate(p, vf, 21.001e-6, s0=-1.0, breaks=br)
        i_w = np.searchsorted(r["t"], 20e-6)
        sim = 0.5 * (r["x"][i_w] - r["x"][-1])
        self.assertLess(abs(0.5 * (x_w - float(dom.w @ s)) - sim), 1e-4)


class TestSpiceFilesInSync(unittest.TestCase):
    def test_parameter_card_matches_generic(self):
        inc = (ROOT / "spice" / "params" / "generic.inc").read_text(encoding="utf-8")
        self.assertEqual(inc, nm.spice_model_card())

    def test_verilog_a_defaults_match_generic(self):
        va = (ROOT / "spice" / "fecap_nls.va").read_text(encoding="utf-8")
        found = dict(re.findall(r"parameter real (\w+)\s*=\s*([-+0-9.eE]+)", va))
        for name, prm in nm.GENERIC.items():
            if prm.va_name in ("FE_ND", "FE_NI", "FE_NQ"):
                self.assertIn(f"`define {prm.va_name} {prm.value}", va)
                continue
            self.assertIn(prm.va_name, found, name)
            self.assertAlmostEqual(float(found[prm.va_name]), prm.value, delta=1e-12 * max(1.0, abs(prm.value)),
                                   msg=name)

    def test_reference_constants_match_verilog_a(self):
        va = (ROOT / "spice" / "fecap_nls.va").read_text(encoding="utf-8")
        for token in ("pow(60.0", "abs(vimpe) + 0.1", "/ 0.1;", "/ 0.01)", "/ 1.0e-10", "min(hu, 6.0) / 5.0",
                      "max(hu, 1.0e-6)", "min(2.0 * `FE_BS", "+ 1.0e-4, wk_kw)", "+ 1.0e-4, fat_kf)",
                      "/ (tfe * tfe) + 1.0e6)", "pow(efl / 1.0e8, fat_nf)"):
            self.assertIn(token, va)
        self.assertEqual((nm.X_MAX, nm.U_XM, nm.V_GATE, nm.U_XW, nm.TAU_X, nm.H_CLIP, nm.H_MAX, nm.H_S),
                         (60.0, 0.1, 0.1, 0.01, 1e-10, 6.0, 5.0, 1e-6))
        self.assertEqual((nm.A_EPS, nm.E_FLOOR**2, nm.E_REF), (1e-4, 1e6, 1e8))
        self.assertAlmostEqual(float(re.search(r"`define FE_BS ([0-9.eE+-]+)", va).group(1)),
                               float(np.sqrt(nm.Q_E / (4 * np.pi * nm.EPS0))), delta=1e-13)

    def test_decks_match_generator(self):
        import sys

        sys.path.insert(0, str(ROOT / "spice"))
        import compare_ngspice as cn

        for d in cn.all_decks():
            self.assertEqual((ROOT / "spice" / "tests" / f"{d.name}.cir").read_text(encoding="utf-8"), d.text(), d.name)


if __name__ == "__main__":
    unittest.main()
