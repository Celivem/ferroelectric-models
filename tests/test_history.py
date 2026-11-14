"""Tests for the cycling and storage history of fe_models.nls_multidomain (E13-E17).

Wake-up (pinned fraction and split internal bias released by switching),
fatigue (field-time stress that pins domains), the pinned pool that keeps the
charge of pinned domains, and the logarithmic-time imprint of log-spaced
trapping states.  Every term is off in the generic card.
"""

import unittest
from unittest import mock

import numpy as np
from scipy.optimize import brentq

from fe_models import nls_multidomain as nm

UC = 1e-2


def triangle_cycles(n, amp=3.0, quarter=2.5e-6):
    """n cycles 0 -> +amp -> -amp -> 0 ... ; returns (v_of_t, breaks, t_end, period, t, v)."""
    t, v = [0.0], [0.0]
    for a in [amp, -amp] * n + [0.0]:
        t.append(t[-1] + abs(a - v[-1]) / amp * quarter)
        v.append(a)
    vf, br = nm.pwl(t, v)
    return vf, br, t[-1], 4 * quarter, np.array(t), np.array(v)


def segments(segs, edge=0.1e-6):
    """[(V, duration), ...] -> (v_of_t, breaks, t_end, segment end times)."""
    t, v, ends, now = [0.0], [segs[0][0]], [], 0.0
    for i, (vv, d) in enumerate(segs):
        if i > 0:
            now += edge
            t.append(now)
            v.append(vv)
        now += d
        t.append(now)
        v.append(vv)
        ends.append(now)
    vf, br = nm.pwl(t, v)
    return vf, br, now, ends


ALL_ON = dict(wk_fp=0.3, wk_vsp=0.4, wk_nw=3.0, wk_kw=0.8, fat_d=0.3, fat_tf=3e-4, fat_kf=0.6, fat_ke=0.2,
              imp_v=0.3, imp_t1=1e-7, imp_t2=1e-2)


class TestDefaultOff(unittest.TestCase):
    def test_generic_card_has_every_history_term_off(self):
        p = nm.generic_card()
        self.assertEqual(nm.aging_flags(p), (False, False, False))
        self.assertFalse(nm.aging_on(p))
        for k, prm in ALL_ON.items():
            self.assertTrue(nm.aging_on(nm.generic_card(**{k: prm})) or k not in ("wk_fp", "wk_vsp", "fat_d", "imp_v"))

    def test_history_argument_is_ignored_when_off(self):
        p = nm.generic_card()
        vf, br, t_end, *_ = triangle_cycles(1, quarter=2.5e-5)
        a = nm.simulate(p, vf, t_end, breaks=br)
        b = nm.simulate(p, vf, t_end, breaks=br, history=nm.History(nw=1e9, af=1e3, si=1.0, pi=0.5))
        np.testing.assert_array_equal(a["qd"], b["qd"])
        np.testing.assert_array_equal(a["t"], b["t"])

    def test_aged_equations_reduce_to_the_original_ones(self):
        p = nm.generic_card()
        dom = nm.make_domains(p)
        rng = np.random.default_rng(1)
        n = len(dom.w)
        taus, signs = nm.imprint_taus(p), nm.split_signs(n)
        for v in (-2.2, -0.3, 0.0, 0.4, 1.7, 3.0):
            s, h = rng.uniform(-1, 1, n), rng.uniform(-3, 3, n)
            y, sig = 0.7 * v, 0.2
            ref = nm.rhs(p, dom, s, h, y, sig, v, p["t0"])
            z = np.concatenate([s, h, [y, sig, 0.0, 0.0, 0.0, 0.0], np.zeros(len(taus))])
            out = nm.rhs_aged(p, dom, z, v, p["t0"], taus, signs)
            for a, b in zip(ref, out[:4]):
                np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-300)
            self.assertEqual([out[4], out[5], out[6], out[7]], [0.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(out[8], 0.0)

    def test_forced_aged_path_matches_original_simulation(self):
        """The extended state vector with every term off integrates to the original result."""
        p = nm.generic_card()
        vf, br, t_end, *_ = triangle_cycles(2, quarter=2.5e-6)
        a = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-9, atol=1e-11)
        with mock.patch.object(nm, "aging_on", return_value=True):
            b = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-9, atol=1e-11)
        self.assertIn("pi", b)
        qb = np.interp(a["t"], b["t"], b["qd"])
        self.assertLess(np.max(np.abs(qb - a["qd"])) / UC, 2e-3)
        self.assertEqual(float(np.max(np.abs(b["f"]))), 0.0)

    def test_cycle_history_is_constant_when_off(self):
        p = nm.generic_card()
        t, v = [0, 2.5e-6, 7.5e-6, 1e-5], [0, 3, -3, 0]
        r = nm.cycle_history(p, t, v, [1, 1e3, 1e8, 1e12], nd=16)
        # only the genuine slow creep of the soft tail remains (0.05 % over 1e12 cycles)
        np.testing.assert_allclose(r["dP"], r["dP"][0], rtol=2e-3)
        np.testing.assert_array_equal(r["W"], 0.0)
        np.testing.assert_array_equal(r["D"], 0.0)


class TestMonotonicDamage(unittest.TestCase):
    def test_laws(self):
        for kappa in (0.3, 1.0, 1.7):
            p = nm.generic_card(fat_d=0.4, fat_kf=kappa, wk_fp=0.2, wk_kw=kappa)
            a = np.logspace(-6, 9, 600)
            D, dD = nm.fatigue_damage(p, a)
            u, du = nm.wake_state(p, a)
            self.assertTrue(np.all(np.diff(D) >= 0) and np.all(np.diff(u) <= 0))
            self.assertTrue(np.all(np.diff(D[(a > 1e-3) & (a < 1)]) > 0))
            self.assertAlmostEqual(float(nm.fatigue_damage(p, 0.0)[0]), 0.0, places=15)
            self.assertAlmostEqual(float(nm.wake_state(p, 0.0)[0]), 1.0, places=15)
            self.assertTrue(np.all(D <= 0.4) and abs(D[-1] - 0.4) < 1e-6)
            m = a < 2.0
            a, D, dD, u, du = a[m], D[m], dD[m], u[m], du[m]
            eps = 1e-4 * (a + nm.A_EPS)
            np.testing.assert_allclose(dD, (nm.fatigue_damage(p, a + eps)[0] - nm.fatigue_damage(p, a - eps)[0]) / (2 * eps),
                                       rtol=1e-5)
            np.testing.assert_allclose(du, (nm.wake_state(p, a + eps)[0] - nm.wake_state(p, a - eps)[0]) / (2 * eps),
                                       rtol=1e-5)

    def test_simulated_ages_never_decrease(self):
        p = nm.generic_card(**ALL_ON)
        vf, br, t_end, period, *_ = triangle_cycles(4)
        r = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-6, max_step=period / 200)
        # monotonic up to the integration tolerance (rtol = 1e-6)
        for key in ("a_w", "a_f", "D"):
            self.assertGreater(float(np.min(np.diff(r[key]))), -1e-6 * float(np.max(r[key])), key)
        self.assertTrue(np.all((r["W"] >= 0) & (r["W"] <= 1)))
        self.assertTrue(np.all((r["f"] >= 0) & (r["f"] < 1)))
        self.assertGreater(r["W"][-1], 0.5)          # it does wake up
        self.assertGreater(r["D"][-1], 0.1)          # and it does fatigue
        self.assertTrue(np.all(np.abs(r["pi"]) <= r["f"] + 1e-9))

    def test_long_cycling_switched_charge_falls_monotonically(self):
        p = nm.generic_card(fat_d=0.5, fat_tf=10.0, fat_kf=0.5, fat_ke=0.2)
        t, v = [0, 2.5e-6, 7.5e-6, 1e-5], [0, 3, -3, 0]
        n = np.logspace(0, 12, 25)
        r = nm.cycle_history(p, t, v, n, nd=16)
        self.assertTrue(np.all(np.diff(r["D"]) >= 0))
        self.assertTrue(np.all(np.diff(r["dP"]) <= 1e-12))
        self.assertLess(r["dP"][-1], 0.6 * r["dP"][0])
        # raising the activation field with damage widens the loop
        self.assertGreater(r["vc_p"][-1], r["vc_p"][0])

    def test_fatigue_onset_scales_with_field_as_the_stress_law(self):
        """Square-wave cycling: cycles to D = D_max/2 scale as (V1/V2)^n_F."""
        nf = 4.0
        p = nm.generic_card(fat_d=0.4, fat_tf=1.0, fat_nf=nf, fat_kf=1.0)
        half = []
        for amp in (2.0, 3.0):
            t = [0, 1e-9, 5e-6, 5e-6 + 2e-9, 1e-5 - 1e-9, 1e-5]
            v = [0, amp, amp, -amp, -amp, 0]
            per = nm.stress_per_period(p, t, v)
            half.append(np.log(2.0) / per)   # D = D_max/2 at a_F = ln 2 (kappa = 1)
        self.assertAlmostEqual(half[0] / half[1], (3.0 / 2.0) ** nf, delta=0.02 * (1.5**nf))


class TestImprint(unittest.TestCase):
    def test_log_time_slope_of_the_trapped_charge(self):
        p = nm.generic_card(imp_v=0.4, imp_t1=1e-7, imp_t2=1e5)
        t = np.logspace(-3, 1, 41)  # well inside [tau_1, tau_2]
        sig = nm.imprint_response(p, t)
        slope = np.polyfit(np.log(t), sig, 1)[0]
        self.assertAlmostEqual(slope * np.log(1e5 / 1e-7), 1.0, delta=0.03)
        local = np.diff(sig) / np.diff(np.log(t)) * np.log(1e12)
        self.assertLess(np.max(np.abs(local - 1.0)), 0.15)  # ripple of 12 states over 12 decades

    def test_coercive_voltage_shift_is_logarithmic_in_storage_time(self):
        p = nm.generic_card(imp_v=0.4, imp_t1=1e-7, imp_t2=1e5)
        holds = np.logspace(-3, 1, 9)
        vc = []
        for th in holds:
            s_i = float(nm.imprint_response(p, th))
            pe, dom, _ = nm.aged_card(p, s_i=s_i)

            def frac(v):
                rp, rn = nm.rates(pe, dom, v, 0.0, p["t0"])
                return float(dom.w @ (1 - np.exp(-(np.abs(rp - rn) * 1e-6) ** pe["beta"]))) - 0.5

            vc.append(brentq(frac, -5.0, -0.5))
        slope = np.polyfit(np.log(holds), vc, 1)[0]
        # stored + : switching to - needs a more negative voltage, by V_I / ln(tau_2/tau_1) per e-fold
        self.assertAlmostEqual(slope, -0.4 / np.log(1e12), delta=0.05 * 0.4 / np.log(1e12))

    def test_storage_reduces_opposite_switching_in_the_ode(self):
        p = nm.generic_card(imp_v=0.4, imp_t1=1e-7, imp_t2=1e-1)
        out = []
        for hold in (1e-5, 1e-2):
            vf, br, t_end, ends = segments([(0.0, 0.5e-6), (3.0, 2e-6), (0.0, hold), (-1.6, 3e-6), (0.0, 2e-6)])
            r = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-7)
            x_before = np.interp(ends[2], r["t"], r["x"])
            out.append(x_before - r["x"][-1])
        self.assertGreater(out[0], out[1] + 0.05)
        p0 = nm.generic_card()
        same = []
        for hold in (1e-5, 1e-2):
            vf, br, t_end, ends = segments([(0.0, 0.5e-6), (3.0, 2e-6), (0.0, hold), (-1.6, 3e-6), (0.0, 2e-6)])
            r = nm.simulate(p0, vf, t_end, breaks=br, rtol=1e-7)
            same.append(np.interp(ends[2], r["t"], r["x"]) - r["x"][-1])
        self.assertAlmostEqual(same[0], same[1], delta=5e-3)

    def test_imprint_recovers_after_rewrite(self):
        p = nm.generic_card(imp_v=0.3, imp_t1=1e-7, imp_t2=1e-2)
        vf, br, t_end, ends = segments([(0.0, 0.5e-6), (3.0, 2e-6), (0.0, 1e-2), (-3.0, 2e-6), (0.0, 1e-2)])
        r = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-7)
        self.assertGreater(np.interp(ends[2], r["t"], r["s_i"]), 0.8)
        self.assertLess(r["s_i"][-1], -0.8)

    def test_thermal_activation(self):
        p = nm.generic_card(imp_v=0.3, imp_ea=0.1)
        ratio = nm.imprint_taus(p, 358.15) / nm.imprint_taus(p, 298.15)
        np.testing.assert_allclose(ratio, np.exp(0.1 / nm.K_B_EV * (1 / 358.15 - 1 / 298.15)), rtol=1e-12)
        self.assertGreater(float(nm.imprint_response(p, 1.0, 358.15)), float(nm.imprint_response(p, 1.0, 298.15)))


class TestConservationAndPassivity(unittest.TestCase):
    def test_pinning_moves_no_charge(self):
        """Pure time aging (n_F = 0) at a fixed bias without switching: D grows, the terminal charge does not move."""
        p = nm.generic_card(fat_d=0.5, fat_tf=1e-4, fat_nf=0.0, fat_kf=1.0, g_oh=0.0, spf_p=0.0, spf_n=0.0)
        for vb in (0.0, 1.0):
            vf, br, t_end, ends = segments([(vb, 1e-3)])
            r = nm.simulate(p, vf, t_end, breaks=br, s0=1.0, rtol=1e-9, atol=1e-12)
            self.assertGreater(r["D"][-1], 0.49)
            self.assertLess(np.max(np.abs(r["qd"] - r["qd"][0])), 1e-9)
            self.assertAlmostEqual(float(r["pi"][-1]), float(r["f"][-1]), places=6)

    def test_release_moves_no_charge(self):
        """Wake-up release: the class states take the pool's mean state; P changes only through switching."""
        p = nm.generic_card(wk_fp=0.4, wk_nw=2.0, wk_kw=1.0)
        dom = nm.make_domains(p)
        n = len(dom.w)
        rng = np.random.default_rng(3)
        taus, signs = nm.imprint_taus(p), nm.split_signs(n)
        for v in (1.6, -1.9, 2.5):
            s, h = rng.uniform(-1, 1, n), rng.uniform(0.1, 2.0, n) * np.sign(v)
            a_w, pi = 0.3, 0.07
            z = np.concatenate([s, h, [v, 0.0, pi, 0.0, a_w, 0.0], np.zeros(len(taus))])
            ds, _, _, _, dpi, _, da_w, _, _ = nm.rhs_aged(p, dom, z, v, p["t0"], taus, signs)
            ds_sw = nm.rhs(p, dom, s, h, v, 0.0, v, p["t0"])[0]
            f, dfw, _, _, _ = nm.pinned_fraction(p, a_w, 0.0)
            fdot = dfw * da_w
            self.assertLess(fdot, 0.0)
            xs = float(dom.w @ s)
            # d/dt [(1 - f) x + pi] minus the switching part
            resid = -fdot * xs + (1.0 - f) * float(dom.w @ (ds - ds_sw)) + dpi
            scale = abs(fdot) + float(np.max(np.abs(ds)))
            self.assertLess(abs(resid), 1e-13 * scale)
            self.assertGreater(abs(dpi), 1e-6 * abs(fdot))
            self.assertAlmostEqual(da_w, float(dom.w @ np.abs(ds_sw)) / (4 * p["wk_nw"]), places=12)

    def test_every_cycle_absorbs_energy(self):
        p = nm.generic_card(**ALL_ON)
        vf, br, t_end, period, *_ = triangle_cycles(6)
        r = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-7, max_step=period / 400)
        for k in range(6):
            m = (r["t"] >= k * period) & (r["t"] <= (k + 1) * period)
            work = float(np.trapezoid(r["v"][m], r["qd"][m]))
            self.assertGreater(work, 0.5)  # J/m^2; one full loop absorbs about 4 P_r V_c

    def test_small_signal_is_passive_with_history(self):
        """Sub-coercive cycling of a cycled, fatigued, imprinted film: no cycle delivers energy.

        The imprint states start at the stored state (x_t = 0.3), so the slow
        states stay put; a relaxing imprint is an internal source of bounded
        energy and is excluded here.
        """
        p = nm.generic_card(**dict(ALL_ON, fat_tf=1e12))
        vf, br, t_end, period, *_ = triangle_cycles(6, amp=0.4, quarter=0.25e-6)
        hist = nm.History(nw=30.0, af=0.5, si=0.3)
        for card, h in ((p, hist), (nm.generic_card(), None)):
            r = nm.simulate(card, vf, t_end, breaks=br, rtol=1e-10, atol=1e-13, history=h, s0=0.3,
                            max_step=period / 200)
            if h is not None:
                self.assertGreater(r["f"][0], 0.1)
                self.assertLess(float(np.ptp(r["s_i"])), 1e-3)
            for k in range(1, 6):
                m = (r["t"] >= k * period) & (r["t"] <= (k + 1) * period)
                vv, qq = r["v"][m], r["qd"][m]
                gross = float(np.sum(np.abs(0.5 * (vv[1:] + vv[:-1]) * np.diff(qq))))
                # net work >= 0 up to the quadrature noise of the adaptive time points
                self.assertGreater(float(np.trapezoid(vv, qq)), -1e-4 * gross)


class TestDrivers(unittest.TestCase):
    def test_switching_fatigue_needs_switching(self):
        """With only the switching driver, unipolar stress does not fatigue, bipolar cycling does."""
        p = nm.generic_card(fat_d=0.4, fat_tf=0.0, fat_ns=2.0, fat_nf=2.0)
        t = [0, 2.5e-6, 5e-6, 7.5e-6, 1e-5]
        uni = nm.cycle_history(p, t, [0, 3, 0, 3, 0], [1, 100], nd=16, s_start=1.0)
        bip = nm.cycle_history(p, t, [0, 3, 0, -3, 0], [1, 100], nd=16, s_start=1.0)
        self.assertLess(uni["D"][-1], 1e-6)
        self.assertGreater(bip["D"][-1], 0.3)

    def test_field_time_fatigue_follows_time_under_field(self):
        """Field-time driver: at the same cycle count, longer pulses fatigue more."""
        p = nm.generic_card(fat_d=0.4, fat_tf=1e-3, fat_nf=2.0, fat_kf=1.0)
        d = []
        for w in (2e-6, 20e-6):
            t = [0, 1e-8, w, w + 2e-8, 2 * w, 2 * w + 1e-8]
            d.append(nm.cycle_history(p, t, [0, 3, 3, -3, -3, 0], [10], nd=16)["D"][0])
        self.assertGreater(d[1], 2 * d[0])

    def test_thermal_activation_of_wake_up(self):
        p = nm.generic_card(wk_fp=0.3, wk_nw=10.0, wk_ea=0.5)
        t, v = [0, 2.5e-6, 7.5e-6, 1e-5], [0, 3, -3, 0]
        w25 = nm.cycle_history(p, t, v, [5], nd=16)["W"][0]
        w85 = nm.cycle_history(p, t, v, [5], nd=16, temp=358.15)["W"][0]
        self.assertGreater(w85, w25 + 0.2)
        self.assertAlmostEqual(nm.arrhenius(p, 0.5, 358.15), float(np.exp(-0.5 / nm.K_B_EV * (1 / 358.15 - 1 / 298.15))))


class TestCycleAveraged(unittest.TestCase):
    def test_matches_direct_simulation(self):
        p = nm.generic_card(wk_fp=0.3, wk_vsp=0.3, wk_nw=4.0, fat_d=0.3, fat_tf=1e-3, fat_kf=0.7, fat_ke=0.2,
                            imp_v=0.3, imp_t1=1e-6, imp_t2=1e-2)
        nd, n = 8, 10
        vf, br, t_end, period, t, v = triangle_cycles(n)
        r = nm.simulate(p, vf, t_end, breaks=br, rtol=1e-7, atol=1e-10, max_step=period / 400, nd=nd)
        c = nm.cycle_history(p, [0, period / 4, 3 * period / 4, period], [0, 3, -3, 0], np.arange(n), nd=nd, sub=400)
        ends = np.arange(n) * period
        for key in ("W", "D", "s_i"):
            direct = np.interp(ends, r["t"], r[key])
            self.assertLess(np.max(np.abs(direct - c[key])), 0.015, key)
        dp = []
        for k in range(n):
            m = (r["t"] >= k * period) & (r["t"] <= (k + 1) * period)
            dp.append(p["ps"] * (1 - r["f"][m].mean()) * np.ptp(r["x"][m]))
        np.testing.assert_allclose(c["dP"][1:], np.array(dp)[1:], rtol=0.03)

    def test_long_runs_are_cheap_and_reach_the_limits(self):
        p = nm.generic_card(wk_fp=0.3, wk_nw=1e3, fat_d=0.3, fat_tf=30.0, fat_kf=0.5)
        r = nm.cycle_history(p, [0, 2.5e-6, 7.5e-6, 1e-5], [0, 3, -3, 0], [1e2, 1e4, 1e6, 1e9, 1e12], nd=16, sub=100)
        self.assertLess(r["W"][0], 0.2)
        self.assertGreater(r["W"][2], 0.99)
        self.assertGreater(r["D"][-1], 0.29)
        i = int(np.argmax(r["dP"]))
        self.assertTrue(0 < i < 4)  # wake-up first raises, fatigue then lowers the switched charge


if __name__ == "__main__":
    unittest.main()
