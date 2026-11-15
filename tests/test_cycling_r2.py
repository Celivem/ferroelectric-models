"""Tests for the second round of cycling terms of fe_models.nls_multidomain (E18-E20) and the loop read-out.

* E18: reversible depinning pool (field depins, re-pinning acts all the time);
* E19: lagged field-time fatigue driver;
* E20: leakage increase with the fatigue age;
* loop_figures: P-V loop read-out on the total charge (the observable of a
  dynamic hysteresis measurement), which is what makes 2E_c grow during wake-up.

Every term is off in the generic card.
"""

import unittest

import numpy as np

from fe_models import nls_multidomain as nm

UC = 1e-2
DEPIN = dict(wk_fq=0.4, wk_rho=0.3, wk_q1=2e-6, wk_q2=2e-4, wk_nq=2.0)
DEPIN_TR = dict(DEPIN, wk_tr=2e-6)


def pulse_train(n, amp, w, period, edge=20e-9):
    """n bipolar periods: +amp for w, 0 V, -amp for w, 0 V (half period each)."""
    t, v = [0.0], [0.0]
    for _ in range(n):
        t0 = t[-1]
        for a, start in ((amp, t0), (-amp, t0 + period / 2)):
            if start > t[-1]:
                t.append(start)
                v.append(0.0)
            t += [start + edge, start + edge + w, start + 2 * edge + w]
            v += [a, a, 0.0]
        t.append(t0 + period)
        v.append(0.0)
    return np.array(t), np.array(v)


def one_period(amp, w, period, edge=20e-9):
    t, v = pulse_train(1, amp, w, period, edge)
    return list(t), list(v)


class TestDefaultOff(unittest.TestCase):
    def test_generic_card_has_the_new_terms_off(self):
        p = nm.generic_card()
        self.assertFalse(nm.depin_on(p))
        self.assertFalse(nm.lag_on(p))
        self.assertEqual(p["fat_lk"], 0.0)
        self.assertFalse(nm.aging_on(p))
        self.assertTrue(nm.aging_on(nm.generic_card(wk_fq=0.1)))
        self.assertTrue(nm.aging_on(nm.generic_card(fat_lk=1.0)))

    def test_round_one_state_vector_is_unchanged(self):
        """With E18 and E19 off the aged right-hand side keeps its round-one layout and values."""
        p = nm.generic_card(wk_fp=0.3, wk_vsp=0.4, fat_d=0.3, fat_tf=1e-3, imp_v=0.2)
        dom = nm.make_domains(p)
        n = len(dom.w)
        taus, signs = nm.imprint_taus(p), nm.split_signs(n)
        z = np.concatenate([np.full(n, 0.3), np.full(n, 0.5), [0.1, 0.0, 0.05, 0.0, 0.2, 0.1], np.full(len(taus), 0.2)])
        out = nm.rhs_aged(p, dom, z, 1.2, p["t0"], taus, signs)
        self.assertEqual(len(out), 9)
        # pinned_fraction with the E18 argument at its default reproduces the old expression exactly
        f, dfw, dff, uw, D = nm.pinned_fraction(p, 0.2, 0.1)
        fw = p["wk_fp"] * uw
        self.assertEqual(f, 1.0 - (1.0 - fw) * (1.0 - D))

    def test_new_history_fields_are_ignored_when_off(self):
        p = nm.generic_card(wk_fp=0.2, wk_nw=3.0)
        t, v = one_period(3.0, 5e-6, 2e-5)
        a = nm.cycle_history(p, t, v, [3], nd=16, sub=40)
        b = nm.cycle_history(p, t, v, [3], nd=16, sub=40, history=nm.History(zq=0.7))
        np.testing.assert_array_equal(a["dP"], b["dP"])
        self.assertEqual(a["Z"][0], 0.0)
        np.testing.assert_array_equal(a["leak"], [1.0])

    def test_leak_factor_is_one_when_off(self):
        p = nm.generic_card()
        np.testing.assert_array_equal(nm.leak_factor(p, np.array([0.0, 3.0, 1e6])), 1.0)


class TestDepinningPool(unittest.TestCase):
    def test_plateau_is_the_duty_cycle_balance(self):
        """Under a pulse train the depinned fraction settles at phi D_c / (phi D_c + rho_q)."""
        p = nm.generic_card(**dict(DEPIN, wk_q1=1e-4, wk_q2=1e-2))   # slow states: little ripple within a period
        period = 2e-5
        for w in (1e-6, 4e-6, 9e-6):
            t, v = one_period(3.0, w, period)
            r = nm.cycle_history(p, t, v, [2e4], nd=8, sub=40)
            duty = 2 * (w + 2 * 20e-9 / 3) / period   # each 20 ns ramp counts 1/3 (phi ~ V^2)
            self.assertAlmostEqual(r["Z"][0], nm.depin_plateau(p, duty, 3.0), delta=3e-3)
        # the plateau grows with the duty cycle and stays below 1 at 100 %
        z = [nm.depin_plateau(p, d, 3.0) for d in (0.04, 0.1, 0.4, 1.0)]
        self.assertTrue(np.all(np.diff(z) > 0))
        self.assertLess(z[-1], 1.0)

    def test_per_reversal_repinning(self):
        """Re-pinning tied to switching: plateau phi D_c / (phi D_c + t_r f), and no re-pinning at rest."""
        p = nm.generic_card(wk_fq=0.4, wk_tr=4e-6, wk_q1=1e-4, wk_q2=1e-2, wk_nq=2.0)
        period = 2e-5
        t, v = one_period(3.0, 4e-6, period)
        r = nm.cycle_history(p, t, v, [2e4], nd=16, sub=80)
        duty = 2 * (4e-6 + 2 * 20e-9 / 3) / period
        # every pulse switches the film fully (act = 4 per period)
        self.assertAlmostEqual(r["Z"][0], nm.depin_plateau(p, duty, 3.0, freq=1 / period), delta=5e-3)
        self.assertLess(r["Z"][0], 0.95)
        z = r["zq"][-1]
        np.testing.assert_allclose(nm.depin_step(p, z, 0.0, 1e4, nm.depin_taus(p)), z, rtol=0, atol=1e-12)
        # in the ODE: a hold at 0 V after a few pulses keeps Z (a small floor field gives phi ~ 1e-10)
        pf = nm.generic_card(wk_fq=0.4, wk_tr=2e-6, wk_q1=2e-6, wk_q2=2e-4, wk_nq=2.0)
        tt, vv = pulse_train(4, 3.0, 4e-6, period)
        tt, vv = np.append(tt, tt[-1] + 3e-4), np.append(vv, 0.0)
        vf, br = nm.pwl(tt, vv)
        rr = nm.simulate(pf, vf, tt[-1], breaks=br, rtol=1e-8, atol=1e-12, max_step=1e-7)
        rest = rr["t"] > tt[-2] + 1e-6
        self.assertGreater(rr["Z"][rest][0], 0.05)
        self.assertLess(np.ptp(rr["Z"][rest]), 1e-6)

    def test_simulate_and_cycle_history_agree(self):
        p = nm.generic_card(**DEPIN_TR)
        period, n = 2e-5, 6
        t, v = pulse_train(n, 3.0, 5e-6, period)
        vf, br = nm.pwl(t, v)
        r = nm.simulate(p, vf, t[-1], breaks=br, rtol=1e-7, atol=1e-10, max_step=1e-7)
        tp, vp = one_period(3.0, 5e-6, period)
        ch = nm.cycle_history(p, tp, vp, np.arange(n + 1), nd=16, sub=200)
        z_sim = np.interp(np.arange(n + 1) * period, r["t"], r["Z"])
        np.testing.assert_allclose(ch["Z"], z_sim, atol=2e-3)
        self.assertGreater(ch["Z"][-1], 0.05)

    def test_rest_repins_and_moves_no_charge(self):
        """At 0 V the pool re-pins (Z falls, f rises) while the terminal charge stays put."""
        p = nm.generic_card(**DEPIN)
        vf, br = nm.pwl([0.0, 1e-6, 1e-6 + 2e-5, 1.02e-4 + 1e-6, 4e-4], [3.0, 3.0, 3.0, 0.0, 0.0])
        r = nm.simulate(p, vf, 4e-4, s0=-1.0, breaks=br, rtol=1e-8, atol=1e-12)
        rest = r["t"] > 1.2e-4
        self.assertTrue(np.all(np.diff(r["Z"][rest]) < 0))
        self.assertTrue(np.all(np.diff(r["f"][rest]) > 0))
        self.assertLess(np.ptp(r["qd"][rest]), 1e-9)
        self.assertLess(abs(np.ptp(r["x_t"][rest])), 1e-7)

    def test_arrhenius_speeds_up_both_directions(self):
        p = nm.generic_card(**dict(DEPIN, wk_q1=1e-3, wk_q2=1e-1), wk_ea=0.3)   # slow states: little ripple
        tau0, tau1 = nm.depin_taus(p), nm.depin_taus(p, 358.15)
        ratio = np.exp(0.3 / nm.K_B_EV * (1 / 298.15 - 1 / 358.15))
        np.testing.assert_allclose(tau0 / tau1, ratio, rtol=1e-12)
        # the plateau does not depend on temperature (same activation energy for depinning and re-pinning)
        t, v = one_period(3.0, 5e-6, 2e-5)
        z = [nm.cycle_history(p, t, v, [1e6], nd=8, sub=40, temp=tk)["Z"][0] for tk in (298.15, 358.15)]
        self.assertAlmostEqual(z[0], z[1], delta=2e-3)

    def test_every_loop_absorbs_energy(self):
        p = nm.generic_card(**DEPIN, wk_fp=0.2, wk_vsp=0.3, wk_nw=3.0)
        tri_t, tri_v = [0.0], [0.0]
        for a in [3.0, -3.0] * 4 + [0.0]:
            tri_t.append(tri_t[-1] + abs(a - tri_v[-1]) / 3.0 * 2.5e-6)
            tri_v.append(a)
        vf, br = nm.pwl(tri_t, tri_v)
        r = nm.simulate(p, vf, tri_t[-1], breaks=br, rtol=1e-7, atol=1e-10, max_step=2e-8)
        period = 1e-5
        for k in range(4):
            m = (r["t"] >= k * period) & (r["t"] <= (k + 1) * period)
            vv, qq = r["v"][m], r["qd"][m]
            work = float(np.sum(0.5 * (vv[1:] + vv[:-1]) * np.diff(qq)))   # closed-loop integral of V dQ
            self.assertGreater(work, 0.0)


class TestFatigueLagAndLeakage(unittest.TestCase):
    def test_lagged_stress_matches_the_closed_form(self):
        """Square pulse after a long 0 V gap, n_F = 2: int E E_L dt / E^2 = w - t_L (1 - e^-w/t_L); nothing at 0 V."""
        tl, w, amp = 2e-6, 7e-6, 3.0
        p = nm.generic_card(fat_d=0.3, fat_tf=1.0, fat_nf=2.0, fat_tl=tl)
        t = [0.0, 1e-12, w, w + 1e-12, 2e-3]
        v = [0.0, amp, amp, 0.0, 0.0]
        got = nm.lagged_stress_per_period(p, t, v)
        e = amp / p["tfe"] / nm.E_REF
        want = e**2 * (w - tl * (1 - np.exp(-w / tl)))
        self.assertAlmostEqual(got / want, 1.0, delta=2e-3)
        # without the lag the stress is the plain field-time integral
        p0 = nm.generic_card(fat_d=0.3, fat_tf=1.0, fat_nf=2.0)
        self.assertAlmostEqual(nm.stress_per_period(p0, t, v) / (e**2 * w), 1.0, delta=1e-3)

    def test_lag_penalises_short_pulses(self):
        """Stress per unit time under field falls for pulses comparable to t_L (Lenox-type dead time)."""
        p = nm.generic_card(fat_d=0.3, fat_tf=1.0, fat_nf=2.0, fat_tl=2e-6)
        per = []
        for w in (2e-6, 5e-6, 20e-6, 50e-6):
            t, v = one_period(2.0, w, 1e-4)
            per.append(nm.stress_per_period(p, t, v) / (2 * w))
        self.assertTrue(np.all(np.diff(per) > 0))
        self.assertLess(per[0] / per[-1], 0.6)

    def test_simulate_and_cycle_history_agree_with_lag(self):
        p = nm.generic_card(fat_d=0.4, fat_tf=3e-5, fat_nf=2.0, fat_kf=0.7, fat_tl=2e-6, fat_lk=5.0)
        period, n = 2e-5, 5
        t, v = pulse_train(n, 3.0, 4e-6, period)
        vf, br = nm.pwl(t, v)
        r = nm.simulate(p, vf, t[-1], breaks=br, rtol=1e-7, atol=1e-10, max_step=1e-7)
        tp, vp = one_period(3.0, 4e-6, period)
        ch = nm.cycle_history(p, tp, vp, np.arange(n + 1), nd=16, sub=200)
        a_sim = np.interp(np.arange(n + 1) * period, r["t"], r["a_f"])
        # the cycle average uses the periodic lag; the first period starts from V_L = 0
        np.testing.assert_allclose(ch["a_f"][1:], a_sim[1:], rtol=0.03)

    def test_leakage_rises_with_the_fatigue_age(self):
        p = nm.generic_card(fat_d=0.2, fat_tf=1e-4, fat_lk=3.0, fat_lm=1.5, g_oh=10.0)
        vf, br = nm.pwl([0.0, 1e-6, 2e-4], [0.0, 2.0, 2.0])
        r = nm.simulate(p, vf, 2e-4, breaks=br, rtol=1e-7)
        ratio = r["i_leak"][1:] / (p["area"] * nm.j_leak(p, r["v"][1:]))
        np.testing.assert_allclose(ratio, 1.0 + 3.0 * r["a_f"][1:] ** 1.5, rtol=1e-12)
        self.assertTrue(np.all(np.diff(r["a_f"]) >= 0))


class TestLoopReadout(unittest.TestCase):
    def test_loop_matches_a_direct_simulation(self):
        p = nm.generic_card(tfe=7e-9, c_d=32 * nm.EPS0 / 7e-9)
        amp, f = 1.61, 1e4
        L = nm.loop_figures(p, amp, f, nd=16, npts=800)
        T = 1 / f
        tt, vv = [0.0], [0.0]
        for a in [amp, -amp, amp, -amp, 0.0]:
            tt.append(tt[-1] + abs(a - vv[-1]) / amp * T / 4)
            vv.append(a)
        vf, br = nm.pwl(tt, vv)
        r = nm.simulate(p, vf, tt[-1], s0=-1.0, nd=16, breaks=br, rtol=1e-8, atol=1e-11, max_step=T / 2000)
        m = r["t"] >= T
        t2, v2, q2 = r["t"][m], r["v"][m], r["qd"][m]
        q2 = q2 - 0.5 * (q2.max() + q2.min())
        rise = (t2 > T + 0.75 * T) | (t2 < T + 0.25 * T)
        order = np.argsort(np.where(t2 < T + 0.25 * T, t2 + T, t2))
        tr = order[rise[order]]
        k = np.nonzero(np.diff(np.sign(q2[tr])) > 0)[0][0]
        vc = v2[tr][k] - q2[tr][k] * (v2[tr][k + 1] - v2[tr][k]) / (q2[tr][k + 1] - q2[tr][k])
        self.assertAlmostEqual(L["vc_p"], vc, delta=0.02)

    def test_coercive_width_grows_as_the_pinched_loop_opens(self):
        """Split populations +-V_sp: the zero crossing of the total charge moves out as the split is removed."""
        p = nm.generic_card(tfe=7e-9, c_d=32 * nm.EPS0 / 7e-9, mul_ea=0.6, wk_vsp=0.6, wk_nw=1.0, wk_kw=1.0)
        width, pr = [], []
        for nw in (0.0, 0.3, 1.0, 3.0, 30.0):
            L = nm.loop_figures(p, 1.61, 1e4, nd=16, history=nm.History(nw=nw))
            width.append(L["vc_p"] - L["vc_m"])
            pr.append(L["pr_p"] - L["pr_m"])
        self.assertTrue(np.all(np.diff(width) > 0), width)
        self.assertTrue(np.all(np.diff(pr) > 0), pr)
        self.assertGreater(width[-1] / width[0], 1.3)

    def test_without_dielectric_charge_the_trend_is_lost(self):
        """The rise of 2E_c comes from the zero crossing of the total charge (P + dielectric)."""
        p = nm.generic_card(tfe=7e-9, c_d=0.0, dc=0.0, mul_ea=0.6, wk_vsp=0.6, wk_nw=1.0, wk_kw=1.0)
        w = [nm.loop_figures(p, 1.61, 1e4, nd=16, history=nm.History(nw=nw)) for nw in (0.0, 30.0)]
        pd = nm.generic_card(tfe=7e-9, c_d=32 * nm.EPS0 / 7e-9, mul_ea=0.6, wk_vsp=0.6, wk_nw=1.0, wk_kw=1.0)
        wd = [nm.loop_figures(pd, 1.61, 1e4, nd=16, history=nm.History(nw=nw)) for nw in (0.0, 30.0)]
        grow_bare = (w[1]["vc_p"] - w[1]["vc_m"]) / (w[0]["vc_p"] - w[0]["vc_m"])
        grow_diel = (wd[1]["vc_p"] - wd[1]["vc_m"]) / (wd[0]["vc_p"] - wd[0]["vc_m"])
        self.assertGreater(grow_diel, grow_bare)

    def test_leakage_inflates_the_apparent_remanence(self):
        p = nm.generic_card(tfe=7e-9, g_oh=300.0)
        a = nm.loop_figures(p, 1.61, 1e4, nd=16)
        b = nm.loop_figures(p, 1.61, 1e4, nd=16, leak=True)
        self.assertGreater(b["pr_p"] - b["pr_m"], a["pr_p"] - a["pr_m"])
        self.assertGreater(b["vc_p"] - b["vc_m"], a["vc_p"] - a["vc_m"])


if __name__ == "__main__":
    unittest.main()
