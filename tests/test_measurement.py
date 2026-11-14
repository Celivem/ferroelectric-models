import unittest

import numpy as np

from fe_models import measurement as ms
from fe_models.preisach import PreisachModel
from fe_models.units import EPS0

AREA = 1e-8        # m^2
THICK = 10e-9      # m
EPS_R = 30.0
VMAX = 3.0         # V  -> 3 MV/cm in 10 nm


def remanent_preisach():
    """Preisach model sitting in its negative remanent state."""
    m = PreisachModel(n_grid=60)
    m.reset(-1)
    m.apply([-m.E_max, 0.0])
    return m


def triangle(f, n=4001):
    t = np.linspace(0.0, 1.0 / f, n)
    return t, ms.triangle_wave(t, VMAX, f)


class TestWaveforms(unittest.TestCase):
    def test_triangle_wave_values(self):
        f = 1e3
        t = np.array([0, 0.25, 0.5, 0.75, 1.0, 0.125]) / f
        np.testing.assert_allclose(ms.triangle_wave(t, 2.0, f), [0, 2, 0, -2, 0, 1], atol=1e-12)
        self.assertIsInstance(ms.triangle_wave(0.1 / f, 2.0, f), float)

    def test_pund_waveform_structure(self):
        t, V, w = ms.pund_waveform(amplitude=2.0)
        self.assertEqual(list(w), ["preset", "P", "U", "N", "D"])
        self.assertTrue(np.all(np.diff(t) > 0))
        for name, sign in (("P", 1), ("U", 1), ("N", -1), ("D", -1)):
            a, top, b = w[name]
            mid = (t > a + 1.1e-6) & (t < top - 1e-9)
            np.testing.assert_allclose(V[mid], sign * 2.0)
            self.assertEqual(V[np.searchsorted(t, b)], 0.0)


class TestPUND(unittest.TestCase):
    def setUp(self):
        self.t, self.V, self.w = ms.pund_waveform(amplitude=VMAX)

    def current(self, polarization=None, leakage=None):
        return ms.capacitor_current(self.t, self.V, area=AREA, thickness=THICK, eps_r=EPS_R,
                                    polarization=polarization, leakage=leakage)

    def test_linear_capacitor_has_no_switching_charge(self):
        r = ms.analyze_pund(self.t, self.current(), self.w, area=AREA)
        scale = EPS0 * EPS_R * VMAX / THICK
        self.assertLess(abs(r["dP_pos"]), 1e-9 * scale)
        self.assertLess(abs(r["dP_neg"]), 1e-9 * scale)
        # but the non-switching (dielectric) charge is there
        self.assertAlmostEqual(r["nonswitching"] / scale, 1.0, places=2)

    def test_switching_charge_equals_twice_remanence(self):
        m = PreisachModel(n_grid=60)
        E, up, dn = m.major_loop()
        pr_model = np.interp(0.0, E, dn)
        for leak in (None, 5e3):
            r = ms.analyze_pund(self.t, self.current(PreisachModel(n_grid=60).apply, leak), self.w, area=AREA)
            self.assertAlmostEqual(r["Pr"] / pr_model, 1.0, places=4)
            self.assertAlmostEqual(r["dP_pos"] / (2 * pr_model), 1.0, places=4)
            self.assertAlmostEqual(r["dP_neg"] / (2 * pr_model), 1.0, places=4)

    def test_full_window_charge_is_remanent_change(self):
        m = PreisachModel(n_grid=60)
        r = ms.analyze_pund(self.t, self.current(m.apply), self.w, area=AREA)
        E, up, dn = PreisachModel(n_grid=60).major_loop()
        pr_model = np.interp(0.0, E, dn)
        # P pulse: -Pr -> +Pr ; U pulse: no net change
        self.assertAlmostEqual(r["Q_full"]["P"] / AREA / (2 * pr_model), 1.0, places=4)
        self.assertLess(abs(r["Q_full"]["U"] / AREA), 1e-4 * pr_model)


class TestLeakageCompensation(unittest.TestCase):
    def loop(self, f, leakage=None):
        t, V = triangle(f)
        I = ms.capacitor_current(t, V, area=AREA, thickness=THICK, eps_r=EPS_R,
                                 polarization=remanent_preisach().apply, leakage=leakage)
        return t, V, I

    def test_ohmic_compensation_recovers_loop(self):
        R = 1e6
        t, V, I0 = self.loop(1e3)
        _, _, I = self.loop(1e3, leakage=R)
        P_true = ms.polarization_from_current(t, I0, AREA)
        P_raw = ms.polarization_from_current(t, I, AREA)
        R_est = ms.estimate_leakage_resistance(t, V, I)
        self.assertLess(abs(R_est / R - 1), 0.01)
        P_comp = ms.polarization_from_current(t, ms.compensate_leakage_ohmic(V, I, R_est), AREA)
        pr = ms.extract_loop_parameters(V / THICK, P_true)["Pr"]
        self.assertGreater(np.max(np.abs(P_raw - P_true)), 0.05 * pr)   # leakage visibly distorts
        self.assertLess(np.max(np.abs(P_comp - P_true)), 0.01 * pr)

    def test_two_frequency_compensation_with_nonlinear_leakage(self):
        def leak(V):
            return 1e-7 * np.sinh(V / 0.8)

        f1, f2 = 1e3, 3e3
        t1, V1, I1 = self.loop(f1, leak)
        t2, V2, I2 = self.loop(f2, leak)
        np.testing.assert_allclose(V1, V2)
        t1_, _, I1_true = self.loop(f1)
        I1c, I2c, I_leak = ms.compensate_leakage_two_frequency(I1, f1, I2, f2)
        np.testing.assert_allclose(I_leak, leak(V1), rtol=1e-6, atol=1e-12)
        P_true = ms.polarization_from_current(t1, I1_true, AREA)
        P_comp = ms.polarization_from_current(t1, I1c, AREA)
        P_raw = ms.polarization_from_current(t1, I1, AREA)
        pr = ms.extract_loop_parameters(V1 / THICK, P_true)["Pr"]
        self.assertGreater(np.max(np.abs(P_raw - P_true)), 0.05 * pr)
        self.assertLess(np.max(np.abs(P_comp - P_true)), 1e-6 * pr)

    def test_linear_capacitor_loop(self):
        t, V = triangle(1e3)
        I = ms.capacitor_current(t, V, area=AREA, thickness=THICK, eps_r=EPS_R)
        P = ms.polarization_from_current(t, I, AREA)
        np.testing.assert_allclose(P, EPS0 * EPS_R * V / THICK, atol=1e-3 * EPS0 * EPS_R * VMAX / THICK)


class TestLoopParameters(unittest.TestCase):
    def test_shifted_loop(self):
        Ps, d = 0.25, 0.3e8
        ecp, ecm = 1.3e8, -0.7e8
        E_up = np.linspace(-4e8, 4e8, 4001)
        E = np.r_[E_up, E_up[::-1]]
        P = np.r_[Ps * np.tanh((E_up - ecp) / d), Ps * np.tanh((E_up[::-1] - ecm) / d)]
        r = ms.extract_loop_parameters(E, P)
        self.assertAlmostEqual(r["Ec_plus"] / ecp, 1.0, places=5)
        self.assertAlmostEqual(r["Ec_minus"] / ecm, 1.0, places=5)
        self.assertAlmostEqual(r["imprint"] / 0.3e8, 1.0, places=5)
        self.assertAlmostEqual(r["Ec"] / 1.0e8, 1.0, places=5)
        self.assertAlmostEqual(r["Pr_plus"] / (Ps * np.tanh(0.7e8 / d)), 1.0, places=5)
        self.assertAlmostEqual(r["Pr_minus"] / (-Ps * np.tanh(1.3e8 / d)), 1.0, places=5)
        # start the record in the middle of the cycle: result must not change
        r2 = ms.extract_loop_parameters(np.roll(E, 1234), np.roll(P, 1234))
        for key in r:
            self.assertAlmostEqual(r[key], r2[key], delta=1e-9 * max(1.0, abs(r[key])))


class TestCycling(unittest.TestCase):
    def test_default_form(self):
        p = dict(Pr0=0.15, dPr_wake=0.05, N_wake=1e3, N_fatigue=1e9, k=1.0)
        y = ms.cycling_pr([0, 1e5, 1e12], **p)
        self.assertAlmostEqual(y[0], 0.15, places=12)
        self.assertAlmostEqual(y[1], 0.20, places=3)
        self.assertLess(y[2], 0.01)

    def test_user_form(self):
        y = ms.cycling_pr([1, 10, 100], form=lambda N, a: a * np.log10(N), a=2.0)
        np.testing.assert_allclose(y, [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
