import unittest

import numpy as np

from fe_models import landau
from fe_models.measurement import extract_loop_parameters

PR = 0.20   # C/m^2  (20 uC/cm^2)
EC = 1.0e8  # V/m    (1 MV/cm)


def first_order_coefficients(P0=0.2, E0=1e8):
    """2-4-6 model with alpha > 0, beta < 0 (first-order, single loop).

    In reduced units E/E0 = 2p - 8p^3 + 6p^5 with p = P/P0, so Pr = P0
    (6x^2 - 8x + 2 = 0 -> x = 1) and P = 0 is also metastable.
    """
    return 1.0 * E0 / P0, -2.0 * E0 / P0**3, 1.0 * E0 / P0**5


class TestStaticLandau(unittest.TestCase):
    def test_helper_reproduces_pr_and_ec(self):
        a, b = landau.coefficients_from_pr_ec(PR, EC)
        self.assertLess(a, 0)
        self.assertGreater(b, 0)
        # independent numerical routes: polynomial roots for Pr, dE/dP = 0 for Ec
        self.assertAlmostEqual(landau.remanent_polarization(a, b) / PR, 1.0, places=12)
        self.assertAlmostEqual(landau.coercive_field(a, b) / EC, 1.0, places=12)

    def test_coercive_field_matches_brute_force_maximum(self):
        # Ec is the largest field the negative-P branch can sustain:
        # max over P < 0 of E(P) on the stable-to-unstable turning point.
        cases = [landau.coefficients_from_pr_ec(PR, EC) + (0.0,), first_order_coefficients()]
        for a, b, g in cases:
            P = np.linspace(-0.5, 0.0, 200001)
            E = landau.field_from_polarization(P, a, b, g)
            # local maxima of E(P) on P < 0
            i = np.where((E[1:-1] > E[:-2]) & (E[1:-1] > E[2:]))[0] + 1
            brute = E[i].max()
            self.assertAlmostEqual(landau.coercive_field(a, b, g) / brute, 1.0, places=6)

    def test_first_order_remanence(self):
        a, b, g = first_order_coefficients()
        self.assertAlmostEqual(landau.remanent_polarization(a, b, g), 0.2, places=12)

    def test_field_is_derivative_of_free_energy(self):
        a, b = landau.coefficients_from_pr_ec(PR, EC)
        g = 1e10
        P = np.linspace(-0.3, 0.3, 20001)
        num = np.gradient(landau.free_energy(P, a, b, g), P)
        ana = landau.field_from_polarization(P, a, b, g)
        np.testing.assert_allclose(num[1:-1], ana[1:-1], rtol=0, atol=1e-5 * np.abs(ana).max())

    def test_free_energy_minimum_at_pr(self):
        a, b = landau.coefficients_from_pr_ec(PR, EC)
        P = np.linspace(0.0, 0.4, 400001)
        self.assertAlmostEqual(P[np.argmin(landau.free_energy(P, a, b))], PR, places=5)

    def test_negative_capacitance_branch(self):
        a, b = landau.coefficients_from_pr_ec(PR, EC)
        P, E, stable = landau.s_curve(a, b, n=4001)
        inner = np.abs(P) < PR / np.sqrt(3)
        outer = np.abs(P) > PR / np.sqrt(3) * (1 + 1e-9)
        self.assertTrue(np.all(~stable[inner]))
        self.assertTrue(np.all(stable[outer]))
        # on the NC branch dE/dP < 0
        self.assertTrue(np.all(np.diff(E[inner]) < 0))

    def test_static_loop_matches_helper(self):
        a, b = landau.coefficients_from_pr_ec(PR, EC)
        n = 2001
        E, P = landau.static_loop(2 * EC, a, b, n=n)
        step = 4 * EC / (n - 1)
        r = extract_loop_parameters(E, P)
        self.assertAlmostEqual(r["Pr"] / PR, 1.0, places=9)
        self.assertLess(abs(r["Ec_plus"] - EC), step)
        self.assertLess(abs(r["Ec_minus"] + EC), step)
        self.assertLess(abs(r["imprint"]), 1e-6 * EC)


class TestLandauKhalatnikov(unittest.TestCase):
    def setUp(self):
        self.a, self.b = landau.coefficients_from_pr_ec(PR, EC)
        self.tau = 1e-8
        self.rho = landau.rho_from_time_constant(self.tau, self.a, self.b)

    def loop_ec(self, f, n_points=4000):
        _, E, P = landau.lk_loop(f, 2 * EC, self.a, self.b, self.rho, n_points=n_points)
        return extract_loop_parameters(E, P)["Ec"]

    def test_time_constant_roundtrip(self):
        self.assertAlmostEqual(landau.lk_time_constant(self.a, self.b, self.rho) / self.tau, 1.0, places=12)

    def test_relaxation_to_remanent_state(self):
        t = np.linspace(0, 50 * self.tau, 501)
        P = landau.lk_simulate(t, lambda tt: 0.0, self.a, self.b, self.rho, P0=0.3 * PR)
        self.assertAlmostEqual(P[-1] / PR, 1.0, places=6)
        P = landau.lk_simulate(t, lambda tt: 0.0, self.a, self.b, self.rho, P0=-0.01 * PR)
        self.assertAlmostEqual(P[-1] / PR, -1.0, places=6)

    def test_slow_sweep_approaches_static_ec(self):
        ec = self.loop_ec(f=1e3)  # f * tau = 1e-5
        self.assertGreater(ec, EC * (1 - 0.005))
        self.assertLess(ec, EC * 1.02)

    def test_fast_sweep_increases_ec(self):
        ecs = [self.loop_ec(f, n_points=2000) for f in (1e3, 1e5, 1e6)]
        self.assertTrue(ecs[0] < ecs[1] < ecs[2])
        self.assertGreater(ecs[2], 1.3 * EC)


if __name__ == "__main__":
    unittest.main()
