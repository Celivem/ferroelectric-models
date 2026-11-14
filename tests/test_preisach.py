import unittest

import numpy as np

from fe_models.measurement import extract_loop_parameters
from fe_models.preisach import PreisachModel, forc_distribution, simulate_forcs

PS = 0.20
EC = 1.0e8


class TestPreisachLoops(unittest.TestCase):
    def setUp(self):
        self.m = PreisachModel(Ps=PS, Ec=EC, n_grid=80)

    def test_saturates_at_ps(self):
        m = self.m.copy().reset(-1)
        self.assertAlmostEqual(m.polarization, -PS, places=12)
        self.assertAlmostEqual(m.apply(m.E_max), PS, places=12)
        self.assertAlmostEqual(m.apply(2 * m.E_max), PS, places=12)
        self.assertAlmostEqual(m.apply(-m.E_max), -PS, places=12)

    def test_weights_normalized_for_both_shapes(self):
        for kind in ("gaussian", "lorentzian"):
            m = PreisachModel(distribution=kind, n_grid=40)
            self.assertAlmostEqual(m.weights.sum(), 1.0, places=12)
            self.assertTrue(np.all(m.weights > 0))
        with self.assertRaises(ValueError):
            PreisachModel(distribution="uniform", n_grid=10)

    def test_unbiased_loop_is_symmetric(self):
        E, up, dn = self.m.major_loop()
        np.testing.assert_allclose(up, -dn[::-1], atol=1e-12)

    def test_bias_shifts_loop(self):
        bias = 0.3e8
        m = PreisachModel(Ps=PS, Ec=EC, bias=bias, n_grid=80)
        E, up, dn = m.major_loop()
        self.assertGreater(np.max(np.abs(up + dn[::-1])), 0.05 * PS)  # no longer symmetric
        r = extract_loop_parameters(np.r_[E, E[::-1]], np.r_[up, dn[::-1]])
        self.assertLess(abs(r["imprint"] - bias), m.cell)
        self.assertLess(abs(r["Ec"] - EC), m.cell)

    def test_rate_independence(self):
        a = self.m.copy().reset(-1).apply(np.linspace(-3e8, 1.1e8, 7))[-1]
        b = self.m.copy().reset(-1).apply(np.linspace(-3e8, 1.1e8, 7001))[-1]
        self.assertAlmostEqual(a, b, places=12)

    def test_minor_loops_inside_major(self):
        E, up, dn = self.m.major_loop()
        Er, Ef, P = simulate_forcs(self.m, E_reversal=E[::4])
        for i in range(Er.size):
            ok = ~np.isnan(P[i])
            lo = np.interp(Ef[ok], E, up)
            hi = np.interp(Ef[ok], E, dn)
            self.assertTrue(np.all(P[i, ok] >= lo - 1e-12))
            self.assertTrue(np.all(P[i, ok] <= hi + 1e-12))
        # at least one FORC lies strictly inside somewhere
        mid = Er.size // 2
        ok = ~np.isnan(P[mid])
        self.assertGreater(np.max(P[mid, ok] - np.interp(Ef[ok], E, up)), 0.05 * PS)

    def test_return_point_memory(self):
        m = self.m.copy().reset(1)
        E1, E2 = -0.9e8, 0.5e8
        P1 = m.apply(np.linspace(3e8, E1, 50))[-1]
        s1 = m.s.copy()
        m.apply(np.linspace(E1, E2, 50))
        P_back = m.apply(np.linspace(E2, E1, 50))[-1]
        self.assertAlmostEqual(P_back, P1, places=12)
        np.testing.assert_array_equal(m.s, s1)
        # the inner excursion changed the state, so the test is not trivial
        m2 = self.m.copy().reset(1)
        m2.apply(np.linspace(3e8, E1, 50))
        self.assertGreater(m2.apply(E2) - P1, 0.01 * PS)

    def test_wiping_out(self):
        # A larger excursion erases the memory of a smaller one.
        path_a = np.r_[np.linspace(3e8, -0.8e8, 40), np.linspace(-0.8e8, 0.4e8, 40), np.linspace(0.4e8, -1.2e8, 40)]
        path_b = np.linspace(3e8, -1.2e8, 40)
        ma = self.m.copy().reset(1)
        mb = self.m.copy().reset(1)
        ma.apply(path_a)
        mb.apply(path_b)
        np.testing.assert_array_equal(ma.s, mb.s)


class TestFORC(unittest.TestCase):
    def test_forc_recovers_distribution(self):
        bias = 0.2e8
        m = PreisachModel(Ps=PS, Ec=EC, sigma_c=0.25e8, sigma_b=0.15e8, bias=bias, n_grid=80)
        Er, E, P = simulate_forcs(m)
        rho = forc_distribution(Er, E, P)
        dE = E[1] - E[0]
        self.assertAlmostEqual(np.nansum(rho) * dE * dE / PS, 1.0, places=2)
        i, j = np.unravel_index(np.nanargmax(rho), rho.shape)
        ec_peak = 0.5 * (E[j] - Er[i])
        bias_peak = 0.5 * (E[j] + Er[i])
        self.assertLess(abs(ec_peak - EC), 1.5 * dE)
        self.assertLess(abs(bias_peak - bias), 1.5 * dE)

    def test_forc_matches_hysteron_weights(self):
        # rho(Er, E) should equal Ps * density at (alpha=E, beta=Er).
        m = PreisachModel(Ps=PS, n_grid=60)
        Er, E, P = simulate_forcs(m)
        rho = forc_distribution(Er, E, P)
        centers, W = m.distribution_grid()
        dE = E[1] - E[0]
        # value of rho at the cell centres (average of the 4 surrounding nodes)
        rc = 0.25 * (rho[:-1, :-1] + rho[1:, :-1] + rho[:-1, 1:] + rho[1:, 1:])  # [beta_idx, alpha_idx]
        target = PS * W.T / dE**2
        mask = np.isfinite(rc) & np.isfinite(target)
        peak = np.nanmax(target)
        err = np.max(np.abs(rc[mask] - target[mask])) / peak
        self.assertLess(err, 0.15)


if __name__ == "__main__":
    unittest.main()
