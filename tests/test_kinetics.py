import unittest

import numpy as np

from fe_models import kinetics as k


class TestKAI(unittest.TestCase):
    def test_value_at_t0(self):
        for n in (0.5, 1.0, 2.0, 3.0):
            self.assertAlmostEqual(float(k.kai_fraction(2e-7, 2e-7, n)), 1 - np.exp(-1), places=14)

    def test_limits_and_monotonic(self):
        t = np.logspace(-12, -3, 500)
        f = k.kai_fraction(t, 1e-7, 2.0)
        self.assertLess(f[0], 1e-9)
        self.assertGreater(f[-1], 1 - 1e-12)
        self.assertTrue(np.all(np.diff(f) >= 0))
        self.assertEqual(float(k.kai_fraction(0.0, 1e-7, 2.0)), 0.0)


class TestNLS(unittest.TestCase):
    t = np.logspace(-10, -3, 300)

    def test_zero_width_is_kai(self):
        for kind in ("lorentzian", "lognormal"):
            np.testing.assert_array_equal(k.nls_fraction(self.t, -6.0, 0.0, n=2.0, kind=kind),
                                          k.kai_fraction(self.t, 1e-6, 2.0))

    def test_narrow_distribution_reduces_to_kai(self):
        for kind in ("lorentzian", "lognormal"):
            for n in (1.0, 2.0):
                f = k.nls_fraction(self.t, -6.0, 1e-3, n=n, kind=kind)
                self.assertLess(np.max(np.abs(f - k.kai_fraction(self.t, 1e-6, n))), 2e-3)

    def test_broad_distribution_differs_from_kai(self):
        f = k.nls_fraction(self.t, -6.0, 1.0, n=2.0, kind="lorentzian")
        kai = k.kai_fraction(self.t, 1e-6, 2.0)
        self.assertGreater(np.max(np.abs(f - kai)), 0.1)
        # stretched: switching starts earlier and ends later than KAI
        i_early = np.searchsorted(self.t, 1e-8)
        i_late = np.searchsorted(self.t, 1e-4)
        self.assertGreater(f[i_early], kai[i_early])
        self.assertLess(f[i_late], kai[i_late])

    def test_bounds(self):
        f = k.nls_fraction(np.logspace(-20, 10, 200), -6.0, 0.5, kind="lognormal")
        self.assertLess(f[0], 1e-6)
        self.assertGreater(f[-1], 1 - 1e-6)
        self.assertTrue(np.all(np.diff(f) >= -1e-15))


class TestMerz(unittest.TestCase):
    def test_decreasing_with_field(self):
        E = np.linspace(0.5e8, 4e8, 50)
        t = k.merz_time(E, 1e-9, 8e8, 1.0)
        self.assertTrue(np.all(np.diff(t) < 0))
        self.assertAlmostEqual(float(k.merz_time(8e8, 1e-9, 8e8, 2.0)) / (1e-9 * np.e), 1.0, places=12)

    def test_field_dependent_nls_uses_merz_centre(self):
        t = np.logspace(-10, -3, 200)
        E = 1.5e8
        u1 = np.log10(k.merz_time(E, 1e-9, 8e8, 1.0))
        np.testing.assert_allclose(k.nls_fraction_field(t, E, 1e-9, 8e8, 1.0, width=0.4),
                                   k.nls_fraction(t, u1, 0.4), atol=1e-14)


class TestFits(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.t = np.logspace(-9, -4, 80)

    def test_fit_kai_recovers_parameters(self):
        f = k.kai_fraction(self.t, 3e-7, 1.7) + self.rng.normal(0, 0.01, self.t.size)
        r = k.fit_kai(self.t, f)
        self.assertLess(abs(r["t0"] / 3e-7 - 1), 0.03)
        self.assertLess(abs(r["n"] - 1.7), 0.1)

    def test_fit_nls_recovers_parameters(self):
        for kind in ("lorentzian", "lognormal"):
            f = k.nls_fraction(self.t, -6.3, 0.6, n=2.0, kind=kind) + self.rng.normal(0, 0.01, self.t.size)
            r = k.fit_nls(self.t, f, n=2.0, kind=kind)
            self.assertLess(abs(r["log10_t1"] + 6.3), 0.05, kind)
            self.assertLess(abs(r["width"] - 0.6), 0.06, kind)

    def test_fit_merz_recovers_parameters(self):
        E = np.linspace(0.8e8, 3.5e8, 15)
        t0 = k.merz_time(E, 1e-9, 8e8, 1.3) * np.exp(self.rng.normal(0, 0.02, E.size))
        r = k.fit_merz(E, t0)
        self.assertLess(abs(r["Ea"] / 8e8 - 1), 0.1)
        self.assertLess(abs(r["m"] - 1.3), 0.15)
        self.assertLess(abs(np.log10(r["t_inf"] / 1e-9)), 0.3)


if __name__ == "__main__":
    unittest.main()
