"""Compiled Verilog-A model (OpenVAF + ngspice) against the Python reference.

Runs only when ngspice is available (``NGSPICE`` or ``ngspice_con`` / ``ngspice``
on PATH) and the model library exists or can be built (``FECAP_OSDI``, an
up-to-date ``spice/build/fecap_nls.osdi``, or ``OPENVAF``; see
``spice/build_osdi.py``).  Otherwise the test is skipped.  It runs all decks of
``spice/compare_ngspice.py`` (about one minute).
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spice"))

import compare_ngspice as cn  # noqa: E402
import build_osdi  # noqa: E402


def _prerequisites():
    if cn.find_ngspice() is None:
        return "ngspice not found (set NGSPICE)"
    try:
        cn.ensure_osdi()
    except SystemExit as e:  # no OpenVAF, or it could not link
        return f"model library not available: {e}"
    return None


class TestNgspiceAgainstPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = _prerequisites() if build_osdi else "build helper missing"
        if reason:
            raise unittest.SkipTest(reason)
        cls.results = cn.run(verbose=False)

    def test_all_checks_within_tolerance(self):
        bad = [(r["name"], c["quantity"], c["difference"], c["tolerance"])
               for r in self.results for c in r["checks"] if not c["ok"]]
        self.assertEqual(bad, [])

    def test_no_singular_matrix_or_timestep_warnings(self):
        self.assertEqual([(r["name"], r["warnings"]) for r in self.results if r["warnings"]], [])

    def test_every_deck_ran(self):
        self.assertEqual(sorted(r["name"] for r in self.results), sorted(d.name for d in cn.all_decks()))


if __name__ == "__main__":
    unittest.main()
