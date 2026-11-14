"""The device card spice/params/mukherjee2023.inc: it reproduces what it was fitted to, in Python and ngspice.

The ngspice part runs only when ngspice and the compiled model are available
(same rules as tests/test_spice_ngspice.py); otherwise it is skipped.
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from fe_models import nls_multidomain as nm
from fe_models import read_disturb as rd

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "spice" / "params" / "mukherjee2023.inc"
sys.path.insert(0, str(ROOT / "spice"))
EPS0, MV, T9 = nm.EPS0, 1e8, 9e-9
V_PEAK, PERIOD = 2.7, 50e-6


def load_card():
    """Python parameter dict from the .model lines of the card (generic values for the rest)."""
    by_va = {prm.va_name: k for k, prm in nm.GENERIC.items()}
    vals = {}
    for line in CARD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\+\s*(\w+)=(\S+)", line)
        if m:
            vals[by_va[m.group(1)]] = float(m.group(2))
    area = float(re.search(r"area = (\S+) m\^2", CARD.read_text(encoding="utf-8")).group(1))
    return nm.generic_card(area=area, **vals)


def eps(p, e_mvcm, x):
    v = e_mvcm * MV * p["tfe"]
    return (nm.q_rev(p, v + 1e-4, x) - nm.q_rev(p, v - 1e-4, x)) / 2e-4 * p["tfe"] / EPS0


class TestCardInPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load_card()

    def test_loop_peaks(self):
        self.assertAlmostEqual(rd.loop_peak(self.p, V_PEAK, PERIOD, -1, nd=128, nstep=3000, total=False), -0.54,
                               delta=0.01)
        self.assertAlmostEqual(rd.loop_peak(self.p, V_PEAK, PERIOD, +1, nd=128, nstep=3000, total=False), 1.35,
                               delta=0.01)

    def test_cv_at_zero_field_and_crossing(self):
        # Fig. 5d means eps+ = 39.9, eps- = 31.2; crossing near 0.87 MV/cm at eps = 36 (fit RMS about 1.5)
        self.assertAlmostEqual(eps(self.p, 0.0, 1.0), 39.9, delta=2.0)
        self.assertAlmostEqual(eps(self.p, 0.0, -1.0), 31.2, delta=2.0)
        self.assertAlmostEqual(eps(self.p, 0.87, 1.0), eps(self.p, 0.87, -1.0), delta=0.05)

    def test_reversible_window_at_minus_100_mV(self):
        # C-V fit only: |Q(+) - Q(-)| without switching, against read 1 of Fig. 7 (4.3 pC)
        cmw = abs(nm.q_rev(self.p, -0.1, 1.0) - nm.q_rev(self.p, -0.1, -1.0)) * self.p["area"] * 1e12
        self.assertAlmostEqual(cmw, 4.3, delta=0.6)

    def test_reads_need_the_stored_state_shift(self):
        # documented limitation (docs/research/tail-fit.md): without a stabilizing shift one -0.5 V read
        # switches about a third of the film; with the fitted 0.41 V shift less than 0.5 %
        f0 = rd.read_sequence(self.p, -0.5, [1], nd=128)["f_plus"][0]
        f1 = rd.read_sequence(self.p, -0.5, [1], nd=128, stored_shift=0.41)["f_plus"][0]
        self.assertGreater(f0, 0.2)
        self.assertLess(f1, 5e-3)


class TestCardInNgspice(unittest.TestCase):
    """A 50 us triangular loop with the card in ngspice against the Python reference (16 classes)."""

    @classmethod
    def setUpClass(cls):
        try:
            import compare_ngspice as cn
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"compare_ngspice not importable: {e}")
        ng = cn.find_ngspice()
        if ng is None:
            raise unittest.SkipTest("ngspice not found (set NGSPICE)")
        try:
            cn.ensure_osdi()
        except SystemExit as e:
            raise unittest.SkipTest(f"model library not available: {e}")
        cls.p = load_card()
        t = [0.0, 0.25 * PERIOD, 0.75 * PERIOD]
        v = [0.0, V_PEAK, -V_PEAK]
        out = cn.OUT
        out.mkdir(parents=True, exist_ok=True)
        deck = out / "card_mukherjee2023.cir"
        deck.write_text("\n".join([
            "* 50 us loop with the Mukherjee 2023 card",
            f".include {Path(os.path.relpath(CARD, out)).as_posix()}",  # relative: ngspice splits paths at spaces
            ".options method=gear reltol=1e-4",
            f"N1 p1 0 q1 fecap_mukherjee2023 s0=-1 area={cls.p['area']:g}",
            f"V1 p1 0 {cn.pwl_text(t, v)}",
            ".control", f"pre_osdi {Path(os.path.relpath(cn.OSDI, out)).as_posix()}", "set wr_singlescale", "set wr_vecnames", "set numdgt=15",
            "tran 2e-9 3.75e-05", "wrdata card_mukherjee2023.dat v(p1) v(q1)", ".endc", ".end", ""]), encoding="utf-8")
        r = subprocess.run([ng, "-b", str(deck)], cwd=out, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        d = cn.read_dat(out / "card_mukherjee2023.dat")
        cls.ng = (d["v(p1)"], d["v(q1)"] * 1e-2)  # monitor output in uC/cm^2 -> C/m^2
        vf, br = nm.pwl(t, v)
        te = np.linspace(0.25 * PERIOD, 0.75 * PERIOD, 4001)
        py = nm.simulate(cls.p, vf, 0.75 * PERIOD, s0=-1.0, nd=16, breaks=br, t_eval=te, rtol=1e-8)
        m = py["t"] >= 0.25 * PERIOD
        cls.py = (py["v"][m], py["qd"][m])

    @staticmethod
    def peak(v, q):
        """Voltage of the largest -dq/dV on the falling branch (+ -> -), smoothed by 30 mV."""
        o = np.argsort(v)
        v, q = v[o], q[o]
        vg = np.linspace(-1.5, 0.3, 1801)
        qg = np.interp(vg, v, q)
        j = np.gradient(qg, vg)
        ker = np.exp(-0.5 * (np.arange(-120, 121) * (vg[1] - vg[0]) / 0.03) ** 2)
        j = np.convolve(j, ker / ker.sum(), mode="same")
        k = int(np.argmax(j[150:-150])) + 150
        return float(vg[k])

    def test_negative_peak_matches_python(self):
        vn, qn = self.ng
        sel = np.arange(len(vn)) > np.argmax(vn)
        v_ng = self.peak(vn[sel], qn[sel])
        v_py = self.peak(*self.py)
        self.assertAlmostEqual(v_ng, v_py, delta=0.01)
        # 16 classes instead of 128: the peak of the total current stays within 60 mV of the fitted -0.54 V
        self.assertAlmostEqual(v_ng, -0.54, delta=0.06)

    def test_charge_after_the_sweep_matches_python(self):
        vn, qn = self.ng
        self.assertAlmostEqual(float(qn[-1]), float(self.py[1][-1]), delta=2e-3 * self.p["ps"])


if __name__ == "__main__":
    unittest.main()
