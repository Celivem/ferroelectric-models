"""Run every example script and write the figures to docs/.

spice_figures.py needs ngspice and the compiled Verilog-A model; it is skipped
when ngspice is not found (see spice/README.md).
"""

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE.parent / "docs").mkdir(exist_ok=True)
for name in ["landau_energy", "lk_loops", "preisach_forc", "kinetics_switching", "pund", "leakage_compensation",
             "literature_comparison"]:
    print(f"--- {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")

sys.path.insert(0, str(HERE.parent / "spice"))
import compare_ngspice  # noqa: E402

if compare_ngspice.find_ngspice():
    print("--- spice_figures")
    runpy.run_path(str(HERE / "spice_figures.py"), run_name="__main__")
else:
    print("--- spice_figures skipped (ngspice not found; set NGSPICE)")
