"""Run every example script and write the figures to docs/."""

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE.parent / "docs").mkdir(exist_ok=True)
for name in ["landau_energy", "lk_loops", "preisach_forc", "kinetics_switching", "pund", "leakage_compensation"]:
    print(f"--- {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
