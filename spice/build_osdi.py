"""Compile spice/fecap_nls.va into an OSDI shared library for ngspice.

Usage (from the repository root)::

    python spice/build_osdi.py              # -> spice/build/fecap_nls.osdi  (FE_ND = 16)
    python spice/build_osdi.py --nd 64      # -> spice/build/fecap_nls_nd64.osdi

The OpenVAF executable is taken from the ``OPENVAF`` environment variable, or
``openvaf-r`` / ``openvaf`` on ``PATH``.

On Linux and macOS OpenVAF links the library itself.  The Windows (MinGW/UCRT)
build of OpenVAF-Reloaded needs an external GNU-style linker; if none is found
it stops with "linker not found" but leaves the compiled object files behind.
In that case this script links them with LLVM lld when ``LLD`` points to an
``ld.lld`` executable and ``MINGW_LIBDIR`` points to a directory containing
``libucrtbase.a`` and ``libkernel32.a`` (any MinGW-w64 or LLVM-MinGW toolchain
has them).  ``CLANG_RT_BUILTINS`` may point to ``libclang_rt.builtins-x86_64.a``
(or ``libgcc.a``) to provide ``___chkstk_ms``.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "fecap_nls.va"


def find_openvaf() -> str | None:
    env = os.environ.get("OPENVAF")
    if env and Path(env).exists():
        return env
    for name in ("openvaf-r", "openvaf"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def link_with_lld(objects: list[str], out: Path) -> None:
    lld = os.environ.get("LLD") or shutil.which("ld.lld")
    libdir = os.environ.get("MINGW_LIBDIR")
    if not lld or not libdir:
        raise SystemExit(
            "OpenVAF could not link the library. Set LLD (ld.lld) and MINGW_LIBDIR "
            "(directory with libucrtbase.a, libkernel32.a) to link the object files manually."
        )
    cmd = [lld, "-m", "i386pep", "--shared", "--entry=", "-o", str(out), *objects,
           "--export-all-symbols", "-L", libdir, "-lucrtbase", "-lkernel32"]
    rt = os.environ.get("CLANG_RT_BUILTINS")
    if rt:
        cmd.append(rt)
    subprocess.run(cmd, check=True)


def build(nd: int = 16, out: Path | None = None, quiet: bool = False) -> Path:
    exe = find_openvaf()
    if exe is None:
        raise SystemExit("OpenVAF not found: set OPENVAF or put openvaf-r on PATH")
    out = out or HERE / "build" / ("fecap_nls.osdi" if nd == 16 else f"fecap_nls_nd{nd}.osdi")
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    for f in glob.glob(str(stem) + ".o*"):
        os.remove(f)
    cmd = [exe, "-D", f"FE_ND={nd}", "-o", str(out), str(SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not quiet:
        sys.stdout.write(r.stdout)
    if r.returncode == 0 and out.exists():
        return out
    objects = sorted(glob.glob(str(stem) + ".o*"))
    if "linker not found" in (r.stdout + r.stderr) and objects:
        link_with_lld(objects, out)
        for f in objects:
            os.remove(f)
        return out
    sys.stderr.write(r.stdout + r.stderr)
    raise SystemExit(f"OpenVAF failed (exit {r.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nd", type=int, default=16, help="number of domain classes FE_ND")
    ap.add_argument("-o", "--output", type=Path, default=None)
    a = ap.parse_args()
    print(build(a.nd, a.output))
