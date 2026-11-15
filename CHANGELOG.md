# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0.0 the
Python API and the Verilog-A parameter names may still change between minor
versions; every such change is listed below.

## [Unreleased]

## [0.3.0] - 2026-11-15

### Added

- Optional cycling and storage history in both the Python reference and the
  Verilog-A model, all off by default (Section 4 of the derivation article):
  - E13 pinned pool: pinned domains keep their polarization and reversible
    charge, so fatigue does not appear as a retention loss;
  - E14 wake-up driven by switching, with a pristine pinned fraction and
    bias split;
  - E15/E16 fatigue with a field-time and a switching driver;
  - E17 logarithmic-time imprint from 12 trapping states, with E17a branch
    asymmetry and E17b field acceleration;
  - E18 reversible depinning pool (pulse-width dependence of wake-up);
  - E19 build-up time of the fatigue stress and E20 leakage growth with
    fatigue.
- `cycle_history` (cycle-averaged history, 10¹² cycles in about 0.1 s),
  `loop_figures` (reads Pr and Ec from the total charge, as an experiment
  does) and `aged_card`.
- Verilog-A instance parameters `nw0`, `af0`, `si0`, `zq0` to start a circuit
  from a prior history; compile-time macros `FE_NI` and `FE_NQ`.
- `fe_models.read_disturb`: read sequences up to 10¹¹ reads, loop calibration
  and dynamic imprint between write and read.
- Device card `spice/params/mukherjee2023.inc` (C–V and loop fit of a
  published TiN/MoOx/La:HZO/Nb₂O₅/TiN capacitor).
- Four research notes in English and Traditional Chinese in `docs/research/`
  (read disturb, wake-up and fatigue, imprint) and fifteen new digitized
  datasets in `data/literature/`.
- Seven ngspice testbenches for the history terms (19 in total, 92 checks)
  and 74 new unit tests (153 in total).
- `CITATION.cff` and this changelog.

### Fixed

- `step()`: a reversed drive below the reset threshold no longer reuses the
  old direction's progress `h` (switched 7.8 % where the ODE switches nothing
  in the reproduction case).

### Changed

- Math in all Markdown files uses GitHub's robust syntax (inline math
  delimited by a dollar sign and a backtick, display math in `math` code
  blocks); inline math next to CJK punctuation and escapes such as `\;`
  inside double-dollar blocks did not render before.
- Version 0.3.0 in `pyproject.toml` and `fe_models.__version__`; project URLs
  added to `pyproject.toml`; README sections on citing and licensing.

## [0.2.0] - 2026-11-15

### Added

- Verilog-A model `spice/fecap_nls.va` of the dynamic multi-domain NLS
  capacitor, compiled with OpenVAF-Reloaded 24.0.2 and verified in ngspice 47
  against the Python reference (12 testbenches).
- Python reference `fe_models.nls_multidomain`, generic parameter card with
  the source of every value, build and comparison scripts.
- Derivation article in English and Traditional Chinese.
- Comparison with published data (switching kinetics, C–V, read endurance,
  back-switching, series-capacitor reads) with digitized values in
  `data/literature/`.

### Fixed

- Initial state at the transient operating point: ngspice/OSDI does not
  report `analysis("static")` there, so the condition also tests
  `$abstime <= 0`.

## [0.1.0] - 2026-11-15

### Added

- `landau`, `preisach`, `kinetics`, `measurement` and `units` modules:
  Landau–Devonshire and Landau–Khalatnikov, Preisach with FORC, KAI/NLS/Merz
  with fitting, simulated PUND and P–E loops with leakage compensation.
- 43 unit tests and six example figures.

[Unreleased]: https://github.com/Celivem/ferroelectric-models/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Celivem/ferroelectric-models/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Celivem/ferroelectric-models/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Celivem/ferroelectric-models/releases/tag/v0.1.0
