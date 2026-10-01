# NeuroVerse CFD

NeuroSim is a desktop CFD workbench built around the [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) lattice Boltzmann solver. FluidX3D stays the high-performance C++/OpenCL core; NeuroSim adds a case system, a native GUI with live GPU-rendered 3D, quantitative slices and monitoring, checkpoints, field export, provenance, and self-benchmarking.

## Quick start

```bash
git clone --recurse-submodules https://github.com/TEJASKUMAR-REDDY-J/NeuroSim.git
```

Double-click `NeuroSim.bat`, or:

```bash
python -m neurosim
```

Pick a preset (car on a moving road, Ahmed body, wing, rotating fan, sphere, cylinder vortex street, lid-driven cavity, particles, dam break, Rayleigh–Bénard convection, Taylor–Green vortex) or import an STL/OBJ/PLY model, then **Preview** or **Run**. The 3D view shows FluidX3D-style vortex lines on black, with an optional faint translucent speed haze (or a full translucent gas cloud for gas and convection cases). See [docs/app.md](docs/app.md).

Requirements: an OpenCL GPU driver, `g++` (MinGW-w64 on Windows), Python 3.10+ with NumPy, PyQt5, Matplotlib.

## Solver optimizations (measured, Intel UHD iGPU, FP16S, D3Q19)

| Change | Effect | Evidence |
|---|---|---|
| Padded DDF stride (patch 0002) | 128³: 150 → 180 MLUP/s (+20 %), 256³: 147 → 189 (+29 %), neutral on other sizes | bit-identical results (`tests/test_padding_identity.py`) |
| No host barrier per time step (patch 0001) | up to 2× on small grids (32³: 108 → 217 MLUP/s), no change on large grids | `docs/phase0/results/` |
| Live-render time budget | 3D frames limited to a set share of wall time (default 20 %); before: rendering took 50 % | in-app insight panel |
| Raw frame transfer | GPU-rendered frames go to the GUI as raw RGB32, no image encoding | — |
| Precision chosen by measurement | `neurosim bench` picks the fastest of FP32/FP16S/FP16C per machine; FP16C is slower than FP32 on this iGPU | Phase 0 report |
| Rejected: workgroup size 32/128/256 | within noise of the default 64; 256 is slower | Phase 0 notes |

## Layout

```
external/FluidX3D/     upstream solver, pinned submodule (never edited in place)
solver/patches/        NeuroSim modifications to FluidX3D, one named patch each
solver/worker/         solver process: case loader, telemetry, live control, slices, checkpoints, exports
neurosim/              Python platform: build cache, cases, geometry, runs, benchmark, desktop app
tests/                 end-to-end preset test, padding bit-identity test, GUI smoke test, drag validation
docs/                  Phase 0 audit report, app guide, measurement logs
tools/phase0/          Phase 0 benchmark harness and probes
```

## Licence

FluidX3D is third-party software under its own non-commercial licence; see [NOTICE.md](NOTICE.md). That licence applies to any distribution of NeuroSim that includes FluidX3D.
