# NeuroVerse CFD

NeuroSim is a simulation platform built around the [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) lattice Boltzmann solver: a headless runtime and CLI, experiment and provenance management, self-benchmarking, simulation analytics, and an optional learned-physics layer. The solver stays a high-performance C++/OpenCL core; the platform around it is built so that the CLI, a GUI, remote/HPC execution and notebooks all drive the same API.

## Status

**Phase 0 (audit) complete.** No platform code yet. The architecture decision, measurements and migration plan are in [docs/phase0/architecture-report.md](docs/phase0/architecture-report.md).

Key measured findings on the reference laptop (Intel UHD iGPU):

- The FluidX3D kernel runs at 81–91 % of attainable memory bandwidth; FP16S storage doubles throughput; FP16C is slower than FP32 on this device, so precision must be chosen by measurement.
- Power-of-two grids lose 15–25 % to DDF stride aliasing; a padded stride recovers +19 % (128³) and +34 % (256³).
- The per-step host barrier costs ~140–230 µs, halving throughput on small grids.
- Multi-GPU halo exchange is host-staged and not overlapped with compute, which is the main scaling limit in the upstream multi-GPU data.

## Layout

```
external/FluidX3D/          upstream solver, pinned submodule (never edited in place)
docs/phase0/                audit report and raw measurement logs
tools/phase0/               benchmark harness, bandwidth/latency probes, experiment patches, reproduction script
```

## Reproducing Phase 0

Requirements: Windows, MinGW-w64 `g++` (C++17), an OpenCL GPU driver; optional `rustc` for the FFI probe.

```bash
git clone --recurse-submodules https://github.com/TEJASKUMAR-REDDY-J/NeuroSim.git
```

```bash
powershell -ExecutionPolicy Bypass -File tools\phase0\run_phase0.ps1
```

Results are written to `build/phase0/phase0_results.txt`.

## Licence

FluidX3D is third-party software under its own non-commercial licence; see [NOTICE.md](NOTICE.md). That licence applies to any distribution of NeuroSim that includes FluidX3D.
