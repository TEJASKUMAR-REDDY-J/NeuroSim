# NeuroSim desktop app

Start it with `NeuroSim.bat` (double-click) or:

```bash
python -m neurosim
```

Requirements: Windows or Linux, an OpenCL driver for your GPU (or a CPU OpenCL runtime), `g++` (MinGW-w64 on Windows), Python 3.10+ with NumPy, PyQt5 and Matplotlib.

## Layout

| Area | What it does |
|---|---|
| Top bar | Presets, open/save case, **Preview** (builds the scene, shows the geometry, does not step), **Run**, **Pause**, **Stop**, run state, checkpoint, field export, PNG snapshot, hardware |
| Left: Scene | Domain size, objects (built-in library or imported STL/OBJ/PLY), size/position/rotation, role (solid, liquid body, heated solid), spin (rotating parts), rest on floor |
| Left: Physics | Flow direction and lattice velocity, Reynolds number, boundaries per face (periodic, wall, freestream in/outflow, moving wall, hot, cold), LES turbulence model, forces, gravity, free-surface liquid boxes and surface tension, heat transfer, tracer particles |
| Left: Solver | Storage precision (auto = fastest measured), velocity set, collision operator, memory-layout padding, device, run length, telemetry/slice/checkpoint intervals, live-3D frame rate and its maximum share of time |
| Center | 3D view rendered by FluidX3D's GPU renderer on black: domain outline, solids (colored by surface force when forces are on), vortices (Q-criterion, colored by velocity), streamlines, velocity field and slices, free surface (rasterized or raytraced), particles. Drag to orbit, wheel to zoom, shift+wheel for field of view. |
| Bottom | Throughput (solver and wall clock), max velocity, drag/lift coefficients, kinetic energy, mass drift, temperature or liquid volume, and an insight panel that names the current bottleneck |
| Right: Slice | Quantitative cut plane (velocity magnitude/components, density, vorticity, temperature or fill level, flags, pressure) with color bar and hover readout |
| Right: Runs | Every run with its status; open a finished run to see its history and last frame, resume from the latest checkpoint, open the run folder |

## What happens when you press Run

1. The case is validated and written to `workspace/runs/<time>-<name>/` together with placed geometry and a provenance record (`run.json`: NeuroSim and FluidX3D commits, patch set, solver variant, derived parameters, host).
2. The solver variant for the selected physics is compiled once (about 30–60 s) and cached in `build/worker/`.
3. The solver process runs the simulation on the GPU, streams telemetry (`telemetry.jsonl`), slices and frames to the app, and accepts live commands (camera, visualization, slice, pause, checkpoint, export).

## Outputs

* `fields/*.npy` — `rho`, `u` (shape 3×Nz×Ny×Nx), `T`, `phi`; load with `numpy.load`.
* `checkpoints/*.nsck` — complete solver state; **Runs → Resume** continues from the latest one.
* `telemetry.jsonl` — one JSON object per line (steps, performance, forces, frames, slices, events).
* Snapshot — PNG of the current 3D view.

## Command line

```bash
python -m neurosim run case.json --steps 5000
```

```bash
python -m neurosim devices
```

```bash
python -m neurosim bench
```

`bench` measures memory bandwidth and the throughput of every storage precision on this machine; the fastest one becomes the default for `precision: auto`.
