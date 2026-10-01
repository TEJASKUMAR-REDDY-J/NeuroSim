# Engineering notes

Findings from building the solver worker and the desktop app. All measurements: Intel UHD iGPU (i3-10110U), FP16S, D3Q19, Windows 11.

## FluidX3D patch series (`solver/patches/`)

| Patch | Purpose | Validation |
|---|---|---|
| 0001 optional per-step sync | `-DNS_NO_STEP_SYNC` removes the host barrier after every step; the worker synchronizes once per chunk and checks `clFinish` status | Phase 0 sweeps; up to 2× on small grids |
| 0002 DDF stride padding | SoA stride `N + pad` (default pad 2112 elements) for `fi` **and** thermal `gi` | bit-identical fields with pad 0 vs 2112 (Taylor–Green and Rayleigh–Bénard, `tests/test_padding_identity.py`) |
| 0003 external main | lets the worker provide `main()` in GRAPHICS builds | build |
| 0004 state accessors | access to DDFs, mass, excess mass, time step for checkpoint/restart and export | checkpoint round trip in the app |
| 0005 render solids on black | solid force coloring uses light gray as its neutral color (FluidX3D used the background color, invisible on black); per-cell force vectors in wireframe mode off by default | visual |

## Bugs found and fixed during integration

* **Thermal DDFs were not padded** in the first version of patch 0002: `gi` is indexed with the same `index_f()` stride, so the padded stride read and wrote past the end of the buffer. The device queue faulted and every later read returned zeros (shown as impossible throughput and all-zero statistics). Fixed by padding `gi`; the worker now checks `clFinish` status so device faults surface as errors instead of silent zeros.
* **Initial velocity inside solids**: initializing the freestream velocity in every cell, including cells later voxelized as solid, made object surfaces behave as moving walls whenever `MOVING_BOUNDARIES` was compiled in (car on a moving road): drag about 2000× too high and striped surfaces. Object cells are now marked (`TYPE_X`) and get zero velocity after voxelization, as in FluidX3D's own setups.
* **Free-surface rendering** needs `skybox8k.png` at `<exe>/../skybox/`; the build cache provides it.
* **Live rendering cost**: Q-criterion plus surfaces at 1280×720 took 50 % of wall time on the iGPU. Frames are now throttled to a configurable share of wall time (default 20 %); camera interaction still renders immediately.

## Rejected optimization

Workgroup size 32 / 64 / 128 / 256 for all kernels: 32–128 within run-to-run noise of the default 64, 256 about 5 % slower (96³, 136³, 192³, two repeats each). Kept 64.

## Physics checks

* Sphere, Re 1000, D = 32 cells, 5 % blockage, LES on: C<sub>d</sub> = 0.583 ± 0.024 (12 000 steps, mean of second half). Unconfined reference ≈ 0.47; the excess is in the direction expected from blockage and staircase voxelization. Not yet a validated number: needs a resolution and blockage study.
* NACA 4412 wing, 8°, aspect ratio ≈ 2.8, Re 20 000: C<sub>l</sub> ≈ 0.37, C<sub>d</sub> ≈ 0.16 on planform area after 3600 steps; lift has the correct sign.
* Every preset runs 200 steps with finite, sub-sonic velocities (`tests/test_presets.py`).
