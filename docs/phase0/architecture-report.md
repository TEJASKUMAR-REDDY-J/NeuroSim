# NeuroSim CFD — Phase 0: Technical Audit, Measurements and Architecture Decision

| | |
|---|---|
| Status | Phase 0 deliverable. No solver code has been changed; experiments live in `tools/phase0/patches/` as clearly marked patches against upstream. |
| Baseline | FluidX3D `4fb540e9ddadaaee2bef8264c57c7f3f5a7fc571` (2026-09-28, v3.8), pinned as git submodule `external/FluidX3D` |
| Test system | Intel Core i3-10110U, integrated Intel UHD Graphics (Gen9.5, 23 EU reported, 1.0 GHz, 0.368 TFLOP/s FP32 est.), 24 GB DDR4-3200 (16+8 GB, asymmetric dual channel), Windows 11 Home 26200, Intel driver 31.0.101.2127, OpenCL C 3.0, MinGW-w64 g++ 13.2 `-O3`, rustc 1.96 (msvc) |
| Raw data | `docs/phase0/results/2026-10-01-uhd620/` |
| Reproduce | `powershell -ExecutionPolicy Bypass -File tools\phase0\run_phase0.ps1` |

The only compute device available for this audit is one low-end integrated GPU. Everything measured here is measured on that device; everything about discrete, datacenter and multi-GPU hardware is taken from the upstream benchmark tables in `external/FluidX3D/README.md` and labelled as such. Conclusions that need hardware we do not have are marked **blocked on hardware**.

---

## 0. Decisions in one page

| Question | Decision | Primary evidence |
|---|---|---|
| Solver core language | **Keep C++ host + OpenCL C kernels from FluidX3D.** No port. | Kernel runs at 81–91 % of attainable memory bandwidth on the test device (§2.3); the host is not on the critical path (§2.5); a port buys nothing measurable and costs permanent divergence from upstream. |
| Platform language (CLI, runtime, project system, analytics, reports, sweeps, ML) | **Python.** | Analytics, visualization and ML ecosystems (NumPy/SciPy/h5py/PyTorch/JAX/VTK) are Python-first; ML forces Python regardless. Host↔driver call cost is language independent (Rust FFI = C++ within noise, §2.6). |
| Rust | **Not adopted now.** Re-evaluate on explicit triggers (§6.4). | No subsystem identified where Rust is objectively better *and* Python is insufficient. A third language has a real maintenance cost. |
| Coupling between platform and solver | **Out-of-process solver worker** (`neurosim-fx`), JSON case in, JSON-lines telemetry out, fields/checkpoints on request. | FluidX3D uses process-global state and `exit()` on error (§3.2); features are compile-time (§3.1); GPU driver resets must not kill the runtime; the same worker serves local, remote and HPC execution. |
| Compute backend | **OpenCL only**, for now. No backend abstraction layer yet. | OpenCL already covers NVIDIA/AMD/Intel/ARM/Apple(1.2)/CPU; upstream data shows near-roofline results on all major GPU vendors. No measured gap that another API closes, except possibly NVIDIA peer-to-peer for multi-GPU (blocked on hardware). |
| Mesa / Rusticl | **Not a dependency.** It is just another OpenCL ICD; supported for free, tested opportunistically on Linux. | Rusticl is Linux-only; adds coverage (Asahi, Panfrost, Freedreno, radeonsi without ROCm, Zink) with zero code on our side. |
| First optimizations to pursue | (1) multi-GPU overlap, (2) stride padding, (3) sync policy, (4) measured precision selection, (5) persistent worker / JIT cache | Ranked bottleneck register, §11. |
| Licensing | **FluidX3D's licence governs the whole product while it ships FluidX3D.** Non-commercial, no military use, altered source must be published and marked. | `external/FluidX3D/LICENSE.md`. See §9 — this is a hard architectural constraint, not a footnote. |

---

## 1. Methodology

* Source audit of every file in `external/FluidX3D/src` that participates in a simulation (`kernel.cpp`, `lbm.cpp/hpp`, `opencl.hpp`, `defines.hpp`, `main.cpp`, `info.cpp`, `setup.cpp`, `graphics.cpp`; `utilities.hpp`, `units.hpp`, `shapes.cpp` skimmed).
* Builds of the unmodified solver with a replacement benchmark `setup.cpp` (`tools/phase0/bench_setup.cpp`), one binary per precision mode. Wall-clock timing over 300 steps after a 10-step warm-up with a final `clFinish`, so asynchronous variants are timed correctly. MLUP/s = cells × steps / seconds.
* Hypothesis experiments as two small patches: `0001-optional-per-step-sync.patch`, `0002-ddf-stride-padding.patch`.
* Independent microbenchmarks with no FluidX3D code: `tools/phase0/bwprobe.cpp` (OpenCL copy/read/triad bandwidth, launch latency) and `tools/phase0/ffi_probe.rs` (the same launch-latency test from Rust through raw FFI).
* All experiments were repeated end-to-end by `run_phase0.ps1` (fresh builds from the pinned submodule + patches) as an independent second run, `results/.../run2_scripted_all.txt`. Every finding reproduced: padding 128³ 150→180 (+20 %), 256³ 147→189 (+29 %); no per-step sync 32³ 108→217 MLUP/s (2.0×); two domains −18 % (160³) / −42 % (64³); Rust/C++ launch latency equal. Single values differ by up to ~10 % between runs (e.g. FP16S 128³: 171 in run 1, 152 in run 2, 148–151 in the four other runs).
* Run-to-run variance on this laptop is real. The bandwidth probe read 16.4–20.0 GB/s in six runs and 24–26 GB/s in one run that did not reproduce (`results/.../bwprobe.txt`). Every number below is quoted with that caveat; the benchmark subsystem must record repeats, spread and power state (§10.3).

---

## 2. Performance architecture (audit item B)

### 2.1 Where the time goes

One LBM time step on a single device is exactly one kernel launch, `stream_collide` (`external/FluidX3D/src/lbm.cpp:924-953`), followed by a host barrier (`lbm.cpp:951`). The kernel is a fused stream+collide with the **Esoteric-Pull** in-place streaming scheme (`kernel.cpp:1325-1338`): each cell loads 19 DDFs, computes ρ/u, equilibrium and collision, and stores 19 DDFs back to the *same* buffer, with odd/even time-step index swapping. There is no second DDF copy and no separate streaming kernel.

Per cell and step the kernel moves (from `bandwidth_bytes_per_cell_device()`, `lbm.cpp:51`):

| Storage | DDF bytes/cell/step | Total incl. flags | Memory footprint |
|---|---|---|---|
| FP32 | 19×2×4 = 152 | 153 B | 93 B/cell |
| FP16S / FP16C | 19×2×2 = 76 | 77 B | 55 B/cell |

`rho`, `u` are only written when a feature needs them (`UPDATE_FIELDS`) or on request (`update_fields`), so the default step does not pay for macroscopic fields.

Arithmetic: a hand count of the D3Q19 SRT path in `kernel.cpp` (moments ≈ 50, equilibrium ≈ 100, collision ≈ 76, clamping/misc ≈ 20; FMA = 2 FLOP; integer index math and FP16 conversion excluded) gives **≈ 250 FLOP/cell**. Arithmetic intensity is therefore ≈ 1.6 FLOP/B (FP32) and ≈ 3.2 FLOP/B (FP16). The ridge point is ≈ 21 FLOP/B on the test iGPU (0.368 TFLOP/s ÷ ~17.7 GB/s) and ≈ 20 FLOP/B on an H100 SXM (66.9 TFLOP/s ÷ 3350 GB/s). **The solver is memory-bandwidth bound on every listed device by roughly an order of magnitude.** Arithmetic optimization of the FP32 path cannot pay off; reducing or better-organizing bytes can.

### 2.2 Measured precision modes (single device, cubic grid, 300 steps)

| N | FP32 MLUP/s | FP16S MLUP/s | FP16C MLUP/s |
|---:|---:|---:|---:|
| 32 | 92 | 112 | 58 |
| 48 | 90 | 140 | 71 |
| 64 | 87 | 176 | 81 |
| 96 | 98 | 185 | 81 |
| 128 | 78 | 171 | 88 |
| 160 | 96 | 196 | 86 |
| 192 | 95 | 190 | 86 |
| 256 | 92 | 150 | 89 |

Source: `results/.../sweep_fp32.log`, `sweep_fp16s.log`, `sweep_fp16c.log`.

* FP16S is 2.0× FP32 at the same effective bandwidth (13–15 GB/s for both): halving bytes halves time. Memory-bound behaviour confirmed by measurement, not only by model.
* **FP16C is slower than FP32 on this GPU** (≈ 6.7 GB/s effective). FP16C emulates its 1-4-11 format in integer code (`kernel.cpp:847-858`); on a 23-EU Gen9 part that pushes the kernel into the compute-bound regime. Upstream data shows the opposite failure on other hardware: Radeon HD 7850 FP16S 120 vs FP16C 635 MLUP/s; EPYC 7352 FP32 739 vs FP16S 106. **The fastest precision mode is device-specific and cannot be predicted from a FLOPS number.** It must be measured (§11, B5).

### 2.3 Roofline position

At mid sizes (96³–192³) FP16S reaches 14.2–15.5 GB/s of modelled traffic against a probe copy bandwidth of 16.8–17.7 GB/s measured interleaved with the solver runs, i.e. **≈ 81–91 % of attainable bandwidth**. Achieved compute is ≈ 47 GFLOP/s, 13 % of peak. Upstream reports (FP32, vs theoretical bandwidth) 69–87 % for NVIDIA datacenter GPUs, 84–87 % for Intel Arc B-series, 47–76 % for AMD RDNA parts. On the test device there is little left to win inside the kernel; on some vendors there may be more, but that needs those devices.

### 2.4 Power-of-two grid sizes lose 15–25 %

Reproduced twice (`pow2_repeat_fp16s.log`):

| N | 120 | 128 | 136 | 240 | 256 | 264 |
|---|---:|---:|---:|---:|---:|---:|
| MLUP/s (2 runs) | 177 / 178 | 151 / 151 | 182 / 173 | 185 / 189 | 148 / 144 | 186 / 188 |

Hypothesis: DDFs are stored SoA with stride exactly `N` (`index_f`, `kernel.cpp:860-862`). For power-of-two `N` the 19 streams of a cell land on addresses that differ by a multiple of a large power of two and contend for the same cache sets / DRAM channels.

Experiment (`0002-ddf-stride-padding.patch`, stride `N+pad`, `ddf_padding_fp16s.txt`):

| pad (elements) | N=128 | N=256 | N=136 (control) |
|---:|---:|---:|---:|
| 0 | 148 | 142 | 181 |
| 64 | 140 | 145 | 186 |
| 256 | 155 | 153 | 177 |
| 2112 | **176 (+19 %)** | **190 (+34 %)** | 184 |

Small pads do not help; a pad above one 4 KiB page does, and leaves the control unaffected. The change is layout-only (it cannot alter arithmetic), but before adoption it still needs a bit-identity check on a non-trivial flow and measurements on other vendors, because the best pad is a hardware property. This makes it an auto-tuning parameter, not a constant.

### 2.5 Host synchronization and launch cost

* `do_time_step()` calls `finish_queue()` after every step in single-device mode (`lbm.cpp:951`), and FluidX3D's own MLUP/s display times each step on the host (`lbm.cpp:970-972`).
* Probe: launch + `clFinish` = 90–102 µs; launch without waiting = 5–7 µs (`ffi_vs_cpp_launch.txt`).
* Experiment (`0001-optional-per-step-sync.patch`, `sweep_fp16s_nosync.log`): removing the per-step barrier gives 32³: 112 → 213 MLUP/s (1.9×), 48³: 140 → 197, 64³: 176 → 201, and no change from 96³ upward. Overhead ≈ 140–230 µs per step.
* Consequence: irrelevant for large 3D runs, significant for 2D, small 3D and **parameter sweeps of many small cases**, which NeuroSim explicitly wants. Removing the barrier also makes the existing per-step host timing meaningless, so the fix must come with event-based (OpenCL profiling) timing.

### 2.6 Startup cost

`LBM` construction (OpenCL C JIT compile + allocation) costs ≈ 1.05 s per domain on this machine (`domains_dx1_fp16s.log`); initialization 6–108 ms. A 64³ × 1000-step case computes in ≈ 1.5 s, so startup is ~40 % of a small sweep case. The kernel source is specialized at runtime with the grid size baked in as `#define`s (`lbm.cpp:334-468`), so every distinct grid compiles anew. Fix candidates: a persistent worker that runs many cases, and a program-binary cache keyed by the generated source.

### 2.7 Multi-device communication

`communicate_field()` (`lbm.cpp:1355-1383`) per field and axis: extract kernel → blocking PCIe read into host buffers → `finish` on all devices → pointer swap on the host → PCIe write → insert kernel. Every step does this after `stream_collide` has finished the whole domain, so **communication is never overlapped with computation**, all traffic is staged through host RAM, and there is no device peer-to-peer path. A single in-order command queue per device (`opencl.hpp:312`) makes overlap structurally impossible today.

* Local proxy (two domains on the same iGPU, so no PCIe at all, `domains_dx*.log`): 160³ per domain 186 → 152 MLUP/s (−18 %), 64³ 168 → 100 (−40 %). Halo kernels plus host barriers already cost this much before any bus is involved.
* Upstream multi-GPU table (same benchmark, larger grids): 2 GPUs give 1.1–2.0× (most 1.4–1.8×); 4× H100 SXM5 2.7–3.0×; 8× B300 SXM6 **2.1× (FP32)** / 3.7× (FP16S) despite NVLink-class hardware; 8× B200 3.1× / 3.9×. Scaling stops being efficient beyond two devices for fast GPUs, because per-step compute time shrinks while host-staged transfer time does not.

This is the largest system-level performance gap in FluidX3D at the high end. Fixing it is **blocked on multi-GPU hardware** for measurement; the design (§11, B1) can be prepared and the single-device proxy used for correctness.

### 2.8 Memory

Host and device allocations are made once (`LBM_Domain::allocate`, `lbm.cpp:121-176`); there is no allocation in the time loop. On iGPUs and CPUs, `rho/u/flags` use zero-copy host buffers (`CL_MEM_USE_HOST_PTR`, `opencl.hpp:379-386`). DDFs never have a host copy, which is why there is no checkpointing (§3.4).

---

## 3. Software architecture (audit item C)

### 3.1 Configuration is compile-time
Features (`VOLUME_FORCE`, `FORCE_FIELD`, `EQUILIBRIUM_BOUNDARIES`, `MOVING_BOUNDARIES`, `SURFACE`, `TEMPERATURE`, `SUBGRID`, `PARTICLES`, graphics), lattice (`D2Q9/D3Q15/D3Q19/D3Q27`), collision (`SRT/TRT`) and precision (`FP16S/FP16C`) are `#define`s in `defines.hpp`. They change struct layouts and host control flow (`#ifdef` throughout `lbm.hpp/.cpp`) *and* are forwarded to the device compiler. The scenario itself is a C++ function `main_setup()` in `setup.cpp`. Every scenario change is a recompile (≈ 65 s for a full `-O3` build on the test machine).

The device side is already runtime-specialized (JIT with injected constants), which is good: feature cost is zero when unused. The host side is the problem for a platform.

### 3.2 Process-global state
`info`, `units`, `camera`, `main_arguments`, `running` are globals; `print_error()` terminates the process; `main()` is owned by FluidX3D (`main.cpp`, or `graphics.cpp` with graphics). Hosting several simulations, or surviving a failed one, in the same process is not supported. This decides the integration model (§6.3).

### 3.3 Modules and dependencies
`kernel.cpp` (OpenCL C as a raw string, 3.2 k lines) · `lbm.cpp/hpp` (simulation object, domain decomposition, transfers, voxelization, VTK output) · `opencl.hpp` (thin C++ wrapper over `opencl.hpp` bindings: `Device_Info`, `Device`, `Memory<T>`, `Kernel`) · `graphics.cpp` (WinAPI/X11/ASCII windowing, camera, input) · `info.cpp` (console status) · `units.hpp` (SI↔lattice units) · `shapes.cpp` · `utilities.hpp` (174 KB header: math, file I/O, `parallel_for`, STL reader, image I/O) · vendored lodepng, OpenCL headers/import libs, X11 headers/libs. Build: `makefile`, `make.sh`, Visual Studio project. No tests, no CMake.

### 3.4 Gaps relevant to NeuroSim
No checkpoint/restart (DDFs have no host buffer). Geometry import is binary STL only (`utilities.hpp:4531`). Field output is legacy big-endian VTK and PNG/QOI/BMP frames. Device auto-selection picks the highest *estimated TFLOPS* device (`lbm.cpp:655-698`, estimate in `opencl.hpp:196`) — the wrong metric for a bandwidth-bound solver.

---

## 4. Numerical architecture (audit item A)

* **LBM**: D2Q9/D3Q15/D3Q19 (default)/D3Q27; SRT (BGK) or TRT collision; Guo forcing (`kernel.cpp:1090`); velocity clamped to the lattice speed of sound for stability (`kernel.cpp:1553-1560`).
* **DDF-shifting**: DDFs stored as `f − w` and density summed with `+1` last (`kernel.cpp:1063-1065`, `1003-1013`) to minimise round-off — this is what makes FP16 storage viable.
* **Storage precision**: arithmetic always FP32. FP16S = IEEE half scaled by 2¹⁵ via hardware `vload_half/vstore_half_rte` (`lbm.cpp:410-414`). FP16C = custom 1-4-11 format in software (`kernel.cpp:847-858`).
* **Boundaries**: bounce-back on `TYPE_S` implicit in Esoteric-Pull; moving walls via momentum correction (`kernel.cpp:1104-1111`); equilibrium in/outflow `TYPE_E`; temperature `TYPE_T`. Periodic everywhere else by index wrap-around (`kernel.cpp:903-914`).
* **Extensions**: Smagorinsky–Lilly LES (`kernel.cpp:1579-1592`); thermal D3Q7 double-distribution with Boussinesq coupling; free surface (volume-of-fluid, PLIC curvature, 4 extra kernels per step); immersed-boundary particles; forces/torque on solids by direct DDF momentum exchange plus device reductions (`object_force/object_torque`).
* **Voxelization** on the device from triangle meshes, including moving/rotating geometry.
* **Built-in validation setups** worth reusing as references: Poiseuille (L2 error vs analytic), Stokes drag, Taylor–Green 2D/3D, lid-driven cavity, Taylor–Couette, Karman street, Rayleigh–Bénard.

---

## 5. Visualization architecture (audit item D)

Rendering is done by OpenCL kernels into an RGBA bitmap + z-buffer on the device (`LBM_Domain::Graphics`, `lbm.cpp:471-602`): flag wireframe, marching-cubes solids, velocity/density/temperature fields (ray-marched single-GPU, rasterized multi-GPU), slices, streamlines, Q-criterion isosurface, rasterized/ray-traced free surface, particles. The frame (e.g. 1920×1080×4 B ≈ 8 MB) is read back once per frame and blitted via GDI or X11; multi-GPU frames are composited on the host by z-buffer. Frame export encodes PNG/QOI/BMP in a detached thread.

This is genuine **in-situ visualization**: no volume ever leaves the GPU for rendering. NeuroSim must preserve that property. What is missing is a client/server boundary (the window, camera and keyboard are hard-wired) and quantitative analysis views (plots, probes, comparisons), which belong to NeuroSim, not to the renderer.

---

## 6. Language and integration decision (audit item F)

### 6.1 Candidate architectures

| | A: C++-first | B: Rust-first (port host) | C: Hybrid C++ core + Rust platform | **Chosen: C++ core + Python platform** |
|---|---|---|---|---|
| Solver speed | = | = (kernels are OpenCL C in all options) | = | = |
| Host→driver overhead | baseline | measured equal (§6.2) | measured equal | not on the per-step path |
| Upstream tracking | easy | lost: ~5 k lines rewritten, permanent fork | easy | easy |
| Analytics / FFT / stats | weak | moderate (ndarray, rustfft) | moderate | **strong (NumPy/SciPy)** |
| Geometry I/O, vis (slices, isosurfaces, streamlines) | VTK C++ | thin | thin | **VTK/PyVista, trimesh** |
| ML (FNO, DeepONet, GNN, active learning) | needs Python anyway | needs Python anyway | needs Python anyway | **native** |
| Languages to maintain | 2 (+Python for ML) | 2 (+Python) +C++ kernels' host glue | 3 | **2** |
| Single-binary distribution | good | best | good | weaker (wheels / env) |
| Safety for a long-running daemon | moderate | best | best | adequate |

### 6.2 Measured: language does not affect host↔GPU cost
Same empty-kernel launch loop from C++ and from Rust via raw `extern "system"` FFI against the same `OpenCL.lib`, 3 alternating runs each (`ffi_vs_cpp_launch.txt`): launch+finish 91–101 µs (Rust) vs 90–102 µs (C++); batched launch 5.3–7.3 µs vs 5.4–7.5 µs. The call itself is a plain C ABI call; driver cost dominates by three orders of magnitude.

### 6.3 Integration model: solver as a worker process
`neurosim-fx` is FluidX3D's sources plus a NeuroSim-owned replacement for `setup.cpp` that:
1. reads a JSON case description (grid, units, physics, boundaries, geometry files, run control, outputs, probes);
2. runs the solver; emits **JSON-lines telemetry** on stdout (step, time, MLUP/s, forces, residual proxies, timings, warnings);
3. writes fields (`.npy` + JSON sidecar) and checkpoints only when asked.

Why a process rather than a Python extension module: global state and `exit()` (§3.2); compile-time feature variants map naturally to multiple worker binaries (§3.1); a GPU reset or driver crash kills the worker, not the runtime, which is what crash recovery needs; and the identical worker runs under a local runtime, a remote agent or a batch scheduler. The cost is IPC, which is small for telemetry and, for fields, equal to the device→host copy that is unavoidable anyway.

Feature variants: the platform generates `defines.hpp` from the case, hashes (FluidX3D commit, patch set, defines, compiler, flags) and builds the variant once into a local cache (≈ 65 s per variant here). The device-side JIT specialization stays as it is.

### 6.4 When Rust becomes the right call
Adopt Rust for a specific component only if one of these is observed: the remote runtime daemon needs concurrency/robustness Python cannot deliver at acceptable effort; single-binary distribution of the CLI becomes a user requirement; a native GUI with performance needs the Python stack cannot meet; or a host-side data path (e.g. high-rate field streaming/compression) profiles as a bottleneck in Python. Each trigger must be shown with a measurement, as in §6.2.

---

## 7. Mesa / Rusticl (audit item G)

Rusticl is Mesa's OpenCL implementation, written in Rust, layered on Gallium drivers: radeonsi, iris, llvmpipe, zink (OpenCL over Vulkan), asahi, panfrost, freedreno, nouveau. Recent Mesa (26.2) advertises OpenCL 3.1 on radeonsi/llvmpipe/zink/iris; FP16 support has landed for asahi, freedreno, llvmpipe and panfrost. It is Linux-only; it does not exist on Windows or macOS. FluidX3D already special-cases it (`opencl.hpp:152`, "rusticl reports EMBEDDED_PROFILE").

Value for NeuroSim: coverage on Linux devices without a vendor OpenCL runtime (Apple Silicon under Asahi, Mali, Adreno, AMD cards without ROCm), and potentially any Vulkan GPU via zink. All of that arrives **through the standard ICD loader with zero code on our side**. Linking Mesa, shipping it, or designing around it would add distribution complexity for no measured gain. Decision: not a dependency; add a Linux CI/benchmark row for Rusticl (and PoCL for CPUs) when a Linux machine is available; treat results as one more device in the benchmark database. That Mesa contains Rust is irrelevant to this decision.

---

## 8. Compute APIs and hardware abstraction (audit items E, H)

**Current hardware abstraction (E):** `get_devices()` enumerates all platforms/devices with one context per device (`opencl.hpp:223-253`); `Device_Info` derives vendor-specific core counts and TFLOPS estimates and applies driver workarounds (Intel >4 GB buffers, NVIDIA FP16 reporting, ARM FMA, AMD dual-CU naming); one in-order queue per device; build options `-cl-finite-math-only -cl-no-signed-zeros -cl-mad-enable`. Multi-device = domain decomposition with one device per domain, preferring identical device types.

| API | Verdict | Reason |
|---|---|---|
| OpenCL | **Keep as the only backend now** | Covers all vendors and CPUs; already near roofline (upstream tables; §2.3); runtime JIT specialization is a strength. |
| CUDA | Candidate **only** for NVIDIA multi-GPU communication (P2P/NVLink), and possibly ML interop | OpenCL has no peer-to-peer on NVIDIA; that is exactly the measured scaling gap (§2.7). Must first try vendor-neutral overlap; blocked on hardware. |
| HIP | No | AMD OpenCL already performs; same P2P argument as CUDA would apply later. |
| Vulkan compute | Defer | Value is coverage where OpenCL is missing (Android without OpenCL, future macOS). If needed, clspv can compile the existing OpenCL C to Vulkan SPIR-V, and Rusticl-on-zink is an alternative path. No performance argument for a memory-bound kernel. |
| SYCL | No | Adds a toolchain; solves no measured problem. |
| wgpu/WebGPU | Not for the solver | No 64-bit integers in WGSL (DDF indices exceed 2³²), binding-size limits. Possible later for a browser viewer only. |
| Native CPU SIMD | Defer | CPU execution already works through OpenCL CPU runtimes (Intel CPU Runtime, PoCL). Upstream CPU efficiencies are 14–74 % of bandwidth and very precision-dependent; a native backend is justified only after measuring those runtimes on the target CPUs. |

A backend abstraction layer is **not** introduced in Phase 1: with one backend it would be an interface with one implementation. The boundary that will exist if a second backend is ever justified is already visible: device enumeration/info, buffers, kernel launch, queue/event, and a kernel source per backend.

---

## 9. Licensing constraints

FluidX3D's licence (`external/FluidX3D/LICENSE.md`) permits use for public research, education or personal use, modification and redistribution, but:
1. origin must not be misrepresented; **altered versions must be plainly marked**;
2. **no commercial use**, including paid hosting/support of a product whose value derives from it;
3. no military use;
4. no training of AI models **on its source code** (simulation output data is not mentioned);
5. if binaries of altered versions, **or data/results generated by them**, are published, the altered source must be published;
6. listed papers must be cited in publications;
7. the notice must be kept.

Architectural consequences adopted here: FluidX3D stays an unmodified, pinned submodule; every NeuroSim change to it is a separate, named patch file; this repository is public so clause 5 is met for any published result; `NOTICE.md` carries the obligations. **While NeuroSim ships FluidX3D, NeuroSim as a whole is non-commercial.** If commercial use is ever a goal, the only route is a solver that does not derive from FluidX3D — a decision the project owner should make early, because it would change the solver strategy entirely. The surrogate/dataset plan (§13) only uses simulation outputs, never FluidX3D source, as training data.

---

## 10. Analytics and telemetry architecture (audit item I)

### 10.1 Data path
```
neurosim-fx (device)                      runtime (Python)                         consumers
 stream_collide / reductions  ──JSONL──▶  run directory                    ──▶   CLI / UI / reports
 forces, probes, slices       ──.npy───▶   run.json        provenance              analytics engine
 frames (in-situ render)      ──PNG────▶   telemetry.jsonl time series             comparison, sweeps
 checkpoints (on request)     ──bin────▶   fields/, frames/, checkpoints/          dataset export
                                           project.sqlite  (index of runs, metrics, benchmarks)
```
Reductions happen on the device whenever possible (forces and torques already do); only scalars and requested slices cross the bus by default. Full fields are exported on an explicit schedule. Disk I/O therefore cannot silently become the bottleneck.

### 10.2 Engines (all operate on the run directory, so they work identically for numerical and learned runs)
* **Performance analytics**: measured MLUP/s × bytes/cell vs the device's *measured* probe bandwidth → bandwidth utilization; event-based kernel time vs wall time → host/sync overhead; transfer and render shares; scaling efficiency vs single-device baseline. Classification rules produce the explanation ("memory-bound at 86 % of measured bandwidth; 31 % of wall time in host barriers").
* **Physics analytics**: forces/coefficients, flow rate and pressure drop over planes, mass-conservation drift, Reynolds/Mach checks, vorticity/Q from exported fields; probes (point, line, plane, surface, volume).
* **Temporal analytics**: moving statistics, FFT/Welch spectra (Strouhal), peak detection, steady-state and statistically-stationary detection.
* **Comparative analytics**: run-to-run deltas of scalars and difference fields on matching grids.

### 10.3 Benchmark database
Every benchmark record stores hardware, OS, driver, backend, compiler and flags, FluidX3D commit + patch set, precision, lattice, collision, grid, steps, **repeats with median and spread**, power source, MLUP/s, effective bandwidth, probe bandwidth, memory. The unpatched pinned FluidX3D build is the permanent baseline row.

---

## 11. Bottleneck register (ranked)

| # | Bottleneck | Evidence | Proposed change | Expected effect | Status |
|---|---|---|---|---|---|
| B1 | Multi-device halo exchange is host-staged, blocking and never overlapped | §2.7: upstream 2×→1.1–2.0×, 8× B300 → 2.1×; local proxy −18 % / −40 % | Split `stream_collide` into boundary shell + interior; exchange halos on a second queue while the interior computes; device P2P where the platform offers it | Approach linear scaling for large domains | Design possible now; **measurement blocked on multi-GPU hardware** |
| B2 | Power-of-two DDF stride aliasing | §2.4: −15…−25 %; pad 2112 → +19 % / +34 % | Padded SoA stride, pad chosen per device by auto-tuning | +19…34 % on affected sizes | Patch exists; needs bit-identity validation and other vendors |
| B3 | Per-step host barrier | §2.5: up to 1.9× at 32³, ~140–230 µs/step | Synchronize only when the host needs data; event-based timing | Large for 2D/small/sweep cases, zero for large | Patch exists; timing rework needed |
| B4 | Startup / JIT per case | §2.6: ≈1.05 s per domain | Persistent worker running many cases; OpenCL program-binary cache | Removes ~40 % overhead from small sweep cases | Not started |
| B5 | Precision and device chosen by FLOPS estimate | §2.2, §3.4 | Few-second micro-benchmark per (device, precision) feeding device/precision selection; user can override | Avoids 2–10× wrong choices on some hardware | Not started |
| B6 | Kernel efficiency | §2.3: 81–91 % of attainable on test device | None now; revisit on AMD RDNA where upstream shows 47–76 % | Unknown | Deprioritized |

Rule for all items: an optimization that changes results must pass the validation suite; layout-only changes must be bit-identical to the baseline.

---

## 12. UI architecture

The UI is a client of the same Python API the CLI uses; it never contains solver logic. Two candidates, to be prototyped in Phase 3 and decided by measurement of interactivity on remote runs:
1. **Desktop** (Qt + VTK/PyVista): richest local post-processing, ParaView-grade filters out of the box.
2. **Browser UI served by the runtime**: one front end for local and remote/HPC, live frames streamed from FluidX3D's in-situ renderer (frames are already produced on the device), plots from telemetry, heavy 3D post-processing delegated to VTK export / ParaView.

FluidX3D's renderer is kept as the live visualization engine; its window and keyboard handling are not.

---

## 13. AI architecture (audit item J)

Optional, behind one interface; the solver and analytics never depend on it.

```
ModelAdapter
  mode: surrogate | corrector | timestepper
  inputs(case, state?) -> tensors        outputs -> fields/scalars on the lattice
  provenance: training runs, data hashes, code version
Validator (shared with the analytics engine)
  mass/momentum conservation, divergence, boundary violation, force error vs reference,
  long-horizon drift; failure -> hand state back to the numerical solver
```
* Surrogate: case → fields/scalars, no solver involvement.
* Corrector: worker pauses every K steps, exports `rho/u` (DDFs reconstructed from equilibrium + stored non-equilibrium if needed), model corrects, worker resumes. Cost per exchange ≈ field size / bus bandwidth (e.g. 256³ `u` FP32 ≈ 200 MB).
* Timestepper: state(t) → state(t+Δt), periodically re-anchored by the solver.
* Datasets come from sweeps: case parameters + geometry hash + fields + scalars + provenance, exported to HDF5 (h5py available) / Zarr / NumPy; PyTorch/JAX loaders.
* The NCA + neural-operator hybrid is a research track with explicit baselines (pure FNO, pure solver) and the same validator; nothing in the core assumes it works.

---

## 14. Migration plan (audit item K)

| Component | Category | Reasoning |
|---|---|---|
| `kernel.cpp` OpenCL kernels | **KEEP** | Numerical reference; at the bandwidth roof on the test device. Changes only via measured, validated patches (B1–B3). |
| `lbm.cpp/hpp` simulation + decomposition | **KEEP + minimal patches** | Patches: checkpoint/restart (host buffer for DDFs + `t` parity), sync policy, event timing, stride padding, later comm/compute overlap. |
| `opencl.hpp` | **KEEP**, later **REFACTOR** | Needs profiling-enabled queue and a second queue for B1. |
| `setup.cpp` scenario functions | **REPLACE** by JSON case loader | Scenarios become data. Upstream setups are kept as reference cases for validation. |
| `defines.hpp` | **WRAP** | Generated per feature variant by the build cache. |
| `main.cpp`, `info.cpp` console output | **WRAP** | Worker emits JSON lines; human console output optional. |
| `graphics.cpp` window/input | **REPLACE** (Phase 3) | Rendering kernels kept; window/camera control moves to the NeuroSim UI. |
| `utilities.hpp`, `units.hpp`, `shapes.cpp` | **KEEP** | Dependencies of the core. Unit conversion mirrored in Python for pre-run validation of cases. |
| Device selection by TFLOPS | **REPLACE** | Measured selection (B5), passed to the worker as explicit device IDs (already supported via arguments). |
| Binary-STL-only import | **WRAP** | Platform converts OBJ/PLY/ASCII STL to binary STL before voxelization. |
| VTK export | **KEEP**, add `.npy` | `.npy` is trivially read by NumPy/PyTorch. |
| make/vcxproj build | **REPLACE** for the worker | One small CMake project builds worker variants on Windows/Linux/macOS. Upstream build files untouched. |
| Anything | **PORT** | Nothing qualifies: no component is faster or meaningfully safer in another language. |
| Anything | **DELETE** | Nothing deleted from upstream; deletions would only make rebasing harder. |

---

## 15. Phased plan

| Phase | Scope | Exit criteria |
|---|---|---|
| 0 Audit | This report, tooling, raw data | Reviewed and accepted by the project owner |
| 1 Shell | `neurosim-fx` worker (CMake, JSON case, JSONL telemetry, variant cache); Python package `neurosim` with `devices`, `benchmark`, `simulate`, `validate`, minimal `report`; run directories with provenance; SQLite benchmark DB; validation cases Poiseuille, Taylor–Green decay, lid-driven cavity (Ghia et al.), cylinder drag | Unpatched worker reproduces Phase 0 numbers within measured spread; validation cases within published tolerances; every run answers "what produced this" from `run.json` |
| 2 Analytics | Telemetry schema, time-series analytics, steady-state detection, probes, run comparison, performance explanation | Performance report explains Phase 0 bottlenecks automatically |
| 3 UI | Prototype both UI candidates; choose by measurement | One UI driving the same API as the CLI |
| 4 Compute abstraction | Only if a second backend is justified by B1/coverage evidence | — |
| 5 Selective migration | Only items with evidence | — |
| 6 Optimization | B1–B5 as validated patches | Each change benchmarked against the baseline row, physics unchanged |
| 7 AI | Model adapter, validator, dataset export | Surrogate rejected/accepted by the validator, not by loss |
| 8 Design exploration | Sweeps, ranking, Bayesian optimization / active learning | User-defined objectives |

### Needed from the project owner
1. Confirmation of the platform-language decision (Python platform, no Rust for now).
2. A licence choice for NeuroSim's own code (FluidX3D's licence still applies to the combined product, §9).
3. Access to at least one discrete GPU, ideally two identical ones, and one Linux machine; multi-GPU work (B1) and Rusticl/PoCL rows cannot be measured on the current laptop.
