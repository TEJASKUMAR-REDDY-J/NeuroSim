"""Builds solver worker variants: pinned FluidX3D sources + NeuroSim patches + generated defines.hpp.

FluidX3D selects features at compile time, so every combination of precision, lattice, collision and
extensions is a separate executable. Variants are cached by a hash of everything that goes into them.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "external" / "FluidX3D"
PATCH_DIR = ROOT / "solver" / "patches"
WORKER_SRC = ROOT / "solver" / "worker" / "neurosim_worker.cpp"
BUILD = ROOT / "build" / "worker"
EXE = "neurosim-fx.exe" if os.name == "nt" else "neurosim-fx"

LATTICES = ("D2Q9", "D3Q15", "D3Q19", "D3Q27")
COLLISIONS = ("SRT", "TRT")
PRECISIONS = ("FP32", "FP16S", "FP16C")
EXTENSIONS = ("VOLUME_FORCE", "FORCE_FIELD", "EQUILIBRIUM_BOUNDARIES", "MOVING_BOUNDARIES", "SURFACE", "TEMPERATURE", "SUBGRID", "PARTICLES")
TOGGLES = LATTICES + COLLISIONS + ("FP16S", "FP16C", "BENCHMARK") + EXTENSIONS + ("INTERACTIVE_GRAPHICS", "INTERACTIVE_GRAPHICS_ASCII", "GRAPHICS")
CXXFLAGS = ["-std=c++17", "-O2", "-pthread", "-Wno-comment", "-DNEUROSIM_WORKER", "-DNS_NO_STEP_SYNC"]
_lock = threading.Lock()


def variant(precision="FP16S", lattice="D3Q19", collision="SRT", extensions=(), frame=(1280, 720)):
    assert precision in PRECISIONS and lattice in LATTICES and collision in COLLISIONS, (precision, lattice, collision)
    ext = sorted(set(extensions))
    assert all(e in EXTENSIONS for e in ext), ext
    if "TEMPERATURE" in ext or "PARTICLES" in ext and "FORCE_FIELD" in ext:
        ext = sorted(set(ext) | {"VOLUME_FORCE"})  # FluidX3D requires it for these combinations
    return {"precision": precision, "lattice": lattice, "collision": collision, "extensions": ext, "frame": [int(frame[0]) // 8 * 8, int(frame[1]) // 8 * 8]}


def _digest(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:12]


def fluidx3d_commit():
    return subprocess.run(["git", "-C", str(FX), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _patches():
    return sorted(PATCH_DIR.glob("*.patch"))


def source_tree():
    """Pinned FluidX3D src/ with the NeuroSim patch series applied (cached)."""
    key = _digest(fluidx3d_commit(), *[p.read_bytes() for p in _patches()])
    tree = BUILD / f"src-{key}"
    if (tree / ".ok").exists():
        return tree
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(FX / "src", tree / "src")
    rel = tree.relative_to(ROOT).as_posix()
    for p in _patches():  # git apply resolves paths from the repository root
        subprocess.run(["git", "-C", str(ROOT), "apply", "-p1", f"--directory={rel}", str(p)], check=True)
    (tree / ".ok").write_text(key)
    return tree


def defines_hpp(v, upstream_text):
    on = {v["lattice"], v["collision"], "GRAPHICS", *v["extensions"]}
    if v["precision"] != "FP32":
        on.add(v["precision"])
    text = upstream_text
    for name in TOGGLES:
        text = re.sub(rf"^(//)?#define {name}\b", ("#define " if name in on else "//#define ") + name, text, count=1, flags=re.M)
    text = re.sub(r"#define GRAPHICS_FRAME_WIDTH \d+", f"#define GRAPHICS_FRAME_WIDTH {v['frame'][0]}", text)
    text = re.sub(r"#define GRAPHICS_FRAME_HEIGHT \d+", f"#define GRAPHICS_FRAME_HEIGHT {v['frame'][1]}", text)
    text = re.sub(r"#define GRAPHICS_BACKGROUND_COLOR 0x[0-9A-Fa-f]+", "#define GRAPHICS_BACKGROUND_COLOR 0x000000", text)  # black (solids stay visible via patch 0005)
    return text


def _compile(src, obj, include, log):
    cmd = ["g++", "-c", str(src), "-o", str(obj), "-I", str(include), "-I", str(include / "OpenCL" / "include")] + CXXFLAGS
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        log(r.stderr[-4000:])
        raise RuntimeError(f"compile failed: {src.name}")
    return obj


def _link_args(tree):
    lib = tree / "src" / "OpenCL" / "lib"
    if os.name == "nt":
        return [str(lib / "OpenCL.lib"), "-static"]
    if sys.platform == "darwin":
        return ["-framework", "OpenCL"]
    return ["-L", str(lib), "-lOpenCL"]


def build(v, log=print):
    """Returns the worker executable for variant v, building it if needed (thread-safe)."""
    with _lock:
        tree = source_tree()
        defines = defines_hpp(v, (tree / "src" / "defines.hpp").read_text())
        key = _digest(tree.name, defines, WORKER_SRC.read_bytes(), CXXFLAGS)
        out = BUILD / key
        exe = out / EXE
        if exe.exists():
            return exe
        log(f"building solver variant {key}: {v['precision']} {v['lattice']} {v['collision']} {' '.join(v['extensions']) or '(no extensions)'}")
        if out.exists():
            shutil.rmtree(out)
        vsrc = out / "src"
        shutil.copytree(tree / "src", vsrc, ignore=shutil.ignore_patterns("*.cpp", "X11"))
        (vsrc / "defines.hpp").write_text(defines)
        shared = tree / "obj"  # translation units that do not include defines.hpp are shared by all variants
        shared.mkdir(exist_ok=True)
        jobs = [(tree / "src" / f"{n}.cpp", shared / f"{n}.o", tree / "src") for n in ("kernel", "lodepng", "shapes") if not (shared / f"{n}.o").exists()]
        for n in ("graphics", "info", "lbm"):
            shutil.copy(tree / "src" / f"{n}.cpp", vsrc / f"{n}.cpp")
            jobs.append((vsrc / f"{n}.cpp", out / f"{n}.o", vsrc))
        jobs.append((WORKER_SRC, out / "worker.o", vsrc))
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
            list(pool.map(lambda j: _compile(*j, log), jobs))
        objs = [str(shared / f"{n}.o") for n in ("kernel", "lodepng", "shapes")] + [str(out / f"{n}.o") for n in ("graphics", "info", "lbm", "worker")]
        r = subprocess.run(["g++", *objs, "-o", str(exe), "-pthread", *_link_args(tree)], capture_output=True, text=True)
        if r.returncode:
            log(r.stderr[-4000:])
            raise RuntimeError("link failed")
        sky = BUILD / "skybox" / "skybox8k.png"  # FluidX3D loads it from <exe>/../skybox/ for free-surface raytracing
        if not sky.exists():
            sky.parent.mkdir(parents=True, exist_ok=True); shutil.copy(FX / "skybox" / "skybox8k.png", sky)
        (out / "variant.json").write_text(json.dumps({**v, "fluidx3d": fluidx3d_commit(), "patches": [p.name for p in _patches()]}, indent=1))
        log(f"built {exe}")
        return exe


if __name__ == "__main__":
    print(build(variant()))
