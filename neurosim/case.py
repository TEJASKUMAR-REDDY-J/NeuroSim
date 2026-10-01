"""Simulation case: user-facing description -> solver variant, lattice parameters and worker case file.

All quantities handed to the solver are lattice units. Optional SI reference values are only used to
convert results for display.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from . import build, geometry

FACES = ("x0", "x1", "y0", "y1", "z0", "z1")
FACE_TYPES = ("periodic", "wall", "freestream", "moving", "hot", "cold")
DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
SLICE_FIELDS = ("|u|", "ux", "uy", "uz", "density", "|vorticity|", "T / fill", "flags", "pressure")
VIS = {"lattice": 0x01, "surface": 0x02, "field": 0x04, "streamlines": 0x08, "q_criterion": 0x10, "free_surface": 0x20, "raytrace": 0x40, "particles": 0x80}

DEFAULT = {
    "name": "untitled",
    "domain": [128, 256, 128],
    "flow": {"direction": "+y", "u": 0.075, "re": 1000.0, "length": None},
    "nu": None,
    "boundaries": {f: "periodic" for f in FACES},
    "wall_velocity": [0.0, 0.0, 0.0],
    "init": "freestream",
    "init_noise": 0.0,
    "gravity": [0.0, 0.0, 0.0],
    "objects": [],
    "fills": [],
    "thermal": None,
    "surface_tension": 0.0,
    "particles": None,
    "les": False,
    "forces": False,
    "solver": {"precision": "auto", "lattice": "auto", "collision": "SRT", "ddf_pad": "auto", "device": -1},
    "run": {"steps": 0, "telemetry_every": 100, "slice_every": 100, "frame_fps": 8, "render_budget": 0.2, "checkpoint_every": 0},
    "view": {"modes": ["lattice", "surface", "q_criterion"], "field": 0, "slice_axis": 0, "slice_field": 0, "camera": [-35.0, 25.0, 60.0, 1.0],
             "cloud": {"on": False, "field": 4, "gain": 1.0, "density": 0.15}},
    "si": None,
    "reference_area": "frontal",  # for force coefficients: "frontal", "planform" or a number in cells^2
}


def normalize(case):
    out = copy.deepcopy(DEFAULT)
    for k, v in case.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def features(c):
    ext = set()
    b = c["boundaries"].values()
    if "freestream" in b:
        ext.add("EQUILIBRIUM_BOUNDARIES")
    if "moving" in b or any(o.get("tip_speed") for o in c["objects"]):
        ext.add("MOVING_BOUNDARIES")
    if c["forces"]:
        ext.add("FORCE_FIELD")
    if any(c["gravity"]):
        ext.add("VOLUME_FORCE")
    if c["fills"] or any(o.get("role") == "fluid" for o in c["objects"]):
        ext.add("SURFACE")
    if c["thermal"] or "hot" in b or "cold" in b or any(o.get("role") == "heat" for o in c["objects"]):
        ext |= {"TEMPERATURE", "VOLUME_FORCE"}
    if c["les"]:
        ext.add("SUBGRID")
    if c["particles"]:
        ext.add("PARTICLES")
    return sorted(ext)


def recommended_precision():
    """Fastest measured precision on this machine (from `neurosim bench`), else FP16S."""
    p = build.ROOT / "workspace" / "benchmarks.json"
    if p.exists():
        best = json.loads(p.read_text()).get("recommended")
        if best:
            return best
    return "FP16S"


def solver_variant(c):
    s = c["solver"]
    precision = recommended_precision() if s["precision"] == "auto" else s["precision"]
    lattice = ("D2Q9" if c["domain"][2] == 1 else "D3Q19") if s["lattice"] == "auto" else s["lattice"]
    return build.variant(precision, lattice, s["collision"], features(c), frame=c["view"].get("resolution", (1280, 720)))


def resolve_model(model):
    """'sphere' (built-in) or a path to an imported mesh."""
    if model in geometry.BUILTIN:
        return geometry.builtin(model)
    return geometry.load(model)


def prepare(case, run_dir):
    """Validates the case, writes geometry + case.cfg into run_dir, returns (variant, derived parameters)."""
    c = normalize(case)
    run_dir = Path(run_dir)
    N = [int(n) for n in c["domain"]]
    if min(N) < 1 or N[0] * N[1] * N[2] < 8:
        raise ValueError("domain too small")
    for f, t in c["boundaries"].items():
        if t not in FACE_TYPES:
            raise ValueError(f"boundary {f}: unknown type {t!r}")
    d = np.asarray(DIRS[c["flow"]["direction"]], float)
    u = float(c["flow"]["u"])
    if not 0 < u <= 0.3:
        raise ValueError("lattice velocity must be in (0, 0.3]; 0.05-0.1 is typical")
    flow_axis = int(np.argmax(np.abs(d)))
    lift_axis = 2 if flow_axis != 2 else 1

    lines, derived = {}, {"flow_axis": flow_axis, "lift_axis": lift_axis, "u": u}
    objects, ref_area, sizes = [], 0.0, []
    geo_dir = run_dir / "geometry"
    for k, o in enumerate(c["objects"]):
        p = f"obj{k}_"
        center = np.asarray(o.get("position", [0.5, 0.5, 0.5]), float) * np.asarray(N) - 0.5
        role = o.get("role", "solid")
        lines[p + "role"] = role
        if "shape" in o:  # analytic primitive, voxelized on the host
            shape = o["shape"]
            lines[p + "kind"] = shape
            lines[p + "p"] = center
            if shape == "sphere":
                r = float(o["radius"]); lines[p + "r"] = r; area = math.pi * r * r; sizes.append(2 * r)
            elif shape == "cylinder":
                r = float(o["radius"]); axis = np.asarray(o.get("axis", [0, 0, 1]), float)
                h = float(o.get("height", N[int(np.argmax(np.abs(axis)))]))
                lines[p + "r"] = r; lines[p + "n"] = axis / np.linalg.norm(axis) * h
                ax = int(np.argmax(np.abs(axis)))
                area = math.pi * r * r if ax == flow_axis else 2 * r * h; sizes.append(2 * r)
            elif shape == "cuboid":
                l = np.asarray(o["size"], float); lines[p + "l"] = l
                area = float(np.prod([l[i] for i in range(3) if i != flow_axis])); sizes.append(float(l.max()))
            else:
                raise ValueError(f"unknown shape {shape!r}")
            meta = {"kind": shape, "center": center.tolist()}
        else:  # mesh: placed here in lattice coordinates so solver and viewer use identical geometry
            tris = geometry.place(resolve_model(o["model"]), float(o.get("size", 0.25 * max(N))), center, o.get("rotation", (0, 0, 0)))
            if o.get("on_ground"):  # rest on the z0 wall (one cell above it)
                tris[:, :, 2] += 1.5 - tris[:, :, 2].min()
                center = center.copy(); center[2] = 0.5 * (tris[:, :, 2].min() + tris[:, :, 2].max())
            geo_dir.mkdir(parents=True, exist_ok=True)
            path = geo_dir / f"obj{k}.stl"
            geometry.save_stl(tris, path)
            lines[p + "kind"] = "stl"
            lines[p + "file"] = path.as_posix()
            lines[p + "center"] = center
            if o.get("tip_speed"):  # rotating: angular velocity from tip speed at the largest radius
                axis = np.asarray(o.get("spin_axis", [0, 0, 1]), float)
                axis /= np.linalg.norm(axis)
                rel = tris.reshape(-1, 3) - center
                radius = float(np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1).max())
                lines[p + "omega"] = axis * float(o["tip_speed"]) / max(radius, 1.0)
            ref_axis = lift_axis if c.get("reference_area") == "planform" else flow_axis  # wings: planform, bluff bodies: frontal
            area = geometry.frontal_area(tris, ref_axis) if role == "solid" else 0.0
            sizes.append(float(o.get("size", 0.25 * max(N))))
            meta = {"kind": "stl", "file": path.name, "center": center.tolist(), "triangles": len(tris)}
        if role == "heat":
            lines[p + "T"] = float(o.get("T", 1.5))
        if role == "solid":
            ref_area += area
        objects.append(meta)
    lines["obj_count"] = len(c["objects"])

    L = c["flow"]["length"] or (max(sizes) if sizes else min(N))
    if c["nu"] is not None:
        nu = float(c["nu"])
        re = u * L / nu
    else:
        re = float(c["flow"]["re"])
        nu = u * L / re
    tau = 3 * nu + 0.5
    if tau <= 0.5 or (tau < 0.5001 and not c["les"]):  # very high Re is only stable with the LES model (as in FluidX3D's aircraft setups)
        raise ValueError(f"viscosity too low for this resolution (tau={tau:.6f}); enable LES, lower Re, raise u or refine the grid")
    if isinstance(c.get("reference_area"), (int, float)):
        ref_area = float(c["reference_area"])
    derived.update(nu=nu, re=re, length=L, tau=tau, ref_area=ref_area or None, cells=N[0] * N[1] * N[2], objects=objects)
    if c["si"]:
        si = c["si"]
        dx = si["length_m"] / L
        dt = dx * u / si["velocity_ms"]
        derived["si"] = {"dx": dx, "dt": dt, "force_scale": si.get("density", 1.225) * dx ** 4 / dt ** 2, **si}

    th = c["thermal"] or {}
    pad = c["solver"]["ddf_pad"]
    view = c["view"]
    lines.update({
        "Nx": N[0], "Ny": N[1], "Nz": N[2], "nu": nu, "device": int(c["solver"]["device"]),
        "ddf_pad": 2112 if pad == "auto" else int(pad),
        "fx": c["gravity"][0], "fy": c["gravity"][1], "fz": c["gravity"][2],
        "sigma": c["surface_tension"] if c["fills"] else 0.0,
        "alpha": th.get("alpha", 0.0), "beta": th.get("beta", 0.0), "T_hot": th.get("T_hot", 1.5), "T_cold": th.get("T_cold", 0.5),
        "init_mode": "taylor_green" if c["init"] == "taylor_green" else "uniform",
        "init_u": d * u if c["init"] == "freestream" else [0, 0, 0], "init_noise": c["init_noise"],
        "init_hydrostatic": bool(c["fills"]) or bool(th),
        "inflow_u": d * u, "wall_u": c["wall_velocity"], "track_forces": bool(c["forces"]),
        "particles": (c["particles"] or {}).get("count", 0), "particles_rho": (c["particles"] or {}).get("rho", 1.0),
        "fill_count": len(c["fills"]),
        "steps": int(c["run"]["steps"]), "telemetry_every": int(c["run"]["telemetry_every"]), "slice_every": int(c["run"]["slice_every"]),
        "frame_fps": float(c["run"]["frame_fps"]), "render_budget": float(c["run"].get("render_budget", 0.2)), "checkpoint_every": int(c["run"]["checkpoint_every"]),
        "export_every": int(c["run"].get("export_every", 0)),
        "vis_modes": sum(VIS[m] for m in view["modes"]), "vis_field": int(view["field"]), "camera": view["camera"],
        "slice_axis": int(view["slice_axis"]), "slice_field": int(view["slice_field"]), "u_ref": u,
        "cloud": bool(view.get("cloud", {}).get("on")), "cloud_field": int(view.get("cloud", {}).get("field", 0)),
        "cloud_gain": float(view.get("cloud", {}).get("gain", 1.0)), "cloud_density": float(view.get("cloud", {}).get("density", 1.0)),
        "has_T": "TEMPERATURE" in features(c), "has_phi": "SURFACE" in features(c) and "TEMPERATURE" not in features(c),
    })
    for f in FACES:
        lines["face_" + f] = c["boundaries"][f]
    for k, b in enumerate(c["fills"]):
        lines[f"fill{k}"] = [b[i] * N[i % 3] for i in range(6)]
    if c.get("restart"):
        lines["restart"] = Path(c["restart"]).as_posix()

    v = solver_variant(c)
    if c["thermal"] is None and "TEMPERATURE" in v["extensions"]:
        raise ValueError("hot/cold boundaries need thermal settings (alpha, beta)")
    if "SURFACE" in v["extensions"] and not (c["fills"] or any(o.get("role") == "fluid" for o in c["objects"])):
        raise ValueError("free surface needs a liquid fill region")
    mem = N[0] * N[1] * N[2] * bytes_per_cell(v) / 2 ** 20
    derived.update(memory_mb=mem, variant=v)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "case.cfg").write_text("\n".join(f"{k} = {_fmt(val)}" for k, val in lines.items()) + "\n")
    return v, derived, c


def bytes_per_cell(v):
    q = {"D2Q9": 9, "D3Q15": 15, "D3Q19": 19, "D3Q27": 27}[v["lattice"]]
    b = q * (4 if v["precision"] == "FP32" else 2) + 17
    ext = v["extensions"]
    b += 12 * ("FORCE_FIELD" in ext) + 12 * ("SURFACE" in ext) + (7 * (4 if v["precision"] == "FP32" else 2) + 4) * ("TEMPERATURE" in ext)
    return b


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(float(x)) for x in v) + "]"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)
