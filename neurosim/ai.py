"""AI Mode backend: neural surrogates of NeuroSim simulations, and how they fail.

An AI experiment is one more entry in the run history (workspace/runs/<stamp>-ai-<slug>/):
  run.json         kind "ai_experiment", provenance and ground-truth parameters (same schema as simulation runs)
  case.json        ground-truth case, carried over from the simulation (geometry, grid, physics, solver)
  experiment.json  dataset settings, train/test conditions, attached models
  dataset/         manifest.json, sim_K.npy [T, C, Z, Y, X] float32, mask_K.npy [Z, Y, X] solid cells
  models/<id>/     weights.pt, train.jsonl (one record per epoch, like solver telemetry), model.json
  eval/<id>.json   measured metrics per step and per simulation; eval/<id>_sK.npy predicted rollouts
  summary.json     surrogate experiment summary

Ground truth is produced by the normal solver runs (one per condition value) with a field-export schedule;
those runs stay in the history as well. Every number shown in AI Mode comes from these files.
"""
import copy
import datetime
import importlib.util
import itertools
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import case as case_mod, runner

KIND = "ai_experiment"
SOLVERS = [("FluidX3D / LBM", True), ("OpenFOAM", False), ("SU2", False)]  # ground-truth sources; only FluidX3D is integrated
FIELDS = ("velocity", "pressure", "density", "temperature", "vorticity")
PARAMS = {  # experiment conditions: what changes between training and testing
    "none": "None: one simulation, split in time",
    "re": "Reynolds number",
    "u": "Inflow velocity (lattice units, fixed viscosity)",
    "geometry": "Geometry",
    "resolution": "Resolution (grid scale factor)",
    "horizon": "Temporal horizon (x recorded steps)",
    "boundary": "Lateral boundaries",
}
NUMERIC = ("re", "u", "resolution", "horizon")
STRESS_TESTS = {
    "interpolation": ("Interpolation", "Test between known training conditions."),
    "extrapolation": ("Extrapolation", "Test beyond the training parameter range."),
    "geometry": ("Unseen geometry", "Train on some geometries, test on a new one."),
    "resolution": ("Resolution shift", "Train at one resolution, test at another."),
    "rollout": ("Long rollout", "Test far more autoregressive steps than seen in training."),
    "boundary": ("Boundary shift", "Test with different boundary conditions."),
    "distribution": ("Distribution shift", "Test against a deliberately different parameter distribution."),
    "custom": ("Custom", "Choose the condition and values yourself."),
}
MODELS = {  # kind: (label, description, available)
    "fno": ("FNO", "Fourier neural operator (spectral convolutions, resolution independent)", True),
    "cnn": ("CNN", "Residual convolutional network (local stencil, translation equivariant)", True),
    "persistence": ("Persistence baseline", "Predicts no change. Any useful surrogate must beat it", True),
    "gnn": ("GNN", "Graph neural network on mesh cells: plugin slot, not in this build", False),
    "nca": ("NCA", "Neural cellular automaton: plugin slot, not in this build", False),
    "torchscript": ("TorchScript model", "Load an exported PyTorch model (.pt)", True),
    "onnx": ("ONNX model", "Load an ONNX model (.onnx) through ONNX Runtime", True),
    "python": ("Python adapter", "Python file with a Surrogate class (optional fit, required predict)", True),
}
EXTERNAL = ("torchscript", "onnx", "python")
DEFAULT_PARAMS = {"fno": {"width": 24, "modes": 12, "depth": 4}, "cnn": {"width": 32, "depth": 4}}
TRAIN_DEFAULTS = {"epochs": 20, "lr": 1e-3, "batch": 8}
STABILITY_THRESHOLD = 0.5  # rollout counts as diverged once the velocity relative L2 error exceeds this


# ====================================================================== experiments


def is_experiment(run_dir):
    try:
        return json.loads((Path(run_dir) / "run.json").read_text()).get("kind") == KIND
    except (OSError, ValueError):
        return False


def list_experiments():
    return [Experiment(p) for p in runner.list_runs() if is_experiment(p)]


class Experiment:
    def __init__(self, d):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "run.json").read_text())
        self.case = json.loads((self.dir / "case.json").read_text())
        self.config = json.loads((self.dir / "experiment.json").read_text())

    @property
    def id(self):
        return self.dir.name

    @property
    def name(self):
        return self.config["name"]

    @classmethod
    def create(cls, case, name=None, source_run=None, stress_test="interpolation"):
        """New experiment from a simulation case (the current setup or a run from the history)."""
        case = case_mod.normalize(copy.deepcopy(case))
        case.pop("restart", None)
        name = name or f"{case.get('name', 'simulation')} surrogate"
        d = runner.new_dir("ai " + name)
        variant, derived, full = case_mod.prepare(case, d / "ground_truth")  # validates the physics once, as a run would
        (d / "case.json").write_text(json.dumps(full, indent=1))
        meta = {"kind": KIND, "created": datetime.datetime.now().isoformat(timespec="seconds"), "status": "created", "derived": derived,
                **runner.provenance(variant)}
        (d / "run.json").write_text(json.dumps(meta, indent=1, default=float))
        cfg = {"name": name, "ground_truth": {"solver": SOLVERS[0][0], "source_run": source_run, "case_name": case.get("name")},
               "stress_test": stress_test, "condition": {}, "dataset": default_dataset(full, derived), "models": []}
        cfg["condition"] = stress_condition(stress_test, full, derived)
        (d / "experiment.json").write_text(json.dumps(cfg, indent=1))
        return cls(d)

    def save(self):
        (self.dir / "experiment.json").write_text(json.dumps(self.config, indent=1))

    def set_status(self, status):
        self.meta["status"] = status
        (self.dir / "run.json").write_text(json.dumps(self.meta, indent=1, default=float))

    # --- stage state, read from disk
    def manifest(self):
        p = self.dir / "dataset" / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def model_meta(self, mid):
        p = self.dir / "models" / mid / "model.json"
        return json.loads(p.read_text()) if p.exists() else None

    def train_log(self, mid, kind="epoch"):
        p = self.dir / "models" / mid / "train.jsonl"
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        return [r for r in recs if r.get("type") == kind]

    def evaluation(self, mid):
        p = self.dir / "eval" / f"{mid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def summary(self):
        p = self.dir / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None

    def model(self, mid):
        return next(m for m in self.config["models"] if m["id"] == mid)

    def add_model(self, kind, params=None, path=None):
        n = 1 + sum(m["kind"] == kind for m in self.config["models"])
        mid = f"{kind}-{n}"
        while any(m["id"] == mid for m in self.config["models"]):
            n += 1; mid = f"{kind}-{n}"
        m = {"id": mid, "kind": kind, "label": MODELS[kind][0] + (f" · {Path(path).name}" if path else ""),
             "params": {**DEFAULT_PARAMS.get(kind, {}), **(params or {})}, "train": dict(TRAIN_DEFAULTS), "path": path}
        self.config["models"].append(m); self.save()
        return m


def default_dataset(case, derived):
    """Sampling that suits the case: about one flow-through to develop, one recorded, 50 snapshots."""
    N = case["domain"]
    through = int(N[derived["flow_axis"]] / derived["u"])
    every = max(10, int(round(through / 50, -1)))
    cells = N[0] * N[1] * N[2]
    fields = ["velocity", "pressure"] + (["temperature"] if case.get("thermal") else [])
    start = int(round(through, -2))
    return {"fields": fields, "every": every, "start": start, "steps": start + 50 * every,
            "downsample": 2 if cells <= 4e6 else 4, "sequence": 3, "val_fraction": 0.1, "time_split": [0.6, 0.1, 0.3], "keep_exports": False}


def stress_condition(kind, case, derived):
    """Train/test values for a stress-test preset, centred on the ground-truth case."""
    re = float(derived["re"])
    r = lambda xs: [float(f"{re * x:.4g}") for x in xs]
    has_obj = bool(case["objects"])
    if kind == "interpolation":
        return {"param": "re", "train": r([0.6, 0.8, 1.0, 1.2, 1.4]), "test": r([0.7, 1.1, 1.3])}
    if kind == "extrapolation":
        return {"param": "re", "train": r([0.6, 0.8, 1.0]), "test": r([1.3, 1.6])}
    if kind == "distribution":  # training clustered low, test drawn log-uniformly over a wider band (seeded)
        test = sorted(np.exp(np.random.default_rng(7).uniform(np.log(0.5), np.log(2.0), 3)).tolist())
        return {"param": "re", "train": r([0.7, 0.8, 0.9, 1.0]), "test": r(test)}
    if kind == "geometry" and has_obj:
        o = case["objects"][0]
        pool = ["cylinder", "cuboid", "sphere"] if "shape" in o else ["sphere", "cube", "cylinder", "torus"]
        return {"param": "geometry", "train": pool[:-1], "test": pool[-1:]}
    if kind == "resolution":
        return {"param": "resolution", "train": [1.0], "test": [2.0]}
    if kind == "rollout":
        return {"param": "horizon", "train": [1.0], "test": [4.0]}
    if kind == "boundary":
        lateral = [f for f in case_mod.FACES if _is_lateral(case, f)]
        cur = case["boundaries"][lateral[0]] if lateral else "freestream"
        return {"param": "boundary", "train": [cur], "test": ["wall" if cur != "wall" else "freestream"]}
    return {"param": "none", "train": [], "test": []}


def condition_for(param, case, derived):
    """Sensible starting values when the user switches the condition parameter."""
    if param == "u":
        u = float(derived["u"])
        f = lambda xs: [float(f"{min(0.3, u * x):.4g}") for x in xs]
        return {"param": "u", "train": f([0.6, 0.8, 1.0, 1.2, 1.4]), "test": f([0.7, 1.1, 1.3])}
    kind = {"re": "interpolation", "geometry": "geometry", "resolution": "resolution", "horizon": "rollout", "boundary": "boundary"}.get(param, "custom")
    return stress_condition(kind, case, derived)


def _is_lateral(case, face):
    axis = "xyz".index(face[0])
    flow_axis = int(np.argmax(np.abs(case_mod.DIRS[case["flow"]["direction"]])))
    return axis != flow_axis and case["domain"][axis] > 1


def parse_values(text, param):
    """'100, 200' or 'a:b:n' (n values from a to b) for numbers; comma separated names otherwise."""
    out = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if param in NUMERIC and part.count(":") == 2:
            a, b, n = part.split(":")
            out += [float(f"{v:.6g}") for v in np.linspace(float(a), float(b), int(n))]
        else:
            out.append(float(part) if param in NUMERIC else part)
    return out


def fmt_value(param, v):
    if v is None:
        return "base case"
    return {"re": f"Re {v:g}", "u": f"u {v:g}", "resolution": f"{v:g}x grid", "horizon": f"{v:g}x horizon", "boundary": f"{v} sides", "geometry": str(v)}.get(param, str(v))


# ====================================================================== dataset


def plan(exp):
    """Simulations the dataset needs: one per condition value (or one, split in time)."""
    cond = exp.config["condition"]
    if cond.get("param", "none") == "none":
        return [{"value": None, "split": "time"}]
    return [{"value": v, "split": "train"} for v in cond["train"]] + [{"value": v, "split": "test"} for v in cond["test"]]


def sim_case(exp, value):
    """Ground-truth case for one condition value: the carried-over case with one parameter changed."""
    c = case_mod.normalize(copy.deepcopy(exp.case))
    ds, p = exp.config["dataset"], exp.config["condition"].get("param", "none")
    steps, every, start = int(ds["steps"]), int(ds["every"]), int(ds["start"])
    if p == "re":
        c["flow"]["re"] = float(value); c["nu"] = None
    elif p == "u":  # same fluid: viscosity fixed, Reynolds number follows the velocity
        nu = exp.meta["derived"]["nu"]; c["flow"]["u"] = float(value); c["nu"] = nu
        c["wall_velocity"] = [w * value / exp.meta["derived"]["u"] for w in c["wall_velocity"]]
    elif p == "geometry" and c["objects"]:
        o = c["objects"][0]
        size = 2 * o["radius"] if "radius" in o else (max(o["size"]) if isinstance(o.get("size"), list) else o.get("size", 0.25 * max(c["domain"])))
        keep = {k: o[k] for k in ("position", "role") if k in o}
        if value in ("cylinder", "cuboid", "sphere") and ("shape" in o or c["domain"][2] == 1):
            o2 = {"shape": value, **keep}
            if value == "cuboid":
                o2["size"] = [size, size, o.get("height", c["domain"][2])]
            else:
                o2["radius"] = size / 2
                if value == "cylinder":
                    o2["axis"] = o.get("axis", [0, 0, 1])
        else:
            o2 = {**{k: v for k, v in o.items() if k not in ("shape", "radius", "axis", "height")}, "model": value, "size": size}
        c["objects"][0] = o2
    elif p == "resolution":
        s = float(value)
        c["domain"] = [n if n == 1 else max(2, int(round(n * s))) for n in c["domain"]]
        for o in c["objects"]:
            for k in ("radius", "height"):
                if k in o:
                    o[k] *= s
            if isinstance(o.get("size"), list):
                o["size"] = [x * s for x in o["size"]]
            elif "size" in o:
                o["size"] *= s
        if c["flow"].get("length"):
            c["flow"]["length"] *= s
        steps, every, start = int(steps * s), max(1, int(every * s)), int(start * s)  # same physical times
    elif p == "horizon":
        steps = start + int((steps - start) * float(value))
    elif p == "boundary":
        for f in case_mod.FACES:
            if _is_lateral(c, f):
                c["boundaries"][f] = value
    c["run"] = {"steps": steps, "telemetry_every": every, "slice_every": 0, "frame_fps": 0, "render_budget": 0.2, "checkpoint_every": 0, "export_every": every}
    c["name"] = f"AI data {exp.name} {fmt_value(p, value)}"
    return c, {"steps": steps, "every": every, "start": start}


def channels(fields, case):
    """Channel names and field groups for the selected fields (2D cases drop uz and keep vorticity z)."""
    two_d = case["domain"][2] == 1
    names, groups = [], {}
    for f in FIELDS:
        if f not in fields:
            continue
        if f == "velocity":
            add = ["ux", "uy"] if two_d else ["ux", "uy", "uz"]
        elif f == "vorticity":
            add = ["wz"] if two_d else ["wx", "wy", "wz"]
        else:
            add = [{"pressure": "p", "density": "rho", "temperature": "T"}[f]]
        groups[f] = list(range(len(names), len(names) + len(add)))
        names += add
    return names, groups


def _curl(u, two_d):
    """Vorticity of u [3, Z, Y, X] (cell units)."""
    d = lambda a, ax: np.gradient(a, axis=ax) if a.shape[ax] > 1 else np.zeros_like(a)
    ux, uy, uz = u
    wz = d(uy, 2) - d(ux, 1)
    if two_d:
        return wz[None]
    return np.stack([d(uz, 1) - d(uy, 0), d(ux, 0) - d(uz, 2), wz])


def _block(a, s, solid):
    """Fluid-weighted block average by s along every axis longer than 1."""
    Z, Y, X = solid.shape
    sz = 1 if Z == 1 else s
    Zc, Yc, Xc = Z // sz * sz, Y // s * s, X // s * s
    w = (~solid[:Zc, :Yc, :Xc]).astype(np.float32)
    a = a[..., :Zc, :Yc, :Xc]
    shp = a.shape[:-3] + (Zc // sz, sz, Yc // s, s, Xc // s, s)
    num = (a * w).reshape(shp).sum(axis=(-5, -3, -1))
    den = w.reshape((Zc // sz, sz, Yc // s, s, Xc // s, s)).sum(axis=(1, 3, 5))
    return num / np.maximum(den, 1e-9), den < 0.5 * sz * s * s


def snapshot(files, fields, two_d, s):
    """Channels [C, Z, Y, X] and solid mask from one solver export."""
    get = {Path(f).name.split("_")[0]: f for f in files}
    rho, u, flags = np.load(get["rho"]), np.load(get["u"]), np.load(get["flags"])
    solid = (flags & 0x01) != 0
    parts = []
    for f in FIELDS:
        if f not in fields:
            continue
        if f == "velocity":
            parts.append(u[:2] if two_d else u)
        elif f == "pressure":
            parts.append(((rho - 1.0) / 3.0)[None])  # lattice units, c_s^2 = 1/3, relative to the reference density
        elif f == "density":
            parts.append(rho[None])
        elif f == "temperature":
            if "T" not in get:
                raise ValueError("temperature selected but the case has no heat transfer")
            parts.append(np.load(get["T"])[None])
        elif f == "vorticity":
            parts.append(_curl(np.where(solid, 0.0, u), two_d))
    x = np.concatenate(parts).astype(np.float32)
    x[:, solid] = 0.0
    if s > 1:
        x, solid = _block(x, s, solid)
        x[:, solid] = 0.0
    return x.astype(np.float32), solid


def estimate(exp):
    """Dataset size and ground-truth cost before anything is generated."""
    ds, sims = exp.config["dataset"], plan(exp)
    C = len(channels(ds["fields"], exp.case)[0])
    bench = json.loads((runner.WORKSPACE / "benchmarks.json").read_text()) if (runner.WORKSPACE / "benchmarks.json").exists() else {}
    mlups = max(bench.get("mlups", {}).values(), default=0) or None
    total_bytes, solver_s, snaps, seqs, raw = 0, 0.0, 0, 0, 0
    for s in sims:
        c, run = sim_case(exp, s["value"])
        N = c["domain"]; f = int(ds["downsample"])
        cells = N[0] * N[1] * N[2]
        cells_ds = max(1, N[0] // f) * max(1, N[1] // f) * (1 if N[2] == 1 else max(1, N[2] // f))
        T = (run["steps"] - run["start"]) // run["every"] + 1
        snaps += T
        seqs += max(0, T - int(ds["sequence"]) + 1)
        total_bytes += T * C * cells_ds * 4
        raw += (run["steps"] // run["every"] + 1) * cells * 17
        if mlups:
            solver_s += cells * run["steps"] / (mlups * 1e6)
    return {"simulations": len(sims), "train": sum(s["split"] == "train" for s in sims), "test": sum(s["split"] == "test" for s in sims),
            "snapshots": snaps, "sequences": seqs, "channels": C, "bytes": total_bytes, "raw_bytes": raw, "solver_s": solver_s if mlups else None}


def generate(exp, progress=lambda f, t: None, cancel=None):
    """Runs the ground-truth simulations through the normal solver and converts their exports."""
    ds = exp.config["dataset"]
    out = exp.dir / "dataset"
    out.mkdir(exist_ok=True)
    sims, entries = plan(exp), []
    names, groups = channels(ds["fields"], exp.case)
    for k, s in enumerate(sims):
        c, run_cfg = sim_case(exp, s["value"])
        two_d = c["domain"][2] == 1
        label = fmt_value(exp.config["condition"].get("param"), s["value"])
        progress(k / len(sims), f"Simulation {k + 1}/{len(sims)} · {label} · preparing")
        r = runner.Run.create(c)
        r.start(log=lambda m: progress(k / len(sims), f"Simulation {k + 1}/{len(sims)} · {m}"))
        while r.status not in ("finished", "failed", "stopped"):
            r.poll(); time.sleep(0.1)
            st = [x for x in r.records if x["type"] == "step"]
            if st:
                progress((k + min(1.0, st[-1]["t"] / max(1, run_cfg["steps"]))) / len(sims),
                         f"Simulation {k + 1}/{len(sims)} · {label} · step {st[-1]['t']:,}/{run_cfg['steps']:,} · {st[-1]['steps_per_s']:.0f} steps/s")
            if cancel is not None and cancel.is_set():
                r.stop(); raise RuntimeError("cancelled")
        r.stop()
        if r.status == "failed":
            raise RuntimeError(f"ground-truth simulation failed: {r.error}")
        exports = [x for x in r.records if x["type"] == "export" and x["t"] >= run_cfg["start"]]
        frames, mask = [], None
        for e in exports:
            x, solid = snapshot(e["files"], ds["fields"], two_d, int(ds["downsample"]))
            frames.append(x); mask = solid if mask is None else mask | solid
        if not ds.get("keep_exports"):
            for e in [x for x in r.records if x["type"] == "export"]:
                for f in e["files"]:
                    Path(f).unlink(missing_ok=True)  # full-resolution copies; the dataset keeps the converted fields
        X = np.stack(frames)
        X[:, :, mask] = 0.0
        np.save(out / f"sim_{k}.npy", X); np.save(out / f"mask_{k}.npy", mask)
        steps = [x for x in r.records if x["type"] == "step"]
        start = next(x for x in r.records if x["type"] == "start")
        mlups = float(np.mean([x["mlups"] for x in steps])) if steps else 0.0
        entries.append({"file": f"sim_{k}.npy", "mask": f"mask_{k}.npy", "value": s["value"], "label": label, "split": s["split"],
                        "run": r.dir.name, "times": [e["t"] for e in exports], "every": run_cfg["every"], "grid": list(X.shape[2:][::-1]),
                        "solver_grid": c["domain"], "solver_cells": int(np.prod(c["domain"])), "solver_mlups": mlups,
                        "solver_s_per_step": int(np.prod(c["domain"])) / (mlups * 1e6) if mlups else None, "solver_memory_mb": start.get("memory_mb"),
                        "device": start.get("device"), "re": r.meta["derived"]["re"]})
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "channels": names, "groups": groups,
                "fields": ds["fields"], "param": exp.config["condition"].get("param", "none"), "dims": 2 if exp.case["domain"][2] == 1 else 3,
                "settings": ds, "sims": entries}
    manifest["stats"] = _stats(out, manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    exp.set_status("dataset ready")
    progress(1.0, f"Dataset ready: {len(entries)} simulations")
    return manifest


def _segments(manifest):
    """(sim, first, last snapshot index, split) used for training windows and evaluation rollouts."""
    segs = []
    for k, s in enumerate(manifest["sims"]):
        T = len(s["times"])
        if s["split"] == "time":
            a, b, _ = manifest["settings"]["time_split"]
            t1, t2 = max(2, int(round(T * a))), max(3, int(round(T * (a + b))))
            segs += [(k, 0, t1 - 1, "train"), (k, t1 - 1, t2 - 1, "val"), (k, t2 - 1, T - 1, "test")]
        else:
            segs.append((k, 0, T - 1, s["split"]))
    return segs


def _stats(out, manifest):
    """Per-channel mean/std and one-step increment std over the training data (fluid cells)."""
    acc = []
    for k, a, b, split in _segments(manifest):
        if split != "train":
            continue
        X = np.load(out / manifest["sims"][k]["file"], mmap_mode="r")[a:b + 1]
        fluid = ~np.load(out / manifest["sims"][k]["mask"])
        acc.append((X[:, :, fluid], np.diff(X, axis=0)[:, :, fluid]))
    v = np.concatenate([x.transpose(1, 0, 2).reshape(x.shape[1], -1) for x, _ in acc], axis=1)
    dv = np.concatenate([d.transpose(1, 0, 2).reshape(d.shape[1], -1) for _, d in acc], axis=1)
    cond = [s["value"] for s in manifest["sims"] if s["split"] == "train" and isinstance(s["value"], (int, float))]
    return {"mean": v.mean(1).tolist(), "std": np.maximum(v.std(1), 1e-8).tolist(), "dstd": np.maximum(dv.std(1), 1e-10).tolist(),
            "cond_mean": float(np.mean(cond)) if cond else 0.0, "cond_std": float(np.std(cond)) if len(cond) > 1 and np.std(cond) > 0 else 1.0}


def load_sim(exp, k):
    m = exp.manifest()
    s = m["sims"][k]
    return np.load(exp.dir / "dataset" / s["file"], mmap_mode="r"), np.load(exp.dir / "dataset" / s["mask"])


def _sq(a, dims):
    """[..., Z, Y, X] -> [..., Y, X] for 2D datasets."""
    return a[..., 0, :, :] if dims == 2 else a


def _cond(manifest, k):
    v = manifest["sims"][k]["value"]
    return [float(v)] if manifest["param"] in ("re", "u") else []


# ====================================================================== models


def _torch():
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch


def build_net(kind, cin, cout, dims, p):
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional
    Conv = nn.Conv2d if dims == 2 else nn.Conv3d

    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            w = int(p["width"])
            s.inp = Conv(cin, w, 3, padding=1, padding_mode="replicate")
            s.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), Conv(w, w, 3, padding=1, padding_mode="replicate"), nn.GELU(), Conv(w, w, 3, padding=1, padding_mode="replicate"))
                                     for _ in range(int(p["depth"])))
            s.out = Conv(w, cout, 1)
            nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)  # starts as the persistence baseline

        def forward(s, x):
            h = s.inp(x)
            for b in s.blocks:
                h = h + b(h)
            return s.out(F.gelu(h))

    class Spectral(nn.Module):
        def __init__(s, w, modes):
            super().__init__()
            s.modes = modes
            s.weight = nn.Parameter(torch.randn(2 ** (dims - 1), w, w, *([modes] * dims), dtype=torch.cfloat) / (w * w))

        def forward(s, x):
            S = x.shape[2:]
            ax = tuple(range(2, x.ndim))
            Xf = torch.fft.rfftn(x, dim=ax)
            out = torch.zeros(x.shape[0], s.weight.shape[2], *Xf.shape[2:], dtype=Xf.dtype)
            for c, signs in enumerate(itertools.product((0, 1), repeat=dims - 1)):  # low modes sit in 2^(d-1) corners of the half spectrum
                sl, wsl = [slice(None), slice(None)], [c, slice(None), slice(None)]
                for d, sg in enumerate(signs):
                    m = min(s.modes, S[d] // 2)
                    sl.append(slice(0, m) if sg == 0 else slice(S[d] - m, S[d])); wsl.append(slice(0, m))
                m = min(s.modes, Xf.shape[-1]); sl.append(slice(0, m)); wsl.append(slice(0, m))
                out[tuple(sl)] = torch.einsum("bi...,io...->bo...", Xf[tuple(sl)], s.weight[tuple(wsl)])
            return torch.fft.irfftn(out, s=S, dim=ax)

    class FNO(nn.Module):
        def __init__(s):
            super().__init__()
            w, L = int(p["width"]), int(p["depth"])
            s.lift = Conv(cin + dims, w, 1)
            s.spec = nn.ModuleList(Spectral(w, int(p["modes"])) for _ in range(L))
            s.pw = nn.ModuleList(Conv(w, w, 1) for _ in range(L))
            s.proj = nn.Sequential(Conv(w, 2 * w, 1), nn.GELU(), Conv(2 * w, cout, 1))
            nn.init.zeros_(s.proj[2].weight); nn.init.zeros_(s.proj[2].bias)

        def forward(s, x):
            S = x.shape[2:]
            grid = torch.meshgrid(*[torch.linspace(0, 1, n) for n in S], indexing="ij")
            h = s.lift(torch.cat([x, torch.stack(grid).unsqueeze(0).expand(x.shape[0], -1, *S)], 1))
            pad = [q for n in reversed(S) for q in (0, max(2, n // 8))]  # non-periodic domain: pad before the spectral layers
            h = F.pad(h, pad)
            for k, (sp, pw) in enumerate(zip(s.spec, s.pw)):
                h = sp(h) + pw(h)
                if k < len(s.spec) - 1:
                    h = F.gelu(h)
            h = h[(slice(None), slice(None)) + tuple(slice(0, n) for n in S)]
            return s.proj(h)

    return {"cnn": CNN, "fno": FNO}[kind]()


class Surrogate:
    """One interface for every model: step(x [C, *S] physical units, solid mask [*S], condition list) -> next state."""

    def __init__(self, exp, m):
        self.exp, self.m, self.kind = exp, m, m["kind"]
        man = exp.manifest()
        self.dims, st = man["dims"], man["stats"]
        self.mean, self.std, self.dstd = (np.asarray(st[k], np.float32) for k in ("mean", "std", "dstd"))
        self.cmean, self.cstd = st["cond_mean"], st["cond_std"]
        self.C = len(man["channels"])
        self.net, self.fn = None, None
        if self.kind in ("cnn", "fno"):
            torch = _torch()
            meta = exp.model_meta(m["id"])
            if not meta:
                raise RuntimeError(f"{m['label']} is not trained yet")
            self.net = build_net(self.kind, meta["cin"], self.C, self.dims, m["params"])
            self.net.load_state_dict(torch.load(exp.dir / "models" / m["id"] / "weights.pt", weights_only=True))
            self.net.eval()
        elif self.kind == "torchscript":
            torch = _torch()
            mod = torch.jit.load(m["path"]).eval()
            self.fn = lambda x: mod(torch.from_numpy(x)).detach().numpy()
        elif self.kind == "onnx":
            import onnxruntime as ort
            sess = ort.InferenceSession(m["path"], providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            self.fn = lambda x: sess.run(None, {name: x})[0]
        elif self.kind == "python":
            self.fn = python_adapter(exp, m).predict

    def parameters(self):
        if self.net is not None:
            return sum(p.numel() for p in self.net.parameters())
        return None

    def rollout(self, x0, mask, cond, n):
        """n autoregressive steps from x0; returns [n+1, C, *S] and the wall time of the n steps."""
        out = np.empty((n + 1,) + x0.shape, np.float32)
        out[0] = x0
        fluid = (~mask).astype(np.float32)
        if self.kind == "persistence":
            t0 = time.perf_counter()
            out[1:] = x0
            return out, time.perf_counter() - t0
        if self.net is not None:
            torch = _torch()
            sh = (-1,) + (1,) * self.dims
            mean, std, k = (torch.from_numpy(a.reshape(sh)) for a in (self.mean, self.std, self.dstd / self.std))
            extra = [torch.from_numpy(mask[None].astype(np.float32))] + [torch.full((1,) + mask.shape, (c - self.cmean) / self.cstd) for c in cond]
            extra = torch.cat(extra).unsqueeze(0)
            fl = torch.from_numpy(fluid)
            with torch.inference_mode():
                x = ((torch.from_numpy(np.array(x0, np.float32)) - mean) / std).unsqueeze(0)
                self.net(torch.cat([x, extra], 1))  # warm-up (allocations, kernel selection) is not timed
                t0 = time.perf_counter()
                for i in range(n):
                    x = x + self.net(torch.cat([x, extra], 1)) * k * fl
                    out[i + 1] = (x[0] * std + mean).numpy()
                dt = time.perf_counter() - t0
            out[1:, :, mask] = 0.0
            return out, dt
        extra = np.concatenate([mask[None].astype(np.float32)] + [np.full((1,) + mask.shape, c, np.float32) for c in cond])
        x = x0
        self.fn(np.concatenate([x, extra])[None].astype(np.float32))
        t0 = time.perf_counter()
        for i in range(n):
            x = np.asarray(self.fn(np.concatenate([x, extra])[None].astype(np.float32)), np.float32).reshape(x0.shape)
            x = np.where(mask, 0.0, x).astype(np.float32)
            out[i + 1] = x
        return out, time.perf_counter() - t0


_ADAPTERS = {}


def python_adapter(exp, m, refit=False):
    """Loads a Python adapter (user file with a Surrogate class) and calls its optional fit() on the training data.
    Kept per process: fitting happens again after the app restarts."""
    key = (str(exp.dir), m["id"])
    if refit or key not in _ADAPTERS:
        man = exp.manifest()
        spec = importlib.util.spec_from_file_location(Path(m["path"]).stem, m["path"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ad = mod.Surrogate(man)
        if hasattr(ad, "fit"):
            data = []
            for k, a, b, sp in _segments(man):
                if sp == "train":
                    X, mask = load_sim(exp, k)
                    data.append((_sq(np.asarray(X[a:b + 1]), man["dims"]), _sq(mask, man["dims"]), _cond(man, k)))
            ad.fit(data)
        _ADAPTERS[key] = ad
    return _ADAPTERS[key]


def train(exp, mid, progress=lambda f, t: None, cancel=None):
    """Trains a built-in model with autoregressive unrolling over each sequence (loss on fluid cells)."""
    m = exp.model(mid)
    man = exp.manifest()
    if not man:
        raise RuntimeError("generate the dataset first")
    out = exp.dir / "models" / mid
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("")
    if m["kind"] in ("persistence", "torchscript", "onnx"):
        (out / "model.json").write_text(json.dumps({"kind": m["kind"], "trained": False, "train_seconds": 0.0}))
        return
    segs = _segments(man)
    if m["kind"] == "python":
        progress(0.1, "Python adapter: fit()")
        t0 = time.time()
        ad = python_adapter(exp, m, refit=True)
        (out / "model.json").write_text(json.dumps({"kind": "python", "trained": hasattr(ad, "fit"), "train_seconds": time.time() - t0}))
        return
    torch = _torch()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    st, dims, L = man["stats"], man["dims"], max(2, int(man["settings"]["sequence"]))
    sh = (-1,) + (1,) * dims
    mean, std = (torch.tensor(st[k], dtype=torch.float32).reshape(sh) for k in ("mean", "std"))
    kdelta = (torch.tensor(st["dstd"]) / torch.tensor(st["std"])).float().reshape(sh)
    # ponytail: whole dataset in memory; stream windows from the .npy memmaps if datasets outgrow RAM
    sims, windows = {}, {"train": [], "val": []}
    for k, a, b, split in segs:
        if split not in ("train", "val"):
            continue
        if k not in sims:
            X, mask = load_sim(exp, k)
            x = (torch.from_numpy(_sq(np.array(X, np.float32), dims)) - mean) / std
            mk = torch.from_numpy(_sq(mask, dims))
            cond = [torch.full(mk.shape, (c - st["cond_mean"]) / st["cond_std"]) for c in _cond(man, k)]
            sims[k] = (x, torch.stack([mk.float()] + cond), (~mk).float())
        windows[split] += [(k, i) for i in range(a, b - L + 2)]
    if man["param"] != "none" and not windows["val"]:  # hold out a fraction of the training sequences
        rng.shuffle(windows["train"])
        nv = int(round(len(windows["train"]) * float(man["settings"]["val_fraction"])))
        windows["val"], windows["train"] = windows["train"][:nv], windows["train"][nv:]
    if not windows["train"]:
        raise RuntimeError("no training sequences: record more snapshots or shorten the sequence length")
    cin = len(man["channels"]) + 1 + len(_cond(man, next(iter(sims))))
    net = build_net(m["kind"], cin, len(man["channels"]), dims, m["params"])
    tr = m["train"]
    opt = torch.optim.Adam(net.parameters(), lr=float(tr["lr"]))
    E, B = int(tr["epochs"]), int(tr["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tr["lr"]), total_steps=E * ((len(windows["train"]) + B - 1) // B))

    def batch_loss(batch, grad, persistence=False):
        x = torch.stack([sims[k][0][i] for k, i in batch])
        extra = torch.stack([sims[k][1] for k, _ in batch])
        fl = torch.stack([sims[k][2] for k, _ in batch]).unsqueeze(1)
        loss = 0.0
        for j in range(1, L):
            if not persistence:
                x = x + net(torch.cat([x, extra], 1)) * kdelta * fl
            y = torch.stack([sims[k][0][i + j] for k, i in batch])
            loss = loss + (((x - y) / kdelta * fl) ** 2).sum() / (fl.sum() * x.shape[1])  # in units of the typical one-step change
        return loss / (L - 1)

    t0, best, log = time.time(), float("inf"), open(out / "train.jsonl", "a")
    ref = windows["val"] or windows["train"]
    with torch.no_grad():  # same loss for "no change" predictions: the level a model must get below
        base = float(np.mean([batch_loss(ref[i:i + B], False, True).item() for i in range(0, len(ref), B)]))
    log.write(json.dumps({"type": "baseline", "persistence_loss": base}) + "\n")
    for e in range(E):
        net.train(); rng.shuffle(windows["train"]); tl = []
        for i in range(0, len(windows["train"]), B):
            if cancel is not None and cancel.is_set():
                log.close(); raise RuntimeError("cancelled")
            loss = batch_loss(windows["train"][i:i + B], True)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl.append(loss.item())
        net.eval()
        with torch.no_grad():
            vl = [batch_loss(windows["val"][i:i + B], False).item() for i in range(0, len(windows["val"]), B)]
        rec = {"type": "epoch", "epoch": e + 1, "train_loss": float(np.mean(tl)), "val_loss": float(np.mean(vl)) if vl else None,
               "seconds": time.time() - t0, "lr": sched.get_last_lr()[0]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        score = rec["val_loss"] if rec["val_loss"] is not None else rec["train_loss"]
        if score <= best:
            best = score; torch.save(net.state_dict(), out / "weights.pt")
        progress((e + 1) / E, f"{m['label']} · epoch {e + 1}/{E} · train {rec['train_loss']:.3g}" + (f" · val {rec['val_loss']:.3g}" if vl else ""))
    log.close()
    params = sum(p.numel() for p in net.parameters())
    (out / "model.json").write_text(json.dumps({"kind": m["kind"], "trained": True, "cin": cin, "parameters": params, "train_seconds": time.time() - t0,
                                                "epochs": E, "sequences": len(windows["train"]), "val_sequences": len(windows["val"]),
                                                "best_loss": best, "unroll": L - 1, "device": "cpu (PyTorch)"}))
    exp.set_status("trained")


# ====================================================================== evaluation


def step_metrics(pred, gt, solid, groups):
    """Measured errors of one predicted state against the ground truth, fluid cells only."""
    fl = ~solid
    r = {}
    if "velocity" in groups:
        v = groups["velocity"]
        g = gt[v][:, fl]
        e = pred[v][:, fl] - g
        em, gm = np.sqrt((e ** 2).sum(0)), np.sqrt((g ** 2).sum(0))
        r.update(rel_l2=float(np.sqrt((e ** 2).sum() / max((g ** 2).sum(), 1e-30))), rel_l1=float(em.sum() / max(gm.sum(), 1e-30)),
                 max_err=float(em.max() / max(gm.max(), 1e-30)), mse=float((e ** 2).mean()), mae=float(np.abs(e).mean()))
        mom_p, mom_g = pred[v][:, fl].sum(1), g.sum(1)
        r["momentum_err"] = float(np.linalg.norm(mom_p - mom_g) / max(np.linalg.norm(mom_g), 1e-30))
        ke_p, ke_g = float((pred[v][:, fl] ** 2).sum()), float((g ** 2).sum())
        r["energy_err"] = abs(ke_p - ke_g) / max(ke_g, 1e-30)
    for name, ch in groups.items():
        pv, g = pred[ch][:, fl], gt[ch][:, fl]
        ref = g if name in ("velocity", "vorticity") else g - g.mean()  # scalars: relative to their fluctuation
        r[f"rel_l2_{name}"] = float(np.sqrt(((pv - g) ** 2).sum() / max((ref ** 2).sum(), 1e-30)))
    rho = None
    if "density" in groups:
        rho = lambda a: a[groups["density"][0]][fl]
    elif "pressure" in groups:
        rho = lambda a: 1.0 + 3.0 * a[groups["pressure"][0]][fl]
    if rho:
        mp, mg = float(rho(pred).sum()), float(rho(gt).sum())
        r["mass_err"] = abs(mp - mg) / mg
    if "rel_l2" not in r:
        r["rel_l2"] = r[f"rel_l2_{next(iter(groups))}"]
    r["finite"] = bool(np.isfinite(pred).all())
    return r


def evaluate(exp, mid, progress=lambda f, t: None):
    """Autoregressive rollouts from each simulation's first state; every metric measured, nothing assumed."""
    m, man = exp.model(mid), exp.manifest()
    sur = Surrogate(exp, m)
    dims, groups = man["dims"], man["groups"]
    out = exp.dir / "eval"
    out.mkdir(exist_ok=True)
    segs = [s for s in _segments(man) if s[3] in ("train", "test")]
    results = []
    for n, (k, a, b, split) in enumerate(segs):
        X, mask = load_sim(exp, k)
        X = _sq(np.asarray(X[a:b + 1], np.float32), dims); mk = _sq(mask, dims)
        progress(n / len(segs), f"{m['label']} · rollout {n + 1}/{len(segs)} · {man['sims'][k]['label']} ({split})")
        pred, secs = sur.rollout(X[0], mk, _cond(man, k), len(X) - 1)
        per = [step_metrics(pred[i], X[i], mk, groups) for i in range(1, len(X))]
        every = man["sims"][k]["every"]
        err = np.array([p["rel_l2"] if p["finite"] else np.inf for p in per])
        bad = np.nonzero(~(err <= STABILITY_THRESHOLD))[0]
        stable = int(bad[0]) * every if bad.size else len(per) * every
        np.save(out / f"{mid}_s{k}_{split}.npy", pred.astype(np.float16))
        s = man["sims"][k]
        gt_s = s["solver_s_per_step"] * len(per) * every if s.get("solver_s_per_step") else None
        results.append({"sim": k, "split": split, "value": s["value"], "label": s["label"], "first": a, "steps": [i * every for i in range(1, len(X))],
                        "per_step": {key: [p.get(key) for p in per] for key in per[0] if key != "finite"},
                        "stable_steps": stable, "horizon_steps": len(per) * every, "diverged": bool(bad.size),
                        "mean": {key: float(np.mean([p[key] for p in per if p["finite"]])) if any(p["finite"] for p in per) else None
                                 for key in per[0] if key != "finite"},
                        "final": per[-1], "surrogate_s": secs, "solver_s": gt_s, "file": f"{mid}_s{k}_{split}.npy",
                        "grid": list(X.shape[1:][::-1]), "solver_grid": s["solver_grid"]})
    params = sur.parameters()
    sim0 = man["sims"][0]
    state_bytes = 2 * len(man["channels"]) * int(np.prod(results[0]["grid"])) * 4
    meta = exp.model_meta(mid) or {}
    perf = {"parameters": params, "parameter_mb": params * 4 / 2 ** 20 if params else None, "state_mb": state_bytes / 2 ** 20,
            "solver_memory_mb": sim0.get("solver_memory_mb"), "train_seconds": meta.get("train_seconds"),
            "runtime": {"torchscript": "PyTorch (TorchScript), CPU", "onnx": "ONNX Runtime, CPU", "python": "Python adapter", "persistence": "none"}.get(sur.kind, "PyTorch, CPU")}
    test = [r for r in results if r["split"] == "test"] or results
    agg = lambda rs, key: float(np.mean([r["mean"][key] for r in rs if r["mean"].get(key) is not None])) if any(r["mean"].get(key) is not None for r in rs) else None
    sur_s, sol_s = sum(r["surrogate_s"] for r in test), sum(r["solver_s"] or 0 for r in test)
    perf.update(surrogate_s=sur_s, solver_s=sol_s or None, speedup=(sol_s / sur_s) if sol_s and sur_s > 0 else None,
                step_ms=1e3 * sur_s / max(1, sum(len(r["steps"]) for r in test)))
    mem_model = (perf["parameter_mb"] or 0) + perf["state_mb"]
    perf["memory_ratio"] = perf["solver_memory_mb"] / mem_model if perf["solver_memory_mb"] and mem_model else None
    worst = max(test, key=lambda r: r["mean"]["rel_l2"] if r["mean"]["rel_l2"] is not None else np.inf)
    keys = ["rel_l1", "rel_l2", "max_err", "mse", "mae", "mass_err", "momentum_err", "energy_err"] + [f"rel_l2_{g}" for g in groups]
    ev = {"model": mid, "label": m["label"], "kind": m["kind"], "evaluated": datetime.datetime.now().isoformat(timespec="seconds"),
          "threshold": STABILITY_THRESHOLD, "sims": results, "performance": perf,
          "aggregate": {split: {k_: agg([r for r in results if r["split"] == split], k_) for k_ in keys} for split in ("train", "test")},
          "stable_steps": min(r["stable_steps"] for r in test), "horizon_steps": max(r["horizon_steps"] for r in test),
          "worst": {"label": worst["label"], "rel_l2": worst["mean"]["rel_l2"]}}
    (out / f"{mid}.json").write_text(json.dumps(ev, indent=1, default=float))
    exp.set_status("evaluated")
    write_summary(exp)
    progress(1.0, f"{m['label']} evaluated")
    return ev


def write_summary(exp):
    """Surrogate experiment summary from the measured evaluations."""
    man = exp.manifest()
    rows = []
    for m in exp.config["models"]:
        ev = exp.evaluation(m["id"])
        if not ev:
            continue
        a, p = ev["aggregate"]["test"], ev["performance"]
        rows.append({"model": m["label"], "id": m["id"], "rel_l2": a["rel_l2"], "rel_l1": a["rel_l1"], "max_err": a["max_err"], "mass_err": a.get("mass_err"),
                     "momentum_err": a.get("momentum_err"), "stable_steps": ev["stable_steps"], "horizon_steps": ev["horizon_steps"],
                     "speedup": p["speedup"], "memory_ratio": p["memory_ratio"], "worst": ev["worst"]["label"], "train_seconds": p.get("train_seconds")})
    sims = man["sims"]
    s = {"ground_truth": exp.config["ground_truth"]["solver"], "case": exp.config["ground_truth"].get("case_name"),
         "stress_test": STRESS_TESTS.get(exp.config.get("stress_test"), ("Custom",))[0], "condition": PARAMS.get(man["param"], man["param"]),
         "train_sims": sum(x["split"] in ("train", "time") for x in sims), "test_sims": sum(x["split"] in ("test", "time") for x in sims),
         "snapshots": sum(len(x["times"]) for x in sims), "grid": sims[0]["grid"], "solver_grid": sims[0]["solver_grid"], "models": rows}
    (exp.dir / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


# ====================================================================== 3D comparison (FluidX3D renderer)


def viewer_case(exp, k):
    """Case for a render-only solver process at the dataset grid of simulation k (same build variant as the ground truth)."""
    man = exp.manifest()
    c, _ = sim_case(exp, man["sims"][k]["value"])
    X, Y, Z = man["sims"][k]["grid"]
    c["domain"] = [X, Y, Z]
    c["objects"], c["particles"] = [], None
    c["flow"]["length"] = min(n for n in (X, Y, Z) if n > 1)
    c["nu"] = None; c["flow"]["re"] = min(c["flow"]["re"], 100.0)  # irrelevant (never stepped) but must be a valid lattice setup
    c["run"] = {"steps": 0, "telemetry_every": 10 ** 9, "slice_every": 0, "frame_fps": 0, "render_budget": 1.0, "checkpoint_every": 0}
    c["view"]["modes"] = ["lattice", "surface", "field"] if Z == 1 else ["lattice", "surface", "q_criterion"]
    c["name"] = "viewer"
    return c


def write_snapshot(path, x, mask, groups, scale=1.0):
    """Raw rho/u/flags for the solver's `load` command from channels [C, Z, Y, X]."""
    Z, Y, X = mask.shape
    u = np.zeros((3, Z, Y, X), np.float32)
    v = groups.get("velocity", [])
    u[:len(v)] = x[v] * scale
    if "density" in groups:
        rho = x[groups["density"][0]]
    elif "pressure" in groups:
        rho = 1.0 + 3.0 * x[groups["pressure"][0]] * scale
    else:
        rho = np.ones((Z, Y, X), np.float32)
    u[:, mask] = 0.0
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(np.ascontiguousarray(rho, np.float32).tobytes()); f.write(u.tobytes()); f.write(mask.astype(np.uint8).tobytes())
    os.replace(tmp, path)


def predictions(exp, mid, k, split):
    p = exp.dir / "eval" / f"{mid}_s{k}_{split}.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def run_all(exp, progress=lambda f, t: None, cancel=None, models=None):
    """Dataset (if missing), then train and evaluate every attached model."""
    if not exp.manifest():
        generate(exp, lambda f, t: progress(0.4 * f, t), cancel)
    ms = models or [m["id"] for m in exp.config["models"]]
    for i, mid in enumerate(ms):
        base = 0.4 + 0.6 * i / max(1, len(ms))
        train(exp, mid, lambda f, t: progress(base + 0.45 * f / len(ms), t), cancel)
        evaluate(exp, mid, lambda f, t: progress(base + (0.45 + 0.15 * f) / len(ms), t))
    progress(1.0, "Experiment complete")


def remove_model(exp, mid):
    exp.config["models"] = [m for m in exp.config["models"] if m["id"] != mid]
    exp.save()
    for p in [exp.dir / "models" / mid] + list((exp.dir / "eval").glob(f"{mid}*")):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
