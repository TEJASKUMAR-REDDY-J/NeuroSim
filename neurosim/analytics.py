"""One analytics layer over the run history: simulation runs and AI experiments, read from their files.

Simulation metrics come from solver telemetry (runtime, MLUP/s, memory, forces, convergence); AI metrics
come from the measured surrogate evaluations. The GUI (both modes) and scripts read the same rows.
"""
import datetime
import json
from pathlib import Path

import numpy as np

from . import ai, case as case_mod, runner

_cache = {}


def _records(run_dir):
    p = Path(run_dir) / "telemetry.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_bytes().splitlines():
        if b'"type":"step"' in line or b'"type":"start"' in line or b'"type":"finished"' in line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def sim_metrics(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    d = meta.get("derived", {})
    recs = _records(run_dir)
    steps = [r for r in recs if r["type"] == "step"]
    start = next((r for r in recs if r["type"] == "start"), {})
    row = {"type": "simulation", "dir": run_dir.name, "name": run_dir.name[16:].replace("-", " "), "created": meta.get("created"), "status": meta.get("status"),
           "cells": d.get("cells"), "re": d.get("re"), "steps": steps[-1]["t"] if steps else 0, "memory_mb": start.get("memory_mb"), "device": start.get("device")}
    if meta.get("started") and meta.get("ended"):
        row["runtime_s"] = (datetime.datetime.fromisoformat(meta["ended"]) - datetime.datetime.fromisoformat(meta["started"])).total_seconds()
    if steps:
        row["mlups"] = float(np.mean([r["mlups"] for r in steps]))
        ke = np.array([r["ke"] for r in steps])
        if len(ke) >= 8:  # convergence: relative change of kinetic energy over the last quarter of the run
            q = ke[-len(ke) // 4:]
            row["ke_change"] = float((q.max() - q.min()) / max(abs(q[-1]), 1e-30))
        if "force" in steps[-1] and d.get("ref_area"):
            F = np.array([r["force"] for r in steps[len(steps) // 2:]])
            qd = 0.5 * d["u"] ** 2 * d["ref_area"]
            case = json.loads((run_dir / "case.json").read_text())
            sign = np.sign(sum(case_mod.DIRS[case["flow"]["direction"]]))
            row["cd"] = float(np.mean(F[:, d["flow_axis"]]) / qd * sign)
            row["cl"] = float(np.mean(F[:, d["lift_axis"]]) / qd)
    return row


def ai_metrics(exp_dir):
    e = ai.Experiment(exp_dir)
    row = {"type": "ai_experiment", "dir": e.dir.name, "name": e.name, "created": e.meta.get("created"), "status": e.meta.get("status"),
           "cells": e.meta.get("derived", {}).get("cells"), "re": e.meta.get("derived", {}).get("re"), "models": len(e.config["models"])}
    evs = [ev for ev in (e.evaluation(m["id"]) for m in e.config["models"]) if ev]
    if evs:
        best = min(evs, key=lambda ev: ev["aggregate"]["test"]["rel_l2"])
        a, p = best["aggregate"]["test"], best["performance"]
        row.update(best_model=best["label"], rel_l2=a["rel_l2"], mass_err=a.get("mass_err"), stable_steps=best["stable_steps"],
                   horizon_steps=best["horizon_steps"], speedup=p.get("speedup"))
    return row


def row(run_dir):
    """Metrics for one history entry (cached by telemetry/eval modification time)."""
    run_dir = Path(run_dir)
    files = [run_dir / "run.json", run_dir / "telemetry.jsonl", run_dir / "summary.json"]
    key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in files)
    hit = _cache.get(run_dir)
    if hit and hit[0] == key:
        return hit[1]
    r = ai_metrics(run_dir) if ai.is_experiment(run_dir) else sim_metrics(run_dir)
    _cache[run_dir] = (key, r)
    return r


def collect(limit=200):
    rows = []
    for p in runner.list_runs()[:limit]:
        try:
            rows.append(row(p))
        except (OSError, ValueError, KeyError):
            pass
    return rows
