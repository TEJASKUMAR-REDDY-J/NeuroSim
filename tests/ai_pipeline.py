"""AI Mode end to end, headless: experiment from a simulation case -> dataset from real solver runs ->
CNN, FNO, persistence baseline and Python adapter -> rollout evaluation -> summary -> 3D viewer snapshot.
Usage: python -m tests.ai_pipeline"""
import copy
import sys
import time

import numpy as np

from neurosim import ai, case as case_mod, presets, runner

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


g = np.random.default_rng(1).normal(size=(3, 8, 8)).astype(np.float32); solid = np.zeros((8, 8), bool); solid[3:5, 3:5] = True
grp = {"velocity": [0, 1], "pressure": [2]}
m0, m1 = ai.step_metrics(g, g, solid, grp), ai.step_metrics(g * 1.1, g, solid, grp)
check("metric definitions", m0["rel_l2"] == 0 and m0["mae"] == 0 and abs(m1["rel_l2"] - 0.1) < 1e-5 and abs(m1["rel_l1"] - 0.1) < 1e-5, (m1["rel_l2"], m1["rel_l1"]))

case = case_mod.normalize(copy.deepcopy(presets.PRESETS["Cylinder 2D: vortex street (AI Mode starter)"]))
case.update(name="ai pipeline test", domain=[64, 256, 1])
case["objects"][0]["radius"] = 6
exp = ai.Experiment.create(case, "pipeline test", stress_test="interpolation")
exp.config["condition"] = {"param": "re", "train": [100.0, 160.0], "test": [130.0]}
exp.config["dataset"].update(every=100, start=2000, steps=4000, downsample=2, sequence=3)
exp.save()
est = ai.estimate(exp)
check("estimate before generation", est["simulations"] == 3 and est["snapshots"] == 3 * 21 and est["bytes"] == 3 * 21 * 3 * 32 * 128 * 4, est)

t0 = time.time()
man = ai.generate(exp, lambda f, t: None)
X, mask = ai.load_sim(exp, 0)
check("dataset from solver exports", X.shape == (21, 3, 1, 128, 32) and np.isfinite(X).all() and mask.any(), f"{X.shape} in {time.time() - t0:.0f}s")
check("ground-truth runs stay in the history", all((runner.RUNS / s["run"]).exists() for s in man["sims"]))
check("solver cost measured", all(s["solver_s_per_step"] for s in man["sims"]))

exp.add_model("persistence")
cnn = exp.add_model("cnn", {"width": 16, "depth": 2}); cnn["train"]["epochs"] = 4
fno = exp.add_model("fno", {"width": 12, "modes": 8, "depth": 2}); fno["train"]["epochs"] = 4
exp.add_model("python", path=str(runner.build.ROOT / "examples" / "surrogate_adapter.py"))
exp.save()
ai.run_all(exp, lambda f, t: None)
evs = {m["id"]: exp.evaluation(m["id"]) for m in exp.config["models"]}
check("every model evaluated", all(evs.values()))
for mid, ev in evs.items():
    a = ev["aggregate"]["test"]
    print(f"      {mid:14s} test relL2 {a['rel_l2']:.4f}  relL1 {a['rel_l1']:.4f}  max {a['max_err']:.3f}  mass {a['mass_err']:.2e}  "
          f"stable {ev['stable_steps']}/{ev['horizon_steps']}  speedup {ev['performance']['speedup'] or 0:.1f}x")
check("metrics finite", all(np.isfinite(ev["aggregate"]["test"]["rel_l2"]) for ev in evs.values()))
check("persistence baseline has real error", evs["persistence-1"]["aggregate"]["test"]["rel_l2"] > 1e-4)
log = exp.train_log("cnn-1")
check("training loss decreases", len(log) == 4 and log[-1]["train_loss"] < log[0]["train_loss"], [round(r["train_loss"], 5) for r in log])
check("CNN beats persistence", evs["cnn-1"]["aggregate"]["test"]["rel_l2"] < evs["persistence-1"]["aggregate"]["test"]["rel_l2"])
s = exp.summary()
check("summary written", s and len(s["models"]) == 4 and s["train_sims"] == 2 and s["test_sims"] == 1)
p = ai.predictions(exp, "cnn-1", 2, "test")
check("predicted rollout stored", p is not None and p.shape == (21, 3, 128, 32))

# 3D view through the FluidX3D renderer: load a ground-truth snapshot into a render-only solver process
vc = ai.viewer_case(exp, 2)
r = runner.Run.create(vc, root=exp.dir / "viewers")
r.start(log=lambda m: None, paused=True)
while not any(x["type"] == "frame" for x in r.records) and r.proc.poll() is None:
    r.poll(); time.sleep(0.05)
ai.write_snapshot(exp.dir / "viewers" / "gt.raw", np.asarray(ai.load_sim(exp, 2)[0][-1]), ai.load_sim(exp, 2)[1], man["groups"])
n = len(r.records)
r.send(f"load {exp.dir / 'viewers' / 'gt.raw'}")
t1 = time.time()
while not any(x["type"] in ("frame", "error") for x in r.records[n:]) and time.time() - t1 < 30:
    r.poll(); time.sleep(0.05)
check("snapshot rendered by the solver renderer", any(x["type"] == "frame" for x in r.records[n:]), [x for x in r.records[n:] if x["type"] == "error"])
r.stop()
check("experiment listed in the run history", any(e.dir == exp.dir for e in ai.list_experiments()))
print(f"\n{sum(results)}/{len(results)} passed  ({exp.dir.name})")
sys.exit(0 if all(results) else 1)
