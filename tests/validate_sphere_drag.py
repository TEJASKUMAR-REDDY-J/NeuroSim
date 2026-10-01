"""Sphere drag at Re = 1000 vs the standard drag curve (Cd ~ 0.47 unconfined).
Expect somewhat higher values here: 5 % blockage, 32-cell diameter, short averaging window."""
import time

import numpy as np

from neurosim import presets, runner

case = dict(presets.PRESETS["Sphere in wind tunnel (drag)"], name="validation sphere drag")
case["run"] = {"steps": 12000, "telemetry_every": 200, "slice_every": 0, "frame_fps": 0.2}
r = runner.Run.create(case)
r.start(log=lambda s: None)
while r.status not in ("finished", "failed"):
    r.poll(); time.sleep(1)
r.stop()
d = r.meta["derived"]
steps = [x for x in r.records if x["type"] == "step"]
t = np.array([x["t"] for x in steps]); F = np.array([x["force"] for x in steps])
cd = F[:, d["flow_axis"]] / (0.5 * d["u"] ** 2 * d["ref_area"])
late = cd[t > t.max() / 2]
print(f"Re {d['re']:.0f}  A_ref {d['ref_area']:.0f} cells^2  Cd(mean of 2nd half) = {late.mean():.3f} +- {late.std():.3f}  (reference ~0.47 unconfined)")
print("Cd history:", " ".join(f"{x:.3f}" for x in cd[::5]))
