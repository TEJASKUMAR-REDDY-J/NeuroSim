"""The padded DDF stride (patch 0002) is a pure memory-layout change: results must be bit-identical."""
import time

import numpy as np

from neurosim import presets, runner


def fields(preset, pad, steps=500):
    case = dict(presets.PRESETS[preset], name=f"identity pad{pad}")
    case["solver"] = {"ddf_pad": pad}
    case["run"] = {"steps": steps, "telemetry_every": steps, "slice_every": 0, "frame_fps": 0}
    r = runner.Run.create(case)
    r.start(log=lambda s: None)
    while r.status != "finished":
        assert r.status != "failed", r.error
        r.poll(); time.sleep(0.1)
    r.send("export")
    while not any(x["type"] == "export" for x in r.records):
        r.poll(); time.sleep(0.1)
    files = next(x for x in r.records if x["type"] == "export")["files"]
    r.stop()
    return {f.split("/")[-1].split("_")[0]: np.load(f) for f in files}


for preset in ("Taylor-Green vortex (validation)", "Rayleigh-Benard convection"):
    a, b = fields(preset, 0), fields(preset, 2112)
    for k in a:
        same = a[k].tobytes() == b[k].tobytes()
        print(f"{preset:35s} {k:4s} bit-identical={same}  max|diff|={np.nanmax(np.abs(a[k] - b[k])):.3g}")
        assert same, (preset, k)
print("padding is bit-identical")
