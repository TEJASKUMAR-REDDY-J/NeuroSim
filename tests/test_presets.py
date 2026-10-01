"""End-to-end check: every preset builds, starts, steps and reports sane telemetry.
Usage: python -m tests.test_presets [name filter]"""
import sys
import time

from neurosim import presets, runner


def run_preset(name, steps=200, timeout=600):
    case = dict(presets.PRESETS[name], name="test " + name)
    case["run"] = {"steps": steps, "telemetry_every": 100, "slice_every": 100, "frame_fps": 2}
    r = runner.Run.create(case)
    r.start(log=lambda s: None)
    t0 = time.time()
    while r.status not in ("finished", "failed", "stopped") and time.time() - t0 < timeout:
        r.poll()
        time.sleep(0.2)
    r.stop()
    steps_rec = [x for x in r.records if x["type"] == "step"]
    start = next((x for x in r.records if x["type"] == "start"), {})
    ok = r.status in ("finished", "stopped") and steps_rec and all(x["u_max"] == x["u_max"] and x["u_max"] < 0.6 for x in steps_rec)
    last = steps_rec[-1] if steps_rec else {}
    print(f"{'OK  ' if ok else 'FAIL'} {name:42s} {r.status:9s} t={last.get('t')} mlups={last.get('mlups', 0):7.1f} u_max={last.get('u_max', float('nan')):.4f} "
          f"force={last.get('force')} setup={start.get('setup_s')} err={r.error}", flush=True)
    return ok


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    results = [run_preset(n) for n in presets.PRESETS if flt.lower() in n.lower()]
    sys.exit(0 if all(results) else 1)
