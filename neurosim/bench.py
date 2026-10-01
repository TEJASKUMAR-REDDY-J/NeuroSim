"""Self-benchmark: attainable memory bandwidth and solver throughput for each storage precision.
The fastest precision becomes the default for `precision: auto`."""
import datetime
import json
import os
import statistics
import subprocess
import time

from . import build, runner

CASE = {"name": "benchmark", "domain": [128, 128, 128], "nu": 0.02, "init": "taylor_green", "flow": {"direction": "+x", "u": 0.05, "re": 100},
        "run": {"steps": 900, "telemetry_every": 150, "slice_every": 0, "frame_fps": 0}, "view": {"modes": []}}


def probe(device=0):
    exe = build.build(build.variant())
    r = subprocess.run([str(exe), "--probe", str(device)], capture_output=True, text=True, timeout=300, creationflags=0x08000000 if os.name == "nt" else 0)
    return json.loads(r.stdout.strip().splitlines()[-1])


def solver_mlups(precision, device=-1):
    case = dict(CASE, name=f"benchmark {precision}", solver={"precision": precision, "device": device})
    r = runner.Run.create(case)
    r.start(log=lambda s: None)
    t0 = time.time()
    while r.status not in ("finished", "failed", "stopped") and time.time() - t0 < 900:
        r.poll(); time.sleep(0.2)
    r.stop()
    steps = [x["mlups"] for x in r.records if x["type"] == "step"][1:]  # skip warm-up interval
    return statistics.median(steps) if steps else 0.0


def run(device=0):
    result = {"date": datetime.datetime.now().isoformat(timespec="seconds"), **probe(device), "mlups": {}}
    for p in build.PRECISIONS:
        result["mlups"][p] = solver_mlups(p, device)
    result["recommended"] = max(result["mlups"], key=result["mlups"].get)
    runner.WORKSPACE.mkdir(parents=True, exist_ok=True)
    (runner.WORKSPACE / "benchmarks.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
