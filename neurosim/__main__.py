"""python -m neurosim [gui | run case.json | devices | bench | build]"""
import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser(prog="neurosim")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="desktop workbench (default)")
    r = sub.add_parser("run", help="run a case file headless")
    r.add_argument("case")
    r.add_argument("--steps", type=int)
    sub.add_parser("devices", help="list OpenCL devices")
    sub.add_parser("bench", help="bandwidth + precision benchmark")
    a = ap.parse_args()
    if a.cmd in (None, "gui"):
        from .app import main as gui
        gui()
    elif a.cmd == "devices":
        from . import runner
        print(json.dumps(runner.devices(), indent=1))
    elif a.cmd == "bench":
        from . import bench
        print(json.dumps(bench.run(), indent=1))
    elif a.cmd == "run":
        from . import runner
        case = json.loads(open(a.case).read())
        if a.steps:
            case.setdefault("run", {})["steps"] = a.steps
        run = runner.Run.create(case)
        run.start()
        print(f"run {run.dir}")
        try:
            while run.status not in ("finished", "failed", "stopped"):
                for rec in run.poll():
                    if rec["type"] == "step":
                        print(f"t={rec['t']:>8} {rec['mlups']:7.1f} MLUP/s  u_max={rec['u_max']:.4f}  rho={rec['rho_mean']:.6f}" + (f"  F={rec['force']}" if "force" in rec else ""), flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        run.stop()
        sys.exit(0 if run.status != "failed" else 1)


if __name__ == "__main__":
    main()
