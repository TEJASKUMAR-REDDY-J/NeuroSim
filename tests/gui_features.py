"""Feature test of the desktop app + proof that interaction does not change the physics.

1. Reference: run a case headless for S steps, export fields.
2. Same case through the real app window while every control is exercised (orbit, zoom, FOV, every
   visualization toggle, color/slice modes, haze modes and intensity, slice panel, pause/resume, Run
   pressed while running, live-FPS/budget changes, checkpoint, snapshot). Export at S and compare
   bit for bit with the reference.
3. Resume from the mid-run checkpoint in the app, run to S, compare bit for bit again.
4. Rotating geometry: same comparison with heavy interaction through the solver command channel.
Usage: python -m tests.gui_features"""
import copy
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from neurosim import app as app_mod, case as case_mod, presets, runner

S = 1500
CASE = case_mod.normalize(copy.deepcopy(presets.PRESETS["Sphere in wind tunnel (drag)"]))
CASE.update(name="feature test sphere", domain=[96, 192, 96])
CASE["objects"][0]["size"] = 24
CASE["run"] = {"steps": S, "telemetry_every": 150, "slice_every": 150, "frame_fps": 8, "render_budget": 0.2, "checkpoint_every": 0}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def wait(run, pred, timeout=600):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        run.poll(); time.sleep(0.05)
    return pred()


def export(run):
    n = len(run.records)
    run.send("export")
    wait(run, lambda: any(x["type"] == "export" for x in run.records[n:]))
    files = next(x for x in run.records[n:] if x["type"] == "export")["files"]
    return {f.split("/")[-1].split("_")[0]: np.load(f) for f in files}


def forces(run):
    return {x["t"]: x.get("force") for x in run.records if x["type"] == "step"}


def same_fields(a, b):
    """Solver state identical: flags and velocity bit for bit everywhere, density bit for bit in fluid cells.
    (Density stored inside solid cells is never read by the solver; for moving geometry it keeps the last
    displayed value of the cell before it became solid, so it legitimately depends on display refreshes.)"""
    fluid = (a["flags"] & 0x21) == 0
    return (a["flags"].tobytes() == b["flags"].tobytes() and a["u"].tobytes() == b["u"].tobytes()
            and a["rho"][fluid].tobytes() == b["rho"][fluid].tobytes()
            and all(a[k].tobytes() == b[k].tobytes() for k in a if k not in ("rho", "u", "flags")))


def reference(case):
    r = runner.Run.create(case)
    r.start(log=lambda s: None)
    assert wait(r, lambda: r.status == "finished"), r.error
    f = export(r)
    r.stop()
    return f, forces(r)


def gui_test(ref_fields, ref_forces):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app_mod.style_app(qapp)
    w = app_mod.Main()
    w.resize(1600, 960); w.show()
    w.case = copy.deepcopy(CASE); w.set_form(w.case)
    check("form round trip keeps the case", w.get_form()["domain"] == CASE["domain"] and w.get_form()["run"]["steps"] == S)

    # --- preview: paused at t=0 with a frame, then Run continues the same process
    w.preview()
    t0 = time.time()
    while (not w.run or not any(x["type"] == "frame" for x in w.run.records)) and time.time() - t0 < 300:
        qapp.processEvents(); time.sleep(0.05)
    check("preview shows geometry without stepping", w.run and w.run.status == "paused" and w.view.image is not None and not any(x["type"] == "step" for x in w.run.records))
    run_id = w.run.id
    w.start_run()
    qapp.processEvents()

    steps = [
        ("orbit", lambda: [w.view.cam.__setitem__(0, w.view.cam[0] + 7) for _ in range(1)] and setattr(w, "_cam_dirty", True)),
        ("zoom", lambda: (w.view.cam.__setitem__(3, w.view.cam[3] * 1.3), setattr(w, "_cam_dirty", True))),
        ("field of view", lambda: (w.view.cam.__setitem__(2, 35.0), setattr(w, "_cam_dirty", True))),
        ("elevation", lambda: (w.view.cam.__setitem__(1, -40.0), setattr(w, "_cam_dirty", True))),
    ] + [(f"toggle {k}", (lambda b=b: b.toggle())) for k, b in w.vis.items()] + [(f"toggle back {k}", (lambda b=b: b.toggle())) for k, b in w.vis.items()] + [
        ("color by density", lambda: w.vis_field.setCurrentIndex(1)),
        ("field slices xyz", lambda: w.vis_slice.setCurrentIndex(5)),
        ("field slice position", lambda: w.vis_pos.setValue(300)),
        ("field volume", lambda: w.vis_slice.setCurrentIndex(0)),
        ("haze on", lambda: w.cloud_btn.setChecked(True)),
    ] + [(f"overlay mode {m}", (lambda i=i: w.cloud_mode.setCurrentIndex(i))) for i, (m, _) in enumerate(app_mod.CLOUD_MODES)] + [
        ("overlay intensity", lambda: w.cloud_amt.setValue(80)),
        ("slice z-normal vorticity", lambda: (w.s_axis.setCurrentIndex(2), w.s_field.setCurrentIndex(5))),
        ("slice position", lambda: w.s_pos.setValue(250)),
        ("slice lock", lambda: w.s_lock.setChecked(True)),
        ("slice pressure", lambda: w.s_field.setCurrentIndex(8)),
        ("live fps 20", lambda: w.f["fps"].setValue(20)),
        ("render budget 50 %", lambda: w.f["budget"].setValue(50)),
        ("pause", lambda: w.cmd("pause")),
        ("still paused", lambda: None),
        ("resume via Run", lambda: w.start_run()),
        ("Run pressed while running", lambda: w.start_run()),
        ("checkpoint", lambda: w.cmd("checkpoint")),
        ("orbit again", lambda: (w.view.cam.__setitem__(0, 120.0), setattr(w, "_cam_dirty", True))),
        ("haze off", lambda: w.cloud_btn.setChecked(False)),
    ]
    i, last = 0, time.time()
    while w.run.status != "finished" and time.time() - t0 < 900:
        qapp.processEvents()
        if i < len(steps) and time.time() - last > 0.35:
            steps[i][1](); i += 1; last = time.time()
        time.sleep(0.02)
    check("all controls exercised during the run", i == len(steps), f"{i}/{len(steps)}")
    check("Run during a live run did not restart it", w.run.id == run_id)
    check("live frames received", sum(x["type"] == "frame" for x in w.run.records) > 10, f"{sum(x['type'] == 'frame' for x in w.run.records)} frames")
    check("slices received", sum(x["type"] == "slice" for x in w.run.records) > 3)
    check("monitor cards populated", len(w._series.get("t", [])) >= S // 150 - 1 and "force" in w._series)
    check("checkpoint written", wait(w.run, lambda: bool(w.run.checkpoints()), 60), str(w.run.checkpoints()[-1].name if w.run.checkpoints() else ""))
    snap = runner.WORKSPACE / "feature_test_snapshot.png"
    check("snapshot saved", w.view.image is not None and w.view.image.save(str(snap)) and snap.exists())
    fields = export(w.run)
    check("interaction leaves the physics bit-identical (rho, u)", same_fields(ref_fields, fields))
    f = forces(w.run)
    bad = [(t, v, f.get(t)) for t, v in ref_forces.items() if f.get(t) != v]
    check("force history identical at every telemetry step", not bad, f"{len(ref_forces)} samples" + (f", mismatches {bad[:2]}" if bad else ""))
    run_dir = w.run.dir

    # --- runs panel: open and resume from checkpoint
    w.stop_run(); w.refresh_runs()
    items = [w.runs.item(k) for k in range(w.runs.count())]
    item = next(it for it in items if it.data(QtCore.Qt.UserRole) == str(run_dir))
    w.runs.setCurrentItem(item)
    w.open_run(); qapp.processEvents()
    check("open finished run restores history and frame", len(w._series.get("t", [])) > 3 and w.view.image is not None)
    w.resume_run()
    t1 = time.time()
    while (w._building or not w.run or w.run.dir == run_dir or w.run.status != "finished") and time.time() - t1 < 900:
        qapp.processEvents(); time.sleep(0.05)
    restart = next((x for x in w.run.records if x["type"] == "restart"), None)
    check("resume starts from the checkpoint", restart is not None and restart["t"] > 0, f"t={restart and restart['t']}")
    fields2 = export(w.run)
    check("checkpoint restart reproduces the reference bit for bit", same_fields(ref_fields, fields2))
    w.stop_run(); w.close()


def rotating_test():
    case = case_mod.normalize(copy.deepcopy(presets.PRESETS["Rotating fan in a box"]))
    case.update(name="feature test fan", domain=[96, 96, 64])
    case["objects"][0]["size"] = 60
    case["run"] = {"steps": 600, "telemetry_every": 100, "slice_every": 0, "frame_fps": 8, "checkpoint_every": 0}
    ref, _ = reference(case)
    r = runner.Run.create(case)
    r.start(log=lambda s: None)
    k = 0
    while r.status != "finished":
        r.poll(); time.sleep(0.03); k += 1
        if k % 5 == 0:
            r.send(f"cam {k % 360} {(k % 120) - 60} 50 {1 + (k % 7) / 5}")
        if k % 23 == 0:
            r.send(f"cloud {k % 2} {k % 5} 1 0.3"); r.send(f"vis {k % 256} {k % 3} {k % 8} 10 20 30"); r.send(f"slice {k % 3} 20 {k % 9}")
        if k % 61 == 0:
            r.send("pause"); time.sleep(0.2); r.send("resume")
    got = export(r)
    r.stop()
    check("rotating geometry: interaction leaves the physics bit-identical", same_fields(ref, got))


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    ref_fields, ref_forces = reference(copy.deepcopy(CASE))
    gui_test(ref_fields, ref_forces)
    rotating_test()
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} passed")
    sys.exit(0 if all(ok for _, ok in results) else 1)
