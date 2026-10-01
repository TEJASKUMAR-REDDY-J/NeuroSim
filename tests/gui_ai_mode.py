"""AI Mode in the real window: enter/leave, every page, comparison slices, 3D comparison through the solver renderer.
Needs one evaluated experiment (run tests.ai_pipeline first). Screenshots go to workspace/ai_mode_*.png.
Usage: python -m tests.gui_ai_mode"""
import sys
import time

from PyQt5 import QtWidgets

from neurosim import ai, app as app_mod, runner

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def pump(qapp, seconds, until=None):
    t0 = time.time()
    while time.time() - t0 < seconds and not (until and until()):
        qapp.processEvents(); time.sleep(0.02)


qapp = QtWidgets.QApplication(sys.argv)
app_mod.style_app(qapp)
w = app_mod.Main(); w.resize(1720, 1000); w.show()
pump(qapp, 0.5)
exp = next(e for e in ai.list_experiments() if any(e.evaluation(m["id"]) for m in e.config["models"]))
check("simulation mode shows AI Mode button", w.ai_btn.isVisible() and w.sim_bar.isVisible())
w.enter_ai(exp.dir); pump(qapp, 1.0)
check("AI Mode transforms the same window", w.body.currentWidget() is w.ai and w.ai_bar.isVisible() and not w.sim_bar.isVisible() and w.sim_btn.isVisible())
check("subtitle", w.brand_sub.text() == "AI Mode · Neural Surrogate Laboratory")
check("experiment opened", w.ai.exp and w.ai.exp.dir == exp.dir and w.ai.title.text() == exp.name)
shots = runner.WORKSPACE
for i, name in enumerate(("experiments", "datasets", "models", "evaluation", "benchmarks", "analytics")):
    w.nav_ai(i); pump(qapp, 0.6)
    w.grab().save(str(shots / f"ai_mode_{name}.png"))
check("pages render", all((shots / f"ai_mode_{n}.png").exists() for n in ("experiments", "evaluation", "analytics")))
check("evaluation table filled", w.ai.e_table.columnCount() == 1 + sum(bool(exp.evaluation(m["id"])) for m in exp.config["models"]) and w.ai.e_table.rowCount() > 15)
check("analytics lists simulations and experiments", {w.ai.a_table.item(r, 0).text() for r in range(w.ai.a_table.rowCount())} >= {"Simulation", "AI experiment"})
w.nav_ai(3); pump(qapp, 0.3)
check("comparison slices drawn", all(s.image is not None for s in w.ai.x_slices))
w.ai.x_t.setValue(5); pump(qapp, 0.2)
w.ai.x_mode.setChecked(True)
pump(qapp, 90, lambda: w.ai.viewers and w.ai.viewers.runs and all(p.image is not None for p in w.ai.x_ports) and not any(w.ai.viewers.busy))
check("3D comparison rendered by three solver renderers", all(p.image is not None for p in w.ai.x_ports), w.ai.viewers and w.ai.viewers.err)
w.ai.x_ports[0].cam[0] += 40; w.ai.sync_camera(w.ai.x_ports[0]); pump(qapp, 3)
check("cameras synchronized", all(p.cam == w.ai.x_ports[0].cam for p in w.ai.x_ports))
w.ai.x_t.setValue(w.ai.x_t.maximum()); pump(qapp, 5, lambda: not any(w.ai.viewers.busy))
w.ai.pages.widget(3).verticalScrollBar().setValue(10000); pump(qapp, 0.5)
w.grab().save(str(shots / "ai_mode_3d.png"))
w.leave_ai(); pump(qapp, 0.5)
check("back to simulation mode", w.body.currentIndex() == 0 and w.sim_bar.isVisible() and w.ai_btn.isVisible() and not w.ai_bar.isVisible())
w.grab().save(str(shots / "ai_mode_back.png"))
w.ai.shutdown(); w.close()
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
