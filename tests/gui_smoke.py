"""Drives the desktop app without a user: load a preset, run it, orbit the camera, save a window screenshot.
Usage: python -m tests.gui_smoke [preset] [out.png] [seconds]"""
import sys
import time

from PyQt5 import QtCore, QtWidgets

from neurosim import app as app_mod


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "Sphere in wind tunnel (drag)"
    out = sys.argv[2] if len(sys.argv) > 2 else "build/gui_smoke.png"
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 25
    qapp = QtWidgets.QApplication(sys.argv)
    app_mod.dark_palette(qapp)
    w = app_mod.Main()
    w.resize(1600, 960)
    w.show()
    w.preset.setCurrentText(preset)
    w.load_preset(preset)
    w.start_run()
    t0 = time.time()
    orbit = [0]

    def step():
        el = time.time() - t0
        if w.run and w.run.status == "running" and el > seconds * 0.5 and orbit[0] < 10:
            w.view.cam[0] += 4; orbit[0] += 1; w._cam_dirty = True
        if el > seconds:
            w.grab().save(out)
            print("status", w.run.status if w.run else None, "frames", sum(1 for r in (w.run.records if w.run else []) if r["type"] == "frame"),
                  "steps", len(w._series.get("t", [])), "error", w.run.error if w.run else None)
            w.stop_run()
            qapp.quit()
    timer = QtCore.QTimer(); timer.timeout.connect(step); timer.start(100)
    qapp.exec_()


if __name__ == "__main__":
    main()
