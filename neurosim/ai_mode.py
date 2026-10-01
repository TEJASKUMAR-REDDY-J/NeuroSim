"""AI Mode: the workbench reconfigured as the Neural Surrogate Laboratory.

Same window, theme and widgets as Simulation Mode. The central workspace switches to experiments that ask:
can this neural model reproduce or accelerate this simulation, and under what conditions does it fail?
Ground truth comes from normal solver runs; the 3D comparison uses the solver's own renderer.
"""
import json
import threading
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import ai, analytics, case as case_mod, runner
from . import app as ui

C = ui.C
NAV = ("Experiments", "Datasets", "Models", "Evaluation", "Benchmarks", "Analytics")
STAGES = [("Physics", 0), ("Dataset", 1), ("Model", 2), ("Training", 2), ("Evaluation", 3), ("Generalization", 3), ("Results", 0)]
PALETTE = [C["accent"], C["orange"], C["ok"], "#c084fc", "#22d3ee", C["warn"], C["bad"]]
PURPLE = "#a855f7"


# ======================================================================== formatting


def pct(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f} %"


def sci(x):
    return "—" if x is None else f"{x:.3g}"


def secs(s):
    if s is None:
        return "—"
    return f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s" if s < 60 else f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.1f} h"


def gb(b):
    return f"{b / 2 ** 30:.2f} GB" if b >= 2 ** 30 else f"{b / 2 ** 20:.1f} MB"


def grid(g):
    return " × ".join(str(n) for n in g if n > 1)


def card(title=None, obj="card"):
    f = QtWidgets.QFrame(); f.setObjectName(obj)
    l = QtWidgets.QVBoxLayout(f); l.setContentsMargins(16, 6 if title else 14, 16, 14); l.setSpacing(6)
    if title:
        l.addWidget(ui.section(title))
    return f, l


def rich(html=""):
    l = QtWidgets.QLabel(html); l.setWordWrap(True); l.setTextFormat(Qt.RichText); l.setTextInteractionFlags(Qt.TextSelectableByMouse)
    l.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    return l


def kv(rows):
    mu = C["muted"]
    return "<table cellspacing='0' cellpadding='3'>" + "".join(
        f"<tr><td style='color:{mu};padding-right:14px'>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def table(cols):
    t = QtWidgets.QTableWidget(0, len(cols)); t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().hide(); t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); t.setShowGrid(False); t.setAlternatingRowColors(True)
    t.horizontalHeader().setStretchLastSection(True); t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    t.setWordWrap(False)
    return t


def fill(t, rows, bold_rows=()):
    t.setRowCount(len(rows))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QtWidgets.QTableWidgetItem("" if v is None else str(v))
            if i in bold_rows:
                f = it.font(); f.setBold(True); it.setFont(f); it.setForeground(QtGui.QColor(C["muted"]))
            t.setItem(i, j, it)
    t.resizeColumnsToContents()


def scroll(widget):
    s = QtWidgets.QScrollArea(); s.setWidgetResizable(True); s.setWidget(widget)
    return s


# ======================================================================== widgets


class Pipeline(QtWidgets.QWidget):
    """Physics -> Dataset -> Model -> Training -> Evaluation -> Generalization -> Results, with the state of each stage."""
    clicked = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(66); self.setMouseTracking(True); self.setCursor(Qt.PointingHandCursor)
        self.stages, self.hover = [(n, "", "todo") for n, _ in STAGES], -1

    def set(self, stages):
        self.stages = stages; self.update()

    def _boxes(self):
        n, gap = len(self.stages), 22
        w = (self.width() - 24 - gap * (n - 1)) / n
        return [QtCore.QRectF(12 + i * (w + gap), 8, w, self.height() - 16) for i in range(n)]

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        col = {"done": C["ok"], "active": C["accent"], "todo": C["dim"], "fail": C["bad"], "run": C["warn"]}
        boxes = self._boxes()
        for i, (r, (name, sub, st)) in enumerate(zip(boxes, self.stages)):
            c = QtGui.QColor(col[st])
            p.setPen(QtGui.QPen(QtGui.QColor(c.red(), c.green(), c.blue(), 150 if st != "todo" else 90), 1.2))
            p.setBrush(QtGui.QColor(C["card2"] if i == self.hover else C["card"])); p.drawRoundedRect(r, 9, 9)
            p.setBrush(c); p.setPen(Qt.NoPen); p.drawEllipse(QtCore.QPointF(r.left() + 14, r.top() + 17), 4, 4)
            p.setPen(QtGui.QColor(C["text"] if st != "todo" else C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.DemiBold))
            p.drawText(r.adjusted(26, 7, -6, 0), Qt.AlignLeft | Qt.AlignTop, name)
            p.setPen(QtGui.QColor(C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 8))
            p.drawText(r.adjusted(12, 0, -6, -7), Qt.AlignLeft | Qt.AlignBottom, p.fontMetrics().elidedText(sub, Qt.ElideRight, int(r.width() - 18)))
            if i < len(boxes) - 1:
                y, x0 = r.center().y(), r.right() + 5
                p.setPen(QtGui.QPen(QtGui.QColor(C["dim"]), 1.4)); p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x0 + 12, y))
                p.drawLine(QtCore.QPointF(x0 + 12, y), QtCore.QPointF(x0 + 8, y - 4)); p.drawLine(QtCore.QPointF(x0 + 12, y), QtCore.QPointF(x0 + 8, y + 4))

    def mouseMoveEvent(self, e):
        h = next((i for i, r in enumerate(self._boxes()) if r.contains(e.pos())), -1)
        if h != self.hover:
            self.hover = h; self.update()

    def leaveEvent(self, e):
        self.hover = -1; self.update()

    def mousePressEvent(self, e):
        i = next((i for i, r in enumerate(self._boxes()) if r.contains(e.pos())), -1)
        if i >= 0:
            self.clicked.emit(STAGES[i][1])


def _ticks(lo, hi, n=5):
    span = max(hi - lo, 1e-30)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((1, 2, 5, 10), key=lambda m: abs(m * mag - raw))
    return np.arange(np.ceil(lo / step) * step, hi + 0.5 * step, step)


class Chart(QtWidgets.QFrame):
    """Line / point chart with axes, ticks and legend. series: [(label, x, y, color, style)], style 'line' | 'dash' | 'points' | 'hollow'."""

    def __init__(self, title, xlabel="", ylabel="", logy=False):
        super().__init__()
        self.setObjectName("card"); self.setMinimumHeight(230)
        self.title, self.xlabel, self.ylabel, self.logy = title, xlabel, ylabel, logy
        self.series, self.hlines, self.xticklabels, self.empty = [], [], None, "No data yet"

    def set(self, series, hlines=(), xticklabels=None, empty="No data yet"):
        self.series = [(l, np.asarray(x, float), np.asarray(y, float), c, s) for l, x, y, c, s in series if len(x)]
        self.hlines, self.xticklabels, self.empty = list(hlines), xticklabels, empty
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QColor(C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.DemiBold))
        p.drawText(14, 22, self.title.upper())
        area = QtCore.QRectF(62, 58, self.width() - 80, self.height() - 100)
        ys = [y[np.isfinite(y) & ((y > 0) if self.logy else True)] for _, _, y, _, _ in self.series]
        ys = np.concatenate(ys) if ys else np.array([])
        if not self.series or ys.size == 0:
            p.setPen(QtGui.QColor(C["dim"])); p.setFont(QtGui.QFont("Segoe UI", 9)); p.drawText(self.rect(), Qt.AlignCenter, self.empty); return
        xs = np.concatenate([x for _, x, _, _, _ in self.series])
        xa, xb = float(xs.min()), float(xs.max())
        if xb <= xa:
            xa, xb = xa - 0.5, xb + 0.5
        tf = (lambda v: np.log10(np.maximum(v, 1e-30))) if self.logy else (lambda v: v)
        hl = [h for h, _ in self.hlines]
        ya, yb = float(tf(np.concatenate([ys, hl])).min()), float(tf(np.concatenate([ys, hl])).max())
        if yb - ya < 1e-9:
            ya, yb = ya - 0.5, yb + 0.5
        pad = 0.06 * (yb - ya); ya -= pad; yb += pad
        if not self.logy and ys.min() >= 0:
            ya = max(ya, 0.0)
        X = lambda v: area.left() + (v - xa) / (xb - xa) * area.width()
        Y = lambda v: area.bottom() - (tf(v) - ya) / (yb - ya) * area.height()
        p.setFont(QtGui.QFont("Cascadia Mono", 7))
        yt = [10.0 ** k for k in range(int(np.floor(ya)), int(np.ceil(yb)) + 1)] if self.logy else _ticks(ya, yb, 4)
        for v in yt:
            y = Y(v)
            if area.top() - 1 <= y <= area.bottom() + 1:
                p.setPen(QtGui.QPen(QtGui.QColor(C["border"]), 1)); p.drawLine(QtCore.QPointF(area.left(), y), QtCore.QPointF(area.right(), y))
                p.setPen(QtGui.QColor(C["muted"])); p.drawText(QtCore.QRectF(4, y - 8, area.left() - 8, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:.3g}")
        if self.xticklabels:
            for i, lab in enumerate(self.xticklabels):
                p.drawText(QtCore.QRectF(X(i) - 50, area.bottom() + 4, 100, 14), Qt.AlignCenter, lab)
        else:
            for v in _ticks(xa, xb, 6):
                p.drawText(QtCore.QRectF(X(v) - 40, area.bottom() + 4, 80, 14), Qt.AlignCenter, f"{v:,.4g}")
        p.setPen(QtGui.QColor(C["dim"])); p.setFont(QtGui.QFont("Segoe UI", 8))
        p.drawText(QtCore.QRectF(area.left(), self.height() - 20, area.width(), 16), Qt.AlignCenter, self.xlabel)
        p.save(); p.translate(14, area.center().y()); p.rotate(-90)
        p.drawText(QtCore.QRectF(-area.height() / 2, -8, area.height(), 16), Qt.AlignCenter, self.ylabel); p.restore()
        for h, label in self.hlines:
            p.setPen(QtGui.QPen(QtGui.QColor(C["bad"]), 1, Qt.DashLine)); y = Y(h)
            p.drawLine(QtCore.QPointF(area.left(), y), QtCore.QPointF(area.right(), y))
            p.setPen(QtGui.QColor(C["bad"])); p.drawText(QtCore.QRectF(area.left() + 4, y - 15, 300, 14), Qt.AlignLeft, label)
        p.setClipRect(area.adjusted(-4, -4, 4, 4))
        for label, x, y, col, style in self.series:
            c = QtGui.QColor(col)
            ok = np.isfinite(y) & ((y > 0) if self.logy else True)
            pts = [QtCore.QPointF(X(a), Y(b)) for a, b in zip(x[ok], y[ok])]
            if style in ("points", "hollow"):
                p.setPen(QtGui.QPen(c, 1.6)); p.setBrush(Qt.NoBrush if style == "hollow" else c)
                for q in pts:
                    p.drawEllipse(q, 4, 4)
            elif len(pts) > 1:
                p.setBrush(Qt.NoBrush); p.setPen(QtGui.QPen(c, 1.8, Qt.DashLine if style == "dash" else Qt.SolidLine)); p.drawPolyline(QtGui.QPolygonF(pts))
        p.setClipping(False)
        p.setFont(QtGui.QFont("Segoe UI", 8)); x0, y = 14.0, 32
        seen = []
        for label, _, _, col, style in self.series:
            if label and label not in seen:
                seen.append(label)
        for label in seen[:8]:  # legend: one row under the title, left to right
            col, style = next((c, s) for l, _, _, c, s in self.series if l == label)
            w = p.fontMetrics().horizontalAdvance(label)
            if x0 + w + 30 > self.width() - 10:
                break
            p.setPen(QtGui.QPen(QtGui.QColor(col), 2, Qt.DashLine if style == "dash" else Qt.SolidLine)); p.setBrush(Qt.NoBrush if style == "hollow" else QtGui.QColor(col))
            if style in ("points", "hollow"):
                p.drawEllipse(QtCore.QPointF(x0 + 8, y + 1), 3.5, 3.5)
            else:
                p.drawLine(QtCore.QPointF(x0, y + 1), QtCore.QPointF(x0 + 16, y + 1))
            p.setPen(QtGui.QColor(C["text"])); p.drawText(QtCore.QPointF(x0 + 20, y + 5), label)
            x0 += w + 36


class Viewers:
    """Ground truth, surrogate and difference rendered by three render-only solver processes (FluidX3D renderer)."""
    NAMES = ("truth", "surrogate", "difference")

    def __init__(self, exp, k, ports):
        self.exp, self.k, self.ports = exp, k, ports
        self.runs, self.err, self.busy, self.pending, self.seq = [], None, [True] * 3, [None] * 3, [0] * 3
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self):
        try:
            case = ai.viewer_case(self.exp, self.k)
            runs = []
            for n in self.NAMES:
                case["name"] = f"viewer {n}"
                r = runner.Run.create(case, root=self.exp.dir / "viewers")
                r.start(log=lambda m: None, paused=True)
                if case["domain"][2] == 1:  # 2D: colored z plane instead of a sparse point volume
                    r.send(f"vis {case_mod.VIS['lattice'] | case_mod.VIS['surface'] | case_mod.VIS['field']} 0 3 0 0 0")
                runs.append(r)
            self.runs = runs
        except Exception as e:  # shown in the comparison panel
            self.err = str(e)

    def show(self, i, writer):
        """writer(path) writes the snapshot; it is sent when the viewer has finished the previous one."""
        self.pending[i] = writer

    def tick(self):
        for i, r in enumerate(self.runs):
            for rec in r.poll():
                if rec["type"] == "frame":
                    data = (r.dir / "live" / rec["file"]).read_bytes()
                    if len(data) == 4 * rec["w"] * rec["h"]:
                        self.ports[i].set_frame(data, rec["w"], rec["h"])
                    self.busy[i] = False
                elif rec["type"] == "error":
                    self.busy[i] = False; self.err = rec.get("message")
            if r.status == "failed":
                self.err = r.error
            if not self.busy[i] and self.pending[i]:
                path = r.dir / f"snap_{self.seq[i] % 2}.raw"; self.seq[i] += 1
                self.pending[i](path); self.pending[i] = None
                r.send(f"load {path}"); self.busy[i] = True

    def send(self, cmd):
        for r in self.runs:
            r.send(cmd)

    def stop(self):
        for r in self.runs:
            r.stop()


# ======================================================================== dialogs


class NewExperiment(QtWidgets.QDialog):
    """+ New AI Experiment: pick the ground truth (current setup or a run) and a stress test; everything physical carries over."""

    def __init__(self, parent, current_case, run_dir=None):
        super().__init__(parent)
        self.setWindowTitle("New AI Experiment"); self.resize(620, 520)
        ui.dark_title_bar(self)
        self.current = current_case
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(20, 18, 20, 18); lay.setSpacing(8)
        h = QtWidgets.QLabel("New AI Experiment"); h.setStyleSheet("font-size: 14pt; font-weight: 600;"); lay.addWidget(h)
        s = QtWidgets.QLabel("Neural Surrogate Laboratory · can a neural model reproduce or accelerate this simulation, and where does it fail?")
        s.setObjectName("hint"); s.setWordWrap(True); lay.addWidget(s)
        self.solver = QtWidgets.QComboBox()
        for name, ok in ai.SOLVERS:
            self.solver.addItem(name if ok else f"{name}  (not integrated)")
            if not ok:
                self.solver.model().item(self.solver.count() - 1).setEnabled(False)
        self.source = QtWidgets.QComboBox()
        self.source.addItem(f"Current setup · {current_case.get('name', 'untitled')}", None)
        for p in runner.list_runs()[:80]:
            if not ai.is_experiment(p) and not p.name[16:].startswith("ai-data"):
                self.source.addItem(f"Run · {p.name[16:].replace('-', ' ')} · {p.name[:8]} {p.name[9:11]}:{p.name[11:13]}", str(p))
        if run_dir:
            self.source.setCurrentIndex(max(0, self.source.findData(str(run_dir))))
        self.kind = QtWidgets.QComboBox()
        for k, (label, desc) in ai.STRESS_TESTS.items():
            self.kind.addItem(label, k)
        self.desc = QtWidgets.QLabel(); self.desc.setObjectName("hint"); self.desc.setWordWrap(True)
        self.name = QtWidgets.QLineEdit()
        for w in (ui.Field("Ground-truth solver", self.solver), ui.Field("Ground truth (carried over: geometry, grid, physics, boundaries, solver)", self.source),
                  ui.Field("Experiment type", self.kind, self.desc), ui.Field("Name", self.name)):
            lay.addWidget(w)
        self.info = rich(); lay.addWidget(self.info, 1)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        ok = QtWidgets.QPushButton("Create experiment"); ok.setObjectName("primary"); ok.clicked.connect(self.accept)
        row.addWidget(cancel); row.addWidget(ok); lay.addLayout(row)
        self.source.currentIndexChanged.connect(self._update); self.kind.currentIndexChanged.connect(self._update)
        self._update()

    def case(self):
        d = self.source.currentData()
        return json.loads((Path(d) / "case.json").read_text()) if d else self.current

    def _update(self):
        c = case_mod.normalize(self.case())
        self.desc.setText(ai.STRESS_TESTS[self.kind.currentData()][1])
        self.name.setText(f"{c.get('name', 'simulation')} · {self.kind.currentText().lower()}")
        objs = ", ".join(o.get("shape") or Path(str(o.get("model"))).stem for o in c["objects"]) or "none"
        self.info.setText(kv([("Grid", f"{grid(c['domain'])} · {np.prod(c['domain']) / 1e6:.2f} M cells"), ("Geometry", objs),
                              ("Flow", f"{c['flow']['direction']} · u {c['flow']['u']} · Re {c['flow']['re']:g}" + (f" · ν {c['nu']}" if c["nu"] else "")),
                              ("Boundaries", " · ".join(f"{f} {t}" for f, t in c["boundaries"].items())),
                              ("Physics", ", ".join(case_mod.features(c)).lower().replace("_", " ") or "plain flow"),
                              ("Note", "Ground truth is produced by the solver with a field-export schedule, one run per condition value.")]))


# ======================================================================== workspace


class AIWorkspace(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.setObjectName("root")
        self.main, self.exp, self.viewers = main, None, None
        self.job, self.jstate, self.cancel = None, {}, threading.Event()
        self._loading, self._cmp, self._gain, self._cam_dirty, self._a_rows = False, None, 1.0, False, []
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        split = QtWidgets.QSplitter(Qt.Horizontal); split.setHandleWidth(1)
        split.addWidget(self._left()); split.addWidget(self._center())
        split.setStretchFactor(1, 1); split.setSizes([300, 1400])
        h.addWidget(split)
        self.refresh_list()

    # ------------------------------------------------------------ layout
    def _left(self):
        f = QtWidgets.QFrame(); f.setObjectName("sidebar"); f.setMinimumWidth(270)
        l = QtWidgets.QVBoxLayout(f); l.setContentsMargins(14, 14, 14, 14); l.setSpacing(8)
        b = self.main._btn("New AI Experiment", "plus", self.new_experiment, "primary"); b.setMinimumHeight(36); l.addWidget(b)
        l.addWidget(ui.section("Experiments"))
        self.list = QtWidgets.QListWidget(); self.list.setWordWrap(True); self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(lambda it, _: it and self.open(it.data(Qt.UserRole)))
        l.addWidget(self.list, 1)
        n = QtWidgets.QLabel("AI experiments live in the run history next to simulations. Ground truth runs on the solver (GPU); "
                             "built-in models train with PyTorch on the CPU.")
        n.setObjectName("hint"); n.setWordWrap(True); l.addWidget(n)
        return f

    def _center(self):
        w = QtWidgets.QWidget(); w.setObjectName("root")
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(16, 12, 16, 0); v.setSpacing(8)
        top = QtWidgets.QHBoxLayout(); top.setSpacing(10)
        self.title = QtWidgets.QLabel("Neural Surrogate Laboratory"); self.title.setStyleSheet("font-size: 15pt; font-weight: 600;")
        top.addWidget(self.title)
        self.chips = QtWidgets.QHBoxLayout(); self.chips.setSpacing(6); top.addLayout(self.chips)
        top.addStretch(1)
        self.job_label = QtWidgets.QLabel(""); self.job_label.setObjectName("hint"); top.addWidget(self.job_label)
        self.job_bar = QtWidgets.QProgressBar(); self.job_bar.setRange(0, 1000); self.job_bar.setFixedWidth(160); self.job_bar.setTextVisible(False); self.job_bar.hide()
        top.addWidget(self.job_bar)
        self.cancel_btn = self.main._btn("Cancel", "stop", lambda: self.cancel.set(), "ghost"); self.cancel_btn.hide(); top.addWidget(self.cancel_btn)
        v.addLayout(top)
        self.pipe = Pipeline(); self.pipe.clicked.connect(lambda i: self.main.nav_ai(i)); v.addWidget(self.pipe)
        self.pages = QtWidgets.QStackedWidget()
        for build in (self._page_overview, self._page_dataset, self._page_models, self._page_eval, self._page_bench, self._page_analytics):
            self.pages.addWidget(build())
        v.addWidget(self.pages, 1)
        return w

    def nav(self, i):
        self.pages.setCurrentIndex(i)
        if i == 5:
            self.refresh_analytics()

    # ------------------------------------------------------------ pages
    def _page_overview(self):
        w = QtWidgets.QWidget(); g = QtWidgets.QGridLayout(w); g.setContentsMargins(0, 4, 4, 16); g.setSpacing(12)
        self.empty, el = card()
        t = QtWidgets.QLabel("Neural Surrogate Laboratory"); t.setStyleSheet("font-size: 18pt; font-weight: 600;"); el.addWidget(t)
        el.addWidget(rich(f"<span style='color:{C['muted']}'>Can a neural model reproduce or accelerate a physics simulation, and under what conditions does it fail?</span><br><br>"
                          "1&nbsp; Pick a simulation as ground truth (the current setup or any run). Geometry, grid, physics, boundaries and solver carry over.<br>"
                          "2&nbsp; Choose what changes between training and testing: Reynolds number, velocity, geometry, resolution, horizon or boundaries.<br>"
                          "3&nbsp; Generate the dataset with the solver, attach models (FNO, CNN, your TorchScript / ONNX / Python model), train, evaluate.<br>"
                          "4&nbsp; Compare against ground truth: field and physics errors, long-rollout stability, speed, memory, 3D side by side."))
        b = self.main._btn("New AI Experiment", "plus", self.new_experiment, "primary"); b.setFixedWidth(220); el.addWidget(b)
        g.addWidget(self.empty, 0, 0, 1, 2)
        self.c_gt, l1 = card("Ground truth"); self.gt_info = rich(); l1.addWidget(self.gt_info)
        self.c_cfg, l2 = card("Experiment"); self.cfg_info = rich(); l2.addWidget(self.cfg_info)
        self.c_sum, l3 = card("Surrogate experiment summary"); self.sum_info = rich(); l3.addWidget(self.sum_info)
        self.c_prov, l4 = card("Provenance"); self.prov_info = rich(); l4.addWidget(self.prov_info)
        g.addWidget(self.c_gt, 1, 0); g.addWidget(self.c_cfg, 1, 1); g.addWidget(self.c_sum, 2, 0, 1, 2); g.addWidget(self.c_prov, 3, 0, 1, 2)
        g.setRowStretch(4, 1); g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)
        return scroll(w)

    def _page_dataset(self):
        w = QtWidgets.QWidget(); g = QtWidgets.QGridLayout(w); g.setContentsMargins(0, 4, 4, 16); g.setSpacing(12)
        c1, l = card("Conditions: what the surrogate sees in training, what it must handle in testing")
        self.d_kind = QtWidgets.QComboBox()
        for k, (label, _) in ai.STRESS_TESTS.items():
            self.d_kind.addItem(label, k)
        self.d_kind_desc = QtWidgets.QLabel(); self.d_kind_desc.setObjectName("hint")
        l.addWidget(ui.Field("Stress test", self.d_kind, self.d_kind_desc))
        self.d_param = QtWidgets.QComboBox()
        for k, label in ai.PARAMS.items():
            self.d_param.addItem(label, k)
        l.addWidget(ui.Field("Condition", self.d_param))
        self.d_train, self.d_test = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        l.addWidget(ui.Field("Training values", self.d_train))
        l.addWidget(ui.Field("Test values", self.d_test, hint="Comma separated; a:b:n gives n evenly spaced numbers. Geometry: sphere, cube, cylinder, torus, … "
                                                         "(primitives: cylinder, cuboid, sphere). Boundaries: freestream, wall, periodic."))
        self.d_err = QtWidgets.QLabel(); self.d_err.setStyleSheet(f"color: {C['bad']};"); l.addWidget(self.d_err)
        c2, l = card("Sampling")
        self.d_fields = {}
        row = QtWidgets.QHBoxLayout()
        for f in ai.FIELDS:
            cb = QtWidgets.QCheckBox(f); self.d_fields[f] = cb; row.addWidget(cb); cb.toggled.connect(self.save_dataset)
        row.addStretch(1)
        fl = QtWidgets.QLabel("Fields"); fl.setStyleSheet(f"color: {C['muted']}; font-size: 8.5pt;"); l.addWidget(fl); l.addLayout(row)
        sp = lambda lo, hi, step: (lambda s: (s.setRange(lo, hi), s.setSingleStep(step), s.valueChanged.connect(self.save_dataset), s)[-1])(QtWidgets.QSpinBox())
        self.d_every, self.d_start, self.d_steps = sp(1, 10 ** 7, 10), sp(0, 10 ** 9, 100), sp(1, 10 ** 9, 100)
        l.addWidget(ui.Field("Temporal sampling: every N steps · record from step · run to step", self.d_every, self.d_start, self.d_steps))
        self.d_ds = QtWidgets.QComboBox(); self.d_ds.addItems(["original", "½ (2× coarser)", "¼ (4× coarser)", "⅛ (8× coarser)"]); self.d_ds.currentIndexChanged.connect(self.save_dataset)
        self.d_seq = QtWidgets.QComboBox(); self.d_seq.setEditable(True); self.d_seq.addItems(["2", "3", "5", "10", "50", "100"])
        self.d_seq.lineEdit().editingFinished.connect(self.save_dataset); self.d_seq.activated.connect(self.save_dataset)
        l.addWidget(ui.Field("Spatial resolution · sequence length (snapshots per training sequence)", self.d_ds, self.d_seq))
        self.d_val = sp(0, 50, 5)
        self.d_split = QtWidgets.QLineEdit(); self.d_split.editingFinished.connect(self.save_dataset)
        l.addWidget(ui.Field("Split: validation % of training sequences · time split train,val,test (no condition)", self.d_val, self.d_split))
        self.d_keep = QtWidgets.QCheckBox("Keep full-resolution solver exports"); self.d_keep.toggled.connect(self.save_dataset); l.addWidget(self.d_keep)
        c3, l = card("Estimate")
        self.d_est = rich(); l.addWidget(self.d_est)
        row = QtWidgets.QHBoxLayout()
        self.d_gen = self.main._btn("Generate dataset", "play", self.generate, "primary"); row.addWidget(self.d_gen); row.addStretch(1)
        l.addLayout(row)
        c4, l = card("Dataset")
        self.d_table = table(["Simulation", "Split", "Snapshots", "Grid (dataset)", "Grid (solver)", "Solver MLUP/s", "Run"])
        self.d_table.setMinimumHeight(170); l.addWidget(self.d_table)
        prow = QtWidgets.QHBoxLayout()
        self.d_psim, self.d_pch = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.d_pt = QtWidgets.QSlider(Qt.Horizontal)
        for x in (self.d_psim, self.d_pch):
            prow.addWidget(x); x.currentIndexChanged.connect(self.preview_dataset)
        prow.addWidget(self.d_pt, 1); self.d_pt.valueChanged.connect(self.preview_dataset)
        self.d_ptl = QtWidgets.QLabel(); self.d_ptl.setObjectName("hint"); prow.addWidget(self.d_ptl)
        l.addLayout(prow)
        self.d_prev = ui.SliceView(); self.d_prev.setMinimumHeight(260); l.addWidget(self.d_prev)
        g.addWidget(c1, 0, 0); g.addWidget(c2, 0, 1); g.addWidget(c3, 1, 0, 1, 2); g.addWidget(c4, 2, 0, 1, 2)
        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1); g.setRowStretch(3, 1)
        self.d_kind.activated.connect(self.apply_stress); self.d_param.activated.connect(self.save_dataset)
        self.d_train.editingFinished.connect(self.save_dataset); self.d_test.editingFinished.connect(self.save_dataset)
        return scroll(w)

    def _page_models(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 4, 4, 16); v.setSpacing(12)
        c, l = card("Model catalog")
        g = QtWidgets.QGridLayout(); g.setSpacing(10)
        for i, (k, (label, desc, ok)) in enumerate(ai.MODELS.items()):
            f = QtWidgets.QFrame(); f.setObjectName("modelcard"); f.setEnabled(ok)
            fl = QtWidgets.QVBoxLayout(f); fl.setContentsMargins(12, 10, 12, 10); fl.setSpacing(4)
            t = QtWidgets.QHBoxLayout(); n = QtWidgets.QLabel(label); n.setStyleSheet("font-weight: 600; font-size: 10pt;"); t.addWidget(n); t.addStretch(1)
            t.addWidget(ui.chip("external" if k in ai.EXTERNAL else "built-in" if ok else "plugin slot", PURPLE if k in ai.EXTERNAL else C["ok"] if ok else C["dim"]))
            fl.addLayout(t)
            d = QtWidgets.QLabel(desc); d.setObjectName("hint"); d.setWordWrap(True); fl.addWidget(d, 1)
            b = self.main._btn("Load…" if k in ai.EXTERNAL else "Attach" if ok else "Not available", "import" if k in ai.EXTERNAL else "plus",
                               (lambda _=False, k=k: self.attach(k)) if ok else None)
            b.setEnabled(ok); fl.addWidget(b)
            g.addWidget(f, i // 4, i % 4)
        l.addLayout(g); v.addWidget(c)
        c, l = card("Attached models")
        self.m_table = table(["Model", "Status", "Parameters", "Training", "Test rel. L2 (velocity)"])
        self.m_table.setMinimumHeight(150); self.m_table.itemSelectionChanged.connect(self.select_model); l.addWidget(self.m_table)
        row = QtWidgets.QHBoxLayout()
        self.m_train = self.main._btn("Train", "play", lambda: self.run_models(train=True, all_=False))
        self.m_eval = self.main._btn("Evaluate", "eye", lambda: self.run_models(train=False, all_=False))
        self.m_all = self.main._btn("Train + evaluate all", "play", lambda: self.run_models(train=True, all_=True), "primary")
        self.m_rm = self.main._btn("Remove", "trash", self.remove_model)
        for b in (self.m_all, self.m_train, self.m_eval, self.m_rm):
            row.addWidget(b)
        row.addStretch(1); l.addLayout(row)
        v.addWidget(c)
        h = QtWidgets.QHBoxLayout(); h.setSpacing(12)
        c, l = card("Hyperparameters")
        self.m_form = {}
        for key, lo, hi, step, dec, label in (("width", 4, 512, 4, 0, "Width (channels)"), ("depth", 1, 16, 1, 0, "Layers / residual blocks"), ("modes", 2, 64, 2, 0, "Fourier modes per axis"),
                                               ("epochs", 1, 10000, 5, 0, "Epochs"), ("lr", 1e-5, 1.0, 1e-4, 5, "Learning rate"), ("batch", 1, 256, 1, 0, "Batch size")):
            s = QtWidgets.QDoubleSpinBox() if dec else QtWidgets.QSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step)
            if dec:
                s.setDecimals(dec)
            s.valueChanged.connect(self.save_model)
            self.m_form[key] = s; l.addWidget(ui.Field(label, s))
        self.m_note = QtWidgets.QLabel(); self.m_note.setObjectName("hint"); self.m_note.setWordWrap(True); l.addWidget(self.m_note); l.addStretch(1)
        c.setFixedWidth(330); h.addWidget(c)
        self.loss = Chart("Training loss (fluid cells, in units of the typical one-step change)", "epoch", "loss", logy=True); h.addWidget(self.loss, 1)
        v.addLayout(h)
        v.addStretch(1)
        return scroll(w)

    def _page_eval(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 4, 4, 16); v.setSpacing(12)
        c, l = card("Surrogate evaluation · test conditions, measured")
        self.e_table = table(["Metric"]); self.e_table.setMinimumHeight(520); self.e_table.setAlternatingRowColors(False); l.addWidget(self.e_table)
        n = QtWidgets.QLabel("Errors are over fluid cells of the dataset grid, averaged over each autoregressive rollout from the first test state. "
                             "Drag and lift errors are not reported: surrogates predict flow fields, while forces need the solver's boundary populations.")
        n.setObjectName("hint"); n.setWordWrap(True); l.addWidget(n)
        v.addWidget(c)
        h = QtWidgets.QHBoxLayout(); h.setSpacing(12)
        lc = QtWidgets.QVBoxLayout()
        self.e_sim = QtWidgets.QComboBox(); self.e_sim.currentIndexChanged.connect(self.refresh_charts); lc.addWidget(self.e_sim)
        self.e_curve = Chart("Long-horizon error · rollout step → relative L2 (velocity)", "solver time steps", "relative L2", logy=True); lc.addWidget(self.e_curve, 1)
        h.addLayout(lc, 1)
        self.e_gen = Chart("Generalization · error per condition (hollow = training, filled = test)", "condition", "mean relative L2", logy=True)
        h.addWidget(self.e_gen, 1)
        v.addLayout(h)
        c, l = card("Ground truth vs surrogate vs difference")
        row = QtWidgets.QHBoxLayout(); row.setSpacing(8)
        self.x_model, self.x_sim, self.x_field, self.x_axis = (QtWidgets.QComboBox() for _ in range(4))
        self.x_axis.addItems(["x-normal", "y-normal", "z-normal"])
        self.x_pos, self.x_t = QtWidgets.QSlider(Qt.Horizontal), QtWidgets.QSlider(Qt.Horizontal)
        self.x_pos.setRange(0, 1000); self.x_pos.setValue(500); self.x_pos.setFixedWidth(100)
        self.x_tl = QtWidgets.QLabel(); self.x_tl.setObjectName("hint"); self.x_tl.setMinimumWidth(150)
        self.x_mode = QtWidgets.QPushButton("3D (solver renderer)"); self.x_mode.setObjectName("toggle"); self.x_mode.setCheckable(True); self.x_mode.setCursor(Qt.PointingHandCursor)
        for x in (self.x_model, self.x_sim, self.x_field, self.x_axis):
            row.addWidget(x)
        row.addWidget(self.x_pos); row.addWidget(QtWidgets.QLabel("t")); row.addWidget(self.x_t, 1); row.addWidget(self.x_tl); row.addWidget(self.x_mode)
        l.addLayout(row)
        self.x_stack = QtWidgets.QStackedWidget()
        sw = QtWidgets.QWidget(); sh = QtWidgets.QHBoxLayout(sw); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(8)
        tw = QtWidgets.QWidget(); th = QtWidgets.QHBoxLayout(tw); th.setContentsMargins(0, 0, 0, 0); th.setSpacing(8)
        self.x_slices, self.x_ports, self.x_titles = [], [], []
        for name in ("Ground truth", "Surrogate", "Difference (surrogate − truth)"):
            col = QtWidgets.QVBoxLayout(); t = QtWidgets.QLabel(name); t.setStyleSheet("font-weight: 600;"); col.addWidget(t)
            s = ui.SliceView(); s.setMinimumHeight(300); col.addWidget(s, 1); self.x_slices.append(s); sh.addLayout(col, 1)
            col = QtWidgets.QVBoxLayout(); t2 = QtWidgets.QLabel(name); t2.setStyleSheet("font-weight: 600;"); col.addWidget(t2); self.x_titles.append(t2)
            p = ui.Viewport(); p.setMinimumSize(200, 300); p.hint = "Starting renderer…"; col.addWidget(p, 1); self.x_ports.append(p); th.addLayout(col, 1)
            p.camera_changed.connect(lambda *a, p=p: self.sync_camera(p))
        self.x_stack.addWidget(sw); self.x_stack.addWidget(tw)
        l.addWidget(self.x_stack)
        self.x_info = QtWidgets.QLabel(); self.x_info.setObjectName("hint"); self.x_info.setWordWrap(True); l.addWidget(self.x_info)
        for x in (self.x_model, self.x_sim):
            x.currentIndexChanged.connect(self.compare_selection)
        for x in (self.x_field, self.x_axis):
            x.currentIndexChanged.connect(self.show_compare)
        self.x_pos.valueChanged.connect(self.show_compare); self.x_t.valueChanged.connect(self.show_compare)
        self.x_mode.toggled.connect(self.toggle_3d)
        v.addWidget(c)
        return scroll(w)

    def _page_bench(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 4, 4, 16); v.setSpacing(12)
        c, l = card("Performance · same simulated time span, measured")
        self.b_table = table(["Model", "Runtime", "Rollout steps", "Surrogate", "Per step", "Solver (same span)", "Speedup", "Dataset grid", "Solver grid",
                              "Parameters", "Model + state", "Solver memory", "Memory ratio", "Training"])
        self.b_table.setMinimumHeight(220); l.addWidget(self.b_table)
        self.b_note = rich(); l.addWidget(self.b_note)
        v.addWidget(c)
        h = QtWidgets.QHBoxLayout(); h.setSpacing(12)
        self.b_speed = Chart("Speedup over the solver (×)", "", "speedup ×", logy=True); h.addWidget(self.b_speed, 1)
        self.b_acc = Chart("Accuracy vs speed · each model (test conditions)", "speedup ×", "relative L2", logy=True); h.addWidget(self.b_acc, 1)
        v.addLayout(h); v.addStretch(1)
        return scroll(w)

    def _page_analytics(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 4, 4, 16); v.setSpacing(12)
        c, l = card("Analytics · simulations and AI experiments, one history")
        row = QtWidgets.QHBoxLayout()
        self.a_filter = QtWidgets.QComboBox(); self.a_filter.addItems(["All", "Simulations", "AI experiments"]); self.a_filter.currentIndexChanged.connect(self.refresh_analytics)
        row.addWidget(self.a_filter); row.addStretch(1); row.addWidget(self.main._btn("", "refresh", self.refresh_analytics, tip="Refresh"))
        l.addLayout(row)
        self.a_table = table(["Type", "Name", "Created", "Cells", "Steps", "Runtime", "MLUP/s", "Memory", "Cd", "Cl", "KE change (last ¼)",
                              "Best surrogate", "Rel. L2", "Mass error", "Stable rollout", "Speedup"])
        self.a_table.setMinimumHeight(560); self.a_table.itemDoubleClicked.connect(self.open_from_analytics); l.addWidget(self.a_table)
        n = QtWidgets.QLabel("Double-click to open. Simulation metrics come from solver telemetry, AI metrics from measured surrogate evaluations.")
        n.setObjectName("hint"); l.addWidget(n)
        v.addWidget(c); v.addStretch(1)
        return scroll(w)

    # ------------------------------------------------------------ experiments
    def refresh_list(self, select=None):
        self.list.blockSignals(True); self.list.clear()
        for e in ai.list_experiments():
            ev = sum(bool(e.evaluation(m["id"])) for m in e.config["models"])
            it = QtWidgets.QListWidgetItem(f"{e.name}\n{e.dir.name[4:6]}/{e.dir.name[6:8]} {e.dir.name[9:11]}:{e.dir.name[11:13]} · {e.meta.get('status')} · {ev}/{len(e.config['models'])} models")
            it.setData(Qt.UserRole, str(e.dir)); self.list.addItem(it)
            if select and Path(select) == e.dir:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)

    def new_experiment(self, run_dir=None):
        dlg = NewExperiment(self, self.main.get_form(), run_dir if isinstance(run_dir, (str, Path)) else None)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            src = dlg.source.currentData()
            e = ai.Experiment.create(dlg.case(), dlg.name.text().strip() or None, Path(src).name if src else None, dlg.kind.currentData())
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Cannot create experiment", str(ex)); return
        self.refresh_list(e.dir); self.open(e.dir); self.main.nav_ai(1)
        self.main.refresh_runs()

    def open(self, d):
        if self.viewers:
            self.viewers.stop(); self.viewers = None; self.x_mode.setChecked(False)
        self.exp = ai.Experiment(d) if d else None
        self.refresh()

    def reload(self):
        if self.exp:
            self.exp = ai.Experiment(self.exp.dir)

    def refresh(self):
        e = self.exp
        for w in (self.c_gt, self.c_cfg, self.c_sum, self.c_prov):
            w.setVisible(e is not None)
        self.empty.setVisible(e is None)
        while self.chips.count():
            self.chips.takeAt(0).widget().deleteLater()
        if not e:
            self.title.setText("Neural Surrogate Laboratory"); self.pipe.set([(n, "", "todo") for n, _ in STAGES]); return
        self.title.setText(e.name)
        kind = ai.STRESS_TESTS.get(e.config.get("stress_test"), ("Custom",))[0]
        for text, col in ((kind, PURPLE), (e.meta.get("status", ""), C["ok"] if e.meta.get("status") == "evaluated" else C["warn"])):
            self.chips.addWidget(ui.chip(text, col))
        self.refresh_overview(); self.load_dataset_form(); self.refresh_dataset(); self.refresh_models(); self.refresh_eval(); self.refresh_bench()
        self.refresh_pipeline()

    def refresh_pipeline(self):
        e = self.exp
        man, ms = e.manifest(), e.config["models"]
        trained = [m for m in ms if e.model_meta(m["id"])]
        evs = [ev for ev in (e.evaluation(m["id"]) for m in ms) if ev]
        busy = self.job and self.job.is_alive()
        d = e.meta["derived"]
        best = min(evs, key=lambda ev: ev["aggregate"]["test"]["rel_l2"]) if evs else None
        gen = best and best["aggregate"]["train"]["rel_l2"] is not None and best["aggregate"]["test"]["rel_l2"]
        self.pipe.set([
            ("Physics", f"{grid(e.case['domain'])} · Re {d['re']:.4g}", "done"),
            ("Dataset", f"{len(man['sims'])} sims · {grid(man['sims'][0]['grid'])}" if man else "not generated", "done" if man else "run" if busy else "active"),
            ("Model", f"{len(ms)} attached" if ms else "none attached", "done" if ms else "todo" if not man else "active"),
            ("Training", f"{len(trained)}/{len(ms)} trained" if ms else "—", "done" if ms and len(trained) == len(ms) else "todo"),
            ("Evaluation", f"{len(evs)}/{len(ms)} evaluated" if ms else "—", "done" if ms and len(evs) == len(ms) else "todo"),
            ("Generalization", f"test {pct(best['aggregate']['test']['rel_l2'], 1)} vs train {pct(best['aggregate']['train']['rel_l2'], 1)}" if gen else "—", "done" if gen else "todo"),
            ("Results", f"best {best['label']} · {best['performance']['speedup']:.3g}× faster" if best and best["performance"].get("speedup") else "—", "done" if best else "todo"),
        ])

    def refresh_overview(self):
        e, mu = self.exp, C["muted"]
        c, d = case_mod.normalize(e.case), e.meta["derived"]
        objs = ", ".join(o.get("shape") or Path(str(o.get("model"))).stem for o in c["objects"]) or "none"
        v = d["variant"]
        gt = e.config["ground_truth"]
        man = e.manifest()
        fields = ["velocity", "pressure", "density", "vorticity"] + (["temperature"] if "TEMPERATURE" in v["extensions"] else [])
        self.gt_info.setText(kv([("Solver", gt["solver"]), ("Source", f"run {gt['source_run']}" if gt.get("source_run") else f"simulation setup “{gt.get('case_name')}”"),
                                 ("Geometry", objs), ("Grid", f"{grid(c['domain'])} · {d['cells'] / 1e6:.2f} M cells · {d['memory_mb']:.0f} MB"),
                                 ("Flow", f"{c['flow']['direction']} · u {d['u']} (lattice) · Re {d['re']:.4g} · ν {d['nu']:.4g} · τ {d['tau']:.4f}"),
                                 ("Boundaries", " · ".join(f"{f} {t}" for f, t in c["boundaries"].items())),
                                 ("Solver", f"{v['precision']} · {v['lattice']} · {v['collision']} · " + (", ".join(x.lower().replace('_', ' ') for x in v['extensions']) or "no extensions")),
                                 ("Fields available", ", ".join(fields)),
                                 ("Time steps", f"export every {e.config['dataset']['every']} · record {e.config['dataset']['start']:,} → {e.config['dataset']['steps']:,}")]))
        cond = e.config["condition"]
        p = cond.get("param", "none")
        self.cfg_info.setText(kv([("Experiment type", ai.STRESS_TESTS.get(e.config.get("stress_test"), ("Custom", ""))[0]),
                                  ("Condition", ai.PARAMS.get(p, p)),
                                  ("Training", ", ".join(ai.fmt_value(p, x) for x in cond.get("train", [])) or "first part of the run"),
                                  ("Testing", ", ".join(ai.fmt_value(p, x) for x in cond.get("test", [])) or "later part of the run"),
                                  ("Dataset", f"{len(man['sims'])} simulations · {sum(len(s['times']) for s in man['sims'])} snapshots · channels {', '.join(man['channels'])}" if man else "not generated"),
                                  ("Models", ", ".join(m["label"] for m in e.config["models"]) or "none attached"),
                                  ("Created", e.meta.get("created"))]))
        s = e.summary()
        if s and s["models"]:
            rows = [("Ground truth", f"{s['ground_truth']} · {s.get('case') or ''}"), ("Stress test", f"{s['stress_test']} · {s['condition']}"),
                    ("Training", f"{s['train_sims']} simulations · {s['snapshots']} snapshots"), ("Evaluation", f"{s['test_sims']} unseen simulations"),
                    ("Grids", f"surrogate {grid(s['grid'])} · solver {grid(s['solver_grid'])}")]
            html = kv(rows) + "<br>"
            hdr = "".join(f"<th style='text-align:left;padding:4px 14px 4px 0;color:{mu}'>{h}</th>" for h in
                          ("Surrogate", "Accuracy (rel. L2)", "Physics (mass)", "Long rollout", "Speed", "Memory", "Worst case"))
            body = "".join("<tr>" + "".join(f"<td style='padding:4px 14px 4px 0'>{x}</td>" for x in (
                f"<b>{r['model']}</b>", pct(r["rel_l2"]), pct(r["mass_err"], 3), f"stable {r['stable_steps']:,}/{r['horizon_steps']:,} steps",
                f"{r['speedup']:.3g}× faster" if r.get("speedup") else "—", f"{r['memory_ratio']:.3g}× lower" if r.get("memory_ratio") else "—", r["worst"])) + "</tr>" for r in s["models"])
            self.sum_info.setText(html + f"<table cellspacing='0'>{'<tr>' + hdr + '</tr>'}{body}</table>")
        else:
            self.sum_info.setText(f"<span style='color:{mu}'>Appears once models are evaluated. Every value comes from measured runs.</span>")
        m = e.meta
        runs = ", ".join(s_["run"] for s_ in man["sims"]) if man else "—"
        self.prov_info.setText(kv([("Directory", str(e.dir)), ("NeuroSim", f"{m.get('neurosim_commit', '')[:10]}" + (" (modified)" if m.get("neurosim_dirty") else "")),
                                   ("FluidX3D", m.get("fluidx3d_commit", "")[:10]), ("Patches", ", ".join(m.get("patches", []))), ("Host", m.get("host", {}).get("os")),
                                   ("Ground-truth runs", runs)]))

    # ------------------------------------------------------------ dataset
    def load_dataset_form(self):
        self._loading = True
        e = self.exp; ds, cond = e.config["dataset"], e.config["condition"]
        self.d_kind.setCurrentIndex(max(0, self.d_kind.findData(e.config.get("stress_test", "custom"))))
        self.d_kind_desc.setText(ai.STRESS_TESTS.get(e.config.get("stress_test"), ("", ""))[1])
        self.d_param.setCurrentIndex(max(0, self.d_param.findData(cond.get("param", "none"))))
        self.d_train.setText(", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in cond.get("train", [])))
        self.d_test.setText(", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in cond.get("test", [])))
        for f, cb in self.d_fields.items():
            cb.setChecked(f in ds["fields"])
        self.d_fields["temperature"].setEnabled("TEMPERATURE" in e.meta["derived"]["variant"]["extensions"])
        self.d_every.setValue(int(ds["every"])); self.d_start.setValue(int(ds["start"])); self.d_steps.setValue(int(ds["steps"]))
        self.d_ds.setCurrentIndex({1: 0, 2: 1, 4: 2, 8: 3}.get(int(ds["downsample"]), 1))
        self.d_seq.setCurrentText(str(ds["sequence"])); self.d_val.setValue(int(round(100 * ds["val_fraction"])))
        self.d_split.setText(",".join(f"{x:g}" for x in ds["time_split"])); self.d_keep.setChecked(bool(ds.get("keep_exports")))
        self._loading = False
        self.update_estimate()

    def apply_stress(self, *_):
        e = self.exp
        if not e:
            return
        k = self.d_kind.currentData()
        e.config["stress_test"] = k
        if k != "custom":
            e.config["condition"] = ai.stress_condition(k, e.case, e.meta["derived"])
        e.save(); self.load_dataset_form(); self.refresh_overview(); self.refresh(); self.refresh_list(e.dir)

    def save_dataset(self, *_):
        e = self.exp
        if self._loading or not e:
            return
        p = self.d_param.currentData()
        if p != e.config["condition"].get("param"):  # new parameter: start from sensible values for it
            c0 = ai.condition_for(p, e.case, e.meta["derived"])
            self._loading = True
            self.d_train.setText(", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in c0["train"]))
            self.d_test.setText(", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in c0["test"]))
            self._loading = False
        try:
            train, test = ai.parse_values(self.d_train.text(), p), ai.parse_values(self.d_test.text(), p)
            split = [float(x) for x in self.d_split.text().split(",")]
            seq = int(self.d_seq.currentText())
            if len(split) != 3 or seq < 2:
                raise ValueError("time split needs three fractions; sequence length at least 2")
            if p != "none" and (not train or not test):
                raise ValueError("give at least one training and one test value")
        except ValueError as ex:
            self.d_err.setText(str(ex)); return
        self.d_err.setText("")
        cond = {"param": p, "train": train if p != "none" else [], "test": test if p != "none" else []}
        if cond != e.config["condition"]:  # hand-edited conditions are a custom experiment
            e.config["stress_test"] = "custom"
            self.d_kind.setCurrentIndex(self.d_kind.findData("custom")); self.d_kind_desc.setText(ai.STRESS_TESTS["custom"][1])
        e.config["condition"] = cond
        e.config["dataset"].update(fields=[f for f, cb in self.d_fields.items() if cb.isChecked()] or ["velocity"], every=self.d_every.value(), start=self.d_start.value(),
                                   steps=max(self.d_steps.value(), self.d_start.value() + self.d_every.value()), downsample=[1, 2, 4, 8][self.d_ds.currentIndex()],
                                   sequence=seq, val_fraction=self.d_val.value() / 100, time_split=split, keep_exports=self.d_keep.isChecked())
        e.save()
        self.update_estimate(); self.refresh_overview()

    def update_estimate(self):
        e = self.exp
        try:
            est = ai.estimate(e)
        except Exception as ex:
            self.d_est.setText(f"<span style='color:{C['bad']}'>{ex}</span>"); return
        man = e.manifest()
        stale = man and (man["settings"] != e.config["dataset"] or man["param"] != e.config["condition"].get("param") or
                         [s["value"] for s in man["sims"]] != [s["value"] for s in ai.plan(e)])
        big = f"<span style='font-size:13pt;font-weight:600'>{est['simulations']} simulations · {est['sequences']:,} sequences · {gb(est['bytes'])}</span>"
        rows = [("Split", f"{est['train']} training · {est['test']} test" if est["test"] else "one run, split in time (train / validation / test)"),
                ("Snapshots", f"{est['snapshots']:,} · {est['channels']} channels"),
                ("Solver exports", gb(est["raw_bytes"]) + ("" if e.config["dataset"].get("keep_exports") else " (temporary)")),
                ("Ground-truth cost", f"≈ {secs(est['solver_s'])} of solver time (benchmark MLUP/s)" if est["solver_s"] else "run Hardware → benchmark for an estimate")]
        note = f"<br><span style='color:{C['warn']}'>Settings changed since the dataset was generated: regenerate to apply.</span>" if stale else ""
        self.d_est.setText(big + "<br>" + kv(rows) + note)

    def generate(self):
        if not self.exp:
            return
        if self.exp.manifest() and QtWidgets.QMessageBox.question(self, "Regenerate dataset", "Replace the existing dataset? Trained models and evaluations become out of date.") != QtWidgets.QMessageBox.Yes:
            return
        self.start_job("Generating dataset", lambda prog, cancel: ai.generate(self.exp, prog, cancel))

    def refresh_dataset(self):
        man = self.exp.manifest()
        self.d_table.setRowCount(0)
        self.d_psim.blockSignals(True); self.d_pch.blockSignals(True); self.d_psim.clear(); self.d_pch.clear()
        if man:
            fill(self.d_table, [(s["label"], s["split"], len(s["times"]), grid(s["grid"]), grid(s["solver_grid"]), f"{s['solver_mlups']:.0f}", s["run"]) for s in man["sims"]])
            for k, s in enumerate(man["sims"]):
                self.d_psim.addItem(f"{s['label']} ({s['split']})", k)
            self.d_pch.addItems(man["channels"])
        self.d_psim.blockSignals(False); self.d_pch.blockSignals(False)
        self.preview_dataset()

    def preview_dataset(self, *_):
        man = self.exp.manifest() if self.exp else None
        if not man or self.d_psim.currentIndex() < 0:
            self.d_prev.image = None; self.d_prev.update(); return
        k = self.d_psim.currentData()
        X, mask = ai.load_sim(self.exp, k)
        self.d_pt.blockSignals(True); self.d_pt.setRange(0, len(X) - 1); self.d_pt.blockSignals(False)
        i = self.d_pt.value()
        self.d_ptl.setText(f"step {man['sims'][k]['times'][i]:,}")
        vals, axis = self._plane(np.asarray(X[i, max(0, self.d_pch.currentIndex())]), mask, 2, 0.5)
        self.d_prev.lock = False
        self.d_prev.set_slice(vals, {"axis": axis, "step": 1, "signed": self.d_pch.currentText() not in ("rho",)}, self.d_pch.currentText() not in ("rho",))

    @staticmethod
    def _plane(a, mask, axis, frac):
        """2D plane of a [Z, Y, X] array with solids as NaN, oriented as SliceView expects."""
        a = np.where(mask, np.nan, a).astype(np.float32)
        if a.shape[0] == 1:
            return a[0], 2
        n = a.shape[2 - axis]
        i = min(n - 1, int(frac * n))
        return (a[:, :, i] if axis == 0 else a[:, i, :] if axis == 1 else a[i]), axis

    # ------------------------------------------------------------ models
    def attach(self, kind):
        if not self.exp:
            self.new_experiment(); return
        path = None
        if kind in ai.EXTERNAL:
            filt = {"torchscript": "TorchScript (*.pt *.pth)", "onnx": "ONNX (*.onnx)", "python": "Python adapter (*.py)"}[kind]
            start = str(runner.build.ROOT / "examples") if kind == "python" else ""
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Load {ai.MODELS[kind][0]}", start, filt)
            if not path:
                return
        self.exp.add_model(kind, path=path)
        self.refresh_models(select=len(self.exp.config["models"]) - 1); self.refresh_pipeline(); self.refresh_overview(); self.refresh_list(self.exp.dir)

    def refresh_models(self, select=None):
        e = self.exp
        rows = []
        for m in e.config["models"]:
            meta, ev = e.model_meta(m["id"]), e.evaluation(m["id"])
            status = "evaluated" if ev else ("trained" if meta and meta.get("trained") else "ready" if meta else "not trained")
            params = (meta or {}).get("parameters")
            rows.append((m["label"] + ("" if m["kind"] not in ai.EXTERNAL else f"  ({Path(m['path']).name})" if m.get("path") else ""), status,
                         f"{params:,}" if params else "—", secs((meta or {}).get("train_seconds")) if meta and meta.get("trained") else "—",
                         pct(ev["aggregate"]["test"]["rel_l2"]) if ev else "—"))
        cur = self.m_table.currentRow() if select is None else select
        self.m_table.blockSignals(True); fill(self.m_table, rows); self.m_table.blockSignals(False)
        if rows:
            self.m_table.selectRow(min(max(cur, 0), len(rows) - 1))
        self.select_model()

    def _sel_model(self):
        r = self.m_table.currentRow()
        ms = self.exp.config["models"] if self.exp else []
        return ms[r] if 0 <= r < len(ms) else None

    def select_model(self):
        m = self._sel_model()
        self._loading = True
        for k, s in self.m_form.items():
            src = (m or {}).get("params", {}) if k in ("width", "depth", "modes") else (m or {}).get("train", {})
            s.setEnabled(bool(m) and k in src); s.setValue(src.get(k, s.minimum()) if m else s.minimum())
        self._loading = False
        if not m:
            self.m_note.setText(""); self.loss.set([]); return
        notes = {"persistence": "No parameters: next state = current state. The reference every model must beat.",
                 "torchscript": "Contract: input [1, C+1+K, *S] (state channels in physical units, solid mask, K condition values) → next state [1, C, *S].",
                 "onnx": "Contract: input [1, C+1+K, *S] (state channels in physical units, solid mask, K condition values) → next state [1, C, *S].",
                 "python": "Surrogate(manifest) with predict(x) and optional fit(train); see examples/surrogate_adapter.py."}
        self.m_note.setText(notes.get(m["kind"], "Changing hyperparameters takes effect on the next training."))
        log = self.exp.train_log(m["id"])
        base = self.exp.train_log(m["id"], "baseline")
        self.loss.set([("train", [r["epoch"] for r in log], [r["train_loss"] for r in log], C["accent"], "line"),
                       ("validation", [r["epoch"] for r in log if r["val_loss"] is not None], [r["val_loss"] for r in log if r["val_loss"] is not None], C["orange"], "line")],
                      hlines=[(base[0]["persistence_loss"], "persistence baseline (measured)")] if log and base else (), empty="Train the model to see its loss curve")

    def save_model(self, *_):
        m = self._sel_model()
        if self._loading or not m:
            return
        for k, s in self.m_form.items():
            if not s.isEnabled():
                continue
            (m["params"] if k in ("width", "depth", "modes") else m["train"])[k] = s.value()
        self.exp.save()

    def remove_model(self):
        m = self._sel_model()
        if m and QtWidgets.QMessageBox.question(self, "Remove model", f"Remove {m['label']} with its weights and evaluation?") == QtWidgets.QMessageBox.Yes:
            ai.remove_model(self.exp, m["id"]); self.reload(); self.refresh()

    def run_models(self, train, all_):
        e = self.exp
        if not e:
            return
        ms = [m["id"] for m in e.config["models"]] if all_ else [m["id"] for m in [self._sel_model()] if m]
        if not ms:
            QtWidgets.QMessageBox.information(self, "No model", "Attach a model first (Models → catalog)."); return
        if not e.manifest() and not all_:
            QtWidgets.QMessageBox.information(self, "No dataset", "Generate the dataset first (Datasets)."); return
        if all_:
            self.start_job("Running experiment", lambda prog, cancel: ai.run_all(e, prog, cancel, ms))
        elif train:
            self.start_job("Training", lambda prog, cancel: ai.train(e, ms[0], prog, cancel))
        else:
            self.start_job("Evaluating", lambda prog, cancel: ai.evaluate(e, ms[0], prog))

    # ------------------------------------------------------------ evaluation
    def _evals(self):
        e = self.exp
        return [(m, e.evaluation(m["id"])) for m in e.config["models"] if e.evaluation(m["id"])] if e else []

    def refresh_eval(self):
        evs = self._evals()
        rows, bold = [], []
        groups = (self.exp.manifest() or {}).get("groups", {})
        spec = [("Field error", None), ("Relative L1 (velocity)", ("test", "rel_l1"), pct), ("Relative L2 (velocity)", ("test", "rel_l2"), pct),
                ("Max error (relative to peak speed)", ("test", "max_err"), pct), ("MSE (lattice units²)", ("test", "mse"), sci), ("MAE (lattice units)", ("test", "mae"), sci)]
        spec += [(f"Relative L2: {g}", ("test", f"rel_l2_{g}"), pct) for g in groups]
        spec += [("Physics", None), ("Mass error", ("test", "mass_err"), lambda x: pct(x, 3)), ("Momentum error", ("test", "momentum_err"), pct),
                 ("Kinetic energy error", ("test", "energy_err"), pct),
                 ("Generalization", None), ("Relative L2 at training conditions", ("train", "rel_l2"), pct), ("Relative L2 at test conditions", ("test", "rel_l2"), pct),
                 ("Worst case", "worst", None),
                 ("Stability", None), ("Stable rollout (rel. L2 < 50 %)", "stable", None),
                 ("Performance", None), ("Ground truth (solver)", ("perf", "solver_s"), secs), ("Surrogate", ("perf", "surrogate_s"), secs),
                 ("Speedup", ("perf", "speedup"), lambda x: "—" if x is None else f"{x:.3g}×"), ("Memory: solver / model", ("perf", "memory_ratio"), lambda x: "—" if x is None else f"{x:.3g}× lower")]
        for item in spec:
            label, key = item[0], item[1]
            if key is None:
                bold.append(len(rows)); rows.append([label.upper()] + [""] * len(evs)); continue
            vals = []
            for m, ev in evs:
                if key == "worst":
                    vals.append(f"{ev['worst']['label']} ({pct(ev['worst']['rel_l2'], 1)})")
                elif key == "stable":
                    vals.append(f"{ev['stable_steps']:,} / {ev['horizon_steps']:,} steps")
                elif key[0] == "perf":
                    vals.append(item[2](ev["performance"].get(key[1])))
                else:
                    vals.append(item[2](ev["aggregate"][key[0]].get(key[1])))
            rows.append([label] + vals)
        self.e_table.setColumnCount(1 + len(evs)); self.e_table.setHorizontalHeaderLabels(["Metric"] + [m["label"] for m, _ in evs])
        fill(self.e_table, rows, bold)
        for c in range(1, 1 + len(evs)):
            self.e_table.setColumnWidth(c, max(self.e_table.columnWidth(c), 170))
        self.e_sim.blockSignals(True); self.e_sim.clear()
        if evs:
            self.e_sim.addItem("Mean over test simulations", None)
            for i, r in enumerate(evs[0][1]["sims"]):
                self.e_sim.addItem(f"{r['label']} ({r['split']})", i)
        self.e_sim.blockSignals(False)
        self.refresh_charts()
        self.x_model.blockSignals(True); self.x_sim.blockSignals(True)
        self.x_model.clear(); self.x_sim.clear()
        for m, ev in evs:
            self.x_model.addItem(m["label"], m["id"])
        if evs:
            for r in evs[0][1]["sims"]:
                self.x_sim.addItem(f"{r['label']} ({r['split']})", (r["sim"], r["split"]))
            test_i = next((i for i, r in enumerate(evs[0][1]["sims"]) if r["split"] == "test"), 0)
            self.x_sim.setCurrentIndex(test_i)
        self.x_model.blockSignals(False); self.x_sim.blockSignals(False)
        self.compare_selection()

    def refresh_charts(self, *_):
        evs = self._evals()
        sel = self.e_sim.currentData()
        series = []
        for n, (m, ev) in enumerate(evs):
            col = PALETTE[n % len(PALETTE)]
            if sel is None:
                tests = [r for r in ev["sims"] if r["split"] == "test"]
                L = min(len(r["steps"]) for r in tests)
                y = np.mean([r["per_step"]["rel_l2"][:L] for r in tests], axis=0)
                series.append((m["label"], tests[0]["steps"][:L], y, col, "line"))
            else:
                r = ev["sims"][sel]
                series.append((m["label"], r["steps"], [np.nan if v is None else v for v in r["per_step"]["rel_l2"]], col, "line"))
        self.e_curve.set(series, hlines=[(ai.STABILITY_THRESHOLD, "divergence threshold")] if series else (), empty="Evaluate a model to see its rollout error")
        man = self.exp.manifest() if self.exp else None
        gseries, labels = [], None
        if evs and man:
            numeric = man["param"] in ai.NUMERIC
            if not numeric:
                labels = [r["label"] for r in evs[0][1]["sims"]]
            for n, (m, ev) in enumerate(evs):
                col = PALETTE[n % len(PALETTE)]
                for split, style in (("train", "hollow"), ("test", "points")):
                    rs = [(i, r) for i, r in enumerate(ev["sims"]) if r["split"] == split]
                    xs = [float(r["value"]) if numeric and r["value"] is not None else i for i, r in rs]
                    gseries.append((m["label"] if split == "test" else "", xs, [r["mean"]["rel_l2"] for _, r in rs], col, style))
        self.e_gen.xlabel = ai.PARAMS.get(man["param"], "") if man else ""
        self.e_gen.set(gseries, xticklabels=labels, empty="Evaluate a model to compare training and test conditions")

    def compare_selection(self, *_):
        evs = dict((m["id"], ev) for m, ev in self._evals())
        mid, sel = self.x_model.currentData(), self.x_sim.currentData()
        self._cmp = None
        if not mid or sel is None:
            for s in self.x_slices:
                s.image = None; s.update()
            self.x_info.setText("Evaluate a model to compare it with the ground truth."); return
        ev = evs[mid]
        r = next(r for r in ev["sims"] if (r["sim"], r["split"]) == tuple(sel))
        man = self.exp.manifest()
        X, mask = ai.load_sim(self.exp, r["sim"])
        pred = ai.predictions(self.exp, mid, r["sim"], r["split"])
        dims = man["dims"]
        gt = np.asarray(X[r["first"]:r["first"] + len(pred)], np.float32)
        pr = np.asarray(pred, np.float32)
        if dims == 2:
            pr = pr[:, :, None]
        self._cmp = {"gt": gt, "pred": pr, "mask": mask, "ev": ev, "r": r, "man": man}
        self.x_field.blockSignals(True); cur = self.x_field.currentText(); self.x_field.clear()
        names = (["|u|"] if "velocity" in man["groups"] else []) + man["channels"]
        self.x_field.addItems(names); self.x_field.setCurrentIndex(max(0, self.x_field.findText(cur))); self.x_field.blockSignals(False)
        self.x_axis.setVisible(dims == 3); self.x_pos.setVisible(dims == 3)
        self.x_t.blockSignals(True); self.x_t.setRange(0, len(pr) - 1); self.x_t.setValue(len(pr) - 1); self.x_t.blockSignals(False)
        v = man["groups"].get("velocity")
        if v:
            dmax = float(np.abs(pr[-1][v] - gt[-1][v]).max()); umax = float(np.abs(gt[-1][v]).max())
            self._gain = float(2 ** np.clip(np.round(np.log2(max(umax, 1e-9) / max(dmax, 1e-9))), 0, 6))
        else:
            self._gain = 1.0
        self.x_titles[2].setText(f"Difference ×{self._gain:g} (surrogate − truth)")
        if self.viewers and self.viewers.k != r["sim"]:
            self.viewers.stop(); self.viewers = None
            if self.x_mode.isChecked():
                self.start_viewers()
        self.show_compare()

    def _channel(self, a, name):
        man = self._cmp["man"]
        if name == "|u|":
            return np.sqrt((a[man["groups"]["velocity"]] ** 2).sum(0))
        return a[man["channels"].index(name)]

    def show_compare(self, *_):
        c = self._cmp
        if not c:
            return
        i = self.x_t.value()
        name = self.x_field.currentText()
        r = c["r"]
        self.x_tl.setText(f"step {r['steps'][i - 1] if i else 0:,} · rollout {i}/{len(c['pred']) - 1}")
        g, p = self._channel(c["gt"][i], name), self._channel(c["pred"][i], name)
        d = p - g if name != "|u|" else np.sqrt(((c["pred"][i] - c["gt"][i])[c["man"]["groups"]["velocity"]] ** 2).sum(0))
        axis, frac = self.x_axis.currentIndex(), self.x_pos.value() / 1000
        gv, ax = self._plane(g, c["mask"], axis, frac)
        pv, _ = self._plane(p, c["mask"], axis, frac)
        dv, _ = self._plane(d, c["mask"], axis, frac)
        signed = name not in ("|u|", "rho")
        fin = gv[np.isfinite(gv)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        if signed:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        meta = {"axis": ax, "step": 1, "signed": signed}
        for view, vals in ((self.x_slices[0], gv), (self.x_slices[1], pv)):
            view.lock, view.vrange = True, (lo, hi if hi > lo else lo + 1e-12)
            view.set_slice(vals, dict(meta), signed)
        dfin = np.abs(dv[np.isfinite(dv)])
        dm = float(dfin.max()) if dfin.size else 1.0
        self.x_slices[2].lock = True; self.x_slices[2].vrange = (0.0 if name == "|u|" else -dm, dm if dm > 0 else 1e-12)
        self.x_slices[2].set_slice(dv, {"axis": ax, "step": 1, "signed": name != "|u|"}, name != "|u|")
        ps = r["per_step"]
        k = max(0, i - 1)
        self.x_info.setText(f"At this step: relative L2 {pct(ps['rel_l2'][k]) if i else '0 %'} · max error {pct(ps['max_err'][k]) if i and 'max_err' in ps else '—'} · "
                            f"mass error {pct(ps['mass_err'][k], 3) if i and 'mass_err' in ps else '—'}. Same color range for truth and surrogate; difference on its own scale. "
                            "Solids are gray." + (f" 3D difference velocities are scaled ×{self._gain:g} for visibility." if self.x_mode.isChecked() else ""))
        if self.viewers:
            self.send_3d(i)

    # ------------------------------------------------------------ 3D comparison
    def toggle_3d(self, on):
        self.x_stack.setCurrentIndex(1 if on else 0)
        if on and self._cmp:
            if "velocity" not in self._cmp["man"]["groups"]:
                self.x_info.setText("3D comparison needs the velocity field in the dataset."); return
            self.start_viewers(); self.show_compare()
        elif not on and self.viewers:
            self.viewers.stop(); self.viewers = None

    def start_viewers(self):
        cam = list(self.exp.case.get("view", {}).get("camera", [-35, 25, 60, 1]))
        for p in self.x_ports:
            p.cam = list(cam); p.image = None; p.hint = "Starting renderer…"; p.update()
        self.viewers = Viewers(self.exp, self._cmp["r"]["sim"], self.x_ports)

    def send_3d(self, i):
        c, man = self._cmp, self._cmp["man"]
        g = man["groups"]
        snaps = (c["gt"][i], c["pred"][i], (c["pred"][i] - c["gt"][i]))
        for n, x in enumerate(snaps):
            if n == 2:
                x = x.copy()
                if "pressure" in g:
                    x[g["pressure"][0]] = 0.0
                if "density" in g:
                    x[g["density"][0]] = 1.0
            self.viewers.show(n, lambda path, x=x, n=n: ai.write_snapshot(path, x, c["mask"], g, self._gain if n == 2 else 1.0))

    def sync_camera(self, src):
        for p in self.x_ports:
            if p is not src:
                p.cam = list(src.cam)
        self._cam_dirty = True

    # ------------------------------------------------------------ benchmarks
    def refresh_bench(self):
        evs = self._evals()
        rows, sp, acc = [], [], []
        for n, (m, ev) in enumerate(evs):
            p = ev["performance"]
            tests = [r for r in ev["sims"] if r["split"] == "test"] or ev["sims"]
            steps = sum(len(r["steps"]) for r in tests)
            rows.append((m["label"], p.get("runtime", ""), f"{steps:,}", secs(p["surrogate_s"]), f"{p['step_ms']:.2f} ms", secs(p["solver_s"]),
                         f"{p['speedup']:.3g}×" if p.get("speedup") else "—", grid(tests[0]["grid"]), grid(tests[0]["solver_grid"]),
                         f"{p['parameters']:,}" if p.get("parameters") else "—", f"{(p['parameter_mb'] or 0) + p['state_mb']:.2f} MB",
                         f"{p['solver_memory_mb']:,} MB" if p.get("solver_memory_mb") else "—", f"{p['memory_ratio']:.3g}×" if p.get("memory_ratio") else "—",
                         secs(p.get("train_seconds")) if p.get("train_seconds") else "—"))
            col = PALETTE[n % len(PALETTE)]
            if p.get("speedup"):
                sp.append((m["label"], [n], [p["speedup"]], col, "points"))
                acc.append((m["label"], [p["speedup"]], [ev["aggregate"]["test"]["rel_l2"]], col, "points"))
        fill(self.b_table, rows)
        man = self.exp.manifest() if self.exp else None
        dev = man["sims"][0].get("device") if man else ""
        self.b_note.setText(f"<span style='color:{C['muted']}'>Solver time is the measured kernel time per step on <b>{dev}</b> at the solver grid, "
                            "multiplied by the steps the rollout covers. Surrogate time is the measured wall time of the autoregressive rollout "
                            "(one untimed warm-up step) on the dataset grid. Model memory counts parameters plus input and output state; "
                            "solver memory is the device memory of the ground-truth run.</span>" if man else "")
        self.b_speed.set(sp, xticklabels=[m["label"] for m, _ in evs] if sp else None, empty="Evaluate models to compare speed")
        self.b_acc.set(acc, empty="Evaluate models to compare accuracy and speed")

    # ------------------------------------------------------------ analytics
    def refresh_analytics(self, *_):
        rows, kinds = [], []
        f = self.a_filter.currentIndex()
        for r in analytics.collect():
            if (f == 1 and r["type"] != "simulation") or (f == 2 and r["type"] != "ai_experiment"):
                continue
            kinds.append(r)
            ai_ = r["type"] == "ai_experiment"
            rows.append(("AI experiment" if ai_ else "Simulation", r["name"], (r.get("created") or "").replace("T", " "), f"{(r.get('cells') or 0) / 1e6:.2f} M",
                         "" if ai_ else f"{r.get('steps', 0):,}", secs(r.get("runtime_s")) if r.get("runtime_s") else "", f"{r['mlups']:.0f}" if r.get("mlups") else "",
                         f"{r['memory_mb']:,} MB" if r.get("memory_mb") else "", f"{r['cd']:.3f}" if r.get("cd") is not None else "", f"{r['cl']:.3f}" if r.get("cl") is not None else "",
                         pct(r["ke_change"], 1) if r.get("ke_change") is not None else "", r.get("best_model", ""), pct(r.get("rel_l2")) if r.get("rel_l2") is not None else "",
                         pct(r.get("mass_err"), 3) if r.get("mass_err") is not None else "",
                         f"{r['stable_steps']:,}/{r['horizon_steps']:,}" if r.get("stable_steps") is not None else "", f"{r['speedup']:.3g}×" if r.get("speedup") else ""))
        self._a_rows = kinds
        fill(self.a_table, rows)

    def open_from_analytics(self, item):
        r = self._a_rows[item.row()]
        d = runner.RUNS / r["dir"]
        if r["type"] == "ai_experiment":
            self.refresh_list(d); self.open(d); self.main.nav_ai(0)
        else:
            self.main.leave_ai(); self.main.open_run_dir(d)

    # ------------------------------------------------------------ jobs
    def start_job(self, label, fn):
        if self.job and self.job.is_alive():
            QtWidgets.QMessageBox.information(self, "Busy", f"{self.jstate.get('text', 'A job')} is still running."); return
        self.cancel = threading.Event()
        st = self.jstate = {"frac": 0.0, "text": label, "error": None, "done": False}

        def progress(frac, text):
            st["frac"], st["text"] = frac, text

        def work():
            try:
                fn(progress, self.cancel)
            except Exception as ex:
                st["error"] = str(ex)
            finally:
                st["done"] = True
        self.job = threading.Thread(target=work, daemon=True); self.job.start()
        self.job_bar.show(); self.cancel_btn.show(); self.refresh_pipeline()

    def tick(self):
        st = self.jstate
        if st:
            self.job_label.setText(st["text"]); self.job_bar.setValue(int(1000 * st["frac"]))
            if st["done"]:
                self.jstate = {}
                self.job_bar.hide(); self.cancel_btn.hide()
                if st["error"]:
                    self.job_label.setText(f"Stopped: {st['error']}")
                    if st["error"] != "cancelled":
                        QtWidgets.QMessageBox.warning(self, "AI experiment", st["error"])
                else:
                    self.job_label.setText("Done")
                self.reload(); self.refresh(); self.refresh_list(self.exp.dir if self.exp else None); self.main.refresh_runs()
        if self.viewers:
            self.viewers.tick()
            if getattr(self, "_cam_dirty", False):
                self._cam_dirty = False
                self.viewers.send("cam " + " ".join(f"{x:.3f}" for x in self.x_ports[0].cam))
            if self.viewers.err:
                for p in self.x_ports:
                    p.hint = f"Renderer: {self.viewers.err}"; p.update()

    def shutdown(self):
        self.cancel.set()
        if self.viewers:
            self.viewers.stop()
