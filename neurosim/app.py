"""NeuroSim desktop workbench (PyQt5).

The 3D view is rendered by FluidX3D's OpenCL renderer inside the solver worker, directly from GPU memory;
frames arrive as raw RGB32 pixels and are painted without decoding. Everything else (setup, slices,
monitoring, runs) is native Qt.
"""
import copy
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import case as case_mod, geometry, presets, runner

MODELS = runner.WORKSPACE / "models"
VIS_BUTTONS = [("Domain", "lattice"), ("Solids", "surface"), ("Vortices", "q_criterion"), ("Streamlines", "streamlines"),
               ("Velocity field", "field"), ("Liquid surface", "free_surface"), ("Raytraced liquid", "raytrace"), ("Particles", "particles")]

CLOUD_MODES = [("Speed haze", 4), ("Vortex cloud", 0), ("Wake cloud", 1), ("Thermal cloud", 2), ("Pressure waves", 3)]

# ------------------------------------------------------------------ design tokens
C = {"bg": "#0a0c10", "panel": "#11141a", "card": "#161a21", "card2": "#1c212a", "border": "#252b36", "text": "#e7eaf0", "muted": "#8b94a5",
     "dim": "#5d6575", "accent": "#4f8cff", "accent2": "#7aa7ff", "ok": "#34d399", "warn": "#fbbf24", "bad": "#f87171", "orange": "#ff9f43"}
MONO = "Cascadia Mono, Consolas, monospace"

STYLE = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif; font-size: 9.5pt; color: {C['text']}; }}
QMainWindow, QWidget#root {{ background: {C['bg']}; }}
QFrame#topbar {{ background: {C['panel']}; border-bottom: 1px solid {C['border']}; }}
QFrame#sidebar, QWidget#sidebarInner, QFrame#rightpanel {{ background: {C['panel']}; }}
QFrame#sidebar {{ border-right: 1px solid {C['border']}; }}
QFrame#rightpanel {{ border-left: 1px solid {C['border']}; }}
QLabel#brand {{ font-size: 13pt; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#brandSub {{ color: {C['muted']}; font-size: 8.5pt; }}
QLabel#section {{ color: {C['muted']}; font-size: 8pt; font-weight: 600; letter-spacing: 1.2px; padding-top: 10px; }}
QLabel#hint {{ color: {C['muted']}; font-size: 8.5pt; }}
QLabel#chip {{ border-radius: 10px; padding: 2px 10px; font-size: 8.5pt; font-weight: 600; }}
QFrame#card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 10px; }}
QPushButton {{ background: {C['card2']}; border: 1px solid {C['border']}; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {C['accent']}; background: #222834; }}
QPushButton:pressed {{ background: #2a3140; }}
QPushButton:disabled {{ color: {C['dim']}; }}
QPushButton#primary {{ background: {C['accent']}; border: 1px solid {C['accent']}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: {C['accent2']}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#ghost:hover {{ background: {C['card2']}; border-color: {C['border']}; }}
QPushButton#toggle {{ background: transparent; border: 1px solid {C['border']}; border-radius: 13px; padding: 4px 12px; color: {C['muted']}; }}
QPushButton#toggle:checked {{ background: rgba(79,140,255,0.18); border-color: {C['accent']}; color: {C['text']}; }}
QPushButton#toggle:hover {{ border-color: {C['accent2']}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 4px 8px; min-height: 20px; selection-background-color: {C['accent']}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: #34405a; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {C['card']}; border: 1px solid {C['border']}; selection-background-color: {C['accent']}; outline: none; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {C['border']}; background: {C['card']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QListWidget {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 8px; padding: 4px; outline: none; }}
QListWidget::item {{ padding: 6px 8px; border-radius: 5px; }}
QListWidget::item:selected {{ background: rgba(79,140,255,0.22); color: {C['text']}; }}
QListWidget::item:hover {{ background: {C['card2']}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {C['muted']}; padding: 8px 14px; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom-color: {C['accent']}; }}
QTabBar::tab:hover {{ color: {C['text']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {C['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: white; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {C['bg']}; }}
QToolTip {{ background: {C['card2']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 4px; }}
QStatusBar {{ background: {C['panel']}; color: {C['muted']}; border-top: 1px solid {C['border']}; }}
QTextEdit {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 8px; }}
QProgressBar {{ background: {C['card']}; border: none; border-radius: 2px; max-height: 3px; }}
QProgressBar::chunk {{ background: {C['accent']}; }}
QPushButton#aimode {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(79,140,255,0.20), stop:1 rgba(168,85,247,0.28));
    border: 1px solid #7c6cf7; border-radius: 7px; padding: 6px 14px; font-weight: 600; }}
QPushButton#aimode:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(79,140,255,0.32), stop:1 rgba(168,85,247,0.42)); }}
QPushButton#nav {{ background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0; padding: 8px 12px; color: {C['muted']}; font-weight: 600; }}
QPushButton#nav:checked {{ color: {C['text']}; border-bottom-color: #a855f7; }}
QPushButton#nav:hover {{ color: {C['text']}; }}
QFrame#modelcard {{ background: {C['card2']}; border: 1px solid {C['border']}; border-radius: 10px; }}
QTableWidget {{ background: {C['card']}; alternate-background-color: {C['card2']}; border: 1px solid {C['border']}; border-radius: 8px; selection-background-color: rgba(79,140,255,0.25); outline: none; }}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{ background: {C['panel']}; color: {C['muted']}; border: none; border-bottom: 1px solid {C['border']}; padding: 6px 8px; font-weight: 600; }}
QDialog {{ background: {C['bg']}; }}
"""


def lut(name):
    from matplotlib import cm
    return (cm.get_cmap(name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def icon(name, color="#e7eaf0", size=18):
    """Small vector icons drawn with QPainter (crisp at any DPI, no asset files)."""
    pm = QtGui.QPixmap(size * 2, size * 2)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(2 * size / 24, 2 * size / 24)
    col = QtGui.QColor(color)
    pen = QtGui.QPen(col, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    path = QtGui.QPainterPath()
    if name == "play":
        path.moveTo(7, 5); path.lineTo(19, 12); path.lineTo(7, 19); path.closeSubpath(); p.fillPath(path, col)
    elif name == "pause":
        p.fillRect(QtCore.QRectF(6, 5, 4, 14), col); p.fillRect(QtCore.QRectF(14, 5, 4, 14), col)
    elif name == "stop":
        p.fillRect(QtCore.QRectF(6, 6, 12, 12), col)
    elif name == "eye":
        path.moveTo(2, 12); path.quadTo(12, 3, 22, 12); path.quadTo(12, 21, 2, 12); p.drawPath(path); p.drawEllipse(QtCore.QPointF(12, 12), 3, 3)
    elif name == "save":
        p.drawRoundedRect(QtCore.QRectF(4, 4, 16, 16), 2, 2); p.drawLine(8, 4, 8, 9); p.drawLine(8, 9, 15, 9); p.drawLine(15, 4, 15, 9)
    elif name == "export":
        p.drawLine(12, 4, 12, 15); p.drawLine(7, 10, 12, 15); p.drawLine(17, 10, 12, 15); p.drawLine(5, 20, 19, 20)
    elif name == "camera":
        p.drawRoundedRect(QtCore.QRectF(3, 7, 18, 12), 2, 2); p.drawEllipse(QtCore.QPointF(12, 13), 3.5, 3.5); p.drawLine(9, 7, 10, 4); p.drawLine(10, 4, 14, 4); p.drawLine(14, 4, 15, 7)
    elif name == "chip":
        p.drawRoundedRect(QtCore.QRectF(6, 6, 12, 12), 2, 2)
        for k in (9, 12, 15):
            p.drawLine(k, 2, k, 6); p.drawLine(k, 18, k, 22); p.drawLine(2, k, 6, k); p.drawLine(18, k, 22, k)
    elif name == "import":
        p.drawLine(12, 15, 12, 4); p.drawLine(7, 9, 12, 4); p.drawLine(17, 9, 12, 4); p.drawLine(5, 20, 19, 20)
    elif name == "plus":
        p.drawLine(12, 5, 12, 19); p.drawLine(5, 12, 19, 12)
    elif name == "trash":
        p.drawLine(5, 7, 19, 7); p.drawLine(10, 4, 14, 4); p.drawRoundedRect(QtCore.QRectF(7, 7, 10, 13), 1.5, 1.5)
    elif name == "folder":
        path.moveTo(3, 7); path.lineTo(9, 7); path.lineTo(11, 9); path.lineTo(21, 9); path.lineTo(21, 19); path.lineTo(3, 19); path.closeSubpath(); p.drawPath(path)
    elif name == "refresh":
        p.drawArc(QtCore.QRectF(5, 5, 14, 14), 30 * 16, 300 * 16); p.drawLine(19, 5, 19, 10); p.drawLine(19, 10, 14, 10)
    elif name == "neural":  # small network: three inputs, two hidden, one output
        nodes = [(4, 6), (4, 12), (4, 18), (12, 8.5), (12, 15.5), (20, 12)]
        p.setPen(QtGui.QPen(col, 1.2))
        for a in nodes[:3]:
            for b in nodes[3:5]:
                p.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*b))
        for a in nodes[3:5]:
            p.drawLine(QtCore.QPointF(*a), QtCore.QPointF(*nodes[5]))
        p.setPen(Qt.NoPen); p.setBrush(col)
        for x, y in nodes:
            p.drawEllipse(QtCore.QPointF(x, y), 2.3, 2.3)
    elif name == "logo":
        g = QtGui.QLinearGradient(0, 0, 24, 24); g.setColorAt(0, QtGui.QColor("#4f8cff")); g.setColorAt(1, QtGui.QColor("#a855f7"))
        p.setPen(QtGui.QPen(QtGui.QBrush(g), 2.4, Qt.SolidLine, Qt.RoundCap))
        for k, a in enumerate((0, 1.3, 2.6)):
            path = QtGui.QPainterPath(); path.moveTo(2, 8 + 4 * k)
            path.cubicTo(8, 2 + 4 * k + a, 14, 14 + 4 * k - a, 22, 8 + 4 * k); p.drawPath(path)
    p.end()
    pm.setDevicePixelRatio(2)
    return QtGui.QIcon(pm)


def section(text):
    l = QtWidgets.QLabel(text.upper()); l.setObjectName("section")
    return l


def rgba(color, alpha):
    """'#rrggbb' with alpha 0..255 as a QSS rgba() (QSS reads 8-digit hex as #AARRGGBB, not #RRGGBBAA)."""
    c = QtGui.QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def chip_style(color):
    return f"background: {rgba(color, 34)}; color: {color}; border: 1px solid {rgba(color, 85)};"


def chip(text, color):
    l = QtWidgets.QLabel(text); l.setObjectName("chip")
    l.setStyleSheet(chip_style(color))
    return l


# ======================================================================== widgets


class Viewport(QtWidgets.QWidget):
    """Solver frames on pure black; drag orbits, wheel zooms, shift+wheel changes field of view."""
    camera_changed = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 300)
        self.image = None
        self._buf = None
        self.cam = [-35.0, 25.0, 60.0, 1.0]
        self.hud = []  # [(label, value)]
        self.status = ("", C["muted"])
        self.hint = "Choose a preset or import a model, then Preview or Run"
        self._drag = None
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setCursor(Qt.OpenHandCursor)

    def set_frame(self, data, w, h):
        self._buf = data  # QImage borrows the buffer
        self.image = QtGui.QImage(self._buf, w, h, 4 * w, QtGui.QImage.Format_RGB32)
        self.update()

    def target_rect(self):
        s = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = int(self.image.width() * s), int(self.image.height() * s)
        return QtCore.QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.image is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            p.drawImage(self.target_rect(), self.image)
        else:
            p.setPen(QtGui.QColor(C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, self.hint)
        x = 14
        if self.status[0]:
            x = self._chip(p, x, 14, self.status[0], QtGui.QColor(self.status[1]))
        for label, value in self.hud:
            x = self._chip(p, x, 14, f"{label}  {value}", QtGui.QColor(C["text"]), subtle=True)
        p.setFont(QtGui.QFont("Segoe UI", 8)); p.setPen(QtGui.QColor(C["dim"]))
        p.drawText(self.rect().adjusted(14, 0, -14, -10), Qt.AlignLeft | Qt.AlignBottom, "drag: orbit   ·   wheel: zoom   ·   shift + wheel: field of view")

    def _chip(self, p, x, y, text, color, subtle=False):
        p.setFont(QtGui.QFont("Cascadia Mono", 8) if subtle else QtGui.QFont("Segoe UI", 8, QtGui.QFont.DemiBold))
        w = p.fontMetrics().horizontalAdvance(text) + 20
        r = QtCore.QRectF(x, y, w, 24)
        bg = QtGui.QColor(18, 22, 30, 200) if subtle else QtGui.QColor(color.red(), color.green(), color.blue(), 45)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 25) if subtle else QtGui.QColor(color.red(), color.green(), color.blue(), 120), 1))
        p.setBrush(bg); p.drawRoundedRect(r, 12, 12)
        p.setPen(color); p.drawText(r, Qt.AlignCenter, text)
        return x + w + 8

    def mousePressEvent(self, e):
        self._drag = (e.pos(), list(self.cam)); self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        p0, c0 = self._drag
        d = e.pos() - p0
        self.cam[0] = c0[0] - 0.3 * d.x()
        self.cam[1] = max(-89.9, min(89.9, c0[1] + 0.3 * d.y()))
        self.camera_changed.emit(*self.cam)

    def mouseReleaseEvent(self, e):
        self._drag = None; self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, e):
        k = 1.12 ** (e.angleDelta().y() / 120)
        if e.modifiers() & Qt.ShiftModifier:
            self.cam[2] = max(5.0, min(170.0, self.cam[2] / k))
        else:
            self.cam[3] = max(0.05, min(50.0, self.cam[3] * k))
        self.camera_changed.emit(*self.cam)


class SliceView(QtWidgets.QWidget):
    """Colormapped 2D slice with colorbar and hover readout."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 220)
        self.setMouseTracking(True)
        self.values, self.meta, self.image = None, {}, None
        self.vrange = (0.0, 1.0)
        self.lock = False
        self.hover = ""
        self.luts = {"seq": lut("turbo"), "div": lut("coolwarm")}

    def set_slice(self, values, meta, signed):
        self.values, self.meta = values, meta
        finite = values[np.isfinite(values)]
        if not self.lock and finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if signed:
                m = max(abs(lo), abs(hi)); lo, hi = -m, m
            self.vrange = (lo, hi if hi > lo else lo + 1e-12)
        lo, hi = self.vrange
        idx = np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        rgb = self.luts["div" if signed else "seq"][idx]
        rgb[~np.isfinite(values)] = (40, 44, 52)  # solid cells
        self._buf = np.ascontiguousarray(rgb[::-1])  # second in-plane axis points up
        h, w = values.shape
        self.image = QtGui.QImage(self._buf.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.update()

    def _rect(self):
        avail = self.rect().adjusted(8, 8, -64, -30)
        s = min(avail.width() / self.image.width(), avail.height() / self.image.height())
        w, h = int(self.image.width() * s), int(self.image.height() * s)
        return QtCore.QRect(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2, w, h)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(C["card"]))
        if self.image is None:
            p.setPen(QtGui.QColor(C["muted"])); p.drawText(self.rect(), Qt.AlignCenter, "Slice appears when a run starts"); return
        r = self._rect()
        p.drawImage(r, self.image)
        lut_ = self.luts["div" if self.meta.get("signed") else "seq"]
        bar = QtCore.QRect(r.right() + 14, r.top(), 10, r.height())
        for i in range(bar.height()):
            c = lut_[int(255 * (1 - i / max(bar.height() - 1, 1)))]
            p.setPen(QtGui.QColor(*c)); p.drawLine(bar.left(), bar.top() + i, bar.right(), bar.top() + i)
        p.setPen(QtGui.QColor(C["text"])); p.setFont(QtGui.QFont("Cascadia Mono", 8))
        lo, hi = self.vrange
        p.drawText(bar.right() + 5, bar.top() + 9, f"{hi:.3g}")
        p.drawText(bar.right() + 5, bar.bottom(), f"{lo:.3g}")
        p.setPen(QtGui.QColor(C["muted"]))
        p.drawText(QtCore.QRect(8, self.height() - 24, self.width() - 16, 20), Qt.AlignLeft | Qt.AlignVCenter, self.hover or "hover for values")

    def mouseMoveEvent(self, e):
        if self.values is None:
            return
        r = self._rect()
        if not r.contains(e.pos()):
            self.hover = ""; self.update(); return
        h, w = self.values.shape
        i = min(int((e.x() - r.left()) / r.width() * w), w - 1)
        j = min(h - 1 - int((e.y() - r.top()) / r.height() * h), h - 1)
        v = self.values[j, i]
        step = self.meta.get("step", 1)
        a, b = {0: ("y", "z"), 1: ("x", "z"), 2: ("x", "y")}[self.meta.get("axis", 0)]
        self.hover = f"{a} = {i * step}   {b} = {j * step}   " + ("solid" if not np.isfinite(v) else f"value = {v:.5g}")
        self.update()


class SparkCard(QtWidgets.QFrame):
    """KPI card: title, large current value, sparkline with gradient fill (up to 2 series)."""
    COLORS = [C["accent"], C["orange"]]

    def __init__(self, title, unit="", fmt="{:.4g}"):
        super().__init__()
        self.setObjectName("card")
        self.title, self.unit, self.fmt = title, unit, fmt
        self.series = {}
        self.setMinimumSize(170, 104)

    def set(self, series):
        self.series = {k: (np.asarray(x, float), np.asarray(y, float)) for k, (x, y) in series.items()}
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QColor(C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.DemiBold))
        p.drawText(14, 22, self.title.upper())
        data = [(k, x, y) for k, (x, y) in self.series.items() if len(x) > 0]
        if not data:
            p.setPen(QtGui.QColor(C["dim"])); p.setFont(QtGui.QFont("Cascadia Mono", 15)); p.drawText(14, 52, "—"); return
        k0, x0_, y0_ = data[0]
        p.setPen(QtGui.QColor(C["text"])); p.setFont(QtGui.QFont("Cascadia Mono", 15))
        val = self.fmt.format(y0_[-1]) if np.isfinite(y0_[-1]) else "—"
        p.drawText(14, 52, val)
        vw = p.fontMetrics().horizontalAdvance(val)
        p.setPen(QtGui.QColor(C["muted"])); p.setFont(QtGui.QFont("Segoe UI", 8))
        sub = self.unit + ("   " + "   ".join(f"{k} {self.fmt.format(y[-1])}" for k, x, y in data[1:]) if len(data) > 1 else "")
        p.drawText(14 + vw + 8, 52, sub)
        area = QtCore.QRectF(10, 60, self.width() - 20, self.height() - 68)
        xs = np.concatenate([x for _, x, _ in data]); ys = np.concatenate([y[np.isfinite(y)] for _, _, y in data])
        if len(xs) < 2 or ys.size == 0:
            return
        xa, xb, ya, yb = xs.min(), xs.max(), ys.min(), ys.max()
        if yb - ya < 1e-12 * max(1.0, abs(yb)):
            ya, yb = ya - 1e-9 - 1e-3 * abs(ya), yb + 1e-9 + 1e-3 * abs(yb)
        for n, (k, x, y) in enumerate(reversed(data)):
            col = QtGui.QColor(self.COLORS[(len(data) - 1 - n) % 2])
            px = area.left() + (x - xa) / max(xb - xa, 1e-12) * area.width()
            py = area.bottom() - (np.nan_to_num(y, nan=ya) - ya) / (yb - ya) * area.height()
            line = QtGui.QPolygonF([QtCore.QPointF(a, b) for a, b in zip(px, py)])
            if n == len(data) - 1:
                fill = QtGui.QPolygonF(line); fill.append(QtCore.QPointF(px[-1], area.bottom())); fill.append(QtCore.QPointF(px[0], area.bottom()))
                g = QtGui.QLinearGradient(0, area.top(), 0, area.bottom())
                c1 = QtGui.QColor(col); c1.setAlpha(70); c2 = QtGui.QColor(col); c2.setAlpha(0)
                g.setColorAt(0, c1); g.setColorAt(1, c2)
                p.setPen(Qt.NoPen); p.setBrush(g); p.drawPolygon(fill)
            p.setBrush(Qt.NoBrush); p.setPen(QtGui.QPen(col, 1.6)); p.drawPolyline(line)


class Field(QtWidgets.QWidget):
    """Label above an input (modern form row)."""

    def __init__(self, label, *widgets, hint=None):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 2, 0, 4); lay.setSpacing(4)
        l = QtWidgets.QLabel(label); l.setStyleSheet(f"color: {C['muted']}; font-size: 8.5pt;")
        lay.addWidget(l)
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        for w in widgets:
            row.addWidget(w)
        lay.addLayout(row)
        if hint:
            h = QtWidgets.QLabel(hint); h.setObjectName("hint"); h.setWordWrap(True); lay.addWidget(h)


# ======================================================================== main window


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroSim CFD")
        self.setWindowIcon(icon("logo", size=32))
        self.resize(1680, 1000)
        MODELS.mkdir(parents=True, exist_ok=True)
        self.run = None
        self._run_case = None
        self.case = case_mod.normalize(presets.PRESETS["Car on moving road"]); self.case["name"] = "Car on moving road"
        self._pending = None
        self._cam_dirty = False
        self._series = {}
        self._start = {}
        self._building = False
        self._probe = self._load_probe()
        self.f, self.o = {}, {}

        root = QtWidgets.QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self._topbar())
        self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 0); self.progress.setTextVisible(False); self.progress.hide(); v.addWidget(self.progress)
        split = QtWidgets.QSplitter(Qt.Horizontal); split.setHandleWidth(1)
        split.addWidget(self._sidebar())
        split.addWidget(self._center())
        split.addWidget(self._rightpanel())
        split.setStretchFactor(1, 1); split.setSizes([400, 920, 400])
        self.body = QtWidgets.QStackedWidget(); self.body.addWidget(split)  # page 1: AI Mode workspace, created on first use
        self.ai = None
        v.addWidget(self.body, 1)
        self.statusBar().showMessage("Ready")
        self.set_form(self.case)
        self.preset.setCurrentText(self.case["name"])
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(30)

    # ------------------------------------------------------------ layout
    def _btn(self, text, ico=None, fn=None, obj=None, tip=None):
        b = QtWidgets.QPushButton(text)
        if ico:
            b.setIcon(icon(ico, "white" if obj == "primary" else C["text"])); b.setIconSize(QtCore.QSize(16, 16))
        if obj:
            b.setObjectName(obj)
        if fn:
            b.clicked.connect(fn)
        if tip:
            b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _topbar(self):
        bar = QtWidgets.QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(58)
        h = QtWidgets.QHBoxLayout(bar); h.setContentsMargins(16, 8, 16, 8); h.setSpacing(10)
        logo = QtWidgets.QLabel(); logo.setPixmap(icon("logo", size=28).pixmap(28, 28)); h.addWidget(logo)
        t = QtWidgets.QVBoxLayout(); t.setSpacing(0)
        a = QtWidgets.QLabel("NeuroSim"); a.setObjectName("brand"); self.brand_sub = QtWidgets.QLabel("CFD workbench · FluidX3D core"); self.brand_sub.setObjectName("brandSub")
        t.addWidget(a); t.addWidget(self.brand_sub); h.addLayout(t)
        h.addSpacing(14)
        self.ai_btn = self._btn("AI Mode", None, lambda: self.enter_ai(), "aimode", "AI Mode · Neural Surrogates & Physics Experiments")
        self.ai_btn.setIcon(icon("neural", "#c4b5fd")); self.ai_btn.setIconSize(QtCore.QSize(18, 18))
        self.sim_btn = self._btn("←  Simulation Mode", None, self.leave_ai, "ghost", "Back to the simulation workbench"); self.sim_btn.hide()
        h.addWidget(self.ai_btn); h.addWidget(self.sim_btn)
        h.addSpacing(10)
        sim = QtWidgets.QWidget(); hs = QtWidgets.QHBoxLayout(sim); hs.setContentsMargins(0, 0, 0, 0); hs.setSpacing(10)
        self.preset = QtWidgets.QComboBox(); self.preset.addItems(list(presets.PRESETS)); self.preset.setMinimumWidth(260)
        self.preset.activated[str].connect(self.load_preset); self.preset.setToolTip("Ready-made cases")
        hs.addWidget(self.preset)
        hs.addWidget(self._btn("Open", "folder", self.open_case, "ghost", "Open a case file"))
        hs.addWidget(self._btn("Save", "save", self.save_case, "ghost", "Save this case as JSON"))
        hs.addSpacing(18)
        hs.addWidget(self._btn("Preview", "eye", self.preview, None, "Build the scene and show geometry without stepping"))
        hs.addWidget(self._btn("Run", "play", self.start_run, "primary", "Start, or continue a paused run"))
        hs.addWidget(self._btn("Pause", "pause", lambda: self.cmd("pause"), None, "Pause the solver"))
        hs.addWidget(self._btn("Stop", "stop", self.stop_run, None, "Stop the solver"))
        self.state_chip = chip("idle", C["muted"]); hs.addWidget(self.state_chip)
        hs.addStretch(1)
        hs.addWidget(self._btn("Checkpoint", "save", lambda: self.cmd("checkpoint"), "ghost", "Save the complete solver state (restartable)"))
        hs.addWidget(self._btn("Export", "export", lambda: self.cmd("export"), "ghost", "Write rho/u/T/phi fields as .npy"))
        hs.addWidget(self._btn("Snapshot", "camera", self.save_image, "ghost", "Save the 3D view as PNG"))
        hs.addWidget(self._btn("Hardware", "chip", self.hardware, "ghost", "Devices and benchmark"))
        h.addWidget(sim, 1); self.sim_bar = sim
        self.ai_bar = QtWidgets.QWidget(); ha = QtWidgets.QHBoxLayout(self.ai_bar); ha.setContentsMargins(0, 0, 0, 0); ha.setSpacing(2)
        self.ai_nav = QtWidgets.QButtonGroup(self)
        from .ai_mode import NAV
        for i, name in enumerate(NAV):
            b = QtWidgets.QPushButton(name); b.setObjectName("nav"); b.setCheckable(True); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.nav_ai(i)); self.ai_nav.addButton(b, i); ha.addWidget(b)
        ha.addStretch(1)
        ha.addWidget(self._btn("Hardware", "chip", self.hardware, "ghost", "Devices and benchmark"))
        self.ai_bar.hide(); h.addWidget(self.ai_bar, 1)
        self.topbar = bar
        return bar

    def _sidebar(self):
        frame = QtWidgets.QFrame(); frame.setObjectName("sidebar"); frame.setMinimumWidth(360)
        lay = QtWidgets.QVBoxLayout(frame); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        tabs = QtWidgets.QTabWidget(); tabs.setDocumentMode(True); tabs.tabBar().setExpanding(True)
        for title, build_fn in (("Scene", self._tab_scene), ("Physics", self._tab_physics), ("Solver", self._tab_solver)):
            inner = QtWidgets.QWidget(); inner.setObjectName("sidebarInner")
            l = QtWidgets.QVBoxLayout(inner); l.setContentsMargins(16, 6, 16, 16); l.setSpacing(2)
            build_fn(l); l.addStretch(1)
            sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(inner)
            tabs.addTab(sc, title)
        lay.addWidget(tabs, 1)
        self.mem_label = QtWidgets.QLabel(); self.mem_label.setWordWrap(True)
        self.mem_label.setStyleSheet(f"background: {C['card']}; border-top: 1px solid {C['border']}; padding: 10px 16px; color: {C['muted']};")
        lay.addWidget(self.mem_label)
        return frame

    def _center(self):
        w = QtWidgets.QWidget(); w.setObjectName("root")
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        bar = QtWidgets.QHBoxLayout(); bar.setContentsMargins(12, 8, 12, 8); bar.setSpacing(6)
        self.vis = {}
        for label, key in VIS_BUTTONS:
            b = QtWidgets.QPushButton(label); b.setObjectName("toggle"); b.setCheckable(True); b.toggled.connect(self.send_vis); b.setCursor(Qt.PointingHandCursor)
            self.vis[key] = b; bar.addWidget(b)
        bar.addSpacing(10)
        self.cloud_btn = QtWidgets.QPushButton("Haze"); self.cloud_btn.setObjectName("toggle"); self.cloud_btn.setCheckable(True); self.cloud_btn.setCursor(Qt.PointingHandCursor)
        self.cloud_btn.setToolTip("Faint translucent overlay rendered on the GPU: speed haze tints only regions faster than the free stream; clouds are for gas/thermal cases")
        self.cloud_mode = QtWidgets.QComboBox(); self.cloud_mode.addItems([m for m, _ in CLOUD_MODES])
        self.cloud_amt = QtWidgets.QSlider(Qt.Horizontal); self.cloud_amt.setRange(0, 100); self.cloud_amt.setValue(44); self.cloud_amt.setFixedWidth(80); self.cloud_amt.setToolTip("Overlay intensity")
        for w_ in (self.cloud_btn, self.cloud_mode, self.cloud_amt):
            bar.addWidget(w_)
        self.cloud_btn.toggled.connect(self.send_cloud); self.cloud_mode.currentIndexChanged.connect(self.send_cloud); self.cloud_amt.valueChanged.connect(self.send_cloud)
        bar.addStretch(1)
        self.vis_field = QtWidgets.QComboBox(); self.vis_field.addItems(["Color: velocity", "Color: density", "Color: temperature"]); self.vis_field.currentIndexChanged.connect(self.send_vis)
        self.vis_slice = QtWidgets.QComboBox(); self.vis_slice.addItems(["Field: volume", "Field: x slice", "Field: y slice", "Field: z slice", "Field: xz", "Field: xyz", "Field: yz", "Field: xy"]); self.vis_slice.currentIndexChanged.connect(self.send_vis)
        self.vis_pos = QtWidgets.QSlider(Qt.Horizontal); self.vis_pos.setRange(0, 1000); self.vis_pos.setValue(500); self.vis_pos.setFixedWidth(110); self.vis_pos.valueChanged.connect(self.send_vis)
        for x in (self.vis_field, self.vis_slice, self.vis_pos):
            bar.addWidget(x)
        v.addLayout(bar)
        vsplit = QtWidgets.QSplitter(Qt.Vertical); vsplit.setHandleWidth(1)
        self.view = Viewport(); self.view.camera_changed.connect(lambda *a: setattr(self, "_cam_dirty", True))
        vsplit.addWidget(self.view)
        mon = QtWidgets.QWidget(); mon.setObjectName("root")
        g = QtWidgets.QGridLayout(mon); g.setContentsMargins(12, 10, 12, 12); g.setSpacing(10)
        self.cards = {"perf": SparkCard("Throughput", "MLUP/s", "{:.0f}"), "u": SparkCard("Max velocity", "lattice"), "force": SparkCard("Drag Cd", "", "{:.3f}"),
                      "ke": SparkCard("Kinetic energy", ""), "mass": SparkCard("Mass drift", "", "{:+.2e}"), "extra": SparkCard("Temperature / liquid", "")}
        for n, c in enumerate(self.cards.values()):
            g.addWidget(c, n // 3, n % 3)
        self.insight = QtWidgets.QLabel("Run a case to see performance and physics insight here.")
        self.insight.setObjectName("card"); self.insight.setWordWrap(True); self.insight.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.insight.setStyleSheet(f"QLabel#card {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 10px; padding: 12px; }}")
        self.insight.setTextInteractionFlags(Qt.TextSelectableByMouse); self.insight.setMinimumWidth(300)
        g.addWidget(self.insight, 0, 3, 2, 1)
        g.setColumnStretch(3, 1)
        vsplit.addWidget(mon)
        vsplit.setStretchFactor(0, 1); vsplit.setSizes([640, 250])
        v.addWidget(vsplit, 1)
        return w

    def _rightpanel(self):
        frame = QtWidgets.QFrame(); frame.setObjectName("rightpanel"); frame.setMinimumWidth(340)
        lay = QtWidgets.QVBoxLayout(frame); lay.setContentsMargins(0, 0, 0, 0)
        tabs = QtWidgets.QTabWidget(); tabs.setDocumentMode(True); lay.addWidget(tabs)
        # slice
        sw = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(sw); sl.setContentsMargins(14, 10, 14, 14); sl.setSpacing(8)
        self.s_axis = QtWidgets.QComboBox(); self.s_axis.addItems(["x-normal", "y-normal", "z-normal"])
        self.s_field = QtWidgets.QComboBox(); self.s_field.addItems(case_mod.SLICE_FIELDS)
        sl.addWidget(Field("Plane and quantity", self.s_axis, self.s_field))
        self.s_pos = QtWidgets.QSlider(Qt.Horizontal); self.s_pos.setRange(0, 1000); self.s_pos.setValue(500)
        sl.addWidget(Field("Position", self.s_pos))
        self.s_lock = QtWidgets.QCheckBox("Lock color range"); sl.addWidget(self.s_lock)
        self.slice = SliceView(); sl.addWidget(self.slice, 1)
        self.s_axis.currentIndexChanged.connect(self.send_slice); self.s_field.currentIndexChanged.connect(self.send_slice); self.s_pos.valueChanged.connect(self.send_slice)
        self.s_lock.toggled.connect(lambda v: setattr(self.slice, "lock", v))
        tabs.addTab(sw, "Slice")
        # runs
        rw = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(rw); rl.setContentsMargins(14, 10, 14, 14); rl.setSpacing(8)
        self.runs = QtWidgets.QListWidget(); self.runs.itemDoubleClicked.connect(lambda *_: self.open_run()); rl.addWidget(self.runs, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._btn("Open", "eye", self.open_run)); row.addWidget(self._btn("Resume", "play", self.resume_run, tip="Continue from the latest checkpoint"))
        row.addWidget(self._btn("", "folder", self.open_folder, tip="Show files")); row.addWidget(self._btn("", "refresh", self.refresh_runs, tip="Refresh"))
        rl.addLayout(row)
        b = self._btn("Use as ground truth · AI experiment", None, self.ai_from_run, "aimode", "Create an AI experiment from the selected run (or the current setup)")
        b.setIcon(icon("neural", "#c4b5fd")); rl.addWidget(b)
        tabs.addTab(rw, "Runs")
        self.refresh_runs()
        return frame

    # ------------------------------------------------------------ setup forms
    def _spin(self, key, lo, hi, step=1, decimals=0):
        s = QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step)
        if decimals:
            s.setDecimals(decimals)
        s.valueChanged.connect(self.form_changed)
        self.f[key] = s
        return s

    def _combo(self, key, items):
        c = QtWidgets.QComboBox(); c.addItems(items); c.currentIndexChanged.connect(self.form_changed); self.f[key] = c
        return c

    def _check(self, key, text):
        c = QtWidgets.QCheckBox(text); c.toggled.connect(self.form_changed); self.f[key] = c
        return c

    def _ospin(self, key, lo, hi, step, dec):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(dec); s.valueChanged.connect(self.object_changed); self.o[key] = s
        return s

    def _tab_scene(self, l):
        l.addWidget(section("Domain"))
        l.addWidget(Field("Cells  x · y · z", self._spin("Nx", 1, 4096, 8), self._spin("Ny", 1, 4096, 8), self._spin("Nz", 1, 4096, 8)))
        l.addWidget(section("Objects"))
        self.obj_list = QtWidgets.QListWidget(); self.obj_list.setFixedHeight(84); self.obj_list.currentRowChanged.connect(self.select_object)
        l.addWidget(self.obj_list)
        self.library = QtWidgets.QComboBox(); self.refresh_library()
        l.addWidget(Field("Model library", self.library))
        row = QtWidgets.QHBoxLayout(); row.setSpacing(6)
        row.addWidget(self._btn("Add", "plus", self.add_object)); row.addWidget(self._btn("Import", "import", self.import_model, tip="STL, OBJ or PLY"))
        row.addWidget(self._btn("Remove", "trash", self.remove_object)); l.addLayout(row)
        l.addWidget(Field("Size (cells, longest side)", self._ospin("size", 1, 4096, 4, 1)))
        l.addWidget(Field("Position (fraction of domain)  x · y · z", *[self._ospin(k, 0, 1, 0.01, 3) for k in ("px", "py", "pz")]))
        l.addWidget(Field("Rotation (degrees)  x · y · z", *[self._ospin(k, -360, 360, 5, 1) for k in ("rx", "ry", "rz")]))
        self.o["role"] = QtWidgets.QComboBox(); self.o["role"].addItems(["solid", "fluid", "heat"]); self.o["role"].currentIndexChanged.connect(self.object_changed)
        l.addWidget(Field("Role", self.o["role"], hint="solid: obstacle · fluid: liquid body (free surface) · heat: heated solid (thermal)"))
        l.addWidget(Field("Spin tip speed (0 = static)", self._ospin("tip", 0, 0.3, 0.01, 3), hint="Rotating geometry, e.g. fans and propellers (lattice units, < 0.2)"))
        self.o["ground"] = QtWidgets.QCheckBox("Rest on the floor"); self.o["ground"].toggled.connect(self.object_changed); l.addWidget(self.o["ground"])

    def _tab_physics(self, l):
        l.addWidget(section("Flow"))
        l.addWidget(Field("Direction · velocity (lattice)", self._combo("direction", list(case_mod.DIRS)), self._spin("u", 0.005, 0.3, 0.005, 3)))
        l.addWidget(Field("Reynolds number · reference length (cells, 0 = object)", self._spin("re", 1, 1e8, 100, 1), self._spin("length", 0, 4096, 8, 1)))
        l.addWidget(Field("Viscosity override (0 = from Re) · initial state", self._spin("nu", 0, 1, 0.001, 5), self._combo("init", ["freestream", "rest", "taylor_green"])))
        l.addWidget(section("Boundaries"))
        for a in ("x", "y", "z"):
            l.addWidget(Field(f"{a} min · {a} max", self._combo(f"face_{a}0", list(case_mod.FACE_TYPES)), self._combo(f"face_{a}1", list(case_mod.FACE_TYPES))))
        l.addWidget(Field("Moving-wall velocity x · y · z", *[self._spin(k, -0.3, 0.3, 0.01, 3) for k in ("wx", "wy", "wz")]))
        l.addWidget(section("Models"))
        l.addWidget(self._check("les", "Turbulence model (LES, Smagorinsky)"))
        l.addWidget(self._check("forces", "Forces on objects (drag, lift)"))
        l.addWidget(Field("Gravity / body force x · y · z", *[self._spin(k, -0.01, 0.01, 0.0001, 5) for k in ("gx", "gy", "gz")]))
        l.addWidget(section("Free surface"))
        e = QtWidgets.QLineEdit(); e.setPlaceholderText("x0,y0,z0,x1,y1,z1; …"); e.editingFinished.connect(self.form_changed); self.f["fills"] = e
        l.addWidget(Field("Liquid boxes (fractions)", e))
        l.addWidget(Field("Surface tension", self._spin("sigma", 0, 0.1, 0.0001, 5)))
        l.addWidget(section("Heat transfer"))
        l.addWidget(Field("Diffusion α · expansion β", self._spin("alpha", 0, 10, 0.05, 3), self._spin("beta", 0, 10, 0.05, 3)))
        l.addWidget(Field("Hot · cold temperature", self._spin("Th", 0, 10, 0.05, 3), self._spin("Tc", 0, 10, 0.05, 3)))
        l.addWidget(section("Particles"))
        l.addWidget(Field("Tracer particles", self._spin("particles", 0, 10_000_000, 1000)))

    def _tab_solver(self, l):
        l.addWidget(section("Numerics"))
        l.addWidget(Field("Storage precision", self._combo("precision", ["auto", "FP32", "FP16S", "FP16C"]), hint="auto = fastest measured on this machine (Hardware → benchmark)"))
        l.addWidget(Field("Velocity set · collision", self._combo("lattice", ["auto", "D3Q19", "D3Q27", "D3Q15", "D2Q9"]), self._combo("collision", ["SRT", "TRT"])))
        l.addWidget(Field("Memory layout padding · device", self._combo("pad", ["auto", "0"]), self._spin("device", -1, 64)))
        l.addWidget(section("Run"))
        l.addWidget(Field("Stop after steps (0 = never)", self._spin("steps", 0, 10 ** 9, 1000)))
        l.addWidget(Field("Telemetry every · slice every (steps)", self._spin("tel", 1, 100000, 50), self._spin("slice_every", 0, 100000, 50)))
        l.addWidget(Field("Checkpoint every (steps, 0 = off)", self._spin("ckpt", 0, 10 ** 9, 1000)))
        l.addWidget(section("Live 3D"))
        l.addWidget(Field("Frames per second · max share of time (%)", self._spin("fps", 0, 60, 1, 1), self._spin("budget", 1, 100, 5),
                          hint="Rendering is throttled so the solver keeps at least the remaining share of the GPU"))
        self.f["budget"].valueChanged.connect(lambda v: self.cmd(f"budget {v / 100}"))
        self.f["fps"].valueChanged.connect(lambda v: self.cmd(f"rates {self.f['tel'].value()} {self.f['slice_every'].value()} {v}"))
        n = QtWidgets.QLabel("Each combination of precision, velocity set and physics is compiled once (~30–60 s) and cached.")
        n.setObjectName("hint"); n.setWordWrap(True); l.addWidget(n)

    # ------------------------------------------------------------ form <-> case
    def set_form(self, c):
        self._loading = True
        f = self.f
        for k, v in zip(("Nx", "Ny", "Nz"), c["domain"]):
            f[k].setValue(int(v))
        fl = c["flow"]
        f["direction"].setCurrentText(fl["direction"]); f["u"].setValue(fl["u"]); f["re"].setValue(fl["re"]); f["length"].setValue(fl.get("length") or 0)
        f["nu"].setValue(c["nu"] or 0); f["init"].setCurrentText(c["init"])
        for face in case_mod.FACES:
            f["face_" + face].setCurrentText(c["boundaries"][face])
        for k, v in zip(("wx", "wy", "wz"), c["wall_velocity"]):
            f[k].setValue(v)
        for k, v in zip(("gx", "gy", "gz"), c["gravity"]):
            f[k].setValue(v)
        f["les"].setChecked(bool(c["les"])); f["forces"].setChecked(bool(c["forces"]))
        f["fills"].setText("; ".join(",".join(f"{x:g}" for x in b) for b in c["fills"]))
        f["sigma"].setValue(c["surface_tension"])
        th = c["thermal"] or {}
        for k, key, d in (("alpha", "alpha", 0), ("beta", "beta", 0), ("Th", "T_hot", 1.5), ("Tc", "T_cold", 0.5)):
            f[k].setValue(th.get(key, d))
        f["particles"].setValue((c["particles"] or {}).get("count", 0))
        s, r = c["solver"], c["run"]
        f["precision"].setCurrentText(s["precision"]); f["lattice"].setCurrentText(s["lattice"]); f["collision"].setCurrentText(s["collision"])
        f["pad"].setCurrentText(str(s["ddf_pad"])); f["device"].setValue(int(s["device"]))
        f["steps"].setValue(int(r["steps"])); f["tel"].setValue(int(r["telemetry_every"])); f["slice_every"].setValue(int(r["slice_every"]))
        f["fps"].setValue(float(r["frame_fps"])); f["budget"].setValue(int(100 * r.get("render_budget", 0.2))); f["ckpt"].setValue(int(r["checkpoint_every"]))
        v = c["view"]
        for key, b in self.vis.items():
            b.blockSignals(True); b.setChecked(key in v["modes"]); b.blockSignals(False)
        for w, val in ((self.vis_field, v["field"]), (self.s_axis, v["slice_axis"]), (self.s_field, v["slice_field"])):
            w.blockSignals(True); w.setCurrentIndex(int(val)); w.blockSignals(False)
        self.view.cam = list(v["camera"])
        cl = v.get("cloud", {})
        for w_ in (self.cloud_btn, self.cloud_mode, self.cloud_amt):
            w_.blockSignals(True)
        self.cloud_btn.setChecked(bool(cl.get("on")))
        self.cloud_mode.setCurrentIndex([f_ for _, f_ in CLOUD_MODES].index(cl.get("field", 4)))
        self.cloud_amt.setValue(int(round(100 * np.log(max(cl.get("density", 0.15), 0.02) / 0.02) / np.log(100))))
        for w_ in (self.cloud_btn, self.cloud_mode, self.cloud_amt):
            w_.blockSignals(False)
        self.obj_list.clear()
        for o in c["objects"]:
            self.obj_list.addItem(self._obj_label(o))
        self._loading = False
        self._obj_enable(bool(c["objects"]))
        if c["objects"]:
            self.obj_list.setCurrentRow(0)
        self.form_changed()

    def get_form(self):
        f, c = self.f, copy.deepcopy(self.case)
        c["domain"] = [f[k].value() for k in ("Nx", "Ny", "Nz")]
        c["flow"] = {"direction": f["direction"].currentText(), "u": f["u"].value(), "re": f["re"].value(), "length": f["length"].value() or None}
        c["nu"] = f["nu"].value() or None
        c["init"] = f["init"].currentText()
        c["boundaries"] = {face: f["face_" + face].currentText() for face in case_mod.FACES}
        c["wall_velocity"] = [f[k].value() for k in ("wx", "wy", "wz")]
        c["gravity"] = [f[k].value() for k in ("gx", "gy", "gz")]
        c["les"] = f["les"].isChecked(); c["forces"] = f["forces"].isChecked()
        boxes = [b for b in f["fills"].text().split(";") if b.strip()]
        c["fills"] = [[float(x) for x in b.split(",")] for b in boxes if len(b.split(",")) == 6]
        c["surface_tension"] = f["sigma"].value()
        alpha, beta = f["alpha"].value(), f["beta"].value()
        c["thermal"] = {"alpha": alpha, "beta": beta, "T_hot": f["Th"].value(), "T_cold": f["Tc"].value()} if (alpha or beta) else None
        n = f["particles"].value(); c["particles"] = {"count": n, "rho": 1.0} if n else None
        pad = f["pad"].currentText()
        c["solver"] = {"precision": f["precision"].currentText(), "lattice": f["lattice"].currentText(), "collision": f["collision"].currentText(),
                       "ddf_pad": pad if pad == "auto" else int(pad), "device": f["device"].value()}
        c["run"] = {"steps": f["steps"].value(), "telemetry_every": f["tel"].value(), "slice_every": f["slice_every"].value(),
                    "frame_fps": f["fps"].value(), "render_budget": f["budget"].value() / 100, "checkpoint_every": f["ckpt"].value()}
        c["view"] = {**self.case.get("view", {}), "modes": [k for k, b in self.vis.items() if b.isChecked()], "field": self.vis_field.currentIndex(),  # keeps settings without a control (e.g. resolution)
                     "slice_axis": self.s_axis.currentIndex(), "slice_field": self.s_field.currentIndex(), "camera": list(self.view.cam),
                     "cloud": {"on": self.cloud_btn.isChecked(), "field": CLOUD_MODES[self.cloud_mode.currentIndex()][1], "gain": 1.0, "density": self._cloud_density()}}
        return c

    def form_changed(self, *_):
        if getattr(self, "_loading", False):
            return
        self.case = self.get_form()
        try:
            v = case_mod.solver_variant(self.case)
            n = int(np.prod(self.case["domain"]))
            mem = n * case_mod.bytes_per_cell(v) / 2 ** 20
            ext = ", ".join(e.lower().replace("_", " ") for e in v["extensions"]) or "no extensions"
            self.mem_label.setText(f"<span style='color:{C['text']};font-weight:600'>{n / 1e6:.2f} M cells · {mem:.0f} MB</span><br>"
                                   f"{v['precision']} · {v['lattice']} · {v['collision']} · {ext}")
        except Exception as e:
            self.mem_label.setText(f"<span style='color:{C['bad']}'>{e}</span>")

    # ------------------------------------------------------------ objects
    def refresh_library(self):
        self.library.clear()
        for k, (name, _) in geometry.BUILTIN.items():
            self.library.addItem(name, k)
        for p in sorted(MODELS.glob("*.stl")):
            self.library.addItem(f"Imported · {p.stem}", str(p))

    def _obj_label(self, o):
        if "shape" in o:
            return f"{o['shape'].capitalize()} (primitive)"
        m = o["model"]
        return geometry.BUILTIN[m][0] if m in geometry.BUILTIN else Path(m).stem

    def add_object(self):
        self.case = self.get_form()
        N = self.case["domain"]
        self.case["objects"].append({"model": self.library.currentData(), "size": round(0.3 * min(N)), "position": [0.5, 0.35, 0.5], "rotation": [0, 0, 0]})
        self.obj_list.addItem(self._obj_label(self.case["objects"][-1]))
        self.obj_list.setCurrentRow(len(self.case["objects"]) - 1)
        self._obj_enable(True)
        self.form_changed()

    def import_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import model", "", "Meshes (*.stl *.obj *.ply)")
        if not path:
            return
        try:
            tris = geometry.load(path)
            dst = MODELS / (Path(path).stem + ".stl")
            geometry.save_stl(tris, dst)  # normalized binary STL
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(e)); return
        self.refresh_library()
        self.library.setCurrentIndex(self.library.findData(str(dst)))
        self.statusBar().showMessage(f"Imported {len(tris):,} triangles from {path}")
        self.add_object()

    def remove_object(self):
        i = self.obj_list.currentRow()
        if i >= 0:
            self.case = self.get_form()
            del self.case["objects"][i]
            self.obj_list.takeItem(i)
            self._obj_enable(bool(self.case["objects"]))
            self.form_changed()

    def _obj_enable(self, on):
        for w in self.o.values():
            w.setEnabled(on)

    def select_object(self, i):
        if i < 0 or i >= len(self.case["objects"]):
            return
        o = self.case["objects"][i]
        self._loading_obj = True
        self.o["size"].setValue(2 * o["radius"] if "radius" in o else (max(o["size"]) if isinstance(o.get("size"), list) else o.get("size", 32)))
        for k, v in zip(("px", "py", "pz"), o.get("position", [0.5] * 3)):
            self.o[k].setValue(v)
        for k, v in zip(("rx", "ry", "rz"), o.get("rotation", [0] * 3)):
            self.o[k].setValue(v)
        self.o["role"].setCurrentText(o.get("role", "solid")); self.o["tip"].setValue(o.get("tip_speed", 0)); self.o["ground"].setChecked(bool(o.get("on_ground")))
        self._loading_obj = False

    def object_changed(self, *_):
        i = self.obj_list.currentRow()
        if getattr(self, "_loading_obj", False) or i < 0 or i >= len(self.case["objects"]):
            return
        o = self.case["objects"][i]
        if "shape" in o:
            if "radius" in o:
                o["radius"] = self.o["size"].value() / 2
        else:
            o["size"] = self.o["size"].value(); o["rotation"] = [self.o[k].value() for k in ("rx", "ry", "rz")]
            o["tip_speed"] = self.o["tip"].value(); o["on_ground"] = self.o["ground"].isChecked()
        o["position"] = [self.o[k].value() for k in ("px", "py", "pz")]
        o["role"] = self.o["role"].currentText()
        self.form_changed()

    # ------------------------------------------------------------ cases
    def load_preset(self, name):
        self.case = case_mod.normalize(presets.PRESETS[name]); self.case["name"] = name
        self.set_form(self.case)

    def open_case(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open case", str(runner.WORKSPACE), "Case (*.json)")
        if path:
            self.case = case_mod.normalize(json.loads(Path(path).read_text())); self.set_form(self.case)

    def save_case(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save case", str(runner.WORKSPACE / f"{self.case['name']}.json"), "Case (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.get_form(), indent=1))

    # ------------------------------------------------------------ run control
    def _set_state(self, text, color):
        self.state_chip.setText(text)
        self.state_chip.setStyleSheet(chip_style(color))
        self.view.status = (text, color); self.view.update()

    def _launch(self, paused, restart=None):
        if self._building:
            return
        self.stop_run()
        case = self.get_form()
        self._building = True
        self.progress.show()
        self._set_state("preparing", C["warn"])
        self.statusBar().showMessage("Preparing solver… the first use of a physics combination compiles it once")
        self.view.hint = "Preparing solver…"; self.view.image = None; self.view.hud = []; self.view.update()
        self._series = {}; self.slice.values = None

        def work():  # geometry placement and compilation can take a while: keep the UI responsive
            run = None
            try:
                run = runner.Run.create(case, restart=restart)
                run.start(log=lambda s: None, paused=paused)
                self._pending = (run, None)
            except Exception as e:
                self._pending = (run, e)
        threading.Thread(target=work, daemon=True).start()

    def preview(self):
        self._launch(paused=True)

    def start_run(self):
        if self._building:
            return
        if self.run and self.run.proc and self.run.proc.poll() is None:  # never discard a live run by pressing Run
            if self.run.status == "paused":
                self.cmd("resume")
            else:
                self.statusBar().showMessage("Already running. Stop it first to run a changed case.")
            return
        self._launch(paused=False)

    def stop_run(self):
        if self.run:
            self.run.stop()
            self.refresh_runs()

    def cmd(self, c):
        if self.run:
            self.run.send(c)

    def send_vis(self, *_):
        modes = sum(case_mod.VIS[k] for k, b in self.vis.items() if b.isChecked())
        N = self.case["domain"]; frac = self.vis_pos.value() / 1000
        self.cmd(f"vis {modes} {self.vis_field.currentIndex()} {self.vis_slice.currentIndex()} {int(frac * (N[0] - 1))} {int(frac * (N[1] - 1))} {int(frac * (N[2] - 1))}")

    def _cloud_density(self):
        return 0.02 * 100 ** (self.cloud_amt.value() / 100)  # 0.02 .. 2, log scale

    def send_cloud(self, *_):
        self.cmd(f"cloud {int(self.cloud_btn.isChecked())} {CLOUD_MODES[self.cloud_mode.currentIndex()][1]} 1 {self._cloud_density():.4f}")

    def send_slice(self, *_):
        axis = self.s_axis.currentIndex()
        pos = int(self.s_pos.value() / 1000 * (self.case["domain"][axis] - 1))
        self.cmd(f"slice {axis} {pos} {self.s_field.currentIndex()}")

    def save_image(self):
        if self.view.image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", str(runner.WORKSPACE / "frame.png"), "PNG (*.png)")
        if path:
            self.view.image.save(path)

    # ------------------------------------------------------------ main loop
    def tick(self):
        if self.ai:
            self.ai.tick()
        if self._pending:
            run, err = self._pending; self._pending = None; self._building = False; self.progress.hide()
            if err:
                self._set_state("failed", C["bad"])
                QtWidgets.QMessageBox.warning(self, "Solver failed to start", str(err)); return
            self.run = run; self._run_case = self.get_form()
            self.statusBar().showMessage(f"Run {run.id}")
            self.refresh_runs()
        if self._cam_dirty and self.run:
            self._cam_dirty = False
            self.cmd("cam " + " ".join(f"{x:.3f}" for x in self.view.cam))
        if not self.run:
            return
        new = self.run.poll()
        st = self.run.status
        color = {"running": C["ok"], "paused": C["warn"], "finished": C["accent"], "starting": C["warn"], "failed": C["bad"]}.get(st, C["muted"])
        if self.state_chip.text() != st:
            self._set_state(st, color)
        if not new:
            if st == "failed":
                self._fail()
            return
        frame = slice_ = None
        for r in new:
            t = r["type"]
            if t == "frame":
                frame = r
            elif t == "slice":
                slice_ = r
            elif t == "step":
                for k, v in r.items():
                    if k != "type":
                        self._series.setdefault(k, []).append(v)
            elif t == "start":
                self._start = r
            elif t in ("checkpoint", "export", "restart"):
                self.statusBar().showMessage(f"{t.capitalize()}: {r.get('file') or ', '.join(r.get('files', []))}")
            elif t == "error":
                self.statusBar().showMessage("Error: " + r.get("message", ""))
        if frame:
            try:
                data = (self.run.dir / "live" / frame["file"]).read_bytes()
                if len(data) == 4 * frame["w"] * frame["h"]:
                    self.view.set_frame(data, frame["w"], frame["h"])
            except OSError:
                pass
        if slice_:
            try:
                vals = np.fromfile(self.run.dir / "live" / slice_["file"], np.float32).reshape(slice_["h"], slice_["w"])
                slice_["signed"] = slice_["field"] in (1, 2, 3, 8)
                self.slice.set_slice(vals, slice_, slice_["signed"])
            except (OSError, ValueError):
                pass
        self.update_monitor()
        if st == "failed":
            self._fail()

    def _fail(self):
        self.statusBar().showMessage(f"Solver failed: {self.run.error}")
        self.view.hint = f"Solver failed: {self.run.error}"; self.view.image = None; self.view.update()
        self.refresh_runs(); self.run = None

    def update_monitor(self):
        s, d = self._series, self.run.meta["derived"]
        t = s.get("t", [])
        if not t:
            return
        dsign = np.sign(sum(case_mod.DIRS[self.run.case["flow"]["direction"]]))
        self.cards["perf"].set({"solver": (t, s["mlups"]), "wall": (t, np.asarray(s["steps_per_s"]) * d["cells"] / 1e6)})
        self.cards["u"].set({"max": (t, s["u_max"])})
        self.cards["ke"].set({"KE": (t, s["ke"])})
        self.cards["mass"].set({"Δρ": (t, np.asarray(s["rho_mean"]) - 1)})
        if "force" in s and d.get("ref_area"):
            F = np.asarray(s["force"]); q = 0.5 * d["u"] ** 2 * d["ref_area"]
            self.cards["force"].set({"Cd": (t, F[:, d["flow_axis"]] / q * dsign), "Cl": (t, F[:, d["lift_axis"]] / q)})
        if "T_mean" in s:
            self.cards["extra"].set({"T": (t, s["T_mean"])})
        elif "liquid_volume" in s:
            self.cards["extra"].set({"V": (t, s["liquid_volume"])})
        last = {k: v[-1] for k, v in s.items()}
        si = d.get("si")
        self.view.hud = [("t", f"{int(last['t']):,}" + (f" ({last['t'] * si['dt']:.4g} s)" if si else "")), ("speed", f"{last['steps_per_s']:.1f} steps/s"),
                         ("solver", f"{last['mlups']:.0f} MLUP/s"), ("grid", f"{d['cells'] / 1e6:.2f} M"), ("Re", f"{d['re']:.4g}")]
        self.view.update()
        self.insight.setText(self.explain(last, d))

    def explain(self, last, d):
        """Performance and physics readout with a bottleneck verdict."""
        start = self._start
        bpc = start.get("bandwidth_bytes_per_cell", 77)
        mlups, sps = last["mlups"], last["steps_per_s"]
        bw = mlups * bpc / 1e3
        wall = sps * d["cells"] / 1e6
        overhead = max(0.0, 1 - wall / mlups) if mlups else 0
        vis = last.get("render_frac", 0) + last.get("slice_frac", 0)
        probe = self._probe.get("copy_gbs")
        if vis > 0.3:
            verdict, col = "Live visualization limits speed — lower FPS or time share", C["warn"]
        elif overhead - vis > 0.2:
            verdict, col = "Host/launch overhead — grid is small for this GPU", C["warn"]
        else:
            verdict, col = "Memory-bandwidth bound (normal for LBM)" + (" · near the device limit" if probe and bw / probe > 0.75 else ""), C["ok"]
        mu = C["muted"]
        rows = [f"<div style='font-weight:600;font-size:10pt'>{start.get('device', '')}</div>",
                f"<div style='color:{mu}'>{d['variant']['precision']} · {d['variant']['lattice']} · {start.get('memory_mb', 0)} MB</div><br>",
                f"<span style='color:{col};font-weight:600'>● {verdict}</span><br><br>",
                f"<span style='color:{mu}'>Kernel</span> {mlups:.0f} MLUP/s · {bw:.1f} GB/s" + (f" ({100 * bw / probe:.0f}% of {probe:.1f} GB/s)" if probe else ""),
                f"<br><span style='color:{mu}'>Wall</span> {wall:.0f} MLUP/s · 3D {100 * vis:.0f}% · other {100 * max(0, overhead - vis):.0f}%",
                f"<br><span style='color:{mu}'>Flow</span> Re {d['re']:.4g} · τ {d['tau']:.4f} · u<sub>max</sub> {last['u_max']:.3f}"
                + (f" <span style='color:{C['bad']}'>(compressibility risk)</span>" if last["u_max"] > 0.17 else "")]
        if "force" in last and d.get("ref_area"):
            F = np.asarray(last["force"]); q = 0.5 * d["u"] ** 2 * d["ref_area"]
            rows.append(f"<br><span style='color:{mu}'>Forces</span> C<sub>d</sub> {abs(F[d['flow_axis']]) / q:.3f} · C<sub>l</sub> {F[d['lift_axis']] / q:.3f}")
        if self.run.error:
            rows.append(f"<br><span style='color:{C['bad']}'>{self.run.error}</span>")
        return "".join(rows)

    # ------------------------------------------------------------ runs
    def refresh_runs(self):
        self.runs.clear()
        for p in runner.list_runs()[:200]:
            try:
                meta = json.loads((p / "run.json").read_text())
            except ValueError:
                continue
            status = "running" if self.run and self.run.dir == p else meta.get("status", "?")
            name = p.name[16:].replace("-", " ") or p.name
            if meta.get("kind") == "ai_experiment":
                name = "AI experiment · " + name[3:]
            it = QtWidgets.QListWidgetItem(f"{name}\n{p.name[:8]} {p.name[9:11]}:{p.name[11:13]} · {status} · {meta['derived']['cells'] / 1e6:.1f} M cells")
            it.setData(Qt.UserRole, str(p)); self.runs.addItem(it)

    def _selected_run(self):
        it = self.runs.currentItem()
        return Path(it.data(Qt.UserRole)) if it else None

    def open_run(self):
        p = self._selected_run()
        if p:
            self.open_run_dir(p)

    def open_run_dir(self, p):
        from . import ai
        if ai.is_experiment(p):
            self.enter_ai(p); return
        if self.run and self.run.dir == p:
            return
        self.stop_run()
        r = runner.Run(p); r.status = "stopped"
        self.run, self._series, self._run_case = r, {}, None
        self.case = case_mod.normalize(r.case); self.set_form(self.case)
        r.load_records()
        for rec in r.records:
            if rec["type"] == "step":
                for k, v in rec.items():
                    if k != "type":
                        self._series.setdefault(k, []).append(v)
            elif rec["type"] == "start":
                self._start = rec
        frames = [x for x in r.records if x["type"] == "frame"]
        if frames:
            f = frames[-1]
            data = (p / "live" / f["file"]).read_bytes()
            if len(data) == 4 * f["w"] * f["h"]:
                self.view.set_frame(data, f["w"], f["h"])
        self._set_state("stopped", C["muted"])
        self.update_monitor()

    # ------------------------------------------------------------ AI Mode (same window, reconfigured workspace)
    def enter_ai(self, exp_dir=None):
        if self.ai is None:
            from .ai_mode import AIWorkspace
            self.ai = AIWorkspace(self); self.body.addWidget(self.ai)
        self.sim_bar.hide(); self.ai_btn.hide(); self.ai_bar.show(); self.sim_btn.show()
        self.brand_sub.setText("AI Mode · Neural Surrogate Laboratory"); self.brand_sub.setStyleSheet("color: #c4b5fd;")
        self.topbar.setStyleSheet("QFrame#topbar { border-bottom: 1px solid #6d4fc2; }")
        self._fade(self.ai)
        self.ai.refresh_list(exp_dir)
        if exp_dir:
            self.ai.open(Path(exp_dir))
        elif self.ai.exp is None and self.ai.list.count():
            self.ai.list.setCurrentRow(0)
        self.nav_ai(self.ai.pages.currentIndex() if not exp_dir else 0)
        self.statusBar().showMessage("AI Mode · Neural Surrogate Laboratory · a running simulation keeps running in Simulation Mode")

    def leave_ai(self):
        self.ai_bar.hide(); self.sim_btn.hide(); self.sim_bar.show(); self.ai_btn.show()
        self.brand_sub.setText("CFD workbench · FluidX3D core"); self.brand_sub.setStyleSheet(""); self.topbar.setStyleSheet("")
        self._fade(self.body.widget(0))
        self.statusBar().showMessage("Simulation Mode")

    def nav_ai(self, i):
        self.ai_nav.button(i).setChecked(True)
        self.ai.nav(i)

    def _fade(self, page):
        self.body.setCurrentWidget(page)
        eff = QtWidgets.QGraphicsOpacityEffect(page); page.setGraphicsEffect(eff)
        a = QtCore.QPropertyAnimation(eff, b"opacity", page); a.setDuration(180); a.setStartValue(0.0); a.setEndValue(1.0)
        a.finished.connect(lambda: page.setGraphicsEffect(None)); a.start()

    def ai_from_run(self):
        self.enter_ai()
        p = self._selected_run()
        from . import ai
        self.ai.new_experiment(p if p and not ai.is_experiment(p) else None)

    def resume_run(self):
        p = self._selected_run()
        if not p:
            return
        ck = runner.Run(p).checkpoints()
        if not ck:
            QtWidgets.QMessageBox.information(self, "No checkpoint", "This run has no checkpoints. Use Checkpoint while it runs, or set an interval in Solver."); return
        self.case = case_mod.normalize(json.loads((p / "case.json").read_text())); self.case.pop("restart", None); self.set_form(self.case)
        self._launch(paused=False, restart=ck[-1])

    def open_folder(self):
        p = self._selected_run() or runner.RUNS
        if os.name == "nt":
            os.startfile(str(p))
        else:
            subprocess.Popen(["xdg-open", str(p)])

    # ------------------------------------------------------------ hardware
    def _load_probe(self):
        p = runner.WORKSPACE / "benchmarks.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def hardware(self):
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Hardware"); dlg.resize(780, 460)
        lay = QtWidgets.QVBoxLayout(dlg); lay.setContentsMargins(16, 16, 16, 16)
        text = QtWidgets.QTextEdit(); text.setReadOnly(True); lay.addWidget(text)
        b = self._btn("Run benchmark  ·  memory bandwidth and all precision modes (~3 min)", "chip", obj="primary"); lay.addWidget(b)

        def show():
            try:
                devs = runner.devices()
            except Exception as e:
                devs = [{"name": f"Device query failed: {e}"}]
            html = "<h3>OpenCL devices</h3>" + "".join(
                f"<p><b>[{d.get('id')}] {d['name']}</b> — {d.get('vendor', '')}<br><span style='color:{C['muted']}'>{d.get('compute_units')} CUs @ {d.get('clock_mhz')} MHz · "
                f"{d.get('tflops')} TFLOP/s · {d.get('memory_mb')} MB {'shared' if d.get('uses_ram') else 'VRAM'} · FP16 {'yes' if d.get('fp16') else 'no'} · "
                f"OpenCL C {d.get('opencl_c')} · driver {d.get('driver')}</span></p>" for d in devs)
            bench = self._load_probe()
            if bench:
                html += (f"<h3>Benchmark · {bench.get('date', '')}</h3><p>Memory bandwidth (copy): <b>{bench.get('copy_gbs', 0):.1f} GB/s</b></p>"
                         + "".join(f"<p>{k}: {v:.0f} MLUP/s</p>" for k, v in bench.get("mlups", {}).items())
                         + f"<p>Recommended precision: <b>{bench.get('recommended')}</b> (used when precision is 'auto')</p>")
            text.setHtml(html)
        show()

        def bench():
            b.setEnabled(False); b.setText("Benchmarking… (compiles missing precision variants first)")
            from . import bench as bench_mod
            threading.Thread(target=lambda: (bench_mod.run(), QtCore.QMetaObject.invokeMethod(dlg, "accept", Qt.QueuedConnection)), daemon=True).start()
        b.clicked.connect(bench)
        dlg.exec_()
        self._probe = self._load_probe()
        self.form_changed()


def style_app(app):
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    for role, c in ((QtGui.QPalette.Window, C["bg"]), (QtGui.QPalette.WindowText, C["text"]), (QtGui.QPalette.Base, C["card"]),
                    (QtGui.QPalette.AlternateBase, C["card2"]), (QtGui.QPalette.Text, C["text"]), (QtGui.QPalette.Button, C["card2"]),
                    (QtGui.QPalette.ButtonText, C["text"]), (QtGui.QPalette.Highlight, C["accent"]), (QtGui.QPalette.HighlightedText, "#ffffff")):
        p.setColor(role, QtGui.QColor(c))
    app.setPalette(p)
    app.setStyleSheet(STYLE)


dark_palette = style_app  # used by tests


def dark_title_bar(widget):
    """Windows 10/11: dark native title bar."""
    if os.name != "nt":
        return
    try:
        import ctypes
        value = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(int(widget.winId()), attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass


def main(case_path=None, autorun=False):
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("NeuroSim")
    style_app(app)
    w = Main()
    if case_path:
        w.case = case_mod.normalize(json.loads(Path(case_path).read_text())); w.set_form(w.case)
        if w.preset.findText(w.case.get("name", "")) < 0:
            w.preset.addItem(w.case["name"])
        w.preset.setCurrentText(w.case["name"])
    dark_title_bar(w)
    w.showMaximized()
    if autorun:
        QtCore.QTimer.singleShot(500, w.start_run)
    code = app.exec_()
    if w.run:
        w.run.stop()
    if w.ai:
        w.ai.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
