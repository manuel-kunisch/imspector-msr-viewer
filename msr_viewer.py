#!/usr/bin/env python3
"""
msr_viewer.py -- drag & drop viewer for LaVision BioTec ImSpector (.msr) files.

    python msr_viewer.py [FILE.msr | FOLDER ...]

Drop .msr files or folders onto the window (or onto MSR_Viewer.bat in Explorer).
Parsing and TIFF export are done by msr_reader.py, which must sit next to this file.

Mouse: wheel = zoom, drag = pan, double-click = fit.
Keys:  Left/Right = previous/next frame, Home/End = first/last frame, Space = play
       (while the image has focus); F = fit, 1 = 100 %, B = scale bar.
"""

from __future__ import annotations

import math
import os
import re
import sys
import traceback
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msr_reader as mr  # noqa: E402

APP_NAME = "MSR Viewer"
__version__ = "0.2.0"


# -----------------------------------------------------------------------------
# lookup tables
# -----------------------------------------------------------------------------

def _ramp_lut(r: int, g: int, b: int) -> np.ndarray:
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.stack([x * r, x * g, x * b], 1).round().astype(np.uint8)


def _anchor_lut(anchors) -> np.ndarray:
    pos = np.array([a[0] for a in anchors], dtype=np.float64)
    rgb = np.array([a[1:] for a in anchors], dtype=np.float64)
    x = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(x, pos, rgb[:, i]) for i in range(3)], 1).round().astype(np.uint8)


LUTS = {
    "Gray": _ramp_lut(255, 255, 255),
    "Green": _ramp_lut(0, 255, 0),
    "Magenta": _ramp_lut(255, 0, 255),
    "Red": _ramp_lut(255, 0, 0),
    "Cyan": _ramp_lut(0, 255, 255),
    "Yellow": _ramp_lut(255, 255, 0),
    "Fire": _anchor_lut([(0, 0, 0, 0), (0.15, 30, 0, 130), (0.3, 100, 0, 220), (0.45, 175, 0, 150),
                         (0.6, 230, 50, 30), (0.75, 255, 140, 0), (0.9, 255, 230, 60), (1, 255, 255, 255)]),
    "Viridis": _anchor_lut([(0, 68, 1, 84), (0.125, 71, 44, 122), (0.25, 59, 81, 139),
                            (0.375, 44, 113, 142), (0.5, 33, 144, 141), (0.625, 39, 173, 129),
                            (0.75, 92, 200, 99), (0.875, 170, 220, 50), (1, 253, 231, 37)]),
    "Inverted": _ramp_lut(255, 255, 255)[::-1].copy(),
}
_hilo = LUTS["Gray"].copy()
_hilo[0] = (0, 0, 255)      # at or below min: blue
_hilo[255] = (255, 0, 0)    # at or above max: red
LUTS["HiLo"] = _hilo
COLOR_TABLES = {name: [0xFF000000 | (int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in lut]
                for name, lut in LUTS.items()}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _specified(v) -> bool:
    return v not in (None, "", "not specified", "not_specified")


def _nice_length(x: float) -> float:
    if not x or x <= 0 or not math.isfinite(x):
        return 0.0
    e = math.floor(math.log10(x))
    candidates = [c * 10.0 ** k for k in (e - 1, e, e + 1) for c in (1, 2, 5)]
    return min(candidates, key=lambda c: abs(math.log(c / x)))


def _fmt_length(v: float, unit: str) -> str:
    if unit in ("µm", "um"):
        if v >= 1000:
            return f"{v / 1000:g} mm"
        if v < 1:
            return f"{v * 1000:g} nm"
        return f"{v:g} µm"
    return f"{v:g} {unit}"


def paint_scalebar(p: QtGui.QPainter, width: float, height: float, per_px: float, unit: str,
                   font_px: int = 12, bar_h: float = 5, margin: float = 18, target: float = 0.18) -> None:
    """Scale bar in the bottom-left corner of a width x height pixel area."""
    if not per_px or per_px <= 0:
        return
    length = _nice_length(target * width * per_px)
    px = length / per_px
    if px < 4:
        return
    font = QtGui.QFont(p.font())
    font.setPixelSize(font_px)
    font.setBold(True)
    fm = QtGui.QFontMetrics(font)
    text = _fmt_length(length, unit)
    tw, th = fm.horizontalAdvance(text), fm.height()
    y_bar = height - margin - bar_h
    box = QtCore.QRectF(margin - 6, y_bar - th - 8, max(px, tw) + 12, th + bar_h + 14)
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor(0, 0, 0, 140))
    p.drawRoundedRect(box, 4, 4)
    p.setBrush(QtGui.QColor(255, 255, 255))
    p.drawRect(QtCore.QRectF(margin, y_bar, px, bar_h))
    p.setPen(QtGui.QColor(255, 255, 255))
    p.setFont(font)
    p.drawText(QtCore.QRectF(margin, y_bar - th - 4, max(px, tw) + 2, th), Qt.AlignLeft | Qt.AlignVCenter, text)
    p.restore()


def _numpy_to_qimage(a8: np.ndarray, colors) -> QtGui.QImage:
    """8-bit array -> indexed QImage (copied, so the array may be freed)."""
    h, w = a8.shape
    if w % 4:
        padded = np.zeros((h, (w + 3) // 4 * 4), np.uint8)
        padded[:, :w] = a8
        a8 = padded
    a8 = np.ascontiguousarray(a8)
    img = QtGui.QImage(a8.data, w, h, a8.strides[0], QtGui.QImage.Format_Indexed8)
    img.setColorTable(colors)
    return img.copy()


def _histogram(frame: np.ndarray, x0: float, x1: float, bins: int = 256):
    """Histogram over [x0, x1]; None if integer data exceed x1 (caller widens)."""
    if frame.dtype in (np.uint8, np.uint16) and x0 == 0 and (int(x1) + 1) % bins == 0:
        c = np.bincount(frame.ravel(), minlength=int(x1) + 1)
        if c.size > int(x1) + 1:
            return None
        return c.reshape(bins, -1).sum(1)
    c, _ = np.histogram(frame, bins=bins, range=(x0, x1))
    return c


def _percentiles(values: np.ndarray, lo_pct: float = 0.1, hi_pct: float = 99.9) -> tuple[float, float]:
    lo, hi = (float(v) for v in np.percentile(values, [lo_pct, hi_pct]))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


@dataclass
class DisplayState:
    lo: float
    hi: float
    lut: str
    x0: float                         # histogram range
    x1: float
    auto: tuple
    imspector: tuple | None


def _initial_state(arr: np.ndarray, stack: mr.DataStack) -> DisplayState:
    """Contrast from up to 7 frames spread over the stack (sub-sampled)."""
    lead = arr.shape[:-2]
    n = int(np.prod(lead)) if lead else 1
    picks = np.unique(np.linspace(0, n - 1, min(n, 7)).round().astype(int))
    h, w = arr.shape[-2:]
    step = max(1, int(math.sqrt(h * w / 250_000)))
    parts = []
    for k in picks:
        idx = tuple(int(i) for i in np.unravel_index(int(k), lead)) if lead else ()
        parts.append(np.asarray(arr[idx][::step, ::step]).ravel())
    sample = np.concatenate(parts)
    lo, hi = _percentiles(sample)
    vmin, vmax = float(sample.min()), float(sample.max())
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        bits = max(8, int(math.ceil(math.log2(vmax + 1)))) if vmax > 0 else 8
        x0, x1 = 0.0, float(2 ** bits - 1)
    else:
        x0, x1 = vmin, (vmax if vmax > vmin else vmin + 1.0)
    ims = None
    if stack.view is not None and stack.view.lut and stack.view.lut[1] > stack.view.lut[0]:
        ims = tuple(float(v) for v in stack.view.lut)
    return DisplayState(lo, hi, "Gray", x0, x1, (lo, hi), ims)


def _thumbnail(stack: mr.DataStack, size: int = 200) -> QtGui.QPixmap:
    arr = stack.asarray()
    lead = arr.shape[:-2]
    frame = np.asarray(arr[(0,) * len(lead)])
    h, w = frame.shape
    step = max(1, int(math.ceil(max(h, w) / (2 * size))))
    f = frame[::step, ::step].astype(np.float32)
    lo, hi = _percentiles(f)
    a8 = np.clip((f - lo) * (255.0 / (hi - lo)) + 0.5, 0, 255).astype(np.uint8)
    pix = QtGui.QPixmap.fromImage(_numpy_to_qimage(a8, COLOR_TABLES["Gray"]))
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _stack_label(pos: int, s: mr.DataStack) -> str:
    return f"S{pos + 1}  {s.channel_id.split(':')[0] or s.source}"


def _shape_text(s: mr.DataStack) -> str:
    return " × ".join(str(n) for n in s.shape)


def _app_icon() -> QtGui.QIcon:
    pix = QtGui.QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setBrush(QtGui.QColor(32, 36, 44))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QtCore.QRectF(2, 2, 60, 60), 12, 12)
    grad = QtGui.QRadialGradient(32, 28, 22)
    grad.setColorAt(0, QtGui.QColor(160, 255, 160))
    grad.setColorAt(1, QtGui.QColor(20, 140, 60))
    p.setBrush(grad)
    p.drawEllipse(QtCore.QPointF(32, 27), 17, 17)
    font = QtGui.QFont()
    font.setPixelSize(15)
    font.setBold(True)
    p.setFont(font)
    p.setPen(QtGui.QColor(235, 235, 235))
    p.drawText(QtCore.QRectF(0, 42, 64, 20), Qt.AlignCenter, "MSR")
    p.end()
    return QtGui.QIcon(pix)


# -----------------------------------------------------------------------------
# widgets
# -----------------------------------------------------------------------------

class ElidedLabel(QtWidgets.QLabel):
    """Single-line label that shortens its text instead of widening the window."""

    def __init__(self, text: str = "", mode=Qt.ElideRight, parent=None):
        super().__init__(text, parent)
        self._mode = mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        r = self.contentsRect()
        p.drawText(r, int(Qt.AlignLeft | Qt.AlignVCenter), self.fontMetrics().elidedText(self.text(), self._mode, r.width()))


class ImageView(QtWidgets.QGraphicsView):
    """Zoomable image with a scale bar overlay."""

    mouseMoved = QtCore.pyqtSignal(float, float)
    mouseLeft = QtCore.pyqtSignal()
    zoomChanged = QtCore.pyqtSignal(float)
    stepRequested = QtCore.pyqtSignal(int)
    jumpRequested = QtCore.pyqtSignal(int)
    playRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self._has_image = False
        self._fit_mode = True
        self.pixel_size: float | None = None
        self.unit = "µm"
        self.show_scalebar = True
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QtGui.QColor(16, 16, 16))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def set_pixmap(self, pix: QtGui.QPixmap, reset: bool) -> None:
        self._item.setPixmap(pix)
        if reset or not self._has_image:
            self._scene.setSceneRect(QtCore.QRectF(0, 0, pix.width(), pix.height()))
            self._has_image = True
            self.fit()
        else:
            self.viewport().update()

    def clear(self) -> None:
        self._item.setPixmap(QtGui.QPixmap())
        self._has_image = False

    def zoom(self) -> float:
        """Screen pixels per image pixel."""
        return self.transform().m11() * self.devicePixelRatioF()

    def fit(self) -> None:
        if not self._has_image:
            return
        self._fit_mode = True
        self.fitInView(self._item, Qt.KeepAspectRatio)
        self._zoomed()

    def zoom_1to1(self) -> None:
        if not self._has_image:
            return
        self._fit_mode = False
        s = 1.0 / self.devicePixelRatioF()
        self.setTransform(QtGui.QTransform.fromScale(s, s))
        self._zoomed()

    def _zoomed(self) -> None:
        z = self.zoom()
        self._item.setTransformationMode(Qt.FastTransformation if z >= 1 else Qt.SmoothTransformation)
        self.zoomChanged.emit(z)
        self.viewport().update()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if not self._has_image:
            return
        factor = 1.25 ** (ev.angleDelta().y() / 120.0)
        cur = self.transform().m11()
        new = min(max(cur * factor, 0.01), 64.0 / self.devicePixelRatioF())
        self.scale(new / cur, new / cur)
        self._fit_mode = False
        self._zoomed()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._fit_mode:
            self.fit()

    def mouseDoubleClickEvent(self, ev) -> None:
        self.fit()

    def mouseMoveEvent(self, ev) -> None:
        super().mouseMoveEvent(ev)
        if self._has_image:
            p = self.mapToScene(ev.pos())
            self.mouseMoved.emit(p.x(), p.y())

    def leaveEvent(self, ev) -> None:
        self.mouseLeft.emit()
        super().leaveEvent(ev)

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k in (Qt.Key_Left, Qt.Key_Right):
            self.stepRequested.emit(-1 if k == Qt.Key_Left else 1)
        elif k in (Qt.Key_Home, Qt.Key_End):
            self.jumpRequested.emit(0 if k == Qt.Key_Home else -1)
        elif k == Qt.Key_Space:
            self.playRequested.emit()
        else:
            super().keyPressEvent(ev)

    def drawForeground(self, painter: QtGui.QPainter, rect) -> None:
        if not (self._has_image and self.show_scalebar and self.pixel_size):
            return
        painter.save()
        painter.resetTransform()
        vp = self.viewport().rect()
        paint_scalebar(painter, vp.width(), vp.height(), self.pixel_size / self.transform().m11(), self.unit)
        painter.restore()


class HistogramWidget(QtWidgets.QWidget):
    """Log histogram with draggable min/max handles and the LUT underneath."""

    rangeEdited = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.setMouseTracking(True)
        self._counts = None
        self._x0, self._x1 = 0.0, 1.0
        self._lo, self._hi = 0.0, 1.0
        self._lut = "Gray"
        self._drag = None
        self.setToolTip("Drag the yellow lines to set the display range")

    def set_data(self, counts, x0: float, x1: float) -> None:
        self._counts = None if counts is None else np.asarray(counts, dtype=np.float64)
        self._x0, self._x1 = float(x0), float(max(x1, x0 + 1e-9))
        self.update()

    def set_range(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self.update()

    def set_lut(self, name: str) -> None:
        self._lut = name
        self.update()

    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.rect()).adjusted(8, 8, -8, -24)

    def _to_x(self, v: float, r: QtCore.QRectF) -> float:
        return r.left() + (v - self._x0) / (self._x1 - self._x0) * r.width()

    def _to_v(self, x: float, r: QtCore.QRectF) -> float:
        return self._x0 + (x - r.left()) / max(r.width(), 1) * (self._x1 - self._x0)

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(24, 24, 26))
        r = self._plot_rect()
        if self._counts is not None and self._counts.size:
            c = np.log1p(self._counts)
            top = c.max() or 1.0
            n = c.size
            path = QtGui.QPainterPath(QtCore.QPointF(r.left(), r.bottom()))
            for i, v in enumerate(c):
                y = r.bottom() - v / top * r.height()
                path.lineTo(r.left() + i / n * r.width(), y)
                path.lineTo(r.left() + (i + 1) / n * r.width(), y)
            path.lineTo(r.right(), r.bottom())
            path.closeSubpath()
            p.fillPath(path, QtGui.QColor(155, 155, 160))
        xl = min(max(self._to_x(self._lo, r), r.left()), r.right())
        xh = min(max(self._to_x(self._hi, r), r.left()), r.right())
        shade = QtGui.QColor(0, 0, 0, 120)
        p.fillRect(QtCore.QRectF(r.left(), r.top(), xl - r.left(), r.height()), shade)
        p.fillRect(QtCore.QRectF(xh, r.top(), r.right() - xh, r.height()), shade)
        # colour ramp: which colour each value is shown with
        width = max(1, int(r.width()))
        values = self._x0 + (np.arange(width) + 0.5) / width * (self._x1 - self._x0)
        t = np.clip((values - self._lo) / max(self._hi - self._lo, 1e-12), 0, 1)
        rgb = LUTS[self._lut][(t * 255 + 0.5).astype(int)].astype(np.uint32)
        line = np.ascontiguousarray((0xFF << 24) | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2])
        img = QtGui.QImage(line.data, width, 1, 4 * width, QtGui.QImage.Format_RGB32)
        p.drawImage(QtCore.QRectF(r.left(), r.bottom() + 6, r.width(), 10), img)
        pen = QtGui.QPen(QtGui.QColor(255, 196, 0), 2)
        p.setPen(pen)
        for x in (xl, xh):
            p.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom() + 16))
        p.setPen(QtGui.QColor(140, 140, 145))
        font = p.font()
        font.setPixelSize(10)
        p.setFont(font)
        p.drawText(QtCore.QRectF(r.left(), r.top(), r.width(), 12), Qt.AlignRight | Qt.AlignTop, f"{self._x1:g}")

    def _handle_at(self, x: float) -> str:
        r = self._plot_rect()
        return "lo" if abs(x - self._to_x(self._lo, r)) <= abs(x - self._to_x(self._hi, r)) else "hi"

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._drag = self._handle_at(ev.x())
            self._drag_to(ev.x())

    def mouseMoveEvent(self, ev) -> None:
        r = self._plot_rect()
        near = min(abs(ev.x() - self._to_x(self._lo, r)), abs(ev.x() - self._to_x(self._hi, r))) < 7
        self.setCursor(Qt.SizeHorCursor if near or self._drag else Qt.ArrowCursor)
        if self._drag:
            self._drag_to(ev.x())

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None

    def _drag_to(self, x: float) -> None:
        r = self._plot_rect()
        v = min(max(self._to_v(x, r), self._x0), self._x1)
        eps = (self._x1 - self._x0) * 1e-4
        lo, hi = self._lo, self._hi
        if self._drag == "lo":
            lo = min(v, hi - eps)
        else:
            hi = max(v, lo + eps)
        self.set_range(lo, hi)
        self.rangeEdited.emit(lo, hi)


class DisplayPanel(QtWidgets.QWidget):
    """LUT, histogram and display range of the current stack."""

    changed = QtCore.pyqtSignal()
    autoRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state: DisplayState | None = None
        self._integer = True
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.lut = QtWidgets.QComboBox()
        for name in LUTS:
            icon = QtGui.QPixmap.fromImage(_numpy_to_qimage(np.tile(np.arange(256, dtype=np.uint8), (12, 1)),
                                                            COLOR_TABLES[name])).scaled(64, 12)
            self.lut.addItem(QtGui.QIcon(icon), name)
        self.lut.setIconSize(QtCore.QSize(64, 12))
        form.addRow("Lookup table", self.lut)
        lay.addLayout(form)
        self.hist = HistogramWidget()
        lay.addWidget(self.hist)
        rng = QtWidgets.QHBoxLayout()
        self.lo = QtWidgets.QDoubleSpinBox()
        self.hi = QtWidgets.QDoubleSpinBox()
        for sb, label in ((self.lo, "Min"), (self.hi, "Max")):
            sb.setRange(-1e12, 1e12)
            sb.setKeyboardTracking(False)
            rng.addWidget(QtWidgets.QLabel(label))
            rng.addWidget(sb, 1)
        lay.addLayout(rng)
        buttons = QtWidgets.QHBoxLayout()
        self.b_auto = QtWidgets.QPushButton("Auto")
        self.b_auto.setToolTip("0.1 – 99.9 % of the current frame")
        self.b_ims = QtWidgets.QPushButton("ImSpector")
        self.b_ims.setToolTip("Display range that was set in ImSpector (stored in the file)")
        self.b_full = QtWidgets.QPushButton("Full")
        self.b_full.setToolTip("Whole value range")
        for b in (self.b_auto, self.b_ims, self.b_full):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.hint)
        lay.addStretch(1)

        self.lut.currentTextChanged.connect(self._on_lut)
        self.lo.valueChanged.connect(self._on_spin)
        self.hi.valueChanged.connect(self._on_spin)
        self.hist.rangeEdited.connect(self._on_hist)
        self.b_auto.clicked.connect(self.autoRequested)
        self.b_ims.clicked.connect(lambda: self._apply(self._state.imspector) if self._state else None)
        self.b_full.clicked.connect(lambda: self._apply((self._state.x0, self._state.x1)) if self._state else None)

    def set_state(self, st: DisplayState, integer: bool) -> None:
        self._state = st
        self._integer = integer
        for sb in (self.lo, self.hi):
            sb.setDecimals(0 if integer else 4)
        self.b_ims.setEnabled(st.imspector is not None)
        self.hint.setText("HiLo shows values at/below Min in blue and at/above Max in red.")
        self._refresh()

    def set_histogram(self, counts, x0: float, x1: float) -> None:
        self.hist.set_data(counts, x0, x1)

    def _refresh(self) -> None:
        st = self._state
        for w in (self.lut, self.lo, self.hi):
            w.blockSignals(True)
        self.lut.setCurrentText(st.lut)
        self.lo.setValue(st.lo)
        self.hi.setValue(st.hi)
        for w in (self.lut, self.lo, self.hi):
            w.blockSignals(False)
        self.hist.set_range(st.lo, st.hi)
        self.hist.set_lut(st.lut)

    def _apply(self, rng) -> None:
        if not rng or self._state is None:
            return
        lo, hi = rng
        if hi <= lo:
            hi = lo + 1
        self._state.lo, self._state.hi = float(lo), float(hi)
        self._refresh()
        self.changed.emit()

    def _on_lut(self, name: str) -> None:
        if self._state is not None:
            self._state.lut = name
            self.hist.set_lut(name)
            self.changed.emit()

    def _on_spin(self) -> None:
        if self._state is not None:
            self._apply((self.lo.value(), self.hi.value()))

    def _on_hist(self, lo: float, hi: float) -> None:
        if self._integer:
            lo, hi = round(lo), max(round(hi), round(lo) + 1)
        self._apply((lo, hi))


class InfoPanel(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Property", "Value"])
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setUniformRowHeights(True)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(True)

    def show_sections(self, sections) -> None:
        self.clear()
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        for title, rows in sections:
            rows = [(k, v) for k, v in rows if _specified(v)]
            if not rows:
                continue
            top = QtWidgets.QTreeWidgetItem([title])
            top.setFont(0, bold)
            top.setFirstColumnSpanned(False)
            self.addTopLevelItem(top)
            for k, v in rows:
                child = QtWidgets.QTreeWidgetItem([k, str(v)])
                child.setToolTip(1, str(v))
                top.addChild(child)
        self.expandAll()

    def keyPressEvent(self, ev) -> None:
        if ev.matches(QtGui.QKeySequence.Copy):
            lines = [f"{it.text(0)}\t{it.text(1)}".rstrip() for it in self.selectedItems()]
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        else:
            super().keyPressEvent(ev)


class CopyTableView(QtWidgets.QTableView):
    def keyPressEvent(self, ev) -> None:
        if ev.matches(QtGui.QKeySequence.Copy):
            rows: dict[int, dict[int, str]] = {}
            for idx in self.selectionModel().selectedIndexes():
                rows.setdefault(idx.row(), {})[idx.column()] = str(idx.data() or "")
            text = "\n".join("\t".join(cols[c] for c in sorted(cols)) for _, cols in sorted(rows.items()))
            QtWidgets.QApplication.clipboard().setText(text)
        else:
            super().keyPressEvent(ev)


class SettingsPanel(QtWidgets.QWidget):
    """All settings of a stack / the file, filterable, with a compare mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources: list[tuple[str, dict]] = []
        self._labels: dict[str, str] = {}
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.source = QtWidgets.QComboBox()
        self.compare = QtWidgets.QComboBox()
        for cb in (self.source, self.compare):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(16)
        self.only_diff = QtWidgets.QCheckBox("only differences")
        self.only_diff.setChecked(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.compare, 1)
        row.addWidget(self.only_diff)
        form.addRow("Show", self.source)
        form.addRow("Compare with", row)
        lay.addLayout(form)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter (setting, value or description)…")
        self.filter.setClearButtonEnabled(True)
        lay.addWidget(self.filter)
        self.model = QtGui.QStandardItemModel(self)
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.table = CopyTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        self.count = QtWidgets.QLabel()
        self.count.setWordWrap(True)
        self.count.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.count)
        self.source.currentIndexChanged.connect(self._rebuild)
        self.compare.currentIndexChanged.connect(self._rebuild)
        self.only_diff.toggled.connect(self._rebuild)
        self.filter.textChanged.connect(self._on_filter)

    def set_sources(self, sources: list[tuple[str, dict]], labels: dict[str, str]) -> None:
        previous = self.compare.currentText()
        self._sources, self._labels = sources, labels
        for cb in (self.source, self.compare):
            cb.blockSignals(True)
            cb.clear()
        self.source.addItems([name for name, _ in sources])
        self.compare.addItem("— none —")
        self.compare.addItems([name for name, _ in sources])
        i = self.compare.findText(previous)
        self.compare.setCurrentIndex(i if i > 0 else 0)
        for cb in (self.source, self.compare):
            cb.blockSignals(False)
        self._rebuild()

    def _on_filter(self, text: str) -> None:
        self.proxy.setFilterFixedString(text)
        self._update_count()

    def _rebuild(self) -> None:
        self.model.clear()
        if not self._sources:
            self._update_count()
            return
        a_name, a = self._sources[max(self.source.currentIndex(), 0)]
        ci = self.compare.currentIndex() - 1
        b_name, b = self._sources[ci] if ci >= 0 else (None, None)
        highlight = QtGui.QBrush(QtGui.QColor(255, 190, 90))

        def item(text, brush=None):
            it = QtGui.QStandardItem(text)
            it.setEditable(False)
            it.setToolTip(text)
            if brush is not None:
                it.setForeground(brush)
            return it

        if b is None:
            self.model.setHorizontalHeaderLabels(["Setting", "Value", "Description"])
            for k in sorted(a, key=str.lower):
                self.model.appendRow([item(k), item(_fmt_value(a[k])), item(self._labels.get(k, ""))])
        else:
            self.model.setHorizontalHeaderLabels(["Setting", a_name.split("  (")[0], b_name.split("  (")[0],
                                                  "Description"])
            for k in sorted(set(a) | set(b), key=str.lower):
                va, vb = a.get(k), b.get(k)
                differ = (k not in a) or (k not in b) or va != vb
                if self.only_diff.isChecked() and not differ:
                    continue
                brush = highlight if differ else None
                self.model.appendRow([item(k), item("—" if k not in a else _fmt_value(va), brush),
                                      item("—" if k not in b else _fmt_value(vb), brush),
                                      item(self._labels.get(k, ""))])
        vw = max(self.table.viewport().width(), 300)
        n_values = self.model.columnCount() - 2
        self.table.setColumnWidth(0, int(vw * (0.42 if n_values == 1 else 0.40)))
        for c in range(1, 1 + n_values):
            self.table.setColumnWidth(c, int(vw * (0.28 if n_values == 1 else 0.22)))
        self._update_count()

    def _update_count(self) -> None:
        shown, total = self.proxy.rowCount(), self.model.rowCount()
        mode = "differing settings" if self.compare.currentIndex() > 0 and self.only_diff.isChecked() else "settings"
        self.count.setText(f"{total} {mode}" + (f", {shown} match the filter" if shown != total else "")
                           + "   ·   Ctrl+C copies selected rows")


class FrameControls(QtWidgets.QWidget):
    """One slider per non-image axis (T, Z, ...), with playback on T."""

    indexChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(10, 4, 10, 6)
        self.grid.setHorizontalSpacing(10)
        self.grid.setColumnStretch(1, 1)
        self.rows: list[dict] = []
        self.stack: mr.DataStack | None = None
        self.play_row = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(lambda: self.step(1))
        self.play_btn = QtWidgets.QPushButton()
        self.play_btn.setMinimumWidth(84)
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Playback speed (defaults to the real acquisition rate, max. 60)")
        self.fps.valueChanged.connect(lambda v: self.timer.setInterval(max(1, round(1000 / v))))
        self.hide()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def configure(self, stack: mr.DataStack, axes: str, shape: tuple) -> None:
        self.stop()
        for r in self.rows:
            for w in (r["label"], r["slider"], r["spin"], r["info"]):
                self.grid.removeWidget(w)
                w.deleteLater()
        self.grid.removeWidget(self.play_btn)
        self.grid.removeWidget(self.fps)
        self.rows, self.stack = [], stack
        for i, (letter, n) in enumerate(zip(axes, shape)):
            label = QtWidgets.QLabel(f"<b>{letter}</b>")
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setRange(0, n - 1)
            slider.setPageStep(max(1, n // 10))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, n)
            spin.setSuffix(f" / {n}")
            info = QtWidgets.QLabel()
            info.setMinimumWidth(120)
            slider.valueChanged.connect(lambda v, i=i: self._on_slider(i, v))
            spin.valueChanged.connect(lambda v, i=i: self.rows[i]["slider"].setValue(v - 1))
            for col, w in enumerate((label, slider, spin, info)):
                self.grid.addWidget(w, i, col)
                for x in [w] + w.findChildren(QtWidgets.QWidget):
                    x.setAcceptDrops(False)  # let file drops reach the main window
            self.rows.append({"letter": letter, "n": n, "label": label, "slider": slider, "spin": spin,
                              "info": info})
        if self.rows:
            self.play_row = next((i for i, r in enumerate(self.rows) if r["letter"] == "T"), 0)
            self.grid.addWidget(self.play_btn, self.play_row, 4)
            self.grid.addWidget(self.fps, self.play_row, 5)
            dt = stack.time_increment if self.rows[self.play_row]["letter"] == "T" else None
            self.fps.setValue(int(min(60, max(1, round(1.0 / dt)))) if dt else 10)
        self._set_icon()
        self._update_info()
        self.setVisible(bool(self.rows))

    def index(self) -> tuple:
        return tuple(r["slider"].value() for r in self.rows)

    def _on_slider(self, i: int, v: int) -> None:
        spin = self.rows[i]["spin"]
        spin.blockSignals(True)
        spin.setValue(v + 1)
        spin.blockSignals(False)
        self._update_info(i)
        self.indexChanged.emit()

    def step(self, delta: int) -> None:
        if self.rows:
            r = self.rows[self.play_row]
            r["slider"].setValue((r["slider"].value() + delta) % r["n"])

    def jump(self, where: int) -> None:
        if self.rows:
            r = self.rows[self.play_row]
            r["slider"].setValue(0 if where == 0 else r["n"] - 1)

    def toggle_play(self) -> None:
        if not self.rows:
            return
        if self.timer.isActive():
            self.stop()
        else:
            self.timer.start(max(1, round(1000 / self.fps.value())))
            self._set_icon()

    def stop(self) -> None:
        self.timer.stop()
        self._set_icon()

    def _set_icon(self) -> None:
        self.play_btn.setText("❚❚  Pause" if self.timer.isActive() else "▶  Play")

    def _update_info(self, only: int | None = None) -> None:
        s = self.stack
        for i, r in enumerate(self.rows):
            if only is not None and i != only:
                continue
            v, text = r["slider"].value(), ""
            if r["letter"] == "T":
                ts = s.timestamps
                if ts and len(ts) == r["n"]:
                    text = f"t = {ts[v] - ts[0]:.3f} s"
                elif s.time_increment:
                    text = f"t = {v * s.time_increment:.3f} s"
            elif r["letter"] == "Z" and s.step("Z"):
                unit = (s.axis_info("Z") or {}).get("unit") or "µm"
                text = f"z = {v * s.step('Z'):.3g} {unit}"
            r["info"].setText(text)


class OverviewPage(QtWidgets.QWidget):
    """All stacks of a workspace side by side."""

    stackActivated = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 8)
        self.title = ElidedLabel(mode=Qt.ElideMiddle)
        font = QtGui.QFont(self.title.font())
        font.setPointSizeF(font.pointSizeF() * 1.35)
        font.setBold(True)
        self.title.setFont(font)
        self.subtitle = QtWidgets.QLabel()
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)
        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.IconMode)
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setIconSize(QtCore.QSize(200, 200))
        self.list.setGridSize(QtCore.QSize(236, 280))
        self.list.setWordWrap(True)
        self.list.setSpacing(6)
        self.list.setUniformItemSizes(True)
        self.list.setDragEnabled(False)
        self.list.itemClicked.connect(lambda it: self.stackActivated.emit(it.data(Qt.UserRole)))
        self.list.itemActivated.connect(lambda it: self.stackActivated.emit(it.data(Qt.UserRole)))
        lay.addWidget(self.list, 1)

    def show_file(self, msr: mr.MSRFile, thumbs: dict[int, QtGui.QPixmap]) -> None:
        ws = ", ".join(ps.get("propset_label", "") for ps in msr.property_sets if ps.get("propset_label"))
        self.title.setText(os.path.basename(msr.path))
        self.title.setToolTip(msr.path)
        self.subtitle.setText(f"{len(msr.stacks)} stacks{' · workspace ' + ws if ws else ''} · "
                              "click a stack to open it")
        self.list.clear()
        for pos, s in enumerate(msr.stacks):
            px = s.pixel_size[0]
            text = f"{_stack_label(pos, s)}\n{_shape_text(s)}  ({s.axes})\n{s.time}" + (f" · {px:.3g} µm/px" if px else "")
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(thumbs[pos]), text)
            it.setData(Qt.UserRole, pos)
            it.setToolTip(f"{s.channel_id}\n{s.meta.get('Instrument Mode', '')} · {s.meta.get('Measurement Mode', '')}")
            self.list.addItem(it)


class WelcomePage(QtWidgets.QWidget):
    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(40, 40, -40, -40)
        p.setPen(QtGui.QPen(QtGui.QColor(90, 90, 96), 2, Qt.DashLine))
        p.drawRoundedRect(r, 18, 18)
        font = QtGui.QFont(self.font())
        font.setPointSize(20)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(210, 210, 215))
        p.drawText(r.adjusted(0, -40, 0, -40), Qt.AlignCenter, "Drop .msr files here")
        font.setPointSize(10)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QtGui.QColor(140, 140, 146))
        p.drawText(r.adjusted(0, 50, 0, 50), Qt.AlignCenter,
                   "or File ▸ Open… (Ctrl+O)  ·  files or whole folders\n"
                   "LaVision BioTec ImSpector workspaces: every stack, source and setting")


class DropOverlay(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()

    def show_over(self, w: QtWidgets.QWidget) -> None:
        self.setGeometry(w.geometry())
        self.raise_()
        self.show()

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(16, 40, 72, 248))
        p.setPen(QtGui.QPen(QtGui.QColor(90, 170, 255), 3, Qt.DashLine))
        p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(14, 14, -14, -14), 16, 16)
        font = QtGui.QFont(self.font())
        font.setPointSize(20)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(235, 243, 255))
        p.drawText(self.rect(), Qt.AlignCenter, "Drop to open")
        font.setPointSize(10)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QtGui.QColor(170, 195, 230))
        p.drawText(self.rect().adjusted(0, 70, 0, 70), Qt.AlignCenter, ".msr files or folders")


class Task(QtCore.QThread):
    """Runs a function in the background (exports)."""

    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self.fn())
        except Exception as exc:  # reported in the GUI
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
# main window
# -----------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.qsettings = QtCore.QSettings("MSRViewer", "MSRViewer")
        self.files: dict[str, mr.MSRFile] = {}
        self.file_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self.thumbs: dict[str, dict[int, QtGui.QPixmap]] = {}
        self.states: dict[tuple, DisplayState] = {}
        self.cur_key: tuple | None = None
        self.arr = None
        self.frame: np.ndarray | None = None
        self._map_key = None
        self._map_lut = None
        self._hist_skip = 0
        self.tasks: set[Task] = set()
        self._build_ui()
        self._build_actions()
        for w in self.findChildren(QtWidgets.QWidget):
            w.setAcceptDrops(False)
        self.setAcceptDrops(True)
        geometry = self.qsettings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1480, 900)
        self._update_actions()

    # -- construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setTextElideMode(Qt.ElideMiddle)
        self.tree.setIconSize(QtCore.QSize(40, 40))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.currentItemChanged.connect(self._on_tree_current)

        self.view = ImageView()
        self.view.mouseMoved.connect(self._on_mouse)
        self.view.mouseLeft.connect(lambda: self.pixel_label.setText(""))
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))
        self.frames = FrameControls()
        self.frames.indexChanged.connect(self._render_frame)
        self.view.stepRequested.connect(self.frames.step)
        self.view.jumpRequested.connect(self.frames.jump)
        self.view.playRequested.connect(self.frames.toggle_play)
        self.viewer_page = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viewer_page)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(self.view, 1)
        vl.addWidget(self.frames)
        self.overview = OverviewPage()
        self.overview.stackActivated.connect(self._select_stack)
        self.welcome = WelcomePage()
        self.center = QtWidgets.QStackedWidget()
        for page in (self.welcome, self.overview, self.viewer_page):
            self.center.addWidget(page)

        self.display = DisplayPanel()
        self.display.changed.connect(lambda: self._render_frame(histogram=False))
        self.display.autoRequested.connect(self._auto_contrast)
        self.info = InfoPanel()
        self.settings_panel = SettingsPanel()
        self.display.setEnabled(False)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.display, "Display")
        self.tabs.addTab(self.info, "Info")
        self.tabs.addTab(self.settings_panel, "Settings")

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.center)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([270, 820, 390])
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        sb = self.statusBar()
        self.pixel_label = ElidedLabel()
        self.zoom_label = QtWidgets.QLabel()
        self.busy = QtWidgets.QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(140)
        self.busy.setFormat("exporting…")
        self.busy.hide()
        sb.addWidget(self.pixel_label, 1)
        sb.addPermanentWidget(self.busy)
        sb.addPermanentWidget(self.zoom_label)
        self.overlay = DropOverlay(self)

    def _build_actions(self) -> None:
        st = self.style()

        def act(text, slot, shortcut=None, icon=None, tip=None):
            a = QtWidgets.QAction(text, self)
            if shortcut:
                a.setShortcut(QtGui.QKeySequence(shortcut))
            if icon is not None:
                a.setIcon(st.standardIcon(icon))
            if tip:
                a.setToolTip(tip)
                a.setStatusTip(tip)
            a.triggered.connect(slot)
            return a

        S = QtWidgets.QStyle
        self.a_open = act("&Open…", self.open_dialog, QtGui.QKeySequence.Open, S.SP_DialogOpenButton,
                          "Open .msr files (or drop them onto the window)")
        self.a_close = act("&Close file", self.close_current_file, "Ctrl+W", S.SP_DialogCloseButton)
        self.a_export = act("Export all stacks as &OME-TIFF…", lambda: self.export_file(False), "Ctrl+E",
                            S.SP_DialogSaveButton, "Export every stack of the file (OME-TIFF + metadata.json)")
        self.a_export_ij = act("Export all stacks as &ImageJ TIFF…", lambda: self.export_file(True))
        self.a_save_stack = act("&Save current stack as…", self.save_stack, QtGui.QKeySequence.Save,
                                tip="Save the displayed stack as OME-TIFF or ImageJ TIFF")
        self.a_save_png = act("Save view as &PNG…", self.save_png, "Ctrl+Shift+S",
                              tip="Current frame with LUT (and scale bar) at full resolution")
        self.a_quit = act("&Quit", self.close, QtGui.QKeySequence.Quit)
        self.a_fit = act("&Fit to window", self.view.fit, "F", tip="Fit (F, or double-click the image)")
        self.a_1to1 = act("&Actual pixels", self.view.zoom_1to1, "1", tip="100 %: one image pixel per screen pixel (1)")
        self.a_scalebar = act("Scale &bar", self._toggle_scalebar, "B", tip="Show scale bar (B)")
        self.a_scalebar.setCheckable(True)
        self.a_scalebar.setChecked(True)
        self.a_play = act("&Play / pause", self.frames.toggle_play, "P")
        self.a_prev = act("Previous frame", lambda: self.frames.step(-1), ",")
        self.a_next = act("Next frame", lambda: self.frames.step(1), ".")
        self.a_about = act("&About", self.about)

        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.a_open, None, self.a_export, self.a_export_ij, self.a_save_stack, self.a_save_png, None,
                  self.a_close, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&View")
        for a in (self.a_fit, self.a_1to1, self.a_scalebar, None, self.a_play, self.a_prev, self.a_next):
            m.addSeparator() if a is None else m.addAction(a)
        mb.addMenu("&Help").addAction(self.a_about)

        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_open, self.a_export, None, self.a_fit, self.a_1to1, self.a_scalebar):
            tb.addSeparator() if a is None else tb.addAction(a)

    # -- file handling -------------------------------------------------------------

    def open_dialog(self) -> None:
        start = self.qsettings.value("last_dir", os.path.expanduser("~"))
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open ImSpector files", start,
                                                          "ImSpector files (*.msr);;All files (*)")
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths) -> None:
        files = []
        for p in paths:
            if os.path.isdir(p):
                files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".msr"))
            elif os.path.isfile(p):
                files.append(p)
        errors, last = [], None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for f in files:
                key = os.path.normcase(os.path.abspath(f))
                if key in self.files:
                    last = self.file_items[key]
                    continue
                try:
                    msr = mr.MSRFile(f)
                    if not msr.stacks:
                        msr.close()
                        raise mr.MSRFormatError("no image data found")
                except Exception as exc:
                    errors.append(f"{os.path.basename(f)}:\n    {exc}")
                    continue
                self.files[key] = msr
                last = self._add_file_item(key, msr)
                self.qsettings.setValue("last_dir", os.path.dirname(os.path.abspath(f)))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if last is not None:
            self.tree.setCurrentItem(last.child(0) if last.childCount() else last)
            self.view.setFocus()
        if errors:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Could not open:\n\n" + "\n\n".join(errors))
        self._update_actions()

    def _add_file_item(self, key: str, msr: mr.MSRFile) -> QtWidgets.QTreeWidgetItem:
        top = QtWidgets.QTreeWidgetItem([os.path.basename(msr.path)])
        font = QtGui.QFont(self.tree.font())
        font.setBold(True)
        top.setFont(0, font)
        top.setData(0, Qt.UserRole, ("file", key))
        warn = f"\n\n{len(msr.warnings)} warning(s):\n" + "\n".join(msr.warnings) if msr.warnings else ""
        top.setToolTip(0, msr.path + warn)
        if msr.warnings:
            top.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning))
        self.thumbs[key] = {}
        for pos, s in enumerate(msr.stacks):
            try:
                thumb = _thumbnail(s)
            except Exception:
                thumb = QtGui.QPixmap(200, 200)
                thumb.fill(QtGui.QColor(60, 30, 30))
            self.thumbs[key][pos] = thumb
            child = QtWidgets.QTreeWidgetItem([f"{_stack_label(pos, s)}\n{_shape_text(s)}  ·  {s.time}"])
            child.setIcon(0, QtGui.QIcon(thumb))
            child.setData(0, Qt.UserRole, ("stack", key, pos))
            px = s.pixel_size[0]
            child.setToolTip(0, f"{s.channel_id}  ({s.source})\naxes {s.axes}, {_shape_text(s)}, {s.dtype}"
                             + (f"\n{px:.4g} µm/px" if px else ""))
            top.addChild(child)
        self.tree.addTopLevelItem(top)
        top.setExpanded(True)
        self.file_items[key] = top
        return top

    def _current_file_key(self) -> str | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        return it.data(0, Qt.UserRole)[1]

    def close_current_file(self) -> None:
        key = self._current_file_key()
        if key is None:
            return
        if self.tasks:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Please wait until the export has finished.")
            return
        if self.cur_key and self.cur_key[0] == key:
            self.frames.stop()
            self.frames.configure(self.files[key].stacks[0], "", ())
            self.arr, self.frame, self.cur_key = None, None, None
            self.view.clear()
        item = self.file_items.pop(key)
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        self.thumbs.pop(key, None)
        for k in [k for k in self.states if k[0] == key]:
            del self.states[k]
        self.files.pop(key).close()
        if not self.files:
            self.center.setCurrentWidget(self.welcome)
            self.info.clear()
            self.settings_panel.set_sources([], {})
            self.setWindowTitle(APP_NAME)
        self._update_actions()

    # -- selection ---------------------------------------------------------------

    def _select_stack(self, pos: int) -> None:
        key = self._current_file_key()
        if key is not None and pos < self.file_items[key].childCount():
            self.tree.setCurrentItem(self.file_items[key].child(pos))
            self.view.setFocus()

    def _on_tree_current(self, cur, prev) -> None:
        if cur is None:
            self.center.setCurrentWidget(self.welcome)
        else:
            data = cur.data(0, Qt.UserRole)
            if data[0] == "file":
                self._show_file(data[1])
            else:
                self._show_stack(data[1], data[2])
        self._update_actions()

    def _show_file(self, key: str) -> None:
        msr = self.files[key]
        self.frames.stop()
        self.overview.show_file(msr, self.thumbs[key])
        self.center.setCurrentWidget(self.overview)
        self.info.show_sections(self._file_sections(msr))
        self.settings_panel.set_sources(self._setting_sources(msr, None), msr.labels)
        self.display.setEnabled(False)
        self.pixel_label.setText("")
        self.setWindowTitle(f"{os.path.basename(msr.path)} — {APP_NAME}")

    def _show_stack(self, key: str, pos: int) -> None:
        msr = self.files[key]
        s = msr.stacks[pos]
        if self.cur_key != (key, pos):
            self.frames.stop()
            self.arr = s.asarray()
            self.cur_key = (key, pos)
            st = self.states.get(self.cur_key)
            if st is None:
                st = self.states[self.cur_key] = _initial_state(self.arr, s)
            self.frames.configure(s, s.axes[:-2], self.arr.shape[:-2])
            self.view.pixel_size = s.pixel_size[0]
            self.view.unit = s.units[0] if s.units and s.units[0] else "µm"
            self.display.set_state(st, np.issubdtype(self.arr.dtype, np.integer))
            self._render_frame(reset=True)
        self.center.setCurrentWidget(self.viewer_page)
        self.display.setEnabled(True)
        self.info.show_sections(self._stack_sections(msr, s, pos))
        self.settings_panel.set_sources(self._setting_sources(msr, s), msr.labels)
        self.setWindowTitle(f"{_stack_label(pos, s)} · {s.channel_id} — {os.path.basename(msr.path)} — {APP_NAME}")

    # -- rendering ------------------------------------------------------------------

    def _map8(self, frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scale = 255.0 / max(hi - lo, 1e-12)
        if frame.dtype in (np.uint8, np.uint16):
            key = (frame.dtype.str, lo, hi)
            if self._map_key != key:
                x = np.arange(256 if frame.dtype == np.uint8 else 65536, dtype=np.float32)
                self._map_lut = np.clip((x - lo) * scale + 0.5, 0, 255).astype(np.uint8)
                self._map_key = key
            return self._map_lut[frame]
        return np.clip((frame.astype(np.float32) - lo) * scale + 0.5, 0, 255).astype(np.uint8)

    def _render_frame(self, reset: bool = False, histogram: bool = True) -> None:
        if self.arr is None or self.cur_key is None:
            return
        idx = self.frames.index()
        self.frame = np.asarray(self.arr[idx] if idx else self.arr)
        st = self.states[self.cur_key]
        img = _numpy_to_qimage(self._map8(self.frame, st.lo, st.hi), COLOR_TABLES[st.lut])
        self.view.set_pixmap(QtGui.QPixmap.fromImage(img), reset)
        if histogram:
            if self.frames.playing and self._hist_skip > 0:
                self._hist_skip -= 1
            else:
                self._hist_skip = 5
                counts = _histogram(self.frame, st.x0, st.x1)
                if counts is None:
                    st.x1 = float(2 ** int(math.ceil(math.log2(float(self.frame.max()) + 1))) - 1)
                    counts = _histogram(self.frame, st.x0, st.x1)
                self.display.set_histogram(counts, st.x0, st.x1)

    def _auto_contrast(self) -> None:
        if self.frame is None or self.cur_key is None:
            return
        f = self.frame
        step = max(1, int(math.sqrt(f.size / 1_000_000)))
        lo, hi = _percentiles(f[::step, ::step])
        self.display._apply((lo, hi))

    def _on_mouse(self, x: float, y: float) -> None:
        if self.frame is None or self.cur_key is None:
            return
        ix, iy = int(math.floor(x)), int(math.floor(y))
        h, w = self.frame.shape
        if not (0 <= ix < w and 0 <= iy < h):
            self.pixel_label.setText("")
            return
        s = self.files[self.cur_key[0]].stacks[self.cur_key[1]]
        v = self.frame[iy, ix]
        text = f"x {ix}   y {iy}"
        px, py = s.pixel_size
        if px and py:
            text += f"   ({(ix + 0.5) * px:.2f}, {(iy + 0.5) * py:.2f} µm)"
        text += f"   value {v:.5g}" if isinstance(v, np.floating) else f"   value {v}"
        self.pixel_label.setText(text)

    def _toggle_scalebar(self) -> None:
        self.view.show_scalebar = self.a_scalebar.isChecked()
        self.view.viewport().update()

    # -- panels ---------------------------------------------------------------------

    @staticmethod
    def _file_sections(msr: mr.MSRFile):
        first = msr.stacks[0] if msr.stacks else None
        rows = [("Path", msr.path),
                ("Size", f"{os.path.getsize(msr.path) / 1e6:.1f} MB"),
                ("Format version", msr.version),
                ("Software", first.meta.get("VersionNr") if first else ""),
                ("Data stacks", len(msr.stacks)),
                ("Display views", len(msr.views)),
                ("Global settings", len(msr.properties))]
        sections = [("File", rows)]
        for ps in msr.property_sets:
            sections.append((f"Workspace '{ps.get('propset_label', '')}'",
                             [("Autosave prefix", ps.get("seq_autosave_prefix")), ("Id", ps.get("propset_id")),
                              ("Settings", len(ps.get("properties", {})))]))
        sections.append(("Stacks", [(_stack_label(i, s), f"{s.channel_id} · {_shape_text(s)} · {s.time}")
                                    for i, s in enumerate(msr.stacks)]))
        if msr.warnings:
            sections.append(("Warnings", [(str(i + 1), w) for i, w in enumerate(msr.warnings)]))
        return sections

    @staticmethod
    def _stack_sections(msr: mr.MSRFile, s: mr.DataStack, pos: int):
        m = s.meta
        px, py = s.pixel_size
        fov = [s.axis_info(a) for a in "XY"]
        dt = s.time_increment if "T" in s.axes else None
        acq = [("Source", s.source), ("Channel ID", s.channel_id), ("Acquired", m.get("Creation Date", s.time)),
               ("Instrument mode", m.get("Instrument Mode")), ("Measurement mode", m.get("Measurement Mode")),
               ("Workspace", f"S{pos + 1} (element {s.index} of the stack array)")]
        geo = [("Axes", " × ".join(f"{n} {a}" for a, n in zip(s.axes, s.shape))),
               ("Data type", s.dtype),
               ("Pixel size", f"{px:.4g} × {py:.4g} µm" if px and py else ""),
               ("Field of view", f"{fov[0]['length']:.4g} × {fov[1]['length']:.4g} µm" if all(fov) else "")]
        if dt:
            geo.append(("Frame interval", f"{dt * 1000:.3f} ms ({1 / dt:.2f} fps)"))
        if len(s.timestamps) > 1:
            geo.append(("Duration", f"{s.timestamps[-1] - s.timestamps[0]:.3f} s "
                                    f"(first frame at {s.timestamps[0]:.3f} s)"))
        if "Z" in s.axes and s.step("Z"):
            geo.append(("Z step", f"{s.step('Z'):.4g} {(s.axis_info('Z') or {}).get('unit') or 'µm'}"))
        geo += [("Dimension labels", ", ".join(s.labels)),
                ("Dimension lengths", ", ".join(f"{v:g}" for v in s.lengths)),
                ("Units", ", ".join(u or "–" for u in s.units))]
        obj = [("Objective", m.get("ObjectiveID")), ("NA", m.get("ObjectiveNA")),
               ("Immersion", m.get("ObjectiveImmersion"))]
        disp = []
        if s.view is not None and s.view.lut:
            disp = [("Display range", f"{s.view.lut[0]:g} – {s.view.lut[1]:g}"),
                    ("Zoom in ImSpector", f"{s.view.header.get('zoom', 0):.3g}")]
        cam = []
        if s.camera_stamp:
            cam = [("First frame (camera clock)", s.camera_stamp["time"]),
                   ("Camera image counter", s.camera_stamp["image_counter"]),
                   ("Note", "the first 14 pixels of each frame hold this BCD time stamp")]
        user = [(k, v) for k, v in m.items() if k in ("First Name", "Last Name", "Email", "Institution",
                                                        "Group", "Description", "Rating")]
        f = [("Path", msr.path), ("Pixel data offset", f"{s.data_offset:,} bytes"),
             ("Pixel data size", f"{s.nbytes / 1e6:.2f} MB"), ("Settings in snapshot", len(s.properties)),
             ("Header parse", s.parse_mode)]
        return [("Acquisition", acq), ("Geometry", geo), ("Objective", obj), ("ImSpector display", disp),
                ("PCO camera", cam), ("User", user), ("File", f)]

    @staticmethod
    def _setting_sources(msr: mr.MSRFile, s: mr.DataStack | None):
        sources = []
        stacks = [(f"{_stack_label(i, t)}  ({t.channel_id}, {t.time})", t.properties) for i, t in enumerate(msr.stacks)]
        if s is not None:
            pos = msr.stacks.index(s)
            sources.append(stacks[pos])
            sources += [x for i, x in enumerate(stacks) if i != pos]
        sources.append(("Global settings  (file)", msr.properties))
        for ps in msr.property_sets:
            sources.append((f"Workspace '{ps.get('propset_label', '')}'  (property set)", ps.get("properties", {})))
        if s is None:
            sources += stacks
        return sources

    # -- export -------------------------------------------------------------------

    def _run_task(self, fn, on_done, what: str) -> None:
        task = Task(fn, self)
        self.tasks.add(task)
        task.done.connect(on_done)
        task.failed.connect(lambda msg: QtWidgets.QMessageBox.critical(self, APP_NAME, f"{what} failed:\n\n{msg}"))
        task.finished.connect(lambda: (self.tasks.discard(task), self._update_actions()))
        self._update_actions()
        task.start()

    def _done_box(self, text: str, details: str, folder: str) -> None:
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, APP_NAME, text, parent=self)
        if details:
            box.setDetailedText(details)
        open_btn = box.addButton("Open folder", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Ok)
        box.exec_()
        if box.clickedButton() is open_btn:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def export_file(self, imagej: bool) -> None:
        key = self._current_file_key()
        if key is None:
            return
        msr = self.files[key]
        start = self.qsettings.value("export_dir", os.path.dirname(msr.path))
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export into folder (a '<name>_tiff' subfolder will be created)", start)
        if not folder:
            return
        self.qsettings.setValue("export_dir", folder)
        path, lines = msr.path, []
        out = os.path.join(folder, os.path.splitext(os.path.basename(path))[0] + "_tiff")

        def done(written):
            self._done_box(f"Exported {len(written)} image file(s) + metadata.json to\n{out}", "\n".join(lines), out)

        self._run_task(lambda: mr.export(path, folder, imagej=imagej, log=lines.append), done, "Export")

    def save_stack(self) -> None:
        if self.cur_key is None or self.center.currentWidget() is not self.viewer_page:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        stem = os.path.splitext(os.path.basename(msr.path))[0]
        name = f"{stem}_S{pos + 1}_{mr._safe(s.channel_id.split(':')[0] or s.source)}.ome.tif"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, "Save stack", start,
                                                          "OME-TIFF (*.ome.tif);;ImageJ TIFF (*.tif)")
        if not path:
            return
        imagej = flt.startswith("ImageJ")
        base = re.sub(r"(\.ome)?\.tiff?$", "", path, flags=re.I)
        path = base + (".tif" if imagej else ".ome.tif")
        self.qsettings.setValue("export_dir", os.path.dirname(path))
        self._run_task(lambda: mr.export_group(msr, [s], path, imagej=imagej),
                       lambda rec: self._done_box(f"Saved {os.path.basename(path)}\naxes {rec['axes']}, "
                                                  f"shape {' × '.join(map(str, rec['shape']))}", "",
                                                  os.path.dirname(path)), "Saving")

    def save_png(self) -> None:
        if self.frame is None or self.cur_key is None:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        st = self.states[self.cur_key]
        rgb = np.ascontiguousarray(LUTS[st.lut][self._map8(self.frame, st.lo, st.hi)])
        h, w = self.frame.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).convertToFormat(QtGui.QImage.Format_RGB32)
        px = s.pixel_size[0]
        if self.view.show_scalebar and px:
            p = QtGui.QPainter(img)
            paint_scalebar(p, w, h, px, self.view.unit, font_px=max(12, h // 28), bar_h=max(3, h // 110),
                           margin=max(8, w // 40))
            p.end()
        stem = os.path.splitext(os.path.basename(msr.path))[0]
        frame_txt = "_".join(f"{a}{i + 1}" for a, i in zip(s.axes[:-2], self.frames.index()))
        name = f"{stem}_S{pos + 1}{'_' + frame_txt if frame_txt else ''}.png"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save view as PNG", start, "PNG image (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            if not img.save(path):
                QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not write {path}")

    # -- misc -------------------------------------------------------------------------

    def _update_actions(self) -> None:
        has_file = self._current_file_key() is not None
        showing_stack = self.cur_key is not None and self.center.currentWidget() is self.viewer_page
        busy = bool(self.tasks)
        for a in (self.a_export, self.a_export_ij):
            a.setEnabled(has_file and not busy)
        self.a_close.setEnabled(has_file and not busy)
        for a in (self.a_save_stack,):
            a.setEnabled(showing_stack and not busy)
        for a in (self.a_save_png, self.a_fit, self.a_1to1):
            a.setEnabled(showing_stack)
        for a in (self.a_play, self.a_prev, self.a_next):
            a.setEnabled(showing_stack and bool(self.frames.rows))
        self.busy.setVisible(busy)

    def _tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        if item.data(0, Qt.UserRole)[0] == "stack":
            menu.addAction(self.a_save_stack)
            menu.addAction(self.a_save_png)
            menu.addSeparator()
        menu.addAction(self.a_export)
        menu.addAction(self.a_export_ij)
        menu.addSeparator()
        key = item.data(0, Qt.UserRole)[1]
        menu.addAction("Show in folder", lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(os.path.dirname(self.files[key].path))))
        menu.addAction(self.a_close)
        menu.exec_(self.tree.viewport().mapToGlobal(pos))

    def about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, APP_NAME,
            f"<b>{APP_NAME} {__version__}</b><br>Viewer for LaVision BioTec ImSpector .msr files.<br><br>"
            f"Reader: msr_reader.py {mr.__version__} (format notes: MSR_FORMAT.md)<br>"
            "Wheel: zoom · drag: pan · double-click: fit<br>"
            "Left/Right: frame · Space: play · F: fit · 1: 100 % · B: scale bar")

    # -- drag & drop ------------------------------------------------------------------

    @staticmethod
    def _paths(mime: QtCore.QMimeData) -> list[str]:
        out = []
        if mime.hasUrls():
            for url in mime.urls():
                p = url.toLocalFile()
                if p and (os.path.isdir(p) or p.lower().endswith(".msr")):
                    out.append(p)
        return out

    def dragEnterEvent(self, ev) -> None:
        if self._paths(ev.mimeData()):
            ev.acceptProposedAction()
            self.overlay.show_over(self.centralWidget())
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:
        if self._paths(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev) -> None:
        self.overlay.hide()

    def dropEvent(self, ev) -> None:
        self.overlay.hide()
        paths = self._paths(ev.mimeData())
        if paths:
            ev.acceptProposedAction()
            QtCore.QTimer.singleShot(0, lambda: self.open_paths(paths))

    def closeEvent(self, ev) -> None:
        if self.tasks:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Please wait until the export has finished.")
            ev.ignore()
            return
        self.qsettings.setValue("geometry", self.saveGeometry())
        self.frames.stop()
        self.arr = self.frame = None
        for msr in self.files.values():
            msr.close()
        ev.accept()


# -----------------------------------------------------------------------------
# application
# -----------------------------------------------------------------------------

def _dark_palette(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    c = QtGui.QColor
    p.setColor(QtGui.QPalette.Window, c(37, 37, 40))
    p.setColor(QtGui.QPalette.WindowText, c(222, 222, 226))
    p.setColor(QtGui.QPalette.Base, c(28, 28, 31))
    p.setColor(QtGui.QPalette.AlternateBase, c(43, 43, 47))
    p.setColor(QtGui.QPalette.ToolTipBase, c(50, 50, 54))
    p.setColor(QtGui.QPalette.ToolTipText, c(230, 230, 230))
    p.setColor(QtGui.QPalette.Text, c(222, 222, 226))
    p.setColor(QtGui.QPalette.Button, c(48, 48, 52))
    p.setColor(QtGui.QPalette.ButtonText, c(222, 222, 226))
    p.setColor(QtGui.QPalette.BrightText, c(255, 80, 80))
    p.setColor(QtGui.QPalette.Link, c(90, 165, 255))
    p.setColor(QtGui.QPalette.Highlight, c(40, 110, 190))
    p.setColor(QtGui.QPalette.HighlightedText, c(255, 255, 255))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        p.setColor(QtGui.QPalette.Disabled, role, c(115, 115, 120))
    app.setPalette(p)
    app.setStyleSheet("QToolTip { color: #e6e6e6; background-color: #323236; border: 1px solid #55555a; }")


def _install_excepthook() -> None:
    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        if sys.stderr:
            sys.stderr.write(text)
        if QtWidgets.QApplication.instance() is not None:
            QtWidgets.QMessageBox.critical(None, APP_NAME, "Unexpected error:\n\n" + text[-4000:])
    sys.excepthook = hook


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    if sys.stdout is None:  # pythonw
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MSRViewer.MSRViewer")
        except Exception:
            pass
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QtGui.QGuiApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    if sys.platform == "win32" and "Segoe UI" in QtGui.QFontDatabase().families():
        app.setFont(QtGui.QFont("Segoe UI", 9))  # Qt5 default is 'MS Shell Dlg 2' 8 pt
    app.setWindowIcon(_app_icon())
    _dark_palette(app)
    _install_excepthook()
    win = MainWindow()
    win.show()
    paths = [a for a in argv[1:] if not a.startswith("-")]
    if paths:
        QtCore.QTimer.singleShot(0, lambda: win.open_paths(paths))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
