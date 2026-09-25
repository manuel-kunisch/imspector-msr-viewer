"""
msr_psf.py -- point spread function analysis, shown on demand.

Click a bead in the image: a cross snaps to its centre (brightest pixel
nearby, refined by the intensity centroid).  Along the two arms of the cross
the lateral profiles are sampled (bilinear, optionally averaged over parallel
neighbouring lines, rotatable); for z-stacks the axial profile through the bead
is the mean of a small box in every slice.  Widths are measured at a level
above a baseline (FWHM, 1/e, 1/e²):

* threshold: the two crossings next to the peak, linearly interpolated
* Gaussian: least-squares fit (scipy), width at the same level

Concepts ported from OME-Tiff-PSF-Evaluator-V2/psf_analyzer_gui.py.  Needs
pyqtgraph for the plots; scipy only for the Gaussian fit.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  (registers pg.exporters)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import msr_reader as mr
from msr_widgets import Handle

LEVELS = (("FWHM (50 %)", 0.5), ("1/e (36.8 %)", 1 / math.e), ("1/e² (13.5 %)", 1 / math.e ** 2))
BASELINES = ("profile ends", "background box", "zero")
COLORS = {"x": "#3ddc84", "y": "#ff5fd2", "z": "#38c7ff"}

try:
    from scipy.optimize import curve_fit
except ImportError:  # Gaussian fit disabled
    curve_fit = None
try:
    from scipy.ndimage import map_coordinates
except ImportError:  # bilinear sampling instead of cubic splines
    map_coordinates = None


# -----------------------------------------------------------------------------
# measurement
# -----------------------------------------------------------------------------

def bilinear(frame: np.ndarray, x, y) -> np.ndarray:
    """Values at array coordinates (x = column, y = row, pixel centres at integers); NaN outside."""
    h, w = frame.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = np.clip(np.floor(x), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(y), 0, h - 1).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    f = frame
    v = (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x1] * fx * (1 - fy)
         + f[y1, x0] * (1 - fx) * fy + f[y1, x1] * fx * fy)
    return np.where(inside, v, np.nan)


def sample_points(frame: np.ndarray, x, y) -> np.ndarray:
    """Values at array coordinates (pixel centres at integers); NaN outside the frame.

    Cubic spline interpolation (scipy) on a window around the points; bilinear
    interpolation would broaden a PSF sampled between pixel centres by ~2 % at
    sigma = 2 px.  Falls back to bilinear without scipy.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h, w = frame.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    if map_coordinates is None or not inside.any():
        return bilinear(frame, x, y)
    m = 8  # margin: the spline prefilter's edge effect decays as 0.27^distance
    x0 = max(0, int(np.floor(x[inside].min())) - m)
    x1 = min(w, int(np.ceil(x[inside].max())) + m + 1)
    y0 = max(0, int(np.floor(y[inside].min())) - m)
    y1 = min(h, int(np.ceil(y[inside].max())) + m + 1)
    win = np.asarray(frame[y0:y1, x0:x1], dtype=np.float64)
    v = map_coordinates(win, [np.clip(y - y0, 0, y1 - y0 - 1), np.clip(x - x0, 0, x1 - x0 - 1)],
                        order=3, mode="nearest")
    return np.where(inside, v, np.nan)


def sample_line(frame: np.ndarray, cx: float, cy: float, angle_deg: float, length_px: float,
                half_width: int = 0, per_px: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Profile through (cx, cy) along angle_deg, averaged over 2*half_width+1 lines 1 px apart.

    (cx, cy) are image-view coordinates (pixel i spans [i, i+1)).  Returns the
    distance from the centre in px and the profile.
    """
    n = max(9, int(math.ceil(length_px * per_px)) + 1)
    t = np.linspace(-length_px / 2.0, length_px / 2.0, n)
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    o = np.arange(-half_width, half_width + 1, dtype=np.float64)[:, None]
    vals = sample_points(frame, cx - 0.5 + t * ux - o * uy, cy - 0.5 + t * uy + o * ux)
    with np.errstate(all="ignore"), _quiet():
        prof = np.nanmean(vals.reshape(o.size, n), axis=0)
    return t, prof


def axial_profile(arr, index: tuple, axis: int, cx: float, cy: float, half: int) -> np.ndarray:
    """Mean of a (2*half+1)² box around (cx, cy) in every slice along *axis*."""
    h, w = arr.shape[-2:]
    ix, iy = int(math.floor(cx)), int(math.floor(cy))
    xs = slice(max(0, ix - half), min(w, ix + half + 1))
    ys = slice(max(0, iy - half), min(h, iy + half + 1))
    idx = list(index)
    out = np.empty(arr.shape[axis])
    for k in range(arr.shape[axis]):
        idx[axis] = k
        out[k] = float(np.asarray(arr[tuple(idx) + (ys, xs)], dtype=np.float64).mean())
    return out


def refine_center(frame: np.ndarray, x: float, y: float, radius: int = 4) -> tuple[float, float]:
    """Brightest pixel within *radius* of (x, y), refined by the 5x5 intensity centroid."""
    h, w = frame.shape
    ix, iy = int(math.floor(x)), int(math.floor(y))
    x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
    win = np.asarray(frame[y0:y1, x0:x1], dtype=np.float64)
    if win.size == 0:
        return x, y
    j = np.unravel_index(int(np.argmax(win)), win.shape)
    py, px = y0 + j[0], x0 + j[1]
    b0, b1, a0, a1 = max(0, py - 2), min(h, py + 3), max(0, px - 2), min(w, px + 3)
    sub = np.asarray(frame[b0:b1, a0:a1], dtype=np.float64)
    sub = sub - sub.min()
    total = sub.sum()
    if total <= 0:
        return px + 0.5, py + 0.5
    yy, xx = np.mgrid[b0:b1, a0:a1]
    return float((sub * (xx + 0.5)).sum() / total), float((sub * (yy + 0.5)).sum() / total)


def edge_baseline(prof: np.ndarray, frac: float = 0.15) -> tuple[float, float]:
    """Baseline (median) and noise (MAD sigma) of both ends of a profile."""
    v = prof[np.isfinite(prof)]
    if v.size < 4:
        return (float(np.nanmin(prof)) if v.size else 0.0), float("nan")
    n = max(2, int(v.size * frac))
    ends = np.concatenate([v[:n], v[-n:]])
    med = float(np.median(ends))
    return med, float(1.4826 * np.median(np.abs(ends - med)))


def threshold_width(d: np.ndarray, prof: np.ndarray, baseline: float, level: float) -> dict | None:
    """Width at baseline + level * (peak - baseline), from the crossings next to the peak."""
    if np.count_nonzero(np.isfinite(prof)) < 3:
        return None
    i_pk = int(np.nanargmax(prof))
    peak = float(prof[i_pk])
    if not peak > baseline:
        return None
    thr = baseline + level * (peak - baseline)

    def crossing(indices) -> float:
        prev = i_pk
        for i in indices:
            v = prof[i]
            if not np.isfinite(v):
                return float("nan")
            if v < thr:
                a, b = prof[i], prof[prev]
                return float(d[i] + (thr - a) / (b - a) * (d[prev] - d[i]))
            prev = i
        return float("nan")

    left = crossing(range(i_pk - 1, -1, -1))
    right = crossing(range(i_pk + 1, prof.size))
    return {"peak": peak, "peak_at": float(d[i_pk]), "threshold": thr, "left": left, "right": right,
            "width": right - left}


def _gauss(x, a, m, s, c):
    return a * np.exp(-0.5 * ((x - m) / s) ** 2) + c


def gaussian_fit(d: np.ndarray, prof: np.ndarray) -> dict | None:
    """Least-squares Gaussian a*exp(-(x-m)²/2s²)+c; None if it fails or scipy is missing."""
    if curve_fit is None:
        return None
    ok = np.isfinite(prof)
    x, y = d[ok], prof[ok]
    if x.size < 6:
        return None
    c0 = float(np.percentile(y, 10))
    a0 = float(y.max() - c0)
    m0 = float(x[np.argmax(y)])
    above = x[y > c0 + a0 / 2]
    s0 = max(float(above.max() - above.min()) / 2.3548 if above.size > 1 else 0.0, float(abs(x[1] - x[0])))
    try:
        with np.errstate(all="ignore"), _quiet():
            p, _ = curve_fit(_gauss, x, y, p0=[a0, m0, s0, c0], maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        return None
    a, m, s, c = (float(v) for v in p)
    s = abs(s)
    if not (np.all(np.isfinite(p)) and a > 0 and 0 < s < x.max() - x.min() and x.min() <= m <= x.max()):
        return None
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - _gauss(x, a, m, s, c)) ** 2)) / ss if ss > 0 else float("nan")
    if not r2 >= 0.5:  # a flat or noise-only profile is not a PSF
        return None
    return {"amp": a, "center": m, "sigma": s, "offset": c, "r2": r2}


def gaussian_width(fit: dict, level: float) -> float:
    """Full width of a Gaussian at *level* of its amplitude (level 0.5 -> FWHM)."""
    return 2.0 * fit["sigma"] * math.sqrt(2.0 * math.log(1.0 / level))


class _quiet:
    """Suppresses 'Mean of empty slice' warnings from nanmean."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


@dataclass
class Measurement:
    axis: str
    coord: np.ndarray           # distance (µm or px), z (µm or slices)
    profile: np.ndarray
    baseline: float
    noise: float
    threshold: dict | None
    fit: dict | None
    fit_width: float | None


# -----------------------------------------------------------------------------
# overlays
# -----------------------------------------------------------------------------

class CrossLines(QtWidgets.QGraphicsItem):
    """The two scan lines (x green, y magenta) with their averaging bands."""

    def __init__(self):
        super().__init__()
        self._c = QtCore.QPointF()
        self._len, self._angle, self._half = 10.0, 0.0, 0
        self._rect = QtCore.QRectF()
        self.setZValue(15)

    def set_geometry(self, cx: float, cy: float, length: float, angle: float, half: int) -> None:
        self.prepareGeometryChange()
        self._c = QtCore.QPointF(cx, cy)
        self._len, self._angle, self._half = length, angle, half
        r = length / 2 + half + 2
        self._rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self._rect

    def paint(self, p: QtGui.QPainter, option, widget=None) -> None:
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        c = self._c
        for ang, key in ((self._angle, "x"), (self._angle + 90, "y")):
            a = math.radians(ang)
            u = QtCore.QPointF(math.cos(a), math.sin(a)) * (self._len / 2)
            v = QtCore.QPointF(-math.sin(a), math.cos(a)) * (self._half + 0.5)
            color = QtGui.QColor(COLORS[key])
            if self._half > 0:
                band = QtGui.QColor(color)
                band.setAlpha(45)
                p.setPen(Qt.NoPen)
                p.setBrush(band)
                p.drawPolygon(QtGui.QPolygonF([c - u - v, c + u - v, c + u + v, c - u + v]))
            pen = QtGui.QPen(color, 1.8)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(c - u, c + u)


class BoxROI(QtWidgets.QGraphicsRectItem):
    """Movable rectangle with a resize corner (background region)."""

    def __init__(self, on_change):
        super().__init__(0, 0, 20, 20)
        self._on_change = on_change
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        pen = QtGui.QPen(QtGui.QColor("#ffd400"), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QtGui.QColor(255, 212, 0, 30))
        self.setZValue(14)
        self.setCursor(Qt.SizeAllCursor)
        self.corner = Handle("#ffd400", radius=5, square=True, parent=self)
        self.corner.setPos(20, 20)
        self.corner.moved.connect(self._resize)

    def _resize(self, x: float, y: float) -> None:
        if x < 2 or y < 2:
            self.corner.setPos(max(2.0, x), max(2.0, y))
            return
        self.setRect(QtCore.QRectF(0, 0, x, y))
        self._on_change()

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._on_change()
        return super().itemChange(change, value)

    def place(self, x: float, y: float, w: float, h: float) -> None:
        self.setPos(x, y)
        self.corner.setPos(w, h)

    def region(self) -> tuple[int, int, int, int]:
        r = self.mapRectToScene(self.rect())
        return (int(math.floor(r.left())), int(math.floor(r.top())),
                int(math.ceil(r.right())), int(math.ceil(r.bottom())))


# -----------------------------------------------------------------------------
# panel
# -----------------------------------------------------------------------------

class PSFPanel(QtWidgets.QWidget):
    """PSF analysis tab; its overlays live in the viewer's image while active."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.stack: mr.DataStack | None = None
        self.arr = None
        self.index: tuple = ()
        self.frame: np.ndarray | None = None
        self.center: tuple[float, float] | None = None
        self.results: dict[str, Measurement] = {}
        self._px, self._unit = 1.0, "px"
        self._z_axis, self._z_step, self._z_unit = -1, 1.0, "slice"
        self._placing = False
        self._box_placed = False
        self.active = False

        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel("Click a bead in the image: the cross snaps to its centre. "
                                "Drag the white circle to fine-tune.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(hint)

        grid = QtWidgets.QGridLayout()
        self.length = QtWidgets.QDoubleSpinBox()
        self.length.setRange(0.05, 5000.0)
        self.length.setDecimals(2)
        self.length.setValue(3.0)
        self.length.setToolTip("Length of each scan line")
        self.avg = QtWidgets.QSpinBox()
        self.avg.setRange(0, 25)
        self.avg.setValue(1)
        self.avg.setPrefix("± ")
        self.avg.setSuffix(" px")
        self.avg.setToolTip("Average over parallel lines on both sides (lateral) / box size (axial)")
        self.angle = QtWidgets.QDoubleSpinBox()
        self.angle.setRange(-180.0, 180.0)
        self.angle.setDecimals(1)
        self.angle.setSuffix(" °")
        self.angle.setWrapping(True)
        self.angle.setToolTip("Rotation of the cross (x arm; y is perpendicular)")
        self.level = QtWidgets.QComboBox()
        self.level.addItems([name for name, _ in LEVELS])
        self.baseline = QtWidgets.QComboBox()
        self.baseline.addItems(BASELINES)
        self.baseline.setToolTip("Reference level for the width: median of both profile ends, the mean in a "
                                 "movable box, or zero")
        self.fit = QtWidgets.QCheckBox("Gaussian fit")
        self.fit.setChecked(curve_fit is not None)
        self.fit.setEnabled(curve_fit is not None)
        if curve_fit is None:
            self.fit.setToolTip("needs scipy")
        self.snap = QtWidgets.QCheckBox("Snap to bead")
        self.snap.setChecked(True)
        for r, (label, w, label2, w2) in enumerate((("Scan length", self.length, "Average", self.avg),
                                                    ("Angle", self.angle, "Width at", self.level),
                                                    ("Baseline", self.baseline, None, None))):
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            grid.addWidget(w, r, 1)
            if label2:
                grid.addWidget(QtWidgets.QLabel(label2), r, 2)
                grid.addWidget(w2, r, 3)
        checks = QtWidgets.QHBoxLayout()
        checks.addWidget(self.fit)
        checks.addWidget(self.snap)
        grid.addLayout(checks, 2, 2, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addLayout(grid)

        pg.setConfigOptions(antialias=True, foreground="#c8c8cc")
        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setBackground("#1c1c1f")
        self.plots.setMinimumHeight(260)
        self.lat = self.plots.addPlot(row=0, col=0)
        self.ax = self.plots.addPlot(row=1, col=0)
        self.curves = {}
        for plot, keys in ((self.lat, "xy"), (self.ax, "z")):
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "intensity")
            plot.addLegend(offset=(-6, 6), labelTextSize="8pt")
            for k in keys:
                color = COLORS[k]
                self.curves[k] = {
                    "data": plot.plot(pen=pg.mkPen(color, width=2), name=k),
                    "fit": plot.plot(pen=pg.mkPen(color, width=1, style=Qt.DashLine)),
                    "thr": plot.plot(pen=pg.mkPen("#ffffff", width=1.5), connect="pairs"),
                    "base": plot.plot(pen=pg.mkPen("#8a8a90", width=1, style=Qt.DotLine)),
                }
        lay.addWidget(self.plots, 1)

        self.table = QtWidgets.QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(["", "threshold", "Gaussian"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 130)
        row_h = self.table.fontMetrics().height() + 8
        self.table.verticalHeader().setDefaultSectionSize(row_h)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setFixedHeight(3 * row_h + self.table.horizontalHeader().sizeHint().height() + 4)
        for r, k in enumerate("xyz"):
            item = QtWidgets.QTableWidgetItem(k)
            item.setForeground(QtGui.QBrush(QtGui.QColor(COLORS[k])))
            self.table.setItem(r, 0, item)
        lay.addWidget(self.table)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.info)
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (("Copy results", self.copy_results), ("Save profiles (CSV)…", self.save_csv),
                           ("Save plot…", self.save_plot)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        lay.addLayout(buttons)

        self.lines = CrossLines()
        self.handle = Handle("#ffffff", radius=7)
        self.handle.setToolTip("Drag to move the measurement point")
        self.box = BoxROI(self.update_all)
        for it in (self.lines, self.handle, self.box):
            it.hide()
            view.scene().addItem(it)
        self.handle.moved.connect(self._handle_moved)
        for w in (self.length, self.angle):
            w.valueChanged.connect(self.update_all)
        self.avg.valueChanged.connect(self.update_all)
        for w in (self.level, self.baseline):
            w.currentIndexChanged.connect(self.update_all)
        for w in (self.fit, self.snap):
            w.toggled.connect(self.update_all)

    # -- lifecycle -------------------------------------------------------------

    def activate(self) -> None:
        if not self.active:
            self.view.clicked.connect(self._clicked)
            self.active = True
        self.update_all()

    def deactivate(self) -> None:
        if self.active:
            self.view.clicked.disconnect(self._clicked)
            self.active = False
        for it in (self.lines, self.handle, self.box):
            it.hide()

    def clear(self) -> None:
        self.stack = self.arr = self.frame = None
        self.results = {}
        for it in (self.lines, self.handle, self.box):
            it.hide()

    # -- data from the viewer --------------------------------------------------

    def set_stack(self, stack: mr.DataStack, arr, index: tuple, frame: np.ndarray) -> None:
        self.stack, self.arr, self.index, self.frame = stack, arr, tuple(index), frame
        px = stack.pixel_size[0]
        new_unit = "µm" if px else "px"
        if new_unit != self._unit:
            self.length.blockSignals(True)
            self.length.setValue(3.0 if px else 8.0)
            self.length.blockSignals(False)
        self._px, self._unit = (px or 1.0), new_unit
        self.length.setSuffix(f" {self._unit}")
        self._z_axis = stack.axes[:-2].find("Z")
        self._z_step = stack.step("Z") or 1.0
        self._z_unit = ((stack.axis_info("Z") or {}).get("unit") or "µm") if stack.step("Z") else "slice"
        h, w = frame.shape
        if self.center is None or not (0 <= self.center[0] < w and 0 <= self.center[1] < h):
            self.center = (w / 2.0, h / 2.0)
        if not self._box_placed:
            self.box.place(w * 0.04, h * 0.04, max(4.0, w * 0.12), max(4.0, h * 0.12))
            self._box_placed = True
        self.update_all()

    def set_frame(self, frame: np.ndarray, index: tuple) -> None:
        self.frame, self.index = frame, tuple(index)
        self.update_all()

    # -- interaction -----------------------------------------------------------

    def _clicked(self, x: float, y: float) -> None:
        if self.frame is None:
            return
        h, w = self.frame.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        if self.snap.isChecked():
            x, y = refine_center(self.frame, x, y)
        self.center = (x, y)
        self.update_all()

    def _handle_moved(self, x: float, y: float) -> None:
        if not self._placing:
            self.center = (x, y)
            self.update_all()

    # -- computation -----------------------------------------------------------

    def _background(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box.region()
        h, w = self.frame.shape
        sub = np.asarray(self.frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)], dtype=np.float64)
        if sub.size == 0:
            return 0.0, float("nan")
        return float(sub.mean()), float(sub.std())

    def _measure(self, axis: str, coord: np.ndarray, prof: np.ndarray, level: float) -> Measurement:
        mode = self.baseline.currentIndex()
        if mode == 1:
            base, noise = self._background()
        elif mode == 2:
            base, noise = 0.0, edge_baseline(prof)[1]
        else:
            base, noise = edge_baseline(prof)
        thr = threshold_width(coord, prof, base, level)
        if thr and np.isfinite(noise) and noise > 0 and thr["peak"] - base < 3 * noise:
            thr = None  # no significant peak (less than 3 sigma above the baseline)
        fit = gaussian_fit(coord, prof) if self.fit.isChecked() else None
        return Measurement(axis, coord, prof, base, noise, thr, fit, gaussian_width(fit, level) if fit else None)

    def update_all(self, *_) -> None:
        if not self.active or self.frame is None or self.center is None:
            return
        cx, cy = self.center
        length_px = self.length.value() / self._px
        half, angle = self.avg.value(), self.angle.value()
        level = LEVELS[self.level.currentIndex()][1]
        res = {}
        for axis, ang in (("x", angle), ("y", angle + 90.0)):
            d, prof = sample_line(self.frame, cx, cy, ang, length_px, half)
            res[axis] = self._measure(axis, d * self._px, prof, level)
        if self._z_axis >= 0:
            zp = axial_profile(self.arr, self.index, self._z_axis, cx, cy, half)
            res["z"] = self._measure("z", np.arange(zp.size) * self._z_step, zp, level)
        self.results = res
        self._placing = True
        self.handle.setPos(cx, cy)
        self._placing = False
        self.lines.set_geometry(cx, cy, length_px, angle, half)
        for it in (self.lines, self.handle):
            it.show()
        self.box.setVisible(self.baseline.currentIndex() == 1)
        self._update_plots()
        self._update_table()

    def _update_plots(self) -> None:
        self.lat.setLabel("bottom", f"distance ({self._unit})")
        self.ax.setLabel("bottom", f"z ({self._z_unit})")
        self.ax.setVisible("z" in self.results)
        for k, c in self.curves.items():
            m = self.results.get(k)
            if m is None:
                for item in c.values():
                    item.setData([], [])
                continue
            c["data"].setData(m.coord, m.profile, connect="finite")
            if m.fit:
                xx = np.linspace(m.coord[0], m.coord[-1], 400)
                c["fit"].setData(xx, _gauss(xx, m.fit["amp"], m.fit["center"], m.fit["sigma"], m.fit["offset"]))
            else:
                c["fit"].setData([], [])
            t = m.threshold
            if t and np.isfinite(t["width"]):
                c["thr"].setData([t["left"], t["right"]], [t["threshold"]] * 2)
            else:
                c["thr"].setData([], [])
            c["base"].setData([m.coord[0], m.coord[-1]], [m.baseline] * 2)
        level_name = LEVELS[self.level.currentIndex()][0].split(" ")[0]
        parts = [f"{k} {self._fmt(self.results[k].threshold, k)}" for k in "xy" if k in self.results]
        self.lat.setTitle(f"lateral {level_name}: " + " · ".join(parts), size="9pt")
        if "z" in self.results:
            self.ax.setTitle(f"axial {level_name}: z {self._fmt(self.results['z'].threshold, 'z')}", size="9pt")

    def _fmt(self, t: dict | None, axis: str, width: float | None = None) -> str:
        w = width if width is not None else (t["width"] if t else float("nan"))
        if w is None or not np.isfinite(w):
            return "—"
        unit = self._z_unit if axis == "z" else self._unit
        if unit == "µm" and w < 1:
            return f"{w * 1000:.0f} nm"
        return f"{w:.3g} {unit}"

    def _update_table(self) -> None:
        for r, k in enumerate("xyz"):
            m = self.results.get(k)
            if m is None:
                texts = ("no Z axis" if k == "z" else "—", "")
            else:
                g = self._fmt(None, k, m.fit_width) if m.fit_width is not None else "—"
                if m.fit and np.isfinite(m.fit["r2"]):
                    g += f"   (R² {m.fit['r2']:.3f})"
                texts = (self._fmt(m.threshold, k), g)
            for c, text in enumerate(texts, start=1):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        m = self.results.get("x")
        if m is None or m.threshold is None:
            self.info.setText("no peak above the baseline")
            return
        peak, base, noise = m.threshold["peak"], m.baseline, m.noise
        snr = f" · SNR {(peak - base) / noise:.0f}" if noise and np.isfinite(noise) and noise > 0 else ""
        cx, cy = self.center
        pos = (f"({cx * self._px:.2f}, {cy * self._px:.2f}) µm" if self._unit == "µm"
               else f"({cx:.1f}, {cy:.1f}) px")
        frame = " · frame " + ", ".join(f"{a} {i + 1}" for a, i in zip(self.stack.axes[:-2], self.index)) \
            if self.index else ""
        self.info.setText(f"peak {peak:.4g} · baseline {base:.4g}{snr} · centre {pos}{frame}")

    # -- output ----------------------------------------------------------------

    def summary_rows(self) -> list[list[str]]:
        level = LEVELS[self.level.currentIndex()][0]
        rows = [["stack", f"{self.stack.channel_id} ({self.stack.source}, {self.stack.time})" if self.stack else ""],
                ["width at", level], ["baseline", BASELINES[self.baseline.currentIndex()]],
                ["scan length", f"{self.length.value():g} {self._unit}"], ["average", f"± {self.avg.value()} px"],
                ["angle", f"{self.angle.value():g} °"], ["info", self.info.text()]]
        for k in "xyz":
            m = self.results.get(k)
            if m is not None:
                rows.append([f"{k} threshold", self._fmt(m.threshold, k)])
                rows.append([f"{k} Gaussian", self._fmt(None, k, m.fit_width) if m.fit_width is not None else "—"])
        return rows

    def copy_results(self) -> None:
        QtWidgets.QApplication.clipboard().setText("\n".join("\t".join(r) for r in self.summary_rows()))

    def _default_name(self, ext: str) -> str:
        if self.stack is None:
            return f"psf.{ext}"
        return f"{os.path.splitext(os.path.basename(self.stack.path))[0]}_psf.{ext}"

    def save_csv(self) -> None:
        if not self.results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PSF profiles", self._default_name("csv"),
                                                        "CSV (*.csv)")
        if not path:
            return
        cols = []
        for k in "xyz":
            m = self.results.get(k)
            if m is not None:
                unit = self._z_unit if k == "z" else self._unit
                cols.append((f"z_position_{unit}" if k == "z" else f"{k}_distance_{unit}", m.coord))
                cols.append((f"{k}_intensity", m.profile))
        with open(mr._fs_path(path), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([name for name, _ in cols])
            for row in itertools.zip_longest(*[c for _, c in cols]):
                w.writerow(["" if v is None or not np.isfinite(v) else f"{v:.6g}" for v in row])
            w.writerow([])
            for r in self.summary_rows():
                w.writerow(["#"] + r)

    def save_plot(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PSF plot", self._default_name("png"),
                                                        "PNG (*.png)")
        if path:
            exporter = pg.exporters.ImageExporter(self.plots.scene())
            exporter.parameters()["width"] = max(800, self.plots.width() * 2)
            exporter.export(path)
