#!/usr/bin/env python3
"""
msr_viewer.py -- drag & drop viewer for LaVision BioTec ImSpector (.msr) files.

    python msr_viewer.py [FILE.msr | FILE.tif | FOLDER ...]

Drop .msr or TIFF files or folders onto the window (or onto MSR_Viewer.bat in Explorer).
Parsing and TIFF export are done by msr_reader.py, which must sit next to this file;
TIFF / OME-TIFF / ImageJ files are read by msr_tiff.py into the same kind of stacks.

Playback runs in real time (recorded frame intervals x speed) or at a fixed frame
rate; File > Export video writes the same as MP4 (msr_video.py).  Tools: PSF analysis
(msr_psf.py) and line scan / correlation across stacks (msr_linescan.py), on demand.
Mosaics (tile scans) appear as one more stack, fused from their tiles at the stage
positions or stitched (msr_mosaic.py, Mosaic tab: msr_stitch.py).  View > 3D view shows
z-stacks and 3D mosaics as a volume (msr_volume*.py, after the Fiji plugin 3Dscript;
OpenGL, loaded on demand).

Mouse: wheel = zoom, drag = pan, double-click = fit.
Keys:  Left/Right = previous/next frame, Home/End = first/last frame, Space = play
       (while the image has focus); F = fit, 1 = 100 %, B = scale bar, 3 = 3D view.
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
import msr_mosaic as mm  # noqa: E402
import msr_reader as mr  # noqa: E402
import msr_stitch as ms  # noqa: E402
import msr_tiff as mt  # noqa: E402
import msr_video as mv  # noqa: E402
from msr_widgets import (COLOR_TABLES, COMPOSITE_COLORS, LUTS, ElidedLabel, ImageView,  # noqa: E402
                         IntensityMapper, composite_rgb, numpy_to_qimage, paint_overlays, render_rgb,
                         rgb_to_qimage)

APP_NAME = "MSR Viewer"
__version__ = "0.6.0"
# LUTs that run from black to one colour: a 3D channel takes that colour (the others become its lookup table)
RAMP_LUTS = ("Gray", "Green", "Magenta", "Red", "Cyan", "Yellow")
VOLUME_HINT = "3D view · drag: rotate · Shift+drag: move · wheel: zoom"
KEEP_VOLUME_BYTES = 2 ** 30        # a 3D volume up to this size stays loaded while 3D is off


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _specified(v) -> bool:
    return v not in (None, "", "not specified", "not_specified")


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
    # first frame, but the middle of a z-stack (the first slice of a bead stack is empty)
    frame = np.asarray(arr[tuple(n // 2 if a == "Z" else 0 for a, n in zip(stack.axes[:-2], lead))])
    h, w = frame.shape
    step = max(1, int(math.ceil(max(h, w) / (2 * size))))
    f = frame[::step, ::step].astype(np.float32)
    lo, hi = _percentiles(f)
    a8 = np.clip((f - lo) * (255.0 / (hi - lo)) + 0.5, 0, 255).astype(np.uint8)
    pix = QtGui.QPixmap.fromImage(numpy_to_qimage(a8, COLOR_TABLES["Gray"]))
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _stack_label(pos: int, s: mr.DataStack) -> str:
    if isinstance(s, mm.MosaicStack):
        return s.mosaic.title
    return f"S{pos + 1}  {s.channel_id.split(':')[0] or s.source}"


def _file_part(pos: int, s: mr.DataStack) -> str:
    """Stack name for file names: 'S3_PMT' or 'mosaic_PMT_0_4_stitched'."""
    if isinstance(s, mm.MosaicStack):
        m = s.mosaic
        return f"mosaic_{mr._safe(m.detector)}_{'stitched' if m.layout == 'aligned' else 'stage'}"
    return f"S{pos + 1}_{mr._safe(s.channel_id.split(':')[0] or s.source)}"


def _frozen(s: mr.DataStack) -> mr.DataStack:
    """For work in another thread: a mosaic fixed to its current layout (other stacks as they are)."""
    return s.mosaic.frozen_stack() if isinstance(s, mm.MosaicStack) else s


def _add_mosaics(msr) -> list:
    """Append a fused stack for every mosaic of an .msr file to its stacks."""
    if not isinstance(msr, mr.MSRFile):
        return []
    try:
        mosaics = mm.find_mosaics(msr)
    except Exception as exc:  # a broken mosaic must not keep the file from opening
        msr._warn(f"mosaic not assembled: {type(exc).__name__}: {exc}")
        return []
    for m in mosaics:
        msr.stacks.append(m.stack)
    return mosaics


def _composite_icon() -> QtGui.QIcon:
    """Three overlapping colour disks, as the merge button in ImSpector."""
    pix = QtGui.QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
    p.setPen(Qt.NoPen)
    for color, x, y in (("#ff3030", 11, 12), ("#30ff30", 21, 12), ("#3070ff", 16, 21)):
        p.setBrush(QtGui.QColor(color))
        p.drawEllipse(QtCore.QPointF(x, y), 8.5, 8.5)
    p.end()
    return QtGui.QIcon(pix)


def _volume_icon() -> QtGui.QIcon:
    """A wire-frame cube, for the 3D view."""
    pix = QtGui.QPixmap(32, 32)
    pix.fill(Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    front = QtCore.QRectF(5.5, 11.5, 16, 16)
    back = front.translated(6, -6)
    p.setPen(QtGui.QPen(QtGui.QColor(120, 170, 255), 1.6))
    p.drawRect(back)
    for a, b in ((front.topLeft(), back.topLeft()), (front.topRight(), back.topRight()),
                 (front.bottomRight(), back.bottomRight())):
        p.drawLine(a, b)
    p.setPen(QtGui.QPen(QtGui.QColor(225, 225, 230), 2.0))
    p.setBrush(QtGui.QColor(120, 170, 255, 70))
    p.drawRect(front)
    p.end()
    return QtGui.QIcon(pix)


def _z_planes(s: mr.DataStack) -> int:
    """Number of Z planes of a stack (0 without a Z axis)."""
    zi = s.axes[:-2].find("Z")
    return int(s.shape[zi]) if zi >= 0 else 0


def _swatch(color: QtGui.QColor, w: int = 30, h: int = 14) -> QtGui.QIcon:
    pix = QtGui.QPixmap(w, h)
    pix.fill(color)
    return QtGui.QIcon(pix)


def _level(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.4g}"


def _stack_tooltip(s: mr.DataStack) -> str:
    px = s.pixel_size[0]
    ident = f"{s.channel_id}  ({s.source})" if s.channel_id else s.source
    return f"{ident}\naxes {s.axes}, {_shape_text(s)}, {s.dtype}" + (f"\n{px:.4g} µm/px" if px else "")


def _openable(path: str) -> bool:
    return path.lower().endswith((".msr",) + mt.TIFF_EXTENSIONS)


def _open_file(path: str):
    """MSRFile for .msr, TIFFFile for .tif / .tiff; any other file as whichever reads it."""
    if mt.is_tiff(path):
        return mt.TIFFFile(path)
    try:
        return mr.MSRFile(path)
    except mr.MSRFormatError:
        if path.lower().endswith(".msr"):
            raise
    return mt.TIFFFile(path)


def _shape_text(s: mr.DataStack) -> str:
    return " × ".join(str(n) for n in s.shape)


def _app_icon() -> QtGui.QIcon:
    """Window and taskbar icon, 16-256 px; the same file is the exe icon (msr_viewer.spec)."""
    return QtGui.QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "msr_viewer.ico"))


def _resource(name: str) -> str | None:
    """Path of a file shipped next to msr_viewer.py (the app bundle's resources when frozen), None if missing."""
    dirs = [os.path.dirname(os.path.abspath(__file__)), getattr(sys, "_MEIPASS", "")]
    if getattr(sys, "frozen", False):  # macOS .app: data files are in Contents/Resources
        dirs.append(os.path.join(os.path.dirname(sys.executable), os.pardir, "Resources"))
    return next((os.path.join(d, name) for d in dirs if d and os.path.isfile(os.path.join(d, name))), None)


# -----------------------------------------------------------------------------
# widgets
# -----------------------------------------------------------------------------

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
            icon = QtGui.QPixmap.fromImage(numpy_to_qimage(np.tile(np.arange(256, dtype=np.uint8), (12, 1)),
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


def _check_action(menu: QtWidgets.QMenu, text: str, icon: QtGui.QIcon, data) -> QtWidgets.QWidgetAction:
    """A tick box with a colour swatch in *menu*: it stays open while boxes are ticked (several in one go),
    and the tick shows next to the colour in every style (a checkable action shows only its icon)."""
    box = QtWidgets.QCheckBox(text)
    box.setIcon(icon)
    box.setIconSize(QtCore.QSize(20, 12))
    holder = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(holder)
    lay.setContentsMargins(10, 3, 16, 3)
    lay.addWidget(box)
    a = QtWidgets.QWidgetAction(menu)
    a.setDefaultWidget(holder)
    a.setData(data)
    a.setProperty("box", box)
    menu.addAction(a)
    return a


class CompositePanel(QtWidgets.QGroupBox):
    """Channels of the composite: on / off, colour and levels; a click picks the channel Min / Max edit."""

    toggled = QtCore.pyqtSignal(int, bool)            # stack position in the file, on
    colorChanged = QtCore.pyqtSignal(int, QtGui.QColor)
    activated = QtCore.pyqtSignal(int)
    autoAll = QtCore.pyqtSignal()
    COLORS = (("Red", "#ff0000"), ("Green", "#00ff00"), ("Blue", "#0000ff"), ("Cyan", "#00ffff"),
              ("Magenta", "#ff00ff"), ("Yellow", "#ffff00"), ("Gray", "#ffffff"))

    def __init__(self, parent=None):
        super().__init__("Composite", parent)
        self._positions: list[int] = []
        lay = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Colour", "Levels"])
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        b_auto = QtWidgets.QPushButton("Auto all")
        b_auto.setToolTip("0.1 – 99.9 % of the current frame, for every channel")
        b_auto.clicked.connect(self.autoAll)
        hint = QtWidgets.QLabel("Ticked channels are merged, in 2D and 3D. Min / Max below set the levels of "
                                "the selected channel.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9a9a9a;")
        row.addWidget(b_auto)
        row.addWidget(hint, 1)
        lay.addLayout(row)
        self.table.itemChanged.connect(self._on_item)
        self.table.cellClicked.connect(lambda r, c: self.activated.emit(self._positions[r]) if c != 1 else None)

    def set_channels(self, rows: list[dict], active: int) -> None:
        """rows: pos, name, color (QColor), on, lo, hi; *active* is the position being edited."""
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 1)
            if w is not None:
                w.hide()  # replaced below; deleteLater only acts once control returns to the event loop
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        self._positions = [r["pos"] for r in rows]
        bold = QtGui.QFont(self.table.font())
        bold.setBold(True)
        for i, r in enumerate(rows):
            it = QtWidgets.QTableWidgetItem(r["name"])
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r["on"] else Qt.Unchecked)
            if r["pos"] == active:
                it.setFont(bold)
            self.table.setItem(i, 0, it)
            self.table.setCellWidget(i, 1, self._color_button(r["pos"], r["color"]))
            levels = QtWidgets.QTableWidgetItem(f"{_level(r['lo'])} – {_level(r['hi'])}")
            levels.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(i, 2, levels)
        if active in self._positions:
            self.table.selectRow(self._positions.index(active))
        self.table.blockSignals(False)
        rows_shown = min(len(rows), 6)
        self.table.setFixedHeight(self.table.horizontalHeader().height() + 24 * rows_shown + 4)

    def set_levels(self, pos: int, lo: float, hi: float) -> None:
        if pos in self._positions:
            self.table.item(self._positions.index(pos), 2).setText(f"{_level(lo)} – {_level(hi)}")

    def _color_button(self, pos: int, color: QtGui.QColor) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setIcon(_swatch(color))
        b.setIconSize(QtCore.QSize(30, 14))
        b.setAutoRaise(True)
        b.setToolTip("Colour of this channel")
        b.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(b)
        for name, hexcol in self.COLORS:
            menu.addAction(_swatch(QtGui.QColor(hexcol), 20, 12), name,
                           lambda c=hexcol: self.colorChanged.emit(pos, QtGui.QColor(c)))
        menu.addSeparator()
        menu.addAction("Other…", lambda: self._pick(pos, color))
        b.setMenu(menu)
        return b

    def _pick(self, pos: int, color: QtGui.QColor) -> None:
        c = QtWidgets.QColorDialog.getColor(color, self, "Channel colour")
        if c.isValid():
            self.colorChanged.emit(pos, c)

    def _on_item(self, it: QtWidgets.QTableWidgetItem) -> None:
        if it.column() == 0:
            self.toggled.emit(self._positions[it.row()], it.checkState() == Qt.Checked)


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
    """One slider per non-image axis (T, Z, ...), with playback on T.

    Playback runs at a fixed frame rate or in real time: every frame stays on
    screen for its actual acquisition interval (per-frame time stamps), scaled
    by a speed factor.  Real time follows the wall clock, so frames are skipped
    when drawing cannot keep up instead of the movie slowing down.
    """

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
        self.timer.timeout.connect(self._tick)
        self.clock = QtCore.QElapsedTimer()
        self._times: np.ndarray | None = None  # acquisition time of each frame on the play axis
        self._anchor = 0.0                      # data time at clock start
        self._ticking = False                   # slider moved by playback, not by the user
        self._preferred_mode = 0                # what the user picked (0 real time, 1 fixed fps)
        self._configuring = False
        self.play_btn = QtWidgets.QPushButton()
        self.play_btn.setMinimumWidth(84)
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["real time", "fixed fps"])
        self.mode.setToolTip("Real time: frames are held for their actual acquisition interval (× speed)\n"
                             "Fixed fps: one frame per tick")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Frames per second in fixed-fps mode")
        self.fps.valueChanged.connect(self._restart_if_playing)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.01, 100.0)
        self.speed.setDecimals(2)
        self.speed.setSingleStep(0.25)
        self.speed.setValue(1.0)
        self.speed.setSuffix(" ×")
        self.speed.setToolTip("Speed in real-time mode: 1 = as recorded, 0.25 = four times slower")
        self.speed.valueChanged.connect(self._restart_if_playing)
        self.hide()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    @property
    def real_time(self) -> bool:
        return self.mode.currentIndex() == 0 and self._times is not None

    def configure(self, stack: mr.DataStack, axes: str, shape: tuple) -> None:
        self.stop()
        for r in self.rows:
            for w in (r["label"], r["slider"], r["spin"], r["info"]):
                self.grid.removeWidget(w)
                w.hide()  # deleteLater only acts once control returns to the event loop
                w.deleteLater()
        for w in (self.play_btn, self.mode, self.fps, self.speed):
            self.grid.removeWidget(w)
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
        self._times = None
        if self.rows:
            self.play_row = next((i for i, r in enumerate(self.rows) if r["letter"] == "T"), 0)
            row = self.rows[self.play_row]
            self._times = mv.frame_times(stack, row["letter"], row["n"])
            for col, w in enumerate((self.play_btn, self.mode, self.fps, self.speed), start=4):
                self.grid.addWidget(w, self.play_row, min(col, 6))
            dt = mv.typical_interval(self._times) if self._times is not None else None
            self.fps.blockSignals(True)
            self.fps.setValue(int(min(60, max(1, round(1.0 / dt)))) if dt else 10)
            self.fps.blockSignals(False)
            for w in (self.play_btn, self.fps, self.speed):  # set_axis_enabled may have turned them off (3D)
                w.setEnabled(True)
            self._configuring = True
            self.mode.setEnabled(self._times is not None)
            self.mode.setCurrentIndex(self._preferred_mode if self._times is not None else 1)
            self._configuring = False
        self._mode_widgets()
        self._set_icon()
        self._update_info()
        self.setVisible(bool(self.rows))

    def index(self) -> tuple:
        return tuple(r["slider"].value() for r in self.rows)

    def set_axis_enabled(self, letter: str, on: bool) -> None:
        """Enable or disable the row of axis *letter* (the 3D view shows every Z plane at once)."""
        for i, r in enumerate(self.rows):
            if r["letter"] != letter or r["slider"].isEnabled() == on:
                continue
            for w in (r["label"], r["slider"], r["spin"], r["info"]):
                w.setEnabled(on)
            r["slider"].setToolTip("" if on else f"The 3D view shows all {r['n']} {letter} planes")
            if i == self.play_row:  # playback steps this axis
                if not on:
                    self.stop()
                for w in (self.play_btn, self.fps, self.speed):
                    w.setEnabled(on)
                self.mode.setEnabled(on and self._times is not None)

    def can_step(self) -> bool:
        """Playback and frame steps are possible: there is a play axis and its row is enabled."""
        return bool(self.rows) and self.rows[self.play_row]["slider"].isEnabled()

    def set_index(self, index: tuple) -> None:
        """Move the sliders without signals (another stack of the same shape keeps the frame)."""
        for r, v in zip(self.rows, index):
            for w, value in ((r["slider"], v), (r["spin"], v + 1)):
                w.blockSignals(True)
                w.setValue(value)
                w.blockSignals(False)
        self._update_info()

    def _on_slider(self, i: int, v: int) -> None:
        spin = self.rows[i]["spin"]
        spin.blockSignals(True)
        spin.setValue(v + 1)
        spin.blockSignals(False)
        self._update_info(i)
        if self.playing and self.real_time and i == self.play_row and not self._ticking:
            self._reanchor()  # the user moved the slider during playback
        self.indexChanged.emit()

    def step(self, delta: int) -> None:
        if self.can_step():
            r = self.rows[self.play_row]
            r["slider"].setValue((r["slider"].value() + delta) % r["n"])

    def jump(self, where: int) -> None:
        if self.can_step():
            r = self.rows[self.play_row]
            r["slider"].setValue(0 if where == 0 else r["n"] - 1)

    def toggle_play(self) -> None:
        if not self.can_step():
            return
        if self.timer.isActive():
            self.stop()
        else:
            self._start()

    def _start(self) -> None:
        if self.real_time:
            self._reanchor()
            dt = (mv.typical_interval(self._times) or 0.05) / self.speed.value()
            self.timer.start(int(max(2, min(20, dt * 1000 / 4))))  # poll a few times per frame
        else:
            self.timer.start(max(1, round(1000 / self.fps.value())))
        self._set_icon()

    def stop(self) -> None:
        self.timer.stop()
        self._set_icon()

    def _restart_if_playing(self, *_) -> None:
        if self.playing:
            self.timer.stop()
            self._start()

    def _mode_changed(self, *_) -> None:
        if not self._configuring:
            self._preferred_mode = self.mode.currentIndex()
        self._mode_widgets()
        self._restart_if_playing()

    def _mode_widgets(self) -> None:
        real = self.mode.currentIndex() == 0 and self._times is not None
        self.speed.setVisible(real and bool(self.rows))
        self.fps.setVisible(not real and bool(self.rows))

    def _reanchor(self) -> None:
        r = self.rows[self.play_row]
        self._anchor = float(self._times[r["slider"].value()])
        self.clock.start()

    def _tick(self) -> None:
        if not self.real_time:
            self.step(1)
            return
        r = self.rows[self.play_row]
        ts = self._times
        t = self._anchor + self.clock.elapsed() / 1000.0 * self.speed.value()
        if t >= ts[-1] + (mv.typical_interval(ts) or 0.0):  # past the last frame: loop
            self._anchor = float(ts[0])
            self.clock.start()
            i = 0
        else:
            i = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
        if i != r["slider"].value():
            self._ticking = True
            r["slider"].setValue(i)
            self._ticking = False

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
        kind = f"{msr.kind} · " if isinstance(msr, mt.TIFFFile) else ""
        self.subtitle.setText(f"{kind}{len(msr.stacks)} stacks{' · workspace ' + ws if ws else ''} · "
                              "click a stack to open it")
        self.list.clear()
        for pos, s in enumerate(msr.stacks):
            px = s.pixel_size[0]
            extra = " · ".join(x for x in (s.time, f"{px:.3g} µm/px" if px else "") if x)
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(thumbs[pos]), f"{_stack_label(pos, s)}\n{_shape_text(s)}  "
                                                                     f"({s.axes})\n{extra}")
            it.setData(Qt.UserRole, pos)
            modes = " · ".join(x for x in (s.meta.get("Instrument Mode"), s.meta.get("Measurement Mode")) if x)
            it.setToolTip("\n".join(x for x in (s.channel_id, modes) if x) or s.source)
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
        p.drawText(r.adjusted(0, -40, 0, -40), Qt.AlignCenter, "Drop .msr or TIFF files here")
        font.setPointSize(10)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QtGui.QColor(140, 140, 146))
        p.drawText(r.adjusted(0, 50, 0, 50), Qt.AlignCenter,
                   "or File ▸ Open… (Ctrl+O)  ·  files or whole folders\n"
                   "LaVision BioTec ImSpector workspaces: every stack, source and setting\n"
                   "TIFF, OME-TIFF and ImageJ stacks for the same analyses")


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
        p.drawText(self.rect().adjusted(0, 70, 0, 70), Qt.AlignCenter, ".msr / TIFF files or folders")


class CalibrationDialog(QtWidgets.QDialog):
    """Pixel size, z step, frame interval and frame meaning for TIFF stacks (Tools ▸ Calibration)."""

    def __init__(self, stack: mt.TIFFStack, n_stacks: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.stack = stack
        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        def spin(value, suffix: str, decimals: int) -> QtWidgets.QDoubleSpinBox:
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(0.0, 1e6)
            sb.setDecimals(decimals)
            sb.setSuffix(suffix)
            sb.setSpecialValueText("unknown")
            sb.setValue(value or 0.0)
            return sb

        self.pixel = spin(stack.pixel_size[0], " µm", 5)
        form.addRow("Pixel size", self.pixel)
        self.frames = QtWidgets.QComboBox()
        self.frames.addItems(["Z slices", "time points"])
        self.frames.setCurrentIndex(1 if stack.assumed == "T" else 0)
        if stack.assumed:
            other = "T" if stack.assumed == "Z" else "Z"
            self.frames.setEnabled(other not in stack.axes)
            form.addRow("Frames are", self.frames)
        self.z_step = spin(stack.step("Z"), " µm", 5)
        self.interval = spin(stack.time_increment if "T" in stack.axes else None, " s", 6)
        if stack.timestamps:
            self.interval.setEnabled(False)
            self.interval.setToolTip("from the per-frame times in the file")
        self.z_label, self.t_label = QtWidgets.QLabel("Z step"), QtWidgets.QLabel("Frame interval")
        form.addRow(self.z_label, self.z_step)
        form.addRow(self.t_label, self.interval)
        self.all = QtWidgets.QCheckBox(f"Apply to all {n_stacks} stacks of this file")
        self.all.setChecked(True)
        self.all.setVisible(n_stacks > 1)
        lay.addWidget(self.all)
        note = QtWidgets.QLabel("Used until the file is closed, not written into it. File ▸ Save current stack "
                                "(OME-TIFF) keeps it.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.frames.currentIndexChanged.connect(self._rows)
        self._rows()

    def _letters(self) -> set:
        letters = set(self.stack.axes[:-2])
        if self.stack.assumed:
            letters.discard(self.stack.assumed)
            letters.add("ZT"[self.frames.currentIndex()])
        return letters

    def _rows(self) -> None:
        letters = self._letters()
        for label, w, letter in ((self.z_label, self.z_step, "Z"), (self.t_label, self.interval, "T")):
            label.setVisible(letter in letters)
            w.setVisible(letter in letters)

    def values(self) -> dict:
        """Keyword arguments for TIFFStack.set_calibration."""
        letters = self._letters()
        return {"pixel_size": self.pixel.value(),
                "frames": "ZT"[self.frames.currentIndex()] if self.stack.assumed else None,
                "z_step": self.z_step.value() if "Z" in letters else None,
                "frame_interval": self.interval.value() if "T" in letters and self.interval.isEnabled() else None}


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
        self.mapper = IntensityMapper()
        self._hist_skip = 0
        self.tasks: set[Task] = set()
        self.psf = None  # msr_psf.PSFPanel, created when first switched on
        self.comp_colors: dict[tuple, QtGui.QColor] = {}   # composite colour and on / off per (file, stack)
        self.comp_on: dict[tuple, bool] = {}
        self._comp_arrays: dict[tuple, np.ndarray] = {}
        self._comp_mappers: dict[tuple, IntensityMapper] = {}
        self.linescan_windows: list = []
        self.mosaics: dict[str, list] = {}       # file key -> msr_mosaic.Mosaic objects
        self._outline_items: list = []
        self._layout_versions: dict[int, int] = {}   # id(mosaic) -> number of layout changes (3D reload key)
        # 3D view (msr_volume_view / msr_volume_panel, created when it is first switched on)
        self.volume_view = None
        self.volume_panel = None
        self.volume_scroll: QtWidgets.QScrollArea | None = None
        self._vol_loader = None                  # VolumeLoadThread of the volume being loaded
        self._vol_threads: set = set()           # loaders still running (cancelled ones finish on their own)
        self._vol_key = None                     # load key of the volume in the 3D view
        self._vol_skip_key = None                # load key that failed or was cancelled: not retried by itself
        self._vol_channels: dict[tuple, object] = {}   # (file, stack) -> its 3D ChannelSettings, kept while open
        self._vol_looks: dict[tuple, tuple] = {}       # (file, stack) -> (colour, lut) last taken from the 2D view
        self._vol_resolution = "auto"            # resolution mode asked for (the 3D tab's Resolution)
        self._vol_mode = None                    # ... of the volume in the 3D view
        self._vol_load_mode = "auto"             # ... of the volume being loaded
        self._vol_gpu = ""                       # renderer of the 3D view (or why OpenGL failed)
        self._vol_gl_ok = True
        self._vol_load_text = ""                 # "Loading <stack>…"
        self._vol_load_message = False           # show the loading progress over the 3D view
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
        self.frames.indexChanged.connect(self._index_changed)
        self.view.stepRequested.connect(self.frames.step)
        self.view.jumpRequested.connect(self.frames.jump)
        self.view.playRequested.connect(self.frames.toggle_play)
        self.view_stack = QtWidgets.QStackedWidget()   # the 2D view, and the 3D view once it is used
        self.view_stack.addWidget(self.view)
        self.viewer_page = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(self.viewer_page)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(self.view_stack, 1)
        vl.addWidget(self.frames)
        self.overview = OverviewPage()
        self.overview.stackActivated.connect(self._select_stack)
        self.welcome = WelcomePage()
        self.center = QtWidgets.QStackedWidget()
        for page in (self.welcome, self.overview, self.viewer_page):
            self.center.addWidget(page)

        self.display = DisplayPanel()
        self.display.changed.connect(self._display_changed)
        self.display.autoRequested.connect(self._auto_contrast)
        self.composite = CompositePanel()
        self.composite.hide()
        self.composite.toggled.connect(self._composite_toggled)
        self.composite.colorChanged.connect(self._composite_color)
        self.composite.activated.connect(self._select_stack)
        self.composite.autoAll.connect(self._composite_auto_all)
        self.info = InfoPanel()
        self.settings_panel = SettingsPanel()
        self.mosaic_panel = ms.MosaicPanel()
        self.mosaic_panel.changed.connect(self._mosaic_changed)
        self.mosaic_panel.outlinesToggled.connect(lambda _: self._update_outlines())
        self.mosaic_panel.taskStarted.connect(self._track_task)
        self.mosaic_scroll = QtWidgets.QScrollArea()
        self.mosaic_scroll.setWidgetResizable(True)
        self.mosaic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mosaic_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mosaic_scroll.setWidget(self.mosaic_panel)
        self.display.setEnabled(False)
        display_tab = QtWidgets.QWidget()
        dl = QtWidgets.QVBoxLayout(display_tab)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.composite)
        dl.addWidget(self.display, 1)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(display_tab, "Display")
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
                          "Open .msr or TIFF files (or drop them onto the window)")
        self.a_close = act("&Close file", self.close_current_file, "Ctrl+W", S.SP_DialogCloseButton)
        self.a_export = act("Export all stacks as &OME-TIFF…", lambda: self.export_file(False), "Ctrl+E",
                            S.SP_DialogSaveButton, "Export every stack of the file (OME-TIFF + metadata.json)")
        self.a_export_ij = act("Export all stacks as &ImageJ TIFF…", lambda: self.export_file(True))
        self.a_save_stack = act("&Save current stack as…", self.save_stack, QtGui.QKeySequence.Save,
                                tip="Save the displayed stack as OME-TIFF or ImageJ TIFF")
        self.a_save_png = act("Save view as &PNG…", self.save_png, "Ctrl+Shift+S",
                              tip="Current frame with LUT (and scale bar) at full resolution")
        self.a_export_ts = act("Export frame &times as text…", self.export_timestamps, "Ctrl+T",
                               tip="Acquisition time of every frame of the current stack, one value (s) per line")
        self.a_video = act("Export &video (MP4)…", self.export_video, "Ctrl+Shift+V",
                           tip="Timelapse or z sweep as MP4: real acquisition timing (× speed) or a fixed frame rate")
        self.a_quit = act("&Quit", self.close, QtGui.QKeySequence.Quit)
        self.a_fit = act("&Fit to window", self._fit, "F", tip="Fit (F, or double-click the image)")
        self.a_1to1 = act("&Actual pixels", self.view.zoom_1to1, "1", tip="100 %: one image pixel per screen pixel (1)")
        self.a_scalebar = act("Scale &bar", self._toggle_scalebar, "B", tip="Show scale bar (B)")
        self.a_scalebar.setCheckable(True)
        self.a_scalebar.setChecked(True)
        self.a_composite = QtWidgets.QAction(_composite_icon(), "&Composite", self)
        self.a_composite.setCheckable(True)
        self.a_composite.setShortcut(QtGui.QKeySequence("C"))
        self.a_composite.setToolTip("Merge the channels of the file that have the same size, each with its own colour "
                                    "and levels (C), in 2D and 3D.\nThe arrow ▾ chooses the channels.")
        self.a_composite.setStatusTip("Merge the channels of the file that have the same size (C), in 2D and 3D; "
                                      "the arrow chooses the channels")
        self.a_composite.toggled.connect(self._toggle_composite)
        # which channels are merged: the arrow of the Composite button, View ▸ Composite channels, the 3D tab
        self.comp_menu = QtWidgets.QMenu("Composite c&hannels", self)
        self.comp_menu.setToolTipsVisible(True)
        self.comp_menu.aboutToShow.connect(self._fill_composite_menu)
        self.a_volume = QtWidgets.QAction(_volume_icon(), "&3D view", self)
        self.a_volume.setCheckable(True)
        self.a_volume.setShortcut(QtGui.QKeySequence("3"))
        self.a_volume.setToolTip("Show z-stacks and 3D mosaics as a volume: drag rotates, Shift+drag moves, "
                                 "wheel zooms (3)")
        self.a_volume.setStatusTip(self.a_volume.toolTip())
        self.a_volume.toggled.connect(self.toggle_volume)
        self.a_play = act("&Play / pause", self.frames.toggle_play, "P")
        self.a_prev = act("Previous frame", lambda: self.frames.step(-1), ",")
        self.a_next = act("Next frame", lambda: self.frames.step(1), ".")
        self.a_about = act("&About", self.about)
        self.a_notices = act("&Third-party notices…", self.show_notices,
                             tip="Licenses of code the viewer is based on (3Dscript for the 3D view)")
        self.a_psf = QtWidgets.QAction("&PSF analysis", self)
        self.a_psf.setCheckable(True)
        self.a_psf.setShortcut(QtGui.QKeySequence("Ctrl+Shift+P"))
        self.a_psf.setToolTip("Measure bead widths (FWHM, 1/e): click a bead, lateral x/y and axial z profiles")
        self.a_psf.toggled.connect(self.toggle_psf)
        self.a_linescan = act("&Line scan / correlation…", self.open_linescan, "Ctrl+Shift+L",
                              tip="Profiles of several stacks along a line, alignment and correlation (own window)")
        self.a_calibrate = act("&Calibration…", self.edit_calibration,
                               tip="Pixel size, z step and frame interval of a TIFF stack that lacks them")
        self.a_mosaic_tiff = act("Save TIFF…", self.save_stack,
                                 tip="The fused mosaic as shown (layout and blending) as OME-TIFF or ImageJ TIFF")
        self.a_mosaic_video = act("Video…", self.export_video,
                                  tip="Z sweep (or timelapse) of the fused mosaic as MP4")
        self.a_mosaic_png = act("PNG…", self._save_2d_png,   # the plane also while the 3D view is shown
                                tip="The current plane of the fused mosaic, as shown (PNG)")
        self.mosaic_panel.add_output_actions([self.a_mosaic_tiff, self.a_mosaic_video, self.a_mosaic_png])

        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.a_open, None, self.a_export, self.a_export_ij, self.a_save_stack, self.a_save_png,
                  self.a_video, self.a_export_ts, None, self.a_close, self.a_quit):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&View")
        for a in (self.a_fit, self.a_1to1, self.a_scalebar, self.a_composite, self.comp_menu.menuAction(),
                  self.a_volume, None, self.a_play, self.a_prev, self.a_next):
            m.addSeparator() if a is None else m.addAction(a)
        self.tools_menu = mb.addMenu("&Tools")
        self.tools_menu.addAction(self.a_psf)
        self.tools_menu.addAction(self.a_linescan)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.a_calibrate)
        m = mb.addMenu("&Help")
        m.addAction(self.a_about)
        m.addAction(self.a_notices)

        tb = self.addToolBar("Main")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        for a in (self.a_open, self.a_export, self.a_video, None, self.a_fit, self.a_1to1, self.a_scalebar,
                  self.a_composite, self.a_volume, None, self.a_psf, self.a_linescan):
            tb.addSeparator() if a is None else tb.addAction(a)
        comp_button = tb.widgetForAction(self.a_composite)   # click: Composite on / off; arrow: its channels
        if isinstance(comp_button, QtWidgets.QToolButton):
            comp_button.setMenu(self.comp_menu)
            comp_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.toolbar = tb

    # -- file handling -------------------------------------------------------------

    def open_dialog(self) -> None:
        start = self.qsettings.value("last_dir", os.path.expanduser("~"))
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open ImSpector or TIFF files", start,
            "ImSpector and TIFF files (*.msr *.tif *.tiff);;ImSpector files (*.msr);;TIFF files (*.tif *.tiff);;"
            "All files (*)")
        if paths:
            self.open_paths(paths)

    def open_paths(self, paths) -> None:
        files = []
        for p in paths:
            if os.path.isdir(mr._fs_path(p)):
                found = sorted(os.path.join(p, f) for f in os.listdir(mr._fs_path(p)) if _openable(f))
                if len(found) > 40 and QtWidgets.QMessageBox.question(
                        self, APP_NAME, f"{len(found)} .msr / TIFF files in\n{p}\n\nOpen all of them?") \
                        != QtWidgets.QMessageBox.Yes:
                    continue
                files += found
            elif os.path.isfile(mr._fs_path(p)):
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
                    msr = _open_file(f)
                    if not msr.stacks:
                        msr.close()
                        raise mr.MSRFormatError("no image data found")
                except Exception as exc:
                    errors.append(f"{os.path.basename(f)}:\n    {exc}")
                    continue
                self.files[key] = msr
                self.mosaics[key] = _add_mosaics(msr)
                last = self._add_file_item(key, msr)
                self.qsettings.setValue("last_dir", os.path.dirname(os.path.abspath(f)))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if last is not None:
            key = last.data(0, Qt.UserRole)[1]
            stacks = self.files[key].stacks
            first = next((i for i, s in enumerate(stacks) if isinstance(s, mm.MosaicStack)), 0)  # a mosaic first
            self.tree.setCurrentItem(last.child(first) if last.childCount() else last)
            self._focus_view()
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
            child = QtWidgets.QTreeWidgetItem([f"{_stack_label(pos, s)}\n{_shape_text(s)}"
                                               + (f"  ·  {s.time}" if s.time else "")])
            child.setIcon(0, QtGui.QIcon(thumb))
            child.setData(0, Qt.UserRole, ("stack", key, pos))
            child.setToolTip(0, _stack_tooltip(s))
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

    def _exporting(self) -> bool:
        """An export (not just a registration) is running."""
        reg = self.mosaic_panel.thread
        return any(t is not reg for t in self.tasks)

    def _stop_registration(self) -> None:
        """Cancel a running registration and wait for it (it stops after the current overlap)."""
        thread = self.mosaic_panel.thread
        if thread is not None and thread.isRunning():
            thread.cancel()
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                thread.wait(30000)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QApplication.processEvents()

    def close_current_file(self) -> None:
        key = self._current_file_key()
        if key is None:
            return
        if self._exporting():
            QtWidgets.QMessageBox.information(self, APP_NAME, "Please wait until the export has finished.")
            return
        self._stop_registration()
        self._release_volume_of(key)  # before the file closes: loaders read from it
        if self.cur_key and self.cur_key[0] == key:
            self.frames.stop()
            self.frames.configure(self.files[key].stacks[0], "", ())
            self.arr, self.frame, self.cur_key = None, None, None
            if self.psf is not None:
                self.psf.clear()
            self.view.clear()
            self.composite.hide()
        if self.mosaic_panel.mosaic is not None and self.mosaic_panel.mosaic in self.mosaics.get(key, []):
            self._show_mosaic_tab(None)
            self.mosaic_panel.set_mosaic(None)
        self.mosaics.pop(key, None)
        self._update_outlines()
        item = self.file_items.pop(key)
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        self.thumbs.pop(key, None)
        for d in (self.states, self._comp_arrays, self._comp_mappers):
            for k in [k for k in d if k[0] == key]:
                del d[k]
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
            self._focus_view()

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
        self.composite.hide()
        self._show_mosaic_tab(None)
        self._leave_volume(release=True)
        self.pixel_label.setText("")
        self.setWindowTitle(f"{os.path.basename(msr.path)} — {APP_NAME}")

    def _show_stack(self, key: str, pos: int) -> None:
        msr = self.files[key]
        s = msr.stacks[pos]
        if self.cur_key != (key, pos):
            prev = self.files[self.cur_key[0]].stacks[self.cur_key[1]] \
                if self.cur_key and self.cur_key[0] in self.files else None
            same = prev is not None and prev.axes == s.axes and prev.shape == s.shape  # another channel
            index = self.frames.index()
            self.frames.stop()
            self.arr = s.asarray()
            self.cur_key = (key, pos)
            st = self.states.get(self.cur_key)
            if st is None:
                st = self.states[self.cur_key] = _initial_state(self.arr, s)
            self.frames.configure(s, s.axes[:-2], self.arr.shape[:-2])
            if same:
                self.frames.set_index(index)
            self.view.pixel_size = s.pixel_size[0]
            self.view.unit = s.units[0] if s.units and s.units[0] else "µm"
            self.display.set_state(st, np.issubdtype(self.arr.dtype, np.integer))
            self._render_frame(reset=not same, new_stack=True)
        self.center.setCurrentWidget(self.viewer_page)
        self.display.setEnabled(True)
        self._update_composite_panel()
        self._show_mosaic_tab(s.mosaic if isinstance(s, mm.MosaicStack) else None, key)
        self._update_outlines()
        self.info.show_sections(self._stack_sections(msr, s, pos))
        self.settings_panel.set_sources(self._setting_sources(msr, s), msr.labels)
        ident = f"{_stack_label(pos, s)} · {s.channel_id}" if s.channel_id else _stack_label(pos, s)
        self.setWindowTitle(f"{ident} — {os.path.basename(msr.path)} — {APP_NAME}")
        self._update_volume()

    # -- rendering ------------------------------------------------------------------

    def _index_changed(self) -> None:
        """A frame slider moved: the 2D frame, and the 3D volume when another T (or other non-Z) index is shown."""
        self._render_frame()
        if self._volume_shown():
            self._update_volume()

    def _display_changed(self) -> None:
        """LUT or levels of the current stack edited in the Display tab."""
        self._render_frame(histogram=False)
        self._sync_volume_looks()

    def _render_frame(self, reset: bool = False, histogram: bool = True, new_stack: bool | None = None) -> None:
        if self.arr is None or self.cur_key is None:
            return
        idx = self.frames.index()
        self.frame = np.asarray(self.arr[idx] if idx else self.arr)
        st = self.states[self.cur_key]
        members = self._composite_members()
        if members:
            img = rgb_to_qimage(self._composite_rgb(members, idx))
            self.composite.set_levels(self.cur_key[1], st.lo, st.hi)
        else:
            img = numpy_to_qimage(self.mapper.map(self.frame, st.lo, st.hi), COLOR_TABLES[st.lut])
        self.view.set_pixmap(QtGui.QPixmap.fromImage(img), reset)
        self._psf_feed(new_stack=reset if new_stack is None else new_stack)
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
        members = self._composite_members()
        if members:  # every channel of the composite
            stacks, idx, parts = self.files[self.cur_key[0]].stacks, self.frames.index(), []
            for k in members:
                if self.comp_on.get(k, True):
                    c = self._member_array(k)[idx + (iy, ix)]
                    name = stacks[k[1]].channel_id.split(":")[0] or stacks[k[1]].source
                    parts.append(f"{name} {c:.5g}" if isinstance(c, np.floating) else f"{name} {c}")
            text += "   " + "  ·  ".join(parts)
        else:
            text += f"   value {v:.5g}" if isinstance(v, np.floating) else f"   value {v}"
        if isinstance(self.arr, mm.MosaicArray):
            tiles = self.arr.coverage(iy, ix, self.frames.index())
            m = self.arr.mosaic
            text += ("   tile " + " + ".join(m.tile_label(k) for k in tiles)) if tiles else "   (no tile)"
        self.pixel_label.setText(text)

    def _toggle_scalebar(self) -> None:
        if self._volume_shown():  # B acts on the view that is shown
            settings = self.volume_view.settings
            settings.show_scalebar = not settings.show_scalebar   # the 3D tab may have changed it meanwhile
            self.a_scalebar.setChecked(settings.show_scalebar)
            self.volume_panel.refresh()
            self.volume_view.settings_changed()
            return
        self.view.show_scalebar = self.a_scalebar.isChecked()
        self.view.viewport().update()

    def _fit(self) -> None:
        if self._volume_shown():
            self.volume_view.set_preset("fit")
        else:
            self.view.fit()

    def _focus_view(self) -> None:
        """Keyboard focus to the 2D or the 3D view, whichever is shown."""
        (self.volume_view if self._volume_shown() else self.view).setFocus()

    # -- composite --------------------------------------------------------------------

    def _composite_candidates(self) -> list[tuple]:
        """(file, stack) keys that can be merged with the current stack: same file, axes and shape."""
        if self.cur_key is None or self.cur_key[0] not in self.files:
            return []
        key, pos = self.cur_key
        stacks = self.files[key].stacks
        s = stacks[pos]
        return [(key, i) for i, t in enumerate(stacks) if t.axes == s.axes and t.shape == s.shape]

    def _composite_members(self) -> list[tuple]:
        """The merged stacks while Composite is on and there is more than one, else []."""
        if not self.a_composite.isChecked():
            return []
        members = self._composite_candidates()
        return members if len(members) > 1 else []

    def _member_array(self, k: tuple) -> np.ndarray:
        if k == self.cur_key and self.arr is not None:
            return self.arr
        if k not in self._comp_arrays:
            self._comp_arrays[k] = self.files[k[0]].stacks[k[1]].asarray()
        return self._comp_arrays[k]

    def _member_state(self, k: tuple) -> DisplayState:
        if k not in self.states:
            self.states[k] = _initial_state(self._member_array(k), self.files[k[0]].stacks[k[1]])
        return self.states[k]

    def _member_color(self, k: tuple, members: list[tuple]) -> QtGui.QColor:
        if k not in self.comp_colors:
            self.comp_colors[k] = QtGui.QColor(COMPOSITE_COLORS[members.index(k) % len(COMPOSITE_COLORS)])
        return self.comp_colors[k]

    def _composite_layers(self, members: list[tuple], idx: tuple) -> tuple[list, list]:
        layers, mappers = [], []
        for k in members:
            if self.comp_on.get(k, True):
                arr, st = self._member_array(k), self._member_state(k)
                layers.append((np.asarray(arr[idx] if idx else arr), st.lo, st.hi,
                               self._member_color(k, members).getRgb()[:3]))
                mappers.append(self._comp_mappers.setdefault(k, IntensityMapper()))
        return layers, mappers

    def _composite_rgb(self, members: list[tuple], idx: tuple) -> np.ndarray:
        rgb = composite_rgb(*self._composite_layers(members, idx))
        return rgb if rgb is not None else np.zeros(self.frame.shape + (3,), dtype=np.uint8)

    def _update_composite_panel(self) -> None:
        members = self._composite_members() if self.center.currentWidget() is self.viewer_page else []
        self.composite.setVisible(bool(members))
        self.display.lut.setEnabled(not members)
        self.display.lut.setToolTip("Composite: each channel has a colour (above)" if members else "")
        if not members:
            return
        stacks = self.files[self.cur_key[0]].stacks
        rows = []
        for k in members:
            st = self._member_state(k)
            rows.append({"pos": k[1], "name": _stack_label(k[1], stacks[k[1]]), "color": self._member_color(k, members),
                         "on": self.comp_on.get(k, True), "lo": st.lo, "hi": st.hi})
        self.composite.set_channels(rows, self.cur_key[1])

    def _toggle_composite(self, on: bool) -> None:
        if (on and self._composite_members() and self.tabs.currentWidget() is not self.psf
                and not self._volume_shown()):   # in 3D the 3D tab stays: it lists the channels too
            self.tabs.setCurrentIndex(0)  # the channel table is in the Display tab
        if on and self.statusBar().currentMessage().startswith("3D shows the selected channel only"):
            self.statusBar().clearMessage()
        self._update_composite_panel()
        self._render_frame(histogram=False)
        self._update_volume()
        self._update_actions()

    def _composite_toggled(self, pos: int, on: bool) -> None:
        """A tick box of the composite table: channel *pos* merged or not (one always stays)."""
        self._set_composite_channel((self.cur_key[0], pos), on)

    def _set_composite_channel(self, k: tuple, on: bool) -> None:
        """Channel *k* in the composite or not; the last one stays.  Picking channels switches Composite on."""
        cands = self._composite_candidates()
        if k not in cands:
            return
        if not on and not any(self.comp_on.get(c, True) for c in cands if c != k):
            # undo the tick box (later: it may be the table item whose signal this is)
            QtCore.QTimer.singleShot(0, self._update_composite_panel)
            self._sync_composite_menu()
            self.statusBar().showMessage("At least one channel stays in the composite", 4000)
            return
        self.comp_on[k] = on
        self._sync_composite_menu()
        if not self.a_composite.isChecked():
            self.a_composite.setChecked(True)   # -> _toggle_composite renders 2D and 3D
            return
        self._update_composite_panel()
        self._render_frame(histogram=False)
        self._update_volume()

    def _all_composite_channels(self) -> None:
        for k in self._composite_candidates():
            self.comp_on[k] = True
        self._sync_composite_menu()
        if not self.a_composite.isChecked():
            self.a_composite.setChecked(True)
            return
        self._update_composite_panel()
        self._render_frame(histogram=False)
        self._update_volume()

    def _fill_composite_menu(self) -> None:
        """The composite channels menu, built when it opens: a tick box per channel of the current size."""
        menu = self.comp_menu
        menu.clear()
        cands = self._composite_candidates()
        if len(cands) < 2:
            a = menu.addAction("No other channel of this size")
            a.setEnabled(False)
            return
        head = menu.addAction("Channels in the composite (2D and 3D)")
        head.setEnabled(False)
        stacks = self.files[self.cur_key[0]].stacks
        for k in cands:
            s = stacks[k[1]]
            a = _check_action(menu, _stack_label(k[1], s), _swatch(self._member_color(k, cands), 20, 12), k)
            box = a.property("box")
            box.setToolTip(_stack_tooltip(s))
            box.toggled.connect(lambda on, k=k: self._set_composite_channel(k, on))
        menu.addSeparator()
        a = menu.addAction(f"All {len(cands)} channels", self._all_composite_channels)
        a.setData("all")
        self._sync_composite_menu()

    def _sync_composite_menu(self) -> None:
        """Tick marks of the composite channels menu as in comp_on; the only ticked channel cannot be unticked."""
        cands = self._composite_candidates()
        ticked = [k for k in cands if self.comp_on.get(k, True)]
        for a in self.comp_menu.actions():
            k = a.data()
            box = a.property("box")
            if isinstance(box, QtWidgets.QCheckBox) and isinstance(k, tuple):
                with QtCore.QSignalBlocker(box):
                    box.setChecked(k in ticked)
                box.setEnabled(ticked != [k])
                box.setToolTip(_stack_tooltip(self.files[k[0]].stacks[k[1]]) if ticked != [k] else
                               "The last channel of the composite stays")
                if k in cands:
                    box.setIcon(_swatch(self._member_color(k, cands), 20, 12))
            elif k == "all":
                a.setEnabled(len(ticked) < len(cands) or not self.a_composite.isChecked())

    def _sync_volume_composite(self) -> None:
        """The composite line of the 3D tab: how many channels are shown, and how to show the others."""
        if self.volume_panel is None:
            return
        cands = self._composite_candidates()
        on = self.a_composite.isChecked() and len(cands) > 1
        shown = sum(self.comp_on.get(k, True) for k in cands) if on else 1
        name = _stack_label(self.cur_key[1], self.files[self.cur_key[0]].stacks[self.cur_key[1]]) \
            if self.cur_key is not None and self.cur_key[0] in self.files else ""
        self.volume_panel.set_composite(len(cands), shown, on, name)

    def _composite_color(self, pos: int, color: QtGui.QColor) -> None:
        self.comp_colors[(self.cur_key[0], pos)] = color
        self._update_composite_panel()
        self._render_frame(histogram=False)
        self._sync_volume_looks()

    def _composite_auto_all(self) -> None:
        idx = self.frames.index()
        for k in self._composite_members():
            arr = self._member_array(k)
            f = np.asarray(arr[idx] if idx else arr)
            step = max(1, int(math.sqrt(f.size / 1_000_000)))
            st = self._member_state(k)
            st.lo, st.hi = _percentiles(f[::step, ::step])
        self.display.set_state(self.states[self.cur_key], np.issubdtype(self.arr.dtype, np.integer))
        self._update_composite_panel()
        self._render_frame(histogram=False)

    # -- mosaics -----------------------------------------------------------------------

    def _show_mosaic_tab(self, mosaic, key: str | None = None) -> None:
        """Mosaic tab next to Display / Info / Settings while a mosaic is shown."""
        i = self.tabs.indexOf(self.mosaic_scroll)
        if mosaic is None:
            if i >= 0:
                self.tabs.removeTab(i)
            if not self.mosaic_panel.busy:
                self.mosaic_panel.set_mosaic(None)
            return
        first = i < 0
        if first:
            i = self.tabs.insertTab(1, self.mosaic_scroll, "Mosaic")
        if self.mosaic_panel.mosaic is not mosaic:
            self.mosaic_panel.set_mosaic(mosaic, self.mosaics.get(key, []), self.frames.index)
            first = True
        if first and self.tabs.currentWidget() is not self.psf:
            self.tabs.setCurrentWidget(self.mosaic_scroll)

    def _mosaic_changed(self, mosaic) -> None:
        """Layout, blending or registration of a mosaic changed: refresh what shows it."""
        key = next((k for k, group in self.mosaics.items() if mosaic in group), None)
        if key is None or key not in self.files:
            return
        self._layout_versions[id(mosaic)] = self._layout_versions.get(id(mosaic), 0) + 1   # a 3D volume reloads
        stacks = self.files[key].stacks
        pos = next((i for i, t in enumerate(stacks) if t is mosaic.stack), None)
        for k in [k for k in self._comp_arrays if k[0] == key and k[1] == pos]:
            del self._comp_arrays[k]
        if pos is not None:
            thumb = _thumbnail(mosaic.stack)
            self.thumbs[key][pos] = thumb
            item = self.file_items[key].child(pos)
            item.setIcon(0, QtGui.QIcon(thumb))
            item.setText(0, f"{_stack_label(pos, mosaic.stack)}\n{_shape_text(mosaic.stack)}"
                         + (f"  ·  {mosaic.stack.time}" if mosaic.stack.time else ""))
            item.setToolTip(0, _stack_tooltip(mosaic.stack))
        showing = self.center.currentWidget() is self.viewer_page
        if pos is not None and self.cur_key == (key, pos) and self.arr is not None and showing:
            old = self.arr.shape
            index = self.frames.index()
            self.arr = mosaic.stack.asarray()
            if self.arr.shape[:-2] != old[:-2]:  # z range changed (tiles shifted in z)
                self.frames.configure(mosaic.stack, mosaic.stack.axes[:-2], self.arr.shape[:-2])
                self.frames.set_index(tuple(min(i, n - 1) for i, n in zip(index, self.arr.shape[:-2])))
            self._update_composite_panel()
            self._render_frame(new_stack=True)
            self._update_outlines()
            self.info.show_sections(self._stack_sections(self.files[key], mosaic.stack, pos))
        elif pos is not None and self.cur_key == (key, pos) and self.arr is not None:
            self.arr = self.frame = None  # reloaded by _show_stack when it is shown again
            self.cur_key = None
        elif self.cur_key and self.cur_key[0] == key and showing:
            self._update_composite_panel()
            self._render_frame(histogram=False)
        self._update_volume()
        self._update_actions()

    def _update_outlines(self) -> None:
        for it in self._outline_items:
            self.view.scene().removeItem(it)
        self._outline_items = []
        if not (isinstance(self.arr, mm.MosaicArray) and self.mosaic_panel.outlines.isChecked()
                and self.center.currentWidget() is self.viewer_page):
            return
        m = self.arr.mosaic
        pen = QtGui.QPen(QtGui.QColor(255, 200, 40, 220), 0)
        pen.setCosmetic(True)
        font = QtGui.QFont(self.font())
        font.setBold(True)
        for k, (x, y, w, h) in enumerate(self.arr.tile_rects()):
            rect = self.view.scene().addRect(QtCore.QRectF(x, y, w, h), pen)
            rect.setZValue(5)
            label = self.view.scene().addSimpleText(f"{m.tile_label(k)}  S{m.tiles[k].pos + 1}", font)
            label.setBrush(QtGui.QColor(255, 200, 40))
            label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            label.setPos(x + 4, y + 4)
            label.setZValue(6)
            self._outline_items += [rect, label]

    def _track_task(self, thread) -> None:
        self.tasks.add(thread)
        thread.finished.connect(lambda: (self.tasks.discard(thread), self._update_actions()))
        self._update_actions()

    # -- 3D view -----------------------------------------------------------------------

    def toggle_volume(self, on: bool) -> None:
        """3D view of z-stacks and 3D mosaics on or off; it stays on as a preference while a stack has no Z."""
        if on and self.volume_view is None and not self._create_volume_view():
            self.a_volume.setChecked(False)
            return
        self._vol_skip_key = None  # switching on again retries a load that failed or was cancelled
        self._update_volume()
        if on and self._volume_shown() and self.tabs.currentWidget() is not self.volume_scroll:
            self.tabs.setCurrentWidget(self.volume_scroll)
        n = len(self._composite_candidates())
        if on and self._volume_shown() and n > 1 and not self.a_composite.isChecked():
            self.statusBar().showMessage(f"3D shows the selected channel only. Composite (C) shows all {n} channels "
                                         "of this size together; its arrow ▾ chooses them.", 10000)
        if self.center.currentWidget() is self.viewer_page:
            self._focus_view()
        self._update_actions()

    def _create_volume_view(self) -> bool:
        """The 3D view and its tab, on first use; False (after a message) if they cannot run here."""
        try:
            import msr_volume_panel
            import msr_volume_view
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "The 3D view could not be loaded (msr_volume.py, "
                                          "msr_volume_view.py and msr_volume_panel.py belong next to "
                                          f"msr_viewer.py):\n\n{exc}")
            return False
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:  # a context that cannot be made would break the whole window once a GL widget is in it
            problem = msr_volume_view.opengl_problem()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if problem:
            QtWidgets.QMessageBox.warning(self, APP_NAME, problem)
            return False
        view = msr_volume_view.VolumeView()
        view.glReady.connect(self._volume_gl_ready)
        self.view_stack.addWidget(view)
        panel = msr_volume_panel.VolumePanel(view)
        panel.set_resolution(self._vol_resolution)
        panel.resolutionChanged.connect(self._volume_resolution)
        panel.fromTwoDRequested.connect(self._volume_from_2d)
        panel.saveRequested.connect(self.save_png)
        panel.copyRequested.connect(self._copy_volume_image)
        panel.cancelRequested.connect(self._cancel_volume_load)
        panel.reloadRequested.connect(self._reload_volume)
        panel.overlayChanged.connect(self._sync_scalebar_action)   # the panel's Scale bar box -> B's check mark
        panel.set_composite_menu(self.comp_menu, _composite_icon())
        panel.compositeRequested.connect(self._composite_from_volume)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setFocusPolicy(Qt.NoFocus)   # a click on the tab's background leaves the keys with the 3D view
        scroll.setWidget(panel)
        for w in [scroll] + scroll.findChildren(QtWidgets.QWidget):
            w.setAcceptDrops(False)
        self.volume_view, self.volume_panel, self.volume_scroll = view, panel, scroll
        return True

    def _current_stack(self) -> mr.DataStack | None:
        """The stack on the viewer page, or None (file overview or welcome page)."""
        if (self.cur_key is None or self.cur_key[0] not in self.files
                or self.center.currentWidget() is not self.viewer_page):
            return None
        return self.files[self.cur_key[0]].stacks[self.cur_key[1]]

    def _volume_shown(self) -> bool:
        return (self.volume_view is not None and self.view_stack.currentWidget() is self.volume_view
                and self.center.currentWidget() is self.viewer_page)

    def _update_volume(self) -> None:
        """Show the 3D view while it is on and the current stack has Z planes; load its volume when needed."""
        s = self._current_stack()
        if not (self.a_volume.isChecked() and self.volume_view is not None and s is not None and _z_planes(s) > 1):
            self._leave_volume()
            return
        self._enter_volume()
        self._sync_volume_composite()
        key, keys, index, factor = self._volume_request(s)
        if key in (self._vol_key, self._vol_skip_key):
            if self._vol_loader is not None:  # back to what is shown (or failed): the other load is not needed
                self._stop_volume_loads()
            if key == self._vol_key:  # what is shown is what is selected: no note about another request
                self.volume_view.set_message(None)
                self.volume_panel.show_reload(False)
                self._volume_info()
            self._sync_volume_looks()
            return
        if (self._vol_loader is not None and self._vol_loader.key == key) or not self._vol_gl_ok:
            return
        self._start_volume_load(key, keys, index, factor)

    def _volume_request(self, s: mr.DataStack) -> tuple:
        """(load key, (file, stack) keys, frame index, downsampling) of the volume to show.

        The channels are the active composite channels while Composite is on, else the
        current stack: what the 2D view shows.  The key changes with anything that changes
        the volume: stacks, non-Z frame index, mosaic layout, calibration and downsampling.
        """
        import msr_volume
        stacks = self.files[self.cur_key[0]].stacks
        members = self._composite_members()
        keys = [k for k in members if self.comp_on.get(k, True)] or [self.cur_key]
        index = self.frames.index()
        others = tuple((a, i) for a, i in zip(s.axes[:-2], index) if a != "Z")
        factor = msr_volume.stack_factors(s, self._vol_resolution, self.volume_view.max_texture_size())

        def ident(k: tuple) -> tuple:
            t = stacks[k[1]]
            layout = self._layout_versions.get(id(t.mosaic), 0) if isinstance(t, mm.MosaicStack) else 0
            return k, layout, t.axes, tuple(t.shape), t.step("X"), t.step("Y"), t.step("Z"), tuple(t.units)

        return (tuple(ident(k) for k in keys), others, factor), keys, index, factor

    def _start_volume_load(self, key: tuple, keys: list, index: tuple, factor: tuple) -> None:
        import msr_volume_view
        self._stop_volume_loads()
        stacks = self.files[self.cur_key[0]].stacks
        sources = [(_stack_label(k[1], stacks[k[1]]), _frozen(stacks[k[1]]), k) for k in keys]
        thread = msr_volume_view.VolumeLoadThread(sources, index, factor=factor, key=key, parent=self)
        thread.progress.connect(lambda done, total, t=thread: self._volume_progress(t, done, total))
        thread.done.connect(lambda volume, t=thread: self._volume_loaded(t, volume))
        thread.failed.connect(lambda msg, t=thread: self._volume_failed(t, msg))
        thread.finished.connect(lambda t=thread: self._volume_thread_finished(t))
        self._vol_loader = thread
        self._vol_threads.add(thread)
        self._vol_skip_key = None
        self._vol_load_mode = self._vol_resolution
        what = sources[0][0] if len(sources) == 1 else f"{len(sources)} channels"
        self._vol_load_text = f"Loading {what}…"
        # other stacks than the volume on screen: say so over it (not for another time point of the same)
        shown = [ident[0] for ident in self._vol_key[0]] if self._vol_key else []
        self._vol_load_message = shown != keys
        self.volume_panel.set_loading(self._vol_load_text, 0, len(keys) * _z_planes(stacks[keys[0][1]]))
        self.volume_view.set_message(self._vol_load_text if self._vol_load_message else None)
        self._volume_info()
        thread.start()
        self._update_actions()

    def _volume_progress(self, thread, done: int, total: int) -> None:
        if thread is not self._vol_loader:
            return
        self.volume_panel.set_loading(self._vol_load_text, done, total)
        if self._vol_load_message:
            self.volume_view.set_message(f"{self._vol_load_text} {100 * done // max(total, 1)} %")

    def _volume_loaded(self, thread, volume) -> None:
        if thread is not self._vol_loader:  # superseded or cancelled
            return
        self._vol_loader = None
        self.volume_panel.set_loading(None)
        self.volume_view.set_message(None)
        if volume is not None:
            self.volume_view.set_volume(volume, self._volume_channel_settings(volume), keep_view=True)
            self.volume_panel.set_volume(volume)
            self._vol_key, self._vol_mode = thread.key, self._vol_load_mode
        self._update_actions()

    def _volume_failed(self, thread, msg: str) -> None:
        if thread is not self._vol_loader:
            return
        self._vol_loader = None
        self.volume_panel.set_loading(None)
        self._load_stopped(thread.key, f"The volume could not be loaded: {msg}")
        self._update_actions()

    def _volume_thread_finished(self, thread) -> None:
        self._vol_threads.discard(thread)
        thread.deleteLater()
        self._update_actions()

    def _stop_volume_loads(self, wait: bool = False) -> None:
        """Cancel the 3D loaders; *wait* until they have stopped (before their file closes)."""
        if self._vol_loader is not None:
            self._vol_loader = None
            self.volume_panel.set_loading(None)
            self.volume_view.set_message(None)
        for t in list(self._vol_threads):
            t.cancel()
        running = [t for t in self._vol_threads if t.isRunning()] if wait else []
        if running:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for t in running:
                    t.wait(30000)  # a cancelled loader stops after the plane it reads
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

    def _cancel_volume_load(self) -> None:
        """Cancel in the 3D tab: stop loading; the same volume is not loaded again by itself."""
        thread = self._vol_loader
        self._stop_volume_loads()
        self.volume_panel.set_loading(None)
        if thread is not None:
            self._load_stopped(thread.key, "Loading was cancelled.")
        self._update_actions()

    def _load_stopped(self, key: tuple, text: str) -> None:
        """The load of *key* was cancelled or failed (*text*); it is not retried by itself.

        If it differs from the volume shown in the resolution only, that volume stays and the
        resolution goes back to its mode.  Otherwise the view is cleared (the old volume is not
        what the tree and the sliders select) and the 3D tab offers Reload.
        """
        self._vol_skip_key = key
        if self._vol_key is not None and key[:2] == self._vol_key[:2]:
            self._vol_resolution = self._vol_mode
            self.volume_panel.set_resolution(self._vol_mode)
            self.volume_view.set_message(None)
            self._volume_info(f"{text} The volume keeps its resolution.")
            return
        self._drop_volume()
        self.volume_view.set_message(f"{text}\nReload in the 3D tab tries again.")
        self.volume_panel.show_reload(True)
        self._volume_info(text)

    def _reload_volume(self) -> None:
        """Reload in the 3D tab, after a cancelled or failed load."""
        self._vol_skip_key = None
        self._update_volume()

    def _drop_volume(self) -> None:
        """No volume in the 3D view, its textures freed (the 3D settings of every stack stay)."""
        if self.volume_view.volume is not None or self.volume_view.channels:
            self.volume_view.release()          # the textures
            self.volume_view.set_volume(None, [])
            self.volume_panel.set_volume(None)
        self._vol_key = self._vol_mode = None

    def _enter_volume(self) -> None:
        """The 3D view in place of the 2D view, its tab, and the Z slider off."""
        if self.view_stack.currentWidget() is not self.volume_view:
            self.view_stack.setCurrentWidget(self.volume_view)
            self._sync_scalebar_action()
            self.pixel_label.setText(VOLUME_HINT)
            self.zoom_label.hide()  # the zoom of the 2D view
        if self.tabs.indexOf(self.volume_scroll) < 0:
            i = self.tabs.indexOf(self.mosaic_scroll)
            self.tabs.insertTab(i + 1 if i >= 0 else 1, self.volume_scroll, "3D")
            if self.psf is None or self.tabs.currentWidget() is not self.psf:
                self.tabs.setCurrentWidget(self.volume_scroll)
        self.frames.set_axis_enabled("Z", False)

    def _leave_volume(self, release: bool = False) -> None:
        """Back to the 2D view without the 3D tab, loading stopped; *release* also drops the volume.

        A volume larger than KEEP_VOLUME_BYTES is dropped too; a smaller one stays (RAM and
        textures), so that switching 3D on again shows it at once.
        """
        if self.volume_view is None:
            return
        self._stop_volume_loads(wait=release)
        if self.view_stack.currentWidget() is self.volume_view:
            self.view_stack.setCurrentWidget(self.view)
            self._sync_scalebar_action()
            self.zoom_label.show()
            if self.pixel_label.text() == VOLUME_HINT:
                self.pixel_label.setText("")
        i = self.tabs.indexOf(self.volume_scroll)
        if i >= 0:
            current = self.tabs.currentIndex() == i
            self.tabs.removeTab(i)
            if current:
                self.tabs.setCurrentIndex(0)  # the Display tab rather than whichever tab is next
        self.frames.set_axis_enabled("Z", True)
        volume = self.volume_view.volume
        if release or (volume is not None and volume.nbytes > KEEP_VOLUME_BYTES):
            self._drop_volume()
            self.volume_view.set_message(None)
            self.volume_panel.show_reload(False)
            self._vol_skip_key = None

    def _release_volume_of(self, key: str) -> None:
        """File *key* is about to close: stop every 3D loader and drop what the 3D view holds of it."""
        for m in self.mosaics.get(key, []):
            self._layout_versions.pop(id(m), None)
        if self.volume_view is None:
            return
        self._stop_volume_loads(wait=True)
        shown = self._vol_key is not None and any(ident[0][0] == key for ident in self._vol_key[0])
        if shown or (self.cur_key is not None and self.cur_key[0] == key):
            self._leave_volume(release=True)
        for d in (self._vol_channels, self._vol_looks):
            for k in [k for k in d if k[0] == key]:
                del d[k]
        self.volume_panel.forget(key)   # Reset snapshots live as long as the settings above

    def _sync_scalebar_action(self) -> None:
        """Scale bar (B) checked as in the view that is shown."""
        on = self.volume_view.settings.show_scalebar if self._volume_shown() else self.view.show_scalebar
        self.a_scalebar.setChecked(bool(on))

    def _volume_look(self, k: tuple, members: list[tuple]) -> tuple:
        """(colour, lut) of a 3D channel as stack *k* looks in 2D: composite colour, or its LUT."""
        if k in members:
            c = self._member_color(k, members)
            return (c.redF(), c.greenF(), c.blueF()), None
        lut = self._member_state(k).lut
        if lut in RAMP_LUTS:
            return tuple(float(v) / 255.0 for v in LUTS[lut][-1]), None
        return (1.0, 1.0, 1.0), lut

    def _follow_look(self, cs, look: tuple) -> bool:
        """Give a 3D channel the colour of the 2D view unless it was changed in the 3D tab; True if it changed."""
        old = self._vol_looks.get(cs.key)
        if look == old or (old is not None and (tuple(cs.color), cs.lut) != old):
            return False
        cs.color, cs.lut = look
        self._vol_looks[cs.key] = look
        return True

    def _volume_channel_settings(self, volume) -> list:
        """3D settings of the volume's channels: kept per stack while its file is open, new ones from 2D."""
        import msr_volume
        members = self._composite_members()
        out = []
        for vc in volume.channels:
            k = vc.key
            look = self._volume_look(k, members)
            cs = self._vol_channels.get(k)
            if cs is None:
                st = self._member_state(k)
                cs = msr_volume.ChannelSettings(name=vc.name, color=look[0], lut=look[1], key=k)
                cs.set_window(st.lo, st.hi)
                self._vol_channels[k] = cs
                self._vol_looks[k] = look
            else:
                self._follow_look(cs, look)
            out.append(cs)
        return out

    def _sync_volume_looks(self) -> None:
        """Colours of the 3D channels after a 2D LUT or composite colour change."""
        if self.volume_view is None or not self.volume_view.channels:
            return
        members = self._composite_members()
        changed = False
        for cs in self.volume_view.channels:
            if cs.key is not None and cs.key[0] in self.files:
                changed |= self._follow_look(cs, self._volume_look(cs.key, members))
        if changed:
            self.volume_panel.refresh()
            self.volume_view.settings_changed()

    def _volume_info(self, error: str | None = None) -> None:
        """Renderer (or why OpenGL failed) and an error under the volume summary of the 3D tab."""
        gpu = (f"OpenGL: {self._vol_gpu}" if self._vol_gl_ok else self._vol_gpu) if self._vol_gpu else ""
        self.volume_panel.set_info("\n".join(t for t in (gpu, error) if t))

    def _volume_gl_ready(self, ok: bool, text: str) -> None:
        self._vol_gl_ok, self._vol_gpu = ok, text
        if not ok:
            self._stop_volume_loads()
        self._volume_info()
        if ok:  # the texture size limit is known now; reloads only if it changes the downsampling
            QtCore.QTimer.singleShot(0, self._update_volume)

    def _composite_from_volume(self, on: bool) -> None:
        """The buttons of the 3D tab's composite line: all channels of this size together, or the selected one."""
        if on:
            self._all_composite_channels()
        else:
            self.a_composite.setChecked(False)

    def _volume_resolution(self, mode: str) -> None:
        self._vol_resolution = mode
        self._vol_skip_key = None  # picked in the 3D tab: always tried, also after a cancel or failure
        self._update_volume()

    def _volume_from_2d(self, i: int) -> None:
        """2D button of the 3D tab: channel *i* gets the Min / Max of its stack in the 2D view."""
        chans = self.volume_view.channels
        if not 0 <= i < len(chans) or chans[i].key is None or chans[i].key[0] not in self.files:
            return
        st = self._member_state(chans[i].key)
        chans[i].set_window(st.lo, st.hi)
        self.volume_panel.refresh()
        self.volume_view.settings_changed()

    def _render_volume_image(self) -> QtGui.QImage | None:
        """Full-quality image of the 3D view with its overlays, or None (after a message)."""
        if self.volume_view.volume is None:
            QtWidgets.QMessageBox.information(self, APP_NAME, "The volume is not loaded yet.")
            return None
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            img, error = self.volume_view.render_image(), None
        except RuntimeError as exc:
            img, error = None, str(exc)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if error:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not render the 3D view:\n\n{error}")
        return img

    def _copy_volume_image(self) -> None:
        img = self._render_volume_image()
        if img is not None:
            QtWidgets.QApplication.clipboard().setImage(img)
            self.statusBar().showMessage(f"3D view copied to the clipboard ({img.width()} × {img.height()} px)", 5000)

    def _save_volume_png(self) -> None:
        """Save view as PNG while the 3D view is shown: the rendered view with its overlays.

        The file is named after the volume shown, which differs from the selection while
        another stack or time point loads.
        """
        keys = [ident[0] for ident in self._vol_key[0]] if self._vol_key else []   # (file, stack) of each channel
        if self.volume_view.volume is None or not keys or keys[0][0] not in self.files:
            QtWidgets.QMessageBox.information(self, APP_NAME, "The volume is not loaded yet.")
            return
        msr = self.files[keys[0][0]]
        pos = keys[0][1]
        s = msr.stacks[pos]
        part = ("composite" if len(keys) > 1 else
                _file_part(pos, s) if isinstance(s, mm.MosaicStack) else f"S{pos + 1}")
        frame_txt = "_".join(f"{a}{i + 1}" for a, i in self._vol_key[1])
        name = f"{mr.file_stem(msr.path)}_{part}{'_' + frame_txt if frame_txt else ''}_3d.png"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 3D view as PNG", start, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        img = self._render_volume_image()
        if img is not None and not img.save(path):
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not write {path}")

    # -- panels ---------------------------------------------------------------------

    @staticmethod
    def _file_sections(msr: mr.MSRFile):
        first = msr.stacks[0] if msr.stacks else None
        stacks = ("Stacks", [(_stack_label(i, s), " · ".join(x for x in (s.channel_id, _shape_text(s), s.time) if x))
                             for i, s in enumerate(msr.stacks)])
        warnings = ("Warnings", [(str(i + 1), w) for i, w in enumerate(msr.warnings)])
        if isinstance(msr, mt.TIFFFile):
            rows = [("Path", msr.path),
                    ("Size", f"{os.path.getsize(mr._fs_path(msr.path)) / 1e6:.1f} MB"),
                    ("Format", msr.version),
                    ("Software", msr.software),
                    ("Image series", msr.series_count),
                    ("Pages", msr.page_count),
                    ("Stacks", len(msr.stacks)),
                    ("Metadata", ", ".join(msr.metadata) or "TIFF tags only"),
                    ("TIFF / ImageJ / OME entries", len(msr.properties))]
            return [("File", rows), stacks, warnings]
        comp = msr.compression
        mosaics = [s for s in msr.stacks if isinstance(s, mm.MosaicStack)]
        rows = [("Path", msr.path),
                ("Size", f"{os.path.getsize(mr._fs_path(msr.path)) / 1e6:.1f} MB"),
                ("Compression", f"zlib ('spc' container), {comp['inflated_bytes'] / 1e6:.1f} MB inflated "
                                "into a temporary file" if comp else "none"),
                ("Format version", msr.version),
                ("Software", first.meta.get("VersionNr") if first else ""),
                ("Data stacks", len(msr.stacks) - len(mosaics)),
                ("Mosaics", ", ".join(s.mosaic.title for s in mosaics)),
                ("Display views", len(msr.views)),
                ("Global settings", len(msr.properties))]
        sections = [("File", rows)]
        for ps in msr.property_sets:
            sections.append((f"Workspace '{ps.get('propset_label', '')}'",
                             [("Autosave prefix", ps.get("seq_autosave_prefix")), ("Id", ps.get("propset_id")),
                              ("Settings", len(ps.get("properties", {})))]))
        return sections + [stacks, warnings]

    @staticmethod
    def _stack_sections(msr: mr.MSRFile, s: mr.DataStack, pos: int):
        m = s.meta
        px, py = s.pixel_size
        fov = [s.axis_info(a) for a in "XY"]
        dt = s.time_increment if "T" in s.axes else None
        tiff = isinstance(s, mt.TIFFStack)
        acq = [("Source", s.source), ("Channel ID", s.channel_id), ("Acquired", m.get("Creation Date", s.time)),
               ("Instrument mode", m.get("Instrument Mode")), ("Measurement mode", m.get("Measurement Mode"))]
        if tiff:
            frames = {"Z": "Z slices", "T": "time points"}.get(s.assumed)
            acq += [("Image series", f"{s.series + 1} of {msr.series_count}" + (f"  ({s.name})" if s.name else "")),
                    ("Frames", f"taken as {frames} (the file has no axis information; Tools ▸ Calibration)"
                     if frames else "")]
        else:
            acq.append(("Workspace", f"S{pos + 1} (element {s.index} of the stack array)"))
        geo = [("Axes", " × ".join(f"{n} {a}" for a, n in zip(s.axes, s.shape))),
               ("Data type", s.dtype),
               ("Pixel size", f"{px:.4g} × {py:.4g} µm" if px and py
                else "not in the file" + (" (Tools ▸ Calibration)" if tiff else "")),
               ("Field of view", f"{fov[0]['length']:.4g} × {fov[1]['length']:.4g} µm" if px and py and all(fov)
                else "")]
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
        if isinstance(s, mm.MosaicStack):
            mo = s.mosaic
            acq[-1] = ("Workspace", f"fused from {len(mo.tiles)} stacks: "
                                    + ", ".join(f"S{t.pos + 1}" for t in mo.tiles))
            mos = [(k[len("Mosaic "):].capitalize() if k != "Mosaic" else "Tiles", v)
                   for k, v in m.items() if k.startswith("Mosaic")]
            f = [("Path", msr.path), ("Pixel data", "composed from the tiles when shown or saved"),
                 ("Size when saved", f"{s.nbytes / 1e6:.2f} MB"), ("Settings", "snapshot of the first tile")]
            return [("Acquisition", acq), ("Mosaic", mos), ("Geometry", geo), ("Objective", obj),
                    ("User", user), ("File", f)]
        if tiff:
            f = [("Path", msr.path), ("Format", msr.version),
                 ("Pixel data", s.access + (f" (offset {s.data_offset:,} bytes)" if s.access == "memory-mapped" else "")),
                 ("Pixel data size", f"{s.nbytes / 1e6:.2f} MB"),
                 ("ImSpector settings", len(s.properties) if s.properties else "")]
        else:
            f = [("Path", msr.path), ("Pixel data offset", f"{s.data_offset:,} bytes"),
                 ("Pixel data size", f"{s.nbytes / 1e6:.2f} MB"), ("Settings in snapshot", len(s.properties)),
                 ("Header parse", s.parse_mode)]
        return [("Acquisition", acq), ("Geometry", geo), ("Objective", obj), ("ImSpector display", disp),
                ("PCO camera", cam), ("User", user), ("File", f)]

    @staticmethod
    def _setting_sources(msr: mr.MSRFile, s: mr.DataStack | None):
        sources = []
        stacks = [(f"{_stack_label(i, t)}  ({', '.join(x for x in (t.channel_id, t.time) if x) or t.source})",
                   t.properties) for i, t in enumerate(msr.stacks)]
        if isinstance(msr, mt.TIFFFile):
            # TIFF stacks have ImSpector settings only when the file is an msr_reader export
            first = [x for t, x in zip(msr.stacks, stacks) if t is s and t.properties]
            others = [x for t, x in zip(msr.stacks, stacks) if t is not s and t.properties]
            return first + [("File metadata  (TIFF tags, ImageJ, OME)", msr.properties)] + others
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
        if isinstance(msr, mt.TIFFFile):  # whole-file export is for .msr files
            return
        start = self.qsettings.value("export_dir", os.path.dirname(msr.path))
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export into folder (a '<name>_tiff' subfolder will be created)", start)
        if not folder:
            return
        self.qsettings.setValue("export_dir", folder)
        path, lines = msr.path, []
        out = os.path.join(folder, mr.file_stem(path) + "_tiff")
        mosaics = []  # the fused mosaics as they are shown now (copies fixed to this layout)
        for i, s in enumerate(msr.stacks):
            if isinstance(s, mm.MosaicStack):
                name = (f"M{len(mosaics) + 1:02d}_{_file_part(i, s)}_{s.mosaic.blend}"
                        + (".tif" if imagej else ".ome.tif"))
                mosaics.append((name, _frozen(s)))

        def work():
            written = mr.export(path, folder, imagej=imagej, log=lines.append, msr=msr)  # no second inflate
            for name, s in mosaics:
                rec = mr.export_group(msr, [s], os.path.join(out, name), imagej=imagej)
                lines.append(f"  -> {name}  ({s.mosaic.title}, {s.meta.get('Mosaic layout')}) "
                             f"axes={rec['axes']} shape={'x'.join(map(str, rec['shape']))}")
                written.append(rec)
            return written

        def done(written):
            self._done_box(f"Exported {len(written)} image file(s) + metadata.json to\n{out}", "\n".join(lines), out)

        self._run_task(work, done, "Export")

    def save_stack(self) -> None:
        if self.cur_key is None or self.center.currentWidget() is not self.viewer_page:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        stem = mr.file_stem(msr.path)
        name = f"{stem}_{_file_part(pos, s)}.ome.tif"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        title = "Save mosaic" if isinstance(s, mm.MosaicStack) else "Save stack"
        path, flt = QtWidgets.QFileDialog.getSaveFileName(self, title, start,
                                                          "OME-TIFF (*.ome.tif);;ImageJ TIFF (*.tif)")
        if not path:
            return
        imagej = flt.startswith("ImageJ")
        base = re.sub(r"(\.ome)?\.tiff?$", "", path, flags=re.I)
        path = base + (".tif" if imagej else ".ome.tif")
        self.qsettings.setValue("export_dir", os.path.dirname(path))
        frozen = _frozen(s)
        self._run_task(lambda: mr.export_group(msr, [frozen], path, imagej=imagej),
                       lambda rec: self._done_box(f"Saved {os.path.basename(path)}\naxes {rec['axes']}, "
                                                  f"shape {' × '.join(map(str, rec['shape']))}", "",
                                                  os.path.dirname(path)), "Saving")

    def save_png(self) -> None:
        """Save view as PNG: the 3D view while it is shown, else the 2D frame."""
        if self._volume_shown():
            self._save_volume_png()
        else:
            self._save_2d_png()

    def _save_2d_png(self) -> None:
        """The current 2D frame (or composite) with its LUT and scale bar, at full resolution."""
        if self.frame is None or self.cur_key is None:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        st = self.states[self.cur_key]
        members = self._composite_members()
        if members:
            img = rgb_to_qimage(self._composite_rgb(members, self.frames.index()))
        else:
            img = rgb_to_qimage(render_rgb(self.frame, st.lo, st.hi, st.lut, self.mapper))
        img = paint_overlays(img, s.pixel_size[0], self.view.unit, self.view.show_scalebar)
        stem = mr.file_stem(msr.path)
        frame_txt = "_".join(f"{a}{i + 1}" for a, i in zip(s.axes[:-2], self.frames.index()))
        part = "composite" if members else (_file_part(pos, s) if isinstance(s, mm.MosaicStack) else f"S{pos + 1}")
        name = f"{stem}_{part}{'_' + frame_txt if frame_txt else ''}.png"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save view as PNG", start, "PNG image (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            if not img.save(path):
                QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not write {path}")

    def export_timestamps(self) -> None:
        if self.cur_key is None or self.center.currentWidget() is not self.viewer_page:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        if not s.timestamps:
            QtWidgets.QMessageBox.information(self, APP_NAME, "This stack has no per-frame times.")
            return
        stem = mr.file_stem(msr.path)
        name = f"{stem}_S{pos + 1}_{mr._safe(s.channel_id.split(':')[0] or s.source)}_timestamps.txt"
        start = os.path.join(self.qsettings.value("export_dir", os.path.dirname(msr.path)), name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export frame times", start, "Text file (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        self.qsettings.setValue("export_dir", os.path.dirname(path))
        try:
            n = mr.write_timestamps(s, path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Could not write {path}:\n{exc}")
            return
        self.statusBar().showMessage(f"Wrote {n} frame times (s, first frame at {s.timestamps[0]:.6g} s) to "
                                     f"{os.path.basename(path)}", 8000)

    def export_video(self) -> None:
        if self.cur_key is None or self.center.currentWidget() is not self.viewer_page or not self.frames.rows:
            return
        msr = self.files[self.cur_key[0]]
        pos = self.cur_key[1]
        s = msr.stacks[pos]
        st = self.states[self.cur_key]
        self.frames.stop()
        stem = mr.file_stem(msr.path)
        defaults = {
            "real_time": self.frames.mode.currentIndex() == 0, "speed": self.frames.speed.value(),
            "fps": self.frames.fps.value(), "out_fps": int(self.qsettings.value("video_out_fps", 60)),
            "scalebar": self.view.show_scalebar, "lut": st.lut, "lo": st.lo, "hi": st.hi,
            "stem": f"{stem}_{_file_part(pos, s)}",
            "folder": self.qsettings.value("export_dir", os.path.dirname(msr.path)),
        }
        members = self._composite_members()
        s = _frozen(s)
        if members:  # the video shows the composite as on screen; timing from the selected channel
            defaults["layers"] = [(_frozen(msr.stacks[k[1]]), self._member_color(k, members).getRgb()[:3],
                                   self._member_state(k).lo, self._member_state(k).hi)
                                  for k in members if self.comp_on.get(k, True)]
            defaults["stem"] = f"{stem}_composite"
        dlg = mv.VideoExportDialog(s, self.arr.shape[:-2], self.frames.index(), self.frames.play_row, defaults, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        settings = dlg.settings()
        self.qsettings.setValue("export_dir", os.path.dirname(settings.path))
        self.qsettings.setValue("video_out_fps", int(settings.out_fps))
        self._start_video_export(s, self.frames.index(), settings)

    def _start_video_export(self, s: mr.DataStack, index: tuple, settings: mv.VideoSettings) -> mv.VideoExportThread:
        thread = mv.VideoExportThread(s, index, settings, self)
        progress = QtWidgets.QProgressDialog(f"Writing {os.path.basename(settings.path)}…", "Cancel", 0, 1, self)
        progress.setWindowTitle("Export video")
        progress.setWindowModality(Qt.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(thread.cancel)
        thread.progress.connect(lambda k, n: (progress.setMaximum(n), progress.setValue(k)))

        def done(info: dict) -> None:
            progress.close()
            if info.get("cancelled"):
                self.statusBar().showMessage("Video export cancelled", 5000)
                return
            w, h = info["size"]
            self._done_box(f"Wrote {os.path.basename(info['path'])}\n{info['duration']:.2f} s, {info['frames']} "
                           f"frames at {info['fps']:g} fps, {w} × {h} px", info["summary"],
                           os.path.dirname(info["path"]))

        def failed(msg: str) -> None:
            progress.close()
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Video export failed:\n\n{msg}")

        thread.done.connect(done)
        thread.failed.connect(failed)
        thread.finished.connect(lambda: (self.tasks.discard(thread), self._update_actions()))
        self.tasks.add(thread)
        self._update_actions()
        progress.show()
        thread.start()
        return thread

    # -- tools ------------------------------------------------------------------------

    def toggle_psf(self, on: bool) -> None:
        """PSF analysis as an extra, closable tab next to Display / Info / Settings."""
        if on:
            if self.psf is None:
                try:
                    import msr_psf
                except ImportError as exc:
                    self.a_psf.setChecked(False)
                    QtWidgets.QMessageBox.warning(self, APP_NAME, "PSF analysis needs pyqtgraph:\n\n"
                                                  f"    pip install pyqtgraph\n\n({exc})")
                    return
                self.psf = msr_psf.PSFPanel(self.view)
                for w in [self.psf] + self.psf.findChildren(QtWidgets.QWidget):
                    w.setAcceptDrops(False)
            i = self.tabs.addTab(self.psf, "PSF")
            close = QtWidgets.QToolButton()
            close.setText("×")
            close.setAutoRaise(True)
            close.setToolTip("Close PSF analysis (Ctrl+Shift+P)")
            close.clicked.connect(lambda: self.a_psf.setChecked(False))
            self.tabs.tabBar().setTabButton(i, QtWidgets.QTabBar.RightSide, close)
            self.tabs.setCurrentWidget(self.psf)
            self.psf.activate()
            self._psf_feed(new_stack=True)
        elif self.psf is not None:
            self.psf.deactivate()
            i = self.tabs.indexOf(self.psf)
            if i >= 0:
                self.tabs.removeTab(i)
        self._update_actions()

    def open_linescan(self) -> None:
        """Pick stacks, then compare them along a line in a separate window."""
        if not self.files:
            return
        try:
            import msr_linescan
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "The line scan needs pyqtgraph:\n\n"
                                          f"    pip install pyqtgraph\n\n({exc})")
            return
        current = self.cur_key if self.center.currentWidget() is self.viewer_page else None
        dlg = msr_linescan.StackPickerDialog(self.files, current, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        stacks, names = dlg.selection()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            win = msr_linescan.LineScanWindow(stacks, names, self)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.linescan_windows.append(win)
        win.destroyed.connect(lambda *_, w=win: self.linescan_windows.remove(w)
                              if w in self.linescan_windows else None)
        win.show()

    def edit_calibration(self) -> None:
        """Set pixel size, z step and frame interval of TIFF stacks; the view and the tools follow."""
        if self.cur_key is None or self.center.currentWidget() is not self.viewer_page:
            return
        key, pos = self.cur_key
        f = self.files[key]
        s = f.stacks[pos]
        if not isinstance(s, mt.TIFFStack):
            return
        dlg = CalibrationDialog(s, len(f.stacks), self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        errors = []
        for t in (f.stacks if dlg.all.isChecked() else [s]):
            try:
                t.set_calibration(**dlg.values())
            except ValueError as exc:
                errors.append(f"{_stack_label(f.stacks.index(t), t)}: {exc}")
        for i, t in enumerate(f.stacks):
            self.file_items[key].child(i).setToolTip(0, _stack_tooltip(t))
        self.cur_key = None  # shown again: sliders, scale bar and the PSF tab use the new values
        self._show_stack(key, pos)
        self._update_actions()
        if errors:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Not changed:\n\n" + "\n".join(errors))

    def _psf_feed(self, new_stack: bool = False) -> None:
        if self.psf is None or not self.psf.active or self.cur_key is None or self.frame is None:
            return
        if new_stack or self.psf.stack is None:
            s = self.files[self.cur_key[0]].stacks[self.cur_key[1]]
            self.psf.set_stack(s, self.arr, self.frames.index(), self.frame)
        else:
            self.psf.set_frame(self.frame, self.frames.index())

    # -- misc -------------------------------------------------------------------------

    def _update_actions(self) -> None:
        has_file = self._current_file_key() is not None
        showing_stack = self.cur_key is not None and self.center.currentWidget() is self.viewer_page
        busy = self._exporting()  # a running registration blocks nothing; exports use a fixed copy
        is_msr = has_file and not isinstance(self.files[self._current_file_key()], mt.TIFFFile)
        for a in (self.a_export, self.a_export_ij):
            a.setEnabled(is_msr and not busy)
        self.a_close.setEnabled(has_file and not busy)
        for a in (self.a_save_stack,):
            a.setEnabled(showing_stack and not busy)
        volume = self._volume_shown()
        for a in (self.a_save_png, self.a_fit):
            a.setEnabled(showing_stack)
        self.a_1to1.setEnabled(showing_stack and not volume)
        stack = self.files[self.cur_key[0]].stacks[self.cur_key[1]] if showing_stack else None
        self.a_volume.setEnabled(stack is not None and _z_planes(stack) > 1)
        for a in (self.a_play, self.a_prev, self.a_next):
            a.setEnabled(showing_stack and self.frames.can_step())  # not on the Z row in 3D
        has_times = showing_stack and bool(self.files[self.cur_key[0]].stacks[self.cur_key[1]].timestamps)
        self.a_export_ts.setEnabled(has_times)
        self.a_video.setEnabled(showing_stack and bool(self.frames.rows) and not busy)
        self.a_psf.setEnabled(showing_stack or self.a_psf.isChecked())
        self.a_linescan.setEnabled(bool(self.files))
        self.a_composite.setEnabled(showing_stack and len(self._composite_candidates()) > 1)
        self.comp_menu.menuAction().setEnabled(self.a_composite.isEnabled())
        self.a_calibrate.setEnabled(showing_stack and not busy
                                    and isinstance(self.files[self.cur_key[0]].stacks[self.cur_key[1]], mt.TIFFStack))
        showing_mosaic = showing_stack and isinstance(self.files[self.cur_key[0]].stacks[self.cur_key[1]],
                                                      mm.MosaicStack)
        self.a_mosaic_tiff.setEnabled(showing_mosaic and not busy)
        self.a_mosaic_video.setEnabled(showing_mosaic and bool(self.frames.rows) and not busy)
        self.a_mosaic_png.setEnabled(showing_mosaic)
        self.busy.setFormat("exporting…" if busy else "registering…" if self.tasks else "loading 3D…")
        self.busy.setVisible(bool(self.tasks) or self._vol_loader is not None)

    def _tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        menu = QtWidgets.QMenu(self)
        if item.data(0, Qt.UserRole)[0] == "stack":
            menu.addAction(self.a_save_stack)
            menu.addAction(self.a_save_png)
            menu.addAction(self.a_video)
            menu.addAction(self.a_export_ts)
            menu.addSeparator()
            menu.addAction(self.a_linescan)
            if self.a_calibrate.isEnabled():
                menu.addAction(self.a_calibrate)
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
            f"<b>{APP_NAME} {__version__}</b><br>Viewer for LaVision BioTec ImSpector .msr files<br>"
            "and TIFF / OME-TIFF / ImageJ stacks, compressed files and mosaics (stitching).<br><br>"
            f"Reader: msr_reader.py {mr.__version__} (format notes: MSR_FORMAT.md)<br>"
            "Wheel: zoom · drag: pan · double-click: fit<br>"
            "Left/Right: frame · Space: play · F: fit · 1: 100 % · B: scale bar · 3: 3D view<br><br>"
            "The 3D view is a Python port of the renderer of the Fiji plugin "
            "<a href='https://github.com/bene51/3Dscript'>3Dscript</a> (Copyright © 2016, Benjamin Schmid; "
            "BSD 2-Clause License, see Help ▸ Third-party notices):<br>"
            "Schmid B et al. 3Dscript: animating 3D/4D microscopy data using a natural-language-based syntax. "
            "<i>Nat Methods</i> 16, 278–280 (2019).")

    def show_notices(self) -> None:
        """Help ▸ Third-party notices: THIRD_PARTY_NOTICES.md (shipped next to the viewer) in a dialog."""
        self._notices_dialog().exec_()

    def _notices_dialog(self) -> QtWidgets.QDialog:
        path = _resource("THIRD_PARTY_NOTICES.md")
        text = "THIRD_PARTY_NOTICES.md was not found next to the viewer."
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except OSError as exc:
                text = f"Could not read {path}: {exc}"
        dlg = QtWidgets.QDialog(self)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.setWindowTitle(f"Third-party notices — {APP_NAME}")
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(text)    # read-only; the license itself is a code block, shown verbatim
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(browser)
        lay.addWidget(buttons)
        dlg.resize(760, 640)
        return dlg

    # -- drag & drop ------------------------------------------------------------------

    @staticmethod
    def _paths(mime: QtCore.QMimeData) -> list[str]:
        out = []
        if mime.hasUrls():
            for url in mime.urls():
                p = url.toLocalFile()
                if p and (os.path.isdir(p) or _openable(p)):
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
        if self._exporting():
            QtWidgets.QMessageBox.information(self, APP_NAME, "Please wait until the export has finished.")
            ev.ignore()
            return
        self._stop_registration()
        self._stop_volume_loads(wait=True)  # they read from the files closed below
        if self.volume_view is not None:
            self.volume_view.release()
        self.qsettings.setValue("geometry", self.saveGeometry())
        self.frames.stop()
        self.arr = self.frame = None
        for msr in self.files.values():
            msr.close()
        ev.accept()


# -----------------------------------------------------------------------------
# application
# -----------------------------------------------------------------------------

class _Application(QtWidgets.QApplication):
    """Opens the files the macOS Finder sends (Open With, drop onto the Dock icon) as QFileOpenEvent."""

    window = None

    def event(self, ev) -> bool:
        if ev.type() == QtCore.QEvent.FileOpen and self.window is not None and ev.file():
            path = ev.file()
            QtCore.QTimer.singleShot(0, lambda: self.window.open_paths([path]))
            return True
        return super().event(ev)


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


def _init_opengl() -> None:
    """OpenGL set-up for the 3D view; must run before the QApplication is created.

    No context is made here: OpenGL is loaded when 3D is first switched on.  (Qt's
    AA_ShareOpenGLContexts would create one at every start, and the 3D view never moves
    to another window, so it needs no shared context.)
    """
    if sys.platform == "darwin":  # macOS offers OpenGL above 2.1 only as a core profile, which must be the default
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    if os.environ.get("MSR_VIEWER_SOFTWARE_GL", "").strip() not in ("", "0"):
        QtCore.QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)  # Qt's Mesa llvmpipe (Windows builds)


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
    _init_opengl()
    app = _Application(argv)
    app.setApplicationName(APP_NAME)
    if sys.platform == "win32" and "Segoe UI" in QtGui.QFontDatabase().families():
        app.setFont(QtGui.QFont("Segoe UI", 9))  # Qt5 default is 'MS Shell Dlg 2' 8 pt
    app.setWindowIcon(_app_icon())
    _dark_palette(app)
    _install_excepthook()
    win = MainWindow()
    win.show()
    app.window = win
    paths = [a for a in argv[1:] if not a.startswith("-")]
    if paths:
        QtCore.QTimer.singleShot(0, lambda: win.open_paths(paths))
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
