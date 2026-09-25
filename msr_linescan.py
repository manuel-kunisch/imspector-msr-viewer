"""
msr_linescan.py -- line scans across several stacks, with alignment and correlation.

Tools > Line scan / correlation: pick stacks of the same size (two detectors,
repeated acquisitions, ...).  A separate window shows them as composite or one
by one; the line ROI gives the intensity profile of every stack along it
(averaged over a band of parallel lines), and the Pearson correlation between
the stacks along the line and over the whole image.  Stacks can be aligned to
the first one by phase correlation (sub-pixel).  Timelapses and z-stacks are
scanned per frame or as mean / maximum projection.

Concepts ported from OME-Tiff-PSF-Evaluator-V2/multichannel_linescan_correlation_app.py.
Needs pyqtgraph; scikit-image and scipy (optional) for sub-pixel registration,
matplotlib (optional) for saving the plot as PNG / PDF / SVG.
"""

from __future__ import annotations

import csv
import itertools
import math
import os

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401  (registers pg.exporters)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import msr_reader as mr
from msr_psf import _quiet, sample_points
from msr_widgets import Handle, ImageView, paint_overlays, qimage_to_rgb, rgb_to_qimage

PALETTE = ("#ff5050", "#3ddc84", "#4da3ff", "#ffb347", "#c77dff", "#20d0d0", "#ff7fbf", "#e0e0e0")
FRAME_MODES = ("single frame", "mean projection", "max projection")


# -----------------------------------------------------------------------------
# computation
# -----------------------------------------------------------------------------

def phase_shift(ref: np.ndarray, mov: np.ndarray, upsample: int = 100) -> tuple[float, float]:
    """(dy, dx) that moves *mov* onto *ref*, sub-pixel.

    Cross-correlation of lightly smoothed (sigma 1 px), Hann-windowed images,
    refined by upsampling (scikit-image) or a parabola fit.  On synthetic shifts
    this was the most accurate of the variants tried (within 0.02 px on clean data);
    phase normalisation without a window can fail by pixels because of the
    image edges and amplifies noise.
    """
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float64))
    mov = np.nan_to_num(np.asarray(mov, dtype=np.float64))
    try:
        from scipy.ndimage import gaussian_filter
        ref, mov = gaussian_filter(ref, 1.0), gaussian_filter(mov, 1.0)
    except ImportError:
        pass
    win = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    ref = (ref - ref.mean()) * win
    mov = (mov - mov.mean()) * win
    try:
        from skimage.registration import phase_cross_correlation
        shift = phase_cross_correlation(ref, mov, upsample_factor=upsample, normalization=None)[0]
        return float(shift[0]), float(shift[1])
    except ImportError:
        pass
    corr = np.fft.ifft2(np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    out = []
    for axis, p in enumerate(peak):
        n = corr.shape[axis]
        idx = [peak[0], peak[1]]
        vals = []
        for k in (p - 1, p, p + 1):
            idx[axis] = k % n
            vals.append(corr[tuple(idx)])
        denom = vals[0] - 2 * vals[1] + vals[2]
        sub = 0.5 * (vals[0] - vals[2]) / denom if denom != 0 else 0.0
        s = p + sub
        out.append(s - n if s > n / 2 else s)
    return float(out[0]), float(out[1])


def shift_image(img: np.ndarray, dy: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Image moved by (dy, dx) and the mask of pixels that still have data."""
    img = np.asarray(img, dtype=np.float64)
    try:
        from scipy.ndimage import shift as nd_shift
        moved = nd_shift(img, (dy, dx), order=1, mode="constant", cval=0.0)
        valid = nd_shift(np.ones_like(img), (dy, dx), order=0, mode="constant", cval=0.0) > 0.5
        return moved, valid
    except ImportError:
        iy, ix = int(round(dy)), int(round(dx))
        moved = np.zeros_like(img)
        valid = np.zeros(img.shape, bool)
        h, w = img.shape
        ys, yd = (slice(0, h - iy), slice(iy, h)) if iy >= 0 else (slice(-iy, h), slice(0, h + iy))
        xs, xd = (slice(0, w - ix), slice(ix, w)) if ix >= 0 else (slice(-ix, w), slice(0, w + ix))
        moved[yd, xd] = img[ys, xs]
        valid[yd, xd] = True
        return moved, valid


def line_profile(img: np.ndarray, p0, p1, half: int = 0, per_px: int = 2,
                 valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Profile from p0 to p1 (image-view coordinates), averaged over 2*half+1 lines 1 px apart.

    Returns the distance from p0 in px and the profile (NaN where there is no data).
    """
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(math.ceil(length * per_px)) + 1)
    t = np.linspace(0.0, 1.0, n)
    ux, uy = ((x1 - x0) / length, (y1 - y0) / length) if length > 0 else (1.0, 0.0)
    o = np.arange(-half, half + 1, dtype=np.float64)[:, None]
    xs = x0 - 0.5 + t * (x1 - x0) - o * uy
    ys = y0 - 0.5 + t * (y1 - y0) + o * ux
    vals = sample_points(img, xs, ys)
    if valid is not None:
        ok = sample_points(valid.astype(np.float64), xs, ys) > 0.999
        vals = np.where(ok, vals, np.nan)
    with np.errstate(all="ignore"), _quiet():
        prof = np.nanmean(vals.reshape(o.size, n), axis=0)
    return t * length, prof


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _display_range(img: np.ndarray) -> tuple[float, float]:
    v = np.asarray(img, dtype=np.float64)
    step = max(1, int(math.sqrt(v.size / 250_000)))
    lo, hi = (float(x) for x in np.nanpercentile(v[::step, ::step], [0.1, 99.9]))
    return lo, (hi if hi > lo else lo + 1.0)


# -----------------------------------------------------------------------------
# stack picker
# -----------------------------------------------------------------------------

class StackPickerDialog(QtWidgets.QDialog):
    """Tick the stacks to compare (from all loaded files)."""

    def __init__(self, files: dict, current: tuple | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line scan / correlation: choose stacks")
        self.resize(560, 420)
        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel("Tick the stacks to compare. They need the same image size (X × Y); timelapses and "
                                "z-stacks can be scanned per frame or as projection.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Stack", "Size", "Pixel", "Time"])
        self.tree.setRootIsDecorated(True)
        self.items: list[tuple[QtWidgets.QTreeWidgetItem, mr.DataStack, str]] = []
        cur_shape = None
        if current is not None:
            s = files[current[0]].stacks[current[1]]
            cur_shape = tuple(s.shape[-2:])
        multi = len(files) > 1
        for key, msr in files.items():
            top = QtWidgets.QTreeWidgetItem([os.path.basename(msr.path)])
            self.tree.addTopLevelItem(top)
            stem = os.path.splitext(os.path.basename(msr.path))[0]
            for pos, s in enumerate(msr.stacks):
                px = s.pixel_size[0]
                label = f"S{pos + 1}  {s.channel_id.split(':')[0] or s.source}"
                it = QtWidgets.QTreeWidgetItem([label, " × ".join(map(str, s.shape)),
                                                f"{px:.4g} µm" if px else "—", s.time])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                same_file = current is not None and key == current[0]
                checked = same_file and (pos == current[1] or tuple(s.shape[-2:]) == cur_shape)
                it.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                top.addChild(it)
                self.items.append((it, s, f"{stem[:24]}… {label}" if multi else label))
            top.setExpanded(True)
            top.setFirstColumnSpanned(True)  # long file names must not widen the stack column
        for c, width in enumerate((190, 120, 95, 80)):
            self.tree.setColumnWidth(c, width)
        lay.addWidget(self.tree, 1)
        self.message = QtWidgets.QLabel()
        self.message.setStyleSheet("color: #ffb347;")
        self.message.setWordWrap(True)
        lay.addWidget(self.message)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Open line scan")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.tree.itemChanged.connect(lambda *_: self._validate())
        self._validate()

    def selection(self) -> tuple[list[mr.DataStack], list[str]]:
        chosen = [(s, name) for it, s, name in self.items if it.checkState(0) == Qt.Checked]
        return [s for s, _ in chosen], [n for _, n in chosen]

    def _validate(self) -> bool:
        stacks, _ = self.selection()
        if not stacks:
            self.message.setText("Tick at least one stack.")
            return False
        shapes = {tuple(s.shape[-2:]) for s in stacks}
        if len(shapes) > 1:
            self.message.setText("The ticked stacks have different image sizes: "
                                 + ", ".join(f"{w} × {h}" for h, w in sorted(shapes)))
            return False
        pxs = [s.pixel_size[0] for s in stacks if s.pixel_size[0]]
        if pxs and (max(pxs) - min(pxs)) > 0.01 * max(pxs):
            self.message.setText("Note: the pixel sizes differ; distances use the first stack's pixel size.")
        else:
            self.message.setText("")
        return True

    def _accept(self) -> None:
        if self._validate():
            self.accept()


# -----------------------------------------------------------------------------
# line ROI
# -----------------------------------------------------------------------------

class LineROI(QtCore.QObject):
    """Line with two end handles, a centre handle to move it, and the averaging band."""

    changed = QtCore.pyqtSignal()

    def __init__(self, scene: QtWidgets.QGraphicsScene):
        super().__init__()
        pen = QtGui.QPen(QtGui.QColor("#ffd400"), 2)
        pen.setCosmetic(True)
        self.line = QtWidgets.QGraphicsLineItem()
        self.line.setPen(pen)
        self.line.setZValue(15)
        self.band = QtWidgets.QGraphicsPolygonItem()
        self.band.setPen(QtGui.QPen(Qt.NoPen))
        self.band.setBrush(QtGui.QColor(255, 212, 0, 40))
        self.band.setZValue(14)
        self.p0 = Handle("#ffd400", radius=6)
        self.p1 = Handle("#ffd400", radius=6)
        self.mid = Handle("#ffd400", radius=5, square=True)
        self.p0.setToolTip("Start of the line (distance 0)")
        self.mid.setToolTip("Drag to move the whole line")
        for it in (self.band, self.line, self.p0, self.p1, self.mid):
            scene.addItem(it)
        self.half = 0
        self._busy = False
        self._center = QtCore.QPointF()
        self.p0.moved.connect(self._end_moved)
        self.p1.moved.connect(self._end_moved)
        self.mid.moved.connect(self._mid_moved)

    def points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (self.p0.pos().x(), self.p0.pos().y()), (self.p1.pos().x(), self.p1.pos().y())

    def set_line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._busy = True
        self.p0.setPos(x0, y0)
        self.p1.setPos(x1, y1)
        self._busy = False
        self._sync()

    def set_half(self, half: int) -> None:
        self.half = half
        self._sync()

    def _end_moved(self, *_) -> None:
        if not self._busy:
            self._sync()

    def _mid_moved(self, x: float, y: float) -> None:
        if self._busy:
            return
        d = QtCore.QPointF(x, y) - self._center
        self._busy = True
        self.p0.setPos(self.p0.pos() + d)
        self.p1.setPos(self.p1.pos() + d)
        self._busy = False
        self._sync()

    def _sync(self) -> None:
        a, b = self.p0.pos(), self.p1.pos()
        self._center = (a + b) / 2
        self._busy = True
        self.mid.setPos(self._center)
        self._busy = False
        self.line.setLine(QtCore.QLineF(a, b))
        length = math.hypot(b.x() - a.x(), b.y() - a.y())
        if self.half > 0 and length > 0:
            n = QtCore.QPointF(-(b.y() - a.y()) / length, (b.x() - a.x()) / length) * (self.half + 0.5)
            self.band.setPolygon(QtGui.QPolygonF([a - n, b - n, b + n, a + n]))
            self.band.show()
        else:
            self.band.hide()
        self.changed.emit()


# -----------------------------------------------------------------------------
# window
# -----------------------------------------------------------------------------

class LineScanWindow(QtWidgets.QMainWindow):
    def __init__(self, stacks: list[mr.DataStack], names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Line scan · " + " / ".join(names))
        self.resize(1400, 820)
        self.channels = []
        for i, (s, name) in enumerate(zip(stacks, names)):
            arr = s.asarray()
            self.channels.append({"stack": s, "arr": arr.reshape((-1,) + arr.shape[-2:]), "name": name,
                                  "color": QtGui.QColor(PALETTE[i % len(PALETTE)]), "enabled": True,
                                  "shift": (0.0, 0.0), "img": None, "valid": None, "range": None, "cache": {}})
        self.h, self.w = stacks[0].shape[-2:]
        self.px = stacks[0].pixel_size[0]
        self.unit = "µm" if self.px else "px"
        self.n_frames = max(ch["arr"].shape[0] for ch in self.channels)
        self.profiles: list[tuple[dict, np.ndarray, np.ndarray]] = []
        self._first_display = True

        # image side
        self.view = ImageView()
        self.view.pixel_size = self.px
        self.roi = LineROI(self.view.scene())
        self.frame_mode = QtWidgets.QComboBox()
        self.frame_mode.addItems(FRAME_MODES)
        self.frame_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, self.n_frames - 1)
        self.frame_label = QtWidgets.QLabel()
        self.frame_label.setMinimumWidth(170)
        frames = QtWidgets.QHBoxLayout()
        frames.addWidget(QtWidgets.QLabel("Frames"))
        frames.addWidget(self.frame_mode)
        frames.addWidget(self.frame_slider, 1)
        frames.addWidget(self.frame_label)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.view, 1)
        self.frames_row = QtWidgets.QWidget()
        self.frames_row.setLayout(frames)
        frames.setContentsMargins(10, 4, 10, 6)
        ll.addWidget(self.frames_row)
        self.frames_row.setVisible(self.n_frames > 1)

        # controls + plot side
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        form = QtWidgets.QGridLayout()
        self.show_combo = QtWidgets.QComboBox()
        self.show_combo.addItem("Composite")
        self.show_combo.addItems([ch["name"] for ch in self.channels])
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(0, 50)
        self.width_spin.setPrefix("± ")
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("Average the profile over parallel lines on both sides")
        self.align = QtWidgets.QCheckBox("Align to first stack")
        self.align.setToolTip("Phase correlation (sub-pixel) of every stack against the first one, per image")
        self.align.setEnabled(len(self.channels) > 1)
        self.normalize = QtWidgets.QCheckBox("Normalize profiles")
        self.normalize.setToolTip("Divide each profile by its maximum")
        form.addWidget(QtWidgets.QLabel("Show"), 0, 0)
        form.addWidget(self.show_combo, 0, 1)
        form.addWidget(QtWidgets.QLabel("Line width"), 0, 2)
        form.addWidget(self.width_spin, 0, 3)
        form.addWidget(self.align, 1, 0, 1, 2)
        form.addWidget(self.normalize, 1, 2, 1, 2)
        form.setColumnStretch(1, 1)
        rl.addLayout(form)

        pg.setConfigOptions(antialias=True, foreground="#c8c8cc")
        self.plot = pg.PlotWidget(background="#1c1c1f")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", f"distance ({self.unit})")
        self.plot.setLabel("left", "intensity")
        self.legend = self.plot.addLegend(offset=(-6, 6), labelTextSize="9pt")
        self.curves = [self.plot.plot(name=ch["name"]) for ch in self.channels]
        rl.addWidget(self.plot, 1)

        self.table = QtWidgets.QTableWidget(len(self.channels), 4)
        self.table.setHorizontalHeaderLabels(["", "colour", "name", "shift"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 30)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 230)
        row_h = self.table.fontMetrics().height() + 10
        self.table.verticalHeader().setDefaultSectionSize(row_h)
        self.table.setFixedHeight(min(6, len(self.channels)) * row_h
                                  + self.table.horizontalHeader().sizeHint().height() + 4)
        self.color_buttons = []
        for r, ch in enumerate(self.channels):
            on = QtWidgets.QTableWidgetItem()
            on.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            on.setCheckState(Qt.Checked)
            self.table.setItem(r, 0, on)
            b = QtWidgets.QPushButton()
            b.setToolTip("Change colour")
            b.clicked.connect(lambda _=False, r=r: self._pick_color(r))
            self.color_buttons.append(b)
            self.table.setCellWidget(r, 1, b)
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(ch["name"]))
            shift = QtWidgets.QTableWidgetItem("—")
            shift.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 3, shift)
            self._paint_color_button(r)
        rl.addWidget(self.table)
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextFormat(Qt.RichText)
        rl.addWidget(self.stats)
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (("Save plot…", self.save_plot), ("Export CSV…", self.save_csv),
                           ("Save image…", self.save_image)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        rl.addLayout(buttons)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([820, 560])
        self.setCentralWidget(splitter)
        self.status = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status, 1)

        self.roi.set_line(self.w * 0.2, self.h * 0.5, self.w * 0.8, self.h * 0.5)
        self.roi.changed.connect(self._update_profiles)
        self.view.mouseMoved.connect(self._on_mouse)
        self.frame_mode.currentIndexChanged.connect(self._frames_changed)
        self.frame_slider.valueChanged.connect(self._frames_changed)
        self.show_combo.currentIndexChanged.connect(self._update_display)
        self.width_spin.valueChanged.connect(self._width_changed)
        self.align.toggled.connect(self._frames_changed)
        self.normalize.toggled.connect(self._update_profiles)
        self.table.itemChanged.connect(self._table_changed)
        self._frames_changed()

    # -- images ----------------------------------------------------------------

    def _raw_image(self, ch: dict) -> np.ndarray:
        mode = self.frame_mode.currentIndex()
        arr = ch["arr"]
        if mode == 0 or arr.shape[0] == 1:
            return np.asarray(arr[min(self.frame_slider.value(), arr.shape[0] - 1)], dtype=np.float64)
        if mode not in ch["cache"]:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                acc = np.zeros(arr.shape[1:], dtype=np.float64)
                for k in range(arr.shape[0]):
                    f = np.asarray(arr[k], dtype=np.float64)
                    acc = acc + f if mode == 1 else np.maximum(acc, f) if k else f
                ch["cache"][mode] = acc / arr.shape[0] if mode == 1 else acc
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        return ch["cache"][mode]

    def _frames_changed(self, *_) -> None:
        mode = self.frame_mode.currentIndex()
        self.frame_slider.setEnabled(mode == 0)
        refs = [self._raw_image(ch) for ch in self.channels]
        for i, (ch, img) in enumerate(zip(self.channels, refs)):
            if self.align.isChecked() and i > 0:
                ch["shift"] = phase_shift(refs[0], img)
                ch["img"], ch["valid"] = shift_image(img, *ch["shift"])
            else:
                ch["shift"] = (0.0, 0.0)
                ch["img"], ch["valid"] = img, None
            if ch["range"] is None or ch.get("range_mode") != mode:
                ch["range"], ch["range_mode"] = _display_range(img), mode
        self._update_frame_label()
        self._update_shift_column()
        self._update_display()
        self._update_profiles()

    def _update_frame_label(self) -> None:
        if self.frame_mode.currentIndex() != 0:
            self.frame_label.setText(f"{FRAME_MODES[self.frame_mode.currentIndex()]} of each stack")
            return
        i = self.frame_slider.value()
        text = f"frame {i + 1} / {self.n_frames}"
        s = self.channels[0]["stack"]
        ts = s.timestamps
        if ts and len(ts) == self.channels[0]["arr"].shape[0] and i < len(ts):
            text += f"   t = {ts[i] - ts[0]:.3f} s"
        self.frame_label.setText(text)

    def _update_shift_column(self) -> None:
        self.table.blockSignals(True)
        for r, ch in enumerate(self.channels):
            dy, dx = ch["shift"]
            if r == 0 or not self.align.isChecked():
                text = "reference" if (r == 0 and self.align.isChecked()) else "—"
            elif self.px:
                text = f"{dx:+.2f}, {dy:+.2f} px  ({dx * self.px:+.3f}, {dy * self.px:+.3f} µm)"
            else:
                text = f"{dx:+.2f}, {dy:+.2f} px"
            self.table.item(r, 3).setText(text)
        self.table.blockSignals(False)

    def _render_rgb(self) -> np.ndarray:
        show = self.show_combo.currentIndex()
        chans = [self.channels[show - 1]] if show > 0 else [ch for ch in self.channels if ch["enabled"]]
        rgb = np.zeros((self.h, self.w, 3), dtype=np.float64)
        for ch in chans:
            lo, hi = ch["range"]
            v = np.clip((np.nan_to_num(ch["img"]) - lo) / (hi - lo), 0.0, 1.0)
            if ch["valid"] is not None:
                v = v * ch["valid"]
            c = ch["color"]
            rgb += v[..., None] * np.array([c.redF(), c.greenF(), c.blueF()])
        return (np.clip(rgb, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)

    def _update_display(self, *_) -> None:
        img = rgb_to_qimage(self._render_rgb())
        self.view.set_pixmap(QtGui.QPixmap.fromImage(img), self._first_display)
        self._first_display = False

    # -- profiles --------------------------------------------------------------

    def _update_profiles(self, *_) -> None:
        p0, p1 = self.roi.points()
        half = self.width_spin.value()
        self.profiles = []
        for ch, curve in zip(self.channels, self.curves):
            if not ch["enabled"] or ch["img"] is None:
                curve.setData([], [])
                continue
            d, prof = line_profile(ch["img"], p0, p1, half, valid=ch["valid"])
            if self.normalize.isChecked():
                with np.errstate(all="ignore"), _quiet():
                    m = np.nanmax(prof) if np.isfinite(prof).any() else np.nan
                prof = prof / m if m and np.isfinite(m) else prof
            coord = d * self.px if self.px else d
            curve.setData(coord, prof, connect="finite", pen=pg.mkPen(ch["color"], width=2))
            self.profiles.append((ch, coord, prof))
        self.plot.setLabel("left", "normalized intensity" if self.normalize.isChecked() else "intensity")
        self._update_legend()
        self._update_stats()
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        size = f"{length * self.px:.2f} µm" if self.px else f"{length:.1f} px"
        self.status.setText(f"line ({p0[0]:.1f}, {p0[1]:.1f}) → ({p1[0]:.1f}, {p1[1]:.1f}) px, length {size}")

    def _update_legend(self) -> None:
        self.legend.clear()
        for ch, curve in zip(self.channels, self.curves):
            if ch["enabled"]:
                self.legend.addItem(curve, ch["name"])

    def _update_stats(self) -> None:
        active = [(ch, prof) for ch, _, prof in self.profiles]
        if len(active) < 2:
            self.stats.setText("")
            return
        rows = []
        for (a, pa), (b, pb) in itertools.combinations(active, 2):
            r_line = pearson(pa, pb)
            va = a["img"] if a["valid"] is None else np.where(a["valid"], a["img"], np.nan)
            vb = b["img"] if b["valid"] is None else np.where(b["valid"], b["img"], np.nan)
            r_img = pearson(va[::2, ::2], vb[::2, ::2])
            rows.append(f"<span style='color:{a['color'].name()}'>{a['name']}</span> – "
                        f"<span style='color:{b['color'].name()}'>{b['name']}</span>: "
                        f"<b>{r_line:.3f}</b> along the line, {r_img:.3f} whole image")
        self.stats.setText("Pearson r<br>" + "<br>".join(rows))

    # -- channel table ---------------------------------------------------------

    def _paint_color_button(self, r: int) -> None:
        self.color_buttons[r].setStyleSheet(f"background-color: {self.channels[r]['color'].name()}; border: 0;")

    def _pick_color(self, r: int) -> None:
        c = QtWidgets.QColorDialog.getColor(self.channels[r]["color"], self, "Colour")
        if c.isValid():
            self.channels[r]["color"] = c
            self._paint_color_button(r)
            self._update_display()
            self._update_profiles()

    def _table_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        r, c = item.row(), item.column()
        ch = self.channels[r]
        if c == 0:
            ch["enabled"] = item.checkState() == Qt.Checked
            self._update_display()
            self._update_profiles()
        elif c == 2 and item.text().strip():
            ch["name"] = item.text().strip()
            self.show_combo.setItemText(r + 1, ch["name"])
            self._update_profiles()

    def _width_changed(self, v: int) -> None:
        self.roi.set_half(v)

    def _on_mouse(self, x: float, y: float) -> None:
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if 0 <= ix < self.w and 0 <= iy < self.h:
            vals = "   ".join(f"{ch['name']} {ch['img'][iy, ix]:.5g}" for ch in self.channels if ch["enabled"])
            pos = f"({(ix + 0.5) * self.px:.2f}, {(iy + 0.5) * self.px:.2f} µm)" if self.px else ""
            self.status.setText(f"x {ix}  y {iy}  {pos}   {vals}")

    # -- output ----------------------------------------------------------------

    def _base_name(self) -> str:
        s = self.channels[0]["stack"]
        return os.path.join(os.path.dirname(s.path), os.path.splitext(os.path.basename(s.path))[0] + "_linescan")

    def _meta_rows(self) -> list[list[str]]:
        p0, p1 = self.roi.points()
        rows = [["stacks", " | ".join(f"{ch['name']} = {ch['stack'].channel_id} ({ch['stack'].time})"
                                      for ch in self.channels if ch["enabled"])],
                ["line (px)", f"({p0[0]:.2f}, {p0[1]:.2f}) -> ({p1[0]:.2f}, {p1[1]:.2f})"],
                ["line width", f"± {self.width_spin.value()} px"],
                ["frames", self.frame_label.text()],
                ["aligned", "yes" if self.align.isChecked() else "no"],
                ["normalized", "yes" if self.normalize.isChecked() else "no"]]
        if self.align.isChecked():
            rows += [[f"shift {ch['name']} (dx, dy px)", f"{ch['shift'][1]:.3f}, {ch['shift'][0]:.3f}"]
                     for ch in self.channels[1:] if ch["enabled"]]
        for (a, pa), (b, pb) in itertools.combinations([(ch, prof) for ch, _, prof in self.profiles], 2):
            rows.append([f"pearson r line {a['name']} / {b['name']}", f"{pearson(pa, pb):.4f}"])
        return rows

    def save_csv(self) -> None:
        if not self.profiles:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export profiles", self._base_name() + ".csv",
                                                        "CSV (*.csv)")
        if not path:
            return
        with open(mr._fs_path(path), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([f"distance_{self.unit}"] + [ch["name"] for ch, _, _ in self.profiles])
            coord = self.profiles[0][1]
            for i in range(coord.size):
                w.writerow([f"{coord[i]:.6g}"] + ["" if not np.isfinite(p[i]) else f"{p[i]:.6g}"
                                                   for _, _, p in self.profiles])
            w.writerow([])
            for r in self._meta_rows():
                w.writerow(["#"] + r)

    def save_plot(self) -> None:
        if not self.profiles:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", self._base_name() + ".png",
                                                        "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:  # no matplotlib: screenshot of the live plot
            pg.exporters.ImageExporter(self.plot.plotItem).export(path)
            return
        fig = Figure(figsize=(6.4, 4.0), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for ch, coord, prof in self.profiles:
            c = ch["color"]
            color = "#000000" if min(c.redF(), c.greenF(), c.blueF()) > 0.9 else c.name()  # white on white
            ax.plot(coord, prof, color=color, lw=1.6, label=ch["name"])
        ax.set_xlabel(f"distance ({self.unit})")
        ax.set_ylabel("normalized intensity" if self.normalize.isChecked() else "intensity (counts)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        with _quiet():
            fig.tight_layout()
        fig.savefig(mr._fs_path(path))

    def save_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", self._base_name() + ".png",
                                                        "PNG (*.png)")
        if not path:
            return
        rgb = self._render_rgb()
        scale = max(1, int(math.ceil(512 / max(self.h, self.w))))
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        img = paint_overlays(rgb_to_qimage(rgb), self.px / scale if self.px else None, "µm", True)
        img = img.convertToFormat(QtGui.QImage.Format_RGB32)
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        (x0, y0), (x1, y1) = self.roi.points()
        p.setPen(QtGui.QPen(QtGui.QColor("#ffd400"), max(2, scale)))
        p.drawLine(QtCore.QPointF(x0 * scale, y0 * scale), QtCore.QPointF(x1 * scale, y1 * scale))
        p.end()
        img.save(path)

    def image_rgb(self) -> np.ndarray:
        """Current display as RGB array (for tests / scripting)."""
        return qimage_to_rgb(rgb_to_qimage(self._render_rgb()))
