"""
msr_stitch.py -- the Mosaic tab of the viewer: stage layout or stitched, blending, registration.

A mosaic (msr_mosaic.Mosaic) is shown as one stack.  This panel switches it between
the stage positions recorded by ImSpector and the stitched layout, where the
overlaps of neighbouring tiles are registered (cross-correlation, as the line scan's
alignment, but for all overlaps at once and solved globally) and blended.  The
registration runs in a background thread; its results (shift and correlation of
every overlap, position of every tile) are listed below.

Only PyQt5 and numpy are needed.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import msr_mosaic as mm


class RegisterThread(QtCore.QThread):
    """Registration (or orientation detection) of one mosaic in the background."""

    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, mosaic: mm.Mosaic, options: mm.RegistrationOptions, orientation: tuple,
                 detect: bool = False, parent=None):
        super().__init__(parent)
        self.mosaic, self.options, self.orientation, self.detect = mosaic, options, orientation, detect
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            report = lambda k, n, text: self.progress.emit(k, n, text)  # noqa: E731
            if self.detect:
                result = mm.detect_orientation(self.mosaic, self.options, report, lambda: self._cancel)
            else:
                result = mm.register(self.mosaic, self.options, self.orientation, report, lambda: self._cancel)
            self.done.emit(result)
        except InterruptedError:
            self.done.emit(None)
        except Exception as exc:  # shown in the panel
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _item(text: str, tip: str = "", align=Qt.AlignLeft | Qt.AlignVCenter) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem(text)
    it.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    it.setTextAlignment(int(align))
    if tip:
        it.setToolTip(tip)
    return it


class MosaicPanel(QtWidgets.QWidget):
    """Layout, blending and registration of the mosaic that is shown."""

    changed = QtCore.pyqtSignal(object)        # a Mosaic whose fused image changed
    outlinesToggled = QtCore.pyqtSignal(bool)
    taskStarted = QtCore.pyqtSignal(object)    # QThread, so the main window waits for it on close

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mosaic: mm.Mosaic | None = None
        self.siblings: list[mm.Mosaic] = []
        self.index_getter = lambda: ()
        self.thread: RegisterThread | None = None
        self._pending_stitch = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.title = QtWidgets.QLabel()
        font = QtGui.QFont(self.title.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.title.setFont(font)
        self.subtitle = QtWidgets.QLabel()
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)

        # -- layout and fusion -----------------------------------------------------------
        box = QtWidgets.QGroupBox("Layout and fusion")
        form = QtWidgets.QFormLayout(box)
        self.stitch = QtWidgets.QCheckBox("Stitch: align the overlapping tiles")
        self.stitch.setToolTip("Off: tiles at the stage positions recorded by ImSpector.\n"
                               "On: the overlaps of neighbouring tiles are registered (cross-correlation),\n"
                               "all shifts are solved together, and the tiles are placed accordingly.")
        self.stitch.toggled.connect(self._stitch_toggled)
        form.addRow(self.stitch)
        self.blend = QtWidgets.QComboBox()
        for key, label in mm.BLEND_MODES.items():
            self.blend.addItem(label, key)
        self.blend.setToolTip("How overlapping tiles are combined.  Linear blending fades from one tile\n"
                              "to the next across the overlap, which hides intensity steps at the seams.")
        self.blend.currentIndexChanged.connect(self._blend_changed)
        form.addRow("Blending", self.blend)
        opts_row = QtWidgets.QHBoxLayout()
        self.subpixel = QtWidgets.QCheckBox("Sub-pixel placement")
        self.subpixel.setToolTip("Place tiles at fractional positions (bilinear interpolation).\n"
                                 "Off: whole pixels, so every pixel outside the overlaps keeps its raw value.")
        self.subpixel.toggled.connect(self._subpixel_toggled)
        self.outlines = QtWidgets.QCheckBox("Tile outlines")
        self.outlines.setToolTip("Draw the outline and name of every tile on the image")
        self.outlines.toggled.connect(self.outlinesToggled)
        opts_row.addWidget(self.subpixel)
        opts_row.addWidget(self.outlines)
        opts_row.addStretch(1)
        form.addRow(opts_row)
        lay.addWidget(box)

        # -- registration ------------------------------------------------------------------
        reg = QtWidgets.QGroupBox("Registration")
        rf = QtWidgets.QFormLayout(reg)
        self.source = QtWidgets.QComboBox()
        self.source.setToolTip("3D: all planes of the overlap are correlated together (most robust for z-stacks).\n"
                               "2D maximum projection: fast, one shift per tile.\n"
                               "2D current plane: only the plane shown in the viewer.")
        self.source.currentIndexChanged.connect(self._source_changed)
        rf.addRow("Use", self.source)
        self.max_shift = QtWidgets.QDoubleSpinBox()
        self.max_shift.setRange(0.0, 100000.0)
        self.max_shift.setDecimals(1)
        self.max_shift.setSuffix(" µm")
        self.max_shift.setSpecialValueText("auto (overlap width)")
        self.max_shift.setToolTip("How far a tile may be from its stage position.  Auto: the width of the overlap.")
        rf.addRow("Search range", self.max_shift)
        self.z_shift = QtWidgets.QSpinBox()
        self.z_shift.setRange(0, 1000)
        self.z_shift.setSuffix(" planes")
        self.z_shift.setSpecialValueText("none")
        self.z_shift.setToolTip("3D only: allow tiles to be shifted in z against each other by up to this many planes")
        rf.addRow("Z shift", self.z_shift)
        self.min_r = QtWidgets.QDoubleSpinBox()
        self.min_r.setRange(0.0, 1.0)
        self.min_r.setSingleStep(0.05)
        self.min_r.setDecimals(2)
        self.min_r.setValue(0.30)
        self.min_r.setToolTip("Overlaps whose normalised cross-correlation (r) is lower are not used; their tiles\n"
                              "follow the median shift of the other overlaps of the same grid step.")
        rf.addRow("Min. correlation", self.min_r)
        orow = QtWidgets.QHBoxLayout()
        self.orientation = QtWidgets.QComboBox()
        for o in mm.ORIENTATIONS:
            self.orientation.addItem(mm.orientation_label(o), o)
        self.orientation.setToolTip("How the stage axes map onto the image.  The default fits TriMScope / ImSpector;\n"
                                    "Detect tries all eight and picks the one whose overlaps correlate best.")
        self.orientation.currentIndexChanged.connect(self._orientation_changed)
        self.detect = QtWidgets.QPushButton("Detect")
        self.detect.setToolTip("Find the orientation from the overlaps")
        self.detect.clicked.connect(lambda: self._start(detect=True))
        orow.addWidget(self.orientation, 1)
        orow.addWidget(self.detect)
        rf.addRow("Orientation", orow)
        brow = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton("Register now")
        self.run.setToolTip("Register the overlaps with these settings and show the stitched mosaic")
        self.run.clicked.connect(lambda: self._start(detect=False))
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self._cancel)
        self.cancel.hide()
        brow.addWidget(self.run)
        brow.addWidget(self.cancel)
        brow.addStretch(1)
        rf.addRow(brow)
        self.progress = QtWidgets.QProgressBar()
        self.progress.hide()
        rf.addRow(self.progress)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rf.addRow(self.summary)
        lay.addWidget(reg)

        # -- results ----------------------------------------------------------------------
        self.results = QtWidgets.QTabWidget()
        self.links = QtWidgets.QTableWidget(0, 4)
        self.links.setHorizontalHeaderLabels(["Overlap", "Shift vs. stage", "r", "Status"])
        self.tiles = QtWidgets.QTableWidget(0, 4)
        self.tiles.setHorizontalHeaderLabels(["Tile", "Stack", "Stage position", "Moved by"])
        for t in (self.links, self.tiles):
            t.verticalHeader().hide()
            t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            t.horizontalHeader().setStretchLastSection(True)
            t.setWordWrap(False)
        self.results.addTab(self.links, "Overlaps")
        self.results.addTab(self.tiles, "Tiles")
        lay.addWidget(self.results, 1)

        # -- output ----------------------------------------------------------------------
        self.out_buttons = QtWidgets.QHBoxLayout()
        lay.addLayout(self.out_buttons)
        more = QtWidgets.QHBoxLayout()
        self.copy_report = QtWidgets.QPushButton("Copy report")
        self.copy_report.setToolTip("Tile positions and all overlap measurements as text")
        self.copy_report.clicked.connect(self._copy_report)
        self.copy_all = QtWidgets.QPushButton("Apply to all mosaics")
        self.copy_all.setToolTip("Use this layout, registration and blending for the other mosaics of the file\n"
                                 "with the same tiles (other detectors), e.g. for a composite")
        self.copy_all.clicked.connect(self._apply_to_siblings)
        more.addWidget(self.copy_report)
        more.addStretch(1)
        more.addWidget(self.copy_all)
        lay.addLayout(more)
        for combo in (self.blend, self.source, self.orientation):  # long entries must not widen the tab
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        self.setEnabled(False)

    # -- public ------------------------------------------------------------------------

    def add_output_actions(self, actions) -> None:
        for a in actions:
            b = QtWidgets.QToolButton()
            b.setDefaultAction(a)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.out_buttons.addWidget(b)

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def set_mosaic(self, mosaic: mm.Mosaic | None, siblings: list | None = None, index_getter=None) -> None:
        if mosaic is not self.mosaic and self.busy:
            self.thread.cancel()
        self.mosaic = mosaic
        self.siblings = [m for m in (siblings or []) if m is not mosaic and mosaic is not None and m.same_grid(mosaic)]
        if index_getter is not None:
            self.index_getter = index_getter
        self.setEnabled(mosaic is not None)
        if mosaic is None:
            return
        first = mosaic.first
        ox, oy = mosaic.overlap_px(mosaic.nominal_positions())
        um = mosaic.px if mosaic.calibrated else None
        self.title.setText(mosaic.title)
        depth = f" × {first.shape[mosaic.zaxis]} planes" if mosaic.zaxis is not None else ""
        over = (f"{ox / mosaic.w * 100:.0f} % ({ox * um:.1f} µm)" if um else f"{ox:.0f} px") if ox else "none"
        self.subtitle.setText(f"{len(mosaic.tiles)} tiles of {mosaic.w} × {mosaic.h} px{depth}, "
                              f"{mosaic.px:.4g} µm/px · overlap {over}"
                              + ("" if all(t.stage for t in mosaic.tiles) else
                                 " · no stage positions in the file: grid from the mosaic settings"))
        self._sync_controls()
        self._fill_tables()

    # -- controls ------------------------------------------------------------------------

    def _sync_controls(self) -> None:
        m = self.mosaic
        widgets = (self.stitch, self.blend, self.subpixel, self.source, self.orientation)
        for w in widgets:
            w.blockSignals(True)
        self.stitch.setChecked(m.layout == "aligned")
        self.blend.setCurrentIndex(max(0, self.blend.findData(m.blend)))
        self.subpixel.setChecked(m.subpixel)
        self.source.clear()
        if m.zaxis is not None:
            self.source.addItem("3D: all planes", "volume")
        self.source.addItem("2D: maximum projection", "max")
        self.source.addItem("2D: current plane", "plane")
        if m.registration is not None:
            i = self.source.findData(m.registration.options.source)
            self.source.setCurrentIndex(max(0, i))
        self.orientation.setCurrentIndex(max(0, mm.ORIENTATIONS.index(tuple(m.orientation))))
        for w in widgets:
            w.blockSignals(False)
        self._source_changed()
        busy = self.busy
        self.run.setEnabled(not busy)
        self.detect.setEnabled(not busy)
        self.copy_all.setVisible(bool(self.siblings))

    def _source_changed(self, *_) -> None:
        self.z_shift.setEnabled(self.source.currentData() == "volume")

    def options(self) -> mm.RegistrationOptions:
        src = self.source.currentData() or "max"
        plane = tuple(self.index_getter()) if src == "plane" else ()
        return mm.RegistrationOptions(source=src, plane=plane, max_shift_um=self.max_shift.value(),
                                      max_z_shift=self.z_shift.value() if src == "volume" else 0,
                                      min_r=self.min_r.value())

    def _stitch_toggled(self, on: bool) -> None:
        m = self.mosaic
        if m is None:
            return
        if on and m.registration is None:
            self._start(detect=False)
            return
        m.set_layout("aligned" if on else "stage")
        self._fill_tables()
        self.changed.emit(m)

    def _blend_changed(self, *_) -> None:
        if self.mosaic is not None:
            self.mosaic.set_blend(self.blend.currentData())
            self.changed.emit(self.mosaic)

    def _subpixel_toggled(self, on: bool) -> None:
        if self.mosaic is not None:
            self.mosaic.set_subpixel(on)
            self.changed.emit(self.mosaic)

    def _orientation_changed(self, *_) -> None:
        m = self.mosaic
        if m is None:
            return
        o = tuple(self.orientation.currentData())
        if o != tuple(m.orientation):
            m.set_orientation(o)
            self.stitch.blockSignals(True)
            self.stitch.setChecked(False)
            self.stitch.blockSignals(False)
            self._fill_tables()
            self.changed.emit(m)

    # -- registration -------------------------------------------------------------------

    def _start(self, detect: bool) -> None:
        m = self.mosaic
        if m is None or self.busy:
            return
        self.thread = RegisterThread(m, self.options(), tuple(self.orientation.currentData()), detect, self)
        self.thread.progress.connect(self._progress)
        self.thread.done.connect(lambda result, m=m, detect=detect: self._done(m, result, detect))
        self.thread.failed.connect(self._failed)
        self.thread.finished.connect(self._finished)
        for w in (self.run, self.detect, self.stitch, self.orientation, self.source):
            w.setEnabled(False)
        self.cancel.show()
        self.progress.setRange(0, 0)
        self.progress.show()
        self.summary.setText("Detecting the orientation…" if detect else "Registering the overlaps…")
        self.taskStarted.emit(self.thread)
        self.thread.start()

    def _progress(self, k: int, n: int, text: str) -> None:
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(k)
        self.progress.setFormat(f"{k} / {n}  {text}")

    def _cancel(self) -> None:
        if self.busy:
            self.thread.cancel()

    def _finished(self) -> None:
        self.cancel.hide()
        self.progress.hide()
        for w in (self.run, self.detect, self.stitch, self.orientation, self.source):
            w.setEnabled(True)
        self.thread = None

    def _failed(self, msg: str) -> None:
        self.summary.setText(f"<span style='color:#ff7070'>Registration failed: {msg}</span>")
        if self.mosaic is not None:
            self.stitch.blockSignals(True)
            self.stitch.setChecked(self.mosaic.layout == "aligned")
            self.stitch.blockSignals(False)

    def _done(self, m: mm.Mosaic, result, detect: bool) -> None:
        if result is None:  # cancelled
            if m is self.mosaic:
                self.summary.setText("Cancelled.")
                self._sync_controls()
            return
        if detect:
            best_r, best = result[0]
            current_r = next((r for r, o in result if tuple(o) == tuple(m.orientation)), 0.0)
            lines = "<br>".join(f"{r:.2f}  {mm.orientation_label(o)}" for r, o in result)
            if m is self.mosaic:
                keep = tuple(best) != tuple(m.orientation) and current_r >= best_r - 0.02  # a tie: no change
                if not keep:
                    self.orientation.setCurrentIndex(mm.ORIENTATIONS.index(tuple(best)))
                self.summary.setText(f"Mean overlap correlation per orientation (maximum projections):<br>{lines}"
                                     + ("<br>The current orientation is as good; kept." if keep else ""))
            return
        m.set_registration(result)
        if m is self.mosaic:
            self._sync_controls()
            self._fill_tables()
        self.changed.emit(m)

    # -- results ---------------------------------------------------------------------------

    def _fill_tables(self) -> None:
        m = self.mosaic
        reg = m.registration
        unit_um = m.calibrated
        if reg is None:
            self.summary.setText("Tiles at the stage positions.  Tick <b>Stitch</b> to register the overlaps.")
        else:
            self.summary.setText(("<b>Stitched.</b> " if m.layout == "aligned" else "Registered (not shown). ")
                                 + reg.summary)
        # overlaps
        links = reg.links if reg is not None else []
        self.links.setRowCount(len(links))
        for r, lk in enumerate(links):
            name = f"{m.tile_label(lk.i)} → {m.tile_label(lk.j)}"
            if lk.measured is not None:
                dx, dy = lk.measured[0] - lk.nominal[0], lk.measured[1] - lk.nominal[1]
                shift = (f"{dx * m.px:+.2f}, {dy * m.py:+.2f} µm" if unit_um else f"{dx:+.2f}, {dy:+.2f} px")
                if abs(lk.measured[2]) >= 0.05:
                    shift += f", z {lk.measured[2]:+.2f}"
                tip = (f"measured displacement {lk.measured[0]:.2f}, {lk.measured[1]:.2f} px "
                       f"(stage: {lk.nominal[0]:.2f}, {lk.nominal[1]:.2f} px)\noverlap {lk.overlap_px} px per plane")
            else:
                shift, tip = "–", ""
            status = "used" if lk.used else "not used"
            if np.isfinite(lk.residual):
                status += f", residual {lk.residual:.2f} px"
            if lk.note:
                status += f" ({lk.note})"
            color = QtGui.QColor("#e0e0e0" if lk.used else "#9a9a9a")
            items = [_item(name), _item(shift, tip), _item(f"{lk.r:.2f}", align=Qt.AlignRight | Qt.AlignVCenter),
                     _item(status, status)]
            for c, it in enumerate(items):
                it.setForeground(color)
                self.links.setItem(r, c, it)
        # tiles
        pos = m.positions()
        nominal = m.nominal_positions()
        moved = np.zeros((len(m.tiles), 2))
        if reg is not None and m.layout == "aligned":
            moved = (np.asarray(reg.positions)[:, :2] - nominal[:, :2])
            moved -= np.median(moved, axis=0)  # relative to the mosaic as a whole
        self.tiles.setRowCount(len(m.tiles))
        for r, t in enumerate(m.tiles):
            stage = f"{t.stage[0]:.2f}, {t.stage[1]:.2f} µm" if t.stage else "–"
            mv = math.hypot(moved[r, 0] * m.px, moved[r, 1] * m.py) if unit_um else math.hypot(*moved[r])
            tip = f"origin in the mosaic: x {pos[r, 0]:.2f}, y {pos[r, 1]:.2f} px" + (
                f", z {pos[r, 2]:+.2f} planes" if pos[r, 2] else "")
            items = [_item(m.tile_label(r), tip), _item(f"S{t.pos + 1}", t.stack.source), _item(stage),
                     _item(f"{mv:.2f} {'µm' if unit_um else 'px'}", tip, Qt.AlignRight | Qt.AlignVCenter)]
            for c, it in enumerate(items):
                self.tiles.setItem(r, c, it)
        for t in (self.links, self.tiles):
            t.resizeColumnsToContents()

    def _apply_to_siblings(self) -> None:
        m = self.mosaic
        if m is None:
            return
        for other in self.siblings:
            other.copy_layout(m)
            self.changed.emit(other)
        self.summary.setText(self.summary.text() + f"<br>Applied to {len(self.siblings)} other mosaic(s).")

    def _copy_report(self) -> None:
        if self.mosaic is not None:
            QtWidgets.QApplication.clipboard().setText(mm.report(self.mosaic))
