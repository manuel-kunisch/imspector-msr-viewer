"""
msr_volume_panel.py -- the 3D tab of the viewer: channels, contrast, rendering and view of the 3D view.

Controls for msr_volume_view.VolumeView: per channel visibility, colour or lookup
table, intensity window (min / max / gamma, edited on the histogram with the transfer
function drawn on top) and weight; the rendering algorithm, sampling, background and
overlays; view presets, rotation and zoom; the resolution of the volume and snapshots.
The panel edits the view's ChannelSettings / RenderSettings / Camera in place and
then calls view.settings_changed(); it does not import the view module, so it works
with any object that has the same attributes and signals (the tests use a stub).

The controls follow the interactive raycaster of the Fiji plugin 3Dscript
(B. Schmid et al., "3Dscript: animating 3D/4D microscopy data using a
natural-language-based syntax", Nature Methods 16:278-280, 2019;
https://github.com/bene51/3Dscript), Copyright (c) 2016, Benjamin Schmid, under the
BSD 2-Clause License; see THIRD_PARTY_NOTICES.md for the full license text.

Only PyQt5 and numpy are needed.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from contextlib import contextmanager

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from msr_volume import ALGORITHM_LABELS, ALGORITHMS, MAX_CHANNELS, ChannelSettings, Volume, VolumeChannel
from msr_widgets import LUTS, ElidedLabel, nice_length, rgb_to_qimage

# colour menu of a channel: plain colours (black to colour ramps) ...
COLORS = (("Red", "#ff0000"), ("Green", "#00ff00"), ("Blue", "#0000ff"), ("Cyan", "#00ffff"),
          ("Magenta", "#ff00ff"), ("Yellow", "#ffff00"), ("Gray", "#ffffff"))
# ... and the lookup tables that are not ramps (ChannelSettings.lut)
LUT_NAMES = ("Fire", "Viridis", "HiLo", "Inverted")
SAMPLING = (("Fast", 0.5), ("Normal", 1.0), ("Fine", 2.0))           # samples per voxel along the ray
RESOLUTIONS = (("Auto", "auto"), ("Full", "full"), ("½", "half"), ("¼", "quarter"))
HINT_STYLE = "color: #9a9a9a;"
HANDLE_COLOR = QtGui.QColor(255, 196, 0)     # as the 2D histogram's handles
ALPHA_COLOR = QtGui.QColor(79, 140, 255)     # opacity curve (3Dscript draws it blue)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.4g}"


def _qcolor(rgb) -> QtGui.QColor:
    """(r, g, b) in 0..1 -> QColor."""
    return QtGui.QColor.fromRgbF(*(min(max(float(c), 0.0), 1.0) for c in tuple(rgb)[:3]))


def _rgb(c: QtGui.QColor) -> tuple[float, float, float]:
    return (c.redF(), c.greenF(), c.blueF())


def _framed(pix: QtGui.QPixmap) -> QtGui.QPixmap:
    """A thin grey frame, so black and dark swatches stay visible on the dark background."""
    p = QtGui.QPainter(pix)
    p.setPen(QtGui.QColor(110, 110, 116))
    p.drawRect(0, 0, pix.width() - 1, pix.height() - 1)
    p.end()
    return pix


def _color_icon(color: QtGui.QColor, w: int = 30, h: int = 14) -> QtGui.QIcon:
    pix = QtGui.QPixmap(w, h)
    pix.fill(color)
    return QtGui.QIcon(_framed(pix))


def _lut_icon(table: np.ndarray, w: int = 30, h: int = 14) -> QtGui.QIcon:
    cols = np.asarray(table)[np.linspace(0, 255, w).round().astype(int)]
    pix = QtGui.QPixmap.fromImage(rgb_to_qimage(np.repeat(cols[None], h, 0)))
    return QtGui.QIcon(_framed(pix))


def _channel_icon(cs: ChannelSettings, w: int = 30, h: int = 14) -> QtGui.QIcon:
    """Swatch of a channel: its colour, or the gradient of its lookup table."""
    if cs.lut and cs.lut in LUTS:
        return _lut_icon(LUTS[cs.lut], w, h)
    return _color_icon(_qcolor(cs.color), w, h)


def _tint(cs: ChannelSettings) -> QtGui.QColor:
    """Histogram colour of a channel: its colour (the middle of a LUT), dark colours lightened."""
    r, g, b = (float(v) for v in cs.lut_table()[128])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    f = min(max((110.0 - luma) / max(255.0 - luma, 1.0), 0.0), 1.0)   # mix with white up to a minimum brightness
    return QtGui.QColor(*(int(round(c + (255.0 - c) * f)) for c in (r, g, b)))


def _no_drops(widget: QtWidgets.QWidget) -> None:
    """Let file drops reach the main window (line edits inside spin boxes accept text drops)."""
    for w in [widget] + widget.findChildren(QtWidgets.QWidget):
        w.setAcceptDrops(False)


class _WheelGuard(QtCore.QObject):
    """Event filter: the wheel edits a combo box, spin box or slider only while it has the keyboard focus.

    Otherwise the event is ignored and goes on to the scroll area of the tab, which scrolls.
    """

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


def _quiet_focus(widget: QtWidgets.QWidget, guard: _WheelGuard) -> None:
    """Controls of *widget* that do not take the keyboard from the 3D view (its arrow keys, R, F, +/-).

    Spin boxes take the focus on a click (to type into) but not from the wheel; combo boxes,
    buttons, check boxes and sliders only by Tab.  The wheel acts on a control only while it
    has the focus (_WheelGuard), so scrolling the tab does not change Resolution or contrast.
    """
    for w in [widget] + widget.findChildren(QtWidgets.QWidget):
        if isinstance(w, QtWidgets.QAbstractButton):
            w.setFocusPolicy(Qt.TabFocus)
        elif isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QSlider)):
            w.setFocusPolicy(Qt.StrongFocus if isinstance(w, QtWidgets.QAbstractSpinBox) else Qt.TabFocus)
            w.installEventFilter(guard)   # (installing it again keeps one)


@contextmanager
def _blocked(*objects):
    """Block the signals of *objects* for programmatic updates."""
    old = [o.blockSignals(True) for o in objects]
    try:
        yield
    finally:
        for o, was in zip(objects, old):
            o.blockSignals(was)


def _tool_button(text: str, tip: str, slot) -> QtWidgets.QToolButton:
    """Text button that may shrink below Fusion's 80 px push-button minimum (rows of several in a narrow tab)."""
    b = QtWidgets.QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setToolButtonStyle(Qt.ToolButtonTextOnly)
    b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    b.clicked.connect(slot)
    return b


def _volume_text(v: Volume) -> str:
    """Size, voxel size and downsampling of a volume for the info line."""
    d, h, w = v.shape
    n = len(v.channels)
    unit = v.unit or "µm"
    ex = v.extent
    lines = [f"{w} × {h} × {d} voxels · {n} channel{'s' if n != 1 else ''} · {v.nbytes / 2 ** 20:.0f} MB",
             f"voxel {v.voxel[0]:.4g} × {v.voxel[1]:.4g} × {v.voxel[2]:.4g} {unit} · "
             f"box {ex[0]:.4g} × {ex[1]:.4g} × {ex[2]:.4g} {unit}" + ("" if v.calibrated else " (uncalibrated)")]
    if tuple(v.factor) != (1, 1, 1):
        sd, sh, sw = v.source_shape
        fx, fy, fz = v.factor
        lines.append(f"binned {fx} × {fy} × {fz} from {sw} × {sh} × {sd} voxels")
    if v.note:
        lines.append(v.note)
    return "\n".join(lines)


class _ValueSpin(QtWidgets.QDoubleSpinBox):
    """Spin box for data values whose size hint does not grow with its (wide) range."""

    def sizeHint(self) -> QtCore.QSize:
        s = super().sizeHint()
        return QtCore.QSize(min(s.width(), 110), s.height())

    def minimumSizeHint(self) -> QtCore.QSize:
        s = super().minimumSizeHint()
        return QtCore.QSize(min(s.width(), 64), s.height())


class _AngleSpin(QtWidgets.QDoubleSpinBox):
    """Angle in degrees, -180..180; stepping past either end wraps round (175 + 10 = -175)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-180.0, 180.0)
        self.setWrapping(True)    # keeps the arrows enabled at the ends; stepBy does the wrapping

    def stepBy(self, steps: int) -> None:
        self.interpretText()      # a typed value not yet committed (keyboard tracking off): step from it
        v = (self.value() + steps * self.singleStep() + 180.0) % 360.0 - 180.0
        self.setValue(180.0 if v == -180.0 and steps > 0 else v)   # 175 + 5 = 180, not -180


# -----------------------------------------------------------------------------
# transfer function
# -----------------------------------------------------------------------------

class TransferFunctionWidget(QtWidgets.QWidget):
    """Histogram of one channel with its transfer function: min / max handles, brightness and opacity curves.

    The x axis spans the histogram range, widened to contain the window (fixed while
    dragging).  The bars are log1p counts tinted in the channel colour; the light grey
    curve is the brightness t^gamma between min and max, the dashed blue one the linked
    opacity (alpha gamma), the strip below the colour every value is shown with.  A
    handle (5 px tolerance) moves min or max, dragging between them moves the window, a
    click elsewhere moves the nearer handle there; *windowEdited* reports every step.
    """

    windowEdited = QtCore.pyqtSignal(float, float)   # min, max, while dragging
    TOLERANCE = 5.0                                  # px around a handle that grab it

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._counts: np.ndarray | None = None       # histogram bin counts
        self._h0, self._h1 = 0.0, 1.0                # histogram range
        self._bars: tuple[np.ndarray, np.ndarray] | None = None   # (edges in values, log1p heights) as drawn
        self._lo, self._hi, self._gamma = 0.0, 1.0, 1.0
        self._alo, self._ahi, self._agamma = 0.0, 1.0, 2.0
        self._table = LUTS["Gray"]
        self._tint = QtGui.QColor(200, 200, 204)
        self._integer = False
        self._empty = True                              # no channel: nothing to draw or drag
        self._axis: tuple[float, float] | None = None   # frozen while dragging
        self._drag: str | None = None                   # "lo" | "hi" | "both" | "either"
        self._hover: str | None = None
        self._mouse_x: float | None = None
        self._press_x = 0.0
        self._press_window = (0.0, 1.0)
        self.setToolTip("Drag the yellow lines to set Min / Max, drag between them to move the window.\n"
                        "Grey curve: brightness (gamma).  Dashed blue: opacity, which follows the window\n"
                        "(dim voxels are more transparent).  The strip shows the colour of each value.")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(300, 110)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(120, 110)

    # -- data ----------------------------------------------------------------------------

    def set_histogram(self, counts, lo: float, hi: float) -> None:
        """Bin counts over [lo, hi] (None: no histogram, only the axis)."""
        self._counts = None if counts is None else np.asarray(counts, dtype=np.float64)
        self._h0 = float(lo)
        self._h1 = float(hi) if hi > lo else float(lo) + 1.0
        self._bars = None
        self.update()

    def set_channel(self, cs: ChannelSettings) -> None:
        """Window, gammas and colours of *cs*."""
        self._lo, self._hi, self._gamma = float(cs.cmin), float(cs.cmax), float(cs.cgamma)
        self._alo, self._ahi, self._agamma = float(cs.amin), float(cs.amax), float(cs.agamma)
        self._table = cs.lut_table()
        self._tint = _tint(cs)
        self._empty = False
        self.update()

    def set_integer(self, on: bool) -> None:
        """Integer data: dragged values are whole numbers, the histogram is drawn per value."""
        if bool(on) != self._integer:
            self._integer = bool(on)
            self._bars = None
            self.update()

    def clear(self) -> None:
        """No channel: an empty plot that ignores the mouse."""
        self._counts = self._bars = None
        self._drag = self._axis = self._hover = self._mouse_x = None
        self._empty = True
        self.update()

    def window(self) -> tuple[float, float]:
        return self._lo, self._hi

    def axis_range(self) -> tuple[float, float]:
        """Values at the left and right end of the plot."""
        if self._axis is not None:
            return self._axis
        x0 = min(self._h0, self._lo, self._alo)
        x1 = max(self._h1, self._hi, self._ahi)
        return (x0, x1) if x1 > x0 else (x0, x0 + 1.0)

    def value_to_x(self, v: float) -> float:
        """Widget x coordinate of a value (also for tests)."""
        return self._to_x(v, self._plot_rect(), self.axis_range())

    # -- geometry ------------------------------------------------------------------------

    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.rect()).adjusted(8, 7, -8, -28)

    @staticmethod
    def _to_x(v: float, r: QtCore.QRectF, axis: tuple[float, float]) -> float:
        return r.left() + (v - axis[0]) / (axis[1] - axis[0]) * r.width()

    @staticmethod
    def _to_v(x: float, r: QtCore.QRectF, axis: tuple[float, float]) -> float:
        return axis[0] + (x - r.left()) / max(r.width(), 1.0) * (axis[1] - axis[0])

    def _histogram_bars(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Bin edges (values) and log1p heights to draw.

        For integer data the bins are narrower than or not a multiple of one count
        (512 bins over e.g. 0..255 or 0..3200), so some hold no whole number or one more
        than their neighbours: they are drawn as counts per value, a bin without a whole
        number joining the previous one (no comb, no ripple).
        """
        if self._bars is None and self._counts is not None and self._counts.size:
            c = self._counts
            edges = np.linspace(self._h0, self._h1, c.size + 1)
            if self._integer:
                first = np.ceil(edges[:-1] - 1e-9)
                per_bin = np.ceil(edges[1:] - 1e-9) - first
                per_bin[-1] = np.floor(edges[-1] + 1e-9) - first[-1] + 1     # numpy's last bin is closed
                keep = (per_bin > 0) | (c > 0)
                c = c[keep] / np.maximum(per_bin[keep], 1)
                edges = np.append(edges[:-1][keep], edges[-1])
            self._bars = (edges, np.log1p(c))
        return self._bars

    def _hit(self, x: float) -> str | None:
        """What a press at *x* grabs: a handle, the window ("both"), or nothing."""
        r, axis = self._plot_rect(), self.axis_range()
        xl, xh = self._to_x(self._lo, r, axis), self._to_x(self._hi, r, axis)
        dl, dh = abs(x - xl), abs(x - xh)
        if dl <= self.TOLERANCE and dh <= self.TOLERANCE:
            if abs(xh - xl) < 1.0:
                return "either"          # the lines coincide: the drag direction decides
            return "lo" if dl < dh else "hi"
        if dl <= self.TOLERANCE:
            return "lo"
        if dh <= self.TOLERANCE:
            return "hi"
        return "both" if xl < x < xh else None

    # -- painting ------------------------------------------------------------------------

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(24, 24, 26))
        r = self._plot_rect()
        if self._empty or r.width() < 4 or r.height() < 4:
            return
        axis = self.axis_range()
        x0, x1 = axis
        # histogram
        bars = self._histogram_bars()
        if bars is not None and bars[1].size and bars[1].max() > 0:
            edges = r.left() + (bars[0] - x0) / (x1 - x0) * r.width()
            ys = r.bottom() - bars[1] / bars[1].max() * r.height()
            poly = QtGui.QPolygonF()
            poly.append(QtCore.QPointF(edges[0], r.bottom()))
            for i in range(ys.size):
                poly.append(QtCore.QPointF(edges[i], ys[i]))
                poly.append(QtCore.QPointF(edges[i + 1], ys[i]))
            poly.append(QtCore.QPointF(edges[-1], r.bottom()))
            path = QtGui.QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            p.save()
            p.setClipRect(r)
            tint = QtGui.QColor(self._tint)
            tint.setAlpha(165)
            p.fillPath(path, tint)
            p.restore()
        p.setPen(QtGui.QColor(62, 62, 68))
        p.drawLine(QtCore.QPointF(r.left(), r.bottom()), QtCore.QPointF(r.right(), r.bottom()))
        # outside the window: shaded
        xl = min(max(self._to_x(self._lo, r, axis), r.left()), r.right())
        xh = min(max(self._to_x(self._hi, r, axis), r.left()), r.right())
        shade = QtGui.QColor(0, 0, 0, 115)
        p.fillRect(QtCore.QRectF(r.left(), r.top(), xl - r.left(), r.height()), shade)
        p.fillRect(QtCore.QRectF(xh, r.top(), r.right() - xh, r.height()), shade)
        # curves, sampled per pixel column
        w = max(2, int(r.width()))
        frac = np.arange(w + 1) / w
        values = x0 + frac * (x1 - x0)
        xs = r.left() + frac * r.width()

        def curve(lo: float, hi: float, g: float) -> QtGui.QPolygonF:
            t = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0) ** g
            ys = r.bottom() - t * r.height()
            return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in zip(xs, ys)])

        p.setRenderHint(QtGui.QPainter.Antialiasing)
        alpha = QtGui.QColor(ALPHA_COLOR)
        alpha.setAlpha(190)
        p.setPen(QtGui.QPen(alpha, 1.3, Qt.DashLine))
        p.drawPolyline(curve(self._alo, self._ahi, self._agamma))
        p.setPen(QtGui.QPen(QtGui.QColor(228, 228, 232), 1.6))
        p.drawPolyline(curve(self._lo, self._hi, self._gamma))
        # colour strip: the colour each value is shown with
        mid = values[:-1] + (values[1] - values[0]) / 2
        t = np.clip((mid - self._lo) / max(self._hi - self._lo, 1e-12), 0.0, 1.0) ** self._gamma
        rgb = np.asarray(self._table)[(t * 255 + 0.5).astype(int)]
        p.drawImage(QtCore.QRectF(r.left(), r.bottom() + 4, r.width(), 8), rgb_to_qimage(rgb[None]))
        # handles: lines with a grip on top
        for which, x in (("lo", xl), ("hi", xh)):
            grabbed = self._drag if self._drag is not None else self._hover
            active = grabbed in (which, "either") or self._drag == "both"
            p.setPen(QtGui.QPen(HANDLE_COLOR, 2.6 if active else 1.7))
            p.drawLine(QtCore.QPointF(x, r.top()), QtCore.QPointF(x, r.bottom() + 12))
            p.setPen(Qt.NoPen)
            p.setBrush(HANDLE_COLOR)
            p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x - 4.5, r.top() - 6), QtCore.QPointF(x + 4.5, r.top() - 6),
                                           QtCore.QPointF(x, r.top())]))
        # axis ends, and the value under the mouse
        font = QtGui.QFont(p.font())
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QtGui.QColor(140, 140, 146))
        ty = r.bottom() + 14
        half = r.width() / 2
        p.drawText(QtCore.QRectF(r.left(), ty, half, 13), Qt.AlignLeft | Qt.AlignTop, _fmt(x0))
        p.drawText(QtCore.QRectF(r.left() + half, ty, half, 13), Qt.AlignRight | Qt.AlignTop, _fmt(x1))
        if self._mouse_x is not None and r.left() <= self._mouse_x <= r.right():
            v = self._to_v(self._mouse_x, r, axis)
            p.setPen(QtGui.QColor(200, 200, 205))
            p.drawText(QtCore.QRectF(r.left(), ty, r.width(), 13), Qt.AlignHCenter | Qt.AlignTop,
                       _fmt(round(v) if self._integer else v))

    # -- mouse ---------------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton or self._empty:
            return
        x = ev.localPos().x()
        r = self._plot_rect()
        self._axis = self.axis_range()
        kind = self._hit(x)
        self._press_x = x
        self._press_window = (self._lo, self._hi)
        if kind is None:  # outside the window: the nearer handle jumps here, then follows the mouse
            kind = "lo" if x < self._to_x(self._lo, r, self._axis) else "hi"
            v = min(max(self._to_v(x, r, self._axis), self._axis[0]), self._axis[1])
            self._press_window = (v, self._hi) if kind == "lo" else (self._lo, v)
        self._drag = kind
        self.setCursor(Qt.ClosedHandCursor if kind == "both" else Qt.SizeHorCursor)
        self._drag_to(x)

    def mouseMoveEvent(self, ev) -> None:
        x = ev.localPos().x()
        if self._drag is not None:
            if ev.buttons() & Qt.LeftButton:
                self._drag_to(x)
                return
            self._drag = self._axis = None   # the release was missed (e.g. a dialog opened during the drag)
        if self._empty:
            return
        self._mouse_x = x
        hover = self._hit(x)
        if hover != self._hover:
            self._hover = hover
        self.setCursor(Qt.SizeHorCursor if hover in ("lo", "hi", "either") else
                       Qt.OpenHandCursor if hover == "both" else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton or self._drag is None:
            return
        self._drag_to(ev.localPos().x())
        self._drag = None
        self._axis = None             # the axis may follow the window again
        self._hover = self._hit(ev.localPos().x())
        self.update()

    def leaveEvent(self, ev) -> None:
        self._mouse_x = None
        self._hover = None
        self.update()
        super().leaveEvent(ev)

    def _drag_to(self, x: float) -> None:
        """Move the grabbed handle(s) by the mouse movement since the press."""
        axis = self._axis or self.axis_range()
        x0, x1 = axis
        r = self._plot_rect()
        dv = (x - self._press_x) / max(r.width(), 1.0) * (x1 - x0)
        lo, hi = self._press_window
        gap = 1.0 if self._integer else (x1 - x0) * 1e-4
        kind = self._drag
        if kind == "either":
            if abs(x - self._press_x) < 1.0:
                return
            kind = self._drag = "hi" if x > self._press_x else "lo"
        if kind == "both":
            width = hi - lo
            lo = min(max(lo + dv, x0), max(x1 - width, x0))
            if self._integer:
                lo = float(round(lo))
            hi = lo + width
        elif kind == "lo":
            lo = min(max(lo + dv, x0), hi - gap)
            if self._integer:
                lo = float(min(round(lo), math.floor(hi - gap)))
        else:
            hi = max(min(hi + dv, x1), lo + gap)
            if self._integer:
                hi = float(max(round(hi), math.ceil(lo + gap)))
        if (lo, hi) != (self._lo, self._hi):
            self._lo, self._hi = lo, hi
            self.update()
            self.windowEdited.emit(lo, hi)


# -----------------------------------------------------------------------------
# channel list
# -----------------------------------------------------------------------------

class _ChannelRow(QtWidgets.QWidget):
    """One channel: visibility, colour menu, name and window; a click selects it for the contrast editor."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = False
        self._hover = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(5, 2, 6, 2)
        lay.setSpacing(6)
        self.visible = QtWidgets.QCheckBox()
        self.visible.setToolTip("Show this channel in 3D")
        self.color = QtWidgets.QToolButton()
        self.color.setIconSize(QtCore.QSize(30, 14))
        self.color.setAutoRaise(True)
        self.color.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.color.setToolTip("Colour or lookup table of this channel")
        self.name = ElidedLabel()
        self.levels = QtWidgets.QLabel()
        self.levels.setStyleSheet(HINT_STYLE)
        self.levels.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.visible)
        lay.addWidget(self.color)
        lay.addWidget(self.name, 1)
        lay.addWidget(self.levels)

    def set_selected(self, on: bool) -> None:
        if on == self.selected and self.name.font().bold() == on:
            return
        self.selected = on
        font = QtGui.QFont(self.name.font())
        font.setBold(on)
        self.name.setFont(font)
        self.update()

    def set_dimmed(self, on: bool) -> None:
        color = QtGui.QColor("#9a9a9a") if on else self.palette().color(QtGui.QPalette.WindowText)
        pal = self.name.palette()
        if pal.color(QtGui.QPalette.WindowText) != color:
            pal.setColor(QtGui.QPalette.WindowText, color)
            self.name.setPalette(pal)
            self.name.update()

    def paintEvent(self, ev) -> None:
        if self.selected or self._hover:
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(40, 110, 190, 120) if self.selected else QtGui.QColor(255, 255, 255, 14))
            p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

    def enterEvent(self, ev) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, ev) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()


# -----------------------------------------------------------------------------
# panel
# -----------------------------------------------------------------------------

class VolumePanel(QtWidgets.QWidget):
    """Controls of the 3D view (the viewer's 3D tab); edits the view's settings in place.

    *view* needs the VolumeView attributes channels, settings, camera, volume, the
    signal cameraChanged and the methods settings_changed() and set_preset(name).
    The info line shows the volume (set_volume) and, below it, the text of set_info
    (e.g. the GPU or an error).
    """

    resolutionChanged = QtCore.pyqtSignal(str)    # "auto" | "full" | "half" | "quarter"
    fromTwoDRequested = QtCore.pyqtSignal(int)    # channel index: copy the 2D display range (viewer, then refresh())
    saveRequested = QtCore.pyqtSignal()           # "Save PNG…"
    copyRequested = QtCore.pyqtSignal()           # "Copy image"
    cancelRequested = QtCore.pyqtSignal()         # cancel loading
    reloadRequested = QtCore.pyqtSignal()         # "Reload" after a cancelled or failed load (show_reload)
    overlayChanged = QtCore.pyqtSignal()          # bounding box, scale bar or axes switched in the panel
    compositeRequested = QtCore.pyqtSignal(bool)  # composite line: True all channels of this size, False selected only

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.volume: Volume | None = None
        self.current = 0                          # channel shown in the contrast editor
        self.rows: list[_ChannelRow] = []
        self._initial: dict = {}                  # snapshot key -> ChannelSettings when first shown (Reset)
        self._loading = False
        self._wheel_guard = _WheelGuard(self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.title = QtWidgets.QLabel("3D view")
        font = QtGui.QFont(self.title.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.title.setFont(font)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet(HINT_STYLE)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.extra = QtWidgets.QLabel()
        self.extra.setWordWrap(True)
        self.extra.setStyleSheet(HINT_STYLE)
        self.extra.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.extra.hide()
        lay.addWidget(self.title)
        lay.addWidget(self.info)
        lay.addWidget(self.extra)
        # loading
        self.load_label = ElidedLabel()
        self.load_label.hide()
        load_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("%v / %m")
        self.progress.hide()
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.setToolTip("Stop loading the volume")
        self.cancel.clicked.connect(self._on_cancel)
        self.cancel.hide()
        load_row.addWidget(self.progress, 1)
        load_row.addWidget(self.cancel)
        lay.addWidget(self.load_label)
        lay.addLayout(load_row)
        self.reload = QtWidgets.QPushButton("Reload")
        self.reload.setToolTip("Load the volume of the current selection again")
        self.reload.clicked.connect(self.reloadRequested)
        self.reload.hide()
        reload_row = QtWidgets.QHBoxLayout()
        reload_row.addWidget(self.reload)
        reload_row.addStretch(1)
        lay.addLayout(reload_row)

        # -- channels --------------------------------------------------------------------
        self.channels_box = QtWidgets.QGroupBox("Channels")
        cl = QtWidgets.QVBoxLayout(self.channels_box)
        cl.setSpacing(4)
        self.rows_lay = QtWidgets.QVBoxLayout()
        self.rows_lay.setSpacing(1)
        cl.addLayout(self.rows_lay)
        # composite line (set_composite): which of the file's channels of this size are shown, and how to add them
        self.comp_frame = QtWidgets.QFrame()
        self.comp_frame.setObjectName("compositeLine")
        fl = QtWidgets.QVBoxLayout(self.comp_frame)
        fl.setContentsMargins(8, 6, 8, 6)
        fl.setSpacing(5)
        self.comp_text = QtWidgets.QLabel()
        self.comp_text.setWordWrap(True)
        fl.addWidget(self.comp_text)
        crow = QtWidgets.QHBoxLayout()
        crow.setSpacing(4)
        self.comp_all = QtWidgets.QPushButton()
        self.comp_all.clicked.connect(lambda: self.compositeRequested.emit(not self._comp_on))
        self.comp_choose = QtWidgets.QPushButton("Choose channels")   # a menu (set_composite_menu)
        self.comp_choose.setToolTip("Tick the channels of the composite (also the arrow ▾ next to Composite\n"
                                    "in the toolbar, or View ▸ Composite channels)")
        crow.addWidget(self.comp_all)
        crow.addWidget(self.comp_choose)
        crow.addStretch(1)
        fl.addLayout(crow)
        self.comp_frame.hide()
        self._comp_on = False
        self._comp_icon: QtGui.QIcon | None = None
        cl.addWidget(self.comp_frame)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #e0a040;")
        self.note.hide()
        cl.addWidget(self.note)
        self.rows_hint = QtWidgets.QLabel("Click a channel to edit its contrast below.")
        self.rows_hint.setWordWrap(True)
        self.rows_hint.setStyleSheet(HINT_STYLE)
        cl.addWidget(self.rows_hint)
        lay.addWidget(self.channels_box)

        # -- contrast --------------------------------------------------------------------
        self.contrast_box = QtWidgets.QGroupBox("Contrast")
        vl = QtWidgets.QVBoxLayout(self.contrast_box)
        head = QtWidgets.QHBoxLayout()
        self.ch_swatch = QtWidgets.QLabel()
        self.ch_name = ElidedLabel()
        bold = QtGui.QFont(self.ch_name.font())
        bold.setBold(True)
        self.ch_name.setFont(bold)
        head.addWidget(self.ch_swatch)
        head.addWidget(self.ch_name, 1)
        vl.addLayout(head)
        self.tf = TransferFunctionWidget()
        self.tf.windowEdited.connect(self._on_tf)
        vl.addWidget(self.tf)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        self.cmin = _ValueSpin()
        self.cmax = _ValueSpin()
        self.cmin.setToolTip("Values at or below Min are black and fully transparent")
        self.cmax.setToolTip("Values at or above Max have the full colour and opacity")
        for sb in (self.cmin, self.cmax):
            sb.setKeyboardTracking(False)
            sb.setRange(-1e9, 1e9)
            sb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.cmin.valueChanged.connect(self._on_min)
        self.cmax.valueChanged.connect(self._on_max)
        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.10, 5.00)
        self.gamma.setSingleStep(0.05)
        self.gamma.setDecimals(2)
        self.gamma.setValue(1.0)
        self.gamma.setKeyboardTracking(False)
        self.gamma.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gamma.setToolTip("Brightness curve between Min and Max: below 1 brightens dim voxels,\n"
                              "above 1 darkens them (1 = linear)")
        self.gamma.valueChanged.connect(self._on_gamma)
        self.weight = QtWidgets.QSlider(Qt.Horizontal)
        self.weight.setRange(0, 100)
        self.weight.setValue(100)
        self.weight.setToolTip("Weight of this channel in the rendering (3Dscript's channel weight); 0 hides it")
        self.weight.valueChanged.connect(self._on_weight)
        self.weight_label = QtWidgets.QLabel("100 %")
        self.weight_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.weight_label.setFixedWidth(self.weight_label.fontMetrics().horizontalAdvance("100 %") + 4)
        wrow = QtWidgets.QHBoxLayout()
        wrow.setSpacing(4)
        wrow.addWidget(self.weight, 1)
        wrow.addWidget(self.weight_label)
        grid.addWidget(QtWidgets.QLabel("Min"), 0, 0)
        grid.addWidget(self.cmin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Max"), 0, 2)
        grid.addWidget(self.cmax, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Gamma"), 1, 0)
        grid.addWidget(self.gamma, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Weight"), 1, 2)
        grid.addLayout(wrow, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        vl.addLayout(grid)
        brow = QtWidgets.QHBoxLayout()
        brow.setSpacing(4)
        self.b_auto = _tool_button("Auto", "0.1 – 99.9 % of the volume\nShift+click: every channel", self._auto)
        self.b_full = _tool_button("Full", "Whole value range of the volume\nShift+click: every channel", self._full)
        self.b_2d = _tool_button("2D", "Min / Max of the 2D view of this channel",
                                 lambda: self.fromTwoDRequested.emit(self.current))
        self.b_reset = _tool_button("Reset", "Every channel back to the colour, window, gamma, weight and visibility\n"
                                    "it had when it was first shown in 3D", self._reset)
        for b in (self.b_auto, self.b_full, self.b_2d, self.b_reset):
            brow.addWidget(b)
        vl.addLayout(brow)
        lay.addWidget(self.contrast_box)

        # -- rendering -------------------------------------------------------------------
        self.render_box = QtWidgets.QGroupBox("Rendering")
        rf = QtWidgets.QFormLayout(self.render_box)
        self.algorithm = QtWidgets.QComboBox()
        for key in ALGORITHMS:
            self.algorithm.addItem(ALGORITHM_LABELS[key], key)
        self.algorithm.setToolTip("Independent transparency: every channel is composited front to back on its own.\n"
                                  "Combined transparency: the opacities of the channels attenuate each other.\n"
                                  "Maximum intensity: the brightest value along each ray (as the 2D maximum "
                                  "projection).")
        self.algorithm.currentIndexChanged.connect(self._on_algorithm)
        rf.addRow("Algorithm", self.algorithm)
        self.sampling = QtWidgets.QComboBox()
        for label, value in SAMPLING:
            self.sampling.addItem(f"{label} ({value:g} per voxel)", value)
        self.sampling.setToolTip("Samples per voxel along each ray: Fast is coarser, Fine smoother but slower.\n"
                                 "While the view moves it is always rendered coarser.")
        self.sampling.currentIndexChanged.connect(self._on_sampling)
        rf.addRow("Sampling", self.sampling)
        self.background = QtWidgets.QToolButton()
        self.background.setIconSize(QtCore.QSize(30, 14))
        self.background.setToolTip("Background colour of the 3D view")
        self.background.clicked.connect(self._pick_background)
        rf.addRow("Background", self.background)
        orow = QtWidgets.QHBoxLayout()
        self.show_box = QtWidgets.QCheckBox("Bounding box")
        self.show_box.setToolTip("Outline the edges of the volume that face the viewer")
        self.show_scalebar = QtWidgets.QCheckBox("Scale bar")
        self.show_scalebar.setToolTip("Scale bar in the lower left corner (in the plane of the screen)")
        self.show_axes = QtWidgets.QCheckBox("Axes")
        self.show_axes.setToolTip("x (red), y (green) and z (blue) axes of the volume, lower right")
        for cb, attr in ((self.show_box, "show_box"), (self.show_scalebar, "show_scalebar"),
                         (self.show_axes, "show_axes")):
            cb.toggled.connect(lambda on, attr=attr: self._on_overlay(attr, on))
            orow.addWidget(cb)
        orow.addStretch(1)
        rf.addRow(orow)
        lay.addWidget(self.render_box)

        # -- view ------------------------------------------------------------------------
        self.view_box = QtWidgets.QGroupBox("View")
        vf = QtWidgets.QFormLayout(self.view_box)
        prow = QtWidgets.QHBoxLayout()
        prow.setSpacing(4)
        tips = {"3D": "Oblique view from above the front (R or Home in the view)",
                "XY": "Looking along z: x right, y down, as the 2D view",
                "XZ": "x right, z down",
                "YZ": "y right, z down",
                "Fit": "Zoom so the volume fills the view, rotation unchanged (F in the view)"}
        self.preset_buttons: dict[str, QtWidgets.QToolButton] = {}
        for name in ("3D", "XY", "XZ", "YZ", "Fit"):
            b = _tool_button(name, tips[name], lambda _=False, n=name: self.view.set_preset("fit" if n == "Fit" else n))
            self.preset_buttons[name] = b
            prow.addWidget(b)
        vf.addRow(prow)
        rot = QtWidgets.QHBoxLayout()
        rot.setSpacing(4)
        self.rot_x, self.rot_y, self.rot_z = (_AngleSpin() for _ in range(3))
        for axis, sb in zip("XYZ", (self.rot_x, self.rot_y, self.rot_z)):
            sb.setDecimals(1)
            sb.setSingleStep(5.0)
            sb.setSuffix("°")
            sb.setKeyboardTracking(False)
            sb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            sb.setToolTip(f"Rotation about {axis} in degrees (3Dscript's angles: R = Ry · Rz · Rx)")
            sb.valueChanged.connect(self._on_rotation)
            label = QtWidgets.QLabel(axis)
            label.setStyleSheet(HINT_STYLE)
            rot.addWidget(label)
            rot.addWidget(sb, 1)
        vf.addRow("Rotation", rot)
        self.zoom = QtWidgets.QDoubleSpinBox()
        self.zoom.setRange(0.05, 50.0)
        self.zoom.setDecimals(3)
        self.zoom.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.zoom.setKeyboardTracking(False)
        self.zoom.setValue(1.0)
        self.zoom.setToolTip("1 = the whole volume fits the view at any rotation (wheel in the view zooms too)")
        self.zoom.valueChanged.connect(self._on_zoom)
        vf.addRow("Zoom", self.zoom)
        hint = QtWidgets.QLabel("In the view: drag rotates · Shift+drag moves · wheel zooms")
        hint.setWordWrap(True)
        hint.setStyleSheet(HINT_STYLE)
        vf.addRow(hint)
        lay.addWidget(self.view_box)

        # -- volume ----------------------------------------------------------------------
        self.volume_box = QtWidgets.QGroupBox("Volume")
        of = QtWidgets.QFormLayout(self.volume_box)
        self.res_combo = QtWidgets.QComboBox()
        for label, key in RESOLUTIONS:
            self.res_combo.addItem(label, key)
        self.res_combo.setToolTip("Resolution of the volume on the graphics card.\n"
                                  "Auto: the finest that fits the memory budget; Full: no downsampling;\n"
                                  "½ and ¼: coarser, for speed.  Changing it reloads the volume.")
        self.res_combo.currentIndexChanged.connect(lambda _: self.resolutionChanged.emit(self.resolution()))
        of.addRow("Resolution", self.res_combo)
        srow = QtWidgets.QHBoxLayout()
        self.b_save = QtWidgets.QPushButton("Save PNG…")
        self.b_save.setToolTip("Save the rendered view as a PNG image, full quality, with the overlays")
        self.b_save.clicked.connect(self.saveRequested)
        self.b_copy = QtWidgets.QPushButton("Copy image")
        self.b_copy.setToolTip("Copy the rendered view to the clipboard")
        self.b_copy.clicked.connect(self.copyRequested)
        srow.addWidget(self.b_save)
        srow.addWidget(self.b_copy)
        srow.addStretch(1)
        of.addRow(srow)
        lay.addWidget(self.volume_box)

        for combo in (self.algorithm, self.sampling, self.res_combo):   # long entries must not widen the tab
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        foot = QtWidgets.QLabel("Rendering after 3Dscript (Schmid et al., Nat. Methods 2019; BSD-2)")
        foot.setWordWrap(True)
        foot.setStyleSheet(HINT_STYLE)
        foot.setToolTip("B. Schmid et al., \"3Dscript: animating 3D/4D microscopy data using a natural-language-"
                        "based syntax\",\nNature Methods 16:278-280 (2019); https://github.com/bene51/3Dscript")
        lay.addWidget(foot)
        lay.addStretch(1)

        view.cameraChanged.connect(self._camera_to_widgets)
        _no_drops(self)
        _quiet_focus(self, self._wheel_guard)
        self.set_volume(getattr(view, "volume", None))

    # -- public ------------------------------------------------------------------------

    def set_volume(self, volume: Volume | None) -> None:
        """Show *volume*: channel rows from view.channels, histograms from volume.channels (same order)."""
        self.volume = volume
        chans = list(self.view.channels)
        if volume is not None:  # the first snapshot of every channel, for Reset (until forget())
            for i, cs in enumerate(chans):
                self._initial.setdefault(self._snapshot_key(i, cs), cs.copy())
            self.reload.hide()
        self.info.setText(_volume_text(volume) if volume is not None and volume.channels else
                          "No volume loaded.")
        self.current = min(max(self.current, 0), max(len(chans) - 1, 0))
        has_vol = volume is not None
        for w in (self.view_box, self.b_save, self.b_copy):
            w.setEnabled(has_vol)
        self.refresh()        # new rows only for another number of channels (a T step keeps them)

    def set_loading(self, text: str | None, done: int = 0, total: int = 0) -> None:
        """Progress of loading the volume, with Cancel; None hides it."""
        if text is None:
            self._loading = False
            for w in (self.load_label, self.progress, self.cancel):
                w.hide()
            return
        if not self._loading:  # a new load
            self._loading = True
            self.cancel.setEnabled(True)
            self.cancel.setText("Cancel")
            self.reload.hide()
        self.load_label.setText(text)
        if total > 0:
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(min(max(done, 0), total)))
        else:
            self.progress.setRange(0, 0)   # busy indicator
        for w in (self.load_label, self.progress, self.cancel):
            w.show()

    def show_reload(self, on: bool) -> None:
        """The Reload button (reloadRequested), after a load was cancelled or failed; hidden by the next load."""
        self.reload.setVisible(bool(on))

    def set_composite_menu(self, menu: QtWidgets.QMenu | None, icon: QtGui.QIcon | None = None) -> None:
        """The menu of *Choose channels* (the viewer's composite channels menu) and the icon of the composite."""
        self.comp_choose.setMenu(menu)
        self._comp_icon = icon

    def set_composite(self, total: int, shown: int, on: bool, name: str = "") -> None:
        """The composite line: *total* channels of the file have the size of the selected stack (*name*);
        Composite is *on* with *shown* of them, or off (only the selected one).  Hidden below two channels."""
        self._comp_on = bool(on)
        if total < 2:
            self.comp_frame.hide()
            return
        icon = self._comp_icon
        if on:
            self.comp_text.setText(f"Composite: {shown} of {total} channels, as in 2D.")
            self.comp_text.setToolTip("")
            self.comp_text.setStyleSheet(HINT_STYLE)
            self.comp_all.setText("Selected channel only")
            self.comp_all.setIcon(QtGui.QIcon())
            self.comp_all.setToolTip("Composite off: 2D and 3D show the channel selected in the list (C)")
            self.comp_frame.setStyleSheet("QFrame#compositeLine { border: 1px solid rgba(128, 128, 128, 110); "
                                          "border-radius: 4px; }")
        else:
            self.comp_text.setText(f"<b>1 of {total} channels shown</b>: the one selected in the list. "
                                   f"Composite (C) shows the {total} channels of this size together.")
            self.comp_text.setToolTip(f"Shown: {name}" if name else "")
            self.comp_text.setStyleSheet("")
            self.comp_all.setText(f"Show all {total} channels")
            self.comp_all.setIcon(icon or QtGui.QIcon())
            self.comp_all.setToolTip("Composite on with every channel of this size, in 2D and 3D (C)")
            self.comp_frame.setStyleSheet("QFrame#compositeLine { background: rgba(40, 110, 190, 55); "
                                          "border: 1px solid rgba(80, 150, 230, 200); border-radius: 4px; }")
        self.comp_frame.show()

    def set_info(self, text: str | None) -> None:
        """Extra line under the volume info (renderer, errors)."""
        self.extra.setText(text or "")
        self.extra.setVisible(bool(text))

    def forget(self, file_key) -> None:
        """Drop the Reset snapshots of the channels of file *file_key* (closed: its settings are gone)."""
        self._initial = {k: v for k, v in self._initial.items()
                         if not (isinstance(k, tuple) and k and k[0] == file_key)}

    def refresh(self) -> None:
        """Re-read the view's channels, settings and camera into the controls (no signals)."""
        if len(self.rows) != len(self.view.channels):
            self._build_rows()
            self.current = min(max(self.current, 0), max(len(self.rows) - 1, 0))
        for i in range(len(self.rows)):
            self._sync_row(i)
        self._update_channel_state()
        self._sync_contrast()
        self._sync_rendering()
        if self.volume is not None:
            self._camera_to_widgets()

    def select_channel(self, i: int) -> None:
        """Edit channel *i* in the contrast editor."""
        if 0 <= i < len(self.rows):
            self.current = i
            for j, row in enumerate(self.rows):
                row.set_selected(j == i)
            self._sync_contrast()

    def current_channel(self) -> int:
        return self.current

    def resolution(self) -> str:
        return self.res_combo.currentData() or "auto"

    def set_resolution(self, mode: str) -> None:
        """Show resolution *mode* without emitting resolutionChanged."""
        i = self.res_combo.findData(mode)
        with _blocked(self.res_combo):
            self.res_combo.setCurrentIndex(max(i, 0))

    # -- helpers -----------------------------------------------------------------------

    @staticmethod
    def _snapshot_key(i: int, cs: ChannelSettings):
        k = cs.key
        try:
            hash(k)
        except TypeError:
            k = None
        return ("#", i, cs.name) if k is None else k

    def _channel(self, i: int | None = None) -> ChannelSettings | None:
        i = self.current if i is None else i
        chans = self.view.channels
        return chans[i] if 0 <= i < len(chans) else None

    def _vchannel(self, i: int | None = None) -> VolumeChannel | None:
        i = self.current if i is None else i
        v = self.volume
        return v.channels[i] if v is not None and 0 <= i < len(v.channels) else None

    def _name(self, i: int) -> str:
        cs, vc = self._channel(i), self._vchannel(i)
        return (cs.name if cs is not None and cs.name else None) or (vc.name if vc is not None and vc.name else
                                                                      f"Channel {i + 1}")

    def _is_integer(self, i: int | None = None) -> bool:
        vc = self._vchannel(i)
        return vc is not None and vc.data.dtype.kind in "ui"

    def _gap(self) -> float:
        """Smallest Min-Max distance: one count for integer data, the last shown decimal otherwise."""
        return 1.0 if self._is_integer() else 10.0 ** -self.cmin.decimals()

    def _changed(self) -> None:
        self.view.settings_changed()

    # -- channel rows --------------------------------------------------------------------

    def _build_rows(self) -> None:
        for row in self.rows:
            self.rows_lay.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.rows = []
        for i in range(len(self.view.channels)):
            row = _ChannelRow()
            row.visible.toggled.connect(lambda on, i=i: self._on_visible(i, on))
            row.color.setMenu(self._color_menu(i, row.color))
            row.clicked.connect(lambda i=i: self.select_channel(i))
            self.rows_lay.addWidget(row)
            self.rows.append(row)
        self.rows_hint.setVisible(len(self.rows) > 1)
        _no_drops(self.channels_box)
        _quiet_focus(self.channels_box, self._wheel_guard)

    def _color_menu(self, i: int, button: QtWidgets.QToolButton) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(button)
        group = QtWidgets.QActionGroup(menu)
        for name, hexcol in COLORS:
            a = menu.addAction(_color_icon(QtGui.QColor(hexcol), 20, 12), name)
            a.setCheckable(True)
            a.setData(("color", hexcol))
            group.addAction(a)
            a.triggered.connect(lambda _=False, c=hexcol: self._set_color(i, QtGui.QColor(c)))
        menu.addSeparator()
        for name in LUT_NAMES:
            a = menu.addAction(_lut_icon(LUTS[name], 20, 12), name)
            a.setCheckable(True)
            a.setData(("lut", name))
            group.addAction(a)
            a.triggered.connect(lambda _=False, n=name: self._set_lut(i, n))
        menu.addSeparator()
        menu.addAction("Other…", lambda: self._other_color_later(i))
        menu.aboutToShow.connect(lambda: self._check_menu(i, menu))
        return menu

    def _check_menu(self, i: int, menu: QtWidgets.QMenu) -> None:
        """Tick the entry of the channel's current colour or lookup table."""
        cs = self._channel(i)
        if cs is None:
            return
        current = _qcolor(cs.color).name()
        for a in menu.actions():
            kind = a.data()
            if a.isCheckable():
                a.setChecked(bool(kind) and ((kind[0] == "lut" and cs.lut == kind[1]) or
                                             (kind[0] == "color" and not cs.lut and kind[1] == current)))

    def _sync_row(self, i: int) -> None:
        cs, row = self._channel(i), self.rows[i]
        if cs is None:
            return
        with _blocked(row.visible):
            row.visible.setChecked(bool(cs.visible))
        row.color.setIcon(_channel_icon(cs))
        row.name.setText(self._name(i))
        row.set_selected(i == self.current)
        self._row_text(i)

    def _row_text(self, i: int) -> None:
        """Window and tooltip of row *i* (cheap: called on every drag step)."""
        cs, row = self._channel(i), self.rows[i]
        if cs is None:
            return
        row.levels.setText(f"{_fmt(cs.cmin)} – {_fmt(cs.cmax)}")
        look = cs.lut or "colour " + _qcolor(cs.color).name()
        row.setToolTip(f"{row.name.text()}\n{look} · window {_fmt(cs.cmin)} – {_fmt(cs.cmax)} · "
                       f"gamma {cs.cgamma:.2f} · weight {cs.weight * 100:.0f} %")

    def _update_channel_state(self) -> None:
        """Dim channels that are not rendered; note when more are on than can be rendered."""
        chans = self.view.channels
        on = [i for i, cs in enumerate(chans) if cs.visible and cs.weight > 0]
        rendered = set(on[:MAX_CHANNELS])
        for i, row in enumerate(self.rows):
            row.set_dimmed(i not in rendered)
        if len(on) > MAX_CHANNELS:
            self.note.setText(f"{len(on)} channels are switched on; only the first {MAX_CHANNELS} are rendered.")
        self.note.setVisible(len(on) > MAX_CHANNELS)
        self.channels_box.setEnabled(bool(chans))

    def _on_visible(self, i: int, on: bool) -> None:
        cs = self._channel(i)
        if cs is not None:
            cs.visible = bool(on)
            self._sync_row(i)
            self._update_channel_state()
            self._changed()

    def _set_color(self, i: int, color: QtGui.QColor) -> None:
        cs = self._channel(i)
        if cs is None or not color.isValid():
            return
        cs.color, cs.lut = _rgb(color), None
        self._color_edited(i)

    def _set_lut(self, i: int, name: str) -> None:
        cs = self._channel(i)
        if cs is not None:
            cs.lut = name
            self._color_edited(i)

    def _color_edited(self, i: int) -> None:
        self._sync_row(i)
        if i != self.current:
            self.select_channel(i)   # show the edited channel (its histogram changes colour too)
        else:
            self._sync_contrast()
        self._changed()

    def _other_color_later(self, i: int) -> None:
        """'Other…' of channel *i*: the dialog opens once the menu has closed.

        A load that finishes while the dialog is open rebuilds the rows; opened from inside the
        menu, Qt would return through the deleted menu and button.
        """
        cs = self._channel(i)
        if cs is not None:
            title = f"Colour of {self._name(i)}"
            QtCore.QTimer.singleShot(0, lambda: self._other_color(cs, title))

    def _other_color(self, cs: ChannelSettings, title: str) -> None:
        """Colour dialog for *cs* itself (not by index: the view may show other channels when it closes)."""
        old = (cs.color, cs.lut)

        def put(color, lut) -> None:
            cs.color, cs.lut = color, lut
            i = next((k for k, c in enumerate(self.view.channels) if c is cs), None)
            if i is not None:   # still shown
                self._color_edited(i)

        self._pick_color(_qcolor(cs.color), title, lambda c: put(_rgb(c), None), lambda: put(*old))

    def _pick_color(self, initial: QtGui.QColor, title: str, apply, restore) -> None:
        """QColorDialog whose current colour is applied live; Cancel restores the previous state."""
        dlg = QtWidgets.QColorDialog(initial, self)
        dlg.setWindowTitle(title)
        dlg.currentColorChanged.connect(lambda c: apply(c) if c.isValid() else None)
        try:
            if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.selectedColor().isValid():
                apply(dlg.selectedColor())
            else:
                restore()
        finally:
            dlg.deleteLater()

    # -- contrast ------------------------------------------------------------------------

    def _sync_contrast(self) -> None:
        """Show the selected channel in the contrast editor."""
        cs, vc = self._channel(), self._vchannel()
        self.contrast_box.setEnabled(cs is not None)
        if cs is None:
            self.ch_name.setText("")
            self.ch_swatch.clear()
            self.tf.clear()
            return
        integer = self._is_integer()
        if vc is not None:
            lo_v, hi_v = min(vc.data_min, vc.hist_lo), max(vc.data_max, vc.hist_hi)
        else:
            lo_v, hi_v = cs.cmin, cs.cmax
        span = max(hi_v - lo_v, abs(cs.cmax - cs.cmin), 1e-12)
        if integer:
            info = np.iinfo(vc.data.dtype)
            decimals, limits = 0, (float(info.min), float(info.max))
            step = max(1.0, round(nice_length(span / 200)))
        else:
            decimals = min(6, max(2, 4 - int(math.floor(math.log10(span)))))
            lim = 10.0 ** (math.ceil(math.log10(max(abs(lo_v), abs(hi_v), abs(cs.cmin), abs(cs.cmax), 1.0))) + 1)
            limits = (-lim, lim)
            step = nice_length(span / 200) or 10.0 ** -decimals
        with _blocked(self.cmin, self.cmax, self.gamma, self.weight):
            for sb in (self.cmin, self.cmax):
                sb.setDecimals(decimals)
                sb.setRange(*limits)
                sb.setSingleStep(step)
            self.cmin.setValue(cs.cmin)
            self.cmax.setValue(cs.cmax)
            self.gamma.setValue(cs.cgamma)
            self.weight.setValue(int(round(cs.weight * 100)))
        self.weight_label.setText(f"{int(round(cs.weight * 100))} %")
        if vc is not None:
            self.tf.set_histogram(vc.histogram, vc.hist_lo, vc.hist_hi)
        else:
            self.tf.set_histogram(None, cs.cmin, cs.cmax)
        self.tf.set_integer(integer)
        self.tf.set_channel(cs)
        self.ch_name.setText(self._name(self.current))
        self.ch_swatch.setPixmap(_channel_icon(cs, 22, 12).pixmap(22, 12))
        for b in (self.b_auto, self.b_full):
            b.setEnabled(vc is not None)

    def _show_window(self) -> None:
        """Window of the selected channel -> spin boxes, transfer function and its row."""
        cs = self._channel()
        if cs is None:
            return
        with _blocked(self.cmin, self.cmax):
            self.cmin.setValue(cs.cmin)
            self.cmax.setValue(cs.cmax)
        self.tf.set_channel(cs)
        self._row_text(self.current)

    def _apply_window(self, lo: float, hi: float, i: int | None = None) -> None:
        """Set the window of channel *i* (default: the selected one) and repaint."""
        i = self.current if i is None else i
        cs = self._channel(i)
        if cs is None:
            return
        if hi <= lo:
            hi = lo + (1.0 if self._is_integer(i) else max(abs(lo) * 1e-6, 1e-12))
        cs.set_window(lo, hi)
        if i == self.current:
            self._show_window()
        else:
            self._row_text(i)

    def _on_tf(self, lo: float, hi: float) -> None:
        self._apply_window(lo, hi)
        self._changed()

    def _on_min(self, v: float) -> None:
        cs = self._channel()
        if cs is None:
            return
        lo, hi = float(v), cs.cmax
        if hi <= lo:  # push Max along
            hi = lo + self._gap()
        self._apply_window(lo, hi)
        self._changed()

    def _on_max(self, v: float) -> None:
        cs = self._channel()
        if cs is None:
            return
        lo, hi = cs.cmin, float(v)
        if hi <= lo:  # push Min along
            lo = hi - self._gap()
        self._apply_window(lo, hi)
        self._changed()

    def _on_gamma(self, v: float) -> None:
        cs = self._channel()
        if cs is not None:
            cs.cgamma = float(v)
            self.tf.set_channel(cs)
            self._row_text(self.current)
            self._changed()

    def _on_weight(self, v: int) -> None:
        cs = self._channel()
        self.weight_label.setText(f"{v} %")
        if cs is not None:
            cs.weight = v / 100.0
            self._row_text(self.current)
            self._update_channel_state()
            self._changed()

    def _targets(self) -> list[int]:
        """The selected channel, or every channel with Shift held."""
        if QtWidgets.QApplication.keyboardModifiers() & Qt.ShiftModifier:
            return list(range(len(self.view.channels)))
        return [self.current]

    def _auto(self) -> None:
        for i in self._targets():
            vc = self._vchannel(i)
            if vc is not None:
                self._apply_window(*vc.window, i=i)
        self._changed()

    def _full(self) -> None:
        for i in self._targets():
            vc = self._vchannel(i)
            if vc is not None:
                self._apply_window(vc.data_min, vc.data_max, i=i)
        self._changed()

    def _reset(self) -> None:
        """Every channel back to its settings when it was first shown (name and key stay)."""
        for i, cs in enumerate(self.view.channels):
            init = self._initial.get(self._snapshot_key(i, cs))
            if init is None:
                continue
            for f in dataclasses.fields(cs):
                if f.name not in ("name", "key"):
                    setattr(cs, f.name, copy.copy(getattr(init, f.name)))
        self.refresh()
        self._changed()

    # -- rendering -----------------------------------------------------------------------

    def _sync_rendering(self) -> None:
        s = self.view.settings
        boxes = (self.algorithm, self.sampling, self.show_box, self.show_scalebar, self.show_axes)
        with _blocked(*boxes):
            self.algorithm.setCurrentIndex(max(0, self.algorithm.findData(s.algorithm)))
            values = [v for _, v in SAMPLING]
            self.sampling.setCurrentIndex(int(np.argmin([abs(math.log(v / max(s.sampling, 1e-6))) for v in values])))
            self.show_box.setChecked(bool(s.show_box))
            self.show_scalebar.setChecked(bool(s.show_scalebar))
            self.show_axes.setChecked(bool(s.show_axes))
        self.background.setIcon(_color_icon(_qcolor(s.background)))

    def _on_algorithm(self, *_) -> None:
        self.view.settings.algorithm = self.algorithm.currentData()
        self._changed()

    def _on_sampling(self, *_) -> None:
        self.view.settings.sampling = float(self.sampling.currentData())
        self._changed()

    def _on_overlay(self, attr: str, on: bool) -> None:
        setattr(self.view.settings, attr, bool(on))
        self._changed()
        self.overlayChanged.emit()

    def _set_background(self, color: QtGui.QColor) -> None:
        if color.isValid():
            self.view.settings.background = _rgb(color)
            self.background.setIcon(_color_icon(color))
            self._changed()

    def _pick_background(self) -> None:
        old = tuple(self.view.settings.background)
        self._pick_color(_qcolor(old), "Background colour", self._set_background,
                         lambda: self._set_background(_qcolor(old)))

    # -- view ----------------------------------------------------------------------------

    def _camera_to_widgets(self) -> None:
        """Rotation and zoom of the view's camera -> spin boxes (no signals)."""
        cam = self.view.camera
        ax, ay, az = cam.euler_deg()
        with _blocked(self.rot_x, self.rot_y, self.rot_z, self.zoom):
            self.rot_x.setValue(ax)
            self.rot_y.setValue(ay)
            self.rot_z.setValue(az)
            self.zoom.setValue(cam.zoom)

    def _on_rotation(self, *_) -> None:
        self.view.camera.set_euler_deg(self.rot_x.value(), self.rot_y.value(), self.rot_z.value())
        self._changed()

    def _on_zoom(self, v: float) -> None:
        self.view.camera.zoom = float(v)
        self._changed()

    # -- loading -------------------------------------------------------------------------

    def _on_cancel(self) -> None:
        self.cancel.setEnabled(False)
        self.cancel.setText("Cancelling…")
        self.cancelRequested.emit()
