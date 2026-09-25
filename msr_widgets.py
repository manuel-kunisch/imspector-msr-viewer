"""
msr_widgets.py -- image display building blocks shared by the viewer and its tools.

Lookup tables, 8-bit intensity mapping, a zoomable image view with scale bar,
and small painting helpers.  Nothing ImSpector-specific lives here.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


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


class IntensityMapper:
    """Maps raw pixel values to 0..255 for a display range (min, max).

    For 8/16-bit data the mapping is a cached lookup table, so repeated frames
    (playback, video export) cost one indexing operation each.  Not shared
    between threads: every thread uses its own instance.
    """

    def __init__(self):
        self._key = None
        self._table = None

    def map(self, frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scale = 255.0 / max(hi - lo, 1e-12)
        if frame.dtype in (np.uint8, np.uint16):
            key = (frame.dtype.str, lo, hi)
            if self._key != key:
                x = np.arange(256 if frame.dtype == np.uint8 else 65536, dtype=np.float32)
                self._table = np.clip((x - lo) * scale + 0.5, 0, 255).astype(np.uint8)
                self._key = key
            return self._table[frame]
        return np.clip((frame.astype(np.float32) - lo) * scale + 0.5, 0, 255).astype(np.uint8)


def render_rgb(frame: np.ndarray, lo: float, hi: float, lut: str,
               mapper: IntensityMapper | None = None) -> np.ndarray:
    """Frame -> (h, w, 3) uint8 RGB with a lookup table and display range."""
    mapper = mapper or IntensityMapper()
    return LUTS[lut][mapper.map(frame, lo, hi)]


# -----------------------------------------------------------------------------
# images and painting
# -----------------------------------------------------------------------------

def numpy_to_qimage(a8: np.ndarray, colors) -> QtGui.QImage:
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


def rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    """(h, w, 3) uint8 array -> QImage (copied)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def qimage_to_rgb(img: QtGui.QImage) -> np.ndarray:
    """QImage -> (h, w, 3) uint8 array."""
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    raw = img.constBits().asstring(h * bpl)
    return np.frombuffer(raw, np.uint8).reshape(h, bpl)[:, :3 * w].reshape(h, w, 3).copy()


def nice_length(x: float) -> float:
    """1, 2 or 5 times a power of ten, closest to x."""
    if not x or x <= 0 or not math.isfinite(x):
        return 0.0
    e = math.floor(math.log10(x))
    candidates = [c * 10.0 ** k for k in (e - 1, e, e + 1) for c in (1, 2, 5)]
    return min(candidates, key=lambda c: abs(math.log(c / x)))


def fmt_length(v: float, unit: str) -> str:
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
    length = nice_length(target * width * per_px)
    px = length / per_px
    if px < 4:
        return
    font = QtGui.QFont(p.font())
    font.setPixelSize(font_px)
    font.setBold(True)
    fm = QtGui.QFontMetrics(font)
    text = fmt_length(length, unit)
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


def paint_label(p: QtGui.QPainter, text: str, font_px: int = 12, margin: float = 18) -> None:
    """Text on a dark box in the top-left corner (frame time etc.)."""
    font = QtGui.QFont(p.font())
    font.setPixelSize(font_px)
    font.setBold(True)
    fm = QtGui.QFontMetrics(font)
    tw, th = fm.horizontalAdvance(text), fm.height()
    p.save()
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor(0, 0, 0, 140))
    p.drawRoundedRect(QtCore.QRectF(margin - 6, margin - 4, tw + 12, th + 8), 4, 4)
    p.setPen(QtGui.QColor(255, 255, 255))
    p.setFont(font)
    p.drawText(QtCore.QRectF(margin, margin, tw + 2, th), Qt.AlignLeft | Qt.AlignVCenter, text)
    p.restore()


def paint_overlays(img: QtGui.QImage, per_px: float | None, unit: str, scalebar: bool = True,
                   label: str | None = None) -> QtGui.QImage:
    """Scale bar and/or label burned into an image, sized relative to the image."""
    if not ((scalebar and per_px) or label):
        return img
    out = img.convertToFormat(QtGui.QImage.Format_RGB32)
    w, h = out.width(), out.height()
    font_px, margin = max(12, h // 28), max(8, w // 40)
    p = QtGui.QPainter(out)
    if scalebar and per_px:
        paint_scalebar(p, w, h, per_px, unit, font_px=font_px, bar_h=max(3, h // 110), margin=margin)
    if label:
        paint_label(p, label, font_px=font_px, margin=margin)
    p.end()
    return out


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
    """Zoomable image with a scale bar overlay.

    Scene coordinates are image pixels (the pixmap sits at 0, 0), so overlay
    items can be placed directly in pixel units.
    """

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
