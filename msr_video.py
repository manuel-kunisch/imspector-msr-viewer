"""
msr_video.py -- MP4 export of a stack axis (timelapse or z sweep).

Two timings:

* real time: every frame stays on screen for its actual acquisition interval
  (per-frame time stamps from the .msr), scaled by a speed factor (0.25 = four
  times slower).  The file has a constant output frame rate so it plays
  everywhere; a frame is repeated for as many output frames as its interval
  lasts, accurate to one output frame.
* fixed frame rate: one video frame per stack frame.

Frames are rendered with the viewer's lookup table and display range and are
streamed to ffmpeg (imageio-ffmpeg's bundled binary, or ffmpeg on PATH), so a
stack of any length is exported without holding it in memory.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtWidgets

import msr_reader as mr
from msr_widgets import IntensityMapper, paint_overlays, qimage_to_rgb, render_rgb, rgb_to_qimage

OUTPUT_RATES = (24, 25, 30, 50, 60, 100, 120)


# -----------------------------------------------------------------------------
# timing
# -----------------------------------------------------------------------------

def frame_times(stack: mr.DataStack, letter: str, n: int) -> np.ndarray | None:
    """Acquisition time (s) of each frame along a T axis, or None."""
    if letter != "T" or n < 2:
        return None
    ts = np.asarray(stack.timestamps, dtype=np.float64)
    if ts.size == n and np.all(np.isfinite(ts)) and np.all(np.diff(ts) >= 0) and ts[-1] > ts[0]:
        return ts
    dt = stack.time_increment
    return np.arange(n, dtype=np.float64) * dt if dt else None


def typical_interval(times: np.ndarray) -> float | None:
    d = np.diff(times)
    d = d[d > 0]
    return float(np.median(d)) if d.size else None


def schedule(first: int, last: int, times: np.ndarray | None = None, speed: float = 1.0,
             out_fps: float = 60.0, fps: float | None = None) -> tuple[np.ndarray, float]:
    """Stack frame shown in each output frame, and the output frame rate.

    With *fps* (fixed frame rate) every stack frame first..last becomes one
    output frame.  Otherwise *times* are used: output frame k (at k / out_fps
    seconds of video) shows the last stack frame whose (time - t_first) / speed
    has already passed.  The last frame is held for one typical interval.
    """
    if fps is not None or times is None:
        return np.arange(first, last + 1), float(fps or out_fps)
    t = (np.asarray(times[first:last + 1], dtype=np.float64) - times[first]) / speed
    hold = (typical_interval(t) or 1.0 / out_fps) if t.size > 1 else 1.0 / out_fps
    n_out = max(1, int(round((t[-1] + hold) * out_fps)))
    tk = np.arange(n_out) / out_fps
    idx = np.searchsorted(t, tk + 1e-9, side="right") - 1
    return first + np.clip(idx, 0, t.size - 1), float(out_fps)


def describe(indices: np.ndarray, rate: float, n_source: int) -> str:
    shown = int(np.unique(indices).size)
    text = f"{indices.size / rate:.2f} s video, {indices.size} frames at {rate:g} fps"
    if shown < n_source:
        text += (f" · {n_source - shown} of {n_source} stack frames are shorter than one output frame and are "
                 "skipped (raise the output rate or slow down)")
    return text


# -----------------------------------------------------------------------------
# ffmpeg
# -----------------------------------------------------------------------------

def find_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW (pythonw)


@functools.lru_cache(maxsize=4)
def _encoders(exe: str) -> str:
    try:
        return subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True,
                              timeout=30, creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


class FFmpegWriter:
    """Streams raw RGB frames into an H.264 MP4 via an ffmpeg process."""

    def __init__(self, path: str, width: int, height: int, fps: float, exe: str | None = None):
        exe = exe or find_ffmpeg()
        if not exe:
            raise RuntimeError("ffmpeg not found: install imageio-ffmpeg (pip/conda) or put ffmpeg on PATH")
        if re.search(r"\blibx264\b", _encoders(exe)):
            codec = ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]
        else:
            codec = ["-c:v", "mpeg4", "-q:v", "2"]
        self.path = path
        self._log = tempfile.TemporaryFile()
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", f"{fps:g}",
               "-i", "-", "-an", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", mr._fs_path(path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log,
                                      creationflags=_NO_WINDOW)

    def _error(self) -> str:
        self._log.seek(0)
        text = self._log.read().decode("utf-8", "replace").strip()
        return text[-1500:] or "ffmpeg failed"

    def write(self, frame: bytes) -> None:
        try:
            self._proc.stdin.write(frame)
        except (BrokenPipeError, OSError):
            self._proc.wait()
            raise RuntimeError(self._error()) from None

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        if self._proc.wait() != 0:
            raise RuntimeError(self._error())
        self._log.close()

    def abort(self) -> None:
        self._proc.kill()
        self._proc.wait()
        self._log.close()
        try:
            os.remove(mr._fs_path(self.path))
        except OSError:
            pass


# -----------------------------------------------------------------------------
# rendering and export
# -----------------------------------------------------------------------------

@dataclass
class VideoSettings:
    path: str
    axis: int                 # which leading axis of the stack is animated
    first: int                # frame range, 0-based, inclusive
    last: int
    real_time: bool = True
    speed: float = 1.0        # real time: 0.5 = twice as slow
    out_fps: float = 60.0     # real time: output frame rate
    fps: float = 25.0         # fixed frame rate
    scale: int = 1            # integer upscaling (nearest neighbour)
    scalebar: bool = True
    label: bool = True        # time (or z) stamp in the corner
    lut: str = "Gray"
    lo: float = 0.0
    hi: float = 1.0


class FrameRenderer:
    """Stack frame -> RGB bytes with LUT, upscaling and overlays (even size)."""

    def __init__(self, stack: mr.DataStack, base_index: tuple, s: VideoSettings):
        self.arr = stack.asarray()
        self.base = list(base_index)
        self.s = s
        self.mapper = IntensityMapper()
        self.letter = stack.axes[s.axis]
        self.times = frame_times(stack, self.letter, self.arr.shape[s.axis])
        self.z_step = stack.step("Z") if self.letter == "Z" else None
        self.z_unit = (stack.axis_info("Z") or {}).get("unit") or "µm"
        px = stack.pixel_size[0]
        self.per_px = px / s.scale if px else None
        self.unit = stack.units[0] if stack.units and stack.units[0] else "µm"

    def label(self, i: int) -> str | None:
        if not self.s.label:
            return None
        if self.times is not None:
            return f"t = {self.times[i] - self.times[0]:.3f} s"
        if self.z_step:
            return f"z = {i * self.z_step:.3g} {self.z_unit}"
        return f"{self.letter} {i + 1}"

    def render(self, i: int) -> np.ndarray:
        idx = list(self.base)
        idx[self.s.axis] = i
        rgb = render_rgb(np.asarray(self.arr[tuple(idx)]), self.s.lo, self.s.hi, self.s.lut, self.mapper)
        if self.s.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.s.scale, axis=0), self.s.scale, axis=1)
        text = self.label(i)
        if (self.s.scalebar and self.per_px) or text:
            rgb = qimage_to_rgb(paint_overlays(rgb_to_qimage(rgb), self.per_px, self.unit, self.s.scalebar, text))
        h, w = rgb.shape[:2]
        if h % 2 or w % 2:  # yuv420p needs even dimensions
            rgb = np.pad(rgb, ((0, h % 2), (0, w % 2), (0, 0)))
        return np.ascontiguousarray(rgb)


class VideoExportThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, stack: mr.DataStack, base_index: tuple, settings: VideoSettings, parent=None):
        super().__init__(parent)
        self.stack, self.base_index, self.settings = stack, tuple(base_index), settings
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        s = self.settings
        writer = None
        try:
            renderer = FrameRenderer(self.stack, self.base_index, s)
            if s.real_time:
                indices, rate = schedule(s.first, s.last, renderer.times, s.speed, s.out_fps)
            else:
                indices, rate = schedule(s.first, s.last, fps=s.fps)
            frame, shown = None, -1
            for k, i in enumerate(indices):
                if self._cancel:
                    if writer is not None:
                        writer.abort()
                    self.done.emit({"cancelled": True, "path": s.path})
                    return
                if i != shown:
                    frame, shown = renderer.render(int(i)), i
                    if writer is None:
                        writer = FFmpegWriter(s.path, frame.shape[1], frame.shape[0], rate)
                    data = frame.tobytes()
                writer.write(data)
                if k % 8 == 0 or k == indices.size - 1:
                    self.progress.emit(k + 1, int(indices.size))
            writer.close()
            self.done.emit({"cancelled": False, "path": s.path, "frames": int(indices.size), "fps": rate,
                            "duration": indices.size / rate, "size": (frame.shape[1], frame.shape[0]),
                            "summary": describe(indices, rate, s.last - s.first + 1)})
        except Exception as exc:
            if writer is not None:
                writer.abort()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class VideoExportDialog(QtWidgets.QDialog):
    """Axis, range, timing, size and overlays for a video export."""

    def __init__(self, stack: mr.DataStack, shape: tuple, index: tuple, play_axis: int, defaults: dict,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.stack = stack
        self.letters = stack.axes[:-2]
        self.shape = tuple(shape)
        self.index = tuple(index)
        self.defaults = defaults
        self._user_path = False
        form = QtWidgets.QFormLayout(self)

        self.axis = QtWidgets.QComboBox()
        for letter, n in zip(self.letters, self.shape):
            self.axis.addItem(f"{letter}  ({n} frames)")
        self.axis.setCurrentIndex(play_axis)
        form.addRow("Animate", self.axis)

        rng = QtWidgets.QHBoxLayout()
        self.first = QtWidgets.QSpinBox()
        self.last = QtWidgets.QSpinBox()
        rng.addWidget(self.first)
        rng.addWidget(QtWidgets.QLabel("to"))
        rng.addWidget(self.last)
        self.range_hint = QtWidgets.QLabel()
        rng.addWidget(self.range_hint, 1)
        form.addRow("Frames", rng)

        self.real = QtWidgets.QRadioButton("Real acquisition time")
        self.fixed = QtWidgets.QRadioButton("Fixed frame rate")
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.01, 100.0)
        self.speed.setDecimals(2)
        self.speed.setSingleStep(0.25)
        self.speed.setSuffix(" ×")
        self.speed.setToolTip("Playback speed: 1 = real time, 0.25 = four times slower, 2 = twice as fast")
        self.out_fps = QtWidgets.QComboBox()
        for r in OUTPUT_RATES:
            self.out_fps.addItem(f"{r} fps", r)
        self.out_fps.setToolTip("Frame rate of the video file; frames are held for their real duration, "
                                "accurate to one output frame")
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setSuffix(" fps")
        real_row = QtWidgets.QHBoxLayout()
        real_row.addWidget(self.real)
        real_row.addWidget(QtWidgets.QLabel("speed"))
        real_row.addWidget(self.speed)
        real_row.addWidget(QtWidgets.QLabel("written at"))
        real_row.addWidget(self.out_fps)
        real_row.addStretch(1)
        fixed_row = QtWidgets.QHBoxLayout()
        fixed_row.addWidget(self.fixed)
        fixed_row.addWidget(self.fps)
        fixed_row.addStretch(1)
        timing = QtWidgets.QVBoxLayout()
        timing.addLayout(real_row)
        timing.addLayout(fixed_row)
        form.addRow("Timing", timing)

        self.scale = QtWidgets.QComboBox()
        h, w = stack.shape[-2:]
        for f in (1, 2, 3, 4):
            self.scale.addItem(f"{f}×  ({w * f} × {h * f} px)", f)
        self.scale.setCurrentIndex(max(0, min(3, int(np.ceil(512 / max(h, w))) - 1)))
        form.addRow("Size", self.scale)

        ov = QtWidgets.QHBoxLayout()
        self.scalebar = QtWidgets.QCheckBox("Scale bar")
        self.scalebar.setChecked(bool(defaults.get("scalebar", True)) and bool(stack.pixel_size[0]))
        self.scalebar.setEnabled(bool(stack.pixel_size[0]))
        self.label = QtWidgets.QCheckBox("Time / z stamp")
        self.label.setChecked(True)
        ov.addWidget(self.scalebar)
        ov.addWidget(self.label)
        ov.addStretch(1)
        form.addRow("Overlays", ov)

        lut = QtWidgets.QLabel(f"{defaults.get('lut', 'Gray')}, {defaults.get('lo', 0):g} – {defaults.get('hi', 1):g}"
                               "  (current display settings)")
        lut.setStyleSheet("color: #9a9a9a;")
        form.addRow("Display", lut)

        path_row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)
        form.addRow("File", path_row)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(460)
        form.addRow("", self.summary)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.speed.setValue(float(defaults.get("speed", 1.0)))
        self.fps.setValue(int(defaults.get("fps", 25)))
        i = self.out_fps.findData(int(defaults.get("out_fps", 60)))
        self.out_fps.setCurrentIndex(i if i >= 0 else OUTPUT_RATES.index(60))
        (self.real if defaults.get("real_time", True) else self.fixed).setChecked(True)

        self.axis.currentIndexChanged.connect(self._axis_changed)
        for w in (self.first, self.last, self.speed, self.fps):
            w.valueChanged.connect(self._update)
        for w in (self.real, self.fixed):
            w.toggled.connect(self._update)
        self.out_fps.currentIndexChanged.connect(self._update)
        self.path.textEdited.connect(lambda _: setattr(self, "_user_path", True))
        browse.clicked.connect(self._browse)
        self._axis_changed()

    def _times(self):
        a = self.axis.currentIndex()
        return frame_times(self.stack, self.letters[a], self.shape[a])

    def _axis_changed(self) -> None:
        n = self.shape[self.axis.currentIndex()]
        for sb in (self.first, self.last):
            sb.blockSignals(True)
            sb.setRange(1, n)
        self.first.setValue(1)
        self.last.setValue(n)
        for sb in (self.first, self.last):
            sb.blockSignals(False)
        has_times = self._times() is not None
        self.real.setEnabled(has_times)
        if not has_times:
            self.fixed.setChecked(True)
        self._update()

    def _update(self) -> None:
        if self.first.value() > self.last.value():
            self.last.setValue(self.first.value())
        real = self.real.isChecked()
        for w in (self.speed, self.out_fps):
            w.setEnabled(real)
        self.fps.setEnabled(not real)
        first, last = self.first.value() - 1, self.last.value() - 1
        times = self._times()
        if real and times is not None:
            idx, rate = schedule(first, last, times, self.speed.value(), float(self.out_fps.currentData()))
            span = times[last] - times[first]
            self.range_hint.setText(f"{span:.3f} s recorded")
        else:
            idx, rate = schedule(first, last, fps=self.fps.value())
            self.range_hint.setText("")
        self.summary.setText(describe(idx, rate, last - first + 1))
        if not self._user_path:
            self.path.setText(self._default_path())

    def _default_path(self) -> str:
        timing = (f"realtime_{self.speed.value():g}x" if self.real.isChecked() else f"{self.fps.value()}fps")
        name = f"{self.defaults.get('stem', 'stack')}_{timing}.mp4"
        return os.path.join(self.defaults.get("folder", ""), name)

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save video", self.path.text(), "MP4 video (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.path.setText(path)
            self._user_path = True

    def _accept(self) -> None:
        if not self.path.text().strip():
            return
        if find_ffmpeg() is None:
            QtWidgets.QMessageBox.warning(self, "Export video", "ffmpeg was not found.\n\nInstall imageio-ffmpeg "
                                          "(conda install imageio-ffmpeg) or put ffmpeg on PATH.")
            return
        self.accept()

    def settings(self) -> VideoSettings:
        path = self.path.text().strip()
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        return VideoSettings(
            path=path, axis=self.axis.currentIndex(), first=self.first.value() - 1, last=self.last.value() - 1,
            real_time=self.real.isChecked(), speed=self.speed.value(), out_fps=float(self.out_fps.currentData()),
            fps=float(self.fps.value()), scale=int(self.scale.currentData()), scalebar=self.scalebar.isChecked(),
            label=self.label.isChecked(), lut=self.defaults.get("lut", "Gray"),
            lo=float(self.defaults.get("lo", 0.0)), hi=float(self.defaults.get("hi", 1.0)))

