"""
msr_volume.py -- volume rendering core of the 3D view (numpy only, no Qt).

A port of the renderer of the Fiji plugin 3Dscript (B. Schmid et al., "3Dscript:
animating 3D/4D microscopy data using a natural-language-based syntax", Nature
Methods 16:278-280, 2019; https://github.com/bene51/3Dscript), without the
animation language.  Orthographic ray casting with per-channel transfer functions
(intensity min/max/gamma for the colour, the same for the opacity), front-to-back
compositing ("independent transparency", "combined transparency") or a maximum
intensity projection, opacity correction for the sampling distance, and 3Dscript's
mouse interaction (drag rotates about the screen axes, shift-drag moves, the wheel
zooms about the cursor).

This module holds everything that is independent of Qt and OpenGL:

* the settings (ChannelSettings, RenderSettings) and the camera (Camera),
* the volume (Volume, VolumeChannel) and how it is read from image stacks, with
  downsampling to a GPU texture budget (load_volume),
* the ray geometry shared by the GLSL shader and the numpy reference renderer
  (ray_setup), the overlay geometry (box_edges, axes_tripod),
* render_cpu: the reference renderer (slow; tests and fallback).

The OpenGL widget is msr_volume_view.py, the controls are msr_volume_panel.py.

Portions are derived from 3Dscript, Copyright (c) 2016, Benjamin Schmid, under the
BSD 2-Clause License; see THIRD_PARTY_NOTICES.md for the full license text.

Conventions
-----------
* Arrays are (Z, Y, X), C order.  Voxel (x, y, z) is data[z, y, x].
* Model space: physical units (µm), origin at the centre of the volume box,
  +x = increasing column, +y = increasing row (down in the 2D view), +z =
  increasing plane index.  The box spans +-extent/2.
* Texture space: u = (x + 0.5) / W, v = (y + 0.5) / H, w = (z + 0.5) / D; the
  volume box is [0, 1]^3.
* Screen space (3Dscript's output space): x right, y down, z into the screen,
  orthographic.  S = zoom * R @ P + (pan_x, pan_y, 0) for a model point P (µm), so
  rotations are about the volume centre and the pan is in screen µm at zoom 1.
* Output pixels: (px, py) with row 0 at the top; the centre of pixel (col, row) is
  (col + 0.5, row + 0.5).  S.xy = ((px, py) - (w, h) / 2) * base_um_per_px(w, h).
  At zoom 1 the whole box fits the window at any rotation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

# rendering algorithms, in the order of 3Dscript's combo box
ALGORITHMS = ("independent", "combined", "mip")
ALGORITHM_LABELS = {
    "independent": "Independent transparency",
    "combined": "Combined transparency",
    "mip": "Maximum intensity",
}
# view presets of Camera.preset
PRESETS = ("3D", "XY", "XZ", "YZ")
RESOLUTIONS = ("auto", "full", "half", "quarter")   # resolution modes of the 3D tab (stack_factors)
MAX_CHANNELS = 8                # channels rendered at once (one 3D texture each)
DEFAULT_BUDGET = 64 * 2 ** 20    # voxels per channel in the GPU texture
DEFAULT_MAX_DIM = 2048           # texture side cap until the GL limit is known
FIT_MARGIN = 0.9                 # fraction of min(w, h) the box diagonal fills at zoom 1
ROT_SPEED = 0.7                  # degrees per dragged pixel (3Dscript)
WHEEL_STEP = 1.2                 # zoom factor per wheel notch
ZOOM_RANGE = (0.05, 50.0)        # zoom limits of the interaction
PRESET_3D = (-25.0, -60.0)       # rotate_by() angles of the "3D" preset from identity
SATURATED = 0.99                 # accumulated opacity at which a ray stops (3Dscript)
TF_EPS = 1e-12                   # smallest transfer-function window (max - min)
MAX_RAY_STEPS = 65536            # safety cap on the samples per ray (a 16384^3 box at 2 samples per voxel needs 56756)
MAX_VOXELS_CHANNEL = 512 * 2 ** 20   # hard cap of a loaded volume, per channel (any resolution mode) ...
MAX_VOXELS_TOTAL = 2 ** 30           # ... and over all channels (load_volume bins x/y further)
_CHUNK = 1 << 22                 # voxels per step of the histogram and NaN passes (bounded temporaries)


# -----------------------------------------------------------------------------
# settings
# -----------------------------------------------------------------------------

@dataclass
class ChannelSettings:
    """Rendering settings of one channel (3Dscript's per-channel properties).

    All intensities are raw data values.  In this version the opacity window
    follows the colour window (alpha_linked, like 3Dscript's start state); the
    opacity and lighting fields are kept for the next version.
    """
    name: str = ""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)   # 0..1, used when lut is None
    lut: str | None = None       # msr_widgets.LUTS key of a non-ramp table (Fire, Viridis, HiLo, Inverted)
    visible: bool = True
    weight: float = 1.0          # 0..1, 3Dscript's channel weight
    cmin: float = 0.0            # colour (intensity) transfer function
    cmax: float = 1.0
    cgamma: float = 1.0
    amin: float = 0.0            # opacity transfer function
    amax: float = 1.0
    agamma: float = 2.0
    alpha_linked: bool = True    # amin/amax follow cmin/cmax
    use_light: bool = False      # lighting (next version; stored, not rendered yet)
    k_object: float = 1.0
    k_diffuse: float = 0.0
    k_specular: float = 0.0
    shininess: float = 5.0
    key: Any = None              # the caller's identifier of the source stack

    def set_window(self, lo: float, hi: float) -> None:
        """Set the colour window (and the opacity window while it is linked)."""
        self.cmin, self.cmax = float(lo), float(hi)
        if self.alpha_linked:
            self.amin, self.amax = self.cmin, self.cmax

    def lut_table(self) -> np.ndarray:
        """(256, 3) uint8: LUTS[lut], or a ramp from black to *color*."""
        if self.lut:
            from msr_widgets import LUTS  # lazy: msr_widgets needs Qt, this module does not
            if self.lut in LUTS:
                return LUTS[self.lut]
        x = np.arange(256, dtype=np.float64)[:, None]
        rgb = np.clip(np.asarray(self.color[:3], dtype=np.float64), 0.0, 1.0)[None, :]
        return np.rint(x * rgb).astype(np.uint8)

    def copy(self) -> "ChannelSettings":
        return replace(self)


@dataclass
class RenderSettings:
    """Global rendering settings."""
    algorithm: str = "independent"                         # one of ALGORITHMS
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sampling: float = 1.0        # samples per voxel along the ray: 0.5 fast, 1 normal, 2 fine
    show_box: bool = True
    box_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    show_scalebar: bool = True
    show_axes: bool = True
    crop: tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)  # x0 x1 y0 y1 z0 z1 (texture space)
    double_alpha: bool = True    # 3Dscript's final A*C + (1-A)*bg (transparency modes); False: C + (1-A)*bg

    def copy(self) -> "RenderSettings":
        return replace(self)


# -----------------------------------------------------------------------------
# camera
# -----------------------------------------------------------------------------

def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _orthonormal(m: np.ndarray) -> np.ndarray:
    """The rotation closest to *m* (removes the drift of many small rotations)."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _extent(extent) -> np.ndarray:
    e = np.asarray(extent, dtype=np.float64).reshape(3)
    return np.where(np.isfinite(e) & (e > 0), e, 1.0)


# rotations of the axis-aligned presets (rows: the model axes that become screen x, y, z)
_PRESET_ROTATIONS = {
    "XY": np.eye(3),                                                          # x right, y down, looking along +z
    "XZ": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),    # x right, z down, looking along -y
    "YZ": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),     # y right, z down, looking along +x
}


@dataclass
class Camera:
    """Orthographic view: rotation about the volume centre, zoom and pan (see the module docstring)."""
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))   # model -> screen
    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)                             # screen µm at zoom 1

    def copy(self) -> "Camera":
        return Camera(np.array(self.rotation, dtype=np.float64), float(self.zoom), tuple(self.pan))

    @staticmethod
    def base_um_per_px(extent: tuple[float, float, float], w: float, h: float) -> float:
        """µm per output pixel at zoom 1: the box diagonal fills FIT_MARGIN * min(w, h)."""
        diag = float(np.linalg.norm(_extent(extent)))
        return diag / (FIT_MARGIN * max(min(float(w), float(h)), 1.0))

    def um_per_px(self, extent, w: float, h: float) -> float:
        """µm per output pixel in the screen plane (for the scale bar)."""
        return self.base_um_per_px(extent, w, h) / self.zoom

    def rotate_by(self, ax_deg: float, ay_deg: float) -> None:
        """R = Ry(ax) @ Rx(ay) @ R: rotation about the screen y and x axes (3Dscript).

        A drag by (dx, dy) pixels is rotate_by(-ROT_SPEED * dx, ROT_SPEED * dy): the front
        of the volume follows the mouse.
        """
        self.rotation = _orthonormal(_rot_y(ax_deg) @ _rot_x(ay_deg) @ np.asarray(self.rotation, dtype=np.float64))

    def pan_by_pixels(self, dx: float, dy: float, extent, w: float, h: float) -> None:
        """Move the volume by (dx, dy) output pixels."""
        mu0 = self.base_um_per_px(extent, w, h)
        self.pan = (float(self.pan[0] + dx * mu0), float(self.pan[1] + dy * mu0))

    def zoom_about(self, px: float, py: float, factor: float, extent, w: float, h: float) -> None:
        """Multiply the zoom by *factor*, keeping the point under output pixel (px, py) in place."""
        new = min(max(self.zoom * float(factor), ZOOM_RANGE[0]), ZOOM_RANGE[1])
        f = new / self.zoom
        mu0 = self.base_um_per_px(extent, w, h)
        sx, sy = (px - w / 2.0) * mu0, (py - h / 2.0) * mu0
        self.pan = (float(sx - f * (sx - self.pan[0])), float(sy - f * (sy - self.pan[1])))
        self.zoom = float(new)

    def fit(self, extent, w: float, h: float, margin: float = 0.92) -> None:
        """Zoom and centre so the projected box fills *margin* of the window (rotation unchanged)."""
        mu0 = self.base_um_per_px(extent, w, h)
        corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * _extent(extent)
        s = corners @ np.asarray(self.rotation, dtype=np.float64).T
        half_x, half_y = float(np.abs(s[:, 0]).max()), float(np.abs(s[:, 1]).max())
        zooms = [margin * n / 2.0 * mu0 / e for n, e in ((max(w, 1.0), half_x), (max(h, 1.0), half_y)) if e > 1e-12]
        zoom = min(zooms) if zooms else 1.0
        self.zoom = float(min(max(zoom, ZOOM_RANGE[0]), ZOOM_RANGE[1]))
        self.pan = (0.0, 0.0)

    def preset(self, name: str, extent, w: float, h: float) -> None:
        """Rotation of a preset (PRESETS), then fit().

        XY: identity (looking along +z, x right, y down: the 2D view).  XZ: x right, z down,
        looking along -y, i.e. from the +y side (the bottom edge of the 2D view); a proper
        rotation cannot look along +y with x right and z down.  YZ: y right, z down, looking
        along +x (from the -x side).  3D: rotate_by(-25, -60) from identity (PRESET_3D): the
        z = 0 face is seen from above at 60 degrees, so x stays to the right, y and z both run
        down the screen; the +y face is the front face below it and the -x face is the side
        face on the left.  This was checked on rendered images of an asymmetric test volume.
        """
        if name == "3D":
            self.rotation = np.eye(3)
            self.rotate_by(*PRESET_3D)
        elif name in _PRESET_ROTATIONS:
            self.rotation = _PRESET_ROTATIONS[name].copy()
        else:
            raise ValueError(f"unknown view preset {name!r}")
        self.fit(extent, w, h)

    def euler_deg(self) -> tuple[float, float, float]:
        """3Dscript's Euler angles (x, y, z) in degrees, R = Ry(y) @ Rz(z) @ Rx(x)."""
        m = np.asarray(self.rotation, dtype=np.float64)
        cz = math.hypot(m[0, 0], m[2, 0])   # cos z; the general formulas hold until z is within ~1e-9 of 90
        if cz < 1e-9:   # gimbal lock: only y + x (or y - x) is defined
            ax, ay, az = 0.0, math.atan2(m[0, 2], m[2, 2]), math.copysign(math.pi / 2, m[1, 0])
        else:
            ay = math.atan2(-m[2, 0], m[0, 0])
            az = math.atan2(m[1, 0], cz)
            ax = math.atan2(-m[1, 2], m[1, 1])
        return math.degrees(ax), math.degrees(ay), math.degrees(az)

    def set_euler_deg(self, ax: float, ay: float, az: float) -> None:
        """Rotation from 3Dscript's Euler angles in degrees (see euler_deg)."""
        self.rotation = _rot_y(ay) @ _rot_z(az) @ _rot_x(ax)

    def screen_from_model(self, extent, w: float, h: float) -> np.ndarray:
        """4x4 matrix: model point (µm, homogeneous) -> (px, py, depth µm, 1)."""
        mu0 = self.base_um_per_px(extent, w, h)
        view = np.eye(4)
        view[:3, :3] = self.zoom * np.asarray(self.rotation, dtype=np.float64)
        view[:2, 3] = self.pan
        pix = np.array([[1.0 / mu0, 0.0, 0.0, w / 2.0],
                        [0.0, 1.0 / mu0, 0.0, h / 2.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        return pix @ view

    def project(self, points: np.ndarray, extent, w: float, h: float) -> np.ndarray:
        """(N, 3) model points (µm) -> (N, 3) output (px, py, depth)."""
        m = self.screen_from_model(extent, w, h)
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return p @ m[:3, :3].T + m[:3, 3]


# -----------------------------------------------------------------------------
# volume
# -----------------------------------------------------------------------------

@dataclass
class VolumeChannel:
    """One channel of a volume, ready for a texture."""
    name: str
    data: np.ndarray             # (Z, Y, X) uint8 | uint16 | float32, C-contiguous
    data_min: float = 0.0
    data_max: float = 1.0
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    hist_lo: float = 0.0         # histogram range (data_min .. 99.99th percentile)
    hist_hi: float = 1.0
    window: tuple[float, float] = (0.0, 1.0)   # suggested display range (0.1 / 99.9 percentiles)
    key: Any = None              # the caller's identifier of the source stack

    @property
    def value_scale(self) -> float:
        """Raw value of a texture value of 1.0: 255 (uint8), 65535 (uint16) or 1 (float32)."""
        return {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(self.data.dtype, 1.0)


@dataclass
class Volume:
    """Channels on one (Z, Y, X) grid with their physical voxel size."""
    channels: list[VolumeChannel]
    voxel: tuple[float, float, float]          # (x, y, z) size of a data voxel in µm (after downsampling)
    unit: str = "µm"
    factor: tuple[int, int, int] = (1, 1, 1)   # downsampling (x, y, z)
    source_shape: tuple[int, int, int] = (1, 1, 1)   # (Z, Y, X) before downsampling
    source_voxel: tuple[float, float, float] = (1.0, 1.0, 1.0)  # voxel size before downsampling
    calibrated: bool = True                    # False when a voxel size was guessed
    note: str = ""                             # e.g. "z step unknown, assumed equal to x"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.channels[0].data.shape) if self.channels else (0, 0, 0)

    @property
    def extent(self) -> tuple[float, float, float]:
        """(x, y, z) size of the box in µm."""
        d, h, w = self.shape
        return (w * self.voxel[0], h * self.voxel[1], d * self.voxel[2])

    @property
    def nbytes(self) -> int:
        return sum(int(c.data.nbytes) for c in self.channels)


def choose_factors(shape_zyx: tuple[int, int, int], voxel_xyz: tuple[float, float, float],
                   max_dim: int = DEFAULT_MAX_DIM, budget: int = DEFAULT_BUDGET) -> tuple[int, int, int]:
    """Integer downsampling (fx, fy, fz) so the volume fits *budget* voxels and *max_dim* per side.

    x and y are reduced together first.  At each x/y factor z is first kept (binned only
    as far as the side cap requires), then binned as far as its voxels stay no coarser
    than the x/y voxels (stacks whose z step is finer than the pixels); only then are x
    and y reduced further.
    """
    nz, ny, nx = (max(1, int(n)) for n in shape_zyx)
    vx, vy, vz = (float(v) if v and math.isfinite(v) and v > 0 else 1.0 for v in voxel_xyz)
    max_dim, budget = max(1, int(max_dim)), max(1, int(budget))
    vxy = min(vx, vy)
    fz_cap = min(-(-nz // max_dim), nz)
    for f in range(1, max(nx, ny) + 1):
        cx, cy = -(-nx // f), -(-ny // f)
        if max(cx, cy) > max_dim:
            continue
        # z voxels never become coarser than the x/y voxels: largest fz with fz * vz <= f * vxy
        fz_iso = min(max(fz_cap, int(f * vxy / vz + 1e-9), 1), nz)
        for fz in (fz_cap, fz_iso):
            if cx * cy * -(-nz // fz) <= budget:
                return f, f, fz
    f = max(nx, ny)   # one voxel in x/y and still too large: only z is left
    return f, f, min(max(fz_cap, -(-nz // budget)), nz)


def stack_factors(stack, mode: str = "auto", max_dim: int = DEFAULT_MAX_DIM,
                  budget: int = DEFAULT_BUDGET) -> tuple[int, int, int]:
    """Downsampling (fx, fy, fz) of a stack's volume for a resolution *mode* (RESOLUTIONS).

    auto: choose_factors with *budget*; full: only what the texture side cap *max_dim*
    requires (in x/y and in z separately); half / quarter: the full factors times 2 / 4
    in x and y, and in z as far as the z voxels stay no coarser than the x/y voxels (the
    rule of choose_factors).  load_volume caps the result at MAX_VOXELS_CHANNEL /
    MAX_VOXELS_TOTAL (cap_factors).
    """
    zi = stack.axes[:-2].find("Z")
    if zi < 0:
        raise ValueError(f"the stack has no Z axis (axes {stack.axes})")
    shape = (int(stack.shape[zi]), int(stack.shape[-2]), int(stack.shape[-1]))
    voxel = _voxel_size(stack)[0]
    if mode == "auto":
        return choose_factors(shape, voxel, max_dim, budget)
    k = {"full": 1, "half": 2, "quarter": 4}.get(mode)
    if k is None:
        raise ValueError(f"unknown resolution {mode!r}")
    nz, ny, nx = (max(1, n) for n in shape)
    max_dim = max(1, int(max_dim))
    fx = fy = min(-(-max(nx, ny) // max_dim), max(nx, ny))   # the side cap only (a fine z step stays)
    fz = min(-(-nz // max_dim), nz)
    if k > 1:
        fx, fy = min(fx * k, nx), min(fy * k, ny)
        coarsest = int(min(fx * voxel[0], fy * voxel[1]) / voxel[2] + 1e-9)   # z voxel <= x/y voxel
        fz = min(max(fz, min(fz * k, coarsest)), nz)
    return fx, fy, fz


def cap_factors(shape_zyx: tuple[int, int, int], factor: tuple[int, int, int], n_channels: int = 1,
                per_channel: int = MAX_VOXELS_CHANNEL,
                total: int = MAX_VOXELS_TOTAL) -> tuple[tuple[int, int, int], bool]:
    """((fx, fy, fz), capped): *factor* binned further in x/y (z only if x/y cannot) so a
    channel has at most *per_channel* voxels and *n_channels* channels at most *total*."""
    nz, ny, nx = (max(1, int(n)) for n in shape_zyx)
    fx, fy, fz = (max(1, int(f)) for f in factor)
    fx, fy, fz = min(fx, nx), min(fy, ny), min(fz, nz)
    limit = max(1, min(int(per_channel), int(total) // max(1, int(n_channels))))

    def voxels(fx: int, fy: int, fz: int) -> int:
        return -(-nz // fz) * -(-ny // fy) * -(-nx // fx)

    if voxels(fx, fy, fz) <= limit:
        return (fx, fy, fz), False
    for f in range(max(fx, fy) + 1, max(nx, ny) + 1):
        gx, gy = min(max(fx, f), nx), min(max(fy, f), ny)
        if voxels(gx, gy, fz) <= limit:
            return (gx, gy, fz), True
    fx, fy = nx, ny   # one voxel in x/y and still too large: z too
    return (fx, fy, min(max(fz, -(-nz // limit)), nz)), True


# length units -> µm, for a z step stored in another unit than x/y
_LENGTH_UM = {"µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0, "micrometer": 1.0, "nm": 1e-3,
              "nanometer": 1e-3, "pm": 1e-6, "å": 1e-4, "mm": 1e3, "millimeter": 1e3, "cm": 1e4, "m": 1e6,
              "meter": 1e6}


def _unit_um(unit) -> float | None:
    u = str(unit or "").strip().replace("μ", "µ").lower()
    return _LENGTH_UM.get(u)


def _voxel_size(stack) -> tuple[tuple[float, float, float], str, bool, str]:
    """((x, y, z) voxel size, unit, calibrated, note) of a stack, with fallbacks for missing steps."""
    def unit(letter: str) -> str:
        return str((stack.axis_info(letter) or {}).get("unit") or "")

    def step(letter: str) -> float | None:
        v = stack.step(letter)
        return float(v) if v is not None and math.isfinite(v) and v > 0 else None

    vx, vy, vz = step("X"), step("Y"), step("Z")
    notes = []
    if vx is None and vy is None:
        return (1.0, 1.0, 1.0), "px", False, "no pixel size: 1 px per voxel"
    if vx is None or vy is None:
        vx = vy = vx or vy
        notes.append("x/y pixel size assumed square")
    xunit = unit("X") or unit("Y") or "µm"
    if vz is None:
        vz = vx
        notes.append("z step unknown, assumed equal to x")
    else:
        sx, sz = _unit_um(xunit), _unit_um(unit("Z") or xunit)
        if sx and sz and sx != sz:
            vz = vz * sz / sx
    return (float(vx), float(vy), float(vz)), xunit, not notes, "; ".join(notes)


def _block_mean(plane: np.ndarray, fy: int, fx: int) -> np.ndarray:
    """Mean of fy x fx blocks; the last row/column of blocks averages the pixels that exist."""
    h, w = plane.shape
    hc, wc = -(-h // fy), -(-w // fx)
    work = np.float32 if fy * fx <= 256 and plane.dtype.itemsize <= 2 else np.float64
    a = plane.astype(work)
    if hc * fy != h or wc * fx != w:
        padded = np.zeros((hc * fy, wc * fx), work)
        padded[:h, :w] = a
        a = padded
        rows = np.minimum(fy, h - np.arange(hc) * fy).astype(work)
        cols = np.minimum(fx, w - np.arange(wc) * fx).astype(work)
        return a.reshape(hc, fy, wc, fx).sum(axis=(1, 3)) / np.outer(rows, cols)
    return a.reshape(hc, fy, wc, fx).mean(axis=(1, 3), dtype=work)


def _store(block: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """A block mean in the output dtype (rounded and clipped for integers)."""
    if dtype == np.float32 or block.dtype == dtype:
        return block
    info = np.iinfo(dtype)
    return np.clip(np.rint(block), info.min, info.max)


def _percentile_counts(cdf: np.ndarray, q: float) -> float:
    """Smallest value whose cumulative count reaches q percent (at least one voxel)."""
    return float(np.searchsorted(cdf, max(q / 100.0 * cdf[-1], 1), side="left"))


def _channel(name: str, data: np.ndarray, key) -> VolumeChannel:
    """A VolumeChannel with the range, histogram and suggested window of *data* (sanitised in place).

    Works in chunks of _CHUNK voxels: np.bincount converts its input to intp, which for the
    whole volume would need 8 bytes per voxel on top of the data.
    """
    flat = data.reshape(-1)
    chunks = range(0, flat.size, _CHUNK)
    if data.dtype in (np.uint8, np.uint16):
        nbins = 256 if data.dtype == np.uint8 else 65536
        counts = np.zeros(nbins, np.int64)
        for i in chunks:
            counts += np.bincount(flat[i:i + _CHUNK], minlength=nbins)
        used = np.flatnonzero(counts)
        dmin, dmax = float(used[0]), float(used[-1])
        cdf = np.cumsum(counts)
        p_lo, p_hi, p_top = (_percentile_counts(cdf, q) for q in (0.1, 99.9, 99.99))
        hist_lo, hist_hi = dmin, (p_top if p_top > dmin else dmax)
        values, weights = np.arange(counts.size, dtype=np.float64), counts
    else:
        lo, hi, bad = math.inf, -math.inf, False
        for i in chunks:
            part = flat[i:i + _CHUNK]
            finite = np.isfinite(part)
            if not finite.all():
                bad = True
                part = part[finite]
            if part.size:
                lo, hi = min(lo, float(part.min())), max(hi, float(part.max()))
        if bad:
            fill = lo if lo <= hi else 0.0
            for i in chunks:                        # the shader must never see NaN or inf
                part = flat[i:i + _CHUNK]
                part[~np.isfinite(part)] = fill
            lo, hi = min(lo, fill), max(hi, fill)
        dmin, dmax = (lo, hi) if flat.size else (0.0, 1.0)
        sample = flat[::max(1, flat.size // 4_000_000)]   # percentiles and histogram from <= ~4 M voxels
        p_lo, p_hi, p_top = (float(v) for v in np.percentile(sample, [0.1, 99.9, 99.99]))
        hist_lo, hist_hi = dmin, (p_top if p_top > dmin else dmax)
        values, weights = sample, None
    if hist_hi <= hist_lo:
        hist_hi = hist_lo + 1.0
    hist = np.histogram(values, bins=512, range=(hist_lo, hist_hi), weights=weights)[0]
    if not p_hi > p_lo:
        p_lo, p_hi = dmin, dmax
    if not p_hi > p_lo:
        p_hi = p_lo + 1.0
    return VolumeChannel(name=name, data=data, data_min=dmin, data_max=dmax,
                         histogram=np.rint(hist).astype(np.int64), hist_lo=float(hist_lo), hist_hi=float(hist_hi),
                         window=(float(p_lo), float(p_hi)), key=key)


def load_volume(sources: list[tuple[str, Any, Any]], index: tuple, *,
                max_dim: int = DEFAULT_MAX_DIM, budget: int = DEFAULT_BUDGET,
                factor: tuple[int, int, int] | None = None,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> Volume:
    """Read the (Z, Y, X) volume of each source at *index*.

    *sources*: (name, stack, key) per channel; stack is an msr_reader.DataStack (mosaics
    already frozen by the caller), all with the same axes and shape.  *index*: one entry
    per axis of stack.axes[:-2] (the entry of the Z axis is ignored and may be a slice;
    missing entries are 0).  *factor* forces a downsampling, otherwise choose_factors
    decides; either is capped by cap_factors (MAX_VOXELS_CHANNEL per channel,
    MAX_VOXELS_TOTAL in all; the volume's note then says "capped").  Planes are read
    one at a time and block averaged (x/y blocks at the right and bottom edges, and the
    last group of planes, average the pixels that exist; the voxel size is factor times
    the source voxel), and *progress(done, total)* /
    *cancelled()* are called per plane (InterruptedError when cancelled).  uint8/uint16
    data stay integer (rounded means), everything else becomes float32 (NaN and inf are
    replaced by the smallest finite value).  Raises ValueError if the stacks have no Z axis
    or differ in axes or shape.
    """
    if not sources:
        raise ValueError("no channels to load")
    first = sources[0][1]
    axes, shape = first.axes, tuple(int(n) for n in first.shape)
    lead = axes[:-2]
    zi = lead.find("Z")
    if zi < 0:
        raise ValueError(f"{sources[0][0] or 'the stack'} has no Z axis (axes {axes})")
    for name, stack, _key in sources[1:]:
        if stack.axes != axes or tuple(int(n) for n in stack.shape) != shape:
            raise ValueError(f"{name}: axes {stack.axes} {tuple(stack.shape)} differ from {axes} {shape}")
    nz, ny, nx = shape[zi], shape[-2], shape[-1]
    index = tuple(index)
    idx = [0 if k == zi or k >= len(index) else int(index[k]) for k in range(len(lead))]
    voxel, unit, calibrated, note = _voxel_size(first)
    if factor is None:
        factor = choose_factors((nz, ny, nx), voxel, max_dim, budget)
    (fx, fy, fz), capped = cap_factors((nz, ny, nx), factor, len(sources), MAX_VOXELS_CHANNEL, MAX_VOXELS_TOTAL)
    if capped:
        note = "; ".join(t for t in (note, f"binned further (capped): at most {MAX_VOXELS_CHANNEL / 2 ** 20:g} M "
                                           f"voxels per channel and {MAX_VOXELS_TOTAL / 2 ** 30:g} G in all") if t)
    cz, cy, cx = -(-nz // fz), -(-ny // fy), -(-nx // fx)
    total, done = len(sources) * nz, 0
    channels = []
    for name, stack, key in sources:
        arr = stack.asarray()
        dt = np.dtype(arr.dtype)
        out_dtype = (np.dtype(np.uint8) if dt.kind == "u" and dt.itemsize == 1 else
                     np.dtype(np.uint16) if dt.kind == "u" and dt.itemsize == 2 else np.dtype(np.float32))
        out = np.empty((cz, cy, cx), out_dtype)
        acc, n_acc = None, 0
        for z in range(nz):
            if cancelled is not None and cancelled():
                raise InterruptedError("loading the volume was cancelled")
            idx[zi] = z
            plane = np.asarray(arr[tuple(idx)])
            if fx == fy == fz == 1:
                out[z] = plane                       # casts float64 etc. to float32
            else:
                block = _block_mean(plane, fy, fx) if fx > 1 or fy > 1 else plane.astype(np.float64)
                if fz == 1:
                    out[z] = _store(block, out_dtype)
                else:
                    acc = block.astype(np.float64) if acc is None else acc + block
                    n_acc += 1
                    if n_acc == fz or z == nz - 1:
                        out[z // fz] = _store(acc / n_acc, out_dtype)
                        acc, n_acc = None, 0
            done += 1
            if progress is not None:
                progress(done, total)
        # a frozen mosaic caches every composed plane; each plane was needed only once
        cache = getattr(arr, "_cache", None)
        if hasattr(cache, "clear"):
            cache.clear()
        del arr
        channels.append(_channel(name, out, key))
    return Volume(channels=channels, voxel=(voxel[0] * fx, voxel[1] * fy, voxel[2] * fz), unit=unit,
                  factor=(fx, fy, fz), source_shape=(nz, ny, nx), source_voxel=voxel, calibrated=calibrated,
                  note=note)


# -----------------------------------------------------------------------------
# ray geometry, overlays, reference renderer
# -----------------------------------------------------------------------------

@dataclass
class RaySetup:
    """Per-frame ray geometry, shared by the shader and render_cpu."""
    tex_from_pixel: np.ndarray   # 4x4: (px, py, s, 1) -> (u, v, w, 1); s = depth along the ray in screen µm
    step: float                  # distance between samples in s units
    alphacorr: float             # opacity correction exponent: physical step / reference length
    max_steps: int               # cap on samples per ray


def ray_setup(camera: Camera, volume: Volume, w: int, h: int, sampling: float = 1.0,
              step_scale: float = 1.0) -> RaySetup:
    """Ray geometry for an output of w x h pixels (see the design notes for the formulas).

    M = T(0.5) @ diag(1/E) @ (1/zoom) [R^T | -R^T pan] @ Spix maps an output pixel and a
    depth s (screen µm) to texture coordinates; it is the inverse of
    Camera.screen_from_model up to the texture affine.  The samples are one voxel apart
    along the ray direction (scaled by step_scale / sampling), and the opacity
    correction refers to the x voxel of the source data, so downsampling keeps the look.
    """
    ext = _extent(volume.extent)
    w, h = max(int(w), 1), max(int(h), 1)
    mu0 = Camera.base_um_per_px(ext, w, h)
    rot = np.asarray(camera.rotation, dtype=np.float64)
    zoom = float(camera.zoom)
    pix = np.array([[mu0, 0.0, 0.0, -w / 2.0 * mu0],
                    [0.0, mu0, 0.0, -h / 2.0 * mu0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    model = np.eye(4)
    model[:3, :3] = rot.T / zoom
    model[:3, 3] = -rot.T @ np.array([camera.pan[0], camera.pan[1], 0.0]) / zoom
    tex = np.diag([1.0 / ext[0], 1.0 / ext[1], 1.0 / ext[2], 1.0])
    tex[:3, 3] = 0.5
    m = tex @ model @ pix
    n = rot[2]                                   # R^T e_z: the ray direction in model space
    vox = np.array([v if v > 0 else 1.0 for v in volume.voxel], dtype=np.float64)
    base = 1.0 / math.sqrt(float(np.sum((n / vox) ** 2)))
    step_phys = base * float(step_scale) / max(float(sampling), 1e-6)
    ref = float(volume.source_voxel[0]) if volume.source_voxel[0] > 0 else float(vox[0])
    step = step_phys * zoom
    # no chord through the box is longer than its diagonal: enough samples for any ray (MAX_RAY_STEPS: safety)
    max_steps = int(min(MAX_RAY_STEPS, math.ceil(float(np.linalg.norm(ext)) * zoom / step) + 2))
    return RaySetup(tex_from_pixel=m, step=step, alphacorr=step_phys / ref, max_steps=max_steps)


_CORNERS = np.array(list(itertools.product((-0.5, 0.5), repeat=3)))      # index = 4 * ix + 2 * iy + iz


def box_edges(camera: Camera, volume: Volume, w: int, h: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Edges of the box faces that face the viewer, in output pixels (3Dscript's bounding box)."""
    if volume is None or not volume.channels:
        return []
    ext = _extent(volume.extent)
    rot = np.asarray(camera.rotation, dtype=np.float64)
    pts = camera.project(_CORNERS * ext, ext, w, h)
    edges: set[tuple[int, int]] = set()
    for axis in range(3):
        for side in (0, 1):
            normal = np.zeros(3)
            normal[axis] = 1.0 if side else -1.0
            if (rot @ normal)[2] >= -1e-9:       # faces away from the viewer, or edge-on
                continue
            face = [i for i in range(8) if (i >> (2 - axis)) & 1 == side]
            for a, b in itertools.combinations(face, 2):
                if bin(a ^ b).count("1") == 1:   # corners differ in one coordinate: an edge
                    edges.add((a, b))
    return [((float(pts[a, 0]), float(pts[a, 1])), (float(pts[b, 0]), float(pts[b, 1]))) for a, b in sorted(edges)]


def axes_tripod(camera: Camera, length: float = 40.0) -> list[tuple[str, tuple[float, float]]]:
    """('x' | 'y' | 'z', (dx, dy)) screen offsets of the three model axes, for an orientation tripod.

    Sorted back to front (the axis pointing furthest into the screen first), so drawing
    in this order puts the axis nearest to the viewer on top.
    """
    rot = np.asarray(camera.rotation, dtype=np.float64)
    order = sorted(range(3), key=lambda i: -rot[2, i])
    return [("xyz"[i], (float(rot[0, i] * length), float(rot[1, i] * length))) for i in order]


def rendered_channels(channels: list[ChannelSettings], volume: Volume | None) -> list[int]:
    """Indices of the channels that are rendered: visible, weight > 0, with data; at most MAX_CHANNELS."""
    if volume is None:
        return []
    n = min(len(channels), len(volume.channels))
    return [i for i in range(n) if channels[i].visible and channels[i].weight > 0][:MAX_CHANNELS]


def _slab(u0: np.ndarray, du: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """(hit, s_near, s_far) of rays u0 + s * du against the box [lo, hi] (slab test)."""
    n = u0.shape[0]
    s0, s1 = np.full(n, -np.inf), np.full(n, np.inf)
    hit = np.ones(n, bool)
    for i in range(3):
        if abs(du[i]) < 1e-12:
            hit &= (u0[:, i] >= lo[i]) & (u0[:, i] <= hi[i])
        else:
            t1, t2 = (lo[i] - u0[:, i]) / du[i], (hi[i] - u0[:, i]) / du[i]
            s0 = np.maximum(s0, np.minimum(t1, t2))
            s1 = np.minimum(s1, np.maximum(t1, t2))
    return hit & (s1 > s0), s0, s1


def _trilinear(flat: np.ndarray, shape: tuple[int, int, int], u: np.ndarray) -> np.ndarray:
    """Values at texture coordinates u (n, 3): trilinear with clamp-to-edge, like GL_LINEAR."""
    d, h, w = shape
    x = np.clip(u[:, 0] * w - 0.5, 0.0, w - 1)
    y = np.clip(u[:, 1] * h - 0.5, 0.0, h - 1)
    z = np.clip(u[:, 2] * d - 0.5, 0.0, d - 1)
    x0, y0, z0 = (np.floor(c).astype(np.intp) for c in (x, y, z))
    fx, fy, fz = x - x0, y - y0, z - z0
    x1, y1, z1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1), np.minimum(z0 + 1, d - 1)

    def at(zz, yy, xx):
        return flat[(zz * h + yy) * w + xx].astype(np.float64)

    c00 = at(z0, y0, x0) * (1 - fx) + at(z0, y0, x1) * fx
    c01 = at(z0, y1, x0) * (1 - fx) + at(z0, y1, x1) * fx
    c10 = at(z1, y0, x0) * (1 - fx) + at(z1, y0, x1) * fx
    c11 = at(z1, y1, x0) * (1 - fx) + at(z1, y1, x1) * fx
    return (c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz


def _lut_lookup(table: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Colours (n, 3) of a (256, 3) 0..1 table, linearly interpolated at index r * 255."""
    x = np.clip(r, 0.0, 1.0) * 255.0
    i0 = np.minimum(np.floor(x).astype(np.intp), 254)
    f = (x - i0)[:, None]
    return table[i0] * (1.0 - f) + table[i0 + 1] * f


def render_cpu(volume: Volume, channels: list[ChannelSettings], settings: RenderSettings,
               camera: Camera, w: int, h: int, step_scale: float = 1.0) -> np.ndarray:
    """Reference renderer: (h, w, 3) uint8, the same math as the shader (slow)."""
    w, h = max(int(w), 1), max(int(h), 1)
    bg = np.clip(np.asarray(settings.background[:3], dtype=np.float64), 0.0, 1.0)
    out = np.empty((h * w, 3))
    out[:] = bg
    active = rendered_channels(channels, volume)
    if active:
        _cast_rays(volume, channels, active, settings, camera, w, h, step_scale, out)
    return np.floor(np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(h, w, 3)


def _cast_rays(volume: Volume, channels: list[ChannelSettings], active: list[int], settings: RenderSettings,
               camera: Camera, w: int, h: int, step_scale: float, out: np.ndarray, chunk: int = 32768) -> None:
    """Fill the hit pixels of *out* ((h * w, 3), background already set), in chunks of rays."""
    rs = ray_setup(camera, volume, w, h, settings.sampling, step_scale)
    m = rs.tex_from_pixel
    du = m[:3, 2]
    crop = np.asarray(settings.crop, dtype=np.float64)
    lo, hi = crop[0::2], crop[1::2]
    bg = out[0].copy()
    mip = settings.algorithm == "mip"
    wcomb = 1.0 if settings.algorithm == "combined" else 0.0
    chans = [channels[i] for i in active]
    datas = [volume.channels[i].data for i in active]
    flats = [d.reshape(-1) for d in datas]
    shape = datas[0].shape
    luts = [c.lut_table().astype(np.float64) / 255.0 for c in chans]
    weights = np.array([c.weight for c in chans], dtype=np.float64)
    nch = len(chans)

    def ramp(v, a, b, gamma):
        t = np.clip((v - a) / max(b - a, TF_EPS), 0.0, 1.0)
        return t if gamma == 1.0 else t ** gamma

    rows = max(1, chunk // w)
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        py, px = np.mgrid[r0:r1, 0:w]
        px, py = px.ravel() + 0.5, py.ravel() + 0.5
        u0 = np.outer(px, m[:3, 0]) + np.outer(py, m[:3, 1]) + m[:3, 3]
        hit, s0, s1 = _slab(u0, du, lo, hi)
        ids = np.flatnonzero(hit)
        if not ids.size:
            continue
        u0, s0, s1 = u0[ids], s0[ids], s1[ids]
        n = ids.size
        # samples k = 0, 1, ... with s0 + (k + 0.5) * step < s1, at most max_steps (as the shader loop)
        nsamp = np.clip(np.ceil((s1 - s0) / rs.step - 0.5), 0, rs.max_steps).astype(np.int64)
        done = nsamp == 0
        if mip:
            best = np.full((n, nch), -1.0)
            vbest = np.zeros((n, nch))
        else:
            color = np.zeros((n, nch, 3))
            alpha = np.zeros((n, nch))
        for k in range(int(nsamp.max(initial=0))):
            live = np.flatnonzero(~done & (nsamp > k))
            if not live.size:
                break
            p = u0[live] + ((s0[live] + (k + 0.5) * rs.step)[:, None]) * du
            for c, cs in enumerate(chans):
                if mip:
                    v = _trilinear(flats[c], shape, p)
                    t = np.clip((v - cs.cmin) / max(cs.cmax - cs.cmin, TF_EPS), 0.0, 1.0)
                    upd = t > best[live, c]          # first maximum wins (t^gamma is monotonic)
                    rays = live[upd]
                    best[rays, c] = t[upd]
                    vbest[rays, c] = v[upd]
                    continue
                gate = alpha[live, c] < SATURATED
                rays = live[gate]
                if not rays.size:
                    continue
                v = _trilinear(flats[c], shape, p[gate])
                rc = ramp(v, cs.cmin, cs.cmax, cs.cgamma)
                ra = ramp(v, cs.amin, cs.amax, cs.agamma)
                ra = 1.0 - (1.0 - ra) ** rs.alphacorr
                a = alpha[rays, c] + wcomb * (alpha[rays].sum(axis=1) - alpha[rays, c])
                tmp = cs.weight * np.maximum(1.0 - a, 0.0) * ra
                color[rays, c] += tmp[:, None] * _lut_lookup(luts[c], rc)
                alpha[rays, c] += tmp
            if mip:
                done[live] = (best[live] >= 1.0).all(axis=1)
            elif wcomb:
                done[live] = alpha[live].sum(axis=1) >= SATURATED
            else:
                done[live] = (alpha[live] >= SATURATED).all(axis=1)
        if mip:
            col = np.zeros((n, 3))
            a_sum = np.zeros(n)
            for c, cs in enumerate(chans):
                got = best[:, c] >= 0.0
                rc = best[got, c] if cs.cgamma == 1.0 else best[got, c] ** cs.cgamma
                col[got] += cs.weight * _lut_lookup(luts[c], rc)
                a_sum[got] += cs.weight * ramp(vbest[got, c], cs.amin, cs.amax, cs.agamma)
            a_tot = np.clip(a_sum, 0.0, 1.0)[:, None]
            out[r0 * w + ids] = col + (1.0 - a_tot) * bg
        else:
            col = np.einsum("c,nck->nk", weights, color)
            a_tot = np.clip(alpha @ weights, 0.0, 1.0)[:, None]
            out[r0 * w + ids] = (a_tot * col if settings.double_alpha else col) + (1.0 - a_tot) * bg
