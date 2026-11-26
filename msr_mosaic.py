#!/usr/bin/env python3
"""
msr_mosaic.py -- mosaics (tile scans) in ImSpector .msr files: layout, stitching and fusion.

ImSpector stores every tile of a mosaic as its own data stack ("PMT - xyz-Table[1] -
xyz-Table[2]").  The stack header says which loop element the stack is (split
index and count per loop, e.g. xyz-Table X 2 of 3) and where the stage was
(offset of X / Y in µm, as ImSpector's own TIFF export writes it).  This module

* finds the tiles of a mosaic (one mosaic per detector) and places them at their
  stage positions,
* optionally registers the overlaps (cross-correlation of high-pass filtered
  overlap regions, best of several peaks by normalised cross-correlation,
  sub-pixel), and solves all pairwise shifts together (weighted least squares,
  outlier links dropped), in 2D or, for z-stacks, in 3D with an optional z shift,
* fuses the tiles into one image (linear blending across the overlaps, average,
  maximum or plain overwrite).

The fused mosaic is a :class:`MosaicStack`, a :class:`msr_reader.DataStack` whose
``asarray()`` composes planes on demand, so the viewer, the TIFF export and the
video export treat it like any other stack.  Only numpy is needed (scipy is used
for the Gaussian filter when installed).

Command line::

    python msr_mosaic.py FILE.msr [--align] [--blend linear|average|max|overwrite]
                         [--source max|volume] [--z-shift N] [--imagej] [-o OUTDIR] [--info]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import math
import os
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msr_reader as mr  # noqa: E402

__version__ = "0.6.1"

BLEND_MODES = {
    "linear": "Linear blend (feather across the overlap)",
    "average": "Average of the overlapping tiles",
    "max": "Maximum of the overlapping tiles",
    "overwrite": "None (later tiles on top, hard seams)",
}

# How stage coordinates map onto mosaic pixels: (transpose, sign x, sign y).
# Default: stage +X -> image right, stage +Y -> image up.  Verified on a 3x3 TriMScope
# mosaic (ImSpector 7.6.4): all 12 overlaps correlate only in this orientation.
DEFAULT_ORIENTATION = (False, 1, -1)
ORIENTATIONS = [(t, sx, sy) for t in (False, True) for sx in (1, -1) for sy in (-1, 1)]


def orientation_label(o: tuple) -> str:
    t, sx, sy = o
    a, b = ("Y", "X") if t else ("X", "Y")
    return (f"stage {a} {'→' if sx > 0 else '←'}, stage {b} {'↓' if sy > 0 else '↑'}"
            + ("  (default)" if tuple(o) == DEFAULT_ORIENTATION else ""))


# -----------------------------------------------------------------------------
# finding the tiles
# -----------------------------------------------------------------------------

_NAME_RE = re.compile(r"^(?P<base>.+?) - (?P<ay>[^\[\]]+?)\[(?P<iy>\d+)\] - (?P<ax>[^\[\]]+?)\[(?P<ix>\d+)\]\s*$")
_CHANNEL_RE = re.compile(r"^(?P<dev>.*?):(?P<a>\d+):(?P<det>\d+):(?P<n>\d+)$")


@dataclass
class Tile:
    stack: mr.DataStack
    pos: int                      # index of the stack in MSRFile.stacks
    ix: int                       # loop index along the stage X / Y axis
    iy: int
    stage: tuple | None = None    # (x, y) stage position in µm, None if unknown

    @property
    def label(self) -> str:
        return f"x{self.ix} y{self.iy}"


def _axis_is(name: str, letter: str) -> bool:
    words = name.strip().upper().split()
    return bool(words) and words[-1] == letter


def _tile_index(s: mr.DataStack) -> tuple[int, int, int, int] | None:
    """(ix, nx, iy, ny) of a mosaic tile, None for a normal stack."""
    split = s.split_axes
    xs = [a for a in split if _axis_is(a[0], "X")]
    ys = [a for a in split if _axis_is(a[0], "Y")]
    if xs or ys:
        ix, nx = (xs[0][1], xs[0][2]) if xs else (0, 1)
        iy, ny = (ys[0][1], ys[0][2]) if ys else (0, 1)
        if nx * ny > 1 and 0 <= ix < nx and 0 <= iy < ny:
            return ix, nx, iy, ny
    m = _NAME_RE.match(s.source or "")  # "<source> - <axis>[iy] - <axis>[ix]", outer loop first
    if m and ("Mosaic" in s.meta.get("Measurement Mode", "") or "table" in m.group("ax").lower()):
        return int(m.group("ix")), 0, int(m.group("iy")), 0
    return None


def _group_key(s: mr.DataStack) -> tuple:
    m = _CHANNEL_RE.match(s.channel_id or "")
    det = (m.group("dev"), m.group("a"), m.group("det")) if m else (s.channel_id,)
    base = (s.source or "").split(" - ")[0]
    return (s.name, base, det, s.meta.get("Creation Date", ""), tuple(s.sizes),
            tuple(round(float(x), 3) for x in s.lengths), tuple(s.labels[2:]), s.dtype)


def find_mosaics(msr) -> list["Mosaic"]:
    """All mosaics of an opened file (one per detector and acquisition)."""
    groups: dict[tuple, list[Tile]] = {}
    for pos, s in enumerate(msr.stacks):
        if isinstance(s, MosaicStack):
            continue
        idx = _tile_index(s)
        if idx is None:
            continue
        off = s.offsets
        stage = (off[0], off[1]) if off and all(math.isfinite(v) for v in off[:2]) else None
        groups.setdefault(_group_key(s), []).append(Tile(s, pos, idx[0], idx[2], stage))
    out = []
    for key, tiles in groups.items():
        if len(tiles) < 2 or len({(t.ix, t.iy) for t in tiles}) != len(tiles):
            continue
        out.append(Mosaic(tiles, msr))
    return out


# -----------------------------------------------------------------------------
# image helpers
# -----------------------------------------------------------------------------

def fast_len(n: int) -> int:
    """Smallest length >= n without prime factors above 5 (fast FFT sizes)."""
    n = max(1, int(n))
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n:
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def highpass(a: np.ndarray, fine: float = 1.0, coarse: float = 8.0) -> np.ndarray:
    """Difference of Gaussians over the last two axes (reflecting edges), as float32.

    Keeps structures of a few pixels (cells, particles, edges) and removes noise and the
    smooth background (vignetting, out-of-focus haze) that would dominate a correlation.
    """
    a = np.asarray(a, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        sig = (0,) * (a.ndim - 2)
        return (gaussian_filter(a, sig + (fine, fine), mode="reflect")
                - gaussian_filter(a, sig + (coarse, coarse), mode="reflect"))
    except ImportError:
        pass
    h, w = a.shape[-2:]
    p = int(math.ceil(3 * coarse))
    ny, nx = fast_len(h + 2 * p), fast_len(w + 2 * p)
    pads = [(0, 0)] * (a.ndim - 2) + [(p, ny - h - p), (p, nx - w - p)]
    ap = np.pad(a, pads, mode="symmetric")
    fy = np.fft.fftfreq(ny)[:, None] ** 2
    fx = np.fft.rfftfreq(nx)[None, :] ** 2
    k = (np.exp(-2.0 * (math.pi * fine) ** 2 * (fy + fx))
         - np.exp(-2.0 * (math.pi * coarse) ** 2 * (fy + fx))).astype(np.float32)
    out = np.fft.irfft2(np.fft.rfft2(ap) * k, s=(ny, nx))
    return np.ascontiguousarray(out[..., p:p + h, p:p + w], dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


# -----------------------------------------------------------------------------
# mosaic model
# -----------------------------------------------------------------------------

@dataclass
class RegistrationOptions:
    source: str = "auto"          # 'volume': all planes (3D), 'max': max projection, 'plane'; auto: 3D for z-stacks
    plane: tuple = ()             # index of the leading axes for source 'plane'
    max_shift_um: float = 0.0     # search radius around the stage position (0: the overlap width)
    max_z_shift: int = 0          # 'volume': allowed z shift between tiles in planes (0: none)
    min_r: float = 0.3            # links with a lower normalised cross-correlation are not used
    consistency: bool = True      # check every link against the other links of the same grid step
    max_residual_px: float = 3.0  # links that disagree more with the global solution are dropped
    coarse_sigma: float = 8.0     # high-pass: background removed above this scale (px)


@dataclass
class Link:
    i: int
    j: int
    nominal: tuple                # (dx, dy, dz) of tile j relative to tile i at the stage positions
    measured: tuple | None        # registered (dx, dy, dz), None if no peak
    r: float                      # normalised cross-correlation of the overlap at the measured shift
    overlap_px: int = 0           # overlap area (px per plane) at the measured shift
    used: bool = False
    residual: float = float("nan")
    note: str = ""
    step: tuple | None = None     # grid step ('x', column) / ('y', row) of neighbouring tiles, else None


@dataclass
class Registration:
    positions: np.ndarray         # (n, 3) tile origins x, y (px) and z (planes), relative
    nominal: np.ndarray           # (n, 3) at the stage positions
    links: list
    options: RegistrationOptions
    orientation: tuple
    summary: str = ""


class Mosaic:
    """The tiles of one mosaic and how they are placed and fused.

    *layout* is 'stage' (tiles at the stage positions from the file) or 'aligned'
    (positions from :meth:`register`).  Changing layout, blending or orientation
    updates :attr:`stack`; arrays from ``stack.asarray()`` are snapshots.
    """

    def __init__(self, tiles: list[Tile], msr=None):
        self.tiles = sorted(tiles, key=lambda t: (t.iy, t.ix))
        first = self.tiles[0].stack
        self.first = first
        self.msr_path = getattr(msr, "path", first.path)
        self.tile_axes = first.axes
        self.tile_shape = tuple(first.shape)
        self.h, self.w = self.tile_shape[-2:]
        px, py = first.pixel_size
        self.px = px or 1.0
        self.py = py or self.px
        self.calibrated = bool(px and py)
        self.nx = max(t.ix for t in self.tiles) + 1
        self.ny = max(t.iy for t in self.tiles) + 1
        m = _CHANNEL_RE.match(first.channel_id or "")
        self.detector = f"{m.group('dev')}:{m.group('a')}:{m.group('det')}" if m else (first.channel_id or first.source)
        self.base_source = (first.source or "").split(" - ")[0] or "mosaic"
        self.orientation = DEFAULT_ORIENTATION
        self.layout = "stage"
        self.blend = "linear"
        self.subpixel = False
        self.registration: Registration | None = None
        self.version = 0
        self.zaxis = self.tile_axes[:-2].find("Z")
        self.zaxis = None if self.zaxis < 0 else self.zaxis
        self.stack = MosaicStack(index=-1, class_name="Mosaic", schema=0, offset=0, path=first.path,
                                 mosaic=self)
        self._update_stack()

    # -- description ------------------------------------------------------------

    @property
    def title(self) -> str:
        return f"Mosaic {self.nx} × {self.ny}  {self.detector}"

    def tile_label(self, k: int) -> str:
        t = self.tiles[k]
        return f"x{t.ix} y{t.iy}"

    # -- positions --------------------------------------------------------------

    def _grid_step_um(self) -> tuple[float, float]:
        """Stage step from the mosaic settings when tiles have no stage position."""
        p = self.first.properties
        steps = []
        for axis, width_px, size in (("X", self.w, self.px), ("Y", self.h, self.py)):
            width = p.get(f"xyz-Table {axis}Width")
            ovl = p.get(f"xyz-Table {axis}Ovl")
            if isinstance(width, (int, float)) and isinstance(ovl, (int, float)) and width > ovl >= 0:
                steps.append(float(width - ovl))
            else:
                pct = p.get("xyz-Table XYOvl", 10.0)
                pct = float(pct) if isinstance(pct, (int, float)) and 0 <= pct < 100 else 10.0
                steps.append(width_px * size * (1 - pct / 100.0))
        return steps[0], steps[1]

    def stage_um(self) -> np.ndarray:
        """(n, 2) stage positions in µm (from the grid indices where the file has none)."""
        if all(t.stage is not None for t in self.tiles) and len({t.stage for t in self.tiles}) == len(self.tiles):
            return np.array([t.stage for t in self.tiles], dtype=np.float64)
        sx, sy = self._grid_step_um()
        # same convention as ImSpector: X grows with the X index, Y falls with the Y index
        return np.array([(t.ix * sx, -t.iy * sy) for t in self.tiles], dtype=np.float64)

    def nominal_positions(self, orientation: tuple | None = None) -> np.ndarray:
        """(n, 3) tile origins in mosaic pixels (x, y) and planes (z) at the stage positions."""
        t, sx, sy = orientation or self.orientation
        st = self.stage_um()
        if t:
            st = st[:, ::-1]
        x = sx * st[:, 0] / self.px
        y = sy * st[:, 1] / self.py
        pos = np.zeros((len(self.tiles), 3))
        pos[:, 0] = x - x.min()
        pos[:, 1] = y - y.min()
        return pos

    def positions(self) -> np.ndarray:
        """Current tile origins (x, y px, z planes), shifted so that the smallest are 0."""
        if self.layout == "aligned" and self.registration is not None:
            pos = np.array(self.registration.positions, dtype=np.float64)
        else:
            pos = self.nominal_positions()
        pos = pos.copy()
        pos[:, :2] -= np.floor(pos[:, :2].min(axis=0))
        pos[:, 2] -= np.round(pos[:, 2]).min()
        return pos

    def overlap_px(self, pos: np.ndarray | None = None) -> tuple[float, float]:
        """Typical overlap (x, y) of neighbouring tiles in pixels, 0 if they do not overlap."""
        pos = self.positions() if pos is None else pos
        ox, oy = [], []
        for a in range(len(self.tiles)):
            for b in range(a + 1, len(self.tiles)):
                dx, dy = abs(pos[b, 0] - pos[a, 0]), abs(pos[b, 1] - pos[a, 1])
                if dx < self.w and dy < self.h:
                    if dx >= dy:
                        ox.append(self.w - dx)
                    else:
                        oy.append(self.h - dy)
        return (float(np.max(ox)) if ox else 0.0, float(np.max(oy)) if oy else 0.0)

    # -- state changes ----------------------------------------------------------

    def set_layout(self, layout: str) -> None:
        if layout == "aligned" and self.registration is None:
            raise ValueError("register the tiles first")
        self.layout = layout
        self._changed()

    def set_blend(self, blend: str) -> None:
        if blend not in BLEND_MODES:
            raise ValueError(blend)
        self.blend = blend
        self._changed()

    def set_subpixel(self, on: bool) -> None:
        self.subpixel = bool(on)
        self._changed()

    def set_orientation(self, orientation: tuple) -> None:
        orientation = tuple(orientation)
        if orientation != self.orientation:
            self.orientation = orientation
            self.registration = None  # registered for another layout
            self.layout = "stage"
            self._changed()

    def set_registration(self, reg: Registration) -> None:
        self.registration = reg
        self.orientation = reg.orientation
        self.layout = "aligned"
        self._changed()

    def same_grid(self, other: "Mosaic") -> bool:
        """Same tiles (grid indices, stage positions, size): one registration fits both."""
        if len(other.tiles) != len(self.tiles) or other.tile_shape[-2:] != self.tile_shape[-2:]:
            return False
        for a, b in zip(self.tiles, other.tiles):
            if (a.ix, a.iy) != (b.ix, b.iy):
                return False
            if (a.stage is None) != (b.stage is None) or (
                    a.stage is not None and max(abs(a.stage[0] - b.stage[0]), abs(a.stage[1] - b.stage[1])) > 0.01):
                return False
        return abs(self.px - other.px) < 1e-6 * self.px

    def copy_layout(self, other: "Mosaic") -> None:
        """Take over layout, registration, blending and orientation of a mosaic with the same grid."""
        if not self.same_grid(other):
            raise ValueError("the mosaics have different tiles")
        self.orientation = other.orientation
        self.registration = other.registration
        self.layout = other.layout if other.registration is not None else "stage"
        self.blend = other.blend
        self.subpixel = other.subpixel
        self._changed()

    def _changed(self) -> None:
        self.version += 1
        self._update_stack()

    # -- geometry of the fused stack ------------------------------------------------

    def origins(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integer tile origins (n, 3: x, y, z; smallest 0) and the sub-pixel rest (n, 2) for *pos*."""
        if self.subpixel:
            o_xy = np.floor(pos[:, :2])
            frac = pos[:, :2] - o_xy
        else:
            o_xy = np.round(pos[:, :2])
            frac = np.zeros((len(pos), 2))
        o = np.zeros((len(pos), 3), dtype=int)
        o[:, :2] = (o_xy - o_xy.min(axis=0)).astype(int)
        oz = np.round(pos[:, 2]).astype(int)
        o[:, 2] = oz - oz.min()
        return o, frac, oz

    def _canvas(self, pos: np.ndarray) -> tuple[int, int, int]:
        """(width, height, depth) of the fused image for *pos*."""
        o, _, _ = self.origins(pos)
        depth = self.tile_shape[self.zaxis] + int(o[:, 2].max()) if self.zaxis is not None else 0
        return int(o[:, 0].max()) + self.w, int(o[:, 1].max()) + self.h, depth

    def _update_stack(self) -> None:
        s, first = self.stack, self.first
        pos = self.positions()
        width, height, depth = self._canvas(pos)
        sizes = list(first.sizes)
        lengths = [float(x) for x in first.lengths]
        sizes[0], sizes[1] = width, height
        lengths[0] = width * self.px if self.calibrated else 0.0
        lengths[1] = height * self.py if self.calibrated else 0.0
        if self.zaxis is not None:
            dim = [i for i, c in enumerate(first.dim_letters) if c == "Z"][0]
            step = first.step("Z")
            sizes[dim] = depth
            lengths[dim] = step * depth if step else lengths[dim]
        s.sizes, s.lengths = sizes, lengths
        s.labels = ["Mosaic X", "Mosaic Y"] + list(first.labels[2:])
        s.units = list(first.units)
        s.dtype = first.dtype
        s.name, s.time = first.name, first.time
        s.source = f"{self.base_source} mosaic"
        s.channel_id = f"{self.detector} mosaic {self.nx}x{self.ny}"
        s.title = f"{first.name} - {first.time} - {self.title}"
        s.timestamps = list(first.timestamps)
        s.properties = first.properties
        ox, oy = self.overlap_px(pos)
        layout = "registered (stitched)" if self.layout == "aligned" else "stage positions"
        s.meta = dict(first.meta)
        s.meta.update({
            "Mosaic": f"{self.nx} x {self.ny} tiles of {self.w} x {self.h} px",
            "Mosaic layout": layout,
            "Mosaic blending": self.blend,
            "Mosaic overlap (px)": f"{ox:.1f} x {oy:.1f}",
            "Mosaic orientation": orientation_label(self.orientation).replace("  (default)", ""),
            "Mosaic tile origins (x, y px; z planes)": "; ".join(
                f"{self.tile_label(k)}: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}" for k, p in enumerate(pos)),
        })
        if self.registration is not None and self.layout == "aligned":
            s.meta["Mosaic registration"] = self.registration.summary
        s.header = {"mosaic": {"tiles": [t.pos for t in self.tiles], "layout": self.layout,
                               "blend": self.blend, "positions": pos.round(3).tolist(),
                               "orientation": list(self.orientation)}}
        s.parse_mode = "mosaic (fused on the fly)"

    # -- fusion -------------------------------------------------------------------

    def plan(self) -> "MosaicArray":
        """A snapshot of the current layout that composes planes on demand."""
        return MosaicArray(self)

    def frozen_stack(self) -> "MosaicStack":
        """A copy of :attr:`stack` fixed to the current layout (for exports in another thread).

        Its geometry, metadata and ``asarray()`` stay consistent even if the layout or the
        blending is changed while the export runs.
        """
        s = copy.copy(self.stack)
        for name in ("sizes", "lengths", "labels", "units", "timestamps"):
            setattr(s, name, list(getattr(self.stack, name)))
        s.meta = dict(self.stack.meta)
        s.header = copy.deepcopy(self.stack.header)
        s.frozen = self.plan()
        return s

    # -- registration --------------------------------------------------------------

    def register(self, options: RegistrationOptions | None = None, orientation: tuple | None = None,
                 progress: Callable[[int, int, str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> Registration:
        """Register all overlapping tile pairs and solve for the tile positions (does not apply them)."""
        return register(self, options or RegistrationOptions(), orientation or self.orientation, progress,
                        cancelled)


@dataclass
class MosaicStack(mr.DataStack):
    """A mosaic as one stack; ``asarray()`` composes the planes from the tiles on demand."""

    virtual: ClassVar[bool] = True   # not stored in the file (msr_reader.export skips it)
    mosaic: Any = field(default=None, repr=False, compare=False)
    frozen: Any = field(default=None, repr=False, compare=False)  # MosaicArray of a frozen copy

    def asarray(self) -> "MosaicArray":
        return self.frozen if self.frozen is not None else self.mosaic.plan()


# -----------------------------------------------------------------------------
# fusion
# -----------------------------------------------------------------------------

def _ramp(n: int, width: float) -> np.ndarray:
    """1 in the middle, falling linearly to ~0 over *width* pixels at both ends."""
    d = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5)
    return np.clip(d / max(width, 1.0), 1e-3, 1.0).astype(np.float32)


def _index_list(k, n: int, ax: int) -> tuple[list[int], bool]:
    """Indices selected by *k* along an axis of length n, and whether the axis is dropped."""
    if isinstance(k, (int, np.integer)) or (isinstance(k, np.ndarray) and k.ndim == 0
                                             and np.issubdtype(k.dtype, np.integer)):
        i = int(k)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"index {int(k)} is out of bounds for axis {ax} with size {n}")
        return [i], True
    if isinstance(k, slice):
        return list(range(*k.indices(n))), False
    a = np.asarray(k)
    if a.size == 0 and a.ndim == 1:  # [] selects nothing, as in numpy
        return [], False
    if a.dtype == bool:
        if a.shape != (n,):
            raise IndexError(f"boolean index of shape {a.shape} does not match axis {ax} with size {n}")
        return [int(i) for i in np.flatnonzero(a)], False
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise IndexError("only integers, slices, Ellipsis and 1-d integer or boolean arrays are valid indices")
    out = []
    for v in a.tolist():
        i = v + n if v < 0 else v
        if not 0 <= i < n:
            raise IndexError(f"index {v} is out of bounds for axis {ax} with size {n}")
        out.append(i)
    return out, False


class _LazyPlanes:
    """numpy-style read-only indexing for arrays whose planes (last two axes) are made on demand.

    Subclasses provide ``shape``, ``dtype`` and ``_plane(lead, y0, y1, x0, x1)``.
    Supported indices: integers, slices (with steps), Ellipsis and 1-d integer or
    boolean arrays, one per axis (orthogonal indexing).
    """

    shape: tuple
    dtype: np.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self[...]
        return out.astype(dtype) if dtype is not None else out

    def reshape(self, *shape) -> "_LazyPlanes":
        """Another arrangement of the leading axes (the last two stay), e.g. (-1, h, w)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(n) for n in shape)
        if len(shape) < 2 or shape[-2:] != tuple(self.shape[-2:]):
            raise ValueError(f"a lazy mosaic can only be reshaped along its leading axes, not to {shape}")
        lead = list(shape[:-2])
        total = int(np.prod(self.shape[:-2])) if self.ndim > 2 else 1
        if lead.count(-1) > 1:
            raise ValueError("can only specify one unknown dimension")
        if -1 in lead:
            known = int(np.prod([n for n in lead if n != -1])) if len(lead) > 1 else 1
            lead[lead.index(-1)] = total // known if known else 0
        if int(np.prod(lead)) != total:
            raise ValueError(f"cannot reshape array of shape {self.shape} into shape {shape}")
        if tuple(lead) == tuple(self.shape[:-2]):
            return self
        return _Reshaped(self, tuple(lead))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if sum(k is Ellipsis for k in key) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if any(k is Ellipsis for k in key):
            i = next(n for n, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        if len(key) > self.ndim:
            raise IndexError(f"too many indices for array: array is {self.ndim}-dimensional, "
                             f"but {len(key)} were indexed")
        key = key + (slice(None),) * (self.ndim - len(key))
        sel, drop = [], []
        for ax, k in enumerate(key):
            idx, d = _index_list(k, self.shape[ax], ax)
            sel.append(idx)
            drop.append(d)
        ys, xs = sel[-2], sel[-1]
        out = np.zeros([len(s) for s in sel], self.dtype)
        if out.size:
            y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            ry = None if ys == list(range(y0, y1)) else np.asarray(ys) - y0
            rx = None if xs == list(range(x0, x1)) else np.asarray(xs) - x0
            for pos, lead in zip(np.ndindex(*out.shape[:-2]), itertools.product(*sel[:-2])):
                plane = self._plane(tuple(lead), y0, y1, x0, x1)
                if ry is not None:
                    plane = plane[ry]
                if rx is not None:
                    plane = plane[:, rx]
                out[pos] = plane
        if any(drop):
            out = out.reshape([n for n, d in zip(out.shape, drop) if not d])
            if out.ndim == 0:
                return out[()]
        return out

    def _plane(self, lead: tuple, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        raise NotImplementedError


class _Reshaped(_LazyPlanes):
    """Leading axes of a lazy array arranged differently (same planes, same order)."""

    def __init__(self, base: _LazyPlanes, lead: tuple):
        self.base = base
        self.shape = tuple(lead) + tuple(base.shape[-2:])
        self.dtype = base.dtype

    def _plane(self, lead, y0, y1, x0, x1):
        flat = int(np.ravel_multi_index(lead, self.shape[:-2])) if lead else 0
        base_lead = tuple(int(i) for i in np.unravel_index(flat, self.base.shape[:-2])) if self.base.ndim > 2 else ()
        return self.base._plane(base_lead, y0, y1, x0, x1)

    def __getattr__(self, name):  # coverage(), tile_rects(), mosaic, ... of the base
        return getattr(self.__dict__["base"], name)


class MosaicArray(_LazyPlanes):
    """Read-only array-like view of a fused mosaic (numpy indexing, planes composed on demand).

    Indexing returns numpy arrays.  Whole planes are cached (least recently used, up to
    ~256 MB).  Each instance is a snapshot of the layout and is meant to be used from
    one thread.
    """

    CACHE_BYTES = 256 << 20

    def __init__(self, mosaic: Mosaic):
        self.mosaic = mosaic
        self.version = mosaic.version
        pos = mosaic.positions()
        width, height, depth = mosaic._canvas(pos)
        o, frac, _ = mosaic.origins(pos)
        self.arrays = [t.stack.asarray() for t in mosaic.tiles]
        self.h, self.w = mosaic.h, mosaic.w
        self.ox, self.oy, self.oz = o[:, 0], o[:, 1], o[:, 2]
        self.fx, self.fy = frac[:, 0], frac[:, 1]
        self.zaxis = mosaic.zaxis
        lead = list(mosaic.tile_shape[:-2])
        self.nz_tile = 1
        if self.zaxis is not None:
            lead[self.zaxis] = depth
            self.nz_tile = mosaic.tile_shape[self.zaxis]
        self.shape = tuple(lead) + (height, width)
        self.dtype = np.dtype(mosaic.first.dtype)
        # float32 holds uint8 / uint16 / float32 exactly; wider types are fused in float64
        self.work = np.float32 if self.dtype.itemsize <= 2 or self.dtype == np.float32 else np.float64
        self.blend = mosaic.blend
        ovx, ovy = mosaic.overlap_px(pos)
        self.weight = (np.outer(_ramp(self.h, ovy), _ramp(self.w, ovx)) if self.blend == "linear"
                       else np.ones((self.h, self.w), np.float32))
        self._cache: OrderedDict = OrderedDict()

    # -- composition --------------------------------------------------------------

    def _plane(self, lead: tuple, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        H, W = self.shape[-2:]
        full = (y0, y1, x0, x1) == (0, H, 0, W)
        if lead in self._cache:
            self._cache.move_to_end(lead)
            plane = self._cache[lead]
            return plane if full else plane[y0:y1, x0:x1]
        plane = self._compose(lead, y0, y1, x0, x1)
        if full:
            self._cache[lead] = plane
            while len(self._cache) > 1 and len(self._cache) * plane.nbytes > self.CACHE_BYTES:
                self._cache.popitem(last=False)
        return plane

    def _tile_rows(self, k: int, tidx: tuple, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Rows r0..r1, columns c0..c1 of tile k at sub-pixel offset (fy, fx), edges replicated."""
        fy, fx = self.fy[k], self.fx[k]
        a = self.arrays[k]
        if not fy and not fx:
            return np.asarray(a[tidx + (slice(r0, r1), slice(c0, c1))], dtype=self.work)
        # canvas pixel (r, c) of the tile shows tile position (r - fy, c - fx)
        rr0, cc0 = max(r0 - 1, 0), max(c0 - 1, 0)
        block = np.asarray(a[tidx + (slice(rr0, r1), slice(cc0, c1))], dtype=self.work)
        if r0 == 0:
            block = np.concatenate([block[:1], block], axis=0)
        if c0 == 0:
            block = np.concatenate([block[:, :1], block], axis=1)
        cur = block[1:, 1:]
        up = block[:-1, 1:]
        left = block[1:, :-1]
        diag = block[:-1, :-1]
        return ((1 - fy) * (1 - fx) * cur + fy * (1 - fx) * up + (1 - fy) * fx * left + fy * fx * diag)

    def _tile_plane(self, k: int, lead: tuple) -> tuple | None:
        """Index of tile k's plane that lies at mosaic position *lead*, None if it has none there."""
        tidx = list(lead)
        if self.zaxis is not None:
            zt = lead[self.zaxis] - self.oz[k]
            if not 0 <= zt < self.nz_tile:
                return None
            tidx[self.zaxis] = zt
        return tuple(tidx)

    def _compose(self, lead: tuple, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        H, W = y1 - y0, x1 - x0
        mode = self.blend
        out = np.zeros((H, W), self.work)
        if mode in ("linear", "average"):
            wsum = np.zeros((H, W), self.work)
        else:
            seen = np.zeros((H, W), bool)
        for k in range(len(self.arrays)):
            tidx = self._tile_plane(k, lead)
            if tidx is None:
                continue
            ty0, ty1 = max(y0, self.oy[k]), min(y1, self.oy[k] + self.h)
            tx0, tx1 = max(x0, self.ox[k]), min(x1, self.ox[k] + self.w)
            if ty0 >= ty1 or tx0 >= tx1:
                continue
            r0, r1 = ty0 - self.oy[k], ty1 - self.oy[k]
            c0, c1 = tx0 - self.ox[k], tx1 - self.ox[k]
            data = self._tile_rows(k, tidx, r0, r1, c0, c1)
            dst = (slice(ty0 - y0, ty1 - y0), slice(tx0 - x0, tx1 - x0))
            if mode in ("linear", "average"):
                wgt = self.weight[r0:r1, c0:c1]
                out[dst] += data * wgt
                wsum[dst] += wgt
            elif mode == "max":
                sub = out[dst]
                np.copyto(sub, np.where(seen[dst], np.maximum(sub, data), data))
                seen[dst] = True
            else:  # overwrite: later tiles on top
                out[dst] = data
                seen[dst] = True
        if mode in ("linear", "average"):
            np.divide(out, wsum, out=out, where=wsum > 0)
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            out = np.rint(out, out=out)
            return np.clip(out, info.min, info.max).astype(self.dtype)
        return out.astype(self.dtype)

    # -- tiles -----------------------------------------------------------------------

    def coverage(self, y: int, x: int, lead: tuple | None = None) -> list[int]:
        """Tiles that contain mosaic pixel (y, x) (in the plane *lead*, if given)."""
        return [k for k in range(len(self.arrays))
                if self.oy[k] <= y < self.oy[k] + self.h and self.ox[k] <= x < self.ox[k] + self.w
                and (lead is None or self.zaxis is None or len(lead) <= self.zaxis
                     or self._tile_plane(k, tuple(lead)) is not None)]

    def tile_rects(self) -> list[tuple[int, int, int, int]]:
        """(x, y, w, h) of every tile in mosaic pixels."""
        return [(int(self.ox[k]), int(self.oy[k]), self.w, self.h) for k in range(len(self.arrays))]


# -----------------------------------------------------------------------------
# registration
# -----------------------------------------------------------------------------

def _reg_image(arr, mosaic: Mosaic, opts: RegistrationOptions, ys: slice, xs: slice, volume: bool) -> np.ndarray:
    """Region of a tile used for registration: (planes, h, w) float, high-pass filtered."""
    lead = mosaic.tile_shape[:-2]
    if opts.source == "plane" and opts.plane and len(opts.plane) == len(lead):
        img = np.asarray(arr[tuple(opts.plane) + (ys, xs)], dtype=np.float32)[None]
    elif not lead:
        img = np.asarray(arr[ys, xs], dtype=np.float32)[None]
    else:
        block = np.asarray(arr[(slice(None),) * len(lead) + (ys, xs)], dtype=np.float32)
        if volume and mosaic.zaxis is not None:
            other = tuple(i for i in range(len(lead)) if i != mosaic.zaxis)
            img = block.max(axis=other) if other else block
        else:
            img = block.reshape((-1,) + block.shape[-2:]).max(axis=0)[None]
    return highpass(img, 1.0, opts.coarse_sigma)


def _local_maxima(c: np.ndarray, k: int) -> list[tuple]:
    """Indices of the k largest local maxima of an n-d array."""
    pad = np.pad(c, 1, mode="constant", constant_values=-np.inf)
    is_max = np.ones(c.shape, bool)
    for off in itertools.product((-1, 0, 1), repeat=c.ndim):
        if any(off):
            sl = tuple(slice(1 + o, pad.shape[d] - 1 + o) for d, o in enumerate(off))
            is_max &= c >= pad[sl]
    idx = np.argwhere(is_max)
    if idx.size == 0:
        return []
    vals = c[tuple(idx.T)]
    order = np.argsort(vals)[::-1][:k]
    return [tuple(int(v) for v in idx[o]) for o in order]


def _overlap_slices(sa: tuple, sb: tuple, s: tuple):
    """Slices of A and B that overlap when B is displaced by s (B[u + s] ~ A[u])."""
    a_sl, b_sl = [], []
    for na, nb, d in zip(sa, sb, s):
        u0, u1 = max(0, -d), min(na, nb - d)
        if u1 <= u0:
            return None
        a_sl.append(slice(u0, u1))
        b_sl.append(slice(u0 + d, u1 + d))
    return tuple(a_sl), tuple(b_sl)


def _overlap_counts(na: int, nb: int, shifts: np.ndarray) -> np.ndarray:
    """Number of overlapping samples along one axis for each displacement s (B[u + s] ~ A[u])."""
    return np.clip(np.minimum(na, nb - shifts) - np.maximum(0, -shifts), 0, None).astype(np.float64)


def _pair_shift(A: np.ndarray, B: np.ndarray, s_lo: tuple, s_hi: tuple, min_area: int,
                n_peaks: int = 8) -> tuple[tuple | None, float, int]:
    """Best displacement s (B[u + s] ~ A[u]) within [s_lo, s_hi], its NCC and overlap area.

    Candidates are the largest peaks of the FFT cross-correlation divided by the overlap
    size (a raw correlation sum favours larger overlaps); each is scored by the normalised
    cross-correlation (NCC) of the parts that actually overlap.  From the best one the NCC
    is climbed to its local maximum on the integer grid, which a parabola through the
    neighbours then refines to sub-pixel.
    """
    ranges = [np.arange(lo, hi + 1) for lo, hi in zip(s_lo, s_hi)]
    if any(r.size == 0 for r in ranges):
        return None, 0.0, 0
    A0, B0 = A - A.mean(), B - B.mean()
    # circular correlation equals the linear one for all searched shifts at these sizes
    shape = tuple(fast_len(max(na + max(hi, 0), nb - min(lo, 0), hi - lo + 1))
                  for na, nb, lo, hi in zip(A.shape, B.shape, s_lo, s_hi))
    if s_lo[0] == s_hi[0] == 0 and A.shape[0] == B.shape[0]:
        # no shift between planes: sum of the 2D correlations of the planes
        shape = (1,) + shape[1:]
        prod = (np.conj(np.fft.rfft2(A0, shape[1:])) * np.fft.rfft2(B0, shape[1:])).sum(axis=0)
        corr = np.fft.irfft2(prod, shape[1:])[None]
    else:
        axes = tuple(range(A.ndim))
        corr = np.fft.irfftn(np.conj(np.fft.rfftn(A0, shape, axes=axes)) * np.fft.rfftn(B0, shape, axes=axes),
                             shape, axes=axes)
    # corr[s] = sum_u A[u] B[u + s], per overlapping sample
    box = corr[np.ix_(*[r % n for r, n in zip(ranges, shape)])]
    # shifts with too little overlap are not candidates; the sum is divided by the square root
    # of the overlap size (the noise of a sum of n terms grows with sqrt(n)), which takes away
    # most of its preference for large overlaps without letting small ones win on noise
    count = np.ones(box.shape)
    plane = np.ones(box.shape)
    for d, (na, nb, r) in enumerate(zip(A.shape, B.shape, ranges)):
        view = [1] * box.ndim
        view[d] = r.size
        c = _overlap_counts(na, nb, r).reshape(view)
        count = count * c
        if d >= box.ndim - 2:
            plane = plane * c
    box = np.where((plane >= min_area) & (count > 0), box / np.sqrt(np.maximum(count, 1.0)), -np.inf)

    scores: dict[tuple, tuple[float, int]] = {}

    def score(s: tuple) -> tuple[float, int]:
        if s in scores:
            return scores[s]
        out = (-1.0, 0)
        if all(lo <= v <= hi for v, lo, hi in zip(s, s_lo, s_hi)):
            sl = _overlap_slices(A.shape, B.shape, s)
            if sl is not None:
                a, b = A[sl[0]], B[sl[1]]
                area = a.shape[-1] * a.shape[-2]
                out = (_ncc(a, b), area) if area >= min_area else (-1.0, area)
        scores[s] = out
        return out

    cands = []
    for p in _local_maxima(box, n_peaks):
        s = tuple(int(r[i]) for r, i in zip(ranges, p))
        cands.append((score(s)[0], s))
    cands = sorted([c for c in cands if c[0] > -1.0], reverse=True)
    if not cands:
        return None, 0.0, 0
    searched = [d for d in range(len(s_lo)) if s_lo[d] < s_hi[d]]
    steps = [off for off in itertools.product((-1, 0, 1), repeat=len(searched)) if any(off)]

    def climb(s: tuple, r: float) -> tuple[tuple, float]:
        for _ in range(64):  # to the local maximum of the NCC on the integer grid
            nxt, nxt_r = s, r
            for off in steps:
                t = list(s)
                for d, o in zip(searched, off):
                    t[d] += o
                rt = score(tuple(t))[0]
                if rt > nxt_r:
                    nxt, nxt_r = tuple(t), rt
            if nxt == s:
                break
            s, r = nxt, nxt_r
        return s, r

    # a peak of the correlation sum can sit a few pixels beside its NCC maximum, so every
    # promising candidate is climbed before they are compared
    best, best_r = None, -1.0
    for r0, s0 in cands[:4]:
        if best is not None and r0 < 0.5 * cands[0][0]:
            break
        s1, r1 = climb(s0, r0)
        if r1 > best_r:
            best, best_r = s1, r1
    sub = []
    for d in range(len(best)):
        if d not in searched:  # not searched along this axis (2D, or no z shift allowed)
            sub.append(float(best[d]))
            continue
        vals = []
        for off in (-1, 1):
            s = list(best)
            s[d] += off
            vals.append(score(tuple(s))[0])
        lo, hi = vals
        den = lo - 2 * best_r + hi
        frac = 0.5 * (lo - hi) / den if den < 0 and lo > -1 and hi > -1 else 0.0
        sub.append(best[d] + float(np.clip(frac, -0.5, 0.5)))
    return tuple(sub), best_r, score(best)[1]


def pair_candidates(mosaic: Mosaic, pos: np.ndarray, min_frac: float = 0.03) -> list[tuple[int, int]]:
    """Tile pairs whose rectangles overlap by at least *min_frac* of a tile (at *pos*)."""
    out = []
    n = len(mosaic.tiles)
    for i in range(n):
        for j in range(i + 1, n):
            ox = mosaic.w - abs(pos[j, 0] - pos[i, 0])
            oy = mosaic.h - abs(pos[j, 1] - pos[i, 1])
            if ox >= 4 and oy >= 4 and ox * oy >= min_frac * mosaic.w * mosaic.h:
                out.append((i, j))
    return out


def measure_pair(mosaic: Mosaic, arrays: list, i: int, j: int, d: np.ndarray, opts: RegistrationOptions,
                 search_px: tuple[int, int]) -> Link:
    """Register tile j against tile i around the nominal displacement d = pos_j - pos_i (x, y)."""
    h, w = mosaic.h, mosaic.w
    dn = (int(round(d[0])), int(round(d[1])))
    mx, my = search_px
    volume = opts.source == "volume" and mosaic.zaxis is not None
    # nominal overlap in tile i coordinates, widened by the search range
    ax0, ax1 = max(0, dn[0]), min(w, w + dn[0])
    ay0, ay1 = max(0, dn[1]), min(h, h + dn[1])
    nominal = (float(d[0]), float(d[1]), 0.0)
    if ax1 - ax0 < 4 or ay1 - ay0 < 4:
        return Link(i, j, nominal, None, 0.0, note="no overlap")
    a_x = (max(0, ax0 - mx), min(w, ax1 + mx))
    a_y = (max(0, ay0 - my), min(h, ay1 + my))
    b_x = (max(0, ax0 - dn[0] - mx), min(w, ax1 - dn[0] + mx))
    b_y = (max(0, ay0 - dn[1] - my), min(h, ay1 - dn[1] + my))
    A = _reg_image(arrays[i], mosaic, opts, slice(*a_y), slice(*a_x), volume)
    B = _reg_image(arrays[j], mosaic, opts, slice(*b_y), slice(*b_x), volume)
    # B[u + s] ~ A[u]  with  s = a0 - b0 - d  (d: true displacement of tile j)
    cx, cy = a_x[0] - b_x[0] - dn[0], a_y[0] - b_y[0] - dn[1]
    mz = max(0, int(opts.max_z_shift)) if volume else 0
    s_lo = (-mz, cy - my, cx - mx)
    s_hi = (mz, cy + my, cx + mx)
    s_lo = tuple(max(lo, -(nb - 1)) for lo, nb in zip(s_lo, B.shape))
    s_hi = tuple(min(hi, na - 1) for hi, na in zip(s_hi, A.shape))
    nominal_area = (ax1 - ax0) * (ay1 - ay0)
    step = _grid_step(mosaic, i, j)
    s, r, area = _pair_shift(A, B, s_lo, s_hi, max(64, int(0.25 * nominal_area)))
    if s is None:
        return Link(i, j, nominal, None, 0.0, note="no correlation peak", step=step)
    if A.shape[0] > 1:
        # score volumes like projections, so that one threshold fits both sources
        sl = _overlap_slices(A.shape, B.shape, tuple(int(round(v)) for v in s))
        if sl is not None:
            r = _ncc(A[sl[0]].max(axis=0), B[sl[1]].max(axis=0))
    measured = (a_x[0] - b_x[0] - s[2], a_y[0] - b_y[0] - s[1], -s[0])
    note = ""
    if max(abs(measured[0] - d[0]), abs(measured[1] - d[1])) >= min(mx, my) - 0.5 and min(mx, my) > 0:
        note = "at the edge of the search range"
    return Link(i, j, nominal, measured, r, overlap_px=area, note=note, step=step)


def _grid_step(mosaic: Mosaic, i: int, j: int) -> tuple | None:
    """('x', column) or ('y', row) for neighbouring tiles of the grid, None otherwise."""
    a, b = mosaic.tiles[i], mosaic.tiles[j]
    dx, dy = b.ix - a.ix, b.iy - a.iy
    if abs(dx) + abs(dy) != 1:
        return None
    return ("x", min(a.ix, b.ix)) if dy == 0 else ("y", min(a.iy, b.iy))


def _error(lk: Link) -> np.ndarray:
    """Measured minus stage displacement of a link (x, y px)."""
    return np.asarray(lk.measured[:2], dtype=np.float64) - np.asarray(lk.nominal[:2], dtype=np.float64)


def _lsq_positions(n: int, rows: list, nominal: np.ndarray, anchor_weight: float = 1e-6) -> np.ndarray:
    """Least-squares positions (n, dims) from rows (i, j, displacement, weight), weakly anchored at *nominal*."""
    dims = nominal.shape[1]
    M = np.zeros((len(rows) + n, n))
    b = np.zeros((len(rows) + n, dims))
    for r, (i, j, d, w) in enumerate(rows):
        M[r, i], M[r, j] = -w, w
        b[r] = w * np.asarray(d[:dims], dtype=np.float64)
    M[len(rows):, :] = anchor_weight * np.eye(n)
    b[len(rows):] = anchor_weight * nominal
    return np.linalg.lstsq(M, b, rcond=None)[0]


def _connected(n: int, edges: list, a: int, b: int) -> bool:
    adj: dict = {}
    for i, j in edges:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    seen, todo = {a}, [a]
    while todo:
        k = todo.pop()
        if k == b:
            return True
        for m in adj.get(k, ()):
            if m not in seen:
                seen.add(m)
                todo.append(m)
    return False


def check_consistency(links: list[Link], tol: float, nominal: np.ndarray | None = None,
                      loop_tol: float = 6.0) -> None:
    """Drop links far off the other links of their grid step, unless the other overlaps confirm them.

    A stage tends to repeat its positioning error for the same move (column c -> c+1 in
    every row, row r -> r+1 in every column), so the links of one step agree; a false
    correlation peak in an overlap without structure does not.  Steps with at least three
    links are judged by the median of their own links, with a tolerance of *tol* px or four
    robust standard deviations of that step, whichever is larger.  A link outside it is
    kept anyway when the other used links connect its two tiles and predict the same
    displacement within *loop_tol* px (a closed loop: no assumption about the stage
    needed); it is dropped only when nothing confirms it.  Smaller disagreements are left
    to the residual test of :func:`solve_positions`.
    """
    groups: dict = {}
    for lk in links:
        if lk.used and lk.step is not None:
            groups.setdefault(lk.step, []).append(lk)
    suspects = []
    for lks in groups.values():
        if len(lks) < 3:
            continue
        errs = np.array([_error(lk) for lk in lks])
        dev = np.linalg.norm(errs - np.median(errs, axis=0), axis=1)
        limit = max(tol, 4.0 * 1.4826 * float(np.median(dev)))
        suspects += [(lk, float(dv)) for lk, dv in zip(lks, dev) if dv > limit]
    if not suspects:
        return
    used = [lk for lk in links if lk.used]
    n = 1 + max(max(lk.i, lk.j) for lk in links)
    nom = (np.zeros((n, 2)) if nominal is None else np.asarray(nominal, dtype=np.float64)[:, :2])
    decisions = []
    for lk, dv in suspects:  # judged on the same set of links, so the order does not matter
        others = [o for o in used if o is not lk]
        confirmed = None
        if _connected(n, [(o.i, o.j) for o in others], lk.i, lk.j):
            pos = _lsq_positions(n, [(o.i, o.j, o.measured, max(o.r, 0.05)) for o in others], nom)
            res = float(np.linalg.norm(pos[lk.j] - pos[lk.i] - np.asarray(lk.measured[:2])))
            confirmed = res if res <= loop_tol else None
        decisions.append((lk, dv, confirmed))
    for lk, dv, confirmed in decisions:
        if confirmed is not None:
            lk.note = (f"{dv:.1f} px off the other links of its grid step, but confirmed by the other "
                       f"overlaps ({confirmed:.1f} px)")
        else:
            lk.used = False
            lk.note = f"dropped: {dv:.1f} px off the other links of this grid step"


def solve_positions(n: int, links: list[Link], nominal: np.ndarray, dims: int, max_residual: float,
                    prior_weight: float = 1e-3, anchor_weight: float = 1e-6) -> tuple[np.ndarray, list[Link]]:
    """Tile positions that best agree with the used links (weighted least squares).

    Pairs without a usable link are held at their stage displacement, corrected by the
    median error of the used links of the same grid step (or direction): a weak prior,
    so that a tile without a good overlap follows the stage's systematic error.  Every
    tile is also anchored, much more weakly, at its stage position, which fixes the
    mosaic as a whole and keeps tiles without any overlap where the stage put them.
    The link that disagrees most is dropped until all residuals are within
    *max_residual* px.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    while True:
        used = [lk for lk in links if lk.used]
        corr: dict[tuple, list] = {}
        for lk in used:
            err = np.asarray(lk.measured[:dims]) - (nominal[lk.j] - nominal[lk.i])[:dims]
            corr.setdefault(_direction(nominal[lk.j] - nominal[lk.i]), []).append(err)
            if lk.step is not None:
                corr.setdefault(lk.step, []).append(err)
        rows, rhs, wts = [], [], []
        for lk in links:
            if lk.used:
                rows.append((lk.i, lk.j))
                rhs.append(np.asarray(lk.measured[:dims], dtype=np.float64))
                wts.append(max(lk.r, 0.05))
            else:
                key = lk.step if lk.step in corr else _direction(nominal[lk.j] - nominal[lk.i])
                c = np.median(corr[key], axis=0) if key in corr else np.zeros(dims)
                rows.append((lk.i, lk.j))
                rhs.append((nominal[lk.j] - nominal[lk.i])[:dims] + c)
                wts.append(prior_weight)
        M = np.zeros((len(rows) + n, n))
        for r, (i, j) in enumerate(rows):
            M[r, i], M[r, j] = -wts[r], wts[r]
        M[len(rows):, :] = anchor_weight * np.eye(n)
        pos = np.zeros((n, dims))
        for d in range(dims):
            b = np.concatenate([[w * v[d] for w, v in zip(wts, rhs)], anchor_weight * nominal[:, d]])
            pos[:, d] = np.linalg.lstsq(M, b, rcond=None)[0]
        worst, worst_res = None, 0.0
        for lk in links:
            if lk.measured is None:
                continue
            res = float(np.linalg.norm(pos[lk.j] - pos[lk.i] - np.asarray(lk.measured[:dims])))
            lk.residual = res
            if lk.used and res > worst_res:
                worst, worst_res = lk, res
        if worst is None or worst_res <= max_residual:
            out = np.zeros((n, 3))
            out[:, :dims] = pos
            return out, links
        worst.used = False
        worst.note = f"dropped: disagrees by {worst_res:.1f} px with the other links"


def _direction(d: np.ndarray) -> tuple:
    """Grid direction class of a displacement (sign of the dominant axis, coarse)."""
    dx, dy = float(d[0]), float(d[1])
    if abs(dx) >= abs(dy):
        return ("x", 1 if dx > 0 else -1, int(round(dy / max(abs(dx), 1.0))))
    return ("y", 1 if dy > 0 else -1, int(round(dx / max(abs(dy), 1.0))))


def search_radius_px(mosaic: Mosaic, opts: RegistrationOptions, nominal: np.ndarray) -> tuple[int, int]:
    if opts.max_shift_um and opts.max_shift_um > 0:
        return (max(1, int(math.ceil(opts.max_shift_um / mosaic.px))),
                max(1, int(math.ceil(opts.max_shift_um / mosaic.py))))
    ox, oy = mosaic.overlap_px(nominal)
    r = max(ox, oy, 16.0)
    return int(math.ceil(r)), int(math.ceil(r))


def register(mosaic: Mosaic, opts: RegistrationOptions, orientation: tuple,
             progress: Callable[[int, int, str], None] | None = None,
             cancelled: Callable[[], bool] | None = None) -> Registration:
    if opts.source == "auto":
        opts = dataclasses.replace(opts, source="volume" if mosaic.zaxis is not None else "max")
    nominal = mosaic.nominal_positions(orientation)
    pairs = pair_candidates(mosaic, nominal)
    if not pairs:
        return Registration(nominal.copy(), nominal, [], opts, tuple(orientation),
                            "no overlapping tiles: kept at the stage positions")
    search = search_radius_px(mosaic, opts, nominal)
    arrays = [t.stack.asarray() for t in mosaic.tiles]
    volume = opts.source == "volume" and mosaic.zaxis is not None
    dims = 3 if volume and opts.max_z_shift > 0 else 2
    links = []
    for k, (i, j) in enumerate(pairs):
        if cancelled is not None and cancelled():
            raise InterruptedError("registration cancelled")
        if progress is not None:
            progress(k, len(pairs), f"{mosaic.tile_label(i)} ↔ {mosaic.tile_label(j)}")
        lk = measure_pair(mosaic, arrays, i, j, nominal[j] - nominal[i], opts, search)
        if lk.measured is not None and lk.r >= opts.min_r and not lk.note:
            lk.used = True
        elif lk.measured is not None and not lk.note:
            lk.note = f"correlation {lk.r:.2f} < {opts.min_r:.2f}"
        links.append(lk)
    if progress is not None:
        progress(len(pairs), len(pairs), "solving")
    if opts.consistency:  # gross outliers only; finer disagreements are judged by the loop residuals
        check_consistency(links, max(5.0, 3.0 * opts.max_residual_px, 0.5 * min(search)), nominal,
                          2.0 * opts.max_residual_px)
    positions, links = solve_positions(len(mosaic.tiles), links, nominal, dims, opts.max_residual_px)
    used = [lk for lk in links if lk.used]
    moved = (positions - nominal)[:, :2]
    moved -= np.median(moved, axis=0)  # relative to the mosaic as a whole
    shift = np.linalg.norm(moved * [mosaic.px, mosaic.py], axis=1) if mosaic.calibrated else np.linalg.norm(moved, axis=1)
    unit = "µm" if mosaic.calibrated else "px"
    rs = [lk.r for lk in used]
    dropped = sum(lk.note.startswith("dropped") for lk in links)
    summary = (f"{len(used)} of {len(pairs)} overlaps used"
               + (f" (r {min(rs):.2f}–{max(rs):.2f})" if rs else "")
               + (f", {dropped} inconsistent dropped" if dropped else "")
               + f", tiles moved by up to {shift.max():.1f} {unit} relative to their stage positions"
               + (f", max residual {max(lk.residual for lk in used):.2f} px" if used else "")
               + f"; source: {'3D volume' if volume else 'plane ' + str(opts.plane) if opts.source == 'plane' else 'max projection'}")
    return Registration(positions, nominal, links, opts, tuple(orientation), summary)


def detect_orientation(mosaic: Mosaic, opts: RegistrationOptions | None = None,
                       progress: Callable[[int, int, str], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> list[tuple[float, tuple]]:
    """Mean overlap correlation of every way the stage axes can map onto the image (best first)."""
    opts = dataclasses.replace(opts or RegistrationOptions(), source="max")  # fast: projections
    arrays = [t.stack.asarray() for t in mosaic.tiles]
    scores = []
    for k, o in enumerate(ORIENTATIONS):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        if progress is not None:
            progress(k, len(ORIENTATIONS), orientation_label(o))
        nominal = mosaic.nominal_positions(o)
        pairs = pair_candidates(mosaic, nominal)
        search = search_radius_px(mosaic, opts, nominal)
        rs = [measure_pair(mosaic, arrays, i, j, nominal[j] - nominal[i], opts, search).r for i, j in pairs]
        scores.append((float(np.mean(rs)) if rs else 0.0, o))
    return sorted(scores, key=lambda x: -x[0])


# -----------------------------------------------------------------------------
# command line
# -----------------------------------------------------------------------------

def report(mosaic: Mosaic) -> str:
    lines = [f"{mosaic.title.replace('×', 'x')}: {len(mosaic.tiles)} tiles of {' x '.join(map(str, mosaic.tile_shape))} "
             f"({mosaic.tile_axes}), {mosaic.px:.4g} µm/px, layout {mosaic.layout}, blend {mosaic.blend}"]
    pos = mosaic.positions()
    for k, t in enumerate(mosaic.tiles):
        st = f"stage ({t.stage[0]:.2f}, {t.stage[1]:.2f}) µm" if t.stage else "no stage position"
        lines.append(f"  {mosaic.tile_label(k):<8} S{t.pos + 1:<3} {st}  ->  x {pos[k, 0]:8.2f}  y {pos[k, 1]:8.2f}"
                     f"  z {pos[k, 2]:5.2f}")
    reg = mosaic.registration
    if reg is not None:
        lines.append(f"  registration: {reg.summary}")
        for lk in reg.links:
            meas = (f"({lk.measured[0]:+8.2f}, {lk.measured[1]:+8.2f}, {lk.measured[2]:+5.2f})"
                    if lk.measured else "-")
            lines.append(f"    {mosaic.tile_label(lk.i):>7} -> {mosaic.tile_label(lk.j):<7} nominal "
                         f"({lk.nominal[0]:+8.2f}, {lk.nominal[1]:+8.2f})  measured {meas}  r {lk.r:5.2f}  "
                         f"{'used' if lk.used else 'not used'}"
                         + (f"  residual {lk.residual:.2f}" if np.isfinite(lk.residual) else "")
                         + (f"  ({lk.note})" if lk.note else ""))
    return "\n".join(lines)


def _non_negative(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fuse the mosaics of ImSpector .msr files into OME-TIFF / ImageJ TIFF.")
    ap.add_argument("inputs", nargs="+", help=".msr files")
    ap.add_argument("-o", "--outdir", help="output folder (default: next to each file)")
    ap.add_argument("--align", action="store_true", help="register the overlaps (stitch) instead of using "
                    "the stage positions as they are")
    ap.add_argument("--blend", choices=list(BLEND_MODES), default="linear")
    ap.add_argument("--source", choices=["auto", "max", "volume"], default="auto",
                    help="registration on all planes (3D, 'volume') or on the maximum projection ('max'); "
                    "auto (default): 3D for z-stacks")
    ap.add_argument("--z-shift", type=_non_negative, default=0,
                    help="allowed z shift between tiles in planes (3D registration only)")
    ap.add_argument("--min-r", type=float, default=0.3, help="minimum overlap correlation of a used link")
    ap.add_argument("--subpixel", action="store_true", help="place tiles with sub-pixel accuracy (interpolated)")
    ap.add_argument("--imagej", action="store_true", help="ImageJ hyperstack TIFF instead of OME-TIFF")
    ap.add_argument("--info", action="store_true", help="only print the mosaics (and the registration)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    failed = 0
    for path in args.inputs:
        try:
            with mr.MSRFile(path) as msr:
                mosaics = find_mosaics(msr)
                if not mosaics:
                    print(f"{path}: no mosaic found")
                    continue
                for n, mo in enumerate(mosaics, 1):
                    mo.set_blend(args.blend)
                    mo.set_subpixel(args.subpixel)
                    if args.align:
                        opts = RegistrationOptions(source=args.source, max_z_shift=args.z_shift, min_r=args.min_r)
                        mo.set_registration(mo.register(opts))
                    print(report(mo))
                    if args.info:
                        continue
                    folder = args.outdir or os.path.dirname(os.path.abspath(path))
                    os.makedirs(mr._fs_path(folder), exist_ok=True)
                    name = (f"{mr.file_stem(path)}_M{n:02d}_mosaic_{mr._safe(mo.detector)}_"
                            f"{'stitched' if mo.layout == 'aligned' else 'stage'}_{mo.blend}"
                            + (".tif" if args.imagej else ".ome.tif"))
                    out = os.path.join(folder, name)
                    rec = mr.export_group(msr, [mo.stack], out, imagej=args.imagej)
                    print(f"  -> {out}  axes={rec['axes']} shape={'x'.join(map(str, rec['shape']))}")
        except (mr.MSRFormatError, OSError, ValueError) as exc:
            failed += 1
            print(f"{path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
