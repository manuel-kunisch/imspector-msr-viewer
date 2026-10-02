#!/usr/bin/env python3
"""
msr_tiff.py -- TIFF, OME-TIFF and ImageJ TIFF files as stacks for the MSR viewer.

Every channel of every image series becomes one stack with the interface of
msr_reader.DataStack (geometry, metadata, asarray), so the viewer and its tools
(PSF analysis, line scan, video, TIFF export) work on TIFFs as on .msr files.

Used where the file has it:
  OME-TIFF     pixel size, z step, frame interval, per-plane times (DeltaT),
               channel names, acquisition date, objective
  ImageJ TIFF  pixel size (resolution tags + unit), z spacing, frame interval
  any TIFF     pixel size from the resolution tags if the unit is cm, mm or µm
Files written by msr_reader.py also carry the ImSpector metadata and settings
(OME map annotation / ImageJ 'Info'); they are read back, so a re-opened export
shows the same information as the .msr file.

The pages of a TIFF without axis information are taken as Z slices, as ImageJ
and Bio-Formats do.  Uncompressed data are memory-mapped, also when written page
by page (as cameras do); compressed, scattered or big-endian data are read into
memory (LZW and JPEG need the imagecodecs package).
"""

from __future__ import annotations

import enum
import itertools
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import tifffile

import msr_reader as mr

TIFF_EXTENSIONS = (".tif", ".tiff")
MAX_READ_BYTES = 8 << 30  # series that cannot be memory-mapped are read only up to this size

_UM = {"µm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0, "micrometer": 1.0, "nm": 1e-3, "nanometer": 1e-3,
       "pm": 1e-6, "å": 1e-4, "mm": 1e3, "millimeter": 1e3, "cm": 1e4, "centimeter": 1e4, "m": 1e6, "meter": 1e6}
_S = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3, "msec": 1e-3, "millisecond": 1e-3,
      "µs": 1e-6, "us": 1e-6, "ns": 1e-9, "min": 60.0, "minute": 60.0, "h": 3600.0, "hr": 3600.0, "hour": 3600.0}
# dimension label per tifffile axis code; msr_reader._axis_letter maps the label back to the letter
_LABELS = {"Z": "Z", "T": "Time", "E": "Lambda", "H": "Lifetime", "R": "Tile", "M": "Tile", "A": "Angle",
           "P": "Phase"}
_RESUNIT_UM = {3: 1e4, 4: 1e3, 5: 1.0}  # TIFF ResolutionUnit centimeter, millimeter, micrometer
_SKIP_TAGS = {"StripOffsets", "StripByteCounts", "TileOffsets", "TileByteCounts", "FreeOffsets", "FreeByteCounts",
              "JPEGTables", "ImageDescription", "IJMetadata", "IJMetadataByteCounts", "ColorMap", "InterColorProfile",
              "XMP", "MicroManagerMetadata", "SubIFDs"}


def is_tiff(path) -> bool:
    return os.fspath(path).lower().endswith(TIFF_EXTENSIONS)


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _unit(text) -> str:
    u = str(text or "").strip().replace("\\u00B5", "µ").replace("\\u00b5", "µ").replace("μ", "µ")
    return u.lower()


def _um(value, unit="µm") -> float | None:
    """Length in µm (None if missing or not a length unit)."""
    try:
        v = float(value) * _UM[_unit(unit) or "µm"]
    except (TypeError, ValueError, KeyError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _seconds(value, unit="s") -> float | None:
    try:
        v = float(value) * _S[_unit(unit) or "s"]
    except (TypeError, ValueError, KeyError):
        return None
    return v if np.isfinite(v) and v >= 0 else None


def _value(text: str):
    """Setting value as written by msr_reader ('0.5', '3', 'text') back to a number where possible."""
    for cast in (int, float):
        try:
            return cast(text)
        except (TypeError, ValueError):
            pass
    return text


def _short(value, limit: int = 300) -> str:
    if isinstance(value, enum.Enum):
        return f"{value.name} ({value.value})"
    if isinstance(value, (bytes, bytearray)):
        return f"{len(value)} bytes"
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) > 16:
        return f"{len(value)} values"
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " …"


def _clock(stamp: str) -> str:
    """HH:MM:SS of '2024-05-01T12:34:56' or of the TIFF DateTime '2024:05:01 12:34:56'."""
    m = re.search(r"[T ](\d{1,2}:\d{2}:\d{2})", stamp or "")
    return m.group(1) if m else ""


def _annotation(pairs: dict) -> dict:
    """ImSpector metadata in files written by msr_reader.py ('meta|…', 'setting|…', 'channel0|id', …)."""
    out = {"meta": {}, "settings": {}, "channel_settings": {}, "channels": {}, "time": "", "camera": {}}
    for k, v in pairs.items():
        group, _, key = str(k).partition("|")
        if not key:
            continue
        m = re.fullmatch(r"setting\[channel(\d+)\]", group)
        if group == "meta":
            out["meta"][key] = v
        elif group == "setting":
            out["settings"][key] = _value(v)
        elif m:
            out["channel_settings"].setdefault(int(m.group(1)), {})[key] = _value(v)
        elif re.fullmatch(r"channel\d+", group):
            out["channels"].setdefault(int(group[7:]), {})[key] = v
        elif group == "stack" and key == "time":
            out["time"] = v
        elif group == "camera":
            out["camera"][key] = v
    return out


def _info_pairs(info: str) -> dict:
    """'key = value' lines of the ImageJ 'Info' property (msr_reader's ImageJ export)."""
    pairs = {}
    for line in (info or "").splitlines():
        k, sep, v = line.partition(" = ")
        if sep and "|" in k:
            pairs[k.strip()] = v.rstrip("\r")  # values exactly as written (may end in spaces)
    return pairs


# -----------------------------------------------------------------------------
# OME-XML
# -----------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name: str) -> list:
    return [c for c in el if _local(c.tag) == name]


def _child(el, name: str):
    return next((c for c in el if _local(c.tag) == name), None)


def _ome_images(root) -> list[dict]:
    """The parts of the OME-XML the viewer uses, one dict per Image."""
    maps, objectives = {}, {}
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "MapAnnotation":
            value = _child(el, "Value")
            if value is not None:
                maps[el.get("ID")] = {m.get("K"): (m.text or "") for m in _children(value, "M")}
        elif tag == "Objective":
            objectives[el.get("ID")] = el
    images = []
    for img in _children(root, "Image"):
        pix = _child(img, "Pixels")
        if pix is None:
            continue
        date = _child(img, "AcquisitionDate")
        d = {"name": img.get("Name", ""), "date": (date.text or "").strip() if date is not None else "",
             "px": _um(pix.get("PhysicalSizeX"), pix.get("PhysicalSizeXUnit", "µm")),
             "py": _um(pix.get("PhysicalSizeY"), pix.get("PhysicalSizeYUnit", "µm")),
             "dz": _um(pix.get("PhysicalSizeZ"), pix.get("PhysicalSizeZUnit", "µm")),
             "dt": _seconds(pix.get("TimeIncrement"), pix.get("TimeIncrementUnit", "s")),
             "channels": [ch.get("Name", "") for ch in _children(pix, "Channel")],
             "planes": [], "annotation": {}, "objective": {}}
        for pl in _children(pix, "Plane"):
            t = _seconds(pl.get("DeltaT"), pl.get("DeltaTUnit", "s"))
            if t is not None:
                d["planes"].append((int(pl.get("TheT", 0)), int(pl.get("TheZ", 0)), int(pl.get("TheC", 0)), t))
        for ref in _children(img, "AnnotationRef"):
            d["annotation"].update(maps.get(ref.get("ID"), {}))
        settings = _child(img, "ObjectiveSettings")
        obj = objectives.get(settings.get("ID")) if settings is not None else None
        if obj is not None:
            d["objective"] = {k: v for k, v in (("ObjectiveID", obj.get("Model", "")), ("ObjectiveNA", obj.get("LensNA", "")),
                                                ("ObjectiveImmersion", obj.get("Immersion", "")),
                                                ("Magnification", obj.get("NominalMagnification", ""))) if v}
        images.append(d)
    if len(images) == 1 and not images[0]["annotation"] and maps:  # annotations not linked to the image
        images[0]["annotation"] = {k: v for m in maps.values() for k, v in m.items()}
    return images


def _ome_flat(root, limit: int = 3000) -> dict:
    """Attributes of the OME-XML as 'OME Image[0]/Pixels SizeX' entries (planes and annotations left out)."""
    out = {}

    def walk(el, path: str) -> None:
        for k, v in el.attrib.items():
            if len(out) < limit:
                out[f"OME {path} {_local(k)}" if path else f"OME {_local(k)}"] = v
        counts: dict[str, int] = {}
        for c in el:
            name = _local(c.tag)
            if name in ("Plane", "TiffData", "BinData", "StructuredAnnotations", "AnnotationRef"):
                continue
            i = counts[name] = counts.get(name, -1) + 1
            sub = f"{path}/{name}[{i}]" if path else f"{name}[{i}]"
            text = (c.text or "").strip()
            if text and not len(c) and len(out) < limit:
                out[f"OME {sub}"] = text
            walk(c, sub)

    walk(root, "")
    return out


# -----------------------------------------------------------------------------
# TIFF tags
# -----------------------------------------------------------------------------

def _tag(page, name: str):
    tag = page.tags.get(name)
    return None if tag is None else tag.value


def _tag_pixel_size(page, unit: str | None = None) -> tuple[float | None, float | None]:
    """(x, y) pixel size in µm from XResolution / YResolution (pixels per unit).

    *unit* is the ImageJ unit; without it the TIFF ResolutionUnit decides, and only
    cm, mm or µm count (an inch value is a print resolution, not a calibration).
    """
    def per_unit(name: str) -> float | None:
        v = _tag(page, name)
        try:
            num, den = v if isinstance(v, tuple) else (v, 1)
            r = float(num) / float(den)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return r if np.isfinite(r) and r > 0 else None

    xr, yr = per_unit("XResolution"), per_unit("YResolution")
    if unit is not None:
        factor = _UM.get(_unit(unit))
    else:
        code = _tag(page, "ResolutionUnit")
        factor = _RESUNIT_UM.get(int(code) if code is not None else 2)
    if not factor or not xr:
        return None, None
    px, py = factor / xr, (factor / yr if yr else factor / xr)
    if unit is None and px > 100:  # 72 dpi stored as px/cm and the like, not a microscope calibration
        return None, None
    return px, py


def _layout(tif, series) -> tuple:
    """How the pixel data can be accessed.

    ('map', offset): one contiguous block, memory-mapped as is.
    ('pages', first, stride): uncompressed pages at a constant spacing (an IFD between
        the pages, as written page by page), memory-mapped as a strided view.
    ('read',): compressed, scattered, big-endian or packed bits, read into memory.
    """
    dtype = np.dtype(series.dtype)
    key = series.keyframe
    if not (tif.byteorder == "<" and dtype.kind in "uif"
            and int(getattr(key, "bitspersample", dtype.itemsize * 8)) == dtype.itemsize * 8):
        return ("read",)
    if series.dataoffset is not None:
        return ("map", int(series.dataoffset))
    if int(getattr(key, "compression", 1)) != 1:
        return ("read",)
    page_bytes = int(np.prod(key.shape)) * dtype.itemsize
    starts = []
    for page in series.pages:
        offsets, counts = (page.dataoffsets, page.databytecounts) if page is not None else ((), ())
        if not offsets or sum(counts) != page_bytes or any(offsets[i] + counts[i] != offsets[i + 1]
                                                           for i in range(len(offsets) - 1)):
            return ("read",)
        starts.append(int(offsets[0]))
    stride = starts[1] - starts[0] if len(starts) > 1 else 0
    if (stride < page_bytes or any(b - a != stride for a, b in zip(starts, starts[1:]))
            or len(starts) * int(np.prod(key.shape)) != int(np.prod(series.shape))):
        return ("read",)
    return ("pages", starts[0], stride)


_ACCESS = {"map": "memory-mapped", "pages": "memory-mapped (page by page)", "read": "read into memory"}


# -----------------------------------------------------------------------------
# stacks
# -----------------------------------------------------------------------------

@dataclass
class TIFFStack(mr.DataStack):
    """One channel of a TIFF image series, with the interface of :class:`msr_reader.DataStack`."""

    series: int = 0
    channel: int = 0
    assumed: str = ""  # axis letter given without metadata ('Z' for the pages of a plain TIFF)
    access: str = ""   # 'memory-mapped' or 'read into memory'
    _file: "TIFFFile | None" = field(default=None, repr=False, compare=False)
    _select: tuple = field(default=(), repr=False, compare=False)

    def asarray(self) -> np.ndarray:
        """Pixel data with axes :attr:`axes` (a view of the memory-mapped or loaded series)."""
        return self._file._series_data(self.series)[self._select]


class TIFFFile:
    """A TIFF file as stacks, with the attributes of :class:`msr_reader.MSRFile`.

    Attributes:
        kind: 'OME-TIFF', 'ImageJ TIFF' or 'TIFF'.
        properties: file metadata {name: value} (TIFF tags, ImageJ and OME entries).
        stacks: list of :class:`TIFFStack`, one per channel of each image series.
        metadata: what was found, e.g. ['OME-XML', 'ImSpector metadata (msr_reader export)'].
        warnings: problems encountered while reading.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self.kind = "TIFF"
        self.version = ""
        self.software = ""
        self.series_count = 0
        self.page_count = 0
        self.metadata: list[str] = []
        self.properties: dict = {}
        self.labels: dict[str, str] = {}
        self.views: list = []
        self.property_sets: list = []
        self.document: dict = {}
        self.stacks: list[TIFFStack] = []
        self.warnings: list[str] = []
        self._data: dict[int, np.ndarray] = {}
        self._layouts: dict[int, tuple] = {}
        self._tif = tifffile.TiffFile(mr._fs_path(self.path))
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._data.clear()
        if getattr(self, "_tif", None) is not None:
            self._tif.close()
            self._tif = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # -- reading ----------------------------------------------------------------

    def _parse(self) -> None:
        tif = self._tif
        page = tif.pages[0]
        self.kind = "OME-TIFF" if tif.is_ome else "ImageJ TIFF" if tif.is_imagej else "TIFF"
        self.version = self.kind + (" (BigTIFF)" if tif.is_bigtiff else "")
        self.software = str(_tag(page, "Software") or "").strip()
        self.page_count = len(tif.pages)
        stamp = str(_tag(page, "DateTime") or "").strip()
        images, root = [], None
        if tif.is_ome:
            try:
                root = ET.fromstring(tif.ome_metadata)
                images = _ome_images(root)
                self.metadata.append("OME-XML")
            except ET.ParseError as exc:
                self._warn(f"OME-XML could not be parsed: {exc}")
        ij = (tif.imagej_metadata or {}) if tif.is_imagej else {}
        if ij:
            self.metadata.append("ImageJ")
        self._file_properties(page, ij, root)
        series = [(si, s) for si, s in enumerate(tif.series) if not getattr(s.keyframe, "is_reduced", False)]
        self.series_count = len(series)
        if len(images) == len(series):
            matched = images
        else:
            by_name = {d["name"]: d for d in images if d["name"]}
            matched = [by_name.get(s.name) for _, s in series]
        for (si, s), ome in zip(series, matched):
            try:
                self._add_series(si, s, ome, ij, stamp)
            except (ValueError, KeyError, IndexError) as exc:
                self._warn(f"series {si} ({s.axes} {'x'.join(map(str, s.shape))}): {exc}")
        if not self.stacks:
            raise ValueError("no readable image data" + (f" ({self.warnings[-1]})" if self.warnings else ""))
        if any(st.properties or st.meta.get("VersionNr") for st in self.stacks):
            self.metadata.append("ImSpector metadata (msr_reader export)")
        if any(st.assumed for st in self.stacks):
            self.metadata.append("no axis information: pages taken as Z slices")

    def _file_properties(self, page, ij: dict, root) -> None:
        for tag in page.tags.values():
            if tag.name not in _SKIP_TAGS:
                key = f"TIFF {tag.name}"
                self.properties[key] = _short(tag.value)
                self.labels[key] = f"TIFF tag {tag.code}"
        for k, v in ij.items():
            if k != "Info":
                self.properties[f"ImageJ {k}"] = _short(v)
                self.labels[f"ImageJ {k}"] = "ImageJ metadata"
        if root is not None:
            for k, v in _ome_flat(root).items():
                self.properties[k] = _short(v)
                self.labels[k] = "OME-XML"

    def _add_series(self, si: int, series, ome: dict | None, ij: dict, stamp: str) -> None:
        tif = self._tif
        axes, shape = series.axes, tuple(int(n) for n in series.shape)
        dims = list(zip(axes, shape))  # slowest first, image plane last
        if len(axes) != len(shape) or "Y" not in axes or "X" not in axes:
            raise ValueError(f"no image plane in axes {axes}")
        dtype = np.dtype(series.dtype)
        if dtype.kind not in "buif":
            raise ValueError(f"{dtype} pixels are not supported")
        layout = self._layouts[si] = _layout(tif, series)
        if layout[0] == "read":
            try:  # fails early if the compression needs imagecodecs
                series.keyframe.asarray()
            except ValueError as exc:
                raise ValueError(f"{exc} (pip install imagecodecs)" if "imagecodecs" in str(exc) else str(exc)) from None
        chan = [i for i, (a, n) in enumerate(dims) if a in "CS" and n > 1]
        drop = [i for i, (a, n) in enumerate(dims) if n == 1 and a not in "YX" and i not in chan]
        keep = [i for i in range(len(dims)) if i not in chan and i not in drop]
        if [dims[i][0] for i in keep[-2:]] != ["Y", "X"]:
            raise ValueError(f"unsupported axis order {axes}")
        other = keep[:-2]

        # calibration: OME, else ImageJ, else the resolution tags
        px = py = dz = dt = None
        names, ann, date = [], _annotation({}), stamp
        if ome:
            px, py, dz, dt = ome["px"], ome["py"] or ome["px"], ome["dz"], ome["dt"]
            names, ann, date = ome["channels"], _annotation(ome["annotation"]), ome["date"] or stamp
        elif ij:
            unit = ij.get("unit")
            px, py = _tag_pixel_size(series.keyframe, unit) if unit else (None, None)
            dz = _um(ij.get("spacing"), unit) if unit else None
            dt = _seconds(ij.get("finterval"), ij.get("tunit", "s"))
            ann = _annotation(_info_pairs(ij.get("Info", "")))
        if px is None:
            px, py = _tag_pixel_size(series.keyframe)
        dt = dt if dt else None

        # one label per remaining axis; pages without axis information become Z (or T) slices
        letters, labels_other, assumed = [], [], ""
        present = {dims[i][0] for i in other}
        for i in other:
            a = dims[i][0]
            if a in _LABELS:
                letter = "R" if a == "M" else a
                label = _LABELS[a]
            elif not assumed and ("Z" not in present or "T" not in present):
                letter = assumed = "Z" if "Z" not in present else "T"
                label = _LABELS[letter]
            else:
                letter, label = "Q", "Frame"
            letters.append(letter)
            labels_other.append(label)

        nx, ny = dims[keep[-1]][1], dims[keep[-2]][1]
        steps = {"Z": (dz, "µm"), "T": (dt, "s")}
        sizes = [nx, ny] + [dims[i][1] for i in reversed(other)]
        labels = ["X", "Y"] + list(reversed(labels_other))
        lengths = [px * nx if px else 0.0, py * ny if py else 0.0]
        units = ["µm" if px else "", "µm" if py else ""]
        for i, letter in zip(reversed(other), reversed(letters)):
            step, unit = steps.get(letter, (None, ""))
            lengths.append(step * dims[i][1] if step else 0.0)
            units.append(unit if step else "")

        rgb = int(getattr(series.keyframe, "photometric", 1)) == 2
        combos = list(itertools.product(*[range(dims[i][1]) for i in chan]))
        n_t = next((dims[i][1] for i, letter in zip(other, letters) if letter == "T"), 0)
        access = _ACCESS[layout[0]]
        acquired = date
        for k, combo in enumerate(combos):
            pick = {dims[i][0]: c for i, c in zip(chan, combo)}
            c = pick.get("C", 0)
            info = ann["channels"].get(k, {})
            name = info.get("id") or (names[c] if c < len(names) and names[c] else "")
            if "S" in pick:
                s = pick["S"]
                sample = "RGBA"[s] if rgb and s < 4 else f"S{s + 1}"
                name = f"{name} {sample}".strip() if not info.get("id") else name
            if not name and len(combos) > 1:
                name = f"C{k + 1}"
            sel = [slice(None)] * len(dims)
            for i in drop:
                sel[i] = 0
            for i, v in zip(chan, combo):
                sel[i] = v
            meta = dict(ann["meta"])
            for key, v in (ome or {}).get("objective", {}).items():
                meta.setdefault(key, v)
            if acquired:
                meta.setdefault("Creation Date", acquired.replace("T", " "))
            settings = dict(ann["settings"])
            settings.update(ann["channel_settings"].get(k, {}))
            cam = None
            if ann["camera"].get("first_frame_stamp"):
                cam = {"time": ann["camera"]["first_frame_stamp"],
                       "image_counter": _value(ann["camera"].get("image_counter", ""))}
            st = TIFFStack(index=len(self.stacks), class_name="TIFF", schema=0, offset=int(series.dataoffset or 0),
                           path=self.path, name=series.name or "", time=ann["time"] or _clock(date),
                           source=info.get("source") or self.kind, title=series.name or "", meta=meta,
                           channel_id=name, sizes=sizes, lengths=lengths, labels=labels, units=units,
                           timestamps=self._times(ome, c if "C" in pick else 0, n_t),
                           dtype="uint8" if dtype.kind == "b" else dtype.name,
                           data_offset=int(series.dataoffset or 0),
                           properties=settings, header={"series": si, "axes": axes, "shape": list(shape),
                                                        "kind": getattr(series, "kind", "")},
                           parse_mode="tifffile", camera_stamp=cam, series=si, channel=k, assumed=assumed,
                           access=access, _file=self, _select=tuple(sel))
            if st.shape != tuple(dims[i][1] for i in keep):
                raise ValueError(f"axes {axes} could not be mapped ({st.axes} {st.shape})")
            self.stacks.append(st)

    @staticmethod
    def _times(ome: dict | None, c: int, n_t: int) -> list[float]:
        """Per-frame times (s) from the OME planes (DeltaT of z = 0), if every frame has one."""
        if not ome or n_t < 2:
            return []
        for channel in (c, 0):
            t = {}
            for the_t, the_z, the_c, delta in ome["planes"]:
                if the_z == 0 and the_c == channel:
                    t.setdefault(the_t, delta)
            if sorted(t) == list(range(n_t)):
                return [t[i] for i in range(n_t)]
        return []

    def _series_data(self, si: int) -> np.ndarray:
        arr = self._data.get(si)
        if arr is None:
            arr = self._data[si] = self._read(si)
        return arr

    def _read(self, si: int) -> np.ndarray:
        series = self._tif.series[si]
        shape = tuple(int(n) for n in series.shape)
        dtype = np.dtype(series.dtype)
        layout = self._layouts.get(si) or _layout(self._tif, series)
        if layout[0] == "map":
            return np.memmap(mr._fs_path(self.path), dtype=dtype.newbyteorder("<"), mode="r",
                             offset=layout[1], shape=shape)
        if layout[0] == "pages":
            page = tuple(int(n) for n in series.keyframe.shape)
            inner, acc = [], dtype.itemsize
            for n in reversed(page):
                inner.insert(0, acc)
                acc *= n
            raw = np.memmap(mr._fs_path(self.path), dtype=np.uint8, mode="r")
            n_pages = int(np.prod(shape)) // int(np.prod(page))
            arr = np.ndarray((n_pages,) + page, dtype=dtype.newbyteorder("<"), buffer=raw, offset=layout[1],
                             strides=(layout[2],) + tuple(inner))
            return arr.reshape(shape)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if nbytes > MAX_READ_BYTES:
            raise ValueError(f"{nbytes / 1e9:.1f} GB of compressed or scattered pixel data is too much to read "
                             "into memory; save the file uncompressed")
        arr = np.asarray(series.asarray()).reshape(shape)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.uint8)
        if not arr.dtype.isnative:
            arr = arr.astype(arr.dtype.newbyteorder("="))
        return arr

    # -- description -------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"{os.path.basename(self.path)}: {self.version}, {self.series_count} series, "
                 f"{len(self.stacks)} stacks" + (f" ({', '.join(self.metadata)})" if self.metadata else "")]
        for s in self.stacks:
            px = s.pixel_size[0]
            dt = s.time_increment if "T" in s.axes else None
            lines.append(f"  S{s.index + 1:<2} {s.channel_id or s.source:<20} series {s.series} {s.axes:<5} "
                         f"{'x'.join(map(str, s.shape)):<16} {s.dtype:<8}" + (f" {px:.4g} um/px" if px else "")
                         + (f"  dt={dt:.4g}s" if dt else "") + (f"  [{s.assumed} assumed]" if s.assumed else ""))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)
