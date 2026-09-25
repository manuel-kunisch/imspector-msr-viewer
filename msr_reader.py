#!/usr/bin/env python3
"""
msr_reader.py -- read LaVision BioTec ImSpector (.msr) files and export the
image data to OME-TIFF or ImageJ-hyperstack TIFF.

An .msr file is a Microsoft MFC ``CArchive`` serialization of an ImSpector
Pro document (the reverse-engineered layout is described in MSR_FORMAT.md):

    int32              file version (8)
    property list      global instrument settings (CProperty objects)
    document fields
    stack array        CDataStack = pixel data + settings snapshot,
                       CImageStack = display view (LUT, zoom) of the stack before it
    CChildFrame array  window layout (ignored)
    CPropArray         property set of the measurement workspace ('Measurement 1')

Command line::

    python msr_reader.py FILE_OR_FOLDER [...] [-o OUTDIR] [--imagej]
                         [--group auto|geometry|none] [--compress zlib] [--timestamps] [--info]

Python::

    from msr_reader import MSRFile
    with MSRFile("measurement.msr") as msr:
        for s in msr.stacks:
            print(s.source, s.axes, s.shape, s.pixel_size)
            arr = s.asarray()          # numpy memmap, axes given by s.axes
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

__version__ = "0.3.0"

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")

# MFC CArchive object tags (afx.h, arcobj.cpp)
_NULL_TAG = 0x0000
_NEW_CLASS_TAG = 0xFFFF
_CLASS_TAG = 0x8000
_BIG_OBJECT_TAG = 0x7FFF
_BIG_CLASS_TAG = 0x80000000

_STACK_CLASSES = ("CDataStack", "CImageStack")
_CHANNEL_ID_RE = re.compile(rb"([\x03-\x7f])([A-Za-z][\x20-\x7e]{0,120}?:\d+:\d+:\d+)")
_NEW_CLASS_RE = re.compile(rb"\xff\xff(..)(..)(C[A-Za-z0-9_]{1,63})", re.S)


class MSRFormatError(ValueError):
    """The file does not match the expected .msr structure."""


def _decode_ansi(raw: bytes) -> str:
    # Some values hold UTF-16 text inside an 8-bit CString ('L\0a\0s\0e\0r\0').
    if len(raw) >= 4 and len(raw) % 2 == 0 and not any(raw[1::2]) and all(raw[0::2]):
        return raw.decode("utf-16-le")
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xml_safe(text: str) -> str:
    return _XML_ILLEGAL.sub("", text)


def _fs_path(path: str) -> str:
    """Path the Win32 API accepts even beyond MAX_PATH (260 characters).

    ImSpector file names are long, so '<name>_tiff/<file>' easily exceeds the
    limit; the '\\\\?\\' prefix lifts it.
    """
    if os.name != "nt":
        return path
    p = os.path.abspath(path)
    if len(p) < 240 or p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):  # UNC path \\server\share\...
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def _printable(s: str) -> bool:
    return all(ch >= " " or ch in "\t\r\n" for ch in s)


def _prod(values) -> int:
    p = 1
    for v in values:
        p *= int(v)
    return p


def _parse_meta(text: str) -> dict[str, str]:
    """Parse ImSpector's 'Key::Value::Key::Value::' metadata string."""
    parts = text.split("::")
    if parts and parts[-1] == "":
        parts.pop()
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


class _Cursor:
    """Little-endian reader over a buffer (bytes or mmap)."""

    __slots__ = ("buf", "pos", "size")

    def __init__(self, buf, pos: int = 0):
        self.buf = buf
        self.pos = pos
        self.size = len(buf)

    def _unpack(self, st: struct.Struct):
        end = self.pos + st.size
        if end > self.size:
            raise MSRFormatError(f"unexpected end of file at offset {self.pos}")
        value = st.unpack_from(self.buf, self.pos)[0]
        self.pos = end
        return value

    def u8(self) -> int:
        return self._unpack(_U8)

    def u16(self) -> int:
        return self._unpack(_U16)

    def i32(self) -> int:
        return self._unpack(_I32)

    def u32(self) -> int:
        return self._unpack(_U32)

    def u64(self) -> int:
        return self._unpack(_U64)

    def f32(self) -> float:
        return self._unpack(_F32)

    def f64(self) -> float:
        return self._unpack(_F64)

    def raw(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0 or end > self.size:
            raise MSRFormatError(f"cannot read {n} bytes at offset {self.pos}")
        data = bytes(self.buf[self.pos:end])
        self.pos = end
        return data

    def _long_length(self, n: int) -> int:
        if n < 0xFFFF:
            return n
        n = self.u32()
        if n < 0xFFFFFFFF:
            return n
        return self.u64()

    def count(self) -> int:
        """CArchive::ReadCount (WORD, escalating to DWORD/QWORD)."""
        return self._long_length(self.u16())

    def cstring(self, max_chars: int = 1 << 28) -> str:
        """CString as written by CArchive::operator<< (AfxReadStringLength)."""
        n = self.u8()
        char_size = 1
        if n == 0xFF:
            n = self.u16()
            if n == 0xFFFE:  # Unicode marker, the real length follows
                char_size = 2
                n = self.u8()
                if n == 0xFF:
                    n = self._long_length(self.u16())
            else:
                n = self._long_length(n)
        if n > max_chars:
            raise MSRFormatError(f"implausible string length {n} at offset {self.pos}")
        raw = self.raw(n * char_size)
        return raw.decode("utf-16-le", "replace") if char_size == 2 else _decode_ansi(raw)


@dataclass
class _ClassInfo:
    name: str
    schema: int
    index: int


class _Archive:
    """Class/object index bookkeeping of an MFC CArchive in load mode.

    Classes and objects share one index space starting at 1 (0 is NULL).  A new
    class is announced by 0xFFFF + CRuntimeClass (schema, name) and gets the
    next index; the object that follows gets the index after that.  Later
    objects of a known class are announced by 0x8000 | class_index.
    """

    def __init__(self, buf):
        self.buf = buf
        self.map: list[Any] = [None]

    def decode_tag(self, pos: int):
        c = _Cursor(self.buf, pos)
        wtag = c.u16()
        if wtag == _NEW_CLASS_TAG:
            schema = c.u16()
            n = c.u16()
            name = c.raw(n) if 0 < n <= 64 else b""
            if not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]*", name):
                raise MSRFormatError(f"invalid class declaration at offset {pos}")
            return "new", (name.decode("ascii"), schema), c.pos
        if wtag == _BIG_OBJECT_TAG:
            obtag = c.u32()
        else:
            obtag = ((wtag & _CLASS_TAG) << 16) | (wtag & 0x7FFF)
        if obtag & _BIG_CLASS_TAG:
            return "class", obtag & 0x7FFFFFFF, c.pos
        if obtag == _NULL_TAG:
            return "null", None, c.pos
        return "object", obtag, c.pos

    def is_class(self, index: int) -> bool:
        return 0 < index < len(self.map) and isinstance(self.map[index], _ClassInfo)

    def class_at(self, pos: int) -> str | None:
        """Name of the class announced by an object tag at *pos* (no side effects)."""
        try:
            kind, value, _ = self.decode_tag(pos)
        except MSRFormatError:
            return None
        if kind == "new":
            return value[0]
        if kind == "class" and self.is_class(value):
            return self.map[value].name
        return None

    def force_class(self, index: int, name: str) -> None:
        """Re-synchronise the map after a region could not be parsed."""
        while len(self.map) <= index:
            self.map.append(None)
        self.map[index] = _ClassInfo(name, -1, index)

    def read_object_tag(self, cur: _Cursor) -> _ClassInfo | None:
        """Consume the tag that precedes a serialized CObject and register it."""
        kind, value, nxt = self.decode_tag(cur.pos)
        if kind == "null":
            cur.pos = nxt
            return None
        if kind == "object":
            raise MSRFormatError(f"unsupported object back-reference #{value} at offset {cur.pos}")
        if kind == "new":
            info = _ClassInfo(value[0], value[1], len(self.map))
            self.map.append(info)
        elif self.is_class(value):
            info = self.map[value]
        else:
            raise MSRFormatError(f"reference to unknown class #{value} at offset {cur.pos}")
        cur.pos = nxt
        self.map.append(info.name)  # index of the object instance
        return info


@dataclass
class Property:
    """One ImSpector setting (CProperty)."""

    key: str
    label: str
    type: int  # 1 double, 2/4 int32, 3 float32, 5 string
    group: int
    flags: int
    value: Any


@dataclass
class ImageView:
    """CImageStack: display state of the data stack stored just before it."""

    index: int
    class_name: str
    schema: int
    offset: int
    name: str = ""
    time: str = ""
    source: str = ""
    title: str = ""
    meta: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    slots: list = field(default_factory=list)

    @property
    def lut(self) -> tuple[float, float] | None:
        """(min, max) display range of the primary layer."""
        if self.slots:
            return self.slots[0]["lut_min"], self.slots[0]["lut_max"]
        return None


@dataclass
class DataStack:
    """CDataStack: one detector channel of a measurement, with its pixel data.

    ``sizes``, ``lengths``, ``labels`` and ``units`` describe the four stored
    dimensions, fastest-varying first (X, Y, then e.g. Z or time).
    """

    index: int
    class_name: str
    schema: int
    offset: int
    path: str
    name: str = ""
    time: str = ""
    source: str = ""
    title: str = ""
    meta: dict = field(default_factory=dict)
    channel_id: str = ""
    sizes: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    units: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    dtype: str = "uint16"
    data_offset: int = 0
    data_end: int = 0
    properties: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    parse_mode: str = "structured"
    view: ImageView | None = None
    camera_stamp: dict | None = None

    # -- geometry ---------------------------------------------------------

    @property
    def dim_letters(self) -> list[str]:
        """Axis code per stored dimension ('' for unused singleton dimensions)."""
        letters = ["X", "Y"]
        for size, label in zip(self.sizes[2:], self.labels[2:]):
            letter = _axis_letter(label) if size > 1 else ""
            if size > 1 and not letter:
                letter = "Q"
            if letter and letter in letters:
                letter = next((c for c in "QZTC" if c not in letters), "Q")
            letters.append(letter)
        return letters

    @property
    def axes(self) -> str:
        """Axes of :meth:`asarray` (slowest first), e.g. 'TYX'."""
        return "".join(reversed([c for c in self.dim_letters if c]))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s, c in zip(reversed(self.sizes), reversed(self.dim_letters)) if c)

    def axis_info(self, letter: str) -> dict | None:
        for i, c in enumerate(self.dim_letters):
            if c == letter:
                return {
                    "size": int(self.sizes[i]),
                    "length": float(self.lengths[i]) if i < len(self.lengths) else 0.0,
                    "label": self.labels[i] if i < len(self.labels) else "",
                    "unit": self.units[i] if i < len(self.units) else "",
                }
        return None

    def step(self, letter: str) -> float | None:
        """Physical step size (length / size) along an axis."""
        info = self.axis_info(letter)
        if not info or info["size"] <= 0 or not np.isfinite(info["length"]) or info["length"] <= 0:
            return None
        return info["length"] / info["size"]

    @property
    def pixel_size(self) -> tuple[float | None, float | None]:
        """(x, y) pixel size in the unit of the X/Y axes (normally µm)."""
        return self.step("X"), self.step("Y")

    @property
    def time_increment(self) -> float | None:
        """Frame interval in seconds, from the per-frame timestamps if present."""
        if len(self.timestamps) >= 2:
            d = np.diff(np.asarray(self.timestamps, dtype=np.float64))
            if np.all(np.isfinite(d)):
                return float(np.median(d))
        return self.step("T")

    @property
    def nbytes(self) -> int:
        return _prod(self.sizes) * np.dtype(self.dtype).itemsize

    # -- data -------------------------------------------------------------

    def asarray(self) -> np.memmap:
        """Pixel data as a read-only memory map with axes :attr:`axes`."""
        full = tuple(int(s) for s in reversed(self.sizes))
        arr = np.memmap(_fs_path(self.path), dtype=np.dtype(self.dtype).newbyteorder("<"),
                        mode="r", offset=self.data_offset, shape=full)
        return arr.reshape(self.shape)


def _axis_letter(label: str) -> str:
    text = label.strip()
    low = text.lower()
    if not text or low == "none":
        return ""
    for letter, words in (("H", ("tcspc", "tdc", "lifetime", "flim")),
                          ("E", ("lambda", "wavelength", "spectr")),
                          ("C", ("channel",))):
        if any(w in low for w in words):
            return letter
    last = text.split()[-1].upper()
    if last in ("X", "Y", "Z", "T"):
        return last
    for letter, words in (("R", ("tile", "mosaic")), ("A", ("angle",)), ("P", ("phase",)),
                          ("T", ("time",))):
        if any(w in low for w in words):
            return letter
    return "Q"


def _decode_pco_stamp(row: np.ndarray) -> dict | None:
    """Decode the BCD time stamp PCO cameras write into the first 14 pixels."""
    if row.size < 14:
        return None
    digits = []
    for v in row[:14].tolist():
        hi, lo = (int(v) >> 4), (int(v) & 0xF)
        if int(v) > 0xFF or hi > 9 or lo > 9:
            return None
        digits.append(hi * 10 + lo)
    counter = digits[0] * 1_000_000 + digits[1] * 10_000 + digits[2] * 100 + digits[3]
    year, month, day = digits[4] * 100 + digits[5], digits[6], digits[7]
    hour, minute, sec = digits[8], digits[9], digits[10]
    usec = digits[11] * 10_000 + digits[12] * 100 + digits[13]
    if not (1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
            and hour < 24 and minute < 60 and sec < 60):
        return None
    return {"image_counter": counter,
            "time": f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}.{usec:06d}"}


class MSRFile:
    """A LaVision BioTec ImSpector .msr file.

    Attributes:
        version: file format version (first int32 of the archive).
        properties: global settings {key: value}.
        stacks: list of :class:`DataStack` (the image data).
        views: list of :class:`ImageView` (display settings).
        property_sets: saved measurement property sets.
        labels: {property key: human readable label}.
        warnings: problems encountered while parsing.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fh = open(_fs_path(self.path), "rb")
        try:
            self._buf = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            self._fh.close()
            raise MSRFormatError(f"{self.path}: empty file") from None
        self.size = len(self._buf)
        self._ar = _Archive(self._buf)
        self.version: int | None = None
        self.properties: dict[str, Any] = {}
        self.document: dict[str, Any] = {}
        self.stacks: list[DataStack] = []
        self.views: list[ImageView] = []
        self.property_sets: list[dict] = []
        self.labels: dict[str, str] = {}
        self.warnings: list[str] = []
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_buf", None) is not None:
            self._buf.close()
            self._buf = None
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # -- top level ------------------------------------------------------------

    def _parse(self) -> None:
        cur = _Cursor(self._buf)
        self.version = cur.i32()
        if self._property_list_score(cur.pos) == 2:
            props, trailer = self._read_property_list(cur)
            self.properties = self._props_to_dict(props)
            self.document["global_property_count"] = len(props)
        array_pos = self._find_stack_array(cur.pos)
        if array_pos is None:
            raise MSRFormatError(f"{self.path}: no image stacks found (not an ImSpector .msr file?)")
        self._read_document_fields(cur.pos, array_pos)
        cur.pos = array_pos
        count = cur.count()
        self.document["stack_array_count"] = count
        last_stack = None
        for i in range(count):
            start = cur.pos
            try:
                self._repair_class_ref(cur.pos)
                info = self._ar.read_object_tag(cur)
                if info is None:
                    continue
                if info.name == "CImageStack":
                    view = self._parse_image_view(cur, i, info, start, i == count - 1)
                    self.views.append(view)
                    if last_stack is not None and last_stack.view is None:
                        last_stack.view = view
                else:
                    last_stack = self._parse_data_stack(cur, i, info, start)
                    self.stacks.append(last_stack)
            except MSRFormatError as exc:
                self._warn(f"stack array element {i} at offset {start}: {exc}")
                nxt = self._resync(start + 2, i == count - 1)
                if nxt is None:
                    break
                cur.pos = nxt
        try:
            self._parse_tail(cur.pos)
        except MSRFormatError as exc:
            self._warn(f"property sets at end of file: {exc}")

    def _find_stack_array(self, pos: int) -> int | None:
        """Offset of the stack-array count: WORD n followed by a new-class tag."""
        for m in _NEW_CLASS_RE.finditer(self._buf, pos, min(self.size, pos + (1 << 20))):
            n = _U16.unpack(m.group(2))[0]
            name = m.group(3)[:n]
            if len(name) != n or name == b"CProperty":
                continue
            p = m.start() - 2
            if p < pos or not 0 < _U16.unpack_from(self._buf, p)[0] < 0xFFFF:
                continue
            if self._base_header_ok(m.start() + 6 + n):
                return p
        return None

    def _read_document_fields(self, start: int, end: int) -> None:
        cur = _Cursor(self._buf, start)
        strings = []
        try:
            while cur.pos < end and len(strings) < 2:
                strings.append(cur.cstring(4096))
        except MSRFormatError:
            pass
        self.document["strings"] = strings
        if cur.pos < end and end - cur.pos <= 4096:
            rest = bytes(self._buf[cur.pos:end])
            if any(rest):
                self.document["raw"] = rest.hex(" ")

    def _parse_tail(self, pos: int) -> None:
        m = re.compile(rb"\xff\xff..\x0a\x00CPropArray", re.S).search(self._buf, pos)
        if m is None:
            return
        cur = _Cursor(self._buf, m.start())
        self._ar.read_object_tag(cur)
        current: dict | None = None
        # Each set: a small header list (propset_label, seq_autosave_prefix,
        # propset_id) followed by the list of settings it stores.
        while cur.pos < self.size - 2 and self._property_list_score(cur.pos) == 2:
            props, _ = self._read_property_list(cur)
            if all(p.key.startswith(("propset_", "seq_")) for p in props):
                current = {p.key: p.value for p in props}
                current["properties"] = {}
                self.property_sets.append(current)
            else:
                if current is None:
                    current = {"properties": {}}
                    self.property_sets.append(current)
                current["properties"].update(self._props_to_dict(props))

    # -- properties -------------------------------------------------------

    def _props_to_dict(self, props: list[Property]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in props:
            self.labels.setdefault(p.key, p.label)
            out[p.key] = p.value
        return out

    def _read_property(self, cur: _Cursor) -> Property:
        key = cur.cstring(4096)
        typ = cur.i32()
        label = cur.cstring(4096)
        group = cur.u16()
        flags = cur.u16()
        if typ == 1:
            value = cur.f64()
        elif typ in (2, 4):
            value = cur.i32()
        elif typ == 3:
            value = cur.f32()
        elif typ == 5:
            value = cur.cstring()
        else:
            value = self._read_unknown_value(cur, key, typ)
        return Property(key, label, typ, group, flags, value)

    def _read_unknown_value(self, cur: _Cursor, key: str, typ: int) -> str:
        # Choose the value size after which the next property tag follows.
        size = 4
        for n in (4, 8, 1, 2, 0, 16):
            if self._ar.class_at(cur.pos + n) == "CProperty":
                size = n
                break
        self._warn(f"property {key!r}: unknown value type {typ}, read {size} raw bytes")
        return cur.raw(size).hex()

    def _read_property_list(self, cur: _Cursor) -> tuple[list[Property], int]:
        """count + CProperty objects + int32 trailer."""
        n = cur.count()
        props = []
        for _ in range(n):
            info = self._ar.read_object_tag(cur)
            if info is None:
                continue
            if info.name != "CProperty":
                raise MSRFormatError(f"expected CProperty, found {info.name} at offset {cur.pos}")
            props.append(self._read_property(cur))
        return props, cur.i32()

    def _property_list_score(self, pos: int) -> int:
        """2: a (non-empty) property list starts at pos, 1: maybe (empty list), 0: no."""
        try:
            c = _Cursor(self._buf, pos)
            n = c.count()
            if n == 0:
                return 1 if c.i32() == 1 else 0
            if n > 10_000_000 or self._ar.class_at(c.pos) != "CProperty":
                return 0
            c.pos = self._ar.decode_tag(c.pos)[2]
            key = c.cstring(4096)
            typ = c.i32()
            label = c.cstring(4096)
            return 2 if 0 <= typ < 64 and key and _printable(key) and _printable(label) else 0
        except MSRFormatError:
            return 0

    # -- common stack header ------------------------------------------------

    def _read_base_header(self, cur: _Cursor, obj) -> None:
        obj.header["base_version"] = cur.i32()
        obj.name = cur.cstring()
        obj.time = cur.cstring()
        obj.source = cur.cstring()
        extra = [cur.cstring(), cur.cstring()]
        obj.title = cur.cstring()
        obj.meta = _parse_meta(cur.cstring())
        if any(extra):
            obj.header["base_strings"] = extra

    def _base_header_ok(self, pos: int) -> bool:
        try:
            c = _Cursor(self._buf, pos)
            if not 0 < c.i32() < 1000:
                return False
            return all(_printable(c.cstring(1 << 16)) for _ in range(7))
        except MSRFormatError:
            return False

    def _repair_class_ref(self, pos: int) -> None:
        """If the index map lost sync, identify a stack class by its layout."""
        try:
            kind, value, nxt = self._ar.decode_tag(pos)
        except MSRFormatError:
            return
        if kind != "class" or self._ar.is_class(value) or not self._base_header_ok(nxt):
            return
        c = _Cursor(self._buf, nxt)
        c.i32()
        for _ in range(7):
            c.cstring()
        version = c.i32()
        name = "CImageStack" if version == 5 else "CDataStack"
        self._ar.force_class(value, name)
        self._warn(f"class index #{value} at offset {pos} re-synchronised as {name}")

    def _is_next_element(self, pos: int, is_last: bool) -> bool:
        if is_last:
            try:
                c = _Cursor(self._buf, pos)
                n = c.count()
            except MSRFormatError:
                return pos >= self.size
            return n == 0 or self._ar.class_at(c.pos) is not None
        name = self._ar.class_at(pos)
        if name is None:
            return False
        return self._base_header_ok(self._ar.decode_tag(pos)[2])

    def _resync(self, pos: int, is_last: bool) -> int | None:
        """Find the start of the next stack-array element (or the array end)."""
        limit = min(self.size, pos + (1 << 26))
        cands = []
        for m in _NEW_CLASS_RE.finditer(self._buf, pos, limit):
            cands.append(m.start() - 2 if is_last else m.start())
            if len(cands) > 64:
                break
        if not is_last:
            for idx, item in enumerate(self._ar.map):
                if isinstance(item, _ClassInfo) and item.name != "CProperty":
                    tag = (_U16.pack(_CLASS_TAG | idx) if idx < _BIG_OBJECT_TAG
                           else _U16.pack(_BIG_OBJECT_TAG) + _U32.pack(_BIG_CLASS_TAG | idx))
                    p = self._buf.find(tag, pos, limit)
                    while p != -1:
                        if self._is_next_element(p, False):
                            cands.append(p)
                            break
                        p = self._buf.find(tag, p + 1, limit)
        for p in sorted(cands):
            if p >= pos and self._is_next_element(p, is_last):
                return p
        return None

    # -- CDataStack -------------------------------------------------------------

    def _parse_data_stack(self, cur: _Cursor, index: int, info: _ClassInfo, start: int) -> DataStack:
        s = DataStack(index=index, class_name=info.name, schema=info.schema, offset=start,
                      path=self.path)
        self._read_base_header(cur, s)
        body = cur.pos
        error = None
        ok = False
        try:
            self._read_stack_header_v17(cur, s)
            self._read_data_array(cur, s)
            ok = self._locate_data_end(s)
            if not ok:
                error = "pixel block does not end at a property list"
        except MSRFormatError as exc:
            error = str(exc)
        if not ok:
            s.header = {"base_version": s.header.get("base_version")}
            s.timestamps, s.units = [], []
            if not self._scan_for_data_array(body, s):
                raise MSRFormatError(f"{info.name} (schema {info.schema}): no pixel block found ({error})")
            s.parse_mode = "scanned"
            self._warn(f"stack {index} ({s.source}): header layout not recognised ({error}); "
                       f"pixel block located by structure scan")
        cur.pos = s.data_end
        props, trailer = self._read_property_list(cur)
        s.properties = self._props_to_dict(props)
        s.header["settings_count"] = len(props)
        if s.dtype == "uint16" and s.sizes[0] >= 14:
            s.camera_stamp = _decode_pco_stamp(s.asarray().reshape(-1)[:14])
        return s

    def _read_stack_header_v17(self, cur: _Cursor, s: DataStack) -> None:
        """Header fields between the metadata string and the data array.

        Layout decoded from ImSpector Pro 7.6 files (CDataStack schema 17).
        Blocks that were zero in all samples are kept as raw bytes and reported
        only when non-zero.
        """
        h = s.header
        unknown: dict[str, Any] = {}

        def raw(name: str, n: int) -> None:
            b = cur.raw(n)
            if any(b):
                unknown[name] = b.hex(" ")

        h["version"] = cur.i32()
        raw("a", 34)
        h["preview_binning"] = [cur.i32(), cur.i32(), cur.i32()]  # e.g. 8,0,8 for 2048 px
        raw("b", 3)
        h["loop_counts"] = [cur.i32(), cur.i32(), cur.i32()]  # e.g. frames, 0, 1
        raw("c", 8)
        h["int_d"] = cur.i32()
        raw("e", 1)
        h["double_f"] = cur.f64()  # 4000.0 in all samples
        raw("g", 16)
        h["axis_ints_1"] = [cur.i32() for _ in range(8)]
        h["records"] = [cur.u64() for _ in range(4)]
        h["axis_ints_2"] = [cur.i32() for _ in range(4)]
        raw("h", 16)
        h["axis_names"] = [cur.cstring(4096) for _ in range(8)]
        raw("i", 32)
        h["axis_ints_3"] = [cur.i32() for _ in range(8)]
        raw("j", 36)
        h["int_k"] = cur.i32()  # -1
        raw("l", 16)
        s.units = [cur.cstring(64) for _ in range(4)]
        raw("m", 44)
        n = cur.i32()
        if not 0 <= n <= (self.size - cur.pos) // 4:
            raise MSRFormatError(f"implausible timestamp count {n}")
        s.timestamps = list(np.frombuffer(cur.raw(4 * n), "<f4").astype(float))
        raw("n", 8)
        n = cur.i32()
        if not 0 <= n <= 64:
            raise MSRFormatError(f"implausible vector count {n}")
        h["vectors"] = [[cur.f32(), cur.f32()] for _ in range(n)]
        if unknown:
            h["unknown_nonzero"] = unknown

    def _read_data_array(self, cur: _Cursor, s: DataStack) -> None:
        s.channel_id = cur.cstring(4096)
        s.header["array_version"] = cur.i32()
        s.header["array_type"] = cur.u16()
        s.sizes = [cur.i32() for _ in range(4)]
        s.lengths = [cur.f32() for _ in range(4)]
        s.labels = [cur.cstring(4096) for _ in range(4)]
        s.data_offset = cur.pos
        if not all(0 < v < (1 << 30) for v in s.sizes):
            raise MSRFormatError(f"implausible dimensions {s.sizes}")

    def _locate_data_end(self, s: DataStack) -> bool:
        """Infer bytes/pixel: the pixel block must be followed by a property list."""
        nvox = _prod(s.sizes)
        best = None
        for bpp in (2, 1, 4, 8):
            end = s.data_offset + nvox * bpp
            if end > self.size:
                continue
            score = self._property_list_score(end)
            if score == 2:
                best = (bpp, end)
                break
            if score == 1 and best is None:
                best = (bpp, end)
        if best is None:
            return False
        bpp, s.data_end = best
        if bpp == 4:
            s.dtype = self._guess_4byte_dtype(s)
        else:
            s.dtype = {1: "uint8", 2: "uint16", 8: "float64"}[bpp]
        return True

    def _guess_4byte_dtype(self, s: DataStack) -> str:
        sample = np.frombuffer(self._buf, "<u4", min(_prod(s.sizes), 1 << 16), s.data_offset)
        exponents = (sample >> 23) & 0xFF
        floaty = np.mean((exponents > 96) & (exponents < 160) | (sample == 0))
        dtype = "float32" if floaty > 0.99 else "uint32"
        self._warn(f"stack {s.index}: 4 bytes/pixel, guessed {dtype}")
        return dtype

    def _scan_for_data_array(self, start: int, s: DataStack) -> bool:
        """Fallback: locate 'channel id, dims, lengths, labels, pixels' by structure."""
        limit = min(self.size, start + (1 << 24))
        for m in _CHANNEL_ID_RE.finditer(self._buf, start, limit):
            if m.group(1)[0] != len(m.group(2)):
                continue
            cur = _Cursor(self._buf, m.start())
            try:
                self._read_data_array(cur, s)
                if not all(np.isfinite(s.lengths)) or not all(_printable(x) for x in s.labels):
                    continue
                if self._locate_data_end(s):
                    self._scan_timestamps(start, m.start(), s)
                    return True
            except MSRFormatError:
                continue
        return False

    def _scan_timestamps(self, lo: int, hi: int, s: DataStack) -> None:
        letters = s.dim_letters
        if "T" not in letters:
            return
        n = s.sizes[letters.index("T")]
        needle = _I32.pack(n)
        p = self._buf.find(needle, lo, hi)
        while p != -1 and p + 4 + 4 * n <= hi:
            with np.errstate(invalid="ignore"):
                ts = np.frombuffer(self._buf, "<f4", n, p + 4).astype(float)
            if np.all(np.isfinite(ts)) and np.all(np.diff(ts) >= 0) and ts[-1] > ts[0]:
                s.timestamps = list(ts)
                return
            p = self._buf.find(needle, p + 1, hi)

    # -- CImageStack ------------------------------------------------------------

    def _parse_image_view(self, cur: _Cursor, index: int, info: _ClassInfo, start: int,
                          is_last: bool) -> ImageView:
        v = ImageView(index=index, class_name=info.name, schema=info.schema, offset=start)
        self._read_base_header(cur, v)
        ok = False
        try:
            h = v.header
            h["version"] = cur.i32()
            h["ints_a"] = [cur.i32(), cur.i32(), cur.i32()]
            h["display_max"] = cur.f64()
            h["display_min"] = cur.f64()
            h["int_b"] = cur.i32()
            h["zoom"] = cur.f32()
            h["current_plane"] = cur.i32()
            h["ints_c"] = [cur.i32(), cur.i32()]
            h["word_d"] = cur.u16()
            while self._view_slot_ok(cur.pos):
                v.slots.append(self._read_view_slot(cur))
            ok = self._is_next_element(cur.pos, is_last)
        except MSRFormatError:
            ok = False
        if not ok:
            nxt = self._resync(start + 2, is_last)
            if nxt is None:
                raise MSRFormatError("cannot find the end of the CImageStack")
            if v.header.get("version") != 5:
                self._warn(f"view {index}: layout not recognised, skipped")
            cur.pos = nxt
        return v

    def _view_slot_ok(self, pos: int) -> bool:
        try:
            c = _Cursor(self._buf, pos)
            if not 0 < c.i32() < 100:
                return False
            if not all(_printable(c.cstring(1 << 16)) for _ in range(7)):
                return False
            return c.pos + 50 <= self.size
        except MSRFormatError:
            return False

    def _read_view_slot(self, cur: _Cursor) -> dict:
        slot: dict[str, Any] = {"base_version": cur.i32()}
        strings = [cur.cstring() for _ in range(7)]
        if any(strings[:6]):
            slot["strings"] = strings[:6]
        slot["meta"] = _parse_meta(strings[6])
        slot["ints"] = [cur.i32() for _ in range(4)]
        slot["lut_min"] = cur.f64()
        slot["lut_max"] = cur.f64()
        slot["byte_a"] = cur.u8()
        slot["mode"] = cur.u8()
        rgb = cur.raw(4)
        slot["color"] = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        slot["word_b"] = cur.u16()
        slot["gamma"] = cur.f32()
        tail = cur.raw(6)
        if any(tail):
            slot["tail"] = tail.hex(" ")
        return slot

    # -- output -----------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"{os.path.basename(self.path)}: format version {self.version}, "
                 f"{len(self.properties)} global settings, {len(self.stacks)} data stacks"]
        for s in self.stacks:
            px = s.pixel_size[0]
            dt = s.time_increment if "T" in s.axes else None
            lines.append(
                f"  #{s.index:<2} {s.source:<12} {s.channel_id:<18} {s.time:<9} "
                f"{s.axes:<5} {'x'.join(map(str, s.shape)):<16} {s.dtype:<8}"
                + (f" {px:.4g} {(s.units[0] if s.units else '').replace('µ', 'u')}/px" if px else "")
                + (f"  dt={dt:.4g}s" if dt else "")
                + ("" if s.parse_mode == "structured" else "  [scanned]"))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def stack_dict(s: DataStack) -> dict:
            return {
                "index": s.index, "class": s.class_name, "schema": s.schema,
                "source": s.source, "channel_id": s.channel_id, "time": s.time,
                "name": s.name, "title": s.title,
                "axes": s.axes, "shape": list(s.shape), "dtype": s.dtype,
                "sizes": s.sizes, "lengths": s.lengths, "labels": s.labels, "units": s.units,
                "pixel_size": list(s.pixel_size), "time_increment": s.time_increment,
                "timestamps": s.timestamps,
                "data_offset": s.data_offset, "data_bytes": s.nbytes,
                "parse_mode": s.parse_mode,
                "camera_stamp": s.camera_stamp,
                "display": ({"lut_min": s.view.lut[0], "lut_max": s.view.lut[1],
                             "zoom": s.view.header.get("zoom")}
                            if s.view is not None and s.view.lut else None),
                "meta": s.meta, "header": s.header, "settings": s.properties,
            }

        return {
            "file": os.path.abspath(self.path),
            "reader": f"msr_reader.py {__version__}",
            "format_version": self.version,
            "document": self.document,
            "stacks": [stack_dict(s) for s in self.stacks],
            "views": [{"index": v.index, "source": v.source, "time": v.time, "meta": v.meta,
                       "header": v.header, "slots": v.slots} for v in self.views],
            "global_settings": self.properties,
            "property_sets": self.property_sets,
            "setting_labels": self.labels,
            "warnings": self.warnings,
        }


# -----------------------------------------------------------------------------
# TIFF export
# -----------------------------------------------------------------------------

_OME_UNITS = {"µm", "um", "nm", "mm", "cm", "m", "pm", "Å"}


def _geometry(s: DataStack) -> tuple:
    return (s.time, tuple(s.sizes), tuple(round(float(x), 4) for x in s.lengths),
            tuple(s.labels), s.dtype)


def _detector(s: DataStack) -> str:
    """Channel id without the copy index, e.g. 'PMT[1]:0:4:0' -> 'PMT:0:4:0'."""
    return re.sub(r"\[\d+\]", "", s.channel_id or s.source)


def group_stacks(stacks: list[DataStack], mode: str = "auto") -> tuple[list[list[DataStack]], list[str]]:
    """Group stacks into multi-channel images.

    mode 'auto': merge only stacks that were acquired together -- same time
    stamp and geometry, different detectors, identical settings snapshot.
    'geometry': merge whenever time stamp and geometry match.  'none': never.
    Returns the groups and notes about stacks that looked similar but were
    kept apart.
    """
    groups: list[list[DataStack]] = []
    notes: list[str] = []
    for s in stacks:
        target = None
        if mode != "none" and "C" not in s.axes:
            for members in groups:
                first = members[0]
                if _geometry(first) != _geometry(s):
                    continue
                if mode == "auto":
                    if any(_detector(m) == _detector(s) for m in members):
                        notes.append(f"#{s.index} {s.channel_id} has the same geometry and time as "
                                     f"#{first.index} but the same detector -> separate file "
                                     f"(repeated acquisition or copy)")
                        continue
                    diff = [k for k in set(first.properties) | set(s.properties)
                            if first.properties.get(k) != s.properties.get(k)]
                    if diff:
                        notes.append(f"#{s.index} {s.channel_id} matches #{first.index} in geometry but "
                                     f"{len(diff)} settings differ (e.g. {', '.join(sorted(diff)[:3])}) "
                                     f"-> separate file")
                        continue
                target = members
                break
        if target is None:
            groups.append([s])
        else:
            target.append(s)
    return groups, notes


def _channel_name(s: DataStack) -> str:
    return s.channel_id or s.source or f"stack{s.index}"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "stack"


def _layout(stacks: list[DataStack], imagej: bool) -> tuple[str, list[int], list[dict]]:
    """Output axes (slowest first), shape and per-stack axis mapping."""
    first = stacks[0]
    own = [c for c in first.axes[:-2]]  # stack axes except Y, X
    sizes = dict(zip(first.axes, first.shape))
    rename = {c: c for c in own}
    if imagej:
        free = [c for c in "ZTC" if c not in own and not (c == "C" and len(stacks) > 1)]
        for c in own:
            if c not in "TZC":
                if not free:
                    raise ValueError(f"axis {c!r} cannot be stored in an ImageJ hyperstack; "
                                     "use OME-TIFF output")
                rename[c] = free.pop(0)
    out = {rename[c]: sizes[c] for c in own}
    if len(stacks) > 1:
        out["C"] = len(stacks)
    order = [c for c in "T" if c in out]
    order += [c for c in out if c not in "TZC"]  # modulo axes (OME) after T
    order += [c for c in "ZC" if c in out]
    axes = "".join(order) + "YX"
    shape = [out[c] for c in order] + [first.shape[-2], first.shape[-1]]
    inverse = {v: k for k, v in rename.items()}
    mapping = [{c: first.axes.index(inverse[c]) for c in order if c in inverse} for _ in stacks]
    return axes, shape, mapping


def _iter_planes(stacks: list[DataStack], axes: str, shape: list[int], mapping) -> Iterator[np.ndarray]:
    arrays = [s.asarray() for s in stacks]
    outer = axes[:-2]
    for idx in np.ndindex(*shape[:-2]):
        pos = dict(zip(outer, idx))
        k = pos.get("C", 0) if len(stacks) > 1 else 0
        sel = [0] * (arrays[k].ndim - 2)
        for letter, axis in mapping[k].items():
            if not (letter == "C" and len(stacks) > 1):
                sel[axis] = pos[letter]
        yield np.ascontiguousarray(arrays[k][tuple(sel)])


def _annotation(stacks: list[DataStack]) -> dict[str, str]:
    """Flat key/value metadata (first channel; other channels only where different)."""
    first = stacks[0]
    ann: dict[str, str] = {}
    for k, v in first.meta.items():
        ann[f"meta|{k}"] = str(v)
    for i, s in enumerate(stacks):
        ann[f"channel{i}|id"] = _channel_name(s)
        ann[f"channel{i}|source"] = s.source
    ann["stack|time"] = first.time
    ann["stack|labels"] = ", ".join(first.labels)
    ann["stack|lengths"] = ", ".join(f"{x:g}" for x in first.lengths)
    ann["stack|units"] = ", ".join(first.units)
    if first.camera_stamp:
        ann["camera|first_frame_stamp"] = first.camera_stamp["time"]
        ann["camera|image_counter"] = str(first.camera_stamp["image_counter"])
    for k, v in first.properties.items():
        ann[f"setting|{k}"] = str(v)
    for i, s in enumerate(stacks[1:], 1):
        for k, v in s.properties.items():
            if first.properties.get(k) != v:
                ann[f"setting[channel{i}]|{k}"] = str(v)
    return {_xml_safe(k): _xml_safe(v) for k, v in ann.items()}


def export_group(msr: MSRFile, stacks: list[DataStack], path: str, imagej: bool = False,
                 compression: str | None = None) -> dict:
    """Write one (possibly multi-channel) image to *path*."""
    import tifffile

    first = stacks[0]
    axes, shape, mapping = _layout(stacks, imagej)
    dtype = np.dtype(first.dtype)
    nbytes = _prod(shape) * dtype.itemsize
    bigtiff = nbytes > (1 << 32) - (1 << 26)
    px, py = first.pixel_size
    xunit = first.units[0] if first.units and first.units[0] else "µm"
    dt = first.time_increment
    names = [_xml_safe(_channel_name(s)) for s in stacks]
    kwargs: dict[str, Any] = dict(photometric="minisblack", bigtiff=bigtiff, dtype=dtype, shape=tuple(shape))
    if compression:
        kwargs["compression"] = compression

    if imagej:
        info = "\n".join(f"{k} = {v}" for k, v in _annotation(stacks).items())
        md: dict[str, Any] = {"axes": axes, "Info": info}
        if xunit in _OME_UNITS:
            md["unit"] = "um" if xunit in ("µm", "um") else xunit
        if "Z" in axes and first.step("Z"):
            md["spacing"] = first.step("Z")
        if "T" in axes and dt:
            md["finterval"] = dt
        if "C" in axes:
            md["mode"] = "composite"
        luts = [s.view.lut for s in stacks if s.view is not None and s.view.lut]
        if len(luts) == len(stacks) and all(hi > lo for lo, hi in luts):
            md["Ranges"] = tuple(v for lut in luts for v in lut)
        if px and py:
            kwargs["resolution"] = (1.0 / px, 1.0 / py)
        if bigtiff:
            msr._warn(f"{os.path.basename(path)}: > 4 GB, ImageJ cannot open BigTIFF hyperstacks "
                      "(Fiji's Bio-Formats importer can)")
        tifffile.imwrite(_fs_path(path), _iter_planes(stacks, axes, shape, mapping), imagej=True,
                         metadata=md, **kwargs)
    else:
        md = {"axes": axes, "Name": _xml_safe(f"{os.path.basename(msr.path)} #{first.index} {first.source} {first.time}")}
        if px and py:
            unit = xunit if xunit in _OME_UNITS else "µm"
            unit = "µm" if unit == "um" else unit
            md.update(PhysicalSizeX=px, PhysicalSizeXUnit=unit, PhysicalSizeY=py, PhysicalSizeYUnit=unit)
        if "Z" in axes and first.step("Z"):
            zunit = first.axis_info("Z")["unit"] or "µm"
            md.update(PhysicalSizeZ=first.step("Z"),
                      PhysicalSizeZUnit="µm" if zunit in ("um", "µm") or zunit not in _OME_UNITS else zunit)
        if "T" in axes and dt:
            md.update(TimeIncrement=dt, TimeIncrementUnit="s")
            ts = first.timestamps
            if len(ts) == shape[axes.index("T")] and axes.index("T") == 0:
                per_t = _prod(shape[1:-2])
                deltas = [float(t) for t in ts for _ in range(per_t)]
                md["Plane"] = {"DeltaT": deltas, "DeltaTUnit": ["s"] * len(deltas)}
        md["Channel"] = {"Name": names if "C" in axes else names[:1]}
        md["MapAnnotation"] = _annotation(stacks)
        tifffile.imwrite(_fs_path(path), _iter_planes(stacks, axes, shape, mapping), ome=True,
                         metadata=md, **kwargs)
    return {"file": os.path.basename(path), "stacks": [s.index for s in stacks], "channels": names,
            "axes": axes, "shape": shape, "dtype": dtype.name,
            "pixel_size": [px, py], "time_increment": dt if "T" in axes else None}


def write_timestamps(stack: DataStack, path: str) -> int:
    """Write the per-frame acquisition times of *stack* to a text file, one per line.

    Values are seconds exactly as stored by ImSpector (not shifted to start at 0),
    printed with the shortest decimal form that reproduces the stored float32.
    """
    with open(_fs_path(path), "w", encoding="ascii", newline="\n") as fh:
        for t in stack.timestamps:
            fh.write(np.format_float_positional(np.float32(t), unique=True, trim="-") + "\n")
    return len(stack.timestamps)


def export(path: str, outdir: str | None = None, imagej: bool = False, group: str = "auto",
           compression: str | None = None, log=print, timestamps_only: bool = False) -> list[dict]:
    """Convert one .msr file; returns a description of the written files.

    Stacks with per-frame times also get a '<name>_timestamps.txt'. With
    *timestamps_only* only those text files are written.
    """
    with MSRFile(path) as msr:
        stem = os.path.splitext(os.path.basename(path))[0]
        folder = os.path.join(outdir or os.path.dirname(os.path.abspath(path)), f"{stem}_tiff")
        if not timestamps_only:
            os.makedirs(_fs_path(folder), exist_ok=True)
        log(msr.summary())
        groups, notes = group_stacks(msr.stacks, group)
        for note in notes:
            log(f"  note: {note}")
        ext = ".tif" if imagej else ".ome.tif"
        written = []
        for gi, members in enumerate(groups, 1):
            s = members[0]
            if len(members) > 1:
                parts = [f"S{gi:02d}", _safe(s.source), _safe(s.time.replace(":", "")), f"{len(members)}ch"]
            else:
                parts = [f"S{gi:02d}", _safe(s.channel_id.split(":")[0] or s.source),
                         _safe(s.time.replace(":", ""))]
            base = os.path.join(folder, "_".join(parts))
            rec = None
            if not timestamps_only:
                out = base + ext
                try:
                    rec = export_group(msr, members, out, imagej=imagej, compression=compression)
                except (ValueError, OSError) as exc:
                    msr._warn(f"could not write {os.path.basename(out)}: {exc}")
                    log(f"  ERROR {os.path.basename(out)}: {exc}")
                    continue
                written.append(rec)
                log(f"  -> {rec['file']}  axes={rec['axes']} shape={'x'.join(map(str, rec['shape']))} "
                    f"channels={', '.join(rec['channels'])}")
            if s.timestamps:
                os.makedirs(_fs_path(folder), exist_ok=True)
                ts_file = base + "_timestamps.txt"
                n = write_timestamps(s, ts_file)
                log(f"  -> {os.path.basename(ts_file)}  ({n} frame times in s)")
                if rec is not None:
                    rec["timestamps_file"] = os.path.basename(ts_file)
                else:
                    written.append({"file": os.path.basename(ts_file), "stacks": [m.index for m in members],
                                    "frames": n})
        if timestamps_only:
            if not written:
                log("  (no stack in this file has per-frame times)")
            return written
        meta = msr.to_dict()
        meta["exports"] = written
        meta["grouping_notes"] = notes
        with open(_fs_path(os.path.join(folder, "metadata.json")), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False, default=_json_default)
        log(f"  -> metadata.json ({len(msr.properties)} global settings, "
            f"{sum(len(s.properties) for s in msr.stacks)} per-stack settings)")
        return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, bytes):
        return o.hex(" ")
    raise TypeError(f"not JSON serializable: {type(o)}")


def _collect(inputs: list[str]) -> list[str]:
    files = []
    for p in inputs:
        if os.path.isdir(_fs_path(p)):
            files += sorted(os.path.join(p, f) for f in os.listdir(_fs_path(p)) if f.lower().endswith(".msr"))
        else:
            files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Export LaVision BioTec ImSpector .msr files to OME-TIFF / ImageJ TIFF.")
    ap.add_argument("inputs", nargs="+", help=".msr files or folders containing .msr files")
    ap.add_argument("-o", "--outdir", help="parent folder for the output (default: next to each .msr); "
                    "each file gets a '<name>_tiff' subfolder")
    ap.add_argument("--imagej", action="store_true",
                    help="write ImageJ hyperstack TIFFs instead of OME-TIFF")
    ap.add_argument("--group", choices=["auto", "geometry", "none"], default="auto",
                    help="merging of stacks into multi-channel files: 'auto' (default) merges only "
                    "channels acquired together (same time, geometry and settings, different "
                    "detectors); 'geometry' merges any stacks with equal time stamp and geometry; "
                    "'none' writes one file per stack")
    ap.add_argument("--compress", choices=["zlib", "lzw", "zstd"], default=None,
                    help="lossless compression (default: none)")
    ap.add_argument("--timestamps", action="store_true",
                    help="only write the per-frame times as text (one value in s per line), no TIFFs")
    ap.add_argument("--info", action="store_true", help="only print the file contents")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    files = _collect(args.inputs)
    if not files:
        print("no .msr files found", file=sys.stderr)
        return 1
    failed = 0
    for f in files:
        try:
            if args.info:
                with MSRFile(f) as msr:
                    print(msr.summary())
            else:
                export(f, args.outdir, imagej=args.imagej, group=args.group,
                       compression=args.compress, timestamps_only=args.timestamps)
        except (MSRFormatError, OSError) as exc:
            failed += 1
            print(f"{f}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
