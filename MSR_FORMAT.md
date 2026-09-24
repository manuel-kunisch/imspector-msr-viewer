# LaVision BioTec ImSpector `.msr` — file format notes

Reverse-engineered from `reso_scan_ao25_um_oscillation_4hz_5V_closed_loop_limit_compensation_timelapse.msr`
(ImSpector Pro 7.6.4, TriM Scope). Implemented in `msr_reader.py`.

## 1. Container: MFC `CArchive`

An `.msr` file is the ImSpector document (the *measurement workspace*, e.g. "Measurement 1")
serialized with Microsoft MFC `CArchive`. Everything is little-endian. There are no offsets or
markers; the file must be read sequentially, and the parser must follow MFC's rules:

| Element | Encoding |
|---|---|
| Object tag, new class | `FFFF`, `uint16 schema`, `uint16 len`, ASCII class name, then the object |
| Object tag, known class | `uint16 0x8000 \| class_index` (or `7FFF` + `uint32 0x80000000 \| index` when index ≥ 0x7FFF) |
| Index bookkeeping | classes **and** objects share one counter that starts at 1. A new class takes the next index, and the object after it takes the one after that. Every object read (including every `CProperty`) must be counted, or later class references resolve to the wrong class. |
| `CString` | `uint8 n`; if `FF`: `uint16 n`; if `FFFF`: `uint32 n`. The bytes `FF FE FF` (`uint8 FF`, `uint16 FFFE`) mark a UTF-16 string, and its length follows. Text is ANSI (cp1252, e.g. `B5 6D` = "µm"). |
| Count (`WriteCount`) | `uint16`; if `FFFF`: `uint32` (if `FFFFFFFF`: `uint64`) |

In this file: `CProperty` = class #1, `CDataStack` = #573, `CImageStack` = #918. That's why the
stack tags are `3D 82` and `96 83`.

## 2. Top-level layout

```
int32                  file version (8)
property list          global settings: count, count × CProperty, int32 1   (571 entries)
CString ""             document fields
CString "not changed"
10 bytes (zero)
stack array            count (8), then objects: CDataStack, CImageStack, CDataStack, CImageStack, ...
CChildFrame array      count (3), window layout of the workspace (not parsed)
CPropArray             workspace property set:
                         list 1: propset_label "Measurement 1", seq_autosave_prefix, propset_id
                         list 2: the settings of the set (1330 entries)
                         12 trailing bytes
```

A *property list* is always `count`, then `count` × `CProperty`, then `int32` (= 1).

## 3. `CProperty` (schema 1): one setting

```
CString key        "PMT NameCH5"
int32   type       1 double, 2 int32, 3 float32, 4 int32 (ms times), 5 CString
CString label      "PMT Name Channel5"
uint16  group      device group (e.g. 9 = PMT, 2 = TriMScope, 11 = ResonantPMT)
uint16  flags      4 = global list, 8 = stack snapshot / property set (0x18 seen once)
value              according to type
```

Quirk: some string values contain UTF-16 inside an 8-bit CString, e.g. `TriMScope ActiveLasers` =
`L\0a\0s\0e\0r\0,\0`. Decode those as UTF-16, otherwise they put NUL bytes into XML.

## 4. `CDataStack` (schema 17): one detector image with its pixels

```
-- base header (also used by CImageStack and by the CImageStack layers) --
int32    4
CString  name            measurement name
CString  time            "22:54:13"
CString  source          "Camera" | "PMT" | "ResonantPMT" | ...
CString  "" × 2
CString  title           "<name> - <time> - <source> "
CString  meta            "First Name::…::Creation Date::…::Instrument Mode::…::ObjectiveID::…::"

-- stack header --
int32    17              version (= schema)
34 B     0
int32×3  8,0,8 / 2,0,2 / 1,0,1     = size/256, probably preview binning
3 B      0
int32×3  250,0,1                   loop counts (frames)
8 B      0
int32    1
1 B      0
double   4000.0
16 B     0
int32×8  1
uint64×4 ?
int32×4  1
16 B     0
CString×8 "None"                   axis names of the scan setup
32 B     0
int32×8  1
36 B     0
int32    -1
16 B     0
CString×4 units                    "µm", "µm", "s", ""
44 B     0
int32 n, float32[n]                per-frame acquisition time in s (timelapse: 250 values, 16.41 ms apart)
8 B      0
int32 n, (float32,float32)[n]      n = 3; entry 1 = XY offset in µm (camera: -FOV/2; meaning unconfirmed)

-- data array --
CString  channel id      "PMT[1]:0:4:0"  (device[copy]:?:detector:?)
int32    1
uint16   1               data type? (only uint16 seen)
int32×4  sizes           X, Y, dim3, dim4 (X varies fastest)
float32×4 lengths        physical extent per dim (µm; the time extent is not the duration)
CString×4 labels         "ResonantPMT X", "ResonantPMT Y", "Time T", "None"
pixels                   prod(sizes) × uint16 little-endian, C order [dim4][dim3][Y][X]

-- settings snapshot at acquisition time --
property list            (343 … 1670 CProperty entries)
```

**Self-check.** The pixel block must end exactly where a property list starts (a `count` followed by a
`CProperty` tag). `msr_reader.py` uses this to confirm the header parse and the bytes per pixel. If the
header layout differs (other ImSpector versions), it falls back to finding the data array by
structure (channel id, then plausible sizes, lengths and labels, then this self-check).

The blocks marked `0` were zero in all four stacks. Their meaning is unknown; the reader reports
them as `unknown_nonzero` if they are ever non-zero.

Camera stacks from PCO cameras contain the camera's BCD timestamp in the first 14 pixels of every frame
(image counter, date, time with µs). It is real camera data and is decoded into `camera_stamp`.

## 5. `CImageStack` (schema 5): display view of the preceding `CDataStack`

```
base header (as above)
int32    5
int32×3
double   display max, double display min
int32; float32 zoom; int32 current plane
int32×2; uint16
2 × layer:
    base header (int32 4, 6 empty CStrings, meta)
    int32×4
    double lut_min, double lut_max          (e.g. 544 / 30810; second layer 0 / 65535)
    uint8, uint8 mode, COLORREF, uint16, float32 gamma (1.0), 6 B
```

No pixel data. The LUT range is reused as the ImageJ display range.

## 6. Semantics seen in the sample

* The stack array holds **everything in the workspace**: acquisitions and copies. `PMT` and `PMT[1]`
  have the same time and geometry but the same detector (`:0:4:0`), different pixels, and different
  settings (OPO power 2.45 → 2.75). They are two acquisitions, not two channels. The exporter
  therefore merges stacks into one multi-channel image only when time, geometry and the whole
  settings snapshot match and the detectors differ.
* Different sources (Camera, PMT, ResonantPMT) are separate stacks with their own geometry.

## 7. Why Bio-Formats (8.4.0) fails

`ImspectorReader.isThisType()` requires the string `CDataStack` within the **first 32 bytes**. ImSpector
Pro 7.x files begin with the global settings list (`CProperty`), and `CDataStack` first appears at
byte 36138, so the file is rejected with `UnknownFormatException`. Its parser also relies on fixed
markers (e.g. `0x8003`, an MFC reference to "class #3", which presumably matched older files that start
directly with the stack) instead of MFC's index bookkeeping, so it would not parse this layout even if
detection were forced.

## 8. Open questions (need more sample files)

* Files with Z-stacks, several simultaneous PMT channels, FLIM/TCSPC, mosaics, other data types.
* Meaning of the `:a:b:c` parts of the channel id, the `uint64×4` block, and the time-axis length.
