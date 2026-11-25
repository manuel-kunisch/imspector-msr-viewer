# LaVision BioTec ImSpector `.msr` — file format notes

Reverse-engineered from `reso_scan_ao25_um_oscillation_4hz_5V_closed_loop_limit_compensation_timelapse.msr`
and `mosaic_10_perc_overlap_403um_fov.msr` (compressed, 3 × 3 mosaic of z-stacks) (ImSpector Pro 7.6.4,
TriM Scope). Implemented in `msr_reader.py` (file) and `msr_mosaic.py` (mosaics).

## 0. Compressed files

ImSpector can save a workspace compressed. Such a file is a 12-byte header and one zlib stream that holds
the archive described below (byte for byte what an uncompressed file contains):

```
char[3]  "spc"
uint8    1                     container version
uint32   uncompressed size     of the archive (113 894 005 in the sample)
uint32   Adler-32              of the compressed bytes that follow (not of the archive)
zlib stream                    78 9C ..., ends with the Adler-32 of the archive (big-endian, as zlib does)
```

`msr_reader.py` inflates the stream once into an anonymous temporary file (removed by the OS when the
last handle or memory map of it closes, also after a crash) and memory-maps the pixels from there. The
sample (85 MB → 114 MB) opens in about a second. Bio-Formats and the previous reader rejected these files,
since they start with `spc` instead of the archive's version number.

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
CString×8 "None"                   loops the measurement was split by ("xyz-Table X", "xyz-Table Y", ...)
int32×8  0                         index of this stack in each loop (mosaic tile: column, row)
int32×8  1                         length of each loop (3, 3 for a 3 × 3 mosaic)
36 B     0                         mosaic sample: float32 4.0 in the first 4 bytes of the first tile only
int32    -1
16 B     0                         mosaic sample: int32 1 in the last 4 bytes of every tile
CString×4 units                    "µm", "µm", "s", ""
44 B     0
int32 n, float32[n]                per-frame acquisition time in s (timelapse: 250 values, 16.41 ms apart)
8 B      0
int32 n, (float32,float32)[n]      n = 3; entries 1 and 2 = offset of the four dimensions (X, Y; dim3,
                                   dim4), as ImSpector's TIFF export writes them (Offset_0..3): the stage
                                   position of a mosaic tile in µm, -FOV/2 for the camera, the z start

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

The blocks marked `0` were zero in all stacks of the first sample. Their meaning is unknown; the reader reports
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

## 8. Mosaics

A mosaic ("Measurement Mode" `2D Mosaic` in the metadata) is stored as one `CDataStack` per tile, named
`PMT - xyz-Table[row] - xyz-Table[column]` (outer loop first), with channel ids `PMT:0:4:0` … `PMT:0:4:8`
(last field: running tile number). Per tile the header gives the loop indices (`split_index`, see §4),
the loop lengths and the stage position (offset vector); the snapshot settings repeat the mosaic setup:
`xyz-Table XRes/YRes` (tiles), `XWidth/YWidth` (tile size, µm), `XOvl/YOvl` (overlap, µm), `XYOvl` (%),
`XPosMin/XPosMax`, `YPosMin/YPosMax` (stage positions of the first and last tile; `YLen` negative: rows
step towards −Y).

In the sample (3 × 3 tiles of 512 × 512 × 22, 0.7886 µm/px, 10 % overlap) the tiles fit together with stage
+X to the right and stage +Y up (row 0 at the top): of the eight ways to map the stage axes onto the image,
only this one makes all twelve overlaps correlate. The actual overlaps differ from the stage positions by up
to 25 px (≈ 20 µm): the scan field is rotated by ≈ 1.4° against the stage axes (every column step also
moves ≈ 10.5 px down, every row step ≈ 12 px left), and the step from column 1 to 2 is ≈ 30 px shorter
than the one from column 0 to 1. `msr_mosaic.py` registers the overlaps to correct this.

## 9. Open questions (need more sample files)

* Files with several simultaneous PMT channels, FLIM/TCSPC, other data types.
* Other values of the `spc` container version; files above 4 GB (the size field is 32 bits).
* Mosaics with several detectors, timelapse mosaics, mosaics that use `xyz-Table FlipXAxis / FlipYAxis`.
* Meaning of the `:a:b:c` parts of the channel id, the `uint64×4` block, and the time-axis length.
