# imspector-msr-viewer

Viewer and TIFF exporter for LaVision BioTec **ImSpector** `.msr` files (ImSpector Pro 7.x, TriM Scope).
Bio-Formats rejects these files ("Unknown file format"), so this reads them directly. No ImSpector needed.

> **v0.2, in development.** Tested so far with ImSpector Pro 7.6.4 files (camera, PMT and resonant
> scanner timelapse). Other acquisition types may not work yet.

## Features

- Drag & drop `.msr` files or folders into the viewer
- Overview of all stacks in the measurement workspace
- Zoom/pan, scale bar, LUTs, histogram/contrast, frame slider and playback with the real frame times
- All metadata: pixel size, frame timestamps, objective and every ImSpector setting (filter, compare two stacks)
- Export to OME-TIFF or ImageJ TIFF with physical pixel size and frame times, plus a `metadata.json`
- Pixel data is exported bit-exact, no rescaling

## Install

Python ≥ 3.10 with numpy, tifffile and PyQt5, for example:

    conda create -n msr python=3.12 numpy tifffile pyqt

Tested with numpy 1.26–2.5, tifffile 2023.4–2026.6 and PyQt5 5.15.

## Usage

Viewer:

    python msr_viewer.py [file.msr ...]

On Windows you can also drop files onto `MSR_Viewer.bat`; adjust its `ENV` line to your conda env.
Wheel = zoom, drag = pan, double-click = fit, ←/→ = frame, Space = play.

Export without GUI:

    python msr_reader.py file.msr            # OME-TIFFs + metadata.json into file_tiff/
    python msr_reader.py folder --imagej     # every .msr in a folder, as ImageJ hyperstacks
    python msr_reader.py file.msr --info     # only list what is inside

From Python:

```python
from msr_reader import MSRFile

with MSRFile("file.msr") as msr:
    for s in msr.stacks:
        print(s.source, s.axes, s.shape, s.pixel_size)
        data = s.asarray()   # numpy memmap, nothing is loaded until you index it
```

## How the file is read

An `.msr` file is the ImSpector workspace (e.g. "Measurement 1") written with Microsoft's MFC `CArchive`
serialization. It is one continuous little-endian object stream, with no offset table and no markers, so it
has to be read front to back, object by object:

- Every object starts with a tag. The first time a class appears, the tag is `FF FF` followed by the schema
  and the class name (e.g. `CDataStack`). After that it is `0x8000 | class index` (e.g. `3D 82`).
- Classes and objects share one running index. Every object has to be counted (each single setting is an
  object), otherwise later tags point to the wrong class.
- Strings are length-prefixed MFC `CString`s, and lists are `count + items`.
- A setting (`CProperty`) is `key, type, label, group, flags, value`. The type says how the value is
  stored: 1 = double, 2/4 = int32, 3 = float, 5 = string.

The stream is divided like this:

```
int32              file version (8)
settings list      count + CProperty × n          global instrument settings (571 here)
workspace fields
stack array        count, then per stack:
  CDataStack         the acquired data        <- pixels live here
  CImageStack        how ImSpector displayed it (LUT, zoom), no pixels
CChildFrame array  window layout
CPropArray         property set of the workspace
```

Inside a `CDataStack`:

```
base header   name, time, source ("Camera", "PMT", "ResonantPMT"), "Key::Value::" metadata
stack header  units, per-frame timestamps (float32), offsets, ...
data array    channel id   "PMT[1]:0:4:0"
              sizes[4]     X, Y, dim3, dim4          e.g. 256, 256, 250, 1
              lengths[4]   physical extent (µm, s)
              labels[4]    "ResonantPMT X", "ResonantPMT Y", "Time T", "None"
              pixels       X·Y·dim3·dim4 × uint16, little-endian, X fastest
settings      count + CProperty × n        snapshot of all settings at acquisition time
```

The pixels are one raw block right after the last axis label, stored in the order `[dim4][dim3][Y][X]`,
so a timelapse is simply 250 consecutive 256×256 frames. There is no marker around the block; it is
separated from the rest purely by its length (`X·Y·dim3·dim4 × bytes per pixel`). As a check, the block
has to end exactly where the stack's settings list starts (a count followed by a `CProperty` tag). The same
check also gives the bytes per pixel. If a header variant is not recognised, the reader locates the data
array by its structure instead, and the same check applies.

Stacks are merged into one multi-channel TIFF only if they were acquired together: same time,
geometry and settings, but different detectors. Copies or repeats inside a workspace (like `PMT[1]`)
stay separate files.

The full byte layout, open questions and why Bio-Formats fails are in [MSR_FORMAT.md](MSR_FORMAT.md).

## Known limitations

- Reverse engineered from one file. Z-stacks, several PMT channels recorded at the same time, FLIM/TCSPC
  and mosaics are not tested yet.
- Only uint16 data seen so far. 8/32-bit data is handled but untested.
- Some header fields are still unknown (they were zero in all samples). The reader reports them if they
  are ever non-zero.
