# imspector-msr-viewer

Viewer and TIFF exporter for LaVision BioTec **ImSpector** `.msr` files (ImSpector Pro 7.x, TriM Scope).
Bio-Formats rejects these files ("Unknown file format"), so this reads them directly. No ImSpector needed.

> **v0.3.1, in development.** Tested so far with ImSpector Pro 7.6.4 files (camera, PMT and resonant
> scanner timelapse). Other acquisition types may not work yet.

## Features

- Drag & drop `.msr` files or folders into the viewer
- Overview of all stacks in the measurement workspace
- Zoom/pan, scale bar, LUTs, histogram/contrast, frame slider
- Timelapse playback in real time (the recorded frame intervals, × speed) or at a fixed frame rate
- All metadata: pixel size, frame timestamps, objective and every ImSpector setting (filter, compare two stacks)
- Export to OME-TIFF or ImageJ TIFF with physical pixel size and frame times, plus a `metadata.json`
- Frame times as plain text, one value (s) per line
- MP4 video of a timelapse or z sweep: real acquisition timing (× speed, e.g. 0.25 = four times slower)
  or a fixed frame rate, with optional scale bar and time stamp
- Pixel data is exported bit-exact, no rescaling
- PSF analysis on demand (Tools ▸ PSF analysis): click a bead, get lateral x/y and (for z-stacks) axial
  widths at FWHM, 1/e or 1/e², from the threshold crossings and a Gaussian fit
- Line scan / correlation on demand (Tools ▸ Line scan): pick stacks (also from different files), compare
  their profiles along a line, Pearson r along the line and over the image, optional sub-pixel alignment;
  per frame or as mean / max projection; plot (PNG/PDF/SVG), CSV and image export

## Install

Python ≥ 3.10 with numpy, tifffile and PyQt5, for example:

    conda create -n msr python=3.12 numpy tifffile pyqt

Tested with numpy 1.26–2.5, tifffile 2023.4–2026.6 and PyQt5 5.15.
Video export needs ffmpeg: `pip install imageio-ffmpeg` (bundles it) or any ffmpeg on PATH.
The PSF analysis and the line scan need pyqtgraph; scipy, scikit-image and matplotlib are optional
(Gaussian fit, spline sampling, sub-pixel alignment, plot export).

## Windows exe

    powershell -ExecutionPolicy Bypass -File build_windows.ps1

builds `dist\MSR_Viewer_v<version>\MSR_Viewer.exe` and a zip of that folder (PyInstaller, one-folder build, no
Python needed on the target PC). The script creates its own `.venv-build` from Python 3.12 (py launcher) with the
packages from requirements.txt, so the exe only contains what the viewer imports: about 320 MB unpacked, mostly
ffmpeg, Qt and scipy. `-SkipInstall` reuses the venv as it is, `-NoZip` skips the zip. Drop .msr files onto the
exe or use "Open with".

## Usage

Viewer:

    python msr_viewer.py [file.msr ...]

On Windows you can also drop files onto `MSR_Viewer.bat`; adjust its `ENV` line to your conda env.
Wheel = zoom, drag = pan, double-click = fit, ←/→ = frame, Space = play.
Next to the play button you choose *real time* (with a speed factor) or *fixed fps*; File ▸ Export video
writes the same as MP4. In real time each frame is held for its recorded interval; the file itself has a
constant frame rate (60 fps by default), so it plays everywhere and the timing is exact to one video frame.

PSF analysis opens as an extra tab. A click on a bead snaps the cross to its centre (brightest pixel nearby,
refined by the intensity centroid); drag the white circle to fine-tune. The profiles along both arms are
sampled with cubic splines (bilinear sampling between pixel centres would broaden a PSF by ~2 % at σ = 2 px),
optionally averaged over parallel lines and rotated. Widths are measured above a baseline (profile ends,
a movable background box, or zero); for z-stacks the axial profile is the mean of a small box in every slice.
Results can be copied or saved as CSV / PNG.

Line scan / correlation first asks which stacks to compare (same X × Y size), then opens its own window: a
composite (or one stack at a time) with a line whose ends and centre can be dragged, the profiles of all
stacks along it (optionally averaged over a band and normalized), and Pearson r for every pair. *Align to
first stack* registers each stack to the first one by cross-correlation of lightly smoothed, Hann-windowed
images with sub-pixel refinement (within 0.02 px on synthetic shifts; plain phase correlation without a
window can be off by pixels because of the image edges). Names and colours are editable in the table.

Export without GUI:

    python msr_reader.py file.msr                # OME-TIFFs, frame times + metadata.json into file_tiff/
    python msr_reader.py folder --imagej         # every .msr in a folder, as ImageJ hyperstacks
    python msr_reader.py file.msr --timestamps   # only the frame times as text, one per line
    python msr_reader.py file.msr --info         # only list what is inside

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
