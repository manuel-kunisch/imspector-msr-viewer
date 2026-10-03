# -*- mode: python ; coding: utf-8 -*-
#
# macOS (Apple Silicon) PyInstaller spec for the MSR Viewer, used by build_macos.sh.
#
# Produces a native arm64 "MSR Viewer.app"; build_macos.sh packs it into a .dmg.
# Same contents as msr_viewer.spec (Windows), plus the .app bundle: .icns icon,
# version and file types in Info.plist, and ffmpeg collected as a binary so it
# stays executable and is signed (ad hoc) with the rest of the app.

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

project_root = Path(SPECPATH).resolve()

# bundle version = __version__ in msr_viewer.py (as in the About box)
_m = re.search(r'^__version__ = "([^"]+)"', (project_root / "msr_viewer.py").read_text(encoding="utf-8"), re.M)
app_version = _m.group(1) if _m else "0.0.0"

# app icon, made by build_macos.sh from msr_viewer.png (or msr_viewer.ico)
_icns = project_root / "msr_viewer.icns"
icon_arg = str(_icns) if _icns.exists() else None

datas = []
datas += collect_data_files("pyqtgraph")
# window icon, loaded next to msr_viewer.py
datas += [(str(project_root / "msr_viewer.ico"), ".")]

# ffmpeg for the video export; imageio_ffmpeg looks for it in its own package folder
binaries = [(src, dest) for src, dest in collect_data_files("imageio_ffmpeg", subdir="binaries")
            if Path(src).name.startswith("ffmpeg")]

excludes = [
    "tkinter",
    "PySide2",
    "PySide6",
    "PyQt6",
    "IPython",
    "pytest",
]

a = Analysis(
    ["msr_viewer.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # imported on demand from the Tools menu
        "msr_psf",
        "msr_linescan",
    ],
    hookspath=[],
    # the line scan plot export draws with Agg and saves PNG / PDF / SVG; no interactive backend
    hooksconfig={"matplotlib": {"backends": ["Agg", "PDF", "SVG"]}},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MSR_Viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,       # files from the Finder / Dock reach Qt as QFileOpenEvent
    target_arch="arm64",        # native Apple Silicon build
    codesign_identity=None,     # "Developer ID Application: ..." to sign; otherwise signed ad hoc
    entitlements_file=None,
    icon=icon_arg,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MSR_Viewer",
)

app = BUNDLE(
    coll,
    name="MSR Viewer.app",
    icon=icon_arg,
    bundle_identifier="io.github.manuel-kunisch.msrviewer",
    info_plist={
        "CFBundleName": "MSR Viewer",
        "CFBundleDisplayName": "MSR Viewer",
        "CFBundleShortVersionString": app_version,
        "CFBundleVersion": app_version,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",  # first macOS for Apple Silicon
        # Open With / drop onto the Dock icon: .msr (own type) and TIFF (as an alternative viewer)
        "CFBundleDocumentTypes": [
            {"CFBundleTypeName": "ImSpector measurement", "CFBundleTypeRole": "Viewer",
             "LSHandlerRank": "Default", "CFBundleTypeExtensions": ["msr"]},
            {"CFBundleTypeName": "TIFF image", "CFBundleTypeRole": "Viewer",
             "LSHandlerRank": "Alternate", "LSItemContentTypes": ["public.tiff"]},
        ],
    },
)
