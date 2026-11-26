# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the MSR Viewer (one-folder build), used by build_windows.ps1.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path(SPECPATH).resolve()

datas = []
datas += collect_data_files("pyqtgraph")
# ffmpeg for the video export; imageio_ffmpeg looks for it in its own package folder
datas += collect_data_files("imageio_ffmpeg", subdir="binaries")
# window / taskbar icon, loaded next to msr_viewer.py (the exe icon is set in EXE below)
datas += [(str(project_root / "msr_viewer.ico"), ".")]
# license of 3Dscript, whose renderer the 3D view ports (BSD 2-Clause: binaries carry the notice);
# Help > Third-party notices shows it (msr_viewer._resource finds it next to the modules)
datas += [(str(project_root / "THIRD_PARTY_NOTICES.md"), ".")]

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
    binaries=[],
    datas=datas,
    hiddenimports=[
        # imported on demand from the Tools menu
        "msr_psf",
        "msr_linescan",
        # the 3D view, imported when it is first switched on
        "msr_volume",
        "msr_volume_view",
        "msr_volume_panel",
        # GL function tables that QOpenGLContext.versionFunctions() imports at run time
        "PyQt5._QOpenGLFunctions_2_0",
        "PyQt5._QOpenGLFunctions_2_1",
        "PyQt5._QOpenGLFunctions_4_1_Core",
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
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "msr_viewer.ico"),
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
