# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ASWIFT Streamlit viewer desktop bundle.

Set DOTNET_RUNTIME_DIR to an app-local .NET runtime directory when you want the
release artifact to work with PalmSens .pssession files without a separate
Microsoft .NET Runtime install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


project_root = Path(SPECPATH).parents[1]
launcher = project_root / "packaging" / "desktop" / "aswift_viewer_launcher.py"
viewer = project_root / "src" / "aswift" / "analysis" / "structured_results_viewer.py"

datas = [
    (str(viewer), "streamlit_app"),
    *collect_data_files("streamlit"),
    *collect_data_files("pypalmsens"),
    *copy_metadata("aswift"),
    *copy_metadata("streamlit"),
    *copy_metadata("pypalmsens"),
]

dotnet_runtime_dir = os.environ.get("DOTNET_RUNTIME_DIR")
if dotnet_runtime_dir:
    datas.append((dotnet_runtime_dir, "dotnet-runtime"))

hiddenimports = [
    *collect_submodules("aswift"),
    *collect_submodules("streamlit"),
    *collect_submodules("pypalmsens"),
]

a = Analysis(
    [str(launcher)],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="ASWIFT Viewer",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="ASWIFT Viewer",
    )
    app = BUNDLE(
        coll,
        name="ASWIFT Viewer.app",
        icon=None,
        bundle_identifier="edu.stanford.sohlab.aswift-viewer",
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="ASWIFT Viewer",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="ASWIFT Viewer",
    )
