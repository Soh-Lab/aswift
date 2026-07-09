# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ASWIFT Streamlit viewer desktop bundle.

Set DOTNET_RUNTIME_DIR to an app-local .NET runtime directory when you want the
release artifact to work with PalmSens .pssession files without a separate
Microsoft .NET Runtime install.
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


project_root = Path(SPECPATH).parents[1]
launcher = project_root / "packaging" / "desktop" / "aswift_viewer_launcher.py"
viewer = project_root / "src" / "aswift" / "analysis" / "structured_results_viewer.py"
assets_dir = project_root / "packaging" / "desktop" / "assets"
logo_png = assets_dir / "aswift-logo.png"
mac_icon = assets_dir / "aswift-logo.icns"
win_icon = assets_dir / "aswift-logo.ico"

pypalmsens_spec = importlib.util.find_spec("pypalmsens")
pypalmsens_lib_dir = None
if pypalmsens_spec and pypalmsens_spec.submodule_search_locations:
    candidate = Path(next(iter(pypalmsens_spec.submodule_search_locations))) / "_libpalmsens"
    if candidate.exists():
        pypalmsens_lib_dir = candidate

datas = [
    (str(viewer), "streamlit_app"),
    (str(logo_png), "assets"),
    *collect_data_files("streamlit"),
    *collect_data_files("plotly"),
    *collect_data_files("pypalmsens"),
    *collect_data_files("pythonnet"),
    *collect_data_files("clr_loader"),
    *copy_metadata("aswift"),
    *copy_metadata("streamlit"),
    *copy_metadata("plotly"),
    *copy_metadata("pypalmsens"),
    *copy_metadata("pythonnet"),
    *copy_metadata("clr_loader"),
]
if pypalmsens_lib_dir is not None:
    datas.append((str(pypalmsens_lib_dir), "pypalmsens/_libpalmsens"))

dotnet_runtime_dir = os.environ.get("DOTNET_RUNTIME_DIR")
if dotnet_runtime_dir:
    datas.append((dotnet_runtime_dir, "dotnet-runtime"))

hiddenimports = [
    "clr",
    *collect_submodules("aswift"),
    *collect_submodules("streamlit"),
    *collect_submodules("plotly"),
    *collect_submodules("pypalmsens"),
    *collect_submodules("pythonnet"),
    *collect_submodules("clr_loader"),
]

binaries = [
    *collect_dynamic_libs("pythonnet"),
    *collect_dynamic_libs("clr_loader"),
]

a = Analysis(
    [str(launcher)],
    pathex=[str(project_root / "src")],
    binaries=binaries,
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
        icon=str(mac_icon),
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
        icon=str(mac_icon),
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
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(win_icon),
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
