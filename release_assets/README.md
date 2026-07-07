# ASWIFT Release Assets

This folder is for downloadable ASWIFT release artifacts, not reusable build
source. The reusable desktop-app packaging files live in `packaging/desktop/`.
The recommended non-coder release artifact is a per-platform zipped PyInstaller
app:

- `ASWIFT-Viewer-windows-x64.zip`
- `ASWIFT-Viewer-macos-x64.zip`
- `ASWIFT-Viewer-macos-arm64.zip`

Each zip contains a frozen copy of Python, ASWIFT, Streamlit, and the viewer's
Python dependencies. Users do not need to install Python, pip, a C++ compiler,
or run `streamlit run`.

## PalmSens and .NET

PalmSens `.pssession` support goes through `pypalmsens`, which needs a .NET 9
runtime. There are two viable release policies:

1. Bundle an app-local .NET runtime in each zip.
   This is the easiest user experience. The launcher sets `DOTNET_ROOT`, adds
   the private runtime to `PATH`, and starts Streamlit in the browser.
2. Do not bundle .NET.
   This makes smaller downloads, but users who open `.pssession` files must
   install Microsoft .NET 9 Runtime separately.

The GitHub Actions workflow in `.github/workflows/build-release-app.yml` uses
policy 1 by default. Manual workflow runs publish the zips as GitHub Actions
artifacts; published-release runs also attach the zips to the GitHub Release.
The workflow builds macOS Intel on `macos-15-intel` and macOS Apple Silicon on
`macos-15`; keep those labels explicit so release builds do not silently move
when GitHub changes `macos-latest`.
macOS artifacts contain `ASWIFT Viewer.app`; Windows artifacts contain the
`ASWIFT Viewer` folder with `ASWIFT Viewer.exe`. The macOS workflow ad-hoc
signs the `.app` bundle and zips it with `ditto` so executable permissions and
bundle metadata survive download/unzip. The workflow also launches the frozen
app in import-check mode before zipping, so a build fails if Streamlit or the
bundled viewer script cannot be loaded. The PyInstaller spec explicitly bundles
package metadata for ASWIFT, Streamlit, and pypalmsens because Streamlit queries
its installed package version at startup.
When launched from Finder, startup errors are written to
`~/Library/Logs/ASWIFT Viewer.log`.

## Local Build

From the repository root, build on the same operating system and CPU type as
the users who will run the app:

```bash
python -m venv .release-venv
.release-venv/bin/python -m pip install --upgrade pip
.release-venv/bin/python -m pip install -e ".[viewer]" pyinstaller
DOTNET_RUNTIME_DIR=/path/to/dotnet-runtime \
  .release-venv/bin/pyinstaller --clean --noconfirm \
  --workpath build/pyinstaller \
  --distpath build/desktop-dist \
  packaging/desktop/aswift_viewer.spec
```

On Windows PowerShell, use:

```powershell
py -3.12 -m venv .release-venv
.\.release-venv\Scripts\python -m pip install --upgrade pip
.\.release-venv\Scripts\python -m pip install -e ".[viewer]" pyinstaller
$env:DOTNET_RUNTIME_DIR = "C:\path\to\dotnet-runtime"
.\.release-venv\Scripts\pyinstaller --clean --noconfirm `
  --workpath build\pyinstaller `
  --distpath build\desktop-dist `
  packaging\desktop\aswift_viewer.spec
```

The Windows app appears under `build/desktop-dist/ASWIFT Viewer/`. On macOS,
ad-hoc sign and zip `build/desktop-dist/ASWIFT Viewer.app` with `ditto`:

```bash
chmod +x "build/desktop-dist/ASWIFT Viewer.app/Contents/MacOS/ASWIFT Viewer"
codesign --force --deep --sign - "build/desktop-dist/ASWIFT Viewer.app"
ditto -c -k --keepParent --sequesterRsrc --rsrc \
  "build/desktop-dist/ASWIFT Viewer.app" \
  ASWIFT-Viewer-macos-arm64.zip
```

## Folder Roles

- `dist/`: generated Python package builds, such as the PyPI wheel and source
  tarball from `python -m build`.
- `packaging/desktop/`: committed source files for building the double-clickable
  Streamlit desktop app.
- `release_assets/`: downloadable release artifacts for end users, such as
  example-data zips and final app zips.

## User Workflow

1. Download the zip for the user's operating system and processor.
2. Unzip it.
3. Double-click `ASWIFT Viewer.app` on macOS or `ASWIFT Viewer.exe` on Windows.
4. The app starts a local Streamlit server and opens the browser.
5. Use upload mode for CSV/JSON files, or enter a PalmSens folder path for
   `.pssession` files.

On macOS, unsigned apps may need right-click > Open the first time. For a public
release, code-sign and notarize the macOS zips and sign the Windows executable.
If replacing a test build on macOS, quit old ASWIFT Viewer processes and delete
the old app before unzipping the new one:

```bash
pkill -f "ASWIFT Viewer" || true
rm -f /tmp/aswift-viewer-instance.json
rm -rf ~/Downloads/"ASWIFT Viewer.app"
```
