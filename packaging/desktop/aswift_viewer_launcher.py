"""Frozen-app launcher for the ASWIFT Streamlit viewer."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8501
APP_URL = f"http://{HOST}:{PORT}"


def _bundle_root() -> Path:
    """Return the directory that contains PyInstaller-collected data files."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _configure_bundled_dotnet(root: Path) -> None:
    """Prefer an app-local .NET runtime when the release bundle includes one."""
    dotnet_root = root / "dotnet-runtime"
    if not dotnet_root.exists():
        return

    os.environ.setdefault("DOTNET_ROOT", str(dotnet_root))
    os.environ.setdefault("DOTNET_MULTILEVEL_LOOKUP", "0")
    os.environ["PATH"] = str(dotnet_root) + os.pathsep + os.environ.get("PATH", "")


def _open_browser_when_ready() -> None:
    """Open the Streamlit URL once the local server accepts connections."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                webbrowser.open(APP_URL)
                return
        except OSError:
            time.sleep(0.5)


def main() -> None:
    root = _bundle_root()
    _configure_bundled_dotnet(root)

    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:
        raise SystemExit(
            "ASWIFT Viewer could not load Streamlit from the bundled application: "
            f"{exc}"
        ) from exc

    viewer = root / "streamlit_app" / "structured_results_viewer.py"
    if not viewer.exists():
        raise SystemExit(f"ASWIFT Viewer could not find the bundled Streamlit app: {viewer}")

    if os.environ.get("ASWIFT_VIEWER_IMPORT_CHECK") == "1":
        print(f"ASWIFT Viewer import check passed: {viewer}")
        return

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    sys.argv = [
        "streamlit",
        "run",
        str(viewer),
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        *sys.argv[1:],
    ]
    streamlit_main()


if __name__ == "__main__":
    main()
