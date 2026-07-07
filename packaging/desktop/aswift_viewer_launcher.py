"""Frozen-app launcher for the ASWIFT Streamlit viewer."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
import atexit
import contextlib
import json
import traceback
import multiprocessing as mp
import tempfile
from pathlib import Path


HOST = "127.0.0.1"
DEFAULT_PORT = 8501


def _log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "ASWIFT Viewer.log"


def _log(message: str) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except OSError:
        return


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


def _app_url(port: int) -> str:
    return f"http://{HOST}:{port}"


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _instance_state_path() -> Path:
    return Path(tempfile.gettempdir()) / "aswift-viewer-instance.json"


def _open_existing_instance() -> bool:
    state_path = _instance_state_path()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        port = int(state["port"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    if not _port_is_open(port):
        with contextlib.suppress(OSError):
            state_path.unlink()
        return False

    webbrowser.open(_app_url(port))
    return True


def _write_instance_state(port: int) -> None:
    state_path = _instance_state_path()
    state_path.write_text(json.dumps({"port": port}), encoding="utf-8")

    def _cleanup() -> None:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if state.get("port") == port:
            with contextlib.suppress(OSError):
                state_path.unlink()

    atexit.register(_cleanup)


def _choose_port() -> int:
    for port in range(DEFAULT_PORT, DEFAULT_PORT + 50):
        if not _port_is_open(port):
            return port
    raise SystemExit("ASWIFT Viewer could not find a free local port to start Streamlit.")


def _open_browser_when_ready(port: int) -> None:
    """Open the Streamlit URL once the local server accepts connections."""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if _port_is_open(port):
            webbrowser.open(_app_url(port))
            return
        else:
            time.sleep(0.5)


def _configure_streamlit_runtime() -> None:
    """Use production Streamlit settings inside the frozen desktop app."""
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


def main() -> None:
    _log("Starting ASWIFT Viewer")
    mp.freeze_support()

    root = _bundle_root()
    _configure_bundled_dotnet(root)
    _configure_streamlit_runtime()

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

    import_check = os.environ.get("ASWIFT_VIEWER_IMPORT_CHECK") == "1"
    if not import_check and _open_existing_instance():
        return

    port = DEFAULT_PORT if import_check else _choose_port()
    if not import_check:
        _write_instance_state(port)

    streamlit_args = [
        "streamlit",
        "run",
        str(viewer),
        "--server.address",
        HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
        *sys.argv[1:],
    ]
    sys.argv = streamlit_args

    if import_check:
        import streamlit.config as st_config

        st_config.get_config_options()
        _log(f"ASWIFT Viewer import check passed: {viewer}")
        print(f"ASWIFT Viewer import check passed: {viewer}")
        return

    threading.Thread(target=_open_browser_when_ready, args=(port,), daemon=True).start()
    streamlit_main()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _log(traceback.format_exc())
        raise
