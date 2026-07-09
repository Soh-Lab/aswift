"""Frozen-app launcher for the ASWIFT Streamlit viewer."""

from __future__ import annotations

import os
import platform
import socket
import sys
import webbrowser
import atexit
import contextlib
import json
import traceback
import multiprocessing as mp
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import signal
from pathlib import Path


HOST = "127.0.0.1"
DEFAULT_PORT = 8501
SERVER_ENV = "ASWIFT_VIEWER_SERVER"
LOGO_NAME = "aswift-logo.png"


def _log_path() -> Path:
    """Return the desktop launcher log path."""
    return Path.home() / "Library" / "Logs" / "ASWIFT Viewer.log"


def _cache_dir() -> Path:
    """Return the desktop launcher cache directory."""
    return Path.home() / "Library" / "Caches" / "ASWIFT Viewer"


def _log(message: str) -> None:
    """Append a timestamped message to the desktop launcher log."""
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
    os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")
    os.environ["PATH"] = str(dotnet_root) + os.pathsep + os.environ.get("PATH", "")


def _pypalmsens_sdk_dir(root: Path) -> Path | None:
    """Return the bundled PalmSens .NET SDK folder for this platform."""
    sdk_root = root / "pypalmsens" / "_libpalmsens"
    if sys.platform == "win32":
        candidate = sdk_root / "win"
    elif sys.platform == "darwin":
        machine = platform.machine().lower()
        candidate = sdk_root / ("osx-arm64" if machine in {"arm64", "aarch64"} else "osx-x64")
    else:
        machine = platform.machine().lower()
        candidate = sdk_root / ("linux-arm64" if machine in {"arm64", "aarch64"} else "linux-x64")
    return candidate if candidate.exists() else None


def _configure_bundled_pypalmsens(root: Path) -> None:
    """Expose bundled PalmSens SDK assemblies for pythonnet dependency resolution."""
    sdk_dir = _pypalmsens_sdk_dir(root)
    if sdk_dir is None:
        return
    os.environ["PATH"] = str(sdk_dir) + os.pathsep + os.environ.get("PATH", "")


def _bundled_asset_path(name: str) -> Path:
    """Return the path to a PyInstaller-collected desktop asset."""
    return _bundle_root() / "assets" / name


def _app_url(port: int) -> str:
    """Build the Streamlit app URL for a local port."""
    return f"http://{HOST}:{port}"


def _health_url(port: int) -> str:
    """Build the Streamlit health-check URL for a local port."""
    return f"{_app_url(port)}/_stcore/health"


def _port_is_open(port: int) -> bool:
    """Return whether a TCP port is accepting local connections."""
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _streamlit_health_is_ready(port: int) -> bool:
    """Return whether Streamlit reports the app as healthy."""
    try:
        with urllib.request.urlopen(_health_url(port), timeout=1.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _instance_state_path() -> Path:
    """Return the launcher state-file path for an existing app instance."""
    return Path(tempfile.gettempdir()) / "aswift-viewer-instance.json"


def _process_is_running(pid: int) -> bool:
    """Return whether a process id appears to still be alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_existing_instance() -> None:
    """Terminate a previously launched viewer process if it is still running."""
    state_path = _instance_state_path()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        port = int(state.get("port", DEFAULT_PORT))
        pid = int(state["pid"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        with contextlib.suppress(OSError):
            state_path.unlink()
        return

    if not _process_is_running(pid):
        _log(f"Discarding ASWIFT Viewer state for stopped pid {pid}")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return
    if not _port_is_open(port):
        _log(f"Discarding ASWIFT Viewer state for pid {pid}; port {port} is closed")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return

    _log(f"Stopping previous ASWIFT Viewer server pid {pid} on port {port}")
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            break
        time.sleep(0.1)

    if _process_is_running(pid):
        _log(f"Force-stopping previous ASWIFT Viewer server pid {pid}")
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)

    with contextlib.suppress(OSError):
        state_path.unlink()


def _open_existing_instance() -> bool:
    """Open the existing viewer server when a healthy instance is already running."""
    state_path = _instance_state_path()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        port = int(state["port"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    bundle_root = state.get("bundle_root")
    if bundle_root is None:
        _log(f"Discarding legacy ASWIFT Viewer instance state for port {port}")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return False
    if not Path(bundle_root).exists():
        _log(f"Discarding ASWIFT Viewer instance with missing bundle root: {bundle_root}")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return False
    if not _streamlit_health_is_ready(port):
        _log(f"Discarding stale ASWIFT Viewer instance state for port {port}")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return False

    _log(f"Opening existing ASWIFT Viewer instance on port {port}")
    webbrowser.open(_app_url(port))
    return True


def _write_instance_state(port: int, *, cleanup: bool, pid: int | None = None) -> None:
    """Persist the active viewer process and port for future launches."""
    state_path = _instance_state_path()
    if pid is None:
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
            if prior_state.get("port") == port:
                pid = prior_state.get("pid")
        except (OSError, json.JSONDecodeError):
            pid = None
    state = {
        "port": port,
        "pid": pid,
        "created_at": time.time(),
        "bundle_root": str(_bundle_root()),
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    if not cleanup:
        return

    def _cleanup() -> None:
        """Stop the launched Streamlit process and remove launcher state."""
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if state.get("port") == port:
            with contextlib.suppress(OSError):
                state_path.unlink()

    atexit.register(_cleanup)


def _terminate_process(proc: subprocess.Popen, *, timeout: float = 5.0) -> None:
    """Terminate a launched subprocess, escalating to kill when needed."""
    if proc.poll() is not None:
        return

    _log(f"Stopping ASWIFT Viewer server pid {proc.pid}")
    with contextlib.suppress(OSError):
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _log(f"Force-stopping ASWIFT Viewer server pid {proc.pid}")
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)


def _clear_instance_state(port: int) -> None:
    """Remove launcher state when it still belongs to the current server port."""
    state_path = _instance_state_path()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if state.get("port") == port:
        with contextlib.suppress(OSError):
            state_path.unlink()


def _run_status_window(port: int, proc: subprocess.Popen) -> None:
    """Show a small desktop window while the local ASWIFT server is running."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        _log(f"Could not load tkinter status window: {exc}")
        try:
            while proc.poll() is None:
                time.sleep(1.0)
        finally:
            _terminate_process(proc)
            _clear_instance_state(port)
        return

    root = tk.Tk()
    root.title("ASWIFT Viewer")
    root.geometry("420x190")
    root.resizable(False, False)

    icon_path = _bundled_asset_path(LOGO_NAME)
    if icon_path.exists():
        with contextlib.suppress(tk.TclError):
            icon = tk.PhotoImage(file=str(icon_path))
            root.iconphoto(True, icon)

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)

    title = ttk.Label(frame, text="ASWIFT Viewer is running", font=("", 15, "bold"))
    title.pack(anchor="w")

    status = ttk.Label(
        frame,
        text=(
            "A local Streamlit server is open in your browser.\n"
            "Keep this window open while using ASWIFT."
        ),
        justify="left",
    )
    status.pack(anchor="w", pady=(10, 14))

    button_row = ttk.Frame(frame)
    button_row.pack(anchor="e", fill="x")

    def open_viewer() -> None:
        """Open the running viewer in the default browser."""
        webbrowser.open(_app_url(port))

    def quit_viewer() -> None:
        """Close the status window and stop the local ASWIFT server."""
        root.destroy()

    ttk.Button(button_row, text="Open Viewer", command=open_viewer).pack(side="right", padx=(8, 0))
    ttk.Button(button_row, text="Quit ASWIFT Viewer", command=quit_viewer).pack(side="right")

    def poll_server() -> None:
        """Close the status window if the Streamlit server exits."""
        if proc.poll() is not None:
            _log(f"ASWIFT Viewer server exited with code {proc.returncode}")
            root.destroy()
            return
        root.after(1000, poll_server)

    root.protocol("WM_DELETE_WINDOW", quit_viewer)
    root.after(1000, poll_server)
    try:
        root.mainloop()
    finally:
        _terminate_process(proc)
        _clear_instance_state(port)


def _choose_port() -> int:
    """Choose an available localhost port for Streamlit."""
    for port in range(DEFAULT_PORT, DEFAULT_PORT + 50):
        if not _port_is_open(port):
            return port
    raise SystemExit("ASWIFT Viewer could not find a free local port to start Streamlit.")


def _loading_page_path(port: int) -> Path:
    """Return the temporary loading-page path for a viewer port."""
    return Path(tempfile.gettempdir()) / f"aswift-viewer-loading-{port}.html"


def _open_loading_page(port: int) -> None:
    """Open a lightweight loading page while the Streamlit server starts."""
    target = _app_url(port)
    health = _health_url(port)
    page = _loading_page_path(port)
    log_path = _log_path()
    page.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Opening ASWIFT Viewer</title>
  <style>
    body {{
      align-items: center;
      background: #f7f7f4;
      color: #1f2933;
      display: flex;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      height: 100vh;
      justify-content: center;
      margin: 0;
    }}
    main {{
      max-width: 34rem;
      padding: 2rem;
      text-align: center;
    }}
    h1 {{
      font-size: 1.6rem;
      margin-bottom: 0.5rem;
    }}
    p {{
      color: #52606d;
      line-height: 1.5;
    }}
    code {{
      background: #eceff1;
      border-radius: 4px;
      display: block;
      margin-top: 1rem;
      overflow-wrap: anywhere;
      padding: 0.75rem;
      text-align: left;
    }}
    .hidden {{
      display: none;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Opening ASWIFT Viewer...</h1>
    <p id="status">The app is starting a local Streamlit server. This can take a little while
    the first time after downloading.</p>
    <p id="debug" class="hidden">ASWIFT Viewer did not finish starting. Quit any old ASWIFT Viewer
    processes, reopen the app, and send this log file to the developer:</p>
    <code id="log" class="hidden">{log_path}</code>
  </main>
  <script>
    const target = "{target}";
    const health = "{health}";
    const started = Date.now();
    const timeoutMs = 45000;
    async function check() {{
      try {{
        await fetch(health, {{ mode: "no-cors", cache: "no-store" }});
        window.location.replace(target);
      }} catch (error) {{
        if (Date.now() - started > timeoutMs) {{
          document.getElementById("status").textContent =
            "Startup is taking longer than expected.";
          document.getElementById("debug").classList.remove("hidden");
          document.getElementById("log").classList.remove("hidden");
        }}
        setTimeout(check, 1000);
      }}
    }}
    check();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    _log(f"Opening loading page {page} for Streamlit port {port}")
    webbrowser.open(page.as_uri())


def _spawn_server(port: int) -> subprocess.Popen:
    """Start the packaged Streamlit server process."""
    env = os.environ.copy()
    env[SERVER_ENV] = "1"
    env["ASWIFT_VIEWER_PORT"] = str(port)
    env["ASWIFT_VIEWER_LOG_PATH"] = str(_log_path())
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    log = _log_path().open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable],
        close_fds=True,
        env=env,
        start_new_session=True,
        stdout=log,
        stderr=log,
    )
    _log(f"Spawned ASWIFT Viewer server on port {port} with pid {proc.pid}")
    return proc


def _configure_streamlit_runtime() -> None:
    """Use production Streamlit settings inside the frozen desktop app."""
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    matplotlib_cache = _cache_dir() / "matplotlib"
    with contextlib.suppress(OSError):
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def main() -> None:
    """Run the command-line entry point for this module."""
    mp.freeze_support()
    _log("Starting ASWIFT Viewer")

    root = _bundle_root()
    _log(
        "Runtime details: "
        f"frozen={getattr(sys, 'frozen', False)} "
        f"platform={sys.platform} "
        f"executable={sys.executable} "
        f"argv={sys.argv} "
        f"cwd={Path.cwd()} "
        f"bundle_root={root}"
    )
    _configure_bundled_dotnet(root)
    _configure_bundled_pypalmsens(root)
    _configure_streamlit_runtime()
    _log(
        "Environment details: "
        f"DOTNET_ROOT={os.environ.get('DOTNET_ROOT', '')} "
        f"PYTHONNET_RUNTIME={os.environ.get('PYTHONNET_RUNTIME', '')} "
        f"ASWIFT_VIEWER_PORT={os.environ.get('ASWIFT_VIEWER_PORT', '')} "
        f"{SERVER_ENV}={os.environ.get(SERVER_ENV, '')}"
    )
    import_check = os.environ.get("ASWIFT_VIEWER_IMPORT_CHECK") == "1"
    server_mode = os.environ.get(SERVER_ENV) == "1"

    viewer = root / "streamlit_app" / "structured_results_viewer.py"
    if not viewer.exists():
        raise SystemExit(f"ASWIFT Viewer could not find the bundled Streamlit app: {viewer}")

    if not import_check and not server_mode:
        _stop_existing_instance()
        port = _choose_port()
        _open_loading_page(port)
        proc = _spawn_server(port)
        _write_instance_state(port, cleanup=False, pid=proc.pid)
        _run_status_window(port, proc)
        return

    port = int(os.environ.get("ASWIFT_VIEWER_PORT", DEFAULT_PORT))

    if not import_check:
        _write_instance_state(port, cleanup=True)

    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:
        raise SystemExit(
            "ASWIFT Viewer could not load Streamlit from the bundled application: "
            f"{exc}"
        ) from exc

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
    _log(f"Starting Streamlit with args: {streamlit_args}")

    if import_check:
        import streamlit.config as st_config

        st_config.get_config_options()
        import pythonnet

        pythonnet.load("coreclr")
        import clr  # noqa: F401
        import pypalmsens  # noqa: F401
        _log(f"ASWIFT Viewer import check passed: {viewer}")
        print(f"ASWIFT Viewer import check passed: {viewer}")
        return

    streamlit_main()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            _log(traceback.format_exc())
        raise
    except BaseException:
        _log(traceback.format_exc())
        raise
