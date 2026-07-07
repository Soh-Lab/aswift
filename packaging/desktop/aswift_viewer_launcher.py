"""Frozen-app launcher for the ASWIFT Streamlit viewer."""

from __future__ import annotations

import os
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
from pathlib import Path


HOST = "127.0.0.1"
DEFAULT_PORT = 8501
SERVER_ENV = "ASWIFT_VIEWER_SERVER"


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


def _health_url(port: int) -> str:
    return f"{_app_url(port)}/_stcore/health"


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _streamlit_health_is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=1.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, TimeoutError):
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

    if not _streamlit_health_is_ready(port):
        _log(f"Discarding stale ASWIFT Viewer instance state for port {port}")
        with contextlib.suppress(OSError):
            state_path.unlink()
        return False

    _log(f"Opening existing ASWIFT Viewer instance on port {port}")
    webbrowser.open(_app_url(port))
    return True


def _write_instance_state(port: int, *, cleanup: bool, pid: int | None = None) -> None:
    state_path = _instance_state_path()
    state = {
        "port": port,
        "pid": pid,
        "created_at": time.time(),
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    if not cleanup:
        return

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


def _loading_page_path(port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"aswift-viewer-loading-{port}.html"


def _open_loading_page(port: int) -> None:
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
    env = os.environ.copy()
    env[SERVER_ENV] = "1"
    env["ASWIFT_VIEWER_PORT"] = str(port)
    env["ASWIFT_VIEWER_LOG_PATH"] = str(_log_path())
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


def main() -> None:
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
    _configure_streamlit_runtime()
    _log(
        "Environment details: "
        f"DOTNET_ROOT={os.environ.get('DOTNET_ROOT', '')} "
        f"ASWIFT_VIEWER_PORT={os.environ.get('ASWIFT_VIEWER_PORT', '')} "
        f"{SERVER_ENV}={os.environ.get(SERVER_ENV, '')}"
    )
    import_check = os.environ.get("ASWIFT_VIEWER_IMPORT_CHECK") == "1"
    server_mode = os.environ.get(SERVER_ENV) == "1"

    viewer = root / "streamlit_app" / "structured_results_viewer.py"
    if not viewer.exists():
        raise SystemExit(f"ASWIFT Viewer could not find the bundled Streamlit app: {viewer}")

    if not import_check and not server_mode:
        if _open_existing_instance():
            return
        port = _choose_port()
        _open_loading_page(port)
        proc = _spawn_server(port)
        _write_instance_state(port, cleanup=False, pid=proc.pid)
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
        _log(f"ASWIFT Viewer import check passed: {viewer}")
        print(f"ASWIFT Viewer import check passed: {viewer}")
        return

    streamlit_main()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _log(traceback.format_exc())
        raise
