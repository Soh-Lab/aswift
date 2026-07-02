"""Command-line launcher for the packaged Streamlit viewer."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the ASWIFT Streamlit viewer."""
    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:
        raise SystemExit(
            "The ASWIFT viewer requires Streamlit. Install it with: pip install 'aswift[streamlit]'"
        ) from exc

    viewer = Path(__file__).with_name("structured_results_viewer.py")
    sys.argv = ["streamlit", "run", str(viewer), *sys.argv[1:]]
    streamlit_main()

