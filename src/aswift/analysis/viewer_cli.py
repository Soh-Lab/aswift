"""Command-line launcher for the packaged Streamlit viewer."""

from __future__ import annotations

import sys
from pathlib import Path


def _run_streamlit_app() -> None:
    """Import and run the viewer with its package context intact."""
    from aswift.analysis.structured_results_viewer import _run_app

    _run_app()


def main() -> None:
    """Launch the ASWIFT Streamlit viewer."""
    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:
        raise SystemExit(
            "The ASWIFT viewer requires the viewer extra. Install it with: pip install 'aswift[viewer]'"
        ) from exc

    # Streamlit executes its target as a top-level script. Target this bootstrap
    # instead of structured_results_viewer.py so that the actual application is
    # imported as part of the aswift package and its relative imports resolve.
    viewer = Path(__file__)
    sys.argv = ["streamlit", "run", str(viewer), *sys.argv[1:]]
    streamlit_main()


if __name__ == "__main__":
    _run_streamlit_app()
