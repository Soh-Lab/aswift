"""Streamlit viewer for batch fit results from the example notebooks.

Run from the repository root with:

    streamlit run analysis/structured_results_viewer.py -- examples/outputs/pssession_demo/pssession_fit_results.json

The input file should be the result dataframe saved by ``fit_dataframe`` or
``fit_pssession_folder`` with ``orient="records"`` JSON, or the same dataframe
saved as CSV.
"""

from __future__ import annotations

import ast
import sys
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peak_extraction.batch import (  # noqa: E402
    order_results_dataframe,
    plot_fit_result_from_row,
    plot_signal_over_time,
)

ARRAY_COLUMNS = {
    "volts",
    "signal",
    "peak_profile",
    "background_profile",
    "fitted_signal",
}

REQUIRED_RESULT_COLUMNS = {
    "method",
    "volts",
    "signal",
    "peak",
    "background",
    "peak_voltage",
    "peak_index",
    "peak_profile",
    "background_profile",
    "success",
}


def _default_results_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    example_path = PROJECT_ROOT / "examples" / "outputs" / "pssession_demo" / "pssession_fit_results.json"
    return str(example_path)


def _parse_array_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return value

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_RESULT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Results file is missing required columns: {sorted(missing)}")

    df = df.copy()
    for column in ARRAY_COLUMNS & set(df.columns):
        df[column] = df[column].map(_parse_array_value)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if "time" not in df.columns and df["timestamp"].notna().any():
            first = df["timestamp"].min()
            df["time"] = (df["timestamp"] - first).dt.total_seconds() / 3600

    return order_results_dataframe(df)


@st.cache_data(show_spinner=False)
def _load_results_from_path(path_text: str) -> pd.DataFrame:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Use a .json or .csv results file.")

    return _prepare_results(df)


@st.cache_data(show_spinner=False)
def _load_results_from_upload(name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        df = pd.read_json(BytesIO(payload))
    elif suffix == ".csv":
        df = pd.read_csv(StringIO(payload.decode("utf-8")))
    else:
        raise ValueError("Use a .json or .csv results file.")

    return _prepare_results(df)


def _select_value(label: str, data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, Any]:
    if column not in data.columns:
        return data, None

    values = sorted(data[column].dropna().unique().tolist())
    if not values:
        return data, None

    selected = st.sidebar.selectbox(label, values)
    return data[data[column] == selected], selected


def _row_label(row: pd.Series) -> str:
    pieces = []
    for column in ("num", "time", "file", "method"):
        if column not in row or pd.isna(row[column]):
            continue
        value = row[column]
        if column == "time":
            pieces.append(f"time={float(value):.3g} h")
        else:
            pieces.append(f"{column}={value}")
    return ", ".join(pieces) or f"row={row.name}"


def _result_summary(row: pd.Series) -> pd.DataFrame:
    columns = [
        "method",
        "success",
        "error",
        "peak",
        "background",
        "peak_voltage",
        "time",
        "timestamp",
        "hz",
        "channel",
        "num",
        "file",
    ]
    present = [column for column in columns if column in row.index]
    return pd.DataFrame({"field": present, "value": [row[column] for column in present]})


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stSlider"] label {
            display: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stTickBar"] {
            display: none;
        }
        .section-gap {
            height: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SWV Fit Results Viewer", layout="wide")
    _inject_style()
    st.title("SWV Fit Results Viewer")

    st.sidebar.header("Results File")
    uploaded = st.sidebar.file_uploader("Upload results JSON or CSV", type=["json", "csv"])
    path_text = st.sidebar.text_input("Or load local results path", value=_default_results_path())

    try:
        if uploaded is not None:
            results = _load_results_from_upload(uploaded.name, uploaded.getvalue())
        else:
            results = _load_results_from_path(path_text)
    except Exception as exc:
        st.error(exc)
        st.stop()

    filtered = results
    filtered, selected_channel = _select_value("Channel", filtered, "channel")
    filtered, selected_hz = _select_value("Frequency (Hz)", filtered, "hz")

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        st.stop()

    sort_cols = [col for col in ("time", "timestamp", "num", "file", "method") if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols)

    selected_position = st.sidebar.slider(
        "Sample index",
        min_value=0,
        max_value=max(len(filtered) - 1, 0),
        value=0,
        label_visibility="collapsed",
    )
    selected_row = filtered.iloc[int(selected_position)]

    title_parts = []
    if selected_channel is not None:
        title_parts.append(f"channel {selected_channel}")
    if selected_hz is not None:
        title_parts.append(f"{selected_hz} Hz")
    title = " | ".join(title_parts) or "selected trace"

    left, right = st.columns([2.4, 1])

    with left:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        plot_fit_result_from_row(selected_row, ax=ax)
        ax.set_title(f"{title} | {_row_label(selected_row)}")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Selected Fit")
        st.dataframe(_result_summary(selected_row), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    if "time" in results.columns or "timestamp" in results.columns:
        st.subheader("Signal Over Time")
        signal_scope = results.copy()
        if selected_channel is not None and "channel" in signal_scope.columns:
            signal_scope = signal_scope[signal_scope["channel"] == selected_channel]
        if selected_hz is not None and "hz" in signal_scope.columns:
            signal_scope = signal_scope[signal_scope["hz"] == selected_hz]

        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        plot_signal_over_time(signal_scope, ax=ax)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    with st.expander("Filtered Result Rows"):
        preview_cols = [
            col for col in ("timestamp", "time", "hz", "channel", "num", "file", "method", "peak", "peak_voltage", "success")
            if col in filtered.columns
        ]
        st.dataframe(filtered[preview_cols].reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
