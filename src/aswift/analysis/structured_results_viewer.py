"""Streamlit viewer for ASWIFT batch inputs and fit results."""

from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from aswift.workflow.batch import (
    fit_dataframe,
    order_results_dataframe,
    pssession_folder_to_dataframe,
    results_to_signal_table,
    strip_mp3_suffix_from_pssession_files,
)

from aswift.analysis.plots import plot_fit_result_from_row, plot_signal_over_time

ARRAY_COLUMNS = {
    "voltage",
    "current",
    "volts",
    "signal",
    "peak_profile",
    "background_profile",
    "fitted_signal",
}

REQUIRED_FIT_COLUMNS = {
    "method",
    "peak",
    "background",
    "peak_voltage",
    "peak_index",
    "peak_profile",
    "background_profile",
    "success",
}
TRACE_COLUMNS = {"voltage", "current"}
SIMPLE_CSV_SOURCE = "simple_csv"
RESULTS_JSON_NAME = "aswift_fit_results.json"
SIGNAL_TABLE_NAME = "aswift_signal_table.csv"
PROGRESS_UPDATE_SECONDS = 0.25
FIT_CHUNKSIZE = 16
DOWNLOAD_DROP_COLUMNS = {
    *ARRAY_COLUMNS,
    "peak_index",
    "peak_idx",
    "peak_window_start",
    "peak_window_end",
    "normalization_start_index",
    "normalization_end_index",
    "normalization_reference_peak",
}


def _cache_data_if_streamlit_runtime(**kwargs):
    def decorator(func):
        if st.runtime.exists():
            return st.cache_data(**kwargs)(func)
        return func

    return decorator


def _startup_results_path() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else ""


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


def _prepare_results(df: pd.DataFrame, *, source_kind: str = "results") -> pd.DataFrame:
    missing = REQUIRED_FIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Results file is missing required columns: {sorted(missing)}")
    if not ({"voltage", "current"}.issubset(df.columns) or {"volts", "signal"}.issubset(df.columns)):
        raise ValueError("Results file must contain array columns voltage/current or legacy volts/signal")

    df = df.copy()
    for column in ARRAY_COLUMNS & set(df.columns):
        df[column] = df[column].map(_parse_array_value)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if "time" not in df.columns and df["timestamp"].notna().any():
            first = df["timestamp"].min()
            df["time"] = (df["timestamp"] - first).dt.total_seconds() / 3600

    results = order_results_dataframe(df)
    results.attrs["aswift_source_kind"] = source_kind
    return results


def _parallel_backend_for_workers(n_workers: int | None) -> str:
    return "process" if n_workers is not None and int(n_workers) > 1 else "thread"


class _StreamlitProgress:
    def __init__(self, label: str = "Processing") -> None:
        self.label = label
        self._last_update = 0.0
        self._text = st.sidebar.empty()
        self._bar = st.sidebar.progress(0)

    def update(self, completed: int, total: int, label: str | None = None, *, force: bool = False) -> None:
        if total <= 0:
            return
        completed = min(max(int(completed), 0), int(total))
        now = time.monotonic()
        if not force and completed < total and now - self._last_update < PROGRESS_UPDATE_SECONDS:
            return
        self._last_update = now
        active_label = label or self.label
        self._text.caption(f"{active_label}: {completed}/{total}")
        self._bar.progress(completed / total)

    def callback(self, label: str | None = None) -> Callable[[int, int], None]:
        return lambda completed, total: self.update(completed, total, label)

    def clear(self) -> None:
        self._text.empty()
        self._bar.empty()


def _read_csv_for_viewer(source) -> pd.DataFrame:
    header_df = _read_csv_with_encoding_fallback(source)
    if REQUIRED_FIT_COLUMNS.issubset(header_df.columns) or TRACE_COLUMNS.issubset(header_df.columns):
        return header_df
    if _simple_csv_to_trace_dataframe(header_df) is not None:
        return header_df

    if hasattr(source, "seek"):
        source.seek(0)
    return _read_csv_with_encoding_fallback(source, header=None)


def _read_csv_payload_for_viewer(payload: bytes) -> pd.DataFrame:
    return _read_csv_for_viewer(BytesIO(payload))


def _read_table_path_for_viewer(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return _read_csv_for_viewer(path)
    raise ValueError("Startup path must be a .json or .csv file.")


def _read_csv_with_encoding_fallback(source, **kwargs) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, **kwargs)  # type: ignore[return-value]
        except UnicodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Could not read CSV file.")


def _is_numeric_label(value: Any) -> bool:
    try:
        float(str(value).strip())
    except ValueError:
        return False
    return True


def _simple_csv_to_trace_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.shape[1] < 2:
        return None

    data = df.copy()
    if all(_is_numeric_label(column) for column in data.columns):
        first_row = pd.DataFrame([list(data.columns)], columns=data.columns)
        data = pd.concat([first_row, data], ignore_index=True)

    numeric = data.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.shape[0] < 5 or numeric.shape[1] < 2:
        return None

    column_names = [str(column).strip().lower() for column in numeric.columns]
    shared_voltage = _shared_voltage_csv_to_trace_dataframe(numeric, column_names)
    if shared_voltage is not None:
        return shared_voltage

    return _paired_simple_csv_to_trace_dataframe(numeric, column_names)


def _shared_voltage_csv_to_trace_dataframe(
    numeric: pd.DataFrame,
    column_names: list[str],
) -> pd.DataFrame | None:
    voltage_idx = 0
    named_voltage_indices = [
        idx for idx, name in enumerate(column_names)
        if "volt" in name or name in {"v", "potential"}
    ]
    if named_voltage_indices:
        voltage_idx = named_voltage_indices[0]
    elif column_names[0] and "current" in column_names[0]:
        return None

    current_indices = [
        idx for idx, name in enumerate(column_names)
        if idx != voltage_idx and ("current" in name or name.startswith("i"))
    ]
    if not current_indices:
        current_indices = [idx for idx in range(numeric.shape[1]) if idx != voltage_idx]

    voltage = numeric.iloc[:, voltage_idx].to_numpy(dtype=float)
    rows = []
    for channel, current_idx in enumerate(current_indices):
        current = numeric.iloc[:, current_idx].to_numpy(dtype=float)
        mask = np.isfinite(current) & np.isfinite(voltage)
        if mask.sum() < 5:
            continue
        rows.append(
            {
                "channel": channel,
                "num": 0,
                "file": "simple_csv",
                "voltage": voltage[mask].tolist(),
                "current": current[mask].tolist(),
            }
        )

    return pd.DataFrame(rows) if rows else None


def _paired_simple_csv_to_trace_dataframe(
    numeric: pd.DataFrame,
    column_names: list[str],
) -> pd.DataFrame | None:
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for idx, name in enumerate(column_names):
        if idx in used or "current" not in name:
            continue
        for candidate in (idx + 1, idx - 1):
            if (
                0 <= candidate < len(column_names)
                and candidate not in used
                and ("volt" in column_names[candidate] or column_names[candidate] in {"v", "potential"})
            ):
                pairs.append((idx, candidate))
                used.update({idx, candidate})
                break

    if not pairs:
        if numeric.shape[1] % 2 != 0:
            return None
        pairs = [(idx, idx + 1) for idx in range(0, numeric.shape[1], 2)]

    rows = []
    for channel, (current_idx, voltage_idx) in enumerate(pairs):
        current = numeric.iloc[:, current_idx].to_numpy(dtype=float)
        voltage = numeric.iloc[:, voltage_idx].to_numpy(dtype=float)
        mask = np.isfinite(current) & np.isfinite(voltage)
        if mask.sum() < 5:
            continue
        rows.append(
            {
                "channel": channel,
                "num": 0,
                "file": "simple_csv",
                "voltage": voltage[mask].tolist(),
                "current": current[mask].tolist(),
            }
        )

    return pd.DataFrame(rows) if rows else None


def _fit_trace_dataframe(
    trace_df: pd.DataFrame,
    *,
    method: str,
    n_workers: int | None,
    _progress: _StreamlitProgress | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if _progress is not None:
        _progress.update(0, len(trace_df), "Fitting rows", force=True)

    kwargs: dict[str, Any] = {
        "method": method,
        "n_workers": n_workers,
        "parallel_backend": _parallel_backend_for_workers(n_workers),
        "chunksize": FIT_CHUNKSIZE,
        "progress_callback": _progress.callback("Fitting rows") if _progress is not None else None,
    }
    if group_cols is not None:
        kwargs["group_cols"] = group_cols

    return fit_dataframe(trace_df, **kwargs)


def _prepare_table_or_fit_traces(
    df: pd.DataFrame,
    *,
    source_path: Path | None = None,
    method: str = "aswift",
    n_workers: int | None = None,
    _progress: _StreamlitProgress | None = None,
) -> pd.DataFrame:
    if REQUIRED_FIT_COLUMNS.issubset(df.columns):
        return _prepare_results(df)
    if TRACE_COLUMNS.issubset(df.columns):
        results = _fit_trace_dataframe(
            df,
            method=method,
            n_workers=n_workers,
            _progress=_progress,
        )
        if source_path is not None:
            output_path = source_path.with_name(f"{source_path.stem}_aswift_fit_results.json")
            results.to_json(output_path, orient="records", indent=2)
        return _prepare_results(results)
    simple_df = _simple_csv_to_trace_dataframe(df)
    if simple_df is not None:
        results = _fit_trace_dataframe(
            simple_df,
            method=method,
            n_workers=n_workers,
            _progress=_progress,
        )
        if source_path is not None:
            output_path = source_path.with_name(f"{source_path.stem}_aswift_fit_results.json")
            results.to_json(output_path, orient="records", indent=2)
        return _prepare_results(results, source_kind=SIMPLE_CSV_SOURCE)
    raise ValueError(
        "Input must be fit results, a structured trace CSV, or a simple voltage/current CSV."
    )


def _upload_cache_key(
    uploads: tuple[tuple[str, bytes], ...],
    method: str,
    n_workers: int | None,
    csv_order: str,
) -> tuple:
    return (
        "upload_results",
        method,
        int(n_workers) if n_workers is not None else None,
        csv_order,
        tuple((name, len(payload), _payload_digest(payload)) for name, payload in uploads),
    )


def _startup_cache_key(
    path_text: str,
    method: str,
    n_workers: int | None,
    signature: tuple[str, int, int],
) -> tuple:
    return (
        "startup_results",
        method,
        int(n_workers) if n_workers is not None else None,
        signature,
        str(Path(path_text).expanduser()),
    )


def _payload_digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _load_results_from_upload_impl(
    uploads: tuple[tuple[str, bytes], ...],
    method: str,
    n_workers: int | None,
    csv_order: str,
    _progress: _StreamlitProgress | None = None,
) -> pd.DataFrame:
    if not uploads:
        raise ValueError("Upload at least one JSON or CSV file.")

    ordered_uploads = _order_uploads(uploads, csv_order)
    if len(ordered_uploads) == 1:
        name, payload = ordered_uploads[0]
        suffix = Path(name).suffix.lower()
        if suffix == ".json":
            df = pd.read_json(BytesIO(payload))
            return _prepare_table_or_fit_traces(
                df,
                method=method,
                n_workers=n_workers,
                _progress=_progress,
            )
        elif suffix == ".csv":
            df = _read_csv_payload_for_viewer(payload)
        else:
            raise ValueError("Use .json or .csv files.")
        if REQUIRED_FIT_COLUMNS.issubset(df.columns):
            return _prepare_results(df)
        is_structured_trace = TRACE_COLUMNS.issubset(df.columns)
        trace_df = df if is_structured_trace else _simple_csv_to_trace_dataframe(df)
        if trace_df is None:
            raise ValueError(f"{name} is not fit results, structured traces, or a simple voltage/current CSV.")
        trace_df = trace_df.copy()
        if not is_structured_trace or "file" not in trace_df.columns:
            trace_df["file"] = name
        if "num" not in trace_df.columns:
            trace_df["num"] = 0
        results = _fit_trace_dataframe(
            trace_df,
            method=method,
            n_workers=n_workers,
            _progress=_progress,
        )
        return _prepare_results(results, source_kind=SIMPLE_CSV_SOURCE)

    trace_frames = []
    result_frames = []
    for num, (name, payload) in enumerate(ordered_uploads):
        suffix = Path(name).suffix.lower()
        if suffix == ".json":
            result_frames.append(_prepare_results(pd.read_json(BytesIO(payload))))
            continue
        if suffix != ".csv":
            raise ValueError("Use .json or .csv files.")

        df = _read_csv_payload_for_viewer(payload)
        if REQUIRED_FIT_COLUMNS.issubset(df.columns):
            result_frames.append(_prepare_results(df))
            continue
        trace_df = df if TRACE_COLUMNS.issubset(df.columns) else _simple_csv_to_trace_dataframe(df)
        if trace_df is None:
            raise ValueError(f"{name} is not fit results, structured traces, or a simple voltage/current CSV.")
        trace_df = trace_df.copy()
        trace_df["file"] = name
        trace_df["num"] = num
        trace_frames.append(trace_df)

    frames = []
    if result_frames:
        frames.extend(result_frames)
    if trace_frames:
        trace_df = pd.concat(trace_frames, ignore_index=True)
        results = _fit_trace_dataframe(
            trace_df,
            method=method,
            n_workers=n_workers,
            _progress=_progress,
        )
        frames.append(_prepare_results(results, source_kind=SIMPLE_CSV_SOURCE))
    if not frames:
        raise ValueError("No usable upload data found.")

    combined = pd.concat(frames, ignore_index=True)
    source_kind = SIMPLE_CSV_SOURCE if trace_frames and not result_frames else "results"
    return _prepare_results(combined, source_kind=source_kind)


@_cache_data_if_streamlit_runtime(show_spinner=False)
def _load_results_from_upload(
    uploads: tuple[tuple[str, bytes], ...],
    method: str,
    n_workers: int | None,
    csv_order: str,
) -> pd.DataFrame:
    return _load_results_from_upload_impl(
        uploads,
        method,
        n_workers,
        csv_order,
    )


def _order_uploads(
    uploads: tuple[tuple[str, bytes], ...],
    csv_order: str,
) -> tuple[tuple[str, bytes], ...]:
    if csv_order == "File name":
        return tuple(sorted(uploads, key=lambda item: item[0]))
    return uploads


# noinspection PyUnusedLocal
def _load_results_from_startup_path_impl(
    path_text: str,
    method: str,
    n_workers: int | None,
    signature: tuple[str, int, int],
    _progress: _StreamlitProgress | None = None,
) -> pd.DataFrame:
    del signature
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError("Startup path must point to a JSON or CSV file.")

    return _prepare_table_or_fit_traces(
        _read_table_path_for_viewer(path),
        method=method,
        n_workers=n_workers,
        _progress=_progress,
    )


@_cache_data_if_streamlit_runtime(show_spinner=False)
def _load_results_from_startup_path(
    path_text: str,
    method: str,
    n_workers: int | None,
    signature: tuple[str, int, int],
) -> pd.DataFrame:
    return _load_results_from_startup_path_impl(
        path_text,
        method,
        n_workers,
        signature,
    )


def _file_signature(path_text: str) -> tuple[str, int, int]:
    path = Path(path_text).expanduser()
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _pssession_file_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return path.name, stat.st_mtime_ns, stat.st_size


def _live_folder_signature_key(folder: Path, method: str) -> str:
    return f"pssession_live_signature::{folder.resolve()}::{method}"


def _current_pssession_files(folder: Path) -> list[Path]:
    strip_mp3_suffix_from_pssession_files(folder)
    return sorted(folder.glob("*.pssession"), key=lambda path: path.stat().st_mtime)


def _pssession_folder_signature(folder: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(_pssession_file_key(path) for path in _current_pssession_files(folder))


def _live_folder_has_changes(folder_text: str, method: str) -> bool:
    folder = Path(folder_text).expanduser()
    if not folder.exists() or not folder.is_dir():
        return True
    signature = _pssession_folder_signature(folder)
    return st.session_state.get(_live_folder_signature_key(folder, method)) != signature


@st.fragment(run_every=2.0)
def _watch_live_pssession_folder(folder_text: str, method: str) -> None:
    if _live_folder_has_changes(folder_text, method):
        st.rerun(scope="app")


def _pssession_file_to_dataframe(path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="aswift-pssession-one-") as tmp:
        temp_path = Path(tmp) / path.name
        temp_path.write_bytes(path.read_bytes())
        df = pssession_folder_to_dataframe(temp_path.parent)
        df["folder"] = str(path.parent)
        df["file"] = path.name
        return df


def _fit_pssession_dataframes(
    frames: list[pd.DataFrame],
    *,
    method: str,
    n_workers: int | None,
    _progress: _StreamlitProgress | None = None,
) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    group_cols = [
        col for col in ["file", "hz", "num", "channel", "timestamp", "time"]
        if col in df.columns
    ]
    return _fit_trace_dataframe(
        df,
        method=method,
        n_workers=n_workers,
        group_cols=group_cols,
        _progress=_progress,
    )


def _load_results_from_live_pssession_folder(
    folder_text: str,
    method: str,
    n_workers: int | None,
    _progress: _StreamlitProgress | None = None,
) -> pd.DataFrame:
    folder = Path(folder_text).expanduser()
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    cache_key = f"pssession_live_cache::{folder.resolve()}::{method}"
    cache = st.session_state.setdefault(cache_key, {})
    files = _current_pssession_files(folder)
    current_signature = tuple(_pssession_file_key(path) for path in files)
    active_keys = {_pssession_file_key(path) for path in files}
    if _progress is not None:
        _progress.update(0, len(files), "Processing files", force=True)

    for key in list(cache):
        if key not in active_keys:
            del cache[key]

    pending: list[tuple[tuple[str, int, int], Path, pd.DataFrame]] = []
    for num, path in enumerate(files):
        key = _pssession_file_key(path)
        if key in cache:
            if _progress is not None:
                _progress.update(num + 1, len(files), "Processing files")
            continue
        try:
            frame = _pssession_file_to_dataframe(path)
        except Exception as exc:
            cache[key] = _failed_pssession_file_result(path, method, exc)
            if _progress is not None:
                _progress.update(num + 1, len(files), "Processing files")
            continue

        frame = frame.copy()
        frame["num"] = num
        pending.append((key, path, frame))
        if _progress is not None:
            _progress.update(num + 1, len(files), "Processing files")

    if pending:
        try:
            fitted = _fit_pssession_dataframes(
                [frame for _, _, frame in pending],
                method=method,
                n_workers=n_workers,
                _progress=_progress,
            )
        except Exception as exc:
            for key, path, _ in pending:
                cache[key] = _failed_pssession_file_result(path, method, exc)
        else:
            for key, path, _ in pending:
                file_results = fitted[fitted["file"] == path.name].copy()
                cache[key] = file_results if not file_results.empty else _failed_pssession_file_result(
                    path,
                    method,
                    ValueError("No fitted rows were returned for this file."),
                )

    frames = []
    for num, path in enumerate(files):
        key = _pssession_file_key(path)
        if key not in cache:
            continue
        frame = cache[key].copy()
        frame["num"] = num
        frames.append(frame)
    if not frames:
        raise ValueError(f"No .pssession files found in {folder}")

    results = _normalize_result_time(order_results_dataframe(pd.concat(frames, ignore_index=True)))
    results.to_json(folder / RESULTS_JSON_NAME, orient="records", indent=2)
    results_to_signal_table(results).to_csv(folder / SIGNAL_TABLE_NAME, index=False)
    st.session_state[_live_folder_signature_key(folder, method)] = current_signature
    return _prepare_results(results)


def _normalize_result_time(results: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in results.columns:
        return results
    normalized = results.copy()
    timestamps = cast(pd.Series, pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True))
    normalized["timestamp"] = timestamps
    if timestamps.notna().any():
        normalized["time"] = (timestamps - timestamps.min()).dt.total_seconds() / 3600
    return normalized


def _failed_pssession_file_result(path: Path, method: str, exc: Exception) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "folder": str(path.parent),
            "file": path.name,
            "num": 0,
            "method": method,
            "success": False,
            "error": str(exc),
            "peak": np.nan,
            "background": np.nan,
            "peak_voltage": np.nan,
            "peak_index": -1,
            "fw_prominence": np.nan,
            "voltage": [],
            "current": [],
            "peak_profile": [],
            "background_profile": [],
            "fitted_signal": [],
        }
    ])


def _row_label(row: pd.Series) -> str:
    pieces = []
    for column in ("num", "time", "file", "method"):
        # noinspection PyPackages
        if column not in row or pd.isna(row[column]):
            continue
        value = row[column]
        if column == "time":
            pieces.append(f"time={float(value):.3g}")
        else:
            pieces.append(f"{column}={value}")
    return ", ".join(pieces) or f"row={cast(Any, row.name)}"


def _result_summary(row: pd.Series) -> pd.DataFrame:
    columns = [
        "method",
        "success",
        "error",
        "peak",
        "norm_signal",
        "background",
        "peak_voltage",
        "fw_prominence",
        "time",
        "timestamp",
        "hz",
        "channel",
        "num",
        "file",
    ]
    present = [column for column in columns if column in row.index]
    return pd.DataFrame({"field": present, "value": [_format_summary_value(row[column]) for column in present]})


def _format_summary_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


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


def _select_sample_position(row_count: int) -> int:
    if row_count <= 1:
        st.sidebar.caption("Sample index: 0")
        return 0

    return int(
        st.sidebar.slider(
            "Sample index",
            min_value=0,
            max_value=row_count - 1,
            value=0,
            label_visibility="collapsed",
        )
    )


def _normalization_controls(row_count: int) -> tuple[bool, int, int]:
    enabled = st.sidebar.checkbox("Normalize", value=False)
    if not enabled:
        return False, 0, 0

    max_idx = max(int(row_count) - 1, 0)
    if max_idx == 0:
        st.sidebar.caption("Normalization range: 0")
        return True, 0, 0

    start = int(
        st.sidebar.slider(
            "Normalize start",
            min_value=0,
            max_value=max_idx,
            value=0,
        )
    )
    end = int(
        st.sidebar.slider(
            "Normalize end",
            min_value=0,
            max_value=max_idx,
            value=max_idx,
        )
    )
    if start > end:
        start, end = end, start
    return True, start, end


def _add_empty_norm_signal(results: pd.DataFrame) -> pd.DataFrame:
    with_norm = results.copy()
    with_norm["norm_signal"] = np.nan
    return with_norm


def _ensure_norm_signal(results: pd.DataFrame) -> pd.DataFrame:
    with_norm = results.copy()
    if "norm_signal" not in with_norm.columns:
        with_norm["norm_signal"] = np.nan
    return with_norm


def _add_norm_signal_by_peak_range(results: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    if "peak" not in results.columns or results.empty:
        return _add_empty_norm_signal(results)

    normalized = _add_empty_norm_signal(results)

    group_cols = [col for col in ("method", "hz", "channel") if col in normalized.columns]
    groups = normalized.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, normalized)]
    sort_cols = [col for col in ("timestamp", "time", "num", "file") if col in normalized.columns]

    for _, group in groups:
        ordered = group.sort_values(sort_cols) if sort_cols else group
        if ordered.empty:
            continue
        lo = min(max(int(start_idx), 0), len(ordered) - 1)
        hi = min(max(int(end_idx), lo), len(ordered) - 1)
        reference = pd.to_numeric(ordered.iloc[lo:hi + 1]["peak"], errors="coerce").mean()
        if not np.isfinite(reference) or reference == 0:
            continue
        normalized.loc[group.index, "norm_signal"] = (
            pd.to_numeric(normalized.loc[group.index, "peak"], errors="coerce") / reference
        )

    return normalized


def _select_global_value(label: str, data: pd.DataFrame, column: str) -> Any:
    if column not in data.columns:
        return None

    values = sorted(data[column].dropna().unique().tolist())
    if not values:
        return None

    return st.sidebar.selectbox(label, values)


def _sample_key_columns(data: pd.DataFrame) -> list[str]:
    for columns in (["num"], ["time"], ["timestamp"], ["file"]):
        if all(column in data.columns for column in columns):
            return columns
    return []


def _sample_options(data: pd.DataFrame) -> pd.DataFrame:
    key_cols = _sample_key_columns(data)
    if not key_cols:
        return pd.DataFrame({"__position": list(range(len(data)))})

    sort_cols = [col for col in ("time", "timestamp", "num", "file") if col in data.columns]
    options = data[key_cols].drop_duplicates().copy()
    if sort_cols:
        options = data[sort_cols].drop_duplicates(subset=key_cols).sort_values(sort_cols)
        options = options[key_cols]
    return options.reset_index(drop=True)


def _filter_global_selection(
    results: pd.DataFrame,
    *,
    selected_channel: Any,
    selected_hz: Any,
) -> pd.DataFrame:
    filtered = results
    if selected_channel is not None and "channel" in filtered.columns:
        filtered = filtered[filtered["channel"] == selected_channel]
    if selected_hz is not None and "hz" in filtered.columns:
        filtered = filtered[filtered["hz"] == selected_hz]
    return filtered


def _filter_selected_results(
    results: pd.DataFrame,
    *,
    selected_channel: Any,
    selected_hz: Any,
    selected_sample: pd.Series | None,
) -> pd.DataFrame:
    filtered = _filter_global_selection(
        results,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
    )
    if selected_sample is not None:
        for column, value in selected_sample.items():
            if cast(Any, column).startswith("__") or column not in filtered.columns:
                continue
            filtered = filtered[filtered[column] == value]
    return filtered


def _blank_fit_plot(message: str):
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    return fig, ax


def _trace_arrays_from_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    if {"voltage", "current"}.issubset(row.index):
        x_values = _parse_array_value(row["voltage"])
        y_values = _parse_array_value(row["current"])
    elif {"volts", "signal"}.issubset(row.index):
        x_values = _parse_array_value(row["volts"])
        y_values = _parse_array_value(row["signal"])
    else:
        return None

    try:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
    except (TypeError, ValueError):
        return None

    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        return None

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return x[mask], y[mask]


def _raw_trace_plot_from_row(row: pd.Series, message: str):
    arrays = _trace_arrays_from_row(row)
    if arrays is None:
        return _blank_fit_plot(message)

    x, y = arrays
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Potential")
    ax.set_ylabel("Current")
    ax.grid(True, alpha=0.25)
    return fig, ax


def _empty_result_summary(message: str) -> pd.DataFrame:
    return pd.DataFrame({"field": ["result"], "value": [message]})


def _download_results(results: pd.DataFrame) -> None:
    st.sidebar.header("Download")
    st.sidebar.download_button(
        "Download results CSV",
        data=_results_csv_bytes(results),
        file_name="aswift_fit_results.csv",
        mime="text/csv",
    )


def _results_csv_bytes(results: pd.DataFrame) -> bytes:
    return _downloadable_results(results).to_csv(index=False).encode("utf-8")


def _downloadable_results(results: pd.DataFrame) -> pd.DataFrame:
    download = _ensure_norm_signal(results)
    drop_cols = [column for column in DOWNLOAD_DROP_COLUMNS if column in download.columns]
    download = download.drop(columns=drop_cols)
    return _order_download_columns(download)


def _order_download_columns(results: pd.DataFrame) -> pd.DataFrame:
    if "peak" not in results.columns or "norm_signal" not in results.columns:
        return results

    columns = [column for column in results.columns if column != "norm_signal"]
    peak_idx = columns.index("peak")
    columns.insert(peak_idx + 1, "norm_signal")
    return results.loc[:, columns]


def main() -> None:
    """Run the Streamlit ASWIFT results viewer application."""
    st.set_page_config(page_title="SWV Fit Results Viewer", layout="wide")
    _inject_style()
    st.title("SWV Fit Results Viewer")

    st.sidebar.header("Input")
    startup_path = _startup_results_path()
    if startup_path:
        source_kind = "Startup JSON/CSV"
        st.sidebar.caption(f"Loaded at startup: {Path(startup_path).name}")
    else:
        source_kind = st.sidebar.radio(
            "Source",
            ["Upload JSON/CSV", "PalmSens .pssession"],
        )
    method = st.sidebar.selectbox("Fit method", ["aswift", "poly_linear"])
    n_workers = st.sidebar.number_input("Workers", min_value=1, value=1, step=1)
    upload_order = st.sidebar.selectbox("Upload order", ["Uploaded order", "File name"])

    progress: _StreamlitProgress | None = None
    try:
        if source_kind == "Startup JSON/CSV":
            startup_signature = _file_signature(startup_path)
            cache_key = _startup_cache_key(startup_path, method, int(n_workers), startup_signature)
            results = st.session_state.get(cache_key)
            if results is None:
                progress = _StreamlitProgress()
                results = _load_results_from_startup_path_impl(
                    startup_path,
                    method,
                    int(n_workers),
                    startup_signature,
                    _progress=progress,
                )
                st.session_state[cache_key] = results
        elif source_kind == "Upload JSON/CSV":
            uploaded_files = st.sidebar.file_uploader(
                "Upload",
                type=["json", "csv"],
                accept_multiple_files=True,
            )
            if not uploaded_files:
                st.info("Upload fit results, structured trace CSVs, or simple voltage/current CSVs to begin.")
                st.stop()
            uploads = tuple((uploaded.name, uploaded.getvalue()) for uploaded in uploaded_files)
            cache_key = _upload_cache_key(uploads, method, int(n_workers), upload_order)
            results = st.session_state.get(cache_key)
            if results is None:
                progress = _StreamlitProgress()
                results = _load_results_from_upload_impl(
                    uploads,
                    method,
                    int(n_workers),
                    upload_order,
                    _progress=progress,
                )
                st.session_state[cache_key] = results
        else:
            folder_text = st.sidebar.text_input(
                "PalmSens folder path",
                help="Path to a folder containing .pssession or .pssession.mp3 files.",
                placeholder="/path/to/pssession/folder",
            ).strip()
            if folder_text:
                folder = Path(folder_text).expanduser()
                if not folder.exists():
                    st.error(f"Folder does not exist: {folder}")
                    st.stop()
                if not folder.is_dir():
                    st.error(f"Path is not a folder: {folder}")
                    st.stop()
                st.sidebar.caption(f"Selected: {folder.name}")
                progress = _StreamlitProgress() if _live_folder_has_changes(folder_text, method) else None
                results = _load_results_from_live_pssession_folder(
                    folder_text,
                    method,
                    int(n_workers),
                    _progress=progress,
                )
                _watch_live_pssession_folder(folder_text, method)
            else:
                st.info("Enter a folder path containing .pssession or .pssession.mp3 files to begin.")
                st.stop()
        if progress is not None:
            progress.clear()
    except Exception as exc:
        st.error(exc)
        st.stop()

    selected_channel = _select_global_value("Channel", results, "channel")
    selected_hz = _select_global_value("Frequency (Hz)", results, "hz")

    sample_scope = _filter_global_selection(
        results,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
    )
    sample_options = _sample_options(sample_scope)
    selected_position = _select_sample_position(len(sample_options))
    selected_sample = sample_options.iloc[selected_position] if not sample_options.empty else None
    normalize, norm_start, norm_end = _normalization_controls(len(sample_options))
    display_results = (
        _add_norm_signal_by_peak_range(results, norm_start, norm_end)
        if normalize
        else _add_empty_norm_signal(results)
    )
    _download_results(display_results)

    filtered = _filter_selected_results(
        display_results,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
        selected_sample=selected_sample,
    )
    sort_cols = [col for col in ("time", "timestamp", "num", "file", "method") if col in filtered.columns]
    if sort_cols and not filtered.empty:
        filtered = filtered.sort_values(sort_cols)
    selected_row: pd.Series | None = filtered.iloc[0] if not filtered.empty else None

    title_parts = []
    if selected_channel is not None:
        title_parts.append(f"channel {selected_channel}")
    if selected_hz is not None:
        title_parts.append(f"{selected_hz} Hz")
    title = " | ".join(title_parts) or "selected trace"

    left, right = st.columns([2.4, 1])

    with left:
        if selected_row is None:
            fig, ax = _blank_fit_plot("No result for the selected sample.")
            ax.set_title(title)
        elif not bool(selected_row.get("success", True)):
            fig, ax = _raw_trace_plot_from_row(selected_row, "No trace data for the selected sample.")
            ax.set_title(f"{title} | {_row_label(selected_row)}")
        else:
            fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            plot_fit_result_from_row(selected_row, ax=ax)
            ax.set_xlabel("Potential")
            ax.set_ylabel("Current")
            ax.set_title(f"{title} | {_row_label(selected_row)}")
            ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Selected Fit")
        if selected_row is None:
            st.dataframe(_empty_result_summary("No result for this frequency/channel/sample."), hide_index=True)
        else:
            st.dataframe(_result_summary(selected_row), hide_index=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    st.subheader("Signal Trend")
    signal_scope = display_results
    if selected_channel is not None and "channel" in signal_scope.columns:
        signal_scope = signal_scope[signal_scope["channel"] == selected_channel]
    if selected_hz is not None and "hz" in signal_scope.columns:
        signal_scope = signal_scope[signal_scope["hz"] == selected_hz]

    if signal_scope.empty:
        fig, ax = _blank_fit_plot("No signal data for this selection.")
    else:
        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        plot_signal_over_time(signal_scope, ax=ax, signal_col="norm_signal" if normalize else "peak")
        ax.set_ylabel("Normalized Signal" if normalize else "Peak Height")
        if "time" in signal_scope.columns or "timestamp" in signal_scope.columns:
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Index Number")
        ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
