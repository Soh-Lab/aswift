"""Streamlit viewer for ASWIFT batch inputs and fit results."""

from __future__ import annotations

import ast
import hashlib
import os
import sys
import tempfile
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, cast, Protocol

import numpy as np
import pandas as pd
import streamlit as st

from ..workflow.batch import (
    fit_dataframe,
    order_results_dataframe,
    pssession_folder_to_dataframe,
    strip_mp3_suffix_from_pssession_files,
)

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
PROGRESS_UPDATE_SECONDS = 0.25
FIT_CHUNKSIZE = 16
UNKNOWN_CPU_WORKER_LIMIT = 8
LOGO_NAME = "aswift-logo.png"
FIT_TRACE_COLORS = {
    "raw": "#0072B2",
    "background": "#E69F00",
    "method": "#009E73",
    "peak": "#CC79A7",
}
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



def _viewer_log_path() -> Path:
    """Return the path used for ASWIFT Viewer diagnostic logs."""
    env_path = os.environ.get("ASWIFT_VIEWER_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / "Library" / "Logs" / "ASWIFT Viewer.log"


def _asset_path(name: str) -> Path:
    """Resolve a packaged or source-tree viewer asset path."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "assets" / name,
        here.parents[3] / "packaging" / "desktop" / "assets" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _log_unhandled_exception(exc: BaseException) -> str:
    """Write an unhandled exception traceback to the viewer log and return it."""
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        path = _viewer_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("Unhandled ASWIFT Viewer error\n")
            handle.write(details.rstrip() + "\n")
    except OSError:
        pass
    return details


def _log_viewer_message(message: str) -> None:
    """Append a diagnostic message to the viewer log if possible."""
    try:
        path = _viewer_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except OSError:
        pass




def _cache_data_if_streamlit_runtime(**kwargs):
    """Apply Streamlit caching only when running inside a Streamlit runtime."""
    def decorator(func):
        """Return the cached or uncached function depending on Streamlit runtime state."""
        if st.runtime.exists():
            return st.cache_data(**kwargs)(func)
        return func

    return decorator


def _startup_results_path() -> str:
    """Read an optional startup results path from command-line arguments."""
    return sys.argv[1] if len(sys.argv) > 1 else ""


def _parse_array_value(value: Any) -> Any:
    """Parse serialized array strings from result files when needed."""
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
    """Validate, parse, order, and annotate fit results for viewer use."""
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
    """Choose the viewer fitting backend from the selected worker count."""
    return "process" if n_workers is not None and int(n_workers) > 1 else "thread"


def _max_worker_count() -> int:
    """Return the positive worker count reported for this machine."""
    return max(1, os.cpu_count() or UNKNOWN_CPU_WORKER_LIMIT)


def _default_worker_count() -> int:
    """Choose a conservative default worker count for the current machine."""
    return min(8, os.cpu_count() or 1)


class _ProgressProtocol(Protocol):
    """Structural interface for progress reporters used during batch fitting."""
    def update(self, completed: int, total: int, label: str | None = None, *, force: bool = False) -> None: ...
    def callback(self, label: str | None = None) -> Callable[[int, int], None]: ...


class _StreamlitProgress:
    """Small wrapper around Streamlit sidebar progress UI elements."""
    def __init__(self, label: str = "Processing") -> None:
        """Initialize the helper object."""
        self.label = label
        self._last_update = 0.0
        self._text = st.sidebar.empty()
        self._bar = st.sidebar.progress(0)

    def update(self, completed: int, total: int, label: str | None = None, *, force: bool = False) -> None:
        """Record or display progress for a fitting operation."""
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
        """Return a callback compatible with batch fitting progress hooks."""
        return lambda completed, total: self.update(completed, total, label)

    def clear(self) -> None:
        """Remove progress UI elements from the sidebar."""
        self._text.empty()
        self._bar.empty()


def _read_csv_for_viewer(source) -> pd.DataFrame:
    """Read an uploaded CSV and infer whether it has headers or simple trace data."""
    header_df = _read_csv_with_encoding_fallback(source)
    if REQUIRED_FIT_COLUMNS.issubset(header_df.columns) or TRACE_COLUMNS.issubset(header_df.columns):
        return header_df
    if _simple_csv_to_trace_dataframe(header_df) is not None:
        return header_df

    if hasattr(source, "seek"):
        source.seek(0)
    return _read_csv_with_encoding_fallback(source, header=None)


def _read_csv_payload_for_viewer(payload: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes into a dataframe for viewer processing."""
    return _read_csv_for_viewer(BytesIO(payload))


def _read_table_path_for_viewer(path: Path) -> pd.DataFrame:
    """Read a startup JSON or CSV results path into a dataframe."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return _read_csv_for_viewer(path)
    raise ValueError("Startup path must be a .json or .csv file.")


def _read_csv_with_encoding_fallback(source, **kwargs) -> pd.DataFrame:
    """Try common CSV encodings until one can be parsed successfully."""
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
    """Return whether a column label can be interpreted as a numeric value."""
    try:
        float(str(value).strip())
    except ValueError:
        return False
    return True


def _simple_csv_to_trace_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Convert simple voltage/current CSV layouts into trace rows."""
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
    """Parse CSVs with one shared voltage column and one or more current columns."""
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
    """Parse CSVs containing repeated current/voltage column pairs."""
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
    _progress: _ProgressProtocol | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fit a trace dataframe with viewer progress and backend settings."""
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
    _progress: _ProgressProtocol | None = None,
) -> pd.DataFrame:
    """Load result tables directly or fit trace tables before preparing results."""
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
    """Build a stable Streamlit cache key for uploaded files and fit options."""
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
    """Build a stable Streamlit cache key for startup files and fit options."""
    return (
        "startup_results",
        method,
        int(n_workers) if n_workers is not None else None,
        signature,
        str(Path(path_text).expanduser()),
    )


def _payload_digest(payload: bytes) -> str:
    """Return a short digest for uploaded file payload bytes."""
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _load_results_from_upload_impl(
    uploads: tuple[tuple[str, bytes], ...],
    method: str,
    n_workers: int | None,
    csv_order: str,
    _progress: _ProgressProtocol | None = None,
) -> pd.DataFrame:
    """Load and optionally fit uploaded JSON or CSV files."""
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
    """Cached wrapper for uploaded viewer inputs."""
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
    """Return uploaded files in the selected processing order."""
    if csv_order == "File name":
        return tuple(sorted(uploads, key=lambda item: item[0]))
    return uploads


# noinspection PyUnusedLocal
def _load_results_from_startup_path_impl(
    path_text: str,
    method: str,
    n_workers: int | None,
    signature: tuple[str, int, int],
    _progress: _ProgressProtocol | None = None,
) -> pd.DataFrame:
    """Load and optionally fit the path supplied to the viewer at startup."""
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
    """Cached wrapper for startup path loading."""
    return _load_results_from_startup_path_impl(
        path_text,
        method,
        n_workers,
        signature,
    )


def _file_signature(path_text: str) -> tuple[str, int, int]:
    """Return a cache signature from a file path, modification time, and size."""
    path = Path(path_text).expanduser()
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _pssession_file_key(path: Path) -> tuple[str, int, int]:
    """Return the live-cache identity for one PalmSens session file."""
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _live_folder_signature_key(folder: Path, method: str) -> str:
    """Return the Streamlit session-state key for a watched PalmSens folder."""
    return f"pssession_live_signature::{folder.resolve()}::{method}"


def _current_pssession_files(folder: Path) -> list[Path]:
    """Discover current PalmSens session files, including renamed .mp3 downloads."""
    strip_mp3_suffix_from_pssession_files(folder, recursive=True)
    return sorted(folder.glob("**/*.pssession"), key=lambda path: path.stat().st_mtime)


def _pssession_folder_signature(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a signature for all PalmSens files in a watched folder."""
    return tuple(_pssession_file_key(path) for path in _current_pssession_files(folder))


def _live_folder_has_changes(folder_text: str, method: str) -> bool:
    """Return whether the watched PalmSens folder has changed since the last load."""
    folder = Path(folder_text).expanduser()
    if not folder.exists() or not folder.is_dir():
        return True
    signature = _pssession_folder_signature(folder)
    return st.session_state.get(_live_folder_signature_key(folder, method)) != signature


@st.fragment(run_every=2.0)
def _watch_live_pssession_folder(folder_text: str, method: str) -> None:
    """Trigger a Streamlit rerun when live PalmSens folder contents change."""
    if _live_folder_has_changes(folder_text, method):
        st.rerun(scope="app")


def _pssession_file_to_dataframe(path: Path) -> pd.DataFrame:
    """Load a single PalmSens session file into trace rows."""
    with tempfile.TemporaryDirectory(prefix="aswift-pssession-one-") as tmp:
        temp_path = Path(tmp) / path.name
        temp_path.write_bytes(path.read_bytes())
        df = pssession_folder_to_dataframe(temp_path.parent)
        df["folder"] = str(path.parent)
        df["source_path"] = str(path)
        df["file"] = path.name
        return df


def _fit_pssession_dataframes(
    frames: list[pd.DataFrame],
    *,
    method: str,
    n_workers: int | None,
    _progress: _ProgressProtocol | None = None,
) -> pd.DataFrame:
    """Fit trace dataframes loaded from one or more PalmSens files."""
    df = pd.concat(frames, ignore_index=True)
    group_cols = [
        col for col in ["source_path", "file", "hz", "num", "channel", "timestamp", "time"]
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
    _progress: _ProgressProtocol | None = None,
) -> pd.DataFrame:
    """Incrementally load, fit, cache, and export live PalmSens folder results."""
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
        frame["relative_folder"] = str(path.parent.relative_to(folder)) if path.parent != folder else ""
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
                if "source_path" in fitted.columns:
                    file_results = fitted[fitted["source_path"] == str(path)].copy()
                else:
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
    st.session_state[_live_folder_signature_key(folder, method)] = current_signature
    return _prepare_results(results)


def _normalize_result_time(results: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps to elapsed hours from the first acquisition."""
    if "timestamp" not in results.columns:
        return results
    normalized = results.copy()
    timestamps = cast(pd.Series, pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True))
    normalized["timestamp"] = timestamps
    if timestamps.notna().any():
        normalized["time"] = (timestamps - timestamps.min()).dt.total_seconds() / 3600
    return normalized


def _failed_pssession_file_result(path: Path, method: str, exc: Exception) -> pd.DataFrame:
    """Build a failed result row for a PalmSens file that could not be loaded."""
    return pd.DataFrame([
        {
            "folder": str(path.parent),
            "relative_folder": "",
            "source_path": str(path),
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
    """Create a compact label for the currently selected result row."""
    pieces = []
    for column in ("num", "time", "method"):
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
    """Build a display-safe summary table for one selected fit result."""
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
    """Format scalar values for display in the selected-fit summary."""
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
    """Inject small CSS adjustments used by the Streamlit viewer."""
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
        .js-plotly-plot .modebar-btn {
            color: #334155 !important;
            background: rgba(255, 255, 255, 0.9) !important;
        }
        .js-plotly-plot .modebar-btn svg path {
            fill: #334155 !important;
        }
        .js-plotly-plot .modebar-btn:hover,
        .js-plotly-plot .modebar-btn.active {
            color: #0f172a !important;
            background: rgba(226, 232, 240, 0.95) !important;
        }
        .js-plotly-plot .modebar-btn:hover svg path,
        .js-plotly-plot .modebar-btn.active svg path {
            fill: #0f172a !important;
        }
        @media (prefers-color-scheme: dark) {
            .js-plotly-plot .modebar-btn {
                color: #e5e7eb !important;
                background: rgba(15, 23, 42, 0.85) !important;
            }
            .js-plotly-plot .modebar-btn svg path {
                fill: #e5e7eb !important;
            }
            .js-plotly-plot .modebar-btn:hover,
            .js-plotly-plot .modebar-btn.active {
                color: #ffffff !important;
                background: rgba(51, 65, 85, 0.95) !important;
            }
            .js-plotly-plot .modebar-btn:hover svg path,
            .js-plotly-plot .modebar-btn.active svg path {
                fill: #ffffff !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _select_sample_position(row_count: int) -> int:
    """Render the sample selector and return the chosen sample position."""
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
    """Render normalization controls and return the selected index range."""
    enabled = st.sidebar.checkbox("Normalize", value=False)
    if not enabled:
        return False, 0, 0

    max_idx = max(int(row_count) - 1, 0)
    if max_idx == 0:
        st.sidebar.caption("Normalization range: 0")
        return True, 0, 0

    start, end = st.sidebar.slider(
        "Normalization range",
        min_value=0,
        max_value=max_idx,
        value=(0, max_idx),
    )
    if start > end:
        start, end = end, start
    return True, int(start), int(end)


def _add_empty_norm_signal(results: pd.DataFrame) -> pd.DataFrame:
    """Return a results dataframe with an empty normalized-signal column."""
    with_norm = results.copy()
    with_norm["norm_signal"] = np.nan
    with_norm["normalization_basis"] = False
    return with_norm


def _ensure_norm_signal(results: pd.DataFrame) -> pd.DataFrame:
    """Ensure a results dataframe has a normalized-signal column."""
    with_norm = results.copy()
    if "norm_signal" not in with_norm.columns:
        with_norm["norm_signal"] = np.nan
    if "normalization_basis" not in with_norm.columns:
        with_norm["normalization_basis"] = False
    return with_norm


def _add_norm_signal_by_peak_range(results: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Normalize peak heights locally by folder, method, frequency, and channel."""
    if "peak" not in results.columns or results.empty:
        return _add_empty_norm_signal(results)

    normalized = _add_empty_norm_signal(results)
    added_folder_group = False
    if {"relative_folder", "folder"} & set(normalized.columns):
        normalized["__norm_folder"] = normalized.apply(_trend_folder_label, axis=1)
        added_folder_group = True

    group_cols = [col for col in ("__norm_folder", "method", "hz", "channel") if col in normalized.columns]
    groups = normalized.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, normalized)]
    sort_cols = [col for col in ("timestamp", "time", "num", "file") if col in normalized.columns]

    for _, group in groups:
        ordered = group.sort_values(sort_cols) if sort_cols else group
        if ordered.empty:
            continue
        lo = min(max(int(start_idx), 0), len(ordered) - 1)
        hi = min(max(int(end_idx), lo), len(ordered) - 1)
        basis_index = ordered.iloc[lo:hi + 1].index
        normalized.loc[basis_index, "normalization_basis"] = True
        reference = pd.to_numeric(ordered.iloc[lo:hi + 1]["peak"], errors="coerce").mean()
        if not np.isfinite(reference) or reference == 0:
            continue
        normalized.loc[group.index, "norm_signal"] = (
            pd.to_numeric(normalized.loc[group.index, "peak"], errors="coerce") / reference
        )

    if added_folder_group:
        normalized = normalized.drop(columns=["__norm_folder"])
    return normalized


def _normalization_row_count(results: pd.DataFrame) -> int:
    """Return the largest local group size used by normalization controls."""
    if results.empty:
        return 0

    data = results.copy()
    group_cols: list[str] = [col for col in ("method", "hz", "channel") if col in data.columns]
    if {"relative_folder", "folder"} & set(data.columns):
        data["__norm_folder"] = data.apply(_trend_folder_label, axis=1)
        group_cols.insert(0, "__norm_folder")

    if not group_cols:
        return len(data)
    return int(data.groupby(group_cols, dropna=False, sort=False).size().max())


def _select_global_value(label: str, data: pd.DataFrame, column: str) -> Any:
    """Render a sidebar selectbox for one optional result column."""
    if column not in data.columns:
        return None

    values = sorted(data[column].dropna().unique().tolist())
    if not values:
        return None

    return st.sidebar.selectbox(label, values)


def _select_folder_label(data: pd.DataFrame) -> str | None:
    """Render a sidebar subfolder selector for nested result data."""
    if data.empty or not ({"relative_folder", "folder"} & set(data.columns)):
        return None

    labels = sorted(data.apply(_trend_folder_label, axis=1).dropna().unique().tolist(), key=str)
    if not labels:
        return None

    selected = st.sidebar.selectbox("Subfolder", ["All folders", *labels])
    return None if selected == "All folders" else str(selected)


def _sample_key_columns(data: pd.DataFrame) -> list[str]:
    """Choose columns that identify a unique sample selection."""
    for columns in (["num"], ["time"], ["timestamp"], ["file"]):
        if all(column in data.columns for column in columns):
            return columns
    return []


def _sample_options(data: pd.DataFrame) -> pd.DataFrame:
    """Build the ordered sample choices available for the current filters."""
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
    selected_folder: str | None = None,
    selected_channel: Any = None,
    selected_hz: Any = None,
) -> pd.DataFrame:
    """Filter result rows by selected folder, channel, and frequency."""
    filtered = results
    if selected_folder is not None and {"relative_folder", "folder"} & set(filtered.columns):
        filtered = filtered[filtered.apply(_trend_folder_label, axis=1) == selected_folder]
    if selected_channel is not None and "channel" in filtered.columns:
        filtered = filtered[filtered["channel"] == selected_channel]
    if selected_hz is not None and "hz" in filtered.columns:
        filtered = filtered[filtered["hz"] == selected_hz]
    return filtered


def _filter_selected_results(
    results: pd.DataFrame,
    *,
    selected_folder: str | None = None,
    selected_channel: Any = None,
    selected_hz: Any = None,
    selected_sample: pd.Series | None = None,
) -> pd.DataFrame:
    """Filter result rows down to the selected folder, channel, frequency, and sample."""
    filtered = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
    )
    if selected_sample is not None:
        for column, value in selected_sample.items():
            if cast(Any, column).startswith("__") or column not in filtered.columns:
                continue
            filtered = filtered[filtered[column] == value]
    return filtered


def _trace_arrays_from_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract finite x/y trace arrays from a result row."""
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


def _profile_array_from_row(row: pd.Series, column: str, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    """Extract a profile array from a row when it matches the raw trace shape."""
    if column not in row.index:
        return None
    try:
        values = np.asarray(_parse_array_value(row[column]), dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != expected_shape:
        return None
    return values


def _selected_fit_figure(row: pd.Series | None, *, title: str, empty_message: str):
    """Build the interactive Plotly figure for the currently selected fit."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if row is None:
        fig.add_annotation(text=empty_message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(height=430, title=title, xaxis={"visible": False}, yaxis={"visible": False})
        return fig

    arrays = _trace_arrays_from_row(row)
    if arrays is None:
        fig.add_annotation(
            text="No trace data for the selected sample.",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
        fig.update_layout(height=430, title=title, xaxis={"visible": False}, yaxis={"visible": False})
        return fig

    x, raw = arrays
    fig.add_trace(
        go.Scatter(
            x=x,
            y=raw,
            mode="lines",
            name="Raw data",
            line={"color": FIT_TRACE_COLORS["raw"], "width": 2},
        )
    )

    background = _profile_array_from_row(row, "background_profile", raw.shape)
    peak_profile = _profile_array_from_row(row, "peak_profile", raw.shape)
    fitted = _profile_array_from_row(row, "fitted_signal", raw.shape)
    if fitted is None and background is not None and peak_profile is not None:
        fitted = background + peak_profile

    if background is not None and np.isfinite(background).any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=background,
                mode="lines",
                name="Background",
                line={"color": FIT_TRACE_COLORS["background"], "width": 2},
            )
        )
    if fitted is not None and np.isfinite(fitted).any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fitted,
                mode="lines",
                name="Method fit",
                line={"color": FIT_TRACE_COLORS["method"], "width": 2.5},
            )
        )

    peak_voltage = row.get("peak_voltage")
    peak = row.get("peak")
    peak_background = row.get("background")
    if pd.notna(peak_voltage) and pd.notna(peak) and pd.notna(peak_background):
        peak_top = float(peak_background) + float(peak)
        fig.add_trace(
            go.Scatter(
                x=[float(peak_voltage), float(peak_voltage)],
                y=[float(peak_background), peak_top],
                mode="lines+markers",
                name="Peak height",
                line={"color": FIT_TRACE_COLORS["peak"], "width": 2, "dash": "dash"},
                marker={"color": FIT_TRACE_COLORS["peak"], "size": 8},
            )
        )

    y_range = _figure_y_range(fig)
    fig.update_layout(
        height=430,
        title={"text": title, "y": 0.98, "x": 0.01, "xanchor": "left", "yanchor": "top"},
        hovermode="closest",
        margin={"l": 10, "r": 20, "t": 110, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "left", "x": 0},
        xaxis={"title": "Potential", "range": _range_with_buffer(pd.Series(x))},
        yaxis={"title": "Current", "range": y_range},
    )
    return fig


def _empty_result_summary(message: str) -> pd.DataFrame:
    """Build a one-row summary table for an empty selected-fit state."""
    return pd.DataFrame({"field": ["result"], "value": [message]})


def _download_results(results: pd.DataFrame) -> None:
    """Render the results CSV download button."""
    st.sidebar.header("Download")
    st.sidebar.download_button(
        "Download results CSV",
        data=_results_csv_bytes(results),
        file_name="aswift_fit_results.csv",
        mime="text/csv",
    )


def _results_csv_bytes(results: pd.DataFrame) -> bytes:
    """Serialize downloadable results to CSV bytes."""
    return _downloadable_results(results).to_csv(index=False).encode("utf-8")


def _download_scope_results(
    results: pd.DataFrame,
    *,
    selected_folder: str | None,
    normalize: bool,
    norm_start: int,
    norm_end: int,
) -> pd.DataFrame:
    """Prepare downloadable results across all frequencies in the selected folder."""
    download_scope = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=None,
        selected_hz=None,
    )
    if normalize:
        return _add_norm_signal_by_peak_range(download_scope, norm_start, norm_end)
    return _add_empty_norm_signal(download_scope)


def _downloadable_results(results: pd.DataFrame) -> pd.DataFrame:
    """Remove internal arrays and normalize columns before result download."""
    download = _ensure_norm_signal(results)
    drop_cols = [column for column in DOWNLOAD_DROP_COLUMNS if column in download.columns]
    download = download.drop(columns=drop_cols)
    return _order_download_columns(download)


def _order_download_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Place normalized signal and basis flag next to peak height in downloaded results."""
    if "peak" not in results.columns or "norm_signal" not in results.columns:
        return results

    adjacent_columns = [column for column in ("norm_signal", "normalization_basis") if column in results.columns]
    columns = [column for column in results.columns if column not in adjacent_columns]
    peak_idx = columns.index("peak")
    columns[peak_idx + 1:peak_idx + 1] = adjacent_columns
    return results.loc[:, columns]


def _trend_metric_options(results: pd.DataFrame, *, normalize: bool) -> dict[str, str]:
    """Return trend metrics available for the current result table."""
    options: dict[str, str] = {}
    signal_col = "norm_signal" if normalize and "norm_signal" in results.columns else "peak"
    if signal_col in results.columns:
        options["Peak height" if signal_col == "peak" else "Normalized peak height"] = signal_col
    if "peak_voltage" in results.columns:
        options["Peak voltage (V)"] = "peak_voltage"
    if "fw_prominence" in results.columns:
        options["Peak width (mV)"] = "fw_prominence"
    return options


def _trend_x_column(data: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    """Choose and prepare the x-axis column for trend plotting."""
    data = data.copy()
    if "time" in data.columns and pd.to_numeric(data["time"], errors="coerce").notna().any():
        data["__trend_x"] = pd.to_numeric(data["time"], errors="coerce")
        return data, "__trend_x", "Time (hours)", "quantitative"
    if "timestamp" in data.columns:
        timestamps = cast(pd.Series, pd.to_datetime(data["timestamp"], errors="coerce"))
        if timestamps.notna().any():
            data["__trend_x"] = timestamps
            return data, "__trend_x", "Timestamp", "temporal"

    sort_cols = [col for col in ("num", "file") if col in data.columns]
    if sort_cols:
        data = data.sort_values(sort_cols)
    group_cols = [col for col in ("hz", "channel") if col in data.columns]
    data["__trend_x"] = (
        data.groupby(group_cols, dropna=False, sort=False).cumcount()
        if group_cols
        else np.arange(len(data))
    )
    return data, "__trend_x", "Sample", "quantitative"


def _format_channel_label(value: Any) -> str:
    """Format channel values for trend legends and hover text."""
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(cast(Any, value))


def _trend_folder_label(row: pd.Series) -> str:
    """Return the folder label used for trend grouping."""
    relative_folder = row.get("relative_folder")
    if isinstance(relative_folder, str) and relative_folder.strip():
        return relative_folder

    folder = row.get("folder")
    if isinstance(folder, str) and folder.strip():
        name = Path(folder).name
        return name or folder

    return "All data"


def _range_with_buffer(values: pd.Series, *, fraction: float = 0.05) -> list[float] | None:
    """Return a numeric plot range with a small visual buffer."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None
    low = float(numeric.min())
    high = float(numeric.max())
    if low == high:
        buffer = max(abs(low) * fraction, 1.0)
    else:
        buffer = (high - low) * fraction
    return [low - buffer, high + buffer]


def _datetime_range_with_buffer(values: pd.Series, *, fraction: float = 0.05) -> list[pd.Timestamp] | None:
    """Return a datetime plot range with a small visual buffer."""
    timestamps = pd.to_datetime(values, errors="coerce")
    timestamps = timestamps[timestamps.notna()]
    if timestamps.empty:
        return None
    low = timestamps.min()
    high = timestamps.max()
    if low == high:
        buffer = pd.Timedelta(minutes=30)
    else:
        buffer = (high - low) * fraction
    return [low - buffer, high + buffer]


def _figure_y_range(fig) -> list[float] | None:
    """Compute a buffered y-axis range from all numeric traces in a Plotly figure."""
    arrays = []
    for trace in fig.data:
        values = getattr(trace, "y", None)
        if values is None:
            continue
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.size:
            arrays.append(arr.ravel())
    if not arrays:
        return None
    return _range_with_buffer(pd.Series(np.concatenate(arrays)))


def _trend_hover_text(row: pd.Series, metric_label: str) -> str:
    """Build hover text for one peak-trend point."""
    pieces = [
        f"{metric_label}: {float(row['__metric']):.6g}",
        f"Folder: {row.get('__folder', 'All data')}",
        f"Channel: {row.get('__channel', 'signal')}",
    ]
    for column, title in [
        ("time", "Time"),
        ("timestamp", "Timestamp"),
        ("hz", "Frequency"),
        ("num", "Sample"),
        ("file", "File"),
        ("method", "Method"),
        ("success", "Success"),
        ("error", "Error"),
    ]:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        pieces.append(f"{title}: {value}")
    return "<br>".join(pieces)


def _interactive_signal_trend(
    results: pd.DataFrame,
    *,
    selected_hz: Any,
    normalize: bool,
) -> None:
    """Render the interactive Plotly peak-trend chart."""
    import plotly.graph_objects as go

    st.subheader("Peak Trends")
    metric_options = _trend_metric_options(results, normalize=normalize)
    if not metric_options:
        st.info("No trend metrics are available for this dataset.")
        return

    metric_col_left, _ = st.columns([1, 2])
    with metric_col_left:
        metric_label = st.selectbox("Trend metric", list(metric_options))
    metric_col = metric_options[metric_label]

    trend_scope = results.copy()
    if selected_hz is not None and "hz" in trend_scope.columns:
        trend_scope = trend_scope[trend_scope["hz"] == selected_hz]

    if not trend_scope.empty:
        trend_scope = trend_scope.copy()
        trend_scope["__folder"] = trend_scope.apply(_trend_folder_label, axis=1)

    trend_scope = trend_scope.copy()
    trend_scope[metric_col] = pd.to_numeric(trend_scope[metric_col], errors="coerce")
    trend_scope = trend_scope[trend_scope[metric_col].notna()]
    if trend_scope.empty:
        st.info("No signal data for this selection.")
        return

    chart_data, x_col, x_title, x_type = _trend_x_column(trend_scope)
    chart_data["__channel"] = (
        chart_data["channel"].map(_format_channel_label)
        if "channel" in chart_data.columns
        else "signal"
    )
    if "__folder" not in chart_data.columns:
        chart_data["__folder"] = chart_data.apply(_trend_folder_label, axis=1)
    chart_data["__series"] = chart_data["__folder"].astype(str) + " | channel " + chart_data["__channel"].astype(str)
    scale = 1000.0 if metric_col == "fw_prominence" else 1.0
    chart_data["__metric"] = chart_data[metric_col] * scale
    chart_data["__hover"] = chart_data.apply(lambda row: _trend_hover_text(row, metric_label), axis=1)

    fig = go.Figure()
    for series, group in chart_data.groupby("__series", dropna=False, sort=True):
        group = group.sort_values(x_col)
        folder = str(group["__folder"].iloc[0])
        channel = str(group["__channel"].iloc[0])
        trace_name = f"{folder} | channel {channel}" if folder != "All data" else f"channel {channel}"
        fig.add_trace(
            go.Scatter(
                x=group[x_col],
                y=group["__metric"],
                mode="lines+markers",
                name=trace_name,
                legendgroup=folder,
                hovertext=group["__hover"],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    x_range = (
        _datetime_range_with_buffer(chart_data[x_col])
        if x_type == "temporal"
        else _range_with_buffer(chart_data[x_col])
    )
    y_range = _range_with_buffer(chart_data["__metric"])
    fig.update_layout(
        height=420,
        dragmode="zoom",
        hovermode="closest",
        margin={"l": 10, "r": 240, "t": 10, "b": 10},
        legend={
            "title": {"text": "Folder / channel"},
            "orientation": "v",
            "yanchor": "top",
            "y": 0.9,
            "xanchor": "left",
            "x": 1.02,
        },
        xaxis={"title": x_title, "range": x_range},
        yaxis={"title": metric_label, "range": y_range},
    )
    chart_config = {
        "displaylogo": False,
        "displayModeBar": True,
        "scrollZoom": True,
        "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"],
    }
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=chart_config,
    )


def _run_app() -> None:
    """Run the Streamlit ASWIFT results viewer application."""
    logo_path = _asset_path(LOGO_NAME)
    st.set_page_config(page_title="SWV Fit Results Viewer", page_icon=str(logo_path), layout="wide")
    _inject_style()
    st.title("SWV Fit Results Viewer")

    input_section = st.sidebar.expander("Input", expanded=True)
    startup_path = _startup_results_path()
    if startup_path:
        source_kind = "Startup JSON/CSV"
        input_section.caption(f"Loaded at startup: {Path(startup_path).name}")
    else:
        source_kind = input_section.radio(
            "Source",
            ["Upload JSON/CSV", "PalmSens .pssession"],
        )
    method = input_section.selectbox("Fit method", ["aswift", "poly_linear"])
    max_workers = _max_worker_count()
    n_workers = input_section.number_input(
        "Workers",
        min_value=1,
        max_value=max_workers,
        value=_default_worker_count(),
        step=1,
    )
    upload_order = input_section.selectbox("Upload order", ["Uploaded order", "File name"])

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
            uploaded_files = input_section.file_uploader(
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
            folder_text = input_section.text_input(
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
                input_section.caption(f"Selected: {folder.name}")
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

    selected_folder = _select_folder_label(results)
    folder_scope = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=None,
        selected_hz=None,
    )
    selected_channel = _select_global_value("Channel", folder_scope, "channel")
    channel_scope = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=selected_channel,
        selected_hz=None,
    )
    selected_hz = _select_global_value("Frequency (Hz)", channel_scope, "hz")

    sample_scope = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
    )
    sample_options = _sample_options(sample_scope)
    selected_position = _select_sample_position(len(sample_options))
    selected_sample = sample_options.iloc[selected_position] if not sample_options.empty else None
    trend_scope = _filter_global_selection(
        results,
        selected_folder=selected_folder,
        selected_channel=None,
        selected_hz=selected_hz,
    )
    normalize, norm_start, norm_end = _normalization_controls(_normalization_row_count(trend_scope))
    display_results = (
        _add_norm_signal_by_peak_range(trend_scope, norm_start, norm_end)
        if normalize
        else _add_empty_norm_signal(trend_scope)
    )
    download_results = _download_scope_results(
        results,
        selected_folder=selected_folder,
        normalize=normalize,
        norm_start=norm_start,
        norm_end=norm_end,
    )
    _download_results(download_results)

    filtered = _filter_selected_results(
        display_results,
        selected_folder=selected_folder,
        selected_channel=selected_channel,
        selected_hz=selected_hz,
        selected_sample=selected_sample,
    )
    sort_cols = [col for col in ("time", "timestamp", "num", "file", "method") if col in filtered.columns]
    if sort_cols and not filtered.empty:
        filtered = filtered.sort_values(sort_cols)
    selected_row: pd.Series | None = filtered.iloc[0] if not filtered.empty else None

    title_parts = []
    if selected_folder is not None:
        title_parts.append(selected_folder)
    if selected_channel is not None:
        title_parts.append(f"channel {selected_channel}")
    if selected_hz is not None:
        title_parts.append(f"{selected_hz} Hz")
    title = " | ".join(title_parts) or "selected trace"

    left, right = st.columns([2.4, 1])

    with left:
        figure_title = title if selected_row is None else f"{title} | {_row_label(selected_row)}"
        fig = _selected_fit_figure(
            selected_row,
            title=figure_title,
            empty_message="No result for the selected sample.",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"],
            },
        )

    with right:
        st.subheader("Selected Fit")
        if selected_row is None:
            st.dataframe(_empty_result_summary("No result for this frequency/channel/sample."), hide_index=True)
        else:
            st.dataframe(_result_summary(selected_row), hide_index=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    _interactive_signal_trend(
        display_results,
        selected_hz=selected_hz,
        normalize=normalize,
    )


def main() -> None:
    """Run the command-line entry point for this module."""
    try:
        _run_app()
    except Exception as exc:
        details = _log_unhandled_exception(exc)
        st.error("ASWIFT Viewer hit an unexpected error.")
        st.caption(f"Diagnostic log: {_viewer_log_path()}")
        with st.expander("Show technical details"):
            st.code(details, language="python")
        st.stop()


if __name__ == "__main__":
    main()
