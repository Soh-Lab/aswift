"""Batch conversion, fitting, and I/O helpers for SWV datasets."""

from __future__ import annotations

import os
import re
import ast
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aswift.peak_extraction.extract_peaks import SUPPORTED_FITTING_METHODS, fit_signal
from aswift.peak_extraction.models import AswiftSettings, FitResult, failed_fit_result
from aswift.workflow.io import get_date, read_swv_csv


@dataclass(frozen=True)
class SwvTrace:
    """One voltage/current trace plus metadata preserved from the source.

    Attributes:
        volts: One-dimensional voltage/potential array.
        current: One-dimensional current/signal array.
        metadata: Source fields to carry into result rows, such as file name,
            frequency, sample number, channel, or acquisition time.
    """

    volts: NDArray[np.float64]
    current: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_workers(n_workers: int | None, n_jobs: int) -> int:
    """Clamp requested worker count to a valid value for the current job count."""
    if n_jobs <= 1:
        return 1
    if n_workers is None:
        n_workers = 1
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1 or None")
    return min(int(n_workers), n_jobs)


def _normalize_backend(parallel_backend: str) -> str:
    """Normalize and validate the requested parallel backend name."""
    backend = parallel_backend.lower()
    if backend == "pool":
        backend = "process"
    if backend not in {"thread", "process", "serial"}:
        raise ValueError("parallel_backend must be 'thread', 'process', 'pool', or 'serial'")
    return backend


def _normalize_chunksize(chunksize: int) -> int:
    """Validate the process-pool task chunk size."""
    chunksize = int(chunksize)
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1")
    return chunksize


def _parse_array_value(value: Any) -> NDArray[np.float64]:
    """Convert list-like dataframe cells into one-dimensional float arrays."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = np.fromstring(text.strip("[]"), sep=" ")
    return np.asarray(value, dtype=float)


def _is_array_cell(value: Any) -> bool:
    """Return whether a dataframe cell appears to hold a one-dimensional array."""
    if isinstance(value, str):
        text = value.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return False
        try:
            parsed = _parse_array_value(text)
        except (TypeError, ValueError):
            return False
        return parsed.ndim == 1 and parsed.size > 1
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _has_array_trace_columns(df: pd.DataFrame, voltage_col: str, current_col: str) -> bool:
    """Detect whether a dataframe already stores one trace per row."""
    if not {voltage_col, current_col}.issubset(df.columns) or df.empty:
        return False
    first = df[[voltage_col, current_col]].dropna().head(1)
    if first.empty:
        return False
    row = first.iloc[0]
    return _is_array_cell(row[voltage_col]) and _is_array_cell(row[current_col])


def dataframe_to_traces(
    df: pd.DataFrame,
    *,
    voltage_col: str = "voltage",
    current_col: str = "current",
    point_col: str = "point",
    group_cols: Sequence[str] | None = None,
    volts_array_col: str = "volts",
    signal_array_col: str = "signal",
) -> list[SwvTrace]:
    """Convert a formatted dataframe into trace objects.

    Two dataframe shapes are supported:
    - Preferred array-row format: one row per trace, with array-like
      voltage/current columns and metadata in the remaining columns.
    - Legacy long format: one row per point, with scalar voltage/current
      columns that are grouped into traces.

    Args:
        df: Input dataframe.
        voltage_col: Column containing voltages or voltage arrays.
        current_col: Column containing currents or current arrays.
        point_col: Optional point-order column for legacy long-form data.
        group_cols: Columns that identify one trace in legacy long-form data.
        volts_array_col: Legacy array-valued voltage column name.
        signal_array_col: Legacy array-valued current/signal column name.

    Returns:
        A list of ``SwvTrace`` objects in dataframe order.
    """
    if _has_array_trace_columns(df, voltage_col, current_col):
        traces = []
        metadata_cols = [c for c in df.columns if c not in {voltage_col, current_col}]
        for _, row in df.iterrows():
            traces.append(SwvTrace(
                volts=_parse_array_value(row[voltage_col]),
                current=_parse_array_value(row[current_col]),
                metadata={col: row[col] for col in metadata_cols},
            ))
        return traces

    if {volts_array_col, signal_array_col}.issubset(df.columns):
        traces = []
        metadata_cols = [c for c in df.columns if c not in {volts_array_col, signal_array_col}]
        for _, row in df.iterrows():
            traces.append(SwvTrace(
                volts=_parse_array_value(row[volts_array_col]),
                current=_parse_array_value(row[signal_array_col]),
                metadata={col: row[col] for col in metadata_cols},
            ))
        return traces

    required = {voltage_col, current_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")

    if group_cols is None:
        excluded = {voltage_col, current_col, point_col}
        group_cols = [col for col in df.columns if col not in excluded]
    if not group_cols:
        group_cols = ["__trace_id"]
        df = df.copy()
        df["__trace_id"] = 0

    sort_cols = [col for col in [*group_cols, point_col] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    traces = []
    for key, group in df.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        metadata = dict(zip(group_cols, key))
        metadata.pop("__trace_id", None)
        traces.append(SwvTrace(
            volts=group[voltage_col].to_numpy(dtype=float),
            current=group[current_col].to_numpy(dtype=float),
            metadata=metadata,
        ))

    return traces


def long_form_to_trace_dataframe(
    df: pd.DataFrame,
    *,
    voltage_col: str = "voltage",
    current_col: str = "current",
    point_col: str = "point",
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert legacy one-row-per-point SWV data into one row per trace.

    Args:
        df: Input dataframe with scalar voltage/current rows.
        voltage_col: Name of the voltage column.
        current_col: Name of the current column.
        point_col: Optional point-order column.
        group_cols: Columns that identify one trace. If omitted, all metadata
            columns except voltage/current/point are used.

    Returns:
        A dataframe with array-valued voltage/current columns.
    """
    required = {voltage_col, current_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")
    if _has_array_trace_columns(df, voltage_col, current_col):
        return df.copy()

    if group_cols is None:
        excluded = {voltage_col, current_col, point_col}
        group_cols = [col for col in df.columns if col not in excluded]
    if not group_cols:
        group_cols = ["__trace_id"]
        df = df.copy()
        df["__trace_id"] = 0

    sort_cols = [col for col in [*group_cols, point_col] if col in df.columns]
    data = df.sort_values(sort_cols) if sort_cols else df
    rows = []
    for key, group in data.groupby(list(group_cols), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.pop("__trace_id", None)
        row[voltage_col] = group[voltage_col].to_numpy(dtype=float).tolist()
        row[current_col] = group[current_col].to_numpy(dtype=float).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def fit_traces(
    traces: Sequence[SwvTrace],
    *,
    method: str = "aswift",
    settings: AswiftSettings | None = None,
    n_workers: int | None = None,
    parallel_backend: str = "thread",
    chunksize: int = 16,
    progress_callback: Callable[[int, int], None] | None = None,
    raise_errors: bool = False,
) -> list[FitResult]:
    """Fit many SWV traces while preserving input order.

    The default single-worker path avoids parallel overhead. For larger batches,
    ``parallel_backend="process"`` fits independent traces in separate Python
    processes and batches work with ``chunksize``.

    Args:
        traces: Sequence of ``SwvTrace`` objects.
        method: Fitting method name, ``"aswift"`` or ``"poly_linear"``.
        settings: Optional settings object for the selected method.
        n_workers: Worker count. ``None`` defaults to the single-worker path.
        parallel_backend: ``"serial"``, ``"thread"``, ``"process"``, or legacy
            alias ``"pool"``.
        chunksize: Process-pool task chunk size.
        progress_callback: Optional callable receiving ``(completed, total)``.
        raise_errors: If true, propagate fit errors instead of returning failed
            ``FitResult`` objects.

    Returns:
        Fitted results in the same order as ``traces``.
    """
    if method not in SUPPORTED_FITTING_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_FITTING_METHODS}")

    parallel_backend = _normalize_backend(parallel_backend)
    chunksize = _normalize_chunksize(chunksize)
    n_workers = _normalize_workers(n_workers, len(traces))
    total = len(traces)
    if parallel_backend == "serial" or n_workers == 1:
        results = []
        for idx, trace in enumerate(traces, start=1):
            results.append(_fit_one_trace(trace, method, settings, raise_errors))
            if progress_callback is not None:
                progress_callback(idx, total)
        return results

    if parallel_backend == "process":
        return _fit_traces_process_pool(
            traces,
            method=method,
            settings=settings,
            n_workers=n_workers,
            chunksize=chunksize,
            progress_callback=progress_callback,
            raise_errors=raise_errors,
        )

    results: list[FitResult | None] = [None] * len(traces)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_fit_one_trace, trace, method, settings, raise_errors): idx
            for idx, trace in enumerate(traces)
        }
        completed = 0
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)

    return [result for result in results if result is not None]


_PROCESS_METHOD: str | None = None
_PROCESS_SETTINGS: AswiftSettings | None = None
_PROCESS_RAISE_ERRORS = False


def _init_fit_process(method: str, settings: AswiftSettings | None, raise_errors: bool) -> None:
    """Initialize shared fitting settings inside a process-pool worker."""
    global _PROCESS_METHOD, _PROCESS_SETTINGS, _PROCESS_RAISE_ERRORS
    _PROCESS_METHOD = method
    _PROCESS_SETTINGS = settings
    _PROCESS_RAISE_ERRORS = raise_errors


def _fit_trace_process_task(task: tuple[int, NDArray[np.float64], NDArray[np.float64]]) -> tuple[int, FitResult]:
    """Fit one serialized trace task inside a process-pool worker."""
    idx, volts, current = task
    if _PROCESS_METHOD is None:
        raise RuntimeError("process worker was not initialized")
    try:
        result = fit_signal(volts, current, _PROCESS_METHOD, settings=_PROCESS_SETTINGS)
    except Exception as exc:
        if _PROCESS_RAISE_ERRORS:
            raise
        result = failed_fit_result(_PROCESS_METHOD, volts, current, exc)
    return idx, result


def _fit_traces_process_pool(
    traces: Sequence[SwvTrace],
    *,
    method: str,
    settings: AswiftSettings | None,
    n_workers: int,
    chunksize: int,
    progress_callback: Callable[[int, int], None] | None,
    raise_errors: bool,
) -> list[FitResult]:
    """Fit traces in a multiprocessing pool while preserving result order."""
    tasks = [
        (idx, np.asarray(trace.volts, dtype=float), np.asarray(trace.current, dtype=float))
        for idx, trace in enumerate(traces)
    ]
    results: list[FitResult | None] = [None] * len(traces)
    with mp.Pool(
        processes=n_workers,
        initializer=_init_fit_process,
        initargs=(method, settings, raise_errors),
    ) as pool:
        completed = 0
        for idx, result in pool.imap_unordered(_fit_trace_process_task, tasks, chunksize=chunksize):
            results[idx] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(tasks))
    return [result for result in results if result is not None]


def _fit_one_trace(
    trace: SwvTrace,
    method: str,
    settings: AswiftSettings | None,
    raise_errors: bool,
) -> FitResult:
    """Fit one trace and convert exceptions into failed results when requested."""
    try:
        return fit_signal(trace.volts, trace.current, method, settings=settings)
    except Exception as exc:
        if raise_errors:
            raise
        return failed_fit_result(method, trace.volts, trace.current, exc)


def fit_dataframe(
    df: pd.DataFrame,
    *,
    method: str = "aswift",
    settings: AswiftSettings | None = None,
    n_workers: int | None = None,
    parallel_backend: str = "thread",
    chunksize: int = 16,
    group_cols: Sequence[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Fit all traces in a dataframe and return result rows.

    The input may already contain one row per trace with array-valued
    voltage/current columns, or legacy one-row-per-point data that can be
    grouped into traces.

    Args:
        df: Input SWV dataframe.
        method: Fitting method name, ``"aswift"`` or ``"poly_linear"``.
        settings: Optional settings object for the selected method.
        n_workers: Worker count. ``None`` defaults to one worker.
        parallel_backend: ``"serial"``, ``"thread"``, ``"process"``, or
            ``"pool"``.
        chunksize: Process-pool task chunk size.
        group_cols: Columns that identify traces for legacy long-form data.
        progress_callback: Optional callable receiving ``(completed, total)``.

    Returns:
        A dataframe with metadata, scalar fit metrics, and array-valued profiles.
    """
    traces = dataframe_to_traces(df, group_cols=group_cols)
    fit_results = fit_traces(
        traces,
        method=method,
        settings=settings,
        n_workers=n_workers,
        parallel_backend=parallel_backend,
        chunksize=chunksize,
        progress_callback=progress_callback,
    )
    return fit_results_to_dataframe(fit_results, traces)


def fit_results_to_dataframe(results: Sequence[FitResult], traces: Sequence[SwvTrace]) -> pd.DataFrame:
    """Convert fitted results and trace metadata into a dataframe.

    Args:
        results: Fit results returned by ``fit_traces``.
        traces: Source traces whose metadata should be preserved.

    Returns:
        A JSON/CSV-friendly dataframe with scalar metrics and array-valued
        voltage, current, peak, background, and fitted-signal profiles.
    """
    rows = []
    for result, trace in zip(results, traces):
        row = {
            **trace.metadata,
            "method": result.method,
            "success": result.success,
            "error": result.error,
            "peak": result.peak_signal,
            "background": result.peak_background,
            "peak_voltage": result.peak_voltage,
            "peak_index": result.peak_index,
            "fw_prominence": result.fw_prominence,
            "voltage": result.volts.tolist(),
            "current": result.current.tolist(),
            "peak_profile": result.peak_profile.tolist(),
            "background_profile": result.background_profile.tolist(),
            "fitted_signal": result.fitted_current.tolist(),
        }
        if "peak_window" in result.params:
            start, end = result.params["peak_window"]
            row["peak_window_start"] = int(start)
            row["peak_window_end"] = int(end)
        rows.append(row)
    return pd.DataFrame(rows)


def results_to_signal_table(
    results_df: pd.DataFrame,
    *,
    calibration_lower_idx: int = 0,
    calibration_upper_idx: int | None = None,
    group_cols: Sequence[str] = ("hz", "channel"),
    order_col: str = "num",
) -> pd.DataFrame:
    """Create a compact signal/gain table from fitted result rows.

    Args:
        results_df: Result dataframe from ``fit_dataframe`` or compatible data.
        calibration_lower_idx: First row index included in the reference signal.
        calibration_upper_idx: Last row index included in the reference signal.
        group_cols: Columns used to create one signal/gain pair per group.
        order_col: Column used to order samples within each group.

    Returns:
        A dataframe containing sample order metadata plus ``signal-*`` and
        ``gain-*`` columns for each group.
    """
    if calibration_upper_idx is None:
        calibration_upper_idx = calibration_lower_idx

    metadata_cols = [col for col in ["time", "trace_type"] if col in results_df.columns]
    if order_col in results_df.columns:
        base = results_df[[order_col, *metadata_cols]].drop_duplicates(order_col).sort_values(order_col)
    else:
        base = pd.DataFrame(index=np.arange(len(results_df)))

    columns: dict[str, Any] = {}
    valid_group_cols = [col for col in group_cols if col in results_df.columns]
    if not valid_group_cols:
        valid_group_cols = ["method"]

    for group_key, group in results_df.groupby(valid_group_cols, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = "-".join(f"{col}{cast(Any, value)}" for col, value in zip(valid_group_cols, group_key))
        if order_col in group.columns:
            group = group.sort_values(order_col)

        signal = group["peak"].to_numpy(dtype=float)
        lo = calibration_lower_idx
        hi = min(calibration_upper_idx + 1, signal.size)
        reference = np.nanmean(signal[lo:hi])
        gain = signal / reference - 1 if np.isfinite(reference) and reference != 0 else np.full_like(signal, np.nan)
        columns[f"signal-{label}"] = pd.Series(signal)
        columns[f"gain-{label}"] = pd.Series(gain)

    return pd.concat([base.reset_index(drop=True), pd.DataFrame(columns)], axis=1)


def formatted_csvs_to_dataframe(
    folder: str | Path,
    *,
    hz_values: Iterable[int] | None = None,
    use_file0: bool = True,
) -> pd.DataFrame:
    """Load formatted SWV CSV files into one row per voltammogram.

    Args:
        folder: Folder containing files named like ``150hz.csv`` or
            ``150hz-1.csv``.
        hz_values: Optional frequency filter.
        use_file0: Whether files without an explicit numeric suffix should be
            treated as sample ``0``.

    Returns:
        A trace dataframe with one row per channel in each matching CSV file.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    pattern = re.compile(r"^(?P<hz>\d+)hz(?:-(?P<num>\d+))?\.csv$")
    rows = []
    files = sorted(folder.glob("*.csv"))
    hz_filter = set(hz_values) if hz_values is not None else None
    ref_time = None

    for path in files:
        match = pattern.match(path.name)
        if match is None:
            continue
        hz = int(match.group("hz"))
        if hz_filter is not None and hz not in hz_filter:
            continue
        raw_num = match.group("num")
        if raw_num is None and not use_file0:
            continue
        num = 0 if raw_num is None else int(raw_num)

        timestamp = get_date(str(path))
        if ref_time is None and timestamp is not None:
            ref_time = timestamp
        elapsed = (
            (timestamp - ref_time).total_seconds() / 3600
            if timestamp is not None and ref_time is not None
            else np.nan
        )

        volts, currents = read_swv_csv(str(path))
        for channel, current in enumerate(currents):
            rows.append({
                "folder": str(folder),
                "file": path.name,
                "hz": hz,
                "num": num,
                "channel": channel,
                "time": elapsed,
                "timestamp": timestamp,
                "voltage": volts.tolist(),
                "current": current.tolist(),
            })

    if not rows:
        raise ValueError(f"No formatted SWV CSV files found in {folder}")
    return pd.DataFrame(rows)


def pssession_folder_to_dataframe(folder: str | Path, *, recursive: bool = False) -> pd.DataFrame:
    """Load PalmSens .pssession files into one row per voltammogram.

    This optional adapter requires `pypalmsens`. It extracts potential arrays
    and net current arrays, preserving file, timestamp, frequency, and channel.
    Set ``recursive=True`` to discover files in nested subfolders while
    preserving ``relative_folder`` and ``source_path`` metadata.
    """
    try:
        # noinspection PyPackageRequirements
        import pypalmsens as ps
    except ImportError as exc:
        raise ImportError(
            "pssession support requires the optional pypalmsens package "
            f"and its .NET bridge dependencies: {exc}"
        ) from exc

    folder = Path(folder)
    strip_mp3_suffix_from_pssession_files(folder, recursive=recursive)
    pattern = "**/*.pssession" if recursive else "*.pssession"
    files = sorted(folder.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise ValueError(f"No .pssession files found in {folder}")

    rows = []
    for num, path in enumerate(files):
        measurements = ps.load_session_file(str(path))
        if len(measurements) != 1:
            continue
        measurement = measurements[0]
        tech_name = getattr(measurement.method, "id", "").lower()
        if "swv" not in tech_name:
            continue

        arrays = list(measurement.dataset.arrays())
        potential_indices = [i for i, arr in enumerate(arrays) if getattr(arr, "unit", "").lower() == "v"]
        current_indices = [
            i for i, arr in enumerate(arrays)
            if getattr(arr, "unit", "").lower() not in {"v", "s"}
            and "forward" not in getattr(arr, "name", "").lower()
            and "reverse" not in getattr(arr, "name", "").lower()
            and "time" not in getattr(arr, "name", "").lower()
        ]
        if not potential_indices or not current_indices:
            continue

        freq = _read_pssession_frequency(path)
        timestamp = _read_pssession_timestamp(path, fallback=getattr(measurement, "timestamp", None))
        device = str(getattr(measurement, "device", ""))
        relative_folder = str(path.parent.relative_to(folder)) if path.parent != folder else ""
        for channel, current_idx in enumerate(current_indices):
            prior_volts = [idx for idx in potential_indices if idx < current_idx]
            volt_idx = prior_volts[-1] if prior_volts else potential_indices[0]
            volts = np.asarray(arrays[volt_idx], dtype=float)
            current = np.asarray(arrays[current_idx], dtype=float)
            label = getattr(arrays[current_idx], "name", f"channel_{channel}")
            rows.append({
                "folder": str(path.parent),
                "relative_folder": relative_folder,
                "source_path": str(path),
                "file": path.name,
                "hz": freq,
                "num": num,
                "channel": channel,
                "label": label,
                "timestamp": timestamp,
                "device": device,
                "voltage": volts.tolist(),
                "current": current.tolist(),
            })

    if not rows:
        raise ValueError(f"No SWV traces could be read from .pssession files in {folder}")
    df = pd.DataFrame(rows)
    return order_swv_dataframe(df)


def strip_mp3_suffix_from_pssession_files(
    folder: str | Path,
    *,
    recursive: bool = False,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Rename `.pssession.mp3` files back to `.pssession`.

    Some transfer/download paths append an `.mp3` suffix to PalmSens session
    files. This helper removes only that final suffix, leaving unrelated `.mp3`
    files untouched. Existing destination files are never overwritten.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    pattern = "**/*.pssession.mp3" if recursive else "*.pssession.mp3"
    renamed: list[tuple[Path, Path]] = []
    for source in sorted(folder.glob(pattern)):
        target = source.with_suffix("")
        if target.exists():
            raise FileExistsError(f"Cannot rename {source}; destination already exists: {target}")
        renamed.append((source, target))
        if not dry_run:
            source.rename(target)

    return renamed


def fit_pssession_folder(
    folder: str | Path,
    *,
    method: str = "aswift",
    settings: AswiftSettings | None = None,
    n_workers: int | None = None,
    parallel_backend: str = "thread",
    chunksize: int = 16,
    recursive: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read `.pssession` files from a folder and fit each SWV trace.

    Returns the trace dataframe and a fitted result dataframe. The
    result rows are ordered by acquisition time, frequency, file number, and
    channel when those fields are available. Set ``recursive=True`` to include
    nested subfolders in the PalmSens file discovery step.
    """
    df = pssession_folder_to_dataframe(folder, recursive=recursive)
    group_cols = [
        col for col in ["source_path", "file", "hz", "num", "channel", "timestamp", "time"]
        if col in df.columns
    ]
    results = fit_dataframe(
        df,
        method=method,
        settings=settings,
        n_workers=n_workers,
        parallel_backend=parallel_backend,
        chunksize=chunksize,
        group_cols=group_cols,
        progress_callback=progress_callback,
    )
    return df, order_results_dataframe(results)


def order_swv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort SWV trace rows by time, frequency, trace number, and channel."""
    df = df.copy()
    if "timestamp" in df.columns:
        parsed = cast(pd.Series, pd.to_datetime(df["timestamp"], errors="coerce", utc=True))
        df["timestamp"] = parsed
        if "time" not in df.columns and parsed.notna().any():
            first = parsed.min()
            df["time"] = (parsed - first).dt.total_seconds() / 3600

    sort_cols = [
        col for col in ["timestamp", "time", "hz", "num", "channel"]
        if col in df.columns
    ]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def order_results_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Sort fitted result rows by time, frequency, trace number, and channel."""
    results_df = results_df.copy()
    if "timestamp" in results_df.columns:
        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], errors="coerce", utc=True)
    sort_cols = [
        col for col in ["timestamp", "time", "hz", "num", "channel", "method"]
        if col in results_df.columns
    ]
    return results_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else results_df


def _read_pssession_frequency(path: Path) -> float | None:
    """Read the SWV frequency from raw PalmSens session metadata."""
    try:
        content = path.read_text(encoding="utf-16", errors="ignore")
    except OSError:
        return None
    match = re.search(r"FREQ=\s*\"?(?P<freq>[-+0-9.eE]+)", content)
    return float(match.group("freq")) if match else None


def _read_pssession_timestamp(path: Path, fallback=None) -> datetime | str | None:
    """Read the UTC PalmSens timestamp from raw session metadata when present."""
    try:
        content = path.read_text(encoding="utf-16", errors="ignore")
    except OSError:
        return _parse_pssession_timestamp(fallback)

    match = re.search(r'"utctimestamp"\s*:\s*(?P<ticks>\d+)', content)
    if match:
        return _datetime_from_dotnet_ticks(int(match.group("ticks")))

    match = re.search(r'"timestamp"\s*:\s*(?P<ticks>\d+)', content)
    if match:
        return _datetime_from_dotnet_ticks(int(match.group("ticks")))

    return _parse_pssession_timestamp(fallback)


def _datetime_from_dotnet_ticks(ticks: int) -> datetime:
    """Convert .NET ticks since 0001-01-01 to a UTC datetime."""
    return datetime(1, 1, 1, tzinfo=UTC) + timedelta(microseconds=ticks // 10)


def _parse_pssession_timestamp(value) -> datetime | str | None:
    """Parse PalmSens timestamp values from known text formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    text = str(value).replace("\u202f", " ").strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return text


def fit_result_from_row(row: pd.Series | dict[str, Any]) -> FitResult:
    """Rebuild a FitResult from one row of a batch results dataframe.

    Batch fitting stores array-valued columns so results can be saved,
    filtered, and plotted later. This helper converts one selected row back
    into the same object returned by ``aswift_fit`` or ``poly_linear_fit``.
    """
    voltage_key = "voltage" if "voltage" in row else "volts"
    current_key = "current" if "current" in row else "signal"
    required = {
        "method",
        "peak",
        "background",
        "peak_voltage",
        "peak_index",
        "peak_profile",
        "background_profile",
        "success",
        voltage_key,
        current_key,
    }
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"Result row is missing required columns: {sorted(missing)}")

    error = row.get("error")
    if pd.isna(error):
        error = None

    params = {}
    # noinspection PyPackages
    if "peak_window_start" in row and "peak_window_end" in row:
        start = row.get("peak_window_start")
        end = row.get("peak_window_end")
        if pd.notna(start) and pd.notna(end):
            # noinspection PyTypeChecker
            params["peak_window"] = (int(start), int(end))

    return FitResult(
        method=str(row["method"]),
        volts=_parse_array_value(row[voltage_key]),
        current=_parse_array_value(row[current_key]),
        peak_signal=float(row["peak"]),
        peak_background=float(row["background"]),
        peak_voltage=float(row["peak_voltage"]),
        peak_index=int(row["peak_index"]),
        fw_prominence=float(row.get("fw_prominence", np.nan)),
        peak_profile=_parse_array_value(row["peak_profile"]),
        background_profile=_parse_array_value(row["background_profile"]),
        params=params,
        success=bool(row["success"]),
        error=error,
    )
