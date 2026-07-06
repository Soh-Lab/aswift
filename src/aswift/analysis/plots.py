"""Matplotlib plotting helpers for SWV fit results and batch datasets."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aswift.peak_extraction.models import FitResult
from aswift.workflow.batch import fit_result_from_row, order_results_dataframe


def plot_signal_over_time(
    results_df: pd.DataFrame,
    *,
    ax=None,
    signal_col: str = "peak",
    time_col: str = "time",
    group_cols: Sequence[str] = ("hz", "channel"),
):
    """Plot fitted peak signal across ordered samples.

    If a relative `time` column is absent, timestamps are converted to a
    relative numeric axis from the first scan. Lines are split by the requested
    grouping columns.
    """
    import matplotlib.pyplot as plt

    if signal_col not in results_df.columns:
        raise ValueError(f"results_df must contain {signal_col!r}")
    data = order_results_dataframe(results_df)

    valid_group_cols = [col for col in group_cols if col in data.columns]
    if not valid_group_cols:
        valid_group_cols = ["method"] if "method" in data.columns else []

    x_col = time_col
    x_label = "Time"
    if x_col not in data.columns:
        if "timestamp" in data.columns:
            x_col = "__relative_time"
            timestamps = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
            data[x_col] = (timestamps - timestamps.min()).dt.total_seconds() / 3600
        else:
            x_col = "__sample_index"
            x_label = "Index Number"
            if valid_group_cols:
                data[x_col] = data.groupby(valid_group_cols, dropna=False, sort=False).cumcount()
            else:
                data[x_col] = np.arange(len(data))
    elif time_col != "time":
        x_label = "Index Number"
    else:
        x_label = "Time"

    if ax is None:
        _, ax = plt.subplots()

    if valid_group_cols:
        grouped = data.groupby(valid_group_cols, dropna=False, sort=True)
    else:
        grouped = [(("signal",), data)]

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = ", ".join(
            f"{col}={value}" for col, value in zip(valid_group_cols, group_key)
        ) if valid_group_cols else "signal"
        ax.plot(group[x_col], group[signal_col], marker="o", label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Signal")
    ax.legend()
    return ax


def _fit_plot_mask(result: FitResult) -> NDArray[np.bool_]:
    mask = np.isfinite(result.fitted_current)
    if result.method != "aswift":
        return mask

    peak_window = result.params.get("peak_window")
    if peak_window is None:
        return np.isfinite(result.peak_profile)

    start, end = peak_window
    window_mask = np.zeros(result.volts.size, dtype=bool)
    window_mask[max(int(start), 0):min(int(end), result.volts.size)] = True
    return window_mask & mask


def plot_fit_result(result: FitResult, ax=None):
    """Plot raw current, background, fitted signal, and peak height."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(result.volts, result.current, label="data", color="tab:blue")
    ax.plot(result.volts, result.background_profile, label="background", color="black")
    fit_mask = _fit_plot_mask(result)
    fit_label = "fit (peak region)" if result.method == "aswift" else "fit"
    ax.plot(result.volts[fit_mask], result.fitted_current[fit_mask], label=fit_label, color="tab:red")
    if result.success and result.peak_index >= 0:
        ax.vlines(
            result.peak_voltage,
            result.peak_background,
            result.peak_background + result.peak_signal,
            color="tab:gray",
            linestyle="--",
            label=f"peak = {result.peak_signal:.3g}",
        )
    ax.set_xlabel("Input")
    ax.set_ylabel("Signal")
    ax.legend()
    return ax


def plot_fit_result_from_row(row: pd.Series | dict[str, Any], ax=None):
    """Plot one row from a batch results dataframe."""
    return plot_fit_result(fit_result_from_row(row), ax=ax)
