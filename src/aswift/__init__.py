"""ASWIFT peak fitting utilities for square-wave voltammetry"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aswift")
except importlib.metadata.PackageNotFoundError:
    # Fallback if the package is imported directly from the source directory
    # without being formally installed via pip
    __version__ = "0.1.0-dev"

from .peak_extraction.extract_peaks import aswift_fit, fit_signal, poly_linear_fit
from .peak_extraction.models import AswiftSettings, FitResult, PolyLinearSettings

# Types in workflow.batch and analysis.plots are lazy-loaded via __getattr__ below to
# prevent large, transitively-loaded libraries imported there (pandas, matplotlib) from
# being loaded by default. However, static type checking gets confused by this, so we
# import these only when this is loaded by a type checker.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow.batch import (SwvTrace, dataframe_to_traces, fit_dataframe,
        fit_pssession_folder, fit_results_to_dataframe, fit_traces, fit_result_from_row,
        formatted_csvs_to_dataframe, long_form_to_trace_dataframe, order_results_dataframe,
        order_swv_dataframe, pssession_folder_to_dataframe, results_to_signal_table,
        strip_mp3_suffix_from_pssession_files,
    )
    from .analysis.plots import (plot_fit_result, plot_fit_result_from_row, plot_signal_over_time)

_CORE_EXPORTS = [
    "AswiftSettings",
    "FitResult",
    "PolyLinearSettings",
    "aswift_fit",
    "fit_signal",
    "poly_linear_fit",
]

_BATCH_EXPORTS = [
    "SwvTrace",
    "dataframe_to_traces",
    "fit_dataframe",
    "fit_pssession_folder",
    "fit_results_to_dataframe",
    "fit_traces",
    "fit_result_from_row",
    "formatted_csvs_to_dataframe",
    "long_form_to_trace_dataframe",
    "order_results_dataframe",
    "order_swv_dataframe",
    "pssession_folder_to_dataframe",
    "results_to_signal_table",
    "strip_mp3_suffix_from_pssession_files",
]

_PLOTS_EXPORTS = [
    "plot_fit_result",
    "plot_fit_result_from_row",
    "plot_signal_over_time",
]

__all__ = _CORE_EXPORTS + _BATCH_EXPORTS + _PLOTS_EXPORTS

def __getattr__(name: str):
    if name in _BATCH_EXPORTS:
        from .workflow import batch

        return getattr(batch, name)
    if name in _PLOTS_EXPORTS:
        from .analysis import plots

        return getattr(plots, name)
    raise AttributeError(f"module 'aswift' has no attribute {name!r}")
