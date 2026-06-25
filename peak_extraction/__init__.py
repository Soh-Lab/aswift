"""ASWIFT peak fitting utilities for square-wave voltammetry."""

from peak_extraction.extract_peaks import aswift_fit, fit_signal, poly_linear_fit
from peak_extraction.models import AswiftSettings, FitResult, PolyLinearSettings

_BATCH_EXPORTS = {
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
    "plot_fit_result",
    "plot_fit_result_from_row",
    "plot_signal_over_time",
    "pssession_folder_to_dataframe",
    "results_to_signal_table",
    "strip_mp3_suffix_from_pssession_files",
}

__all__ = [
    "AswiftSettings",
    "FitResult",
    "PolyLinearSettings",
    "SwvTrace",
    "aswift_fit",
    "dataframe_to_traces",
    "fit_dataframe",
    "fit_pssession_folder",
    "fit_results_to_dataframe",
    "fit_signal",
    "fit_traces",
    "fit_result_from_row",
    "formatted_csvs_to_dataframe",
    "long_form_to_trace_dataframe",
    "order_results_dataframe",
    "order_swv_dataframe",
    "plot_fit_result",
    "plot_fit_result_from_row",
    "plot_signal_over_time",
    "poly_linear_fit",
    "pssession_folder_to_dataframe",
    "results_to_signal_table",
    "strip_mp3_suffix_from_pssession_files",
]


def __getattr__(name: str):
    if name in _BATCH_EXPORTS:
        from peak_extraction import batch

        return getattr(batch, name)
    raise AttributeError(f"module 'peak_extraction' has no attribute {name!r}")
