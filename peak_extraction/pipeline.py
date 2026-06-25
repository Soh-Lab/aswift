"""Pipeline orchestration for applying signal methods to discovered SWV files."""

import os
import time

import numpy as np

from peak_extraction.config import config
from peak_extraction.extract_peaks import SUPPORTED_FITTING_METHODS, aswift_settings_from_config, fit_signal
from peak_extraction.io import discover_data_folders, get_date, list_csv_files, read_swv_csv
from peak_extraction.models import failed_fit_result


def fit_data(path, method: str):
    """Fit every channel in a single CSV file."""
    volts, currents = read_swv_csv(path)
    fit_results = []

    for channel_index, current in enumerate(currents):
        try:
            settings = aswift_settings_from_config() if method == "aswift" else None
            result = fit_signal(volts, current, method, settings=settings)
            fit_results.append((channel_index, result))
        except Exception as exc:
            filename = os.path.basename(path)
            print(f"Failed fit: {filename}, channel {channel_index}, reason: {exc}")
            fit_results.append((channel_index, failed_fit_result(method, volts, current, exc)))

    return fit_results


def get_peaks(input_dir):
    """Fit every configured method and frequency in one input directory."""
    parameters = config.parameters
    files = list_csv_files(input_dir)
    hz_values = parameters.hz_values
    methods = parameters.fitting_methods
    invalid_methods = sorted(set(methods) - set(SUPPORTED_FITTING_METHODS))
    if invalid_methods:
        allowed = ", ".join(SUPPORTED_FITTING_METHODS)
        raise ValueError(f"Unsupported fitting methods {invalid_methods}. Allowed methods: {allowed}.")

    ref_time = None
    method_results = []
    times = []

    for method in methods:
        print("Running method: " + method)
        results = []

        for hz_index, hz_set in enumerate(files):
            for file_index, file_name in enumerate(hz_set):
                path = os.path.join(input_dir, file_name)
                if hz_index == 0 and file_index == 0 and method == methods[0]:
                    ref_time = get_date(path)

                if hz_index == 0 and method == methods[0]:
                    current_time = get_date(path)
                    if current_time is not None and ref_time is not None:
                        times.append((current_time - ref_time).total_seconds() / 3600)
                    else:
                        times.append(0.0)

                volts, _ = read_swv_csv(path)
                trace_type = "full" if np.nanmin(volts) < config.parameters.full_cutoff else "partial"

                for channel_index, fit_result in fit_data(path, method=method):
                    results.append(fit_result.to_record(
                        hz=hz_values[hz_index],
                        num=file_index,
                        channel=channel_index,
                        trace_type=trace_type,
                    ))

        method_results.append(results)

    return method_results, times


def extract_peaks():
    """Recursively process every data folder under the configured input directory."""
    peak_dict = {}

    for root in discover_data_folders(config.locations.input_dir):
        print(f"Extracting peaks from {root}")
        start = time.time()
        results, times = get_peaks(input_dir=root)
        print(f"time: {time.time() - start}")
        print(f"voltammograms: {np.array(results).shape}")
        peak_dict[root] = (results, times)

    return peak_dict
