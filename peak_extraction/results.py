"""Convert fit dictionaries into detailed and method-comparison DataFrames."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from peak_extraction.config import config
from peak_extraction.io import read_swv_csv, result_file_name


def generate_detailed_df(times, json_file):
    df = pd.DataFrame({"time": times})

    try:
        with open(json_file, "r") as file:
            results = json.load(file)
    except Exception as exc:
        raise RuntimeError("Failed to load json file") from exc

    hz_values = sorted(set(d["hz"] for d in results))
    num_values = sorted(set(d["num"] for d in results))
    trace_type_by_num = {
        d["num"]: d.get("trace_type", np.nan)
        for d in results
    }

    df["trace_type"] = [trace_type_by_num.get(num, np.nan) for num in num_values]
    channel_values = sorted(set(d["channel"] for d in results))

    hz_to_index = {hz: i for i, hz in enumerate(hz_values)}
    num_to_index = {num: i for i, num in enumerate(num_values)}
    input_dir = Path(json_file).parent

    channel_data = {}
    for channel in channel_values:
        channel_data[channel] = np.full((len(hz_values), len(num_values), 5), np.nan)

    for d in results:
        hz = d["hz"]
        num = d["num"]
        channel = d["channel"]
        arr = channel_data[channel]
        arr[hz_to_index[hz], num_to_index[num], 0] = d["peak"]
        arr[hz_to_index[hz], num_to_index[num], 4] = d["background"]

        bg_idx_val = d["bg_idx"]
        if not np.isnan(bg_idx_val):
            bg_idx = int(bg_idx_val)
            peak_arr = np.array(d["popt"][:bg_idx])
            nonzero = peak_arr != 0
            if np.any(nonzero):
                lower = np.argmax(nonzero)
                upper = len(nonzero) - np.argmax(nonzero[::-1]) - 1
                peak_idx = np.nanargmax(peak_arr)

                file_name = result_file_name(hz, num)
                volts, _ = read_swv_csv(os.path.join(input_dir, file_name))
                arr[hz_to_index[hz], num_to_index[num], 2] = volts[peak_idx]
                arr[hz_to_index[hz], num_to_index[num], 3] = volts[upper] - volts[lower]
        else:
            arr[hz_to_index[hz], num_to_index[num], 2] = d["popt"][1]
            arr[hz_to_index[hz], num_to_index[num], 3] = d["popt"][2]

    new_columns = {}
    for channel, arr in channel_data.items():
        lower_idx = config.parameters.calibration_lower_idx
        upper_idx = config.parameters.calibration_upper_idx + 1
        sb_mean = np.nanmean(arr[:, lower_idx:upper_idx, 0], axis=1)
        arr[..., 1] = arr[..., 0] / sb_mean[:, np.newaxis] - 1

        for i, hz in enumerate(hz_values):
            new_columns[f"signal-{hz}hz-ch{channel}"] = arr[i, :, 0]
            new_columns[f"peak-{hz}hz-ch{channel}"] = arr[i, :, 0] + arr[i, :, 4]
            new_columns[f"gain-{hz}hz-ch{channel}"] = arr[i, :, 1]
            new_columns[f"location-{hz}hz-ch{channel}"] = arr[i, :, 2]
            new_columns[f"width-{hz}hz-ch{channel}"] = arr[i, :, 3]
            new_columns[f"background-{hz}hz-ch{channel}"] = arr[i, :, 4]

    return pd.concat([df, pd.DataFrame(new_columns)], axis=1).copy()


def generate_methods_df(times, results):
    df = pd.DataFrame({"time": times})

    hz_values = sorted(set(d["hz"] for d in results[0]))
    num_values = sorted(set(d["num"] for d in results[0]))
    trace_type_by_num = {
        d["num"]: d.get("trace_type", np.nan)
        for method_results in results
        for d in method_results
    }

    df["trace_type"] = [trace_type_by_num.get(num, np.nan) for num in num_values]
    channel_values = sorted(set(d["channel"] for d in results[0]))
    methods = config.parameters.fitting_methods

    hz_to_index = {hz: i for i, hz in enumerate(hz_values)}
    num_to_index = {num: i for i, num in enumerate(num_values)}

    channel_data = {}
    for channel in channel_values:
        channel_data[channel] = np.full((len(hz_values), len(num_values), 2 * len(methods)), np.nan)

    for method_index, method_results in enumerate(results):
        for d in method_results:
            arr = channel_data[d["channel"]]
            arr[hz_to_index[d["hz"]], num_to_index[d["num"]], 2 * method_index] = d["peak"]

    new_columns = {}
    for channel, arr in channel_data.items():
        lower_idx = config.parameters.calibration_lower_idx
        upper_idx = config.parameters.calibration_upper_idx + 1

        for i in range(len(methods)):
            sb_mean = np.nanmean(arr[:, lower_idx:upper_idx, 2 * i], axis=1)
            arr[..., 2 * i + 1] = arr[..., 2 * i] / sb_mean[:, np.newaxis] - 1

        for i, method in enumerate(methods):
            for j, hz in enumerate(hz_values):
                new_columns[f"signal-{hz}hz-ch{channel}-{method}"] = arr[j, :, 2 * i]
                new_columns[f"gain-{hz}hz-ch{channel}-{method}"] = arr[j, :, 2 * i + 1]

    return pd.concat([df, pd.DataFrame(new_columns)], axis=1).copy()
