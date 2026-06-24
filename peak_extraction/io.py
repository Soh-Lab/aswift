"""CSV loading, folder discovery, and filename helpers for SWV data."""

import csv
import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd

from peak_extraction.config import config


_volts_array_cache = None


def read_swv_csv(path):
    """Read one potentiostat CSV into voltage and channel-current arrays."""
    df = (
        pd.read_csv(path, skiprows=6, encoding="utf-16")
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    volts = df.iloc[:, 0].astype(float).to_numpy()
    currents = df.iloc[:, 1::2].dropna(how="all").astype(float).to_numpy().T
    return volts, currents


def get_volts_array(input_dir=None):
    global _volts_array_cache

    if input_dir is None:
        input_dir = config.locations.input_dir

    if _volts_array_cache is not None:
        return np.array(_volts_array_cache)

    save_path = os.path.join(input_dir, "volts.pkl")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            _volts_array_cache = pickle.load(f)
        return np.array(_volts_array_cache)

    hz = config.parameters.hz_values[0]
    candidates = [f"{hz}hz-1.csv", f"{hz}hz.csv"]
    for file_name in candidates:
        path = os.path.join(input_dir, file_name)
        if os.path.exists(path):
            _volts_array_cache, _ = read_swv_csv(path)
            with open(save_path, "wb") as f:
                pickle.dump(_volts_array_cache, f)
            return np.array(_volts_array_cache)

    raise FileNotFoundError(f"No voltage source CSV found for {hz} Hz in {input_dir}")


def list_csv_files(folder_path):
    """Return CSV files ordered by configured frequency and replicate number."""
    files = []
    for hz in config.parameters.hz_values:
        prefix = f"{hz}hz-"
        csv_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".csv") and f.startswith(prefix)
        ]
        csv_files = sorted(csv_files, key=lambda x: int(x.split("-")[1].split(".")[0]))

        if config.parameters.use_file0:
            file0 = f"{hz}hz.csv"
            if os.path.exists(os.path.join(folder_path, file0)):
                csv_files.insert(0, file0)

        files.append(csv_files)

    min_len = min(len(file_set) for file_set in files)
    files = [file_set[:min_len] for file_set in files]
    return np.array(files)


def get_date(file):
    target = "Date and time measurement:"
    encodings = ("utf-16", "utf-8-sig")

    for encoding in encodings:
        try:
            with open(file, "r", encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue

                    if str(row[0]).strip() != target:
                        continue

                    date_str = str(row[1]).strip()
                    try:
                        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        return None
        except UnicodeError:
            continue

    return None


def discover_data_folders(input_dir):
    """Yield folders that contain configured SWV CSV filename patterns."""
    hz_pattern = "|".join(str(hz) for hz in config.parameters.hz_values)
    pattern = re.compile(rf"^({hz_pattern})hz-(\d+)\.csv$")

    for root, _, files in os.walk(input_dir):
        if any(pattern.match(f) for f in files):
            yield root


def result_file_name(hz, num):
    if config.parameters.use_file0:
        suffix = f"-{num}" if num else ""
        return f"{hz}hz{suffix}.csv"

    suffix = f"-{num + 1}" if num else "-1"
    return f"{hz}hz{suffix}.csv"
