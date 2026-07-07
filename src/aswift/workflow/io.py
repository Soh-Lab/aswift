"""CSV loading and timestamp helpers for SWV data."""

import csv
from datetime import datetime


def read_swv_csv(path):
    """Read one potentiostat CSV into voltage and channel-current arrays.

    Args:
        path: CSV path in the formatted SWV export layout.

    Returns:
        ``(volts, currents)`` where ``volts`` is one-dimensional and
        ``currents`` has shape ``(n_channels, n_points)``.
    """

    # Lazy import pandas here instead of in the imports so that core code that doesn't
    # need this function doesn't pick up pandas by default.
    import pandas as pd
    
    df = (
        pd.read_csv(path, skiprows=6, encoding="utf-16")  # type: ignore[union-attr]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    volts = df.iloc[:, 0].astype(float).to_numpy()
    currents = df.iloc[:, 1::2].dropna(how="all").astype(float).to_numpy().T
    return volts, currents


def get_date(file):
    """Read the measurement timestamp from a formatted SWV CSV.

    Args:
        file: CSV path to inspect.

    Returns:
        A ``datetime`` when the expected metadata row is present and parseable,
        otherwise ``None``.
    """
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
