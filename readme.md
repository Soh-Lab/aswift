# ASWIFT

ASWIFT fits square-wave voltammetry traces and returns peak signal, background,
and full fitted profiles that can be plotted directly.

## Install

```bash
pip install -r requirements.txt
```

## Fit One Trace

```python
import numpy as np
from peak_extraction import aswift_fit, poly_linear_fit

volts = np.asarray([...], dtype=float)
current = np.asarray([...], dtype=float)

result = aswift_fit(volts, current)
# or:
result = poly_linear_fit(volts, current)

print(result.peak_signal, result.peak_voltage)
```

`result` is a `FitResult` with:

- `peak_signal`: background-subtracted peak height
- `peak_background`: background current at the peak
- `peak_voltage`: voltage at the peak maximum
- `peak_profile`: fitted peak-only profile
- `background_profile`: fitted background profile
- `fitted_current`: `peak_profile + background_profile`

## Plot One Trace

```python
from peak_extraction.batch import plot_fit_result

ax = plot_fit_result(result)
```

## Fit A DataFrame

Use long-form data with one row per point:

```python
from peak_extraction.batch import fit_dataframe, results_to_signal_table

results = fit_dataframe(
    df,
    method="aswift",
    group_cols=["file", "hz", "num", "channel"],
    n_workers=4,
)

signals = results_to_signal_table(
    results,
    calibration_lower_idx=0,
    calibration_upper_idx=4,
)
```

Expected long-form columns are:

- `voltage`
- `current`
- optional `point`
- any metadata columns used for grouping, such as `file`, `hz`, `num`, `channel`

Rows can also contain array-like `volts` and `signal` columns, one row per trace.

## Load Existing Files

```python
from peak_extraction.batch import (
    fit_pssession_folder,
    formatted_csvs_to_dataframe,
    pssession_folder_to_dataframe,
    plot_signal_over_time,
)

df = formatted_csvs_to_dataframe("path/to/csv/folder", hz_values=[150, 200, 250])
results = fit_dataframe(df, method="aswift", n_workers=4)

# Optional PalmSens adapter; requires pypalmsens.
ps_df = pssession_folder_to_dataframe("path/to/pssession/folder")
ps_df, ps_results = fit_pssession_folder("path/to/pssession/folder", n_workers=4)
plot_signal_over_time(ps_results)
```

The `.pssession` workflow orders traces by timestamp, frequency, file number,
and channel when those fields are available. If files were downloaded with a
spurious `.mp3` suffix, files named `.pssession.mp3` are renamed to `.pssession`
before discovery. PalmSens UTC timestamps are used when present, and `time` is
normalized in hours so the earliest measurement is `0.0`.

## Examples

Runnable examples live in the top-level `examples/` folder:

- `examples/01_single_fit.ipynb`
- `examples/02_dataframe_and_csv_batch_fitting.ipynb`
- `examples/03_pssession_folder_workflow.ipynb`

They are intentionally outside the importable package. The wheel only includes
`peak_extraction*`, so examples and future notebooks can remain in the GitHub
repo without being installed for every package user.

## Legacy CLI

The previous config-driven workflow is still available:

```bash
python -m peak_extraction.app -c peak_extraction/config/example_config.toml --save
streamlit run analysis/fit_visualization.py "peak_extraction/config/example_config.toml"
```
