# ASWIFT

ASWIFT fits square-wave voltammetry traces and returns peak signal, background,
and full fitted profiles that can be plotted directly.

## Install

Install the core utilities via pip:
```bash
pip install aswift
```

Install the core utilities with visualization via pip:
```bash
pip install "aswift[all]"
```

## Fit one trace

```python
import numpy as np
from aswift import aswift_fit, poly_linear_fit

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

## Plot one trace

```python
from aswift import plot_fit_result

ax = plot_fit_result(result)
```

## Open the Streamlit viewer

Install with Streamlit support:

```bash
pip install "aswift[streamlit]"
```

Then launch the viewer with no preloaded data:

```bash
aswift-viewer
```

The sidebar lets you load:

- one or more uploaded fit-results `.json` or `.csv` files
- one or more uploaded structured trace `.csv` files with `voltage` and
  `current` array columns
- one or more uploaded simple sweep `.csv` files with shared voltage in the
  first column and channel currents in the following columns
- uploaded PalmSens `.pssession` files, or a PalmSens folder path for live
  refresh

Simple sweep CSVs may include a title/header row or no header row. The viewer
reads them as `voltage, current_0, current_1, ...`, fits each current column as
one channel, and shows the selected fit summary and plot without the full
results table. UTF-8, UTF-8 BOM, UTF-16, and Latin-1 encoded CSVs are accepted.
The older adjacent `current, voltage` pair layout is still accepted as a
fallback. Multiple uploaded CSVs can be ordered by upload order or file name.

When pointed at a `.pssession` folder path, the viewer fits the folder, writes
`aswift_fit_results.json` and `aswift_signal_table.csv` into that folder, and
can auto-refresh so newly added `.pssession` files appear in the interface.
During live refresh, unchanged files are reused from the viewer session cache,
new or changed files are fit, and deleted files are removed from the displayed
results. Live refresh checks the folder every 2 seconds.

Fit results can be downloaded from the bottom of the viewer sidebar as a CSV for
any loaded input source.

## Run tests

From the repository root:

```bash
NUMBA_CACHE_DIR=/private/tmp/numba-cache python -m pytest -q
```

If your active environment has binary package conflicts, install the test extras
in the project virtual environment, then run the same tests there:

```bash
venv/bin/python -m pip install -e ".[test,streamlit]"
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/mpl-cache venv/bin/python -m pytest -q
```

To verify the package still builds:

```bash
venv/bin/python -m build
```

## Fit a structured dataframe

Use one row per voltammogram. The `voltage` column contains the full potential
array for that trace, and `current` contains the full current array:

```python
from aswift import fit_dataframe, long_form_to_trace_dataframe, results_to_signal_table

results = fit_dataframe(
    df,
    method="aswift",
    n_workers=1,
)

signals = results_to_signal_table(
    results,
    calibration_lower_idx=0,
    calibration_upper_idx=4,
)
```

Expected columns are:

- `voltage`: array-like potentials for one voltammogram
- `current`: array-like currents for one voltammogram
- any metadata columns to preserve, such as `file`, `hz`, `num`, `channel`, `time`

Older long-form data with one row per sampled point is still supported for
compatibility. For that shape, pass `group_cols` to define which rows belong to
one trace, or convert it first:

```python
df = long_form_to_trace_dataframe(
    legacy_df,
    group_cols=["file", "hz", "num", "channel", "time"],
)
```

## Load existing PalmSens files

```python
from aswift import (
    fit_pssession_folder,
    formatted_csvs_to_dataframe,
    pssession_folder_to_dataframe,
    plot_signal_over_time,
)

df = formatted_csvs_to_dataframe("path/to/csv/folder", hz_values=[150, 200, 250])
results = fit_dataframe(df, method="aswift", n_workers=1)

# Optional PalmSens adapter; requires pypalmsens.
ps_df = pssession_folder_to_dataframe("path/to/pssession/folder")
ps_df, ps_results = fit_pssession_folder("path/to/pssession/folder", n_workers=1)
plot_signal_over_time(ps_results)
```

The `.pssession` workflow orders traces by timestamp, frequency, file number,
and channel when those fields are available. If files were downloaded with a
spurious `.mp3` suffix, files named `.pssession.mp3` are renamed to `.pssession`
before discovery. PalmSens UTC timestamps are used when present, and `time` is
normalized in hours so the earliest measurement is `0.0`.

ASWIFT defaults to single-worker fitting because the core numerical work already
spends most of its time inside SciPy/NumPy/pybaselines routines. In local
benchmarks, Python thread-pool workers were slower for typical trace sizes. Use
`n_workers > 1` only after benchmarking your dataset.

For larger batches, use process-based parallelism to fit independent trace rows
in separate workers:

```python
results = fit_dataframe(
    df,
    method="aswift",
    n_workers=4,
    parallel_backend="process",
    chunksize=16,
)
```

This mirrors the older simulation benchmark workflow: each row is fit
independently, and `chunksize` reduces scheduling overhead. It can be faster for
large CPU-bound batches, but process startup and array serialization can still
make it slower for small interactive uploads.

## Examples

Runnable examples live in the top-level `examples/` folder:

- `examples/01_single_fit.ipynb`
- `examples/02_csv_workflow.ipynb`
- `examples/03_pssession_workflow.ipynb`

They are intentionally outside the importable package. The wheel only includes
`aswift*`, so examples and future notebooks can remain in the GitHub
repo without being installed for every package user.

## Acknowledgments
This work was supported by resources provided by the [Soh Lab](https://sohlab.stanford.edu/) 
within the School of Engineering at Stanford University.

## Author
* **Max Yates** - *PhD Candidate* - [GitHub Profile](https://github.com/max-giraffe)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
