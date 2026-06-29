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

## Fit a structured dataframe

Use one row per voltammogram. The `voltage` column contains the full potential
array for that trace, and `current` contains the full current array:

```python
from aswift import fit_dataframe, long_form_to_trace_dataframe, results_to_signal_table

results = fit_dataframe(
    df,
    method="aswift",
    n_workers=4,
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
- `examples/02_csv_workflow.ipynb`
- `examples/03_pssession_folder_workflow.ipynb`

They are intentionally outside the importable package. The wheel only includes
`aswift*`, so examples and future notebooks can remain in the GitHub
repo without being installed for every package user.

## Legacy CLI

The previous config-driven workflow is still available:

```bash
python -m peak_extraction.app -c peak_extraction/config/example_config.toml --save
streamlit run analysis/fit_visualization.py "peak_extraction/config/example_config.toml"
```
## Acknowledgments
This work was supported by resources provided by the [Soh Lab](https://sohlab.stanford.edu/) 
within the School of Engineering at Stanford University.

## Author
* **Max Yates** - *PhD Candidate* - [GitHub Profile](https://github.com/max-giraffe)

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
