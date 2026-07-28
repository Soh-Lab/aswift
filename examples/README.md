# Example Notebooks

These notebooks live outside the importable `aswift` package. The
`pyproject.toml` only includes `aswift*` in the wheel, so notebooks are
available in the GitHub repo without being installed for normal package users.

## Notebooks

- `01_single_fit.ipynb`: fit one SWV trace with ASWIFT and poly-linear.
- `02_csv_workflow.ipynb`: fit structured CSV/dataframe inputs.
- `03_pssession_workflow.ipynb`: convert a folder of PalmSens
  `.pssession` files, fit traces, and plot peak signal over time.

## Structured CSV/DataFrame Format

Notebook 02 expects one row per voltammogram. The required measured columns are:

- `voltage`: full one-dimensional array/list of potentials for the trace.
- `current`: full one-dimensional array/list of measured currents for the trace.

Metadata columns describe the trace and are carried through to the results.
Common metadata columns are:

- `file`: source file or trace name.
- `hz`: SWV frequency.
- `num`: sample/acquisition number, usually time ordered.
- `channel`: electrode/channel identifier.
- `time`: elapsed acquisition time, usually hours from the first measurement.

`fit_dataframe` treats each row as one trace when `voltage` and `current`
contain arrays. Older long-form point tables are still supported for
compatibility through `long_form_to_trace_dataframe`, but new examples should
prefer the array-per-trace shape.

PalmSens `.pssession` files do not need to be manually converted into this CSV
format first. Notebook 03 shows how to point ASWIFT at a folder of `.pssession`
files and automatically convert them into the same array-per-trace dataframe and
fit-results JSON.

## Public Example Data

The notebooks use the public
[`example_data.zip`](https://github.com/Soh-Lab/aswift/releases/download/v1.0.2/example_data.zip)
archive. Each notebook checks for its required extracted file or folder and,
when it is missing, extracts the repository's bundled archive or downloads and
unzips it automatically when the notebook is being run on its own.

- Notebook 01 uses `simple_csv_example/250hz-1.csv`.
- Notebook 02 uses
  `structured_csv_example/dox_invitro_shouldering_structured.csv` and fits
  channels 0–3 at 10 Hz and 200 Hz.
- Notebook 03 uses the `pssession_example` folder.

## Optional Streamlit Viewer

Batch notebooks can save their fit results as JSON and launch the packaged
`aswift.analysis.structured_results_viewer` app from a notebook cell. The viewer
reads fit results, filters by channel and frequency, steps through samples in
time order, and plots the stored fit profiles.

PalmSens `.pssession` support uses `pypalmsens`, which requires the Microsoft
.NET 9 Runtime to be installed on the computer. If the viewer reports that it
failed to create a .NET runtime/CoreCLR, install the .NET 9 Runtime and restart
the viewer.

After installing with viewer support, open the viewer without preloaded data:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "aswift[viewer]"
aswift-viewer
```

From the sidebar, point the viewer to one of:

- uploaded fit-results JSON or CSV files created by `fit_dataframe` or
  `fit_pssession_folder`
- uploaded structured trace CSV files with array-valued `voltage` and `current`
  columns
- uploaded simple sweep CSV files with shared voltage in the first column and
  one or more current columns after it, with or without a header row
- a PalmSens folder path containing `.pssession` or `.pssession.mp3` files

For simple sweep CSVs, columns are interpreted as
`voltage, current_0, current_1, ...`. Each current column is fit as one channel,
and the viewer shows the selected fit summary and plot without the filtered
results table. UTF-8, UTF-8 BOM, UTF-16, and Latin-1 encoded CSVs are accepted.
The older adjacent `current, voltage` pair layout is still accepted as a
fallback. Multiple uploaded CSVs can be ordered by upload order or file name.

For PalmSens data, enter the path to a folder containing `.pssession` or
`.pssession.mp3` files. Files ending in `.pssession.mp3` are renamed to
`.pssession` before processing.
If a fit fails for a trace but raw voltage/current data are available, the
viewer still plots the raw trace so the failed measurement can be inspected.

A fit-results CSV can be downloaded directly from the bottom of the viewer
sidebar for uploaded CSV sequences or PalmSens folder paths.
