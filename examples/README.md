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

## Public Data Suggestions

Each notebook has a `DATA_URL`, `CSV_ZIP_URL`, or `PSSESSION_ZIP_URL` placeholder.
Good places to host freely downloadable example data:

- GitHub Releases: easiest for small zipped demo datasets tied to package tags.
- Zenodo: good for citable research data and DOI-backed releases.
- OSF or Figshare: good for larger public experimental datasets.

For a PyPI package, keep larger data files and notebooks out of the importable
package. Link to them from the README and notebooks instead.

## Optional Streamlit Viewer

Batch notebooks can save their fit results as JSON and launch the packaged
`aswift.analysis.structured_results_viewer` app from a notebook cell. The viewer
reads fit results, filters by channel and frequency, steps through samples in
time order, and plots the stored fit profiles.

After installing with Streamlit support, open the viewer without preloaded data:

```bash
pip install "aswift[streamlit]"
aswift-viewer
```

From the sidebar, point the viewer to one of:

- uploaded fit-results JSON or CSV files created by `fit_dataframe` or
  `fit_pssession_folder`
- uploaded structured trace CSV files with array-valued `voltage` and `current`
  columns
- uploaded simple sweep CSV files with shared voltage in the first column and
  one or more current columns after it, with or without a header row
- uploaded PalmSens `.pssession` files, or a PalmSens folder path for live
  refresh

For simple sweep CSVs, columns are interpreted as
`voltage, current_0, current_1, ...`. Each current column is fit as one channel,
and the viewer shows the selected fit summary and plot without the filtered
results table. UTF-8, UTF-8 BOM, UTF-16, and Latin-1 encoded CSVs are accepted.
The older adjacent `current, voltage` pair layout is still accepted as a
fallback. Multiple uploaded CSVs can be ordered by upload order or file name.

When a `.pssession` folder is selected, the viewer writes these files back into
that folder and can auto-refresh as new session files are added. Unchanged files
are reused from the viewer session cache, new or changed files are fit, and
deleted files are removed from the displayed results. Live refresh checks the
folder every 2 seconds:

```text
aswift_fit_results.json
aswift_signal_table.csv
```

A fit-results CSV can be downloaded directly from the bottom of the viewer
sidebar for uploaded CSV sequences, uploaded `.pssession` files, or live
`.pssession` folders.
