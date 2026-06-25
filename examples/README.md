# Example Notebooks

These notebooks live outside the importable `peak_extraction` package. The
`pyproject.toml` only includes `peak_extraction*` in the wheel, so notebooks are
available in the GitHub repo without being installed for normal package users.

## Notebooks

- `01_single_fit.ipynb`: fit one SWV trace with ASWIFT and poly-linear.
- `02_csv_workflow.ipynb`: fit structured CSV/dataframe inputs with
  threads.
- `03_pssession_folder_workflow.ipynb`: convert a folder of PalmSens
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

Batch notebooks can save their fit results as JSON and launch
`analysis/structured_results_viewer.py` from a notebook cell. The viewer reads
the structured results dataframe, filters by channel and frequency, steps
through samples in time order, and plots the stored fit profiles.

Example from the repository root:

```bash
streamlit run analysis/structured_results_viewer.py -- examples/outputs/pssession_demo/pssession_fit_results.json
```
