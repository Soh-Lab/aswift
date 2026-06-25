# Example Notebooks

These notebooks live outside the importable `peak_extraction` package. The
`pyproject.toml` only includes `peak_extraction*` in the wheel, so notebooks are
available in the GitHub repo without being installed for normal package users.

## Notebooks

- `01_single_fit.ipynb`: fit one SWV trace with ASWIFT and poly-linear.
- `02_dataframe_and_csv_batch_fitting.ipynb`: fit long-form dataframes with
  threads and optionally convert a formatted CSV folder.
- `03_pssession_folder_workflow.ipynb`: convert a folder of PalmSens
  `.pssession` files, fit traces, and plot peak signal over time.

## Public Data Suggestions

Each notebook has a `DATA_URL`, `CSV_ZIP_URL`, or `PSSESSION_ZIP_URL` placeholder.
Good places to host freely downloadable example data:

- GitHub Releases: easiest for small zipped demo datasets tied to package tags.
- Zenodo: good for citable research data and DOI-backed releases.
- OSF or Figshare: good for larger public experimental datasets.

For a PyPI package, keep larger data files and notebooks out of the importable
package. Link to them from the README and notebooks instead.

## Optional Streamlit Viewer

Batch notebooks can save their fit results as JSON and print a launch command
for `analysis/structured_results_viewer.py`. The viewer reads the structured
results dataframe, filters by channel and frequency, steps through samples in
time order, and plots the stored fit profiles.

Example from the repository root:

```bash
streamlit run analysis/structured_results_viewer.py -- examples/outputs/pssession_demo/pssession_fit_results.json
```
