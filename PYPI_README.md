# ASWIFT

<p align="center">
  <img src="https://raw.githubusercontent.com/Soh-Lab/aswift/main/docs/assets/aswift-logo-banner.png" alt="ASWIFT: automated square-wave voltammetry signal fitting" width="640">
</p>

ASWIFT fits square-wave voltammetry (SWV) traces and extracts peak signal,
background, peak voltage, full-prominence peak width, and baseline-subtracted
full-prominence peak area. Use the no-code ASWIFT Viewer for interactive
analysis, or install the Python package to build ASWIFT into a custom workflow.

## Installation

ASWIFT requires Python 3.12 or newer. Install the core fitting utilities with:

```bash
pip install aswift
```

Install the viewer, pandas CSV helpers, plotting tools, and PalmSens
`.pssession` support with:

```bash
pip install "aswift[viewer]"
```

PalmSens support uses `pypalmsens`, which requires the
[Microsoft .NET 9 Runtime](https://dotnet.microsoft.com/en-us/download/dotnet/9.0).

## ASWIFT Viewer

Launch the viewer from an activated Python environment:

```bash
aswift-viewer
```

![Annotated overview of the ASWIFT Viewer interface](https://raw.githubusercontent.com/Soh-Lab/aswift/main/docs/assets/aswift-viewer-overview.png)

The viewer loads CSV, JSON, and PalmSens `.pssession` data; fits individual or
batched voltammograms; visualizes fitted backgrounds and peaks; and exports
raw and normalized results with their associated metadata.

## Python API

```python
import numpy as np
from aswift import aswift_fit, plot_fit_result

volts = np.asarray([...], dtype=float)
current = np.asarray([...], dtype=float)

result = aswift_fit(volts, current)
print(result.peak_signal, result.peak_voltage)
ax = plot_fit_result(result)
```

The returned `FitResult` includes the peak signal, background, voltage,
full-prominence width and area, peak profile, background profile, and combined
fitted current. ASWIFT also provides dataframe and PalmSens helpers for batch
workflows.

## Documentation and support

See the [full documentation and examples](https://github.com/Soh-Lab/aswift#readme)
on GitHub. Please [open an issue](https://github.com/Soh-Lab/aswift/issues) for
bug reports, questions, feedback, or feature requests.

## Citation

If you use ASWIFT in your research, please cite:

> Yates, M., Ji, J., Yee, S., & Soh, H. T. (2026). Robust Regularization
> Enables Automated, Real-Time Square-Wave Voltammetry Signal Quantification.
> *bioRxiv*. [https://doi.org/10.64898/2026.07.25.740173](https://doi.org/10.64898/2026.07.25.740173)

Machine-readable citation metadata is available in
[`CITATION.cff`](https://github.com/Soh-Lab/aswift/blob/main/CITATION.cff).

## License

ASWIFT is licensed under the MIT License. See the
[license](https://github.com/Soh-Lab/aswift/blob/main/LICENSE) for details.
