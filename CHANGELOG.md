# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## - 2026-06-29

### Added
- Initial public release of `aswift` to PyPI.
- Core API supporting ASWIFT peak fitting utilities for square-wave voltammetry.
- Documentation, setup instructions, and deployment examples in `README.md`.


## 2026-07-01

### Added
- Added pytest coverage for fitting workflows, example data, CSV parsing, and the streamlit viewer.
- Added the `aswift-viewer` command for launching the Streamlit interface.
- Added Streamlit support for uploaded JSON/CSV files, simple voltage/current CSV files, PalmSens `.pssession` files, and live folder monitoring.
- Added functionality for normalizing peak height by specified range.
- Added column for peak full-width prominence length in results file.

### Fixed
- Improved batch fitting performance with process-based row-level parallelism for larger datasets.
- Improved Streamlit live-folder behavior so unchanged files are cached and not refit.

### Removed
- Removed the legacy config-driven CLI CSV processing pipeline.
- Removed `loguru` and `pydantic` from core runtime dependencies.
