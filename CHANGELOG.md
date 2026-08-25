# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- Add an input control that multiplies raw current by -1 before fitting.
- Track, plot, and export baseline-subtracted full-prominence peak area.
- Add the giraffe ASWIFT banner to the README used for the PyPI project page.

### Fixed
- Keep live PalmSens folder, channel, frequency, sample, and trend selections
  synchronized with fit and trend plots as new files arrive.

## [1.0.2] - 2026-07-27

### Added
- Machine-readable citation metadata and the bioRxiv preprint citation.
- Self-contained example notebooks that download and extract the versioned
  GitHub Release example-data archive when it is not available locally.

### Changed
- Pin README images and example-data links to the `v1.0.2` release.

## [1.0.1] - 2026-07-15

### Fixed
- Use an absolute, version-pinned URL for the annotated ASWIFT Viewer image so
  it renders on package-index project pages after the repository is public.
- Point the example-data download link at the `v1.0.1` GitHub release.
- Update Streamlit chart sizing and pandas JSON date serialization for current
  APIs, and install Watchdog with the optional viewer dependencies.

## [1.0.0] - 2026-07-14

### Added
- Cross-platform desktop release builds for Apple Silicon macOS, Intel macOS,
  and 64-bit Windows.
- Package-safe Streamlit launcher support for relative imports.
- Release-readiness documentation and packaging checks.
- Example datasets attached automatically to the GitHub release.

### Changed
- Consolidated overlapping viewer tests and removed environment-sensitive
  timing assertions.

## [0.1.0] - 2026-07-01

### Added
- Core API supporting ASWIFT peak fitting utilities for square-wave voltammetry.
- Documentation, setup instructions, and deployment examples in `README.md`.
- Added pytest coverage for fitting workflows, example data, CSV parsing, and the streamlit viewer.
- Added the `aswift-viewer` command for launching the Streamlit interface.
- Added Streamlit support for uploaded JSON/CSV files, simple voltage/current
  CSV files, PalmSens `.pssession` files, and live folder monitoring.
- Added functionality for normalizing peak height by specified range.
- Added column for peak full-width prominence length in results file.

### Fixed
- Improved batch fitting performance with process-based row-level parallelism for larger datasets.
- Improved Streamlit live-folder behavior so unchanged files are cached and not refit.

### Removed
- Removed the legacy config-driven CLI CSV processing pipeline.
- Removed `loguru` and `pydantic` from core runtime dependencies.
