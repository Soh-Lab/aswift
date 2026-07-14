# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

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
