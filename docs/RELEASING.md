# ASWIFT Release Checklist

ASWIFT publishes four distinct artifact types:

1. GitHub automatically generates source-code `.zip` and `.tar.gz` archives
   from every release tag.
2. PyPI receives the Python source distribution and platform-independent wheel
   built by `python -m build`.
3. `.github/workflows/build-release-app.yml` builds native ASWIFT Viewer ZIPs
   for Apple Silicon macOS, Intel macOS, and 64-bit Windows when a GitHub
   release is published.
4. Example data is a separate research-data artifact. It is intentionally not
   installed with the Python package.

## Before the first public release

- Make the repository public and confirm that its history contains no secrets,
  credentials, private paths, participant information, or data that cannot be
  redistributed.
- Confirm ownership and redistribution terms for every example dataset,
  including PalmSens session files.
- Decide whether example data belongs on the GitHub release, Zenodo, or another
  research-data archive. Prefer Zenodo when a DOI, citation, or long-term data
  record is important.
- Configure branch protection and require passing tests before merging to
  `main`.
- Configure a `pypi` GitHub environment with trusted maintainers and optional
  manual approval.
- Create the `aswift` project or a pending Trusted Publisher on PyPI. Configure
  it with owner `Soh-Lab`, repository `aswift`, workflow filename
  `publish-pypi.yml`, and environment `pypi`.
- Replace ad-hoc macOS signing and unsigned Windows packaging with production
  signing before presenting the desktop applications as fully trusted public
  downloads. macOS distribution should also be notarized.

## Prepare a release

In the steps below, replace `X.Y.Z` with the release version.

1. Update `version` in `pyproject.toml` to `X.Y.Z`.
2. Review `README.md` and `PYPI_README.md` together. Keep the PyPI description
   free of links back to its own project page, and confirm that its banner and
   viewer screenshot URLs resolve to the current assets on `main`.
3. Move completed changelog entries from `Unreleased` to
   `[X.Y.Z] - YYYY-MM-DD`.
4. Run the test and package checks:

   ```bash
   python -m pytest -q
   python -m build
   python -m twine check --strict dist/*
   ```

5. Test installation from the wheel in a clean environment, including the
   core package and the optional viewer extra.
6. Configure a TestPyPI Trusted Publisher with owner `Soh-Lab`, repository
   `aswift`, workflow filename `publish-testpypi.yml`, and environment
   `testpypi`. Run `Publish to TestPyPI` for the candidate tag and verify its
   metadata, README, and installation. TestPyPI accounts are separate from
   PyPI accounts.
7. Confirm that the approved `release_assets/example_data.zip` archive is
   tracked by Git and that the README download link targets the `vX.Y.Z`
   release asset.
8. Commit the version, changelog, and URL updates; merge them to `main`; then
   create and push the `vX.Y.Z` tag.
9. Publish the GitHub release from `vX.Y.Z`. The desktop workflow will build,
   smoke-test, zip, and attach all three native applications plus the tracked
   example-data archive.
10. Run the `Publish to PyPI` workflow with the existing release tag. The
   dedicated Trusted Publishing job uses the GitHub `pypi` environment and
   only the `id-token: write` permission. Confirm the environment approval, if
   configured, only after checking the tag and build job output.

## Example data policy

The local `examples/data/` directory and arbitrary `release_assets/*.zip`
files are ignored because experimental files may be large or restricted.
`release_assets/example_data.zip` is the explicit exception: it is tracked and
the release workflow attaches it automatically whenever a GitHub release is
published. Replace that file and commit the replacement when updating the
public example dataset.

Do not add large example data to the PyPI wheel or source distribution. Keep
the notebooks in the source repository and point them to the separately hosted
data archive.
