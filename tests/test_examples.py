"""Regression tests for ASWIFT fitting, batch processing, and viewer helpers."""
from __future__ import annotations

from contextlib import nullcontext
from io import StringIO
import importlib.util
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from aswift import aswift_fit, poly_linear_fit


def _load_desktop_launcher():
    """Load the desktop launcher without requiring it to be a package module."""
    launcher_path = (
        Path(__file__).parents[1] / "packaging" / "desktop" / "aswift_viewer_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("aswift_viewer_launcher", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_desktop_launcher_unblocks_downloaded_dotnet_assemblies(
    monkeypatch, tmp_path: Path
) -> None:
    """Verify the Windows launcher removes downloaded-file markers from DLLs."""
    launcher = _load_desktop_launcher()
    monkeypatch.setattr(launcher.sys, "platform", "win32")

    pythonnet_dir = tmp_path / "pythonnet" / "runtime"
    palmsens_dir = tmp_path / "pypalmsens" / "_libpalmsens" / "win"
    unrelated_dir = tmp_path / "unrelated"
    for directory, name in (
        (pythonnet_dir, "Python.Runtime.dll"),
        (palmsens_dir, "PalmSens.Core.dll"),
        (unrelated_dir, "Other.dll"),
    ):
        directory.mkdir(parents=True)
        (directory / name).touch()
        (directory / f"{name}:Zone.Identifier").touch()

    launcher._unblock_bundled_windows_assemblies(tmp_path)

    assert not (pythonnet_dir / "Python.Runtime.dll:Zone.Identifier").exists()
    assert not (palmsens_dir / "PalmSens.Core.dll:Zone.Identifier").exists()
    assert (unrelated_dir / "Other.dll:Zone.Identifier").exists()


def test_viewer_cli_preserves_package_context(monkeypatch) -> None:
    """Verify CLI and Streamlit execution preserve package-relative imports."""
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer
    import aswift.analysis.viewer_cli as viewer_cli
    import streamlit.web.cli

    launched = []
    monkeypatch.setattr(streamlit.web.cli, "main", lambda: launched.append(sys.argv.copy()))
    monkeypatch.setattr(sys, "argv", ["aswift-viewer", "--server.headless", "true"])

    viewer_cli.main()

    assert launched == [
        ["streamlit", "run", str(Path(viewer_cli.__file__)), "--server.headless", "true"]
    ]

    app_runs = []
    monkeypatch.setattr(viewer, "_run_app", lambda: app_runs.append(True))

    runpy.run_path(viewer_cli.__file__, run_name="__main__")

    assert app_runs == [True]


def synthetic_single_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return one reproducible synthetic SWV-like trace for tests."""
    rng = np.random.default_rng(seed)
    volts = np.linspace(-0.4, 0.0, 400)
    background = 3.0 * (volts + 0.1) ** 2 + 1.0
    peak = 0.45 * np.exp(-0.5 * ((volts + 0.2) / 0.045) ** 2)
    current = background + peak + rng.normal(loc=0.0, scale=0.01, size=volts.size)
    return volts, current


def synthetic_trace_dataframe():
    """Return a small synthetic multi-frequency trace dataframe for tests."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(4)
    volts = np.linspace(-0.4, 0.0, 220)
    rows = []
    for hz in (150, 250):
        for channel in range(2):
            for num in range(4):
                amp = 0.35 + 0.03 * num + 0.04 * channel
                center = -0.18 + 0.015 * channel
                background = 1.0 + 0.08 * volts - 0.1 * num * volts
                peak = amp * np.exp(-0.5 * ((volts - center) / 0.02) ** 2)
                current = background + peak + rng.normal(0.0, 0.004, size=volts.size)
                rows.append(
                    {
                        "file": f"{hz}hz-{num + 1}.csv",
                        "hz": hz,
                        "num": num,
                        "channel": channel,
                        "time": num * 0.25,
                        "voltage": volts.tolist(),
                        "current": current.tolist(),
                    }
                )
    return pd.DataFrame(rows)


def test_core_fitting_methods_fit_synthetic_trace() -> None:
    """Verify both public fitting methods return plausible results."""
    volts, current = synthetic_single_trace()

    aswift_result = aswift_fit(volts, current)
    poly_result = poly_linear_fit(volts, current)

    for result in (aswift_result, poly_result):
        assert result.success
        assert result.error is None
        assert np.isfinite(result.peak_signal)
        assert np.isfinite(result.fw_prominence)
        assert result.fw_prominence > 0
        assert 0.25 < result.peak_signal < 0.6
        assert -0.25 < result.peak_voltage < -0.15
        assert result.fitted_current.shape == volts.shape


def test_dataframe_fitting_builds_signal_table() -> None:
    """Verify dataframe fitting produces ordered signal-table columns."""
    df = synthetic_trace_dataframe()
    from aswift import fit_dataframe, results_to_signal_table

    results = fit_dataframe(df, method="aswift", n_workers=1)

    assert len(results) == len(df)
    assert results["success"].all()
    assert {"hz", "num", "channel", "peak", "peak_voltage"}.issubset(results.columns)
    assert results["peak"].between(0.2, 0.7).all()
    assert "fw_prominence" in results.columns
    assert results["fw_prominence"].notna().all()

    signal_table = results_to_signal_table(
        results,
        calibration_lower_idx=0,
        calibration_upper_idx=1,
    )
    expected_signal_columns = {
        "signal-hz150-channel0",
        "signal-hz150-channel1",
        "signal-hz250-channel0",
        "signal-hz250-channel1",
    }
    assert expected_signal_columns.issubset(signal_table.columns)
    assert len(signal_table) == 4


@pytest.mark.parametrize(
    ("source", "expected_channels"),
    [
        ("shared_header", 2),
        ("headerless", 2),
        ("utf16_shared", 1),
        ("current_voltage_pairs", 2),
    ],
)
def test_simple_csv_formats_convert_to_trace_dataframe(source: str, expected_channels: int) -> None:
    """Verify simple csv formats convert to trace dataframe."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import (
        _read_csv_for_viewer,
        _read_csv_payload_for_viewer,
        _simple_csv_to_trace_dataframe,
    )

    volts = np.linspace(-0.4, 0.0, 20)
    if source == "shared_header":
        df = pd.DataFrame({"voltage": volts, "current_0": 1.0 + volts, "current_1": 1.1 + volts})
    elif source == "headerless":
        raw = pd.DataFrame({0: volts, 1: 1.0 + volts, 2: 1.1 + volts})
        df = _read_csv_for_viewer(StringIO(raw.to_csv(index=False, header=False)))
    elif source == "utf16_shared":
        raw = pd.DataFrame({"V": volts, "uA": 1.0 + volts})
        df = _read_csv_payload_for_viewer(raw.to_csv(index=False).encode("utf-16"))
    else:
        df = pd.DataFrame(
            {
                "current_channel_0": 1.0 + volts,
                "volts_channel_0": volts,
                "current_channel_1": 1.1 + volts,
                "volts_channel_1": volts,
            }
        )

    traces = _simple_csv_to_trace_dataframe(df)

    assert traces is not None
    assert len(traces) == expected_channels
    assert traces["channel"].tolist() == list(range(expected_channels))
    assert traces.iloc[0]["current"] == pytest.approx((1.0 + volts).tolist())
    assert traces.iloc[0]["voltage"] == pytest.approx(volts.tolist())


def test_single_row_sample_selection_does_not_use_slider(monkeypatch) -> None:
    """Verify single row sample selection does not use slider."""
    pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    class Sidebar:
        """Minimal sidebar stub used to assert Streamlit sample-selection behavior."""
        def __init__(self) -> None:
            """Initialize the helper object."""
            self.slider_called = False

        @staticmethod
        def caption(text: str) -> None:
            """Validate the caption text written by the code under test."""
            assert text == "Sample index: 0"

        def slider(self, *_args, **_kwargs) -> int:
            """Fail if a slider is rendered when only one sample is available."""
            self.slider_called = True
            raise AssertionError("slider should not be used for one row")

    sidebar = Sidebar()
    monkeypatch.setattr(viewer.st, "sidebar", sidebar)

    assert viewer._select_sample_position(1) == 0
    assert not sidebar.slider_called


def test_signal_table_pads_unequal_group_lengths() -> None:
    """Verify signal table pads unequal group lengths."""
    pd = pytest.importorskip("pandas")
    from aswift import results_to_signal_table

    results = pd.DataFrame(
        {
            "num": [0, 1, 0],
            "hz": [150, 150, 250],
            "channel": [0, 0, 0],
            "peak": [1.0, 2.0, 3.0],
        }
    )

    table = results_to_signal_table(results)

    assert len(table) == 2
    assert table["signal-hz150-channel0"].tolist() == pytest.approx([1.0, 2.0])
    assert table["signal-hz250-channel0"].iloc[0] == pytest.approx(3.0)
    assert pd.isna(table["signal-hz250-channel0"].iloc[1])


def test_fit_dataframe_default_uses_single_worker_fast_path(monkeypatch) -> None:
    """Verify fit dataframe default uses single worker fast path."""
    df = synthetic_trace_dataframe().head(2)
    import aswift.workflow.batch as batch

    def fail_process_pool(*_args, **_kwargs):
        """Fail if the default fitting path unexpectedly uses process workers."""
        raise AssertionError("default fit_dataframe should not use process workers")

    monkeypatch.setattr(batch, "_fit_traces_process_pool", fail_process_pool)

    results = batch.fit_dataframe(df, method="aswift")

    assert len(results) == len(df)
    assert results["success"].all()


def test_fit_dataframe_reports_progress_for_single_worker() -> None:
    """Verify fit dataframe reports progress for single worker."""
    df = synthetic_trace_dataframe().head(3)
    from aswift import fit_dataframe

    updates = []
    results = fit_dataframe(
        df,
        method="aswift",
        n_workers=1,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(results) == len(df)
    assert updates[-1] == (len(df), len(df))


def test_plot_helpers_use_expected_axis_labels() -> None:
    """Verify plot helpers use expected axis labels."""
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    from aswift import aswift_fit, plot_fit_result, plot_signal_over_time

    volts, current = synthetic_single_trace()
    result = aswift_fit(volts, current)
    _, fit_ax = plt.subplots()
    plot_fit_result(result, ax=fit_ax)

    results = synthetic_trace_dataframe().head(2).copy()
    results["peak"] = [1.0, 1.1]
    results["method"] = "aswift"
    _, trend_ax = plt.subplots()
    plot_signal_over_time(results, ax=trend_ax)

    index_results = pd.DataFrame({"method": ["aswift", "aswift"], "num": [0, 1], "peak": [1.0, 1.2]})
    _, index_ax = plt.subplots()
    plot_signal_over_time(index_results, ax=index_ax)

    assert fit_ax.get_xlabel() == "Input"
    assert fit_ax.get_ylabel() == "Signal"
    assert trend_ax.get_xlabel() == "Time"
    assert trend_ax.get_ylabel() == "Signal"
    assert index_ax.get_xlabel() == "Index Number"
    assert index_ax.get_ylabel() == "Signal"
    plt.close("all")


def test_viewer_normalization_scopes_groups_and_preserves_raw_data() -> None:
    """Verify normalization groups independently without changing raw results."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _add_norm_signal_by_peak_range, _normalization_row_count

    results = pd.DataFrame(
        {
            "relative_folder": ["a", "a", "a", "b", "b"],
            "method": ["aswift"] * 5,
            "hz": [100, 100, 200, 100, 100],
            "channel": [0] * 5,
            "num": [0, 1, 0, 0, 1],
            "peak": [2.0, 4.0, 10.0, 100.0, 200.0],
            "background": [1.0, 2.0, 5.0, 50.0, 100.0],
            "current": [[2.0, 4.0], [4.0, 8.0], [10.0, 20.0], [100.0, 200.0], [200.0, 400.0]],
            "peak_profile": [[2.0], [4.0], [10.0], [100.0], [200.0]],
            "background_profile": [[1.0], [2.0], [5.0], [50.0], [100.0]],
            "fitted_signal": [[3.0], [6.0], [15.0], [150.0], [300.0]],
        }
    )

    normalized = _add_norm_signal_by_peak_range(results, 0, 0)

    assert normalized["peak"].tolist() == pytest.approx([2.0, 4.0, 10.0, 100.0, 200.0])
    assert normalized["background"].tolist() == pytest.approx([1.0, 2.0, 5.0, 50.0, 100.0])
    assert normalized["norm_signal"].tolist() == pytest.approx([1.0, 2.0, 1.0, 1.0, 2.0])
    assert normalized["normalization_basis"].tolist() == [True, False, True, True, False]
    assert _normalization_row_count(results) == 2
    assert normalized.iloc[0]["current"] == pytest.approx([2.0, 4.0])
    assert normalized.iloc[2]["background_profile"] == pytest.approx([5.0])


def test_normalization_controls_preserve_range_when_live_data_grows(monkeypatch) -> None:
    """Verify a live-data rerun retains the selected normalization indices."""
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    class Sidebar:
        """Minimal normalization-control stub that records slider defaults."""

        def __init__(self) -> None:
            self.defaults = []

        @staticmethod
        def checkbox(_label, *, key):
            """Return the persisted enabled setting."""
            assert key == viewer.NORMALIZE_ENABLED_KEY
            return True

        def slider(self, _label, *, min_value, max_value, value):
            """Record the default and emulate the current widget selection."""
            assert min_value == 0
            self.defaults.append((max_value, value))
            return (1, 2) if len(self.defaults) == 1 else value

    sidebar = Sidebar()
    monkeypatch.setattr(viewer.st, "session_state", {})
    monkeypatch.setattr(viewer.st, "sidebar", sidebar)

    assert viewer._normalization_controls(4) == (True, 1, 2)
    assert viewer._normalization_controls(5) == (True, 1, 2)
    assert sidebar.defaults == [(3, (0, 3)), (4, (1, 2))]
    assert viewer.st.session_state[viewer.NORMALIZATION_RANGE_KEY] == (1, 2)


def test_viewer_results_download_hides_internal_columns_and_preserves_norm_signal() -> None:
    """Verify viewer results download hides internal columns and preserves norm signal."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _add_norm_signal_by_peak_range, _downloadable_results

    results = pd.DataFrame(
        {
            "method": ["aswift", "aswift"],
            "hz": [100, 100],
            "channel": [0, 0],
            "num": [0, 1],
            "peak": [2.0, 4.0],
            "background": [1.0, 2.0],
            "peak_index": [2, 3],
            "peak_idx": [2, 3],
            "peak_window_start": [1, 2],
            "peak_window_end": [4, 5],
            "normalization_start_index": [0, 0],
            "normalization_end_index": [1, 1],
            "normalization_reference_peak": [3.0, 3.0],
            "voltage": [[0.0, 1.0], [0.0, 1.0]],
            "current": [[1.0, 2.0], [2.0, 3.0]],
            "peak_profile": [[0.0, 1.0], [1.0, 2.0]],
        }
    )

    download = _downloadable_results(results)
    normalized_download = _downloadable_results(_add_norm_signal_by_peak_range(results, 0, 1))

    assert "peak" in download.columns
    assert "norm_signal" in download.columns
    assert "normalization_basis" in download.columns
    peak_index = download.columns.get_loc("peak")
    assert download.columns[peak_index : peak_index + 4].tolist() == [
        "peak",
        "norm_signal",
        "normalization_basis",
        "background",
    ]
    assert download["norm_signal"].isna().all()
    assert download["normalization_basis"].tolist() == [False, False]
    for column in (
        "peak_index",
        "peak_idx",
        "peak_window_start",
        "peak_window_end",
        "normalization_start_index",
        "normalization_end_index",
        "normalization_reference_peak",
        "voltage",
        "current",
        "peak_profile",
    ):
        assert column not in download.columns
    peak_index = normalized_download.columns.get_loc("peak")
    assert normalized_download.columns[peak_index : peak_index + 4].tolist() == [
        "peak",
        "norm_signal",
        "normalization_basis",
        "background",
    ]
    assert normalized_download["norm_signal"].tolist() == pytest.approx([2 / 3, 4 / 3])
    assert normalized_download["normalization_basis"].tolist() == [True, True]


def test_viewer_download_scope_includes_all_frequencies_for_selected_folder() -> None:
    """Verify viewer download includes all frequencies rather than the selected one."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _download_scope_results

    results = pd.DataFrame(
        {
            "relative_folder": ["plate-a", "plate-a", "plate-a", "plate-a", "plate-b"],
            "method": ["aswift"] * 5,
            "hz": [150, 150, 250, 250, 150],
            "channel": [0, 0, 0, 0, 0],
            "num": [0, 1, 0, 1, 0],
            "peak": [2.0, 4.0, 10.0, 20.0, 99.0],
        }
    )

    download = _download_scope_results(
        results,
        selected_folder="plate-a",
        normalize=True,
        norm_start=0,
        norm_end=0,
    )

    assert download["relative_folder"].tolist() == ["plate-a", "plate-a", "plate-a", "plate-a"]
    assert sorted(download["hz"].unique().tolist()) == [150, 250]
    assert download["norm_signal"].tolist() == pytest.approx([1.0, 2.0, 1.0, 2.0])
    assert download["normalization_basis"].tolist() == [True, False, True, False]


def test_result_summary_values_are_display_safe_strings() -> None:
    """Verify result summary values are display safe strings."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _result_summary

    summary = _result_summary(
        pd.Series(
            {
                "method": "aswift",
                "success": np.bool_(True),
                "peak": np.float64(1.23456789),
                "background": np.float64(0.1),
            }
        )
    )

    assert summary["value"].tolist() == ["aswift", "True", "1.23457", "0.1"]


def test_failed_fit_row_builds_interactive_raw_trace() -> None:
    """Verify failed fit row builds interactive raw trace."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("plotly")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _selected_fit_figure

    row = pd.Series(
        {
            "success": False,
            "voltage": [0.0, 1.0, 2.0],
            "current": [3.0, 4.0, 5.0],
        }
    )

    fig = _selected_fit_figure(row, title="Example", empty_message="No trace")

    assert len(fig.data) == 1
    assert list(fig.data[0].x) == pytest.approx([0.0, 1.0, 2.0])
    assert list(fig.data[0].y) == pytest.approx([3.0, 4.0, 5.0])
    assert fig.data[0].name == "Raw data"
    assert fig.layout.xaxis.title.text == "Potential"
    assert fig.layout.yaxis.title.text == "Current"


def test_successful_fit_row_builds_interactive_method_fit() -> None:
    """Verify successful fit row builds interactive method fit."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("plotly")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import FIT_TRACE_COLORS, _selected_fit_figure

    row = pd.Series(
        {
            "success": True,
            "peak": 1.0,
            "background": 2.0,
            "peak_voltage": 1.0,
            "voltage": [0.0, 1.0, 2.0],
            "current": [3.0, 4.0, 5.0],
            "background_profile": [2.0, 2.0, 2.0],
            "peak_profile": [1.0, 2.0, 3.0],
            "fitted_signal": [3.0, 4.0, 5.0],
        }
    )

    fig = _selected_fit_figure(row, title="Example", empty_message="No trace")

    assert {trace.name for trace in fig.data} >= {
        "Raw data",
        "Background",
        "Method fit",
        "Peak height",
    }
    assert "Peak profile" not in {trace.name for trace in fig.data}
    background_trace = next(trace for trace in fig.data if trace.name == "Background")
    assert background_trace.line.color == FIT_TRACE_COLORS["background"]
    assert background_trace.line.dash is None
    fit_trace = next(trace for trace in fig.data if trace.name == "Method fit")
    assert fit_trace.line.color == FIT_TRACE_COLORS["method"]
    assert fit_trace.line.dash is None
    peak_trace = next(trace for trace in fig.data if trace.name == "Peak height")
    assert list(peak_trace.x) == pytest.approx([1.0, 1.0])
    assert list(peak_trace.y) == pytest.approx([2.0, 3.0])
    assert peak_trace.line.color == FIT_TRACE_COLORS["peak"]
    assert peak_trace.line.dash == "dash"


def test_upload_trace_csv_uses_selected_fit_method(monkeypatch) -> None:
    """Verify upload trace csv uses selected fit method."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    volts = np.linspace(-0.4, 0.0, 20)
    trace_df = pd.DataFrame(
        {
            "voltage": volts,
            "current_0": 1.0 + volts,
        }
    )
    calls = []

    def fake_fit_dataframe(_df, **kwargs):
        """Return deterministic fitted rows while recording call behavior."""
        calls.append(kwargs["method"])
        return pd.DataFrame(
            [
                {
                    "file": "trace.csv",
                    "num": 0,
                    "channel": 0,
                    "method": kwargs["method"],
                    "success": True,
                    "error": None,
                    "peak": 1.0,
                    "background": 0.0,
                    "peak_voltage": 0.0,
                    "peak_index": 0,
                    "fw_prominence": 1.0,
                    "voltage": volts.tolist(),
                    "current": (1.0 + volts).tolist(),
                    "peak_profile": (1.0 + volts).tolist(),
                    "background_profile": np.zeros_like(volts).tolist(),
                    "fitted_signal": (1.0 + volts).tolist(),
                }
            ]
        )

    monkeypatch.setattr(viewer, "fit_dataframe", fake_fit_dataframe)

    loaded = viewer._load_results_from_upload(
        (("trace.csv", trace_df.to_csv(index=False).encode("utf-8")),),
        "poly_linear",
        1,
        "Uploaded order",
    )

    assert calls == ["poly_linear"]
    assert loaded["method"].tolist() == ["poly_linear"]


def test_upload_trace_csv_impl_reports_progress(monkeypatch) -> None:
    """Verify upload trace csv impl reports progress."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    volts = np.linspace(-0.4, 0.0, 20)
    trace_df = pd.DataFrame(
        {
            "file": ["trace.csv", "trace.csv"],
            "num": [0, 1],
            "channel": [0, 0],
            "voltage": [volts.tolist(), volts.tolist()],
            "current": [(1.0 + volts).tolist(), (1.1 + volts).tolist()],
        }
    )
    updates = []

    class Progress:
        """Progress stub that records updates from viewer fitting helpers."""
        @staticmethod
        def update(completed, total, label=None, *, force=False):
            """Record or display progress for a fitting operation."""
            updates.append((completed, total, label, force))

        def callback(self, label=None):
            """Return a callback compatible with batch fitting progress hooks."""
            return lambda completed, total: self.update(completed, total, label)

    def fake_fit_dataframe(df, **kwargs):
        """Return deterministic fitted rows while recording call behavior."""
        callback = kwargs.get("progress_callback")
        assert callback is not None
        callback(len(df), len(df))
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "file": row["file"],
                    "num": row["num"],
                    "channel": row["channel"],
                    "method": kwargs["method"],
                    "success": True,
                    "error": None,
                    "peak": 1.0,
                    "background": 0.0,
                    "peak_voltage": 0.0,
                    "peak_index": 0,
                    "fw_prominence": 1.0,
                    "voltage": row["voltage"],
                    "current": row["current"],
                    "peak_profile": row["current"],
                    "background_profile": [0.0] * len(row["current"]),
                    "fitted_signal": row["current"],
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(viewer, "fit_dataframe", fake_fit_dataframe)

    loaded = viewer._load_results_from_upload_impl(
        (("trace.csv", trace_df.to_csv(index=False).encode("utf-8")),),
        "aswift",
        1,
        "Uploaded order",
        _progress=Progress(),
    )

    assert len(loaded) == 2
    assert updates[0] == (0, 2, "Fitting rows", True)
    assert updates[-1] == (2, 2, "Fitting rows", False)


def test_viewer_worker_configuration_helpers(monkeypatch) -> None:
    """Verify viewer backend selection and bounded worker defaults."""
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    assert viewer._parallel_backend_for_workers(None) == "thread"
    assert viewer._parallel_backend_for_workers(1) == "thread"
    assert viewer._parallel_backend_for_workers(2) == "process"

    monkeypatch.setattr(viewer.os, "cpu_count", lambda: 12)
    assert viewer._max_worker_count() == 12
    assert viewer._default_worker_count() == 8

    monkeypatch.setattr(viewer.os, "cpu_count", lambda: 4)
    assert viewer._max_worker_count() == 4
    assert viewer._default_worker_count() == 4

    monkeypatch.setattr(viewer.os, "cpu_count", lambda: None)
    assert viewer._max_worker_count() == viewer.UNKNOWN_CPU_WORKER_LIMIT
    assert viewer._default_worker_count() == 1


def test_process_backend_matches_single_worker_results() -> None:
    """Verify process backend matches single worker results."""
    df = synthetic_trace_dataframe().head(4)
    from aswift import fit_dataframe

    serial = fit_dataframe(df, method="aswift", n_workers=1)
    updates = []
    process = fit_dataframe(
        df,
        method="aswift",
        n_workers=2,
        parallel_backend="process",
        chunksize=1,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert process["file"].tolist() == serial["file"].tolist()
    assert process["success"].tolist() == serial["success"].tolist()
    assert process["peak"].to_numpy() == pytest.approx(serial["peak"].to_numpy())
    assert process["peak_voltage"].to_numpy() == pytest.approx(serial["peak_voltage"].to_numpy())
    assert updates[-1] == (len(df), len(df))


def test_viewer_filtering_builds_dense_options_and_rejects_missing_sample() -> None:
    """Verify filtered sample options are dense and invalid combinations stay empty."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import (
        _filter_global_selection,
        _filter_selected_results,
        _sample_options,
    )

    results = pd.DataFrame(
        {
            "num": [0, 1, 0],
            "hz": [150, 150, 250],
            "channel": [0, 0, 1],
            "method": ["aswift", "aswift", "aswift"],
        }
    )
    shorter_group = _filter_global_selection(results, selected_channel=1, selected_hz=250)
    options = _sample_options(shorter_group)

    assert len(options) == 1
    assert options.iloc[0]["num"] == 0

    selected_sample = pd.Series({"num": 1})

    filtered = _filter_selected_results(
        results,
        selected_channel=1,
        selected_hz=250,
        selected_sample=selected_sample,
    )

    assert filtered.empty


def test_live_pssession_time_normalization_after_cache_changes() -> None:
    """Verify live pssession time normalization after cache changes."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _normalize_result_time

    results = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
            "time": [99.0, 100.0],
        }
    )

    normalized = _normalize_result_time(results)

    assert normalized["time"].tolist() == pytest.approx([0.0, 2.0])


def test_live_pssession_new_files_are_fit_in_one_batch(tmp_path, monkeypatch) -> None:
    """Verify live pssession new files are fit in one batch."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    for idx in range(3):
        path = tmp_path / f"sample-{idx}.pssession"
        path.write_text("fake")
        path.touch()

    monkeypatch.setattr(viewer.st, "session_state", {})

    def fake_file_to_dataframe(f):
        """Return deterministic trace rows for a fake PalmSens file."""
        return pd.DataFrame(
            [
                {
                    "file": f.name,
                    "num": 0,
                    "channel": 0,
                    "voltage": [0.0, 1.0],
                    "current": [1.0, 2.0],
                }
            ]
        )

    calls = []

    def fake_fit_dataframe(df, **kwargs):
        """Return deterministic fitted rows while recording call behavior."""
        calls.append((len(df), kwargs))
        if kwargs.get("progress_callback") is not None:
            kwargs["progress_callback"](len(df), len(df))
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "file": row["file"],
                    "num": row["num"],
                    "channel": row["channel"],
                    "method": kwargs["method"],
                    "success": True,
                    "error": None,
                    "peak": 1.0,
                    "background": 0.0,
                    "peak_voltage": 0.0,
                    "peak_index": 0,
                    "voltage": row["voltage"],
                    "current": row["current"],
                    "peak_profile": row["current"],
                    "background_profile": [0.0, 0.0],
                    "fitted_signal": row["current"],
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(viewer, "_pssession_file_to_dataframe", fake_file_to_dataframe)
    monkeypatch.setattr(viewer, "fit_dataframe", fake_fit_dataframe)

    results = viewer._load_results_from_live_pssession_folder(str(tmp_path), "aswift", 16)

    assert len(results) == 3
    assert not viewer._live_folder_has_changes(str(tmp_path), "aswift")
    assert len(calls) == 1
    assert calls[0][0] == 3
    assert calls[0][1]["parallel_backend"] == "process"
    assert calls[0][1]["chunksize"] == viewer.FIT_CHUNKSIZE

    viewer._load_results_from_live_pssession_folder(str(tmp_path), "aswift", 16)

    assert len(calls) == 1

    new_path = tmp_path / "sample-3.pssession"
    new_path.write_text("fake")
    assert viewer._live_folder_has_changes(str(tmp_path), "aswift")


def test_startup_json_path_loads_precomputed_results(tmp_path) -> None:
    """Verify startup json path loads precomputed results."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _file_signature, _load_results_from_startup_path

    volts, current = synthetic_single_trace()
    result = aswift_fit(volts, current)
    rows = pd.DataFrame(
        [
            {
                "method": result.method,
                "success": result.success,
                "error": result.error,
                "peak": result.peak_signal,
                "background": result.peak_background,
                "peak_voltage": result.peak_voltage,
                "peak_index": result.peak_index,
                "voltage": result.volts.tolist(),
                "current": result.current.tolist(),
                "peak_profile": result.peak_profile.tolist(),
                "background_profile": result.background_profile.tolist(),
            }
        ]
    )
    path = tmp_path / "fit_results.json"
    rows.to_json(path, orient="records", indent=2)

    loaded = _load_results_from_startup_path(str(path), "aswift", 1, _file_signature(str(path)))

    assert len(loaded) == 1
    assert loaded.iloc[0]["success"]
    assert loaded.iloc[0]["peak"] == pytest.approx(result.peak_signal)


def test_viewer_trend_and_filter_helpers() -> None:
    """Verify viewer trend and filter helpers."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import (
        _filter_global_selection,
        _trend_folder_label,
        _trend_metric_options,
    )

    metric_results = pd.DataFrame(
        {
            "peak": [1.0],
            "norm_signal": [1.0],
            "peak_voltage": [-0.2],
            "fw_prominence": [0.05],
        }
    )

    assert _trend_metric_options(metric_results, normalize=False) == {
        "Peak height": "peak",
        "Peak voltage (V)": "peak_voltage",
        "Peak width (mV)": "fw_prominence",
    }
    assert _trend_metric_options(metric_results, normalize=True)["Normalized peak height"] == "norm_signal"

    assert _trend_folder_label(
        pd.Series({"relative_folder": "plate-a/day-1", "folder": "/tmp/root"})
    ) == "plate-a/day-1"
    assert _trend_folder_label(pd.Series({"relative_folder": "", "folder": "/tmp/root/day-2"})) == "day-2"
    assert _trend_folder_label(pd.Series({})) == "All data"

    filter_results = pd.DataFrame(
        {
            "relative_folder": ["plate-a", "plate-b", "plate-a"],
            "folder": ["/tmp/root/plate-a", "/tmp/root/plate-b", "/tmp/root/plate-a"],
            "channel": [0, 0, 1],
            "hz": [150, 150, 250],
            "peak": [1.0, 2.0, 3.0],
        }
    )

    filtered = _filter_global_selection(
        filter_results,
        selected_folder="plate-a",
        selected_channel=0,
        selected_hz=150,
    )

    assert filtered["peak"].tolist() == [1.0]


def test_signal_trend_legend_toggles_individual_channels(monkeypatch) -> None:
    """Verify clicking a trend legend item does not hide every channel in its folder."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    import aswift.analysis.structured_results_viewer as viewer

    results = pd.DataFrame(
        {
            "relative_folder": ["plate-a", "plate-a", "plate-a", "plate-a"],
            "channel": [0, 0, 1, 1],
            "hz": [250, 250, 250, 250],
            "num": [0, 1, 0, 1],
            "peak": [1.0, 1.1, 2.0, 2.1],
        }
    )
    captured = {}

    monkeypatch.setattr(viewer.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(viewer.st, "columns", lambda *args, **kwargs: [nullcontext(), nullcontext()])
    monkeypatch.setattr(viewer.st, "selectbox", lambda label, options: options[0])
    monkeypatch.setattr(viewer.st, "plotly_chart", lambda figure, **kwargs: captured.setdefault("figure", figure))

    viewer._interactive_signal_trend(results, selected_hz=250, normalize=False)

    figure = captured["figure"]
    assert len(figure.data) == 2
    assert figure.data[0].legendgroup == figure.data[1].legendgroup == "plate-a"
    assert figure.layout.legend.groupclick == "toggleitem"


def test_live_pssession_folder_discovers_nested_files(tmp_path, monkeypatch) -> None:
    """Verify live pssession folder discovers nested files."""
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    nested = tmp_path / "experiment-a" / "day-1"
    nested.mkdir(parents=True)
    top_file = tmp_path / "top.pssession"
    nested_file = nested / "nested.pssession"
    mp3_file = nested / "renamed.pssession.mp3"
    top_file.write_text("fake")
    nested_file.write_text("fake")
    mp3_file.write_text("fake")

    monkeypatch.setattr(viewer, "strip_mp3_suffix_from_pssession_files", lambda folder, recursive=False: [])

    discovered = viewer._current_pssession_files(tmp_path)

    assert top_file in discovered
    assert nested_file in discovered
