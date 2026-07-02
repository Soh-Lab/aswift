from __future__ import annotations

import time
from io import StringIO

import numpy as np
import pytest

from aswift import aswift_fit, poly_linear_fit


def synthetic_single_trace(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    volts = np.linspace(-0.4, 0.0, 400)
    background = 3.0 * (volts + 0.1) ** 2 + 1.0
    peak = 0.45 * np.exp(-0.5 * ((volts + 0.2) / 0.045) ** 2)
    current = background + peak + rng.normal(loc=0.0, scale=0.01, size=volts.size)
    return volts, current


def synthetic_trace_dataframe():
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


def test_notebook_01_synthetic_single_trace_fits() -> None:
    volts, current = synthetic_single_trace()

    started = time.perf_counter()
    aswift_result = aswift_fit(volts, current)
    poly_result = poly_linear_fit(volts, current)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0
    for result in (aswift_result, poly_result):
        assert result.success
        assert result.error is None
        assert np.isfinite(result.peak_signal)
        assert np.isfinite(result.fw_prominence)
        assert result.fw_prominence > 0
        assert 0.25 < result.peak_signal < 0.6
        assert -0.25 < result.peak_voltage < -0.15
        assert result.fitted_current.shape == volts.shape


def test_notebook_02_synthetic_dataframe_batch_and_signal_table() -> None:
    df = synthetic_trace_dataframe()
    from aswift import fit_dataframe, results_to_signal_table

    started = time.perf_counter()
    results = fit_dataframe(df, method="aswift", n_workers=1)
    elapsed = time.perf_counter() - started

    assert elapsed < 20.0
    assert len(results) == len(df)
    assert results["success"].all()
    assert set(["hz", "num", "channel", "peak", "peak_voltage"]).issubset(results.columns)
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


def test_simple_csv_with_shared_voltage_header_converts_to_trace_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _simple_csv_to_trace_dataframe

    volts = np.linspace(-0.4, 0.0, 20)
    df = pd.DataFrame(
        {
            "voltage": volts,
            "current_0": 1.0 + volts,
            "current_1": 1.1 + volts,
        }
    )

    traces = _simple_csv_to_trace_dataframe(df)

    assert traces is not None
    assert len(traces) == 2
    assert traces["channel"].tolist() == [0, 1]
    assert traces.iloc[0]["current"] == pytest.approx((1.0 + volts).tolist())
    assert traces.iloc[1]["voltage"] == pytest.approx(volts.tolist())


def test_simple_csv_without_headers_preserves_first_row() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _read_csv_for_viewer, _simple_csv_to_trace_dataframe

    volts = np.linspace(-0.4, 0.0, 20)
    raw = pd.DataFrame({0: volts, 1: 1.0 + volts, 2: 1.1 + volts})

    traces = _simple_csv_to_trace_dataframe(_read_csv_for_viewer(StringIO(raw.to_csv(index=False, header=False))))

    assert traces is not None
    assert len(traces) == 2
    assert traces.iloc[0]["current"] == pytest.approx((1.0 + volts).tolist())
    assert traces.iloc[0]["voltage"] == pytest.approx(volts.tolist())


def test_simple_csv_upload_reads_utf16_shared_voltage() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _read_csv_payload_for_viewer, _simple_csv_to_trace_dataframe

    volts = np.linspace(-0.4, 0.0, 20)
    raw = pd.DataFrame({"V": volts, "uA": 1.0 + volts})
    payload = raw.to_csv(index=False).encode("utf-16")

    traces = _simple_csv_to_trace_dataframe(_read_csv_payload_for_viewer(payload))

    assert traces is not None
    assert len(traces) == 1
    assert traces.iloc[0]["current"] == pytest.approx((1.0 + volts).tolist())
    assert traces.iloc[0]["voltage"] == pytest.approx(volts.tolist())


def test_simple_csv_with_current_voltage_pairs_still_converts() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _simple_csv_to_trace_dataframe

    volts = np.linspace(-0.4, 0.0, 20)
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
    assert len(traces) == 2
    assert traces["channel"].tolist() == [0, 1]
    assert traces.iloc[0]["current"] == pytest.approx((1.0 + volts).tolist())
    assert traces.iloc[1]["voltage"] == pytest.approx(volts.tolist())


def test_single_row_sample_selection_does_not_use_slider(monkeypatch) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    class Sidebar:
        def __init__(self) -> None:
            self.slider_called = False

        def caption(self, text: str) -> None:
            assert text == "Sample index: 0"

        def slider(self, *args, **kwargs) -> int:
            self.slider_called = True
            raise AssertionError("slider should not be used for one row")

    sidebar = Sidebar()
    monkeypatch.setattr(viewer.st, "sidebar", sidebar)

    assert viewer._select_sample_position(1) == 0
    assert not sidebar.slider_called


def test_signal_table_pads_unequal_group_lengths() -> None:
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


def test_batch_default_workers_are_single_threaded() -> None:
    pytest.importorskip("pandas")
    from aswift.peak_extraction.batch import _normalize_workers

    assert _normalize_workers(None, 10) == 1


def test_fit_dataframe_default_uses_single_worker_fast_path(monkeypatch) -> None:
    df = synthetic_trace_dataframe().head(2)
    import aswift.peak_extraction.batch as batch

    def fail_process_pool(*args, **kwargs):
        raise AssertionError("default fit_dataframe should not use process workers")

    monkeypatch.setattr(batch, "_fit_traces_process_pool", fail_process_pool)

    results = batch.fit_dataframe(df, method="aswift")

    assert len(results) == len(df)
    assert results["success"].all()


def test_fit_dataframe_reports_progress_for_single_worker() -> None:
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


def test_plot_labels_are_unit_agnostic() -> None:
    pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    from aswift import aswift_fit
    from aswift.peak_extraction.batch import plot_fit_result, plot_signal_over_time

    volts, current = synthetic_single_trace()
    result = aswift_fit(volts, current)
    _, fit_ax = plt.subplots()
    plot_fit_result(result, ax=fit_ax)

    results = synthetic_trace_dataframe().head(2).copy()
    results["peak"] = [1.0, 1.1]
    results["method"] = "aswift"
    _, trend_ax = plt.subplots()
    plot_signal_over_time(results, ax=trend_ax)

    assert fit_ax.get_xlabel() == "Input"
    assert fit_ax.get_ylabel() == "Signal"
    assert trend_ax.get_xlabel() == "Time"
    assert trend_ax.get_ylabel() == "Signal"
    plt.close("all")


def test_signal_trend_uses_index_axis_without_time() -> None:
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")
    from aswift.peak_extraction.batch import plot_signal_over_time

    results = pd.DataFrame(
        {
            "method": ["aswift", "aswift"],
            "num": [0, 1],
            "peak": [1.0, 1.2],
        }
    )
    _, ax = plt.subplots()
    plot_signal_over_time(results, ax=ax)

    assert ax.get_xlabel() == "Index Number"
    assert ax.get_ylabel() == "Signal"
    plt.close("all")


def test_viewer_peak_range_normalization_adds_norm_signal_per_group() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _add_norm_signal_by_peak_range

    results = pd.DataFrame(
        {
            "method": ["aswift"] * 4,
            "hz": [100, 100, 200, 200],
            "channel": [0, 0, 0, 0],
            "num": [0, 1, 0, 1],
            "peak": [2.0, 4.0, 10.0, 20.0],
            "background": [1.0, 2.0, 5.0, 10.0],
            "current": [[2.0, 4.0], [4.0, 8.0], [10.0, 20.0], [20.0, 40.0]],
            "peak_profile": [[2.0], [4.0], [10.0], [20.0]],
            "background_profile": [[1.0], [2.0], [5.0], [10.0]],
            "fitted_signal": [[3.0], [6.0], [15.0], [30.0]],
        }
    )

    normalized = _add_norm_signal_by_peak_range(results, 0, 1)

    assert normalized["peak"].tolist() == pytest.approx([2.0, 4.0, 10.0, 20.0])
    assert normalized["background"].tolist() == pytest.approx([1.0, 2.0, 5.0, 10.0])
    assert normalized["norm_signal"].tolist() == pytest.approx([2 / 3, 4 / 3, 10 / 15, 20 / 15])
    assert normalized.iloc[0]["current"] == pytest.approx([2.0, 4.0])
    assert normalized.iloc[2]["background_profile"] == pytest.approx([5.0])


def test_viewer_results_download_hides_internal_columns_and_keeps_norm_signal() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _downloadable_results

    results = pd.DataFrame(
        {
            "peak": [3.0],
            "background": [1.0],
            "peak_index": [2],
            "peak_idx": [2],
            "peak_window_start": [1],
            "peak_window_end": [4],
            "normalization_start_index": [0],
            "normalization_end_index": [1],
            "normalization_reference_peak": [3.0],
            "voltage": [[0.0, 1.0]],
            "current": [[1.0, 2.0]],
            "peak_profile": [[0.0, 1.0]],
        }
    )

    download = _downloadable_results(results)

    assert "peak" in download.columns
    assert "norm_signal" in download.columns
    assert list(download.columns[:3]) == ["peak", "norm_signal", "background"]
    assert pd.isna(download.loc[0, "norm_signal"])
    assert "peak_index" not in download.columns
    assert "peak_idx" not in download.columns
    assert "peak_window_start" not in download.columns
    assert "peak_window_end" not in download.columns
    assert "normalization_start_index" not in download.columns
    assert "normalization_end_index" not in download.columns
    assert "normalization_reference_peak" not in download.columns
    assert "voltage" not in download.columns
    assert "current" not in download.columns
    assert "peak_profile" not in download.columns


def test_viewer_results_download_preserves_computed_norm_signal() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import (
        _add_norm_signal_by_peak_range,
        _downloadable_results,
    )

    results = pd.DataFrame(
        {
            "method": ["aswift", "aswift"],
            "hz": [100, 100],
            "channel": [0, 0],
            "num": [0, 1],
            "peak": [2.0, 4.0],
            "background": [1.0, 2.0],
        }
    )

    normalized = _add_norm_signal_by_peak_range(results, 0, 1)
    download = _downloadable_results(normalized)

    assert download[["peak", "norm_signal", "background"]].columns.tolist() == [
        "peak",
        "norm_signal",
        "background",
    ]
    assert download["norm_signal"].tolist() == pytest.approx([2 / 3, 4 / 3])


def test_result_summary_values_are_display_safe_strings() -> None:
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


def test_upload_trace_csv_uses_selected_fit_method(monkeypatch) -> None:
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

    def fake_fit_dataframe(df, **kwargs):
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


def test_structured_trace_processing_reports_progress(monkeypatch) -> None:
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
        def update(self, completed, total, label=None, *, force=False):
            updates.append((completed, total, label, force))

        def callback(self, label=None):
            return lambda completed, total: self.update(completed, total, label)

    def fake_fit_dataframe(df, **kwargs):
        callback = kwargs.get("progress_callback")
        assert callback is not None
        callback(1, len(df))
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

    loaded = viewer._prepare_table_or_fit_traces(
        trace_df,
        method="aswift",
        n_workers=1,
        _progress=Progress(),
    )

    assert len(loaded) == 2
    assert updates[0] == (0, 2, "Fitting rows", True)
    assert updates[-1] == (2, 2, "Fitting rows", False)


def test_upload_trace_csv_impl_reports_progress(monkeypatch) -> None:
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
        def update(self, completed, total, label=None, *, force=False):
            updates.append((completed, total, label, force))

        def callback(self, label=None):
            return lambda completed, total: self.update(completed, total, label)

    def fake_fit_dataframe(df, **kwargs):
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


def test_viewer_uses_process_backend_only_when_workers_exceed_one() -> None:
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _parallel_backend_for_workers

    assert _parallel_backend_for_workers(None) == "thread"
    assert _parallel_backend_for_workers(1) == "thread"
    assert _parallel_backend_for_workers(2) == "process"


def test_process_backend_matches_single_worker_results() -> None:
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


def test_missing_frequency_channel_sample_combination_returns_empty_filter() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _filter_selected_results

    results = pd.DataFrame(
        {
            "num": [0, 1, 0],
            "hz": [150, 150, 250],
            "channel": [0, 0, 1],
            "method": ["aswift", "aswift", "aswift"],
        }
    )
    selected_sample = pd.Series({"num": 1})

    filtered = _filter_selected_results(
        results,
        selected_channel=1,
        selected_hz=250,
        selected_sample=selected_sample,
    )

    assert filtered.empty


def test_sample_options_are_dense_after_frequency_channel_filtering() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    from aswift.analysis.structured_results_viewer import _filter_global_selection, _sample_options

    results = pd.DataFrame(
        {
            "num": [0, 1, 2, 0],
            "hz": [150, 150, 150, 250],
            "channel": [0, 0, 0, 0],
            "method": ["aswift", "aswift", "aswift", "aswift"],
        }
    )

    shorter_group = _filter_global_selection(results, selected_channel=0, selected_hz=250)
    options = _sample_options(shorter_group)

    assert len(options) == 1
    assert options.iloc[0]["num"] == 0


def test_live_pssession_time_normalization_after_cache_changes() -> None:
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
    pd = pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import aswift.analysis.structured_results_viewer as viewer

    for idx in range(3):
        path = tmp_path / f"sample-{idx}.pssession"
        path.write_text("fake")
        path.touch()

    monkeypatch.setattr(viewer.st, "session_state", {})

    def fake_file_to_dataframe(path):
        return pd.DataFrame(
            [
                {
                    "file": path.name,
                    "num": 0,
                    "channel": 0,
                    "voltage": [0.0, 1.0],
                    "current": [1.0, 2.0],
                }
            ]
        )

    calls = []

    def fake_fit_dataframe(df, **kwargs):
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
