"""Typed return objects and settings for public fitting APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AswiftSettings:
    """Numerical settings for the ASWIFT fit.

    ASWIFT first estimates a smooth current trace, uses derpsalsa to estimate
    the slowly varying background, and then fits the positive peak with a
    Tikhonov-smoothed profile inside the detected peak window.
    """

    baseline_boundary: float = 0.05
    bg_buffer: int = 0
    huber_reweight: bool = True
    huber_cutoff: float = 2.0
    mad_window: int = 21
    peak_lambda_scale: float = 1.0
    peak_prominence: float = 0.5


@dataclass(frozen=True)
class PolyLinearSettings:
    """Numerical settings for the polynomial plus local linear baseline fit."""

    savgol_window: int = 15
    savgol_degree: int = 3
    polynomial_degree: int = 15


@dataclass
class FitResult:
    """One fitted voltammogram, with arrays ready for plotting."""

    method: str
    volts: NDArray[np.float64]
    current: NDArray[np.float64]
    peak_signal: float
    peak_background: float
    peak_voltage: float
    peak_index: int
    peak_profile: NDArray[np.float64]
    background_profile: NDArray[np.float64]
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    @property
    def fitted_current(self) -> NDArray[np.float64]:
        return self.peak_profile + self.background_profile

    def to_record(self, **metadata: Any) -> dict[str, Any]:
        """Return a JSON-friendly row with optional caller-supplied metadata."""
        record = {
            **metadata,
            "method": self.method,
            "success": self.success,
            "error": self.error,
            "peak": self.peak_signal,
            "background": self.peak_background,
            "peak_voltage": self.peak_voltage,
            "peak_index": self.peak_index,
            "popt": self.popt.tolist(),
            "bg_idx": self.bg_idx,
        }
        if "peak_window" in self.params:
            start, end = self.params["peak_window"]
            record["peak_window_start"] = int(start)
            record["peak_window_end"] = int(end)
        return record

    @property
    def bg_idx(self) -> int:
        return int(self.peak_profile.size)

    @property
    def popt(self) -> NDArray[np.float64]:
        return np.concatenate([self.peak_profile, self.background_profile])


def failed_fit_result(
    method: str,
    volts,
    current,
    error: Exception | str,
) -> FitResult:
    """Build a FitResult that preserves input arrays when fitting fails."""
    volts_arr = np.asarray(volts, dtype=float)
    current_arr = np.asarray(current, dtype=float)
    empty = np.full_like(current_arr, np.nan, dtype=float)
    return FitResult(
        method=method,
        volts=volts_arr,
        current=current_arr,
        peak_signal=np.nan,
        peak_background=np.nan,
        peak_voltage=np.nan,
        peak_index=-1,
        peak_profile=empty.copy(),
        background_profile=empty.copy(),
        success=False,
        error=str(error),
    )
