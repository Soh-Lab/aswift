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

    Attributes:
        baseline_boundary: Fraction of points to protect at each edge when
            locating the background/peak window.
        bg_buffer: Extra points to include on each side of the prominence-based
            peak window before background fitting.
        huber_reweight: Whether to downweight large residuals during smoothing.
        huber_cutoff: Scaled residual threshold used by the Huber weighting rule.
        mad_window: Rolling median absolute deviation window for local noise
            estimation.
        peak_lambda_scale: Multiplier applied to the selected peak-smoothing
            lambda.
        peak_prominence: Relative height used when narrowing the final peak
            fitting window.
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
    """Numerical settings for the polynomial plus local linear baseline fit.

    Attributes:
        savgol_window: Odd-length window used by the Savitzky-Golay smoother.
        savgol_degree: Polynomial degree for the Savitzky-Golay smoother.
        polynomial_degree: Degree of the global polynomial fitted to the
            smoothed trace.
    """

    savgol_window: int = 15
    savgol_degree: int = 3
    polynomial_degree: int = 15


@dataclass
class FitResult:
    """One fitted voltammogram, with arrays ready for plotting.

    Attributes:
        method: Fitting method name, such as ``"aswift"`` or ``"poly_linear"``.
        volts: One-dimensional input voltage/potential array.
        current: One-dimensional input current/signal array.
        peak_signal: Background-subtracted fitted peak height.
        peak_background: Fitted background value at ``peak_index``.
        peak_voltage: Voltage at the detected peak maximum.
        peak_index: Integer index of the detected peak maximum, or ``-1`` for
            a failed fit.
        fw_prominence: Full-prominence peak width in voltage units.
        peak_profile: Full-length fitted peak-only profile. ASWIFT stores
            ``NaN`` outside the fitted peak window.
        background_profile: Full-length fitted background profile.
        params: Method-specific diagnostic values.
        success: Whether fitting completed successfully.
        error: Error text for failed fits, otherwise ``None``.
    """

    method: str
    volts: NDArray[np.float64]
    current: NDArray[np.float64]
    peak_signal: float
    peak_background: float
    peak_voltage: float
    peak_index: int
    fw_prominence: float
    peak_profile: NDArray[np.float64]
    background_profile: NDArray[np.float64]
    params: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    @property
    def fitted_current(self) -> NDArray[np.float64]:
        """Full fitted current profile: ``peak_profile + background_profile``."""
        return self.peak_profile + self.background_profile

    def to_record(self, **metadata: Any) -> dict[str, Any]:
        """Return a JSON-friendly result row.

        Args:
            **metadata: Extra key/value pairs to include in the returned record.

        Returns:
            A dictionary containing scalar fit metrics, serialized fitting
            profiles, and any supplied metadata.
        """
        record = {
            **metadata,
            "method": self.method,
            "success": self.success,
            "error": self.error,
            "peak": self.peak_signal,
            "background": self.peak_background,
            "peak_voltage": self.peak_voltage,
            "peak_index": self.peak_index,
            "fw_prominence": self.fw_prominence,
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
        """Compatibility index separating peak and background values in ``popt``."""
        return int(self.peak_profile.size)

    @property
    def popt(self) -> NDArray[np.float64]:
        """Compatibility vector containing peak profile followed by background."""
        return np.concatenate([self.peak_profile, self.background_profile])


def failed_fit_result(
    method: str,
    volts,
    current,
    error: Exception | str,
) -> FitResult:
    """Build a failed ``FitResult`` that preserves input arrays.

    Args:
        method: Name of the attempted fitting method.
        volts: Input voltage/potential values.
        current: Input current/signal values.
        error: Exception or message describing the failure.

    Returns:
        A ``FitResult`` with ``success=False`` and ``NaN`` fit profiles.
    """
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
        fw_prominence=np.nan,
        peak_profile=empty.copy(),
        background_profile=empty.copy(),
        success=False,
        error=str(error),
    )
