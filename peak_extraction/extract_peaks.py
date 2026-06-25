"""Signal calculation methods.

ASWIFT models SWV peak extraction as a sequence of weighted Tikhonov
regularization problems: robust whole-trace smoothing, lower-envelope
derpsalsa background fitting, local peak smoothing, and final peak-height
calculation. The public functions return full profiles so callers can inspect
or plot every fitted component.
"""

import math

import numpy as np
from pybaselines import Baseline
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

from peak_extraction.config import config
from peak_extraction.io import get_volts_array
from peak_extraction.models import ASwiftSettings, FitResult, PolyLinearSettings


ASWIFT_BACKGROUND_METHOD = "derpsalsa_iter"
ASWIFT_PEAK_METHOD = "tikhonov"
SUPPORTED_FITTING_METHODS = ("aswift", "poly_linear")


def _as_array(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if arr.size < 5:
        raise ValueError(f"{name} must contain at least 5 points")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _validate_trace(volts, current) -> tuple[np.ndarray, np.ndarray]:
    volts_arr = _as_array("volts", volts)
    current_arr = _as_array("current", current)
    if volts_arr.shape != current_arr.shape:
        raise ValueError("volts and current must have the same shape")
    if np.ptp(volts_arr) == 0:
        raise ValueError("volts must span more than one value")
    return volts_arr, current_arr


def aswift_settings_from_config() -> ASwiftSettings:
    params = config.parameters
    return ASwiftSettings(
        baseline_boundary=params.baseline_boundary,
        bg_buffer=params.bg_buffer,
        huber_reweight=params.huber_reweight,
        huber_cutoff=params.huber_cutoff,
        mad_window=params.mad_window,
        peak_lambda_scale=params.peak_lambda_scale,
        peak_prominence=params.peak_prominence,
    )


def poly_calculator(volts, *coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    return np.polyval(coeffs, volts)


def linear_calculator(volts, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    return coeffs[0] * volts + coeffs[1]


def calculate_solved_peak(volts, *peak):
    if isinstance(volts, float):
        volts_array = get_volts_array()
        idx = np.argmin(np.abs(volts_array - volts))
        return list(peak)[idx]

    peak_data = np.array(list(peak), dtype=float)
    peak_data[peak_data <= 0] = np.nan
    return peak_data


def calculate_solved_background(volts, background):
    if isinstance(volts, float):
        volts_array = get_volts_array()
        idx = np.argmin(np.abs(volts_array - volts))
        return background[idx]

    return np.asarray(background, dtype=float)


def make_smoother_D2(size: int):
    """Create a second-derivative Tikhonov smoother.

    The returned solver minimizes ||W(y - g)||^2 + lambda ||D2 g||^2,
    where `g` is the smooth trace and `D2` penalizes curvature. The banded
    solve keeps repeated lambda evaluations fast enough for L-curve searches.
    """
    if size < 5:
        raise ValueError("size must be >= 5")

    main = np.empty(size, dtype=float)
    main[0] = 1.0
    main[1] = 5.0
    main[2:-2] = 6.0
    main[-2] = 5.0
    main[-1] = 1.0

    off1 = np.empty(size - 1, dtype=float)
    off1[0] = -2.0
    off1[1:-1] = -4.0
    off1[-1] = -2.0

    off2 = np.ones(size - 2, dtype=float)

    def smoother(y: np.ndarray, lam: float, weights: np.ndarray | None = None):
        y = np.asarray(y, dtype=float)
        if y.shape[0] != size:
            raise ValueError(f"y must have length {size}")

        if weights is None:
            weights = np.ones(size, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape[0] != size:
                raise ValueError(f"weights must have length {size}")
            if np.any(weights < 0):
                raise ValueError("weights must be >= 0")

        ab = np.zeros((3, size), dtype=float)
        ab[0, 2:] = lam * off2
        ab[1, 1:] = lam * off1
        ab[2, :] = weights + lam * main
        rhs = weights * y

        return solveh_banded(ab, rhs, lower=False, check_finite=False)

    return smoother


def mad_scale(residuals: np.ndarray, eps: float = 1e-12) -> float:
    residuals = np.asarray(residuals, dtype=float)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad + eps


def rolling_mad_scale(residuals: np.ndarray, window: int, eps: float = 1e-12) -> np.ndarray:
    window = max(int(window), 5)
    if window % 2 == 0:
        window += 1

    residuals = np.asarray(residuals, dtype=float)
    size = residuals.size
    if size == 0:
        return residuals.copy()

    if window > size:
        window = size if size % 2 == 1 else max(5, size - 1)
    if window < 5:
        return np.full(size, mad_scale(residuals, eps=eps), dtype=float)

    pad = window // 2
    padded = np.pad(residuals, pad_width=pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    med = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - med[:, None]), axis=-1)
    return np.maximum(1.4826 * mad + eps, eps)


def huber_irls_weights(residuals: np.ndarray, scale, cutoff: float) -> np.ndarray:
    """Return Huber-style IRLS weights from locally scaled residuals.

    Residuals below `cutoff * scale` keep unit weight; larger residuals are
    downweighted in inverse proportion to their scaled magnitude so outliers and
    distorted regions have less influence on lambda selection.
    """
    eps = 1e-12

    residuals = np.asarray(residuals, dtype=float)
    scale = np.asarray(scale, dtype=float)

    if scale.ndim == 0:
        u = residuals / scale if np.isfinite(scale) and scale > eps else residuals
    else:
        scale = np.broadcast_to(scale, residuals.shape)
        valid = np.isfinite(scale) & (scale > eps)
        u = np.empty_like(residuals, dtype=float)
        u[valid] = residuals[valid] / scale[valid]
        u[~valid] = residuals[~valid]

    abs_u = np.abs(u)
    if cutoff == 0:
        weights = 1.0 / np.maximum(abs_u, eps)
    else:
        weights = np.ones_like(abs_u)
        mask = abs_u > cutoff
        weights[mask] = cutoff / np.maximum(abs_u[mask], eps)

    weights[~np.isfinite(weights)] = 1.0
    return weights


def huber_smoother_D2(
    y: np.ndarray,
    lam: float,
    settings: ASwiftSettings,
    base_w: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
    min_w: float = 1e-8,
):
    y = np.asarray(y, dtype=float)
    size = y.size
    smoother = make_smoother_D2(size)

    if base_w is None:
        base_w = np.ones(size, dtype=float)
    else:
        base_w = np.asarray(base_w, dtype=float).copy()
        if base_w.shape[0] != size:
            raise ValueError("base_w must have the same length as y")
        if np.any(base_w < 0):
            raise ValueError("base_w must be >= 0")

    weights = np.maximum(base_w, min_w)
    smooth = smoother(y, lam, weights)

    for _ in range(max_iter):
        residuals = y - smooth
        scale = rolling_mad_scale(residuals, settings.mad_window)
        robust_weights = huber_irls_weights(residuals, scale, settings.huber_cutoff)
        new_weights = np.maximum(base_w * robust_weights, min_w)
        new_smooth = smoother(y, lam, new_weights)

        denom = np.linalg.norm(smooth) + 1e-12
        if np.linalg.norm(new_smooth - smooth) / denom < tol:
            return new_smooth, new_weights

        smooth, weights = new_smooth, new_weights

    return smooth, weights


def choose_lambda_lcurve(
    values,
    lam_bounds=(1e-5, 1e8),
    n_grid=200,
    weights=None,
    eps=1e-300,
):
    """Select a Tikhonov lambda from the L-curve corner.

    The L-curve compares fidelity to the observed trace against roughness of
    the smoothed trace. ASWIFT uses the maximum-curvature point as a data-driven
    tradeoff between following noise and oversmoothing the peak.
    """
    values = np.asarray(values, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("Input contains NaN or inf.")
    if weights is not None and not np.all(np.isfinite(weights)):
        raise ValueError("weights contains NaN or inf.")

    value_scale = np.nanmedian(np.abs(values))
    if not np.isfinite(value_scale) or value_scale <= 0:
        value_scale = np.nanmax(np.abs(values))
    if not np.isfinite(value_scale) or value_scale <= 0:
        value_scale = 1.0

    scaled_values = values / value_scale
    smoother = make_smoother_D2(len(scaled_values))
    lambdas = np.logspace(np.log10(lam_bounds[0]), np.log10(lam_bounds[1]), n_grid)
    residual_norm = np.full(n_grid, np.nan)
    roughness_norm = np.full(n_grid, np.nan)

    for i, lam in enumerate(lambdas):
        try:
            smooth = smoother(scaled_values, lam, weights)
            if not np.all(np.isfinite(smooth)):
                continue

            residual = scaled_values - smooth
            residual_norm[i] = np.sum(residual**2 if weights is None else weights * residual**2)
            roughness_norm[i] = np.sum(np.diff(smooth, n=2) ** 2)
        except Exception:
            continue

    valid = (
        np.isfinite(residual_norm)
        & np.isfinite(roughness_norm)
        & (residual_norm > 0)
        & (roughness_norm > 0)
    )
    if valid.sum() < 5:
        raise RuntimeError("Too few valid lambda values for L-curve selection.")

    valid_lambdas = lambdas[valid]
    x = np.log(residual_norm[valid] + eps)
    y = np.log(roughness_norm[valid] + eps)
    t = np.log(valid_lambdas)

    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt**2 + dy_dt**2) ** 1.5 + eps)

    finite = np.isfinite(curvature)
    if not np.any(finite):
        raise RuntimeError("Curvature calculation failed.")

    if np.any(curvature[finite] > 0):
        candidate_idx = np.where(finite & (curvature > 0))[0]
        best_idx = candidate_idx[np.argmax(curvature[candidate_idx])]
    else:
        candidate_idx = np.where(finite)[0]
        best_idx = candidate_idx[np.argmax(np.abs(curvature[candidate_idx]))]

    best_lam = valid_lambdas[best_idx]
    best_smooth = smoother(scaled_values, best_lam, weights) * value_scale
    params = {
        "lambdas": lambdas,
        "residual_norm": residual_norm,
        "roughness_norm": roughness_norm,
        "valid": valid,
        "curvature_lambdas": valid_lambdas,
        "curvature": curvature,
        "weights": weights,
        "value_scale": value_scale,
    }

    return best_lam, best_smooth, params


def huber_reweighted_lcurve(
    current,
    pilot_smooth,
    settings: ASwiftSettings,
    n_grid=200,
    lam_bounds=(1e-1, 1e8),
):
    """Repeat L-curve smoothing with Huber IRLS weights.

    Large residuals receive smaller weights so sharp peak structure and
    occasional artifacts do not dominate the smooth-background estimate.
    """
    residuals = current - pilot_smooth
    scale = rolling_mad_scale(residuals, settings.mad_window)
    weights = np.maximum(huber_irls_weights(residuals, scale, settings.huber_cutoff), 1e-8)

    best_lam, _, params = choose_lambda_lcurve(
        current,
        weights=weights,
        n_grid=n_grid,
        lam_bounds=lam_bounds,
    )
    best_smooth, final_weights = huber_smoother_D2(current, best_lam, settings)
    params["weights"] = final_weights
    return best_lam, best_smooth, params


def get_background_range(current, settings: ASwiftSettings, rel_height=1.0):
    """Find the dominant prominence-based peak window.

    ASWIFT assumes one primary redox peak and uses the largest-prominence peak
    to define the region excluded from baseline fitting and included in local
    peak smoothing.
    """
    boundary = settings.baseline_boundary
    peaks, props = find_peaks(current, prominence=(None, None))
    if len(peaks) == 0:
        raise ValueError("No peaks found")

    best_idx = np.argmax(props["prominences"])
    best_peak = peaks[best_idx]
    padding = math.ceil(boundary * len(current))

    widths = peak_widths(current, [best_peak], rel_height=min(rel_height, 1.0))
    lower, upper = math.floor(widths[2][0]), math.ceil(widths[3][0])
    idx_bound = max(best_peak - lower, upper - best_peak) + settings.bg_buffer
    lower = max(best_peak - idx_bound, padding)
    upper = min(best_peak + idx_bound, current.shape[0] - padding - 1)

    return lower, upper, best_peak


def choose_lambda_area(
    current,
    volts,
    settings: ASwiftSettings,
    lam_bounds=(1e-1, 1e7),
    search_space=50,
    refine_space=10,
    threshold=0.1,
):
    """Choose derpsalsa lambda by minimizing background area under the peak.

    After the smoothed trace identifies a peak window, candidate derpsalsa
    backgrounds are scored by the area they place inside that window. The
    selected background is the lowest plausible curve that still has support on
    both sides of the peak.
    """
    lower, upper, peak_idx = get_background_range(current, settings, rel_height=1.0)
    peak_indices = np.zeros_like(current, dtype=bool)
    peak_indices[lower:upper] = True
    baseline_fitter = Baseline(x_data=volts)

    def lambda_area(lam):
        background, params = baseline_fitter.derpsalsa(current, lam=lam)
        weights = params["weights"]
        if np.any(weights[:peak_idx] > threshold) and np.any(weights[peak_idx:] > threshold):
            return np.sum(background[peak_indices])
        return np.inf

    def eval_grid(lam_lo, lam_hi, n):
        lam_lo = max(lam_lo, np.finfo(float).tiny)
        lam_hi = max(lam_hi, lam_lo * 1e3)
        lambdas = np.logspace(np.log10(lam_lo), np.log10(lam_hi), n)
        areas = np.array([lambda_area(lam) for lam in lambdas])
        return lambdas, areas

    lambdas, areas = eval_grid(lam_bounds[0], lam_bounds[1], search_space)
    min_idx = np.argmin(areas)
    lower_refine = max(0, min_idx - 1)
    upper_refine = min(min_idx + 1, len(areas) - 1)

    refined_lambdas, refined_areas = eval_grid(
        lambdas[lower_refine],
        lambdas[upper_refine],
        refine_space,
    )
    best_lam = refined_lambdas[np.argmin(refined_areas)]
    best_background, _ = baseline_fitter.derpsalsa(current, lam=best_lam)
    return best_lam, best_background


def fit_derpsalsa_background_iterative(current, volts, settings: ASwiftSettings):
    """ASWIFT background fit: smooth the trace, then fit derpsalsa baseline."""
    lam_smoother, smooth, _ = choose_lambda_lcurve(current)
    if settings.huber_reweight:
        lam_smoother, smooth, _ = huber_reweighted_lcurve(current, smooth, settings)

    _, background = choose_lambda_area(smooth, volts, settings)
    lower, upper, _ = get_background_range(smooth, settings, rel_height=settings.peak_prominence)
    peak_indices = np.zeros_like(smooth, dtype=bool)
    peak_indices[lower:upper] = True

    return background, peak_indices, smooth


def fit_tikhonov_peak(current, volts, background, indices, settings: ASwiftSettings, smooth_current=None):
    """ASWIFT peak fit inside the detected background-excluded peak window."""
    peak_region = current[indices] - background[indices]
    peak_lambda_scale = settings.peak_lambda_scale

    if settings.huber_reweight:
        if smooth_current is None:
            _, smooth_current, _ = choose_lambda_lcurve(current)

        residuals = current - smooth_current
        scale = rolling_mad_scale(residuals, settings.mad_window)
        weights = np.maximum(huber_irls_weights(residuals, scale, settings.huber_cutoff), 1e-8)
        lam_peak, _, _ = choose_lambda_lcurve(peak_region, weights=weights[indices])
        peak_fit, _ = huber_smoother_D2(
            peak_region,
            lam=lam_peak * peak_lambda_scale,
            settings=settings,
            base_w=weights[indices],
        )
        noise_reference = residuals
    else:
        lam_peak, smooth_peak, _ = choose_lambda_lcurve(peak_region)
        smoother = make_smoother_D2(len(peak_region))
        peak_fit = smoother(peak_region, lam_peak * peak_lambda_scale)
        noise_reference = peak_region - smooth_peak

    peak_idx = np.argmax(peak_fit)
    peak_signal = peak_fit[peak_idx]
    if peak_signal < 2 * np.mean(np.abs(noise_reference)):
        raise ValueError("Peak smaller than 2 * MAE smoothed fit residuals")

    popt = np.zeros_like(current)
    popt = np.full_like(current, np.nan, dtype=float)
    popt[indices] = peak_fit
    return popt, peak_signal, len(popt)


def aswift_fit(volts, current, settings: ASwiftSettings | None = None) -> FitResult:
    """Fit one SWV trace with ASWIFT.

    Parameters are one-dimensional voltage and current arrays. The result
    contains the peak signal, the background at the peak, and full-length
    background/peak profiles suitable for plotting.
    """
    volts, current = _validate_trace(volts, current)
    settings = settings or ASwiftSettings()

    background, peak_indices, smooth_current = fit_derpsalsa_background_iterative(current, volts, settings)
    peak_profile, peak_signal, bg_idx = fit_tikhonov_peak(
        current,
        volts,
        background,
        peak_indices,
        settings,
        smooth_current=smooth_current,
    )

    # The reported signal is max_i(p_i - b_i) in the detected peak window.
    peak_idx = int(np.nanargmax(peak_profile))
    peak_background = background[peak_idx]
    return FitResult(
        method="aswift",
        volts=volts,
        current=current,
        peak_signal=float(peak_signal),
        peak_background=float(peak_background),
        peak_voltage=float(volts[peak_idx]),
        peak_index=peak_idx,
        peak_profile=peak_profile,
        background_profile=background,
        params={"settings": settings},
    )


def poly_linear_fit(volts, current, settings: PolyLinearSettings | None = None) -> FitResult:
    """Fit the legacy polynomial signal with a linear local baseline."""
    volts, current = _validate_trace(volts, current)
    settings = settings or PolyLinearSettings()
    if settings.savgol_window % 2 == 0:
        raise ValueError("savgol_window must be odd")
    if settings.savgol_window > current.size:
        raise ValueError("savgol_window cannot be longer than current")
    if settings.savgol_degree >= settings.savgol_window:
        raise ValueError("savgol_degree must be less than savgol_window")

    smooth_current = savgol_filter(current, settings.savgol_window, settings.savgol_degree)
    polynomial_coeffs = np.polyfit(volts, smooth_current, settings.polynomial_degree)
    polynomial_fit = np.polyval(polynomial_coeffs, volts)

    peak_idxs, _ = find_peaks(polynomial_fit)
    if len(peak_idxs) == 0:
        raise ValueError("No peaks found.")

    prominences, left_bases, right_bases = peak_prominences(polynomial_fit, peak_idxs)
    best = int(np.argmax(prominences))
    peak_idx = int(peak_idxs[best])
    left_base = int(left_bases[best])
    right_base = int(right_bases[best])

    x_background = np.array([volts[left_base], volts[right_base]])
    y_background = np.array([polynomial_fit[left_base], polynomial_fit[right_base]])
    baseline_coeffs = np.polyfit(x_background, y_background, 1)

    peak_background = np.polyval(baseline_coeffs, volts[peak_idx]).tolist()
    peak_signal = polynomial_fit[peak_idx] - peak_background

    adjusted_polynomial = np.asarray(polynomial_coeffs, dtype=float).copy()
    slope, intercept = baseline_coeffs
    adjusted_polynomial[-2] -= slope
    adjusted_polynomial[-1] -= intercept
    peak_profile = np.polyval(adjusted_polynomial, volts)
    background_profile = np.polyval(baseline_coeffs, volts)

    return FitResult(
        method="poly_linear",
        volts=volts,
        current=current,
        peak_signal=float(peak_signal),
        peak_background=float(peak_background),
        peak_voltage=float(volts[peak_idx]),
        peak_index=peak_idx,
        peak_profile=peak_profile,
        background_profile=background_profile,
        params={
            "polynomial_coeffs": polynomial_coeffs,
            "baseline_coeffs": baseline_coeffs,
            "settings": settings,
        },
    )


def fit_signal(volts, current, method: str, settings=None) -> FitResult:
    """Dispatch to one of the two supported fitting methods."""
    fitting_methods = {
        "aswift": aswift_fit,
        "poly_linear": poly_linear_fit,
    }

    try:
        fitting_method = fitting_methods[method]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_FITTING_METHODS)
        raise ValueError(f"Unsupported fitting method '{method}'. Allowed methods: {allowed}.") from exc

    return fitting_method(volts, current, settings=settings)
