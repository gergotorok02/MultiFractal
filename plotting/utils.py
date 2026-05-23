from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def unpack_obj(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    return x


def as_list_of_dicts(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [unpack_obj(v) for v in x.tolist()]
    if isinstance(x, dict):
        return [x]
    return []


def safe_arr(x: Any, dtype=np.float64) -> np.ndarray:
    try:
        return np.asarray(x, dtype=dtype)
    except Exception:
        return np.asarray([], dtype=dtype)


def maybe_scalar(x: Any, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if np.isscalar(x):
            return float(x)
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return default
    except Exception:
        return default


def nan_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def linear_fit_with_band(x: np.ndarray, y: np.ndarray, n_grid: int = 200):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return None

    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = beta
    yhat = slope * x + intercept
    resid = y - yhat
    dof = max(1, x.size - 2)
    s2 = float(np.sum(resid ** 2) / dof)

    xg = np.linspace(np.min(x), np.max(x), n_grid)
    xbar = np.mean(x)
    sxx = np.sum((x - xbar) ** 2) + 1e-18
    yg = slope * xg + intercept
    se_mean = np.sqrt(s2 * (1.0 / x.size + (xg - xbar) ** 2 / sxx))
    lo = yg - 1.96 * se_mean
    hi = yg + 1.96 * se_mean
    return xg, yg, lo, hi, float(slope), float(intercept)


def robust_limits(arrays: Sequence[np.ndarray], pad_frac: float = 0.06) -> Tuple[float, float]:
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays if np.asarray(a).size > 0]
    if len(arrays) == 0:
        return -1.0, 1.0
    vals = np.concatenate(arrays)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo = float(np.quantile(vals, 0.02))
    hi = float(np.quantile(vals, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    span = max(1e-12, hi - lo)
    return lo - pad_frac * span, hi + pad_frac * span


def quantile_summary(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float64)
    return (
        np.nanmedian(mat, axis=0),
        np.nanquantile(mat, 0.25, axis=0),
        np.nanquantile(mat, 0.75, axis=0),
    )


def quartile_summary(mat: np.ndarray):
    """
    Median and 25%-75% pointwise envelope.

    This is a distributional interquartile band over retained windows/batches/banks,
    not a formal statistical confidence interval.
    """
    mat = np.asarray(mat, dtype=np.float64)
    return (
        np.nanmedian(mat, axis=0),
        np.nanquantile(mat, 0.25, axis=0),
        np.nanquantile(mat, 0.75, axis=0),
    )


def choose_index(n: int, frac: float) -> int:
    if n <= 1:
        return 0
    frac = min(max(float(frac), 0.0), 1.0)
    return int(round(frac * (n - 1)))


def moving_average_nan(x: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if k <= 1 or x.size == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    h = k // 2
    for i in range(x.size):
        sl = x[max(0, i - h): min(x.size, i + h + 1)]
        if np.isfinite(sl).any():
            out[i] = np.nanmean(sl)
    return out
