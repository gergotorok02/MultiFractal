from __future__ import annotations

from typing import List, Tuple

import numpy as np

from plotting.utils import quartile_summary, safe_arr
from plotting.analysis.extract import collect_checkpoint_distributions


def tau_curve_branch_color(q: np.ndarray, curve: np.ndarray) -> str:
    """
    For tau(q):
      - upper branch on q < 0 -> blue
      - lower branch on q < 0 -> red

    This branch assignment is also reused in alpha(q) and f(alpha)
    so corresponding curves keep the same color across columns.
    """
    q = np.asarray(q, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)

    TAU_UPPER = "#1f77b4"   # blue
    TAU_LOWER = "#d62728"   # red

    mask = np.isfinite(q) & np.isfinite(curve)
    if mask.sum() < 3:
        return TAU_LOWER

    neg_mask = mask & (q < 0)
    if neg_mask.sum() >= 2:
        level = float(np.nanmedian(curve[neg_mask]))
    else:
        level = float(np.nanmedian(curve[mask]))

    return TAU_UPPER if level > 0.0 else TAU_LOWER


def tau_curve_color_list(q: np.ndarray, tau: np.ndarray) -> List[str]:
    """
    Assign one persistent color per curve based on its tau(q) branch.
    """
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if tau.size == 0:
        return []

    return [tau_curve_branch_color(q, curve) for curve in tau]


def tau_curvature_sign_changes(q: np.ndarray, tau_curve: np.ndarray) -> float:
    """
    Count sign changes in a discrete second-derivative estimate of tau(q).

    This is only a diagnostic for finite-scale estimated tau curves.
    It should not be interpreted as a strict theorem-level convexity test.
    """
    q = np.asarray(q, dtype=np.float64)
    tau_curve = np.asarray(tau_curve, dtype=np.float64)

    mask = np.isfinite(q) & np.isfinite(tau_curve)
    q = q[mask]
    tau_curve = tau_curve[mask]

    if q.size < 5:
        return np.nan

    order = np.argsort(q)
    q = q[order]
    tau_curve = tau_curve[order]

    dq = np.diff(q)
    if np.sum(np.isfinite(dq) & (np.abs(dq) > 1e-12)) < 4:
        return np.nan

    slopes = np.diff(tau_curve) / (np.diff(q) + 1e-12)
    denom = 0.5 * (q[2:] - q[:-2]) + 1e-12
    curv = np.diff(slopes) / denom

    curv = curv[np.isfinite(curv)]
    if curv.size < 2:
        return np.nan

    # Ignore tiny curvature values to avoid counting numerical flicker.
    scale = np.nanmedian(np.abs(curv)) + 1e-12
    curv[np.abs(curv) < 0.05 * scale] = 0.0

    signs = np.sign(curv)
    signs = signs[signs != 0]

    if signs.size < 2:
        return 0.0

    return float(np.sum(signs[1:] != signs[:-1]))


def summarize_tau_curvature_sign_changes(q: np.ndarray, tau: np.ndarray, label: str) -> None:
    """
    Print curvature sign-change statistics for a collection of tau curves.
    """
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if q.size == 0 or tau.size == 0:
        print(f"[fig03 curvature] {label}: no tau curves available")
        return

    scores = []
    for curve in tau:
        score = tau_curvature_sign_changes(q, curve)
        if np.isfinite(score):
            scores.append(score)

    if len(scores) == 0:
        print(f"[fig03 curvature] {label}: no finite curvature sign-change scores")
        return

    scores = np.asarray(scores, dtype=np.float64)

    print(
        f"[fig03 curvature] {label}: "
        f"n={scores.size}, "
        f"median={np.nanmedian(scores):.2f}, "
        f"mean={np.nanmean(scores):.2f}, "
        f"q25={np.nanquantile(scores, 0.25):.2f}, "
        f"q75={np.nanquantile(scores, 0.75):.2f}, "
        f"max={np.nanmax(scores):.0f}, "
        f"frac_nonzero={np.mean(scores > 0):.2f}"
    )


def _finite_curve_xy(a_curve: np.ndarray, f_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return finite f(alpha) curve points sorted by alpha.

    The f(alpha) curves are originally parameterized by q.  For plotting and
    filling a swept region between two whole curves, we sort each curve by its
    alpha coordinate and drop non-finite points.
    """
    a_curve = np.asarray(a_curve, dtype=np.float64)
    f_curve = np.asarray(f_curve, dtype=np.float64)
    mask = np.isfinite(a_curve) & np.isfinite(f_curve)
    x = a_curve[mask]
    y = f_curve[mask]
    if x.size < 2:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def falpha_alpha_sweep_bounds(alpha: np.ndarray, f: np.ndarray):
    """
    Build the grey sweep for the f(alpha) panel.

    Important: this is intentionally NOT the alpha-grid interpolation band.
    The user-facing plot is a family of parabola-like f(alpha) curves.  We first
    compute the usual median curve from pointwise medians over q:

        med_alpha(q), med_f(q).

    Then we select two *whole curves* from the family: the curve that lies
    farthest to the left of med_alpha(q) on the alpha-axis and the curve that
    lies farthest to the right.  The grey region is the swept polygon between
    these two selected curves.  This gives a visual sweep of a parabola-family,
    instead of a pointwise vertical band on an artificial alpha grid.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    if alpha.size == 0 or f.size == 0 or alpha.shape != f.shape:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty

    med_a, _, _ = quartile_summary(alpha)
    med_f, _, _ = quartile_summary(f)

    offsets = []
    indices = []
    for i, (a_curve, f_curve) in enumerate(zip(alpha, f)):
        mask = np.isfinite(a_curve) & np.isfinite(f_curve) & np.isfinite(med_a)
        if np.sum(mask) < 3:
            continue
        # Robust horizontal displacement from the median alpha curve.
        offsets.append(float(np.nanmedian(a_curve[mask] - med_a[mask])))
        indices.append(i)

    if len(indices) < 2:
        empty = np.asarray([], dtype=np.float64)
        return med_a, med_f, empty, empty, empty, empty

    offsets = np.asarray(offsets, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)

    left_idx = int(indices[np.nanargmin(offsets)])
    right_idx = int(indices[np.nanargmax(offsets)])

    left_a, left_f = _finite_curve_xy(alpha[left_idx], f[left_idx])
    right_a, right_f = _finite_curve_xy(alpha[right_idx], f[right_idx])

    return med_a, med_f, left_a, left_f, right_a, right_f


def falpha_bootstrap_confidence_band(
    alpha: np.ndarray,
    f: np.ndarray,
    n_boot: int = 1200,
    ci: float = 0.95,
    grid_size: int = 260,
    min_coverage: float = 0.80,
    seed: int = 12345,
):
    """
    Bootstrap confidence band for the median f(alpha) profile.

    The raw curves are naturally parameterized by q, but the panel is shown in
    the transformed alpha-coordinate.  A vertical pointwise band over raw
    alpha-values would mix different q-levels and can produce misleading filled
    regions.  Here each bootstrap sample first forms a median parametric curve
    (median alpha(q), median f(q)), sorts it by alpha, interpolates it onto a
    common alpha-grid, and only then takes percentile intervals.

    Returns
    -------
    x_grid, median_on_grid, lower, upper
        Arrays containing the common alpha-grid, the original median profile on
        that grid, and the percentile bootstrap confidence interval.  Grid
        locations with insufficient bootstrap support are removed.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    if alpha.size == 0 or f.size == 0 or alpha.shape != f.shape or alpha.ndim != 2:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty

    # Keep only curves with enough finite parametric points.
    good_rows = []
    for i in range(alpha.shape[0]):
        if np.sum(np.isfinite(alpha[i]) & np.isfinite(f[i])) >= 4:
            good_rows.append(i)
    if len(good_rows) < 3:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty

    alpha = alpha[good_rows]
    f = f[good_rows]
    n = alpha.shape[0]

    med_a = np.nanmedian(alpha, axis=0)
    med_f = np.nanmedian(f, axis=0)
    finite_med = np.isfinite(med_a) & np.isfinite(med_f)
    if np.sum(finite_med) < 4:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty

    # Use the robust central alpha-range of the observed curves as the display
    # grid.  The final coverage filter below removes unsupported tails.
    avals = alpha[np.isfinite(alpha)]
    if avals.size < 4:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    lo = float(np.nanquantile(avals, 0.03))
    hi = float(np.nanquantile(avals, 0.97))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty

    x_grid = np.linspace(lo, hi, grid_size)

    def interp_parametric(a_curve: np.ndarray, f_curve: np.ndarray) -> np.ndarray:
        mask = np.isfinite(a_curve) & np.isfinite(f_curve)
        if np.sum(mask) < 4:
            return np.full_like(x_grid, np.nan, dtype=np.float64)
        a = a_curve[mask]
        y = f_curve[mask]
        order = np.argsort(a)
        a = a[order]
        y = y[order]

        # Collapse duplicate / nearly duplicate alpha locations by averaging f.
        uniq_a = []
        uniq_y = []
        for val in np.unique(a):
            same = a == val
            if np.any(same):
                uniq_a.append(float(val))
                uniq_y.append(float(np.nanmean(y[same])))
        a = np.asarray(uniq_a, dtype=np.float64)
        y = np.asarray(uniq_y, dtype=np.float64)
        if a.size < 4 or np.nanmax(a) <= np.nanmin(a):
            return np.full_like(x_grid, np.nan, dtype=np.float64)
        return np.interp(x_grid, a, y, left=np.nan, right=np.nan)

    med_grid = interp_parametric(med_a, med_f)

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, grid_size), np.nan, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.integers(0, n, size=n)
        ba = np.nanmedian(alpha[sample], axis=0)
        bf = np.nanmedian(f[sample], axis=0)
        boot[b] = interp_parametric(ba, bf)

    alpha_level = 1.0 - float(ci)
    lower = np.nanquantile(boot, 0.5 * alpha_level, axis=0)
    upper = np.nanquantile(boot, 1.0 - 0.5 * alpha_level, axis=0)
    coverage = np.mean(np.isfinite(boot), axis=0)

    keep = (
        np.isfinite(x_grid)
        & np.isfinite(med_grid)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (coverage >= min_coverage)
    )

    return x_grid[keep], med_grid[keep], lower[keep], upper[keep]


def _tau_zoom_ylim(q: np.ndarray, tau: np.ndarray, qmin: float = 0.0, qmax: float = 2.0):
    q = np.asarray(q, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if q.size == 0 or tau.size == 0:
        return None

    mask = np.isfinite(q) & (q >= qmin) & (q <= qmax)
    if mask.sum() < 2:
        return None

    vals = tau[:, mask].reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lo = float(np.nanquantile(vals, 0.02))
    hi = float(np.nanquantile(vals, 0.98))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))

    span = max(1e-12, hi - lo)
    return lo - 0.12 * span, hi + 0.12 * span


def _trapz_branch_mean_abs(q: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean absolute spectral displacement over a q-branch.

    Uses a trapezoidal integral divided by branch length, so the negative and
    positive branches are comparable even if they contain different numbers
    of sampled q-values.
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(q) & np.isfinite(y)

    if mask.sum() < 2:
        return np.nan

    x = q[mask]
    v = np.abs(y[mask])

    order = np.argsort(x)
    x = x[order]
    v = v[order]

    span = float(np.nanmax(x) - np.nanmin(x))
    if not np.isfinite(span) or span <= 1e-12:
        return np.nan

    return float(np.trapz(v, x) / span)


def _finite_summary(vals: np.ndarray):
    """
    Median and interquartile summary for a 1D array.
    """
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.nanmedian(vals)),
        float(np.nanquantile(vals, 0.25)),
        float(np.nanquantile(vals, 0.75)),
    )


def collect_tau_spectrum_time_series(entries, family_name, variant_name):
    """
    Collect median tau(q,t) curves and the retained tau(q) curve distribution
    at each checkpoint.

    Returns:
      q_ref:
        common q-grid

      tau_med, tau_q25, tau_q75:
        arrays of shape [num_checkpoints, num_q]

      tau_curves_by_t:
        list where each element is a matrix [num_curves, num_q]
    """
    q_ref = None
    tau_curves_by_t = []

    for e in entries:
        d = collect_checkpoint_distributions(e, family_name, variant_name)

        q = np.asarray(d["q"], dtype=np.float64)

        # Prefer window-level curves, because they preserve the distribution.
        curves = d["window_tau_vectors"]
        if len(curves) == 0:
            curves = d["tau_vectors"]

        if q_ref is None and q.size > 0:
            q_ref = q.copy()

        if q_ref is None or q_ref.size == 0:
            tau_curves_by_t.append(np.empty((0, 0), dtype=np.float64))
            continue

        valid_curves = []
        for curve in curves:
            curve = np.asarray(curve, dtype=np.float64)
            if curve.size == q_ref.size:
                valid_curves.append(curve)

        if len(valid_curves) == 0:
            tau_curves_by_t.append(np.empty((0, q_ref.size), dtype=np.float64))
        else:
            tau_curves_by_t.append(np.stack(valid_curves, axis=0))

    if q_ref is None:
        q_ref = np.asarray([], dtype=np.float64)

    tau_med = []
    tau_q25 = []
    tau_q75 = []

    for mat in tau_curves_by_t:
        if mat.size == 0 or mat.shape[0] == 0:
            tau_med.append(np.full_like(q_ref, np.nan, dtype=np.float64))
            tau_q25.append(np.full_like(q_ref, np.nan, dtype=np.float64))
            tau_q75.append(np.full_like(q_ref, np.nan, dtype=np.float64))
        else:
            med, q25, q75 = quartile_summary(mat)
            tau_med.append(med)
            tau_q25.append(q25)
            tau_q75.append(q75)

    tau_med = np.stack(tau_med, axis=0) if len(tau_med) else np.empty((0, 0))
    tau_q25 = np.stack(tau_q25, axis=0) if len(tau_q25) else np.empty((0, 0))
    tau_q75 = np.stack(tau_q75, axis=0) if len(tau_q75) else np.empty((0, 0))

    return q_ref, tau_med, tau_q25, tau_q75, tau_curves_by_t


def extract_signed_q_spectral_evolution(entries, family_name, variant_name):
    """
    Extract signed-q localization of spectral evolution.

    Reference spectrum:
      The first checkpoint with a finite median tau(q) curve.

    For each later checkpoint and each retained tau(q) curve, compute:
      delta_tau(q,t) = tau(q,t) - tau_ref(q)

      C_-(t) = mean absolute displacement over q < 0
      C_+(t) = mean absolute displacement over q > 0

      rho_-(t) = C_-(t) / (C_-(t) + C_+(t) + eps)

    This measures where the temporal change of the spectrum is concentrated.
    """
    q, tau_med, tau_q25, tau_q75, tau_curves_by_t = collect_tau_spectrum_time_series(
        entries,
        family_name,
        variant_name,
    )

    n = tau_med.shape[0] if tau_med.ndim == 2 else 0

    out = {
        "q": q,
        "tau_med": tau_med,
        "tau_q25": tau_q25,
        "tau_q75": tau_q75,
        "delta_tau_med": np.full_like(tau_med, np.nan, dtype=np.float64),
        "ref_index": 0,
        "cneg_median": np.full(n, np.nan),
        "cneg_q25": np.full(n, np.nan),
        "cneg_q75": np.full(n, np.nan),
        "cpos_median": np.full(n, np.nan),
        "cpos_q25": np.full(n, np.nan),
        "cpos_q75": np.full(n, np.nan),
        "rho_neg_median": np.full(n, np.nan),
        "rho_neg_q25": np.full(n, np.nan),
        "rho_neg_q75": np.full(n, np.nan),
    }

    if q.size == 0 or tau_med.size == 0:
        return out

    finite_rows = np.where(np.isfinite(tau_med).sum(axis=1) >= max(3, q.size // 2))[0]
    if finite_rows.size == 0:
        return out

    ref_index = int(finite_rows[0])
    tau_ref = tau_med[ref_index].copy()

    out["ref_index"] = ref_index
    out["delta_tau_med"] = tau_med - tau_ref[None, :]

    neg_mask = q < 0
    pos_mask = q > 0

    eps = 1e-12

    for i, mat in enumerate(tau_curves_by_t):
        if mat.size == 0 or mat.shape[0] == 0:
            continue

        cneg_vals = []
        cpos_vals = []
        rho_vals = []

        for curve in mat:
            if curve.size != q.size:
                continue

            delta = curve - tau_ref

            cneg = _trapz_branch_mean_abs(q, delta, neg_mask)
            cpos = _trapz_branch_mean_abs(q, delta, pos_mask)

            if np.isfinite(cneg):
                cneg_vals.append(cneg)
            if np.isfinite(cpos):
                cpos_vals.append(cpos)

            if np.isfinite(cneg) and np.isfinite(cpos) and (cneg + cpos) > eps:
                rho_vals.append(cneg / (cneg + cpos + eps))

        m, q25, q75 = _finite_summary(cneg_vals)
        out["cneg_median"][i] = m
        out["cneg_q25"][i] = q25
        out["cneg_q75"][i] = q75

        m, q25, q75 = _finite_summary(cpos_vals)
        out["cpos_median"][i] = m
        out["cpos_q25"][i] = q25
        out["cpos_q75"][i] = q75

        m, q25, q75 = _finite_summary(rho_vals)
        out["rho_neg_median"][i] = m
        out["rho_neg_q25"][i] = q25
        out["rho_neg_q75"][i] = q75

    return out
