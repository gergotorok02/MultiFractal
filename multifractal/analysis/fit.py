import numpy as np

def line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    x, y = x[mask], y[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-18)
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(1.0 - ss_res / ss_tot)}

def tau_nonlinearity(q, tau) -> float:
    q, tau = np.asarray(q, float), np.asarray(tau, float)
    mask = np.isfinite(q) & np.isfinite(tau)
    if mask.sum() < 3: return np.nan
    fit = line(q[mask], tau[mask])
    pred = fit["slope"] * q[mask] + fit["intercept"]
    return float(np.sqrt(np.mean((tau[mask] - pred) ** 2)))

def per_direction_alpha(eps, raw, window, floor: float):
    start, end = window
    x = np.log(eps[start:end])
    yraw = np.maximum(raw[start:end], floor)
    alpha, r2 = np.full(yraw.shape[1], np.nan), np.full(yraw.shape[1], np.nan)
    for di in range(yraw.shape[1]):
        fit = line(x, np.log(yraw[:, di]))
        alpha[di], r2[di] = fit["slope"], fit["r2"]
    return alpha, r2
