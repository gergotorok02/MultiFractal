import numpy as np

def nanmean(x, default=np.nan):
    x = np.asarray(x, dtype=float); m = np.isfinite(x)
    return float(np.mean(x[m])) if np.any(m) else float(default)

def nanstd(x, default=np.nan):
    x = np.asarray(x, dtype=float); m = np.isfinite(x)
    return float(np.std(x[m])) if np.any(m) else float(default)

def nanrange(x, default=np.nan):
    x = np.asarray(x, dtype=float); m = np.isfinite(x)
    return float(np.max(x[m]) - np.min(x[m])) if np.any(m) else float(default)

def weighted_nanmean(weights, values, default=np.nan):
    w, v = np.asarray(weights, float), np.asarray(values, float)
    m = np.isfinite(w) & np.isfinite(v)
    if not np.any(m): return float(default)
    sw = float(np.sum(w[m]))
    return float(np.sum(w[m] * v[m]) / sw) if sw > 0 and np.isfinite(sw) else float(default)

def weighted_axis0(weights, values):
    values = np.asarray(values, float)
    flat = values.reshape(values.shape[0], -1)
    out = np.array([weighted_nanmean(weights, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape(values.shape[1:])
