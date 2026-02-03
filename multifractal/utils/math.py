import numpy as np
import torch

def safe_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm() + eps)

def fit_alpha_loglog(eps, y) -> float:
    eps = np.asarray(eps, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = (eps > 0) & (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    lx = np.log(eps[mask])
    ly = np.log(y[mask])
    A = np.vstack([lx, np.ones_like(lx)]).T
    alpha, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(alpha)

def alpha_local_adjacent(eps, y):
    eps = np.asarray(eps, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.full(len(eps) - 1, np.nan, dtype=np.float64)
    for k in range(len(eps) - 1):
        if eps[k] > 0 and eps[k+1] > 0 and y[k] > 0 and y[k+1] > 0:
            out[k] = (np.log(y[k+1]) - np.log(y[k])) / (np.log(eps[k+1]) - np.log(eps[k]))
    return out

def collapse_score_quantiles(eps, deltas, alpha_hat, qs=(0.1,0.25,0.5,0.75,0.9)):
    eps = np.asarray(eps, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)   # [K,D]
    scaled = deltas / (eps[:, None] ** float(alpha_hat) + 1e-12)
    qmat = np.quantile(scaled, qs, axis=1)          # [Q,K]
    std_per_q = np.std(qmat, axis=1)                # [Q]
    score = float(np.mean(std_per_q))
    return score, std_per_q.astype(np.float32), qmat.astype(np.float32)
