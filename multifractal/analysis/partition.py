import numpy as np
from multifractal.analysis.fit import line
from multifractal.analysis.stats import nanmean

def partition_tau(q_list, eps, p, window, p_floor: float):
    start, end = window
    p_safe = np.where(np.isfinite(p), np.maximum(p, p_floor), np.nan)
    z = np.full((len(q_list), len(eps)), np.nan)
    tau = np.full(len(q_list), np.nan); r2 = np.full(len(q_list), np.nan)
    for qi, q in enumerate(q_list):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z[qi] = np.nansum(np.power(p_safe, q), axis=1)
        valid = np.isfinite(z[qi, start:end]) & (z[qi, start:end] > 0)
        if valid.sum() >= 2:
            fit = line(np.log(eps[start:end][valid]), np.log(z[qi, start:end][valid]))
            tau[qi], r2[qi] = fit["slope"], fit["r2"]
    return {"Z": z, "tau": tau, "tau_r2": r2}

def chhabra_jensen(q_list, eps, p, window, p_floor: float):
    start, end = window
    p_safe = np.where(np.isfinite(p), np.maximum(p, p_floor), np.nan)
    alpha_q, f_q = np.full(len(q_list), np.nan), np.full(len(q_list), np.nan)
    log_eps = np.log(np.asarray(eps, float))
    for qi, q in enumerate(q_list):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pq = np.power(p_safe, q)
            denom = np.nansum(pq, axis=1, keepdims=True)
            mu = pq / np.where(denom > 0, denom, np.nan)
            alpha = np.nansum(mu * np.log(p_safe), axis=1) / (log_eps + 1e-30)
            ff = np.nansum(mu * np.log(np.maximum(mu, p_floor)), axis=1) / (log_eps + 1e-30)
        alpha_q[qi], f_q[qi] = nanmean(alpha[start:end]), nanmean(ff[start:end])
    return {"alpha_q": alpha_q, "f_q": f_q}
