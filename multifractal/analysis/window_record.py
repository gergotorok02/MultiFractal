import numpy as np
from multifractal.analysis.fit import per_direction_alpha, tau_nonlinearity
from multifractal.analysis.local_tau import local_tau_map
from multifractal.analysis.partition import chhabra_jensen, partition_tau
from multifractal.analysis.stats import nanmean, nanrange, nanstd
from multifractal.analysis.windows import span_decades

def score(tau_r2, tau_nonlin, local_curv):
    r2m = nanmean(tau_r2)
    if not np.isfinite(r2m): return -1e9
    return r2m + 0.10 * finite_or_zero(tau_nonlin) + 0.10 * finite_or_zero(local_curv)

def finite_or_zero(x):
    return 0.0 if not np.isfinite(x) else float(x)

def make_record(q, eps, p, raw, trust, window, cfg):
    tau_info = partition_tau(q, eps, p, window, cfg.p_floor)
    cj = chhabra_jensen(q, eps, p, window, cfg.p_floor)
    alpha_dir, alpha_r2 = per_direction_alpha(eps, raw, window, cfg.eps_floor)
    local_tau, local_mid, local_curv = local_tau_map(tau_info["Z"], eps)
    nonlin = tau_nonlinearity(q, tau_info["tau"])
    return {
        "window": window, "fixed_window_mode": True,
        "window_span_decades": span_decades(eps, window),
        "window_trust_fraction": float(np.mean(trust[window[0]:window[1]])),
        "score": score(tau_info["tau_r2"], nonlin, local_curv),
        "partition_tau": {"tau_q": tau_info["tau"], "tau_q_r2": tau_info["tau_r2"],
                          "tau_nonlinearity": nonlin, "Z_q_eps": tau_info["Z"],
                          "local_tau_q_eps": local_tau, "local_tau_eps_mid": local_mid,
                          "local_tau_q_eps_mean_abs": nanmean(np.abs(local_tau)),
                          "local_tau_curvature": local_curv},
        "chhabra_jensen": cj,
        "alpha_distribution": {"alpha_dir": alpha_dir, "alpha_dir_r2": alpha_r2,
                               "alpha_std": nanstd(alpha_dir), "alpha_span": nanrange(alpha_dir)},
    }

def has_enough_tau(record, min_finite: int = 3) -> bool:
    tau = np.asarray(record["partition_tau"]["tau_q"], dtype=float)
    return int(np.isfinite(tau).sum()) >= min_finite
