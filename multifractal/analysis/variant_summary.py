import numpy as np
from multifractal.analysis.stats import nanrange, weighted_axis0
from multifractal.analysis.window_record import has_enough_tau, make_record

def stack(records, group, key):
    return np.stack([np.asarray(r[group][key], dtype=float) for r in records], axis=0)

def summarize_variant(name, variant, windows, trust, q_array, eps_grid, cfg):
    if not windows:
        return {"valid": False, "reason": "no_fixed_windows", "trust_mask": trust}
    records = [make_record(q_array, eps_grid, variant["p"], variant["raw"], trust, w, cfg)
               for w in windows]
    if not any(has_enough_tau(r) for r in records):
        return {"valid": False, "reason": "fixed_windows_produced_no_finite_tau",
                "trust_mask": trust, "all_windows": records}
    records = records[:cfg.max_windows_to_keep]
    weights = np.ones(len(records)) / float(len(records))
    tau = stack(records, "partition_tau", "tau_q")
    tau_r2 = stack(records, "partition_tau", "tau_q_r2")
    alpha_q = stack(records, "chhabra_jensen", "alpha_q")
    f_q = stack(records, "chhabra_jensen", "f_q")
    return {"valid": True, "aggregated": aggregate(records, weights, tau, tau_r2, alpha_q, f_q),
            "selected_windows": records, "trust_mask": trust}

def aggregate(records, weights, tau, tau_r2, alpha_q, f_q):
    nonlin = np.asarray([r["partition_tau"]["tau_nonlinearity"] for r in records])
    local_curv = np.asarray([r["partition_tau"]["local_tau_curvature"] for r in records])
    alpha_std = np.asarray([r["alpha_distribution"]["alpha_std"] for r in records])
    alpha_span = np.asarray([r["alpha_distribution"]["alpha_span"] for r in records])
    f_width = np.asarray([nanrange(r["chhabra_jensen"]["f_q"]) for r in records])
    return {"window_weights": weights, "tau_q": weighted_axis0(weights, tau),
            "tau_q_r2": weighted_axis0(weights, tau_r2),
            "alpha_q": weighted_axis0(weights, alpha_q),
            "f_q": weighted_axis0(weights, f_q),
            "tau_nonlinearity": weighted_axis0(weights, nonlin)[0],
            "local_tau_curvature": weighted_axis0(weights, local_curv)[0],
            "alpha_std": weighted_axis0(weights, alpha_std)[0],
            "alpha_span": weighted_axis0(weights, alpha_span)[0],
            "f_width": weighted_axis0(weights, f_width)[0]}
