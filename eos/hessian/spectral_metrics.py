import numpy as np
from eos.hessian.eigens import hutchinson_trace, power_extreme

def add_extreme_eigenvalues(cfg, model, batch, criterion, shapes, numels, entry, device):
    if cfg.do_lambda_max:
        eig = power_extreme(model, batch, criterion, shapes, numels,
                            cfg.num_power_iters, cfg.num_power_restarts, True, device)
        entry["lambda_max"] = eig["eig"]
        entry["lambda_max_residual"] = eig["residual"]
        entry["eta_lambda_max"] = entry["lr"] * eig["eig"] if np.isfinite(entry["lr"]) else np.nan
    if cfg.do_lambda_min:
        eig = power_extreme(model, batch, criterion, shapes, numels,
                            cfg.num_power_iters, cfg.num_power_restarts, False, device)
        entry["lambda_min"] = eig["eig"]
        entry["lambda_min_residual"] = eig["residual"]

def add_trace(cfg, model, batch, criterion, shapes, numels, entry, device):
    if not cfg.do_trace:
        return
    vals = hutchinson_trace(model, batch, criterion, shapes, numels,
                            cfg.num_trace_samples, device)
    entry["trace_H"] = float(np.mean(vals))
    entry["trace_H_std"] = float(np.std(vals))
    entry["trace_H_per_dim"] = float(entry["trace_H"] / sum(numels))
