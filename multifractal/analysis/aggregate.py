import numpy as np
from multifractal.analysis.stats import nanmean, nanstd

def aggregate_variant(records):
    valid = [r for r in records if r["valid"]]
    if not valid: return {"valid": False}
    aggs = [r["aggregated"] for r in valid]
    keys = ["tau_q", "tau_q_r2", "alpha_q", "f_q", "tau_nonlinearity",
            "local_tau_curvature", "alpha_std", "alpha_span", "f_width"]
    out = {"valid": True, "num_valid_banks": len(valid)}
    for key in keys:
        vals = np.asarray([a[key] for a in aggs], dtype=float)
        out[key] = np.nanmean(vals, axis=0)
        out[key + "_std"] = np.nanstd(vals, axis=0)
    return out

def aggregate_banks(bank_results):
    names = sorted({name for bank in bank_results for name in bank["variants"]})
    return {name: aggregate_variant([bank["variants"][name] for bank in bank_results
                                     if name in bank["variants"]]) for name in names}

def checkpoint_summary(variants):
    base = variants.get("global_abs_d2", {})
    if not base.get("valid", False): return {}
    return {"tau_nonlinearity": float(np.asarray(base["tau_nonlinearity"])),
            "alpha_std": float(np.asarray(base["alpha_std"])),
            "alpha_span": float(np.asarray(base["alpha_span"])),
            "f_width": float(np.asarray(base["f_width"]))}
