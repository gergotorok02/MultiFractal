import numpy as np

def normalize(raw, p_floor: float):
    raw = np.maximum(np.where(np.isfinite(raw), raw, 0.0), 0.0)
    row_sum = np.sum(raw, axis=1)
    valid = np.isfinite(row_sum) & (row_sum > p_floor)
    p = np.full_like(raw, np.nan, dtype=float)
    if np.any(valid):
        p[valid] = raw[valid] / row_sum[valid, None]
    return p, valid

def build_variants(abs_d2, abs_d1, cfg):
    variants = {}
    for gamma in cfg.gamma_variants:
        raw = np.maximum(abs_d2, cfg.eps_floor)
        name = "global_abs_d2"
        if abs(gamma - 1.0) >= 1e-12:
            raw, name = raw ** float(gamma), f"gamma{gamma:g}_abs_d2"
        p, valid = normalize(raw, cfg.p_floor)
        variants[name] = {"raw": raw, "p": p, "valid_rows": valid}
    if cfg.include_abs_d1_variant:
        raw = np.maximum(abs_d1, cfg.eps_floor)
        p, valid = normalize(raw, cfg.p_floor)
        variants["global_abs_d1"] = {"raw": raw, "p": p, "valid_rows": valid}
    return variants

def trustworthy(p, raw, cfg):
    mask = np.zeros(raw.shape[0], dtype=bool)
    for i in range(raw.shape[0]):
        if not np.any(np.isfinite(p[i])): continue
        if float(np.mean(raw[i] > cfg.eps_floor)) < cfg.min_positive_fraction: continue
        mu, sd = np.nanmean(raw[i]), np.nanstd(raw[i])
        mask[i] = np.isfinite(mu) and np.isfinite(sd) and mu / max(sd, cfg.eps_floor) >= cfg.snr_threshold
    return mask
