import numpy as np

def span_decades(eps, window) -> float:
    start, end = window
    e = np.asarray(eps[start:end])
    return 0.0 if len(e) < 2 or np.any(e <= 0) else float(np.log10(e.max()) - np.log10(e.min()))

def analysis_windows(eps, trust_mask, cfg):
    out = []
    for start, end in cfg.fixed_windows:
        if start < 0 or end > len(eps) or end <= start:
            raise ValueError(f"Invalid fixed window {(start, end)}")
        if cfg.fixed_window_require_trust and not np.all(trust_mask[start:end]):
            continue
        out.append((int(start), int(end)))
    return out
