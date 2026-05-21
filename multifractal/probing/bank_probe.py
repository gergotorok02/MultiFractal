import numpy as np
from common.torch_utils import cross_entropy_on_batch
from multifractal.probing.directions import apply_, sample_bank
from multifractal.analysis.measures import build_variants, trustworthy
from multifractal.probing.state import snapshot
from multifractal.analysis.variant_summary import summarize_variant
from multifractal.analysis.windows import analysis_windows

def directional_responses(model, batch, eps_grid, directions):
    base_loss = cross_entropy_on_batch(model, batch)
    d2 = np.zeros((len(eps_grid), len(directions)))
    d1 = np.zeros_like(d2)
    for di, direction in enumerate(directions):
        if di % 12 == 0: print(f"      direction {di+1}/{len(directions)}", flush=True)
        for ei, eps in enumerate(eps_grid):
            apply_(model, direction, +float(eps)); lp = cross_entropy_on_batch(model, batch)
            apply_(model, direction, -2.0 * float(eps)); lm = cross_entropy_on_batch(model, batch)
            apply_(model, direction, +float(eps))
            d2[ei, di] = abs(lp + lm - 2.0 * base_loss)
            d1[ei, di] = abs(0.5 * (lp - lm))
    return d2, d1

def probe_single_bank(model, batch, eps_grid, q_array, cfg, bank_index: int):
    snapshot(model)
    directions = sample_bank(model, cfg, bank_index)
    print(f"    [bank {bank_index}] start | directions={len(directions)}, scales={len(eps_grid)}")
    d2, d1 = directional_responses(model, batch, eps_grid, directions)
    out = {}
    for name, variant in build_variants(d2, d1, cfg).items():
        trust = trustworthy(variant["p"], variant["raw"], cfg)
        windows = analysis_windows(eps_grid, trust, cfg)
        out[name] = summarize_variant(name, variant, windows, trust, q_array, eps_grid, cfg)
    return {"bank_index": bank_index, "base_raw": {"abs_d2": d2, "abs_d1": d1}, "variants": out}
