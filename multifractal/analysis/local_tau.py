import numpy as np
from multifractal.analysis.stats import nanmean

def local_tau_map(z_q_eps, eps):
    qn, en = z_q_eps.shape
    out = np.full((qn, max(0, en - 1)), np.nan)
    if en < 2:
        return out, np.asarray([], dtype=float), np.nan
    eps_mid = np.sqrt(eps[:-1] * eps[1:])
    for qi in range(qn):
        for ei in range(en - 1):
            z1, z2 = z_q_eps[qi, ei], z_q_eps[qi, ei + 1]
            if not (np.isfinite(z1) and np.isfinite(z2) and z1 > 0 and z2 > 0):
                continue
            den = np.log(eps[ei + 1]) - np.log(eps[ei])
            if abs(den) > 1e-30: out[qi, ei] = (np.log(z2) - np.log(z1)) / den
    return out, eps_mid, curvature(out)

def curvature(local_tau):
    vals = []
    if local_tau.shape[1] < 3: return np.nan
    for row in local_tau:
        for j in range(1, len(row) - 1):
            triple = [row[j - 1], row[j], row[j + 1]]
            if np.all(np.isfinite(triple)):
                vals.append(abs(row[j + 1] - 2 * row[j] + row[j - 1]))
    return nanmean(vals)
