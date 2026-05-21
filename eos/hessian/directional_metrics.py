import numpy as np
from common.torch_utils import safe_normalize
from eos.hessian.linalg import directional_curvature, flat_parameters

def add_directional_curvatures(cfg, model, batch, criterion, shapes, numels,
                               entry, gflat, previous_theta):
    theta = flat_parameters(model).detach()
    if not cfg.do_directional_curvatures:
        return theta
    grad_dir = safe_normalize(gflat)
    update = theta - previous_theta if previous_theta is not None else gflat
    update_dir = safe_normalize(update)
    entry["update_norm"] = float(update.norm().item()) if previous_theta is not None else np.nan
    entry["gHg_over_gg"] = directional_curvature(model, batch, criterion, grad_dir, shapes, numels)
    entry["uHu_over_uu"] = directional_curvature(model, batch, criterion, update_dir, shapes, numels)
    return theta
