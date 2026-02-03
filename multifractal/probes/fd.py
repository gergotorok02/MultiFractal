import numpy as np
import torch

from multifractal.utils.math import safe_normalize
from multifractal.utils.params import get_param_list, get_flat_param_vector, set_flat_param_vector
from multifractal.probes.curves import loss_on_batch, loss_for_vector

def sample_random_direction_raw(base_vec):
    v = torch.randn_like(base_vec)
    return safe_normalize(v)

def sample_random_direction_filter_norm(model):
    params = get_param_list(model)
    chunks = []
    with torch.no_grad():
        for p in params:
            r = torch.randn_like(p)
            scale = (p.data.norm() / (r.norm() + 1e-12))
            chunks.append((r * scale).reshape(-1))
        v = torch.cat(chunks)
    return safe_normalize(v)

def fd_probe_rich(
    model, batch, criterion,
    epsilons, num_directions,
    shapes, numels,
    direction_mode="filter_norm",
):
    base_vec = get_flat_param_vector(model).detach()
    base_loss = loss_on_batch(model, batch, criterion)

    K = len(epsilons)
    D = int(num_directions)

    deltas = np.zeros((K, D), dtype=np.float64)
    fd_mean = np.zeros(K, dtype=np.float64)
    fd_var  = np.zeros(K, dtype=np.float64)

    for k, eps in enumerate(epsilons):
        losses = np.zeros(D, dtype=np.float64)
        for i in range(D):
            if direction_mode == "filter_norm":
                v = sample_random_direction_filter_norm(model)
            else:
                v = sample_random_direction_raw(base_vec)
            new_vec = base_vec + float(eps) * v
            l = loss_for_vector(model, new_vec, batch, criterion, shapes, numels)
            losses[i] = l
            deltas[k, i] = l - base_loss

        fd_mean[k] = float(np.mean(np.abs(losses - base_loss)))
        fd_var[k]  = float(np.var(losses))

    return float(base_loss), fd_mean, fd_var, deltas
