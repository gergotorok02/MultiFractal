import numpy as np
import torch

from multifractal.utils.math import safe_normalize
from multifractal.utils.params import (
    get_param_list, flatten_params_from_list,
    unflatten_vector_to_list
)

def hessian_vector_product(model, batch, criterion, v_list, device: torch.device):
    x, y = batch
    params = get_param_list(model)
    logits = model(x)
    loss = criterion(logits, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    g_dot_v = torch.zeros(1, device=device)
    for g, v in zip(grads, v_list):
        g_dot_v = g_dot_v + (g * v).sum()

    hvps = torch.autograd.grad(g_dot_v, params, retain_graph=False)
    return hvps

def estimate_hessian_metrics(
    model, batch, criterion,
    shapes, numels,
    device: torch.device,
    num_power_iters=5,
    num_trace_samples=2,
):
    total_dim = sum(numels)
    v = safe_normalize(torch.randn(total_dim, device=device))
    lam = torch.tensor(0.0, device=device)
    v_top = v.clone()

    for _ in range(int(num_power_iters)):
        v_list = unflatten_vector_to_list(v, shapes, numels)
        hvp_list = hessian_vector_product(model, batch, criterion, v_list, device=device)
        hv = flatten_params_from_list(hvp_list).detach()
        lam = (v * hv).sum() / (v.norm() ** 2 + 1e-12)
        v = safe_normalize(hv)
        v_top = v.clone()

    lambda_max = float(lam.detach().cpu().item())

    trace_vals = []
    for _ in range(int(num_trace_samples)):
        z = torch.randint(0, 2, (total_dim,), device=device, dtype=torch.float32)
        z = 2.0 * z - 1.0
        z_list = unflatten_vector_to_list(z, shapes, numels)
        hvp_list = hessian_vector_product(model, batch, criterion, z_list, device=device)
        hv = flatten_params_from_list(hvp_list).detach()
        trace_vals.append(float((z * hv).sum().detach().cpu().item()))
    trace_H = float(np.mean(trace_vals)) if trace_vals else float("nan")

    return lambda_max, trace_H, v_top.detach()
