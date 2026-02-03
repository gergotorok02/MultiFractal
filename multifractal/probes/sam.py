import torch

from multifractal.utils.params import get_flat_param_vector, set_flat_param_vector
from multifractal.probes.curves import loss_on_batch, loss_for_vector, grad_vector

def sam_sharpness_probe(
    model, batch, criterion,
    rho_list, shapes, numels,
    steps=1, lr_ascent=1.0,
):
    base_vec = get_flat_param_vector(model).detach()
    base_loss = loss_on_batch(model, batch, criterion)

    results = []
    for rho in rho_list:
        theta = base_vec.clone()
        delta = torch.zeros_like(theta)

        for _ in range(int(steps)):
            set_flat_param_vector(model, theta + delta, shapes, numels)
            g, _ = grad_vector(model, batch, criterion, create_graph=False)
            g = g.detach()
            delta = delta + float(lr_ascent) * g
            dn = delta.norm() + 1e-12
            if dn > float(rho):
                delta = delta * (float(rho) / dn)

        sharp_loss = loss_for_vector(model, theta + delta, batch, criterion, shapes, numels)
        results.append({
            "rho": float(rho),
            "sharpness": float(sharp_loss - base_loss),
            "loss_at_pert": float(sharp_loss),
            "delta_norm": float(delta.norm().detach().cpu().item()),
        })
        set_flat_param_vector(model, base_vec, shapes, numels)

    return {"base_loss": float(base_loss), "items": results}
