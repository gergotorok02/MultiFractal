import numpy as np
import torch

from multifractal.probes.curves import grad_vector

def grad_noise_proxy(model, loader_iter, k_batches, criterion, device):
    grads, norms, losses = [], [], []
    for _ in range(int(k_batches)):
        x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)
        g, l = grad_vector(model, (x, y), criterion, create_graph=False)
        g = g.detach()
        grads.append(g)
        norms.append(float(g.norm().cpu().item()))
        losses.append(float(l))

    G = len(grads)
    if G < 2:
        return {
            "k": G,
            "mean_norm": float(np.mean(norms)) if norms else 0.0,
            "std_norm": 0.0,
            "pairwise_cos_mean": float("nan"),
            "diff_norm_mean": float("nan"),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    cos_vals, diff_vals = [], []
    for i in range(G):
        for j in range(i + 1, G):
            gi, gj = grads[i], grads[j]
            cos = float(torch.dot(gi, gj).cpu().item() / ((gi.norm() * gj.norm()).cpu().item() + 1e-12))
            cos_vals.append(cos)
            diff_vals.append(float((gi - gj).norm().cpu().item()))

    return {
        "k": G,
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "pairwise_cos_mean": float(np.mean(cos_vals)),
        "diff_norm_mean": float(np.mean(diff_vals)),
        "mean_loss": float(np.mean(losses)),
    }
