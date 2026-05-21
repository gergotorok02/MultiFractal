import torch
from common.torch_utils import safe_normalize
from eos.hessian.linalg import hvp

def power_extreme(model, batch, criterion, shapes, numels, iters: int, restarts: int, largest: bool, device):
    total_dim, best = sum(numels), None
    for _ in range(restarts):
        v = safe_normalize(torch.randn(total_dim, device=device))
        previous_v = v.clone()
        for _ in range(iters):
            h = hvp(model, batch, criterion, v, shapes, numels)
            previous_v = v.clone()
            v = safe_normalize(h if largest else -h)
        h = hvp(model, batch, criterion, previous_v, shapes, numels)
        lam = torch.dot(previous_v, h)
        residual = (h - lam * previous_v).norm().item()
        candidate = {"eig": float(lam.detach().cpu().item()),
                     "vec": previous_v.detach().clone(), "residual": float(residual)}
        if best is None or (largest and candidate["eig"] > best["eig"]):
            best = candidate
        if best is None or ((not largest) and candidate["eig"] < best["eig"]):
            best = candidate
    return best

def hutchinson_trace(model, batch, criterion, shapes, numels, samples: int, device):
    values, total_dim = [], sum(numels)
    for _ in range(samples):
        z = torch.randint(0, 2, (total_dim,), device=device, dtype=torch.float32)
        z = 2.0 * z - 1.0
        values.append(float(torch.dot(z, hvp(model, batch, criterion, z, shapes, numels)).detach().cpu().item()))
    return values
