import math
import torch

@torch.no_grad()
def weight_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + p.detach().pow(2).sum()
    return float(torch.sqrt(total).item())

def grad_norm(model) -> float:
    value = 0.0
    for p in model.parameters():
        if p.grad is not None:
            value += float(p.grad.detach().pow(2).sum().item())
    return float(math.sqrt(max(value, 0.0)))

def current_lrs(optimizer):
    return [float(group["lr"]) for group in optimizer.param_groups]
