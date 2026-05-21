import torch
import torch.nn.functional as F

def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def move_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])

def safe_normalize(v, eps: float = 1e-12):
    return v / (v.norm() + eps)

@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval(); total = correct = 0
    for batch in loader:
        x, y = move_batch(batch, device)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)

@torch.no_grad()
def cross_entropy_on_batch(model, batch) -> float:
    x, y = batch
    return float(F.cross_entropy(model(x), y).item())
