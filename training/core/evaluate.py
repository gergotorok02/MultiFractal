import torch
from common.torch_utils import move_batch

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval(); total = correct = 0; loss_sum = 0.0
    for batch in loader:
        x, y = move_batch(batch, device)
        logits = model(x)
        loss_sum += float(criterion(logits, y).item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.size(0))
    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total)}
