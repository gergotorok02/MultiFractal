import torch
from training.core.metrics import current_lrs, grad_norm, weight_norm

def train_step(model, batch, criterion, optimizer, scaler, cfg, device, global_step):
    x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(cfg.use_amp and device.type == "cuda")):
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    gnorm = None
    if cfg.store_step_grad_norm and global_step % cfg.step_grad_norm_every == 0:
        scaler.unscale_(optimizer); gnorm = grad_norm(model)
    scaler.step(optimizer); scaler.update()
    with torch.no_grad():
        acc = float((logits.argmax(1) == y).float().mean().item())
    wnorm = None
    if cfg.store_weight_norm and global_step % cfg.weight_norm_every == 0:
        wnorm = weight_norm(model)
    return {
        "loss": float(loss.item()), "acc": acc, "grad_norm": gnorm,
        "weight_norm": wnorm, "lr": current_lrs(optimizer),
    }
