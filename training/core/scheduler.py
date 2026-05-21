import math
import torch.optim as optim

def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "cosine_warmup":
        warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            den = float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / den))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if cfg.scheduler == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=total_steps,
            pct_start=cfg.onecycle_pct_start, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1e4)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
