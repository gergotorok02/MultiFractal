import time
from dataclasses import asdict

def initial_metadata(cfg, device, steps_per_epoch: int):
    return {
        "config": asdict(cfg), "device": str(device),
        "steps_per_epoch": steps_per_epoch,
        "step_log": [], "epoch_log": [], "best": None,
    }

def step_entry(global_step, epoch, batch_idx, steps_per_epoch, stats, t0):
    return {
        "global_step": global_step, "epoch": epoch,
        "epoch_progress": epoch - 1 + batch_idx / steps_per_epoch,
        "batch_idx": batch_idx, "lr": stats["lr"],
        "batch_loss": stats["loss"], "batch_acc": stats["acc"],
        "grad_norm": stats["grad_norm"], "weight_norm": stats["weight_norm"],
        "wallclock_seconds": float(time.time() - t0),
    }

def append_recent(recent, entry, max_len: int = 200):
    recent.append(entry)
    return recent[-max_len:]
