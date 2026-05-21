import random
import numpy as np
import torch
from dataclasses import asdict

def rng_state():
    state = {"python_random": random.getstate(),
             "numpy_random": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def save_checkpoint(path, model, optimizer, scheduler, scaler, cfg,
                    epoch, global_step, epoch_step, history_tail,
                    train_summary, eval_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": asdict(cfg), "epoch": epoch, "global_step": global_step,
        "epoch_step": epoch_step, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if cfg.save_optimizer_state else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_summary": train_summary, "eval_summary": eval_summary,
        "history_tail": history_tail,
    }
    if cfg.save_rng_state:
        payload["rng_state"] = rng_state()
    torch.save(payload, path)
