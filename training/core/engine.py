import time
from pathlib import Path
import torch
from common.io import save_json
from common.seed import set_seed
from training.core.builder import build_objects
from training.data import build_loaders
from training.core.epoch import run_epoch
from training.io.finalize import save_final
from training.io.run_state import initial_metadata
from training.core.scheduler import build_scheduler

def run_training(cfg):
    set_seed(cfg.seed)
    out, ckpt_dir = Path(cfg.out_dir), Path(cfg.out_dir) / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    train_loader, test_loader = build_loaders(cfg)
    device, model, criterion, eval_criterion, opt = build_objects(cfg)
    scheduler = build_scheduler(opt, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))
    save_json(out / "config.json", cfg.__dict__)
    meta = initial_metadata(cfg, device, len(train_loader))
    state = {"best": -1.0, "global_step": 0, "recent": [], "t0": time.time()}
    for epoch in range(1, cfg.epochs + 1):
        run_epoch(cfg, epoch, state, model, criterion, eval_criterion, opt,
                  scheduler, scaler, train_loader, test_loader,
                  device, ckpt_dir, meta)
        save_json(out / "training_log.json", meta)
    save_final(cfg, ckpt_dir, model, opt, scheduler, scaler,
               state["global_step"], state["recent"], meta)
    save_json(out / "training_log.json", meta)
    print(f"Finished. Artifacts written to: {out}")
