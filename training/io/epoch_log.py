import time
from training.io.checkpoint import save_checkpoint
from training.core.metrics import current_lrs

def log_epoch(cfg, epoch, state, train, eval_m, meta, model, opt, scheduler, scaler, ckpt_dir):
    row = {"epoch": epoch, "global_step": state["global_step"],
           "train_loss": train["epoch_loss"], "train_acc": train["epoch_acc"],
           "test_loss": eval_m["loss"], "test_acc": eval_m["acc"],
           "lr_end_epoch": current_lrs(opt),
           "wallclock_seconds": float(time.time() - state["t0"])}
    meta["epoch_log"].append(row)
    print(f"[Epoch {epoch:03d}/{cfg.epochs:03d}] train_loss={row['train_loss']:.4f} "
          f"train_acc={row['train_acc']:.4f} test_loss={row['test_loss']:.4f} "
          f"test_acc={row['test_acc']:.4f}")
    maybe_save_best(cfg, epoch, state, train, eval_m, meta, model, opt, scheduler, scaler, ckpt_dir)

def maybe_save_best(cfg, epoch, state, train, eval_m, meta, model, opt, scheduler, scaler, ckpt_dir):
    if eval_m["acc"] <= state["best"]:
        return
    state["best"] = eval_m["acc"]
    meta["best"] = {"epoch": epoch, "global_step": state["global_step"],
                    "test_acc": state["best"], "test_loss": eval_m["loss"]}
    save_checkpoint(ckpt_dir / "best.pt", model, opt, scheduler, scaler, cfg,
                    epoch, state["global_step"], -1, state["recent"], train, eval_m)
