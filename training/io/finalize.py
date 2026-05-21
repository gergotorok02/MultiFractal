from training.io.checkpoint import save_checkpoint

def save_step_checkpoint(cfg, ckpt_dir, model, opt, scheduler, scaler,
                         epoch, global_step, batch_idx, recent, stats):
    summary = {"epoch_running_loss": stats["epoch_loss"],
               "epoch_running_acc": stats["epoch_acc"],
               "latest_batch_loss": stats["last_loss"],
               "latest_batch_acc": stats["last_acc"], "latest_lr": stats["lr"]}
    save_checkpoint(ckpt_dir / f"step_{global_step:07d}.pt", model, opt,
                    scheduler, scaler, cfg, epoch, global_step, batch_idx,
                    recent, summary, None)

def save_final(cfg, ckpt_dir, model, opt, scheduler, scaler, global_step, recent, meta):
    last = meta["epoch_log"][-1]
    save_checkpoint(ckpt_dir / "last.pt", model, opt, scheduler, scaler, cfg,
                    cfg.epochs, global_step, -1, recent, last,
                    {"loss": last["test_loss"], "acc": last["test_acc"]})
