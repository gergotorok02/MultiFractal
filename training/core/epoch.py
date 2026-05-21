from training.core.evaluate import evaluate
from training.io.epoch_log import log_epoch
from training.io.run_state import append_recent, step_entry
from training.core.step import train_step

def running_stats(loss_sum, correct, total, batch_loss, batch_acc, lr):
    return {"epoch_loss": loss_sum / max(1, total), "epoch_acc": correct / max(1, total),
            "last_loss": batch_loss, "last_acc": batch_acc, "lr": lr}

def maybe_save_step(ctx, batch_idx, stats):
    cfg = ctx["cfg"]
    if not (cfg.keep_step_checkpoints and cfg.save_every_steps > 0):
        return
    if ctx["state"]["global_step"] % cfg.save_every_steps != 0:
        return
    from training.io.finalize import save_step_checkpoint
    save_step_checkpoint(cfg, ctx["ckpt_dir"], ctx["model"], ctx["opt"], ctx["scheduler"],
                         ctx["scaler"], ctx["epoch"], ctx["state"]["global_step"],
                         batch_idx, ctx["state"]["recent"], stats)

def run_minibatches(ctx):
    loss_sum = correct = total = 0; last = None
    for batch_idx, batch in enumerate(ctx["loader"], start=1):
        last = run_one_batch(ctx, batch, batch_idx)
        n = int(batch[1].size(0)); loss_sum += last["loss"] * n
        correct += int(last["acc"] * n); total += n
        maybe_save_step(ctx, batch_idx, running_stats(loss_sum, correct, total,
                        last["loss"], last["acc"], last["lr"]))
    return running_stats(loss_sum, correct, total, last["loss"], last["acc"], last["lr"])

def run_one_batch(ctx, batch, batch_idx):
    cfg, state = ctx["cfg"], ctx["state"]
    stats = train_step(ctx["model"], batch, ctx["criterion"], ctx["opt"],
                       ctx["scaler"], cfg, ctx["device"], state["global_step"])
    state["global_step"] += 1
    if cfg.scheduler in ("cosine_warmup", "onecycle"): ctx["scheduler"].step()
    entry = step_entry(state["global_step"], ctx["epoch"], batch_idx,
                       len(ctx["loader"]), stats, state["t0"])
    ctx["meta"]["step_log"].append(entry)
    state["recent"] = append_recent(state["recent"], entry)
    return stats

def run_epoch(cfg, epoch, state, model, criterion, eval_criterion, opt,
              scheduler, scaler, train_loader, test_loader, device, ckpt_dir, meta):
    model.train()
    ctx = locals() | {"loader": train_loader, "meta": meta}
    train = run_minibatches(ctx)
    eval_m = evaluate(model, test_loader, device, eval_criterion)
    if cfg.scheduler == "cosine": scheduler.step()
    log_epoch(cfg, epoch, state, train, eval_m, meta, model, opt, scheduler, scaler, ckpt_dir)
