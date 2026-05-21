import numpy as np
import torch
from common.lr import infer_lr
from common.torch_utils import accuracy, move_batch
from multifractal.analysis.aggregate import aggregate_banks, checkpoint_summary
from multifractal.probing.bank_probe import probe_single_bank

def probe_checkpoint(model, ckpt_path, loaders, cfg, q_array, eps_grid, device, step_index, index):
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    step = int(payload.get("global_step", index))
    row = {"checkpoint_index": index, "checkpoint_name": ckpt_path.name,
           "path": str(ckpt_path), "step": step,
           "epoch": int(payload.get("epoch", -1)),
           "epoch_step": int(payload.get("epoch_step", -1)),
           "lr": infer_lr(payload, step_index)}
    if cfg.compute_test_acc:
        row["test_acc"] = float(accuracy(model, loaders["test_loader"], device))
    batches = loaders["fixed_train_batches"]
    bank_results = []
    for bi in range(cfg.num_direction_banks):
        batch = move_batch(batches[bi % len(batches)], device)
        bank_results.append(probe_single_bank(model, batch, eps_grid, q_array, cfg, bi))
    variants = aggregate_banks(bank_results)
    row["variants"] = variants
    row["summary"] = checkpoint_summary(variants)
    return row, bank_results
