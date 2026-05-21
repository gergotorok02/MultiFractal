from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from common.checkpoints import load_training_log, step_log_index
from common.constants import default_device
from common.model import CIFARResNet18
from common.seed import set_seed
from multifractal.probing.checkpoint_probe import probe_checkpoint
from multifractal.io.checkpoints import selected_checkpoints
from multifractal.data.loaders import build_probe_loaders

def offline_probe(cfg):
    set_seed(cfg.seed)
    device = torch.device(default_device())
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    loaders = build_probe_loaders(cfg)
    step_index = step_log_index(load_training_log(cfg.checkpoint_root, cfg.training_log_path))
    ckpts = selected_checkpoints(cfg)
    model = CIFARResNet18().to(device)
    q_array, eps_grid = np.asarray(cfg.q_list, float), np.asarray(cfg.eps_abs, float)
    rows, raw_banks = [], []
    for i, ckpt in enumerate(ckpts):
        print(f"[{i+1}/{len(ckpts)}] {ckpt.name}", flush=True)
        row, banks = probe_checkpoint(model, ckpt, loaders, cfg, q_array, eps_grid, device, step_index, i)
        rows.append(row); raw_banks.append(banks)
    save_payload(cfg, rows, raw_banks, loaders, q_array, eps_grid, device)
    print(f"Saved: {cfg.out_path}")
    return cfg.out_path

def save_payload(cfg, rows, raw_banks, loaders, q_array, eps_grid, device):
    np.savez(cfg.out_path,
             logs=np.asarray(rows, dtype=object),
             raw_bank_results=np.asarray(raw_banks, dtype=object),
             q_list=q_array, eps_abs=eps_grid,
             fixed_train_indices_all=loaders["fixed_train_indices_all"],
             fixed_test_indices_all=loaders["fixed_test_indices_all"],
             meta=np.asarray({"cfg": asdict(cfg), "device": str(device),
                              "version": "refactored_fixed_window_probe"}, dtype=object))
