import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from common.checkpoints import discover, load_training_log, step_log_index
from common.constants import default_device
from common.model import CIFARResNet18
from common.seed import set_seed
from common.torch_utils import trainable_params
from eos.hessian.linalg import shapes_numels
from eos.data.loaders import batch_getter, build_loaders
from eos.probing.probe_checkpoint import probe_checkpoint

def eos_probe(cfg):
    set_seed(cfg.seed)
    device = torch.device(default_device())
    out_path = Path(cfg.out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(cfg); get_batch = batch_getter(loaders, device)
    step_index = step_log_index(load_training_log(cfg.checkpoint_root, cfg.training_log_path))
    paths = discover(cfg.checkpoint_root, cfg.checkpoint_kind, cfg.probe_every_n_checkpoints,
                     cfg.max_checkpoints, cfg.include_best, cfg.include_last)
    model = CIFARResNet18().to(device)
    shapes, numels = shapes_numels(trainable_params(model))
    logs, previous_theta = [], None
    for i, path in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] {path.name}", flush=True)
        entry, previous_theta = probe_checkpoint(cfg, model, path, i, get_batch, loaders,
                                                 step_index, shapes, numels, device, previous_theta)
        logs.append(entry)
    save_outputs(cfg, out_path, logs, loaders, device)
    print(f"Saved: {out_path}")
    return str(out_path)

def save_outputs(cfg, out_path, logs, loaders, device):
    np.savez(out_path, logs=np.asarray(logs, dtype=object),
             meta=np.asarray({"cfg": asdict(cfg), "device": str(device)}, dtype=object),
             probe_indices=np.asarray({"fixed_train_indices": loaders["fixed_train_indices"],
                                       "fixed_test_indices": loaders["fixed_test_indices"]}, dtype=object))
    if cfg.out_json_summary:
        summary = {"num_checkpoints": len(logs),
                   "best_test_acc": stat(logs, "test_acc", np.nanmax),
                   "max_eta_lambda": stat(logs, "eta_lambda_max", np.nanmax),
                   "mean_lambda_max": stat(logs, "lambda_max", np.nanmean),
                   "mean_gHg_over_gg": stat(logs, "gHg_over_gg", np.nanmean)}
        out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))

def stat(logs, key, fn):
    vals = [x.get(key, np.nan) for x in logs]
    return float(fn(vals)) if vals else None
