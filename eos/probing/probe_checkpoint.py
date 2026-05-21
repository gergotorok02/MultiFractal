import torch
import torch.nn as nn
from common.lr import infer_lr
from common.torch_utils import accuracy
from eos.model.bn import measurement_mode
from eos.hessian.directional_metrics import add_directional_curvatures
from eos.hessian.linalg import loss_and_grad
from eos.hessian.spectral_metrics import add_extreme_eigenvalues, add_trace

def base_entry(path, payload, index, lr):
    return {"checkpoint_index": index, "checkpoint_name": path.name, "path": str(path),
            "step": int(payload.get("global_step", index)),
            "epoch": int(payload.get("epoch", -1)),
            "epoch_step": int(payload.get("epoch_step", -1)), "lr": float(lr)}

def probe_checkpoint(cfg, model, path, index, get_batch, loaders,
                     step_index, shapes, numels, device, previous_theta):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    criterion = nn.CrossEntropyLoss()
    batch = get_batch(cfg.batch_source)
    with measurement_mode(model, cfg.model_mode, cfg.bn_mode,
                          cfg.use_batch_stats_without_running_updates):
        loss, gflat = loss_and_grad(model, batch, criterion)
        entry = base_entry(path, payload, index, infer_lr(payload, step_index))
        entry["loss"] = loss
        entry["grad_norm"] = float(gflat.norm().item())
        entry["test_acc"] = float(accuracy(model, loaders["test_loader"], device))
        add_extreme_eigenvalues(cfg, model, batch, criterion, shapes, numels, entry, device)
        add_trace(cfg, model, batch, criterion, shapes, numels, entry, device)
        theta = add_directional_curvatures(cfg, model, batch, criterion, shapes,
                                           numels, entry, gflat, previous_theta)
    return entry, theta
