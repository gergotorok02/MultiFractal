import numpy as np
import torch
import torch.nn as nn

from ultramf.utils.params import (
    get_param_list, flatten_params_from_list,
    get_flat_param_vector, set_flat_param_vector
)

@torch.no_grad()
def loss_on_batch(model, batch, criterion):
    x, y = batch
    logits = model(x)
    loss = criterion(logits, y)
    return float(loss.item())

@torch.no_grad()
def margin_stats(model, batch):
    x, y = batch
    logits = model(x)
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().item())

    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), float("-inf"))
    other = masked.max(dim=1).values
    margin = (true - other).detach().cpu().numpy()
    qs = np.quantile(margin, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "acc": acc,
        "mean": float(margin.mean()),
        "median": float(qs[2]),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
    }

def grad_vector(model, batch, criterion, create_graph=False):
    params = get_param_list(model)
    x, y = batch
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=create_graph)
    gflat = flatten_params_from_list([g.reshape(-1) for g in grads])
    return gflat, float(loss.detach().item())

@torch.no_grad()
def logits_preds_for_vector(model, param_vec, batch, shapes, numels):
    backup = get_flat_param_vector(model).detach()
    set_flat_param_vector(model, param_vec, shapes, numels)
    x, y = batch
    logits = model(x)
    preds = logits.argmax(dim=1)
    set_flat_param_vector(model, backup, shapes, numels)
    return logits.detach(), preds.detach()

def loss_for_vector(model, param_vec, batch, criterion, shapes, numels):
    backup = get_flat_param_vector(model).detach()
    with torch.no_grad():
        set_flat_param_vector(model, param_vec, shapes, numels)
        l = loss_on_batch(model, batch, criterion)
        set_flat_param_vector(model, backup, shapes, numels)
    return l

def directional_1d_curve(
    model, batch, criterion,
    dir_vec, eps_grid, shapes, numels,
    also_minus=True, measure_flip=True
):
    base_vec = get_flat_param_vector(model).detach()
    base_loss = loss_on_batch(model, batch, criterion)
    base_margin = margin_stats(model, batch)
    _, base_preds = logits_preds_for_vector(model, base_vec, batch, shapes, numels)

    eps_grid = np.asarray([float(e) for e in eps_grid], dtype=np.float64)

    def eval_at(vec):
        l = loss_for_vector(model, vec, batch, criterion, shapes, numels)
        out = {"loss": l, "delta": l - base_loss}
        if measure_flip:
            _, preds = logits_preds_for_vector(model, vec, batch, shapes, numels)
            out["flip_rate"] = float((preds != base_preds).float().mean().item())
        return out

    plus = [eval_at(base_vec + eps * dir_vec) for eps in eps_grid]
    minus = [eval_at(base_vec - eps * dir_vec) for eps in eps_grid] if also_minus else None

    out = {
        "base_loss": float(base_loss),
        "base_margin": base_margin,
        "eps": eps_grid.astype(np.float32),
        "plus_loss": np.array([p["loss"] for p in plus], dtype=np.float32),
        "plus_delta": np.array([p["delta"] for p in plus], dtype=np.float32),
    }
    if measure_flip:
        out["plus_flip"] = np.array([p["flip_rate"] for p in plus], dtype=np.float32)

    if minus is not None:
        out["minus_loss"] = np.array([m["loss"] for m in minus], dtype=np.float32)
        out["minus_delta"] = np.array([m["delta"] for m in minus], dtype=np.float32)
        if measure_flip:
            out["minus_flip"] = np.array([m["flip_rate"] for m in minus], dtype=np.float32)

    return out
