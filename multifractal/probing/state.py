import torch

def snapshot(model):
    return {"params": {k: v.detach().clone() for k, v in model.named_parameters()},
            "buffers": {k: v.detach().clone() for k, v in model.named_buffers()}}

def restore(model, state) -> None:
    params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    with torch.no_grad():
        for key, value in state["params"].items():
            params[key].copy_(value)
        for key, value in state["buffers"].items():
            buffers[key].copy_(value)
