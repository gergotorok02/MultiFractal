import torch
from common.torch_utils import flatten, safe_normalize, trainable_params

def shapes_numels(params):
    return [tuple(p.shape) for p in params], [p.numel() for p in params]

def flat_parameters(model):
    return flatten([p.detach().reshape(-1) for p in trainable_params(model)])

def loss_and_grad(model, batch, criterion):
    params = trainable_params(model)
    x, y = batch
    model.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
    return float(loss.detach().item()), flatten([g.reshape(-1) for g in grads]).detach()

def unflatten_like(vector, shapes, numels):
    out, offset = [], 0
    for shape, numel in zip(shapes, numels):
        out.append(vector[offset:offset + numel].view(shape))
        offset += numel
    return out

def hvp(model, batch, criterion, direction, shapes, numels):
    params = trainable_params(model)
    loss = criterion(model(batch[0]), batch[1])
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_dot_v = sum((g * v).sum() for g, v in zip(grads, unflatten_like(direction, shapes, numels)))
    hv_parts = torch.autograd.grad(g_dot_v, params, retain_graph=False)
    return flatten([h.reshape(-1) for h in hv_parts]).detach()

def directional_curvature(model, batch, criterion, direction, shapes, numels) -> float:
    d = safe_normalize(direction.detach())
    return float(torch.dot(d, hvp(model, batch, criterion, d, shapes, numels)).detach().cpu().item())
