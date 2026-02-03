import torch
import torch.nn as nn

def get_param_list(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def get_param_shapes_and_numels(params):
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    return shapes, numels

def flatten_params_from_list(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])

def unflatten_vector_to_list(vec, shapes, numels):
    out = []
    off = 0
    for sh, n in zip(shapes, numels):
        out.append(vec[off:off+n].view(sh))
        off += n
    return out

def get_flat_param_vector(model: nn.Module):
    params = get_param_list(model)
    return flatten_params_from_list([p.data for p in params])

def set_flat_param_vector(model: nn.Module, flat_vec, shapes, numels):
    params = get_param_list(model)
    assert flat_vec.numel() == sum(numels)
    off = 0
    with torch.no_grad():
        for p, sh, n in zip(params, shapes, numels):
            p.data.copy_(flat_vec[off:off+n].view(sh))
            off += n
