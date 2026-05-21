import math
import torch
from common.torch_utils import trainable_params

def sample_bank(model, cfg, bank_index: int):
    base_seed = cfg.seed + cfg.direction_seed_stride * (bank_index + 1)
    directions = []
    for di in range(cfg.directions_per_bank):
        torch.manual_seed(base_seed + di)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(base_seed + di)
        chunks, sq = [], 0.0
        for p in trainable_params(model):
            z = torch.randn_like(p)
            if cfg.direction_mode == "filter_norm":
                z = z * (p.detach().norm() / (z.norm() + 1e-12))
            chunks.append(z); sq += float((z.float() ** 2).sum().cpu())
        directions.append([z / math.sqrt(max(sq, 1e-30)) for z in chunks])
    return directions

def apply_(model, direction, scale: float) -> None:
    with torch.no_grad():
        for p, d in zip(trainable_params(model), direction):
            p.add_(d, alpha=scale)
