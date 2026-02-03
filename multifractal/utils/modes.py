from contextlib import contextmanager
import torch.nn as nn

def set_bn_eval_only(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

@contextmanager
def measurement_eval_mode(model: nn.Module, bn_eval: bool = True):
    prev = model.training
    model.eval()
    try:
        if bn_eval:
            set_bn_eval_only(model)
        yield
    finally:
        model.train(prev)

@contextmanager
def train_mode_with_bn_frozen(model: nn.Module, bn_eval: bool = True):
    prev = model.training
    model.train(True)
    try:
        if bn_eval:
            set_bn_eval_only(model)
        yield
    finally:
        model.train(prev)
