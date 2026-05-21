from contextlib import contextmanager
from dataclasses import dataclass
from common.constants import BN_TYPES

@dataclass
class BNState:
    training_flags: dict
    track_running_stats: dict

def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, BN_TYPES)]

@contextmanager
def measurement_mode(model, model_mode: str, bn_mode: str, no_running_updates: bool = False):
    previous_model_training = model.training
    previous = BNState(training_flags={}, track_running_stats={})
    model.train(model_mode == "train")
    for module in bn_modules(model):
        previous.training_flags[id(module)] = module.training
        previous.track_running_stats[id(module)] = bool(getattr(module, "track_running_stats", True))
        module.train(bn_mode == "train")
        if bn_mode == "train" and no_running_updates and hasattr(module, "track_running_stats"):
            module.track_running_stats = False
    try:
        yield
    finally:
        model.train(previous_model_training)
        for module in bn_modules(model):
            module.train(previous.training_flags[id(module)])
            if hasattr(module, "track_running_stats"):
                module.track_running_stats = previous.track_running_stats[id(module)]
