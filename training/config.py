from dataclasses import dataclass
from common.constants import default_device

@dataclass
class TrainConfig:
    seed: int = 0
    out_dir: str = "./cifar10_resnet18_strong_baseline"
    data_dir: str = "./data"
    epochs: int = 200
    batch_size: int = 128
    test_batch_size: int = 256
    num_workers: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.0
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    onecycle_pct_start: float = 0.2
    use_amp: bool = True
    device: str = default_device()
    log_every: int = 50
    save_every_steps: int = 100
    save_every_epochs: int = 1
    keep_step_checkpoints: bool = True
    save_optimizer_state: bool = False
    save_rng_state: bool = False
    store_step_grad_norm: bool = True
    step_grad_norm_every: int = 25
    store_weight_norm: bool = True
    weight_norm_every: int = 50
