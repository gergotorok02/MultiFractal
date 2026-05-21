from dataclasses import dataclass

@dataclass
class EosConfig:
    checkpoint_root: str = "./cifar10_resnet18_strong_baseline"
    training_log_path: str | None = None
    data_dir: str = "./data"
    out_path: str = "./eos_diagnostics_from_checkpoints.npz"
    seed: int = 0
    num_workers: int = 2
    test_batch_size: int = 256
    checkpoint_kind: str = "step"
    probe_every_n_checkpoints: int = 1
    max_checkpoints: int | None = None
    include_best: bool = False
    include_last: bool = False
    fixed_train_probe_size: int = 256
    fixed_test_probe_size: int = 256
    current_noaug_batch_size: int = 256
    batch_source: str = "fixed_train"
    model_mode: str = "eval"
    bn_mode: str = "eval"
    use_batch_stats_without_running_updates: bool = False
    do_lambda_max: bool = True
    do_lambda_min: bool = True
    do_trace: bool = True
    do_directional_curvatures: bool = True
    num_power_iters: int = 24
    num_power_restarts: int = 2
    num_trace_samples: int = 24
    out_json_summary: bool = True
