from dataclasses import dataclass, field
import numpy as np

@dataclass
class ProbeConfig:
    checkpoint_root: str = "./cifar10_resnet18_strong_baseline"
    training_log_path: str | None = None
    data_dir: str = "./data"
    out_path: str = "./offline_multifractal_fixed_windows_fig03_only.npz"
    seed: int = 0
    num_workers: int = 2
    test_batch_size: int = 256
    checkpoint_kind: str = "step"
    probe_every_n_checkpoints: int = 1
    max_checkpoints: int | None = None
    include_best: bool = False
    include_last: bool = False
    fig03_only_checkpoints: bool = True
    fig03_fractions: tuple[float, ...] = (0.10, 0.50, 0.90)
    representative_fractions: tuple[float, ...] = tuple([0.00,0.015,0.030,0.045,0.060,0.080,0.100,0.125,0.150,0.180,0.210,0.245,0.280,0.320,0.360,0.405,0.450,0.500,0.550,0.605,0.660,0.715,0.770,0.820,0.870,0.910,0.940,0.965,0.985,1.000])
    fixed_train_probe_size: int = 256
    fixed_test_probe_size: int = 256
    fixed_train_probe_batches: int = 3
    fixed_test_probe_batches: int = 2
    num_direction_banks: int = 8
    directions_per_bank: int = 128
    direction_seed_stride: int = 100_000
    direction_mode: str = "filter_norm"
    eps_abs: tuple[float, ...] = field(default_factory=lambda: tuple(np.logspace(-6.5, -1.9, 12)))
    fixed_windows: tuple[tuple[int, int], ...] = ((5,9),(6,10),(7,11),(8,12),(5,10),(6,11),(7,12),(5,11))
    fixed_window_require_trust: bool = False
    q_list: tuple[float, ...] = (-3,-2,-1.5,-1,-.5,.25,.5,.75,1,1.25,1.5,2,3,4)
    gamma_variants: tuple[float, ...] = (1.0, 2.0, 4.0)
    include_abs_d1_variant: bool = True
    eps_floor: float = 1e-20
    p_floor: float = 1e-10
    snr_threshold: float = 0.35
    min_positive_fraction: float = 0.20
    max_windows_to_keep: int = 8
    compute_test_acc: bool = True
