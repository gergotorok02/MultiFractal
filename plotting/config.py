from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PlotConfig:
    mf_npz_path: str = "./offline_multifractal_fixed_windows_full.npz"
    train_acc_npz_path: str = "./train_acc_30pts.npz"
    eos_path_candidates: Tuple[str, ...] = ("./eos_diagnostics_from_checkpoints.npz",)
    out_dir: Path = Path("./neurips_figures_bold_fixed_windows_v7_alpha_sweep")

    primary_protocol: str = "eval_eval_fixed_train"
    primary_family: str = "absolute"
    primary_variant: str = "global_abs_d2"
    variant_priority: Tuple[str, ...] = ("global_abs_d2", "gamma2_abs_d2", "gamma4_abs_d2", "global_abs_d1")

    early_frac: float = 0.10
    mid_frac: float = 0.50
    late_frac: float = 0.90
    appendix_fracs: Tuple[float, ...] = (0.00, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00)

    balanced_epoch_breaks: Tuple[float, float, float] = (5.0, 10.0, 30.0)
    balanced_color_levels: Tuple[float, float, float, float, float] = (0.00, 0.25, 0.50, 0.75, 1.00)

    @property
    def main_dir(self) -> Path:
        return self.out_dir / "main"

    @property
    def appendix_dir(self) -> Path:
        return self.out_dir / "appendix"

    def prepare_dirs(self) -> None:
        self.main_dir.mkdir(parents=True, exist_ok=True)
        self.appendix_dir.mkdir(parents=True, exist_ok=True)


CFG = PlotConfig()
