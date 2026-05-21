# Refactor notes

The original three monolithic scripts were reorganized into three top-level workflows:

- `training/`
- `multifractal/`
- `eos/`

Shared code moved into `common/`, including CIFAR-10 transforms, the CIFAR-adapted ResNet-18, checkpoint discovery, learning-rate recovery, seed handling, and tensor utilities.

## Original-to-new entry points

| Original script | New entry point |
|---|---|
| `cifar10_resnet18_regular_cosine_training.py` | `scripts/train_cifar10.py` |
| `offline_probe_fixed_fig03_multiwindow_final.py` | `scripts/run_multifractal_probe.py` |
| `eos_diagnostics_from_checkpoints.py` | `scripts/run_eos_diagnostics.py` |

## Internal grouping

- `training/core/` contains the actual training mechanics.
- `training/io/` contains checkpointing and log/final-artifact writing.
- `multifractal/probing/` contains perturbation and direction-bank mechanics.
- `multifractal/analysis/` contains mathematical estimators and aggregation.
- `multifractal/io/` contains checkpoint-selection logic.
- `eos/hessian/` contains HVP, eigenvalue, trace, and curvature routines.
- `eos/model/` contains BatchNorm/measurement-mode handling.
- `eos/probing/` contains the per-checkpoint diagnostic routine.

All Python files are kept at or below 50 lines in this refactor.
