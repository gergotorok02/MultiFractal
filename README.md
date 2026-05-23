# Multifractal Structure of Neural Network Loss Landscapes

Code for the MSc thesis **“Multifractal Structure of Neural Network Loss Landscapes”** - Gergő Török.

This repository contains three main components:

1. training a CIFAR-10 ResNet-18 model and saving checkpoints,
2. offline multifractal probing of the saved checkpoints,
3. edge-of-stability / Hessian diagnostics on the same checkpoints.


## Installation

Create a fresh Python environment:

```bash
conda create -n multifractal-landscape python=3.10
conda activate multifractal-landscape
```

Install PyTorch according to your CUDA version from the official PyTorch instructions.

For example, for a CUDA-enabled setup:

```bash
pip install torch torchvision torchaudio
```

Then install the remaining dependencies:

```bash
pip install numpy matplotlib tqdm
```

If the repository contains a `pyproject.toml`, the package can also be installed in editable mode:

```bash
pip install -e .
```

## Data

The experiments use CIFAR-10. The dataset is downloaded automatically by `torchvision` when running the training or probing scripts.

By default, data is stored under:

```bash
./data
```

This can be changed in the corresponding config files.

## 1. Training

Train the CIFAR-10 ResNet-18 baseline:

```bash
python scripts/train_cifar10.py --config configs/training_default.json
```

This creates an output directory containing:

```text
checkpoints/
training_log.json
config.json
```

The checkpoints are later used by the multifractal probing and EoS diagnostic scripts.

## 2. Multifractal Probing

Run offline multifractal probing on the saved checkpoints:

```bash
python scripts/run_multifractal_probe.py --config configs/multifractal_default.json
```

The probing code evaluates finite-scale loss changes along random parameter-space directions and computes multifractal quantities such as:

```text
Z(q, epsilon)
tau(q)
Chhabra-Jensen alpha(q), f(q)
directional scaling exponents
local tau maps
scalar heterogeneity summaries
```

The main output is an `.npz` file, for example:

```text
offline_multifractal_fixed_windows_fig03_only.npz
```

## 3. EoS Diagnostics

Run Hessian / edge-of-stability diagnostics on the same checkpoints:

```bash
python scripts/run_eos_diagnostics.py --config configs/eos_default.json
```

This computes quantities such as:

```text
lambda_max
lambda_min
eta * lambda_max
Hessian trace estimates
gradient-direction curvature
update-direction curvature
test accuracy
```

The main output is an `.npz` file, for example:

```text
eos_diagnostics_from_checkpoints.npz
```

## 4. Plotting

The repository also contains a dedicated plotting module for generating the final thesis figures from the saved experiment outputs.

The plotting code expects three result files:

```text
offline_multifractal_fixed_windows_full.npz
train_acc_30pts.npz
eos_diagnostics_from_checkpoints.npz
```

These files are not included directly in the GitHub repository because they are experiment artifacts. They are stored separately in a Google Drive folder:

[https://drive.google.com/drive/u/0/folders/1rRWmPNu-5VRQBx0tUGxuQPBw0yTRb6Jk]()


After downloading the three `.npz` files, place them in the project root or provide their paths explicitly when running the plotting script.

Run the plotting pipeline with:

```bash
python scripts/plot_results.py \
  --mf offline_multifractal_fixed_windows_full.npz \
  --train-acc train_acc_30pts.npz \
  --eos eos_diagnostics_from_checkpoints.npz \
  --out-dir figures
```

This creates the final figures under:

```text
figures/main/
figures/appendix/
```

The plotting code is separated from the measurement code. The training, multifractal probing, and EoS diagnostic scripts produce numerical artifacts, while the plotting module loads these artifacts and generates the thesis figures.


## Recommended Workflow

The intended workflow is:

```bash
# 1. Train model and save checkpoints
python scripts/train_cifar10.py --config configs/training_default.json

# 2. Run multifractal probing
python scripts/run_multifractal_probe.py --config configs/multifractal_default.json

# 3. Run EoS diagnostics
python scripts/run_eos_diagnostics.py --config configs/eos_default.json

# 4. Generate thesis figures from the saved .npz files
python scripts/plot_main_figures.py --config configs/plotting_default.json
```

## Configuration

Experiment settings are stored in JSON config files under:

```text
configs/
```

Typical parameters include:

```text
data_dir
checkpoint_root
out_path
seed
number of checkpoints
number of directions
epsilon scales
q values
probe batch size
Hessian power-iteration settings
```

For reproducibility, the same checkpoint sequence, probe batches, direction banks, and scale windows should be used when comparing training stages.
