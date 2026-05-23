# Plotting

Publication-style figure generation for the thesis experiments.

Run all figures with:

```bash
python scripts/plot_results.py \
  --mf offline_multifractal_fixed_windows_full.npz \
  --train-acc train_acc_30pts.npz \
  --eos eos_diagnostics_from_checkpoints.npz \
  --out-dir figures
```

The script writes main figures to `figures/main/` and appendix figures to `figures/appendix/`.

The plotting package separates result loading, alignment, extraction, styling, and figure builders. The plotting code assumes that the multifractal `.npz`, training-accuracy `.npz`, and EoS `.npz` files were produced by the corresponding repository pipelines.
