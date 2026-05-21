from common.checkpoints import discover, select_by_fraction

def selected_checkpoints(cfg):
    paths = discover(cfg.checkpoint_root, cfg.checkpoint_kind,
                     cfg.probe_every_n_checkpoints, cfg.max_checkpoints,
                     cfg.include_best, cfg.include_last)
    paths = select_by_fraction(paths, cfg.representative_fractions)
    if cfg.fig03_only_checkpoints:
        paths = select_by_fraction(paths, cfg.fig03_fractions)
    if not paths:
        raise FileNotFoundError("No checkpoints found.")
    for path in paths: print(f"  - {path}", flush=True)
    return paths
