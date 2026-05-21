import numpy as np
from common.data import cifar10, loader, subset_loader

def build_probe_loaders(cfg):
    train_ds = cifar10(cfg.data_dir, train=True, augment=False)
    test_ds = cifar10(cfg.data_dir, train=False, augment=False)
    test_loader = loader(test_ds, cfg.test_batch_size, False, cfg.num_workers)
    rng = np.random.default_rng(cfg.seed)
    train_batches, test_batches, train_idx, test_idx = [], [], [], []
    for _ in range(cfg.fixed_train_probe_batches):
        idx = rng.choice(len(train_ds), size=cfg.fixed_train_probe_size, replace=False)
        train_idx.append(idx.astype(np.int64))
        train_batches.append(next(iter(subset_loader(train_ds, idx, cfg.fixed_train_probe_size, cfg.num_workers))))
    for _ in range(cfg.fixed_test_probe_batches):
        idx = rng.choice(len(test_ds), size=cfg.fixed_test_probe_size, replace=False)
        test_idx.append(idx.astype(np.int64))
        test_batches.append(next(iter(subset_loader(test_ds, idx, cfg.fixed_test_probe_size, cfg.num_workers))))
    return {"test_loader": test_loader, "fixed_train_batches": train_batches,
            "fixed_test_batches": test_batches, "fixed_train_indices_all": np.asarray(train_idx, dtype=object),
            "fixed_test_indices_all": np.asarray(test_idx, dtype=object)}
