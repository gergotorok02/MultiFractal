import numpy as np
from common.data import cifar10, loader, subset_loader
from common.torch_utils import move_batch

def build_loaders(cfg):
    train_ds = cifar10(cfg.data_dir, train=True, augment=False)
    test_ds = cifar10(cfg.data_dir, train=False, augment=False)
    train = loader(train_ds, cfg.current_noaug_batch_size, True, cfg.num_workers, True)
    test = loader(test_ds, cfg.test_batch_size, False, cfg.num_workers)
    rng = np.random.default_rng(cfg.seed)
    tr_idx = rng.choice(len(train_ds), cfg.fixed_train_probe_size, replace=False)
    te_idx = rng.choice(len(test_ds), cfg.fixed_test_probe_size, replace=False)
    return {"train_noaug_loader": train, "test_loader": test,
            "fixed_train_batch": next(iter(subset_loader(train_ds, tr_idx, cfg.fixed_train_probe_size, cfg.num_workers))),
            "fixed_test_batch": next(iter(subset_loader(test_ds, te_idx, cfg.fixed_test_probe_size, cfg.num_workers))),
            "fixed_train_indices": tr_idx.astype(np.int64),
            "fixed_test_indices": te_idx.astype(np.int64)}

def batch_getter(loaders, device):
    iterator = iter(loaders["train_noaug_loader"])
    def get(source: str):
        nonlocal iterator
        if source == "fixed_train": return move_batch(loaders["fixed_train_batch"], device)
        if source == "fixed_test": return move_batch(loaders["fixed_test_batch"], device)
        if source == "current_train_noaug":
            try: batch = next(iterator)
            except StopIteration:
                iterator = iter(loaders["train_noaug_loader"]); batch = next(iterator)
            return move_batch(batch, device)
        raise ValueError(f"Unknown batch source: {source}")
    return get
