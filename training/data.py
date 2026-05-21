from common.data import cifar10, loader

def build_loaders(cfg):
    train_ds = cifar10(cfg.data_dir, train=True, augment=True)
    test_ds = cifar10(cfg.data_dir, train=False, augment=False)
    train = loader(train_ds, cfg.batch_size, True, cfg.num_workers, drop_last=True)
    test = loader(test_ds, cfg.test_batch_size, False, cfg.num_workers)
    return train, test
