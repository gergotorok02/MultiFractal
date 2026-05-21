from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .constants import CIFAR_MEAN, CIFAR_STD

def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def cifar10(root: str, train: bool, augment: bool = False):
    tfm = train_transform() if (train and augment) else eval_transform()
    return datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)

def loader(ds, batch_size: int, shuffle: bool, workers: int, drop_last: bool = False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True,
                      drop_last=drop_last, persistent_workers=workers > 0)

def subset_loader(ds, indices, batch_size: int, workers: int):
    return loader(Subset(ds, list(indices)), batch_size, False, workers)
