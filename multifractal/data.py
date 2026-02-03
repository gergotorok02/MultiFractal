from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def build_loaders(batch_size, use_randaugment, ra_num_ops, ra_magnitude, num_workers=2, root="./data"):
    tf_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        tf_train.append(transforms.RandAugment(num_ops=int(ra_num_ops), magnitude=int(ra_magnitude)))
    tf_train += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    train_transform = transforms.Compose(tf_train)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    probe_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    probe_batch = next(iter(probe_loader))
    return train_loader, test_loader, probe_batch
