import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from typing import Tuple, Union
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


def get_mnist_loaders(
    data_root: Union[str, Path] = "./data", batch_size=128
) -> Tuple[DataLoader[MNIST], DataLoader[MNIST]]:
    """Returns MNIST train and test dataloaders.
    Args:
        data_root (Union[str, Path], optional): Path to MNIST data. Defaults to "./data".
        batch_size (int, optional): Batch size. Defaults to 128.
    Returns:
        Tuple[DataLoader, DataLoader]: MNIST train and test dataloaders.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root=data_root, train=False, transform=transform),
        batch_size=batch_size,
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    data_root: Union[str, Path] = "./data", batch_size: int = 128
) -> Tuple[DataLoader[CIFAR10], DataLoader[CIFAR10]]:
    """Returns CIFAR10 train and test dataloaders.
    Args:
        data_root (Union[str, Path], optional): Path to CIFAR10 data. Defaults to "./data"
        batch_size (int, optional): Batch size. Defaults to 128.
    Returns:
        Tuple[DataLoader, DataLoader]: CIFAR10 train and test dataloaders.
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, test_loader
