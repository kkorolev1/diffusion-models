import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split


class ChannelDuplicater:
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


def get_MNIST(batch_size, val_ratio=0.1):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        ChannelDuplicater(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MNIST('data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def get_CIFAR10(batch_size, val_ratio=0.1):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def get_loaders(dataset_name, batch_size, val_ratio=0.1):
    dataset_fetchers = {
        "mnist": get_MNIST,
        "cifar10": get_CIFAR10
    }

    if dataset_name not in dataset_fetchers:
        raise ValueError(f"Unknown dataset name {dataset_name}")
    
    return dataset_fetchers[dataset_name](batch_size, val_ratio)