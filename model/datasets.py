import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split, Subset

import wandb


class ChannelDuplicater:
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


def get_dataset(Dataset, batch_size, val_ratio=None, train=True, subset=None, transform=None):    
    num_workers = wandb.config["num_workers"]
    dataset = Dataset('data', train=train, download=True, transform=transform)
    
    if subset is not None:
        indices = torch.randint(len(dataset), size=(int(len(dataset) * subset),))
        dataset = Subset(dataset, indices)
    
    if train:
        if val_ratio is not None:
            train_dataset, val_dataset = random_split(dataset, [1 - val_ratio, val_ratio])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            return train_loader, val_loader
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
def get_MNIST(batch_size, val_ratio=0.1, train=True, subset=None):
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        ChannelDuplicater(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return get_dataset(MNIST, batch_size, val_ratio, train, subset, transform)


def get_CIFAR10(batch_size, val_ratio=0.1, train=True, subset=None):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    return get_dataset(CIFAR10, batch_size, val_ratio, train, subset, transform)
    

def get_loaders(dataset_name, batch_size, val_ratio=None, train=True, subset=None):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_fetchers = {
        "mnist": get_MNIST,
        "cifar10": get_CIFAR10
    }

    if dataset_name not in dataset_fetchers:
        raise ValueError(f"Unknown dataset name {dataset_name}")
    
    return dataset_fetchers[dataset_name](batch_size, val_ratio, train, subset)