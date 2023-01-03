import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


def get_MNIST(batch_size):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    dataset = MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader

def get_CIFAR10(batch_size):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        lambda image: image[0, :, :],
        transforms.Normalize((0.5,), (0.5,))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset = CIFAR10('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return dataloader