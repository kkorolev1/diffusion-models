import torch
import torch.nn as nn
import numpy as np
import scipy

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from tqdm import tqdm


def fid_score(real_data, fake_data, feature):
    device = fake_data.get_device()
    fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device if device >= 0 else 'cpu')

    print("Calculate activations for real data")
    fid.update(real_data, real=True)
    
    print("Calculate activations for fake data")
    fid.update(fake_data, real=False)
    
    print("Compute FID")
    return fid.compute().item()

def inception_score(fake_data):
    inception = InceptionScore(feature=64, normalize=True)
    inception.update(fake_data)
    return inception.compute()