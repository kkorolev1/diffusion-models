import torch
import torch.nn as nn
import numpy as np
import scipy

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from tqdm import tqdm


def fid_score(real_data, fake_data):
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid.update(real_data, real=True)
    fid.update(fake_data, real=False)
    return fid.compute().item()

def inception_score(fake_data):
    inception = InceptionScore(feature=64, normalize=True)
    inception.update(fake_data)
    return inception.compute()