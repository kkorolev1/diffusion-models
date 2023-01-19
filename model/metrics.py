import torch
import torch.nn as nn
import numpy as np
import scipy

from tqdm import tqdm


class InceptionHeadless(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=7)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.fc = nn.Identity()  # remove last fc layer
    
    def forward(self, x):
        x = self.upsample(x)
        return self.model(x)

inception = InceptionHeadless()


def calculate_activations(data, batch_size=32):
    # Calculate activations of Pool3 layer of InceptionV3
    output = []
    slicer = range(0, len(data), batch_size)

    inception.eval()
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        inception.cuda()

    with torch.no_grad():
        for i in tqdm(slicer):
            x = data[i:i + batch_size]
            if use_gpu:
                x = x.cuda()
            y = inception(x)
            if use_gpu:
                y = y.cpu()
            output.append(y)
    
    if use_gpu:
        inception.cpu()

    output = torch.cat(output, dim=0)
    return output

def calculate_activation_statistics(activations):
    mu = torch.mean(activations, axis=0)
    sigma = torch.from_numpy(np.cov(activations, rowvar=False))
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    sigma1_sigma2 = scipy.linalg.sqrtm(np.dot(sigma1, sigma2))

    if np.iscomplexobj(sigma1_sigma2):
        sigma1_sigma2 = sigma1_sigma2.real

    if not np.isfinite(sigma1_sigma2).all():
        offset = np.eye(sigma1.shape[0]) * eps
        sigma1_sigma2 = scipy.linalg.sqrtm(np.dot(sigma1 + offset, sigma2 + offset))

    diff = mu1 - mu2
    
    return  (diff**2).sum() + np.trace(sigma1 + sigma2 - 2 * sigma1_sigma2)

def fid_score(real_data, fake_data):
    # Run inception on real and fake data to obtain activations
    real_activations = calculate_activations(real_data)
    fake_activations = calculate_activations(fake_data)

    # Calculate mu and sigma for both real and fake activations
    real_mu, real_sigma = calculate_activation_statistics(real_activations)
    fake_mu, fake_sigma = calculate_activation_statistics(fake_activations)

    return frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)