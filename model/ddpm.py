import torch
import torch.nn as nn
import numpy as np


class DDPM(nn.Module):
    def __init__(self, model, T=1000, beta_1=1e-4, beta_T=0.02, device=None):
        super().__init__()

        self.model = model

        self.T = T

        # linear variance scheduling
        self.betas = torch.linspace(beta_1, beta_T, T).to(device)
        self.alphas = 1 - self.betas
        self.alpha_prods = torch.from_numpy(np.cumprod(self.alphas.cpu().numpy())).to(device)

        self.device = device

    def forward(self, x_0, t, eps=None):
        if eps is None:
            eps = torch.randn(x_0.shape).to(self.device)

        alpha_prod = self.alpha_prods[t]

        if len(alpha_prod.shape) > 0:
            alpha_prod = alpha_prod.reshape(x_0.shape[0], 1, 1, 1)

        # using closed form to compute x_t using x_0 and noise
        return alpha_prod.sqrt() * x_0 + (1 - alpha_prod).sqrt() * eps

    def backward(self, x, t):
        return self.model(x, t)