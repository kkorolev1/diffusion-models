import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DiffusionTrainer(nn.Module):
    def __init__(self, model, T=1000, beta_1=1e-4, beta_T=0.02):
        super().__init__()

        self.model = model
        self.T = T

        # linear variance scheduling
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        
        alphas = 1. - self.betas
        alpha_prods = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alpha_prods', torch.sqrt(alpha_prods))
        self.register_buffer('sqrt_one_minus_alpha_prods', torch.sqrt(1. - alpha_prods))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        eps = torch.randn_like(x_0)

        # using closed form to compute x_t using x_0 and noise
        x_t = extract(self.sqrt_alpha_prods, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alpha_prods, t, x_0.shape) * eps

        return F.mse_loss(self.model(x_t, t), eps, reduction='none')

class DiffusionSampler(nn.Module):
    def __init__(self, model, T=1000, beta_1=1e-4, beta_T=0.02, img_size=32):
        super().__init__()

        self.model = model
        self.T = T

        # linear variance scheduling
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        
        alphas = 1. - self.betas
        alpha_prods = torch.cumprod(alphas, dim=0)

        alpha_prods_prev = F.pad(alpha_prods, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alpha_prods))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alpha_prods - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alpha_prods_prev) / (1. - alpha_prods))

        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))

        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alpha_prods_prev) * self.betas / (1. - alpha_prods))

        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alpha_prods_prev) / (1. - alpha_prods))

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def q_mean_variance(self, x_0, x_t, t):
        assert x_0.shape == x_t.shape

        # mean of posterior q(x_{t-1} | x_t, x_0)
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + \
                         extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

        # log var of posterior q(x_{t-1} | x_t, x_0)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def p_mean_variance(self, x_t, t):
        # posterior log_var
        model_log_var = torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]]))
        model_log_var = extract(model_log_var, t, x_t.shape)

        # predict noise
        eps = self.model(x_t, t)
        x_0 = self.predict_xstart_from_eps(x_t, t, eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)

        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T):
        x_t = x_T

        for timestamp in reversed(range(self.T)):
            t = torch.ones((x_T.shape[0], ), dtype=torch.long) * timestamp
            mean, log_var = self.p_mean_variance(x_t, t)
            eps = torch.randn_like(x_t) if timestamp > 0 else 0
            x_t = mean + torch.exp(0.5 * log_var) * eps
        
        return torch.clip(x_t, -1, 1)
