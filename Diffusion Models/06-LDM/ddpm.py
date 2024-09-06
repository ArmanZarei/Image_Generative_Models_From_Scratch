import torch
from torch import nn


class DDPM:
    def __init__(self, T=1000, beta_1=1e-4, beta_T=2e-2, device='cpu', posterior_var_type='fixedlarge'):
        self.T = T
        
        self.betas = torch.linspace(beta_1, beta_T, steps=T, dtype=torch.double).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.prev_alpha_bars = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0).clone().to(device)

        # Calculation of q(x_t | x_0)  ->  [Yellow Equation]
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).float()
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars).float()

        # Calculation of x_0 from x_t and epsilon  ->  [Green Equation]
        self.sqrt_recip_alpha_bars = torch.sqrt(1. / self.alpha_bars).float()
        self.sqrt_recip_alpha_bars_minus_one = torch.sqrt(1. / self.alpha_bars - 1).float()

        # Calculation of q(x_{t-1} | x_t , x_0)  ->  [Red Equation]
        self.posterior_mean_coef_x0 = (self.betas * torch.sqrt(self.prev_alpha_bars) / (1 - self.alpha_bars)).float()
        self.posterior_mean_coef_xt = (torch.sqrt(self.alphas) * (1 - self.prev_alpha_bars) / (1 - self.alpha_bars)).float()
        if posterior_var_type == 'fixedlarge':
            self.posterior_log_variance = torch.log(self.betas).float()
        elif posterior_var_type == 'fixedsmall':
            posterior_variance = (((1 - self.prev_alpha_bars) / (1 - self.alpha_bars)) * self.betas).float()
            self.posterior_log_variance = torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20))).float() # This was done in the original TF repo.
        else:
            raise Exception("Unknown posterior_var_type argument")


    # Calculation of q(x_t | x_0)  ->  [Yellow Equation]
    def calculate_xt_from_x0(self, x_0, t, eps=None):
        t = self._fix_t_view(t)

        if eps is None:
            eps = torch.randn_like(x_0)

        x_t = self.sqrt_alpha_bars[t] * x_0 + self.sqrt_one_minus_alpha_bars[t] * eps

        return x_t, eps

    # Calculation of x_0 from x_t and epsilon  ->  [Green Equation]
    def calculate_x0_from_xt_and_eps(self, x_t, t, eps):
        t = self._fix_t_view(t)

        x_0 = self.sqrt_recip_alpha_bars[t] * x_t - self.sqrt_recip_alpha_bars_minus_one[t] * eps

        return x_0
    
    # Calculation of q(x_{t-1} | x_t , x_0)  ->  [Red Equation]
    def calculate_xprev_from_xt_and_x0(self, x_t, t, x_0):
        t = self._fix_t_view(t)
        
        mean = self.posterior_mean_coef_x0[t] * x_0 + self.posterior_mean_coef_xt[t] * x_t
        std = torch.exp(0.5 * self.posterior_log_variance[t])
        std[t == 0] = 0

        xprev = mean + std * torch.randn_like(x_t)

        return xprev
    
    def _fix_t_view(self, t):
        assert t.ndim == 1 and torch.all(t >= 0) and torch.all(t < self.T)
        return t[:, None, None, None]
    
    def reverse_process_from_xt_using_model(self, model: nn.Module, x_t, t_end: int):
        out = x_t

        for t in range(t_end, -1, -1):
            t_tensor = torch.full((x_t.shape[0], ), t).to(x_t.device)
            eps = model(out, t_tensor)
            x_0_pred = self.calculate_x0_from_xt_and_eps(out, t_tensor, eps)
            out = self.calculate_xprev_from_xt_and_x0(out, t_tensor, x_0_pred).detach()

        return out