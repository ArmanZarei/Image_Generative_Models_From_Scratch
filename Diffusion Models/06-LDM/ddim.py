import torch


class DDIM:
    def __init__(self, T=1000, beta_1=1e-4, beta_T=2e-2, device='cpu'):
        self.device = device
        self.T = T

        betas = torch.linspace(beta_1, beta_T, steps=T, dtype=torch.double).to(device)
        betas = torch.cat([torch.zeros(1).to(device), betas], dim=0)
        self.shifted_alpha_bars = (1 - betas).cumprod(dim=0)

    def get_alpha_bars_t(self, t):
        return self.shifted_alpha_bars[t+1][:, None, None, None]
    
    def sample(self, x, model, sequence, eta=0):
        x_t = x
        for t, t_prev in zip(reversed(sequence), reversed([-1] + list(sequence[:-1]))):
            t, t_prev = [(torch.ones((x.shape[0], )) * val).long().to(self.device) for val in [t, t_prev]]

            abar_t, abar_tprev = self.get_alpha_bars_t(t), self.get_alpha_bars_t(t_prev)

            eps = model(x_t, t)
            sigma = eta * torch.sqrt((1 - abar_tprev) / (1 - abar_t)) * torch.sqrt(1 - abar_t/abar_tprev)

            # Blue Equation
            x0_pred = (x_t - torch.sqrt(1 - abar_t) * eps) / torch.sqrt(abar_t)
            first_term = torch.sqrt(abar_tprev) * x0_pred

            # Green Equation
            second_term = torch.sqrt(1 - abar_tprev - sigma**2) * eps

            # Pink Equation
            third_term = sigma * torch.randn_like(x)

            x_t = (first_term + second_term + third_term).float().detach()
        
        return x_t

            
    def sample_using_uniform_skip(self, x, model, num_steps=100, eta=0):
        sequence = list(range(0, self.T, self.T//num_steps))
        return self.sample(x, model, sequence, eta)
