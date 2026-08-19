import torch
import torch.nn as nn

class DDPM_Scheduler(nn.Module):
    """Linear beta-schedule noise scheduler for denoising diffusion probabilistic models.

    Precomputes the beta (noise variance) and alpha (cumulative product of `1 - beta`)
    schedules used to add and remove noise at each diffusion timestep.
    """

    def __init__(self, num_time_steps: int=1000):
        """Initializes the linear beta schedule and its cumulative alpha product.

        Args:
            num_time_steps: Number of diffusion timesteps to precompute the schedule
                for.
        """
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
        self.num_time_steps = num_time_steps

    def forward(self, t):
        """Looks up the beta and cumulative alpha values for a batch of timesteps.

        Args:
            t: Tensor of timestep indices to look up.

        Returns:
            A tuple `(beta, alpha)` of tensors containing `self.beta[t]` and
            `self.alpha[t]`.
        """
        return self.beta[t], self.alpha[t]
