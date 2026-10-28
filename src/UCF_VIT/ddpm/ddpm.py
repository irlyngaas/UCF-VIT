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


def diffusion_reverse_step(x_t, predicted_noise, timestep, scheduler, variance_mode="current", noise=None):
    """Applies one epsilon-prediction DDPM/DDIM reverse step.

    `scheduler.alpha[timestep]` is the cumulative product (alpha_bar_t), matching
    the forward-noising convention used during training (see UCF_VIT.training's
    DiffusionVIT forward_step: `data_noisy = sqrt(alpha_bar_t)*data_clean +
    sqrt(1-alpha_bar_t)*e`).

    Args:
        x_t: Current noisy sample at `timestep`.
        predicted_noise: The model's epsilon prediction for `x_t` at `timestep`,
            same shape as `x_t`.
        timestep: The current timestep (int, or a 0-d/1-element tensor).
        scheduler: A `DDPM_Scheduler` instance, read for its `.beta`/`.alpha`.
        variance_mode: One of:
            - `"current"`: ancestral DDPM sampling using the raw forward
              variance `beta_t` (matches the original DDPM paper's simplest
              choice).
            - `"posterior"`: ancestral DDPM sampling using the tighter
              posterior variance `beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)`.
            - `"ddim"`: deterministic DDIM step (no injected noise); predicts
              x0 from `x_t`/`predicted_noise`, then re-noises it to
              `alpha_bar_{t-1}`.
        noise: Fresh Gaussian noise, same shape as `x_t`, required when
            `variance_mode` is `"current"`/`"posterior"` and `timestep > 0`
            (unused for `"ddim"` or when `timestep == 0`).

    Returns:
        The sample one step closer to clean data (`x_{t-1}`, or `x0` when
        `timestep == 0`).

    Raises:
        ValueError: If `variance_mode` is not one of the three above, or if
            `noise` is required but not given.
    """
    timestep = int(timestep)
    if variance_mode not in ("current", "posterior", "ddim"):
        raise ValueError(f"Unknown variance mode: {variance_mode}")

    dtype = x_t.dtype
    device = x_t.device
    beta_t = scheduler.beta[timestep].to(device=device, dtype=dtype)
    alpha_bar_t = scheduler.alpha[timestep].to(device=device, dtype=dtype)
    alpha_t = 1.0 - beta_t
    alpha_bar_prev = (
        scheduler.alpha[timestep - 1].to(device=device, dtype=dtype)
        if timestep > 0 else torch.ones((), device=device, dtype=dtype)
    )
    eps = torch.finfo(dtype).eps

    if variance_mode == "ddim":
        x0_prediction = (
            x_t - torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=eps))
            * predicted_noise
        ) / torch.sqrt(torch.clamp(alpha_bar_t, min=eps))
        if timestep == 0:
            return x0_prediction
        return (
            torch.sqrt(alpha_bar_prev) * x0_prediction
            + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=eps))
            * predicted_noise
        )

    mean = (
        x_t
        - beta_t
        / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=eps))
        * predicted_noise
    ) / torch.sqrt(torch.clamp(alpha_t, min=eps))
    if timestep == 0:
        return mean
    if noise is None:
        raise ValueError("A noise tensor is required for stochastic DDPM")

    if variance_mode == "current":
        variance = beta_t
    else:
        variance = (
            beta_t
            * (1.0 - alpha_bar_prev)
            / torch.clamp(1.0 - alpha_bar_t, min=eps)
        )
    return mean + torch.sqrt(torch.clamp(variance, min=0.0)) * noise
