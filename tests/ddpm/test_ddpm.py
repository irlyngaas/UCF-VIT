"""Tests for diffusion_reverse_step, the reverse-diffusion step used by
training_scripts/inference_diffusion.py to sample from a trained DiffusionVIT
checkpoint.

No heavy deps (timm/monai/xformers) -- unlike tests/model/test_diffusion_vit.py,
this only needs torch, so it runs unconditionally.
"""

import pytest
import torch

from UCF_VIT.ddpm.ddpm import DDPM_Scheduler, diffusion_reverse_step


@pytest.fixture
def scheduler():
    return DDPM_Scheduler(num_time_steps=100)


def test_unknown_variance_mode_raises(scheduler):
    x_t = torch.randn(2, 1, 4, 4)
    with pytest.raises(ValueError, match="variance mode"):
        diffusion_reverse_step(x_t, torch.randn_like(x_t), 50, scheduler, variance_mode="bogus")


def test_stochastic_modes_require_noise_above_timestep_zero(scheduler):
    x_t = torch.randn(2, 1, 4, 4)
    for mode in ("current", "posterior"):
        with pytest.raises(ValueError, match="noise"):
            diffusion_reverse_step(x_t, torch.randn_like(x_t), 50, scheduler, variance_mode=mode, noise=None)


def test_stochastic_modes_do_not_require_noise_at_timestep_zero(scheduler):
    # timestep==0 returns the mean directly, with no injected noise term --
    # must not raise even though noise=None.
    x_t = torch.randn(2, 1, 4, 4)
    for mode in ("current", "posterior"):
        out = diffusion_reverse_step(x_t, torch.randn_like(x_t), 0, scheduler, variance_mode=mode, noise=None)
        assert torch.isfinite(out).all()


def test_ddim_never_requires_noise(scheduler):
    x_t = torch.randn(2, 1, 4, 4)
    for timestep in (0, 50, 99):
        out = diffusion_reverse_step(x_t, torch.randn_like(x_t), timestep, scheduler, variance_mode="ddim", noise=None)
        assert torch.isfinite(out).all()
        assert out.shape == x_t.shape


def test_ddim_is_deterministic(scheduler):
    x_t = torch.randn(2, 1, 4, 4)
    predicted_noise = torch.randn_like(x_t)
    out1 = diffusion_reverse_step(x_t, predicted_noise, 50, scheduler, variance_mode="ddim")
    out2 = diffusion_reverse_step(x_t, predicted_noise, 50, scheduler, variance_mode="ddim")
    torch.testing.assert_close(out1, out2)


def test_ddim_inverts_the_forward_noising_formula_exactly(scheduler):
    """x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*e (UCF_VIT.training's
    DiffusionVIT forward_step formula). Feeding the real e back in as
    predicted_noise, a "ddim" step at the SAME timestep must recover x0
    exactly (it is an exact algebraic inverse, not an approximation).
    """
    timestep = 50
    x0 = torch.randn(2, 1, 4, 4)
    e = torch.randn(2, 1, 4, 4)
    alpha_bar_t = scheduler.alpha[timestep]
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * e

    x0_recovered = diffusion_reverse_step(x_t, e, timestep, scheduler, variance_mode="ddim")
    # timestep > 0: ddim's branch re-noises to alpha_bar_prev, so this is the
    # one-step-earlier noisy sample, not raw x0 -- check via the same forward
    # formula at timestep-1 instead of expecting x0 directly.
    alpha_bar_prev = scheduler.alpha[timestep - 1]
    expected = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1 - alpha_bar_prev) * e
    torch.testing.assert_close(x0_recovered, expected, atol=1e-5, rtol=1e-5)


def test_ddim_at_timestep_zero_recovers_x0_exactly(scheduler):
    x0 = torch.randn(2, 1, 4, 4)
    e = torch.randn(2, 1, 4, 4)
    alpha_bar_0 = scheduler.alpha[0]
    x_0 = torch.sqrt(alpha_bar_0) * x0 + torch.sqrt(1 - alpha_bar_0) * e

    x0_recovered = diffusion_reverse_step(x_0, e, 0, scheduler, variance_mode="ddim")
    torch.testing.assert_close(x0_recovered, x0, atol=1e-4, rtol=1e-4)


def test_current_and_posterior_variance_differ(scheduler):
    # Same mean, different variance -- same noise tensor must produce
    # different outputs (otherwise one of the two branches is dead code).
    x_t = torch.randn(2, 1, 4, 4)
    predicted_noise = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)
    out_current = diffusion_reverse_step(x_t, predicted_noise, 50, scheduler, variance_mode="current", noise=noise)
    out_posterior = diffusion_reverse_step(x_t, predicted_noise, 50, scheduler, variance_mode="posterior", noise=noise)
    assert not torch.equal(out_current, out_posterior)


def test_output_shape_and_dtype_match_input(scheduler):
    x_t = torch.randn(3, 1, 4, 4, 4, dtype=torch.float64)
    predicted_noise = torch.randn_like(x_t)
    noise = torch.randn_like(x_t)
    out = diffusion_reverse_step(x_t, predicted_noise, 10, scheduler, variance_mode="current", noise=noise)
    assert out.shape == x_t.shape
    assert out.dtype == x_t.dtype
