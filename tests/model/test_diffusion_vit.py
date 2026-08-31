"""Tests for Block_diffusion, DiffusionVIT's per-block timestep conditioning.

Block_diffusion (UCF_VIT.model.building_blocks) adds a per-block sinusoidal
timestep embedding (SinusoidalEmbeddings + a learned EmbeddingDenseLayer
projection), re-injected into the token sequence before every block's
attention -- rather than DiffusionVIT's earlier design, which added a single
timestep embedding once, before the first block, and let it propagate
implicitly through the residual stream. get_model (model/utils.py) passes
block_fn=Block_diffusion only for conf["model"]["type"] == "DiffusionVIT" --
every other model type still uses the plain Block.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see tests/distributed/test_tensor_parallel_correctness.py's
module docstring for why this skips cleanly via importorskip instead of
erroring at collection.
"""

import pytest

arch_mod = pytest.importorskip(
    "UCF_VIT.model.arch",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)
building_blocks_mod = pytest.importorskip(
    "UCF_VIT.model.building_blocks",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)
DiffusionVIT = arch_mod.DiffusionVIT
Block_diffusion = building_blocks_mod.Block_diffusion

import torch

_COMMON_KWARGS = dict(
    img_size=(16, 16), patch_size=4, twoD=True, in_chans=1,
    embed_dim=8, depth=2, num_heads=1, mlp_ratio=1.0,
    fixed_length=16, adaptive_patching=False,
    weight_init="skip", class_token=False, pos_embed="learn",
    block_fn=Block_diffusion,
)


def _make_diffusion_vit(**overrides):
    kwargs = dict(_COMMON_KWARGS, num_time_steps=100, linear_decoder=True)
    kwargs.update(overrides)
    return DiffusionVIT(**kwargs)


def test_diffusion_vit_uses_block_diffusion_instances():
    model = _make_diffusion_vit()
    assert len(model.blocks) > 0
    assert all(isinstance(blk, Block_diffusion) for blk in model.blocks)


def test_diffusion_vit_forward_backward_linear_decoder():
    model = _make_diffusion_vit()
    x = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 100, (2,))

    output = model(x, t, ["v0"])
    output.sum().backward()

    assert output.shape == (2, model.num_patches, model.patch_dim)
    assert torch.isfinite(output).all()


def test_diffusion_vit_forward_backward_transformer_decoder():
    # linear_decoder=False: also exercises decoder_blocks, which get the
    # same block_fn/num_time_steps threading as the encoder's self.blocks.
    model = _make_diffusion_vit(
        linear_decoder=False, decoder_depth=2, decoder_embed_dim=8,
        decoder_num_heads=1, decoder_mlp_ratio=1.0,
    )
    assert all(isinstance(blk, Block_diffusion) for blk in model.decoder_blocks)

    x = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 100, (2,))

    output = model(x, t, ["v0"])
    output.sum().backward()

    assert torch.isfinite(output).all()


def test_diffusion_vit_different_timesteps_change_the_output():
    """The actual point of threading t through every block: different
    timesteps must produce genuinely different outputs, not just be
    plumbed through and ignored.
    """
    model = _make_diffusion_vit()
    model.eval()
    x = torch.randn(2, 1, 16, 16)

    with torch.no_grad():
        out_t0 = model(x, torch.zeros(2, dtype=torch.long), ["v0"])
        out_t_last = model(x, torch.full((2,), 99, dtype=torch.long), ["v0"])

    assert not torch.equal(out_t0, out_t_last)
