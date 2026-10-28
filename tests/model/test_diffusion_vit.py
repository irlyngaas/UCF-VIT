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


# ---------------------------------------------------------------------------
# Opt-in decoder refinements (residual_mlp/conv3d/feature_conv3d_decoder,
# spatial_feature_conv3d/conv2d_decoder) -- ported from the colleague's
# AKZ-upd4 branch (src/UCF_VIT/fsdp/arch.py there), adapted onto this
# repo's current DiffusionVIT. Real correctness property each residual
# option must have: an untrained model with the option on is *exactly*
# (not approximately) equivalent to the plain linear decoder, since the
# new branch's final layer is zero-initialized -- verified here the same
# way the original branch's own tests did, bit-for-bit (atol=0, rtol=0).
# ---------------------------------------------------------------------------

_COMMON_KWARGS_3D = dict(
    img_size=(8, 8, 8), patch_size=4, twoD=False, in_chans=1,
    embed_dim=12, depth=1, num_heads=1, mlp_ratio=1.0,  # embed_dim must be divisible by 3 for get_3d_sincos_pos_embed
    fixed_length=8, adaptive_patching=False,
    weight_init="skip", class_token=False, pos_embed="learn",
    block_fn=Block_diffusion,
)


def _make_diffusion_vit_3d(**overrides):
    kwargs = dict(_COMMON_KWARGS_3D, num_time_steps=100, linear_decoder=True)
    kwargs.update(overrides)
    return DiffusionVIT(**kwargs)


def test_legacy_linear_model_has_no_residual_parameters():
    model = _make_diffusion_vit_3d()
    assert not any(name.startswith("decoder_residual") for name in model.state_dict())
    assert hasattr(model, "decoder_pred")


def test_residual_mlp_decoder_step_zero_matches_legacy():
    legacy = _make_diffusion_vit_3d()
    refined = _make_diffusion_vit_3d(residual_mlp_decoder=True)
    incompatible = refined.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(k.startswith("decoder_residual.") for k in incompatible.missing_keys)

    features = torch.randn(2, legacy.num_patches, legacy.embed_dim)
    torch.testing.assert_close(refined.forward_head(features, torch.zeros(2, dtype=torch.long)),
                                legacy.forward_head(features, torch.zeros(2, dtype=torch.long)),
                                atol=0, rtol=0)


def test_residual_conv3d_decoder_step_zero_matches_legacy_and_round_trips():
    legacy = _make_diffusion_vit_3d()
    refined = _make_diffusion_vit_3d(residual_conv3d_decoder=True)
    incompatible = refined.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(k.startswith("decoder_residual_conv3d.") for k in incompatible.missing_keys)

    features = torch.randn(2, legacy.num_patches, legacy.embed_dim)
    t = torch.zeros(2, dtype=torch.long)
    torch.testing.assert_close(refined.forward_head(features, t), legacy.forward_head(features, t), atol=0, rtol=0)

    # _tokens_to_3d_volume/_volume_to_3d_tokens must round-trip exactly
    # (this is what lets the residual conv operate on real spatial
    # structure without corrupting the underlying patch layout).
    tokens = torch.randn(2, refined.num_patches, refined.patch_dim)
    volume = refined._tokens_to_3d_volume(tokens)
    assert volume.shape == (2, 1, 8, 8, 8)
    torch.testing.assert_close(refined._volume_to_3d_tokens(volume), tokens)


def test_residual_conv3d_decoder_requires_3d():
    with pytest.raises(ValueError, match="requires a 3D model"):
        _make_diffusion_vit_3d(residual_conv3d_decoder=True, twoD=True, img_size=(8, 8))


def test_residual_feature_conv3d_decoder_step_zero_matches_legacy():
    legacy = _make_diffusion_vit_3d()
    refined = _make_diffusion_vit_3d(residual_feature_conv3d_decoder=True)
    incompatible = refined.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert all(k.startswith("decoder_residual_feature_") for k in incompatible.missing_keys)

    features = torch.randn(2, legacy.num_patches, legacy.embed_dim)
    t = torch.zeros(2, dtype=torch.long)
    torch.testing.assert_close(refined.forward_head(features, t), legacy.forward_head(features, t), atol=0, rtol=0)


def test_residual_feature_conv3d_decoder_receives_gradient():
    model = _make_diffusion_vit_3d(residual_feature_conv3d_decoder=True)
    with torch.no_grad():
        model.decoder_residual_feature_conv3d[-1].weight.normal_(std=1e-3)
    features = torch.randn(2, model.num_patches, model.embed_dim, requires_grad=True)
    target = torch.randn(2, model.num_patches, model.patch_dim)
    t = torch.zeros(2, dtype=torch.long)
    torch.nn.functional.mse_loss(model.forward_head(features, t), target).backward()

    projection = model.decoder_residual_feature_proj[1]
    output_conv = model.decoder_residual_feature_conv3d[-1]
    assert projection.weight.grad.abs().sum().item() > 0
    assert output_conv.weight.grad.abs().sum().item() > 0


def test_spatial_feature_conv3d_decoder_replaces_decoder_pred():
    model = _make_diffusion_vit_3d(spatial_feature_conv3d_decoder=True)
    assert not hasattr(model, "decoder_pred")

    features = torch.randn(2, model.num_patches, model.embed_dim, requires_grad=True)
    target = torch.randn(2, model.num_patches, model.patch_dim)
    t = torch.zeros(2, dtype=torch.long)
    prediction = model.forward_head(features, t)
    assert prediction.shape == target.shape

    torch.nn.functional.mse_loss(prediction, target).backward()
    assert model.decoder_spatial_feature_proj[1].weight.grad.abs().sum().item() > 0
    assert model.decoder_spatial_feature_conv3d[-1].weight.grad.abs().sum().item() > 0


def test_spatial_feature_conv2d_decoder_layout_and_gradients():
    model = _make_diffusion_vit(spatial_feature_conv2d_decoder=True, spatial_feature_conv2d_channels=3)
    assert not hasattr(model, "decoder_pred")

    tokens = torch.randn(2, model.num_patches, 4 ** 2 * 3)
    image = model._tokens_to_2d_image(tokens, num_channels=3)
    assert image.shape == (2, 3, 16, 16)
    torch.testing.assert_close(model._image_to_2d_tokens(image), tokens)

    features = torch.randn(2, model.num_patches, model.embed_dim, requires_grad=True)
    target = torch.randn(2, model.num_patches, model.patch_dim)
    prediction = model.forward_head(features, torch.zeros(2, dtype=torch.long))
    assert prediction.shape == target.shape

    torch.nn.functional.mse_loss(prediction, target).backward()
    assert model.decoder_spatial_feature_proj[1].weight.grad.abs().sum().item() > 0
    assert model.decoder_spatial_feature_conv2d[-1].weight.grad.abs().sum().item() > 0


def test_spatial_feature_conv2d_decoder_rejects_3d_model():
    with pytest.raises(ValueError, match="requires a 2D model"):
        _make_diffusion_vit_3d(spatial_feature_conv2d_decoder=True)


def test_spatial_decoder_is_mutually_exclusive_with_residual():
    with pytest.raises(ValueError, match="cannot be combined"):
        _make_diffusion_vit_3d(spatial_feature_conv3d_decoder=True, residual_conv3d_decoder=True)


def test_residual_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one residual decoder"):
        _make_diffusion_vit_3d(residual_mlp_decoder=True, residual_conv3d_decoder=True)
    with pytest.raises(ValueError, match="only one residual decoder"):
        _make_diffusion_vit_3d(residual_conv3d_decoder=True, residual_feature_conv3d_decoder=True)


def test_spatial_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one direct spatial decoder"):
        _make_diffusion_vit_3d(spatial_feature_conv3d_decoder=True, spatial_feature_conv2d_decoder=True, twoD=True, img_size=(8, 8))


def test_all_decoder_refinements_require_linear_decoder():
    with pytest.raises(ValueError, match="requires linear_decoder=True"):
        _make_diffusion_vit_3d(
            spatial_feature_conv3d_decoder=True, linear_decoder=False,
            decoder_depth=1, decoder_embed_dim=8, decoder_num_heads=1, decoder_mlp_ratio=1.0,
        )


# ---------------------------------------------------------------------------
# Opt-in per-block modality conditioning (use_modality_conditioning), for a
# paired multi-modality dataset (e.g. "nct"/"xct") -- ported from the
# colleague's AKZ-upd4 branch (its Block.forward's cond_map/cond_emb,
# DiffusionVIT's own var_map). Unlike the decoder refinements above, this
# has no step-zero-equivalence property in the reference it was ported from
# (cond_map/condEmbeddingMap are freshly, randomly initialized, not
# zero-init) -- the real correctness property to check is that it's wired
# end to end: different modalities must produce different output, the same
# way test_diffusion_vit_different_timesteps_change_the_output already
# checks for t.
# ---------------------------------------------------------------------------


def test_modality_conditioning_off_by_default_adds_no_parameters():
    model = _make_diffusion_vit()
    assert not any(name.startswith(("cond_map", "condEmbeddingMap")) for name, _ in model.blocks[0].named_parameters())
    assert model.blocks[0].num_modalities is None


def test_modality_conditioning_requires_at_least_two_default_vars():
    with pytest.raises(ValueError, match="at least 2 modalities"):
        _make_diffusion_vit(use_modality_conditioning=True, default_vars=["nct"])
    with pytest.raises(ValueError, match="at least 2 modalities"):
        _make_diffusion_vit(use_modality_conditioning=True)


def test_modality_conditioning_builds_per_block_sublayers_and_var_map():
    model = _make_diffusion_vit(use_modality_conditioning=True, default_vars=["nct", "xct"])
    assert model.var_map == {"nct": 0, "xct": 1}
    for blk in model.blocks:
        assert blk.num_modalities == 2
        assert hasattr(blk, "cond_map") and hasattr(blk, "condEmbeddingMap")


def test_different_modalities_change_the_output():
    """The actual point of modality conditioning: "nct" and "xct" batches
    must produce genuinely different output, not just be plumbed through
    and ignored -- mirrors test_diffusion_vit_different_timesteps_change_
    the_output's own check for t.
    """
    model = _make_diffusion_vit(use_modality_conditioning=True, default_vars=["nct", "xct"])
    model.eval()
    x = torch.randn(2, 1, 16, 16)
    t = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        out_nct = model(x, t, ["nct"])
        out_xct = model(x, t, ["xct"])

    assert not torch.equal(out_nct, out_xct)


def test_modality_conditioning_receives_gradient():
    model = _make_diffusion_vit(use_modality_conditioning=True, default_vars=["nct", "xct"])
    x = torch.randn(2, 1, 16, 16, requires_grad=True)
    t = torch.zeros(2, dtype=torch.long)

    output = model(x, t, ["xct"])
    output.sum().backward()

    blk = model.blocks[0]
    assert blk.cond_map.weight.grad is not None
    assert blk.cond_map.weight.grad.abs().sum().item() > 0


def test_block_diffusion_rejects_mismatched_modality_id():
    plain_block = _make_diffusion_vit().blocks[0]
    with pytest.raises(ValueError, match="if and only if"):
        plain_block(torch.randn(2, 4, 8), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))

    modality_block = _make_diffusion_vit(use_modality_conditioning=True, default_vars=["nct", "xct"]).blocks[0]
    with pytest.raises(ValueError, match="if and only if"):
        modality_block(torch.randn(2, 4, 8), torch.zeros(2, dtype=torch.long))
