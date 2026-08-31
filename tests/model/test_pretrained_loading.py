"""Tests for pretrained-checkpoint loading at a different h:w[:d] resolution.

Covers three things introduced/fixed together:

1. `UCF_VIT.utils.pos_embed.interpolate_pos_embed`/`interpolate_pos_embed_3d` --
   rewritten to take the original grid shape as an explicit argument instead of
   guessing it from a hardcoded 2:1 width:height ratio (the old code was wrong
   for every shipped config, all of which are square). Generalizes to any
   independent height:width[:depth] ratio, for both the pretrained and the new
   grid.
2. `UCF_VIT.model.utils.extract_encoder_state_dict` -- an allowlist-based
   generalization of the encoder-extraction step that previously only worked
   for `model_type == "MAE"` (every other pretrained source silently
   transplanted nothing at all).
3. `UCF_VIT.model.utils._transplant_pos_embed` -- resizes the extracted
   `pos_embed` entry to the new model's own shape, dispatching between the 2D/3D
   grid case and the flat (adaptive, non-sqrt_len_method) case, and rejecting
   `sqrt_len_method:True` (`SAP`/`UNETR+do_ap`) explicitly rather than silently
   producing a wrong-shaped result (see this session's own investigation notes
   for why that regime's `grid_size` doesn't reflect its real token count).

Note: the flat (adaptive, non-sqrt_len_method) case used to interpolate
`pos_embed` via 1D linear interpolation along the sequence-slot-index axis
(`UCF_VIT.utils.misc.interpolate_pos_embed_adaptive`) instead of the grid
case's spatial resize. Reviewing that function found two real problems: no
principled notion of "adjacent slots" exists for adaptive patching at all
(`FixedQuadTree`/`FixedOctTree`'s node order reflects greedy-split order,
not spatial position), so the interpolation rested on an assumption the
data doesn't satisfy; and, independently, it never sliced out
`num_prefix_tokens` first, so a class-token row got blended into the
interpolation whenever `class_token:True`. Replaced with simply dropping
the pretrained `pos_embed` on a `fixed_length` mismatch (the new model
keeps its own fresh init) -- `interpolate_pos_embed_adaptive` was deleted,
archived at
`../UCF-VIT-claude-archive/src/UCF_VIT/utils/misc.py`.

Note: `get_model` needs a conversion step between `conf["data"]["tile_size"]`
and `PatchEmbed`'s `img_size` argument -- originally `_patch_embed_img_size`,
which handled `tile_size` being `[width, height]` for `imagenet`/`catsdogs`
while `PatchEmbed` expects `(H, W)` (caught by a real non-square `catsdogs`
resize on Frontier, job 5348773). The config/data-loading layer was later
changed to store `img_size`/`resize`/`tile_size` as `[height, width]`
throughout, which made the width/height-swap half of `_patch_embed_img_size`
a no-op -- but its other job, truncating `basic_ct`'s 3-tuple `tile_size`
(`(H_tile, W_tile, Z_native)`, kept 3-wide even when `twoD` is True so
`TileDataIter`'s dispatch logic still works) down to a genuine 2-tuple
whenever `twoD` is True, was *not* a no-op, and got dropped along with the
whole function. That regressed every `basic_ct`+`twoD` UNETR run (its
decoder's `nn.Upsample(size=self.img_size, ...)` needs an exact-rank size
match) -- caught by a real Frontier smoke run (job 5388433):
`ValueError: Input and output must have the same number of spatial
dimensions, but got input with spatial dimensions of [128, 128] and output
size of (256, 256, 256)`. Fixed by reinstating the truncation as
`_model_img_size(tile_size, twoD)` (dataset-name argument dropped -- the
2-vs-3 truncation only depends on `twoD` now, not which dataset it is).

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports, transitively pulled in by model/utils.py itself) -- see
tests/distributed/test_tensor_parallel_correctness.py's module docstring for
why this skips cleanly via importorskip instead of erroring at collection.
"""

import pytest

pos_embed_mod = pytest.importorskip(
    "UCF_VIT.utils.pos_embed",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)
model_utils_mod = pytest.importorskip(
    "UCF_VIT.model.utils",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)
arch_mod = pytest.importorskip(
    "UCF_VIT.model.arch",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)

interpolate_pos_embed = pos_embed_mod.interpolate_pos_embed
interpolate_pos_embed_3d = pos_embed_mod.interpolate_pos_embed_3d
extract_encoder_state_dict = model_utils_mod.extract_encoder_state_dict
_transplant_pos_embed = model_utils_mod._transplant_pos_embed
_prune_incompatible_cls_token = model_utils_mod._prune_incompatible_cls_token
_model_img_size = model_utils_mod._model_img_size
VIT = arch_mod.VIT
MAE = arch_mod.MAE
SAP = arch_mod.SAP
UNETR = arch_mod.UNETR
DiffusionVIT = arch_mod.DiffusionVIT

import torch


# ---------------------------------------------------------------------------
# _model_img_size
# ---------------------------------------------------------------------------


def test_model_img_size_2d_tuple_passes_through_unchanged():
    # imagenet/catsdogs: tile_size is always already a genuine 2-tuple.
    assert _model_img_size((128, 64), twoD=True) == (128, 64)


def test_model_img_size_basic_ct_twod_truncates_stale_z_depth():
    # basic_ct+twoD: tile_size keeps a raw, undivided z-depth as its 3rd
    # entry (parse.py's own dataloader-dispatch requirement) -- the model's
    # img_size must not include it, or decoder heads that use it as an exact
    # target size (UNETR's nn.Upsample) crash on a spatial-dimension-count
    # mismatch. Regression test for job 5388433.
    assert _model_img_size((128, 128, 256), twoD=True) == (128, 128)


def test_model_img_size_basic_ct_3d_keeps_all_three_dims():
    assert _model_img_size((64, 64, 64), twoD=False) == (64, 64, 64)


# ---------------------------------------------------------------------------
# interpolate_pos_embed / interpolate_pos_embed_3d
# ---------------------------------------------------------------------------


def test_interpolate_pos_embed_2d_non_square_independent_ratio():
    embed_dim = 4
    orig_grid = (8, 16)
    new_grid = (16, 8)  # swapped ratio, not just a uniform rescale
    pos_embed = torch.randn(1, orig_grid[0] * orig_grid[1], embed_dim)

    resized = interpolate_pos_embed(pos_embed, orig_grid, new_grid)

    assert resized.shape == (1, new_grid[0] * new_grid[1], embed_dim)


def test_interpolate_pos_embed_2d_another_independent_ratio():
    embed_dim = 4
    orig_grid = (4, 8)
    new_grid = (8, 4)
    pos_embed = torch.randn(1, orig_grid[0] * orig_grid[1], embed_dim)

    resized = interpolate_pos_embed(pos_embed, orig_grid, new_grid)

    assert resized.shape == (1, new_grid[0] * new_grid[1], embed_dim)


def test_interpolate_pos_embed_2d_same_size_is_a_true_noop():
    embed_dim = 4
    grid = (5, 7)
    pos_embed = torch.randn(1, grid[0] * grid[1], embed_dim)

    resized = interpolate_pos_embed(pos_embed, grid, grid)

    assert resized is pos_embed


def test_interpolate_pos_embed_2d_preserves_class_token_prefix():
    embed_dim = 4
    orig_grid = (4, 8)
    new_grid = (8, 4)
    cls_row = torch.full((1, 1, embed_dim), 99.0)
    grid_tokens = torch.randn(1, orig_grid[0] * orig_grid[1], embed_dim)
    pos_embed = torch.cat([cls_row, grid_tokens], dim=1)

    resized = interpolate_pos_embed(pos_embed, orig_grid, new_grid, num_prefix_tokens=1)

    assert resized.shape == (1, 1 + new_grid[0] * new_grid[1], embed_dim)
    assert torch.equal(resized[:, :1], cls_row)


def test_interpolate_pos_embed_3d_non_cubic_independent_ratio():
    embed_dim = 4
    orig_grid = (4, 8, 16)
    new_grid = (8, 4, 4)
    pos_embed = torch.randn(1, orig_grid[0] * orig_grid[1] * orig_grid[2], embed_dim)

    resized = interpolate_pos_embed_3d(pos_embed, orig_grid, new_grid)

    assert resized.shape == (1, new_grid[0] * new_grid[1] * new_grid[2], embed_dim)


def test_interpolate_pos_embed_3d_same_size_is_a_true_noop():
    embed_dim = 4
    grid = (4, 6, 8)
    pos_embed = torch.randn(1, grid[0] * grid[1] * grid[2], embed_dim)

    resized = interpolate_pos_embed_3d(pos_embed, grid, grid)

    assert resized is pos_embed


# ---------------------------------------------------------------------------
# extract_encoder_state_dict
# ---------------------------------------------------------------------------


def test_extract_encoder_state_dict_keeps_only_shared_vit_attrs():
    fake_state_dict = {
        "patch_embed.proj.weight": 1,
        "pos_embed": 2,
        "cls_token": 3,
        "blocks.0.attn.qkv.weight": 4,
        "norm.weight": 5,
        "var_embed": 6,
        "head.weight": 7,  # VIT's own classification head -- task-specific
        "decoder_pred.weight": 8,  # MAE/DiffusionVIT decoder
        "decoder_norm.weight": 9,
        "mask_token": 10,
        "neck.0.weight": 11,  # SAP decoder
        "mask_header.0.weight": 12,  # SAP decoder
        # UNETR's own skip-connection convs -- decoder-side, NOT the
        # transformer encoder, despite the misleading "encoder" name. The
        # exact case a substring-based ("decoder" not in k) denylist would
        # get wrong; this is why extract_encoder_state_dict is an allowlist.
        "encoder1.layer.conv1.weight": 13,
        "out.conv.weight": 14,
    }

    result = extract_encoder_state_dict(fake_state_dict)

    assert result == {
        "patch_embed.proj.weight": 1,
        "pos_embed": 2,
        "cls_token": 3,
        "blocks.0.attn.qkv.weight": 4,
        "norm.weight": 5,
        "var_embed": 6,
    }


# ---------------------------------------------------------------------------
# extract_encoder_state_dict against every real model type as a pretrained
# source, not just MAE/VIT -- the real workflow is "pretrain via MAE, fine-
# tune into any downstream type," but the encoder-extraction step is equally
# reachable with SAP/UNETR/DiffusionVIT as the pretrained source (nothing in
# get_model restricts which type that is), and each has its own real
# decoder/task-specific keys the fake-dict test above only approximated.
# ---------------------------------------------------------------------------

_ENCODER_ONLY_PREFIXES = {"patch_embed", "token_embeds", "cls_token", "pos_embed", "var_embed", "adaptive_pos_dep_emb", "blocks", "norm"}


def test_extract_encoder_state_dict_strips_real_sap_decoder():
    # weight_init="skip": SAP.__init__ calls self.init_weights('') itself
    # after building neck/mask_header (matches get_model's own
    # weight_init='skip' for every non-VIT type).
    model = SAP(
        img_size=(16, 16), patch_size=4, interp_size=4, twoD=True,
        num_classes=2, class_token=False, pos_embed="learn",
        adaptive_patching=True, fixed_length=4, sqrt_len=2, sqrt_len_method=True,
        weight_init="skip", embed_dim=8, depth=1, num_heads=1, mlp_ratio=1.0, in_chans=1,
    )

    result = extract_encoder_state_dict(model.state_dict())

    assert result.keys() and all(k.split(".")[0] in _ENCODER_ONLY_PREFIXES for k in result)
    # SAP's own neck/mask_header (its segmentation decoder) must be gone.
    assert not any(k.startswith("neck") or k.startswith("mask_header") for k in result)


def test_extract_encoder_state_dict_strips_real_unetr_decoder():
    # linear_decoder=True, skip_connection=False: avoids building UNETR's
    # real monai UnetrBasicBlock/UnetrPrUpBlock conv skip-connection path --
    # unrelated to what's being verified here (which real keys
    # extract_encoder_state_dict keeps/drops), and this way doesn't need a
    # real feature_size-shaped 3D volume to construct.
    model = UNETR(
        img_size=(16, 16), patch_size=4, twoD=True, num_classes=2,
        class_token=False, pos_embed="learn", adaptive_patching=False,
        fixed_length=4, linear_decoder=True, feature_size=4, skip_connection=False,
        weight_init="skip", embed_dim=8, depth=1, num_heads=1, mlp_ratio=1.0, in_chans=1,
    )

    result = extract_encoder_state_dict(model.state_dict())

    assert result.keys() and all(k.split(".")[0] in _ENCODER_ONLY_PREFIXES for k in result)
    # UNETR's own decoder-side modules (its literal "encoderN" conv blocks
    # feed the decoder, not the transformer encoder -- see the fake-dict
    # test above) and linear-decoder head must both be gone.
    assert not any(
        k.startswith(("encoder1", "encoder2", "encoder3", "encoder4", "decoder", "out", "upsample", "mlp_head"))
        for k in result
    )


def test_extract_encoder_state_dict_strips_real_diffusionvit_decoder():
    # linear_decoder=True: avoids building DiffusionVIT's real transformer
    # decoder_blocks -- unrelated to what's being verified here.
    model = DiffusionVIT(
        img_size=(16, 16), patch_size=4, twoD=True, num_classes=None,
        class_token=False, pos_embed="learn", adaptive_patching=False,
        fixed_length=4, linear_decoder=True, num_time_steps=10,
        decoder_depth=None, decoder_embed_dim=None, decoder_num_heads=None, decoder_mlp_ratio=None,
        weight_init="skip", embed_dim=8, depth=1, num_heads=1, mlp_ratio=1.0, in_chans=1,
    )

    result = extract_encoder_state_dict(model.state_dict())

    assert result.keys() and all(k.split(".")[0] in _ENCODER_ONLY_PREFIXES for k in result)
    # DiffusionVIT's own timestep-conditioning/reconstruction-decoder
    # modules must all be gone.
    assert not any(
        k.startswith(("temporalEmbeddings", "timeEmbeddingMap", "decoder"))
        for k in result
    )


# ---------------------------------------------------------------------------
# End-to-end: pretrain at one h:w ratio, load at a different h:w[:d] ratio
# ---------------------------------------------------------------------------

_COMMON_KWARGS = dict(
    embed_dim=8, depth=1, num_heads=1, mlp_ratio=1.0, in_chans=1,
    fixed_length=16, adaptive_patching=False,
)


def _load_pretrained_encoder(pretrained_model, model):
    """Mirrors get_model's pretrained branch: extract + transplant + merge."""
    encoder_dict = extract_encoder_state_dict(pretrained_model.state_dict())
    _transplant_pos_embed(encoder_dict, pretrained_model, model)
    _prune_incompatible_cls_token(encoder_dict, model)
    model_dict = model.state_dict()
    model_dict.update(encoder_dict)
    model.load_state_dict(model_dict)


def test_pretrained_loading_2d_non_square_ratio_change_same_architecture():
    pretrained = VIT(
        img_size=(16, 32), patch_size=4, twoD=True, num_classes=2,
        class_token=True, pos_embed="learn", **_COMMON_KWARGS,
    )
    new_model = VIT(
        img_size=(32, 16), patch_size=4, twoD=True, num_classes=3,
        class_token=True, pos_embed="learn", **_COMMON_KWARGS,
    )
    fresh_new_pos_embed = new_model.pos_embed.clone()

    _load_pretrained_encoder(pretrained, new_model)

    assert new_model.pos_embed.shape == fresh_new_pos_embed.shape
    # Actually replaced (interpolated from the pretrained checkpoint), not
    # left at its own freshly-initialized values.
    assert not torch.equal(new_model.pos_embed, fresh_new_pos_embed)
    # head is task-specific (num_classes differs, 2 vs 3) -- must NOT have
    # been overwritten by the pretrained model's own head.
    assert new_model.head.out_features == 3


def test_pretrained_loading_cross_architecture_mae_encoder_into_vit():
    pretrained = MAE(
        img_size=(32, 64), patch_size=4, twoD=True, class_token=False,
        pos_embed="learn", mask_ratio=0.75, linear_decoder=True,
        decoder_depth=None, decoder_embed_dim=None, decoder_num_heads=None,
        decoder_mlp_ratio=None, num_classes=None,
        # MAE.__init__ calls self.init_weights('') itself, after setting its
        # own decoder_pos_embed -- weight_init must be 'skip' here or
        # VIT.__init__'s own call to the same (polymorphic) init_weights
        # fires first, before decoder_pos_embed exists at all. Matches
        # get_model's own real construction (weight_init='skip' for every
        # non-VIT type).
        weight_init="skip",
        **_COMMON_KWARGS,
    )
    new_model = VIT(
        img_size=(64, 32), patch_size=4, twoD=True, num_classes=5,
        class_token=True, pos_embed="learn", **_COMMON_KWARGS,
    )

    _load_pretrained_encoder(pretrained, new_model)

    assert new_model.pos_embed.shape[1] == new_model.num_patches + new_model.num_prefix_tokens
    # MAE's decoder must not have leaked into the VIT model at all.
    assert not any(k.startswith("decoder") or k == "mask_token" for k in new_model.state_dict())


def test_pretrained_loading_cross_architecture_vit_encoder_into_mae():
    """The reverse prefix-token direction from the MAE->VIT case above: 1
    class-token prefix row (VIT) down to 0 (MAE) -- _transplant_pos_embed's
    mismatched-prefix-count branch slices `resized[:, pretrained_model.
    num_prefix_tokens:]` (dropping the pretrained model's own leading cls
    row) rather than prepending a fresh one, a genuinely different branch
    than the 0->1 direction the MAE->VIT test above exercises.
    """
    pretrained = VIT(
        img_size=(32, 64), patch_size=4, twoD=True, num_classes=2,
        class_token=True, pos_embed="learn", **_COMMON_KWARGS,
    )
    new_model = MAE(
        img_size=(64, 32), patch_size=4, twoD=True, class_token=False,
        pos_embed="learn", mask_ratio=0.75, linear_decoder=True,
        decoder_depth=None, decoder_embed_dim=None, decoder_num_heads=None,
        decoder_mlp_ratio=None, num_classes=None,
        weight_init="skip",
        **_COMMON_KWARGS,
    )
    fresh_new_pos_embed = new_model.pos_embed.clone()

    _load_pretrained_encoder(pretrained, new_model)

    # No prefix row at all (class_token=False) -- not 1 + grid, just grid.
    assert new_model.pos_embed.shape == fresh_new_pos_embed.shape
    assert new_model.pos_embed.shape[1] == new_model.num_patches
    assert not torch.equal(new_model.pos_embed, fresh_new_pos_embed)
    # VIT's own classification head must not have leaked into MAE at all.
    assert not any(k.startswith("head") for k in new_model.state_dict())


def test_pretrained_loading_3d_non_cubic_ratio_change():
    # embed_dim=12 (not _COMMON_KWARGS' 8): get_3d_sincos_pos_embed requires
    # embed_dim % 3 == 0.
    kwargs_3d = dict(_COMMON_KWARGS, embed_dim=12)
    pretrained = VIT(
        img_size=(8, 16, 32), patch_size=4, twoD=False, num_classes=2,
        class_token=False, pos_embed="learn", **kwargs_3d,
    )
    new_model = VIT(
        img_size=(32, 8, 16), patch_size=4, twoD=False, num_classes=2,
        class_token=False, pos_embed="learn", **kwargs_3d,
    )
    fresh_new_pos_embed = new_model.pos_embed.clone()

    _load_pretrained_encoder(pretrained, new_model)

    assert new_model.pos_embed.shape == fresh_new_pos_embed.shape
    assert not torch.equal(new_model.pos_embed, fresh_new_pos_embed)


def test_pretrained_loading_adaptive_patching_fixed_length_mismatch_drops_pos_embed():
    """adaptive_patching:True + not sqrt_len_method: pos_embed is a flat,
    learned, per-sequence-slot-index embedding with no spatial/geometric
    meaning attached to any given slot (FixedQuadTree/FixedOctTree's own
    node order reflects greedy-split order, not spatial adjacency) --
    unlike the grid case above, there's no principled way to resize it, so
    a fixed_length mismatch drops it entirely (new model keeps its own
    fresh init) rather than attempting an interpolation. An earlier
    implementation instead did a 1D linear interpolation along the
    slot-index axis, resting on an adjacency assumption the data doesn't
    satisfy, and -- independently -- never sliced out num_prefix_tokens
    first, corrupting the class-token row into the interpolation whenever
    class_token:True (exercised directly here via class_token=True on both
    sides).
    """
    kwargs = dict(
        patch_size=4, interp_size=4, twoD=True, num_classes=2,
        class_token=True, pos_embed="learn", adaptive_patching=True,
        sqrt_len_method=False, embed_dim=8, depth=1, num_heads=1,
        mlp_ratio=1.0, in_chans=1,
    )
    pretrained = VIT(img_size=(32, 32), fixed_length=16, **kwargs)
    new_model = VIT(img_size=(32, 32), fixed_length=24, **kwargs)
    fresh_new_pos_embed = new_model.pos_embed.clone()

    _load_pretrained_encoder(pretrained, new_model)

    # Dropped, not interpolated -- new model keeps its own fresh init exactly.
    assert torch.equal(new_model.pos_embed, fresh_new_pos_embed)
    assert new_model.pos_embed.shape[1] == new_model.fixed_length + new_model.num_prefix_tokens


def test_pretrained_loading_sqrt_len_method_mismatch_raises_clearly():
    # embed_dim=12: get_3d_sincos_pos_embed requires embed_dim % 3 == 0.
    sqrt_len_kwargs = dict(
        patch_size=4, interp_size=4, twoD=False, num_classes=2,
        class_token=False, pos_embed="learn", adaptive_patching=True,
        sqrt_len_method=True, fixed_length=8,
        embed_dim=12, depth=1, num_heads=1, mlp_ratio=1.0, in_chans=1,
    )
    pretrained = VIT(img_size=(32, 32, 32), **sqrt_len_kwargs)
    new_model = VIT(img_size=(64, 64, 64), **sqrt_len_kwargs)

    encoder_dict = extract_encoder_state_dict(pretrained.state_dict())
    with pytest.raises(NotImplementedError, match="sqrt_len_method"):
        _transplant_pos_embed(encoder_dict, pretrained, new_model)
