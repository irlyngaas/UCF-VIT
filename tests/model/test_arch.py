"""Tests for UCF_VIT.model.arch: VIT.effective_patch_size, SAP.mask_head, UNETR.proj_feat.

Covers the interp_size/patch_size dispatch introduced to stop patch_size
from being silently reused (with a different meaning) whenever
adaptive_patching is turned on: `interp_size` (the size adaptive
quadtree/octree leaf patches are interpolated to, and every dependent
model-layer calculation is based on) takes over patch_size's role entirely
in that mode, and is required in that mode.

Also covers the fix for adaptive patching's flat token sequence (greedy-split
tree order, not raster order) being wrongly treated as a dense spatial grid:
SAP.mask_head now returns per-token predictions (indexed by real sequence
position) instead of a dense grid built by raw reshape, and UNETR.proj_feat
now reconstructs each feat_size canvas cell from its true owning token (via
_adaptive_token_grid_index's point-in-box lookup against real pos/size)
instead of assuming sequence index == raster position.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see tests/distributed/test_tensor_parallel_correctness.py's
module docstring for why this skips cleanly via importorskip instead of
erroring at collection. Imports UCF_VIT.model.arch itself (not just the bare
top-level packages) so this also skips cleanly if an installed xformers
version has dropped a submodule building_blocks.py needs (as happened during
this test's own development).
"""

import pytest
import torch

_arch = pytest.importorskip(
    "UCF_VIT.model.arch",
    reason="needs the real UCF_VIT.model.building_blocks deps (timm/monai/xformers) -- run in the forge-vit env",
)
VIT = _arch.VIT
SAP = _arch.SAP
UNETR = _arch.UNETR


def _make_vit(**overrides):
    kwargs = dict(
        img_size=16,
        patch_size=4,
        in_chans=1,
        num_classes=2,
        embed_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        twoD=True,
        adaptive_patching=False,
        fixed_length=16,
        class_token=False,
        pos_embed="none",
    )
    kwargs.update(overrides)
    return VIT(**kwargs)


def test_effective_patch_size_non_adaptive_uses_patch_size():
    model = _make_vit(adaptive_patching=False, patch_size=4)
    assert model.effective_patch_size == 4


def test_effective_patch_size_adaptive_uses_interp_size():
    model = _make_vit(adaptive_patching=True, interp_size=8, fixed_length=16)
    assert model.effective_patch_size == 8
    # patch_size itself must stay untouched/truthful, not silently overwritten.
    assert model.patch_size == 4


def test_effective_patch_size_adaptive_missing_interp_size_raises():
    with pytest.raises(AssertionError, match="interp_size is required"):
        _make_vit(adaptive_patching=True, interp_size=None, fixed_length=16)


# ---------------------------------------------------------------------------
# SAP.mask_head -- per-token output (not a dense grid built by raw reshape)
# ---------------------------------------------------------------------------


def _make_sap(**overrides):
    kwargs = dict(
        img_size=8,
        patch_size=4,
        interp_size=4,
        in_chans=1,
        num_classes=3,
        embed_dim=4,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        twoD=True,
        adaptive_patching=True,
        fixed_length=4,
        class_token=False,
        pos_embed="none",
        sqrt_len=2,
    )
    kwargs.update(overrides)
    return SAP(**kwargs)


def test_sap_mask_head_returns_per_token_predictions_2d():
    model = _make_sap()
    pooled = torch.randn(2, model.fixed_length, model.embed_dim)
    out = model.mask_head(pooled)
    assert out.shape == (2, model.fixed_length, model.num_classes, model.effective_patch_size, model.effective_patch_size)


def test_sap_mask_head_returns_per_token_predictions_3d():
    model = _make_sap(twoD=False, img_size=8, sqrt_len=2, fixed_length=8)
    pooled = torch.randn(2, model.fixed_length, model.embed_dim)
    out = model.mask_head(pooled)
    assert out.shape == (
        2, model.fixed_length, model.num_classes,
        model.effective_patch_size, model.effective_patch_size, model.effective_patch_size,
    )


# ---------------------------------------------------------------------------
# UNETR.proj_feat / _adaptive_token_grid_index -- reconstruct true positions,
# not raw sequence order
# ---------------------------------------------------------------------------


def _make_unetr(**overrides):
    kwargs = dict(
        img_size=(12, 12),
        patch_size=4,
        interp_size=3,
        in_chans=1,
        num_classes=2,
        embed_dim=4,
        depth=4,
        num_heads=1,
        mlp_ratio=1.0,
        twoD=True,
        adaptive_patching=True,
        fixed_length=16,
        class_token=False,
        pos_embed="none",
        feature_size=4,
        skip_connection=True,
        linear_decoder=False,
        sqrt_len=4,
    )
    kwargs.update(overrides)
    return UNETR(**kwargs)


def test_unetr_proj_feat_reconstructs_true_spatial_positions_from_shuffled_token_order():
    """Regression test for the fake-grid bug: tokens are given in a
    deliberately shuffled (non-raster) order with their real (pos, size)
    boxes -- the same shape of mismatch real greedy-split tree order
    produces -- and proj_feat must still place each token's feature at the
    canvas cell its true box covers, not at its raw sequence position.
    """
    model = _make_unetr()
    feat_size = model.feat_size  # (4, 4)

    # Boxes exactly tiling img_size=(12,12) at feat_size=(4,4) resolution:
    # box (r, c) covers canvas cell (r, c) exactly (3x3 native pixels each).
    boxes = []
    for r in range(feat_size[0]):
        for c in range(feat_size[1]):
            x1, x2 = c * 3, (c + 1) * 3
            y1, y2 = r * 3, (r + 1) * 3
            boxes.append((x1, x2, y1, y2))

    perm = torch.randperm(len(boxes))
    shuffled = [boxes[i] for i in perm.tolist()]
    size = torch.tensor([[b[1] - b[0] for b in shuffled]], dtype=torch.float32)
    pos = torch.tensor([[[(b[0] + b[1]) / 2, (b[2] + b[3]) / 2] for b in shuffled]], dtype=torch.float32)
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)  # (1, S, 3)

    # Each token's feature is its ORIGINAL (raster, pre-shuffle) box index,
    # so recovering the right value at each canvas cell proves the gather
    # followed true position, not sequence order.
    tokens = torch.zeros(1, len(boxes), model.embed_dim)
    for seq_idx, orig_idx in enumerate(perm.tolist()):
        tokens[0, seq_idx, :] = orig_idx

    token_grid_index = model._adaptive_token_grid_index(seq_ps, feat_size)
    canvas = model.proj_feat(tokens, model.embed_dim, feat_size, token_grid_index)

    expected = torch.arange(feat_size[0] * feat_size[1]).view(1, 1, feat_size[0], feat_size[1]).float()
    expected = expected.expand(1, model.embed_dim, feat_size[0], feat_size[1])
    assert torch.equal(canvas, expected)


def test_unetr_proj_feat_non_adaptive_unchanged():
    """Non-adaptive case doesn't need (or use) token_grid_index -- confirms
    the new gather path is gated strictly behind self.adaptive_patching."""
    model = _make_unetr(adaptive_patching=False)
    # Non-adaptive feat_size = img_size / patch_size = 12 / 4 = (3, 3).
    n_tokens = model.feat_size[0] * model.feat_size[1]
    tokens = torch.arange(n_tokens * model.embed_dim, dtype=torch.float32).view(1, n_tokens, model.embed_dim)

    canvas = model.proj_feat(tokens, model.embed_dim, model.feat_size)

    expected = tokens.view(1, model.feat_size[0], model.feat_size[1], model.embed_dim).permute(0, 3, 1, 2)
    assert torch.equal(canvas, expected)
