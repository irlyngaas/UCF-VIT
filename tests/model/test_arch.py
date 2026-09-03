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


# ---------------------------------------------------------------------------
# UNETR.token_selection -- "point" (default), "smallest_overlap", "area_weighted"
# ---------------------------------------------------------------------------


def test_unetr_token_selection_defaults_to_point():
    model = _make_unetr()
    assert model.token_selection == "point"


def test_unetr_token_selection_smallest_overlap_prefers_small_token_over_large():
    """"point" samples a fixed grid of cell-center points, so a token's odds
    of winning any cell scale with its area -- small (typically detail-rich)
    tokens can lose to a large token that happens to cover more sample
    points. "smallest_overlap" tests real box-vs-cell overlap instead of a
    single point, and picks the smallest overlapping token, so a small token
    always wins any cell it touches regardless of how much area a
    competing large token covers.
    """
    model = _make_unetr(token_selection="smallest_overlap")
    assert model.token_selection == "smallest_overlap"
    feat_size = model.feat_size  # (4, 4) over img_size=(12, 12) -> 3x3-pixel cells

    # One token covering the whole domain, and one tiny token sitting inside
    # cell (0, 0) (native pixels [0,3)x[0,3), center (1.5, 1.5)).
    big = (0.0, 12.0, 0.0, 12.0)
    small = (1.4, 1.6, 1.4, 1.6)
    size = torch.tensor([[big[1] - big[0], small[1] - small[0]]])
    pos = torch.tensor([[
        [(big[0] + big[1]) / 2, (big[2] + big[3]) / 2],
        [(small[0] + small[1]) / 2, (small[2] + small[3]) / 2],
    ]])
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    token_grid_index = model._adaptive_token_grid_index(seq_ps, feat_size).view(feat_size[0], feat_size[1])

    assert token_grid_index[0, 0].item() == 1  # the small token wins the cell it overlaps
    assert (token_grid_index.flatten() == 1).sum().item() == 1  # and only that one cell
    assert (token_grid_index.flatten() == 0).sum().item() == feat_size[0] * feat_size[1] - 1  # big token gets the rest


def test_unetr_token_selection_point_can_lose_small_token_entirely():
    """Companion to the test above: under "point", the same tiny token can
    lose every cell (including the one it physically overlaps) if no cell
    *center* happens to fall inside it -- demonstrating the bias
    "smallest_overlap" exists to fix, not just a difference in style.
    """
    model = _make_unetr(token_selection="point")
    feat_size = model.feat_size  # cell (0,0)'s center is at (1.5, 1.5)

    big = (0.0, 12.0, 0.0, 12.0)
    small = (0.1, 0.3, 0.1, 0.3)  # tiny box in cell (0,0), doesn't contain (1.5, 1.5)
    size = torch.tensor([[big[1] - big[0], small[1] - small[0]]])
    pos = torch.tensor([[
        [(big[0] + big[1]) / 2, (big[2] + big[3]) / 2],
        [(small[0] + small[1]) / 2, (small[2] + small[3]) / 2],
    ]])
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    token_grid_index = model._adaptive_token_grid_index(seq_ps, feat_size)
    assert (token_grid_index == 1).sum().item() == 0  # the small token (index 1) wins no cell at all


def test_unetr_token_selection_area_weighted_matches_gather_when_tiling_is_exact():
    """When tokens exactly tile the canvas 1:1 (no ambiguity), area-weighted
    blending's per-cell weights collapse to one-hot -- proj_feat's weighted-
    sum reconstruction must produce the exact same result the gather-based
    methods do, not just a close approximation.
    """
    model = _make_unetr(token_selection="area_weighted")
    assert model.token_selection == "area_weighted"
    feat_size = model.feat_size  # (4, 4)

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
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    weights = model._adaptive_token_grid_weights(seq_ps, feat_size)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, feat_size[0] * feat_size[1]))
    assert torch.equal(weights, (weights > 0.999).to(weights.dtype))  # one-hot

    tokens = torch.zeros(1, len(boxes), model.embed_dim)
    for seq_idx, orig_idx in enumerate(perm.tolist()):
        tokens[0, seq_idx, :] = orig_idx

    canvas = model.proj_feat(tokens, model.embed_dim, feat_size, weights)
    expected = torch.arange(feat_size[0] * feat_size[1]).view(1, 1, feat_size[0], feat_size[1]).float()
    expected = expected.expand(1, model.embed_dim, feat_size[0], feat_size[1])
    assert torch.equal(canvas, expected)


def test_unetr_token_selection_area_weighted_blends_proportionally_by_area():
    """Two real, non-overlapping (square, tiling) leaves both landing inside
    one canvas cell -- the cell's weight for each must be proportional to
    how much of the cell's own area that leaf actually covers, not
    winner-take-all.
    """
    model = _make_unetr(token_selection="area_weighted", img_size=(12, 12))
    feat_size = (2, 2)  # cell (0,0) = [0,6)x[0,6), area 36

    # leafA: square side 4, box [0,4)x[0,4) -- fully inside cell (0,0), area 16.
    # leafB: square side 2, box [4,6)x[0,2) -- fully inside cell (0,0), area 4,
    # non-overlapping with leafA (leafA's x ends exactly where leafB's starts).
    size = torch.tensor([[4.0, 2.0]])
    pos = torch.tensor([[[2.0, 2.0], [5.0, 1.0]]])
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    weights = model._adaptive_token_grid_weights(seq_ps, feat_size).view(feat_size[0], feat_size[1], 2)
    cell00 = weights[0, 0]
    total_covered = 16.0 + 4.0  # only 20 of the cell's 36 area is covered by these two leaves
    assert cell00[0].item() == pytest.approx(16.0 / total_covered, abs=1e-5)
    assert cell00[1].item() == pytest.approx(4.0 / total_covered, abs=1e-5)


def test_unetr_area_weighted_alpha_defaults_to_zero():
    model = _make_unetr(token_selection="area_weighted")
    assert model.area_weighted_alpha == 0.0


def test_unetr_area_weighted_alpha_zero_matches_plain_area_weighting():
    """alpha=0 must reduce to exactly the original (pre-alpha) area-proportional formula."""
    model = _make_unetr(token_selection="area_weighted", area_weighted_alpha=0.0, img_size=(12, 12))
    feat_size = (2, 2)
    size = torch.tensor([[4.0, 2.0]])
    pos = torch.tensor([[[2.0, 2.0], [5.0, 1.0]]])
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    weights = model._adaptive_token_grid_weights(seq_ps, feat_size).view(feat_size[0], feat_size[1], 2)
    cell00 = weights[0, 0]
    total_covered = 16.0 + 4.0
    assert cell00[0].item() == pytest.approx(16.0 / total_covered, abs=1e-5)
    assert cell00[1].item() == pytest.approx(4.0 / total_covered, abs=1e-5)


def test_unetr_area_weighted_alpha_increases_small_token_weight_toward_smallest_overlap():
    """As alpha grows, the smaller of two overlapping tokens' weight should
    monotonically increase past its plain area-proportional share, toward 1
    -- "smallest_overlap"'s winner-take-all outcome in the limit.
    """
    feat_size = (4, 4)
    big = (0.0, 12.0, 0.0, 12.0)
    small = (1.0, 2.0, 1.0, 2.0)  # side 1, fully inside cell (0,0) = [0,3)x[0,3)
    size = torch.tensor([[big[1] - big[0], small[1] - small[0]]])
    pos = torch.tensor([[
        [(big[0] + big[1]) / 2, (big[2] + big[3]) / 2],
        [(small[0] + small[1]) / 2, (small[2] + small[3]) / 2],
    ]])
    seq_ps = torch.cat([size.unsqueeze(-1), pos], dim=-1)

    small_weights = []
    for alpha in (0.0, 1.0, 5.0, 50.0):
        model = _make_unetr(token_selection="area_weighted", area_weighted_alpha=alpha, img_size=(12, 12))
        w = model._adaptive_token_grid_weights(seq_ps, feat_size).view(feat_size[0], feat_size[1], 2)[0, 0]
        small_weights.append(w[1].item())

    assert small_weights == sorted(small_weights)  # monotonically increasing
    assert small_weights[0] == pytest.approx(0.1, abs=1e-5)  # alpha=0: plain area share (1/(1+9))
    assert small_weights[-1] == pytest.approx(1.0, abs=1e-3)  # alpha=50: effectively all the weight


# ---------------------------------------------------------------------------
# UNETR.token_selection -- "cross_attention" (learned soft pooling)
# ---------------------------------------------------------------------------


def test_unetr_cross_attention_builds_expected_submodules():
    model = _make_unetr(token_selection="cross_attention")
    assert model.token_selection == "cross_attention"
    assert isinstance(model.token_query_mlp, torch.nn.Sequential)
    assert isinstance(model.token_key_proj, torch.nn.Linear)
    assert isinstance(model.token_value_proj, torch.nn.Linear)


def test_unetr_cross_attention_not_built_for_other_selections():
    model = _make_unetr(token_selection="point")
    assert not hasattr(model, "token_query_mlp")
    assert not hasattr(model, "token_key_proj")
    assert not hasattr(model, "token_value_proj")


# 3 real, non-overlapping tokens exactly covering a 2x2 canvas over a 12x12
# domain, with token A spanning both cells in column 0 -- reused by both
# tests below.
_CROSS_ATTN_FEAT_SIZE = (2, 2)
_CROSS_ATTN_IMG_SIZE = (12, 12)
_CROSS_ATTN_TOKEN_BOXES = [
    (0.0, 4.0, 0.0, 12.0),   # A: x[0,4), y[0,12) -- spans cell(0,0) and cell(1,0)
    (7.0, 9.0, 7.0, 9.0),    # B: x[7,9), y[7,9) -- cell(1,1) only
    (7.0, 9.0, 1.0, 3.0),    # C: x[7,9), y[1,3) -- cell(0,1) only
]


def _cross_attn_seq_ps():
    size = torch.tensor([[b[1] - b[0] for b in _CROSS_ATTN_TOKEN_BOXES]])
    pos = torch.tensor([[[(b[0] + b[1]) / 2, (b[2] + b[3]) / 2] for b in _CROSS_ATTN_TOKEN_BOXES]])
    return torch.cat([size.unsqueeze(-1), pos], dim=-1)


def test_unetr_cross_attention_query_and_mask_shapes_and_masking():
    model = _make_unetr(token_selection="cross_attention", img_size=_CROSS_ATTN_IMG_SIZE)
    seq_ps = _cross_attn_seq_ps()

    query, mask = model._cross_attention_query_and_mask(seq_ps, _CROSS_ATTN_FEAT_SIZE)

    assert query.shape == (_CROSS_ATTN_FEAT_SIZE[0] * _CROSS_ATTN_FEAT_SIZE[1], model.embed_dim)
    mask = mask.view(_CROSS_ATTN_FEAT_SIZE[0], _CROSS_ATTN_FEAT_SIZE[1], 3)
    # cell(0,0)->A, cell(0,1)->C, cell(1,0)->A, cell(1,1)->B; each cell has
    # exactly one real overlapping token, A owns two cells.
    assert mask[0, 0].tolist() == [True, False, False]
    assert mask[0, 1].tolist() == [False, False, True]
    assert mask[1, 0].tolist() == [True, False, False]
    assert mask[1, 1].tolist() == [False, True, False]


def test_unetr_cross_attention_proj_feat_respects_mask_and_multi_cell_overlap():
    """End-to-end through proj_feat, with token_key_proj/token_value_proj
    fixed to the identity (rather than random init) so the masked-attention
    output is fully deterministic: each cell's reconstructed feature must
    exactly equal its one real owning token's embedding, and the token
    spanning two cells (A) must produce the identical result in both.
    """
    model = _make_unetr(token_selection="cross_attention", img_size=_CROSS_ATTN_IMG_SIZE)
    with torch.no_grad():
        model.token_key_proj.weight.copy_(torch.eye(model.embed_dim))
        model.token_key_proj.bias.zero_()
        model.token_value_proj.weight.copy_(torch.eye(model.embed_dim))
        model.token_value_proj.bias.zero_()

    seq_ps = _cross_attn_seq_ps()
    token_selection_state = model._compute_token_selection_state(seq_ps, _CROSS_ATTN_FEAT_SIZE)

    tokens = torch.stack([
        torch.full((model.embed_dim,), 1.0),  # A
        torch.full((model.embed_dim,), 2.0),  # B
        torch.full((model.embed_dim,), 3.0),  # C
    ]).unsqueeze(0)  # (1, 3, embed_dim)

    canvas = model.proj_feat(tokens, model.embed_dim, _CROSS_ATTN_FEAT_SIZE, token_selection_state)

    assert torch.allclose(canvas[0, :, 0, 0], torch.full((model.embed_dim,), 1.0), atol=1e-5)  # cell(0,0) -> A
    assert torch.allclose(canvas[0, :, 0, 1], torch.full((model.embed_dim,), 3.0), atol=1e-5)  # cell(0,1) -> C
    assert torch.allclose(canvas[0, :, 1, 0], torch.full((model.embed_dim,), 1.0), atol=1e-5)  # cell(1,0) -> A (same as cell(0,0))
    assert torch.allclose(canvas[0, :, 1, 1], torch.full((model.embed_dim,), 2.0), atol=1e-5)  # cell(1,1) -> B
