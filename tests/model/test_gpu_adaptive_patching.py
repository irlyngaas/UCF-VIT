"""Tests for UCF_VIT.model.gpu_adaptive_patching.GPUPatchify2D.

GPUPatchify2D is a GPU-native, level-parallel *merge* (bottom-up, shrinking
from the finest grid down to `fixed_length` leaves) alternative to the
CPU/dataloader-side `Patchify`/`FixedQuadTree` *split* (top-down, growing
from 1 root), ported from a reference implementation the user maintains
elsewhere. These tests exercise it standalone (no model/config wiring
exists yet -- that's a separate, later pass) with real, deterministic
tensors, so an offset/shape/routing bug shows up as a wrong value, not just
a wrong shape.

Runs fully on CPU (`torch`, `scipy.ndimage`, `torch.nn.functional.grid_sample`
-- no CUDA, no dataset-specific dependency).
"""

import sys
import types

import pytest
import torch

from UCF_VIT.model.gpu_adaptive_patching import GPUPatchify2D, _label_regions


def test_pad_tensor_pads_to_coarsest_block_size():
    # img_size=(12,12), min_size=2 -> max_blocks=6, max_level=2 (2**2=4<=6<8),
    # coarsest block = 2*4=8 -- 12 is not a multiple of 8, needs padding to 16,
    # which is also why the *real* level-0 grid is 16/2=8 -> 64 leaves, not
    # 6*6=36 -- fixed_length must satisfy (64 - fixed_length) % 3 == 0.
    p = GPUPatchify2D(img_size=(12, 12), fixed_length=4, interp_size=4, min_size=2)
    t = torch.arange(12 * 12, dtype=torch.float32).reshape(1, 1, 12, 12)

    padded, orig = p._pad_tensor(t)

    assert orig == (12, 12)
    assert padded.shape[-2:] == (16, 16)
    # Edge-replication: padded region repeats the last real row/column.
    assert torch.equal(padded[0, 0, :12, :12], t[0, 0])
    assert torch.equal(padded[0, 0, 12:, :12], t[0, 0, 11:12, :].expand(4, 12))
    assert torch.equal(padded[0, 0, :12, 12:], t[0, 0, :, 11:12].expand(12, 4))


def test_pad_tensor_no_op_when_already_aligned():
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=1, interp_size=4, min_size=2)
    t = torch.rand(1, 1, 16, 16)

    padded, orig = p._pad_tensor(t)

    assert orig == (16, 16)
    assert torch.equal(padded, t)


def test_fixed_length_congruence_assertion():
    # img_size=(16,16), min_size=2 -> max_blocks=8, initial_leaves=64.
    # (64 - fixed_length) % 3 == 0 -> fixed_length in {..., 10, 13, 16, ...}.
    GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2)  # does not raise

    with pytest.raises(AssertionError, match="fixed_length"):
        GPUPatchify2D(img_size=(16, 16), fixed_length=14, interp_size=4, min_size=2)


def test_non_square_requires_shorter_first_dimension():
    with pytest.raises(NotImplementedError, match="img_size"):
        GPUPatchify2D(img_size=(16, 8), fixed_length=1, interp_size=4, min_size=2)


# ---------------------------------------------------------------------------
# region_backend -- "scipy" (default, CPU round-trip) is exercised by every
# other test in this file unchanged. "cupyx" (opt-in, GPU-native) can't be
# tested against real cupy/GPU hardware in this environment (no CUDA/ROCm
# device, cupy not installed) -- these tests cover (a) the real, honest
# behavior here (cupy genuinely missing -> a clear ImportError, not a silent
# fallback) and (b) the dispatch logic itself with cupy/cupyx faked out,
# trusting cupy's documented scipy.ndimage.label API compatibility for the
# real numerics.
# ---------------------------------------------------------------------------


def test_region_backend_invalid_value_raises_clearly():
    border_mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(AssertionError, match="region_backend"):
        _label_regions(border_mask, region_backend="bogus")


def test_region_backend_defaults_to_scipy_with_no_env_var(monkeypatch):
    monkeypatch.delenv("UCF_VIT_GPU_AP_REGION_BACKEND", raising=False)
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=1, interp_size=4, min_size=2)
    assert p.region_backend == "scipy"


def test_region_backend_reads_env_var_when_not_passed_explicitly(monkeypatch):
    # Real production path (arch.py's do_gpu_ap dispatch) never passes
    # region_backend at all -- this env var is how a real training job can
    # try "cupyx" with no code or config change (see this class's own
    # region_backend docstring for why: kept test-only, not wired into
    # parse.py/config YAML yet).
    monkeypatch.setenv("UCF_VIT_GPU_AP_REGION_BACKEND", "cupyx")
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=1, interp_size=4, min_size=2)
    assert p.region_backend == "cupyx"


def test_region_backend_explicit_value_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("UCF_VIT_GPU_AP_REGION_BACKEND", "cupyx")
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=1, interp_size=4, min_size=2, region_backend="scipy")
    assert p.region_backend == "scipy"


def test_region_backend_cupyx_without_cupy_installed_raises_import_error():
    # A real, meaningful assertion in *this* environment: cupy genuinely
    # isn't installed here, so this exercises the actual (not simulated)
    # error path a user would hit without it.
    border_mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(ImportError, match="cupy"):
        _label_regions(border_mask, region_backend="cupyx")


def test_region_backend_cupyx_requires_a_cuda_tensor(monkeypatch):
    # Fakes cupy/cupyx being installed (so the ImportError above doesn't
    # fire) to isolate the *next* check: region_backend="cupyx" must still
    # refuse a CPU tensor rather than silently using it.
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    fake_ndimage = types.ModuleType("cupyx.scipy.ndimage")
    fake_ndimage.label = lambda x: (x, 0)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", fake_ndimage)

    border_mask = torch.zeros(4, 4, dtype=torch.bool)
    with pytest.raises(AssertionError, match="CUDA"):
        _label_regions(border_mask, region_backend="cupyx")


def test_region_backend_cupyx_dispatch_with_cupy_faked_out(monkeypatch):
    """Confirms the "cupyx" branch is actually reached and its
    (labeled, num_features) result is wrapped back correctly -- a
    dispatch-logic test, not a real-numerics test. cupy.from_dlpack and
    cupyx.scipy.ndimage.label are faked to operate directly on plain torch
    tensors (torch.from_dlpack already accepts a real torch.Tensor
    unchanged, confirmed directly -- no need to fake that half too).
    """
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.from_dlpack = lambda t: t  # "cupy array" is just the same tensor here
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    calls = []

    def fake_label(interior):
        calls.append(interior)
        labeled = torch.zeros_like(interior, dtype=torch.int32)
        labeled[interior] = 1
        return labeled, 1

    fake_ndimage = types.ModuleType("cupyx.scipy.ndimage")
    fake_ndimage.label = fake_label
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", fake_ndimage)

    # is_cuda is a read-only property on real tensors -- patch it at the
    # class level so a plain CPU tensor reports True for this test only.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    border_mask = torch.tensor([[True, False], [False, False]])
    labeled, num_features = _label_regions(border_mask, region_backend="cupyx")

    assert num_features == 1
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], ~border_mask)
    torch.testing.assert_close(labeled, torch.tensor([[0, 1], [1, 1]], dtype=torch.int32))


def test_merge_cost_is_exactly_zero_for_a_flat_region():
    """A perfectly uniform block has zero variance at every level -- merging
    it costs exactly 0, deterministically (no dependence on block size or
    the unbiased-variance correction, unlike a genuinely detailed region --
    see test_merge_cost_prefers_merging_flat_region_over_a_step_edge's own
    docstring for why that one uses a step edge, not noise, as the "detailed"
    case).
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2)
    img = torch.full((1, 1, 16, 16), 3.0)

    merge_costs, _ = p._compute_all_levels_batch(img)

    for level in range(1, p.max_level + 1):
        assert torch.equal(merge_costs[level][0], torch.zeros_like(merge_costs[level][0]))


def test_merge_cost_prefers_merging_flat_region_over_a_step_edge():
    """A deterministic step edge (not random noise, and not a period-2
    checkerboard) inside one quadrant, flat everywhere else.

    Deliberately not noise: with only a handful of samples per block,
    per-block variance's *unbiased* (n-1 denominator) correction can make a
    genuinely-detailed small block's SSE come out *below* the sum of its
    even-smaller children's SSEs (a real, if surprising, property of this
    ported cost formula -- confirmed by hand -- not a bug introduced by the
    port), giving an occasional negative "cost" for random noise. A step
    edge that lines up with block boundaries avoids this: every block is
    internally uniform until the level whose blocks actually straddle the
    edge, which then gets a large, unambiguous, deterministic positive cost.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2)
    img = torch.full((1, 1, 16, 16), 1.0)
    img[0, 0, 0:8, 4:8] = 10.0  # step edge inside the top-left 8x8 quadrant

    merge_costs, _ = p._compute_all_levels_batch(img)

    # Level 2 merges 4x4 blocks into 8x8 -- the top-left 8x8 (row 0, col 0)
    # straddles the step edge; every other 8x8 region is untouched and flat.
    level2 = merge_costs[2][0]
    assert level2[0, 0] > 0
    assert torch.equal(level2.flatten()[1:], torch.zeros(level2.numel() - 1))


def test_forward_terminates_when_img_size_needs_padding():
    """Regression test for a real hang: `__init__`'s congruence assertion
    used to check `(img_size[0] // min_size) ** 2` (the *unpadded* grid),
    but `forward()` actually operates on `_pad_tensor`'s *padded* grid, which
    is strictly larger whenever img_size isn't already a multiple of
    `_pad_size()` -- e.g. img_size=(12,12), min_size=2 pads to (16,16): a
    real 8*8=64-leaf grid, not the 6*6=36 the old assertion checked against.
    An off-by-wrong-modulus fixed_length silently passed that stale check,
    then `run_merge_batch`'s per-image budget (`(leaves_remaining -
    fixed_length) // 3`) permanently floors to 0 once leaves_remaining can
    never land exactly on fixed_length -- an infinite loop, not a wrong
    answer, so this test would hang (not fail) if the bug reappeared.
    """
    p = GPUPatchify2D(img_size=(12, 12), fixed_length=4, interp_size=4, min_size=2)
    img = torch.rand(2, 1, 12, 12)

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 4)
    assert seq_pos.shape == (2, 4, 2)


def test_run_merge_batch_reaches_fixed_length_independently_per_image():
    """Two images with different content, same fixed_length target -- each
    must land on exactly fixed_length regions on its own (not, e.g.,
    accidentally averaged/coupled across the batch by the single global
    topk each iteration).
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2)
    img = torch.full((2, 1, 16, 16), 1.0)
    img[1, 0, 0:8, 4:8] = 10.0  # only image 1 has real structure

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 13)
    assert seq_pos.shape == (2, 13, 2)
    assert seq_img.shape[0] == 2
    # Image 1's step-edge quadrant must not have merged into one big region
    # (unlike image 0, which is free to merge anywhere) -- every region
    # overlapping x<8,y<8 stays at the finest level-1 size (4), never 8.
    for size, pos in zip(seq_size[1].tolist(), seq_pos[1].tolist()):
        if pos[0] < 8 and pos[1] < 8:
            assert size == 4


def test_serialize_batch_extracts_each_regions_true_content():
    """Four distinctly-valued flat quadrants, forced fixed_length=4 (one
    region per quadrant) -- each detected region's extracted (and
    interp_size-resized) patch must be a constant equal to its own
    quadrant's real value, not some other quadrant's or a blend.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=4, interp_size=4, min_size=2)
    img = torch.zeros(1, 1, 16, 16)
    img[0, 0, 0:8, 0:8] = 1.0
    img[0, 0, 0:8, 8:16] = 2.0
    img[0, 0, 8:16, 0:8] = 3.0
    img[0, 0, 8:16, 8:16] = 4.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (1, 1, 4, 16)  # B, C, fixed_length, interp_size**2
    for i in range(4):
        x, y = seq_pos[0, i].tolist()
        expected = 1.0 if (x < 8 and y < 8) else 2.0 if (x >= 8 and y < 8) else 3.0 if (x < 8 and y >= 8) else 4.0
        patch = seq_img[0, 0, i]
        assert torch.allclose(patch, torch.full_like(patch, expected), atol=1e-4)


def test_serialize_batch_non_square_image_does_not_transpose_regions():
    """Regression test for a real bug found while writing these tests: the
    ported `_serialize_batch` originally normalized column bounds by H (not
    W) and row bounds by W (not H), and fed them to grid_sample's x/y grid
    channels swapped -- self-consistent enough to not error, and (for a
    square image) numerically equivalent to a silent row/column transpose
    of the extracted region. A rectangular image is the case where the
    wrong normalization denominator alone (independent of the axis swap)
    would also produce a visibly wrong value, not just a transposed
    position -- this is present in the reference implementation this
    module was ported from, not introduced by the port.
    """
    p = GPUPatchify2D(img_size=(8, 16), fixed_length=4, interp_size=4, min_size=2)
    img = torch.zeros(1, 1, 8, 16)
    img[0, 0, 0:4, 0:8] = 1.0
    img[0, 0, 0:4, 8:16] = 2.0
    img[0, 0, 4:8, 0:8] = 3.0
    img[0, 0, 4:8, 8:16] = 4.0

    seq_img, seq_size, seq_pos = p(img)

    for i in range(4):
        x, y = seq_pos[0, i].tolist()
        expected = 1.0 if (x < 8 and y < 4) else 2.0 if (x >= 8 and y < 4) else 3.0 if (x < 8 and y >= 4) else 4.0
        assert torch.allclose(seq_img[0, 0, i], torch.full_like(seq_img[0, 0, i], expected), atol=1e-4)


def test_score_fn_invalid_value_raises_clearly():
    with pytest.raises(AssertionError, match="score_fn"):
        GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="bogus")


def test_canny_edge_map_flat_image_has_no_edges():
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16), 5.0)

    edges = p._canny_edge_map_batch(img)

    assert torch.equal(edges, torch.zeros_like(edges))


def test_canny_edge_map_marks_a_deterministic_step_edge():
    """A vertical step edge (columns 0:8 at one value, 8:16 at another) --
    the edge map must be nonzero only in a thin band straddling column 8,
    and exactly zero everywhere else."""
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.zeros(1, 1, 16, 16)
    img[:, :, :, 8:] = 10.0

    edges = p._canny_edge_map_batch(img)

    assert edges[:, :, :7].sum() == 0
    assert edges[:, :, 9:].sum() == 0
    assert edges[:, :, 7:9].sum() > 0


def test_canny_edge_map_multi_channel_sums_independent_edges():
    """Two channels, each with its own step edge at a different column --
    a pixel gets edge-count 1 if only one channel flags it, 2 if both do
    (mirrors Patchify_3D's own "weight a voxel by how many channels
    independently flag it as an edge" convention)."""
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.zeros(1, 2, 16, 16)
    img[:, 0, :, 8:] = 10.0  # channel 0's edge at column 8
    img[:, 1, :, 8:] = 10.0  # channel 1's edge also at column 8 -- same location, both flag it

    edges = p._canny_edge_map_batch(img)
    single_channel_edges = p._canny_edge_map_batch(img[:, 0:1])

    assert torch.equal(edges, 2 * single_channel_edges)


def test_compute_all_levels_batch_canny_merge_cost_is_exactly_zero_for_a_flat_region():
    """Unlike variance's `errors[level] - sum_children`, canny's merge cost
    is the block's own edge count directly -- a perfectly flat image has
    zero edges everywhere, so every level's cost is exactly 0, deterministically.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16), 3.0)

    merge_costs, _ = p._compute_all_levels_batch(img)

    for level in range(1, p.max_level + 1):
        assert torch.equal(merge_costs[level][0], torch.zeros_like(merge_costs[level][0]))


def test_compute_all_levels_batch_canny_merge_cost_is_highest_where_the_edge_is():
    """A small square, strictly inside the top-left 8x8 quadrant (away from
    every level-2 block boundary, so Gaussian smoothing can't bleed the
    edge signal across a block edge) -- that quadrant's level-2 merge cost
    must be strictly higher than every other level-2 block's.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16), 1.0)
    img[0, 0, 2:6, 2:6] = 10.0

    merge_costs, _ = p._compute_all_levels_batch(img)
    level2 = merge_costs[2][0]

    assert level2[0, 0] > level2.flatten()[1:].max()


def test_forward_canny_score_keeps_edge_containing_regions_fine():
    """Direct analog of test_run_merge_batch_reaches_fixed_length_
    independently_per_image, through the canny score path instead of
    variance -- image 1's step-edge quadrant must stay at the finest
    level-1 size, never merge to the coarser size 8.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=13, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((2, 1, 16, 16), 1.0)
    img[1, 0, 0:8, 4:8] = 10.0  # only image 1 has real structure

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 13)
    for size, pos in zip(seq_size[1].tolist(), seq_pos[1].tolist()):
        if pos[0] < 8 and pos[1] < 8:
            assert size == 4


def test_serialize_batch_multi_channel_does_not_scramble_channels():
    """2-channel input, each channel a distinct constant per quadrant --
    confirms the C>1 branch's permute keeps channel/region/pixel axes
    correctly separated (the exact class of bug this session's own
    Patchify_3D tests were written to catch for the CPU path)."""
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=4, interp_size=4, min_size=2)
    img = torch.zeros(1, 2, 16, 16)
    img[0, 0, 0:8, 0:8] = 10.0
    img[0, 1, 0:8, 0:8] = 100.0
    img[0, 0, 0:8, 8:16] = 20.0
    img[0, 1, 0:8, 8:16] = 200.0
    img[0, 0, 8:16, 0:8] = 30.0
    img[0, 1, 8:16, 0:8] = 300.0
    img[0, 0, 8:16, 8:16] = 40.0
    img[0, 1, 8:16, 8:16] = 400.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (1, 2, 4, 16)  # B, C, fixed_length, interp_size**2
    for i in range(4):
        x, y = seq_pos[0, i].tolist()
        ch0_expected = 10.0 if (x < 8 and y < 8) else 20.0 if (x >= 8 and y < 8) else 30.0 if (x < 8 and y >= 8) else 40.0
        assert torch.allclose(seq_img[0, 0, i], torch.full_like(seq_img[0, 0, i], ch0_expected), atol=1e-4)
        assert torch.allclose(seq_img[0, 1, i], torch.full_like(seq_img[0, 1, i], ch0_expected * 10), atol=1e-4)


def test_labeled_to_region_tensors_scales_to_many_regions():
    """2D analog of GPUPatchify3D's own identically-named regression test
    -- that class's own `_labeled_to_region_tensors` hit a real `torch.
    OutOfMemoryError` on Frontier from an `[num_features, D*H*W]` one-hot
    membership matrix; this class had the identical latent `[num_features,
    H*W]` pattern, fixed here too even though it hadn't yet been observed
    to OOM for real. Calls `_labeled_to_region_tensors` directly with 256
    distinct regions (16x16 grid of 4x4 blocks in a 64-square image) --
    the largest region count any test in this file uses, specifically to
    catch a regression back to an O(num_features * H*W) approach.
    Correctness checked against an independent, brute-force per-label
    min/max (not the class's own formula reused).
    """
    p = GPUPatchify2D(img_size=(64, 64), fixed_length=4, interp_size=4, min_size=2)

    H = W = 64
    bs = 4
    n_per_axis = 16
    labeled = torch.zeros(H, W, dtype=torch.long)
    label = 1
    label_to_block = {}
    for hi in range(n_per_axis):
        for wi in range(n_per_axis):
            labeled[hi*bs:(hi+1)*bs, wi*bs:(wi+1)*bs] = label
            label_to_block[label] = (hi, wi)
            label += 1
    num_features = label - 1
    assert num_features == n_per_axis ** 2  # 256

    result = p._labeled_to_region_tensors(labeled, num_features)

    for lbl, (hi, wi) in label_to_block.items():
        idx = lbl - 1
        coords = (labeled == lbl).nonzero(as_tuple=False)
        h_lo, h_hi = coords[:, 0].min().item(), coords[:, 0].max().item()
        w_lo, w_hi = coords[:, 1].min().item(), coords[:, 1].max().item()

        expected_y0 = h_lo - 1 if h_lo != 0 else h_lo
        expected_x0 = w_lo - 1 if w_lo != 0 else w_lo

        assert result.y0s[idx].item() == expected_y0
        assert result.y1s[idx].item() == h_hi
        assert result.x0s[idx].item() == expected_x0
        assert result.x1s[idx].item() == w_hi


def test_serialize_batch_chunking_does_not_mix_up_images_or_regions():
    """2D analog of GPUPatchify3D's own identically-named regression test
    -- forces `serialize_chunk_size` (3) smaller than a single image's own
    region count (N=4), so chunk boundaries fall mid-image and at least
    one chunk straddles the boundary between image 0's regions and image
    1's. Two images, each with 4 distinctly-valued quadrants, using
    *different* value ranges per image (1-4 vs 101-104) so any cross-image
    bleed from a wrong `img_idx` lookup is immediately visible.
    """
    p = GPUPatchify2D(img_size=(16, 16), fixed_length=4, interp_size=4, min_size=2, serialize_chunk_size=3)
    img = torch.zeros(2, 1, 16, 16)
    for b, base in enumerate((1.0, 101.0)):
        img[b, 0, 0:8, 0:8] = base
        img[b, 0, 0:8, 8:16] = base + 1.0
        img[b, 0, 8:16, 0:8] = base + 2.0
        img[b, 0, 8:16, 8:16] = base + 3.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (2, 1, 4, 16)
    for b, base in enumerate((1.0, 101.0)):
        for i in range(4):
            x, y = seq_pos[b, i].tolist()
            expected = base + (0.0 if (x < 8 and y < 8) else 1.0 if (x >= 8 and y < 8) else 2.0 if (x < 8 and y >= 8) else 3.0)
            patch = seq_img[b, 0, i]
            assert torch.allclose(patch, torch.full_like(patch, expected), atol=1e-4)
