"""Tests for UCF_VIT.model.gpu_adaptive_patching.GPUPatchify3D.

3D generalization of GPUPatchify2D (see that module's own test file for
the full design rationale, shared unchanged here) -- 8 children merge
into 1 parent (not 4), block "borders" become block "faces" (2D planes on
a 3D boundary volume, not 1D lines), Canny's non-max suppression uses 13
canonical directions (not 4), and _serialize_batch uses a 5D (volumetric)
grid_sample call. Mirrors test_gpu_adaptive_patching.py's structure and
rigor exactly, extended to 3 axes.

Runs fully on CPU (torch, scipy.ndimage -- no CUDA, no dataset-specific
dependency).
"""

import pytest
import torch

from UCF_VIT.model.gpu_adaptive_patching import GPUPatchify3D


def test_pad_tensor_pads_to_coarsest_block_size():
    # img_size=(12,12,12), min_size=2 -> max_blocks=6, max_level=2,
    # coarsest block=2*4=8 -- 12 is not a multiple of 8, needs padding to
    # 16, which is also why the *real* level-0 grid is (16/2)**3=512
    # leaves, not 6**3=216 -- fixed_length must satisfy (512-fixed_length)%7==0.
    p = GPUPatchify3D(img_size=(12, 12, 12), fixed_length=8, interp_size=4, min_size=2)
    t = torch.arange(12 ** 3, dtype=torch.float32).reshape(1, 1, 12, 12, 12)

    padded, orig = p._pad_tensor(t)

    assert orig == (12, 12, 12)
    assert padded.shape[-3:] == (16, 16, 16)
    assert torch.equal(padded[0, 0, :12, :12, :12], t[0, 0])


def test_pad_tensor_no_op_when_already_aligned():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=1, interp_size=4, min_size=2)
    t = torch.rand(1, 1, 16, 16, 16)

    padded, orig = p._pad_tensor(t)

    assert orig == (16, 16, 16)
    assert torch.equal(padded, t)


def test_fixed_length_congruence_assertion():
    # img_size=(16,16,16), min_size=2 -> max_blocks=8, initial_leaves=512.
    # (512 - fixed_length) % 7 == 0 -> fixed_length in {..., 8, 15, 22, ...}.
    GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2)  # does not raise

    with pytest.raises(AssertionError, match="fixed_length"):
        GPUPatchify3D(img_size=(16, 16, 16), fixed_length=16, interp_size=4, min_size=2)


def test_forward_terminates_when_img_size_needs_padding():
    """3D analog of GPUPatchify2D's own regression test for a real hang
    (see that module's test file for the full story): the fixed_length
    congruence must reflect the *padded* grid, not the raw img_size //
    min_size -- this would hang (not fail) if that regressed."""
    p = GPUPatchify3D(img_size=(12, 12, 12), fixed_length=8, interp_size=4, min_size=2)
    img = torch.rand(2, 1, 12, 12, 12)

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 8)
    assert seq_pos.shape == (2, 8, 3)


def test_non_cuboid_requires_sorted_dimensions():
    with pytest.raises(NotImplementedError, match="img_size"):
        GPUPatchify3D(img_size=(16, 8, 16), fixed_length=1, interp_size=4, min_size=2)
    with pytest.raises(NotImplementedError, match="img_size"):
        GPUPatchify3D(img_size=(16, 16, 8), fixed_length=1, interp_size=4, min_size=2)


def test_score_fn_invalid_value_raises_clearly():
    with pytest.raises(AssertionError, match="score_fn"):
        GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="bogus")


# ---------------------------------------------------------------------------
# Variance scoring -- merge cost is exactly 0 for a flat region, and the law
# of total variance guarantees it can never be negative *as long as the
# tested region doesn't straddle a block boundary* (torch.var's unbiased
# n-1 correction can otherwise inflate small children's SSE above the
# parent's own -- the same documented, hand-confirmed property GPUPatchify2D's
# own tests already flagged, not a 3D-specific issue). Every fixture below
# deliberately keeps its "detail" region aligned to block boundaries, same
# reasoning as GPUPatchify2D's own step-edge tests.
# ---------------------------------------------------------------------------


def test_merge_cost_is_exactly_zero_for_a_flat_region():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2)
    img = torch.full((1, 1, 16, 16, 16), 3.0)

    merge_costs, _ = p._compute_all_levels_batch(img)

    for level in range(1, p.max_level + 1):
        assert torch.equal(merge_costs[level][0], torch.zeros_like(merge_costs[level][0]))


def test_merge_cost_prefers_merging_flat_region_over_a_step_cube():
    """A deterministic, block-boundary-aligned step (not noise, and not
    a region straddling a block boundary -- see this file's own module
    docstring above for why) inside one octant, flat everywhere else.
    """
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2)
    img = torch.full((1, 1, 16, 16, 16), 1.0)
    img[0, 0, 0:8, 0:8, 0:4] = 10.0  # step inside the (0,0,0) octant, aligned to its own midpoint

    merge_costs, _ = p._compute_all_levels_batch(img)

    level2 = merge_costs[2][0]
    assert level2[0, 0, 0] > 0
    assert torch.equal(level2.flatten()[1:], torch.zeros(level2.numel() - 1))


def test_run_merge_batch_reaches_fixed_length_independently_per_image():
    """Two images with different content, same fixed_length target -- each
    must land on exactly fixed_length regions on its own."""
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2)
    img = torch.full((2, 1, 16, 16, 16), 1.0)
    img[1, 0, 0:8, 0:8, 0:4] = 10.0  # only image 1 has real structure

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 15)
    assert seq_pos.shape == (2, 15, 3)
    assert seq_img.shape[0] == 2
    # Image 1's step-cube octant must not have merged into one big region --
    # every region overlapping x<8,y<8,z<8 stays at the finest level-1 size.
    for size, pos in zip(seq_size[1].tolist(), seq_pos[1].tolist()):
        x, y, z = pos
        if x < 8 and y < 8 and z < 8:
            assert size == 4


def test_serialize_batch_extracts_each_regions_true_content():
    """Eight distinctly-valued flat octants, forced fixed_length=8 (one
    region per octant) -- each detected region's extracted (and
    interp_size-resized) patch must be a constant equal to its own
    octant's real value, not some other octant's or a blend. The exact
    class of test that already caught a real axis-transpose bug in
    GPUPatchify2D's own _serialize_batch.
    """
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=8, interp_size=4, min_size=2)
    img = torch.zeros(1, 1, 16, 16, 16)
    val = 1.0
    for dd in (0, 1):
        for hh in (0, 1):
            for ww in (0, 1):
                img[0, 0, dd * 8:dd * 8 + 8, hh * 8:hh * 8 + 8, ww * 8:ww * 8 + 8] = val
                val += 1.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (1, 1, 8, 64)  # B, C, fixed_length, interp_size**3
    for i in range(8):
        x, y, z = seq_pos[0, i].tolist()
        dd, hh, ww = (1 if z >= 8 else 0), (1 if y >= 8 else 0), (1 if x >= 8 else 0)
        expected = dd * 4 + hh * 2 + ww + 1.0
        patch = seq_img[0, 0, i]
        assert torch.allclose(patch, torch.full_like(patch, expected), atol=1e-4)


def test_serialize_batch_non_cuboid_volume_does_not_transpose_axes():
    """Non-cuboid volume (D=8, H=8, W=16, i.e. W is twice H/D) -- the 3D
    analog of GPUPatchify2D's own non-square regression test, the case
    where a wrong axis mapping produces a visibly wrong value, not just a
    transposed position.
    """
    p = GPUPatchify3D(img_size=(8, 8, 16), fixed_length=8, interp_size=4, min_size=2)
    img = torch.zeros(1, 1, 8, 8, 16)
    val = 1.0
    for hh in (0, 1):
        for ww in (0, 1):
            img[0, 0, :, hh * 4:hh * 4 + 4, ww * 8:ww * 8 + 8] = val
            val += 1.0

    seq_img, seq_size, seq_pos = p(img)

    for i in range(8):
        x, y, z = seq_pos[0, i].tolist()
        hh, ww = (1 if y >= 4 else 0), (1 if x >= 8 else 0)
        expected = hh * 2 + ww + 1.0
        assert torch.allclose(seq_img[0, 0, i], torch.full_like(seq_img[0, 0, i], expected), atol=1e-4)


def test_serialize_batch_multi_channel_does_not_scramble_channels():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=8, interp_size=4, min_size=2)
    img = torch.zeros(1, 2, 16, 16, 16)
    val = 1.0
    for dd in (0, 1):
        for hh in (0, 1):
            for ww in (0, 1):
                img[0, 0, dd * 8:dd * 8 + 8, hh * 8:hh * 8 + 8, ww * 8:ww * 8 + 8] = val
                img[0, 1, dd * 8:dd * 8 + 8, hh * 8:hh * 8 + 8, ww * 8:ww * 8 + 8] = val * 10
                val += 1.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (1, 2, 8, 64)  # B, C, fixed_length, interp_size**3
    for i in range(8):
        x, y, z = seq_pos[0, i].tolist()
        dd, hh, ww = (1 if z >= 8 else 0), (1 if y >= 8 else 0), (1 if x >= 8 else 0)
        expected = dd * 4 + hh * 2 + ww + 1.0
        assert torch.allclose(seq_img[0, 0, i], torch.full_like(seq_img[0, 0, i], expected), atol=1e-4)
        assert torch.allclose(seq_img[0, 1, i], torch.full_like(seq_img[0, 1, i], expected * 10), atol=1e-4)


# ---------------------------------------------------------------------------
# Canny scoring -- 13-direction non-max suppression, hysteresis approximated
# via a few F.max_pool3d-based dilation passes (see GPUPatchify2D's own
# tests/README.md entry for the identical reasoning, extended to 3D).
# ---------------------------------------------------------------------------


def test_canny_edge_map_flat_volume_has_no_edges():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16, 16), 5.0)

    edges = p._canny_edge_map_batch(img)

    assert torch.equal(edges, torch.zeros_like(edges))


def test_canny_edge_map_marks_a_deterministic_step_plane():
    """A step plane (values 0 for w<8, 10 for w>=8) -- the edge map must be
    nonzero only in a thin band straddling column 8, and exactly zero
    everywhere else."""
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.zeros(1, 1, 16, 16, 16)
    img[:, :, :, :, 8:] = 10.0

    edges = p._canny_edge_map_batch(img)

    assert edges[:, :, :, :7].sum() == 0
    assert edges[:, :, :, 9:].sum() == 0
    assert edges[:, :, :, 7:9].sum() > 0


def test_canny_edge_map_multi_channel_sums_independent_edges():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.zeros(1, 2, 16, 16, 16)
    img[:, 0, :, :, 8:] = 10.0
    img[:, 1, :, :, 8:] = 10.0  # same location -- both channels flag it

    edges = p._canny_edge_map_batch(img)
    single_channel_edges = p._canny_edge_map_batch(img[:, 0:1])

    assert torch.equal(edges, 2 * single_channel_edges)


def test_compute_all_levels_batch_canny_merge_cost_is_exactly_zero_for_a_flat_region():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16, 16), 3.0)

    merge_costs, _ = p._compute_all_levels_batch(img)

    for level in range(1, p.max_level + 1):
        assert torch.equal(merge_costs[level][0], torch.zeros_like(merge_costs[level][0]))


def test_compute_all_levels_batch_canny_merge_cost_is_highest_where_the_edge_is():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((1, 1, 16, 16, 16), 1.0)
    img[0, 0, 0:8, 0:8, 0:4] = 10.0

    merge_costs, _ = p._compute_all_levels_batch(img)
    level2 = merge_costs[2][0]

    assert level2[0, 0, 0] > level2.flatten()[1:].max()


def test_forward_canny_score_keeps_edge_containing_regions_fine():
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, score_fn="canny")
    img = torch.full((2, 1, 16, 16, 16), 1.0)
    img[1, 0, 0:8, 0:8, 0:4] = 10.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_size.shape == (2, 15)
    for size, pos in zip(seq_size[1].tolist(), seq_pos[1].tolist()):
        x, y, z = pos
        if x < 8 and y < 8 and z < 8:
            assert size == 4


def test_region_backend_cupyx_without_cupy_installed_raises_import_error():
    # Real (not simulated) in this environment -- cupy genuinely isn't
    # installed here. See test_gpu_adaptive_patching.py's identical tests
    # for the full region_backend design rationale (shared _label_regions
    # helper, exercised there in depth) -- this file just confirms
    # GPUPatchify3D wires region_backend through the same way.
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=15, interp_size=4, min_size=2, region_backend="cupyx")
    img = torch.rand(1, 1, 16, 16, 16)

    with pytest.raises(ImportError, match="cupy"):
        p(img)


def test_labeled_to_region_tensors_scales_to_many_regions():
    """Real Frontier regression test: a real basic_ct/unetr do_gpu_ap:True
    run hit `torch.OutOfMemoryError: HIP out of memory. Tried to allocate
    64.00 GiB` inside the old implementation's `[num_features, D*H*W]`
    one-hot membership matrix -- fine for every existing test's small
    (fixed_length <= 15) region count, catastrophic at real image scale
    where both factors are large. Every existing test in this file only
    ever exercises a handful of regions (bounded by `fixed_length`); this
    calls `_labeled_to_region_tensors` directly with 512 distinct regions
    (8x8x8 grid of 4x4x4 blocks in a 32-cube volume) -- still tiny next to
    a real 256-cube volume's finest level, but the largest region count
    any test here uses, specifically to catch a regression back to an
    O(num_features * D*H*W) approach. Correctness checked against an
    independent, brute-force per-label min/max (not the class's own
    formula reused, so this can't trivially pass by construction).
    """
    p = GPUPatchify3D(img_size=(32, 32, 32), fixed_length=8, interp_size=4, min_size=2)

    D = H = W = 32
    bs = 4
    n_per_axis = 8
    labeled = torch.zeros(D, H, W, dtype=torch.long)
    label = 1
    label_to_block = {}
    for di in range(n_per_axis):
        for hi in range(n_per_axis):
            for wi in range(n_per_axis):
                labeled[di*bs:(di+1)*bs, hi*bs:(hi+1)*bs, wi*bs:(wi+1)*bs] = label
                label_to_block[label] = (di, hi, wi)
                label += 1
    num_features = label - 1
    assert num_features == n_per_axis ** 3  # 512

    result = p._labeled_to_region_tensors(labeled, num_features)

    for lbl, (di, hi, wi) in label_to_block.items():
        idx = lbl - 1
        # Independent brute-force reference -- true min/max coords for this
        # label, not the class's own bounds() formula.
        coords = (labeled == lbl).nonzero(as_tuple=False)
        d_lo, d_hi = coords[:, 0].min().item(), coords[:, 0].max().item()
        h_lo, h_hi = coords[:, 1].min().item(), coords[:, 1].max().item()
        w_lo, w_hi = coords[:, 2].min().item(), coords[:, 2].max().item()

        expected_z0 = d_lo - 1 if d_lo != 0 else d_lo
        expected_y0 = h_lo - 1 if h_lo != 0 else h_lo
        expected_x0 = w_lo - 1 if w_lo != 0 else w_lo

        assert result.z0s[idx].item() == expected_z0
        assert result.z1s[idx].item() == d_hi
        assert result.y0s[idx].item() == expected_y0
        assert result.y1s[idx].item() == h_hi
        assert result.x0s[idx].item() == expected_x0
        assert result.x1s[idx].item() == w_hi


def test_serialize_batch_chunking_does_not_mix_up_images_or_regions():
    """Real regression test for the OOM fix's own chunking loop: with
    `serialize_chunk_size` forced smaller than a single image's own region
    count, chunk boundaries fall mid-image (chunk_size=3 doesn't evenly
    divide either N=8 or B*N=16) and at least one chunk straddles the
    boundary between image 0's regions and image 1's -- exactly the case
    that would silently mix up which image a region's patch comes from if
    `img_idx`'s indexing (not `imgs_batch`'s own repeat_interleave
    ordering) were wrong. Two images, each with 8 distinctly-valued
    octants, using *different* value ranges per image (1-8 vs 101-108) so
    any cross-image bleed is immediately visible, not coincidentally
    correct.
    """
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=8, interp_size=4, min_size=2, serialize_chunk_size=3)
    img = torch.zeros(2, 1, 16, 16, 16)
    for b, base in enumerate((1.0, 101.0)):
        val = base
        for dd in (0, 1):
            for hh in (0, 1):
                for ww in (0, 1):
                    img[b, 0, dd*8:dd*8+8, hh*8:hh*8+8, ww*8:ww*8+8] = val
                    val += 1.0

    seq_img, seq_size, seq_pos = p(img)

    assert seq_img.shape == (2, 1, 8, 64)
    for b, base in enumerate((1.0, 101.0)):
        for i in range(8):
            x, y, z = seq_pos[b, i].tolist()
            dd, hh, ww = (1 if z >= 8 else 0), (1 if y >= 8 else 0), (1 if x >= 8 else 0)
            expected = base + dd * 4 + hh * 2 + ww
            patch = seq_img[b, 0, i]
            assert torch.allclose(patch, torch.full_like(patch, expected), atol=1e-4)


def test_region_backend_defaults_to_scipy_with_no_env_var(monkeypatch):
    monkeypatch.delenv("UCF_VIT_GPU_AP_REGION_BACKEND", raising=False)
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=1, interp_size=4, min_size=2)
    assert p.region_backend == "scipy"


def test_region_backend_reads_env_var_when_not_passed_explicitly(monkeypatch):
    # Same mechanism as GPUPatchify2D's own identical test -- see that
    # class's own region_backend docstring for why (real production path
    # never passes region_backend at all; this env var lets a real
    # training job try "cupyx" with no code or config change).
    monkeypatch.setenv("UCF_VIT_GPU_AP_REGION_BACKEND", "cupyx")
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=1, interp_size=4, min_size=2)
    assert p.region_backend == "cupyx"


def test_region_backend_explicit_value_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("UCF_VIT_GPU_AP_REGION_BACKEND", "cupyx")
    p = GPUPatchify3D(img_size=(16, 16, 16), fixed_length=1, interp_size=4, min_size=2, region_backend="scipy")
    assert p.region_backend == "scipy"
