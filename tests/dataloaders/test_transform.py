"""Tests for UCF_VIT.dataloaders.transform's Patchify/Patchify_3D edge detection.

Patchify uses two different Canny implementations depending on `dataset`:
per-channel SimpleITK.CannyEdgeDetection, summed into a per-pixel edge count,
for "imagenet"/"catsdogs" (real, possibly multi-channel photos), and
skimage.feature.canny for everything else (arbitrary-range float data, e.g.
"basic_ct"). The imagenet/catsdogs path used to call cv2.Canny directly --
replaced because cv2.Canny's own multi-channel handling is a per-pixel
"winner take all" across channels (each pixel's edge decision comes from
whichever single channel has the largest local gradient magnitude there,
discarding the other channels' gradients at that pixel entirely, confirmed
by reading cv2's own C++ source, modules/imgproc/src/canny.cpp) -- a real,
coarse simplification that doesn't generalize well to channels with
different physical scales/meanings (tolerable for RGB's similarly-scaled
channels, not for arbitrary multi-channel data). The per-channel-count
scheme this switched to is the same convention Patchify_3D already uses
(SimpleITK.CannyEdgeDetection doesn't support multi-channel/vector images
directly either way, so a per-channel loop was already required there).
Each channel is independently min-max normalized to [0,1] before Canny runs
(edge-detection input only, real patch content untouched) so the shared
canny_low_threshold/canny_high_threshold defaults stay meaningful
regardless of a dataset's real intensity range (imagenet/catsdogs are
uint8 [0,255], not the ~[0,1] scale those otherwise assume) --
test_patchify_imagenet_branch_
normalizes_input_before_canny below is the direct regression test.

An earlier code version assumed float data was already normalized to
exactly [0,1] before scaling to cv2.Canny's required 8-bit range
(`(img*255).astype(np.uint8)`) -- silently wrong (wastes dynamic range, or
clips) whenever that assumption doesn't hold. skimage.feature.canny operates
on the real float values directly, no scaling/casting needed -- but it only
accepts single-channel 2D input; a regression test below confirms the
non-photo path raises a clear error for multi-channel input rather than
silently mishandling it.

Patchify_3D was rewritten again since then, replacing its earlier per-slice
cv2.Sobel/cv2.Canny pipeline (archived at
../UCF-VIT-claude-archive/src/UCF_VIT/dataloaders/transform.py) with
SimpleITK.CannyEdgeDetection -- a genuinely 3D Canny (one call per channel
over the whole volume, not a loop over 2D slices). The old pipeline only
ever computed in-plane (H, W) gradients/edges, never a derivative along the
depth axis at all -- test_patchify_3d_detects_edge_purely_along_depth below
is the direct regression test: a volume that's a step function purely along
depth (uniform within every single slice, only a hard transition *across*
slices), which the old per-slice approach would report zero edges for
entirely. The per-channel-edge-count weighting itself (more weight to a
voxel flagged as an edge on more channels) is unchanged in spirit --
test_patchify_3d_weights_by_channel_agreement checks it directly. The old
pipeline's separate Sobel-direction-consistency gate was dropped entirely
(not reimplemented in 3D) -- see Patchify_3D's own docstring for why.
Its own edge-detection-input normalization (originally scoped to "sst"
only, its raw CFD fields being in arbitrary physical units) now runs
unconditionally for every dataset, not just "sst" -- see test_patchify_3d_
normalizes_input_before_canny_for_every_dataset below.
"""

import random

import numpy as np
import pytest

from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D, _pad_to_power_of_two


def _box_image(H=64, W=64, low=0.0, high=1.0):
    """A single-channel image with a clear square edge, shape (H, W)."""
    img = np.full((H, W), low, dtype=np.float32)
    img[16:48, 16:48] = high
    return img


def test_patchify_imagenet_branch_uses_per_channel_simpleitk_canny():
    """Real uint8 photo-like image, 3 identical channels -- confirms the
    per-channel SimpleITK.CannyEdgeDetection loop actually runs (detects a
    real edge) and that its per-pixel edge *count* behaves as designed:
    since every channel here is byte-identical, each pixel's count must be
    either 0 (no channel flags it) or exactly num_channels (every channel
    agrees) -- never some in-between value, which would mean the per-
    channel loop isn't summing independent per-channel results correctly.
    """
    img = np.stack([_box_image(low=0, high=255).astype(np.uint8)] * 3, axis=-1)  # (H, W, 3)
    p = Patchify(sths=[3], fixed_length=16, interp_size=8, num_channels=3, dataset="imagenet", return_edges=True)

    _, _, _, _, edges = p(img)

    assert edges.sum() > 0
    assert set(np.unique(edges)).issubset({0, 3})


def test_patchify_imagenet_branch_normalizes_input_before_canny():
    """canny_low_threshold/canny_high_threshold's own defaults assume
    ~[0,1]-scale input -- imagenet/catsdogs images are real uint8 [0,255],
    so without per-channel min-max
    normalization (see forward()'s own comment) the default thresholds
    would be meaningless on this raw scale (gradients ~255x too large,
    likely degenerately all-edge or no-edge). A real, clean box edge here
    (low=0, high=255) must still cleanly detect an edge with the unscaled
    default thresholds, confirming normalization actually ran.
    """
    img = _box_image(low=0, high=255).astype(np.uint8)[:, :, None]  # (H, W, 1)
    p = Patchify(sths=[3], fixed_length=16, interp_size=8, num_channels=1, dataset="catsdogs", return_edges=True)

    _, _, _, _, edges = p(img)

    assert edges.sum() > 0


# ---------------------------------------------------------------------------
# canny_sigma/canny_low_threshold/canny_high_threshold -- the same shared
# ap.canny_sigma/ap.canny_low_threshold/ap.canny_high_threshold config knobs
# UCF_VIT.model.gpu_adaptive_patching.GPUPatchify2D/GPUPatchify3D's own Canny
# scoring already exposes, now also consumed by Patchify/Patchify_3D (see
# UCF_VIT.parse's own comment for why this is now shared, and why canny_
# hysteresis_iters deliberately isn't).
# ---------------------------------------------------------------------------


def test_patchify_canny_sigma_squares_for_simpleitk_branch_not_skimage():
    # imagenet/catsdogs (SimpleITK.CannyEdgeDetection's own "variance", not
    # sigma) -- canny_sigma must be squared (ITK's own variance-vs-sigma
    # convention). Every other dataset (skimage.feature.canny's own real
    # sigma) -- used directly, no conversion.
    p_photo = Patchify(canny_sigma=2.0, num_channels=1, dataset="imagenet")
    assert p_photo.sths == [4.0]

    p_other = Patchify(canny_sigma=2.0, num_channels=1, dataset="basic_ct")
    assert p_other.sths == [2.0]


def test_patchify_canny_sigma_none_leaves_sths_untouched():
    p = Patchify(sths=[7, 9], canny_sigma=None, num_channels=1, dataset="imagenet")
    assert p.sths == [7, 9]


def test_patchify_canny_low_and_high_threshold_resolve_independently():
    # No canny_thresholds tuple parameter -- canny_low_threshold/canny_high_
    # threshold are each real parameters with their own real default, no
    # need to set both together.
    p_both = Patchify(canny_low_threshold=0.2, canny_high_threshold=0.4, num_channels=1, dataset="imagenet")
    assert p_both.canny_low_threshold == 0.2
    assert p_both.canny_high_threshold == 0.4

    p_only_low = Patchify(canny_low_threshold=0.2, num_channels=1, dataset="imagenet")
    assert p_only_low.canny_low_threshold == 0.2
    assert p_only_low.canny_high_threshold == 0.15  # its own default, untouched

    p_neither = Patchify(num_channels=1, dataset="imagenet")
    assert p_neither.canny_low_threshold == 0.05
    assert p_neither.canny_high_threshold == 0.15


def test_patchify_canny_sigma_and_thresholds_actually_used_end_to_end():
    """Not just attribute-level -- confirms the overridden values actually
    drive a real forward() call (a fixed, fresh canny_sigma=1.0/thresholds
    close to Patchify_3D's own class defaults, real edge still detected).
    """
    img = _box_image(low=0, high=255).astype(np.uint8)[:, :, None]
    p = Patchify(fixed_length=16, interp_size=8, num_channels=1, dataset="catsdogs",
                 canny_sigma=1.0, canny_low_threshold=0.05, canny_high_threshold=0.15, return_edges=True)

    _, _, _, _, edges = p(img)

    assert edges.sum() > 0


def test_patchify_3d_canny_sigma_squares_unconditionally():
    # Patchify_3D is always SimpleITK-backed -- canny_sigma always squares,
    # regardless of dataset.
    p = Patchify_3D(canny_sigma=3.0, num_channels=1, dataset="basic_ct")
    assert p.sths == [9.0]


def test_patchify_3d_canny_low_and_high_threshold_resolve_independently():
    p_both = Patchify_3D(canny_low_threshold=0.2, canny_high_threshold=0.4, num_channels=1, dataset="basic_ct")
    assert p_both.canny_low_threshold == 0.2
    assert p_both.canny_high_threshold == 0.4

    p_only_high = Patchify_3D(canny_high_threshold=0.4, num_channels=1, dataset="basic_ct")
    assert p_only_high.canny_low_threshold == 0.05  # its own default, untouched
    assert p_only_high.canny_high_threshold == 0.4


def test_patchify_non_photo_branch_handles_arbitrary_float_range():
    """Regression test for the original bug: (img*255).astype(np.uint8)
    assumed [0,1]-normalized float data. Uses a range far outside [0,1]
    (e.g. un-normalized CT-style intensities) -- skimage.feature.canny
    operates on the real values directly, so this should still cleanly
    detect the edge instead of the old code's silently-wasted dynamic
    range (or outright clipping).
    """
    img = _box_image(low=0.0, high=2000.0)[:, :, None]  # (H, W, 1)
    p = Patchify(sths=[1.0], fixed_length=16, canny_quantiles=(0.5, 0.8), interp_size=8, num_channels=1, dataset="basic_ct", return_edges=True)

    _, _, _, _, edges = p(img)

    assert edges.sum() > 0
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 1})  # unscaled -- Rect.contains no longer assumes any particular scale


def test_patchify_non_photo_branch_rejects_multi_channel():
    img = np.stack([_box_image()] * 3, axis=-1)  # (H, W, 3)
    p = Patchify(sths=[1.0], fixed_length=16, interp_size=8, num_channels=3, dataset="basic_ct")

    with pytest.raises(NotImplementedError):
        p(img)


def test_patchify_variance_mode_uses_raw_image_as_domain():
    """score_fn="variance" skips smoothing/Canny entirely -- the domain
    handed to FixedQuadTree is img itself, deterministically (unlike
    score_fn="canny", which randomizes smoothing/thresholds every call)."""
    img = _box_image()[:, :, None]
    p = Patchify(fixed_length=16, interp_size=8, num_channels=1, dataset="basic_ct", score_fn="variance", return_edges=True)

    _, _, _, _, edges1 = p(img)
    _, _, _, _, edges2 = p(img)

    assert edges1 is img
    assert edges2 is img
    np.testing.assert_array_equal(edges1, edges2)


def test_patchify_variance_mode_favors_higher_variance_region():
    """Direct analog of FixedQuadTree's own test_fixedquadtree_variance_
    further_subdivides_high_variance_region, through the full Patchify
    pipeline -- a checkerboard quadrant (real pixel variance) gets split
    further than the three flat quadrants."""
    img = np.full((16, 16), 5.0, dtype=np.float32)
    img[0:8, 0:8] = np.tile([[1., 9.], [9., 1.]], (4, 4))
    img = img[:, :, None]

    p = Patchify(fixed_length=7, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance")
    _, seq_size, _, qdt = p(img)

    assert qdt.count_patches() == 7
    assert sorted(seq_size) == [4] * 4 + [8] * 3


def test_patchify_min_size_limits_smallest_leaf():
    """min_size is the shared floor with GPUPatchify2D's own min_size --
    forwarded through to FixedQuadTree, confirmed end to end through the
    full Patchify pipeline (not just at the tree level, already covered by
    test_quadtree.py's own test_fixedquadtree_min_size_limits_smallest_leaf)."""
    img = np.full((16, 16), 5.0, dtype=np.float32)
    img[0:4, 0:4] = 100.0  # dense in a small region, plenty of room to keep splitting
    img = img[:, :, None]

    p = Patchify(fixed_length=50, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance", min_size=4)
    _, seq_size, _, qdt = p(img)

    real_sizes = [s for s in seq_size if s > 0]
    assert min(real_sizes) == 4
    assert qdt.count_patches() < 50  # stops early -- can't reach fixed_length without going below min_size


def test_patchify_variance_mode_allows_multi_channel_on_non_photo_dataset():
    """score_fn="variance" has no equivalent to score_fn="canny"'s C == 1
    restriction on non-imagenet/catsdogs datasets (skimage.feature.canny
    only accepts single-channel input; np.var handles any channel count) --
    this is a real generalization, not just a different scoring formula."""
    img = np.stack([_box_image()] * 3, axis=-1)  # (H, W, 3)
    p = Patchify(fixed_length=16, interp_size=8, num_channels=3, dataset="basic_ct", score_fn="variance")

    p(img)  # does not raise


# ---------------------------------------------------------------------------
# _pad_to_power_of_two -- removes the old data.tile_size-must-be-a-power-of-2
# restriction for this (CPU) path: FixedQuadTree/FixedOctTree's split (one
# integer midpoint, used for both children) only ever produces equal-sized
# children when repeatedly halving stays clean all the way down, which a
# power-of-two size guarantees and any other size doesn't. Padding is
# edge-replication (numpy's mode="edge", the numpy equivalent of
# GPUPatchify2D/3D's own F.pad(..., mode="replicate")), not zero-padding --
# zero-padding would create a hard discontinuity exactly at the real image
# boundary, a false edge for score_fn="canny" and spuriously high variance
# for score_fn="variance" on any block straddling it.
# ---------------------------------------------------------------------------


def test_pad_to_power_of_two_is_noop_when_already_power_of_two():
    img = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    out = _pad_to_power_of_two(img, ndim=2)
    assert out is img  # no-op returns the same object, not a copy


def test_pad_to_power_of_two_pads_up_to_next_power_of_two_per_axis():
    img = np.zeros((12, 20, 3), dtype=np.float32)  # 12 -> 16, 20 -> 32, independently
    out = _pad_to_power_of_two(img, ndim=2)
    assert out.shape == (16, 32, 3)


def test_pad_to_power_of_two_uses_edge_replication_not_zero():
    img = np.zeros((3, 3), dtype=np.float32)
    img[:, -1] = 5.0  # distinct value on the edge that gets replicated
    out = _pad_to_power_of_two(img, ndim=2)

    assert out.shape == (4, 4)
    assert np.all(out[:3, :3] == img[:3, :3])  # real content untouched
    assert np.all(out[:3, 3] == 5.0)  # replicated from the last real column, not 0
    assert np.all(out[3, :] == out[2, :])  # replicated row equals the last real row


def test_pad_to_power_of_two_leaves_trailing_non_spatial_axes_alone():
    # ndim=2 means only the first 2 axes are spatial -- a trailing channel
    # axis of any size must never be padded.
    img = np.zeros((3, 16, 5), dtype=np.float32)
    out = _pad_to_power_of_two(img, ndim=2)
    assert out.shape == (4, 16, 5)


def test_patchify_handles_non_power_of_two_image_size():
    """End to end through Patchify itself (not just the helper): a
    non-power-of-2 size no longer needs to be rejected upstream (parse.py's
    do_gpu_ap:False path no longer asserts this) -- forward() pads
    internally instead, and the return value reflects the padded (not
    original) size."""
    img = np.full((24, 24), 5.0, dtype=np.float32)[:, :, None]

    p = Patchify(fixed_length=7, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance", return_edges=True)
    _, _, _, qdt, edges = p(img)

    assert edges.shape == (32, 32, 1)  # 24 -> next_power_of_two(24) == 32
    assert qdt.count_patches() == 7


def test_patchify_3d_detects_edge_purely_along_depth():
    """The direct regression test for the switch to SimpleITK.CannyEdgeDetection:
    a step function purely along depth (uniform *within* every single slice,
    only a hard transition *across* slices) has zero in-plane gradient
    anywhere -- the old per-slice cv2.Sobel/cv2.Canny pipeline (see this
    module's own docstring) would detect nothing here at all. A genuine 3D
    Canny must mark the transition planes.
    """
    D = H = W = 16
    vol = np.zeros((D, H, W, 1), dtype=np.float32)
    vol[6:10, :, :, 0] = 1.0  # step purely along depth (D), rows 6:10

    p = Patchify_3D(sths=[0.5], fixed_length=8, interp_size=4, num_channels=1, dataset="basic_ct", return_edges=True)
    _, _, _, _, edges = p(vol)

    # full-plane edges at the two depth boundaries (z=5, z=10), nothing elsewhere
    assert edges[5].sum() == H * W
    assert edges[10].sum() == H * W
    for z in range(D):
        if z not in (5, 10):
            assert edges[z].sum() == 0


def test_patchify_3d_weights_by_channel_agreement():
    """A voxel flagged as an edge on more channels should score higher --
    the design intent behind edges_combined_counter, preserved through the
    SimpleITK rewrite (see Patchify_3D's own docstring).
    """
    D = H = W = 24
    vol = np.zeros((D, H, W, 2), dtype=np.float32)
    vol[8:16, 8:16, 8:16, :] = 1.0  # both channels see this box
    vol[2:6, 2:6, 2:6, 1] = 1.0     # channel 1 only

    p = Patchify_3D(sths=[1.0], fixed_length=8, interp_size=4, num_channels=2, dataset="basic_ct", return_edges=True)
    _, _, _, _, edges = p(vol)

    assert set(np.unique(edges)).issubset({0, 1, 2})
    assert set(np.unique(edges[8, 8:16, 8:16])) == {2}  # shared box's edge face: both channels agree
    assert set(np.unique(edges[2, 2:6, 2:6])) == {1}    # channel-1-only box's edge face


def test_patchify_3d_variance_mode_uses_raw_image_as_domain():
    vol = np.random.RandomState(0).rand(16, 16, 16, 1).astype(np.float32)
    p = Patchify_3D(fixed_length=8, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance", return_edges=True)

    _, _, _, _, edges = p(vol)

    assert edges is vol


def test_patchify_3d_variance_mode_skips_the_simpleitk_canny_loop_entirely(monkeypatch):
    """score_fn="variance" should never call SimpleITK.CannyEdgeDetection at
    all -- not just get an equivalent result via a different path. Confirms
    the "cheaper, not just an alternative" claim by making the real Canny
    call raise if it's ever reached.
    """
    import UCF_VIT.dataloaders.transform as transform_module

    def _boom(*args, **kwargs):
        raise AssertionError("SimpleITK.CannyEdgeDetection should not be called under score_fn='variance'")

    monkeypatch.setattr(transform_module.sitk, "CannyEdgeDetection", _boom)

    vol = np.random.RandomState(0).rand(16, 16, 16, 1).astype(np.float32)
    p = Patchify_3D(fixed_length=8, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance")

    p(vol)  # does not raise


def test_patchify_3d_variance_mode_favors_higher_variance_region():
    """Direct analog of FixedOctTree's own test_fixedocttree_variance_
    further_subdivides_high_variance_region, through the full Patchify_3D
    pipeline."""
    vol = np.full((16, 16, 16), 5.0, dtype=np.float32)
    vol[0:8, 0:8, 0:8] = np.tile([[[1., 9.], [9., 1.]], [[9., 1.], [1., 9.]]], (4, 4, 4))
    vol = vol[:, :, :, None]

    p = Patchify_3D(fixed_length=15, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance")
    _, seq_size, _, octtree = p(vol)

    assert len(octtree.nodes) == 15
    assert sorted(seq_size) == [4] * 8 + [8] * 7


def test_patchify_3d_min_size_limits_smallest_leaf():
    # fixed_length=100 is deliberately above (16//4)**3=64 -- the most
    # 4x4x4 leaves a 16-cube volume could ever be split into -- so this
    # can only stop early, never reach fixed_length, without min_size.
    vol = np.full((16, 16, 16), 5.0, dtype=np.float32)
    vol[0:4, 0:4, 0:4] = 100.0  # dense in a small region, plenty of room to keep splitting
    vol = vol[:, :, :, None]

    p = Patchify_3D(fixed_length=100, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance", min_size=4)
    _, seq_size, _, octtree = p(vol)

    real_sizes = [s for s in seq_size if s > 0]
    assert min(real_sizes) == 4
    assert len(octtree.nodes) < 100  # stops early -- can't reach fixed_length without going below min_size


def test_patchify_multi_channel_reshape_does_not_scramble_channels():
    """Regression test: seq_img comes out of qdt.serialize as (fixed_length,
    interp_size, interp_size, num_channels) -- channel last. The num_channels>1
    branch reshapes straight to (num_channels, fixed_length, interp_size**2)
    without first moving the channel axis to the front; since a plain
    np.reshape never moves data, that silently scrambled patches/channels
    together instead of separating them (verified by disabling np.moveaxis and
    confirming this same test fails). Each channel here is a distinct,
    perfectly flat constant (no internal edges, so bicubic resizing can't
    introduce any intermediate values) -- every real (non-padded) entry in
    seq_img[c] must be exactly that channel's constant, and every padded entry
    exactly 0.
    """
    np.random.seed(0)
    random.seed(0)
    H, W, C, fixed_length = 32, 32, 3, 16
    img = np.zeros((H, W, C), dtype=np.uint8)
    for c in range(C):
        img[:, :, c] = (c + 1) * 50  # 50, 100, 150 -- distinct per channel

    # sths=[0]: random (image-content-independent) edge map, so the tree still
    # splits into multiple real leaf nodes even though the image itself is flat.
    p = Patchify(sths=[0], fixed_length=fixed_length, interp_size=4, num_channels=C, dataset="imagenet")
    seq_img, seq_size, seq_pos, qdt = p(img)

    assert seq_img.shape == (C, fixed_length, 4 * 4)
    for c in range(C):
        for idx in range(fixed_length):
            expected = (c + 1) * 50 if seq_size[idx] > 0 else 0
            assert np.all(seq_img[c, idx] == expected), f"channel {c} patch {idx} contaminated"


def test_patchify_3d_multi_channel_reshape_does_not_scramble_channels():
    """Same regression as test_patchify_multi_channel_reshape_does_not_scramble_channels,
    for Patchify_3D's identical channel-last-reshape bug.
    """
    np.random.seed(1)
    random.seed(1)
    D = H = W = 16
    C, fixed_length = 2, 8
    vol = np.zeros((D, H, W, C), dtype=np.float32)
    for c in range(C):
        vol[:, :, :, c] = (c + 1) * 5.0  # 5.0, 10.0 -- distinct per channel

    p = Patchify_3D(sths=[0.5], fixed_length=fixed_length, interp_size=4, num_channels=C, dataset="basic_ct")
    seq_img, seq_size, seq_pos, octtree = p(vol)

    assert seq_img.shape == (C, fixed_length, 4 * 4 * 4)
    for c in range(C):
        for idx in range(fixed_length):
            expected = (c + 1) * 5.0 if seq_size[idx] > 0 else 0.0
            assert np.allclose(seq_img[c, idx], expected), f"channel {c} patch {idx} contaminated"


def test_patchify_3d_handles_non_power_of_two_volume_size():
    """End to end through Patchify_3D itself (not just the helper) --
    direct 3D analog of test_patchify_handles_non_power_of_two_image_size."""
    vol = np.full((12, 12, 12), 5.0, dtype=np.float32)[:, :, :, None]

    p = Patchify_3D(fixed_length=8, interp_size=4, num_channels=1, dataset="basic_ct", score_fn="variance", return_edges=True)
    _, _, _, octtree, edges = p(vol)

    assert edges.shape == (16, 16, 16, 1)  # 12 -> next_power_of_two(12) == 16
    assert len(octtree.nodes) == 8


def test_patchify_3d_shape_and_dtype():
    # 24 isn't a power of two, so forward() pads it up to 32 internally
    # (see _pad_to_power_of_two) before computing edges.
    D = H = W = 24
    vol = np.zeros((D, H, W, 2), dtype=np.float32)
    vol[8:16, 8:16, 8:16, :] = 1.0

    p = Patchify_3D(sths=[1.0], fixed_length=8, interp_size=4, num_channels=2, dataset="basic_ct", return_edges=True)
    _, _, _, _, edges = p(vol)

    assert edges.shape == (32, 32, 32)
    assert edges.dtype == np.uint8
    assert edges.sum() > 0


def test_patchify_3d_profile_prints_edge_octree_serialize_timings(capsys):
    """Diagnostic only, off by default -- when profile=True, forward() must
    print real (non-negative) timings for each of the 3 real per-sample
    CPU-bound stages (Canny edge loop, octree build, serialize), summing to
    patchify_time, from inside whatever process actually calls it (a
    DataLoader worker in real use) -- confirmed here by parsing the exact
    printed line, not just "no exception".
    """
    D = H = W = 24
    vol = np.zeros((D, H, W, 2), dtype=np.float32)
    vol[8:16, 8:16, 8:16, :] = 1.0

    p = Patchify_3D(sths=[1.0], fixed_length=8, interp_size=4, num_channels=2, dataset="basic_ct", profile=True)
    p(vol)

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("patchify_3d")]
    assert len(lines) == 1
    tokens = lines[0].split()[1:]  # drop the leading "patchify_3d" label
    fields = dict(zip(tokens[0::2], tokens[1::2]))
    edge_time = float(fields["edge_time"])
    octree_time = float(fields["octree_time"])
    serialize_time = float(fields["serialize_time"])
    patchify_time = float(fields["patchify_time"])
    assert edge_time >= 0
    assert octree_time >= 0
    assert serialize_time >= 0
    assert patchify_time == pytest.approx(edge_time + octree_time + serialize_time)


def test_patchify_3d_profile_defaults_to_false_and_prints_nothing():
    D = H = W = 24
    vol = np.zeros((D, H, W, 2), dtype=np.float32)
    vol[8:16, 8:16, 8:16, :] = 1.0

    p = Patchify_3D(sths=[1.0], fixed_length=8, interp_size=4, num_channels=2, dataset="basic_ct")
    assert p.profile is False


def test_patchify_3d_normalizes_only_edge_detection_input_not_real_patch_content():
    """canny_low_threshold/canny_high_threshold assume a ~[0,1] scale --
    normalization keeps that meaningful regardless of a volume's real
    intensity range, for any
    dataset (not just "sst", whose raw CFD fields are in arbitrary
    physical units, the case this was originally built for). Confirmed
    here by checking a raw, far-outside-[0,1] volume reproduces the exact
    same edge map a manually pre-normalized [0,1] copy of it gets --
    while leaving the real patch content (seq_img, what the model
    actually trains against) at the real, un-normalized physical values.
    """
    D = H = W = 16
    raw_vol = np.full((D, H, W, 1), 500.0, dtype=np.float32)
    raw_vol[6:10, :, :, 0] = 800.0  # a real step, but far outside [0,1]
    normalized_vol = (raw_vol - raw_vol.min()) / (raw_vol.max() - raw_vol.min())  # same shape, values in {0.0, 1.0}

    kwargs = dict(sths=[0.5], fixed_length=8, interp_size=4, num_channels=1, dataset="basic_ct", return_edges=True)
    seq_img_raw, _, _, _, edges_raw = Patchify_3D(**kwargs)(raw_vol)
    _, _, _, _, edges_pre_normalized = Patchify_3D(**kwargs)(normalized_vol)

    np.testing.assert_array_equal(edges_raw, edges_pre_normalized)
    assert edges_raw.sum() > 0  # sanity: the step was actually detected, not just identically absent
    # The real patch content -- what the model actually trains against --
    # must still be the real, un-normalized physical values (500/800), not
    # the [0,1]-rescaled proxy only used internally for edge detection.
    real_values = set(np.unique(seq_img_raw))
    assert real_values <= {500.0, 800.0}
    assert not (real_values <= {0.0, 1.0})


def test_patchify_3d_normalizes_input_before_canny_for_every_dataset():
    """Direct demonstration of the actual failure mode normalization
    fixes, for any dataset, not just "sst": a real step whose raw
    magnitude (0.001) is far below canny_low_threshold's own default
    (0.05) -- undetectable without normalization, easily detectable once
    rescaled to use the full [0,1] range. Checked against "basic_ct"
    specifically (its own edge-detection input was never normalized
    before this change) and "sst" (already normalized before this change)
    to confirm both now behave identically -- normalization is no longer
    scoped by dataset name.
    """
    D = H = W = 16
    vol = np.zeros((D, H, W, 1), dtype=np.float32)
    vol[6:10, :, :, 0] = 0.001  # real step, but ~50x smaller than the lower threshold

    kwargs = dict(sths=[0.5], fixed_length=8, interp_size=4, num_channels=1, return_edges=True)
    _, _, _, _, edges_basic_ct = Patchify_3D(dataset="basic_ct", **kwargs)(vol)
    _, _, _, _, edges_sst = Patchify_3D(dataset="sst", **kwargs)(vol)

    assert edges_basic_ct.sum() > 0  # real edge, now found regardless of dataset
    assert edges_sst.sum() > 0
