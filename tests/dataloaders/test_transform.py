"""Tests for UCF_VIT.dataloaders.transform's Patchify/Patchify_3D edge detection.

Patchify uses two different Canny implementations depending on `dataset`:
cv2.Canny for "imagenet"/"catsdogs" (real, already-uint8 photos, possibly
multi-channel -- unchanged by this session's work), and skimage.feature.canny
for everything else (arbitrary-range float data, e.g. "basic_ct"). The old
code assumed that float data was already normalized to exactly [0,1] before
scaling to cv2.Canny's required 8-bit range (`(img*255).astype(np.uint8)`) --
silently wrong (wastes dynamic range, or clips) whenever that assumption
doesn't hold. skimage.feature.canny operates on the real float values
directly, no scaling/casting needed -- but it only accepts single-channel 2D
input, unlike cv2.Canny, which silently combines multi-channel gradients; a
regression test below confirms the non-photo path raises a clear error for
multi-channel input rather than silently mishandling it.

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
"""

import random

import numpy as np
import pytest

from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D


def _box_image(H=64, W=64, low=0.0, high=1.0):
    """A single-channel image with a clear square edge, shape (H, W)."""
    img = np.full((H, W), low, dtype=np.float32)
    img[16:48, 16:48] = high
    return img


def test_patchify_imagenet_branch_uses_cv2_canny_unchanged():
    # Real uint8 photo-like image, possibly multi-channel -- untouched by
    # this change; sanity-checks it still runs and detects a real edge.
    img = np.stack([_box_image(low=0, high=255).astype(np.uint8)] * 3, axis=-1)  # (H, W, 3)
    p = Patchify(sths=[3], fixed_length=16, cannys=[50, 100], interp_size=8, num_channels=3, dataset="imagenet", return_edges=True)

    _, _, _, _, edges = p(img)

    assert edges.sum() > 0


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

    p = Patchify_3D(sths=[0.5], fixed_length=8, canny_thresholds=(0.05, 0.15), interp_size=4, num_channels=1, dataset="basic_ct", return_edges=True)
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

    p = Patchify_3D(sths=[1.0], fixed_length=8, canny_thresholds=(0.05, 0.15), interp_size=4, num_channels=2, dataset="basic_ct", return_edges=True)
    _, _, _, _, edges = p(vol)

    assert set(np.unique(edges)).issubset({0, 1, 2})
    assert set(np.unique(edges[8, 8:16, 8:16])) == {2}  # shared box's edge face: both channels agree
    assert set(np.unique(edges[2, 2:6, 2:6])) == {1}    # channel-1-only box's edge face


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
    p = Patchify(sths=[0], fixed_length=fixed_length, cannys=[50, 100], interp_size=4, num_channels=C, dataset="imagenet")
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

    p = Patchify_3D(sths=[0.5], fixed_length=fixed_length, canny_thresholds=(0.05, 0.15), interp_size=4, num_channels=C, dataset="basic_ct")
    seq_img, seq_size, seq_pos, octtree = p(vol)

    assert seq_img.shape == (C, fixed_length, 4 * 4 * 4)
    for c in range(C):
        for idx in range(fixed_length):
            expected = (c + 1) * 5.0 if seq_size[idx] > 0 else 0.0
            assert np.allclose(seq_img[c, idx], expected), f"channel {c} patch {idx} contaminated"


def test_patchify_3d_shape_and_dtype():
    D = H = W = 24
    vol = np.zeros((D, H, W, 2), dtype=np.float32)
    vol[8:16, 8:16, 8:16, :] = 1.0

    p = Patchify_3D(sths=[1.0], fixed_length=8, canny_thresholds=(0.05, 0.15), interp_size=4, num_channels=2, dataset="basic_ct", return_edges=True)
    _, _, _, _, edges = p(vol)

    assert edges.shape == (D, H, W)
    assert edges.dtype == np.uint8
    assert edges.sum() > 0
