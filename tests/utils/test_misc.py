import os

import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image

from UCF_VIT.utils.misc import (
    calculate_tile_overlap,
    detect_img_size,
    detect_num_channels,
    is_power_of_two,
    patchify,
    process_root_dirs,
    shard_attention_state_dict,
    shard_mlp_state_dict,
    unpatchify,
)


@pytest.mark.parametrize(
    "n,expected",
    [(0, False), (1, True), (2, True), (3, False), (4, True), (5, False), (16, True), (1023, False), (1024, True)],
)
def test_is_power_of_two(n, expected):
    assert is_power_of_two(n) is expected


def test_calculate_tile_overlap_even():
    start, end = calculate_tile_overlap((4, 4))
    assert start == [2, 2]
    assert end == [2, 2]


def test_calculate_tile_overlap_odd():
    start, end = calculate_tile_overlap((5, 3))
    assert start == [2, 1]
    assert end == [3, 2]


def test_calculate_tile_overlap_zero():
    start, end = calculate_tile_overlap((0, 0, 0))
    assert start == [0, 0, 0]
    assert end == [0, 0, 0]


def test_patchify_unpatchify_roundtrip_2d():
    torch.manual_seed(0)
    patch_size = 4
    data = torch.randn(2, 3, 8, 8)  # 2x2 grid of patches

    patches = patchify(data, patch_size, twoD=True)
    assert patches.shape == (2, 4, patch_size * patch_size * 3)

    recon = unpatchify(patches, data, patch_size, twoD=True)
    assert recon.shape == data.shape
    torch.testing.assert_close(recon, data)


def test_patchify_unpatchify_roundtrip_3d():
    torch.manual_seed(0)
    patch_size = 4
    data = torch.randn(1, 2, 8, 8, 8)  # 2x2x2 grid of patches

    patches = patchify(data, patch_size, twoD=False)
    assert patches.shape == (1, 8, patch_size ** 3 * 2)

    recon = unpatchify(patches, data, patch_size, twoD=False)
    assert recon.shape == data.shape
    torch.testing.assert_close(recon, data)


# ---------------------------------------------------------------------------
# process_root_dirs
# ---------------------------------------------------------------------------


def _make_imagenet_dir(tmp_path, num_classes, images_per_class):
    root = tmp_path / "imagenet_root"
    for c in range(num_classes):
        cdir = root / f"class{c:03d}"
        cdir.mkdir(parents=True)
        for i in range(images_per_class):
            (cdir / f"img{i}.JPEG").write_text("")
    return str(root)


def test_process_root_dirs_imagenet_more_classes_than_data_par_size_evenly_divisible(tmp_path):
    root = _make_imagenet_dir(tmp_path, num_classes=16, images_per_class=2)
    result = process_root_dirs("imagenet", {"imagenet": root}, data_par_size=4)

    assert set(result.keys()) == {0, 1, 2, 3}  # exactly data_par_size buckets
    for bucket in result.values():
        assert len(bucket) == 8  # 4 classes/bucket (16/4) * 2 images/class
    # every image appears in exactly one bucket, none dropped (evenly divisible)
    all_images = [img for bucket in result.values() for img in bucket]
    assert len(all_images) == len(set(all_images)) == 32


def test_process_root_dirs_imagenet_more_classes_than_data_par_size_not_evenly_divisible(tmp_path):
    """Documents current (not fixed here -- see the TODO in process_root_dirs
    itself) behavior: leftover classes past data_par_size * classes_to_combine
    are silently dropped, not distributed among the buckets.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=10, images_per_class=1)
    result = process_root_dirs("imagenet", {"imagenet": root}, data_par_size=4)

    assert set(result.keys()) == {0, 1, 2, 3}  # still exactly data_par_size buckets
    classes_to_combine = 10 // 4  # == 2, per process_root_dirs' own formula
    for bucket in result.values():
        assert len(bucket) == classes_to_combine
    all_images = [img for bucket in result.values() for img in bucket]
    assert len(all_images) == classes_to_combine * 4 == 8  # last 2 of 10 classes dropped


@pytest.mark.parametrize("num_classes", [1, 4, 7, 8])
def test_process_root_dirs_imagenet_classes_at_or_below_data_par_size(tmp_path, num_classes):
    """Regression test: process_root_dirs used to raise UnboundLocalError
    for len(classes) <= data_par_size (classes_to_combine was only assigned
    in the len(classes) > data_par_size branch). Fixed to combine 1 class
    per bucket in that case, giving len(classes) buckets instead of
    data_par_size -- matches this function's own docstring ("data_par_size
    (or fewer) buckets"). num_classes == data_par_size (8) is included since
    it's the boundary the buggy `>` comparison got wrong.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=num_classes, images_per_class=3)
    result = process_root_dirs("imagenet", {"imagenet": root}, data_par_size=8)

    assert set(result.keys()) == set(range(num_classes))  # one bucket per class
    for bucket in result.values():
        assert len(bucket) == 3  # exactly that one class's images, uncombined
    all_images = [img for bucket in result.values() for img in bucket]
    assert len(all_images) == len(set(all_images)) == num_classes * 3  # nothing dropped


def test_process_root_dirs_imagenet_bucket_contents_are_correct_classes(tmp_path):
    """Beyond counts: each bucket must contain the *right* classes' images,
    combined in sorted-class order, not just the right totals.
    """
    root = tmp_path / "imagenet_root"
    # class00 and class01 -> bucket 0; class02 and class03 -> bucket 1
    for c in range(4):
        cdir = root / f"class{c:02d}"
        cdir.mkdir(parents=True)
        (cdir / "img0.JPEG").write_text("")

    result = process_root_dirs("imagenet", {"imagenet": str(root)}, data_par_size=2)

    assert os.path.basename(os.path.dirname(result[0][0])) == "class00"
    assert os.path.basename(os.path.dirname(result[0][1])) == "class01"
    assert os.path.basename(os.path.dirname(result[1][0])) == "class02"
    assert os.path.basename(os.path.dirname(result[1][1])) == "class03"


def test_process_root_dirs_non_imagenet_lists_imagesTr(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    for i in range(3):
        (images_dir / f"image{i}.nii").write_text("")

    result = process_root_dirs("basic_ct", {"ct1": str(tmp_path)}, data_par_size=8)

    assert set(result.keys()) == {"ct1"}  # keyed by dict_root_dirs key, not bucket index
    assert len(result["ct1"]) == 3


# ---------------------------------------------------------------------------
# detect_num_channels
# ---------------------------------------------------------------------------


def test_detect_num_channels_imagenet_hardcodes_three_without_touching_filesystem(tmp_path):
    """imagenet always forces 3 channels via dataset.py's
    Image.open(path).convert("RGB") -- no file needs to exist for this to
    return the right answer.
    """
    result = detect_num_channels("imagenet", {"imagenet": str(tmp_path / "does-not-exist")})
    assert result == {"imagenet": 3}


@pytest.mark.parametrize("mode,expected_channels", [("RGB", 3), ("L", 1), ("RGBA", 4)])
def test_detect_num_channels_catsdogs_reads_real_file_band_count(tmp_path, mode, expected_channels):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    Image.new(mode, (4, 4)).save(images_dir / "image0.png")

    result = detect_num_channels("catsdogs", {"catsdogs": str(tmp_path)})

    assert result == {"catsdogs": expected_channels}


def test_detect_num_channels_basic_ct_reads_real_3d_file_as_one_channel(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine=np.eye(4)), images_dir / "image0.nii")

    result = detect_num_channels("basic_ct", {"ct1": str(tmp_path)})

    assert result == {"ct1": 1}


def test_detect_num_channels_basic_ct_raises_for_ambiguous_4d_shape(tmp_path):
    """No shipped config or test exercises basic_ct with num_channels > 1,
    so there's no verified channel-axis convention to infer a 4D+ shape
    from -- must raise rather than silently guess.
    """
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 2), dtype=np.float32), affine=np.eye(4)), images_dir / "image0.nii")

    with pytest.raises(RuntimeError, match="ambiguous"):
        detect_num_channels("basic_ct", {"ct1": str(tmp_path)})


def test_detect_num_channels_raises_for_missing_imagesTr_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_num_channels("catsdogs", {"catsdogs": str(tmp_path)})


def test_detect_num_channels_multiple_keys_detected_independently(tmp_path):
    """Different dataset keys can genuinely have different channel counts
    (e.g. one grayscale source, one RGB source) -- each key's detection
    must be independent, not short-circuited by the first key checked.
    """
    grey_dir = tmp_path / "grey"
    (grey_dir / "imagesTr").mkdir(parents=True)
    Image.new("L", (4, 4)).save(grey_dir / "imagesTr" / "image0.png")

    rgb_dir = tmp_path / "rgb"
    (rgb_dir / "imagesTr").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(rgb_dir / "imagesTr" / "image0.png")

    result = detect_num_channels("catsdogs", {"grey": str(grey_dir), "rgb": str(rgb_dir)})

    assert result == {"grey": 1, "rgb": 3}


# ---------------------------------------------------------------------------
# detect_img_size
# ---------------------------------------------------------------------------


def test_detect_img_size_basic_ct_reads_real_native_shape(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((10, 20, 30), dtype=np.float32), affine=np.eye(4)), images_dir / "image0.nii")

    result = detect_img_size("basic_ct", {"ct1": str(tmp_path)})

    assert result == [10, 20, 30]


def test_detect_img_size_imagenet_reads_real_file_native_pixel_size(tmp_path):
    """Deliberately non-square (12x7, not e.g. 12x12) so this actually
    verifies the [width, height] order rather than trivially passing on a
    square fixture -- PIL's Image.size is (width, height), and that's the
    order dataset.py's own cv.resize(..., dsize=[resize[0], resize[1]])
    call already uses unswapped (cv2's dsize is itself (width, height)),
    so detect_img_size must match it, not datamodule.py's own (incorrect)
    docstring claim of [height, width].
    """
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    Image.new("RGB", (12, 7)).save(images_dir / "image0.png")

    result = detect_img_size("imagenet", {"imagenet": str(tmp_path)})

    assert result == [12, 7]


def test_detect_img_size_catsdogs_reads_real_file_native_pixel_size(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    Image.new("RGB", (12, 7)).save(images_dir / "image0.png")

    result = detect_img_size("catsdogs", {"catsdogs": str(tmp_path)})

    assert result == [12, 7]


def test_detect_img_size_uses_first_dict_root_dirs_key_only(tmp_path):
    """img_size is a single value shared across the whole dataset (unlike
    num_channels' per-key dict) -- detection samples only the first key,
    mirroring parse.py's own "first key wins" num_channels-to-in_chans
    convention.
    """
    first_dir = tmp_path / "first"
    (first_dir / "imagesTr").mkdir(parents=True)
    Image.new("RGB", (12, 7)).save(first_dir / "imagesTr" / "image0.png")

    second_dir = tmp_path / "second"
    (second_dir / "imagesTr").mkdir(parents=True)
    Image.new("RGB", (99, 99)).save(second_dir / "imagesTr" / "image0.png")

    result = detect_img_size("catsdogs", {"first": str(first_dir), "second": str(second_dir)})

    assert result == [12, 7]


def test_detect_img_size_raises_for_missing_imagesTr_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_img_size("catsdogs", {"catsdogs": str(tmp_path)})


def test_detect_img_size_raises_for_empty_dict_root_dirs():
    with pytest.raises(FileNotFoundError):
        detect_img_size("catsdogs", {})


# ---------------------------------------------------------------------------
# shard_mlp_state_dict / shard_attention_state_dict
# ---------------------------------------------------------------------------


def _fake_mlp_state_dict(in_features, hidden_features, out_features):
    torch.manual_seed(0)
    return {
        "fc1.weight": torch.randn(hidden_features, in_features),
        "fc1.bias": torch.randn(hidden_features),
        "fc2.weight": torch.randn(out_features, hidden_features),
        "fc2.bias": torch.randn(out_features),
    }


@pytest.mark.parametrize("tensor_par_size", [1, 2, 4])
def test_shard_mlp_state_dict_reconstructs_full_weights(tensor_par_size):
    full = _fake_mlp_state_dict(in_features=8, hidden_features=16, out_features=8)

    shards = [shard_mlp_state_dict(full, tensor_par_size, r) for r in range(tensor_par_size)]

    torch.testing.assert_close(torch.cat([s["fc1.weight"] for s in shards], dim=0), full["fc1.weight"])
    torch.testing.assert_close(torch.cat([s["fc1.bias"] for s in shards], dim=0), full["fc1.bias"])
    torch.testing.assert_close(torch.cat([s["fc2.weight"] for s in shards], dim=1), full["fc2.weight"])


@pytest.mark.parametrize("tensor_par_size", [1, 2, 4])
def test_shard_mlp_state_dict_fc2_bias_sums_back_to_full(tensor_par_size):
    """The property Mlp.forward's post-fc2 all-reduce relies on: summing every
    shard's fc2.bias across the tensor-parallel group must reproduce the
    original, unsharded bias exactly (not tensor_par_size copies of it).
    """
    full = _fake_mlp_state_dict(in_features=8, hidden_features=16, out_features=8)

    shards = [shard_mlp_state_dict(full, tensor_par_size, r) for r in range(tensor_par_size)]

    summed_bias = sum(s["fc2.bias"] for s in shards)
    torch.testing.assert_close(summed_bias, full["fc2.bias"])
    # exactly one shard carries the real values, the rest are exactly zero
    nonzero = [r for r, s in enumerate(shards) if not torch.all(s["fc2.bias"] == 0)]
    assert nonzero == [0]


def _fake_attention_state_dict(dim):
    torch.manual_seed(0)
    return {
        "qkv.weight": torch.randn(dim * 3, dim),
        "qkv.bias": torch.randn(dim * 3),
        "proj.weight": torch.randn(dim, dim),
        "proj.bias": torch.randn(dim),
    }


def _expected_qkv_shard(full_qkv, dim, num_heads, tensor_par_size, tp_rank):
    """Reference reimplementation of the head-range slicing
    shard_attention_state_dict's qkv.weight/qkv.bias must match: reshape the
    flat (3 * dim, ...) tensor into (3, num_heads, head_dim, ...), select
    this rank's head range from every one of the 3 (Q/K/V) blocks, and
    flatten back -- the inverse of Attention.forward's own per-rank
    `.reshape(B, N, 3, num_heads // tensor_par_size, head_dim)` of a
    *sharded* qkv output.
    """
    head_dim = dim // num_heads
    heads_per_shard = num_heads // tensor_par_size
    head_start, head_end = tp_rank * heads_per_shard, (tp_rank + 1) * heads_per_shard
    trailing_shape = full_qkv.shape[1:]
    reshaped = full_qkv.reshape(3, num_heads, head_dim, *trailing_shape)
    sliced = reshaped[:, head_start:head_end]
    return sliced.reshape(3 * heads_per_shard * head_dim, *trailing_shape)


@pytest.mark.parametrize("tensor_par_size", [1, 2, 4])
def test_shard_attention_state_dict_slices_qkv_by_head_range_not_flat_chunk(tensor_par_size):
    """Regression test for a real bug found on a real Frontier run (job
    5341031): shard_attention_state_dict used to take one flat contiguous
    row-slice of qkv's (dim * 3)-sized output. qkv's output is actually 3
    contiguous dim-sized blocks (Q, K, V), each internally split into
    num_heads head-groups -- a flat row-slice mixes head ranges across
    Q/K/V for tensor_par_size > 1 (e.g. all of Q plus part of K on early
    ranks, part of K plus all of V on late ranks) instead of the SAME head
    range from each of Q, K, and V independently, which is exactly what
    Attention.forward's own per-rank reshape(..., 3, num_heads //
    tensor_par_size, head_dim) requires. This produced a 99.9%-of-elements
    mismatch in test_tensor_parallel_correctness.py's real multi-rank
    Attention check before being fixed.
    """
    dim = 16
    num_heads = 4
    full = _fake_attention_state_dict(dim=dim)

    for tp_rank in range(tensor_par_size):
        shard = shard_attention_state_dict(full, num_heads, tensor_par_size, tp_rank)
        torch.testing.assert_close(
            shard["qkv.weight"],
            _expected_qkv_shard(full["qkv.weight"], dim, num_heads, tensor_par_size, tp_rank),
        )
        torch.testing.assert_close(
            shard["qkv.bias"],
            _expected_qkv_shard(full["qkv.bias"], dim, num_heads, tensor_par_size, tp_rank),
        )


@pytest.mark.parametrize("tensor_par_size", [1, 2, 4])
def test_shard_attention_state_dict_reconstructs_full_proj_weight(tensor_par_size):
    """proj's input-dim column-slice IS a plain contiguous chunk (unlike
    qkv's row-slice) -- see shard_attention_state_dict's docstring for why
    head ranges and contiguous column ranges coincide there.
    """
    dim = 16
    num_heads = 4
    full = _fake_attention_state_dict(dim=dim)

    shards = [shard_attention_state_dict(full, num_heads, tensor_par_size, r) for r in range(tensor_par_size)]

    torch.testing.assert_close(torch.cat([s["proj.weight"] for s in shards], dim=1), full["proj.weight"])


@pytest.mark.parametrize("tensor_par_size", [1, 2, 4])
def test_shard_attention_state_dict_proj_bias_sums_back_to_full(tensor_par_size):
    full = _fake_attention_state_dict(dim=16)

    shards = [shard_attention_state_dict(full, num_heads=4, tensor_par_size=tensor_par_size, tp_rank=r) for r in range(tensor_par_size)]

    summed_bias = sum(s["proj.bias"] for s in shards)
    torch.testing.assert_close(summed_bias, full["proj.bias"])
    nonzero = [r for r, s in enumerate(shards) if not torch.all(s["proj.bias"] == 0)]
    assert nonzero == [0]


def test_shard_attention_state_dict_rejects_qk_norm_params():
    full = _fake_attention_state_dict(dim=16)
    full["q_norm.weight"] = torch.randn(4)

    with pytest.raises(NotImplementedError):
        shard_attention_state_dict(full, num_heads=4, tensor_par_size=2, tp_rank=0)
