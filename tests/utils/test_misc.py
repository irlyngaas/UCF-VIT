import os

import pytest
import torch

from UCF_VIT.utils.misc import (
    calculate_tile_overlap,
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
