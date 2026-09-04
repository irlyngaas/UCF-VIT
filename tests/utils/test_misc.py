import os

import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image

from UCF_VIT.utils.misc import (
    bucket_file_list,
    calculate_load_balancing_on_the_fly,
    calculate_tile_bounds,
    calculate_tile_overlap,
    detect_img_size,
    detect_num_channels,
    is_power_of_two,
    patchify,
    process_root_dirs,
    shard_attention_state_dict,
    shard_mlp_state_dict,
    slice_file_list,
    unpatchify,
)


@pytest.mark.parametrize(
    "n,expected",
    [(0, False), (1, True), (2, True), (3, False), (4, True), (5, False), (16, True), (1023, False), (1024, True)],
)
def test_is_power_of_two(n, expected):
    assert is_power_of_two(n) is expected


def test_slice_file_list_matches_fixed_length_reader_formula():
    # Mirrors FileReader.__init__'s own slice exactly -- see its docstring/
    # implementation (UCF_VIT.dataloaders.dataset.FileReader).
    files = [f"f{i}" for i in range(10)]
    assert slice_file_list(files, 0.0, 1.0) == files
    assert slice_file_list(files, 0.0, 0.5) == files[:5]
    assert slice_file_list(files, 0.5, 1.0) == files[5:]
    assert slice_file_list(files, 0.8, 0.9) == files[8:9]


def test_slice_file_list_empty_range_returns_empty_list():
    files = [f"f{i}" for i in range(10)]
    assert slice_file_list(files, 0.9, 0.9) == []


def _make_basic_ct_dir(tmp_path, num_files):
    root = tmp_path / "ct_root"
    images_tr = root / "imagesTr"
    images_tr.mkdir(parents=True)
    for i in range(num_files):
        (images_tr / f"img{i:03d}.nii.gz").write_text("")
    return str(root)


def _basic_ct_conf(root, data_par_size, allow_file_reuse, batch_size=1):
    return {
        "data": {
            "dict_root_dirs": {"ct1": root},
            "img_size": [64, 64, 64],
            "tile_size": (64, 64, 64),
            "twoD": False,
            "dataset": "basic_ct",
            "patch_size": 16,
            "interp_size": None,
        },
        "dataloader": {
            "dict_start_idx": {"ct1": 0.0},
            "dict_end_idx": {"ct1": 1.0},
            "batch_size": batch_size,
            "num_workers": 1,
            "allow_file_reuse": allow_file_reuse,
        },
        "tiling": {"div": 1},
        "ap": {"do_ap": False},
        "parallelism": {"data_par_size": data_par_size},
    }


def test_calculate_load_balancing_raises_when_not_enough_files_and_reuse_disabled(tmp_path):
    root = _make_basic_ct_dir(tmp_path, num_files=3)
    conf = _basic_ct_conf(root, data_par_size=5, allow_file_reuse=False)
    with pytest.raises(AssertionError, match="not all GPUs have at least one image"):
        calculate_load_balancing_on_the_fly(conf)


def test_calculate_load_balancing_floors_to_one_when_reuse_allowed(tmp_path):
    root = _make_basic_ct_dir(tmp_path, num_files=3)
    conf = _basic_ct_conf(root, data_par_size=5, allow_file_reuse=True)
    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)
    # Every rank gets at least 1 image (reused) -> at least 1 batch/rank/epoch,
    # not the 0 that would otherwise make the whole dataset key unusable.
    assert batches_per_rank_epoch["ct1"] >= 1


def test_calculate_load_balancing_raises_when_zero_batches_per_rank(tmp_path):
    # 8 files / data_par_size=8 -> exactly 1 image/rank, enough to clear the
    # "at least one image per rank" check above -- but batch_size=2 needs 2
    # images/rank for even one full batch (drop_last=True), so
    # batches_per_rank_epoch["ct1"] floors to 0. Without this assert, that
    # propagates into a bare ZeroDivisionError deep in
    # NativePytorchDataModule._compute_keys_to_add instead of a clear
    # message here -- the real failure mode that surfaced on Frontier when
    # dataloader.val_split_ratio/test_split_ratio's automatic split narrowed
    # an already-tight basic_ct allocation below one batch/rank.
    root = _make_basic_ct_dir(tmp_path, num_files=8)
    conf = _basic_ct_conf(root, data_par_size=8, allow_file_reuse=False, batch_size=2)
    with pytest.raises(AssertionError, match="0 batches per rank"):
        calculate_load_balancing_on_the_fly(conf)


def test_calculate_load_balancing_raises_on_zero_files_even_with_reuse_allowed(tmp_path):
    root = _make_basic_ct_dir(tmp_path, num_files=0)
    conf = _basic_ct_conf(root, data_par_size=1, allow_file_reuse=True)
    with pytest.raises(AssertionError, match="zero files"):
        calculate_load_balancing_on_the_fly(conf)


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


def test_calculate_tile_bounds_div_one_returns_full_dimension():
    """div == 1 means no tiling -- (0, tile_size) regardless of tile_idx or
    overlap, exercised by both UCF_VIT.dataloaders.dataset.TileDataIter and
    UCF_VIT.datasets.catsdogs.CatsDogsDataset as their shared "no tiling"
    default.
    """
    assert calculate_tile_bounds(tile_idx=0, div=1, tile_size=64, overlap_start=4, overlap_end=4) == (0, 64)


def test_calculate_tile_bounds_no_overlap():
    div = 4
    tile_size = 16
    bounds = [calculate_tile_bounds(i, div, tile_size, 0, 0) for i in range(div)]
    assert bounds == [(0, 16), (16, 32), (32, 48), (48, 64)]


def test_calculate_tile_bounds_with_overlap_first_middle_last():
    """First tile only gets extra overlap on its right/end edge, the last
    tile only on its left/start edge, and middle tiles get it on both --
    matching adjacent tiles' edges without extending past the image on the
    outer edges.
    """
    div = 3
    tile_size = 10
    overlap_start, overlap_end = 2, 3

    first = calculate_tile_bounds(0, div, tile_size, overlap_start, overlap_end)
    middle = calculate_tile_bounds(1, div, tile_size, overlap_start, overlap_end)
    last = calculate_tile_bounds(2, div, tile_size, overlap_start, overlap_end)

    assert first == (0, 10 + overlap_start * 2)
    assert middle == (10 - overlap_start, 20 + overlap_end)
    assert last == (20 - overlap_end * 2, 30)


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


def test_process_root_dirs_imagenet_lists_every_image_unbucketed(tmp_path):
    """process_root_dirs no longer buckets imagenet by data_par_size (that used
    to make which images count as train/val/test depend on data_par_size --
    see NativePytorchDataModule.__init__'s own comment) -- one entry per
    dict_root_dirs key, same shape as the non-imagenet branch, with every
    image present exactly once, regardless of data_par_size.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=16, images_per_class=2)
    result = process_root_dirs("imagenet", {"imagenet": root})

    assert set(result.keys()) == {"imagenet"}
    assert len(result["imagenet"]) == len(set(result["imagenet"])) == 32


def test_process_root_dirs_imagenet_order_is_sorted_class_then_image(tmp_path):
    """Beyond counts: images must come out in deterministic sorted-class,
    sorted-image-within-class order (not glob's own unspecified order) -- the
    determinism the whole train/val/test split now depends on.
    """
    root = tmp_path / "imagenet_root"
    for c in range(4):
        cdir = root / f"class{c:02d}"
        cdir.mkdir(parents=True)
        (cdir / "img0.JPEG").write_text("")

    result = process_root_dirs("imagenet", {"imagenet": str(root)})

    classes_in_order = [os.path.basename(os.path.dirname(p)) for p in result["imagenet"]]
    assert classes_in_order == ["class00", "class01", "class02", "class03"]


@pytest.mark.parametrize("num_classes", [1, 4, 7, 8])
def test_process_root_dirs_imagenet_data_par_size_argument_is_ignored(tmp_path, num_classes):
    """Regardless of data_par_size passed (even omitted), the full image list
    is always returned unbucketed -- data_par_size only affects bucket_file_list
    now, called later, after train/val/test membership is already resolved.
    """
    root = _make_imagenet_dir(tmp_path, num_classes=num_classes, images_per_class=3)
    result_no_arg = process_root_dirs("imagenet", {"imagenet": root})
    result_with_arg = process_root_dirs("imagenet", {"imagenet": root}, data_par_size=8)

    assert result_no_arg == result_with_arg
    assert len(result_no_arg["imagenet"]) == num_classes * 3


# ---------------------------------------------------------------------------
# bucket_file_list
# ---------------------------------------------------------------------------


def test_bucket_file_list_evenly_divisible():
    files = [f"f{i}" for i in range(12)]
    result = bucket_file_list(files, num_buckets=4)

    assert set(result.keys()) == {0, 1, 2, 3}
    assert all(len(bucket) == 3 for bucket in result.values())
    all_files = [f for bucket in result.values() for f in bucket]
    assert sorted(all_files) == sorted(files)  # every file present exactly once, none dropped


def test_bucket_file_list_uneven_division_keeps_every_file():
    # Unlike the old per-class bucketing this replaced, no file is ever
    # silently dropped for not dividing evenly -- np.array_split gives the
    # first `remainder` buckets one extra item instead.
    files = [f"f{i}" for i in range(10)]
    result = bucket_file_list(files, num_buckets=4)

    assert set(result.keys()) == {0, 1, 2, 3}
    sizes = sorted(len(bucket) for bucket in result.values())
    assert sizes == [2, 2, 3, 3]
    all_files = [f for bucket in result.values() for f in bucket]
    assert sorted(all_files) == files


def test_bucket_file_list_preserves_order_within_and_across_buckets():
    files = [f"f{i}" for i in range(6)]
    result = bucket_file_list(files, num_buckets=3)

    assert result[0] == ["f0", "f1"]
    assert result[1] == ["f2", "f3"]
    assert result[2] == ["f4", "f5"]


def test_bucket_file_list_caps_buckets_to_file_count_not_empty_buckets():
    """Regression guard: more buckets requested than files available must
    never produce an empty bucket purely from over-bucketing (that would trip
    FileReader's zero-files guard even under allow_file_reuse, which is meant
    for "not enough files for this many ranks", not "zero files for this key
    at all") -- capped to len(file_list) buckets instead.
    """
    files = ["f0", "f1", "f2"]
    result = bucket_file_list(files, num_buckets=10)

    assert set(result.keys()) == {0, 1, 2}
    assert all(len(bucket) == 1 for bucket in result.values())


def test_bucket_file_list_empty_input_returns_single_empty_bucket():
    # A single {0: []} (not {}) so downstream zero-files asserts still fire
    # clearly instead of silently iterating zero buckets.
    assert bucket_file_list([], num_buckets=5) == {0: []}


def test_bucket_file_list_shuffle_seed_is_deterministic():
    files = [f"f{i}" for i in range(20)]
    first = bucket_file_list(files, num_buckets=4, shuffle_seed=42)
    second = bucket_file_list(files, num_buckets=4, shuffle_seed=42)
    assert first == second


def test_bucket_file_list_shuffle_seed_still_covers_every_file_once():
    files = [f"f{i}" for i in range(20)]
    result = bucket_file_list(files, num_buckets=4, shuffle_seed=42)
    all_files = [f for bucket in result.values() for f in bucket]
    assert sorted(all_files) == sorted(files)


def test_bucket_file_list_shuffle_seed_breaks_up_contiguous_order():
    # Without a seed, bucketing is a contiguous split -- bucket 0 gets the
    # first N/num_buckets elements verbatim. With a seed, it shouldn't.
    files = [f"f{i}" for i in range(20)]
    unshuffled = bucket_file_list(files, num_buckets=4)
    shuffled = bucket_file_list(files, num_buckets=4, shuffle_seed=42)
    assert unshuffled[0] == files[:5]
    assert shuffled[0] != files[:5]


def test_bucket_file_list_no_seed_preserves_contiguous_order():
    files = [f"f{i}" for i in range(20)]
    result = bucket_file_list(files, num_buckets=4)
    assert result[0] == files[:5]
    assert result[1] == files[5:10]


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
    """Deliberately non-square (width=12, height=7, not e.g. 12x12) so this
    actually verifies the [height, width] order rather than trivially
    passing on a square fixture -- PIL's Image.size is natively
    (width, height); detect_img_size swaps it to (height, width) to match
    the height-first convention used throughout the config and
    data-loading layers (img_size/resize/tile_size), with the width-first
    swap-back happening only locally at the actual cv2.resize call sites
    (dataset.py, catsdogs.py).
    """
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    Image.new("RGB", (12, 7)).save(images_dir / "image0.png")

    result = detect_img_size("imagenet", {"imagenet": str(tmp_path)})

    assert result == [7, 12]


def test_detect_img_size_catsdogs_reads_real_file_native_pixel_size(tmp_path):
    images_dir = tmp_path / "imagesTr"
    images_dir.mkdir()
    Image.new("RGB", (12, 7)).save(images_dir / "image0.png")

    result = detect_img_size("catsdogs", {"catsdogs": str(tmp_path)})

    assert result == [7, 12]


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

    assert result == [7, 12]


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
