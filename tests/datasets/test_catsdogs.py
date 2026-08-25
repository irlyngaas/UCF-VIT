"""Correctness tests for UCF_VIT.datasets.catsdogs (CatsDogsDataset,
CatsDogsCollate).

catsdogs is the only shipped dataset using dataloader.type == "dataloader"
(a plain torch.utils.data.Dataset + DistributedSampler, not the
iterative_dataloader/NativePytorchDataModule stack tests/dataloaders/
test_dataset.py and test_datamodule.py cover) -- so this file is its own
small, self-contained suite, built against small real JPEG files written to
a temp directory (CatsDogsDataset.__getitem__ genuinely opens/resizes/reads
each file via PIL/cv2, so there's no meaningful way to stub that out and
still test the class).

Writing the adaptive_patching=True tests here surfaced a real bug in how
training_scripts/train.py wires CatsDogsDataset up, not in catsdogs.py
itself: it passed `num_channels=conf["data"]["num_channels"]` -- the whole
{key: count} dict -- instead of the per-key int
`conf["data"]["num_channels"][dkey_train]`. Harmless when adaptive_patching
is False (num_channels is unused then), but adaptive_patching=True stores it
as `Patchify.num_channels` and immediately does `if self.num_channels > 1`,
which raises `TypeError: '>' not supported between instances of 'dict' and
'int'`. The shipped catsdogs config ships with `ap.do_ap: False`, so this
was dormant, not an active failure. Fixed in train.py; the tests below
construct CatsDogsDataset directly with the correct (int) num_channels, so
they document the working, intended contract.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from UCF_VIT.datasets.catsdogs import CatsDogsCollate, CatsDogsDataset

TILE_SIZE = (64, 64)
PATCH_SIZE = 16
FIXED_LENGTH = 16
NUM_CHANNELS = 3


def _write_fake_images(tmp_path, names, size=(32, 32)):
    """Writes one small, real, random RGB JPEG per name (e.g. "dog.0.jpg")
    and returns their paths, in the given order. `size` is `(height,
    width)` (matches `np.random.randint`'s own array-shape convention, as
    opposed to `img_size`/`resize`'s `[width, height]`).
    """
    paths = []
    rng = np.random.RandomState(0)
    for name in names:
        path = tmp_path / name
        img = Image.fromarray(rng.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8))
        img.save(path)
        paths.append(str(path))
    return paths


# ---------------------------------------------------------------------------
# CatsDogsDataset
# ---------------------------------------------------------------------------


def test_catsdogs_dataset_len_matches_file_list(tmp_path):
    paths = _write_fake_images(tmp_path, ["dog.0.jpg", "cat.1.jpg", "dog.2.jpg"])
    ds = CatsDogsDataset(paths, variables=("r", "g", "b"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS)
    assert len(ds) == 3


@pytest.mark.parametrize("filename,expected_label", [
    ("dog.0.jpg", 1),
    ("cat.0.jpg", 0),
    ("dog.123.jpg", 1),
    ("cat.999.jpg", 0),
])
def test_catsdogs_dataset_label_derived_from_filename(tmp_path, filename, expected_label):
    paths = _write_fake_images(tmp_path, [filename])
    ds = CatsDogsDataset(paths, variables=("r", "g", "b"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS)
    image, label, variables, dataset_name = ds[0]
    assert label == expected_label


def test_catsdogs_dataset_non_adaptive_shape_and_passthrough(tmp_path):
    paths = _write_fake_images(tmp_path, ["dog.0.jpg"])
    variables = ("red", "green", "blue")
    ds = CatsDogsDataset(
        paths, variables=variables, tile_size=TILE_SIZE, adaptive_patching=False,
        num_channels=NUM_CHANNELS, dataset="catsdogs", resize=TILE_SIZE,
    )
    image, label, returned_variables, dataset_name = ds[0]
    assert image.shape == (NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])  # channel-first, resized
    assert image.dtype == np.uint8
    assert label == 1
    assert returned_variables == variables
    assert dataset_name == "catsdogs"


def test_catsdogs_dataset_resize_none_leaves_native_size(tmp_path):
    """resize is now a separate, optional step (decoupled from tile_size,
    which just drives tiling/patch-size math) -- when omitted, images stay
    at their real native size instead of being forced to tile_size.
    """
    paths = _write_fake_images(tmp_path, ["dog.0.jpg"])  # fabricated at 32x32, see _write_fake_images
    ds = CatsDogsDataset(
        paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=False,
        num_channels=NUM_CHANNELS, dataset="catsdogs",
    )
    image, label, returned_variables, dataset_name = ds[0]
    assert image.shape == (NUM_CHANNELS, 32, 32)


def test_catsdogs_dataset_div_one_is_untiled_default():
    """div's default (1) must reproduce today's exact behavior: __len__
    unchanged (not multiplied by 1*1, but literally the same value/code
    path), matching every pre-existing catsdogs config (which never set
    div/tile_overlap at all before this feature).
    """
    ds = CatsDogsDataset([f"/fake/{i}.jpg" for i in range(5)], variables=("r", "g", "b"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS)
    assert len(ds) == 5


def test_catsdogs_dataset_tiling_len_scales_by_div_squared(tmp_path):
    paths = _write_fake_images(tmp_path, ["dog.0.jpg", "cat.1.jpg"])
    div = 2
    ds = CatsDogsDataset(
        paths, variables=("r", "g", "b"), tile_size=(16, 16), num_channels=NUM_CHANNELS,
        resize=(32, 32), div=div, tile_overlap=(0, 0),
    )
    assert len(ds) == len(paths) * div * div


def test_catsdogs_dataset_tiling_non_square_full_coverage_no_overlap(tmp_path):
    """Regression test mirroring tests/dataloaders/test_dataset.py's
    test_tiledataiter_2d_non_square_tile_size_axes_arent_swapped: uses a
    deliberately non-square image (height != width) so a swapped
    width/height axis mapping (the exact bug just fixed in TileDataIter's
    own 2D branch) would produce truncated/incomplete tiles instead of
    full coverage, rather than passing trivially on a square fixture.

    Native image: height=24, width=36 (np.random.randint shape
    convention). resize=(36, 24) is [width, height] (a no-op resize to
    the same native size, just exercising the resize path too). div=3,
    tile_overlap=(0, 0) -> tile_size is the *per-tile* [width, height]
    (36 // 3, 24 // 3) = (12, 8) -- matches parse.py's own tile_size
    computation (already divided by div, not divided again internally) --
    i.e. each tile is (8, 12) as a channel-first (C, H, W) array.
    """
    paths = _write_fake_images(tmp_path, ["dog.0.jpg"], size=(24, 36))  # (height, width)
    div = 3
    ds = CatsDogsDataset(
        paths, variables=("r", "g", "b"), tile_size=(12, 8), num_channels=NUM_CHANNELS,
        resize=(36, 24), div=div, tile_overlap=(0, 0),
    )
    assert len(ds) == div * div

    covered = np.zeros((24, 36), dtype=bool)
    for idx in range(len(ds)):
        image, label, variables, dataset_name = ds[idx]
        assert image.shape == (NUM_CHANNELS, 24 // div, 36 // div)  # (C, H, W)
        w_idx, h_idx = divmod(idx, div)
        sh, sw = h_idx * (24 // div), w_idx * (36 // div)
        covered[sh:sh + 24 // div, sw:sw + 36 // div] = True
        assert label == 1
        assert dataset_name == "catsdogs"
    assert covered.all()  # every pixel covered; would fail under a swapped-axis bug


def test_catsdogs_dataset_tiling_with_adaptive_patching(tmp_path):
    """Tiling composes with adaptive_patching -- Patchify runs on each
    tile individually, not the whole image (matching imagenet's own
    tile-then-patchify pipeline order via TileDataIter -> ProcessChannels).
    """
    paths = _write_fake_images(tmp_path, ["cat.0.jpg"], size=(32, 32))
    div = 2
    ds = CatsDogsDataset(
        paths, variables=("r", "g", "b"), tile_size=(16, 16), adaptive_patching=True,
        fixed_length=FIXED_LENGTH, interp_size=PATCH_SIZE, num_channels=NUM_CHANNELS,
        dataset="catsdogs", resize=(32, 32), div=div, tile_overlap=(0, 0),
    )
    assert len(ds) == div * div
    for idx in range(len(ds)):
        image, seq_img, seq_size, seq_pos, label, variables, dataset_name = ds[idx]
        assert image.shape == (NUM_CHANNELS, 16, 16)  # one tile, not the whole 32x32 image
        assert seq_img.shape == (NUM_CHANNELS, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
        assert label == 0


def test_catsdogs_dataset_adaptive_patching_shapes(tmp_path):
    paths = _write_fake_images(tmp_path, ["cat.0.jpg"])
    ds = CatsDogsDataset(
        paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=True,
        fixed_length=FIXED_LENGTH, interp_size=PATCH_SIZE, num_channels=NUM_CHANNELS, dataset="catsdogs",
        resize=TILE_SIZE,
    )
    image, seq_img, seq_size, seq_pos, label, variables, dataset_name = ds[0]
    assert image.shape == (NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert seq_img.shape == (NUM_CHANNELS, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert seq_size.shape == (FIXED_LENGTH,)
    assert seq_pos.shape == (FIXED_LENGTH, 2)
    assert label == 0


# ---------------------------------------------------------------------------
# CatsDogsCollate
# ---------------------------------------------------------------------------


def _non_adaptive_batch(tmp_path, names):
    paths = _write_fake_images(tmp_path, names)
    ds = CatsDogsDataset(paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS, resize=TILE_SIZE)
    return [ds[i] for i in range(len(paths))]


def _adaptive_batch(tmp_path, names):
    paths = _write_fake_images(tmp_path, names)
    ds = CatsDogsDataset(
        paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=True,
        fixed_length=FIXED_LENGTH, interp_size=PATCH_SIZE, num_channels=NUM_CHANNELS, resize=TILE_SIZE,
    )
    return [ds[i] for i in range(len(paths))]


def test_catsdogs_collate_non_adaptive_with_label(tmp_path):
    batch = _non_adaptive_batch(tmp_path, ["dog.0.jpg", "cat.1.jpg", "dog.2.jpg"])
    inp, label, variables, dict_key = CatsDogsCollate(batch, adaptive_patching=False, return_label=True)
    assert inp.shape == (3, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert label.tolist() == [1, 0, 1]  # dog, cat, dog -- order preserved (no internal buffering, unlike ProcessChannels)
    assert variables == ("red", "green", "blue")
    assert dict_key == "catsdogs"


def test_catsdogs_collate_non_adaptive_no_label(tmp_path):
    batch = _non_adaptive_batch(tmp_path, ["dog.0.jpg", "cat.1.jpg"])
    inp, variables, dict_key = CatsDogsCollate(batch, adaptive_patching=False, return_label=False)
    assert inp.shape == (2, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert dict_key == "catsdogs"


def test_catsdogs_collate_adaptive_with_label(tmp_path):
    batch = _adaptive_batch(tmp_path, ["dog.0.jpg", "cat.1.jpg"])
    inp, seq, size, pos, label, variables, dict_key = CatsDogsCollate(batch, adaptive_patching=True, return_label=True)
    assert inp.shape == (2, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert seq.shape == (2, NUM_CHANNELS, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (2, 1, FIXED_LENGTH)
    assert pos.shape == (2, 1, FIXED_LENGTH, 2)
    assert label.tolist() == [1, 0]
    assert dict_key == "catsdogs"


def test_catsdogs_collate_adaptive_no_label(tmp_path):
    batch = _adaptive_batch(tmp_path, ["dog.0.jpg", "cat.1.jpg"])
    inp, seq, size, pos, variables, dict_key = CatsDogsCollate(batch, adaptive_patching=True, return_label=False)
    assert inp.shape == (2, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert seq.shape == (2, NUM_CHANNELS, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert dict_key == "catsdogs"
