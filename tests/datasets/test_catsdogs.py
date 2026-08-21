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


def _write_fake_images(tmp_path, names):
    """Writes one small, real, random RGB JPEG per name (e.g. "dog.0.jpg")
    and returns their paths, in the given order.
    """
    paths = []
    rng = np.random.RandomState(0)
    for name in names:
        path = tmp_path / name
        img = Image.fromarray(rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8))
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
        num_channels=NUM_CHANNELS, dataset="catsdogs",
    )
    image, label, returned_variables, dataset_name = ds[0]
    assert image.shape == (NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])  # channel-first, resized
    assert image.dtype == np.uint8
    assert label == 1
    assert returned_variables == variables
    assert dataset_name == "catsdogs"


def test_catsdogs_dataset_adaptive_patching_shapes(tmp_path):
    paths = _write_fake_images(tmp_path, ["cat.0.jpg"])
    ds = CatsDogsDataset(
        paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=True,
        fixed_length=FIXED_LENGTH, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS, dataset="catsdogs",
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
    ds = CatsDogsDataset(paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS)
    return [ds[i] for i in range(len(paths))]


def _adaptive_batch(tmp_path, names):
    paths = _write_fake_images(tmp_path, names)
    ds = CatsDogsDataset(
        paths, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=True,
        fixed_length=FIXED_LENGTH, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS,
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
