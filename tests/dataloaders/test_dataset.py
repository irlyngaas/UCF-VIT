"""Correctness tests for the iterable-dataset pipeline in UCF_VIT.dataloaders.dataset.

Covers TileDataIter, ShuffleIterableDataset, ProcessChannels, and FileReader's
worker-sharding math -- all against small, synthetic, in-memory data, so these
run anywhere in well under a second with no GPU/SLURM/real dataset needed.

TileDataIter gets the deepest coverage here because it's where a real, live
bug was found and fixed this session (see tests/README.md item 8): a 3D
basic_ct volume being twoD-sliced into 2D z-planes was silently treated as
genuinely-2D data, leaving the whole z-axis untouched on every tile and
producing a 5D batch several layers downstream. The regression tests below
(`test_tiledataiter_3d_twod_true_*`) exercise exactly that path.
"""

import itertools

import numpy as np
import pytest
from torch.utils.data import IterableDataset

from UCF_VIT.dataloaders.dataset import (
    FileReader,
    ProcessChannels,
    ShuffleIterableDataset,
    TileDataIter,
)


class _FakeSource(IterableDataset):
    """Yields pre-built samples, standing in for FileReader/TileDataIter/etc.

    in tests that only care about exercising one class in isolation.
    """

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        yield from self.samples


# ---------------------------------------------------------------------------
# TileDataIter
# ---------------------------------------------------------------------------


def test_tiledataiter_2d_full_coverage_no_overlap():
    H, W = 12, 12
    div = 3
    tile_h, tile_w = H // div, W // div
    data = np.arange(H * W, dtype=np.float32).reshape(1, H, W)
    source = _FakeSource([(data, ("v0",))])
    tdi = TileDataIter(source, tile_size=(tile_h, tile_w), twoD=True, return_label=False, div=div, tile_overlap=(0, 0), classification=False)

    results = list(tdi)
    assert len(results) == div * div

    covered = np.zeros((H, W), dtype=bool)
    for (tile, variables), (x_idx, y_idx) in zip(results, itertools.product(range(div), range(div))):
        sx, ex = x_idx * tile_h, (x_idx + 1) * tile_h
        sy, ey = y_idx * tile_w, (y_idx + 1) * tile_w
        np.testing.assert_array_equal(tile, data[:, sx:ex, sy:ey])
        assert variables == ("v0",)
        covered[sx:ex, sy:ey] = True
    assert covered.all()  # every pixel covered; disjoint by construction since overlap=0


def test_tiledataiter_2d_segmentation_label_tiled_same_as_data():
    H, W = 8, 8
    div = 2
    tile_h, tile_w = H // div, W // div
    data = np.arange(H * W, dtype=np.float32).reshape(1, H, W)
    label = np.arange(H * W, dtype=np.int64).reshape(H, W) + 1000
    source = _FakeSource([(data, label, ("v0",))])
    tdi = TileDataIter(source, tile_size=(tile_h, tile_w), twoD=True, return_label=True, div=div, tile_overlap=(0, 0), classification=False)

    results = list(tdi)
    assert len(results) == div * div
    for (tile, tile_label, variables), (x_idx, y_idx) in zip(results, itertools.product(range(div), range(div))):
        sx, ex = x_idx * tile_h, (x_idx + 1) * tile_h
        sy, ey = y_idx * tile_w, (y_idx + 1) * tile_w
        np.testing.assert_array_equal(tile, data[:, sx:ex, sy:ey])
        np.testing.assert_array_equal(tile_label, label[sx:ex, sy:ey])


def test_tiledataiter_2d_classification_label_passed_through_whole():
    H, W = 8, 8
    div = 2
    data = np.arange(H * W, dtype=np.float32).reshape(1, H, W)
    label = 3  # a whole-image class index, as used for classification
    source = _FakeSource([(data, label, ("v0",))])
    tdi = TileDataIter(source, tile_size=(H // div, W // div), twoD=True, return_label=True, div=div, tile_overlap=(0, 0), classification=True)

    results = list(tdi)
    assert len(results) == div * div
    assert all(tile_label == label for _, tile_label, _ in results)


def test_tiledataiter_3d_twod_true_slices_into_2d_tiles_no_leftover_z_axis():
    """Regression test for the 5D-batch bug (see module docstring).

    3D data with twoD=True must be sliced one z-index at a time into plain
    2D tiles -- not left with the untouched z-axis still attached.
    """
    C, X, Y, Z = 1, 8, 8, 3
    div = 2
    tile_x, tile_y = X // div, Y // div
    data = np.arange(C * X * Y * Z, dtype=np.float32).reshape(C, X, Y, Z)
    source = _FakeSource([(data, ("v0",))])
    # tile_size must stay a 3-tuple (z entry = full, untiled depth) -- this is
    # what parse.py now produces for twoD=True 3D data; see its tile_size
    # computation and TileDataIter's own len(self.tile_size) == 3 dispatch.
    tdi = TileDataIter(source, tile_size=(tile_x, tile_y, Z), twoD=True, return_label=False, div=div, tile_overlap=(0, 0), classification=False)

    results = list(tdi)
    assert len(results) == Z * div * div
    for tile, _ in results:
        assert tile.shape == (C, tile_x, tile_y)  # no leftover z axis

    covered = np.zeros((X, Y, Z), dtype=bool)
    for (tile, _), (z_idx, x_idx, y_idx) in zip(results, itertools.product(range(Z), range(div), range(div))):
        sx, ex = x_idx * tile_x, (x_idx + 1) * tile_x
        sy, ey = y_idx * tile_y, (y_idx + 1) * tile_y
        np.testing.assert_array_equal(tile, data[:, sx:ex, sy:ey, z_idx])
        covered[sx:ex, sy:ey, z_idx] = True
    assert covered.all()


def test_tiledataiter_3d_twod_true_with_segmentation_label():
    C, X, Y, Z = 1, 8, 8, 2
    div = 2
    tile_x, tile_y = X // div, Y // div
    data = np.arange(C * X * Y * Z, dtype=np.float32).reshape(C, X, Y, Z)
    label = np.arange(X * Y * Z, dtype=np.int64).reshape(X, Y, Z) + 1000
    source = _FakeSource([(data, label, ("v0",))])
    tdi = TileDataIter(source, tile_size=(tile_x, tile_y, Z), twoD=True, return_label=True, div=div, tile_overlap=(0, 0), classification=False)

    results = list(tdi)
    assert len(results) == Z * div * div
    for (tile, tile_label, _), (z_idx, x_idx, y_idx) in zip(results, itertools.product(range(Z), range(div), range(div))):
        assert tile.shape == (C, tile_x, tile_y)
        assert tile_label.shape == (tile_x, tile_y)
        sx, ex = x_idx * tile_x, (x_idx + 1) * tile_x
        sy, ey = y_idx * tile_y, (y_idx + 1) * tile_y
        np.testing.assert_array_equal(tile, data[:, sx:ex, sy:ey, z_idx])
        np.testing.assert_array_equal(tile_label, label[sx:ex, sy:ey, z_idx])


def test_tiledataiter_3d_twod_false_full_3d_tiles():
    C, X, Y, Z = 1, 8, 8, 8
    div = 2
    t = X // div
    data = np.arange(C * X * Y * Z, dtype=np.float32).reshape(C, X, Y, Z)
    source = _FakeSource([(data, ("v0",))])
    tdi = TileDataIter(source, tile_size=(t, t, t), twoD=False, return_label=False, div=div, tile_overlap=(0, 0, 0), classification=False)

    results = list(tdi)
    assert len(results) == div ** 3

    covered = np.zeros((X, Y, Z), dtype=bool)
    for (tile, _), (x_idx, y_idx, z_idx) in zip(results, itertools.product(range(div), range(div), range(div))):
        assert tile.shape == (C, t, t, t)
        sx, ex = x_idx * t, (x_idx + 1) * t
        sy, ey = y_idx * t, (y_idx + 1) * t
        sz, ez = z_idx * t, (z_idx + 1) * t
        np.testing.assert_array_equal(tile, data[:, sx:ex, sy:ey, sz:ez])
        covered[sx:ex, sy:ey, sz:ez] = True
    assert covered.all()


def test_tiledataiter_div_one_returns_full_image():
    C, H, W = 1, 6, 6
    data = np.arange(C * H * W, dtype=np.float32).reshape(C, H, W)
    source = _FakeSource([(data, ("v0",))])
    tdi = TileDataIter(source, tile_size=(H, W), twoD=True, return_label=False, div=1, tile_overlap=(0, 0), classification=False)

    results = list(tdi)
    assert len(results) == 1
    tile, variables = results[0]
    np.testing.assert_array_equal(tile, data)
    assert variables == ("v0",)


def test_tiledataiter_2d_overlap_tiles_have_configured_shape_and_reach_edges():
    H, W = 12, 12
    div = 3
    overlap = (2, 2)
    base = H // div
    tile_h = tile_w = base + overlap[0]  # matches parse.py's tile_size = img_size//div + overlap
    data = np.arange(H * W, dtype=np.float32).reshape(1, H, W)
    source = _FakeSource([(data, ("v0",))])
    tdi = TileDataIter(source, tile_size=(tile_h, tile_w), twoD=True, return_label=False, div=div, tile_overlap=overlap, classification=False)

    results = list(tdi)
    assert len(results) == div * div
    for tile, _ in results:
        assert tile.shape == (1, tile_h, tile_w)

    # the first tile starts exactly at the image origin and the last tile
    # ends exactly at the image's far edge -- overlap only extends inward
    # between tiles, never off the edge of the source array
    first_tile, _ = results[0]  # x_idx=0, y_idx=0
    np.testing.assert_array_equal(first_tile, data[:, 0:tile_h, 0:tile_w])
    last_tile, _ = results[-1]  # x_idx=div-1, y_idx=div-1
    np.testing.assert_array_equal(last_tile, data[:, H - tile_h:H, W - tile_w:W])


# ---------------------------------------------------------------------------
# ShuffleIterableDataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buffer_size", [1, 5, 30, 1000])
def test_shuffle_iterable_dataset_no_data_loss_or_duplication(buffer_size):
    items = list(range(30))
    shuffled = ShuffleIterableDataset(_FakeSource(items), buffer_size=buffer_size)
    assert sorted(shuffled) == items


def test_shuffle_iterable_dataset_buffer_size_one_preserves_order():
    # With buffer_size=1 the reservoir always holds exactly one item and
    # immediately swaps it out for the next, so output order == input order.
    items = list(range(20))
    shuffled = ShuffleIterableDataset(_FakeSource(items), buffer_size=1)
    assert list(shuffled) == items


def test_shuffle_iterable_dataset_requires_positive_buffer_size():
    with pytest.raises(AssertionError):
        ShuffleIterableDataset(_FakeSource([1, 2, 3]), buffer_size=0)


# ---------------------------------------------------------------------------
# ProcessChannels
# ---------------------------------------------------------------------------


def test_processchannels_no_ap_no_label_passes_tiles_through():
    batch_size = 4
    tiles = [np.full((1, 2, 2), float(i), dtype=np.float32) for i in range(batch_size * 2)]
    source = _FakeSource([(tile, ("v0",)) for tile in tiles])
    pc = ProcessChannels(
        source, num_channels=1, batch_size=batch_size, return_label=False,
        adaptive_patching=False, separate_channels=False, patch_size=4,
        fixed_length=16, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    results = list(pc)
    assert len(results) == len(tiles)
    got_values = sorted(float(tile[0, 0, 0]) for tile, _ in results)
    expected_values = sorted(float(tile[0, 0, 0]) for tile in tiles)
    assert got_values == expected_values


def test_processchannels_drops_incomplete_trailing_internal_batch():
    """Documents current behavior: ProcessChannels buffers `batch_size`
    samples before draining, so a trailing group smaller than `batch_size`
    (because the upstream source ran out) is silently never yielded.
    """
    batch_size = 4
    tiles = [np.zeros((1, 2, 2), dtype=np.float32) for _ in range(batch_size + 2)]
    source = _FakeSource([(tile, ("v0",)) for tile in tiles])
    pc = ProcessChannels(
        source, num_channels=1, batch_size=batch_size, return_label=False,
        adaptive_patching=False, separate_channels=False, patch_size=4,
        fixed_length=16, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    results = list(pc)
    assert len(results) == batch_size


def test_processchannels_no_ap_with_label_basic_ct():
    batch_size = 3
    tiles = [np.full((1, 2, 2), float(i), dtype=np.float32) for i in range(batch_size)]
    labels = [np.full((2, 2), i + 100, dtype=np.int64) for i in range(batch_size)]
    source = _FakeSource([(t, l, ("v0",)) for t, l in zip(tiles, labels)])
    pc = ProcessChannels(
        source, num_channels=1, batch_size=batch_size, return_label=True,
        adaptive_patching=False, separate_channels=False, patch_size=4,
        fixed_length=16, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    results = list(pc)
    assert len(results) == batch_size
    got = {(float(t[0, 0, 0]), float(l[0, 0])) for t, l, v in results}
    expected = {(float(i), float(i + 100)) for i in range(batch_size)}
    assert got == expected


def test_processchannels_adaptive_patching_produces_fixed_length_sequence():
    patch_size = 4
    fixed_length = 16
    img = np.random.RandomState(0).uniform(0, 1, size=(1, 32, 32)).astype(np.float32)
    source = _FakeSource([(img, ("ct_res1",))])
    pc = ProcessChannels(
        source, num_channels=1, batch_size=1, return_label=False,
        adaptive_patching=True, separate_channels=False, patch_size=patch_size,
        fixed_length=fixed_length, twoD=True, _dataset="basic_ct", return_qdt=False,
    )
    (np_image, seq_image, seq_size, seq_pos, variables), = list(pc)
    assert np_image.shape == img.shape
    assert seq_image.shape == (fixed_length, patch_size * patch_size)
    assert seq_size.shape[0] == fixed_length
    assert seq_pos.shape[0] == fixed_length
    assert variables == ("ct_res1",)


# ---------------------------------------------------------------------------
# FileReader worker sharding
# ---------------------------------------------------------------------------


class _FakeWorkerInfo:
    def __init__(self, num_workers, id):
        self.num_workers = num_workers
        self.id = id


def test_filereader_worker_shards_are_disjoint_and_cover_the_file_list(monkeypatch):
    # Stub out actual file I/O so __iter__'s real sharding logic can run
    # against a fake file_list without touching disk -- each "sample" is just
    # the path itself, tagged with the variables it was read with.
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)

    file_list = [f"/fake/path/{i}.jpg" for i in range(20)]
    num_workers = 4

    shards = []
    for worker_id in range(num_workers):
        monkeypatch.setattr(
            "torch.utils.data.get_worker_info",
            lambda worker_id=worker_id: _FakeWorkerInfo(num_workers=num_workers, id=worker_id),
        )
        reader = FileReader(
            file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx="1",
            ddp_group=None, data_par_size=1, dataset="imagenet",
        )
        shard = {path for path, variables in reader}
        shards.append(shard)

    # disjoint
    for a, b in itertools.combinations(shards, 2):
        assert a.isdisjoint(b)
    # every file covered by exactly one shard (evenly divides here: 20 / 4 == 5)
    covered = set().union(*shards)
    assert covered == set(file_list)
    assert all(len(shard) == len(file_list) // num_workers for shard in shards)


def test_filereader_keys_to_add_walks_each_replicated_copy_once(monkeypatch):
    """In production, `file_list` arriving at FileReader has *already* been
    replicated `keys_to_add` times by the caller
    (NativePytorchDataModule.set_iterative_dataloader/reset), to balance
    epoch length across differently-sized dict_root_dirs keys. FileReader's
    own keys_to_add loop then walks the same per-worker slice within each of
    those `keys_to_add` copies -- it does not itself repeat a shorter
    file_list.
    """
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)
    monkeypatch.setattr("torch.utils.data.get_worker_info", lambda: _FakeWorkerInfo(num_workers=1, id=0))

    copy_size = 4
    keys_to_add = 3
    file_list = [f"/fake/copy{c}/{i}.jpg" for c in range(keys_to_add) for i in range(copy_size)]
    reader = FileReader(
        file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx="1",
        ddp_group=None, data_par_size=1, dataset="imagenet", keys_to_add=keys_to_add,
    )
    seen = [path for path, variables in reader]
    assert sorted(seen) == sorted(file_list)  # every file in every copy visited exactly once
    assert len(seen) == len(set(seen))  # no duplicates


def test_filereader_keys_to_add_with_no_dataloader_workers(monkeypatch):
    """Regression test: keys_to_add > 1 combined with num_workers=0 (no
    DataLoader multiprocessing workers, so torch.utils.data.get_worker_info()
    is None) used to walk past the end of file_list and raise IndexError --
    __iter__'s no-worker-info branch didn't divide iter_end by keys_to_add
    before the repetition loop, unlike the real-worker-info branch. Fixed by
    routing num_workers=0 through the same sharding math as num_workers >= 1
    (see test_filereader_num_workers_zero_shards_by_ddp_rank below for the
    closely-related bug this shared a root cause with).
    """
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)
    monkeypatch.setattr("torch.utils.data.get_worker_info", lambda: None)

    copy_size = 4
    keys_to_add = 3
    file_list = [f"/fake/copy{c}/{i}.jpg" for c in range(keys_to_add) for i in range(copy_size)]
    reader = FileReader(
        file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx="1",
        ddp_group=None, data_par_size=1, dataset="imagenet", keys_to_add=keys_to_add,
    )
    seen = [path for path, variables in reader]
    assert sorted(seen) == sorted(file_list)
    assert len(seen) == len(set(seen))


def test_filereader_num_workers_zero_shards_by_ddp_rank(monkeypatch):
    """Regression test for a real bug: FileReader.__iter__'s num_workers=0
    (torch.utils.data.get_worker_info() is None) branch used to skip
    DDP-rank sharding entirely -- iter_start=0, iter_end=len(file_list)
    unconditionally -- so *every* DDP rank read the whole file_list instead
    of its own shard. This is live today, not just a landmine: basic_ct/sap
    and basic_ct/unetr both ship with num_workers: 0 and simple_ddp_size: 8.
    """
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)
    monkeypatch.setattr("torch.utils.data.get_worker_info", lambda: None)

    file_list = [f"/fake/{i}.jpg" for i in range(24)]
    data_par_size = 4

    shards = []
    for ddp_rank in range(data_par_size):
        monkeypatch.setattr("torch.distributed.get_rank", lambda ddp_rank=ddp_rank: ddp_rank)
        reader = FileReader(
            file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx=str(data_par_size),
            ddp_group=None, data_par_size=data_par_size, dataset="imagenet",
        )
        shards.append({path for path, variables in reader})

    for a, b in itertools.combinations(shards, 2):
        assert a.isdisjoint(b)  # previously every shard == the full file_list
    covered = set().union(*shards)
    assert covered == set(file_list)
    assert all(len(shard) == len(file_list) // data_par_size for shard in shards)


@pytest.mark.parametrize("num_dataloader_workers", [1, 2, 7])
def test_filereader_shards_combine_ddp_rank_and_dataloader_workers(monkeypatch, num_dataloader_workers):
    """The real Frontier setup: data_par_size DDP ranks, each with its own
    DataLoader running num_workers subprocess workers (up to 7 here, matching
    Frontier's cores/node). The two axes combine multiplicatively via
    num_shards = group_size * num_workers_per_ddp, and every (ddp_rank,
    worker_id) pair must get a disjoint slice that together cover the whole
    file_list.
    """
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)

    data_par_size = 2
    total_shards = data_par_size * num_dataloader_workers
    files_per_shard = 3
    file_list = [f"/fake/{i}.jpg" for i in range(total_shards * files_per_shard)]

    shards = []
    for ddp_rank in range(data_par_size):
        monkeypatch.setattr("torch.distributed.get_rank", lambda ddp_rank=ddp_rank: ddp_rank)
        for worker_id in range(num_dataloader_workers):
            monkeypatch.setattr(
                "torch.utils.data.get_worker_info",
                lambda worker_id=worker_id: _FakeWorkerInfo(num_workers=num_dataloader_workers, id=worker_id),
            )
            reader = FileReader(
                file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx=str(data_par_size),
                ddp_group=None, data_par_size=data_par_size, dataset="imagenet",
            )
            shards.append({path for path, variables in reader})

    for a, b in itertools.combinations(shards, 2):
        assert a.isdisjoint(b)
    covered = set().union(*shards)
    assert covered == set(file_list)
    assert all(len(shard) == files_per_shard for shard in shards)
