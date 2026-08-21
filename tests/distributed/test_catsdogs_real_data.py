"""Real-multi-rank check of the catsdogs data pipeline, against real
CatsDogs JPEGs on Frontier.

catsdogs is the only shipped dataset using dataloader.type == "dataloader"
-- a plain torch.utils.data.Dataset (CatsDogsDataset) sharded by PyTorch's
own DistributedSampler, not by any UCF_VIT-custom sharding logic like
FileReader's (see test_dataloader_real_data.py for that). There's no known
bug to regression-test here the way there was for FileReader's
num_workers=0 case -- DistributedSampler is well-tested upstream PyTorch
code -- so the value of this file is different: confirming the *real*
production wiring (training_scripts/train.py's DistributedSampler(...,
num_replicas=data_par_size, rank=world_rank) + DataLoader(...,
num_workers=...) + CatsDogsCollate construction) actually works end to end
against real files, real multi-rank ranks, and real JPEG decode/edge
detection (tests/datasets/test_catsdogs.py's adaptive_patching=True
coverage uses small synthetic random-noise JPEGs, which don't exercise
Canny edge detection the way a real photo's actual edges do).

Unlike test_dataloader_real_data.py, file reading is *not* stubbed out here
-- CatsDogsDataset.__getitem__ has no meaningful decode-free path (it opens,
resizes, and label-derives from the real file inline), and real decode cost
against a handful of narrowed files is fast enough not to need stubbing.

The real CatsDogs directory can have tens of thousands of files, so (same
"real data, just less of it" principle as
tests/integration/run_training_smoke.py's create_narrow_catsdogs_dir) this
narrows to a small, real subset -- sized to an exact multiple of world_size
so DistributedSampler's default drop_last=False padding (which repeats
samples to round up to a multiple of num_replicas) never kicks in, keeping
the disjointness check unambiguous.
"""

import glob
import os

import pytest
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from UCF_VIT.datasets.catsdogs import CatsDogsCollate, CatsDogsDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "catsdogs", "classification", "base_config.yaml")

TILE_SIZE = (64, 64)
PATCH_SIZE = 16
FIXED_LENGTH = 16
NUM_CHANNELS = 3
FILES_PER_RANK = 4

_file_list_cache = None


def _real_file_list(world_size):
    """Globs the real catsdogs directory the same way train.py does
    (glob.glob(root_dir/*.jpg)), narrowed to an exact multiple of
    world_size real files.
    """
    global _file_list_cache
    if _file_list_cache is not None:
        return _file_list_cache

    with open(CONFIG_PATH) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    dict_root_dirs = conf["data"]["dict_root_dirs"]
    dkey = next(iter(dict_root_dirs))
    root_dir = dict_root_dirs[dkey]

    files = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    needed = FILES_PER_RANK * world_size
    if len(files) < needed:
        pytest.skip(
            f"only {len(files)} real catsdogs files found under {root_dir!r}, need "
            f"at least {needed} ({FILES_PER_RANK} x world_size={world_size}) -- "
            f"dict_root_dirs in {CONFIG_PATH} is stale or unmounted here"
        )
    _file_list_cache = files[:needed]
    return _file_list_cache


def _rank_indices(file_list, world_size, world_rank, shuffle):
    # shuffle=False here (production uses shuffle=True) since shuffling only
    # reorders each rank's assigned indices, it doesn't change the
    # partitioning itself -- disjointness/coverage hold identically either
    # way, and False keeps this test's expectations simple to state.
    ds = CatsDogsDataset(file_list, variables=("red", "green", "blue"), tile_size=TILE_SIZE, num_channels=NUM_CHANNELS)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=world_rank, shuffle=shuffle)
    assert len(sampler) == FILES_PER_RANK  # exact multiple of world_size -- no drop_last padding
    return ds, sampler, list(sampler)


@pytest.mark.parametrize("num_workers", [0, 1, 4])
def test_catsdogs_distributed_sampler_shards_real_files_disjointly(dist_info, num_workers):
    """The real DistributedSampler/DataLoader wiring train.py uses must give
    every rank a disjoint set of real files, and every rank must actually be
    able to read and collate its share through the real
    CatsDogsDataset/CatsDogsCollate pipeline.
    """
    world_size = dist_info["world_size"]
    world_rank = dist_info["world_rank"]
    file_list = _real_file_list(world_size)

    ds, sampler, indices = _rank_indices(file_list, world_size, world_rank, shuffle=False)
    my_files = {file_list[i] for i in indices}

    all_files = [None] * world_size
    dist.all_gather_object(all_files, my_files)
    for other_rank, other_files in enumerate(all_files):
        if other_rank == world_rank:
            continue
        assert my_files.isdisjoint(other_files), (
            f"rank {world_rank}'s real files overlap rank {other_rank}'s -- "
            f"DistributedSampler sharding isn't behaving as expected"
        )
    if world_rank == 0:
        covered = set().union(*all_files)
        assert covered == set(file_list)

    loader = DataLoader(
        ds, sampler=sampler, num_workers=num_workers, batch_size=FILES_PER_RANK, drop_last=True,
        collate_fn=lambda batch: CatsDogsCollate(batch, adaptive_patching=False, return_label=True),
    )
    (inp, label, variables, dict_key), = list(loader)
    assert inp.shape == (FILES_PER_RANK, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert set(label.tolist()) <= {0, 1}
    assert dict_key == "catsdogs"


def test_catsdogs_adaptive_patching_against_real_images(dist_info):
    """adaptive_patching=True against real photos, not tests/datasets/
    test_catsdogs.py's synthetic random-noise JPEGs -- real Canny edge
    detection on real image content, at real multi-rank scale.
    """
    world_size = dist_info["world_size"]
    world_rank = dist_info["world_rank"]
    file_list = _real_file_list(world_size)

    ds = CatsDogsDataset(
        file_list, variables=("red", "green", "blue"), tile_size=TILE_SIZE, adaptive_patching=True,
        fixed_length=FIXED_LENGTH, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS, dataset="catsdogs",
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=world_rank, shuffle=False)
    loader = DataLoader(
        ds, sampler=sampler, num_workers=0, batch_size=FILES_PER_RANK, drop_last=True,
        collate_fn=lambda batch: CatsDogsCollate(batch, adaptive_patching=True, return_label=True),
    )
    (inp, seq, size, pos, label, variables, dict_key), = list(loader)
    assert inp.shape == (FILES_PER_RANK, NUM_CHANNELS, TILE_SIZE[0], TILE_SIZE[1])
    assert seq.shape == (FILES_PER_RANK, NUM_CHANNELS, FIXED_LENGTH, PATCH_SIZE * PATCH_SIZE)
    assert size.shape == (FILES_PER_RANK, 1, FIXED_LENGTH)
    assert pos.shape == (FILES_PER_RANK, 1, FIXED_LENGTH, 2)
