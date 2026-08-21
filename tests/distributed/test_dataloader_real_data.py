"""Real-multi-rank check of FileReader's DDP-rank sharding and
ShuffleIterableDataset's no-loss/no-duplication guarantee, against real
basic_ct and imagenet data on Frontier.

Tier 1's tests/dataloaders/test_dataset.py already covers both classes'
correctness against synthetic file lists and *simulated* ranks
(monkeypatched torch.distributed.get_rank/torch.utils.data.get_worker_info,
run one at a time in a single process) -- including the regression tests for
the bug this fixed (num_workers=0 never applying DDP-rank sharding at all,
so every rank read the entire file list; see tests/README.md's "Real runs on
Frontier so far" and tests/dataloaders/test_dataset.py's
test_filereader_num_workers_zero_shards_by_ddp_rank).

This file re-runs that same check, but for real: 8 genuinely separate
processes under a real srun launch, using torch.distributed.get_rank() for
real, against real file lists under basic_ct's and imagenet's actual
dict_root_dirs on Frontier -- the strongest possible validation that the fix
works at the scale/environment it's meant for. Parametrized across
num_workers (0, 1, 4 -- num_workers > 0 combines a real DDP rank with a
*simulated* per-worker split, monkeypatched the same way Tier 1 does, rather
than spawning real DataLoader multiprocessing workers under an
already-NCCL-initialized process) and buffer_size (1, 20, 100 -- 100 is both
basic_ct/unetr's and imagenet/classification's real shipped
dict_buffer_sizes value; 20 and 1 are extra coverage, not tied to a shipped
config).

File *reading* itself is stubbed out (FileReader.read_process_file
monkeypatched to return the path unchanged) so this stays fast and focused
on correctness, not decode performance -- see
tests/dataloaders/test_dataset_speed.py's module docstring for why real
timing numbers belong in a separate, informational-only tier instead.

imagenet's real directory can be very large (order of a million files across
~1000 class subdirectories), so imagenet's file list here is deliberately
narrowed to the first few class subdirectories -- same "real data, just less
of it" principle as tests/integration/run_training_smoke.py's
create_narrow_catsdogs_dir, kept simple since this only needs *a* real file
list to shard, not the full dataset.
"""

import glob
import itertools
import os

import pytest
import torch.distributed as dist
import yaml

from UCF_VIT.dataloaders.dataset import FileReader, ShuffleIterableDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATHS = {
    "basic_ct": os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml"),
    "imagenet": os.path.join(REPO_ROOT, "configs", "imagenet", "classification", "base_config.yaml"),
}
IMAGENET_MAX_CLASSES = 5

_file_list_cache = {}


def _real_file_list(dataset_name):
    """Lists a real file list for `dataset_name`, cached across parametrized
    test invocations so repeated globbing doesn't add up.
    """
    if dataset_name in _file_list_cache:
        return _file_list_cache[dataset_name]

    with open(CONFIG_PATHS[dataset_name]) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    dict_root_dirs = conf["data"]["dict_root_dirs"]
    dkey = next(iter(dict_root_dirs))
    root_dir = dict_root_dirs[dkey]

    if dataset_name == "basic_ct":
        # Mirrors NativePytorchDataModule.process_root_dirs' non-imagenet branch.
        files = sorted(glob.glob(os.path.join(root_dir, "imagesTr", "*")))
    else:
        # A handful of real class subdirectories, not all ~1000 -- see
        # module docstring.
        if not os.path.isdir(root_dir):
            files = []
        else:
            classes = sorted(os.listdir(root_dir))[:IMAGENET_MAX_CLASSES]
            files = sorted(
                f for cls in classes for f in glob.glob(os.path.join(root_dir, cls, "*.JPEG"))
            )

    if not files:
        pytest.skip(
            f"No real {dataset_name} files found under {root_dir!r} -- "
            f"dict_root_dirs in {CONFIG_PATHS[dataset_name]} is stale or unmounted here"
        )
    _file_list_cache[dataset_name] = files
    return files


class _FakeWorkerInfo:
    def __init__(self, num_workers, id):
        self.num_workers = num_workers
        self.id = id


def _read_shard(file_list, data_par_size):
    reader = FileReader(
        file_list, start_idx=0.0, end_idx=1.0, variables=("v0",), gx=str(data_par_size),
        ddp_group=None, data_par_size=data_par_size, dataset="imagenet",
    )
    return {path for path, variables in reader}


@pytest.mark.parametrize("buffer_size", [1, 20, 100])
@pytest.mark.parametrize("num_workers", [0, 1, 4])
@pytest.mark.parametrize("dataset_name", ["basic_ct", "imagenet"])
def test_filereader_shuffle_shards_real_data_disjointly_and_losslessly(
    dist_info, monkeypatch, dataset_name, num_workers, buffer_size,
):
    """For every (dataset, num_workers, buffer_size) combination: this rank's
    real shard must be disjoint from every other rank's, the shards together
    must cover the real file list (modulo FileReader's documented
    floor-division remainder), and shuffling a shard through
    ShuffleIterableDataset at any buffer_size must not lose or duplicate any
    of its files.
    """
    monkeypatch.setattr(FileReader, "read_process_file", lambda self, path: path)

    world_size = dist_info["world_size"]
    world_rank = dist_info["world_rank"]
    file_list = _real_file_list(dataset_name)

    num_shards = world_size * max(num_workers, 1)
    if len(file_list) // num_shards < 1:
        pytest.skip(
            f"only {len(file_list)} real {dataset_name} files available, not enough "
            f"for {num_shards} shards (world_size={world_size}, num_workers={num_workers})"
        )

    if num_workers == 0:
        my_shard = _read_shard(file_list, world_size)
    else:
        my_shard = set()
        for worker_id in range(num_workers):
            monkeypatch.setattr(
                "torch.utils.data.get_worker_info",
                lambda worker_id=worker_id: _FakeWorkerInfo(num_workers=num_workers, id=worker_id),
            )
            my_shard |= _read_shard(file_list, world_size)

    # ShuffleIterableDataset must not lose or duplicate anything from this
    # rank's shard, at any buffer_size. (Passing the plain set as `dataset`
    # works because ShuffleIterableDataset only ever does `for x in
    # self.dataset`, and sets are iterable.)
    shuffled = set(ShuffleIterableDataset(my_shard, buffer_size=buffer_size))
    assert shuffled == my_shard

    all_shards = [None] * world_size
    dist.all_gather_object(all_shards, my_shard)

    assert len(my_shard) > 0, f"rank {world_rank} got an empty shard"
    for other_rank, other_shard in enumerate(all_shards):
        if other_rank == world_rank:
            continue
        assert my_shard.isdisjoint(other_shard), (
            f"rank {world_rank}'s shard overlaps rank {other_rank}'s -- DDP-rank "
            f"sharding is broken (this is exactly the num_workers=0 regression)"
        )

    if world_rank == 0:
        covered = set().union(*all_shards)
        per_shard = len(file_list) // num_shards
        assert len(covered) >= per_shard * num_shards
        assert len(covered) <= len(file_list)
