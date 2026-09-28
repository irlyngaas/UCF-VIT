"""Standalone utility to precompute per-(dataset key, variable) z-score normalization stats.

Computes mean/std over the *train* split only (never val/test -- see this
module's own `--split` default and `UCF_VIT.utils.normalize`'s module
docstring for why), by reusing the real training data pipeline directly --
`UCF_VIT.parse.parse_config` to resolve the config exactly like `train.py`
does, then either `UCF_VIT.dataloaders.datamodule.NativePytorchDataModule`
(`dataloader.type: "iterative_dataloader"`, i.e. imagenet/basic_ct/sst) or
`UCF_VIT.datasets.catsdogs.CatsDogsDataset` (`dataloader.type: "dataloader"`)
-- the same two dataloader-construction paths `train.py` itself uses, run
single-process (no real SLURM allocation needed, same trick `utils/
validate_config.py` uses). Accumulates count/sum/sum-of-squares in float64
per (dataset key, variable name) across one full pass, then writes the
result as a YAML file shaped exactly the way `config.data.
normalize_stats_path` expects:

    ct1:
      ct_res1: {mean: ..., std: ...}

Deliberately builds the dataloader with `adaptive_patching`/`tiling` forced
off and `normalize_stats` left empty regardless of what the config says --
stats are computed over the *raw* per-sample data (not adaptively-patched
sequences, not already-normalized data from some existing
`normalize_stats_path` the config might already point at, and not tiled --
tiling doesn't change the underlying value distribution when tile_overlap
is 0, and actively biases it when tile_overlap > 0, since overlapping
voxels would be double-counted; either way it multiplies the number of
samples iterated for zero benefit to the computed stats).

Scales across real, separate processes/nodes for free: if launched under a
real multi-task `srun -n N` (detected via the same `SLURM_PROCID`/
`SLURM_NTASKS` env vars `train.py` itself reads), each rank gets its own
file shard via `NativePytorchDataModule`'s own existing `data_par_size`/
`gx` sharding (the exact mechanism real distributed training already uses)
and accumulates its own local count/sum/sum-of-squares; the ranks'
partial sums are then combined -- exactly, not approximately, since a sum
of sums is still exact regardless of how the underlying values were
partitioned -- via one `dist.all_gather_object` at the end. Uses "gloo"
(CPU), not `train.py`'s "nccl" -- this is pure CPU/file-IO work, no GPU
tensor communication happens anywhere, so there's no reason to require
real GPUs just to get multi-process/multi-node parallelism. Launched as a
single plain process (no `srun`, `SLURM_NTASKS` unset or 1), it falls back
to the exact single-process behavior this always had.

Usage:
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --split val
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --max-samples 2000
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --progress-interval 5
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --num-workers 7
"""

import argparse
import copy
import functools
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from UCF_VIT.parse import parse_config
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.training import get_batch
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly, find_repo_root, init_par_groups

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_config import init_single_process_dist


class _RunningStats:
    """Per-(dataset key, variable) count/sum/sum-of-squares accumulator, in float64.

    A batch's worth of a channel's values are folded in via `update`, all
    channels/keys sharing one flat accumulator dict rather than holding
    every value seen in memory -- this is the whole point, since the train
    split is real training data, potentially far larger than fits in RAM.
    """

    def __init__(self):
        self._count = {}
        self._sum = {}
        self._sumsq = {}

    def update(self, key, variable, values):
        values = values.astype(np.float64).ravel()
        self._count[(key, variable)] = self._count.get((key, variable), 0) + values.size
        self._sum[(key, variable)] = self._sum.get((key, variable), 0.0) + values.sum()
        self._sumsq[(key, variable)] = self._sumsq.get((key, variable), 0.0) + (values ** 2).sum()

    def finalize(self):
        """Returns `{key: {variable: {"mean":..., "std":...}}}`, one entry per (key, variable) `update` ever saw."""
        stats = {}
        for (key, variable), count in self._count.items():
            mean = self._sum[(key, variable)] / count
            var = self._sumsq[(key, variable)] / count - mean ** 2
            std = float(np.sqrt(max(var, 0.0)))  # max(...): guards a tiny negative from float roundoff, not real data
            stats.setdefault(key, {})[variable] = {"mean": float(mean), "std": std}
        return stats

    def sample_counts(self):
        """`{(key, variable): count}` -- for the end-of-run summary print, not written to the stats file."""
        return dict(self._count)

    def state(self):
        """Returns this instance's raw accumulator state, for combining across ranks via `merge_state`."""
        return dict(self._count), dict(self._sum), dict(self._sumsq)

    def merge_state(self, state):
        """Folds another instance's `state()` (e.g. gathered from another rank) into this one.

        Exact, not approximate: a sum of per-rank sums (and sums-of-squares,
        and counts) is mathematically identical to accumulating over the
        same values in one place, regardless of how they were partitioned
        across ranks.
        """
        count, sum_, sumsq = state
        for k in count:
            self._count[k] = self._count.get(k, 0) + count[k]
            self._sum[k] = self._sum.get(k, 0.0) + sum_[k]
            self._sumsq[k] = self._sumsq.get(k, 0.0) + sumsq[k]


def _var_name(entry):
    return entry[0] if isinstance(entry, tuple) else entry


def _accumulate_array(stats, key, variables, array, max_per_channel):
    """Folds one `[..., C, ...]`-shaped batch of channel-first data into `stats`.

    Args:
        array: `[B, C, ...]` (or a list of `[B, ...]` arrays, one per
            channel -- "sst"'s own convention, see `UCF_VIT.utils.normalize.
            zscore_normalize`'s identical `data` contract).
        max_per_channel: Stop feeding a given (key, variable) once its
            running count reaches this many values (still counts whatever
            it already has) -- bounds the accumulation pass's own cost for
            `--max-samples`, not a hard global sample cap.
    """
    names = [_var_name(v) for v in variables]
    channels = array if isinstance(array, list) else [np.asarray(array)[:, i] for i in range(len(names))]
    for name, channel in zip(names, channels):
        if max_per_channel is not None and stats.sample_counts().get((key, name), 0) >= max_per_channel:
            continue
        stats.update(key, name, np.asarray(channel))


def _init_distributed():
    """Initializes a real multi-rank `gloo` process group when launched under `srun -n N>1`, else a single-process one.

    Detects a real SLURM multi-task launch via `SLURM_PROCID`/`SLURM_NTASKS`
    -- the same env vars `training_scripts/train.py`'s own `dist_init` reads
    (its Slurm branch only; this utility never needs the mpi4py path, since
    it's launched the same simple way every other `launch/*/*.sh` script
    is). Uses `"gloo"` (CPU), not `train.py`'s `"nccl"` -- this is pure CPU/
    file-IO work, no GPU tensor communication happens anywhere, so there's
    no reason to require real GPUs just to get multi-process/multi-node
    parallelism.

    Returns:
        `(world_rank, world_size)`.
    """
    if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        os.environ.setdefault("MASTER_ADDR", os.environ["HOSTNAME"])
        os.environ.setdefault("MASTER_PORT", "29500")
        world_size = int(os.environ["SLURM_NTASKS"])
        world_rank = int(os.environ["SLURM_PROCID"])
        dist.init_process_group("gloo", timeout=timedelta(seconds=7200), rank=world_rank, world_size=world_size)
        return world_rank, world_size
    init_single_process_dist()
    return 0, 1


def _iterate_iterative_dataloader(conf, split, num_workers, world_rank, world_size):
    """Yields `(dict_key, variables, data_array[, (variables_out, label_array)])` per batch, sharded across `world_size` ranks."""
    start_key, end_key = {
        "train": ("dict_start_idx", "dict_end_idx"),
        "val": ("dict_val_start_idx", "dict_val_end_idx"),
        "test": ("dict_test_start_idx", "dict_test_end_idx"),
    }[split]

    # calculate_load_balancing_on_the_fly reads dataloader.dict_start_idx/
    # dict_end_idx unconditionally (it's train.py's own train-split-only
    # helper) -- swap this split's bounds into those exact keys on a shallow
    # copy so a --split val/test run gets correctly-sized load balancing too.
    split_conf = copy.deepcopy(conf)
    split_conf["dataloader"]["dict_start_idx"] = conf["dataloader"][start_key]
    split_conf["dataloader"]["dict_end_idx"] = conf["dataloader"][end_key]
    split_conf["parallelism"]["data_par_size"] = world_size  # matches this function's own NativePytorchDataModule construction below
    split_conf["dataloader"]["num_workers"] = num_workers  # calculate_load_balancing_on_the_fly's own sizing must match what's actually constructed below

    # world_size == 1 (no real srun -n N>1 launch): ddp_group=None, exactly
    # today's single-process behavior. world_size > 1: a real ddp_group
    # spanning every rank -- tensor_par_size/fsdp_size both 1 (no tensor
    # parallelism or FSDP sharding, only plain data-parallel file sharding
    # across all ranks) -- built the same way train.py builds its own,
    # just with the trivial 1x1xworld_size shape.
    ddp_group = None
    if world_size > 1:
        ddp_group, _, _, _, _ = init_par_groups(
            world_rank=world_rank, data_par_size=world_size, tensor_par_size=1, fsdp_size=1, simple_ddp_size=world_size,
        )

    # Tiling doesn't matter for stats (mean/std over the union of tiles'
    # values is identical to computing it over the whole untiled image, as
    # long as there's no overlap) -- and is actively wrong to leave on: a
    # real config's tile_overlap > 0 would double-count overlapping voxels,
    # biasing the stats, and even with overlap:0 it multiplies the number
    # of samples/batches iterated for zero benefit. Force div=1/overlap=0
    # here, which also means data.tile_size must be recomputed (it's
    # already pre-divided by the real div in parse.py) -- with div=1 and
    # zero overlap, parse.py's own tile_size formula always collapses to
    # exactly effective_size (resize, if any, else img_size), regardless of
    # dimensionality, so no need to replicate its 2D/twoD-3D/3D branches.
    effective_size = conf["dataset_options"]["resize"].get(conf["data"]["dataset"], conf["data"]["img_size"])
    split_conf["data"]["tile_size"] = tuple(effective_size)
    split_conf["tiling"]["div"] = 1
    split_conf["tiling"]["tile_overlap"] = tuple(0 for _ in effective_size)

    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(split_conf)

    data_module = NativePytorchDataModule(
        dict_root_dirs=conf["data"]["dict_root_dirs"],
        dict_start_idx=conf["dataloader"][start_key],
        dict_end_idx=conf["dataloader"][end_key],
        batches_per_rank_epoch=batches_per_rank_epoch,
        dataset_group_list=dataset_group_list,
        # Stats don't care about sample order, so there's no reason to pay
        # for real shuffling -- and with num_workers=0 (single-process,
        # unlike a real training run), ShuffleIterableDataset.__iter__ has
        # to serially fill the whole buffer before yielding anything at
        # all, so a real config's (possibly large) buffer size would add
        # pure startup latency here for zero benefit. 1 is the minimum
        # ShuffleIterableDataset accepts (assert buffer_size > 0) and
        # collapses it to a plain passthrough (see its own __iter__:
        # buffer_size=1 yields each sample immediately, no real shuffling).
        dict_buffer_sizes={k: 1 for k in conf["dataloader"]["dict_buffer_sizes"]},
        dict_in_variables=conf["data"]["dict_in_variables"],
        num_channels_used=conf["data"]["num_channels"],
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=num_workers,
        pin_memory=False,
        interp_size=conf["data"]["interp_size"],
        tile_size=split_conf["data"]["tile_size"],
        twoD=conf["data"]["twoD"],
        return_label=conf["dataloader"]["return_label"],
        div=split_conf["tiling"]["div"],
        tile_overlap=split_conf["tiling"]["tile_overlap"],
        adaptive_patching=False,  # stats need raw per-sample data, not adaptively-patched sequences
        fixed_length=conf["ap"]["fixed_length"],
        data_par_size=world_size,
        ddp_group=ddp_group,
        dataset=conf["data"]["dataset"],
        resize=conf["dataset_options"]["resize"],
        num_classes=conf["model"]["kwargs"].get("num_classes"),
        dict_out_variables=conf["data"]["dict_out_variables"],
        img_size=conf["data"]["img_size"],
        full_domain_size=conf["dataset_options"]["full_domain_size"],
        time_offsets=conf["data"]["time_offsets"],
        normalize_stats=None,  # raw data only -- never accumulate over already-normalized values
    )
    data_module.setup()
    it_loader = iter(data_module.train_dataloader())

    ap_off_conf = copy.deepcopy(conf)
    ap_off_conf["ap"]["do_ap"] = False  # matches adaptive_patching=False above -- get_batch's simpler unpack

    while True:
        try:
            batch = get_batch(ap_off_conf, it_loader)
        except StopIteration:
            return
        dict_key = batch["dict_key"]
        variables = conf["data"]["dict_in_variables"][dict_key]
        data = batch["data"].numpy()
        out = None
        if conf["dataloader"]["return_label"] and conf["data"]["dataset"] == "sst" and conf["data"]["dict_out_variables"]:
            out = (conf["data"]["dict_out_variables"][dict_key], batch["label"].numpy())
        yield dict_key, variables, data, out


def _iterate_catsdogs(conf, split, num_workers, world_rank, world_size):
    """Yields `(dict_key, variables, data_array)` per batch, for `dataloader.type: "dataloader"`, sharded across `world_size` ranks."""
    from UCF_VIT.datasets.catsdogs import CatsDogsDataset, CatsDogsCollate
    from UCF_VIT.utils.misc import slice_file_list
    import glob

    start_key, end_key = {
        "train": ("dict_start_idx", "dict_end_idx"),
        "val": ("dict_val_start_idx", "dict_val_end_idx"),
        "test": ("dict_test_start_idx", "dict_test_end_idx"),
    }[split]
    dkey = next(iter(conf["data"]["dict_root_dirs"]))
    file_list = sorted(glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey], "*.jpg")))
    file_list = slice_file_list(file_list, conf["dataloader"][start_key][dkey], conf["dataloader"][end_key][dkey])
    # No NativePytorchDataModule/gx sharding here (catsdogs' own map-style
    # Dataset path never uses that machinery) -- a plain interleaved slice
    # is enough, since stats don't care about order or which rank sees
    # which file, only that every file is seen exactly once across ranks.
    file_list = file_list[world_rank::world_size]

    # See _iterate_iterative_dataloader's identical comment -- tiling is
    # irrelevant (and, with overlap > 0, actively biasing) for stats, so
    # forced off here too; tile_size with div=1 must be the post-resize
    # whole-image size, matching CatsDogsDataset.__init__'s own docstring
    # ("When div == 1 ... this is the size the whole image actually is once
    # resize ... has been applied").
    resize = conf["dataset_options"]["resize"].get(conf["data"]["dataset"])
    effective_size = resize or conf["data"]["img_size"]
    ds = CatsDogsDataset(
        file_list, conf["data"]["dict_in_variables"][dkey], tuple(effective_size),
        adaptive_patching=False, num_channels=conf["data"]["num_channels"][dkey],
        dataset=conf["data"]["dataset"], resize=resize,
        div=1, tile_overlap=(0, 0),
    )
    loader = DataLoader(
        ds, batch_size=conf["dataloader"]["batch_size"], shuffle=False, num_workers=num_workers,
        collate_fn=functools.partial(CatsDogsCollate, adaptive_patching=False, return_label=conf["dataloader"]["return_label"]),
    )
    for batch in loader:
        data = batch[0].numpy()
        yield dkey, conf["data"]["dict_in_variables"][dkey], data, None


def compute_stats(config_path, split="train", max_samples=None, progress_interval=20, num_workers=0):
    """Computes per-(dataset key, variable) z-score stats over one split of `config_path`'s data.

    Args:
        config_path: Path to a training config YAML -- parsed exactly like
            `train.py`'s own entry point (`UCF_VIT.parse.parse_config`).
        split: `"train"` (default -- the only statistically correct choice
            for stats actually used to train with; see this module's own
            docstring), `"val"`, or `"test"`.
        max_samples: Stop accumulating a given (dataset key, variable) once
            *this rank* has seen at least this many real values -- for a
            quick approximate run over a large dataset; `None` (default)
            uses the entire split. Per-rank, not global: with `world_size`
            real ranks (see `_init_distributed`), the true total accumulated
            for a given (key, variable) can be up to `world_size` times this
            value, not exactly this value.
        progress_interval: Print a progress line every this many batches --
            per-sample decode cost (a real NIfTI/CT volume vs. a small
            imagenet crop) varies wildly enough that a real run's total
            batch count isn't knowable up front, so this reports elapsed
            time/rate and running per-key counts rather than a percentage.
            `0` disables progress printing entirely (still prints the final
            per-(key, variable) summary). Only rank 0 prints (with more
            than one real rank, this reflects only rank 0's own shard's
            progress, not a true global aggregate -- ranks' shards can
            finish at different times depending on file-size imbalance).
        num_workers: DataLoader worker processes for real per-sample file
            decode. `0` (default) keeps everything in this one process --
            safe to run anywhere (a login node, a laptop), but every file
            decode is serial. On a real dedicated node (see `launch/utils/
            run_compute_normalization_stats.sh`), set this to actually use
            the node's other CPU cores. Combines with real multi-rank
            parallelism (below): each rank gets its own `num_workers`.

    Returns:
        `{dataset_key: {variable_name: {"mean": float, "std": float}}}` --
        the same, fully combined result on every rank when running with
        `world_size > 1` real ranks (see `_init_distributed`'s own
        docstring), not just rank 0's own shard.
    """
    world_rank, world_size = _init_distributed()
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args, load_balance_offline=True)

    stats = _RunningStats()
    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches = _iterate_iterative_dataloader(conf, split, num_workers, world_rank, world_size)
    else:
        batches = _iterate_catsdogs(conf, split, num_workers, world_rank, world_size)

    if world_rank == 0:
        print(f"Computing normalization stats over the {split!r} split of {config_path} ({world_size} rank(s))...", flush=True)
    start = time.perf_counter()
    num_batches = 0
    num_samples = 0

    for dict_key, variables, data, out in batches:
        _accumulate_array(stats, dict_key, variables, data, max_samples)
        if out is not None:
            out_variables, label = out
            _accumulate_array(stats, dict_key, out_variables, label, max_samples)

        num_batches += 1
        num_samples += data.shape[0] if not isinstance(data, list) else data[0].shape[0]

        if world_rank == 0 and progress_interval and num_batches % progress_interval == 0:
            elapsed = time.perf_counter() - start
            counts = ", ".join(f"{k}/{v}={c}" for (k, v), c in sorted(stats.sample_counts().items()))
            print(
                f"  batch {num_batches} ({num_samples} samples, {elapsed:.1f}s elapsed, "
                f"{num_samples / elapsed:.1f} samples/s) -- {counts}"
                + (" [rank 0 only]" if world_size > 1 else ""),
                flush=True,
            )

        if max_samples is not None and all(c >= max_samples for c in stats.sample_counts().values()):
            break

    if world_size > 1:
        # Every rank's own local state(), gathered onto every rank -- a
        # fresh accumulator merging all world_size entries (including this
        # rank's own) replaces the local-only one, so what follows (and the
        # return value) reflects the true combined total, not just this
        # rank's shard.
        gathered = [None] * world_size
        dist.all_gather_object(gathered, stats.state())
        merged = _RunningStats()
        for state in gathered:
            merged.merge_state(state)
        stats = merged

    if world_rank == 0:
        elapsed = time.perf_counter() - start
        print(f"Done: {num_batches} batches, {num_samples} samples, {elapsed:.1f}s elapsed" + (" (rank 0)" if world_size > 1 else "") + ".", flush=True)
        for (key, variable), count in sorted(stats.sample_counts().items()):
            print(f"{key}/{variable}: {count} values")

    return stats.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a training config YAML")
    parser.add_argument("--output", required=True, help="Where to write the computed stats YAML")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap per (dataset key, variable) values accumulated -- for a quick approximate run")
    parser.add_argument("--progress-interval", type=int, default=20, help="Print progress every this many batches; 0 disables progress printing")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes for file decode -- 0 (default) is single-process, safe anywhere; set higher on a dedicated node to actually use its other CPU cores")
    args = parser.parse_args()

    stats = compute_stats(
        args.config, split=args.split, max_samples=args.max_samples,
        progress_interval=args.progress_interval, num_workers=args.num_workers,
    )

    # With multiple real ranks (compute_stats's own _init_distributed),
    # `stats` is already the same fully-combined result on every rank --
    # only rank 0 actually writes, so N ranks don't clobber the same file.
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(find_repo_root(), args.output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(stats, f, sort_keys=False)
        print(f"Wrote stats for {sum(len(v) for v in stats.values())} variable(s) across {len(stats)} dataset key(s) to {output_path}")


if __name__ == "__main__":
    main()
