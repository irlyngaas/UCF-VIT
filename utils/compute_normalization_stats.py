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

Deliberately builds the dataloader with `adaptive_patching` forced off and
`normalize_stats` left empty regardless of what the config says -- stats
are computed over the *raw* per-sample data (not adaptively-patched
sequences, and not already-normalized data from some existing
`normalize_stats_path` the config might already point at).

Usage:
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --split val
    python utils/compute_normalization_stats.py path/to/config.yaml --output stats.yaml --max-samples 2000
"""

import argparse
import copy
import functools
import os
import sys

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


def _iterate_iterative_dataloader(conf, split):
    """Yields `(dict_key, variables, data_array[, (variables_out, label_array)])` per batch."""
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
    split_conf["parallelism"]["data_par_size"] = 1  # matches this function's own single-process NativePytorchDataModule below
    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(split_conf)

    data_module = NativePytorchDataModule(
        dict_root_dirs=conf["data"]["dict_root_dirs"],
        dict_start_idx=conf["dataloader"][start_key],
        dict_end_idx=conf["dataloader"][end_key],
        batches_per_rank_epoch=batches_per_rank_epoch,
        dataset_group_list=dataset_group_list,
        dict_buffer_sizes=conf["dataloader"]["dict_buffer_sizes"],
        dict_in_variables=conf["data"]["dict_in_variables"],
        num_channels_used=conf["data"]["num_channels"],
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        interp_size=conf["data"]["interp_size"],
        tile_size=conf["data"]["tile_size"],
        twoD=conf["data"]["twoD"],
        return_label=conf["dataloader"]["return_label"],
        div=conf["tiling"]["div"],
        tile_overlap=conf["tiling"]["tile_overlap"],
        adaptive_patching=False,  # stats need raw per-sample data, not adaptively-patched sequences
        fixed_length=conf["ap"]["fixed_length"],
        data_par_size=1,
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


def _iterate_catsdogs(conf, split):
    """Yields `(dict_key, variables, data_array)` per batch, for `dataloader.type: "dataloader"`."""
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

    ds = CatsDogsDataset(
        file_list, conf["data"]["dict_in_variables"][dkey], conf["data"]["tile_size"],
        adaptive_patching=False, num_channels=conf["data"]["num_channels"][dkey],
        dataset=conf["data"]["dataset"], resize=conf["dataset_options"]["resize"].get(conf["data"]["dataset"]),
        div=conf["tiling"]["div"], tile_overlap=conf["tiling"]["tile_overlap"],
    )
    loader = DataLoader(
        ds, batch_size=conf["dataloader"]["batch_size"], shuffle=False, num_workers=0,
        collate_fn=functools.partial(CatsDogsCollate, adaptive_patching=False, return_label=conf["dataloader"]["return_label"]),
    )
    for batch in loader:
        data = batch[0].numpy()
        yield dkey, conf["data"]["dict_in_variables"][dkey], data, None


def compute_stats(config_path, split="train", max_samples=None):
    """Computes per-(dataset key, variable) z-score stats over one split of `config_path`'s data.

    Args:
        config_path: Path to a training config YAML -- parsed exactly like
            `train.py`'s own entry point (`UCF_VIT.parse.parse_config`).
        split: `"train"` (default -- the only statistically correct choice
            for stats actually used to train with; see this module's own
            docstring), `"val"`, or `"test"`.
        max_samples: Stop accumulating a given (dataset key, variable) once
            it's seen at least this many real values -- for a quick
            approximate run over a large dataset; `None` (default) uses the
            entire split.

    Returns:
        `{dataset_key: {variable_name: {"mean": float, "std": float}}}`.
    """
    init_single_process_dist()
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args, load_balance_offline=True)

    stats = _RunningStats()
    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches = _iterate_iterative_dataloader(conf, split)
    else:
        batches = _iterate_catsdogs(conf, split)

    for dict_key, variables, data, out in batches:
        _accumulate_array(stats, dict_key, variables, data, max_samples)
        if out is not None:
            out_variables, label = out
            _accumulate_array(stats, dict_key, out_variables, label, max_samples)
        if max_samples is not None and all(c >= max_samples for c in stats.sample_counts().values()):
            break

    counts = stats.sample_counts()
    for (key, variable), count in sorted(counts.items()):
        print(f"{key}/{variable}: {count} values")

    return stats.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to a training config YAML")
    parser.add_argument("--output", required=True, help="Where to write the computed stats YAML")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap per (dataset key, variable) values accumulated -- for a quick approximate run")
    args = parser.parse_args()

    stats = compute_stats(args.config, split=args.split, max_samples=args.max_samples)

    output_path = args.output if os.path.isabs(args.output) else os.path.join(find_repo_root(), args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(stats, f, sort_keys=False)
    print(f"Wrote stats for {sum(len(v) for v in stats.values())} variable(s) across {len(stats)} dataset key(s) to {output_path}")


if __name__ == "__main__":
    main()
