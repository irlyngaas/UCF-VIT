"""Informational REAL-decode throughput measurements, against real basic_ct,
imagenet, and catsdogs data on Frontier.

test_dataset_speed.py's num_workers test uses a synthetic, made-up
time.sleep() delay standing in for real disk I/O -- useful for confirming
num_workers parallelizes *something*, but not for judging whether a real
config's num_workers setting is actually well-chosen. This file replaces
that stand-in with genuine NIfTI/JPEG decode, through the same real
parse_config-based construction tests/distributed/test_dataloader_real_
pipeline.py uses (parse_config + calculate_load_balancing_on_the_fly +
NativePytorchDataModule for basic_ct/imagenet's "iterative_dataloader";
CatsDogsDataset + DistributedSampler + DataLoader straight from parse_config
output for catsdogs's "dataloader" type) -- just single-process instead of
across real 8 ranks, since throughput timing doesn't need real DDP-rank
sharding to be meaningful, only real decode cost. parse_config is called
with load_balance_offline=True for exactly that reason: the real shipped
configs all have data_par_size=8, and this file only ever runs as rank 0 of
1 real process, not 8.

Like test_dataset_speed.py: no pass/fail threshold, not run by default (see
addopts in pyproject.toml), and each test times pulling a few real batches
per num_workers setting and prints the result -- read the numbers by eye
(`pytest -m dataloader_speed -s`) rather than relying on an assertion.
Unlike test_dataset_speed.py, this needs real Frontier data to mean
anything: each test skips (not fails) if the real config's data isn't
reachable from wherever this runs, same as tests/distributed/'s real-data
files.

num_workers sweeps up to 7 (not just up to 4, as tests/distributed/
test_dataloader_real_data.py's real-*rank* sharding test does) to match
Frontier's real node core counts -- this file doesn't have that test's
real-multi-rank cost to worry about, since it's single-process by design.

Caveat: with num_workers > 0, iterating the loader lazily spawns worker
subprocesses on first use, and that startup cost is included in the timed
region (matching how a real training loop experiences it once per epoch,
not excluded as a one-time setup cost) -- for small NUM_BATCHES_TO_PULL,
worker startup can dominate the measurement rather than steady-state decode
throughput. Interpret num_workers > 0 results with that in mind, especially
at higher worker counts.

test_real_decode_throughput_config is a fourth, generic test, separate from
the three fixed per-dataset ones above: it runs against whatever real config
you point it at via --speed-config, instead of one of the three fixed
configs above. For investigating one specific real config/problem (e.g. "is
basic_ct/sap's buffer_size:100 too large for its real per-rank shard
size?") without paying for a sweep over every shipped config -- skipped
entirely unless --speed-config is given. See its own docstring, and
conftest.py's --speed-config/--speed-buffer-sizes/--speed-num-workers
option docs, for usage.
"""

import argparse
import glob
import itertools
import os
import sys
import time

import pytest
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "integration"))

from run_training_smoke import (  # noqa: E402
    NoRealDataFoundError,
    compute_narrow_dict_idx,
    create_narrow_catsdogs_dir,
    inflate_min_files_for_train_split,
)

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule  # noqa: E402
from UCF_VIT.parse import parse_config  # noqa: E402
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly  # noqa: E402

pytestmark = pytest.mark.dataloader_speed

BASIC_CT_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml")
IMAGENET_CONFIG = os.path.join(REPO_ROOT, "configs", "imagenet", "classification", "base_config.yaml")
CATSDOGS_CONFIG = os.path.join(REPO_ROOT, "configs", "catsdogs", "classification", "base_config.yaml")

NUM_WORKERS_VALUES = [0, 1, 4, 7]
NUM_BATCHES_TO_PULL = 4

# Same min_files reasoning as tests/distributed/test_dataloader_real_pipeline.py:
# basic_ct's baseline config now has do_tiling=False/twoD=False/do_ap=False,
# so each real file is exactly one sample, no tiling/z-slice multiplication
# -- min_files is a *total* count that the config's declared data_par_size=8
# then divides (same as catsdogs below). imagenet's min_files targets
# samples per *bucket* -- process_root_dirs no longer buckets internally
# (see its own docstring), so _narrowed_config_path multiplies by
# data_par_size before narrowing. All three also get scaled up via
# inflate_min_files_for_train_split so this many files survive the automatic
# train/val/test split, not just the initial narrowing -- without that,
# these tight, no-margin-by-design targets can end up with 0 real batches to
# time (see that function's own docstring for the real Frontier
# ZeroDivisionError this fixes).
BASIC_CT_MIN_FILES = 4 * NUM_BATCHES_TO_PULL * 8  # basic_ct/unetr's real batch_size=4 (UNETR's own decoder-memory exception -- see its config comment)
IMAGENET_MIN_FILES = 32 * NUM_BATCHES_TO_PULL + 32  # >= 32 (real batch_size) * NUM_BATCHES_TO_PULL, with margin -- per bucket
CATSDOGS_MIN_FILES = 32 * NUM_BATCHES_TO_PULL * 8


def _narrowed_config_path(base_config_path, min_files, num_workers, tag):
    with open(base_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    narrow_min_files = min_files
    if conf["data"]["dataset"] == "imagenet":
        # data_par_size isn't a real key on this raw, un-parsed YAML dict --
        # see test_dataloader_real_pipeline.py's identical comment.
        narrow_min_files *= conf["parallelism"]["fsdp_size"] * conf["parallelism"]["simple_ddp_size"]
    narrow_min_files = inflate_min_files_for_train_split(conf, narrow_min_files)

    try:
        narrow_end_idx = compute_narrow_dict_idx(conf, narrow_min_files)
    except NoRealDataFoundError as e:
        pytest.skip(str(e))

    conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
    conf["dataloader"]["dict_end_idx"] = narrow_end_idx
    conf["dataloader"]["num_workers"] = num_workers

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/dataloader_speed_real"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"{tag}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _build_data_module(config_path):
    args = argparse.Namespace(config=config_path, pretrained_config="")
    # load_balance_offline=True: this file runs single-process, but the real
    # shipped configs all declare data_par_size=8 -- offline mode skips the
    # data_par_size * tensor_par_size == dist.get_world_size() assertion that
    # would otherwise fail. calculate_load_balancing_on_the_fly and
    # NativePytorchDataModule both derive their own sharding from the
    # config's stated data_par_size, not the real world size, so this
    # process still ends up decoding a real, correctly-shaped ~1/8th share
    # of the narrowed data -- not the full config's worth, but genuinely
    # real work, not a full-scale measurement.
    conf = parse_config(args, load_balance_offline=True)
    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)

    data_module = NativePytorchDataModule(
        dict_root_dirs=conf["data"]["dict_root_dirs"],
        dict_start_idx=conf["dataloader"]["dict_start_idx"],
        dict_end_idx=conf["dataloader"]["dict_end_idx"],
        dict_buffer_sizes=conf["dataloader"]["dict_buffer_sizes"],
        dict_in_variables=conf["data"]["dict_in_variables"],
        num_channels_used=conf["data"]["num_channels"],
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        pin_memory=conf["dataloader"]["pin_memory"],
        interp_size=conf["data"]["interp_size"],
        tile_size=conf["data"]["tile_size"],
        twoD=conf["data"]["twoD"],
        return_label=conf["dataloader"]["return_label"],
        dataset_group_list=dataset_group_list,
        batches_per_rank_epoch=batches_per_rank_epoch,
        div=conf["tiling"]["div"],
        tile_overlap=conf["tiling"]["tile_overlap"],
        adaptive_patching=conf["ap"]["do_ap"],
        fixed_length=conf["ap"]["fixed_length"],
        separate_channels=conf["ap"]["separate_channels"],
        data_par_size=conf["parallelism"]["data_par_size"],
        dataset=conf["data"]["dataset"],
        resize=conf["dataset_options"]["resize"],
        num_classes=conf["model"]["kwargs"]["num_classes"] if conf["model"]["type"] in ["UNETR", "SAP"] else None,
    )
    data_module.setup()
    return conf, data_module


def _narrowed_catsdogs_config_path(num_workers, tag):
    with open(CATSDOGS_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/dataloader_speed_real/{tag}"
    try:
        dkey, narrowed_dir = create_narrow_catsdogs_dir(
            conf, scratch_dir, inflate_min_files_for_train_split(conf, CATSDOGS_MIN_FILES),
        )
    except NoRealDataFoundError as e:
        pytest.skip(str(e))
    conf["data"]["dict_root_dirs"][dkey] = narrowed_dir
    conf["dataloader"]["num_workers"] = num_workers

    out_path = os.path.join(scratch_dir, "catsdogs.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _build_catsdogs_loader(config_path):
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args, load_balance_offline=True)

    dkey_train = list(conf["data"]["dict_root_dirs"])[0]
    train_list = glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey_train], "*.jpg"))
    train_data = conf["dataloader"]["dataset_module"](
        train_list, conf["data"]["dict_in_variables"][dkey_train], conf["data"]["tile_size"],
        adaptive_patching=conf["ap"]["do_ap"], fixed_length=conf["ap"]["fixed_length"],
        interp_size=conf["data"]["interp_size"], num_channels=conf["data"]["num_channels"][dkey_train],
        dataset=conf["data"]["dataset"],
    )
    train_sampler = DistributedSampler(
        train_data, shuffle=True, num_replicas=conf["parallelism"]["data_par_size"], rank=dist.get_rank(),
    )
    loader = DataLoader(
        dataset=train_data, sampler=train_sampler, num_workers=conf["dataloader"]["num_workers"],
        pin_memory=conf["dataloader"]["pin_memory"], batch_size=conf["dataloader"]["batch_size"], drop_last=True,
        collate_fn=lambda batch: conf["dataloader"]["collate_fn"](
            batch, adaptive_patching=conf["ap"]["do_ap"], return_label=conf["dataloader"]["return_label"],
        ),
    )
    return conf, loader


def _time_batches(loader, num_batches):
    start = time.perf_counter()
    batches = list(itertools.islice(loader, num_batches))
    elapsed = time.perf_counter() - start
    assert len(batches) == num_batches, (
        f"only pulled {len(batches)}/{num_batches} batches -- not enough real narrowed "
        f"data available for this many batches (see the *_MIN_FILES constants)"
    )
    return elapsed


def _report(dataset_label, num_workers, elapsed, num_batches, batch_size, extra=""):
    samples = num_batches * batch_size
    rate = samples / elapsed if elapsed > 0 else float("inf")
    print(
        f"\n{dataset_label} num_workers={num_workers}: {elapsed:.3f}s for {num_batches} "
        f"batches ({samples} samples, {rate:,.1f} samples/s){extra}"
    )


@pytest.mark.parametrize("num_workers", NUM_WORKERS_VALUES)
def test_real_decode_throughput_basic_ct_unetr(num_workers):
    """Real NIfTI decode, no tiling/adaptive patching (baseline config)."""
    config_path = _narrowed_config_path(BASIC_CT_CONFIG, BASIC_CT_MIN_FILES, num_workers, f"basic_ct-{num_workers}")
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()
    elapsed = _time_batches(loader, NUM_BATCHES_TO_PULL)
    _report("basic_ct/unetr", num_workers, elapsed, NUM_BATCHES_TO_PULL, conf["dataloader"]["batch_size"])


@pytest.mark.parametrize("num_workers", NUM_WORKERS_VALUES)
def test_real_decode_throughput_imagenet_classification(num_workers):
    """Real JPEG decode + resize, non-adaptive-patching."""
    config_path = _narrowed_config_path(IMAGENET_CONFIG, IMAGENET_MIN_FILES, num_workers, f"imagenet-{num_workers}")
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()
    elapsed = _time_batches(loader, NUM_BATCHES_TO_PULL)
    _report("imagenet/classification", num_workers, elapsed, NUM_BATCHES_TO_PULL, conf["dataloader"]["batch_size"])


@pytest.mark.parametrize("num_workers", NUM_WORKERS_VALUES)
def test_real_decode_throughput_catsdogs_classification(num_workers):
    """Real JPEG decode + resize via CatsDogsDataset, the "dataloader" type
    -- a different production code path from the other two, and (unlike
    them) no ShuffleIterableDataset/buffer_size in the mix at all.
    """
    config_path = _narrowed_catsdogs_config_path(num_workers, f"catsdogs-{num_workers}")
    conf, loader = _build_catsdogs_loader(config_path)
    elapsed = _time_batches(loader, NUM_BATCHES_TO_PULL)
    _report("catsdogs/classification", num_workers, elapsed, NUM_BATCHES_TO_PULL, conf["dataloader"]["batch_size"])


def _narrowed_generic_config_path(base_config_path, num_workers, buffer_size, tag):
    """Same idea as `_narrowed_config_path`, but for an arbitrary
    `iterative_dataloader`-type config instead of one of the three fixed
    ones above -- `min_files` is derived from the config's own
    `batch_size`/`fsdp_size`/`simple_ddp_size` (there's no shipped-config-
    specific constant to reach for), and inflated enough to comfortably
    exceed `buffer_size` too, not just `NUM_BATCHES_TO_PULL` batches --
    otherwise every swept `buffer_size` would exceed the narrowed per-rank
    shard and `ShuffleIterableDataset` would never reach its steady-state
    swap-and-yield behavior for *any* of them, hiding the exact effect this
    is meant to measure (see this module's own docstring).

    Returns:
        `(config_path, real_buffer_size)` -- `real_buffer_size` is what was
        actually written into the config (the given `buffer_size`, or the
        config's own shipped value when `buffer_size` is `None`), for
        `_report`'s label.
    """
    with open(base_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    assert conf["dataloader"]["type"] == "iterative_dataloader", (
        f"{base_config_path}: --speed-config only supports dataloader.type:"
        f"\"iterative_dataloader\" configs (got {conf['dataloader']['type']!r}) -- "
        f"\"dataloader\"-type configs (catsdogs) have no buffer_size/ShuffleIterableDataset "
        f"in the pipeline at all, so this test has nothing meaningful to sweep for them."
    )

    dkey = next(iter(conf["dataloader"]["dict_buffer_sizes"]))
    real_buffer_size = buffer_size if buffer_size is not None else conf["dataloader"]["dict_buffer_sizes"][dkey]

    data_par_size = conf["parallelism"]["fsdp_size"] * conf["parallelism"]["simple_ddp_size"]
    per_rank_target = max(real_buffer_size, conf["dataloader"]["batch_size"] * NUM_BATCHES_TO_PULL) * 2
    min_files = per_rank_target * data_par_size
    if conf["data"]["dataset"] == "imagenet":
        # data_par_size isn't a real key on this raw, un-parsed YAML dict --
        # see test_dataloader_real_pipeline.py's identical comment.
        min_files *= data_par_size
    min_files = inflate_min_files_for_train_split(conf, min_files)

    try:
        narrow_end_idx = compute_narrow_dict_idx(conf, min_files)
    except NoRealDataFoundError as e:
        pytest.skip(str(e))

    conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
    conf["dataloader"]["dict_end_idx"] = narrow_end_idx
    conf["dataloader"]["num_workers"] = num_workers
    conf["dataloader"]["dict_buffer_sizes"] = {k: real_buffer_size for k in conf["dataloader"]["dict_buffer_sizes"]}

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/dataloader_speed_real"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"{tag}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path, real_buffer_size


def test_real_decode_throughput_config(request, speed_buffer_size, speed_num_workers):
    """Real decode throughput for one specific config you point at, instead
    of one of the three fixed configs above -- for investigating one real
    problem (e.g. "is basic_ct/sap's buffer_size:100 too large for its real
    per-rank shard size?") without paying for a sweep over every shipped
    config, which would be far more expensive than needed for a single
    question. Skipped entirely unless --speed-config is given.

    Usage (see launch/tests/run_dataloader_speed.sh for the full sbatch
    wrapper -- add these flags to its own `pytest -m dataloader_speed ...`
    line):
        --speed-config ../../configs/basic_ct/sap/base_config.yaml
        --speed-buffer-sizes 16,32,64,100   # optional -- defaults to the config's own shipped value
        --speed-num-workers 0,1             # optional -- defaults to the config's own shipped value

    Only supports dataloader.type:"iterative_dataloader" configs (asserted
    in _narrowed_generic_config_path) -- "dataloader"-type configs (catsdogs)
    have no buffer_size/ShuffleIterableDataset in the pipeline to sweep.
    """
    speed_config = request.config.getoption("--speed-config")
    if not speed_config:
        pytest.skip("no --speed-config given -- see this test's own docstring for usage")

    label = os.path.splitext(os.path.basename(speed_config))[0]
    config_path, real_buffer_size = _narrowed_generic_config_path(
        speed_config, speed_num_workers, speed_buffer_size, f"{label}-{speed_num_workers}-{speed_buffer_size}",
    )
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()
    elapsed = _time_batches(loader, NUM_BATCHES_TO_PULL)
    _report(
        label, speed_num_workers, elapsed, NUM_BATCHES_TO_PULL, conf["dataloader"]["batch_size"],
        extra=f", buffer_size={real_buffer_size}",
    )
