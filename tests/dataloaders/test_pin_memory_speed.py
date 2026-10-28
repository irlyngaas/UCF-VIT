"""Informational pin_memory / non_blocking transfer-speed measurements.

Answers the question that came up while looking at training.py's
dataloader-sourced `.to(device)` calls: `non_blocking=True` only produces a
real asynchronous host->device copy when the source CPU tensor is pinned
(`DataLoader(pin_memory=True)`) -- every shipped config currently sets
`dataloader.pin_memory: False`, so as of today `non_blocking=True` would be
a no-op there. This file measures the actual delta with real numbers,
rather than deciding by assumption whether flipping `pin_memory:True` (and
wiring `non_blocking=True` into `training.py`'s `.to(device)` calls) is
worth it.

Like the other `tests/dataloaders/*speed*.py` files: no pass/fail threshold,
informational only, not run by default (see `addopts` in `pyproject.toml`).
Read the printed numbers by eye:

    pytest -m dataloader_speed -s

Needs a real GPU -- unlike `test_dataset_speed.py`, pinning's whole effect
is on the host->device copy, so there's nothing meaningful to measure
without one. Skips cleanly (module-level) if none is visible.

Two levels:
  - `test_pinned_vs_pageable_transfer_time`: a synthetic microbenchmark
    isolating pure H2D transfer cost (pinned vs. pageable source memory,
    crossed with `non_blocking` True/False) from any real decode/dataloader
    overhead. This measures raw transfer bandwidth and per-call CPU/GPU
    handoff overhead -- it does NOT simulate overlapping the transfer with
    real GPU compute (there's no compute here to overlap with), so it can't
    show the "hide the copy behind other work" half of `non_blocking`'s
    benefit -- see the module docstring discussion in the PR/commit this
    file shipped with for why that part is limited anyway in this
    codebase's current `process_batch` (data moves to device, then
    `forward_step` uses it almost immediately, with little CPU work
    in between to overlap with).
  - `test_real_decode_then_transfer_basic_ct_unetr`: the same real
    basic_ct/unetr `NativePytorchDataModule` construction
    `test_dataset_speed_real_data.py` uses, with `dataloader.pin_memory`
    swept True/False, timing pulling N real batches *and* moving each to
    device -- what a training run's per-iteration cost would actually look
    like end to end, decode included.
"""

import argparse
import itertools
import os
import sys
import time

import pytest
import torch
import yaml

pytestmark = pytest.mark.dataloader_speed

if not torch.cuda.is_available():
    pytest.skip("pin_memory's effect is on host->device transfer -- needs a real GPU", allow_module_level=True)

DEVICE = torch.device("cuda:0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "integration"))

from run_training_smoke import NoRealDataFoundError, compute_narrow_dict_idx, inflate_min_files_for_train_split  # noqa: E402

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule  # noqa: E402
from UCF_VIT.parse import parse_config  # noqa: E402
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly  # noqa: E402

BASIC_CT_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml")
NUM_BATCHES_TO_PULL = 4
# basic_ct/unetr's real batch_size=4 (UNETR's own decoder-memory exception),
# data_par_size=8 -- same reasoning as test_dataset_speed_real_data.py's own
# BASIC_CT_MIN_FILES.
BASIC_CT_MIN_FILES = 4 * NUM_BATCHES_TO_PULL * 8

NUM_TRANSFERS = 50
# ~4MB and ~64MB float32 batches -- small enough to run fast, large enough
# that transfer time isn't dominated by fixed per-call launch overhead.
BATCH_SHAPES = {
    "small (~4MB)": (8, 3, 224, 224),
    "large (~64MB)": (32, 3, 512, 512),
}


def _time_transfers(source, non_blocking):
    # Warm up -- the first CUDA op on a process pays a one-time context/
    # allocator setup cost unrelated to what's being measured here.
    source.to(DEVICE, non_blocking=non_blocking)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(NUM_TRANSFERS):
        source.to(DEVICE, non_blocking=non_blocking)
    torch.cuda.synchronize()
    return time.perf_counter() - start


@pytest.mark.parametrize("shape_label", list(BATCH_SHAPES))
def test_pinned_vs_pageable_transfer_time(shape_label):
    shape = BATCH_SHAPES[shape_label]

    print(f"\nH2D transfer, {shape_label} batches ({NUM_TRANSFERS} transfers each):")
    for pinned, non_blocking in itertools.product((False, True), (False, True)):
        source = torch.empty(shape, dtype=torch.float32, pin_memory=pinned)
        elapsed = _time_transfers(source, non_blocking)
        print(
            f"  pinned={pinned!s:5} non_blocking={non_blocking!s:5}: "
            f"{elapsed:.4f}s ({elapsed / NUM_TRANSFERS * 1000:.2f}ms/transfer)"
        )


def _narrowed_config_path(pin_memory):
    with open(BASIC_CT_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    try:
        # inflate_min_files_for_train_split: see its own docstring -- without
        # it, this tight, no-margin-by-design target can lose its only batch
        # to the automatic train/val/test split.
        narrow_end_idx = compute_narrow_dict_idx(conf, inflate_min_files_for_train_split(conf, BASIC_CT_MIN_FILES))
    except NoRealDataFoundError as e:
        pytest.skip(str(e))

    conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
    conf["dataloader"]["dict_end_idx"] = narrow_end_idx
    conf["dataloader"]["pin_memory"] = pin_memory

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/pin_memory_speed"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"basic_ct-pin_memory_{pin_memory}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _build_loader(pin_memory):
    config_path = _narrowed_config_path(pin_memory)
    args = argparse.Namespace(config=config_path, pretrained_config="")
    # load_balance_offline=True: see test_dataset_speed_real_data.py's own
    # _build_data_module for why (this file is single-process too).
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
        num_classes=conf["model"]["kwargs"]["num_classes"],
    )
    data_module.setup()
    return conf, data_module.train_dataloader()


@pytest.mark.parametrize("pin_memory", [False, True])
def test_real_decode_then_transfer_basic_ct_unetr(pin_memory):
    """Real NIfTI decode (same construction as
    test_dataset_speed_real_data.py's basic_ct/unetr case) plus the
    .to(device) transfer every batch actually goes through in training --
    end-to-end per-iteration cost, not just decode.
    """
    conf, loader = _build_loader(pin_memory)
    batch_size = conf["dataloader"]["batch_size"]

    start = time.perf_counter()
    count = 0
    for inp, label, variables, dict_key in itertools.islice(loader, NUM_BATCHES_TO_PULL):
        # non_blocking=True is required here -- without it these copies are
        # blocking regardless of pin_memory, and the whole point of this
        # test is measuring pin_memory's real (async-copy) benefit, not just
        # the smaller, always-present blocking-transfer bandwidth edge
        # pinned memory also happens to have.
        inp.to(DEVICE, non_blocking=True)
        label.to(DEVICE, non_blocking=True)
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert count == NUM_BATCHES_TO_PULL, (
        f"only pulled {count}/{NUM_BATCHES_TO_PULL} batches -- not enough real narrowed "
        f"data available (see BASIC_CT_MIN_FILES)"
    )
    samples = NUM_BATCHES_TO_PULL * batch_size
    rate = samples / elapsed if elapsed > 0 else float("inf")
    print(
        f"\nbasic_ct/unetr pin_memory={pin_memory}: {elapsed:.3f}s for "
        f"{NUM_BATCHES_TO_PULL} batches ({samples} samples, {rate:,.1f} samples/s) "
        f"decode+transfer combined"
    )
