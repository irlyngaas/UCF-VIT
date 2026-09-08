"""Real, multi-rank decode-throughput measurement for one specific config you
point it at, under a genuine `srun`/NCCL launch -- the real-environment
counterpart to tests/dataloaders/test_dataset_speed_real_data.py's
`test_real_decode_throughput_config`.

That single-process version runs as one plain Python process under a bare
`sbatch` (no `srun`), with `parse_config(..., load_balance_offline=True)`
simulating a single 1/`data_par_size`th rank's share and gloo standing in
just to satisfy `dist.is_initialized()`/`get_rank()` calls inside the
dataloader's own sharding logic. That's a real, valid measurement of decode
cost in isolation, but it can't show two things real SAP training actually
has: (1) genuine per-node CPU contention -- `data_par_size` real ranks' worth
of `num_workers` subprocesses all fighting for the same physical cores at
once, versus one process getting the whole node's cores to itself, and (2)
the real launch mechanics (`srun`, real NCCL init, real
`data_par_size * tensor_par_size == world_size` assertion in `parse_config`)
production actually runs under.

This file is that real version: launched via `launch/tests/
run_distributed_tests.sh`'s real `srun -n 8`, reusing the exact same
`dist_info` fixture (real SLURM env vars, real NCCL process group) every
other file in this directory already relies on, and the exact same
real-pipeline construction `test_dataloader_real_pipeline.py` uses
(`parse_config` with no offline flag, `calculate_load_balancing_on_the_fly`,
`NativePytorchDataModule`) -- every rank decodes its own real shard,
concurrently, same as a real training epoch's first few batches. Only
supports `dataloader.type:"iterative_dataloader"` configs, same restriction
as the single-process version, for the same reason (no buffer_size/
ShuffleIterableDataset concept for "dataloader"-type configs).

Like the single-process version: informational, no pass/fail threshold
(`-m dataloader_speed`, excluded by default -- see addopts in pyproject.toml),
skipped entirely unless --speed-config is given, and uses lenient batch-count
timing since an arbitrary config's real data may not have enough narrowed
files to produce a full NUM_BATCHES_TO_PULL per rank regardless of
buffer_size (see _time_batches_lenient's own docstring, and job 5421875's
real basic_ct/sap example, for why this is the right leniency rather than a
hard assertion).

Usage (see launch/tests/run_distributed_tests.sh for the full sbatch
wrapper):
    sbatch run_distributed_tests.sh -m dataloader_speed \\
        -k test_real_decode_throughput_config_distributed \\
        --speed-config ../../configs/basic_ct/sap/base_config.yaml \\
        --speed-buffer-sizes 16,32,64,100
"""

import argparse
import itertools
import os
import sys
import time

import pytest
import torch.distributed as dist
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "integration"))

from run_training_smoke import (  # noqa: E402
    NoRealDataFoundError,
    compute_narrow_dict_idx,
    inflate_min_files_for_train_split,
)

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule  # noqa: E402
from UCF_VIT.parse import parse_config  # noqa: E402
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly  # noqa: E402

pytestmark = pytest.mark.dataloader_speed

NUM_BATCHES_TO_PULL = 4


def _narrowed_generic_config_path(base_config_path, num_workers, buffer_size, world_rank, tag):
    """Same idea as tests/dataloaders/test_dataset_speed_real_data.py's
    `_narrowed_generic_config_path`, but sized for this file's real
    `data_par_size` real ranks (no single-simulated-rank factor needed --
    every rank here really is its own real rank). Per-rank tag (via
    `world_rank`) avoids concurrent ranks racing the same scratch file, same
    as `test_dataloader_real_pipeline.py`'s own narrowing helpers.

    Returns:
        `(config_path, real_buffer_size)`.
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
    scratch_dir = f"/tmp/{job_id}/dataloader_speed_real_pipeline"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"{tag}-{world_rank}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path, real_buffer_size


def _build_data_module(config_path):
    """Same construction as test_dataloader_real_pipeline.py's own
    `_build_data_module` -- real `parse_config` (no offline flag), so this
    rank's real world rank/size must match the config's real
    `data_par_size * tensor_par_size`, exactly as real training enforces.
    """
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args)
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


def _time_batches_lenient(loader, max_batches):
    """Same idea (and same reasoning) as tests/dataloaders/
    test_dataset_speed_real_data.py's identical helper -- duplicated rather
    than imported across sibling test-module files, matching this repo's
    convention. See that file's own docstring for the real Frontier example
    (job 5421875) that motivated leniency over a hard batch-count assert.

    Returns:
        `(elapsed_total, batches_pulled, time_to_first_batch)`.
    """
    it = iter(loader)
    start = time.perf_counter()
    try:
        first = next(it)
    except StopIteration:
        pytest.fail("pulled 0 batches -- no real narrowed data available at all (see --speed-config's own real data)")
    time_to_first_batch = time.perf_counter() - start

    batches = [first] + list(itertools.islice(it, max_batches - 1))
    elapsed_total = time.perf_counter() - start
    return elapsed_total, len(batches), time_to_first_batch


def test_real_decode_throughput_config_distributed(dist_info, request, speed_buffer_size, speed_num_workers):
    """Real, concurrent multi-rank decode throughput for --speed-config.

    Every real rank narrows/builds/times its own real shard independently,
    synchronized by a barrier immediately beforehand so the timed region
    starts at roughly the same wall-clock moment across ranks -- the whole
    point being to observe real per-node CPU contention across
    `data_par_size` ranks' worth of `num_workers` subprocesses, which the
    single-process version (tests/dataloaders/test_dataset_speed_real_data.py)
    cannot produce at all. Results from every rank are gathered onto rank 0,
    which prints a per-rank breakdown plus an aggregate (real total samples
    pulled across every rank, divided by the *slowest* rank's own elapsed
    time) -- that aggregate approximates real steady-state per-node
    throughput under contention, which is the number that actually matters
    for judging whether a buffer_size/num_workers setting is well-chosen for
    real SAP-style training, not any single rank's isolated number.
    """
    speed_config = request.config.getoption("--speed-config")
    if not speed_config:
        pytest.skip("no --speed-config given -- see this test's own docstring for usage")

    label = os.path.splitext(os.path.basename(speed_config))[0]
    config_path, real_buffer_size = _narrowed_generic_config_path(
        speed_config, speed_num_workers, speed_buffer_size, dist_info["world_rank"],
        f"{label}-{speed_num_workers}-{speed_buffer_size}",
    )
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()

    dist.barrier()
    elapsed, batches_pulled, time_to_first_batch = _time_batches_lenient(loader, NUM_BATCHES_TO_PULL)

    result = (dist_info["world_rank"], elapsed, batches_pulled, time_to_first_batch)
    gathered = [None] * dist_info["world_size"]
    dist.all_gather_object(gathered, result)

    if dist_info["world_rank"] == 0:
        batch_size = conf["dataloader"]["batch_size"]
        print(
            f"\n{label} num_workers={speed_num_workers} buffer_size={real_buffer_size} "
            f"(world_size={dist_info['world_size']}, real concurrent ranks):"
        )
        total_samples = 0
        max_elapsed = 0.0
        for world_rank, rank_elapsed, rank_batches, rank_ttfb in sorted(gathered):
            samples = rank_batches * batch_size
            total_samples += samples
            max_elapsed = max(max_elapsed, rank_elapsed)
            rate = samples / rank_elapsed if rank_elapsed > 0 else float("inf")
            print(
                f"  rank {world_rank}: {rank_elapsed:.3f}s for {rank_batches} batches "
                f"({samples} samples, {rate:,.1f} samples/s), time_to_first_batch={rank_ttfb:.3f}s"
                + (f" (only {rank_batches}/{NUM_BATCHES_TO_PULL} batches available from real data)"
                   if rank_batches < NUM_BATCHES_TO_PULL else "")
            )
        aggregate_rate = total_samples / max_elapsed if max_elapsed > 0 else float("inf")
        print(f"  aggregate (under real contention): {total_samples} samples / {max_elapsed:.3f}s = {aggregate_rate:,.1f} samples/s")
