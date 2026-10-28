import os
import socket

import pytest
import torch.distributed as dist


def _slurm_launch_available():
    return all(k in os.environ for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))


@pytest.fixture(scope="session", autouse=True)
def _single_process_distributed():
    """Initializes a single-process (world_size=1) torch.distributed group for the whole test session.

    Several UCF_VIT functions (e.g. `parse_config`) call `dist.get_rank()` even
    outside of an actual multi-process training launch. This fixture lets those
    calls succeed locally, without requiring a real multi-GPU/SLURM allocation.

    This conftest.py is a parent of tests/distributed/, so this autouse fixture
    would otherwise also fire for Tier 2 tests, racing tests/distributed/conftest.py's
    own real multi-process init to call `dist.init_process_group` first and making
    the second call fail with "trying to initialize the default process group
    twice!". Step aside entirely under a real SLURM launch and let that conftest
    own initialization instead.
    """
    if _slurm_launch_available():
        yield
    elif dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        yield
        dist.destroy_process_group()
    else:
        yield


def pytest_addoption(parser):
    """CLI options shared by tests/dataloaders/test_dataset_speed_real_data.py's
    single-process test_real_decode_throughput_config and tests/distributed/
    test_dataloader_speed_real_pipeline.py's real multi-rank counterpart.

    Kept here (a shared ancestor of both) rather than in either directory's
    own conftest.py: pytest loads every conftest.py under `testpaths` (see
    pyproject.toml -- testpaths = ["tests"]) for a bare `pytest` invocation,
    so registering the same --speed-config/etc. options in two sibling
    conftest.py files would crash argparse with a duplicate-option error the
    moment both tests/dataloaders/ and tests/distributed/ are collected
    together, not just when either speed test actually runs.
    """
    parser.addoption(
        "--speed-config",
        action="store",
        default=None,
        help=(
            "Path to a real config YAML to run test_real_decode_throughput_config "
            "against (e.g. ../../configs/basic_ct/sap/base_config.yaml). Only that "
            "one config runs -- deliberately not a sweep over every shipped config, "
            "since real decode timing is expensive. Skipped entirely if omitted."
        ),
    )
    parser.addoption(
        "--speed-buffer-sizes",
        action="store",
        default="",
        help=(
            "Comma-separated dict_buffer_sizes values to sweep against "
            "--speed-config's own dataset key (e.g. '16,32,64,100'). Only meaningful "
            "for dataloader.type:\"iterative_dataloader\" configs (ShuffleIterableDataset's "
            "buffer_size) -- ignored (single no-op run) for \"dataloader\"-type configs "
            "(catsdogs), which have no buffer_size concept at all. Defaults to just "
            "--speed-config's own shipped value if omitted."
        ),
    )
    parser.addoption(
        "--speed-num-workers",
        action="store",
        default="",
        help=(
            "Comma-separated num_workers values to sweep against --speed-config "
            "(e.g. '0,1,4'). Defaults to just --speed-config's own shipped value if "
            "omitted, so the default cost is one run, not the full "
            "NUM_WORKERS_VALUES matrix -- pass this explicitly to also sweep it."
        ),
    )
