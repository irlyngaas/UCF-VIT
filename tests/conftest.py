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
