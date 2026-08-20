import os
import socket

import pytest
import torch.distributed as dist


@pytest.fixture(scope="session", autouse=True)
def _single_process_distributed():
    """Initializes a single-process (world_size=1) torch.distributed group for the whole test session.

    Several UCF_VIT functions (e.g. `parse_config`) call `dist.get_rank()` even
    outside of an actual multi-process training launch. This fixture lets those
    calls succeed locally, without requiring a real multi-GPU/SLURM allocation.
    """
    if dist.is_available() and not dist.is_initialized():
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
