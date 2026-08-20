import os

import pytest
import torch
import torch.distributed as dist


def _slurm_launch_available():
    return all(k in os.environ for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))


@pytest.fixture(scope="session", autouse=True)
def dist_info():
    """Initializes torch.distributed from SLURM env vars for the whole test session.

    Mirrors training_scripts/train.py's `init_dist` (Slurm branch): reads
    SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID, sets the CUDA device, and inits an
    NCCL process group. Skips the entire tests/distributed/ suite -- with a clear
    reason -- if not actually launched under `srun` (see
    launch/tests/run_distributed_tests.sh), since these tests are meaningless as
    a single local process.

    Yields:
        Dict with "world_rank", "world_size", "local_rank" for the current process.
    """
    if not _slurm_launch_available():
        pytest.skip(
            "tests/distributed requires an srun launch (SLURM_PROCID/SLURM_NTASKS/"
            "SLURM_LOCALID not set) -- run via launch/tests/run_distributed_tests.sh"
        )

    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ.setdefault("MASTER_PORT", "29500")

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)

    yield {"world_rank": world_rank, "world_size": world_size, "local_rank": local_rank}

    dist.barrier()
    dist.destroy_process_group()
