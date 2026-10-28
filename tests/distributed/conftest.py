import os

import pytest
import torch
import torch.distributed as dist
import yaml


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


def _config_default(config, option, key):
    """Same idea as tests/dataloaders/conftest.py's identical helper --
    duplicated rather than imported (conftest.py files aren't meant to import
    each other across sibling directories) for
    test_dataloader_speed_real_pipeline.py's speed_buffer_size/
    speed_num_workers fixtures. --speed-config/etc. options themselves are
    registered once, in the shared tests/conftest.py (see its own
    pytest_addoption docstring for why they can't live in either directory's
    own conftest.py).
    """
    speed_config = config.getoption("--speed-config")
    if not speed_config:
        return None
    with open(speed_config) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    if key == "num_workers":
        return raw["dataloader"]["num_workers"]
    dict_buffer_sizes = raw["dataloader"].get("dict_buffer_sizes")
    if not dict_buffer_sizes:
        return None  # "dataloader"-type config (catsdogs) -- no buffer_size concept
    return next(iter(dict_buffer_sizes.values()))


def pytest_generate_tests(metafunc):
    """Same idea as tests/dataloaders/conftest.py's identical hook -- see its
    own docstring. Duplicated (not shared) since it only needs to apply to
    this directory's own test_dataloader_speed_real_pipeline.py.
    """
    if "speed_buffer_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--speed-buffer-sizes")
        if raw:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        else:
            default = _config_default(metafunc.config, "speed_buffer_size", "buffer_size")
            values = [default]
        metafunc.parametrize("speed_buffer_size", values)

    if "speed_num_workers" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--speed-num-workers")
        if raw:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        else:
            default = _config_default(metafunc.config, "speed_num_workers", "num_workers")
            values = [default]
        metafunc.parametrize("speed_num_workers", values)
