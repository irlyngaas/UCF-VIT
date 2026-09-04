import os
import socket

import pytest
import torch.distributed as dist
import yaml


@pytest.fixture(scope="session", autouse=True)
def _ensure_single_process_distributed():
    """Guarantees a single-process `torch.distributed` group for every test in
    this directory, regardless of SLURM environment variables.

    `tests/conftest.py`'s own autouse fixture steps aside (does no init at
    all) whenever `SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID` are all set,
    on the assumption that means real multi-rank `tests/distributed/` tests
    are running, which do their own real init instead. But
    `launch/tests/run_dataloader_speed.sh` submits via a real `sbatch` job
    too, just without `srun` -- and a real Frontier run (job 5421658) showed
    this specific cluster setup populates those same `SLURM_*` variables for
    a bare `sbatch` script even with no `srun` involved at all, so that
    fixture wrongly stepped aside here too, leaving `torch.distributed` never
    initialized -- `NativePytorchDataModule._my_dataset_key()`'s own
    `if not torch.distributed.is_initialized()` guard then raised
    `NotImplementedError` before any real decode/timing work ever ran.

    None of `tests/dataloaders/`'s own tests are ever real multi-rank (see
    `test_dataset_speed_real_data.py`'s own module docstring -- deliberately
    single-process, even under a real Frontier job) and this file only
    applies within `tests/dataloaders/` and its subdirectories (pytest scopes
    conftest.py fixtures to the directory tree they're defined in), so this
    can't affect `tests/distributed/`'s own real multi-rank init at all --
    safe to just always ensure single-process init here, independent of the
    SLURM-launch heuristic that exists for the parent conftest's own,
    different reasons.
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


def pytest_addoption(parser):
    """CLI options for test_dataset_speed_real_data.py's generic, config-driven test.

    Kept in this directory's own conftest.py (not the top-level tests/conftest.py)
    since they're only meaningful for tests/dataloaders/'s dataloader_speed-marked
    tests. See test_real_decode_throughput_config's own docstring for usage.
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


def _config_default(config, option, key):
    """Reads a single int default out of --speed-config's own raw YAML, for
    whichever of --speed-buffer-sizes/--speed-num-workers wasn't given
    explicitly. Returns None if --speed-config itself wasn't given either --
    the test function's own skip (no --speed-config) handles that case, this
    just needs to not blow up at collection time.
    """
    speed_config = config.getoption("--speed-config")
    if not speed_config:
        return None
    with open(speed_config) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    if key == "num_workers":
        return raw["dataloader"]["num_workers"]
    # buffer_size: dict_buffer_sizes is keyed per dataset-dict-key -- take the
    # first (real shipped configs here only ever have one real key).
    dict_buffer_sizes = raw["dataloader"].get("dict_buffer_sizes")
    if not dict_buffer_sizes:
        return None  # "dataloader"-type config (catsdogs) -- no buffer_size concept
    return next(iter(dict_buffer_sizes.values()))


def pytest_generate_tests(metafunc):
    """Dynamically parametrizes test_real_decode_throughput_config's
    speed_buffer_size/speed_num_workers fixtures from --speed-buffer-sizes/
    --speed-num-workers, since pytest.mark.parametrize can't read a CLI
    option (or --speed-config's own YAML) at collection time on its own.
    """
    if "speed_buffer_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--speed-buffer-sizes")
        if raw:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        else:
            default = _config_default(metafunc.config, "speed_buffer_size", "buffer_size")
            values = [default]  # None if --speed-config omitted or catsdogs-style -- test skips/no-ops on it
        metafunc.parametrize("speed_buffer_size", values)

    if "speed_num_workers" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--speed-num-workers")
        if raw:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        else:
            default = _config_default(metafunc.config, "speed_num_workers", "num_workers")
            values = [default]  # None if --speed-config omitted -- test skips on it
        metafunc.parametrize("speed_num_workers", values)
