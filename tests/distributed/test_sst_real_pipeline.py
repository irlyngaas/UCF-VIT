"""Real, full-pipeline dataloader check for "sst": memmap read -> chunk
slice -> tile -> collate, against real CFD data on Frontier, through the
exact production `NativePytorchDataModule` construction
`training_scripts/train.py` uses (including the new "sst"-specific
`dict_out_variables`/`img_size`/`full_domain_size` kwargs) -- not a
hand-assembled substitute.

Same idea as this directory's own `test_dataloader_real_pipeline.py`
(basic_ct/imagenet/catsdogs), kept as a separate file rather than added to
that one since "sst" is a genuinely different data format (memmap binary,
not NIfTI/JPEG) with its own config (`configs/sst/unetr/base_config.yaml`).
`configs/sst/unetr/base_config.yaml`'s own `parallelism.simple_ddp_size` is
`"auto"` (derives from real world_size at launch, so the same config works
at any node count) -- overridden to a fixed `8` here specifically, to match
this file's real 8-rank `srun` launch the same way `test_dataloader_real_
pipeline.py`'s basic_ct/imagenet configs already do with a fixed value, and
because `compute_narrow_dict_idx` reads `parallelism.simple_ddp_size`
straight off the raw (un-parsed) config, which can't itself resolve
`"auto"` the way `parse_config` does.

This is also the first real-data exercise of two things added without any
real-data test yet: the `+2` x-axis padding trim (confirmed with the user
to be trailing-only, not symmetric -- see `FileReader.read_process_file`'s
own comment) and `model.loss_fn:"MSE"` regression output shape/dtype
(float, not the usual int64 class-index label).
"""

import argparse
import itertools
import os
import sys

import pytest
import torch
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

SST_CONFIG = os.path.join(REPO_ROOT, "configs", "sst", "unetr", "base_config.yaml")
NUM_BATCHES_TO_CHECK = 2

# batch_size (2) * NUM_BATCHES_TO_CHECK * data_par_size (8, fixed below) --
# same formula test_dataloader_real_pipeline.py's basic_ct uses. For "sst",
# process_root_dirs already returns the *chunked* entry list (see its own
# docstring), so "files" here really means chunk-entries -- with this
# config's real 1*2*2=4 chunks per real timestamp, this many chunk-entries
# needs roughly a third that many real timestamps on disk.
SST_MIN_FILES = 2 * NUM_BATCHES_TO_CHECK * 8


def _narrowed_sst_config_path(world_rank):
    with open(SST_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # See this module's own docstring for why this override is needed.
    conf["parallelism"]["simple_ddp_size"] = 8

    narrow_min_files = inflate_min_files_for_train_split(conf, SST_MIN_FILES)
    try:
        narrow_end_idx = compute_narrow_dict_idx(conf, narrow_min_files)
    except NoRealDataFoundError as e:
        pytest.skip(str(e))

    conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
    conf["dataloader"]["dict_end_idx"] = narrow_end_idx

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/sst_real_pipeline"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"sst-{world_rank}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _build_sst_data_module(config_path):
    """Same construction as test_dataloader_real_pipeline.py's own
    `_build_data_module`, plus the "sst"-specific `dict_out_variables`/
    `img_size`/`full_domain_size` kwargs `training_scripts/train.py`'s real
    construction also passes for this dataset.
    """
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args)  # load_balance_offline=False: this rank's real world_size must match the config

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
        dict_out_variables=conf["data"]["dict_out_variables"],
        img_size=conf["data"]["img_size"],
        full_domain_size=conf["dataset_options"]["full_domain_size"],
    )
    data_module.setup()
    return conf, data_module


def _assert_finite(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} has non-finite values (NaN/Inf)"


def test_real_pipeline_sst_unetr(dist_info):
    """Real "sst": memmap decode, chunk-splitting, real regression label
    (continuous, not a discrete class index)."""
    config_path = _narrowed_sst_config_path(dist_info["world_rank"])
    conf, data_module = _build_sst_data_module(config_path)
    loader = data_module.train_dataloader()

    dkey = next(iter(conf["data"]["dict_root_dirs"]))
    batch_size = conf["dataloader"]["batch_size"]
    num_channels = conf["data"]["num_channels"][dkey]
    tile_size = conf["data"]["tile_size"]
    num_out_channels = len(conf["data"]["dict_out_variables"][dkey])

    batches = list(itertools.islice(loader, NUM_BATCHES_TO_CHECK))
    assert len(batches) == NUM_BATCHES_TO_CHECK

    for inp, label, variables, dict_key in batches:
        assert inp.shape == (batch_size, num_channels, *tile_size)
        assert label.shape == (batch_size, num_out_channels, *tile_size)

        assert inp.dtype == torch.float32
        assert label.dtype == torch.float32  # regression target, not a class-index label
        _assert_finite("inp", inp)
        _assert_finite("label", label)

        assert dict_key == dkey
        assert variables == conf["data"]["dict_in_variables"][dkey]
