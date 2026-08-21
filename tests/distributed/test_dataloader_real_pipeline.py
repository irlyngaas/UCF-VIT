"""Real, full-pipeline dataloader check: decode -> tile -> patch -> collate,
against real basic_ct and imagenet data on Frontier, through the exact
production `NativePytorchDataModule` construction `training_scripts/train.py`
uses -- not a hand-assembled substitute.

The other two real-data files in this directory deliberately stub out file
I/O (`test_dataloader_real_data.py`) so they can focus on sharding
correctness at low cost. This file is the complement: no stubbing anywhere,
real NIfTI/JPEG decode, real tiling, real (for basic_ct) adaptive patching,
real collation -- checking that what comes out the other end is actually
correct, not just that sharding is. This was the biggest gap in dataloader
test coverage as of this session: Tier 1 exercises the pipeline's logic with
synthetic arrays, and the other Tier 2 files exercise sharding with decode
stubbed out, but nothing previously decoded a real file and checked the
resulting batch.

Three configs, chosen to cover meaningfully different branches:
  - basic_ct/unetr: adaptive_patching=True, twoD=False (full 3D tiling, no
    z-slice explosion -- see tests/README.md's "Real runs on Frontier so
    far" for why twoD=True configs need much smaller min_files), and real
    segmentation labels (the other real gap flagged this session --
    test_dataloader_real_data.py only ever globs imagesTr, never touches
    labelsTr).
  - imagenet/classification: adaptive_patching=False, real classification
    labels, real JPEG decode + resize.
  - catsdogs/classification: the "dataloader" type -- a completely different
    production code path from the other two (dataloader.type ==
    "iterative_dataloader"). tests/distributed/test_catsdogs_real_data.py
    already does real, unstubbed catsdogs decode, but with hand-picked
    tile_size/patch_size/fixed_length rather than the real shipped config's
    values, and without going through parse_config at all. This one is built
    the same faithful way as the two above: real parse_config output, real
    narrowing mechanism, and the exact construction train.py itself makes.

basic_ct/imagenet's parallelism settings (data_par_size=8, tensor_par_size=1)
exactly match this file's real 8-rank srun launch, so -- unlike Tier 3's
smoke test -- no parallelism overrides are needed for them, only data
narrowing (reusing tests/integration/run_training_smoke.py's
compute_narrow_dict_idx, the same real-file-count-aware narrowing Tier 3
uses, rather than a third reimplementation). catsdogs/classification's real
config already matches too (also data_par_size=8).

For dataloader.type == "iterative_dataloader" (basic_ct/imagenet), the data
module is built directly via UCF_VIT.parse.parse_config +
UCF_VIT.utils.misc.calculate_load_balancing_on_the_fly +
UCF_VIT.dataloaders.datamodule.NativePytorchDataModule -- the same three
calls training_scripts/train.py itself makes for that dataloader type,
skipping only the model/optimizer/training-loop parts this file has no need
for. For dataloader.type == "dataloader" (catsdogs), train.py never calls
calculate_load_balancing_on_the_fly or NativePytorchDataModule at all -- it
builds a plain CatsDogsDataset + DistributedSampler + DataLoader directly,
so this file does exactly that instead, reusing Tier 3's
create_narrow_catsdogs_dir for narrowing.
"""

import argparse
import glob
import itertools
import os
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "integration"))

from run_training_smoke import (  # noqa: E402
    NoRealDataFoundError,
    compute_narrow_dict_idx,
    create_narrow_catsdogs_dir,
)

from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule  # noqa: E402
from UCF_VIT.parse import parse_config  # noqa: E402
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly  # noqa: E402

BASIC_CT_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml")
IMAGENET_CONFIG = os.path.join(REPO_ROOT, "configs", "imagenet", "classification", "base_config.yaml")
CATSDOGS_CONFIG = os.path.join(REPO_ROOT, "configs", "catsdogs", "classification", "base_config.yaml")
NUM_BATCHES_TO_CHECK = 2

# compute_narrow_dict_idx's min_files means different things for the two
# datasets: for basic_ct (a single dict_root_dirs key), it's a total file
# count that FileReader's own DDP-rank sharding then divides across ranks,
# and tiling multiplies each real file into div**3 = 64 samples (unetr's
# real div=4) -- so even 1 file/rank is comfortably enough for
# NUM_BATCHES_TO_CHECK batches at batch_size=2. For imagenet,
# process_root_dirs already buckets real files one bucket per rank before
# min_files narrows *within* each bucket, and there's no tiling
# multiplication (div=1 for classification) -- so min_files here must
# directly cover batch_size * NUM_BATCHES_TO_CHECK samples per rank, not
# just per dataset.
BASIC_CT_MIN_FILES = 16
IMAGENET_MIN_FILES = 100  # >= 32 (real batch_size) * 2 (NUM_BATCHES_TO_CHECK), with margin

# create_narrow_catsdogs_dir's min_files is a floor under max(min_files,
# batch_size * data_par_size) -- real catsdogs/classification's batch_size=32
# * data_par_size=8 = 256 already dominates any min_files smaller than that,
# giving 256 / 8 = 32 files/rank = exactly 1 batch. Set explicitly to what's
# actually needed (batch_size * NUM_BATCHES_TO_CHECK * data_par_size) so
# NUM_BATCHES_TO_CHECK batches/rank are available regardless of how the
# batch_size * data_par_size floor happens to compare.
CATSDOGS_MIN_FILES = 32 * NUM_BATCHES_TO_CHECK * 8


def _narrowed_config_path(base_config_path, min_files, tag):
    """Writes a copy of `base_config_path` with dict_start_idx/dict_end_idx
    narrowed to `min_files` real files -- same mechanism (and same function)
    Tier 3's smoke test uses. Per-rank scratch path (via `tag`, which
    includes the world rank) so concurrent ranks never race writing/reading
    the same file, even though every rank computes an identical result.
    """
    with open(base_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    try:
        narrow_end_idx = compute_narrow_dict_idx(conf, min_files)
    except NoRealDataFoundError as e:
        pytest.skip(str(e))

    conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
    conf["dataloader"]["dict_end_idx"] = narrow_end_idx

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/dataloader_real_pipeline"
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, f"{tag}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _narrowed_catsdogs_config_path(world_rank):
    """Same idea as `_narrowed_config_path`, but for catsdogs -- symlink-based
    narrowing (create_narrow_catsdogs_dir) instead of dict_start_idx/
    dict_end_idx fractions, since dataloader.type == "dataloader" has no
    index-based trimming knob at all.
    """
    with open(CATSDOGS_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_dir = f"/tmp/{job_id}/dataloader_real_pipeline/catsdogs-{world_rank}"
    try:
        dkey, narrowed_dir = create_narrow_catsdogs_dir(conf, scratch_dir, CATSDOGS_MIN_FILES)
    except NoRealDataFoundError as e:
        pytest.skip(str(e))
    conf["data"]["dict_root_dirs"][dkey] = narrowed_dir

    out_path = os.path.join(scratch_dir, "catsdogs.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def _build_data_module(config_path):
    """Builds a real NativePytorchDataModule the same way train.py does for
    dataloader.type == "iterative_dataloader" -- parse_config +
    calculate_load_balancing_on_the_fly + the constructor call itself,
    skipping only the model/optimizer/training-loop parts.

    Returns:
        (conf, data_module), with data_module already .setup().
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
        patch_size=conf["data"]["patch_size"],
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


def _assert_finite(name, tensor):
    assert torch.isfinite(tensor).all(), f"{name} has non-finite values (NaN/Inf)"


def test_real_pipeline_basic_ct_unetr(dist_info):
    """Real basic_ct: NIfTI decode, full 3D tiling, adaptive patching, and
    (the other real-data gap this closes) real segmentation labels -- one
    of the two never previously exercised by any test with real content.
    """
    config_path = _narrowed_config_path(BASIC_CT_CONFIG, BASIC_CT_MIN_FILES, f"basic_ct-{dist_info['world_rank']}")
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()

    batch_size = conf["dataloader"]["batch_size"]
    num_channels = conf["data"]["num_channels"]["ct1"]
    tile_size = conf["data"]["tile_size"]
    fixed_length = conf["ap"]["fixed_length"]
    patch_size = conf["data"]["patch_size"]
    num_classes = conf["model"]["kwargs"]["num_classes"]

    batches = list(itertools.islice(loader, NUM_BATCHES_TO_CHECK))
    assert len(batches) == NUM_BATCHES_TO_CHECK

    for inp, seq, size, pos, label, seq_label, variables, dict_key in batches:
        assert inp.shape == (batch_size, num_channels, *tile_size)
        assert seq.shape == (batch_size, 1, fixed_length, patch_size ** 3)
        assert size.shape == (batch_size, 1, fixed_length)
        assert pos.shape == (batch_size, 1, fixed_length, 3)
        assert label.shape == (batch_size, 1, *tile_size)
        assert seq_label.shape == (batch_size, num_classes, patch_size ** 3, fixed_length)

        _assert_finite("inp", inp)
        # basic_ct's FileReader min-max-normalizes each volume to [0, 1]
        assert inp.min() >= 0.0 and inp.max() <= 1.0

        assert label.dtype == torch.uint8
        assert label.min() >= 0 and label.max() < num_classes  # real labels, shifted to [0, num_classes)

        # seq_label is one-hot over the class axis (dim=1)
        torch.testing.assert_close(
            seq_label.sum(dim=1), torch.ones(batch_size, patch_size ** 3, fixed_length, dtype=seq_label.dtype)
        )

        assert dict_key == "ct1"
        assert variables == conf["data"]["dict_in_variables"]["ct1"]


def test_real_pipeline_imagenet_classification(dist_info):
    """Real imagenet: JPEG decode + resize, non-adaptive-patching, real
    classification labels.
    """
    config_path = _narrowed_config_path(IMAGENET_CONFIG, IMAGENET_MIN_FILES, f"imagenet-{dist_info['world_rank']}")
    conf, data_module = _build_data_module(config_path)
    loader = data_module.train_dataloader()

    batch_size = conf["dataloader"]["batch_size"]
    num_channels = conf["data"]["num_channels"]["imagenet"]
    tile_size = conf["data"]["tile_size"]
    num_classes = conf["model"]["kwargs"]["num_classes"]

    batches = list(itertools.islice(loader, NUM_BATCHES_TO_CHECK))
    assert len(batches) == NUM_BATCHES_TO_CHECK

    for inp, label, variables, dict_key in batches:
        assert inp.shape == (batch_size, num_channels, *tile_size)
        assert label.shape == (batch_size,)

        _assert_finite("inp", inp)
        assert inp.min() >= 0.0  # real (resized, uint8-sourced) pixel data

        assert label.min() >= 0 and label.max() < num_classes

        # imagenet buckets by numeric index (process_root_dirs splits it into
        # data_par_size buckets), not by the "imagenet" dict_root_dirs key
        # itself -- unlike basic_ct, whose single key ("ct1") is the dict_key.
        assert isinstance(dict_key, int) and 0 <= dict_key < conf["parallelism"]["data_par_size"]
        assert variables == conf["data"]["dict_in_variables"]["imagenet"]


def _build_catsdogs_loader(config_path, world_rank):
    """Builds a real DataLoader the same way train.py does for
    dataloader.type == "dataloader" -- there's no NativePytorchDataModule or
    calculate_load_balancing_on_the_fly call for this type in production, so
    this doesn't make one either; just CatsDogsDataset + DistributedSampler +
    DataLoader, straight from parse_config's real output.

    Returns:
        (conf, loader).
    """
    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args)

    dkey_train = list(conf["data"]["dict_root_dirs"])[0]
    train_list = glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey_train], "*.jpg"))
    train_data = conf["dataloader"]["dataset_module"](
        train_list, conf["data"]["dict_in_variables"][dkey_train], conf["data"]["tile_size"],
        adaptive_patching=conf["ap"]["do_ap"], fixed_length=conf["ap"]["fixed_length"],
        patch_size=conf["data"]["patch_size"], num_channels=conf["data"]["num_channels"][dkey_train],
        dataset=conf["data"]["dataset"],
    )
    train_sampler = DistributedSampler(
        train_data, shuffle=True, num_replicas=conf["parallelism"]["data_par_size"], rank=world_rank,
    )
    loader = DataLoader(
        dataset=train_data, sampler=train_sampler, num_workers=conf["dataloader"]["num_workers"],
        pin_memory=conf["dataloader"]["pin_memory"], batch_size=conf["dataloader"]["batch_size"], drop_last=True,
        collate_fn=lambda batch: conf["dataloader"]["collate_fn"](
            batch, adaptive_patching=conf["ap"]["do_ap"], return_label=conf["dataloader"]["return_label"],
        ),
    )
    return conf, loader


def test_real_pipeline_catsdogs_classification(dist_info):
    """Real catsdogs: JPEG decode + resize, non-adaptive-patching (the real
    shipped config's setting), real classification labels -- the
    dataloader.type == "dataloader" code path, completely separate from
    basic_ct/imagenet's "iterative_dataloader" path above. Regression
    coverage, as a side effect of using real parse_config output, for the
    num_channels dict-vs-int train.py wiring bug fixed earlier this session
    (see tests/datasets/test_catsdogs.py's module docstring) -- a wrong
    argument there would surface here as a real crash, not a synthetic one.
    """
    config_path = _narrowed_catsdogs_config_path(dist_info["world_rank"])
    conf, loader = _build_catsdogs_loader(config_path, dist_info["world_rank"])

    batch_size = conf["dataloader"]["batch_size"]
    num_channels = conf["data"]["num_channels"]["catsdogs"]
    tile_size = conf["data"]["tile_size"]

    batches = list(itertools.islice(loader, NUM_BATCHES_TO_CHECK))
    assert len(batches) == NUM_BATCHES_TO_CHECK

    for inp, label, variables, dict_key in batches:
        assert inp.shape == (batch_size, num_channels, *tile_size)
        assert label.shape == (batch_size,)

        _assert_finite("inp", inp.float())
        assert inp.min() >= 0 and inp.max() <= 255  # real uint8-sourced pixel data

        assert set(label.tolist()) <= {0, 1}  # cat=0, dog=1

        assert dict_key == "catsdogs"
        assert variables == conf["data"]["dict_in_variables"]["catsdogs"]
