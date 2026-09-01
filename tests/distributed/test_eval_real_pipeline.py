"""Real (not fake-model) coverage for the two things val.py/test.py depend on
that had zero Tier 2 coverage before this: get_model's
`resume_from_checkpoint:True` load path (val.py/test.py force this
regardless of what the config says), and `eval_epoch`'s forward-only loop
running against a real model under real FSDP, rather than the fake model
tests/test_eval_epoch.py (Tier 1) uses.

Deliberately does NOT re-test dataloader/sharding correctness -- val.py/
test.py build their dataloader via `get_split_conf` feeding straight into the
exact same `calculate_load_balancing_on_the_fly`/`NativePytorchDataModule`
construction `test_dataloader_real_pipeline.py` already exercises with real,
unstubbed data; re-deriving narrowed real files here would only pay real I/O
cost for coverage that already exists. A fake, in-memory dataloader stands in
here instead, matching `tests/test_eval_epoch.py`'s own scope split (Tier 1
covers `eval_epoch`'s no_grad/loop-summing logic against a fake model; this
file's job is only the two real-integration gaps named above).

Deliberately scoped to the simplest baseline parallelism and a tiny synthetic
VIT classification model (same scope discipline as
test_pretrained_loading_real.py's own module docstring) -- get_model's
resume_from_checkpoint branch and eval_epoch's loop don't meaningfully depend
on data_par_size or model architecture in ways not already covered by the
existing dataloader/pretrained-loading Tier 2 tests, and a real 8-rank srun
launch is expensive, so this is kept to one test.

IMPORTANT: same collective-call-safety note as test_init_par_groups.py --
init_par_groups and FSDP's own collectives require every process in the job
to make the same calls in the same order, so nothing here may branch on this
rank's own rank id (only on WORLD_SIZE, identical on every rank).
"""

import os

import pytest
import torch
import torch.distributed as dist
import yaml

pytest.importorskip("xformers", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("timm", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("monai", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")

import argparse  # noqa: E402

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402

from UCF_VIT.model.utils import get_model  # noqa: E402
from UCF_VIT.parse import parse_config  # noqa: E402
from UCF_VIT.training import eval_epoch  # noqa: E402
from UCF_VIT.utils.misc import init_par_groups  # noqa: E402

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))
SCRATCH_ROOT = f"/tmp/{os.environ.get('SLURM_JOB_ID', 'local')}/test_eval_real_pipeline"

IMG_SIZE = [16, 16]
NUM_CLASSES = 2
BATCH_SIZE = 2
NUM_BATCHES_TO_CHECK = 2
SAVED_EPOCH = 3
SAVED_LOSS_LIST = [0.5, 0.4]


def _base_config(checkpoint_path):
    """A minimal, real, self-contained config dict (no real data files needed
    -- img_size/num_channels given explicitly, so detect_img_size/
    detect_num_channels never fire) -- same shape as
    test_pretrained_loading_real.py's own _base_config, trimmed to just the
    catsdogs/VIT classification combination this file needs.
    """
    return {
        "trainer": {
            "max_epochs": 1, "data_type": "float32", "gpu_type": "amd",
            "checkpoint_path": checkpoint_path, "resume_from_checkpoint": False,
            "checkpoint_filename": "epoch_0", "use_pretrained_model": False,
            "pretrained_checkpoint_filename": "", "save_frequency": 1,
        },
        "parallelism": {"fsdp_size": 1, "simple_ddp_size": WORLD_SIZE, "tensor_par_size": 1},
        "optimizer": {"type": "AdamW", "lr": 0.0001, "beta_1": 0.9, "beta_2": 0.95, "weight_decay": 1e-5},
        "scheduler": {"type": "linear-warmup-cosine-annealing", "warmup_epochs": 1, "warmup_start_lr": 1e-8, "eta_min": 1e-8},
        "grad_scaler": {"use_grad_scaler": False, "init_scale": 8192, "min_scale": 128, "growth_interval": 100},
        "model": {
            "type": "VIT", "embed_dim": 8, "depth": 1, "num_heads": 1, "mlp_ratio": 1.0,
            "drop_path": 0.0, "drop_rate": 0.0, "use_channel_aggregation": False, "num_classes": NUM_CLASSES,
        },
        "tiling": {"do_tiling": False, "div": 1, "tile_overlap": 0, "use_all_data": False},
        "ap": {"do_ap": False, "fixed_length": 196, "separate_channels": False, "use_adaptive_pos_emb": False, "interp_size": 16},
        "data": {
            "dataset": "catsdogs", "img_size": IMG_SIZE, "twoD": True, "patch_size": 4,
            "dict_root_dirs": {"catsdogs": "/nonexistent/never-read-since-get_model-and-the-fake-dataloader-below-never-touch-real-files"},
            "num_channels": {"catsdogs": 1}, "dict_in_variables": {"catsdogs": ["v0"]},
        },
        "dataloader": {"type": "dataloader", "batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": False},
        "dataset_options": {},
    }


class _FakeCatsDogsDataloader:
    """Always yields a fresh random (data, label, variables, dict_key) 4-tuple
    -- get_batch's VIT/do_ap:False branch unpacks exactly this shape.
    Real random data/labels (not fixed, unlike test_eval_epoch.py's Tier 1
    fixture) since this file's job is checking real integration runs
    end-to-end without erroring, not exact loss values.
    """

    def __iter__(self):
        return self

    def __next__(self):
        data = torch.randn(BATCH_SIZE, 1, IMG_SIZE[0], IMG_SIZE[1])
        label = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
        return data, label, ["v0"], "catsdogs"


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
def test_eval_real_pipeline_resume_from_checkpoint_and_forward_pass(dist_info):
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = torch.device(f"cuda:{local_rank}")

    checkpoint_dir = os.path.join(SCRATCH_ROOT, "ckpt")
    config_path = os.path.join(SCRATCH_ROOT, "config.yaml")

    if world_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(_base_config(checkpoint_dir), f)
    dist.barrier()

    args = argparse.Namespace(config=config_path, pretrained_config="")
    conf = parse_config(args)

    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(
        world_rank=world_rank, data_par_size=WORLD_SIZE, tensor_par_size=1, fsdp_size=1, simple_ddp_size=WORLD_SIZE,
    )

    # Build fresh (resume_from_checkpoint:False, get_model's own "train from
    # scratch" branch) to get a real, correctly-shaped model_state_dict to
    # round-trip through resume_from_checkpoint below, without needing an
    # actual prior training run.
    fresh_model, _, _ = get_model(conf, {}, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)

    if world_rank == 0:
        # Real save_checkpoint format/filename convention (epoch_{N}_rank_{R}
        # .ckpt, {"epoch","model_state_dict","optimizer_state_dict",
        # "scheduler_state_dict","loss_list"}) -- get_model's
        # resume_from_checkpoint branch only ever reads "model_state_dict"/
        # "loss_list"/"epoch" out of this, matching val.py/test.py's own real
        # usage (they never build an optimizer/scheduler at all).
        torch.save(
            {
                "epoch": SAVED_EPOCH,
                "model_state_dict": fresh_model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "loss_list": SAVED_LOSS_LIST,
            },
            os.path.join(checkpoint_dir, "epoch_0_rank_0.ckpt"),
        )
    dist.barrier()

    # Same override val.py/test.py apply unconditionally, regardless of what
    # the config says.
    resumed_conf = dict(conf)
    resumed_conf["trainer"] = dict(conf["trainer"], resume_from_checkpoint=True)

    model, epoch_start, loss_list = get_model(
        resumed_conf, {}, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group,
    )

    # The actual gap: get_model's resume_from_checkpoint:True branch had
    # never run under a real distributed launch before this test.
    assert epoch_start == SAVED_EPOCH + 1
    assert loss_list == SAVED_LOSS_LIST

    assert isinstance(model, FSDP)
    with FSDP.summon_full_params(model, writeback=False), FSDP.summon_full_params(fresh_model, writeback=False):
        for (name, resumed_param), (fresh_name, fresh_param) in zip(model.named_parameters(), fresh_model.named_parameters()):
            assert name == fresh_name
            torch.testing.assert_close(resumed_param, fresh_param)

    # eval_epoch, against the real resumed model -- the other gap.
    model.eval()
    epoch_loss, epoch_accuracy = eval_epoch(
        resumed_conf, model, _FakeCatsDogsDataloader(), epoch=epoch_start - 1,
        iterations_per_epoch=NUM_BATCHES_TO_CHECK, device=device,
        tensor_par_group=tensor_par_group, ddpm_scheduler=None,
    )

    assert torch.isfinite(epoch_loss)
    # Summed (not averaged) over NUM_BATCHES_TO_CHECK iterations, each a
    # per-batch accuracy in [0, 1] -- matching eval_epoch's own accumulation
    # semantics (see its docstring).
    assert 0.0 <= epoch_accuracy.item() <= NUM_BATCHES_TO_CHECK

    # No gradients should have accumulated -- eval_epoch wraps every
    # iteration in torch.no_grad(). tests/test_eval_epoch.py already checks
    # this mechanically (torch.is_grad_enabled() at the moment of a fake
    # model's forward call); this is the same property checked the other
    # way, against real FSDP-wrapped parameters, where a regression would
    # otherwise only surface as silently-wrong eval results (no optimizer
    # ever runs in val.py/test.py to consume/zero the gradients).
    with FSDP.summon_full_params(model, writeback=False):
        for p in model.parameters():
            assert p.grad is None
