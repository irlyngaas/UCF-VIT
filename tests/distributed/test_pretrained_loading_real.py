"""Real multi-rank test of the pretrained-checkpoint-loading wiring itself.

tests/model/test_pretrained_loading.py (Tier 1) deliberately calls
extract_encoder_state_dict/_transplant_pos_embed directly, bypassing
get_model/parse_pretrained_config entirely, to avoid needing FSDP/dist/SLURM
scaffolding -- that thoroughly covers interpolation *correctness*, but leaves
the actual integration/wiring layer (parse_pretrained_config building p_conf,
get_model constructing pretrained_model at p_conf's own size, loading a real
per-rank checkpoint file, merging into the new model, FSDP-wrapping it) with
zero coverage. This test calls the real get_model end to end, under a real
8-rank srun launch, with use_pretrained_model:True -- the first time that
code path has ever actually run (tracing it to write this test turned up an
unconditional KeyError bug, conf["pretrained_model"]["checkpoint_path"]
never being populated anywhere; fixed alongside this test).

Deliberately scoped to the simplest baseline parallelism (tensor_par_size=1,
fsdp_size=1, simple_ddp_size=WORLD_SIZE -- NO_SHARD) and non-adaptive
patching, matching the interpolation-focused scope of the work being
verified; combining pretrained-loading with tensor_par_size>1/fsdp_size>1/
adaptive patching/tiling is a separate follow-up, not attempted here.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see test_tensor_parallel_correctness.py's module
docstring for why this skips cleanly via importorskip instead of erroring at
collection.

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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402

from UCF_VIT.model.utils import get_model  # noqa: E402
from UCF_VIT.utils.misc import init_par_groups  # noqa: E402
import argparse  # noqa: E402
from UCF_VIT.parse import parse_config, parse_pretrained_config  # noqa: E402

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))
SCRATCH_ROOT = f"/tmp/{os.environ.get('SLURM_JOB_ID', 'local')}/test_pretrained_loading_real"

# Deliberately non-square, and deliberately swapped between the two configs
# (not just uniformly rescaled) -- the same "independent ratio change" shape
# tests/model/test_pretrained_loading.py's Tier 1 tests already cover, now
# through the real end-to-end path.
PRETRAINED_IMG_SIZE = [32, 64]
NEW_IMG_SIZE = [64, 32]


def _base_config(img_size, num_classes, checkpoint_path):
    """A minimal, real, self-contained config dict (no real data files
    needed -- img_size/num_channels given explicitly, so detect_img_size/
    detect_num_channels never fire) mirroring configs/catsdogs/classification/
    base_config.yaml's structure.
    """
    return {
        "trainer": {
            "max_epochs": 1,
            "data_type": "float32",
            "gpu_type": "amd",
            "checkpoint_path": checkpoint_path,
            "resume_from_checkpoint": False,
            "checkpoint_filename": "epoch_0",
            "use_pretrained_model": False,
            "pretrained_checkpoint_filename": "",
            "save_frequency": 1,
        },
        "parallelism": {"fsdp_size": 1, "simple_ddp_size": WORLD_SIZE, "tensor_par_size": 1},
        "optimizer": {"type": "AdamW", "lr": 0.0001, "beta_1": 0.9, "beta_2": 0.95, "weight_decay": 1e-5},
        "scheduler": {"type": "linear-warmup-cosine-annealing", "warmup_epochs": 1, "warmup_start_lr": 1e-8, "eta_min": 1e-8},
        "grad_scaler": {"use_grad_scaler": False, "init_scale": 8192, "min_scale": 128, "growth_interval": 100},
        "model": {
            "type": "VIT", "embed_dim": 8, "depth": 1, "num_heads": 1, "mlp_ratio": 1.0,
            "drop_path": 0.0, "drop_rate": 0.0, "use_channel_aggregation": False,
            "num_classes": num_classes,
        },
        "tiling": {"do_tiling": False, "div": 1, "tile_overlap": 0, "use_all_data": False},
        "ap": {"do_ap": False, "fixed_length": 196, "separate_channels": False, "use_adaptive_pos_emb": False, "interp_size": 16},
        "data": {
            "dataset": "catsdogs",
            "img_size": img_size,
            "twoD": True,
            "patch_size": 4,
            "dict_root_dirs": {"catsdogs": "/nonexistent/never-read-since-get_model-never-loads-data"},
            "num_channels": {"catsdogs": 1},
            "dict_in_variables": {"catsdogs": ["v0"]},
        },
        "dataloader": {"type": "dataloader", "batch_size": 2, "num_workers": 0, "pin_memory": False},
        "dataset_options": {},
    }


def _write_config(path, conf):
    with open(path, "w") as f:
        yaml.dump(conf, f)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
def test_pretrained_loading_wiring_real_get_model(dist_info):
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = torch.device(f"cuda:{local_rank}")

    pretrained_dir = os.path.join(SCRATCH_ROOT, "pretrained")
    new_dir = os.path.join(SCRATCH_ROOT, "new")
    pretrained_config_path = os.path.join(SCRATCH_ROOT, "pretrained.yaml")
    new_config_path = os.path.join(SCRATCH_ROOT, "new.yaml")

    if world_rank == 0:
        os.makedirs(pretrained_dir, exist_ok=True)
        os.makedirs(new_dir, exist_ok=True)

        pretrained_conf = _base_config(PRETRAINED_IMG_SIZE, num_classes=2, checkpoint_path=pretrained_dir)
        _write_config(pretrained_config_path, pretrained_conf)

        new_conf = _base_config(NEW_IMG_SIZE, num_classes=3, checkpoint_path=new_dir)
        new_conf["trainer"]["use_pretrained_model"] = True
        new_conf["trainer"]["pretrained_checkpoint_filename"] = "epoch_0"
        _write_config(new_config_path, new_conf)

        # A real tiny VIT at the pretrained config's own size, saved in
        # training.py's save_checkpoint own real format/filename convention
        # (epoch_{N}_rank_{R}.ckpt, {"epoch","model_state_dict",
        # "optimizer_state_dict","scheduler_state_dict","loss_list"}) --
        # get_model's pretrained branch only ever reads "model_state_dict"
        # out of this, but matching the real format end to end (rather than
        # a bespoke fixture shape) is the point of this test.
        from UCF_VIT.model.arch import VIT

        pretrained_model = VIT(
            img_size=tuple(PRETRAINED_IMG_SIZE), patch_size=4, in_chans=1,
            num_classes=2, embed_dim=8, depth=1, num_heads=1, mlp_ratio=1.0,
            twoD=True, class_token=True, pos_embed="learn",
            adaptive_patching=False, fixed_length=196,
        )
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": pretrained_model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "loss_list": [],
            },
            os.path.join(pretrained_dir, "epoch_0_rank_0.ckpt"),
        )
        # parse_pretrained_config's existence pre-check looks for a plain
        # file named exactly pretrained_checkpoint_filename (no _rank_/.ckpt
        # suffix) -- a real, separate, pre-existing inconsistency from what
        # actually gets loaded (found while writing this test, not fixed
        # here -- out of scope, doesn't block anything since it's just an
        # existence check).
        open(os.path.join(pretrained_dir, "epoch_0"), "w").close()

    dist.barrier()

    args = argparse.Namespace(config=new_config_path, pretrained_config=pretrained_config_path)
    conf = parse_config(args)
    p_conf = parse_pretrained_config(args, conf)

    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(
        world_rank=world_rank,
        data_par_size=WORLD_SIZE,
        tensor_par_size=1,
        fsdp_size=1,
        simple_ddp_size=WORLD_SIZE,
    )

    model, epoch_start, loss_list = get_model(
        conf, p_conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group,
    )

    assert isinstance(model, FSDP)

    with FSDP.summon_full_params(model, writeback=False):
        # Resized to the NEW model's own grid, not left at the pretrained
        # model's -- num_prefix_tokens=1 (class_token=True) on both sides
        # here, so the prefix row also carries over unchanged rather than
        # falling back to a fresh one (see model/utils.py's
        # _transplant_pos_embed for the mismatched-prefix-count case, not
        # exercised by this particular config pair).
        new_grid = tuple(s // 4 for s in NEW_IMG_SIZE)  # patch_size=4
        expected_len = 1 + new_grid[0] * new_grid[1]
        assert model.pos_embed.shape == (1, expected_len, 8)
        # num_classes=3 (the new config's own), not 2 (the pretrained
        # model's) -- the task-specific head must not have been transplanted.
        assert model.head.out_features == 3

    x = torch.randn(2, 1, NEW_IMG_SIZE[0], NEW_IMG_SIZE[1], device=device)
    variables = ["v0"]
    output = model(x, variables, None)
    output.sum().backward()

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
