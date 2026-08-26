"""Tests that UCF_VIT.parse.parse_pretrained_config works for every pretrained
source model type, not just VIT/MAE.

get_kwargs (parse.py) has a distinct branch per model type (VIT/SAP/MAE/
UNETR/DiffusionVIT), and parse_pretrained_config calling get_kwargs(model_
type, pretrained_conf) for that branch had -- before this session's work --
never been exercised for SAP/UNETR/DiffusionVIT as the pretrained source at
all (only VIT/MAE, both at Tier 1 in tests/model/test_pretrained_loading.py
and Tier 2 in tests/distributed/test_pretrained_loading_real.py). The real
workflow is "pretrain via MAE, fine-tune into any downstream type," but
nothing in get_model/parse_pretrained_config actually restricts which type
the pretrained source itself is -- just because no shipped config uses SAP/
UNETR/DiffusionVIT as a pretrained source doesn't mean it isn't a real,
reachable code path.

Pure parse.py-level tests -- no timm/monai/xformers needed (parse.py itself
never imports UCF_VIT.model.arch/building_blocks), so these run in any
environment, unlike tests/model/test_pretrained_loading.py.
"""

import argparse
import os
import tempfile

import yaml

from UCF_VIT.parse import parse_config, parse_pretrained_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _base_config(model_type, do_ap, checkpoint_path, extra_model=None, extra_ap=None):
    """A minimal, real, self-contained config dict (no real data files
    needed -- img_size/num_channels given explicitly) covering every field
    get_kwargs/parse_config actually reads, for any of the 5 model types.
    """
    conf = {
        "trainer": {
            "max_epochs": 1, "data_type": "float32", "gpu_type": "amd",
            "checkpoint_path": checkpoint_path, "resume_from_checkpoint": False,
            "checkpoint_filename": "epoch_0", "use_pretrained_model": False,
            "pretrained_checkpoint_filename": "", "save_frequency": 1,
        },
        "parallelism": {"fsdp_size": 1, "simple_ddp_size": 8, "tensor_par_size": 1},
        "optimizer": {"type": "AdamW", "lr": 0.0001, "beta_1": 0.9, "beta_2": 0.95, "weight_decay": 1e-5},
        "scheduler": {"type": "linear-warmup-cosine-annealing", "warmup_epochs": 1, "warmup_start_lr": 1e-8, "eta_min": 1e-8},
        "grad_scaler": {"use_grad_scaler": False, "init_scale": 8192, "min_scale": 128, "growth_interval": 100},
        "model": {
            "type": model_type, "embed_dim": 8, "depth": 1, "num_heads": 1, "mlp_ratio": 1.0,
            "drop_path": 0.0, "drop_rate": 0.0, "use_channel_aggregation": False,
        },
        "tiling": {"do_tiling": False, "div": 1, "tile_overlap": 0, "use_all_data": False},
        "ap": {
            "do_ap": do_ap, "fixed_length": 4, "separate_channels": False,
            "use_adaptive_pos_emb": False, "interp_size": 4,
        },
        "data": {
            "dataset": "catsdogs", "img_size": [16, 16], "twoD": True, "patch_size": 4,
            "dict_root_dirs": {"catsdogs": "/nonexistent"},
            "num_channels": {"catsdogs": 1},
            "dict_in_variables": {"catsdogs": ["v0"]},
        },
        "dataloader": {"type": "dataloader", "batch_size": 2, "num_workers": 0, "pin_memory": False},
        "dataset_options": {},
    }
    if extra_model:
        conf["model"].update(extra_model)
    if extra_ap:
        conf["ap"].update(extra_ap)
    return conf


def _write(path, conf):
    with open(path, "w") as f:
        yaml.dump(conf, f)


def _parse_as_pretrained_source(pretrained_conf, downstream_conf):
    """Writes both configs to temp files, then runs the real parse_config +
    parse_pretrained_config against them (downstream config's own
    use_pretrained_model:True, pointed at the pretrained config's file).

    Returns:
        p_conf, as returned by parse_pretrained_config.
    """
    scratch = tempfile.mkdtemp()
    pretrained_path = os.path.join(scratch, "pretrained.yaml")
    downstream_path = os.path.join(scratch, "downstream.yaml")

    pretrained_conf["trainer"]["checkpoint_path"] = os.path.join(scratch, "pretrained_ckpt")
    os.makedirs(pretrained_conf["trainer"]["checkpoint_path"])
    # parse_pretrained_config's checkpoint-existence pre-check needs
    # "<pretrained_checkpoint_filename>_rank_0.ckpt" to exist (the real
    # filename save_checkpoint writes) -- content is never read here, this
    # test only exercises get_kwargs, not an actual checkpoint load.
    open(os.path.join(pretrained_conf["trainer"]["checkpoint_path"], "epoch_0_rank_0.ckpt"), "w").close()

    downstream_conf["trainer"]["checkpoint_path"] = os.path.join(scratch, "downstream_ckpt")
    downstream_conf["trainer"]["use_pretrained_model"] = True
    downstream_conf["trainer"]["pretrained_checkpoint_filename"] = "epoch_0"

    _write(pretrained_path, pretrained_conf)
    _write(downstream_path, downstream_conf)

    args = argparse.Namespace(config=downstream_path, pretrained_config=pretrained_path)
    conf = parse_config(args, load_balance_offline=True)
    return parse_pretrained_config(args, conf)


def test_parse_pretrained_config_sap_source():
    # SAP requires do_ap:True (get_kwargs's own assert); the downstream
    # config must match (parse_pretrained_config's own do_ap-equality
    # assert).
    pretrained_conf = _base_config(
        "SAP", do_ap=True, checkpoint_path="",
        extra_model={"num_classes": 2},
    )
    downstream_conf = _base_config("VIT", do_ap=True, checkpoint_path="", extra_model={"num_classes": 3})

    p_conf = _parse_as_pretrained_source(pretrained_conf, downstream_conf)

    assert p_conf["model_type"] == "SAP"
    assert p_conf["kwargs"]["sqrt_len_method"] is True
    # fixed_length=4 (2D quadtree, twoD:True) -> sqrt_len=2.
    assert p_conf["kwargs"]["sqrt_len"] == 2


def test_parse_pretrained_config_unetr_source():
    pretrained_conf = _base_config(
        "UNETR", do_ap=False, checkpoint_path="",
        extra_model={"num_classes": 2, "linear_decoder": True, "skip_connection": False, "feature_size": 4},
    )
    downstream_conf = _base_config("VIT", do_ap=False, checkpoint_path="", extra_model={"num_classes": 3})

    p_conf = _parse_as_pretrained_source(pretrained_conf, downstream_conf)

    assert p_conf["model_type"] == "UNETR"
    assert p_conf["kwargs"]["feature_size"] == 4
    assert p_conf["kwargs"]["linear_decoder"] is True
    # do_ap:False on the pretrained source -> sqrt_len_method must be False
    # (only True when UNETR itself has do_ap:True; see get_kwargs's own
    # UNETR branch).
    assert p_conf["kwargs"]["sqrt_len_method"] is False


def test_parse_pretrained_config_diffusionvit_source():
    # DiffusionVIT requires do_ap:False (get_kwargs's own assert).
    pretrained_conf = _base_config(
        "DiffusionVIT", do_ap=False, checkpoint_path="",
        extra_model={"num_time_steps": 100, "linear_decoder": True},
    )
    downstream_conf = _base_config("VIT", do_ap=False, checkpoint_path="", extra_model={"num_classes": 3})

    p_conf = _parse_as_pretrained_source(pretrained_conf, downstream_conf)

    assert p_conf["model_type"] == "DiffusionVIT"
    assert p_conf["kwargs"]["time_steps"] == 100
    assert p_conf["kwargs"]["linear_decoder"] is True
