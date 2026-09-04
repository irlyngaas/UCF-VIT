"""Runs every shipped example config through UCF_VIT.parse.parse_config.

This is a regression net against the parsers in src/UCF_VIT/parse.py: since
those functions mostly aren't unit-testable in isolation (they read entire
config files and cross-check many fields against each other), the cheapest
real coverage is to make sure every config we actually ship still parses.
"""

import argparse
import glob
import os
import tempfile

import pytest
import yaml

from validate_config import validate_config

from UCF_VIT.parse import parse_config
from UCF_VIT.utils.misc import find_repo_root

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "**", "*.yaml"), recursive=True))
SAP_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "sap", "base_config.yaml")
UNETR_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml")


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_shipped_config_parses(config_path):
    validate_config(config_path)


def test_resume_and_pretrained_both_true_raises_clearly():
    """Regression test: resume_from_checkpoint:True and use_pretrained_model:True
    together used to silently drop use_pretrained_model with no warning at all
    (trainer_conf's own "use_pretrained_model" ternary, and parse_pretrained_config's
    identical silent override) -- now rejected explicitly instead, since the two are
    mutually exclusive (resume continues an existing run's own checkpoint;
    use_pretrained_model starts a new run from a different model's weights)."""
    with open(SAP_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["trainer"]["resume_from_checkpoint"] = True
    conf["trainer"]["use_pretrained_model"] = True

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        with pytest.raises(SystemExit, match="resume_from_checkpoint and trainer.use_pretrained_model cannot both be True"):
            parse_config(args, load_balance_offline=True)
    finally:
        os.remove(path)


def test_missing_interp_size_under_do_ap_raises_clearly():
    """Regression test for interp_size's requiredness: do_ap:True with no
    ap.interp_size set must fail with a clear, actionable error (not a bare
    KeyError), the same way other required-when-conditional fields do
    elsewhere in parse.py (e.g. feature_size for UNETR).
    """
    with open(SAP_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    del conf["ap"]["interp_size"]

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        with pytest.raises(SystemExit, match="interp_size"):
            parse_config(args, load_balance_offline=True)
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# parallelism.simple_ddp_size: "auto" -- derives simple_ddp_size from world_size
# (fsdp_size/tensor_par_size fixed by the model, simple_ddp_size fills the
# rest) instead of needing hand-editing every time a run's node count changes.
# ---------------------------------------------------------------------------


def _config_with_parallelism(fsdp_size, simple_ddp_size, tensor_par_size):
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["parallelism"] = {
        "fsdp_size": fsdp_size,
        "simple_ddp_size": simple_ddp_size,
        "tensor_par_size": tensor_par_size,
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    with open(path, "w") as f:
        yaml.dump(conf, f)
    return path


def test_simple_ddp_size_auto_derives_from_world_size(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 16)
    path = _config_with_parallelism(fsdp_size=2, simple_ddp_size="auto", tensor_par_size=1)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args)
        assert parsed["parallelism"]["simple_ddp_size"] == 8  # 16 / (2*1)
        assert parsed["parallelism"]["data_par_size"] == 16  # fsdp_size * simple_ddp_size
    finally:
        os.remove(path)


def test_simple_ddp_size_auto_accounts_for_tensor_par_size(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 16)
    path = _config_with_parallelism(fsdp_size=1, simple_ddp_size="auto", tensor_par_size=2)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args)
        assert parsed["parallelism"]["simple_ddp_size"] == 8  # 16 / (1*2)
    finally:
        os.remove(path)


def test_simple_ddp_size_auto_raises_when_not_evenly_divisible(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 17)  # prime -- not divisible by 2
    path = _config_with_parallelism(fsdp_size=2, simple_ddp_size="auto", tensor_par_size=1)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        with pytest.raises(AssertionError, match="evenly divisible"):
            parse_config(args)
    finally:
        os.remove(path)


def test_simple_ddp_size_auto_raises_with_no_live_process_group(monkeypatch):
    # utils/load_balance.py's offline precompute has no live process group to
    # read world_size from -- "auto" must fail clearly there, not silently
    # compute against the wrong (or no) world_size. Gated on
    # dist.is_initialized() itself, not load_balance_offline -- validate_config.py
    # also passes load_balance_offline=True but does have a live (single-process)
    # group by then, so that flag alone can't be the signal.
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    path = _config_with_parallelism(fsdp_size=1, simple_ddp_size="auto", tensor_par_size=1)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        with pytest.raises(AssertionError, match="process group.*none is initialized"):
            parse_config(args, load_balance_offline=True)
    finally:
        os.remove(path)


def test_simple_ddp_size_auto_works_under_load_balance_offline_with_live_group(monkeypatch):
    # The validate_config.py case: load_balance_offline=True *and* a live
    # (single-process) group already initialized -- "auto" must resolve fine
    # here (against that group's own world_size), matching every shipped
    # unetr_token_selection_experiment config now using "auto" with
    # test_shipped_config_parses's load_balance_offline=True validation.
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    path = _config_with_parallelism(fsdp_size=1, simple_ddp_size="auto", tensor_par_size=1)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["parallelism"]["simple_ddp_size"] == 1
    finally:
        os.remove(path)


def test_simple_ddp_size_explicit_int_unaffected_by_auto_support(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 4)
    path = _config_with_parallelism(fsdp_size=1, simple_ddp_size=4, tensor_par_size=1)
    try:
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args)
        assert parsed["parallelism"]["simple_ddp_size"] == 4
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# UNETR's do_ap token_selection / area_weighted_alpha -- get_kwargs' own
# plumbing, not covered by test_arch.py's model-level tests (which construct
# UNETR directly, bypassing parse.py entirely)
# ---------------------------------------------------------------------------


def test_token_selection_defaults_to_point_under_do_ap():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["ap"]["do_ap"] = True
    conf["ap"]["interp_size"] = 32  # matches this config's real do_ap:True feature-matrix cell

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["model"]["kwargs"]["token_selection"] == "point"
    finally:
        os.remove(path)


def test_token_selection_smallest_overlap_and_area_weighted_thread_through():
    for value in ("smallest_overlap", "area_weighted", "cross_attention"):
        with open(UNETR_CONFIG) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        conf["ap"]["do_ap"] = True
        conf["ap"]["interp_size"] = 32
        conf["model"]["token_selection"] = value

        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        try:
            with open(path, "w") as f:
                yaml.dump(conf, f)
            args = argparse.Namespace(config=path, pretrained_config="")
            parsed = parse_config(args, load_balance_offline=True)
            assert parsed["model"]["kwargs"]["token_selection"] == value
        finally:
            os.remove(path)


def test_token_selection_invalid_value_raises_clearly():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["ap"]["do_ap"] = True
    conf["ap"]["interp_size"] = 32
    conf["model"]["token_selection"] = "bogus"

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        with pytest.raises(AssertionError, match="Unknown token_selection"):
            parse_config(args, load_balance_offline=True)
    finally:
        os.remove(path)


def test_area_weighted_alpha_defaults_to_zero():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["ap"]["do_ap"] = True
    conf["ap"]["interp_size"] = 32

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["model"]["kwargs"]["area_weighted_alpha"] == 0.0
    finally:
        os.remove(path)


def test_area_weighted_alpha_threads_through():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["ap"]["do_ap"] = True
    conf["ap"]["interp_size"] = 32
    conf["model"]["token_selection"] = "area_weighted"
    conf["model"]["area_weighted_alpha"] = 2.5

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["model"]["kwargs"]["area_weighted_alpha"] == 2.5
    finally:
        os.remove(path)


def test_token_capacity_warning_fires_for_point_but_not_area_weighted(capsys):
    """fixed_length:512 (this config's default) is a perfect cube (8**3) --
    no shortfall, no warning. Bumping to the next octree-valid value
    (fixed_length % 7 == 1) that isn't a perfect cube (519: sqrt_len stays
    8, 8**3=512 < 519) must trigger parse.py's "N tokens will never reach
    the reconstructed feature map" warning for the default ("point")
    token_selection, but not for "area_weighted"/"cross_attention" -- see
    parse.py's own comment on why those methods aren't subject to it.
    """
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["ap"]["do_ap"] = True
    conf["ap"]["interp_size"] = 32
    conf["ap"]["fixed_length"] = 519

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")

        parse_config(args, load_balance_offline=True)
        assert "will never reach the reconstructed feature map" in capsys.readouterr().out

        for value in ("area_weighted", "cross_attention"):
            conf["model"]["token_selection"] = value
            with open(path, "w") as f:
                yaml.dump(conf, f)
            parse_config(args, load_balance_offline=True)
            assert "will never reach the reconstructed feature map" not in capsys.readouterr().out
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# epoch_shuffle_seed -- seeds FileReader's own per-epoch reshuffle, see
# UCF_VIT.dataloaders.dataset.FileReader's own docstring entry
# ---------------------------------------------------------------------------


def test_epoch_shuffle_seed_defaults_to_42_when_omitted():
    # No shipped config sets this -- matches bucket_shuffle_seed's own default
    # convention (a real seed by default, since some per-epoch reshuffle is a
    # strict improvement over none).
    parsed = validate_config(UNETR_CONFIG)
    assert parsed["dataloader"]["epoch_shuffle_seed"] == 42


def test_epoch_shuffle_seed_threads_through_when_set():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["dataloader"]["epoch_shuffle_seed"] = None  # opts out of reshuffling

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["dataloader"]["epoch_shuffle_seed"] is None
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# inference_output -- optional test.py/val.py sample-inference dump, see
# UCF_VIT.utils.inference_output.save_inference_batch
# ---------------------------------------------------------------------------


def test_inference_output_defaults_to_off_when_omitted():
    """No shipped config sets inference_output -- confirms the except KeyError
    default keeps it off and harmless for every config we ship."""
    parsed = validate_config(UNETR_CONFIG)
    assert parsed["inference_output"]["save"] is False
    assert parsed["inference_output"]["all_batches"] is False
    assert parsed["inference_output"]["num_batches"] == 1
    # Resolved against the repo root -- see trainer.checkpoint_path's identical
    # resolution and its own tests below for the full rationale.
    assert parsed["inference_output"]["output_dir"] == os.path.join(find_repo_root(), "inference_output")


def test_inference_output_threads_through_when_set():
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["inference_output"] = {
        "save": True,
        "all_batches": True,
        "num_batches": 3,
        "output_dir": "my_inference_dump",
    }

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["inference_output"] == {
            "save": True,
            "all_batches": True,
            "num_batches": 3,
            "output_dir": os.path.join(find_repo_root(), "my_inference_dump"),
        }
    finally:
        os.remove(path)


def test_inference_output_save_false_ignores_other_fields():
    """save:False forces all_batches/num_batches/output_dir back to their
    defaults even if the config sets them to something else -- matches
    tiling_conf/ap_conf's same "off means off" convention elsewhere in
    parse.py."""
    with open(UNETR_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["inference_output"] = {
        "save": False,
        "all_batches": True,
        "num_batches": 3,
        "output_dir": "my_inference_dump",
    }

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["inference_output"] == {
            "save": False,
            "all_batches": False,
            "num_batches": 1,
            "output_dir": os.path.join(find_repo_root(), "inference_output"),
        }
    finally:
        os.remove(path)
