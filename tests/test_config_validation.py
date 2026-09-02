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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "**", "*.yaml"), recursive=True))
SAP_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "sap", "base_config.yaml")
UNETR_CONFIG = os.path.join(REPO_ROOT, "configs", "basic_ct", "unetr", "base_config.yaml")


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_shipped_config_parses(config_path):
    validate_config(config_path)


def test_multiprocessing_context_defaults_to_none_when_omitted():
    """Every shipped config leaves dataloader.multiprocessing_context unset --
    DataLoader's own default (fork on Linux) stays in effect for all of them. (Tried
    "spawn" for basic_ct/sap specifically, to work around a real fork-after-CUDA-init
    segfault -- job 5390076 -- but that traded it for a different, faster,
    whole-job-killing crash, job 5394881; basic_ct/sap uses num_workers:0 instead now.
    See NativePytorchDataModule's multiprocessing_context docstring and
    basic_ct/sap/base_config.yaml's own num_workers comment for the full story.)"""
    conf = validate_config(SAP_CONFIG)
    assert conf["dataloader"]["multiprocessing_context"] is None


def test_multiprocessing_context_read_from_config_when_set():
    """No shipped config currently sets this (see test_multiprocessing_context_defaults_
    to_none_when_omitted for why) -- covers the parse.py plumbing itself via a synthetic
    override, so it doesn't silently bit-rot if a future config does need it."""
    with open(SAP_CONFIG) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf["dataloader"]["multiprocessing_context"] = "spawn"

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(conf, f)
        args = argparse.Namespace(config=path, pretrained_config="")
        parsed = parse_config(args, load_balance_offline=True)
        assert parsed["dataloader"]["multiprocessing_context"] == "spawn"
    finally:
        os.remove(path)


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
    for value in ("smallest_overlap", "area_weighted"):
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
    token_selection, but not for "area_weighted" -- see parse.py's own
    comment on why that method isn't subject to it.
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

        conf["model"]["token_selection"] = "area_weighted"
        with open(path, "w") as f:
            yaml.dump(conf, f)
        parse_config(args, load_balance_offline=True)
        assert "will never reach the reconstructed feature map" not in capsys.readouterr().out
    finally:
        os.remove(path)
