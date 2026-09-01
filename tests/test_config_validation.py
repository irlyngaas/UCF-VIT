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
