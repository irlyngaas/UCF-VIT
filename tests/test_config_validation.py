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
