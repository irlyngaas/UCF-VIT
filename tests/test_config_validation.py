"""Runs every shipped example config through UCF_VIT.parse.parse_config.

This is a regression net against the parsers in src/UCF_VIT/parse.py: since
those functions mostly aren't unit-testable in isolation (they read entire
config files and cross-check many fields against each other), the cheapest
real coverage is to make sure every config we actually ship still parses.
"""

import glob
import os

import pytest

from validate_config import validate_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "**", "*.yaml"), recursive=True))


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_shipped_config_parses(config_path):
    validate_config(config_path)
