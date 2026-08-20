"""Standalone utility to validate that a UCF_VIT training config file parses correctly.

Runs it through `UCF_VIT.parse.parse_config` (and, if requested, also
`parse_pretrained_config`) and reports success or the exact error. Initializes a
single-process torch.distributed group itself, so it can be run directly from a
login node/laptop without an actual multi-GPU/SLURM allocation -- useful for
quickly catching config typos or parse.py regressions before submitting a real
training job.

Usage:
    python utils/validate_config.py path/to/config.yaml
    python utils/validate_config.py path/to/config.yaml --pretrained-config path/to/pretrained.yaml
"""

import argparse
import os
import socket
import sys

import torch.distributed as dist

from UCF_VIT.parse import parse_config, parse_pretrained_config


def init_single_process_dist():
    """Initializes a single-process (world_size=1) torch.distributed group, if not already initialized.

    Lets code that calls `dist.get_rank()`/`dist.get_world_size()` (e.g.
    `parse_config`) run outside of an actual multi-process launch.
    """
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def validate_config(config_path, pretrained_config_path=""):
    """Parses a training config file (and optionally a pretrained-model config) and returns the result.

    Args:
        config_path: Path to a training config YAML file.
        pretrained_config_path: Path to a pretrained-model config YAML file to
            also validate, or "" to skip that check.

    Returns:
        The parsed config dict, as returned by `parse_config`.

    Raises:
        Whatever exception `parse_config`/`parse_pretrained_config` raises when
        the config is invalid (commonly `KeyError`, `AssertionError`, or
        `SystemExit` for the checks that call `sys.exit` directly).
    """
    init_single_process_dist()

    args = argparse.Namespace(config=config_path, pretrained_config=pretrained_config_path)
    conf = parse_config(args, load_balance_offline=True)

    if pretrained_config_path:
        parse_pretrained_config(args, conf)

    return conf


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=str, help="Path to the training config YAML file to validate")
    parser.add_argument(
        "--pretrained-config",
        type=str,
        default="",
        help="Path to a pretrained-model config YAML file to also validate",
    )
    args = parser.parse_args()

    print(f"Validating {args.config} ...")
    try:
        conf = validate_config(args.config, args.pretrained_config)
    except SystemExit as e:
        print(f"FAILED: config parsing called sys.exit({e.code!r}) -- see message above")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

    print("OK")
    print(f"  model type:        {conf['model']['type']}")
    print(f"  dataset:            {conf['data']['dataset']}")
    print(f"  data_par_size:       {conf['parallelism']['data_par_size']}")
    print(f"  tensor_par_size:     {conf['parallelism']['tensor_par_size']}")
    print(f"  fsdp_size:           {conf['parallelism']['fsdp_size']}")
    print(f"  simple_ddp_size:     {conf['parallelism']['simple_ddp_size']}")
    print(f"  adaptive_patching:   {conf['ap']['do_ap']}")


if __name__ == "__main__":
    main()
