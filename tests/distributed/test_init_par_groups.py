"""Tests for UCF_VIT.utils.misc.init_par_groups.

IMPORTANT: init_par_groups makes collective calls (dist.new_group), so every
process in the job must make the *same* calls in the *same* order. That means
which tests run/parametrize here must be decided identically on every rank --
i.e. only from world-size-derived values (uniform across ranks, and available
from SLURM_NTASKS at collection time, before any rank has actually connected),
never from this process's own rank. Do not add per-rank skips/parametrization
to this file.
"""

import os

import pytest
import torch.distributed as dist

from UCF_VIT.utils.misc import init_par_groups

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))


def _valid_splits(world_size):
    """(tensor_par_size, fsdp_size, simple_ddp_size) triples with tensor_par_size * fsdp_size * simple_ddp_size == world_size."""
    if world_size <= 0:
        return []
    splits = [(1, world_size, 1), (1, 1, world_size)]
    if world_size % 2 == 0:
        splits.append((2, world_size // 2, 1))
    if world_size % 4 == 0:
        splits.append((2, world_size // 4, 2))
    seen = set()
    unique = []
    for s in splits:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size,fsdp_size,simple_ddp_size", _valid_splits(WORLD_SIZE))
def test_group_membership(tensor_par_size, fsdp_size, simple_ddp_size, dist_info):
    world_rank = dist_info["world_rank"]
    data_par_size = fsdp_size * simple_ddp_size

    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(
        world_rank=world_rank,
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
    )

    # Hand-derived from init_par_groups' rank layout (tensor-parallel-size-major:
    # consecutive `tensor_par_size` ranks share a tensor-parallel block).
    tp_idx = world_rank % tensor_par_size  # this rank's position within its tensor-parallel block
    block = world_rank // tensor_par_size  # this rank's data-parallel replica index

    # tensor_par_group: the tensor_par_size consecutive ranks sharing this rank's block
    assert dist.get_world_size(group=tensor_par_group) == tensor_par_size
    assert dist.get_rank(group=tensor_par_group) == tp_idx

    # ddp_group: all ranks sharing this rank's tp_idx (one per data-parallel replica)
    assert dist.get_world_size(group=ddp_group) == data_par_size
    assert dist.get_rank(group=ddp_group) == block

    # data_seq_ort_group: built from the exact same rank set as ddp_group
    assert dist.get_world_size(group=data_seq_ort_group) == data_par_size
    assert dist.get_rank(group=data_seq_ort_group) == block

    # fsdp_group: the fsdp_size ranks (within this rank's tp_idx) sharing this
    # rank's simple-ddp replica index (block // fsdp_size)
    assert dist.get_world_size(group=fsdp_group) == fsdp_size
    assert dist.get_rank(group=fsdp_group) == block % fsdp_size

    # simple_ddp_group: the simple_ddp_size ranks (within this rank's tp_idx)
    # sharing this rank's fsdp-shard index (block % fsdp_size)
    assert dist.get_world_size(group=simple_ddp_group) == simple_ddp_size
    assert dist.get_rank(group=simple_ddp_group) == block // fsdp_size
