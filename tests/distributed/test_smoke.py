"""Basic connectivity sanity checks -- run these first if Tier 2 ever misbehaves.

If these fail, the problem is in the launch/environment (NCCL setup, network,
SLURM env vars), not in UCF_VIT code, so it's worth ruling out before chasing
failures in the more specific test files.
"""

import torch
import torch.distributed as dist


def test_world_size_and_rank_are_sane(dist_info):
    assert dist.get_world_size() == dist_info["world_size"]
    assert dist.get_rank() == dist_info["world_rank"]
    assert 0 <= dist_info["world_rank"] < dist_info["world_size"]


def test_all_reduce_sum_across_all_ranks(dist_info):
    world_size = dist_info["world_size"]
    rank = dist_info["world_rank"]

    x = torch.tensor([float(rank)], device=f"cuda:{dist_info['local_rank']}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))
    torch.testing.assert_close(x, torch.tensor([float(expected)], device=x.device))
