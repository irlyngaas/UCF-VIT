"""Tests for UCF_VIT.utils.dist_functions -- the custom collective autograd ops
that model/building_blocks.py and model/arch.py use for tensor parallelism.

Every input tensor is built as `leaf.clone()` rather than passed as a leaf
directly: several of these ops (broadcast, all_reduce) mutate their input
tensor in place, which real usage in this codebase never does to an actual
leaf/parameter tensor either (it's always applied to an intermediate
activation). Mirroring that keeps these tests representative of real usage
and avoids an unrelated "in-place op on a leaf that requires grad" error.

Same collective-call-safety note as test_init_par_groups.py: any
skip/parametrize decisions here must depend only on world-size-derived values,
never on this rank's own rank id.
"""

import os

import pytest
import torch
import torch.distributed as dist

from UCF_VIT.utils.dist_functions import (
    F_AllReduce_B_Identity,
    F_Identity_B_AllReduce,
    all_gather,
    all_reduce,
    broadcast,
    gather,
)

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))
requires_slurm = pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")


def _leaf_value(rank, local_rank, offset=0.0):
    leaf = torch.tensor([float(rank) + offset], device=f"cuda:{local_rank}", requires_grad=True)
    return leaf, leaf.clone()


@requires_slurm
def test_all_reduce_forward_sum(dist_info):
    world_size = dist_info["world_size"]
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"])

    y = all_reduce(x, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))
    torch.testing.assert_close(y, torch.tensor([float(expected)], device=y.device))


@requires_slurm
def test_all_reduce_backward_is_another_all_reduce(dist_info):
    world_size = dist_info["world_size"]
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"])

    y = all_reduce(x, op=dist.ReduceOp.SUM)
    # Every rank runs the *same* loss formula against its (identical) copy of
    # y, so the local grad_output flowing into y is always 1.0 here -- easy to
    # reason about: total expected grad = world_size (one contribution per rank).
    loss = y.sum()
    loss.backward()

    torch.testing.assert_close(leaf.grad, torch.tensor([float(world_size)], device=leaf.grad.device))


@requires_slurm
def test_broadcast_forward_all_ranks_get_src_value(dist_info):
    src = 0
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"], offset=100.0)

    y = broadcast(x, src=src)

    torch.testing.assert_close(y, torch.tensor([100.0 + src], device=y.device))


@requires_slurm
def test_broadcast_backward_reduce_sums_to_src_and_zeros_elsewhere(dist_info):
    world_size = dist_info["world_size"]
    rank = dist_info["world_rank"]
    src = 0
    leaf, x = _leaf_value(rank, dist_info["local_rank"], offset=100.0)

    y = broadcast(x, src=src)
    # Weight the loss by (rank + 1) so each rank's local grad_output is distinct
    # and the expected sum-at-src is unambiguous (not just world_size copies of 1).
    loss = y.sum() * (rank + 1)
    loss.backward()

    if rank == src:
        expected = sum(r + 1 for r in range(world_size))
        torch.testing.assert_close(leaf.grad, torch.tensor([float(expected)], device=leaf.grad.device))
    else:
        torch.testing.assert_close(leaf.grad, torch.zeros_like(leaf.grad))


@requires_slurm
def test_all_gather_forward_orders_by_rank(dist_info):
    world_size = dist_info["world_size"]
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"])

    gathered = all_gather(x)

    assert len(gathered) == world_size
    for i, t in enumerate(gathered):
        torch.testing.assert_close(t, torch.tensor([float(i)], device=t.device))


@requires_slurm
def test_gather_only_dst_gets_real_data(dist_info):
    world_size = dist_info["world_size"]
    rank = dist_info["world_rank"]
    dst = 0
    # offset by 1 so every value is nonzero -- distinguishes "real gathered
    # value" from the zero-filled placeholder gather() leaves on non-dst ranks.
    leaf, x = _leaf_value(rank, dist_info["local_rank"], offset=1.0)

    gathered = gather(x, dst=dst)

    assert len(gathered) == world_size
    if rank == dst:
        for i, t in enumerate(gathered):
            torch.testing.assert_close(t, torch.tensor([float(i) + 1.0], device=t.device))
    else:
        for t in gathered:
            torch.testing.assert_close(t, torch.zeros_like(t))


@requires_slurm
def test_F_Identity_B_AllReduce_forward_is_unchanged(dist_info):
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"])

    y = F_Identity_B_AllReduce(x)

    torch.testing.assert_close(y, x.detach())


@requires_slurm
def test_F_AllReduce_B_Identity_forward_matches_all_reduce(dist_info):
    world_size = dist_info["world_size"]
    leaf, x = _leaf_value(dist_info["world_rank"], dist_info["local_rank"])

    y = F_AllReduce_B_Identity(x, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))
    torch.testing.assert_close(y, torch.tensor([float(expected)], device=y.device))
