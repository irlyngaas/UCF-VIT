"""Real multi-rank numerical-correctness tests for tensor parallelism in
UCF_VIT.model.building_blocks (Mlp, Attention).

Verifies that a tensor_par_size > 1 forward pass, given weights sliced from
an identical tensor_par_size=1 reference module and an identical input,
produces (approximately) the same output as the reference's forward pass --
using a real NCCL process group and real collectives
(F_Identity_B_AllReduce, F_AllReduce_B_Identity, dist.all_reduce), not a
local single-process simulation. This is deliberately model-level
(Mlp/Attention directly, not the full VIT/training loop/FSDP/checkpointing
pipeline): those two classes are the entire set of code paths in
building_blocks.py that touch tensor-parallel collectives, and this
session's earlier bugs (in training.py's process_batch and arch.py's
_pos_embed) were only reachable via a real multi-rank tensor_par_size > 1
launch -- a local math simulation would not have caught that class of bug.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- skips this whole file, rather than erroring at
collection, if they're not installed (see tests/README.md's "lighter"
Tier 1-only install note).

IMPORTANT: same collective-call-safety note as test_init_par_groups.py --
init_par_groups and the broadcasts/all-reduces below are collective calls,
so every process in the job must make the same calls in the same order.
Which tests run/parametrize here must be decided identically on every rank,
i.e. only from world-size-derived values available at collection time,
never from this rank's own rank id.
"""

import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("xformers", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("timm", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("monai", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")

from UCF_VIT.model.building_blocks import Attention, Mlp  # noqa: E402
from UCF_VIT.utils.fused_attn import FusedAttn  # noqa: E402
from UCF_VIT.utils.misc import init_par_groups, shard_attention_state_dict, shard_mlp_state_dict  # noqa: E402

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))

SEED = 1234
DIM = 64
NUM_HEADS = 8
MLP_RATIO = 4
BATCH = 2
SEQ_LEN = 16
TOL = dict(rtol=1e-3, atol=1e-4)


def _valid_tensor_par_sizes(world_size):
    """tensor_par_size values that evenly divide world_size, excluding the
    trivial 1 (that's the reference path itself, not an interesting case).
    """
    if world_size <= 0:
        return []
    return [n for n in (2, 4, 8) if world_size % n == 0]


def _build_tensor_par_group(world_rank, tensor_par_size):
    data_par_size = WORLD_SIZE // tensor_par_size
    _, tensor_par_group, _, _, _ = init_par_groups(
        world_rank=world_rank,
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        fsdp_size=data_par_size,
        simple_ddp_size=1,
    )
    return tensor_par_group


def _build_input(local_rank):
    """A fixed input, identical across every rank -- deterministic CPU RNG
    given the same seed, same technique as the reference-weight construction
    below.
    """
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x = torch.randn(BATCH, SEQ_LEN, DIM, generator=g)
    return x.to(f"cuda:{local_rank}")


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
def test_reference_weights_are_deterministic_and_identical_across_ranks(dist_info):
    """Canary for the core assumption the rest of this file relies on: that
    torch.manual_seed(SEED) really does produce bit-identical CPU-init
    weights on every rank of this real job. Uses dist.all_reduce (already
    covered by test_smoke.py) rather than re-deriving new broadcast
    plumbing here, to check this without adding another hand-rolled
    collective code path.
    """
    torch.manual_seed(SEED)
    reference = Mlp(in_features=DIM, hidden_features=DIM * MLP_RATIO)
    fingerprint = sum(v.detach().float().sum() for v in reference.state_dict().values())
    fingerprint = fingerprint.reshape(1).to(f"cuda:{dist_info['local_rank']}")

    fp_min, fp_max = fingerprint.clone(), fingerprint.clone()
    dist.all_reduce(fp_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(fp_max, op=dist.ReduceOp.MAX)

    torch.testing.assert_close(fp_min, fp_max, rtol=0, atol=0)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size", _valid_tensor_par_sizes(WORLD_SIZE))
def test_mlp_tensor_parallel_forward_matches_reference(tensor_par_size, dist_info):
    world_rank = dist_info["world_rank"]
    device = f"cuda:{dist_info['local_rank']}"

    tensor_par_group = _build_tensor_par_group(world_rank, tensor_par_size)
    tp_rank = dist.get_rank(group=tensor_par_group)

    torch.manual_seed(SEED)
    reference = Mlp(in_features=DIM, hidden_features=DIM * MLP_RATIO)
    reference_state = reference.state_dict()
    reference = reference.to(device).eval()

    x = _build_input(dist_info["local_rank"])
    with torch.no_grad():
        expected = reference(x)

    sharded = Mlp(
        in_features=DIM,
        hidden_features=DIM * MLP_RATIO,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )
    sharded.load_state_dict(shard_mlp_state_dict(reference_state, tensor_par_size, tp_rank))
    sharded = sharded.to(device).eval()

    with torch.no_grad():
        actual = sharded(x)

    torch.testing.assert_close(actual, expected, **TOL)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size", _valid_tensor_par_sizes(WORLD_SIZE))
def test_attention_tensor_parallel_forward_matches_reference(tensor_par_size, dist_info):
    world_rank = dist_info["world_rank"]
    device = f"cuda:{dist_info['local_rank']}"

    tensor_par_group = _build_tensor_par_group(world_rank, tensor_par_size)
    tp_rank = dist.get_rank(group=tensor_par_group)

    torch.manual_seed(SEED)
    # qkv_bias=True deliberately overrides Attention's default (False) --
    # the bias-duplication bug shard_attention_state_dict guards against
    # only manifests when biases are present. fused_attn=FusedAttn.NONE is
    # the plain softmax(QK^T)V path, chosen so this test's tolerance budget
    # is about tensor parallelism, not fused-kernel numerical variance.
    reference = Attention(dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, fused_attn=FusedAttn.NONE)
    reference_state = reference.state_dict()
    reference = reference.to(device).eval()

    x = _build_input(dist_info["local_rank"])
    with torch.no_grad():
        expected = reference(x)

    sharded = Attention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qkv_bias=True,
        fused_attn=FusedAttn.NONE,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )
    sharded.load_state_dict(shard_attention_state_dict(reference_state, tensor_par_size, tp_rank))
    sharded = sharded.to(device).eval()

    with torch.no_grad():
        actual = sharded(x)

    torch.testing.assert_close(actual, expected, **TOL)
