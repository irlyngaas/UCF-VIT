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

Also verifies the backward pass: forward-only correctness says the
sharding math is right, but says nothing about whether gradients are
correctly synchronized across the tensor-parallel group during backprop
(F_Identity_B_AllReduce's/F_AllReduce_B_Identity's whole reason for
existing) -- a bug there wouldn't crash training, just silently corrupt
gradients. See the `_backward_matches_reference` tests' own docstrings for
how each parameter's expected gradient is derived.

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


def _build_grad_weight(local_rank):
    """A fixed, non-uniform per-output-element weighting for the
    backward-pass tests' loss (`(output * weight).sum()`), identical across
    every rank -- same determinism technique as `_build_input`, different
    seed offset so it's independent of it. Deliberately not a plain
    `.sum()`: that hands every output element an incoming gradient of
    exactly 1, a special/symmetric case a broken all-reduce (e.g. one that
    silently double-counts or drops a rank's contribution) could still
    accidentally satisfy; a non-uniform weighting is more sensitive to
    that class of bug.
    """
    g = torch.Generator(device="cpu").manual_seed(SEED + 2)
    w = torch.randn(BATCH, SEQ_LEN, DIM, generator=g)
    return w.to(f"cuda:{local_rank}")


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
def test_mlp_tensor_parallel_backward_matches_reference(tensor_par_size, dist_info):
    """Companion to test_mlp_tensor_parallel_forward_matches_reference,
    covering the backward pass: forward-only correctness says the sharding
    math is right, but says nothing about whether gradients are correctly
    synchronized across the tensor-parallel group during backprop --
    exactly what Mlp.forward's F_Identity_B_AllReduce (before fc1) and
    F_AllReduce_B_Identity (after fc2) exist to do. A bug there wouldn't
    crash training, just silently corrupt gradients.

    Reference gradients are available in full on every rank without any
    extra communication (the reference module is built identically and
    redundantly on every rank, same as the forward test), so this compares
    entirely with local tensors -- no gather/broadcast needed beyond what
    the sharded model's own forward/backward already does internally.
    """
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = f"cuda:{local_rank}"

    tensor_par_group = _build_tensor_par_group(world_rank, tensor_par_size)
    tp_rank = dist.get_rank(group=tensor_par_group)

    torch.manual_seed(SEED)
    reference = Mlp(in_features=DIM, hidden_features=DIM * MLP_RATIO)
    reference = reference.to(device).eval()

    x_ref = _build_input(local_rank).requires_grad_(True)
    weight = _build_grad_weight(local_rank)
    expected = reference(x_ref)
    (expected * weight).sum().backward()

    sharded = Mlp(
        in_features=DIM,
        hidden_features=DIM * MLP_RATIO,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )
    sharded.load_state_dict(shard_mlp_state_dict(reference.state_dict(), tensor_par_size, tp_rank))
    sharded = sharded.to(device).eval()

    x_actual = _build_input(local_rank).requires_grad_(True)
    actual = sharded(x_actual)
    (actual * weight).sum().backward()

    # x's gradient: F_Identity_B_AllReduce's backward all-reduce-sums this
    # rank's local partial contribution with every other rank's, so
    # x_actual.grad should already be the fully-reduced, correct value on
    # every rank -- no further reduction needed in the test itself.
    torch.testing.assert_close(x_actual.grad, x_ref.grad, **TOL)

    # fc1.weight/fc1.bias (row-sharded) and fc2.weight (column-sharded)
    # gradients are computed independently per shard from purely local
    # quantities (this rank's own weight rows/columns and the shared,
    # already-correct upstream gradient) -- reuse shard_mlp_state_dict's
    # already-verified slicing to compute the expected shard directly from
    # the reference's full gradients. Passing only these three keys (no
    # fc2.bias) means the helper's bias-zeroing logic (a VALUE
    # reconstruction concern) never runs -- see the fc2.bias comment below
    # for why that logic doesn't apply to gradients.
    expected_grad_shard = shard_mlp_state_dict(
        {
            "fc1.weight": reference.fc1.weight.grad,
            "fc1.bias": reference.fc1.bias.grad,
            "fc2.weight": reference.fc2.weight.grad,
        },
        tensor_par_size,
        tp_rank,
    )
    torch.testing.assert_close(sharded.fc1.weight.grad, expected_grad_shard["fc1.weight"], **TOL)
    torch.testing.assert_close(sharded.fc1.bias.grad, expected_grad_shard["fc1.bias"], **TOL)
    torch.testing.assert_close(sharded.fc2.weight.grad, expected_grad_shard["fc2.weight"], **TOL)

    # fc2.bias is NOT sharded (Mlp.forward's post-fc2 F_AllReduce_B_Identity
    # sums every rank's local fc2 output, bias included, in the forward
    # pass) -- by linearity, d(sum_r local_out_r)/d(bias_r) = 1 for every
    # rank r, so dL/d(fc2.bias) is the SAME value on every rank, matching
    # the reference's full fc2.bias gradient directly. This holds
    # regardless of which rank's forward bias value was real vs zeroed by
    # shard_mlp_state_dict -- the gradient doesn't depend on the bias's own
    # value, only on the shared upstream gradient.
    torch.testing.assert_close(sharded.fc2.bias.grad, reference.fc2.bias.grad, **TOL)


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
    sharded.load_state_dict(shard_attention_state_dict(reference_state, NUM_HEADS, tensor_par_size, tp_rank))
    sharded = sharded.to(device).eval()

    with torch.no_grad():
        actual = sharded(x)

    torch.testing.assert_close(actual, expected, **TOL)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size", _valid_tensor_par_sizes(WORLD_SIZE))
def test_attention_tensor_parallel_backward_matches_reference(tensor_par_size, dist_info):
    """Companion to test_attention_tensor_parallel_forward_matches_reference,
    covering the backward pass -- same rationale as
    test_mlp_tensor_parallel_backward_matches_reference's docstring.
    """
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = f"cuda:{local_rank}"

    tensor_par_group = _build_tensor_par_group(world_rank, tensor_par_size)
    tp_rank = dist.get_rank(group=tensor_par_group)

    torch.manual_seed(SEED)
    reference = Attention(dim=DIM, num_heads=NUM_HEADS, qkv_bias=True, fused_attn=FusedAttn.NONE)
    reference = reference.to(device).eval()

    x_ref = _build_input(local_rank).requires_grad_(True)
    weight = _build_grad_weight(local_rank)
    expected = reference(x_ref)
    (expected * weight).sum().backward()

    sharded = Attention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qkv_bias=True,
        fused_attn=FusedAttn.NONE,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
    )
    sharded.load_state_dict(shard_attention_state_dict(reference.state_dict(), NUM_HEADS, tensor_par_size, tp_rank))
    sharded = sharded.to(device).eval()

    x_actual = _build_input(local_rank).requires_grad_(True)
    actual = sharded(x_actual)
    (actual * weight).sum().backward()

    # x's gradient: F_Identity_B_AllReduce's backward all-reduce-sums this
    # rank's local partial contribution with every other rank's (same
    # reasoning as the Mlp backward test).
    torch.testing.assert_close(x_actual.grad, x_ref.grad, **TOL)

    # qkv.weight/qkv.bias (head-sharded, not a flat row range -- see
    # shard_attention_state_dict's docstring) and proj.weight
    # (column-sharded) gradients are computed independently per shard from
    # purely local quantities -- reuse shard_attention_state_dict's
    # already-verified head-range slicing to compute the expected shard
    # directly from the reference's full gradients. Passing only these
    # three keys (no proj.bias) means the helper's bias-zeroing logic never
    # runs -- see the proj.bias comment below for why that doesn't apply
    # to gradients.
    expected_grad_shard = shard_attention_state_dict(
        {
            "qkv.weight": reference.qkv.weight.grad,
            "qkv.bias": reference.qkv.bias.grad,
            "proj.weight": reference.proj.weight.grad,
        },
        NUM_HEADS,
        tensor_par_size,
        tp_rank,
    )
    torch.testing.assert_close(sharded.qkv.weight.grad, expected_grad_shard["qkv.weight"], **TOL)
    torch.testing.assert_close(sharded.qkv.bias.grad, expected_grad_shard["qkv.bias"], **TOL)
    torch.testing.assert_close(sharded.proj.weight.grad, expected_grad_shard["proj.weight"], **TOL)

    # proj.bias is NOT sharded (Attention.forward's post-proj
    # dist.all_reduce sums every rank's local proj output, bias included) --
    # same linearity argument as Mlp's fc2.bias: dL/d(proj.bias) is the same
    # value on every rank, matching the reference's full gradient directly,
    # independent of which rank's forward bias value was real vs zeroed.
    torch.testing.assert_close(sharded.proj.bias.grad, reference.proj.bias.grad, **TOL)
