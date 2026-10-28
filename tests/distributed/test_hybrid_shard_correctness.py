"""Real multi-rank numerical-correctness test for the combined
fsdp_size > 1 + tensor_par_size > 1 case (production's HYBRID_SHARD
branch).

Verifies that a small stack of real UCF_VIT.model.building_blocks.Block
instances, built with real tensor-parallel sharding (tensor_par_size > 1,
weights sliced from an identical tensor_par_size=1 reference via
UCF_VIT.utils.misc.shard_attention_state_dict/shard_mlp_state_dict) and
then wrapped in PyTorch's own FSDP with sharding_strategy=HYBRID_SHARD
(fsdp_size > 1, simple_ddp_size > 1), produces (approximately) the same
forward AND backward result as the same, unwrapped, unsharded reference --
using real NCCL process groups and real FSDP/tensor-parallel collectives,
not a simulation. This is the deferred follow-up explicitly called out in
test_tensor_parallel_correctness.py's and test_fsdp_correctness.py's own
module docstrings, now that both of those are independently proven on
real Frontier data.

Deliberately scoped to the minimal combination that forces production's
real model/utils.py get_model HYBRID_SHARD branch (fsdp_size > 1 and
simple_ddp_size > 1) together with real tensor-parallel sharding: for this
repo's actual 8-rank run_distributed_tests.sh launch, the only
(tensor_par_size, fsdp_size, simple_ddp_size) triple with all three > 1
multiplying to world_size is (2, 2, 2) -- see _valid_hybrid_configs.

Does not re-verify the torch.manual_seed(SEED) reference-determinism
assumption (test_tensor_parallel_correctness.py's
test_reference_weights_are_deterministic_and_identical_across_ranks
canary already covers that, in the same real job/environment).

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see test_tensor_parallel_correctness.py's module
docstring for why this skips cleanly via importorskip instead of erroring
at collection.

IMPORTANT: same collective-call-safety note as test_init_par_groups.py --
init_par_groups and FSDP's/tensor-parallel's own collectives require every
process in the job to make the same calls in the same order, so which
tests run/parametrize here must be decided identically on every rank, only
from world-size-derived values, never from this rank's own rank id.
"""

import functools
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

pytest.importorskip("xformers", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("timm", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("monai", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy  # noqa: E402
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402

from UCF_VIT.model.building_blocks import Block  # noqa: E402
from UCF_VIT.utils.misc import init_par_groups, shard_attention_state_dict, shard_mlp_state_dict  # noqa: E402

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))

SEED = 9012
DIM = 64
NUM_HEADS = 8
MLP_RATIO = 4
DEPTH = 2
BATCH = 2
SEQ_LEN = 16
# Combines test_tensor_parallel_correctness.py's forward tolerance (looser
# of the two -- more collectives/reordering here than FSDP alone) with its
# backward tolerance; retune after the first real run like every other
# tolerance in this file's siblings.
TOL = dict(rtol=1e-3, atol=1e-4)

# float32-only, matching both sibling files' reasoning for a tight,
# meaningful tolerance instead of production's bfloat16.
FLOAT32_POLICY = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)


def _valid_hybrid_configs(world_size):
    """(tensor_par_size, fsdp_size, simple_ddp_size) triples, all > 1,
    whose product is world_size -- the minimal combination that forces
    production's HYBRID_SHARD branch (fsdp_size > 1 and simple_ddp_size >
    1) together with real tensor-parallel sharding. For world_size=8
    (this repo's actual run_distributed_tests.sh launch), the only such
    triple is (2, 2, 2).
    """
    if world_size <= 0:
        return []
    configs = []
    for t in (2, 4):
        if world_size % t != 0:
            continue
        remainder = world_size // t
        for f in (2, 4):
            if remainder % f != 0:
                continue
            s = remainder // f
            if s > 1:
                configs.append((t, f, s))
    return configs


def _build_hybrid_groups(world_rank, tensor_par_size, fsdp_size, simple_ddp_size):
    # data_par_size = fsdp_size * simple_ddp_size (init_par_groups's own
    # docstring), and must span the entire world alongside tensor_par_size
    # (parse.py's data_par_size*tensor_par_size == world_size assertion) --
    # same reasoning already verified in test_fsdp_correctness.py's own
    # data_par_size fix.
    data_par_size = fsdp_size * simple_ddp_size
    _, tensor_par_group, _, fsdp_group, simple_ddp_group = init_par_groups(
        world_rank=world_rank,
        data_par_size=data_par_size,
        tensor_par_size=tensor_par_size,
        fsdp_size=fsdp_size,
        simple_ddp_size=simple_ddp_size,
    )
    return tensor_par_group, fsdp_group, simple_ddp_group


def _build_model(tensor_par_size=1, tensor_par_group=None):
    # qkv_bias=True deliberately overrides Attention's default (False) --
    # same reasoning as test_tensor_parallel_correctness.py's Attention
    # test: the bias-duplication bug shard_attention_state_dict guards
    # against only manifests when biases are present.
    return nn.Sequential(*[
        Block(
            dim=DIM,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            qkv_bias=True,
            tensor_par_size=tensor_par_size,
            tensor_par_group=tensor_par_group,
        )
        for _ in range(DEPTH)
    ])


def _build_input(local_rank):
    """A fixed input, identical across every rank -- deterministic CPU RNG
    given the same seed, same technique as every sibling test file.
    """
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x = torch.randn(BATCH, SEQ_LEN, DIM, generator=g)
    return x.to(f"cuda:{local_rank}")


def _build_grad_weight(local_rank):
    """Non-uniform, deterministic loss weighting for the backward-pass
    test -- same rationale as test_tensor_parallel_correctness.py's/
    test_fsdp_correctness.py's own _build_grad_weight.
    """
    g = torch.Generator(device="cpu").manual_seed(SEED + 2)
    w = torch.randn(BATCH, SEQ_LEN, DIM, generator=g)
    return w.to(f"cuda:{local_rank}")


def _shard_block_state_dict(full_state_dict, tensor_par_size, tp_rank):
    """Slices a single Block's full (tensor_par_size=1) state_dict into
    the shard TP rank tp_rank should load. norm1/norm2 (never sharded --
    Block.__init__ builds them at full size regardless of
    tensor_par_size) are copied verbatim; attn.*/mlp.* are delegated to
    shard_attention_state_dict/shard_mlp_state_dict, stripping/re-adding
    their submodule prefix since those helpers expect bare
    qkv.weight/fc1.weight-style keys.
    """
    attn_state = {k[len("attn."):]: v for k, v in full_state_dict.items() if k.startswith("attn.")}
    mlp_state = {k[len("mlp."):]: v for k, v in full_state_dict.items() if k.startswith("mlp.")}
    sharded_attn = shard_attention_state_dict(attn_state, NUM_HEADS, tensor_par_size, tp_rank)
    sharded_mlp = shard_mlp_state_dict(mlp_state, tensor_par_size, tp_rank)

    sharded = {k: v for k, v in full_state_dict.items() if k.startswith("norm1.") or k.startswith("norm2.")}
    sharded.update({f"attn.{k}": v for k, v in sharded_attn.items()})
    sharded.update({f"mlp.{k}": v for k, v in sharded_mlp.items()})
    return sharded


def _expected_block_grad_state(reference_block, tensor_par_size, tp_rank):
    """Computes the expected per-rank gradient state for a single Block,
    given the reference (tensor_par_size=1) Block's own gradients after
    backward. norm1/norm2 (never sharded) and attn.proj.bias/mlp.fc2.bias
    (unsharded, all-reduce-summed forward => identical gradient on every
    rank by linearity -- same reasoning as
    test_tensor_parallel_correctness.py's backward tests) are compared
    directly; attn.qkv/proj and mlp.fc1/fc2 weight (and qkv/fc1 bias)
    gradients are sliced via shard_attention_state_dict/
    shard_mlp_state_dict, reusing the exact logic those backward tests
    already verified on real Frontier data -- just applied to Block's
    submodules instead of standalone Attention/Mlp instances.
    """
    grads = {name: p.grad for name, p in reference_block.named_parameters()}

    attn_grads = {
        k[len("attn."):]: v for k, v in grads.items()
        if k.startswith("attn.") and k != "attn.proj.bias"
    }
    mlp_grads = {
        k[len("mlp."):]: v for k, v in grads.items()
        if k.startswith("mlp.") and k != "mlp.fc2.bias"
    }
    sharded_attn = shard_attention_state_dict(attn_grads, NUM_HEADS, tensor_par_size, tp_rank)
    sharded_mlp = shard_mlp_state_dict(mlp_grads, tensor_par_size, tp_rank)

    expected = {k: v for k, v in grads.items() if k.startswith("norm1.") or k.startswith("norm2.")}
    expected.update({f"attn.{k}": v for k, v in sharded_attn.items()})
    expected.update({f"mlp.{k}": v for k, v in sharded_mlp.items()})
    expected["attn.proj.bias"] = grads["attn.proj.bias"]
    expected["mlp.fc2.bias"] = grads["mlp.fc2.bias"]
    return expected


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size,fsdp_size,simple_ddp_size", _valid_hybrid_configs(WORLD_SIZE))
def test_hybrid_shard_forward_matches_reference(tensor_par_size, fsdp_size, simple_ddp_size, dist_info):
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = f"cuda:{local_rank}"

    torch.manual_seed(SEED)
    reference = _build_model().to(device).eval()

    x = _build_input(local_rank)
    with torch.no_grad():
        expected = reference(x)

    tensor_par_group, fsdp_group, simple_ddp_group = _build_hybrid_groups(
        world_rank, tensor_par_size, fsdp_size, simple_ddp_size
    )
    tp_rank = dist.get_rank(group=tensor_par_group)

    # Same seed as the reference -> bit-identical pre-wrap full weights,
    # then load_state_dict slices each Block down to this rank's real TP
    # shard (not just re-seeding -- tensor_par_size > 1 already builds
    # smaller Linear layers, so the random init itself would differ from a
    # true slice of the reference).
    torch.manual_seed(SEED)
    to_wrap = _build_model(tensor_par_size=tensor_par_size, tensor_par_group=tensor_par_group)
    for ref_block, shard_block in zip(reference, to_wrap):
        shard_block.load_state_dict(_shard_block_state_dict(ref_block.state_dict(), tensor_par_size, tp_rank))

    # auto_wrap_policy targets Block only (not Sequential), matching
    # test_fsdp_correctness.py's own convention.
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    wrapped = FSDP(
        to_wrap,
        device_id=local_rank,
        process_group=(fsdp_group, simple_ddp_group),
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=FLOAT32_POLICY,
        forward_prefetch=True,
        limit_all_gathers=False,
    ).eval()

    with torch.no_grad():
        actual = wrapped(x)

    torch.testing.assert_close(actual, expected, **TOL)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size,fsdp_size,simple_ddp_size", _valid_hybrid_configs(WORLD_SIZE))
def test_hybrid_shard_backward_matches_reference(tensor_par_size, fsdp_size, simple_ddp_size, dist_info):
    """Companion to test_hybrid_shard_forward_matches_reference, covering
    the backward pass -- same rationale as
    test_tensor_parallel_correctness.py's/test_fsdp_correctness.py's own
    backward tests: forward-only correctness says nothing about whether
    gradients are correctly synchronized through BOTH the tensor-parallel
    collectives and FSDP's own reduce-scatter at once.
    """
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = f"cuda:{local_rank}"

    torch.manual_seed(SEED)
    reference = _build_model().to(device).eval()

    x_ref = _build_input(local_rank).requires_grad_(True)
    weight = _build_grad_weight(local_rank)
    expected = reference(x_ref)
    (expected * weight).sum().backward()

    tensor_par_group, fsdp_group, simple_ddp_group = _build_hybrid_groups(
        world_rank, tensor_par_size, fsdp_size, simple_ddp_size
    )
    tp_rank = dist.get_rank(group=tensor_par_group)

    torch.manual_seed(SEED)
    to_wrap = _build_model(tensor_par_size=tensor_par_size, tensor_par_group=tensor_par_group)
    for ref_block, shard_block in zip(reference, to_wrap):
        shard_block.load_state_dict(_shard_block_state_dict(ref_block.state_dict(), tensor_par_size, tp_rank))

    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    wrapped = FSDP(
        to_wrap,
        device_id=local_rank,
        process_group=(fsdp_group, simple_ddp_group),
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=FLOAT32_POLICY,
        forward_prefetch=True,
        limit_all_gathers=False,
        # Required for summon_full_params(..., with_grads=True) below --
        # same fix as test_fsdp_correctness.py's own backward test
        # (confirmed against a real Frontier run, job 5341269).
        use_orig_params=True,
    ).eval()

    x_actual = _build_input(local_rank).requires_grad_(True)
    actual = wrapped(x_actual)
    (actual * weight).sum().backward()

    # x's gradient needs no FSDP-specific unsharding -- it's a normal,
    # never-sharded activation tensor. F_Identity_B_AllReduce's own
    # backward all-reduce (inside Attention/Mlp) already sums this rank's
    # local partial contribution with every other rank's in its
    # tensor_par_group before this point.
    torch.testing.assert_close(x_actual.grad, x_ref.grad, **TOL)

    # Build the expected gradient state for every block up front, indexed
    # by nn.Sequential's own "{index}.{submodule_path}" naming, so a
    # single summon_full_params pass over the whole wrapped Sequential can
    # look each parameter's expected value up by name directly -- avoids
    # relying on named_parameters()' iteration order matching a
    # differently-ordered dict.
    expected_grads = {}
    for i, ref_block in enumerate(reference):
        block_expected = _expected_block_grad_state(ref_block, tensor_par_size, tp_rank)
        expected_grads.update({f"{i}.{k}": v for k, v in block_expected.items()})

    # summon_full_params(with_grads=True) undoes FSDP's own sharding,
    # revealing each rank's full (but still tensor-parallel-sized, since
    # that's a model-architecture property FSDP doesn't know about)
    # gradient for direct comparison against expected_grads above.
    with FSDP.summon_full_params(wrapped, writeback=False, with_grads=True):
        for name, p in wrapped.named_parameters():
            torch.testing.assert_close(p.grad, expected_grads[name], **TOL)
