"""Real multi-rank numerical-correctness test for FSDP (fsdp_size > 1,
sharding_strategy=FULL_SHARD).

Verifies that wrapping a small stack of real UCF_VIT.model.building_blocks.
Block instances in torch.distributed.fsdp.FullyShardedDataParallel
(mirroring model/utils.py's get_model FULL_SHARD branch: fsdp_size > 1,
simple_ddp_size == 1) produces (approximately) the same forward-pass output
as running the same, unwrapped model directly -- using real FSDP parameter
sharding/gathering, not a simulation.

Deliberately scoped to fsdp_size > 1 alone (tensor_par_size=1) -- combined
fsdp_size > 1 + tensor_par_size > 1 (closer to production's HYBRID_SHARD
branch) is a deferred follow-up once this and
test_tensor_parallel_correctness.py are both proven independently.

Requires timm/monai/xformers (building_blocks.py's real, unconditional
top-level imports) -- see test_tensor_parallel_correctness.py's module
docstring for why this skips cleanly via importorskip instead of erroring
at collection.

IMPORTANT: same collective-call-safety note as test_init_par_groups.py --
init_par_groups and FSDP's own collectives require every process in the job
to make the same calls in the same order, so which tests run/parametrize
here must be decided identically on every rank, only from world-size-derived
values, never from this rank's own rank id.
"""

import functools
import os

import pytest
import torch
import torch.nn as nn

pytest.importorskip("xformers", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("timm", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")
pytest.importorskip("monai", reason="needs the real UCF_VIT.model.building_blocks deps -- run in the forge-vit env")

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy  # noqa: E402
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402

from UCF_VIT.model.building_blocks import Block  # noqa: E402
from UCF_VIT.utils.misc import init_par_groups  # noqa: E402

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))

SEED = 5678
DIM = 64
NUM_HEADS = 8
MLP_RATIO = 4
DEPTH = 2
BATCH = 2
SEQ_LEN = 16
TOL = dict(rtol=1e-4, atol=1e-5)

# float32-only, unlike model/utils.py's real bfloatPolicy -- keeps this
# test's tolerance tight and meaningful (same reasoning as
# test_tensor_parallel_correctness.py's float32 choice).
FLOAT32_POLICY = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)


def _valid_fsdp_sizes(world_size):
    """fsdp_size values that evenly divide world_size, excluding the
    trivial 1 (that's the reference path itself, not an interesting case).
    """
    if world_size <= 0:
        return []
    return [n for n in (2, 4, 8) if world_size % n == 0]


def _build_model():
    return nn.Sequential(*[Block(dim=DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO) for _ in range(DEPTH)])


def _build_input(local_rank):
    """A fixed input, identical across every rank -- deterministic CPU RNG
    given the same seed, same technique as the reference-weight construction
    below.
    """
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x = torch.randn(BATCH, SEQ_LEN, DIM, generator=g)
    return x.to(f"cuda:{local_rank}")


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("fsdp_size", _valid_fsdp_sizes(WORLD_SIZE))
def test_fsdp_full_shard_forward_matches_reference(fsdp_size, dist_info):
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = f"cuda:{local_rank}"

    torch.manual_seed(SEED)
    reference = _build_model().to(device).eval()

    x = _build_input(local_rank)
    with torch.no_grad():
        expected = reference(x)

    # data_par_size == fsdp_size and simple_ddp_size == 1 is exactly the
    # combination model/utils.py's get_model uses to select FULL_SHARD
    # (fsdp_size > 1 and simple_ddp_size == 1) rather than HYBRID_SHARD or
    # NO_SHARD.
    _, _, _, fsdp_group, _ = init_par_groups(
        world_rank=world_rank,
        data_par_size=fsdp_size,
        tensor_par_size=1,
        fsdp_size=fsdp_size,
        simple_ddp_size=1,
    )

    # Same seed as the reference -> bit-identical pre-wrap weights (same
    # technique test_tensor_parallel_correctness.py uses). sync_module_states
    # below is a real backstop on top of that, mirroring production's own
    # FSDP(...) call, not a substitute for it.
    torch.manual_seed(SEED)
    to_wrap = _build_model()
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    wrapped = FSDP(
        to_wrap,
        device_id=local_rank,
        process_group=fsdp_group,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=FLOAT32_POLICY,
        forward_prefetch=True,
        limit_all_gathers=False,
    ).eval()

    with torch.no_grad():
        actual = wrapped(x)

    torch.testing.assert_close(actual, expected, **TOL)
