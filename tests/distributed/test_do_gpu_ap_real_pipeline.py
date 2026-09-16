"""Real multi-rank coverage for ap.do_gpu_ap's tensor-parallel interaction.

training.py's own comment on this (see process_batch's do_gpu_ap-vs-do_ap
docstring note) states the design directly: under do_gpu_ap:True,
process_batch broadcasts the *raw* image across a tensor-parallel group
exactly like it already does for do_ap:False (dataloader_do_ap is False
either way, so it's the same code path) -- then every rank in the group
independently calls the model's own on-device patchify (GPUPatchify2D/
GPUPatchify3D) on its own now-identical, broadcast copy of the data,
redundantly but "deterministically identically" rather than computing
seq_ps once and broadcasting the result.

That's an assumption process_batch's own design relies on, never actually
checked against real (potentially different) GPU hardware in one job --
a local single-process run (as in test_gpu_adaptive_patching.py/
test_gpu_adaptive_patching_3d.py) can't tell you whether two different
physical GPUs, given bit-identical input, produce bit-identical output
from the same ops (e.g. non-deterministic reduction order in a fused
conv/pooling kernel). This file checks exactly that, isolated from
attention/MLP's own tensor-parallel sharding correctness (already covered
by test_tensor_parallel_correctness.py) -- constructs GPUPatchify2D
directly (no model, no FSDP) rather than going through get_model, since
the property under test is about GPUPatchify2D's own cross-device
determinism, not about anything get_model/VIT would add.

IMPORTANT: same collective-call-safety note as test_init_par_groups.py --
init_par_groups and the broadcasts inside process_batch are collective
calls, so every process in the job must make the same calls in the same
order. Which tests run/parametrize here must be decided identically on
every rank, i.e. only from world-size-derived values available at
collection time, never from this rank's own rank id.
"""

import os

import pytest
import torch
import torch.distributed as dist

from UCF_VIT.model.gpu_adaptive_patching import GPUPatchify2D
from UCF_VIT.training import process_batch
from UCF_VIT.utils.misc import init_par_groups

WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "0"))

SEED = 4321
IMG_SIZE = (16, 16)
NUM_CHANNELS = 1
BATCH_SIZE = 2
FIXED_LENGTH = 13  # (16//2)**2=64 leaves; (64-13)%3==0
INTERP_SIZE = 4


def _valid_tensor_par_sizes(world_size):
    """Same helper as test_tensor_parallel_correctness.py's own -- tensor_par_size
    values that evenly divide world_size, excluding the trivial 1 (that's a
    single-process patchify call, already covered by the local test files)."""
    if world_size <= 0:
        return []
    return [n for n in (2, 4, 8) if world_size % n == 0]


def _build_conf(tensor_par_size):
    return {
        "trainer": {"data_type": "float32"},
        "parallelism": {"tensor_par_size": tensor_par_size},
        "ap": {
            "do_ap": True, "do_gpu_ap": True, "fixed_length": FIXED_LENGTH,
            "separate_channels": False,
        },
        "data": {
            "dataset": "catsdogs", "twoD": True, "tile_size": IMG_SIZE,
            "num_channels": {"catsdogs": NUM_CHANNELS}, "interp_size": INTERP_SIZE,
        },
        "dataloader": {"batch_size": BATCH_SIZE, "return_label": False},
        "model": {"type": "VIT"},
    }


class _FakeSingleBatchLoader:
    """Yields one fixed (deterministic across ranks, same seeding technique
    as test_tensor_parallel_correctness.py's own _build_input), real-shaped
    (data, label, variables, dict_key) 4-tuple -- get_batch's VIT/
    dataloader_do_ap:False branch unpacks exactly this shape (dataloader_do_ap
    is False under do_gpu_ap:True, same code path as do_ap:False entirely).
    Only ever read on tensor_par_group-rank-0 (process_batch's own dispatch);
    every other rank's copy of this loader is never touched.
    """

    def __iter__(self):
        return self

    def __next__(self):
        g = torch.Generator(device="cpu").manual_seed(SEED)
        data = torch.randn(BATCH_SIZE, NUM_CHANNELS, *IMG_SIZE, generator=g)
        label = torch.zeros(BATCH_SIZE, dtype=torch.int64)  # return_label:False -- never read
        return data, label, ["v0"], "catsdogs"


def _assert_identical_within_group(tensor, tensor_par_group):
    """Fingerprints `tensor` and confirms every rank within `tensor_par_group`
    (not the whole world -- different tensor-parallel groups are expected to
    get different batches under real data parallelism) computed the exact
    same value -- same MIN/MAX all_reduce technique as test_tensor_parallel_
    correctness.py's own test_reference_weights_are_deterministic_and_
    identical_across_ranks, scoped to this group via `group=`.
    """
    fingerprint = tensor.detach().float().sum().reshape(1)
    fp_min, fp_max = fingerprint.clone(), fingerprint.clone()
    dist.all_reduce(fp_min, op=dist.ReduceOp.MIN, group=tensor_par_group)
    dist.all_reduce(fp_max, op=dist.ReduceOp.MAX, group=tensor_par_group)
    torch.testing.assert_close(fp_min, fp_max, rtol=0, atol=0)


@pytest.mark.skipif(WORLD_SIZE == 0, reason="requires SLURM_NTASKS (run via srun)")
@pytest.mark.parametrize("tensor_par_size", _valid_tensor_par_sizes(WORLD_SIZE))
def test_do_gpu_ap_broadcast_and_patchify_are_identical_within_tensor_par_group(tensor_par_size, dist_info):
    world_rank = dist_info["world_rank"]
    local_rank = dist_info["local_rank"]
    device = torch.device(f"cuda:{local_rank}")

    data_par_size = WORLD_SIZE // tensor_par_size
    _, tensor_par_group, _, fsdp_group, simple_ddp_group = init_par_groups(
        world_rank=world_rank, data_par_size=data_par_size, tensor_par_size=tensor_par_size,
        fsdp_size=1, simple_ddp_size=data_par_size,
    )

    conf = _build_conf(tensor_par_size)
    it_loader = _FakeSingleBatchLoader()

    batch = process_batch(conf, it_loader, device, tensor_par_group, ddpm_scheduler=None)

    # Precondition: process_batch's own broadcast (dataloader_do_ap:False's
    # code path, taken here because do_gpu_ap:True) really did give every
    # rank in the group the identical raw image.
    _assert_identical_within_group(batch["data"], tensor_par_group)

    # The actual thing under test: GPUPatchify2D, constructed identically
    # (no randomness -- purely a function of img_size/fixed_length/interp_size/
    # min_size, all from conf) and run independently on every rank's own
    # (already-confirmed-identical) copy of batch["data"], must produce
    # bit-identical output on every rank in the group -- confirming the
    # "redundant but deterministically identical" design assumption on real
    # (and possibly physically different) GPU hardware, not just in theory.
    gpu_patchify = GPUPatchify2D(
        img_size=IMG_SIZE, fixed_length=FIXED_LENGTH, interp_size=INTERP_SIZE, min_size=2,
    ).to(device)
    seq_img, seq_size, seq_pos = gpu_patchify(batch["data"])

    _assert_identical_within_group(seq_img, tensor_par_group)
    _assert_identical_within_group(seq_size, tensor_par_group)
    _assert_identical_within_group(seq_pos, tensor_par_group)
