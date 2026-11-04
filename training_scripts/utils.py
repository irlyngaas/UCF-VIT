
import os
import random
import torch
import numpy as np
import torch.distributed as dist
import time
from datetime import timedelta
import copy

#DIST_TIMEOUT = timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_S", "600")))
#_GROUP_TIMEOUT = DIST_TIMEOUT

def warm_up_process_groups(groups, device, world_rank):
    """Force eager NCCL/RCCL communicator setup for each process group.

    dist.new_group() only allocates a group object; PyTorch/NCCL lazily
    builds the real communicator (bootstrap handshake, ring/tree channel
    construction, GDR/DMA-BUF memory registration) on that group's FIRST
    actual collective call. Left alone, that first call happens to fall
    inside the first training iteration's forward/backward -- competing
    with the (often much shorter, see init_par_groups' dist.new_group()
    calls, which get PyTorch's 600s default rather than the long timeout
    passed to dist.init_process_group) collective watchdog timeout, and
    scaling up with job size to the point of blowing through it entirely.

    This issues one dummy all_reduce per group, right after group creation
    and before the model/FSDP is even built, so that cost is paid once,
    up front, in an explicit and separately-timed phase instead.

    IMPORTANT -- world barrier between group types: init_par_groups()
    creates MANY disjoint instances of most group types (e.g. at N nodes
    there are N separate 8-rank fsdp_group communicators, one per node,
    not one shared fsdp_group). Each rank here only warms up its OWN
    instance, so without a global synchronization point, ranks that finish
    quickly race ahead into the NEXT group type (or into the real training
    loop) while OTHER ranks' bootstrap for the CURRENT group type is still
    in flight. That lets many independent communicator bootstraps overlap
    in wall-clock time and contend for shared rendezvous/bootstrap
    resources (a plausible explanation for why a real, post-warm-up
    fsdp_group all_gather still hit the full NCCL watchdog timeout at 32
    nodes, immediately after this function reported success). A
    dist.barrier() on the default WORLD group after each group type below
    forces every rank to wait for ALL instances of that type to finish
    before any rank starts the next one, trading a bit of extra warm-up
    wall time for strictly serialized (per-type) bootstrap storms instead
    of everything overlapping at once.

    Args:
        groups: dict of {name: group}. init_par_groups() partitions all
            world ranks into exactly one instance of each group type per
            rank, so every rank always holds a real (non-None) handle here
            -- the None check below is just a defensive fallback.
        device: CUDA device (index or torch.device) to allocate the dummy
            tensor on.
        world_rank: this process's global rank; only rank 0 prints timing.
            Every group type in init_par_groups() assigns world_rank 0 to
            its "first" instance, so rank 0's timing is representative.
    """
    if world_rank == 0:
        print("warm_up_process_groups: forcing eager NCCL/RCCL comm init", flush=True)

    overall_tic = time.perf_counter()

    # Bootstrap the default WORLD group first and unconditionally, even if
    # it's not in `groups` -- dist.barrier() below defaults to WORLD, so
    # WORLD itself must already be warm before we can use a barrier to
    # synchronize warm-up of everything else.
    torch.cuda.synchronize(device=device)
    tic = time.perf_counter()

    dummy = torch.ones(1, device=device)
    dist.all_reduce(dummy, group=dist.group.WORLD)

    torch.cuda.synchronize(device=device)
    toc = time.perf_counter()

    if world_rank == 0:
        print(
            f"warm_up_process_groups: 'WORLD (bootstrap)' (nRanks={dist.get_world_size()}) "
            f"dummy all_reduce took {(toc - tic):0.4f} seconds",
            flush=True,
        )

    for name, group in groups.items():
        if group is None:
            continue

        torch.cuda.synchronize(device=device)
        tic = time.perf_counter()

        dummy = torch.ones(1, device=device)
        dist.all_reduce(dummy, group=group)

        torch.cuda.synchronize(device=device)
        toc = time.perf_counter()

        if world_rank == 0:
            print(
                f"warm_up_process_groups: '{name}' (nRanks={dist.get_world_size(group)}) "
                f"dummy all_reduce took {(toc - tic):0.4f} seconds",
                flush=True,
            )

        # See "world barrier between group types" above: wait for every
        # instance of this group type across the whole job before any
        # rank moves on. Timed separately: at 512+ nodes intermittent hangs
        # have been observed after data_seq_ort_group's all_reduce completes
        # but before fsdp_group's print -- empty FR traces imply the stall
        # is here (WORLD timeout is hours) rather than in fsdp's 600s group.
        torch.cuda.synchronize(device=device)
        barrier_tic = time.perf_counter()
        dist.barrier(group=dist.group.WORLD)
        torch.cuda.synchronize(device=device)
        barrier_toc = time.perf_counter()

        if world_rank == 0:
            print(
                f"warm_up_process_groups: post-'{name}' WORLD barrier took "
                f"{(barrier_toc - barrier_tic):0.4f} seconds",
                flush=True,
            )

    if world_rank == 0:
        overall_toc = time.perf_counter()
        print(
            f"warm_up_process_groups: total warm-up wall time "
            f"{(overall_toc - overall_tic):0.4f} seconds",
            flush=True,
        )
