
import os
import random
import torch
import numpy as np
import torch.distributed as dist
import time
import copy
from datetime import timedelta

from UCF_VIT.training import process_batch, forward_step

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


def warm_up_kernel_jit(conf, train_dataloader, warm_it_loader, model, device, optimizer, grad_scaler, min_scale, tensor_par_group, ddpm_scheduler, world_rank):
    """Force eager MIOpen/CK GPU-kernel selection & JIT compilation before batch 0.

    warm_up_process_groups() above only exercises collective communication
    (all_reduce on dummy tensors) -- it never runs a real forward/backward
    pass, so it can't pre-trigger MIOpen/CK kernel compilation for the
    model's actual layer shapes. Left alone, every unique (layer type,
    shape, dtype) combination gets JIT-compiled from scratch inside batch 0
    of real training.

    Runs one real forward pass, backward pass, and optimizer step through
    the exact same calls train_epoch() uses per batch (process_batch,
    forward_step, loss.backward()/grad_scaler, optimizer.step()/
    grad_scaler.step()), on one throwaway batch.

    That batch is pulled from `warm_it_loader` when train.py already built
    one (dataloader.num_workers > 0), not a second, independent
    `iter(train_dataloader)` -- FileReader.__iter__ (UCF_VIT.dataloaders.dataset)
    mutates shared state (self._epoch, self.file_list) on every call, and
    with persistent_workers:True a second DataLoader-level iterator is not
    guaranteed to coexist safely with the first. train_epoch's epoch_start
    then resumes consuming the same `warm_it_loader`, one batch further
    along than it otherwise would be. Falls back to a fresh
    `iter(train_dataloader)` only when `warm_it_loader` is None but
    `train_dataloader` isn't (dataloader.num_workers == 0 -- no persistent
    worker to desync).

    The optimizer/scaler step is made a no-op on the real weights: every
    param group's `lr` is forced to 0.0 for this one call (p -= 0 *
    step_size == p, bit-exact), and optimizer.state_dict()/
    grad_scaler.state_dict() are backed up beforehand and restored
    afterward, so the momentum buffers/step-counter/scale/growth-tracker
    this call would otherwise seed with the throwaway batch's gradients are
    undone too. Still exercises AdamW's lazy per-parameter state allocation
    and its real update kernels for the first time here instead of in batch
    0.

    Must be called after the model/optimizer/grad_scaler are fully built
    and after model.train() (so the same kernels/dropout paths batch 0 will
    hit are the ones warmed), and before the training loop.

    A dist.barrier(WORLD) after each phase (dataloader, forward, backward,
    optimizer step) guarantees every rank has finished that phase's
    collectives before any rank moves on -- same reasoning as
    warm_up_process_groups' own per-group-type barrier above: FSDP handles
    unshard/reshard independently per rank, so a fast rank racing ahead
    while a slow rank is still mid-collective on the current phase is the
    same class of race that caused a real post-warm-up watchdog timeout
    there.

    Args:
        conf: Parsed training configuration dict (as returned by parse_config).
        train_dataloader: The real train dataloader, or None on ranks that
            never read real data (see train.py's own dataloader section).
        warm_it_loader: train.py's own pre-CUDA-init warm-up iterator over
            train_dataloader, if one was built (None when
            dataloader.num_workers == 0, or on ranks that never read real
            data) -- reused here instead of building a second iterator.
        model: The model (already FSDP-wrapped, if applicable).
        device: CUDA device the model/data live on.
        optimizer: The real optimizer. .step() IS called here, but with
            lr forced to 0.0 and its state_dict() restored afterward.
        grad_scaler: The real ShardedGradScaler, or None when
            conf["grad_scaler"]["use_grad_scaler"] is False.
        min_scale: Minimum allowed grad scaler scale -- same clamp
            train_epoch applies after grad_scaler.update().
        tensor_par_group: Process group for tensor-parallel batch distribution.
        ddpm_scheduler: DDPM_Scheduler used when conf["model"]["type"] == "DiffusionVIT", else None.
        world_rank: This process's global rank; only rank 0 prints timing.
    """
    if world_rank == 0:
        print(
            "warm_up_kernel_jit: forcing eager MIOpen/CK kernel JIT compilation "
            "via one real (discarded) forward+backward pass",
            flush=True,
        )

    overall_tic = time.perf_counter()

    # See docstring: reuse train.py's own warm_it_loader when it exists --
    # only build a fresh iterator when there isn't one to reuse (which is
    # also then the num_workers == 0 case, so no persistent worker to
    # desync). process_batch never dereferences it_loader on ranks that
    # don't read real data (train_dataloader is None there too).
    if warm_it_loader is not None:
        warmup_it_loader = warm_it_loader
    elif train_dataloader is not None:
        warmup_it_loader = iter(train_dataloader)
    else:
        warmup_it_loader = None

    torch.cuda.synchronize(device=device)
    tic = time.perf_counter()
    batch = process_batch(conf, warmup_it_loader, device, tensor_par_group, ddpm_scheduler)
    torch.cuda.synchronize(device=device)
    dataloader_s = time.perf_counter() - tic

    dist.barrier(group=dist.group.WORLD)

    if conf["model"]["type"] in ["VIT", "UNETR"]:
        loss, _ = forward_step(conf, batch, model)
    else:
        loss = forward_step(conf, batch, model)

    torch.cuda.synchronize(device=device)
    tic_fwd = time.perf_counter()
    forward_s = tic_fwd - overall_tic - dataloader_s

    dist.barrier(group=dist.group.WORLD)

    if conf["grad_scaler"]["use_grad_scaler"]:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    torch.cuda.synchronize(device=device)
    tic_step = time.perf_counter()
    backward_s = tic_step - tic_fwd

    dist.barrier(group=dist.group.WORLD)

    # Force lr=0.0 so this step is an exact no-op on the real weights, and
    # restore optimizer/scaler state afterward -- see docstring.
    opt_state_backup = copy.deepcopy(optimizer.state_dict())
    scaler_state_backup = copy.deepcopy(grad_scaler.state_dict()) if grad_scaler is not None else None
    original_lrs = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0

    if conf["grad_scaler"]["use_grad_scaler"]:
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if grad_scaler._scale < min_scale:
            grad_scaler._scale = torch.tensor(min_scale).to(grad_scaler._scale)
    else:
        optimizer.step()

    for group, lr in zip(optimizer.param_groups, original_lrs):
        group["lr"] = lr
    optimizer.load_state_dict(opt_state_backup)
    if grad_scaler is not None:
        grad_scaler.load_state_dict(scaler_state_backup)
    optimizer.zero_grad()

    torch.cuda.synchronize(device=device)
    toc = time.perf_counter()
    optimizer_step_s = toc - tic_step

    dist.barrier(group=dist.group.WORLD)

    del batch, loss

    if world_rank == 0:
        print(
            "warm_up_kernel_jit: "
            f"dataloader={dataloader_s:0.4f}s forward={forward_s:0.4f}s "
            f"backward={backward_s:0.4f}s optimizer_step={optimizer_step_s:0.4f}s "
            f"total warm-up wall time {(toc - overall_tic):0.4f} seconds",
            flush=True,
        )

