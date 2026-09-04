import functools
import os
import sys
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import time
import yaml
import math
import glob
from argparse import ArgumentParser

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from UCF_VIT.parse import parse_config, parse_pretrained_config
from UCF_VIT.model.utils import get_model
from UCF_VIT.training import load_optimizer_scheduler_from_checkpoint, train_epoch
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, init_par_groups, calculate_load_balancing_on_the_fly, slice_file_list
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler




def init_dist(args):
    """Determines this process's rank/local_rank and initializes the NCCL process group.

    Supports two launch mechanisms: MPI (via mpi4py, deriving rank/world size from
    `MPI.COMM_WORLD` and broadcasting the master address from rank 0) and Slurm
    (via `SLURM_*` environment variables).

    Deliberately does *not* call `torch.cuda.set_device`/touch a CUDA device --
    see `set_cuda_device`'s own docstring for why that's a separate, later step.
    `dist.init_process_group('nccl', ...)` itself doesn't establish a CUDA context
    either (no `device_id` is passed, so NCCL communicator creation stays lazy,
    deferred to each process group's first real collective) -- only pure
    rank/world-size bookkeeping happens here, which is exactly what lets
    `train.py`'s `main()` build (and fork the workers of) the training
    `DataLoader` *before* any CUDA context exists in this process at all.

    Args:
        args: Parsed command-line arguments; must have a `launcher` attribute equal
            to "mpi" or any other value (treated as Slurm).

    Returns:
        This process's `local_rank`.
    """
    if args.launcher == "mpi":
        from mpi4py import MPI
        import socket

        num_gpus_per_node = torch.cuda.device_count()
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        world_rank = rank = comm.Get_rank()
        local_rank = int(rank) % int(num_gpus_per_node) if num_gpus_per_node>0 else 0 # local_rank and device are 0 when using 1 GPU per task
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(world_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        master_addr = None
        if rank == 0:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            master_addr = ip_address
        master_addr = comm.bcast(master_addr, root=0)
        os.environ['MASTER_ADDR'] = master_addr

    else:#elif launcher == "slurm":

        os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['RANK'] = os.environ['SLURM_PROCID']

        world_size = int(os.environ['SLURM_NTASKS'])
        world_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])

    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    return local_rank


def set_cuda_device(local_rank):
    """Binds this process to its GPU, establishing this process's first real CUDA context.

    Split out from `init_dist` so `main()` can build the training `DataLoader`
    (and, with `dataloader.num_workers > 0`, fork its worker processes) *before*
    calling this -- forking a process that already has an active CUDA context is
    a documented hazard (CUDA/NCCL keep background threads that can be
    mid-critical-section, holding a lock, at the instant of the fork; the forked
    child inherits that lock held forever by a thread that no longer exists,
    causing hangs/segfaults with no relation to what the child actually runs --
    see `tests/README.md`'s "Fixed a real, intermittent basic_ct-sap+tensor_par
    segfault" entry for the original incident). Calling this only once the
    `DataLoader`'s worker pool already exists (and is reused for the rest of the
    run via `persistent_workers`, never re-forked) avoids the hazard entirely,
    instead of trading it for `num_workers:0`'s lost dataloader/compute overlap.

    Args:
        local_rank: This process's local rank, as returned by `init_dist`.

    Returns:
        This process's `torch.device`.
    """
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")



#def main(device, local_rank):
def main():
    """Parses CLI args and config, builds the dataloader/model/optimizer, and runs the full training loop.

    Entry point for FSDP-based training: initializes distributed process groups
    (data/tensor/FSDP parallel), builds either the iterative or standard PyTorch
    dataloader (before this process's first CUDA context exists -- see
    `set_cuda_device`), binds this process to its GPU, builds the model via
    `get_model`, sets up the optimizer/scheduler/grad scaler (restoring from a
    checkpoint if configured), and then runs `train_epoch` for each remaining
    epoch, reusing the same dataloader (and its worker pool, if any) for the
    whole run.
    """
#1. Load arguments from config file and setup parallelization
##############################################################################################################
    parser = ArgumentParser(description="")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    parser.add_argument(
        "--pretrained_config",
        type=str,
        default="",
        help="Path to configuration YAML file for pre-trained model",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="slurm",
        help="Type of launching to use ",
    )

    args = parser.parse_args()

    local_rank = init_dist(args)
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    conf = parse_config(args)
    pretrained_conf = parse_pretrained_config(args, conf)
    #TODO: Add function parse dataset specific options separately

    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)

    #Set up communication groups based on the parallelism settings chosen
    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = conf["parallelism"]["data_par_size"], tensor_par_size = conf["parallelism"]["tensor_par_size"], fsdp_size = conf["parallelism"]["fsdp_size"], simple_ddp_size = conf["parallelism"]["simple_ddp_size"])

#2. Initialize Dataloader
##############################################################################################################
    # Deliberately built before set_cuda_device() below establishes this process's
    # first real CUDA context -- see set_cuda_device's own docstring for why.
    if conf["dataloader"]["type"] == "iterative_dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            data_module = NativePytorchDataModule(dict_root_dirs=conf["data"]["dict_root_dirs"],
                dict_start_idx = conf["dataloader"]["dict_start_idx"],
                dict_end_idx = conf["dataloader"]["dict_end_idx"],
                dict_buffer_sizes = conf["dataloader"]["dict_buffer_sizes"],
                dict_in_variables = conf["data"]["dict_in_variables"],
                num_channels_used = conf["data"]["num_channels"],
                batch_size = conf["dataloader"]["batch_size"],
                num_workers = conf["dataloader"]["num_workers"],
                pin_memory = conf["dataloader"]["pin_memory"],
                interp_size = conf["data"]["interp_size"],
                tile_size = conf["data"]["tile_size"],
                twoD = conf["data"]["twoD"],
                return_label = conf["dataloader"]["return_label"],
                dataset_group_list = dataset_group_list,
                batches_per_rank_epoch = batches_per_rank_epoch,
                div = conf["tiling"]["div"],
                tile_overlap = conf["tiling"]["tile_overlap"],
                adaptive_patching = conf["ap"]["do_ap"],
                fixed_length = conf["ap"]["fixed_length"],
                separate_channels = conf["ap"]["separate_channels"],
                data_par_size = conf["parallelism"]["data_par_size"],
                dataset = conf["data"]["dataset"],
                resize = conf["dataset_options"]["resize"],
                num_classes = conf["model"]["kwargs"]["num_classes"] if conf["model"]["type"] in ["UNETR", "SAP"] else None,
                ddp_group = ddp_group,
                allow_file_reuse = conf["dataloader"]["allow_file_reuse"],
                bucket_shuffle_seed = conf["dataloader"]["bucket_shuffle_seed"],
                epoch_shuffle_seed = conf["dataloader"]["epoch_shuffle_seed"],
            )

            data_module.setup()

            train_dataloader = data_module.train_dataloader()
            if conf["dataloader"]["num_workers"] > 0:
                # Forces the DataLoader's persistent worker pool to fork right now,
                # while this process still has no CUDA context at all. Iterating a
                # DataLoader immediately queues real prefetch work (and, via
                # epoch_shuffle_seed, a real reshuffle) in the background worker --
                # not just the fork itself -- so this iterator is kept (as
                # warm_it_loader) and handed to train_epoch for epoch_start below,
                # instead of being discarded and having that same work redone from
                # scratch the moment the real training loop calls iter() again.
                # train_dataloader itself is never rebuilt after this -- every epoch
                # after epoch_start builds its own fresh iterator from it as usual
                # (NativePytorchDataModule's epoch_shuffle_seed is what lets
                # FileReader.__iter__ reshuffle per-epoch on its own, instead of the
                # old reset() rebuilding a whole new DataLoader -- see its own
                # docstring entry).
                warm_it_loader = iter(train_dataloader)
            else:
                warm_it_loader = None
        else:
            # Only tensor_par_group-rank-0 reads real data (see
            # UCF_VIT.training.process_batch's docstring); the rest of each
            # tensor-parallel group never touches train_dataloader/
            # data_module directly (process_batch only dereferences them on
            # tensor_par_group-rank-0), but both names must still be bound
            # to *something* since train_epoch() references them
            # unconditionally for every rank.
            data_module = None
            train_dataloader = None
            warm_it_loader = None

    elif conf["dataloader"]["type"] == "dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            #TODO: Loop over dict keys
            dkey_train = list(conf["data"]["dict_root_dirs"])[0]
            # sorted(): glob.glob's order isn't guaranteed stable across separate
            # process launches -- for iterative_dataloader, NativePytorchDataModule
            # sorts before slicing for exactly this reason (see its own __init__
            # comment); this path needs the same determinism so train/val/test
            # membership (below) doesn't silently shift across runs.
            train_list = sorted(glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey_train],'*.jpg')))
            # The map-style "dataloader" path has no FileReader of its own to apply
            # dict_start_idx/dict_end_idx the way iterative_dataloader does -- slice
            # the globbed list directly instead, so this path respects the same
            # train/val/test split parse_config's _resolve_dataset_splits resolved
            # (a no-op slice, [0.0,1.0), for any config not using val.py/test.py's
            # auto-split at all).
            train_list = slice_file_list(train_list, conf["dataloader"]["dict_start_idx"][dkey_train], conf["dataloader"]["dict_end_idx"][dkey_train])

            train_data = conf["dataloader"]["dataset_module"](train_list, conf["data"]["dict_in_variables"][dkey_train], conf["data"]["tile_size"], adaptive_patching=conf["ap"]["do_ap"], fixed_length=conf["ap"]["fixed_length"], interp_size=conf["data"]["interp_size"], num_channels=conf["data"]["num_channels"][dkey_train], dataset=conf["data"]["dataset"], resize=conf["dataset_options"]["resize"].get(conf["data"]["dataset"]), div=conf["tiling"]["div"], tile_overlap=conf["tiling"]["tile_overlap"])

            # rank=world_rank is only correct when tensor_par_size == 1 (world_rank
            # then equals this replica's position among data_par_size replicas).
            # With tensor_par_size > 1, DistributedSampler needs the rank *within
            # the data-parallel dimension* -- dist.get_rank(ddp_group) gives
            # exactly that (ddp_group, from init_par_groups, contains one rank per
            # data-parallel replica at this tensor-parallel index), and reduces to
            # world_rank when tensor_par_size == 1 since ddp_group then spans the
            # whole world.
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, num_replicas=conf["parallelism"]["data_par_size"],rank=dist.get_rank(ddp_group))

            # persistent_workers: True (whenever num_workers > 0) so this DataLoader's
            # worker pool -- forked below, still before any CUDA context exists in
            # this process -- is reused for the whole run rather than being
            # re-forked every epoch. This map-style path has no per-epoch reset() of
            # its own to worry about (it was never rebuilt between epochs).
            train_dataloader = DataLoader(dataset = train_data, sampler=train_sampler, num_workers=conf["dataloader"]["num_workers"], persistent_workers=conf["dataloader"]["num_workers"] > 0, pin_memory=conf["dataloader"]["pin_memory"], batch_size=conf["dataloader"]["batch_size"], drop_last=True, collate_fn=functools.partial(conf["dataloader"]["collate_fn"], adaptive_patching=conf["ap"]["do_ap"], return_label=conf["dataloader"]["return_label"]))
            if conf["dataloader"]["num_workers"] > 0:
                # Kept as warm_it_loader (not discarded) -- see the
                # iterative_dataloader branch's own comment above for why.
                warm_it_loader = iter(train_dataloader)
            else:
                warm_it_loader = None
        else:
            # Same reasoning as the iterative_dataloader branch above.
            train_dataloader = None
            warm_it_loader = None

#3. Bind this process to its GPU, then initialize model, optimizer, and scheduler
##############################################################################################################
    device = set_cuda_device(local_rank)
    if conf["dataloader"]["type"] == "iterative_dataloader" and data_module is not None:
        data_module.to(device)  # no-op movement (no real nn.Parameters/buffers) -- kept for parity with NativePytorchDataModule being an nn.Module

    model, epoch_start, loss_list = get_model(conf, pretrained_conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)

    optimizer = configure_optimizer(model, conf["trainer"]["optimizer_type"], conf["optimizer"])
    scheduler = configure_scheduler(optimizer, conf["trainer"]["scheduler_type"], conf["scheduler"])

    if conf["trainer"]["resume_from_checkpoint"]:
        optimizer, scheduler, loss_list, epoch_start = load_optimizer_scheduler_from_checkpoint(conf, optimizer, scheduler, data_seq_ort_group, device)

    if conf["grad_scaler"]["use_grad_scaler"]:
        grad_scaler = ShardedGradScaler(init_scale=conf["grad_scaler"]["init_scale"], growth_interval=conf["grad_scaler"]["growth_interval"])
        min_scale = conf["grad_scaler"]["min_scale"]
    else:
        grad_scaler = None
        min_scale = None

#4. Training Loop
##############################################################################################################
    #Find iterations per epoch
    if conf["dataloader"]["type"] == "iterative_dataloader":
        iterations_per_epoch = 0
        for i,k in enumerate(batches_per_rank_epoch):
            if batches_per_rank_epoch[k] > iterations_per_epoch:
                iterations_per_epoch = batches_per_rank_epoch[k]

    elif conf["dataloader"]["type"] == "dataloader":
        # Only tensor_par_group-rank-0 built a real train_dataloader above, but
        # every rank needs the *same* iterations_per_epoch to loop in lockstep
        # (process_batch's per-tensor-parallel-group broadcasts require every
        # rank in a group to call it the same number of times) -- broadcast it
        # from each group's rank-0 the same way process_batch broadcasts batch
        # tensors. Reduces to a same-value no-op broadcast when
        # tensor_par_size == 1.
        iterations_per_epoch_tensor = torch.tensor(
            len(train_dataloader) if dist.get_rank(tensor_par_group) == 0 else 0, device=device
        )
        dist.broadcast(iterations_per_epoch_tensor, src=(dist.get_rank()//conf["parallelism"]["tensor_par_size"]*conf["parallelism"]["tensor_par_size"]), group=tensor_par_group)
        iterations_per_epoch = iterations_per_epoch_tensor.item()

    if conf["model"]["type"] == "DiffusionVIT":
        ddpm_scheduler = DDPM_Scheduler(num_time_steps=conf["model"]["kwargs"]["num_time_steps"]).to(device)
    else:
        ddpm_scheduler = None

    for epoch in range(epoch_start,conf["trainer"]["max_epochs"]):
        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        epoch_accuracy = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if dist.get_rank() == 0:
            print("epoch ",epoch,flush=True)

        # warm_it_loader (the pre-CUDA-init warm-up iterator, if one was built
        # above) is only valid for epoch_start -- every later epoch builds its
        # own fresh iterator inside train_epoch as usual.
        train_epoch(conf, model, train_dataloader, epoch, iterations_per_epoch, optimizer, scheduler, grad_scaler, min_scale, loss_list, device, tensor_par_group, ddpm_scheduler, it_loader=(warm_it_loader if epoch == epoch_start else None))

if __name__ == "__main__":

    main()
    dist.destroy_process_group()
