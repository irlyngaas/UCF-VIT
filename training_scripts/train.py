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
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, init_par_groups, calculate_load_balancing_on_the_fly
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler




def init_dist(args):
    """Determines this process's rank/device info and initializes the NCCL process group.

    Supports two launch mechanisms: MPI (via mpi4py, deriving rank/world size from
    `MPI.COMM_WORLD` and broadcasting the master address from rank 0) and Slurm
    (via `SLURM_*` environment variables).

    Args:
        args: Parsed command-line arguments; must have a `launcher` attribute equal
            to "mpi" or any other value (treated as Slurm).

    Returns:
        A tuple `(device, local_rank)`.
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

        torch.cuda.set_device(local_rank)
        device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

    else:#elif launcher == "slurm":

        os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['RANK'] = os.environ['SLURM_PROCID']

        world_size = int(os.environ['SLURM_NTASKS'])
        world_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        torch.cuda.set_device(local_rank)
        device = torch.cuda.current_device()

    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    return device, local_rank



#def main(device, local_rank):
def main():
    """Parses CLI args and config, builds the model/optimizer/dataloader, and runs the full training loop.

    Entry point for FSDP-based training: initializes distributed process groups
    (data/tensor/FSDP parallel), builds the model via `get_model`, sets up the
    optimizer/scheduler/grad scaler (restoring from a checkpoint if configured),
    builds either the iterative or standard PyTorch dataloader, and then runs
    `train_epoch` for each remaining epoch, resetting the dataloader between epochs.
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

    device, local_rank = init_dist(args)
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    conf = parse_config(args)
    pretrained_conf = parse_pretrained_config(args, conf)
    #TODO: Add function parse dataset specific options separately

    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)

    #Set up communication groups based on the parallelism settings chosen
    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = conf["parallelism"]["data_par_size"], tensor_par_size = conf["parallelism"]["tensor_par_size"], fsdp_size = conf["parallelism"]["fsdp_size"], simple_ddp_size = conf["parallelism"]["simple_ddp_size"])

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
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

#3. Initialize Dataloader
##############################################################################################################
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
                patch_size = conf["data"]["patch_size"],
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
            ).to(device)

            data_module.setup()

            train_dataloader = data_module.train_dataloader()
        else:
            # Only tensor_par_group-rank-0 reads real data (see
            # UCF_VIT.training.process_batch's docstring); the rest of each
            # tensor-parallel group never touches train_dataloader/
            # data_module directly (process_batch only dereferences them on
            # tensor_par_group-rank-0), but both names must still be bound
            # to *something* since train_epoch()/the per-epoch reset below
            # reference them unconditionally for every rank.
            data_module = None
            train_dataloader = None

    elif conf["dataloader"]["type"] == "dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            #TODO: Loop over dict keys
            dkey_train = list(conf["data"]["dict_root_dirs"])[0]
            train_list = glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey_train],'*.jpg'))

            train_data = conf["dataloader"]["dataset_module"](train_list, conf["data"]["dict_in_variables"][dkey_train], conf["data"]["tile_size"], adaptive_patching=conf["ap"]["do_ap"], fixed_length=conf["ap"]["fixed_length"], patch_size=conf["data"]["patch_size"], num_channels=conf["data"]["num_channels"][dkey_train], dataset=conf["data"]["dataset"])

            # rank=world_rank is only correct when tensor_par_size == 1 (world_rank
            # then equals this replica's position among data_par_size replicas).
            # With tensor_par_size > 1, DistributedSampler needs the rank *within
            # the data-parallel dimension* -- dist.get_rank(ddp_group) gives
            # exactly that (ddp_group, from init_par_groups, contains one rank per
            # data-parallel replica at this tensor-parallel index), and reduces to
            # world_rank when tensor_par_size == 1 since ddp_group then spans the
            # whole world.
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, num_replicas=conf["parallelism"]["data_par_size"],rank=dist.get_rank(ddp_group))

            train_dataloader = DataLoader(dataset = train_data, sampler=train_sampler, num_workers=conf["dataloader"]["num_workers"], pin_memory=conf["dataloader"]["pin_memory"], batch_size=conf["dataloader"]["batch_size"], drop_last=True, collate_fn=lambda batch: conf["dataloader"]["collate_fn"](batch, adaptive_patching=conf["ap"]["do_ap"], return_label=conf["dataloader"]["return_label"]))
        else:
            # Same reasoning as the iterative_dataloader branch above.
            train_dataloader = None

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
        ddpm_scheduler = DDPM_Scheduler(num_time_steps=conf["model"]["kwargs"]["time_steps"]).to(device)
    else:
        ddpm_scheduler = None

    for epoch in range(epoch_start,conf["trainer"]["max_epochs"]):
        #Reset dataloader module every epoch to ensure all files get used
        if epoch != epoch_start:
            if conf["dataloader"]["type"] == "iterative_dataloader":
                if dist.get_rank(tensor_par_group) == 0:
                    data_module.reset()
                    train_dataloader = data_module.train_dataloader()

        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        epoch_accuracy = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if dist.get_rank() == 0:
            print("epoch ",epoch,flush=True)

        train_epoch(conf, model, train_dataloader, epoch, iterations_per_epoch, optimizer, scheduler, grad_scaler, min_scale, loss_list, device, tensor_par_group, ddpm_scheduler)

if __name__ == "__main__":

    main()
    dist.destroy_process_group()
