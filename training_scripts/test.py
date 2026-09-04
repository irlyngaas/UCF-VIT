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

from torch.utils.data import DataLoader

from UCF_VIT.parse import parse_config, get_split_conf
from UCF_VIT.model.utils import get_model
from UCF_VIT.training import eval_epoch
from UCF_VIT.utils.misc import init_par_groups, calculate_load_balancing_on_the_fly, slice_file_list
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler

from train import init_dist, set_cuda_device


#def main(device, local_rank):
def main():
    """Parses CLI args and config, builds the model/dataloader, and runs one forward-only test pass.

    Entry point for evaluating an existing checkpoint against a held-out test split
    (either a separate `data.dict_test_root_dirs`, or, if that's absent for a given
    dataset key, an automatic split carved out of that dataset's own training data --
    see `UCF_VIT.parse._resolve_dataset_splits`/`get_split_conf`). Identical to
    val.py except it evaluates the *test* split instead of the *val* one -- see
    val.py's own docstring for the full rationale (forcing
    trainer.resume_from_checkpoint, no optimizer/scheduler, forward pass only).
    """
#1. Load arguments from config file and setup parallelization
##############################################################################################################
    parser = ArgumentParser(description="")
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
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
    # See val.py's identical override for the full rationale.
    conf["trainer"]["resume_from_checkpoint"] = True

    test_conf = get_split_conf(conf, "test")
    # See val.py's identical override for the full rationale (comparable
    # apples-to-apples results across different amounts of training requires a
    # fixed, exact computation over the same files every time, not one distorted
    # by rank/worker file duplication).
    if test_conf["dataloader"].get("allow_file_reuse"):
        if dist.get_rank() == 0:
            print("Note: dataloader.allow_file_reuse is set in this config, but test.py always evaluates "
                  "with it off -- see val.py's own comment for why.", flush=True)
    test_conf["dataloader"] = dict(test_conf["dataloader"], allow_file_reuse=False)

    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(test_conf)

    #Set up communication groups based on the parallelism settings chosen
    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = conf["parallelism"]["data_par_size"], tensor_par_size = conf["parallelism"]["tensor_par_size"], fsdp_size = conf["parallelism"]["fsdp_size"], simple_ddp_size = conf["parallelism"]["simple_ddp_size"])

#2. Initialize Dataloader
##############################################################################################################
    # Deliberately built before set_cuda_device() below establishes this process's
    # first real CUDA context -- see train.py's set_cuda_device docstring for why.
    if test_conf["dataloader"]["type"] == "iterative_dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            data_module = NativePytorchDataModule(dict_root_dirs=test_conf["data"]["dict_root_dirs"],
                dict_start_idx = test_conf["dataloader"]["dict_start_idx"],
                dict_end_idx = test_conf["dataloader"]["dict_end_idx"],
                dict_buffer_sizes = test_conf["dataloader"]["dict_buffer_sizes"],
                dict_in_variables = test_conf["data"]["dict_in_variables"],
                num_channels_used = test_conf["data"]["num_channels"],
                batch_size = test_conf["dataloader"]["batch_size"],
                num_workers = test_conf["dataloader"]["num_workers"],
                pin_memory = test_conf["dataloader"]["pin_memory"],
                interp_size = test_conf["data"]["interp_size"],
                tile_size = test_conf["data"]["tile_size"],
                twoD = test_conf["data"]["twoD"],
                return_label = test_conf["dataloader"]["return_label"],
                dataset_group_list = dataset_group_list,
                batches_per_rank_epoch = batches_per_rank_epoch,
                div = test_conf["tiling"]["div"],
                tile_overlap = test_conf["tiling"]["tile_overlap"],
                adaptive_patching = test_conf["ap"]["do_ap"],
                fixed_length = test_conf["ap"]["fixed_length"],
                separate_channels = test_conf["ap"]["separate_channels"],
                data_par_size = test_conf["parallelism"]["data_par_size"],
                dataset = test_conf["data"]["dataset"],
                resize = test_conf["dataset_options"]["resize"],
                num_classes = test_conf["model"]["kwargs"]["num_classes"] if test_conf["model"]["type"] in ["UNETR", "SAP"] else None,
                ddp_group = ddp_group,
                allow_file_reuse = test_conf["dataloader"]["allow_file_reuse"],
                bucket_shuffle_seed = test_conf["dataloader"]["bucket_shuffle_seed"],
                epoch_shuffle_seed = test_conf["dataloader"]["epoch_shuffle_seed"],
            )

            data_module.setup()

            eval_dataloader = data_module.train_dataloader()
            if test_conf["dataloader"]["num_workers"] > 0:
                # Forces the DataLoader's worker pool to fork right now, while this
                # process still has no CUDA context at all -- see train.py's
                # set_cuda_device docstring for why that matters.
                iter(eval_dataloader)
        else:
            # Only tensor_par_group-rank-0 reads real data (see
            # UCF_VIT.training.process_batch's docstring); the rest of each
            # tensor-parallel group never touches eval_dataloader/data_module
            # directly.
            data_module = None
            eval_dataloader = None

    elif test_conf["dataloader"]["type"] == "dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            #TODO: Loop over dict keys
            dkey_test = list(test_conf["data"]["dict_root_dirs"])[0]
            # sorted(): see train.py's identical glob.glob call for why.
            test_list = sorted(glob.glob(os.path.join(test_conf["data"]["dict_root_dirs"][dkey_test],'*.jpg')))
            test_list = slice_file_list(test_list, test_conf["dataloader"]["dict_start_idx"][dkey_test], test_conf["dataloader"]["dict_end_idx"][dkey_test])

            test_data = test_conf["dataloader"]["dataset_module"](test_list, test_conf["data"]["dict_in_variables"][dkey_test], test_conf["data"]["tile_size"], adaptive_patching=test_conf["ap"]["do_ap"], fixed_length=test_conf["ap"]["fixed_length"], interp_size=test_conf["data"]["interp_size"], num_channels=test_conf["data"]["num_channels"][dkey_test], dataset=test_conf["data"]["dataset"], resize=test_conf["dataset_options"]["resize"].get(test_conf["data"]["dataset"]), div=test_conf["tiling"]["div"], tile_overlap=test_conf["tiling"]["tile_overlap"])

            # Same reasoning as train.py's own DistributedSampler construction.
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False, num_replicas=test_conf["parallelism"]["data_par_size"],rank=dist.get_rank(ddp_group))

            eval_dataloader = DataLoader(dataset = test_data, sampler=test_sampler, num_workers=test_conf["dataloader"]["num_workers"], persistent_workers=test_conf["dataloader"]["num_workers"] > 0, pin_memory=test_conf["dataloader"]["pin_memory"], batch_size=test_conf["dataloader"]["batch_size"], drop_last=False, collate_fn=lambda batch: test_conf["dataloader"]["collate_fn"](batch, adaptive_patching=test_conf["ap"]["do_ap"], return_label=test_conf["dataloader"]["return_label"]))
            if test_conf["dataloader"]["num_workers"] > 0:
                iter(eval_dataloader)  # forces the fork now -- see the iterative_dataloader branch's own comment above
        else:
            eval_dataloader = None

#3. Bind this process to its GPU, then load the model from checkpoint
##############################################################################################################
    device = set_cuda_device(local_rank)
    if test_conf["dataloader"]["type"] == "iterative_dataloader" and data_module is not None:
        data_module.to(device)  # no-op movement (no real nn.Parameters/buffers) -- see train.py's own identical comment

    model, epoch_start, loss_list = get_model(conf, {}, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)

#4. Test Pass
##############################################################################################################
    if test_conf["dataloader"]["type"] == "iterative_dataloader":
        iterations_per_epoch = 0
        for i,k in enumerate(batches_per_rank_epoch):
            if batches_per_rank_epoch[k] > iterations_per_epoch:
                iterations_per_epoch = batches_per_rank_epoch[k]

    elif test_conf["dataloader"]["type"] == "dataloader":
        # Same reasoning as train.py's own iterations_per_epoch broadcast.
        iterations_per_epoch_tensor = torch.tensor(
            len(eval_dataloader) if dist.get_rank(tensor_par_group) == 0 else 0, device=device
        )
        dist.broadcast(iterations_per_epoch_tensor, src=(dist.get_rank()//conf["parallelism"]["tensor_par_size"]*conf["parallelism"]["tensor_par_size"]), group=tensor_par_group)
        iterations_per_epoch = iterations_per_epoch_tensor.item()

    if conf["model"]["type"] == "DiffusionVIT":
        ddpm_scheduler = DDPM_Scheduler(num_time_steps=conf["model"]["kwargs"]["num_time_steps"]).to(device)
    else:
        ddpm_scheduler = None

    #tell the model that we are in eval mode. Matters because we have the dropout
    model.eval()
    if dist.get_rank() == 0:
        print("testing checkpoint from epoch ", epoch_start - 1, flush=True)

    eval_epoch(conf, model, eval_dataloader, epoch_start - 1, iterations_per_epoch, device, tensor_par_group, ddpm_scheduler)

if __name__ == "__main__":

    main()
    dist.destroy_process_group()
