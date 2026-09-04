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
    """Parses CLI args and config, builds the model/dataloader, and runs one forward-only validation pass.

    Entry point for evaluating an existing checkpoint against a held-out validation
    split (either a separate `data.dict_val_root_dirs`, or, if that's absent for a
    given dataset key, an automatic split carved out of that dataset's own training
    data -- see `UCF_VIT.parse._resolve_dataset_splits`/`get_split_conf`). Takes the
    same config file as the training run being evaluated (same `checkpoint_path`/
    `checkpoint_filename`); `trainer.resume_from_checkpoint` is forced to True here
    regardless of what the config says, since evaluating a checkpoint that doesn't
    exist yet isn't meaningful. Builds no optimizer/scheduler -- val.py only ever
    runs a forward pass (`eval_epoch`, under `torch.no_grad()`), never a backward one.
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
    # Evaluating a checkpoint is only meaningful if one already exists -- force this
    # regardless of what the config says, so the same config used for training can
    # be reused here unmodified. Safe w.r.t. get_model's own resume_from_checkpoint/
    # use_pretrained_model branching: get_model only ever reads use_pretrained_model
    # inside its `if not resume_from_checkpoint:` branch, which this override skips
    # entirely, so a stale use_pretrained_model value (if the original config had
    # one set) is never read.
    conf["trainer"]["resume_from_checkpoint"] = True

    val_conf = get_split_conf(conf, "val")
    # Unlike train.py, val.py never reuses (duplicates) files across ranks/workers,
    # regardless of what the config says -- the whole point of a val pass is a
    # fixed, exact computation over the same files every time, so results are
    # directly comparable across different amounts of training (different
    # checkpoints/epochs). Reuse would make the aggregate loss/accuracy a
    # weighted, not uniform, average over the val set (files landing on more than
    # one rank/worker counted more than once) -- and which files get duplicated
    # can shift with data_par_size, so even the *bias* wouldn't stay comparable
    # across runs. If there genuinely aren't enough val files for the requested
    # parallelism, this should fail loudly (same as before allow_file_reuse
    # existed), not silently evaluate on a distorted set.
    if val_conf["dataloader"].get("allow_file_reuse"):
        if dist.get_rank() == 0:
            print("Note: dataloader.allow_file_reuse is set in this config, but val.py always evaluates "
                  "with it off -- see val.py's own comment for why.", flush=True)
    val_conf["dataloader"] = dict(val_conf["dataloader"], allow_file_reuse=False)

    if conf["dataloader"]["type"] == "iterative_dataloader":
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(val_conf)

    #Set up communication groups based on the parallelism settings chosen
    ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = conf["parallelism"]["data_par_size"], tensor_par_size = conf["parallelism"]["tensor_par_size"], fsdp_size = conf["parallelism"]["fsdp_size"], simple_ddp_size = conf["parallelism"]["simple_ddp_size"])

#2. Initialize Dataloader
##############################################################################################################
    # Deliberately built before set_cuda_device() below establishes this process's
    # first real CUDA context -- see train.py's set_cuda_device docstring for why.
    if val_conf["dataloader"]["type"] == "iterative_dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            data_module = NativePytorchDataModule(dict_root_dirs=val_conf["data"]["dict_root_dirs"],
                dict_start_idx = val_conf["dataloader"]["dict_start_idx"],
                dict_end_idx = val_conf["dataloader"]["dict_end_idx"],
                dict_buffer_sizes = val_conf["dataloader"]["dict_buffer_sizes"],
                dict_in_variables = val_conf["data"]["dict_in_variables"],
                num_channels_used = val_conf["data"]["num_channels"],
                batch_size = val_conf["dataloader"]["batch_size"],
                num_workers = val_conf["dataloader"]["num_workers"],
                pin_memory = val_conf["dataloader"]["pin_memory"],
                interp_size = val_conf["data"]["interp_size"],
                tile_size = val_conf["data"]["tile_size"],
                twoD = val_conf["data"]["twoD"],
                return_label = val_conf["dataloader"]["return_label"],
                dataset_group_list = dataset_group_list,
                batches_per_rank_epoch = batches_per_rank_epoch,
                div = val_conf["tiling"]["div"],
                tile_overlap = val_conf["tiling"]["tile_overlap"],
                adaptive_patching = val_conf["ap"]["do_ap"],
                fixed_length = val_conf["ap"]["fixed_length"],
                separate_channels = val_conf["ap"]["separate_channels"],
                data_par_size = val_conf["parallelism"]["data_par_size"],
                dataset = val_conf["data"]["dataset"],
                resize = val_conf["dataset_options"]["resize"],
                num_classes = val_conf["model"]["kwargs"]["num_classes"] if val_conf["model"]["type"] in ["UNETR", "SAP"] else None,
                ddp_group = ddp_group,
                allow_file_reuse = val_conf["dataloader"]["allow_file_reuse"],
                bucket_shuffle_seed = val_conf["dataloader"]["bucket_shuffle_seed"],
                epoch_shuffle_seed = val_conf["dataloader"]["epoch_shuffle_seed"],
            )

            data_module.setup()

            eval_dataloader = data_module.train_dataloader()
            if val_conf["dataloader"]["num_workers"] > 0:
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

    elif val_conf["dataloader"]["type"] == "dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            #TODO: Loop over dict keys
            dkey_val = list(val_conf["data"]["dict_root_dirs"])[0]
            # sorted(): see train.py's identical glob.glob call for why.
            val_list = sorted(glob.glob(os.path.join(val_conf["data"]["dict_root_dirs"][dkey_val],'*.jpg')))
            val_list = slice_file_list(val_list, val_conf["dataloader"]["dict_start_idx"][dkey_val], val_conf["dataloader"]["dict_end_idx"][dkey_val])

            val_data = val_conf["dataloader"]["dataset_module"](val_list, val_conf["data"]["dict_in_variables"][dkey_val], val_conf["data"]["tile_size"], adaptive_patching=val_conf["ap"]["do_ap"], fixed_length=val_conf["ap"]["fixed_length"], interp_size=val_conf["data"]["interp_size"], num_channels=val_conf["data"]["num_channels"][dkey_val], dataset=val_conf["data"]["dataset"], resize=val_conf["dataset_options"]["resize"].get(val_conf["data"]["dataset"]), div=val_conf["tiling"]["div"], tile_overlap=val_conf["tiling"]["tile_overlap"])

            # Same reasoning as train.py's own DistributedSampler construction.
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False, num_replicas=val_conf["parallelism"]["data_par_size"],rank=dist.get_rank(ddp_group))

            eval_dataloader = DataLoader(dataset = val_data, sampler=val_sampler, num_workers=val_conf["dataloader"]["num_workers"], persistent_workers=val_conf["dataloader"]["num_workers"] > 0, pin_memory=val_conf["dataloader"]["pin_memory"], batch_size=val_conf["dataloader"]["batch_size"], drop_last=False, collate_fn=lambda batch: val_conf["dataloader"]["collate_fn"](batch, adaptive_patching=val_conf["ap"]["do_ap"], return_label=val_conf["dataloader"]["return_label"]))
            if val_conf["dataloader"]["num_workers"] > 0:
                iter(eval_dataloader)  # forces the fork now -- see the iterative_dataloader branch's own comment above
        else:
            eval_dataloader = None

#3. Bind this process to its GPU, then load the model from checkpoint
##############################################################################################################
    device = set_cuda_device(local_rank)
    if val_conf["dataloader"]["type"] == "iterative_dataloader" and data_module is not None:
        data_module.to(device)  # no-op movement (no real nn.Parameters/buffers) -- see train.py's own identical comment

    model, epoch_start, loss_list = get_model(conf, {}, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)

#4. Validation Pass
##############################################################################################################
    if val_conf["dataloader"]["type"] == "iterative_dataloader":
        iterations_per_epoch = 0
        for i,k in enumerate(batches_per_rank_epoch):
            if batches_per_rank_epoch[k] > iterations_per_epoch:
                iterations_per_epoch = batches_per_rank_epoch[k]

    elif val_conf["dataloader"]["type"] == "dataloader":
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
        print("validating checkpoint from epoch ", epoch_start - 1, flush=True)

    eval_epoch(conf, model, eval_dataloader, epoch_start - 1, iterations_per_epoch, device, tensor_par_group, ddpm_scheduler)

if __name__ == "__main__":

    main()
    dist.destroy_process_group()
