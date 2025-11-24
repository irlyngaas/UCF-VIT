import os
import sys
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import time
import yaml
import math
import functools
from argparse import ArgumentParser

from timm.layers import use_fused_attn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
   size_based_auto_wrap_policy, wrap, transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from torch.nn import Sequential
from UCF_VIT.fsdp.building_blocks import Block

from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, init_par_groups, calculate_load_balancing_on_the_fly, is_power_of_two
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn

from torch.utils.data import DataLoader


#Use for both adaptive and non-adaptive patching
def training_step(data, variables, label, net: VIT, seq_ps):

    output = net.forward(data, variables, seq_ps)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output,label)

    return loss, output

def init_dist(args):
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

def parse_config(args):

    conf = yaml.load(open(args.config_path,'r'),Loader=yaml.FullLoader)
    if args.pretrained_config_path != "":
        pretrained_conf = yaml.load(open(args.pretrained_config_path,'r'),Loader=yaml.FullLoader)

# ---------------------------- TRAINER -------------------------------------------
    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    #If not resuming from checkpoint, check if starting from a pretrained model
    if not resume_from_checkpoint:
        try:
            use_pretrained_model = conf["trainer"]["use_pretrained_model"]
            if use_pretrained_model:
                assert args.pretrained_config_path != "", "If training using a pretrained model, you must pass in the config file from the pretrained model"
                
                try:
                    pretrained_checkpoint_filename = trainer_conf["pretrained_checkpoint_filename"]

                except KeyError:
                    if dist.get_rank() == 0:
                        sys.exit("pretrained_checkpoint_filename needs to be set in the trainer section")
                #Check if pretrained checkpoint file exists
                assert os.path.isfile(os.path.join(pretrained_conf["trainer"]["checkpoint_path"],trainer_conf["pretrained_checkpoint_filename"])), "Pretrained checkpoint file does not exist"

        except KeyError:
            if dist.get_rank() == 0:
                print("Since use_pretrained_model wasn't given, defaulting to not using a pretrained model")
            use_pretrained_model = False

    trainer_conf = {
        "max_epochs": conf['trainer']['max_epochs'],
        "data_type": conf['trainer']['data_type'],
        "gpu_type": conf['trainer']['gpu_type'],
        "checkpoint_path": conf['trainer']['checkpoint_path'],
        "checkpoint_filename": conf['trainer']['checkpoint_filename'],
        "resume_from_checkpoint": resume_from_checkpoint,
        "use_pretrained_model": use_pretrained_model if not resume_from_checkpoint else False,
    }

    #Check if checkpoint file exists
    if resume_from_checkpoint:
        checkpointExists = os.path.isfile(os.path.join(trainer_conf["checkpoint_path"],trainer_conf["checkpoint_filename"])), 
        if not checkpointExists:
            sys.exit("Checkpoint file does not exist")

# ---------------------------- PARALLELISM ---------------------------------------

    fsdp_size = conf['parallelism']['fsdp_size']
    simple_ddp_size = conf['parallelism']['simple_ddp_size']
    data_par_size = fsdp_size * simple_ddp_size
    tensor_par_size = conf['parallelism']['tensor_par_size']
    assert (data_par_size * seq_par_size * tensor_par_size) == dist.get_world_size(), "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal world_size"


    parallelism_conf = {
        "fsdp_size": fsdp_size,
        "simple_ddp_size": simple_ddp_size,
        "data_par_size": data_par_size,
        "tensor_par_size": tensor_par_size,
    }

# ---------------------------- OPTIMIZER -----------------------------------------
    optimizer_conf = {
        "lr": float(conf['model']['lr']),
        "beta_1": float(conf['model']['beta_1']),
        "beta_2": float(conf['model']['beta_2']),
        "weight_decay": float(conf['model']['weight_decay']),
    }

# ---------------------------- SCHEDULER -----------------------------------------
    scheduler_conf = {
        "warmup_epochs": conf['model']['warmup_epochs'],
        "warmup_start_lr": float(conf['model']['warmup_start_lr']),
        "eta_min": float(conf['model']['eta_min']),
    }

# ---------------------------- GRAD SCALER ---------------------------------------
    try:
        use_grad_scaler = conf["grad_scaler"]["use_grad_scaler"]
        grad_scaler_conf = {
            "use_grad_scaler": use_grad_scaler,
            "init_scale": conf["grad_scaler"]["init_scale"] if use_grad_scaler else None,
            "min_scale": conf["grad_scaler"]["min_scale"] if use_grad_scaler else None,
            "growth_interval": conf["grad_scaler"]["growth_interval"] if use_grad_scaler else None,
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no grad_scaler_conf was given in the config file, defaulting to not using a grad_scaler ")
        grad_scaler_conf = {"use_grad_scaler": False, "init_scale": None, "min_scale": None, "growth_interval": None}

# ---------------------------- MODEL ---------------------------------------------
    model_type = conf['model']['type']

    assert model_type in ['VIT', 'SAP', 'MAE', 'UNETR', 'DiffusionVIT'], "Model architecture type not supported. Choose a model architecture from the following choices: VIT, SAP, MAE, UNETR, DiffusionVIT"


    #Check Arguments that are required across all different architectures
    #TODO: Add checking on each argument, e.g. > 0
    try:
        embed_dim = conf['model']['embed_dim'],
    except KeyError:
        sys.exit("embed_dim is required")

    try:
        depth = conf['model']['depth']
    except KeyError:
        sys.exit("depth is required")

    try:
        num_heads = conf['model']['num_heads']
    except KeyError:
        sys.exit("num_heads is required")

    try:
        mlp_ratio = conf['model']['mlp_ratio']
    except KeyError:
        sys.exit("mlp_ratio is required")

    try:
        drop_path = conf['model']['drop_path']
    except KeyError:
        sys.exit("drop_path is required")

    try:
        drop_rate = conf['model']['drop_rate']
    except KeyError:
        sys.exit("drop_rate is required")

    try:
        use_channel_aggregation = conf['model']['use_channel_aggregation'],
    except KeyError:
        if dist.get_rank() == 0:
            print("use_channel_aggregation is not set, by default this is set to False")
        use_channel_aggregation = False
    
    #Set model specific arguments via a model_kwarg config
    kwargs = {}
    #if model_type == "SAP":
    #TODO: Add SAP specific arguments
    elif model_type == "MAE":
        try:
            linear_decoder = conf["model"]["linear_decoder"]
        except KeyError:
            if dist.get_rank() == 0:
                print("linear_decoder is not set, by default this is set to False")
            linear_decoder = False

        if not linear_decoder:
            try:
                decoder_depth = conf['model']['decoder_depth']
            except KeyError:
                sys.exit("decoder_depth is required")
            kwargs.update({"decoder_depth": decoder_depth})

            try:
                decoder_embed_dim = conf['model']['decoder_embed_dim']
            except KeyError:
                sys.exit("decoder_embed_dim is required")
            kwargs.update({"decoder_embed_dim": decoder_embed_dim})

            try:
                decoder_num_heads = conf['model']['decoder_num_heads']
            except KeyError:
                sys.exit("decoder_num_heads is required")
            kwargs.update({"decoder_num_heads": decoder_num_heads})

            try:
                decoder_mlp_ratio = conf['model']['decoder_mlp_ratio']
            except KeyError:
                if dist.get_rank() == 0:
                    print("decoder_mlp_ratio not set, by default setting to mlp_ratio")
                decoder_mlp_ratio = mlp_ratio
            kwargs.update({"decoder_mlp_ratio": decoder_mlp_ratio})
        else:
            kwargs.update({"decoder_embed_dim": None})
            kwargs.update({"decoder_depth": None})
            kwargs.update({"decoder_num_heads": None})
            kwargs.update({"decoder_mlp_ratio": None})
    #elif model_type == "UNETR":
    #TODO: Add UNETR specific arguments
    #elif model_type == "DiffusionVIT":
    #TODO: Add DiffusionVIT specific arguments
            
    
    #TODO: Check for given choice of model architecture check that proper inputs are provided

    model_conf = {
        "type": model_type,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "drop_path": drop_path,
        "drop_rate": drop_rate,
        "use_channel_aggregation": use_channel_aggregation,
        "kwargs": kwargs,
    }


# ---------------------------- TILING ------------------------------------------
    
    try:
        do_tiling = conf["tiling"]["do_tiling"]
        tiling_conf = {
            "do_tiling": do_tiling,
            "div": con["tiling"]["div"] if do_tiling else 1,
            "tile_overlap": conf["tiling"]["tile_overlap"] if do_tiling else 0.0,
            "use_all_data": conf["tiling"]["use_all_data"] if do_tiling else False,
        }
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no tiling_conf was given in the config file, this is defaulting to be ran without tiling")
        tiling_conf = {"do_tiling": False, "div": 1, "tile_overlap": 0.0, "use_all_data": False}
        
# ---------------------------- AP ------------------------------------------------
    try:
        do_ap = conf['ap']['do_ap']
        ap_conf = {
            "do_ap": do_ap,
            "fixed_length": conf['ap']['fixed_length'] if do_ap else None
            "separate_channels": conf['ap']['separate_channels'] if do_ap else False
            "use_adaptive_pos_emb": conf['ap']['use_adaptive_pos_emb'] if do_ap else False
        }
    except KeyError:
        if dist.get_rank() == 0:
            print("Since no ap_conf was given in the config file, this is defaulting to be ran with standard patching")
        ap_conf = {"do_ap": False, "fixed_length": None, "separate_channels": False, "use_adaptive_pos_emb": False}

    if ap_conf["do_ap"]:
        if ap_conf["separate_channels"]:
            assert not ap_conf["use_adaptive_pos_emb"], "Capability to use separate channels and adaptive pos_emb not implemented yet"
            
# ---------------------------- DATA ----------------------------------------------
    dataset = conf['data']['dataset']
    assert dataset in ["imagenet"], "This training script only supports imagenet"

    #To remove the need for specifying img_size in the config can add check_data_size function. The issue with automating this process is that raw data files come in various forms that are not consistent. This requires special functionality for each different dataset. Additionaly, it can be expensive to read individual datafiles that are very large on the fly.
    img_size = conf['data']['img_size']

    assert len(img_size) == 2 or len(img_size) == 3, "Img_size needs to be 2D or 3D"
    if len(img_size) == 2:
        twoD = True
    elif len(img_size) == 3:
        twoD = conf['data']['twoD'],

    patch_size = conf['data']['patch_size'],

    if twoD:
        tile_size = (img_size[0]/tiling_conf["div"], img_size[1]/tiling_conf["div"])
    else:
        tile_size = (img_size[0]/tiling_conf["div"], img_size[1]/tiling_conf["div"], img_size[2]/tiling_conf["div"])

        
    #If doing standard patching, check if img_size/tile_size is divisible by patch_size
    if not ap_conf["do_ap"]:
        for i in range(len(tile_size)):
            assert tile_size[i] % patch_size == 0, "img_size/tile_size not divisible by patch_size which is required when doing standard patching"

    #Check if overlapping splits up image evenly
    if tiling_conf["do_tiling"] and tiling_conf["tile_overlap"] > 0.0:
        for i in range(len(tile_size)):
            assert tile_size[i] % int(tile_size[i]*tiling_conf["tile_overlap"]) == 0, "Tile overlap doesn't divide up tile evenly. This assert can be turned off . However, to use all of the data turn on the use_all_data flag"

    #Check if num_channels is valid for the architecture being used, for classification we expect all images to have the same amount of channels, unless channel aggregation is used.
    num_channels = conf['data']['num_channels']
    if not model_conf["use_channel_aggregation"]
        for i,k in enumerate(num_channels):
            if i == 0:
                num_chan = num_channels[k]
            else:
                assert num_chan == num_channels[k], "If not using channel aggregation, num_channels across different datasets must be the same"
    #in_chans is the num_channels to be acrossed all datasets 
    in_chans = num_channels
        
    #Create default dict_in variables if it doesn't exist that assumes the channels are the same across different datasets
    try:
        dict_in_variables = conf['data']['dict_in_variables'],
        #Check if number of variables is valid 
        for i,k in enumerate(dict_in_variables):
            assert len(dict_in_variables[k]) == num_channels, "dict_in_variables must have the same amount as the num_channels"
    except KeyError:
        if dist.get_rank() == 0:
            print("Using a default in_variables, which assumes the datasets have channels that are all arranged in the same order. If you want to track input channels for uses such as training multi-modal data in a flexible manner it is recommended to create your own dict_in_variables given each channel appropriate unique labels")
        dict_in_variables = {}
        for i,k in enumerate(num_channels):
            in_variables_list = []
            for i in range(num_channels[k]):
                in_variables_list.append(str(i))
            dict_in_variables.update(in_variables_list)
    #Create default_vars
    for i,k in enumerate(dict_in_variables):
        if i == 0:
            default_vars = dict_in_variables[k]
        else:
            default_vars = list(set(default_vars + dict_in_variables[k]))

    #If using adaptive patching check if fixed length is compatible with tile_size
    if ap_conf['do_ap']:
        for i in range(len(tile_size)):
            p2 = is power_of_two(tile_size[i])
            assert p2, f"Tile Size in the {i} dimension must be a power of 2"

        if twoD:
            assert ap_conf["fixed_length"] % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
        else:
            assert ap_conf["fixed_length"] % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"
            
        
    data_conf = {    
        "dataset": dataset,
        "tile_size": tile_size,
        "patch_size": patch_size,
        "default_vars": default_vars,
        "twoD": twoD,
        "dict_root_dirs": conf['data']['dict_root_dirs'],
        "num_channels": num_channels,
        "dict_in_variables": dict_in_variables,
        "in_chans": in_chans,
        "num_classes": conf['data']['num_classes'], # Move to kwargs
    }

# ---------------------------- DATALOADER ----------------------------------------
    dataloader_type = conf['dataloader']['type']
    if dataloader_type == "dataloader":
        assert dataset in ["catsdogs"], "Only the catsdogs datset is supported with the standard torch dataloader, add your dataset to UCF_VIT/datasets"
        if data_conf['dataset'] == 'catsdogs':
            from UCF_VIT.datasets.catsdogs import CatsDogsDataset as dataset_module
            from UCF_VIT.datasets.catsdogs import CatsDogsCollate as collate_fn
        #elif add other datasets here
        #TODO: else dataset doesn't exist 
    dataloader_conf = {
        "type": dataloader_type,
        "dict_start_idx": conf['dataloader']['dict_start_idx'],
        "dict_end_idx": conf['dataloader']['dict_end_idx'],
        "dict_buffer_sizes": conf['dataloader']['dict_buffer_sizes'],
        "batch_size": conf['dataloader']['batch_size'],
        "num_workers": conf['dataloader']['num_workers'],
        "pin_memory": conf['dataloader']['pin_memory'],
        "dataset_module": dataset_module if dataloader_type == "dataloader" else None,
        "collate_fn": collate_fn if dataloader_type == "dataloader" else None,

    }

    
# ---------------------------- DATASET SPECIFIC OPTIONS --------------------------
    dataset_option_conf = {
        "imagenet_resize": conf['dataset_options']['imagenet_resize'] if dataset == "imagenet" else None, 
    }

# ---------------------------- PRETRAINED MODEL ----------------------------------
#TODO: Check pretrained_model for compatibility
#TODO: Add config for pretrained model checking whether the appropriate arguments are given and they are compatible with arguments for the given model to be trained
#TODO: Model, Data, and Parallelism compatibility
    if trainer_conf["use_pretrained_model"]:
        #Get and check model arguments
        embed_dim=pretrained_conf["model"]["embed_dim"],
        assert embed_dim == model_conf["embed_dim"], "Pretrained embed_dim does not match model to be trained"

        depth=pretrained_conf["model"]["depth"],
        assert depth == model_conf["depth"], "Pretrained depth does not match model to be trained"

        num_heads=pretrained_conf["model"]["num_heads"],
        assert num_heads == model_conf["num_heads"], "Pretrained num_heads does not match model to be trained"

        mlp_ratio=pretrained_conf["model"]["mlp_ratio"],
        assert mlp_ratio == model_conf["mlp_ratio"], "Pretrained num_heads does not match model to be trained"
        #TODO: Check these aren't needed
        drop_path_rate=pretrained_conf["model"]["drop_path"],
        drop_rate=pretrained_conf["model"]["drop_rate"],

        #Get and check data arguments
        pretrained_img_size = pretrained_conf["data"]["img_size"],
        if len(pretrained_img_size) == 2:
            twoD = True
        elif len(pretrained_img_size) == 3:
            twoD = pretrained_conf['data']['twoD'],

        pretrained_do_ap = pretrained_conf["ap"]["do_ap"]
        if pretrained_do_ap:
            assert ap_conf["do_ap"], "If pretrained model was trained with adaptive patching, the downstream model needs to use adaptive patching"
        else:
            assert not ap_conf["do_ap"], "If pretrained model was trained with standard patching, this model needs to use standard patching"

        pretrained_do_tiling = pretraind_conf["tiling"]["do_tiling"]
        if pretrained_do_tiling:
            pretrained_div = pretrained_conf["tiling"]["div"]
        else:
            pretrained_div = 1

        if twoD:
            pretrained_tile_size = (pretrained_img_size[0]/pretrained_div, img_size[1]/pretrained_div)
        else:
            pretrained_tile_size = (img_size[0]/pretrained_div, img_size[1]/pretrained_div, img_size[2]/pretrained_div)
             
        for i in range(len(pretrained_tile_size)):
            assert pretrained_tile_size[i] == data_conf["tile_size"], "Image/Tile size does not match between pretrained model and this model"
        
        pretrained_patch_size=pretrained_conf["data"]["patch_size"],
        assert pretrained_patch_size == data_conf["patch_size"], "Patch size does not match between pretrained model and this model"

        #Move num_classes to kwargs
        ###num_classes=conf["data"]["num_classes"],
        ###in_chans=conf["data"]["max_channels"],
        ###twoD=conf["data"]["twoD"],
        ###default_vars=conf["data"]["default_vars"],
        ###use_varemb=["pretrained_model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
        ###adaptive_patching=conf["ap"]["do_ap"],
        ###fixed_length=conf["ap"]["fixed_length"],
        ###FusedAttn_option=FusedAttn_option,
        ###use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
        ###weight_init='' if model_type == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
        ###**conf['pretrained_model']['kwargs']

    else:
        pretrained_conf = {}

    return { 
        "trainer": trainer_conf, 
        "parallelism": parallelism_conf, 
        "optimizer", optimizer_conf, 
        "scheduler": scheduler_conf, 
        "model": model_conf, 
        "tiling": tiling_conf, 
        "ap": ap_conf, 
        "data": data_conf, 
        "dataloader": dataloader_conf, 
        "dataset_option", dataset_option_conf,
        "pretrained_model", pretrained_conf,
    } 

def create_model(conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group):
    world_rank = dist.get_rank()

    if conf["trainer"]["data_type"] == "bfloat16":
        if conf["trainer"]["gpu_type"] == "amd":
            FusedAttn_option = FusedAttn.CK
        elif conf["trainer"]["gpu_type"] == "nvidia":
            FusedAttn_option = FusedAttn.FLASH #Can be switched to Default if Xformers is causing errors
    else:
        #Check whether flash attention is installed, if it's not use a basic python implementation
        if use_fused_attn():
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.NONE

    if conf["model"]["model_type"] == "VIT":
        from UCF_VIT.fsdp.arch import VIT as model_arch
    elif conf["model"]["model_type"] == "SAP":
        from UCF_VIT.fsdp.arch import SAP as model_arch
    elif conf["model"]["model_type"] == "MAE":
        from UCF_VIT.fsdp.arch import MAE as model_arch
    elif conf["model"]["model_type"] == "UNETR":
        from UCF_VIT.fsdp.arch import UNETR as model_arch
    elif conf["model"]["model_type"] == "DiffusionVIT":
        from UCF_VIT.fsdp.arch import DiffusionVIT as model_arch

    model = model_arch(
        img_size=conf["data"]["tile_size"],
        patch_size=conf["data"]["patch_size"],
        num_classes=conf["data"]["num_classes"], #Move to kwargs
        in_chans=conf["data"]["in_chans"],
        embed_dim=conf["model"]["embed_dim"],
        depth=conf["model"]["depth"],
        num_heads=conf["model"]["num_heads"],
        mlp_ratio=conf["model"]["mlp_ratio"],
        drop_path_rate=conf["model"]["drop_path"],
        drop_rate=conf["model"]["drop_rate"],
        twoD=conf["data"]["twoD"],
        default_vars=conf["data"]["default_vars"],
        single_channel=False, #TODO: Take out single channel option altogether
        use_varemb=["model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
        adaptive_patching=conf["ap"]["do_ap"],
        fixed_length=conf["ap"]["fixed_length"],
        FusedAttn_option=FusedAttn_option,
        use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
        weight_init='' if model_type == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
        **conf['model']['kwargs']
    ).to(device)

    if not conf["trainer"]["resume_from_checkpoint"]:
        epoch_start = 0
        loss_list = []
        if conf["trainer"]["use_pretrained_model"]: #Train from pre-trained model
            if world_rank==0:       
                print("Train starting from pretrained model.",flush=True)

            if conf["pretrained_model"]["model_type"] == "VIT":
                from UCF_VIT.fsdp.arch import VIT as pretrained_model_arch
            elif conf["pretrained_model"]["model_type"] == "SAP":
                from UCF_VIT.fsdp.arch import SAP as pretrained_model_arch
            elif conf["pretrained_model"]["model_type"] == "MAE":
                from UCF_VIT.fsdp.arch import MAE as pretrained_model_arch
            elif conf["pretrained_model"]["model_type"] == "UNETR":
                from UCF_VIT.fsdp.arch import UNETR as pretrained_model_arch
            elif conf["pretrained_model"]["model_type"] == "DiffusionVIT":
                from UCF_VIT.fsdp.arch import DiffusionVIT as pretrained_model_arch
            pretrained_model = pretrained_model_arch(
                img_size=conf["data"]["tile_size"],
                patch_size=conf["data"]["patch_size"],
                num_classes=conf["data"]["num_classes"], #move to kwargs
                in_chans=conf["data"]["in_chans"],
                embed_dim=conf["pretrained_model"]["embed_dim"],
                depth=conf["pretrained_model"]["depth"],
                num_heads=conf["pretrained_model"]["num_heads"],
                mlp_ratio=conf["pretrained_model"]["mlp_ratio"],
                drop_path_rate=conf["pretrained_model"]["drop_path"],
                drop_rate=conf["pretrained_model"]["drop_rate"],
                twoD=conf["data"]["twoD"],
                default_vars=conf["data"]["default_vars"],
                single_channel=False, #TODO: Take out single channel option altogether
                use_varemb=["pretrained_model"]["use_channel_aggregation"], #TODO: Change use_varemb to use_channel_aggregation in arch.py
                adaptive_patching=conf["ap"]["do_ap"],
                fixed_length=conf["ap"]["fixed_length"],
                FusedAttn_option=FusedAttn_option,
                use_adaptive_pos_emb=conf["ap"]["use_adaptive_pos_emb"],
                weight_init='' if model_type == "VIT" else 'skip', #Choose ['' or 'skip'] If using VIT use '' otherwise use 'skip'. Option whether to use VITs weight initialization or use the one corresponding to the architecture you choose
                **conf['pretrained_model']['kwargs']
            ).to(device)

            if world_rank< conf["parallelism"]["tensor_par_size"]:
                map_location = 'cpu' #TODO: Choose cpu or cuda+str
                #map_location = 'cuda:'+str(device)
                pretrained_checkpoint = torch.load(conf["pretrained_model"]["checkpoint_path"]+"/"+conf["trainer"]["pretrained_checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

            new_state_dict = OrderedDict()
            encoder_dict = new_state_dict
            model_dict = pretrained_model.state_dict()

            #Taking out encoder states from pretrained model. The decoder to take out from the model dict is different depending on the pretrained model
            #TODO: Add encoder_dict for different pretrained model types
            if conf["pretrained_model"]["model_type"] == "MAE":
                #decoder_dict = {k: v for k, v in model_dict.items() if ('decoder_pred' in k or 'attn_layers_decoder' in k or 'mask_token' in k)}
                encoder_dict = {k: v for k,v in model_dict.items() if ('decoder' not in k and 'mask_token' not in k)}
                #state_dict.append(encoder_dict)

            #Load encoder states from pretrained model into the model we want to train
            model_dict = model.state_dict()
            model_dict.update(encoder_dict)
            model.load_state_dict(model_dict)

        else: #Train from scratch
            if world_rank==0:       
                print("Train from scratch.",flush=True)

                #Check whether the specified checkpointing path exists or not
                isExist = os.path.exists(conf["trainer"]["checkpoint_path"])
                if not isExist:
                    #Create a new directory because it does not exist
                    os.makedirs(conf["trainer"]["checkpoint_path"])
                    print("The new checkpoint directory is created!")

                #Save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
                init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

                torch.save(init_model_dict,
                        checkpoint_path+'/initial_'+str(dist.get_rank())+'.pth')

                del init_model_dict

            dist.barrier()

            if world_rank!=0 and world_rank < conf["parallelism"]["tensor_par_size"]:

               #load initial model weights and synchronize model weights that are not in the training block among sequence parallel GPUs
               src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

               map_location = 'cpu' #TODO: Choose cpu or cuda+str
               #map_location = 'cuda:'+str(device)
               model.load_state_dict(torch.load(checkpoint_path+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)

    else: #Resume from checkpoint
        if world_rank < conf["parallelism"]["tensor_par_size"]:
            if os.path.exists(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                map_location = 'cpu' #TODO: Choose cpu or cuda+str
                #map_location = 'cuda:'+str(device)

                checkpoint = torch.load(conf["trainer"]["checkpoint_path"]+"/"+conf["trainer"]["checkpoint_filename"]+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch_start = checkpoint['epoch']
                del checkpoint

            else:
                print("resume from checkpoint was set to True. But the checkpoint path does not exist.",flush=True)
                sys.exit("checkpoint path does not exist")

    dist.barrier()
    
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block, Sequential   # < ---- Your Transformer layer class
        },
    )

    if conf["trainer"]["data_type"] == "float32":
        precision_dt = torch.float32
    elif conf["trainer"]["data_type"] == "bfloat16":
        precision_dt = torch.bfloat16
    else:
        raise RuntimeError("Data type not supported")

    bfloatPolicy = MixedPrecision(
        param_dtype=precision_dt,
        # Gradient communication precision.
        reduce_dtype=precision_dt,
        # Buffer precision.
        buffer_dtype=precision_dt,
    )

    #add hybrid sharded FSDP
    if conf["parallelism"["fsdp_size"] > 1 and conf["parallelism"]["simple_ddp_size"] > 1:
        model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add fully sharded FSDP
    elif conf["parallelism"["fsdp_size"] > 1 and conf["parallelism"]["simple_ddp_size"] == 1:
        model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add unsharded DDP
    else:
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )

    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )

    return model, epoch_start, loss_list

def main(device, local_rank):
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

    config_path = args.config
    pretrained_config_path = args.pretrained_config

    #pretrained_conf = parse_pretrained_config(pretrained_config_path)
    conf = parse_config(args)
    #TODO: Add function parse dataset specific options separately
    #TODO: Add function to parse model options separately, adding capability for different architecutres
    batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)

#2. Initialize model, optimizer, and scheduler
##############################################################################################################


    #Set up communication groups based on the parallelism settings chosen
    seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = conf["parallelism"]["data_par_size"], tensor_par_size = conf["parallelism"]["tensor_par_size"], seq_par_size = 1, fsdp_size = conf["parallelism"]["fsdp_size"], simple_ddp_size = conf["parallelism"]["simple_ddp_size"]) #TODO: Take out seq_par_group and seq_par_size

    model, epoch_start, loss_list = create_model(conf, device, local_rank, fsdp_group, simple_ddp_group, tensor_par_group)

    optimizer = configure_optimizer(model,conf["optimizer"]["lr"],conf["optimizer"]["beta_1"],conf["optimizer"]["beta_2"],conf["optimizer"]["weight_decay"])
    scheduler = configure_scheduler(optimizer,conf["scheduler"]["warmup_epochs"],conf["trainer"]["max_epoch"],conf["scheduler"]["warmup_start_lr"],conf["scheduler"]["eta_min"])

    #TODO: Add function for loading optimizer and scheduler from checkpoint
    if conf["trainer"]["resume_from_checkpoint"]:

        print("optimizer resume from checkpoint was set to True",flush=True)

        src_rank = world_rank - tensor_par_size * dist.get_rank(group=data_seq_ort_group)

        map_location = 'cpu' #TODO: Choose cpu or cuda+str
        #map_location = 'cuda:'+str(device)

        checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #TODO: Load loss_list and epoch_start in create_model loading from checkpoint rather than here
        loss_list = checkpoint['loss_list']
        epoch_start = checkpoint['epoch'] + 1
        del checkpoint

    if conf["grad_scaler"]["use_grad_scaler"]:
        scaler = ShardedGradScaler(init_scale=conf["grad_scaler"]["init_scale"], growth_interval=conf["grad_scaler"]["growth_interval"]
        min_scale = conf["grad_scaler"]["min_scale"]

#3. Initialize Dataloader
##############################################################################################################
    if conf["dataloader"]["type"] == "iterative_dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            data_module = NativePytorchDataModule(dict_root_dirs=conf["data"]dict_root_dirs,
                dict_start_idx = conf["dataloader"]["dict_start_idx"],
                dict_end_idx = conf["dataloader"]["dict_end_idx"],
                dict_buffer_sizes = conf["dataloader"]["dict_buffer_sizes"],
                dict_in_variables = conf["data"]["dict_in_variables"],
                num_channels_used = conf["data"]["num_channels"],
                batch_size = conf["dataloader"]["batch_size"],
                num_workers = conf["dataloader"]["num_workers"],
                pin_memory = conf["dataloader"]["pin_memory"],
                patch_size = conf["data"]["patch_size"],
                tile_size_x = conf["data"]["tile_size"][0], #TODO: move tile_size into one variable
                tile_size_y = conf["data"]["tile_size"][1],
                tile_size_z = conf["data"]["tile_size"][2] if len(conf["data"]["tile_size"]) == 3 else None,
                twoD = conf["data"]["twoD"],
                single_channel = False, #TODO: Take out single_channel option altogether
                return_label = True, #TODO: Add to config
                dataset_group_list = dataset_group_list,
                batches_per_rank_epoch = batches_per_rank_epoch,
                tile_overlap = conf["tiling"]["tile_overlap"],
                use_all_data = conf["tiling"]["use_all_data"],
                adaptive_patching = conf["ap"]["do_ap"],
                fixed_length = conf["ap"]["fixed_length"],
                separate_channels = conf["ap"]["separate_channels"],
                data_par_size = conf["parallelism"]["data_par"],
                dataset = conf["data"]["dataset"],
                imagenet_resize = conf["dataset_option"]["imagenet_resize"],
            ).to(device)

            data_module.setup()

            train_dataloader = data_module.train_dataloader()

    elif conf["dataloader"]["type"] == "dataloader":
        if dist.get_rank(tensor_par_group) == 0:
            #TODO: Loop over dict keys
            dkey_train = list(conf["data"]["dict_root_dirs"])[0]
            train_list = glob.glob(os.path.join(conf["data"]["dict_root_dirs"][dkey_train],'*.jpg'))

            train_data = conf["dataloader"]["dataset_module"](train_list, conf["data"]["dict_in_variables"][dkey_train], conf["data"]["tile_size"][0], adaptive_patching=conf["ap"]["do_ap"], fixed_length=conf["ap"]["fixed_length"], patch_size=conf["data"]["patch_size"], num_channels=conf["data"]["num_channels"], dataset=conf["data"]["dataset"])

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, num_replicas=conf["paralelism"]["data_par"],rank=world_rank)

            train_loader = DataLoader(dataset = train_data, sampler=train_sampler,num_workers=conf["dataloader"]["num_workers"], pin_memory=conf["dataloader"]["pin_memory"], batch_size=conf["dataloader"]["batch_size"], drop_last=True, collate_fn=lambda batch: conf["dataloader"]["collate_fn"](batch, adaptive_patching=conf["ap"]["do_ap"]))

#4. Training Loop
##############################################################################################################

    #Find iterations per epoch
    if conf["dataloader"]["type"] == "iterative_dataloader":
        iterations_per_epoch = 0
        for i,k in enumerate(batches_per_rank_epoch):
            if batches_per_rank_epoch[k] > iterations_per_epoch:
                iterations_per_epoch = batches_per_rank_epoch[k]
    #TODO: elif conf["dataloader"]["type"] == "standard":


    #TODO: Move to training loop function

    for epoch in range(epoch_start,conf["trainer"]["max_epochs"]):
        #Reset dataloader module every epoch to ensure all files get used
        if epoch != epoch_start:
            data_module.reset()
            train_dataloader = data_module.train_dataloader()

        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        epoch_accuracy = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)

        counter = 0
        for batch_idx, batch in enumerate(train_dataloader):
            counter = counter + 1
            if counter > iterations_per_epoch:
                print("A GPU ran out of data, moving to next epoch", flush=True)
                break

            if adaptive_patching:
                data, seq, seq_size, seq_pos, label, variables, _ = batch
                seq = seq.to(device)
                label = label.to(device)
                if separate_channels:
                    #TODO: Move seq_size and seq_pos to a single channel
                    seq_ps = None
                else:
                    seq_size = torch.squeeze(seq_size)
                    seq_size = seq_size.to(torch.float32)
                    seq_size = seq_size.to(device)
                    seq_pos = torch.squeeze(seq_pos)
                    seq_pos = seq_pos.to(torch.float32)
                    seq_pos = seq_pos.to(device)
                    seq_size = seq_size.unsqueeze(-1)
                    seq_ps = torch.concat([seq_size, seq_pos],dim=-1)
            else:
                data, label, variables, _ = batch
                data = data.to(device)
                label = label.to(device)
                seq_ps = None

            loss, output = training_step(data, variables, label, model, seq_ps)

            acc = (output.argmax(dim=1) == label).float().mean()

            epoch_accuracy += acc.detach()
            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch, "batch_idx", batch_idx, "it_loss ",loss, "it_acc", acc, flush=True)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        loss_list.append(epoch_loss)


        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss, "epoch_accuracy ", epoch_accuracy, flush=True)

        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()
        scheduler_states = scheduler.state_dict()


        #Alternating saving in to odd and even checkpoint file to avoid losing progress
        if world_rank == 0 and epoch % 2 == 0:
     
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                'scheduler_state_dict': scheduler_states,
                'loss_list' : loss_list,
                }, checkpoint_path+"/epoch_even.ckpt")

        if world_rank == 0 and epoch % 2 == 1:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                'scheduler_state_dict': scheduler_states,
                'loss_list' : loss_list,
                }, checkpoint_path+"/epoch_odd.ckpt")
     
        dist.barrier()
        del model_states
        del optimizer_states
        del scheduler_states

if __name__ == "__main__":

    main()

    
    main(device, local_rank)

    dist.destroy_process_group()
