
import glob
import os
import sys
import random
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import save_image
import time
from collections import OrderedDict
import yaml
from torch.nn import Sequential
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
import functools
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from timm.layers import use_fused_attn

from functools import partial

from UCF_VIT.fsdp.arch import UNETR, MAE
from UCF_VIT.fsdp.building_blocks import Block
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, init_par_groups
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn



def training_step(data, label, variables, net: UNETR, patch_size, twoD):

    output = net.forward(data, variables)

    criterion = nn.MSELoss()
    loss = criterion(output,label)

    return loss, output

#def test_step(data, label, variables, net: UNETR, tile_size, twoD):
#    if twoD:
#        inf_size = [tile_size[0], tile_size[1]]
#    else:
#        inf_size = [tile_size[0], tile_size[1], tile_size[2]]
#    model_inferer = partial(
#                        sliding_window_inference,
#                        roi_size=inf_size,
#                        #sw_batch_size=args.sw_batch_size,
#                        sw_batch_size=1,
#                        predictor=net,
#                        #overlap=args.infer_overlap,
#                        overlap=0.5,
#                        variables = variables,
#                    )
#
#    output = model_inferer(data)
#
#    return output


def main():
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 

    config_path = sys.argv[1]

    print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    print(conf,flush=True)

    max_epochs=conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    checkpoint_path =conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    seq_par_size = conf['parallelism']['seq_par_size']

    use_pretrained_mae_model = conf['trainer']['use_pretrained_mae_model']

    mae_checkpoint_path = conf['trainer']['mae_checkpoint_path']

    mae_checkpoint_filename = conf['trainer']['mae_checkpoint_filename']
 
    lr = float(conf['model']['lr'])

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps =  conf['model']['warmup_steps']

    max_steps =  conf['model']['max_steps']

    warmup_start_lr =  float(conf['model']['warmup_start_lr'])

    eta_min =  float(conf['model']['eta_min'])

    default_vars =  conf['model']['net']['init_args']['default_vars']

    tile_size =  conf['model']['net']['init_args']['tile_size']

    patch_size =  conf['model']['net']['init_args']['patch_size']
 
    emb_dim =  conf['model']['net']['init_args']['embed_dim']

    depth =  conf['model']['net']['init_args']['depth']

    num_heads = conf['model']['net']['init_args']['num_heads']

    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    drop_path = conf['model']['net']['init_args']['drop_path']

    linear_decoder = conf['model']['net']['init_args']['linear_decoder']

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    feature_size = conf['model']['net']['init_args']['feature_size']

    skip_connection = conf['model']['net']['init_args']['skip_connection']

    dataset = conf['data']['dataset']
    assert dataset in ["sst"], "This training script only supports sst dataset"

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_start_idx = conf['data']['dict_start_idx']

    dict_end_idx = conf['data']['dict_end_idx']

    dict_buffer_sizes = conf['data']['dict_buffer_sizes']

    num_channels_available = conf['data']['num_channels_available']

    num_channels_used = conf['data']['num_channels_used']

    dict_in_variables = conf['data']['dict_in_variables']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    single_channel = conf['data']['single_channel']

    num_classes = conf['data']['num_classes']

    tile_overlap = conf['data']['tile_overlap']

    use_all_data = conf['data']['use_all_data']

    assert not use_all_data, "Don't use all_data=True, need to make some changes within the code for this option to work"

    batches_per_rank_epoch = conf['load_balancing']['batches_per_rank_epoch']

    dataset_group_list = conf['load_balancing']['dataset_group_list']

    #These configs need only for finetuning with pre-trained MAE model
    decoder_embed_dim = conf['model']['net']['init_args']['decoder_embed_dim']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    decoder_num_heads = conf['model']['net']['init_args']['decoder_num_heads']

    mlp_ratio_decoder = conf['model']['net']['init_args']['mlp_ratio_decoder']

    mask_ratio = conf['model']['net']['init_args']['mask_ratio']

    #use_scaler = conf['model']['net']['init_args']['use_scaler']
    use_scaler = True

    #Datset specific options
    if dataset == "sst":
        nx = conf['dataset_options']['nx']

        ny = conf['dataset_options']['ny']

        nz = conf['dataset_options']['nz']

        nx_skip = conf['dataset_options']['nx_skip']

        ny_skip = conf['dataset_options']['ny_skip']

        nz_skip = conf['dataset_options']['nz_skip']
        
        dict_out_variables = conf['dataset_options']['dict_out_variables']

        chunk_size = conf['dataset_options']['chunk_size']

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
    if data_type == "bfloat16":
        FusedAttn_option = FusedAttn.CK
    else:
        if use_fused_attn():
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.NONE

    device = torch.cuda.current_device()
    #Find correct in_chans to use
    if single_channel:
        max_channels = 1
    else:
        max_channels = 1
        for i,k in enumerate(num_channels_used):
            if num_channels_used[k] > 1:
                max_channels = num_channels_used[k]

    model = UNETR(
        img_size=tile_size,
        patch_size=patch_size,
        in_chans=max_channels,
        num_classes=num_classes,
        embed_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path,
        linear_decoder=linear_decoder,
        twoD=twoD,
        default_vars=default_vars,
        feature_size=feature_size,
        skip_connection=skip_connection,
        single_channel=single_channel,
        use_varemb=use_varemb,
        tensor_par_size=tensor_par_size,
        #tensor_par_group=tensor_par_group,
        FusedAttn_option=FusedAttn_option,
        class_token=False,
        weight_init='skip',
    ).to(device)

    total_params = torch.tensor(0,dtype=torch.long)
    decoder_params = torch.tensor(0,dtype=torch.long)
    decoder_attn_params = torch.tensor(0,dtype=torch.long)
    encoder_params = torch.tensor(0,dtype=torch.long)
    encoder_attn_params = torch.tensor(0,dtype=torch.long)
    params_per_gpu = torch.tensor(0,dtype=torch.long)

    for name, param in model.named_parameters():
        print("parameter name ",name," requires_gradient ",param.requires_grad, "size",param.shape, flush=True)

        params_per_gpu = params_per_gpu + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)

        if 'decoder' not in name:
            encoder_params = encoder_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)
            if 'attn' in name:
                encoder_attn_params = encoder_attn_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)
        else:
            decoder_params = decoder_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)
            if 'attn' in name:
                decoder_attn_params = decoder_attn_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)

        #if 'attn' not in name and 'mlp' not in name and 'var_agg' not in name:
        #    total_params = total_params + torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)
        #else:
        #    total_params = total_params + tensor_par_size * torch.prod(torch.tensor(list(param.size()),dtype=torch.int),dtype=torch.long)

    print("Attn Params Per Encoder Block:", encoder_attn_params/depth)
    print("Attn Params Per Decoder Block:", decoder_attn_params/decoder_depth)
    print("Params Per Encoder Attention Head:", encoder_attn_params/depth/num_heads)
    print("Params Per Decoder Attention Head:", encoder_attn_params/decoder_depth/decoder_num_heads)
    print("Encoder Params:",encoder_params,flush=True)
    print("Decoder Params:",decoder_params,flush=True)
    #print("total_params before FSDP",total_params,"params_per_gpu",params_per_gpu,flush=True)
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024),flush=True)

if __name__ == "__main__":
    main()
