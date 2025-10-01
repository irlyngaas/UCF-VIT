
import glob
import os
import sys
import random
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import time
import yaml
import math
from einops import rearrange
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

from UCF_VIT.fsdp.arch import MAE
from UCF_VIT.fsdp.building_blocks import Block
from UCF_VIT.utils.metrics import masked_mse
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, patchify, unpatchify, init_par_groups, calculate_load_balancing_on_the_fly, is_power_of_two
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn



def training_step_adaptive(seq, variables, net: MAE, patch_size, twoD, loss_fn, seq_ps):

    output, mask = net.forward(seq, variables, seq_ps)
    criterion = nn.MSELoss()
    target = rearrange(seq, 'b c s p -> b s (p c)')
    loss = criterion(output, target)

    return loss

def training_step(data, variables, net: MAE, patch_size, twoD, loss_fn):

    if loss_fn == "maskMSE":
        output, mask = net.forward(data, variables, None)
        criterion = masked_mse
        target = patchify(data, patch_size, twoD)
        loss = criterion(output,target,mask)

    else: #Default use full MSE
        output, _ = net.forward(data, variables, None)
        criterion = nn.MSELoss()
        target = patchify(data, patch_size, twoD)
        loss = criterion(output,target)

    return loss


def main(device, local_rank):
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    config_path = sys.argv[1]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if world_rank==0: 
        print(conf,flush=True)

    max_epochs = conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    gpu_type = conf['trainer']['gpu_type']

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    seq_par_size = conf['parallelism']['seq_par_size']

    lr = float(conf['model']['lr'])

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps = conf['model']['warmup_steps']

    max_steps = conf['model']['max_steps']

    warmup_start_lr = float(conf['model']['warmup_start_lr'])

    eta_min = float(conf['model']['eta_min'])

    loss_fn = conf['model']['loss_fn']

    default_vars =  conf['model']['net']['init_args']['default_vars']

    tile_size = conf['model']['net']['init_args']['tile_size']

    patch_size = conf['model']['net']['init_args']['patch_size']
 
    emb_dim = conf['model']['net']['init_args']['embed_dim']

    depth = conf['model']['net']['init_args']['depth']

    num_heads = conf['model']['net']['init_args']['num_heads']
    
    decoder_embed_dim = conf['model']['net']['init_args']['decoder_embed_dim']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    decoder_num_heads = conf['model']['net']['init_args']['decoder_num_heads']

    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    mlp_ratio_decoder = conf['model']['net']['init_args']['mlp_ratio_decoder']

    drop_path = conf['model']['net']['init_args']['drop_path']

    mask_ratio = conf['model']['net']['init_args']['mask_ratio']

    linear_decoder = conf['model']['net']['init_args']['linear_decoder']

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    adaptive_patching = conf['model']['net']['init_args']['adaptive_patching']

    if adaptive_patching:
        fixed_length = conf['model']['net']['init_args']['fixed_length']
        separate_channels = conf['model']['net']['init_args']['separate_channels']
        use_adaptive_pos_emb = conf['model']['net']['init_args']['use_adaptive_pos_emb']
        if separate_channels:
            assert not use_adaptive_pos_emb, "Capability to use separate channels and adaptive pos_emb not implemented yet"
    else:
        fixed_length = None
        separate_channels = None
        use_adaptive_pos_emb = None

    use_grad_scaler = conf['model']['use_grad_scaler']

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct", "imagenet"], "This training script only supports basic_ct and imagenet datasets"

    if dataset == "imagenet":
        assert twoD, "twoD must be True if using imagenet"

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_start_idx = conf['data']['dict_start_idx']

    dict_end_idx = conf['data']['dict_end_idx']

    dict_buffer_sizes = conf['data']['dict_buffer_sizes']

    num_channels_used = conf['data']['num_channels_used']

    dict_in_variables = conf['data']['dict_in_variables']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    single_channel = conf['data']['single_channel']

    tile_overlap = conf['data']['tile_overlap']

    use_all_data = conf['data']['use_all_data']

    #Datset specific options
    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['imagenet_resize']
    else:
        imagenet_resize = None

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]
    if dataset == "imagenet":
        tile_size_z = None
    else:
        tile_size_z = tile_size[2]

    
    assert (tile_size_x%patch_size)==0, "tile_size_x % patch_size must be 0"
    assert (tile_size_y%patch_size)==0, "tile_size_y % patch_size must be 0"
    if dataset != "imagenet":
        assert (tile_size_z%patch_size)==0, "tile_size_z % patch_size must be 0"

    data_par_size = fsdp_size * simple_ddp_size
    assert seq_par_size == 1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) == 0, "model heads % tensor parallel size must be 0"
    assert (decoder_num_heads % tensor_par_size) == 0, "decoder model heads % tensor parallel size must be 0"

    if adaptive_patching:
        x_p2 = is_power_of_two(tile_size_x)
        assert x_p2, "tile_size_x must be a power of 2"
        y_p2 = is_power_of_two(tile_size_y)
        assert y_p2, "tile_size_y must be a power of 2"
        if dataset != "imagenet":
            z_p2 = is_power_of_two(tile_size_z)
            assert z_p2, "tile_size_z must be a power of 2"

        if twoD:
            assert fixed_length % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
        else:
            sqrt_len=int(np.rint(math.pow(fixed_length,1/3)))
            assert fixed_length % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"

    auto_load_balancing = conf['load_balancing']['auto_load_balancing']
    if auto_load_balancing:
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(config_path, data_par_size, batch_size)
    else:
        batches_per_rank_epoch = conf['load_balancing']['batches_per_rank_epoch']
        dataset_group_list = conf['load_balancing']['dataset_group_list']

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
    if data_type == "bfloat16":
        if gpu_type == "amd":
            FusedAttn_option = FusedAttn.CK
        elif gpu_type == "nvidia":
            FusedAttn_option = FusedAttn.FLASH
        else:
            print("Invalid gpu_type used, reverting to using default FMHA")
            FusedAttn_option = FusedAttn.DEFAULT
    else:
        if use_fused_attn():
            FusedAttn_option = FusedAttn.DEFAULT
        else:
            FusedAttn_option = FusedAttn.NONE

    #Find correct in_chans to use
    if single_channel:
        max_channels = 1
    else:
        max_channels = 1
        for i,k in enumerate(num_channels_used):
            if num_channels_used[k] > 1:
                max_channels = num_channels_used[k]

    seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size)

    model = MAE(
        img_size=tile_size,
        patch_size=patch_size,
        in_chans=max_channels,
        embed_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_depth=decoder_depth,
        decoder_embed_dim=decoder_embed_dim, 
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path,
        mask_ratio=mask_ratio,
        linear_decoder=linear_decoder,
        twoD=twoD,
        mlp_ratio_decoder=mlp_ratio_decoder,
        default_vars=default_vars,
        single_channel=single_channel,
        use_varemb=use_varemb,
        adaptive_patching=adaptive_patching,
        fixed_length=fixed_length,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
        FusedAttn_option=FusedAttn_option,
        use_adaptive_pos_emb=use_adaptive_pos_emb,
        class_token=False,
        weight_init='skip',
    ).to(device)

    if not resume_from_checkpoint: #train from scratch

        epoch_start = 0
        loss_list = []
        if world_rank==0:       
            print("resume from checkpoint was set to False. Pretrain from scratch.",flush=True)

        if world_rank==0:

            #Check whether the specified checkpointing path exists or not
            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                #Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")

            #Save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
            init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

            torch.save(init_model_dict,
                    checkpoint_path+'/initial_'+str(dist.get_rank())+'.pth')

            del init_model_dict

        dist.barrier()

        if world_rank!=0 and world_rank <tensor_par_size:


           #load initial model weights and synchronize model weights that are not in the training block among sequence parallel GPUs
           src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

           map_location = 'cpu'
           #map_location = 'cuda:'+str(device)
           model.load_state_dict(torch.load(checkpoint_path+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)
    else:  
        if world_rank < tensor_par_size:
            if os.path.exists(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt"):
                print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

                print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

                map_location = 'cpu'
                #map_location = 'cuda:'+str(device)

                checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
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

    if data_type == "float32":
        precision_dt = torch.float32
    elif data_type == "bfloat16":
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
    if fsdp_size > 1 and simple_ddp_size > 1:
        model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add fully sharded FSDP
    elif fsdp_size > 1 and simple_ddp_size == 1:
        model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )
    #add unsharded DDP
    else:
        model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, mixed_precision=bfloatPolicy, forward_prefetch=True, limit_all_gathers = False )

    check_fn = lambda submodule: isinstance(submodule, Block)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
    )

    optimizer = configure_optimizer(model,lr,beta_1,beta_2,weight_decay)
    scheduler = configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min)

    if resume_from_checkpoint:

        print("optimizer resume from checkpoint was set to True",flush=True)

        src_rank = world_rank - tensor_par_size * dist.get_rank(group=data_seq_ort_group)

        map_location = 'cpu'
        #map_location = 'cuda:'+str(device)

        checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(src_rank)+".ckpt",map_location=map_location)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_list = checkpoint['loss_list']
        epoch_start = checkpoint['epoch'] + 1
        del checkpoint

    if use_grad_scaler:
        scaler = ShardedGradScaler(init_scale=8192, growth_interval=100)
        min_scale= 128


#3. Initialize Dataloader
##############################################################################################################
    if dist.get_rank(tensor_par_group) == 0:
        data_module = NativePytorchDataModule(dict_root_dirs=dict_root_dirs,
            dict_start_idx=dict_start_idx,
            dict_end_idx=dict_end_idx,
            dict_buffer_sizes=dict_buffer_sizes,
            dict_in_variables=dict_in_variables,
            num_channels_used = num_channels_used,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            patch_size = patch_size,
            tile_size_x = tile_size_x,
            tile_size_y = tile_size_y,
            tile_size_z = tile_size_z,
            twoD = twoD,
            single_channel = single_channel,
            return_label = False,
            dataset_group_list = dataset_group_list,
            batches_per_rank_epoch = batches_per_rank_epoch,
            tile_overlap = tile_overlap,
            use_all_data = use_all_data,
            adaptive_patching = adaptive_patching,
            fixed_length = fixed_length,
            separate_channels = separate_channels,
            data_par_size = data_par_size,
            ddp_group = ddp_group,
            dataset = dataset,
            imagenet_resize = imagenet_resize,
        ).to(device)

        data_module.setup()

        train_dataloader = data_module.train_dataloader()

#4. Training Loop
##############################################################################################################
    #Find max batches
    iterations_per_epoch = 0
    for i,k in enumerate(batches_per_rank_epoch):
        if batches_per_rank_epoch[k] > iterations_per_epoch:
            iterations_per_epoch = batches_per_rank_epoch[k]

    for epoch in range(epoch_start,max_epochs):
        #Reset dataloader module every epoch to ensure all files get used
        if epoch != epoch_start:
            if dist.get_rank(tensor_par_group) == 0:
                data_module.reset()
                train_dataloader = data_module.train_dataloader()

        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)

        if dist.get_rank(tensor_par_group) == 0:
            it_loader = iter(train_dataloader)

        counter = 0
        while counter < iterations_per_epoch:
            counter = counter + 1
            if adaptive_patching:
                if tensor_par_size > 1:
                    if dist.get_rank(tensor_par_group) == 0:
                        #seq, variables, dict_key = next(it_loader)
                        data, seq, seq_size, seq_pos, variables, _ = next(it_loader)
                        seq = seq.to(precision_dt)
                        seq = seq.to(device)
                        if dataset != "imagenet":
                            dict_key_len = torch.tensor(len(dict_key)).to(device)
                        else:
                            dict_key = "imagenet"
                    else:
                        if dataset != "imagenet":
                            dict_key_len = torch.tensor(0).to(device)
                        else: 
                            dict_key = "imagenet"

                    if dataset != "imagenet":
                        dist.broadcast(dict_key_len, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group = tensor_par_group)
                        if dist.get_rank(tensor_par_group) != 0:
                            dict_key = [None] * dict_key_len.item()
                        dist.broadcast_object_list(dict_key, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                        if dist.get_rank(tensor_par_group) != 0:
                            dict_key = ''.join(dict_key)

                    if dist.get_rank(tensor_par_group) != 0:
                        if twoD:
                            seq = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, patch_size*patch_size, dtype=precision_dt).to(device)
                            if separate_channels:
                                seq_size = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, dtype=precision_dt).to(device)
                                seq_pos = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, 2, dtype=precision_dt).to(device)
                            else:
                                seq_size = torch.zeros(batch_size, 1, fixed_length, 1, dtype=precision_dt).to(device)
                                seq_pos = torch.zeros(batch_size, 1, fixed_length, 1, 1, dtype=precision_dt).to(device)
                        else:
                            seq = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, patch_size*patch_size*patch_size, dtype=precision_dt).to(device)
                            if separate_channels:
                                seq_size = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, dtype=precision_dt).to(device)
                                seq_pos = torch.zeros(batch_size, num_channels_used[dict_key], fixed_length, 3, dtype=precision_dt).to(device)
                            else:
                                seq_size = torch.zeros(batch_size, 1, fixed_length, 1, dtype=precision_dt).to(device)
                                seq_pos = torch.zeros(batch_size, 1, fixed_length, 1, 1, 1, dtype=precision_dt).to(device)
                        variables = [None] * num_channels_used[dict_key]
                    dist.broadcast(seq, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast(seq_size, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast(seq_pos, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                else: #Avoid unnecesary broadcasts if not using tensor parallelism
                    #seq, variables, _ = next(it_loader)
                    data, seq, seq_size, seq_pos, variables, _ = next(it_loader)
                    seq = seq.to(precision_dt)
                    seq = seq.to(device)

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

                loss = training_step_adaptive(seq, variables, model, patch_size, twoD, loss_fn, seq_ps)

            else:
                if tensor_par_size > 1:
                    if dist.get_rank(tensor_par_group) == 0:
                        data, variables, dict_key = next(it_loader)
                        data = data.to(precision_dt)
                        data = data.to(device)
                        if dataset != "imagenet":
                            dict_key_len = torch.tensor(len(dict_key)).to(device)
                        else:
                            dict_key = "imagenet"
                    else:
                        if dataset != "imagenet":
                            dict_key_len = torch.tensor(0).to(device)
                        else:
                            dict_key = "imagenet"

                    if dataset != "imagenet":
                        dist.broadcast(dict_key_len, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group = tensor_par_group)

                        if dist.get_rank(tensor_par_group) != 0:
                            dict_key = [None] * dict_key_len.item()
                        dist.broadcast_object_list(dict_key, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                        if dist.get_rank(tensor_par_group) != 0:
                            dict_key = ''.join(dict_key)

                    if dist.get_rank(tensor_par_group) != 0:
                        if twoD:
                            data = torch.zeros(batch_size, num_channels_used[dict_key], tile_size_x, tile_size_y, dtype=precision_dt).to(device)
                        else:
                            data = torch.zeros(batch_size, num_channels_used[dict_key], tile_size_x, tile_size_y, tile_size_z, dtype=precision_dt).to(device)
                        variables = [None] * num_channels_used[dict_key]
                    dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                else: #Avoid unnecesary broadcasts if not using tensor parallelism
                    data, variables, _ = next(it_loader)
                    data = data.to(precision_dt)
                    data = data.to(device)

                loss = training_step(data, variables, model, patch_size, twoD, loss_fn)

            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",counter,"world_rank",world_rank,"it_loss ",loss,flush=True)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scaler._scale < min_scale:
                    scaler._scale = torch.tensor(min_scale).to(scaler._scale)
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
        
        loss_list.append(epoch_loss)

        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)

        model_states = model.state_dict()
        optimizer_states = optimizer.state_dict()
        scheduler_states = scheduler.state_dict()


        #Alternating saving in to odd and even checkpoint file to avoid losing progress
        if epoch % 2 == 0:
     
            if world_rank < tensor_par_size:
                torch.save({
                    'epoch': epoch_start+max_epochs,
                    'model_state_dict': model_states,
                    'optimizer_state_dict': optimizer_states,
                    'scheduler_state_dict': scheduler_states,
                    'loss_list' : loss_list,
                    }, checkpoint_path+"/"+checkpoint_filename+"_even_rank_"+str(world_rank)+".ckpt")

        if epoch % 2 == 1:
            if world_rank < tensor_par_size:
                torch.save({
                    'epoch': epoch_start+max_epochs,
                    'model_state_dict': model_states,
                    'optimizer_state_dict': optimizer_states,
                    'scheduler_state_dict': scheduler_states,
                    'loss_list' : loss_list,
                    }, checkpoint_path+"/"+checkpoint_filename+"_odd_rank_"+str(world_rank)+".ckpt")

        dist.barrier()
        del model_states
        del optimizer_states
        del scheduler_states

if __name__ == "__main__":

    if len(sys.argv) > 2:
        LAUNCHER = sys.argv[2]
    else:
        LAUNCHER = None

    if LAUNCHER == "MPI":
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

    else:#elif LAUNCHER == "SLURM":

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
    
    main(device, local_rank)

    dist.destroy_process_group()
