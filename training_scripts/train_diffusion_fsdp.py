
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
import time
import yaml
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

from UCF_VIT.fsdp.arch import DiffusionVIT
from UCF_VIT.fsdp.building_blocks import Block
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, unpatchify, init_par_groups
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler

def training_step(data, variables, t, e, net: DiffusionVIT, patch_size, twoD, loss_fn):


    output = net.forward(data, t, variables)
    output = unpatchify(output, data, patch_size, twoD)
    criterion = nn.MSELoss()
    loss = criterion(output,e)

    return loss


def main(device):
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()

    config_path = sys.argv[1]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if world_rank==0: 
        print(conf,flush=True)

    max_epochs = conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    seq_par_size = conf['parallelism']['seq_par_size']

    cpu_offload_flag = conf['parallelism']['cpu_offloading']
 
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

    linear_decoder = conf['model']['net']['init_args']['linear_decoder'] 

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    adaptive_patching = conf['model']['net']['init_args']['adaptive_patching']

    assert not adaptive_patching, "Adaptive Patching not implemented for DiffusionVIT yet"

    if adaptive_patching:
        fixed_length = conf['model']['net']['init_args']['fixed_length']
        separate_channels = conf['model']['net']['init_args']['separate_channels']

        if not twoD:
            assert not separate_channels, "Adaptive Patching in 3D with multiple channels (non-separated) is not currently implemented"
    else:
        fixed_length = None
        separate_channels = None

    #use_scaler = conf['model']['net']['init_args']['use_scaler']
    use_scaler = True

    num_time_steps = conf['model']['net']['init_args']['num_time_steps']

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct", "imagenet"], "This training script only supports basic_ct, imagenet datasets"


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

    tile_overlap = conf['data']['tile_overlap']

    use_all_data = conf['data']['use_all_data']

    batches_per_rank_epoch = conf['load_balancing']['batches_per_rank_epoch']

    dataset_group_list = conf['load_balancing']['dataset_group_list']

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

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
    if data_type == "bfloat16":
        FusedAttn_option = FusedAttn.CK
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

    ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)
    model = DiffusionVIT(
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
        time_steps=num_time_steps,
        class_token=False,
        weight_init='skip',
    ).to(device)

    if not resume_from_checkpoint: #train from scratch
        epoch_start = 0
        loss_list = []
        if world_rank==0:       
            print("resume from checkpoint was set to False. Pretrain from scratch.",flush=True)

        if world_rank==0:

            # Check whether the specified checkpointing path exists or not

            isExist = os.path.exists(checkpoint_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(checkpoint_path)
                print("The new checkpoint directory is created!")

            #save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
            init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}

            print("rank",dist.get_rank(),"init_model_dict.keys()",init_model_dict.keys(),flush=True)

            torch.save(init_model_dict,
                    checkpoint_path+'/initial_'+str(dist.get_rank())+'.pth')

            print("rank", dist.get_rank(),"after torch.save for initial",flush=True)

            del init_model_dict

        dist.barrier()

        if world_rank!=0 and world_rank <tensor_par_size:


           #load initial model weights and synchronize model weights that are not in the training block among sequence parallel GPUs
           src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

           print("rank",dist.get_rank(),"src_rank",src_rank,flush=True)

           map_location = 'cpu'
           #map_location = 'cuda:'+str(device)
           model.load_state_dict(torch.load(checkpoint_path+'/initial_'+str(0)+'.pth',map_location=map_location),strict=False)
    else:  
        if world_rank< tensor_par_size:
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

    if use_scaler:
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
            num_channels_available = num_channels_available,
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
        with torch.autograd.set_detect_anomaly(False):
            while counter < iterations_per_epoch:
                counter = counter + 1
                if tensor_par_size > 1:
                    if dist.get_rank(tensor_par_group) == 0:
                        data, variables, dict_key = next(it_loader)
                        data = data.to(precision_dt)
                        data = data.to(device)
                        if dataset != "imagenet":
                            dict_key_len = torch.tensor(len(dict_key)).to(device)
                        else:
                            dict_key = "imagenet"
                        t = torch.randint(0,num_time_steps,(batch_size,))
                        e = torch.randn_like(data, requires_grad=False)
                        if twoD:
                            a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1).to(precision_dt).to(device)
                        else:
                            a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1,1).to(precision_dt).to(device)
                        t = t.to(device)
                        data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)
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
                        t = torch.zeros(batch_size, dtype=torch.int).to(device)
                        e = torch.zeros_like(data, requires_grad=False)
                    dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    dist.broadcast(t, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
                    t = t.to('cpu')
                    dist.broadcast(e, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

                else: #Avoid unnecesary broadcasts if not using tensor parallelism
                    data, variables, _ = next(it_loader)
                    data = data.to(precision_dt)
                    data = data.to(device)
                    t = torch.randint(0,num_time_steps,(batch_size,))
                    e = torch.randn_like(data, requires_grad=False)
                    if twoD:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1).to(precision_dt).to(device)
                    else:
                        a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1,1).to(precision_dt).to(device)
                    data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)

                loss = training_step(data, variables, t, e, model, patch_size, twoD, loss_fn)

                epoch_loss += loss.detach()
    
                if world_rank==0:
                    print("epoch: ",epoch,"batch_idx",counter,"world_rank",world_rank,"it_loss ",loss,flush=True)
    
                if use_scaler:
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
                    'epoch': epoch,
                    'model_state_dict': model_states,
                    'optimizer_state_dict': optimizer_states,
                    'scheduler_state_dict': scheduler_states,
                    'loss_list' : loss_list,
                    }, checkpoint_path+"/"+checkpoint_filename+"_even_rank_"+str(world_rank)+".ckpt")

        if epoch % 2 == 1:

            if world_rank < tensor_par_size:
                torch.save({
                    'epoch': epoch,
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

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()



    #torch.backends.cudnn.benchmark = True

    dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

#    initialize_process()

    print("Using dist.init_process_group. world_size ",world_size,flush=True)
    
    main(device)

    dist.destroy_process_group()
