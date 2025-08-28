import glob
import os
import sys
import copy
import random
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.utils import save_image
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

# sys.path.insert(0, '/home/ziabariak/git/UCF-VIT/src')

from UCF_VIT.fsdp.arch import DiffusionVIT
from UCF_VIT.fsdp.building_blocks import Block
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, unpatchify, init_par_groups, calculate_load_balancing_on_the_fly
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler#, sample_images#,save_intermediate_data

# get all plottings
sys.path.append(os.path.expanduser('~/git/UCF-VIT/utils'))
from plotting_generations import *
### training
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

    gpu_type = conf['trainer']['gpu_type']

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    inference_path = conf['trainer']['inference_path']

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

    num_time_steps = conf['model']['net']['init_args']['num_time_steps']

    use_grad_scaler = conf['model']['use_grad_scaler']

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct", "imagenet", "xct"], "This training script only supports basic_ct, imagenet, or xct datasets"


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

                #map_location = 'cuda:'+str(device)
                map_location = 'cpu'

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

        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'

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

    isExist = os.path.exists(inference_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(inference_path, exist_ok=True)
        print("The new inference directory is created!")

    #Find max batches
    iterations_per_epoch = 0
    for i,k in enumerate(batches_per_rank_epoch):
        if batches_per_rank_epoch[k] > iterations_per_epoch:
            iterations_per_epoch = batches_per_rank_epoch[k]

    ### to get the best loss saved
    fid_scores = []
    fid_epochs = []
    fid_eval_period = 10
    
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    max_patience = 50000
    patience = 2000
    lr_decay_count = 0
    save_period = 1 # this allows for us ensuring the data is plotted and saved correctly, and then we cahnge it to 50 or some other number for less frequent saving!
    save_period_main = 50
    decay_factor = 0.9
    patience_inc_rate = 1.25 
        
    best_model_state = None
    best_optimizer_state = None
    best_scheduler_state = None

    for epoch in range(epoch_start,max_epochs+1):
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
            print("Starting epoch ",epoch,flush=True)

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

                if epoch % 100 == 0:
                    if world_rank==1:
                        print("epoch: ",epoch,"batch_idx",counter,"world_rank",world_rank,"it_loss ",loss,flush=True)
                    if world_rank==2:
                        print("epoch: ",epoch,"batch_idx",counter,"world_rank",world_rank,"it_loss ",loss,flush=True)
                    if world_rank==3:
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
        
        # if epoch % fid_eval_period == 0 and world_rank == 0:
        #     model.eval()
        #     for var in default_vars:
        #         fid = save_intermediate_data_with_fid(model, var, device, tile_size, precision_dt, patch_size,
        #                         epoch=epoch, num_samples=4, twoD=twoD, save_path=inference_path,
        #                         num_time_steps=num_time_steps,
        #                         test_volume_path="/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/single_validation/XCT_Concrete/XCT_Concrete_256x256_Z00730_pixel30.8985um_Pair_97.npy") # ,downscale=1#add in config
        #     model.train()
        #     if fid is not None and np.isfinite(fid):
        #         log_and_plot_fid(fid, epoch, fid_scores, fid_epochs, inference_path)
        #     else:
        #         print(f"FID computation failed at epoch {epoch}")
        
        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)
            if epoch % 100 == 0:
                plotLoss(loss_list, save_path=os.path.join(checkpoint_path, f'loss_N{simple_ddp_size//8}_BS{batch_size}_PS{patch_size}_ED{emb_dim}.png'))
        if world_rank==1:
            if epoch % 100 == 0:
                print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)
                plotLoss(loss_list, save_path=os.path.join(checkpoint_path, f'loss_N{simple_ddp_size//8}_BS{batch_size}_PS{patch_size}_ED{emb_dim}_rank1.png'))

        if ((epoch==1) or (epoch % 50 == 0)) and (dist.get_rank(tensor_par_group) == 0):
            # grab a small batch from the current loader (only this rank has it)
            it_eval = iter(train_dataloader)
            x_eval, variables_eval, _ = next(it_eval)
            x_eval = x_eval.to(precision_dt).to(device)

            # plotPerformance
            plotPerformance(
                model=model,
                device=device,
                x=x_eval,
                variables=variables_eval,
                num_time_steps=num_time_steps,
                patch_size=patch_size,
                twoD=twoD,
                scheduler=ddpm_scheduler,   # your DDPM_Scheduler from above
                epoch=epoch,
                savefol=inference_path,     # where to save the figure
                precision_dt=precision_dt,
                Ntimes=9
            )

            # plotPerformance
            plotPerformanceImgs(
                model=model,
                device=device,
                x=x_eval,
                variables=variables_eval,
                num_time_steps=num_time_steps,
                patch_size=patch_size,
                twoD=twoD,
                scheduler=ddpm_scheduler,   # your DDPM_Scheduler from above
                epoch=epoch,
                savefol=inference_path,     # where to save the figure
                precision_dt=precision_dt,
                Ntimes=9
            )


        # Track best model independently
        if epoch_loss.item() < best_loss:
            best_loss = epoch_loss.item()
            best_epoch = epoch
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        else:
            epochs_without_improvement += 1

            # Reduce LR if no improvement for `patience` epochs
            if epochs_without_improvement >= patience:
                
                # model.load_state_dict(best_model_state)
                # optimizer.load_state_dict(best_optimizer_state)
                # scheduler.load_state_dict(best_scheduler_state)
                # if world_rank == 0:
                #     print(f"[Epoch {epoch}] Reloading best model from epoch {best_epoch} after LR decay.")
                    
                lr_decay_count += 1
                for i, param_group in enumerate(optimizer.param_groups):
                    if world_rank == 0:
                        print(f"LR for param group {i} was: {param_group['lr']:.2e}")
                    param_group['lr'] *= decay_factor
                    if world_rank == 0:
                        print(f"LR for param group {i} updated to: {param_group['lr']:.2e}")
                    # Log patience change (optional)
                    if world_rank == 0:
                        print(f"Increasing patience: old={patience}", flush=True)

                    # Update patience
                    patience = min(int(patience * patience_inc_rate),max_patience)
                epochs_without_improvement = 0

        # Save the best model periodically
        if epoch > 0 and epoch % save_period == 0 and world_rank < tensor_par_size:
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': best_optimizer_state,
                'scheduler_state_dict': best_scheduler_state,
                'loss_list': loss_list,
            }, f"{checkpoint_path}/{checkpoint_filename}_BEST_{epoch}_rank_{world_rank}.ckpt")

            model.load_state_dict(best_model_state)

            for var in default_vars:
                model.eval()
                sample_images(model, var, device, tile_size, precision_dt, patch_size,
                            epoch=epoch, num_samples=5, twoD=twoD, save_path=inference_path,
                            num_time_steps=num_time_steps)
                model.train()
            if save_period<save_period_main:
                save_period = save_period_main

        dist.barrier()

    # Final save of best model
    if world_rank == 0:
        print(f"Training completed. Best loss: {best_loss:.6f} at epoch {best_epoch}")

    if best_model_state is not None and world_rank < tensor_par_size:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'scheduler_state_dict': best_scheduler_state,
            'loss_list': loss_list,
        }, checkpoint_path+"/"+checkpoint_filename+"_FINALBEST_"+str(best_epoch)+"_rank_"+str(world_rank)+".ckpt".format(best_epoch)) 

        model.load_state_dict(best_model_state)

        for var in default_vars:
            model.eval()
            sample_images(model, var, device, tile_size, precision_dt, patch_size,
                                epoch=best_epoch, num_samples=10, twoD=twoD, save_path=inference_path,
                                num_time_steps=num_time_steps)
            model.train()
            # save_intermediate_data(model, var, device, tile_size, precision_dt, patch_size,
            #                     epoch=best_epoch, num_samples=2, twoD=twoD, save_path=inference_path,
            #                     num_time_steps=num_time_steps)




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


