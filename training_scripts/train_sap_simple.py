
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
import math
from collections import OrderedDict
from timm.layers import use_fused_attn

from UCF_VIT.simple.arch import SAP, MAE
from UCF_VIT.utils.metrics import DiceBLoss
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, interpolate_pos_embed_adaptive, calculate_load_balancing_on_the_fly, is_power_of_two
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn

#TODO: Add qdt_list back for visualization
def training_step_adaptive(seq, seq_label, variables, net: SAP, patch_size, twoD, num_classes, sqrt_len):

    #seq = torch.reshape(seq, shape=(-1,1,patch_size*sqrt_len, patch_size*sqrt_len))
    if twoD:
        seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len))
    else:
        seq_label = torch.reshape(seq_label, shape=(-1,num_classes,patch_size*sqrt_len, patch_size*sqrt_len, patch_size*sqrt_len))
        
    output = net.forward(seq, variables)
    criterion = DiceBLoss(num_class=num_classes)
    loss = criterion(output,seq_label)
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

    assert data_type == "float32", "Only float32 training supported in this training script"

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']

    use_pretrained_mae_model = conf['trainer']['use_pretrained_mae_model']

    mae_checkpoint_path = conf['trainer']['mae_checkpoint_path']

    mae_checkpoint_filename = conf['trainer']['mae_checkpoint_filename']
 
    lr = float(conf['model']['lr'])

    beta_1 = float(conf['model']['beta_1'])

    beta_2 = float(conf['model']['beta_2'])

    weight_decay = float(conf['model']['weight_decay'])

    warmup_steps = conf['model']['warmup_steps']

    max_steps = conf['model']['max_steps']

    warmup_start_lr = float(conf['model']['warmup_start_lr'])

    eta_min = float(conf['model']['eta_min'])

    default_vars =  conf['model']['net']['init_args']['default_vars']

    tile_size = conf['model']['net']['init_args']['tile_size']

    patch_size = conf['model']['net']['init_args']['patch_size']
 
    emb_dim = conf['model']['net']['init_args']['embed_dim']

    depth = conf['model']['net']['init_args']['depth']

    num_heads = conf['model']['net']['init_args']['num_heads']
    
    mlp_ratio = conf['model']['net']['init_args']['mlp_ratio']

    drop_path = conf['model']['net']['init_args']['drop_path']

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    adaptive_patching = conf['model']['net']['init_args']['adaptive_patching']

    assert adaptive_patching, "SAP requires adaptive_patching"

    fixed_length = conf['model']['net']['init_args']['fixed_length']
    separate_channels = conf['model']['net']['init_args']['separate_channels']

    if not twoD:
        assert not separate_channels, "Adaptive Patching in 3D with multiple channels (non-separated) is not currently implemented"

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct"], "This training script only supports basic_ct dataloader for now"

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_start_idx = conf['data']['dict_start_idx']

    dict_end_idx = conf['data']['dict_end_idx']

    dict_buffer_sizes = conf['data']['dict_buffer_sizes']

    num_channels_used = conf['data']['num_channels_used']

    dict_in_variables = conf['data']['dict_in_variables']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    num_classes = conf['data']['num_classes']

    single_channel = conf['data']['single_channel']

    tile_overlap = conf['data']['tile_overlap']

    use_all_data = conf['data']['use_all_data']

    #These configs need only for finetuning with pre-trained MAE model
    decoder_embed_dim = conf['model']['net']['init_args']['decoder_embed_dim']

    decoder_depth = conf['model']['net']['init_args']['decoder_depth']

    decoder_num_heads = conf['model']['net']['init_args']['decoder_num_heads']

    mlp_ratio_decoder = conf['model']['net']['init_args']['mlp_ratio_decoder']

    mask_ratio = conf['model']['net']['init_args']['mask_ratio']

    linear_decoder = conf['model']['net']['init_args']['linear_decoder']

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]
    tile_size_z = tile_size[2]

    assert (tile_size_x%patch_size)==0, "tile_size_x % patch_size must be 0"
    assert (tile_size_y%patch_size)==0, "tile_size_y % patch_size must be 0"
    assert (tile_size_z%patch_size)==0, "tile_size_z % patch_size must be 0"

    if adaptive_patching:
        x_p2 = is_power_of_two(tile_size_x)
        assert x_p2, "tile_size_x must be a power of 2"
        y_p2 = is_power_of_two(tile_size_y)
        assert y_p2, "tile_size_y must be a power of 2"
        z_p2 = is_power_of_two(tile_size_z)
        assert z_p2, "tile_size_z must be a power of 2"

        if twoD:
            assert math.sqrt(fixed_length) % 1 == 0, "sqrt of fixed length needs to be a whole number"
            sqrt_len=int(math.sqrt(fixed_length))
            assert fixed_length % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
        else:
            assert np.abs(np.rint(math.pow(fixed_length,1/3)) - math.pow(fixed_length, 1/3)) < 0.0001, "cube root of fixed length needs to be a whole number"
            sqrt_len=int(np.rint(math.pow(fixed_length,1/3)))
            assert fixed_length % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"

    auto_load_balancing = conf['load_balancing']['auto_load_balancing']
    if auto_load_balancing:
        batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(config_path, world_size, batch_size)
    else:
        batches_per_rank_epoch = conf['load_balancing']['batches_per_rank_epoch']
        dataset_group_list = conf['load_balancing']['dataset_group_list']

#2. Initialize model, optimizer, and scheduler
##############################################################################################################

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
        

    model = SAP(
        img_size=tile_size,
        patch_size=patch_size,
        in_chans=max_channels,
        num_classes=num_classes,
        embed_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path,
        twoD=twoD,
        default_vars=default_vars,
        single_channel=single_channel,
        use_varemb=use_varemb,
        adaptive_patching=adaptive_patching,
        fixed_length=fixed_length,
        sqrt_len=sqrt_len,
        FusedAttn_option=FusedAttn_option,
        class_token=False,
        weight_init='skip',
    ).to(device)

    #model = DDP(model,device_ids=[local_rank],output_device=[local_rank])
    #find_unused_parameters=True is needed under these circumstances
    if resume_from_checkpoint:
        model = DDP(model,device_ids=[local_rank],output_device=[local_rank],find_unused_parameters=True)
 
    optimizer = configure_optimizer(model,lr,beta_1,beta_2,weight_decay)
    scheduler = configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min)

    if not resume_from_checkpoint:
        epoch_start = 0
        isExist = os.path.exists(checkpoint_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(checkpoint_path,exist_ok=True)
            print("The new checkpoint directory is created!")        
        loss_list = []

        if use_pretrained_mae_model:
            if single_channel:
                max_channels = 1
            else:
                max_channels = 1
                for i,k in enumerate(num_channels_used):
                    if num_channels_used[k] > 1:
                        max_channels = num_channels_used[k]
            mae_model = MAE(
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
                global_pool='no_reduce',
                class_token=False,
                mlp_ratio_decoder=mlp_ratio_decoder,
                weight_init='skip',
                default_vars=default_vars,
                single_channel=single_channel,
                use_varemb=use_varemb,
                adaptive_patching=adaptive_patching,
                fixed_length=fixed_length,
            ).to(device)
            #mae_model = DDP(mae_model,device_ids=[local_rank],output_device=[local_rank],find_unused_parameters=True)
            #dist.barrier()

            map_location = 'cpu'
            mae_checkpoint = torch.load(mae_checkpoint_path+"/"+mae_checkpoint_filename+".ckpt",map_location=map_location)

            
            mae_checkpoint_state_dict = mae_checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            for k, v in mae_checkpoint_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
                new_state_dict[name] = v
           
            interpolate_pos_embed_adaptive(model, new_state_dict, new_size=model.fixed_length)

            mae_model.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            encoder_dict = new_state_dict
            model_dict = mae_model.state_dict()
            encoder_dict = {k: v for k,v in model_dict.items() if ('decoder' not in k and 'mask_token' not in k)}

            model_dict = model.state_dict()
            model_dict.update(encoder_dict)
            model.load_state_dict(model_dict)
            del mae_checkpoint
            del mae_model
            del encoder_dict
            model = DDP(model,device_ids=[local_rank],output_device=[local_rank],find_unused_parameters=True)
            dist.barrier()
    else:
        dist.barrier()
        #map_location = 'cuda:'+str(device)
        map_location = 'cpu'
        checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename_for_loading+".ckpt",map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']
        epoch_start += 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_list = checkpoint['loss_list']
        del checkpoint


    dist.barrier()

#3. Initialize Dataloader
##############################################################################################################
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
        return_label = True,
        dataset_group_list = dataset_group_list,
        batches_per_rank_epoch = batches_per_rank_epoch,
        tile_overlap = tile_overlap,
        use_all_data = use_all_data,
        adaptive_patching = adaptive_patching,
        fixed_length = fixed_length,
        separate_channels = separate_channels,
        data_par_size = dist.get_world_size(),
        dataset = dataset,
        num_classes = num_classes,
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
            data_module.reset()
            train_dataloader = data_module.train_dataloader()

        #tell the model that we are in train mode. Matters because we have the dropout
        model.train()
        loss = 0.0
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)

        counter = 0
        for batch_idx, batch in enumerate(train_dataloader):
            counter = counter + 1
            if counter > iterations_per_epoch:
                print("A GPU ran out of data, moving to next epoch", flush=True)
                break

            #TODO: Add qdt_list back for visualization
            #data, seq, size, pos, label, seq_label, variables, qdt_list = batch
            seq, seq_label, variables, _ = batch
            seq = seq.to(device)
            seq_label = seq_label.to(device)
            loss = training_step_adaptive(seq, seq_label, variables, model, patch_size, twoD, num_classes, sqrt_len)

            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch,"batch_idx",batch_idx,"world_rank",world_rank,"it_loss ",loss,flush=True)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        loss_list.append(epoch_loss)


        if world_rank==0:
            print("epoch: ",epoch," epoch_loss ",epoch_loss, flush=True)

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
                }, checkpoint_path+"/"+checkpoint_filename+"_even.ckpt")

        if world_rank == 0 and epoch % 2 == 1:

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_states,
                'optimizer_state_dict': optimizer_states,
                'scheduler_state_dict': scheduler_states,
                'loss_list' : loss_list,
                }, checkpoint_path+"/"+checkpoint_filename+"_odd.ckpt")
     
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
