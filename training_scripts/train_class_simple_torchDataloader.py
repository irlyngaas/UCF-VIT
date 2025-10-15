
import glob
import os
import sys
import random
import argparse
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
from timm.layers import use_fused_attn

from UCF_VIT.simple.arch import VIT
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, is_power_of_two
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.utils.quanto_quantization import (
    setup_quanto_quantization, 
    add_quantization_args, 
    create_quantization_config_from_args
)
from UCF_VIT.utils.torchao_quantization import (
    setup_torchao_quantization,
    add_torchao_quantization_args,
    create_torchao_config_from_args
)

from torch.utils.data import DataLoader


def training_step(data, variables, label, net: VIT, seq_ps):

    output = net.forward(data, variables, seq_ps)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output,label)

    return loss, output


def main(device, local_rank):
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UCF-VIT Training with Quantization Support')
    parser.add_argument('config', help='Path to config file')
    parser = add_quantization_args(parser)
    parser = add_torchao_quantization_args(parser)
    args = parser.parse_args()

    config_path = args.config

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
    
    # Create quantization configs from command line arguments
    quantization_config = create_quantization_config_from_args(args)
    torchao_config = create_torchao_config_from_args(args)
    conf['quantization'] = quantization_config
    conf['torchao_quantization'] = torchao_config

    if world_rank==0: 
        print(conf,flush=True)

    max_epochs = conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    assert data_type == "float32", "Only float32 training supported in this training script"

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']
 
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

    drop_rate = conf['model']['net']['init_args']['drop_rate']

    twoD = conf['model']['net']['init_args']['twoD']

    use_varemb = conf['model']['net']['init_args']['use_varemb']

    adaptive_patching = conf['model']['net']['init_args']['adaptive_patching']

    if adaptive_patching:
        fixed_length = conf['model']['net']['init_args']['fixed_length']
        use_adaptive_pos_emb = conf['model']['net']['init_args']['use_adaptive_pos_emb']
    else:
        fixed_length = None
        use_adaptive_pos_emb = None

    dataset = conf['data']['dataset']
    assert dataset in ["catsdogs"], "This training script only supports catsdogs"
    
    if dataset == "catsdogs":
        from UCF_VIT.datasets.catsdogs import CatsDogsDataset as Dataset_
        from UCF_VIT.datasets.catsdogs import CatsDogsCollate as collate_fn
        assert twoD, "twoD must be True if using catsdogs"

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_in_variables = conf['data']['dict_in_variables']

    batch_size = conf['data']['batch_size']

    num_workers = conf['data']['num_workers']

    pin_memory = conf['data']['pin_memory']

    num_classes = conf['data']['num_classes']

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]

    if dataset == "catsdogs":
        tile_size_z = None
    else:
        tile_size_z = tile_size[2]
    
    assert (tile_size_x%patch_size)==0, "tile_size_x % patch_size must be 0"
    assert (tile_size_y%patch_size)==0, "tile_size_y % patch_size must be 0"
    if dataset != "catsdogs":
        assert (tile_size_z%patch_size)==0, "tile_size_z % patch_size must be 0"

    if adaptive_patching:
        x_p2 = is_power_of_two(tile_size_x)
        assert x_p2, "tile_size_x must be a power of 2"
        y_p2 = is_power_of_two(tile_size_y)
        assert y_p2, "tile_size_y must be a power of 2"
        if dataset != "catsdogs":
            z_p2 = is_power_of_two(tile_size_z)
            assert z_p2, "tile_size_z must be a power of 2"

        if twoD:
            assert fixed_length % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
        else:
            sqrt_len=int(np.rint(math.pow(fixed_length,1/3)))
            assert fixed_length % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"

#2. Initialize model, optimizer, and scheduler
##############################################################################################################

    if use_fused_attn():
        FusedAttn_option = FusedAttn.DEFAULT
    else:
        FusedAttn_option = FusedAttn.NONE

    #Find correct in_chans to use
    in_channels = len(dict_in_variables["catsdogs"])

    model = VIT(
        img_size=tile_size,
        patch_size=patch_size,
        num_classes=num_classes,
        in_chans=in_channels,
        embed_dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path,
        drop_rate=drop_rate,
        twoD=twoD,
        weight_init='',
        default_vars=default_vars,
        #single_channel=single_channel,
        use_varemb=use_varemb,
        adaptive_patching=adaptive_patching,
        fixed_length=fixed_length,
        FusedAttn_option=FusedAttn_option,
        use_adaptive_pos_emb=use_adaptive_pos_emb,
    ).to(device)

    #model = DDP(model,device_ids=[local_rank],output_device=[local_rank])
    #find_unused_parameters=True is needed under these circumstances
    model = DDP(model,device_ids=[local_rank],output_device=[local_rank],find_unused_parameters=True)

    # QUANTIZATION INTEGRATION
    # Apply quantization after DDP wrapping
    if quantization_config.get('enabled', False):
        if world_rank == 0:
            print("=" * 80, flush=True)
            print("APPLYING QUANTO QUANTIZATION", flush=True)
            print(f"Target: {quantization_config.get('bits', 8)}-bit with quanto", flush=True)
            print("=" * 80, flush=True)
            
        # Apply quanto quantization to the DDP-wrapped model
        model = setup_quanto_quantization(model, quantization_config)
        
        if world_rank == 0:
            print("Quanto quantization setup complete!", flush=True)
    
    elif torchao_config.get('enabled', False):
        if world_rank == 0:
            print("=" * 80, flush=True)
            print("APPLYING TORCH.AO QUANTIZATION", flush=True)
            print(f"Target: {torchao_config.get('bits', 8)}-bit with torch.ao", flush=True)
            print(f"Method: {torchao_config.get('method', 'dynamic')}", flush=True)
            print("=" * 80, flush=True)
            
        # Apply torch.ao quantization to the DDP-wrapped model
        model = setup_torchao_quantization(model, torchao_config)
        
        if world_rank == 0:
            print("Torch.ao quantization setup complete!", flush=True)
 
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
    else:
        dist.barrier()
        map_location = 'cpu'
        #map_location = 'cuda:'+str(device)
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
    dkey_train = list(dict_root_dirs)[0]
    #dkey_test = list(dict_root_dirs_test)[0]
    train_list = glob.glob(os.path.join(dict_root_dirs[dkey_train],'*.jpg'))
    #test_list = glob.glob(os.path.join(dict_root_dirs_test[dkey_test], '*.jpg'))

    train_data = Dataset_(train_list, dict_in_variables[dkey_train], tile_size, adaptive_patching=adaptive_patching, fixed_length=fixed_length, patch_size=patch_size, num_channels=len(dict_in_variables[dkey_train]), dataset=dataset)
    #test_data = Dataset(test_list, transform=test_transforms, adaptive_patching=adaptive_patching, fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=dataset)



    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, num_replicas=dist.get_world_size(),rank=dist.get_rank())
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=True, num_replicas=dist.get_world_size(),rank=dist.get_rank())

    train_loader = DataLoader(dataset = train_data, sampler=train_sampler,num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, drop_last=True, collate_fn=lambda batch: collate_fn(batch, adaptive_patching=adaptive_patching))
    #test_loader = DataLoader(dataset = test_data, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size,drop_last=True, collate_fn=lambda batch: collate_fn(batch, adaptive_patching=adaptive_patching))

    len_train_loader = torch.tensor(len(train_loader)).to(device)


#4. Training Loop
##############################################################################################################

    for epoch in range(epoch_start,max_epochs):

        train_loader_iter = iter(train_loader)
        model.train()
        epoch_loss = torch.tensor(0.0 , dtype=torch.float32, device=device)
        epoch_accuracy = torch.tensor(0.0 , dtype=torch.float32, device=device)
        if world_rank==0:
            print("epoch ",epoch,flush=True)
            
        for it in range(len_train_loader):

            if adaptive_patching:
                data, seq, seq_size, seq_pos, label, variables = next(train_loader_iter)
                seq = seq.to(device)
                label = label.to(device)
                seq_size = torch.squeeze(seq_size)
                seq_size = seq_size.to(torch.float32)
                seq_size = seq_size.to(device)
                seq_pos = torch.squeeze(seq_pos)
                seq_pos = seq_pos.to(torch.float32)
                seq_pos = seq_pos.to(device)
                seq_size = seq_size.unsqueeze(-1)
                seq_ps = torch.concat([seq_size, seq_pos],dim=-1)

            else:
                data, label, variables = next(train_loader_iter)
                data = data.to(device)
                data = data.to(torch.float32)
                label = label.to(device)
                seq_ps = None

            loss, output = training_step(data, variables, label, model, seq_ps)

            acc = (output.argmax(dim=1) == label).float().mean()

            epoch_accuracy += acc.detach()
            epoch_loss += loss.detach()
    
            if world_rank==0:
                print("epoch: ",epoch, "batch_idx", it, "it_loss ",loss, "it_acc", acc, flush=True)
    
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
