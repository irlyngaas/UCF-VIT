
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
import yaml
from einops import rearrange

from UCF_VIT.simple.arch import MAE
from UCF_VIT.utils.metrics import masked_mse, adaptive_patching_mse
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, patchify, unpatchify
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule

def training_step_adaptive(data, seq, size, pos, variables, net: MAE, patch_size, twoD, loss_fn, batch_idx, inference_path, epoch, world_rank, single_channel):

        
    output, mask = net.forward(seq, variables)
    if loss_fn == "realMSE": #Real Loss
        criterion = adaptive_patching_mse
        loss = criterion(output, data, size, pos, patch_size, twoD)
    else: #Compression Loss
        criterion = nn.MSELoss()
        target = rearrange(seq, 'b c s p -> b s (p c)')
        loss = criterion(output, target)


    return loss

def training_step(data, variables, net: MAE, patch_size, twoD, loss_fn, batch_idx, inference_path, epoch, world_rank, single_channel):

    if loss_fn == "maskMSE":
        output, mask = net.forward(data, variables)
        criterion = masked_mse
        target = patchify(data, patch_size, twoD)
        loss = criterion(output,target,mask)

    else: #Default use full MSE
        output, _ = net.forward(data, variables)
        criterion = nn.MSELoss()
        target = patchify(data, patch_size, twoD)
        loss = criterion(output,target)

    if world_rank == 0:
        if single_channel:
            if batch_idx == 0 or batch_idx == 1 or batch_idx == 2 or batch_idx == 3:
                pred_unpatched = unpatchify(output, data, patch_size, twoD)
                pred_unpatched = torch.clip(pred_unpatched, min=0.0, max=1.0)

                num_channels = data.shape[1]
                #Save a single item of batch for visualization
                for i in range(num_channels):
                    if twoD:
                        save_image(data[0,i,:,:], inference_path+'/gt_epoch'+str(epoch)+'_channel'+str(i)+'.png')
                        save_image(pred_unpatched[0,i,:,:], inference_path+'/pred_epoch'+str(epoch)+'_channel'+str(i)+'.png')

                    else:
                        #Save a middle slice for 3D visualization
                        save_image(data[0,i,:,int(patch_size/2),:], inference_path+'/gt_epoch'+str(epoch)+'_channel'+str(i)+'.png')
                        save_image(pred_unpatched[0,i,:,int(patch_size/2),:], inference_path+'/pred_epoch'+str(epoch)+'_channel'+str(i)+'.png')
        else:
            if batch_idx == 0:
                pred_unpatched = unpatchify(output, data, patch_size, twoD)
                pred_unpatched = torch.clip(pred_unpatched, min=0.0, max=1.0)

                num_channels = data.shape[1]
                #Save a single item of batch for visualization
                for i in range(num_channels):
                    if twoD:
                        save_image(data[0,i,:,:], inference_path+'/gt_epoch'+str(epoch)+'_channel'+str(i)+'.png')
                        save_image(pred_unpatched[0,i,:,:], inference_path+'/pred_epoch'+str(epoch)+'_channel'+str(i)+'.png')

                    else:
                        #Save a middle slice for 3D visualization
                        save_image(data[0,i,:,int(patch_size/2),:], inference_path+'/gt_epoch'+str(epoch)+'_channel'+str(i)+'.png')
                        save_image(pred_unpatched[0,i,:,int(patch_size/2),:], inference_path+'/pred_epoch'+str(epoch)+'_channel'+str(i)+'.png')
        

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

    checkpoint_path = conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    inference_path = conf['trainer']['inference_path']

    resume_from_checkpoint = conf['trainer']['resume_from_checkpoint']
 
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
        gauss_filter_order = conf['model']['net']['init_args']['gauss_filter_order']

        if not twoD:
            assert separate_channels, "Adaptive Patching in 3D with multiple channels (non-separated) is not currently implemented"
    else:
        fixed_length = None
        separate_channels = None
        gauss_filter_order = None

    dataset = conf['data']['dataset']
    assert dataset in ["basic_ct", "imagenet", "turb", "s8d", "s8d_2d"], "This training script only supports basic_ct, imagenet, s8d, s8d_2d, and turb datasets"


    dict_root_dirs = conf['data']['dict_root_dirs']

    if dataset == "turb":
        nx = conf['data']['nx']

        ny = conf['data']['ny']

        nz = conf['data']['nz']

        nx_skip = conf['data']['nx_skip']

        ny_skip = conf['data']['ny_skip']

        nz_skip = conf['data']['nz_skip']
    elif dataset == "s8d" or dataset == "s8d_2d":
        nx = conf['data']['nx']

        ny = conf['data']['ny']

        nz = None

        nx_skip = None

        ny_skip = None

        nz_skip = None
    else:
        nx = None

        ny = None

        nz = None

        nx_skip = None

        ny_skip = None

        nz_skip = None

    if dataset == "s8d":
        num_samples_to_stitch = conf['data']['num_samples_to_stitch']
        chunk_size = conf['data']['chunk_size']
    else:
        num_samples_to_stitch = None
        chunk_size = None
        

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

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]

    if dataset == "imagenet" or dataset == "s8d_2d":
        tile_size_z = None
    else:
        tile_size_z = tile_size[2]
    
    assert (tile_size_x%patch_size)==0, "tile_size_x % patch_size must be 0"
    assert (tile_size_y%patch_size)==0, "tile_size_y % patch_size must be 0"
    if dataset != "imagenet" and dataset != "s8d_2d":
        assert (tile_size_z%patch_size)==0, "tile_size_z % patch_size must be 0"

#2. Initialize model, optimizer, and scheduler
##############################################################################################################
    #Find correct in_chans to use
    if single_channel:
        max_channels = 1
    else:
        max_channels = 1
        for i,k in enumerate(num_channels_used):
            if num_channels_used[k] > 1:
                max_channels = num_channels_used[k]
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
        class_token=False,
        weight_init='skip',
    ).to(device)

    local_rank = int(os.environ['SLURM_LOCALID'])
    #model = DDP(model,device_ids=[local_rank],output_device=[local_rank])
    #find_unused_parameters=True is needed under these circumstances
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
        gauss_filter_order = gauss_filter_order,
        data_par_size = dist.get_world_size(),
        dataset = dataset,
        nx = nx,
        ny = ny,
        nz = nz,
        nx_skip = nx_skip,
        ny_skip = ny_skip,
        nz_skip = nz_skip,
        num_samples_to_stitch = num_samples_to_stitch,
        chunk_size = chunk_size,
    ).to(device)

    data_module.setup()

    train_dataloader = data_module.train_dataloader()

#4. Training Loop
##############################################################################################################
    isExist = os.path.exists(inference_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(inference_path,exist_ok=True)
        print("The new inference directory is created!")        

    #Find max batches
    iterations_per_epoch = 0
    for i,k in enumerate(batches_per_rank_epoch):
        if batches_per_rank_epoch[k] > iterations_per_epoch:
            iterations_per_epoch = batches_per_rank_epoch[k]

    for epoch in range(epoch_start,max_epochs):
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

            if adaptive_patching:
                data, seq, size, pos, variables = batch
                seq = seq.to(device)
                loss = training_step_adaptive(data, seq, size, pos, variables, model, patch_size, twoD, loss_fn, batch_idx, inference_path, epoch, world_rank, single_channel)

            else:
                data, variables = batch
                data = data.to(device)
                loss = training_step(data, variables, model, patch_size, twoD, loss_fn, batch_idx, inference_path, epoch, world_rank, single_channel)

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
