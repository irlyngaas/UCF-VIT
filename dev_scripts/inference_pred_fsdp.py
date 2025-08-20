
import glob
import os
import sys
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
from UCF_VIT.utils.misc import configure_optimizer, configure_scheduler, init_par_groups, get_test_data, stitch_data
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.fused_attn import FusedAttn

from sklearn.metrics import mean_squared_error, r2_score
import math
from UCF_VIT.utils.plotting import *

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def training_step(data, label, variables, net: UNETR, patch_size, twoD):

    output = net.forward(data, variables)

    criterion = nn.MSELoss()
    loss = criterion(output,label)

    return loss, output

def test_step(data, variables, net: UNETR):
    output = net.forward(data, variables)

    return output

def main(device, local_rank):
#1. Load arguments from config file and setup parallelization
##############################################################################################################

    print("in main()","sys.argv[1] ",sys.argv[1],flush=True) 
    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = dist.get_rank()

    config_path = sys.argv[1]

    snapshot = sys.argv[2]

    gravity = sys.argv[3]

    if world_rank==0:
        print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    if world_rank==0: 
        print(conf,flush=True)

    max_epochs=conf['trainer']['max_epochs']

    data_type = conf['trainer']['data_type']

    checkpoint_path =conf['trainer']['checkpoint_path']
  
    checkpoint_filename = conf['trainer']['checkpoint_filename']

    checkpoint_filename_for_loading = conf['trainer']['checkpoint_filename_for_loading']

    assert checkpoint_filename_for_loading not None, "Checkpoint needs to be specified for inferencing"

    inference_path =conf['trainer']['inference_path']

    fsdp_size = conf['parallelism']['fsdp_size']

    simple_ddp_size = conf['parallelism']['simple_ddp_size']

    tensor_par_size = conf['parallelism']['tensor_par_size']

    seq_par_size = conf['parallelism']['seq_par_size']

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

    dict_root_dirs = conf['data']['dict_root_dirs']

    dict_in_variables = conf['data']['dict_in_variables']

    num_channels_used = conf['data']['num_channels_used']

    single_channel = conf['data']['single_channel']

    tile_overlap = conf['data']['tile_overlap']

    dataset = conf['data']['dataset']
    assert dataset in ["sst"], "This inference script only supports sst dataset"

    
    batch_size = conf['data']['batch_size']

    num_classes = conf['data']['num_classes']

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



    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]
    tile_size_z = tile_size[2]
    
    assert (tile_size_x%patch_size)==0, "tile_size_x must be divisible by patch_size"
    assert (tile_size_y%patch_size)==0, "tile_size_y must be divisible by patch_size"
    assert (tile_size_z%patch_size)==0, "tile_size_z must be divisible by patch_size"

    data_par_size = fsdp_size * simple_ddp_size
    assert seq_par_size == 1, "Sequence parallelism not implemented"
    assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
    assert (num_heads % tensor_par_size) == 0, "model heads % tensor parallel size must be 0"

#2. Initialize model
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
        tensor_par_group=tensor_par_group,
        FusedAttn_option=FusedAttn_option,
        class_token=False,
        weight_init='skip',
    ).to(device)

    if world_rank < tensor_par_size:
        if os.path.exists(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt"):
            print("resume from checkpoint was set to True. Checkpoint path found.",flush=True)

            print("rank",dist.get_rank(),"src_rank",world_rank,flush=True)

            map_location = 'cpu'
            #map_location = 'cuda:'+str(device)

            checkpoint = torch.load(checkpoint_path+"/"+checkpoint_filename_for_loading+"_rank_"+str(world_rank)+".ckpt",map_location=map_location)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch_start = checkpoint['epoch']
            loss_list = checkpoint['loss_list']
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

    isExist = os.path.exists(inference_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(inference_path, exist_ok=True)
        print("The new inference directory is created!")

    model.eval()

    key = list(dict_in_variables.keys())[0]

    data, label = get_test_data(dict_root_dirs[key], snapshot, dict_in_variables[key], dict_out_variables[key], tile_size_z, tile_size_y, tile_size_x, int(nz[key]), int(ny[key]), int(nx[key]), int(nz_skip[key]), int(ny_skip[key]), int(nx_skip[key]), overlap=tile_overlap)
    print(f"Shape of data: {data.shape}; label: {label.shape}")

    # batched inference
    batch_inference = batch_size
    output = np.zeros(label.shape)
    start_time_i = time.time()
    for i in range(0, data.shape[0], batch_inference):
        idx_start = i
        idx_end = min(i + batch_inference, data.shape[0])
        data_test = data[idx_start:idx_end, :, :, :, :]
        if tensor_par_size > 1:
            asdf = 123
        else:
            data_test = torch.from_numpy(data_test)
            data_test = data_test.to(precision_dt)
            data_test = data_test.to(device)
        start_time = time.time()
        #output[idx_start:idx_end, :, :, :, :] = model_seq_par(torch.from_numpy( np.array(data_test) ) ).detach().cpu().numpy()
        output[idx_start:idx_end, :, :, :, :] = test_step(data_test, dict_in_variables[key], model).detach().cpu().numpy()
        elpsdt = time.time() - start_time
        print(f'Time elapsed for testing batch samples: {int(elpsdt//60)} min {elpsdt%60:.2f} sec')

    # Flatten the arrays across the desired dimensions
    label_flat = label.reshape(label.shape[0], -1)
    output_flat = output.reshape(output.shape[0], -1)

    # Compute inference metrics
    mse = mean_squared_error(label_flat, output_flat, multioutput='uniform_average') # compute the averaged mse over all dims
    msefull = mean_squared_error(label_flat, output_flat, multioutput='raw_values') # compute the mse over each dim

    print(f"Test MSE: {mse:1.3e} (RMSE: {math.sqrt(mse):1.3e})") # mse and RMSE (same unit as the vars compared)
    print(f"R2 values = {np.round([r2_score(label_flat[:,i], output_flat[:,i]) for i in range(4)], 2)}") # R2 score (coeff. determination) over each dim

    # Plot results
    print("Plotting results...")
    # Plot learning curve(s)
    plot_learning_curve(loss_list)
    plt.savefig(f'{inference_path}/loss_curves.png', bbox_inches='tight', dpi=200)

    # Plot box contours
    sampleID = 0  # sample (sub-cube) to plot
    varID = 0  # variable to plot
    # Grid
    nxoffset = 0
    nyoffset = 0
    nzoffset = 0
    nxsl = tile_size_x
    nysl = tile_size_y
    nzsl = tile_size_z

    Lh = 1.0
    Lv = 0.5


    #x = get_1Dgrid(Lh, nx-2, nxoffset, nxsl, nxskip)
    x = get_1Dgrid(Lh, int(nx[key]), nxoffset, nxsl, int(nx_skip[key]))
    if gravity == 'y':
        y = get_1Dgrid(Lv, int(ny[key]), nyoffset, nysl, int(ny_skip[key]))
        z = get_1Dgrid(Lh, int(nz[key]), nzoffset, nzsl, int(nz_skip[key]))
    elif gravity == 'z':
        y = get_1Dgrid(Lh, int(ny[key]), nyoffset, nysl, int(ny_skip[key]))
        z = get_1Dgrid(Lv, int(nz[key]), nzoffset, nzsl, int(nz_skip[key]))
    else:
        raise Exception("Gravity should be defined")

    for varID in range(len(dict_out_variables[key])):
        # Plot true results
        datacube = label[sampleID, varID, :]
        plot_contour_box(x, y, z, datacube, gravity)
        plt.savefig(f'{inference_path}/{snapshot}_test_box_plot_var-{dict_out_variables[key][varID]}_true.png', bbox_inches='tight', dpi=200)
        # Plot ML results
        datacubeML = output[sampleID, varID, :]
        mse_plot = mean_squared_error(datacube.reshape(datacube.shape[0], -1), datacubeML.reshape(datacubeML.shape[0], -1), multioutput='uniform_average') # compute the averaged mse over all dims
        ax = plot_contour_box(x, y, z, datacubeML, gravity)
        ax.set_title(f'MSE: {mse_plot:1.3e} (RMSE: {math.sqrt(mse_plot):1.3e})')
        plt.savefig(f'{inference_path}/{snapshot}_test_box_plot_var-{dict_out_variables[key][varID]}_ML.png', bbox_inches='tight', dpi=200)

    # Plot full data
    data_full = stitch_data(data, int(nz[key]), int(ny[key]), int(nx[key]), tile_size_z, tile_size_y, tile_size_x, int(nz_skip[key]), int(ny_skip[key]), int(nx_skip[key]), overlap=tile_overlap)
    label_full = stitch_data(label, int(nz[key]), int(ny[key]), int(nx[key]), tile_size_z, tile_size_y, tile_size_x, int(nz_skip[key]), int(ny_skip[key]), int(nx_skip[key]), overlap=tile_overlap)
    output_full = stitch_data(output, int(nz[key]), int(ny[key]), int(nx[key]), tile_size_z, tile_size_y, tile_size_x, int(nz_skip[key]), int(ny_skip[key]), int(nx_skip[key]), overlap=tile_overlap)

    # Full grid
    nxoffset = 0
    nyoffset = 0
    nzoffset = 0
    #nxsl = nx-2
    nxsl = int(nx[key])
    nysl = int(ny[key])
    nzsl = int(nz[key])
    #x = get_1Dgrid(Lh, nx-2, nxoffset, nxsl, nx_skip)
    x = get_1Dgrid(Lh, int(nx[key]), nxoffset, nxsl, int(nx_skip[key]))
    if gravity == 'y':
        y = get_1Dgrid(Lv, int(ny[key]), nyoffset, nysl, int(ny_skip[key]))
        z = get_1Dgrid(Lh, int(nz[key]), nzoffset, nzsl, int(nz_skip[key]))
    elif gravity == 'z':
        y = get_1Dgrid(Lh, int(ny[key]), nyoffset, nysl, int(ny_skip[key]))
        z = get_1Dgrid(Lv, int(nz[key]), nzoffset, nzsl, int(nz_skip[key]))
    else:
        raise Exception("Gravity should be defined")

    for varID in range(len(dict_out_variables[key])):
        # Plot true results
        datacube = label_full[:,:,:, varID]
        plot_contour_box(x, y, z, datacube, gravity)
        plt.savefig(f'{inference_path}/{snapshot}_test_box_plotFull_var-{dict_out_variables[key][varID]}_true.png', bbox_inches='tight', dpi=200)
        # Plot ML results
        datacubeML = output_full[:,:,:, varID] 
        mse_plot = mean_squared_error(datacube.reshape(datacube.shape[0], -1), datacubeML.reshape(datacubeML.shape[0], -1), multioutput='uniform_average') # compute the averaged mse over all dims
        ax = plot_contour_box(x, y, z, datacubeML, gravity)
        ax.set_title(f'MSE: {mse_plot:1.3e} (RMSE: {math.sqrt(mse_plot):1.3e})')
        plt.savefig(f'{inference_path}/{snapshot}_test_box_plotFull_var-{dict_out_variables[key][varID]}_ML.png', bbox_inches='tight', dpi=200)

    # Plot input data
    for varID in range(len(dict_in_variables[key])):
        datacube = data_full[:,:,:, varID]
        plot_contour_box(x, y, z, datacube, gravity)
        plt.savefig(f'{inference_path}/{snapshot}_input_box_plotFull_var-{dict_in_variables[key][varID]}.png', bbox_inches='tight', dpi=200)

    print("Finished plotting. Done!")

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
    
    main(device, local_rank)

    dist.destroy_process_group()
