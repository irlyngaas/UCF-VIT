import torch
import torch.distributed as dist
import yaml
import os
import numpy as np
import nibabel as nib
import math
import glob
from PIL import Image
import cv2 as cv
import torchdata.datapipes as dp
from UCF_VIT.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

def patchify( data, patch_size, twoD):
    batch_size = data.shape[0]
    num_channels = data.shape[1]
    dim_x = data.shape[2]
    dim_y = data.shape[3]
    if not twoD:
        dim_z = data.shape[4]
    num_patches_x = dim_x // patch_size
    num_patches_y = dim_y // patch_size
    if not twoD:
        num_patches_z = dim_z // patch_size
    if twoD:
        patchified_pixel_values = data.reshape(batch_size, num_channels, num_patches_x, patch_size, num_patches_y, patch_size)
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape( batch_size, num_patches_x * num_patches_y, patch_size**2 * num_channels)
    else:
        patchified_pixel_values = data.reshape(batch_size, num_channels, num_patches_x, patch_size, num_patches_y, patch_size, num_patches_z, patch_size)
        patchified_pixel_values = torch.einsum("nchpwqdr->nhwdpqrc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape( batch_size, num_patches_x * num_patches_y * num_patches_z, patch_size**3 * num_channels)
    return patchified_pixel_values

def unpatchify(patchified_pixel_values,data, patch_size, twoD):
    if twoD:
        original_x, original_y = data.shape[2], data.shape[3]
    else:
        original_x, original_y, original_z = data.shape[2], data.shape[3], data.shape[4]

    num_patches_x = original_x // patch_size
    num_patches_y = original_y // patch_size
    if not twoD:
        num_patches_z = original_z // patch_size
    
    batch_size = patchified_pixel_values.shape[0]
    num_channels = data.shape[1]
    if twoD:
        patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_x, num_patches_y, patch_size, patch_size, num_channels)
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_x*patch_size, num_patches_y*patch_size)
    else:
        patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_x, num_patches_y, num_patches_z, patch_size, patch_size, patch_size, num_channels)
        patchified_pixel_values = torch.einsum("nhwdpqrc->nchpwqdr", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_x*patch_size, num_patches_y*patch_size, num_patches_z*patch_size)
    return pixel_values

def configure_optimizer(model,lr,beta_1,beta_2,weight_decay):
    decay = []
    no_decay = []
    for name, m in model.named_parameters():
        if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
            no_decay.append(m)
        else:
            decay.append(m)

    optimizer = torch.optim.AdamW(
        [
        {
            "params": decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay,
            "lr": lr,
            "betas": (beta_1, beta_2),
            "weight_decay": 0,
        },
        ]
    )

    return optimizer

def configure_scheduler(optimizer,warmup_steps,max_steps,warmup_start_lr,eta_min):
    
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps,
        max_steps,
        warmup_start_lr,
        eta_min,
    )

    return lr_scheduler

def interpolate_pos_embed_adaptive(model, checkpoint_model, new_size=127):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]

        if orig_num_patches != new_size:
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_num_patches, embedding_size).permute(0, 2, 1)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode="linear", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0,2,1)
            checkpoint_model["pos_embed"] = new_pos_tokens

            del new_pos_tokens

    if "decoder_pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["decoder_pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]

        if orig_num_patches != new_size:
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_num_patches, embedding_size).permute(0, 2, 1)
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode="linear", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0,2,1)
            checkpoint_model["decoder_pos_embed"] = new_pos_tokens

            del new_pos_tokens

def init_par_groups(world_rank, data_par_size, tensor_par_size, seq_par_size, fsdp_size, simple_ddp_size):

    tensor_par_group = None

    for i in range(data_par_size *seq_par_size):
        ranks = [j for j in range(i*tensor_par_size,(i+1)*tensor_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," tensor_par_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:
            tensor_par_group = group




    seq_par_group = None

    for t in range(data_par_size):
        for i in range(tensor_par_size):
            ranks = [t*tensor_par_size*seq_par_size+i+j*tensor_par_size for j in range(seq_par_size)]

            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size, " TENSOR_PAR_SIZE ",tensor_par_size," seq_par_group ranks ",ranks,flush=True)

            group = dist.new_group(ranks)

            if world_rank in ranks:

                seq_par_group = group




    ddp_group = None

    fsdp_group = None

    simple_ddp_group = None

    for i in range(tensor_par_size *seq_par_size):
        ranks = [i+j*tensor_par_size *seq_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]


            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," fsdp_ranks",fsdp_ranks)


            group = dist.new_group(fsdp_ranks)

            if world_rank in fsdp_ranks:

                fsdp_group = group


        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]


            #if world_rank==0:
            #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," simple_ddp_ranks",simple_ddp_ranks)


            group = dist.new_group(simple_ddp_ranks)

            if world_rank in simple_ddp_ranks:

                simple_ddp_group = group



        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," ddp_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            ddp_group = group






    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size * seq_par_size)]

        #if world_rank==0:
        #    print("i ",i," data_par_size ",data_par_size," SEQ_PAR_SIZE ",seq_par_size," TENSOR_PAR_SIZE ",tensor_par_size," data_seq_ort_group ranks ",ranks)

        group = dist.new_group(ranks)

        if world_rank in ranks:

            data_seq_ort_group = group


    return seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group

def process_root_dirs(dataset, dict_root_dirs, data_par_size, nx, ny, nz, chunk_size, num_samples, num_slices_per_sample):
    if dataset == "imagenet":
        dict_lister_trains = {}
        for k, root_dir in dict_root_dirs.items():
            #TODO: Add shuffling for data_par_size if it doesn't divide 1000 equally
            classes = sorted(os.listdir(root_dir))
            if len(classes) > data_par_size:
                classes_to_combine = int(len(classes) // data_par_size)
            img_list = []
            classes_counter = 0
            num_data_roots = 0
            for cls_name in classes: 
                if classes_counter == classes_to_combine:
                    classes_counter = 0
                    img_list = []
                cls_dir = os.path.join(root_dir, cls_name)
                for img_path in glob.glob(os.path.join(cls_dir,"*.JPEG")):
                    img_list.append(img_path)
                classes_counter += 1
            
                if classes_counter == classes_to_combine:
                    img_dict = {num_data_roots: img_list}
                    dict_lister_trains.update(img_dict)
                    num_data_roots +=1

                if num_data_roots > data_par_size-1:
                    break
    elif dataset == "s8d_2d":
        dict_lister_trains = {}
        for k, root_dir in dict_root_dirs.items():
            samples = sorted(os.listdir(root_dir))
            img_list = []
            for sample in samples: 
                sample_dir = os.path.join(root_dir, sample)
                for img_path in glob.glob(os.path.join(sample_dir,"*.raw")):
                    img_list.append(img_path)
            
            img_dict = {k: img_list}
            dict_lister_trains.update(img_dict)
            
    elif dataset == "s8d_2d_label":
        dict_lister_trains = {}
        for k, root_dir in dict_root_dirs.items():
            img_list = []
            for img_path in glob.glob(os.path.join(root_dir,"*.npy")):
                img_list.append(img_path)
            
            img_dict = {k: img_list}
            dict_lister_trains.update(img_dict)

    elif dataset == "s8d_3d":
        dict_lister_trains = {}
        dict_chunk_trains = {}
        for k, root_dir in dict_root_dirs.items():
            num_chunks_x = int(nx[k]/chunk_size[k][0])
            num_chunks_y = int(ny[k]/chunk_size[k][1])
            num_chunks_z = int(nz[k]/chunk_size[k][2])
            samples = sorted(os.listdir(root_dir))
            #In 3D rather than passing individual files pass list of files with corresponding connected samples
            img_list = []
            chunk_list = []

            sample_iterator = 0
            slice_iterator = 0
            for i in range(num_chunks_z):
                img_sample_list = []
                while sample_iterator < num_samples[k]:
                    sample_dir = os.path.join(root_dir, samples[sample_iterator])
                    img_paths = sorted(glob.glob(os.path.join(sample_dir,"*.raw")))
                    while slice_iterator < num_slices_per_sample[k]: 
                        if len(img_sample_list) < chunk_size[k][2]:
                            img_sample_list.append(img_paths[slice_iterator])
                            slice_iterator = slice_iterator + 1
                            
                        if len(img_sample_list) == chunk_size[k][2]:
                            break
                        
                    if len(img_sample_list) == chunk_size[k][2]:
                        for x_chunk in range(num_chunks_x):
                            for y_chunk in range(num_chunks_y):
                                img_list.append(img_sample_list)
                                chunk_list.append([x_chunk,y_chunk,0])
                        break
                    else:
                        slice_iterator = 0
                        sample_iterator = sample_iterator + 1
            
            img_dict = {k: img_list}
            dict_lister_trains.update(img_dict)
            chunk_dict = {k: chunk_list}
            dict_chunk_trains.update(chunk_dict)
    else:
        dict_lister_trains = { k: list(dp.iter.FileLister(os.path.join(root_dir, "imagesTr"))) for k, root_dir in dict_root_dirs.items() }

    if dataset != "s8d_3d":
        dict_chunk_trains = None
    return dict_lister_trains, dict_chunk_trains

def read_process_file(dataset, path, imagenet_resize, nx, ny, chunk_size):
    if dataset == "imagenet":
        data = Image.open(path).convert("RGB")
        data = np.array(data) 
        data = cv.resize(data, dsize=[imagenet_resize["imagenet"][0],imagenet_resize["imagenet"][1]])
    elif dataset == "s8d_2d":
        data = np.fromfile(path, dtype=np.uint16).reshape([nx,ny])
    elif dataset == "s8d_2d_label":
        data = np.load(path)
        data = np.moveaxis(data,0,-1)
    elif dataset == "s8d_3d":
        #data_list = []
        #for idx in range(len(keys[0])):
        #    chunk_idx = chunks[0]
        #    
        #    data = np.fromfile(keys[0][idx], dtype=np.uint16).reshape(nx[k],ny[k])
        #    data = (data[chunk_idx[0]*chunk_size[k][0]:(chunk_idx[0]+1)*chunk_size[k][0], chunk_idx[1]*chunk_size[k][1]:(chunk_idx[1]+1)*chunk_size[k][1]])
        #    data_list.append(data)
        #data = np.stack([data_list[idx] for idx in range(len(data_list))])
        ##z stacked on first dimension, so move to last dimension
        #data = np.moveaxis(data, 0, -1)

        #Use this alternative to not need to load from files which is time consuming
        data = np.zeros(shape=(chunk_size[0],chunk_size[1],chunk_size[2]), dtype=np.uint8)
    else:
        data = nib.load(path)
        data = np.array(data.dataobj).astype(np.float32)
    return data

def calculate_load_balancing_on_the_fly(yaml_file, data_par_size, batch_size, VERBOSE=False):
    conf = yaml.load(open(yaml_file,'r'),Loader=yaml.FullLoader)
    num_total_ddp_ranks = data_par_size

    dict_root_dirs = conf['data']['dict_root_dirs']
    dict_start_idx = conf['data']['dict_start_idx']
    dict_end_idx = conf['data']['dict_end_idx']
    tile_size =  conf['model']['net']['init_args']['tile_size']
    twoD = conf['model']['net']['init_args']['twoD']
    num_channels_used = conf['data']['num_channels_used']
    single_channel = conf['data']['single_channel']
    batch_size = conf['data']['batch_size']
    tile_overlap = conf['data']['tile_overlap']
    use_all_data = conf['data']['use_all_data']
    patch_size =  conf['model']['net']['init_args']['patch_size']
    dataset = conf['data']['dataset']

    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['imagenet_resize']
    else:
        imagenet_resize = None

    if dataset == "s8d_2d" or dataset == "s8d_3d":
        nx = conf['dataset_options']['nx']
        ny = conf['dataset_options']['ny']
    else:
        nx = None
        ny = None

    if dataset == "s8d_3d":
        nz = conf['dataset_options']['nz']
        chunk_size = conf['dataset_options']['chunk_size']
        num_samples = conf['dataset_options']['num_samples']
        num_slices_per_sample = conf['dataset_options']['num_slices_per_sample']
    else:
        nz = None
        chunk_size = None
        num_samples = None
        num_slices_per_sample = None

    tile_size_x = int(tile_size[0])
    tile_size_y = int(tile_size[1])
    if dataset != "imagenet" and dataset != "s8d_2d":
        tile_size_z = int(tile_size[2])

    dict_lister_trains, dict_chunk_trains = process_root_dirs(dataset, dict_root_dirs, num_total_ddp_ranks, nx, ny, nz, chunk_size, num_samples, num_slices_per_sample)

    num_total_tiles = []
    num_total_images = []
    tiles_per_image = []
    num_channels_per_dataset = []
    for i, k in enumerate(dict_lister_trains.keys()):
        lister_train = dict_lister_trains[k]
        if dataset == "imagenet":
            start_idx = int(dict_start_idx["imagenet"] * len(lister_train))
            end_idx = int(dict_end_idx["imagenet"] * len(lister_train))
        else:
            start_idx = int(dict_start_idx[k] * len(lister_train))
            end_idx = int(dict_end_idx[k] * len(lister_train))
        keys = lister_train[start_idx:end_idx]
        num_total_images.append(len(keys))

        #Assume all channels have the same data size
        if dataset == "s8d_3d":
            data_path = None
        else:
            data_path = keys[0]
       
        if chunk_size != None:
            data = read_process_file(dataset, data_path, imagenet_resize, nx[k], ny[k], chunk_size[k])

        else:
            if nx != None:
                data = read_process_file(dataset, data_path, imagenet_resize, nx[k], ny[k], chunk_size)
            else:
                data = read_process_file(dataset, data_path, imagenet_resize, nx, ny, chunk_size)

        tile_overlap_size_x = int(tile_size_x*tile_overlap)
        tile_overlap_size_y = int(tile_size_y*tile_overlap)
        
        if tile_overlap == 0.0:
            OTP2_x = 1
            tile_overlap_size_x = tile_size_x
        else:
            OTP2_x = int(tile_size_x/tile_overlap_size_x)

        if tile_overlap == 0.0:
            OTP2_y = 1
            tile_overlap_size_y = tile_size_y
        else:
            OTP2_y = int(tile_size_y/tile_overlap_size_y)
            
        #USE THIS IF RAW FILES ARE 2D
        if dataset == "imagenet" or dataset == "s8d_2d":
            #Total Tiles Evenly Spaced
            TTE_x = data.shape[0]//tile_size_x
            TTE_y = data.shape[1]//tile_size_y
            num_blocks_x = (TTE_x-1)*OTP2_x + 1
            num_blocks_y = (TTE_y-1)*OTP2_y + 1
            if use_all_data:
                #Total Tiles
                TT_x = data.shape[0]/(tile_size_x)
                TT_y = data.shape[1]/(tile_size_y)
                # Number of leftover overlap patches for last tile
                LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                if data.shape[0] % tile_overlap_size_x != 0:
                    LTOP_x += 1
                if data.shape[1] % tile_overlap_size_y != 0:
                    LTOP_y += 1
                num_blocks_x = int(num_blocks_x + LTOP_x)
                num_blocks_y = int(num_blocks_y + LTOP_y)

            if VERBOSE:
                print("KEY", k, "DATA_SHAPE", data.shape,"NUM_BLOCKS:", num_blocks_x, num_blocks_y, flush=True)

            tiles_per_image.append(num_blocks_x*num_blocks_y)
            if dataset == "imagenet":
                num_channels_per_dataset.append(num_channels_used["imagenet"])
            else:
                num_channels_per_dataset.append(num_channels_used[k])
        #USE THIS IF RAW FILES ARE 3D
        else:
            tile_overlap_size_z = int(tile_size_z*tile_overlap)
            if tile_overlap == 0.0:
                OTP2_z = 1
                tile_overlap_size_z = tile_size_z
            else:
                OTP2_z = int(tile_size_z/tile_overlap_size_z)

            #Total Tiles Evenly Spaced
            TTE_x = data.shape[0]//tile_size_x
            TTE_y = data.shape[1]//tile_size_y
            num_blocks_x = (TTE_x-1)*OTP2_x + 1
            num_blocks_y = (TTE_y-1)*OTP2_y + 1
            if use_all_data:
                #Total Tiles
                TT_x = data.shape[0]/(tile_size_x)
                TT_y = data.shape[1]/(tile_size_y)
                # Number of leftover overlap patches for last tile
                LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                if data.shape[0] % tile_overlap_size_x != 0:
                    LTOP_x += 1
                if data.shape[1] % tile_overlap_size_y != 0:
                    LTOP_y += 1
                num_blocks_x = int(num_blocks_x + LTOP_x)
                num_blocks_y = int(num_blocks_y + LTOP_y)

            if twoD:
                if use_all_data:
                    num_blocks_z = data.shape[2]//tile_size_z
                    leftover_z_tiles = data.shape[2] % tile_size_z
                else:
                    num_blocks_z = data.shape[2]//tile_size_z
            else:
                TTE_z = data.shape[2]//tile_size_z
                num_blocks_z = (TTE_z-1)*OTP2_z + 1
                if use_all_data:
                    #Total Tiles Rounded Up
                    TT_z = data.shape[2]/(tile_size_z)
                    # Number of leftover overlap patches for last tile
                    LTOP_z = np.floor((TT_z-TTE_z)*OTP2_z)
                    if data.shape[2] % tile_overlap_size_z != 0:
                        LTOP_z += 1
                    num_blocks_z = int(num_blocks_z + LTOP_z)

            if VERBOSE:
                print("KEY", k, "DATA_SHAPE", data.shape,"NUM_BLOCKS:", num_blocks_x, num_blocks_y, num_blocks_z, flush=True)

            if twoD:
                if use_all_data:
                    tiles_per_image.append(num_blocks_x*num_blocks_y*num_blocks_z*tile_size_z + num_blocks_x*num_blocks_y*leftover_z_tiles)
                else:
                    tiles_per_image.append(num_blocks_x*num_blocks_y*num_blocks_z*tile_size_z)
            else:
                tiles_per_image.append(num_blocks_x*num_blocks_y*num_blocks_z)
            
            num_channels_per_dataset.append(num_channels_used[k])

        if single_channel:
            num_total_tiles.append(tiles_per_image[i] * num_channels_per_dataset[i] * num_total_images[i])
        else:
            num_total_tiles.append(tiles_per_image[i] * num_total_images[i])

    if VERBOSE:
        print("Total Images", num_total_images)
        print("Tiles Per Image", tiles_per_image)
        print("Total Tiles per Dataset", num_total_tiles)
        print("Total Tiles", sum(num_total_tiles))
        if twoD:
            print("Total Tokens", sum(num_total_tiles)*(tile_size_x/patch_size)*(tile_size_y/patch_size))
        else:
            print("Total Tokens", sum(num_total_tiles)*(tile_size_x/patch_size)*(tile_size_y/patch_size)*(tile_size_z/patch_size))
        
    total_tiles_all_data = sum(num_total_tiles)
        
    ddp_ratio = []
    ddp_rank_ratio = []
    ratio_diff = []
    for i in range(len(num_total_tiles)):
        ratio = num_total_tiles[i]/total_tiles_all_data
        ddp_ratio.append(ratio*num_total_ddp_ranks)
        ddp_rank_ratio.append(int(np.rint(ddp_ratio[i])))
        ratio_diff.append(ddp_rank_ratio[i] - ddp_ratio[i])
    if VERBOSE:
        print("DDP RATIO", ddp_ratio)
        print("DDP RANK RATIO", ddp_rank_ratio)

    rank_sum = sum(ddp_rank_ratio)

    #Rebalance till ranks equal actually amount wanted to use
    while rank_sum != num_total_ddp_ranks:
        leftover = []
        for i in range(len(num_total_tiles)):
            if ddp_ratio[i] > ddp_rank_ratio[i]:
                leftover.append((-1.0)*(ddp_ratio[i]-ddp_rank_ratio[i]))
            else:
                leftover.append(ddp_rank_ratio[i]-ddp_ratio[i])
        if rank_sum > num_total_ddp_ranks:
            rank_to_decrease = -1
            for i in range(len(num_total_tiles)):
                if leftover[i] < 0:
                    continue
                else:
                    if rank_to_decrease == -1:
                        rank_to_decrease = i
                        continue
                    if ddp_rank_ratio[rank_to_decrease] == 1:
                        rank_to_decrease = i
                        continue
                    if leftover[i] > leftover[rank_to_decrease] and ddp_rank_ratio[i] > 1:
                        rank_to_decrease = i
            if VERBOSE:
                print("Rank to decrease", rank_to_decrease)
            ddp_rank_ratio[rank_to_decrease] -= 1

        if rank_sum < num_total_ddp_ranks:
            rank_to_increase = -1
            for i in range(len(num_total_tiles)):
                if leftover[i] > 0:
                    continue
                else:
                    if rank_to_increase == -1:
                        rank_to_increase = i
                        continue
                    if leftover[i] < leftover[rank_to_increase]:
                        rank_to_increase = i
            if VERBOSE:
                print("Rank to increase", rank_to_increase)
            ddp_rank_ratio[rank_to_increase] += 1

        rank_sum = sum(ddp_rank_ratio)

    if VERBOSE:
        print("DDP RANKS:", ddp_rank_ratio)
    assert rank_sum == num_total_ddp_ranks, "All DDP ranks not used"

    for i in range(len(ddp_rank_ratio)):
        assert ddp_rank_ratio[i] > 0, "All Datasets need at least one GPU. Add more GPUs to the training to resolve this issue, or consider removing datasets with small amounts of data"

    num_images_per_rank = []
    for i in range(len(num_total_tiles)):
        num_images_per_rank.append(int(math.floor(num_total_images[i] / float(ddp_rank_ratio[i]))))
    #print("Num Images Per Rank", num_images_per_rank)
    assert min(num_images_per_rank) >= 1.0, "Decrease number of GPUs, not all GPUs have their own image"

    batches_per_rank = []
    tiles_per_rank = []
    for i in range(len(num_total_tiles)):
        if single_channel:
            batches_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i]*num_channels_per_dataset[i]/batch_size)
        else:
            batches_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i]/batch_size)
        tiles_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i])
    if VERBOSE:
        print("Tiles Per Rank", tiles_per_rank)
        print("USE BELOW IN CONFIG FILE")
        print("batches_per_rank_epoch: {")
    batches_per_rank_epoch = {}
    if dataset == "imagenet":
        new_data = [("imagenet", int(min(batches_per_rank)))]
        batches_per_rank_epoch.update(new_data)
    else:
        for i,k in enumerate(dict_lister_trains.keys()):
            new_data = [(k, int(batches_per_rank[i]))]
            batches_per_rank_epoch.update(new_data)

    if VERBOSE:
        if dataset == "imagenet":
            print("'%s': %i," % ("imagenet", int(min(batches_per_rank))))
        else:
            for i, k in enumerate(dict_lister_trains.keys()):
                print("'%s': %i," % (k, int(batches_per_rank[i])))
        print('}')

    grouplist_str = ''
    for i in range(len(ddp_rank_ratio)):
        grouplist_str += str(ddp_rank_ratio[i])+':'
    if VERBOSE:
        print("dataset_group_list: '%s'" % (grouplist_str[:-1]))
    grouplist_str = grouplist_str[:-1]

    return batches_per_rank_epoch, grouplist_str

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)



