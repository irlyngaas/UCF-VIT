import os
import sys
import numpy as np
import torch
import yaml
import nibabel as nib
from pathlib import Path
import math
import glob
from PIL import Image
import cv2 as cv


def main():
    config_path = sys.argv[1]
    num_total_ddp_ranks = int(sys.argv[2])

    print("config_path ",config_path,flush=True)

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)
    dict_root_dirs = conf['data']['dict_root_dirs']
    dict_start_idx = conf['data']['dict_start_idx']
    dict_end_idx = conf['data']['dict_end_idx']
    num_channels_available = conf['data']['num_channels_available']
    dict_in_variables = conf['data']['dict_in_variables']
    tile_size =  conf['model']['net']['init_args']['tile_size']
    twoD = conf['model']['net']['init_args']['twoD']
    num_channels_used = conf['data']['num_channels_used']
    single_channel = conf['data']['single_channel']
    batch_size = conf['data']['batch_size']
    tile_overlap = conf['data']['tile_overlap']
    use_all_data = conf['data']['use_all_data']
    patch_size =  conf['model']['net']['init_args']['patch_size']
    max_epochs=conf['trainer']['max_epochs']

    #seeds = np.random.randint(2**32-1,size=max_epochs)

    tile_size_x = int(tile_size[0])
    tile_size_y = int(tile_size[1])

    dict_lister_trains = {}
    for k, root_dir in dict_root_dirs.items():
        classes = sorted(os.listdir(root_dir))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        if len(classes) > num_total_ddp_ranks:
            classes_to_combine = int(len(classes) // num_total_ddp_ranks)
        print("CLASSES_TO_COMBINE", classes_to_combine)
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
                print("LEN_IMG_LIST",len(img_list))
                img_dict = {num_data_roots: img_list}
                dict_lister_trains.update(img_dict)
                num_data_roots +=1

            if num_data_roots > num_total_ddp_ranks-1:
                break
        
    print("KEYS", dict_lister_trains.keys())
    print("NUM_DATA_ROOTS", num_data_roots, flush=True)

    num_total_tiles = []
    num_total_images = []
    tiles_per_image = []
    num_channels_per_dataset = []
    for i, k in enumerate(dict_lister_trains.keys()):
        lister_train = dict_lister_trains[k]
        start_idx = int(dict_start_idx['imagenet'] * len(lister_train))
        end_idx = int(dict_end_idx['imagenet'] * len(lister_train))
        keys = lister_train[start_idx:end_idx]
        num_total_images.append(len(keys))

        #Assume all channels have the same data size
        data_path = keys[0]
        data = Image.open(data_path).convert("RGB")
        data = np.array(data) 
        data = cv.resize(data, dsize=[256,256])
        #data = np.moveaxis(data,-1,0)

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

        print("KEY", k, "DATA_SHAPE", data.shape,"NUM_BLOCKS:", num_blocks_x, num_blocks_y, flush=True)

        tiles_per_image.append(num_blocks_x*num_blocks_y)
        
        num_channels_per_dataset.append(num_channels_used["imagenet"])
        if single_channel:
            num_total_tiles.append(tiles_per_image[i] * num_channels_per_dataset[i] * num_total_images[i])
        else:
            num_total_tiles.append(tiles_per_image[i] * num_total_images[i])

    print("Total Images", num_total_images)
    print("Tiles Per Image", tiles_per_image)
    print("Total Tiles per Dataset", num_total_tiles)
    print("Total Tiles", sum(num_total_tiles))
    print("Total Tokens", sum(num_total_tiles)*(tile_size_x/patch_size)*(tile_size_y/patch_size))
        
    total_tiles_all_data = sum(num_total_tiles)
        
    ddp_ratio = []
    ddp_rank_ratio = []
    ratio_diff = []
    for i in range(len(num_total_tiles)):
        ratio = num_total_tiles[i]/total_tiles_all_data
        ddp_ratio.append(ratio*num_total_ddp_ranks)
        ddp_rank_ratio.append(int(np.rint(ddp_ratio[i])))
        ratio_diff.append(ddp_rank_ratio[i] - ddp_ratio[i])
    print("DDP RATIO", ddp_ratio)
    print("DDP RANK RATIO", ddp_rank_ratio)

    rank_sum = sum(ddp_rank_ratio)
    #First Rebalance, if necessary
    if rank_sum != num_total_ddp_ranks:
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
                    if leftover[i] > leftover[rank_to_decrease]:
                        rank_to_decrease = i
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
            print("Rank to increase", rank_to_increase)
            ddp_rank_ratio[rank_to_increase] += 1

    rank_sum = sum(ddp_rank_ratio)

    #Second Rebalance, if necessary, could add more rebalances
    if rank_sum != num_total_ddp_ranks:
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
                    if leftover[i] > leftover[rank_to_decrease]:
                        rank_to_decrease = i
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
            print("Rank to increase", rank_to_increase)
            ddp_rank_ratio[rank_to_increase] += 1

    print("DDP RANKS:", ddp_rank_ratio)
    assert rank_sum == num_total_ddp_ranks, "All DDP ranks not used"

    num_images_per_rank = []
    for i in range(len(num_total_tiles)):
        num_images_per_rank.append(int(math.floor(num_total_images[i] / float(ddp_rank_ratio[i]))))
    print("Num Images Per Rank", num_images_per_rank)
    assert min(num_images_per_rank) >= 1.0, "Decrease number of GPUs, not all GPUs have their own image"

    batches_per_rank = []
    tiles_per_rank = []
    for i in range(len(num_total_tiles)):
        if single_channel:
            batches_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i]*num_channels_per_dataset[i]/batch_size)
        else:
            batches_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i]/batch_size)
        tiles_per_rank.append(np.floor(num_images_per_rank[i])*tiles_per_image[i])
    print("Tiles Per Rank", tiles_per_rank)
    print("Batches Per Rank", batches_per_rank)
    print("Min Batches Per Rank", min(batches_per_rank))

if __name__ == "__main__":
    main()
