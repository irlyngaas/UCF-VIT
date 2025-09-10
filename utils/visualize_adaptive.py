# USAGE: python visualize_adaptive.py ../configs/sst/pred/base_config.yaml /lustre/orion/stf006/world-shared/muraligm/CFD135/data_iso/super_res/binary_data/P1F4R32_nx512ny512nz256_6vars/28.040000

import yaml
import os
import sys

import numpy as np
import torch
from pathlib import Path
import nibabel as nib

from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D
from PIL import Image
import cv2 as cv

from matplotlib import pyplot as plt
import math

def read_process_file(path, dataset, imagenet_resize, num_channels_available, nx, ny, nz, variables):
    if dataset == "imagenet":
        data = Image.open(path).convert("RGB")
        data = np.array(data) 
        data = cv.resize(data, dsize=[imagenet_resize[0],imagenet_resize[1]])
        data = np.moveaxis(data,-1,0)


        return data

    elif dataset == "basic_ct":
        data = nib.load(path)
        data = np.array(data.dataobj).astype(np.float32)
        data = (data-data.min())/(data.max()-data.min())

        if num_channels_available == 1:
            return np.expand_dims(data,axis=0)
        else:
            return data

    elif dataset == "sst":
        root_path = Path(path)
        parent = root_path.parent
        stem = path.split('/')[-1]
        data_list = []
        for i in range(len(variables)):
            channel_path = os.path.join(parent, variables[i]+"_"+stem)
            data_memmap = np.memmap(channel_path, dtype=np.float32, mode='r', shape=(nz, ny, nx+2))
            data_list.append(data_memmap)

        return data_list

def get_data(data, dataset, twoD, tile_size_x, tile_size_y, tile_size_z, nx_skip, ny_skip, nz_skip, chunk_size):
    if dataset == "sst":
        chunk_offset_z = 0 #chunk_idx[0]*chunk_size[0]
        chunk_offset_y = 0 #chunk_idx[1]*chunk_size[1]
        chunk_offset_x = 0 #chunk_idx[2]*chunk_size[2]
        kk = 0 #which tile in z dim
        jj = 0 #which tile in y dim
        ii = 0 #which tile in x dim
        x_step_size = tile_size_x #self.tile_size_x-tile_overlap_size_x
        y_step_size = tile_size_y #self.tile_size_y-tile_overlap_size_y
        if twoD:
            kkk = 0 #which slice
            datalist = []
            for cc in range(len(data)):
                data_cube = data[cc][chunk_offset_z+kkk+kk*tile_size_z, chunk_offset_y+jj*y_step_size:chunk_offset_y+(tile_size_y*ny_skip)+jj*y_step_size:ny_skip, chunk_offset_x+ii*x_step_size:chunk_offset_x+(tile_size_x*nx_skip)+ii*x_step_size:nx_skip]
                datalist.append(data_cube.copy().transpose(1,0))
            return np.stack(datalist, axis=0)
        else:
            x_step_size = tile_size_x #self.tile_size_x-tile_overlap_size_x
            y_step_size = tile_size_y #self.tile_size_y-tile_overlap_size_y
            z_step_size = tile_size_z #self.tile_size_z-tile_overlap_size_z
            datalist = []
            for cc in range(len(data)):
                data_cube = data[cc][chunk_offset_z+kk*z_step_size:chunk_offset_z+(tile_size_z*nz_skip)+kk*z_step_size:nz_skip, chunk_offset_y+jj*y_step_size:chunk_offset_y+(tile_size_y*ny_skip)+jj*y_step_size:ny_skip,chunk_offset_x+ii*x_step_size:chunk_offset_x+(tile_size_x*nx_skip)+ii*x_step_size:nx_skip]
                datalist.append(data_cube.copy().transpose(2,1,0))
            return np.stack(datalist, axis=0)

def main():
    config_path = sys.argv[1]
    img_path = sys.argv[2]
    dict_key = 'P1F4R32'

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    dataset = conf['data']['dataset']

    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['imagenet_resize']
    else:
        imagenet_resize = None

    if dataset == "sst":
        nx = conf['dataset_options']['nx']

        ny = conf['dataset_options']['ny']

        nz = conf['dataset_options']['nz']

        nx_skip = conf['dataset_options']['nx_skip']

        ny_skip = conf['dataset_options']['ny_skip']

        nz_skip = conf['dataset_options']['nz_skip']

        chunk_size = conf['dataset_options']['chunk_size']
    else:
        nx = None

        ny = None

        nz = None

        nx_skip = None

        ny_skip = None

        nz_skip = None
        
        chunk_size = None

    twoD = conf['model']['net']['init_args']['twoD']
    variables = conf['data']['dict_in_variables']
    tile_size =  conf['model']['net']['init_args']['tile_size']
    patch_size =  conf['model']['net']['init_args']['patch_size']
    fixed_length = conf['model']['net']['init_args']['fixed_length']
    if twoD:
        assert fixed_length % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"
    else:
        sqrt_len=int(np.rint(math.pow(fixed_length,1/3)))
        assert fixed_length % 7 == 1 % 7, "Octtree fixed length needs to be 7n+1, where n is some integer"

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]
    if dataset == "imagenet":
        tile_size_z = None
    else:
        tile_size_z = tile_size[2]

    data = read_process_file(img_path, dataset, imagenet_resize, len(variables[dict_key]), nx[dict_key], ny[dict_key], nz[dict_key], variables[dict_key])

    np_image = get_data(data, dataset, twoD, tile_size_x, tile_size_y, tile_size_z, nx_skip[dict_key], ny_skip[dict_key], nz_skip[dict_key], chunk_size[dict_key])
    print(np_image.shape)

    smooth_factor = [0,1,3,5,7]
    canny1 = 50
    canny2 = 100

    if twoD:
        patchify = Patchify(sths=smooth_factor,cannys=[canny1,canny2],fixed_length=fixed_length, patch_size=patch_size, num_channels=len(variables[dict_key]), dataset=dataset, return_edges = True)
    else:
        patchify = Patchify_3D(sths=smooth_factor,cannys=[canny1,canny2],fixed_length=fixed_length, patch_size=patch_size, num_channels=len(variables[dict_key]), dataset=dataset, return_edges = True)

    seq_image, seq_size, seq_pos, qdt, edges = patchify(np.moveaxis(np_image,0,-1))
    print(seq_size)
    print("NNZ", np.count_nonzero(seq_size))
    print(seq_pos)
    print(edges.shape)

    if twoD:
        fig, ax = plt.subplots()
        ax.imshow(edges)
        qdt.draw(ax=ax)
        plt.savefig(f'images/edges.png', bbox_inches='tight', dpi=200)
        for i in range(len(variables[dict_key])):
            fig, ax = plt.subplots()
            ax.imshow(np_image[i])
            qdt.draw(ax=ax)
            plt.savefig(f'images/qdt_image_{variables[dict_key][i]}.png', bbox_inches='tight', dpi=200)
    else:
        z_slice = 0
        fig, ax = plt.subplots()
        ax.imshow(edges[:,:,z_slice])
        plt.savefig(f'images/edges.png', bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    main()








