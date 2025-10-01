# USAGE: python visualize_adaptive.py ../configs/basic_ct/sap/base_config.yaml /lustre/orion/world-shared/nro108/anikat/dataset/Tr8_Training/imagesTr/image_100.nii ct1
# USAGE: python visualize_adaptive.py ../configs/imagenet/classification/base_config.yaml /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet/train/n01806143/n01806143_13402.JPEG imagenet

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

def read_process_file(path, dataset, num_channels_available, variables, imagenet_resize=None):
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

def get_data(data, dataset, twoD, tile_size_x, tile_size_y, tile_size_z):
    if dataset == "basic_ct":
        kk = 0 #which tile in z dim
        jj = 0 #which tile in y dim
        ii = 0 #which tile in x dim
        x_step_size = tile_size_x #self.tile_size_x-tile_overlap_size_x
        y_step_size = tile_size_y #self.tile_size_y-tile_overlap_size_y
        if twoD:
            kkk = 0 #which slice
            return data[:, ii*x_step_size:tile_size_x+ii*x_step_size, jj*y_step_size:tile_size_y+jj*y_step_size, kkk+kk*tile_size_z]
        else:
            z_step_size = tile_size_z #self.tile_size_z-tile_overlap_size_z
            return data[:, ii*x_step_size:tile_size_x+ii*x_step_size, jj*y_step_size:tile_size_y+jj*y_step_size, kk*z_step_size:tile_size_z+kk*z_step_size]

    elif dataset == "imagenet":
        return data


def main():
    config_path = sys.argv[1]
    img_path = sys.argv[2]
    dict_key = sys.argv[3]

    conf = yaml.load(open(config_path,'r'),Loader=yaml.FullLoader)

    dataset = conf['data']['dataset']

    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['imagenet_resize']
    else:
        imagenet_resize = None

    twoD = conf['model']['net']['init_args']['twoD']
    if dataset == "imagenet":
        assert twoD, "twoD must be True if using imagenet"
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

    if dataset == "imagenet":
        data = read_process_file(img_path, dataset, len(variables[dict_key]), variables[dict_key], imagenet_resize=imagenet_resize[dict_key])
    else:
        data = read_process_file(img_path, dataset, len(variables[dict_key]), variables[dict_key])

    np_image = get_data(data, dataset, twoD, tile_size_x, tile_size_y, tile_size_z)

    #Default
    #smooth_factor = [0,1,3,5]
    smooth_factor = [1]
    #Default
    #canny1 = 50
    #canny2 = 100
    canny1 = 50
    canny2 = 51

    if twoD:
        patchify = Patchify(sths=smooth_factor,cannys=[canny1,canny2],fixed_length=fixed_length, patch_size=patch_size, num_channels=len(variables[dict_key]), dataset=dataset, return_edges = True)
    else:
        patchify = Patchify_3D(sths=smooth_factor,cannys=[canny1,canny2],fixed_length=fixed_length, patch_size=patch_size, num_channels=len(variables[dict_key]), dataset=dataset, return_edges = True)

    seq_image, seq_size, seq_pos, qdt, edges = patchify(np.moveaxis(np_image,0,-1))
    print(seq_size)
    print("NNZ Patches: ", np.count_nonzero(seq_size))
    print(seq_pos)

    isExist = os.path.exists("images")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("images",exist_ok=True)
    if twoD:
        fig, ax = plt.subplots()
        ax.imshow(edges)
        qdt.draw(ax=ax)
        plt.savefig(f'images/edges.png', bbox_inches='tight', dpi=200)
        if dataset != "imagenet":
            for i in range(len(variables[dict_key])):
                fig, ax = plt.subplots()
                ax.imshow(np_image[i])
                qdt.draw(ax=ax)
                plt.savefig(f'images/qdt_image_{variables[dict_key][i]}.png', bbox_inches='tight', dpi=200)
        else:
            fig, ax = plt.subplots()
            ax.imshow(np.moveaxis(np_image,0,-1))
            qdt.draw(ax=ax)
            plt.savefig(f'images/qdt_image.png', bbox_inches='tight', dpi=200)
    else:
        z_slice = 32
        fig, ax = plt.subplots()
        ax.imshow(edges[:,:,z_slice])
        #qdt.draw(ax=ax)
        plt.savefig(f'images/edges.png', bbox_inches='tight', dpi=200)
        for i in range(len(variables[dict_key])):
            fig, ax = plt.subplots()
            ax.imshow(np_image[i,:,:,z_slice])
            #qdt.draw(ax=ax)
            plt.savefig(f'images/qdt_image_{variables[dict_key][i]}.png', bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    main()








