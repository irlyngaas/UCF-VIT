# USAGE: python visualize_adaptive.py ../configs/basic_ct/sap/base_config.yaml /lustre/orion/world-shared/nro108/anikat/dataset/Tr8_Training/imagesTr/image_100.nii ct1
# USAGE: python visualize_adaptive.py ../configs/imagenet/classification/base_config.yaml /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet/train/n01806143/n01806143_13402.JPEG imagenet
#
# 2D only: visualizes the quadtree produced by adaptive patching. There is no
# 3D (octree) equivalent here since FixedOctTree has no draw() method to
# overlay octree boundaries on a volume slice.

import argparse
import os
import sys
import yaml

import numpy as np
import torch
from pathlib import Path
import nibabel as nib

from UCF_VIT.dataloaders.transform import Patchify
from UCF_VIT.parse import parse_config
from PIL import Image
import cv2 as cv

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_config import init_single_process_dist

def read_process_file(path, dataset, num_channels_available, variables, imagenet_resize=None):
    """Reads and preprocesses a single image/CT slice-volume for visualization.

    Args:
        path: Path to the file to read.
        dataset: Dataset type, "imagenet" or "basic_ct".
        num_channels_available: Number of channels available; if 1 for "basic_ct",
            a channel dimension is added.
        variables: Unused for "imagenet"; kept for interface symmetry.
        imagenet_resize: `[height, width]` to resize to; only used for
            `dataset == "imagenet"`.

    Returns:
        Channel-first data array, or None if `dataset` is neither "imagenet" nor
        "basic_ct".
    """
    if dataset == "imagenet":
        data = Image.open(path).convert("RGB")
        data = np.array(data)
        # imagenet_resize is (height, width); cv2's dsize is natively
        # (width, height), so swap locally right here.
        data = cv.resize(data, dsize=[imagenet_resize[1],imagenet_resize[0]])
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

def get_data(data, dataset, tile_size_x, tile_size_y):
    """Extracts a single 2D tile to visualize.

    For "basic_ct" (a 3D volume sliced into 2D tiles), this is the first x/y
    tile of the first z-slice. For "imagenet", the full (already tile-sized)
    image is returned unchanged.

    Args:
        data: Data array as returned by `read_process_file`.
        dataset: Dataset type, "basic_ct" or "imagenet".
        tile_size_x: Tile size along the x dimension.
        tile_size_y: Tile size along the y dimension.

    Returns:
        For "basic_ct", the first x/y tile of the first z-slice of `data`. For
        "imagenet", `data` unchanged.
    """
    if dataset == "basic_ct":
        return data[:, 0:tile_size_x, 0:tile_size_y, 0]

    elif dataset == "imagenet":
        return data


def main():
    """Visualizes adaptive (quadtree) patchification of a single 2D image or CT slice.

    Reads the config path, image/volume path, and dataset key from `sys.argv[1:4]`,
    loads and tiles one 2D sample, adaptively patchifies it (with the edge map
    returned), and saves the detected edges and per-channel quadtree overlay
    images under an "images" directory.
    """
    config_path = sys.argv[1]
    img_path = sys.argv[2]
    dict_key = sys.argv[3]

    init_single_process_dist()
    args = argparse.Namespace(config=config_path)
    conf = parse_config(args, load_balance_offline=True)

    # parse_config only validates/keeps ap.fixed_length when ap.do_ap is True
    # (it's irrelevant to the model architecture otherwise), but this script
    # always previews adaptive patching regardless of do_ap, so read the raw
    # value directly instead of the (possibly do_ap-gated-to-None) parsed one.
    with open(config_path, 'r') as f:
        raw_conf = yaml.load(f, Loader=yaml.FullLoader)
    fixed_length = raw_conf['ap']['fixed_length']
    # Same reasoning as fixed_length above: interp_size is also
    # do_ap-gated-to-None in the parsed conf, so read it raw too. Falls back
    # to data.patch_size for configs that don't ship ap.interp_size at all
    # (i.e. every do_ap:False baseline this script can still be pointed at
    # to preview what adaptive patching would look like) -- interp_size is
    # only required in the config once do_ap:True is actually turned on.
    interp_size = raw_conf['ap'].get('interp_size', raw_conf['data'].get('patch_size'))

    dataset = conf['data']['dataset']

    if dataset == "imagenet":
        imagenet_resize = conf['dataset_options']['resize']
    else:
        imagenet_resize = None

    twoD = conf['data']['twoD']
    assert twoD, "This script only supports 2D visualization (FixedOctTree has no draw() method for 3D)"
    variables = conf['data']['dict_in_variables']
    tile_size = conf['data']['tile_size']
    assert fixed_length % 3 == 1 % 3, "Quadtree fixed length needs to be 3n+1, where n is some integer"

    tile_size_x = tile_size[0]
    tile_size_y = tile_size[1]

    if dataset == "imagenet":
        data = read_process_file(img_path, dataset, len(variables[dict_key]), variables[dict_key], imagenet_resize=imagenet_resize[dict_key])
    else:
        data = read_process_file(img_path, dataset, len(variables[dict_key]), variables[dict_key])

    np_image = get_data(data, dataset, tile_size_x, tile_size_y)

    #Default
    #smooth_factor = [0,1,3,5]
    smooth_factor = [1]
    #Default
    #canny1 = 50
    #canny2 = 100
    canny1 = 50
    canny2 = 51

    patchify = Patchify(sths=smooth_factor,cannys=[canny1,canny2],fixed_length=fixed_length, interp_size=interp_size, num_channels=len(variables[dict_key]), dataset=dataset, return_edges = True)

    seq_image, seq_size, seq_pos, qdt, edges = patchify(np.moveaxis(np_image,0,-1))
    print(seq_size)
    print("NNZ Patches: ", np.count_nonzero(seq_size))
    print(seq_pos)

    isExist = os.path.exists("images")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("images",exist_ok=True)

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


if __name__ == "__main__":
    main()
