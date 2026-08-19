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
    """Splits a batch of images into a sequence of flattened, non-overlapping patches.

    Args:
        data: Input tensor, shape (Batch, Channel, X, Y) for 2D or (Batch, Channel,
            X, Y, Z) for 3D.
        patch_size: Side length of each square/cube patch.
        twoD: Whether `data` is 2D (True) or 3D (False).

    Returns:
        Tensor of shape (Batch, num_patches, patch_size**2 * Channel) for 2D, or
        (Batch, num_patches, patch_size**3 * Channel) for 3D.
    """
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
    """Reassembles a sequence of flattened patches back into a batch of images.

    Inverse of `patchify`.

    Args:
        patchified_pixel_values: Patch sequence tensor, shape (Batch, num_patches,
            patch_size**2 * Channel) for 2D, or (Batch, num_patches, patch_size**3 *
            Channel) for 3D.
        data: Reference tensor whose shape gives the target channel count and
            original spatial dimensions, e.g. (Batch, Channel, X, Y[, Z]).
        patch_size: Side length of each square/cube patch.
        twoD: Whether the data is 2D (True) or 3D (False).

    Returns:
        Reconstructed image tensor, shape (Batch, Channel, X, Y) for 2D or (Batch,
        Channel, X, Y, Z) for 3D.
    """
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

def configure_optimizer(model, optimizer_type, optimizer_kwargs):
    """Builds a PyTorch optimizer for a model's parameters.

    Args:
        model: Model whose `.parameters()` will be optimized.
        optimizer_type: Optimizer type, case-insensitive; one of "sgd", "adam",
            "adamw".
        optimizer_kwargs: Keyword arguments passed through to the optimizer
            constructor (e.g. `lr`, `betas`, `weight_decay`).

    Returns:
        The instantiated `torch.optim.Optimizer`.
    """
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    return optimizer

def configure_scheduler(optimizer, scheduler_type, scheduler_kwargs):
    """Builds a learning rate scheduler for an optimizer.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: One of "constant", "linear", "exponential",
            "linear-warmup-cosine-annealing", "reduce-lr-on-plateau".
        scheduler_kwargs: Keyword arguments passed through to the scheduler
            constructor.

    Returns:
        The instantiated learning rate scheduler.
    """

    if scheduler_type == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **scheduler_kwargs)
    elif scheduler_type == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_kwargs)
    elif scheduler_type == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
    elif scheduler_type == "linear-warmup-cosine-annealing":
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, **scheduler_kwargs)
    elif scheduler_type == "reduce-lr-on-plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

    return lr_scheduler

def interpolate_pos_embed_adaptive(model, checkpoint_model, new_size=127):
    """Interpolates adaptive-patching positional embeddings from a checkpoint to a new length.

    Resizes the "pos_embed" and, if present, "decoder_pos_embed" entries of
    `checkpoint_model` (in place) via 1D linear interpolation along the sequence
    dimension, so they can be loaded into a model whose sequence length differs from
    the checkpoint's.

    Args:
        model: Unused; kept for interface compatibility with `interpolate_pos_embed`.
        checkpoint_model: State dict loaded from a checkpoint; modified in place.
        new_size: Target sequence length for the interpolated embeddings.
    """
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

def init_par_groups(world_rank, data_par_size, tensor_par_size, fsdp_size, simple_ddp_size):
    """Creates the distributed process groups used for hybrid tensor/data/FSDP parallelism.

    Partitions the world of ranks (assumed laid out as tensor-parallel-size-major, i.e.
    consecutive ranks belong to the same tensor-parallel group) into tensor-parallel
    groups, FSDP shard groups, simple-DDP replica groups, a combined DDP group per
    tensor-parallel index, and an orthogonal "data_seq" group that collects one rank
    per tensor-parallel group across the data-parallel dimension.

    Args:
        world_rank: Global rank of the current process.
        data_par_size: Number of data-parallel replicas (`fsdp_size * simple_ddp_size`).
        tensor_par_size: Number of ranks participating in tensor parallelism.
        fsdp_size: Number of ranks over which parameters are FSDP-sharded within a
            replica.
        simple_ddp_size: Number of FSDP shard groups that are DDP-synchronized with
            each other.

    Returns:
        A tuple `(ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group,
        simple_ddp_group)` of `torch.distributed.ProcessGroup` objects (or None where
        the current rank does not belong to a given group), each containing only the
        groups that `world_rank` is a member of.
    """

    tensor_par_group = None

    for i in range(data_par_size):
        ranks = [j for j in range(i*tensor_par_size,(i+1)*tensor_par_size)]

        group = dist.new_group(ranks)
        if world_rank in ranks:
            tensor_par_group = group

    ddp_group = None

    fsdp_group = None

    simple_ddp_group = None

    for i in range(tensor_par_size):
        ranks = [i+j*tensor_par_size for j in range(data_par_size)]

        for k in range(simple_ddp_size):
            fsdp_begin_idx = k*fsdp_size
            fsdp_end_idx = (k+1)*fsdp_size
            fsdp_ranks = ranks[fsdp_begin_idx:fsdp_end_idx]

            group = dist.new_group(fsdp_ranks)
            if world_rank in fsdp_ranks:
                fsdp_group = group


        for k in range(fsdp_size):
            simple_ddp_begin_idx = k
            simple_ddp_end_idx = len(ranks)
            simple_ddp_ranks = ranks[simple_ddp_begin_idx:simple_ddp_end_idx:fsdp_size]

            group = dist.new_group(simple_ddp_ranks)

            if world_rank in simple_ddp_ranks:

                simple_ddp_group = group

        group = dist.new_group(ranks)

        if world_rank in ranks:

            ddp_group = group

    data_seq_ort_group = None

    for i in range(tensor_par_size):
        ranks = [i+tensor_par_size*j for j in range(data_par_size)]

        group = dist.new_group(ranks)

        if world_rank in ranks:
            data_seq_ort_group = group

    return ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group

def process_root_dirs(dataset, dict_root_dirs, data_par_size):
    """Builds per-data-parallel-group lists of image file paths for a dataset.

    For "imagenet", groups classes under each root directory into `data_par_size`
    (or fewer) buckets of combined class image lists. For other datasets, lists all
    files under each root directory's "imagesTr" subfolder, one entry per key in
    `dict_root_dirs`.

    Args:
        dataset: Dataset name, e.g. "imagenet" or another supported dataset key.
        dict_root_dirs: Dict mapping a dataset key to its root directory path.
        data_par_size: Number of data-parallel groups to split the imagenet classes
            across; unused for non-imagenet datasets.

    Returns:
        Dict mapping a group/dataset key to a list of file paths.
    """
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
    else:
        dict_lister_trains = { k: list(dp.iter.FileLister(os.path.join(root_dir, "imagesTr"))) for k, root_dir in dict_root_dirs.items() }
    return dict_lister_trains

def calculate_load_balancing_on_the_fly(conf, VERBOSE=False):
    """Computes how many DDP ranks and batches-per-epoch each dataset should get.

    Given the relative size (in tiles) of each dataset, allocates data-parallel ranks
    proportionally (rounding to whole ranks while ensuring every dataset gets at
    least one rank and the total matches `data_par_size`), then computes how many
    images/batches per epoch each rank and each dataloader worker should process.

    Args:
        conf: Parsed training configuration dict (as returned by `parse_config`).
        VERBOSE: If True, print intermediate values useful for filling in the
            `batches_per_rank_epoch` and `dataset_group_list` config entries.

    Returns:
        A tuple `(batches_per_rank_epoch, grouplist_str)` where `batches_per_rank_epoch`
        is a dict mapping each dataset key to the number of batches per rank per
        epoch, and `grouplist_str` is a colon-separated string of the number of DDP
        ranks assigned to each dataset, in the same order as `dict_root_dirs`.
    """

    dict_root_dirs = conf['data']['dict_root_dirs']
    dict_start_idx = conf['dataloader']['dict_start_idx']
    dict_end_idx = conf['dataloader']['dict_end_idx']
    img_size =  conf['data']['img_size']
    tile_size =  conf['data']['tile_size']
    twoD = conf['data']['twoD']
    batch_size = conf['dataloader']['batch_size']
    div = conf['tiling']['div']
    patch_size =  conf['data']['patch_size']
    dataset = conf['data']['dataset']
    num_total_ddp_ranks = conf['parallelism']['data_par_size']
    num_workers = conf['dataloader']['num_workers']

    #num_workers = 0, uses 1 worker. However, setting it to 0 means it uses the main process as the dataloader worker in the iterative dataloader.
    #Need to set the local version of num_workers in this function to 1 in this case to calculate load balancing correctly
    if num_workers == 0:
        num_workers = 1

    if dataset == "imagenet":
        resize = conf['dataset_options']['resize']
    else:
        resize = None

    dict_lister_trains = process_root_dirs(dataset, dict_root_dirs, num_total_ddp_ranks)

    num_total_tiles = []
    num_total_images = []
    tiles_per_image = []
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

        if len(tile_size) == 3: #3D images
            if twoD: #Slice on one of the dimensions
                #The current implementation slices on the z dimension but, could do x or y as well
                #TODO: Add an option on which dimension to slice
                tiles_per_image.append(div*div*img_size[2])

            else:
                tiles_per_image.append(div*div*div)

        else: #2D images
            tiles_per_image.append(div*div)

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
    num_images_per_rank_worker = []
    actual_num_images_per_rank = []
    for i in range(len(num_total_tiles)):
        num_images_per_rank.append(int(math.floor(num_total_images[i] / float(ddp_rank_ratio[i]))))
        num_images_per_rank_worker.append(int(math.floor(num_images_per_rank[i] / float(num_workers))))
        actual_num_images_per_rank.append(num_images_per_rank_worker[i]*num_workers)
    if VERBOSE:
        print("Num Images Per Rank", num_images_per_rank)
        print("Num Images Per Worker", num_images_per_rank_worker)
        print("Actual Num Images Per Rank", actual_num_images_per_rank)
    assert min(num_images_per_rank) >= 1.0, "Decrease number of GPUs, not all GPUs have at least one image"
    assert min(num_images_per_rank_worker) >= 1.0, "Decrease number of GPUs or num_workers, not all dataloader workers have at least one image"

    batches_per_rank = []
    batches_per_worker = []
    tiles_per_rank = []
    for i in range(len(num_total_tiles)):
        batches_per_rank.append(np.floor(actual_num_images_per_rank[i])*tiles_per_image[i]/batch_size)
        batches_per_worker.append(np.floor(num_images_per_rank_worker[i]*tiles_per_image[i]/batch_size))
        tiles_per_rank.append(np.floor(actual_num_images_per_rank[i])*tiles_per_image[i])
    if VERBOSE:
        print("Tiles Per Rank", tiles_per_rank)
        print("USE BELOW IN CONFIG FILE")
        print("batches_per_rank_epoch: {")
    batches_per_rank_epoch = {}
    if dataset == "imagenet":
        new_data = [("imagenet", int(min(batches_per_worker*num_workers)))]
        batches_per_rank_epoch.update(new_data)
    else:
        for i,k in enumerate(dict_lister_trains.keys()):
            new_data = [(k, int(batches_per_worker[i]*num_workers))]
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
    """Checks whether an integer is a power of two.

    Args:
        n: Integer to check.

    Returns:
        True if `n` is a nonzero power of two, False otherwise.
    """
    return (n != 0) and (n & (n-1) == 0)


def calculate_tile_overlap(overlap):
    """Splits a total per-dimension overlap into start/end padding amounts.

    Even overlap is split symmetrically; odd overlap pads one extra unit onto the
    end rather than the start.

    Args:
        overlap: Sequence of total overlap amounts, one per spatial dimension.

    Returns:
        A tuple `(start_overlap, end_overlap)` of lists, each the same length as
        `overlap`, giving the overlap to apply at the start and end of each
        dimension respectively.
    """
    start_overlap = []
    end_overlap = []
    for i in range(len(overlap)):
        if overlap[i] % 2 == 0:
            # Even overlap: symmetric padding
            start_overlap.append(overlap[i] // 2)
            end_overlap.append(overlap[i] // 2)
        else:
            # Odd overlap: asymmetric padding
            # Chose to pad one more to the end rather than the start. Could switch this order
            start_overlap.append(overlap[i] // 2)
            end_overlap.append(overlap[i] // 2 + 1)

    return start_overlap, end_overlap
