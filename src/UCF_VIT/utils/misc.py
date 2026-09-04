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

def find_repo_root():
    """Returns this repository's root directory (the parent of `src/`).

    Computed from *this file's own* location (`src/UCF_VIT/utils/misc.py` is
    always exactly 3 directories below the repo root), not the caller's --
    so any script, anywhere in the repo (including ones added later, at any
    depth someone chooses), gets the same correct answer just by importing
    and calling this, without needing to work out its own location relative
    to the repo root itself.

    Config files' own relative paths (`trainer.checkpoint_path`,
    `inference_output.output_dir`) are resolved against this in `parse.py`
    (not against the process's current working directory, which depends on
    wherever a launch script happens to invoke `python` from) -- see
    `parse_config`'s own comments at each of those fields for the exact
    resolution.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

def shard_mlp_state_dict(full_state_dict, tensor_par_size, tp_rank):
    """Slices a full (tensor_par_size=1) UCF_VIT.model.building_blocks.Mlp
    state_dict into the shard TP rank `tp_rank` of a `tensor_par_size`-way
    tensor-parallel group should load.

    Mirrors Mlp's actual sharding: `fc1`'s hidden dimension (dim 0 of its
    weight/bias) is row-sliced into `tensor_par_size` contiguous chunks, and
    `fc2`'s hidden dimension (dim 1 of its weight) is column-sliced the same
    way, matching `fc1 = Linear(in_features, hidden_features //
    tensor_par_size)` / `fc2 = Linear(hidden_features // tensor_par_size,
    out_features)`.

    `fc2.bias` is a special case: Mlp does not shard it (every rank's `fc2`
    keeps a full-size `out_features` bias, since `out_features` itself is
    never divided by `tensor_par_size`), and Mlp.forward's post-fc2
    all-reduce (`F_AllReduce_B_Identity(..., op=SUM, ...)`) sums every
    rank's fc2 output -- including that bias -- across the group. To
    reconstruct the reference bias exactly (not `tensor_par_size` copies of
    it), only `tp_rank == 0` keeps the real bias; every other rank's shard
    gets a zero bias of the same shape, so the sum across the group is
    unaffected. This is purely a test-construction technique for comparing
    a sharded forward pass against a known reference.

    Args:
        full_state_dict: `state_dict()` of an `Mlp` built with
            `tensor_par_size=1` (i.e. unsharded, "reference" weights).
        tensor_par_size: Number of tensor-parallel ranks to shard across.
        tp_rank: This rank's index within its tensor-parallel group
            (`0 <= tp_rank < tensor_par_size`).

    Returns:
        A dict with the same keys as `full_state_dict`, sliced (or, for
        `fc2.bias` on `tp_rank != 0`, zeroed) for TP rank `tp_rank`.
    """
    hidden_features = full_state_dict["fc1.weight"].shape[0]
    assert hidden_features % tensor_par_size == 0, (
        f"hidden_features ({hidden_features}) must be divisible by tensor_par_size ({tensor_par_size})"
    )
    shard = hidden_features // tensor_par_size
    start, end = tp_rank * shard, (tp_rank + 1) * shard

    sharded = {}
    sharded["fc1.weight"] = full_state_dict["fc1.weight"][start:end].clone()
    if "fc1.bias" in full_state_dict:
        sharded["fc1.bias"] = full_state_dict["fc1.bias"][start:end].clone()
    sharded["fc2.weight"] = full_state_dict["fc2.weight"][:, start:end].clone()
    if "fc2.bias" in full_state_dict:
        if tp_rank == 0:
            sharded["fc2.bias"] = full_state_dict["fc2.bias"].clone()
        else:
            sharded["fc2.bias"] = torch.zeros_like(full_state_dict["fc2.bias"])
    return sharded


def shard_attention_state_dict(full_state_dict, num_heads, tensor_par_size, tp_rank):
    """Slices a full (tensor_par_size=1) UCF_VIT.model.building_blocks.Attention
    state_dict into the shard TP rank `tp_rank` of a `tensor_par_size`-way
    tensor-parallel group should load.

    Mirrors Attention's actual sharding, which shards `num_heads`, not a
    flat row range of `qkv`'s output: `qkv`'s output (dim 0 of its
    weight/bias, size `dim * 3`) is laid out as 3 contiguous blocks (Q, K,
    V, each size `dim`), and Attention.forward reshapes each rank's *own*
    `qkv` output as `(3, num_heads // tensor_par_size, head_dim)` -- i.e.
    every rank's shard must contain the SAME contiguous range of heads
    from each of Q, K, and V, not one contiguous slice of the flattened
    `dim * 3` vector (which would span all of Q plus part of K for low
    tensor_par_size, and part of K plus all of V for the last rank --
    completely wrong). `proj`'s input dimension (dim 1 of its weight,
    `dim`) IS just a plain contiguous column-slice by head range, though:
    Attention.forward's `x.reshape(B, N, C // tensor_par_size)` before
    `self.proj(x)` flattens this rank's own `num_heads // tensor_par_size`
    heads head-major, and head ranges are assigned to ranks in contiguous
    order, so `tp_rank`'s head range and its `dim // tensor_par_size`-sized
    contiguous column range of the reference `proj.weight` coincide.

    `proj.bias` is a special case, exactly like Mlp's `fc2.bias` (see
    `shard_mlp_state_dict`): Attention does not shard it (`proj`'s output
    dimension, `dim`, is never divided by `tensor_par_size`), and
    Attention.forward's post-proj `dist.all_reduce(..., op=SUM, ...)` sums
    every rank's proj output -- including that bias -- across the group.
    Only `tp_rank == 0` keeps the real bias; every other rank's shard gets a
    zero bias.

    Only supports `qk_norm=False` (the default): `q_norm`/`k_norm` are then
    `nn.Identity` with no parameters, so there is nothing to shard or copy.
    If `full_state_dict` contains `q_norm.*`/`k_norm.*` keys, this raises
    rather than silently dropping them -- those parameters operate on
    `head_dim`, which tensor parallelism does not shard (only `num_heads`
    is sharded), so they would need to be copied verbatim, not sliced; not
    implemented here since no shipped config currently uses `qk_norm=True`.

    Args:
        full_state_dict: `state_dict()` of an `Attention` built with
            `tensor_par_size=1` (i.e. unsharded, "reference" weights).
        num_heads: Number of attention heads (the same value the reference
            `Attention` was built with) -- needed to locate head boundaries
            within `qkv`'s output; not derivable from `full_state_dict`'s
            shapes alone (`dim = num_heads * head_dim` has no unique
            factorization).
        tensor_par_size: Number of tensor-parallel ranks to shard across.
        tp_rank: This rank's index within its tensor-parallel group
            (`0 <= tp_rank < tensor_par_size`).

    Returns:
        A dict with the same keys as `full_state_dict`, sliced (or, for
        `proj.bias` on `tp_rank != 0`, zeroed) for TP rank `tp_rank`.

    Raises:
        NotImplementedError: If `full_state_dict` has `q_norm.*`/`k_norm.*`
            keys (i.e. came from an `Attention` built with `qk_norm=True`).
    """
    unsupported = [k for k in full_state_dict if k.startswith("q_norm.") or k.startswith("k_norm.")]
    if unsupported:
        raise NotImplementedError(
            f"shard_attention_state_dict only supports qk_norm=False (no q_norm/k_norm "
            f"parameters); got keys {unsupported}"
        )

    dim = full_state_dict["proj.weight"].shape[1]  # proj.weight is (dim, dim) at tensor_par_size=1
    assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
    head_dim = dim // num_heads
    assert num_heads % tensor_par_size == 0, (
        f"num_heads ({num_heads}) must be divisible by tensor_par_size ({tensor_par_size})"
    )
    heads_per_shard = num_heads // tensor_par_size
    head_start, head_end = tp_rank * heads_per_shard, (tp_rank + 1) * heads_per_shard
    elem_start, elem_end = head_start * head_dim, head_end * head_dim  # == tp_rank/(tp_rank+1) * (dim // tensor_par_size)

    def _qkv_head_slice(full_tensor):
        # full_tensor's dim 0 is 3 contiguous dim-sized blocks (Q, K, V);
        # take the same head range out of each block, then re-concatenate
        # in Q, K, V order -- matching Attention.forward's own (3,
        # num_heads // tensor_par_size, head_dim) reshape of this rank's
        # qkv output.
        blocks = [full_tensor[block_idx * dim:(block_idx + 1) * dim][elem_start:elem_end] for block_idx in range(3)]
        return torch.cat(blocks, dim=0)

    sharded = {}
    sharded["qkv.weight"] = _qkv_head_slice(full_state_dict["qkv.weight"]).clone()
    if "qkv.bias" in full_state_dict:
        sharded["qkv.bias"] = _qkv_head_slice(full_state_dict["qkv.bias"]).clone()
    sharded["proj.weight"] = full_state_dict["proj.weight"][:, elem_start:elem_end].clone()
    if "proj.bias" in full_state_dict:
        if tp_rank == 0:
            sharded["proj.bias"] = full_state_dict["proj.bias"].clone()
        else:
            sharded["proj.bias"] = torch.zeros_like(full_state_dict["proj.bias"])
    return sharded


def process_root_dirs(dataset, dict_root_dirs, data_par_size=None):
    """Builds per-dataset-key lists of image file paths.

    For "imagenet", lists every image under each root directory (grouped by
    class, in deterministic sorted-class/sorted-image order), one entry per key
    in `dict_root_dirs`. For other datasets, lists all files under each root
    directory's "imagesTr" subfolder, same shape.

    Deliberately *not* bucketed by rank count here -- that would make *which
    files count as train/val/test* depend on `data_par_size` (a real problem on
    a checkpoint restart with a different node count, or running `val.py`/
    `test.py` at a different parallelism than the training run being evaluated).
    Bucketing happens via `bucket_file_list`, called by `calculate_load_
    balancing_on_the_fly`/`NativePytorchDataModule.__init__` *after* they apply
    `dict_start_idx`/`dict_end_idx` slicing to this function's (sorted,
    deterministic) output -- so split membership never depends on rank count,
    only how it's divided across ranks does.

    Args:
        dataset: Dataset name, e.g. "imagenet" or another supported dataset key.
        data_par_size: Unused -- kept only so existing callers passing it
            positionally don't need updating.

    Returns:
        Dict mapping each `dict_root_dirs` key to its (sorted) list of file
        paths.
    """
    if dataset == "imagenet":
        dict_lister_trains = {}
        for k, root_dir in dict_root_dirs.items():
            classes = sorted(os.listdir(root_dir))
            img_list = []
            for cls_name in classes:
                cls_dir = os.path.join(root_dir, cls_name)
                img_list.extend(sorted(glob.glob(os.path.join(cls_dir, "*.JPEG"))))
            dict_lister_trains[k] = img_list
    else:
        dict_lister_trains = { k: list(dp.iter.FileLister(os.path.join(root_dir, "imagesTr"))) for k, root_dir in dict_root_dirs.items() }
    return dict_lister_trains


def bucket_file_list(file_list, num_buckets, shuffle_seed=None):
    """Splits `file_list` into up to `num_buckets` roughly-equal, deterministic chunks.

    Used to divide a dataset key's *already train/val/test-sliced* file list into
    per-DDP-rank-group buckets for imagenet -- kept separate from slicing itself
    (`slice_file_list`) so it can run identically, on the same resolved
    membership, from both `calculate_load_balancing_on_the_fly` (per-bucket rank
    ratios) and `NativePytorchDataModule.__init__` (real per-bucket dataloader
    pipelines). The two must agree exactly on bucket count/order
    (`train_dataloader` matches a rank to its bucket via `gx` against
    `dict_data_train`'s enumeration order) -- calling this one shared function
    guarantees that by construction.

    Args:
        file_list: List of file paths, already sliced to the desired train/
            val/test membership (order doesn't matter for bucket *sizes*, but
            both callers pass an already-sorted list so bucket *contents* also
            end up identical between the two, not just bucket counts).
        num_buckets: Requested number of buckets.
        shuffle_seed: Optional int. If given, `file_list` is shuffled (a seeded,
            deterministic permutation, independent of `data_par_size`/restarts)
            before chunking, so each bucket gets a representative cross-section
            instead of a contiguous range. Matters for imagenet: `file_list`
            arrives sorted class-by-class (`process_root_dirs`), so without
            shuffling, each bucket/rank only sees a narrow range of classes
            every epoch (skews BatchNorm, correlates gradients). `None`
            preserves the original ordering.

    Returns:
        Dict mapping bucket index (`0` to `min(num_buckets, len(file_list)) - 1`,
        or just `{0: []}` if `file_list` is empty) to that bucket's file list --
        every file appears in exactly one bucket, bucket sizes differing by at
        most one file. Never fewer than 1 file per bucket purely from
        over-bucketing (capped to `len(file_list)` buckets instead) -- an empty
        bucket for that reason would otherwise trip `FileReader`'s zero-files
        guard even under `allow_file_reuse`, which is meant for "not enough
        files for this many ranks", not "zero files for this key at all".
    """
    if not file_list:
        return {0: []}
    if shuffle_seed is not None:
        file_list = list(file_list)
        np.random.RandomState(shuffle_seed).shuffle(file_list)
    actual_buckets = min(num_buckets, len(file_list))
    chunks = np.array_split(np.array(file_list, dtype=object), actual_buckets)
    return {i: list(chunks[i]) for i in range(actual_buckets)}

def slice_file_list(file_list, start_idx, end_idx):
    """Slices `file_list` to the `[start_idx, end_idx)` fraction requested.

    Mirrors `FileReader.__init__`'s own slicing formula exactly (see
    `UCF_VIT.dataloaders.dataset.FileReader`) -- the map-style `"dataloader"`
    path (catsdogs) has no `FileReader` of its own, so `train.py`/`val.py`/
    `test.py` call this directly on the globbed file list instead, to apply the
    same train/val/test split ratios `parse_config`'s `_resolve_dataset_splits`
    already resolved for the `iterative_dataloader` path.

    Args:
        file_list: List of file paths, in listing order (no shuffle applied
            here or by the caller -- the slice is deterministic).
        start_idx: Fraction (0.0-1.0) of `file_list` to start reading from.
        end_idx: Fraction (0.0-1.0) of `file_list` to stop reading at.

    Returns:
        The `[int(start_idx*len(file_list)), int(end_idx*len(file_list)))`
        slice of `file_list`.
    """
    start = int(start_idx * len(file_list))
    end = int(end_idx * len(file_list))
    return file_list[start:end]

def _find_representative_file(root_dir, key, what):
    """Finds one real file to sample for auto-detecting a per-dataset-key
    property (channel count, image size, ...), using the same
    `imagesTr/`-listing convention `process_root_dirs` uses to list
    non-imagenet datasets.

    Args:
        root_dir: Root directory for this dataset key (a
            `dict_root_dirs` value).
        key: The dataset key `root_dir` belongs to, only used to make the
            error message specific.
        what: Short description of what's being detected, only used to
            make the error message specific (e.g. "num_channels").

    Returns:
        Path to one real file under `root_dir/imagesTr/`.

    Raises:
        FileNotFoundError: If `root_dir/imagesTr/` is missing or empty.
    """
    images_dir = os.path.join(root_dir, "imagesTr")
    try:
        filename = sorted(os.listdir(images_dir))[0]
    except (FileNotFoundError, IndexError) as e:
        raise FileNotFoundError(
            f"Could not auto-detect {what} for dataset key '{key}': "
            f"no files found under {images_dir}. Set {what} manually "
            f"in the config for this key instead."
        ) from e
    return os.path.join(images_dir, filename)

def detect_num_channels(dataset, dict_root_dirs):
    """Auto-detects the number of channels for each dataset key by reading
    one real representative data file, for use when
    conf['data']['num_channels'] is omitted from a training config.

    Behavior is dataset-type-specific, matching how each dataset's own
    dataset.py.read_process_file (or datasets/catsdogs.py's __getitem__)
    actually decodes files:
      - "imagenet": always 3, no file read needed -- dataset.py's
        read_process_file always calls Image.open(path).convert("RGB"),
        forcing 3 channels regardless of the source file.
      - Every other dataset (e.g. "catsdogs", "basic_ct"): reads one real
        file, found the same way process_root_dirs lists non-imagenet
        datasets (the first file under dict_root_dirs[k]/imagesTr/).
        - "basic_ct": reads the file's shape via nibabel's lazy
          ArrayProxy (nib.load(path).shape reads only the NIfTI header,
          not the voxel data, so this is cheap even for large volumes).
          A 3D shape means 1 channel, matching dataset.py's
          read_process_file num_channels_available==1 branch (which
          prepends a channel dim via np.expand_dims). A 4D+ shape is
          ambiguous: dataset.py's own handling of num_channels_available
          > 1 assumes the raw array is already channel-first with no
          verified real-file example anywhere in this repo (no shipped
          config or test exercises basic_ct with num_channels > 1), so
          this raises rather than guessing a channel-axis convention.
        - Other datasets (e.g. "catsdogs"): reads the file's band count
          via PIL (Image.open(path).getbands()), a cheap, lazy read that
          doesn't decode full pixel data and works regardless of the
          file's mode (RGB, RGBA, grayscale, ...).

    Args:
        dataset: Dataset name ("imagenet", "catsdogs", "basic_ct", ...).
        dict_root_dirs: Dict mapping each dataset key to its root
            directory path.

    Returns:
        Dict mapping each dataset key to its detected channel count,
        shaped like conf['data']['num_channels'].

    Raises:
        FileNotFoundError: If a key's imagesTr/ directory is missing or
            empty.
        RuntimeError: If a basic_ct file's shape doesn't unambiguously
            imply a channel count (4 or more dimensions).
    """
    if dataset == "imagenet":
        return {k: 3 for k in dict_root_dirs}

    num_channels = {}
    for k, root_dir in dict_root_dirs.items():
        path = _find_representative_file(root_dir, k, "num_channels")

        if dataset == "basic_ct":
            shape = nib.load(path).shape
            if len(shape) == 3:
                num_channels[k] = 1
            else:
                raise RuntimeError(
                    f"Could not auto-detect num_channels for dataset key '{k}': "
                    f"{path} has shape {shape} (not 3D), so the channel axis is "
                    f"ambiguous -- no verified convention for multi-channel "
                    f"basic_ct files exists in this repo yet. Set num_channels "
                    f"manually in the config for this key instead."
                )
        else:
            num_channels[k] = len(Image.open(path).getbands())

    return num_channels

def detect_img_size(dataset, dict_root_dirs):
    """Auto-detects the real/native size of the data by reading one real
    representative file, for use when conf['data']['img_size'] is
    omitted from a training config.

    Unlike num_channels (a per-dataset-key dict), img_size is a single
    flat 2- or 3-element list shared across the whole dataset ([height,
    width] for imagenet/catsdogs -- see the "Every other dataset" bullet
    below for why; axis order as-is for basic_ct) -- parse.py reads it
    once and uses it directly (e.g. for tile_size), never per
    dict_root_dirs key. Detection therefore samples
    just one representative file from the *first* dict_root_dirs key,
    mirroring the same "first key wins" convention parse.py's own
    num_channels-to-in_chans reduction already uses (dict iteration order
    in Python 3.7+ is insertion order, so this is deterministic).

    Unlike num_channels, img_size is meaningful to auto-detect for every
    dataset type: it always represents the actual size of the real data,
    never a resize target (resizing to a different size than native is a
    separate, optional step -- see parse.py's dataset_options.resize
    handling).

    Behavior, using the same representative-file lookup as
    detect_num_channels (dict_root_dirs[k]/imagesTr/):
      - "basic_ct": the file's shape via nibabel's lazy ArrayProxy
        (nib.load(path).shape reads only the NIfTI header, not the voxel
        data, so this is cheap even for large volumes) -- returned as-is,
        matching how img_size's elements are used directly as array axes
        throughout tiling/patching code, with no axis reordering.
      - Every other dataset (e.g. "imagenet", "catsdogs"): the file's
        pixel dimensions via PIL (Image.open(path).size, a cheap, lazy
        read that doesn't decode full pixel data). PIL's own .size is
        natively (width, height); this function swaps it to (height,
        width) before returning, matching the convention used everywhere
        else in this codebase for real tensors/arrays (PyTorch's
        (B,C,H,W), numpy's (H,W,C)) -- img_size/resize/tile_size are
        height-first throughout the config and data-loading layers.
        cv2's dsize parameter (used by dataset.py's/catsdogs.py's own
        cv.resize calls) is the only place that still needs width-first
        order, since that's cv2's own native convention -- those call
        sites swap back to width-first locally, right at the cv2 call.

    Args:
        dataset: Dataset name ("imagenet", "catsdogs", "basic_ct", ...).
        dict_root_dirs: Dict mapping each dataset key to its root
            directory path.

    Returns:
        The detected size as a list of ints, shaped like
        conf['data']['img_size'].

    Raises:
        FileNotFoundError: If dict_root_dirs is empty, or its first key's
            imagesTr/ directory is missing or empty.
    """
    try:
        k, root_dir = next(iter(dict_root_dirs.items()))
    except StopIteration:
        raise FileNotFoundError("Could not auto-detect img_size: dict_root_dirs is empty.")
    path = _find_representative_file(root_dir, k, "img_size")

    if dataset == "basic_ct":
        return list(nib.load(path).shape)
    else:
        width, height = Image.open(path).size
        return [height, width]

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
    # patch_size is unused (None) when do_ap:True -- interp_size takes over
    # its role in that mode, same as everywhere else that dispatches on
    # adaptive_patching (see UCF_VIT.model.arch.VIT.effective_patch_size).
    effective_patch_size = conf['data']['interp_size'] if conf['ap']['do_ap'] else conf['data']['patch_size']
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

    dict_lister_trains = process_root_dirs(dataset, dict_root_dirs)

    # For imagenet, dict_start_idx/dict_end_idx's ratio slice (on a sorted,
    # deterministic order) must happen *before* any rank-count-dependent bucketing,
    # or which images count as train/val/test would depend on data_par_size -- see
    # process_root_dirs's/bucket_file_list's own comments. dict_root_dirs always has
    # exactly one key for imagenet, so slicing then re-keying dict_lister_trains by
    # bucket index replaces that single entry entirely -- the loop below's imagenet
    # branch just consumes each bucket's already-correct membership as-is.
    # Non-imagenet keys are unaffected (1:1 dict_root_dirs->dict_lister_trains, no
    # bucketing); their slicing happens unchanged in the loop below.
    if dataset == "imagenet":
        k = next(iter(dict_lister_trains))
        sliced = slice_file_list(sorted(dict_lister_trains[k]), dict_start_idx["imagenet"], dict_end_idx["imagenet"])
        assert len(sliced) > 0, f"Dataset '{k}' has zero files -- check dict_root_dirs/dict_start_idx/dict_end_idx."
        dict_lister_trains = bucket_file_list(sliced, num_total_ddp_ranks, shuffle_seed=conf['dataloader'].get('bucket_shuffle_seed', 42))

    num_total_tiles = []
    num_total_images = []
    tiles_per_image = []
    for i, k in enumerate(dict_lister_trains.keys()):
        lister_train = dict_lister_trains[k]
        if dataset == "imagenet":
            # Already sliced to the correct train/val/test membership and
            # bucketed above.
            keys = lister_train
        else:
            start_idx = int(dict_start_idx[k] * len(lister_train))
            end_idx = int(dict_end_idx[k] * len(lister_train))
            keys = sorted(lister_train)[start_idx:end_idx]
            # Fails clearly here rather than falling through to a bare
            # ZeroDivisionError further down (total_tiles_all_data would be 0) --
            # a dataset key resolving to zero files is always a config problem,
            # regardless of allow_file_reuse (which can't reuse files that don't
            # exist -- see its own comment further down).
            assert len(keys) > 0, f"Dataset '{k}' has zero files -- check dict_root_dirs/dict_start_idx/dict_end_idx."
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
            print("Total Tokens", sum(num_total_tiles)*(tile_size[0]/effective_patch_size)*(tile_size[1]/effective_patch_size))
        else:
            print("Total Tokens", sum(num_total_tiles)*(tile_size[0]/effective_patch_size)*(tile_size[1]/effective_patch_size)*(tile_size[2]/effective_patch_size))
        
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

    # Optional -- default False (see parse.py's own "allow_file_reuse" comment for
    # the full rationale). A dataset key with fewer files than DDP ranks/workers
    # assigned to it isn't just a small/debug-dataset concern -- at the node counts
    # this repo targets, data_par_size can exceed a real dataset's file count too.
    allow_file_reuse = conf['dataloader'].get('allow_file_reuse', False)

    num_images_per_rank = []
    num_images_per_rank_worker = []
    actual_num_images_per_rank = []
    dict_keys = list(dict_lister_trains.keys())
    for i in range(len(num_total_tiles)):
        # num_total_images[i] > 0 already asserted above, right where it's built.
        this_num_images_per_rank = int(math.floor(num_total_images[i] / float(ddp_rank_ratio[i])))
        this_num_images_per_rank_worker = int(math.floor(this_num_images_per_rank / float(num_workers)))
        if allow_file_reuse:
            if this_num_images_per_rank_worker < 1 and dist.get_rank() == 0:
                print(f"WARNING: dataset '{dict_keys[i]}' has only {num_total_images[i]} files for "
                      f"{ddp_rank_ratio[i]} assigned DDP ranks x {num_workers} workers "
                      f"({num_total_images[i] / (ddp_rank_ratio[i] * num_workers):.3g} files per rank/worker) -- "
                      f"allow_file_reuse:True means files will be reused (duplicated) across ranks/workers this "
                      f"epoch, each rank/worker getting exactly 1 file.", flush=True)
            this_num_images_per_rank = max(1, this_num_images_per_rank)
            this_num_images_per_rank_worker = max(1, this_num_images_per_rank_worker)
        num_images_per_rank.append(this_num_images_per_rank)
        num_images_per_rank_worker.append(this_num_images_per_rank_worker)
        actual_num_images_per_rank.append(this_num_images_per_rank_worker*num_workers)
    if VERBOSE:
        print("Num Images Per Rank", num_images_per_rank)
        print("Num Images Per Worker", num_images_per_rank_worker)
        print("Actual Num Images Per Rank", actual_num_images_per_rank)
    assert min(num_images_per_rank) >= 1.0, "Decrease number of GPUs, not all GPUs have at least one image. Or set dataloader.allow_file_reuse: True to let ranks reuse (duplicate) files instead."
    assert min(num_images_per_rank_worker) >= 1.0, "Decrease number of GPUs or num_workers, not all dataloader workers have at least one image. Or set dataloader.allow_file_reuse: True to let workers reuse (duplicate) files instead."

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

    # The asserts above only guarantee at least 1 *image* per rank/worker -- not
    # at least 1 full *batch* (drop_last=True means fewer than batch_size images
    # yields zero batches, not a smaller one). Without this check that silently
    # propagates into a bare ZeroDivisionError deep in NativePytorchDataModule.
    # _compute_keys_to_add instead of a clear message here. Real way to hit
    # this: the automatic train/val/test split (val_split_ratio/test_split_ratio)
    # can push an already-tight allocation below one batch/rank even though it
    # had enough images before the split.
    zero_batch_keys = [k for k, v in batches_per_rank_epoch.items() if v < 1]
    assert not zero_batch_keys, (
        f"Dataset key(s) {zero_batch_keys} yield 0 batches per rank with "
        f"dataloader.batch_size={batch_size} -- too few images per rank/worker after "
        f"DDP-rank/worker sharding, and (if dataloader.val_split_ratio/test_split_ratio are "
        f"set) after the automatic train/val/test split narrowed how many images go to "
        f"training. Increase the available data, decrease dataloader.batch_size, decrease "
        f"parallelism.data_par_size/dataloader.num_workers, set "
        f"dataloader.allow_file_reuse: True to let ranks/workers reuse (duplicate) images "
        f"instead, or lower dataloader.val_split_ratio/test_split_ratio so more data stays "
        f"in training."
    )

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

def calculate_tile_bounds(tile_idx, div, tile_size, overlap_start, overlap_end):
    """Calculates the [start, end) bounds of one tile along a single dimension.

    Extracted from `UCF_VIT.dataloaders.dataset.TileDataIter` (an
    `IterableDataset`, which expands one sample into `div * div` tiles on
    the fly) so the exact same tile-bounds math can also be reused by
    `UCF_VIT.datasets.catsdogs.CatsDogsDataset` (a map-style `Dataset`,
    which instead maps a flat `__getitem__` index to one `(file, tile)`
    pair -- see its own docstring for why tiling needs a different
    mechanism there).

    Args:
        tile_idx: Index of this tile along this dimension (0-based).
        div: Number of tiles this dimension is divided into. `div == 1`
            means no tiling -- the whole dimension is returned unchanged.
        tile_size: Size of one tile in this dimension, excluding overlap.
        overlap_start: Overlap amount to add at the start of this
            dimension's tile grid.
        overlap_end: Overlap amount to add at the end of this dimension's
            tile grid.

    Returns:
        A `(start, end)` tuple of bounds for this tile.
    """
    if div == 1:
        # No tiling: use full dimension
        return 0, tile_size
    # Base tile boundaries without overlap
    start = tile_size * tile_idx
    end = tile_size * (tile_idx + 1)

    # Add overlap based on tile position
    if tile_idx == 0:
        # First tile: only overlap on right
        end += overlap_start * 2
    elif tile_idx == div - 1:
        # Last tile: only overlap on left
        start -= overlap_end * 2
    else:
        # Middle tiles: overlap on both sides
        start -= overlap_start
        end += overlap_end
    return start, end
