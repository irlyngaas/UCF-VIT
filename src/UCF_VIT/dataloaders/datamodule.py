import functools
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

from .dataset import (
    FileReader,
    TileDataIter,
    ShuffleIterableDataset,
    ProcessChannels,
)
from UCF_VIT.utils.misc import bucket_file_list, slice_file_list
from UCF_VIT.utils.misc import process_root_dirs as process_root_dirs_shared

def collate_fn(batch, return_label, adaptive_patching, separate_channels, dataset, num_classes, num_labels, return_qdt, dict_key):
    """Collate function for `NativePytorchDataModule`'s iterative dataloaders.

    Stacks per-sample numpy arrays into batched tensors, handling several
    combinations of adaptive patching, labeling, and dataset type (e.g. one-hot
    encoding segmentation masks for "basic_ct", stacking per-channel label lists for
    other segmentation datasets).

    Args:
        batch: List of samples as yielded by the underlying `ProcessChannels`
            iterable dataset.
        return_label: Whether samples include a label and whether it should be
            included in the returned tuple.
        adaptive_patching: Whether samples include adaptive-patching sequence data
            (patch sequence, size, and position) in addition to the raw tile.
        separate_channels: Whether adaptive patching was done per channel
            (True) or jointly across channels (False); affects how `size`/`pos`
            are stacked.
        dataset: Dataset name, e.g. "imagenet" or "basic_ct"; determines label/
            seq_label handling.
        num_classes: Number of segmentation classes, used to one-hot encode masks
            for "basic_ct".
        num_labels: Number of per-channel label tensors to stack for non-"basic_ct"
            segmentation datasets.
        return_qdt: Whether to also collect and return the list of quadtree/octree
            objects for each sample.
        dict_key: Dataset key to attach to the batch, returned as the final tuple
            element.

    Returns:
        A tuple of batched tensors/values whose exact composition depends on
        `adaptive_patching`, `return_label`, `dataset`, and `return_qdt`; always ends
        with `dict_key`.
    """
    if adaptive_patching:
        if return_label:
            inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])

            #TODO: Generalize this
            if dataset == "basic_ct":
                if separate_channels:
                    seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
                else:
                    seq = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
            else:
                seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])

            #TODO: Finish and Test separate_channels implementation
            if separate_channels:
                size = torch.stack([torch.from_numpy(batch[i][2]) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(batch[i][3]) for i in range(len(batch))])
            else:
                size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])

            if dataset == "imagenet":
                label = torch.stack([torch.tensor(batch[i][4]) for i in range(len(batch))])
                variables = batch[0][5]
                if return_qdt:
                    qdt_list = []
                    for i in range(len(batch)):
                        qdt_list.append(batch[i][6])
            else:
                label = torch.stack([torch.from_numpy(batch[i][4]) for i in range(len(batch))])
                seq_label_list = []
                for i in range(len(batch)):
                    if dataset == "basic_ct":
                        seq_mask = torch.from_numpy(batch[i][5][0]).long()
                        seq_mask = F.one_hot(seq_mask.squeeze(-1), num_classes=num_classes)
                        seq_label_list.append(seq_mask.permute(2, 0, 1).float())
                    else:
                        seq_label_list.append([])
                        for j in range(num_labels):
                            seq_label_list[i].append(torch.from_numpy(batch[i][5][j]))
                if dataset == "basic_ct":
                    seq_label = torch.stack([seq_label_list[i] for i in range(len(seq_label_list))])
                else:
                    channel_list = []
                    for i in range(len(batch)):
                        channel_list.append(torch.stack([seq_label_list[i][j] for j in range(num_labels)]))
                    seq_label = torch.stack([channel_list[i] for i in range(len(batch))])

                variables = batch[0][6]
                if return_qdt:
                    qdt_list = []
                    for i in range(len(batch)):
                        qdt_list.append(batch[i][7])

            if dataset == "imagenet":                
                if return_qdt:
                    return (inp, seq, size, pos, label, variables, qdt_list, dict_key)
                else:
                    return (inp, seq, size, pos, label, variables, dict_key)
            else:
                if return_qdt:
                    return (inp, seq, size, pos, label, seq_label, variables, qdt_list, dict_key)
                else:
                    return (inp, seq, size, pos, label, seq_label, variables, dict_key)
        else:
            inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
            #TODO: Generalize this
            if dataset == "basic_ct":
                if separate_channels:
                    seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
                else:
                    seq = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
            else:
                seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
            #TODO: Finish and Test separate_channels implementation
            if separate_channels:
                size = torch.stack([torch.from_numpy(batch[i][2]) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(batch[i][3]) for i in range(len(batch))])
            else:
                size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])
            variables = batch[0][4]

            if return_qdt:
                qdt_list = []
                for i in range(len(batch)):
                    qdt_list.append(batch[i][5])
                return (inp, seq, size, pos, variables, qdt_list, dict_key)
            else:
                return (inp, seq, size, pos, variables, dict_key)
    else:
        if return_label:
            inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
            if dataset == "imagenet":
                label = torch.stack([torch.tensor(batch[i][1]) for i in range(len(batch))])
            else:
                if num_labels == 1:
                    label = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
                else:
                    label = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
            variables = batch[0][2]
                
            return (inp, label, variables, dict_key)
        else:
            inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
            variables = batch[0][1]

            return (inp, variables, dict_key)

class NativePytorchDataModule(torch.nn.Module):
    """Native pytorch data module for multi-source data.

    Args:
        dict_root_dirs (Dict): Dictionary of root directories for each source.
        dict_start_idx (Dict): Dictionary of start indices ratio (between 0.0 and 1.0) for each source.
        dict_end_idx (Dict): Dictionary of end indices ratio (between 0.0 and 1.0) for each source.
        dict_buffer_sizes (Dict): Dictionary of shuffle-buffer sizes for each source, used by
            `ShuffleIterableDataset`.
        dict_in_variables (Dict): Dictionary of input modality variables for each source
        num_channels_used (Dict): Dictionary of number of channels used from each source.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
        interp_size (int, optional): Side length each adaptive leaf patch is
            interpolated to, used by the adaptive-patching transform
            (`Patchify`/`Patchify_3D`), when `adaptive_patching` is True.
        tile_size (tuple[int,...], optional): the tile size in each dimension
        twoD (bool, optional): Variable for indicating two or three dimensionsal input, if False, three dimensional input.
        dataset_group_list (string, optional): How to split available GPUs amongst the available datasets, run "python utils/preprocess_load_balancing.py CONFIG_FILE NUM_GPUS" to obtain
        batches_per_rank_epoch (Dict, optional): Dict mapping each dataset key to the number of
            batches per rank per epoch (as returned by `calculate_load_balancing_on_the_fly`),
            used in `setup`/`reset` to determine how many times each dataset's file listing
            needs to be replicated to balance dataset sizes.
        div (int, optional): How many tiles to divide each image into
        tile_overlap (tuple[int,...], optional): Amount of tile overlapping to use in each dimension. Use 0 in each dimension for no overlapping
        adaptive_patching (bool, optional): Whether to adaptively patchify each sample via a
            quadtree/octree instead of returning fixed-size tiles.
        fixed_length (int, optional): Fixed output sequence length for adaptive patching, when
            `adaptive_patching` is True.
        separate_channels (bool, optional): Whether adaptive patching is done independently per
            channel (True) or jointly across all channels (False).
        data_par_size (int, optional): the size of the data parallelism
        dataset (str, optional): Dataset name, e.g. "imagenet" or "basic_ct"; determines how
            files are listed and how samples are processed/collated.
        return_label (bool, optional): Whether or not the dataloader returns segmentation labels
        return_qdt (bool, optional): Whether to also yield the quadtree/octree object(s) used for
            each adaptively-patched sample.
        ddp_group (ProcessGroup, optional): Process group used to determine this rank's assigned
            dataset in `train_dataloader`; defaults to the default world group if None.
        num_classes (int, optional): Number of segmentation classes; required when `dataset` is
            "basic_ct" and `return_label` is True, to one-hot encode masks in `collate_fn`.
        resize (Dict, optional): Dict mapping "imagenet" to the `[height, width]` size to resize
            images to (cv2's own `dsize` convention is `(width, height)`; `dataset.py`'s
            `FileReader.read_process_file` swaps locally right before its `cv.resize` call);
            only used when `dataset` is "imagenet". If absent or has no "imagenet" entry,
            images are left at their native size.
        multiprocessing_context (str, optional): `DataLoader`'s own `multiprocessing_context`
            argument -- `None` (the default) leaves PyTorch's own default in place (`fork` on
            Linux). Only worth setting to `"spawn"` for a config combining `num_workers > 0`
            with heavy per-sample CPU work (e.g. adaptive-patching a 3D volume) run alongside
            `tensor_par_size > 1`: forking a worker process after CUDA/NCCL is already
            initialized in the parent (as happens here -- `get_model` runs before
            `train_dataloader` in every training script) is a known hazard (PyTorch's own
            docs warn about it) -- CUDA/NCCL keep background threads that can hold a lock at
            the instant of the fork, which the child then inherits stuck forever, causing a
            segfault the next time it touches libc's allocator (i.e. almost immediately, in
            unrelated-looking code). Root-caused this way after a real, intermittent
            `basic_ct`+`SAP`+`tensor_par_size:2` Frontier segfault (job 5390076) that predated
            this option -- see `configs/basic_ct/sap/base_config.yaml`'s own
            `multiprocessing_context: "spawn"` for the one shipped config that actually needs
            it. `"spawn"` costs real per-worker startup latency (a fresh Python interpreter
            re-imports the whole `torch`/`timm`/`monai`/`xformers` chain), so it's opt-in per
            config rather than a new global default.
        allow_file_reuse (bool, optional): If False (default), a dataset key with fewer
            files than the DDP ranks/workers assigned to it fails loudly
            (`calculate_load_balancing_on_the_fly`'s and `FileReader.__iter__`'s own
            asserts) rather than silently letting some ranks/workers train on no data
            at all. If True, every rank/worker gets at least one file instead, reusing
            (duplicating) files round-robin as needed, with a printed warning
            quantifying how much reuse is happening. Not just a small/debug-dataset
            concern -- at the node counts this repo targets, `data_par_size` can
            exceed a real dataset's file count too. See `FileReader.__iter__`'s own
            comment for the reuse mechanism.
        bucket_shuffle_seed (int, optional): Imagenet only -- seeds the shuffle
            `bucket_file_list` applies (to the already train/val/test-sliced image
            list) before dividing it into per-DDP-rank-group buckets. Without this,
            bucketing is a contiguous split of a class-sorted list, so each bucket
            (and therefore each rank) only ever sees a narrow range of classes every
            epoch -- a real concern for data-parallel SGD (class-homogeneous local
            batches skew BatchNorm statistics and correlate gradients within a
            rank's own step sequence). A fixed seed keeps this fully deterministic
            (same seed always gives the same shuffle, independent of
            `data_par_size`/process restarts, same as everything else about the
            split) while giving each bucket a representative cross-section of
            classes instead. `None` preserves the original contiguous ordering.
    """

    def __init__(
        self,
        dict_root_dirs: Dict = None,
        dict_start_idx: Dict = None,
        dict_end_idx: Dict = None,
        dict_buffer_sizes: Dict = None,
        dict_in_variables: Dict = None,
        num_channels_used: Dict = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        interp_size: int = 16,
        tile_size: tuple[int, ...] = (64, 64),
        twoD: bool = True,
        dataset_group_list: str = '',
        batches_per_rank_epoch: Dict = None,
        div: int = 1,
        tile_overlap: tuple[int, ...] = (0, 0),
        adaptive_patching: bool = False,
        fixed_length: int = 4096,
        separate_channels: bool = False,
        data_par_size: int = 1,
        dataset: str = "imagenet",
        return_label: Optional[bool] = False,
        return_qdt: Optional[bool] = False,
        ddp_group: Optional[dist.ProcessGroup] = None,
        num_classes: Optional[int] = None,
        resize: Optional[Dict] = None,
        multiprocessing_context: Optional[str] = None,
        allow_file_reuse: bool = False,
        bucket_shuffle_seed: Optional[int] = None,
    ):
        """Initializes the data module and builds the per-dataset file listings.

        See the class docstring for a description of each argument. Splits
        data-parallel ranks across datasets according to `dataset_group_list` (or
        evenly if not given) and calls `process_root_dirs` to list each dataset's
        files.
        """
        super().__init__()

        assert len(dict_root_dirs) <= data_par_size, "the number of data parallel GPUs (data_par_size) needs to be at least equal to the number of datasets. Try to increase data_par_size"

        #Default: Split ddp ranks evenly across datasets
        if dataset_group_list == '':
            self.gx = ":".join(["%d"%(data_par_size//len(dict_root_dirs)),]*len(dict_root_dirs))
        else:
            self.gx = dataset_group_list

        self.dict_root_dirs = dict_root_dirs
        self.dict_start_idx = dict_start_idx
        self.dict_end_idx = dict_end_idx
        self.dict_buffer_sizes = dict_buffer_sizes 
        self.num_channels_used = num_channels_used
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multiprocessing_context = multiprocessing_context
        self.allow_file_reuse = allow_file_reuse
        self.bucket_shuffle_seed = bucket_shuffle_seed
        self.interp_size = interp_size
        self.tile_size = tile_size
        self.twoD = twoD
        self.return_label = return_label
        self.return_qdt = return_qdt
        self.batches_per_rank_epoch = batches_per_rank_epoch
        self.div = div
        self.tile_overlap = tile_overlap
        self.adaptive_patching = adaptive_patching
        self.fixed_length = fixed_length
        self.separate_channels = separate_channels
        self.data_par_size = data_par_size
        self.ddp_group = ddp_group
        self.dataset = dataset

        #Optional Inputs
        self.num_classes = num_classes
        if self.dataset == "basic_ct":
            if return_label:
                assert num_classes != None, "If using segmentation with basic_ct need to pass the number of classes"

        if self.dataset == "imagenet":
            self.resize = resize

        in_variables = {}
        for k, list_out in dict_in_variables.items():
            if list_out is not None:
                in_variables[k] = list_out
            #TODO: Add checking and mapping for in_variables
            #in_variables[k] = [ x for x in in_variables[k] if x in DEFAULT_VARIABLE_LIST ]
            in_variables[k] = [ x for x in in_variables[k] ]
        self.dict_in_variables = in_variables

        self.dict_lister_trains = self.process_root_dirs()

        # Fixes train/val/test membership once, from a deterministic (sorted) file
        # order, before setup()/reset() ever run their own epoch-to-epoch reshuffle
        # (np.random.choice, unseeded -- needed for genuine per-epoch training
        # variety and per-rank sharding fairness, so deliberately not removed).
        # Without this, dict_start_idx/dict_end_idx's ratio slice (applied inside
        # FileReader, called from set_iterative_dataloader below) would be applied
        # to a freshly-randomized order on every single setup()/reset() call --
        # harmless when dict_start_idx/dict_end_idx was always the full [0,1) range
        # (every shipped config, before auto train/val/test splitting existed), but
        # real data leakage once it can be a genuine partial range: which files
        # counted as "train" vs held-out "val"/"test" would silently change on every
        # checkpoint restart, and even between epochs of the *same* run. Sorting
        # here also makes this independent of process_root_dirs's own listing order
        # (os.listdir/glob.glob/FileLister aren't guaranteed stable across calls).
        # set_iterative_dataloader passes a no-op start_idx=0.0/end_idx=1.0 to
        # FileReader now that slicing happens here instead.
        for k in self.dict_lister_trains:
            if self.dataset == "imagenet":
                start_idx, end_idx = self.dict_start_idx["imagenet"], self.dict_end_idx["imagenet"]
            else:
                start_idx, end_idx = self.dict_start_idx[k], self.dict_end_idx[k]
            self.dict_lister_trains[k] = slice_file_list(sorted(self.dict_lister_trains[k]), start_idx, end_idx)

        if self.dataset == "imagenet":
            # Buckets are purely a rank/GPU-group assignment mechanism
            # (FileReader.read_process_file derives each image's label from its
            # own parent directory at read time, not from which bucket it's
            # in) -- bucketing here, after the slice above (process_root_dirs
            # used to bucket internally, *before* any slicing, which is
            # exactly what made membership depend on data_par_size), keeps
            # train/val/test membership independent of data_par_size entirely:
            # only how the already-resolved membership gets divided across
            # ranks depends on it now, not which images are in it. Must call
            # bucket_file_list identically (same already-sorted-and-sliced
            # input, same data_par_size) to calculate_load_balancing_on_the_fly's
            # own call, or train_dataloader's gx-based bucket selection breaks
            # -- see bucket_file_list's own docstring for why.
            k = next(iter(self.dict_lister_trains))
            self.dict_lister_trains = bucket_file_list(self.dict_lister_trains[k], self.data_par_size, shuffle_seed=self.bucket_shuffle_seed)

        self.dict_data_train: Optional[Dict] = None

    def process_root_dirs(self):
        """Builds per-dataset-key lists of image file paths for `self.dataset`.

        Thin wrapper over the shared `UCF_VIT.utils.misc.process_root_dirs` --
        used to also be a separate, near-duplicate implementation of the same
        imagenet-bucketing logic, which had to be manually kept in sync with
        `calculate_load_balancing_on_the_fly`'s own bucketing (both ultimately
        feed `train_dataloader`'s `gx`-based rank-to-bucket selection, which
        requires the two to agree on bucket count/order exactly). Delegating to
        one shared function eliminates that synchronization risk by
        construction. Bucketing itself (imagenet only) now happens later, in
        `__init__`, *after* `dict_start_idx`/`dict_end_idx` slicing -- see that
        code's own comment for why.

        Returns:
            Dict mapping each `self.dict_root_dirs` key to its (sorted) list of
            file paths.
        """
        return process_root_dirs_shared(self.dataset, self.dict_root_dirs)

    def set_iterative_dataloader(self, dict_data_train, k, lister_train, keys_to_add):
        """Builds the iterable dataset pipeline (file read -> tile -> shuffle -> channel processing) for one dataset key.

        Args:
            dict_data_train: Dict of dataset-key -> iterable dataset to update in
                place with the new pipeline for `k`.
            k: Dataset key to build the pipeline for.
            lister_train: List of file paths for this dataset key (already
                shuffled/replicated by the caller as needed).
            keys_to_add: Number of times `lister_train` was replicated to balance
                dataset sizes; passed through to `FileReader` as `keys_to_add`.

        Returns:
            `dict_data_train`, with `dict_data_train[k]` set to the new
            `ProcessChannels`-wrapped iterable dataset.
        """
        # start_idx/end_idx are a no-op [0.0, 1.0) here -- dict_start_idx/dict_end_idx
        # were already applied once, deterministically, in __init__ (before
        # setup()/reset()'s own epoch-to-epoch reshuffle of lister_train could
        # re-randomize which files count as this key's train/val/test membership --
        # see __init__'s own comment for the full rationale). FileReader still takes
        # start_idx/end_idx as real parameters for its other, direct callers/tests.
        start_idx = 0.0
        end_idx = 1.0
        if self.dataset == "imagenet":
            buffer_size = self.dict_buffer_sizes["imagenet"]
            variables = self.dict_in_variables["imagenet"]
            num_channels_used = self.num_channels_used["imagenet"]
            resize = self.resize.get("imagenet") if self.resize else None
        else:
            buffer_size = self.dict_buffer_sizes[k]
            variables = self.dict_in_variables[k]
            num_channels_used = self.num_channels_used[k]
        return_label = self.return_label
        if self.dataset == "imagenet":
            dict_data_train[k] = ProcessChannels(
                ShuffleIterableDataset(
                    TileDataIter(
                            FileReader(
                                lister_train,
                                gx = self.gx,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                data_par_size=self.data_par_size,
                                return_label=return_label,
                                keys_to_add=keys_to_add,
                                ddp_group=self.ddp_group,
                                dataset=self.dataset,
                                resize=resize,
                                allow_file_reuse=self.allow_file_reuse,
                            ),
                        self.tile_size,
                        self.twoD,
                        return_label = return_label,
                        div = self.div,
                        tile_overlap = self.tile_overlap,
                        classification = True,
                    ),
                    buffer_size
                ),
                num_channels_used,
                self.batch_size,
                return_label,
                self.adaptive_patching,
                self.separate_channels,
                self.interp_size,
                self.fixed_length,
                self.twoD,
                self.dataset,
                self.return_qdt,
            )
        else:
            dict_data_train[k] = ProcessChannels(
                ShuffleIterableDataset(
                    TileDataIter(
                            FileReader(
                                lister_train,
                                gx = self.gx,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                data_par_size = self.data_par_size,
                                return_label = return_label,
                                keys_to_add = keys_to_add,
                                ddp_group = self.ddp_group,
                                dataset=self.dataset,
                                allow_file_reuse=self.allow_file_reuse,
                            ),
                        self.tile_size,
                        self.twoD,
                        return_label = return_label,
                        div = self.div,
                        tile_overlap = self.tile_overlap,
                    ),
                    buffer_size
                ),
                num_channels_used,
                self.batch_size,
                return_label,
                self.adaptive_patching,
                self.separate_channels,
                self.interp_size,
                self.fixed_length,
                self.twoD,
                self.dataset,
                self.return_qdt,
            )
        return dict_data_train
        

    def setup(self):
        """Builds the iterable training datasets for every dataset key, if not already built.

        Computes `self.max_balance`, the largest `batches_per_rank_epoch` across
        datasets, then replicates each dataset's file listing enough times
        (`keys_to_add`) so that dataloading can continue reusing files until the
        largest dataset is exhausted, and builds each dataset's pipeline via
        `set_iterative_dataloader`. No-op if `self.dict_data_train` is already set.
        """
        # load datasets only if they're not loaded already
        if not self.dict_data_train:

            #Choice to made at this point. Imagenet uses 1) The default option is to use 2)
            #1) Use the dataset with the smallest amount of data tiles. In this case dataloading stops once all tiles are yielded from the smallest dataset
            #2) Add more files to each dataset. Allowing dataloading to continue reusing files from the dataset until all tiles are yielded from the largest dataset
            self.max_balance = 0
            if self.dataset == "imagenet":
                self.max_balance = self.batches_per_rank_epoch["imagenet"]
            else:
                for i, k in enumerate(self.dict_lister_trains.keys()):
                    if self.batches_per_rank_epoch[k] > self.max_balance:
                          self.max_balance = self.batches_per_rank_epoch[k]

            dict_data_train = {}
            for i, k in enumerate(self.dict_lister_trains.keys()):
                lister_train = self.dict_lister_trains[k]
                if self.dataset == "imagenet":
                    keys_to_add = 1
                else:
                    keys_to_add = int(np.ceil(self.max_balance/self.batches_per_rank_epoch[k]))
                _lister_train = np.random.choice(lister_train, size=len(lister_train), replace=False).tolist()
                if keys_to_add > 1:
                    for i in range(keys_to_add-1):
                        _balance_train = np.random.choice(lister_train, size=len(lister_train), replace=False).tolist()
                        _lister_train.extend(_balance_train)

                lister_train = _lister_train

                dict_data_train = self.set_iterative_dataloader(dict_data_train, k, lister_train, keys_to_add)

            self.dict_data_train = dict_data_train

    def reset(self):
        """Rebuilds each dataset's iterable pipeline with a freshly shuffled file order.

        Called between epochs to randomize file order and reintroduce data that may
        have been missed in prior epochs (files get dropped when a dataset's file
        count isn't evenly divisible by the number of GPUs splitting it up).
        """
        #Reset data file list to randomize order of files. Needed in order to introduce data that was potentially missed in prior epochs. Some data files are missed when the number of files for each dataset is not divisible by the number of GPUs that's splitting up those files
        dict_data_train = {}
        for i, k in enumerate(self.dict_lister_trains.keys()):
            lister_train = self.dict_lister_trains[k]
            if self.dataset == "imagenet":
                keys_to_add = 1
            else:
                keys_to_add = int(np.ceil(self.max_balance/self.batches_per_rank_epoch[k]))
            _lister_train = np.random.choice(lister_train, size=len(lister_train), replace=False).tolist()
            if keys_to_add > 1:
                for i in range(keys_to_add-1):
                    _balance_train = np.random.choice(lister_train, size=len(lister_train), replace=False).tolist()
                    _lister_train.extend(_balance_train)

            lister_train = _lister_train

            dict_data_train = self.set_iterative_dataloader(dict_data_train, k, lister_train, keys_to_add)

        self.dict_data_train = dict_data_train

    def train_dataloader(self):
        """Builds the `DataLoader` for the dataset assigned to this rank's data-parallel group.

        Requires `torch.distributed` to be initialized. Determines which dataset
        this rank belongs to from `self.gx` (the colon-separated GPU-per-dataset
        split) and this rank's position within `self.ddp_group`, then wraps that
        dataset's iterable pipeline in a `DataLoader` using `collate_fn`.

        Returns:
            A `torch.utils.data.DataLoader` over this rank's assigned dataset.

        Raises:
            NotImplementedError: If `torch.distributed` is not initialized.
        """
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")

        assert torch.distributed.is_initialized()

        if self.ddp_group == None:
            ddp_rank = torch.distributed.get_rank()
        else:
            ddp_rank = torch.distributed.get_rank(group=self.ddp_group)

        group_list = list(map(lambda x: int(x), self.gx.split(":")))

        assert self.data_par_size == sum(group_list), "data_par_size, group_list: %d %d"%(self.data_par_size, sum(group_list))
        group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
        group_size = group_list[group_id]
        group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]

        for idx, k in enumerate(self.dict_data_train.keys()):
            if idx == group_id:
                data_train = self.dict_data_train[k]
                num_labels = 1
                break

        return DataLoader(
            data_train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # A plain function + functools.partial (rather than a closure lambda) so this
            # is picklable, which multiprocessing_context="spawn" requires -- see the
            # multiprocessing_context docstring entry above for why that matters.
            collate_fn=functools.partial(collate_fn, return_label=self.return_label, adaptive_patching=self.adaptive_patching, separate_channels=self.separate_channels, dataset=self.dataset, num_classes=self.num_classes, num_labels=num_labels, return_qdt=self.return_qdt, dict_key=k),
            multiprocessing_context=self.multiprocessing_context,
        )

