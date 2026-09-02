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
            argument -- `None` (default) leaves PyTorch's default (`fork`) in place. Set to
            `"spawn"` if `num_workers > 0` forks a worker after CUDA/NCCL is already
            initialized in the parent (`get_model` runs before `train_dataloader`), which
            can deadlock/segfault the worker -- see `configs/basic_ct/sap/base_config.yaml`.
            Costs real per-worker startup latency (re-imports torch/timm/monai/xformers),
            so it's opt-in, not a new default.
        allow_file_reuse (bool, optional): If False (default), a dataset key with fewer
            files than its assigned DDP ranks/workers fails loudly instead of silently
            training some ranks on no data. If True, every rank/worker gets at least one
            file, reusing (duplicating) files round-robin, with a printed warning. Relevant
            at scale too, not just small datasets -- `data_par_size` can exceed a real
            dataset's file count. See `FileReader.__iter__` for the reuse mechanism.
        bucket_shuffle_seed (int, optional): Imagenet only -- seeds the shuffle applied
            before dividing the sliced image list into per-rank buckets. Without it,
            bucketing is a contiguous split of a class-sorted list, so each rank only sees
            a narrow range of classes (skews BatchNorm stats, correlates gradients within a
            rank). Deterministic regardless of `data_par_size`/restarts. `None` disables
            shuffling (original contiguous ordering).
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

        # Slices train/val/test membership once (sorted, deterministic) before
        # setup()/reset()'s own per-epoch reshuffle (np.random.choice, unseeded --
        # kept for training variety/per-rank sharding fairness) ever runs, so which
        # files count as "train" vs held-out "val"/"test" doesn't drift across
        # restarts or epochs. Sorting also avoids relying on process_root_dirs's own
        # (unstable) listing order. set_iterative_dataloader passes a no-op
        # start_idx=0.0/end_idx=1.0 to FileReader since slicing happens here.
        for k in self.dict_lister_trains:
            if self.dataset == "imagenet":
                start_idx, end_idx = self.dict_start_idx["imagenet"], self.dict_end_idx["imagenet"]
            else:
                start_idx, end_idx = self.dict_start_idx[k], self.dict_end_idx[k]
            self.dict_lister_trains[k] = slice_file_list(sorted(self.dict_lister_trains[k]), start_idx, end_idx)

        if self.dataset == "imagenet":
            # Buckets are purely a rank/GPU-group assignment mechanism (labels
            # come from each image's parent directory at read time, not its
            # bucket) -- bucketing after the slice above keeps train/val/test
            # membership independent of data_par_size: only how it's divided
            # across ranks depends on it. Must call bucket_file_list identically
            # to calculate_load_balancing_on_the_fly's own call (same input, same
            # data_par_size), or train_dataloader's gx-based bucket selection breaks.
            k = next(iter(self.dict_lister_trains))
            self.dict_lister_trains = bucket_file_list(self.dict_lister_trains[k], self.data_par_size, shuffle_seed=self.bucket_shuffle_seed)

        self.dict_data_train: Optional[Dict] = None

    def process_root_dirs(self):
        """Builds per-dataset-key lists of image file paths for `self.dataset`.

        Thin wrapper over the shared `UCF_VIT.utils.misc.process_root_dirs`.
        Delegating to one shared function (rather than a local reimplementation)
        keeps this in sync with `calculate_load_balancing_on_the_fly`'s own
        bucketing by construction -- both ultimately feed `train_dataloader`'s
        `gx`-based rank-to-bucket selection, which requires the two to agree on
        bucket count/order exactly. Bucketing itself (imagenet only) happens
        later, in `__init__`, after `dict_start_idx`/`dict_end_idx` slicing --
        see that code's own comment for why.

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
        # start_idx/end_idx are a no-op [0.0, 1.0) here -- already applied once,
        # deterministically, in __init__ (before setup()/reset()'s reshuffle could
        # re-randomize membership; see __init__'s comment). FileReader still takes
        # them as real parameters for its other direct callers/tests.
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
        

    def _my_dataset_key(self):
        """Determines which single `dict_lister_trains`/`dict_data_train` key this rank owns.

        Uses the same `gx`-based `group_id` matching `train_dataloader` relies on --
        factored out so `setup()`/`reset()` can shuffle and build only *this* rank's
        own key's pipeline, instead of every key's, keeping per-epoch cost O(1) per
        rank rather than O(number of keys) (O(`data_par_size`) for imagenet, since
        it has up to `data_par_size` bucket keys -- O(`data_par_size`^2) cluster-wide
        if every rank did this redundantly).

        Returns:
            The single `self.dict_lister_trains` key this rank's DDP-rank
            group is assigned to.

        Raises:
            NotImplementedError: If `torch.distributed` is not initialized.
        """
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")

        if self.ddp_group == None:
            ddp_rank = torch.distributed.get_rank()
        else:
            ddp_rank = torch.distributed.get_rank(group=self.ddp_group)

        group_list = list(map(lambda x: int(x), self.gx.split(":")))
        assert self.data_par_size == sum(group_list), "data_par_size, group_list: %d %d"%(self.data_par_size, sum(group_list))
        group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]

        for idx, k in enumerate(self.dict_lister_trains.keys()):
            if idx == group_id:
                return k

    def _shuffle_and_replicate(self, k):
        """Shuffles this rank's own key's file listing and replicates it `keys_to_add` times.

        Shared by `setup()`/`reset()` -- see their own docstrings for when/why
        each calls this.

        Args:
            k: The dataset key to shuffle/replicate -- always `self._my_dataset_key()`'s
                result in practice, but takes it as an argument rather than
                recomputing it, since `setup()` also needs `k` for `self.batches_per_rank_epoch[k]`.

        Returns:
            A tuple `(lister_train, keys_to_add)`: the shuffled (and, if
            `keys_to_add > 1`, replicated -- each repetition independently
            shuffled) file list, and how many times it was replicated.
        """
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
        return _lister_train, keys_to_add

    def setup(self):
        """Builds the iterable training dataset for this rank's own dataset key, if not already built.

        Computes `self.max_balance`, the largest `batches_per_rank_epoch` across
        every joint dataset (not just this rank's own -- a smaller joint
        dataset's `keys_to_add` replication needs to know the largest one's
        epoch length to match it), then replicates and shuffles *only this
        rank's own dataset key's* file listing (`keys_to_add` times) and
        builds its pipeline via `set_iterative_dataloader`. No-op if
        `self.dict_data_train` is already set. See `_my_dataset_key`'s own
        docstring for why every other key is skipped entirely.
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

            k = self._my_dataset_key()
            lister_train, keys_to_add = self._shuffle_and_replicate(k)
            self.dict_data_train = self.set_iterative_dataloader({}, k, lister_train, keys_to_add)

    def reset(self):
        """Rebuilds this rank's own dataset-key pipeline with a freshly shuffled file order.

        Called between epochs to randomize file order and reintroduce data that may
        have been missed in prior epochs (files get dropped when a dataset's file
        count isn't evenly divisible by the number of GPUs splitting it up). See
        `_my_dataset_key`'s own docstring for why every other key is skipped
        entirely.
        """
        k = self._my_dataset_key()
        lister_train, keys_to_add = self._shuffle_and_replicate(k)
        self.dict_data_train = self.set_iterative_dataloader({}, k, lister_train, keys_to_add)

    def train_dataloader(self):
        """Builds the `DataLoader` for the dataset assigned to this rank's data-parallel group.

        `setup()`/`reset()` already built `self.dict_data_train` for only this
        rank's own dataset key (see `_my_dataset_key`), so this just wraps
        that single pipeline in a `DataLoader` using `collate_fn`.

        Returns:
            A `torch.utils.data.DataLoader` over this rank's assigned dataset.

        Raises:
            NotImplementedError: If `torch.distributed` is not initialized, or
                if `setup()`/`reset()` haven't been called yet.
        """
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
        if not self.dict_data_train:
            raise NotImplementedError("dict_data_train is empty -- call setup() (or reset()) first")

        k = next(iter(self.dict_data_train))
        data_train = self.dict_data_train[k]
        num_labels = 1

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

