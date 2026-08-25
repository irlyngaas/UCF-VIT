import os
from typing import Dict, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader
from pathlib import Path
import glob
import torch.nn.functional as F
import torch.distributed as dist

from .dataset import (
    FileReader,
    TileDataIter,
    ShuffleIterableDataset,
    ProcessChannels,
)

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
        resize (Dict, optional): Dict mapping "imagenet" to the `[width, height]` size to resize
            images to (matches cv2's own `dsize` convention -- see `dataset.py`'s
            `FileReader.read_process_file`); only used when `dataset` is "imagenet". If absent
            or has no "imagenet" entry, images are left at their native size.
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
           

        self.dict_data_train: Optional[Dict] = None

    def process_root_dirs(self):
        """Builds per-data-parallel-group lists of image file paths for `self.dataset`.

        For "imagenet", groups classes under each root directory into
        `self.data_par_size` (or fewer) buckets of combined class image lists. For
        other datasets, lists all files under each root directory's "imagesTr"
        subfolder, one entry per key in `self.dict_root_dirs`.

        Returns:
            Dict mapping a group/dataset key to a list of file paths.
        """
        if self.dataset == "imagenet":
            dict_lister_trains = {}
            for k, root_dir in self.dict_root_dirs.items():
                #TODO: Add shuffling for data_par_size if it doesn't divide 1000 equally
                classes = sorted(os.listdir(root_dir))
                if len(classes) > self.data_par_size:
                    classes_to_combine = int(len(classes) // self.data_par_size)
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

                    if num_data_roots > self.data_par_size-1:
                        break
        else:
            dict_lister_trains = { k: list(dp.iter.FileLister(os.path.join(root_dir, "imagesTr"))) for k, root_dir in self.dict_root_dirs.items() }
        return dict_lister_trains

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
        if self.dataset == "imagenet":
            start_idx = self.dict_start_idx["imagenet"]
            end_idx = self.dict_end_idx["imagenet"]
            buffer_size = self.dict_buffer_sizes["imagenet"]
            variables = self.dict_in_variables["imagenet"]
            num_channels_used = self.num_channels_used["imagenet"]
            resize = self.resize.get("imagenet") if self.resize else None
        else:
            start_idx = self.dict_start_idx[k]
            end_idx = self.dict_end_idx[k]
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
                                dataset=self.dataset
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
            collate_fn=lambda batch: collate_fn(batch, return_label=self.return_label, adaptive_patching = self.adaptive_patching, separate_channels=self.separate_channels, dataset=self.dataset, num_classes=self.num_classes, num_labels=num_labels, return_qdt=self.return_qdt, dict_key=k),
        )

