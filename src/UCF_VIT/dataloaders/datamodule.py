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
    ImageBlockDataIter_2D,
    ImageBlockDataIter_3D,
    ShuffleIterableDataset,
    ProcessChannels,
)

def collate_fn(batch, return_label, single_channel, adaptive_patching, separate_channels, dataset, num_classes, num_labels, return_qdt):
    if adaptive_patching:
        if return_label:
            if single_channel:
                inp = torch.stack([torch.from_numpy(np.expand_dims(batch[i][0],axis=0)) for i in range(len(batch))])
                seq = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])

                size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])
                if dataset == "imagenet":
                    label = torch.stack([torch.tensor(batch[i][4]) for i in range(len(batch))])
                    variables = []
                    variables.append(batch[0][5])
                    if return_qdt:
                        qdt_list = []
                        for i in range(len(batch)):
                            qdt_list.append(batch[i][6])
                else:
                    if num_labels == 1:
                        label = torch.stack([torch.from_numpy(np.expand_dims(batch[i][4],axis=0)) for i in range(len(batch))])
                    else:
                        label = torch.stack([torch.from_numpy(batch[i][4]) for i in range(len(batch))])
                    seq_label_list = []
                    for i in range(len(batch)):
                        if dataset == "basic_ct":
                            seq_mask = torch.from_numpy(batch[i][5]).long()
                            seq_mask = F.one_hot(seq_mask.squeeze(-1), num_classes=num_classes)
                            seq_label_list.append(seq_mask.permute(2, 0, 1).float())
                        else:
                            seq_label_list.append([])
                            for j in range(num_labels):
                                seq_label_list[i].append(torch.from_numpy(batch[i][5][j]))
                    seq_label = torch.stack([seq_label_list[i] for i in range(len(seq_label_list))])
                    variables = []
                    variables.append(batch[0][6])
                    if return_qdt:
                        qdt_list = []
                        for i in range(len(batch)):
                            qdt_list.append(batch[i][7])
            else:
                inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
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
                    if num_labels == 1:
                        label = torch.stack([torch.from_numpy(np.expand_dims(batch[i][4],axis=0)) for i in range(len(batch))])
                    else:
                        label = torch.stack([torch.from_numpy(batch[i][4]) for i in range(len(batch))])
                    seq_label_list = []
                    for i in range(len(batch)):
                        if dataset == "basic_ct":
                            seq_mask = torch.from_numpy(batch[i][5]).long()
                            seq_mask = F.one_hot(seq_mask.squeeze(-1), num_classes=num_classes)
                            seq_label_list.append(seq_mask.permute(2, 0, 1).float())
                        else:
                            seq_label_list.append([])
                            for j in range(num_labels):
                                seq_label_list[i].append(torch.from_numpy(batch[i][5][j]))
                    seq_label = torch.stack([seq_label_list[i] for i in range(len(seq_label_list))])
                    variables = batch[0][6]
                    if return_qdt:
                        qdt_list = []
                        for i in range(len(batch)):
                            qdt_list.append(batch[i][7])
            if dataset == "imagenet":                
                if return_qdt:
                    return (inp, seq, size, pos, label, variables, qdt_list)
                else:
                    return (inp, seq, size, pos, label, variables)
            else:
                if return_qdt:
                    return (inp, seq, size, pos, label, seq_label, variables, qdt_list)
                else:
                    return (inp, seq, size, pos, label, seq_label, variables)
        else:
            if single_channel:
                inp = torch.stack([torch.from_numpy(np.expand_dims(batch[i][0],axis=0)) for i in range(len(batch))])
                seq = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
                size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
                pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])
                variables = []
                variables.append(batch[0][4])
            else:
                inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
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
                return (inp, seq, size, pos, variables, qdt_list)
            else:
                return (inp, seq, size, pos, variables)
    else:
        if return_label:
            if single_channel:
                inp = torch.stack([torch.from_numpy(np.expand_dims(batch[i][0],axis=0)) for i in range(len(batch))])
                if dataset == "imagenet":
                    label = torch.stack([torch.tensor(batch[i][1]) for i in range(len(batch))])
                else:
                    if num_labels == 1:
                        label = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
                    else:
                        label = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
                variables = []
                variables.append(batch[0][2])
            else:
                inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
                if dataset == "imagenet":
                    label = torch.stack([torch.tensor(batch[i][1]) for i in range(len(batch))])
                else:
                    if num_labels == 1:
                        label = torch.stack([torch.from_numpy(np.expand_dims(batch[i][1],axis=0)) for i in range(len(batch))])
                    else:
                        label = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
                variables = batch[0][2]
                
            return (inp, label, variables)
        else:
            if single_channel:
                inp = torch.stack([torch.from_numpy(np.expand_dims(batch[i][0],axis=0)) for i in range(len(batch))])
                variables = []
                variables.append(batch[0][1])
            else:
                inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
                variables = batch[0][1]

            return (inp, variables)

class NativePytorchDataModule(torch.nn.Module):
    """Native pytorch data module for multi-source data.

    Args:
        dict_root_dirs (Dict): Dictionary of root directories for each source.
        dict_start_idx (Dict): Dictionary of start indices ratio (between 0.0 and 1.0) for each source.
        dict_end_idx (Dict): Dictionary of end indices ratio (between 0.0 and 1.0) for each source.
        dict_in_variables (Dict): Dictionary of input modality variables for each source
        dict_buffer_sizes (Dict): Dictionary of buffer sizes for each source.
        num_channels_available (Dict): Dictionary of number of channels available for each source.
        num_channels_used (Dict): Dictionary of number of channels used from each source.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
        data_par_size (int, optional): the size of the data parallelism
        tile_size_x (int, optional): the tile size in the x dimension
        tile_size_y (int, optional): the tile size in the y dimension
        tile_size_z (int, optional): the tile size in the z dimension
        twoD (bool, optional): Variable for indicating two or three dimensionsal input, if False, three dimensional input.
        return_label (bool, optional): Whether or not the dataloader returns segmentation labels 
        single_channel (bool, optional): Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality
        dataset_group_list (string, optional): How to split available GPUs amongst the available datasets, run "python utils/preprocess_load_balancing.py CONFIG_FILE NUM_GPUS" to obtain
        tile_overlap (float, optional): Amount of tile overlapping to use, takes decimal values, multiples tile_size by tile_overlap to determine step size. Use 0.0 for no overlapping
        use_all_data (bool, optional): Whether or not to use all data in dataloading. Including if tile size doesn't evenly split images. If tile size splits an image unevenly on last tile of a dimension go from last pixel backwards to get a full tile
    """

    def __init__(
        self,
        dict_root_dirs: Dict = None,
        dict_start_idx: Dict = None,
        dict_end_idx: Dict = None,
        dict_buffer_sizes: Dict = None,
        dict_in_variables: Dict = None,
        num_channels_available: Dict = None, 
        num_channels_used: Dict = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        patch_size: int = 16,
        tile_size_x: int = 64,
        tile_size_y: int = 64,
        tile_size_z: int = None,
        twoD: bool = True,
        single_channel: bool = False,
        dataset_group_list: str = '',
        batches_per_rank_epoch: Dict = None,
        tile_overlap: float = 0.0,
        use_all_data: bool = False,
        adaptive_patching: bool = False,
        fixed_length: int = 4096,
        separate_channels: bool = False,
        data_par_size: int = 1,
        dataset: str = "imagenet",
        return_label: Optional[bool] = False,
        return_qdt: Optional[bool] = False,
        ddp_group: Optional[dist.ProcessGroup] = None,
        num_classes: Optional[int] = None,
        imagenet_resize: Optional[Dict] = None,
    ):
        super().__init__()
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

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
        self.num_channels_available = num_channels_available
        self.num_channels_used = num_channels_used
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_size = patch_size
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_size_z = tile_size_z
        self.twoD = twoD
        self.single_channel = single_channel
        self.return_label = return_label
        self.return_qdt = return_qdt
        self.batches_per_rank_epoch = batches_per_rank_epoch
        self.tile_overlap = tile_overlap
        self.use_all_data = use_all_data
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
            self.imagenet_resize = imagenet_resize

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
        if self.dataset == "imagenet":
            start_idx = self.dict_start_idx["imagenet"]
            end_idx = self.dict_end_idx["imagenet"]
            buffer_size = self.dict_buffer_sizes["imagenet"]
            variables = self.dict_in_variables["imagenet"]
            num_channels_available = self.num_channels_available["imagenet"]
            num_channels_used = self.num_channels_used["imagenet"]
            imagenet_resize = self.imagenet_resize["imagenet"]
        else:
            start_idx = self.dict_start_idx[k]
            end_idx = self.dict_end_idx[k]
            buffer_size = self.dict_buffer_sizes[k]
            variables = self.dict_in_variables[k]
            num_channels_available = self.num_channels_available[k]
            num_channels_used = self.num_channels_used[k]
        single_channel = self.single_channel
        return_label = self.return_label
        if self.dataset == "imagenet":
            dict_data_train[k] = ProcessChannels(
                ShuffleIterableDataset(
                    ImageBlockDataIter_2D(
                            FileReader(
                                lister_train,
                                num_channels_available,
                                gx = self.gx,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                multi_dataset_training=True,
                                data_par_size=self.data_par_size,
                                return_label=return_label,
                                keys_to_add=keys_to_add,
                                ddp_group=self.ddp_group,
                                dataset=self.dataset,
                                imagenet_resize=imagenet_resize,
                            ),
                        self.tile_size_x,
                        self.tile_size_y,
                        self.tile_size_z,
                        return_label = return_label,
                        tile_overlap = self.tile_overlap,
                        use_all_data = self.use_all_data,
                        classification = True,
                    ),
                    buffer_size
                ),
                num_channels_used,
                single_channel,
                self.batch_size,
                return_label,
                self.adaptive_patching,
                self.separate_channels,
                self.patch_size,
                self.fixed_length,
                self.twoD,
                self.dataset,
                self.return_qdt,
            )
        else:
            dict_data_train[k] = ProcessChannels(
                ShuffleIterableDataset(
                    ImageBlockDataIter_3D(
                            FileReader(
                                lister_train,
                                num_channels_available,
                                gx = self.gx,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                multi_dataset_training=True,
                                data_par_size = self.data_par_size,
                                return_label = return_label,
                                keys_to_add = keys_to_add,
                                ddp_group = self.ddp_group,
                                dataset=self.dataset
                            ),
                        self.tile_size_x,
                        self.tile_size_y,
                        self.tile_size_z,
                        self.twoD,
                        return_label = return_label,
                        tile_overlap = self.tile_overlap,
                        use_all_data = self.use_all_data,
                    ),
                    buffer_size
                ),
                num_channels_used,
                single_channel,
                self.batch_size,
                return_label,
                self.adaptive_patching,
                self.separate_channels,
                self.patch_size,
                self.fixed_length,
                self.twoD,
                self.dataset,
                self.return_qdt,
            )
        return dict_data_train
        

    def setup(self):
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

    def train_dataloader(self):
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
            collate_fn=lambda batch: collate_fn(batch, return_label=self.return_label, single_channel=self.single_channel, adaptive_patching = self.adaptive_patching, separate_channels=self.separate_channels, dataset=self.dataset, num_classes=self.num_classes, num_labels=num_labels, return_qdt=self.return_qdt),
        )

