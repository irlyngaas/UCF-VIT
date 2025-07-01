import math
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import nibabel as nib

from .transform import Patchify, Patchify_3D
from PIL import Image
import cv2 as cv

class FileReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        gx,
        ddp_group,
        multi_dataset_training=False,
        data_par_size: int = 1,
        twoD: bool = False,
        return_label: bool = False,
        keys_to_add: int = 1,
        dataset: str = "imagenet",
        imagenet_resize: Optional[list] = [256,256],
    ) -> None:
        super().__init__()
        self.num_channels_available = len(variables)
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = file_list
        self.multi_dataset_training = multi_dataset_training
        self.data_par_size = data_par_size
        self.twoD = twoD
        self.return_label = return_label
        self.variables = variables
        self.gx = gx
        self.keys_to_add = keys_to_add
        self.ddp_group = ddp_group
        self.dataset = dataset

        #Optional Inputs
        if self.dataset == "imagenet":
            self.imagenet_resize = imagenet_resize

    def read_process_file(self, path):
        if self.dataset == "imagenet":
            data = Image.open(path).convert("RGB")
            data = np.array(data) 
            data = cv.resize(data, dsize=[self.imagenet_resize[0],self.imagenet_resize[1]])
            data = np.moveaxis(data,-1,0)


            if self.return_label:
                data_path = Path(path)
                parent = data_path.parent.absolute()
                parent2 = parent.parent.absolute()
                stem1 = parent.stem
                classes = sorted(os.listdir(os.path.join(parent2)))
                class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
                label = class_to_idx[stem1]
                return data, label
            else:
                return data

        elif self.dataset == "basic_ct":
            data = nib.load(path)
            data = np.array(data.dataobj).astype(np.float32)
            data = (data-data.min())/(data.max()-data.min())

            if self.return_label:
                data_path = Path(path)
                path2 = data_path.parent.absolute()
                path3 = path2.parent.absolute()
                label_stem = data_path.stem.split('image')[-1]
                path4= os.path.join(path3,'labelsTr', "label"+label_stem+".nii")
                label = nib.load(path4)
                label = np.array(label.dataobj).astype(np.int64)
                label = label - 1 # subtract 1 as original labels are [1,4], new will be [0,3]

            if self.num_channels_available == 1:
                if self.return_label:
                    return np.expand_dims(data,axis=0), label
                else:
                    return np.expand_dims(data,axis=0)
            else:
                if self.return_label:
                    return data, label
                else:
                    return data

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            assert torch.distributed.is_initialized()
            class dummy:
                num_workers = 1
                id = 0
            worker_info = dummy()
            iter_start = 0
            iter_end = len(self.file_list)

        else:
            if not torch.distributed.is_initialized():
                ddp_rank = 0
                self.data_par_size = 1
            else:
                if self.ddp_group == None:
                    ddp_rank = torch.distributed.get_rank()
                else:
                    ddp_rank = torch.distributed.get_rank(group=self.ddp_group)

            num_workers_per_ddp = worker_info.num_workers
            assert num_workers_per_ddp == 1
            if self.multi_dataset_training:
                group_list = list(map(lambda x: int(x), self.gx.split(":")))
                group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
                group_size = group_list[group_id]
                group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]
                num_shards = group_size
                rank = group_rank
            else:
                num_shards = num_workers_per_ddp * self.data_par_size
                rank = ddp_rank
            per_worker = int(math.floor(len(self.file_list)/ float(self.keys_to_add) / float(num_shards)))
            if per_worker == 0:
                self.file_list = (self.file_list * math.ceil(num_shards/len(self.file_list)))[:num_shards]
                per_worker = 1
            assert per_worker > 0
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        #print ("global rank %d: ddp rank %d, num_workers_per_ddp %d, worker_info_id %d, worker id %d, iter_start,iter_end = %d %d"%(torch.distributed.get_rank(), ddp_rank, num_workers_per_ddp, worker_info.id, worker_id, iter_start, iter_end), flush=True)
        for m in range(self.keys_to_add):
            start_it = iter_start + m*int(len(self.file_list)/self.keys_to_add)
            end_it = iter_end + m*int(len(self.file_list)/self.keys_to_add)
            for idx in range(start_it, end_it):
                if self.return_label:
                    data, label = self.read_process_file(self.file_list[idx])
                    yield data, label, self.variables
                else:
                    data = self.read_process_file(self.file_list[idx])
                    yield data, self.variables

class ImageBlockDataIter_2D(IterableDataset):
    def __init__(
        self, dataset: FileReader, tile_size_x: int = 64, tile_size_y: int = 64, tile_size_z: int = None, return_label: bool = False, tile_overlap: float = 0.0, use_all_data: bool = False, classification: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_size_z = tile_size_z
        self.return_label = return_label
        self.tile_overlap = tile_overlap
        self.use_all_data = use_all_data
        self.classification = classification

    def __iter__(self):
        tile_overlap_size_x = int(self.tile_size_x*self.tile_overlap)
        tile_overlap_size_y = int(self.tile_size_y*self.tile_overlap)
        if tile_overlap_size_x == 0.0:
            OTP2_x = 1
            tile_overlap_size_x = 0
        else:
            OTP2_x = int(self.tile_size_x/tile_overlap_size_x)
        if tile_overlap_size_y == 0.0:
            OTP2_y = 1
            tile_overlap_size_y = 0
        else:
            OTP2_y = int(self.tile_size_y/tile_overlap_size_y)

        if self.return_label:
            for (data,label,variables) in self.dataset:
                #Total Tiles Evenly Spaced
                TTE_x = data.shape[1]//self.tile_size_x
                TTE_y = data.shape[2]//self.tile_size_y
                num_blocks_x = (TTE_x-1)*OTP2_x + 1
                num_blocks_y = (TTE_y-1)*OTP2_y + 1
                if self.use_all_data:
                    #Total Tiles
                    TT_x = data.shape[1]/(self.tile_size_x)
                    TT_y = data.shape[2]/(self.tile_size_y)
                    # Number of leftover overlap patches for last tile
                    LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                    LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                    if tile_overlap_size_x == 0:
                        if data.shape[1] % self.tile_size_x != 0:
                            LTOP_x += 1
                    else: #>0
                        if data.shape[1] % tile_overlap_size_x != 0:
                            LTOP_x += 1
                    if tile_overlap_size_y == 0:
                        if data.shape[2] % self.tile_size_y != 0:
                            LTOP_y += 1
                    else: #>0
                        if data.shape[2] % tile_overlap_size_y != 0:
                            LTOP_y += 1
                    num_blocks_x = int(num_blocks_x + LTOP_x)
                    num_blocks_y = int(num_blocks_y + LTOP_y)

                channels, datalen_x, datalen_y = data.shape

                x_step_size = self.tile_size_x-tile_overlap_size_x
                y_step_size = self.tile_size_y-tile_overlap_size_y
                for ii in range(num_blocks_x):
                    for jj in range(num_blocks_y):
                        if not self.use_all_data:
                            if self.classification:
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables
                            else:
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                        else:
                            if self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                    #xy
                                    if self.classification:
                                        yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], label, variables
                                    else:
                                        yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], label[datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], variables
                                else:
                                #x
                                    if self.classification:
                                        yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables
                                    else:
                                        yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                            elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                            #y
                                if self.classification:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], label, variables
                                else:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], variables
                            else:
                                if self.classification:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables
                                else:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables

        else:
            for (data,variables) in self.dataset:
                #Total Tiles Evenly Spaced
                TTE_x = data.shape[1]//self.tile_size_x
                TTE_y = data.shape[2]//self.tile_size_y
                num_blocks_x = (TTE_x-1)*OTP2_x + 1
                num_blocks_y = (TTE_y-1)*OTP2_y + 1
                if self.use_all_data:
                    #Total Tiles
                    TT_x = data.shape[1]/(self.tile_size_x)
                    TT_y = data.shape[2]/(self.tile_size_y)
                    # Number of leftover overlap patches for last tile
                    LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                    LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                    if tile_overlap_size_x == 0:
                        if data.shape[1] % self.tile_size_x != 0:
                            LTOP_x += 1
                    else: #>0
                        if data.shape[1] % tile_overlap_size_x != 0:
                            LTOP_x += 1
                    if tile_overlap_size_y == 0:
                        if data.shape[2] % self.tile_size_y != 0:
                            LTOP_y += 1
                    else: #>0
                        if data.shape[2] % tile_overlap_size_y != 0:
                            LTOP_y += 1
                    num_blocks_x = int(num_blocks_x + LTOP_x)
                    num_blocks_y = int(num_blocks_y + LTOP_y)

                channels, datalen_x, datalen_y = data.shape

                x_step_size = self.tile_size_x-tile_overlap_size_x
                y_step_size = self.tile_size_y-tile_overlap_size_y
                for ii in range(num_blocks_x):
                    for jj in range(num_blocks_y):
                        if not self.use_all_data:
                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                        else:
                            if self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                #xy
                                    yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], variables
                                else:
                                #x
                                    yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                            elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                #y
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], variables
                            else:
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables

class ImageBlockDataIter_3D(IterableDataset):
    def __init__(
        self, dataset: FileReader, tile_size_x: int = 64, tile_size_y: int = 64, tile_size_z: int = 64, twoD: bool = True, return_label: bool = False, tile_overlap: float = 0.0, use_all_data: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.twoD = twoD
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.tile_size_z = tile_size_z
        self.return_label = return_label
        self.tile_overlap = tile_overlap
        self.use_all_data = use_all_data

    def __iter__(self):
        tile_overlap_size_x = int(self.tile_size_x*self.tile_overlap)
        tile_overlap_size_y = int(self.tile_size_y*self.tile_overlap)
        tile_overlap_size_z = int(self.tile_size_z*self.tile_overlap)
        if tile_overlap_size_x == 0.0:
            OTP2_x = 1
            tile_overlap_size_x = 0
        else:
            OTP2_x = int(self.tile_size_x/tile_overlap_size_x)
        if tile_overlap_size_y == 0.0:
            OTP2_y = 1
            tile_overlap_size_y = 0
        else:
            OTP2_y = int(self.tile_size_y/tile_overlap_size_y)
        if tile_overlap_size_z == 0.0:
            OTP2_z = 1
            tile_overlap_size_z = 0
        else:
            OTP2_z = int(self.tile_size_z/tile_overlap_size_z)

        if self.return_label:
            for (data,label,variables) in self.dataset:
                #Total Tiles Evenly Spaced
                TTE_x = data.shape[1]//self.tile_size_x
                TTE_y = data.shape[2]//self.tile_size_y
                num_blocks_x = (TTE_x-1)*OTP2_x + 1
                num_blocks_y = (TTE_y-1)*OTP2_y + 1
                if self.use_all_data:
                    #Total Tiles
                    TT_x = data.shape[1]/(self.tile_size_x)
                    TT_y = data.shape[2]/(self.tile_size_y)
                    # Number of leftover overlap patches for last tile
                    LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                    LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                    if tile_overlap_size_x == 0:
                        if data.shape[1] % self.tile_size_x != 0:
                            LTOP_x += 1
                    else: #>0
                        if data.shape[1] % tile_overlap_size_x != 0:
                            LTOP_x += 1
                    if tile_overlap_size_y == 0:
                        if data.shape[2] % self.tile_size_y != 0:
                            LTOP_y += 1
                    else: #>0
                        if data.shape[2] % tile_overlap_size_y != 0:
                            LTOP_y += 1
                    num_blocks_x = int(num_blocks_x + LTOP_x)
                    num_blocks_y = int(num_blocks_y + LTOP_y)

                if self.twoD:
                    if self.use_all_data:
                        num_blocks_z = int(np.ceil(data.shape[3]/self.tile_size_z))
                    else:
                        num_blocks_z = data.shape[3]//self.tile_size_z
                else:
                    TTE_z = data.shape[3]//self.tile_size_z
                    num_blocks_z = (TTE_z-1)*OTP2_z + 1
                    if self.use_all_data:
                        #Total Tiles
                        TT_z = data.shape[3]/(self.tile_size_z)
                        # Number of leftover overlap patches for last tile
                        LTOP_z = np.floor((TT_z-TTE_z)*OTP2_z)
                        if tile_overlap_size_z == 0:
                            if data.shape[3] % self.tile_size_z != 0:
                                LTOP_z += 1
                        else: #>0
                            if data.shape[3] % tile_overlap_size_z != 0:
                                LTOP_z += 1
                        num_blocks_z = int(num_blocks_z + LTOP_z)

                channels, datalen_x, datalen_y, datalen_z = data.shape

                x_step_size = self.tile_size_x-tile_overlap_size_x
                y_step_size = self.tile_size_y-tile_overlap_size_y
                if not self.twoD:
                    z_step_size = self.tile_size_z-tile_overlap_size_z
                for ii in range(num_blocks_x):
                    for jj in range(num_blocks_y):
                        for kk in range(num_blocks_z):
                            if self.twoD:
                                for kkk in range(self.tile_size_z):
                                    if not self.use_all_data:
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables
                                    else:
                                        if kkk+kk*self.tile_size_z > (datalen_z-1):
                                            continue
                                        elif self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                            if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                            #xy
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], label[datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], variables
                                            else:
                                            #x
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], label[datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables
                                        elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                        #y
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], variables
                                        else:
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables

                            else:
                                if not self.use_all_data:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                else:
                                    if self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                        if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                            if self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                            #xyz
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], label[datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], variables
                                            else:
                                            #xy
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], label[datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                        elif self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                        #xz
                                            yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], label[datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], variables
                                        else:
                                        #x
                                            yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], label[datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                    elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                        if self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                        #yz
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], variables
                                        else:
                                        #y
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                    elif self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                    #z
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], variables
                                    else:
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables

        else:
            for (data,variables) in self.dataset:
                #Total Tiles Evenly Spaced
                TTE_x = data.shape[1]//self.tile_size_x
                TTE_y = data.shape[2]//self.tile_size_y
                num_blocks_x = (TTE_x-1)*OTP2_x + 1
                num_blocks_y = (TTE_y-1)*OTP2_y + 1
                if self.use_all_data:
                    #Total Tiles
                    TT_x = data.shape[1]/(self.tile_size_x)
                    TT_y = data.shape[2]/(self.tile_size_y)
                    # Number of leftover overlap patches for last tile
                    LTOP_x = np.floor((TT_x-TTE_x)*OTP2_x)
                    LTOP_y = np.floor((TT_y-TTE_y)*OTP2_y)
                    if tile_overlap_size_x == 0:
                        if data.shape[1] % self.tile_size_x != 0:
                            LTOP_x += 1
                    else: #>0
                        if data.shape[1] % tile_overlap_size_x != 0:
                            LTOP_x += 1
                    if tile_overlap_size_y == 0:
                        if data.shape[2] % self.tile_size_y != 0:
                            LTOP_y += 1
                    else: #>0
                        if data.shape[2] % tile_overlap_size_y != 0:
                            LTOP_y += 1
                    num_blocks_x = int(num_blocks_x + LTOP_x)
                    num_blocks_y = int(num_blocks_y + LTOP_y)

                if self.twoD:
                    if self.use_all_data:
                        num_blocks_z = np.ceil(data.shape[3]/self.tile_size_z).astype(int)
                    else:
                        num_blocks_z = data.shape[3]//self.tile_size_z
                else:
                    TTE_z = data.shape[3]//self.tile_size_z
                    num_blocks_z = (TTE_z-1)*OTP2_z + 1
                    if self.use_all_data:
                        #Total Tiles
                        TT_z = data.shape[3]/(self.tile_size_z)
                        # Number of leftover overlap patches for last tile
                        LTOP_z = np.floor((TT_z-TTE_z)*OTP2_z)
                        if tile_overlap_size_z == 0:
                            if data.shape[3] % self.tile_size_z != 0:
                                LTOP_z += 1
                        else: #>0
                            if data.shape[3] % tile_overlap_size_z != 0:
                                LTOP_z += 1
                        num_blocks_z = int(num_blocks_z + LTOP_z)

                channels, datalen_x, datalen_y, datalen_z = data.shape

                x_step_size = self.tile_size_x-tile_overlap_size_x
                y_step_size = self.tile_size_y-tile_overlap_size_y
                if not self.twoD:
                    z_step_size = self.tile_size_z-tile_overlap_size_z
                for ii in range(num_blocks_x):
                    for jj in range(num_blocks_y):
                        for kk in range(num_blocks_z):
                            if self.twoD:
                                for kkk in range(self.tile_size_z):
                                    if not self.use_all_data:
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables
                                    else:
                                        if kkk+kk*self.tile_size_z > (datalen_z-1):
                                            continue
                                        elif self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                            if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                            #xy
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], variables
                                            else:
                                            #x
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables
                                        elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                        #y
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kkk+kk*self.tile_size_z], variables
                                        else:
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kkk+kk*self.tile_size_z], variables

                            else:
                                if not self.use_all_data:
                                    yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                else:
                                    if self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                        if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                            if self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                            #xyz
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], variables
                                            else:
                                            #xy
                                                yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                        elif self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                        #xz
                                            yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], variables
                                        else:
                                        #x
                                            yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                    elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                        if self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                        #yz
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, datalen_z-self.tile_size_z:datalen_z], variables
                                        else:
                                        #y
                                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
                                    elif self.tile_size_z+kk*z_step_size > (datalen_z-1):
                                    #z
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, datalen_z-self.tile_size_z:datalen_z], variables
                                    else:
                                        yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size, kk*z_step_size:self.tile_size_z+kk*z_step_size], variables
            
class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []

        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

class ProcessChannels(IterableDataset):
    def __init__(self, dataset, num_channels: int, single_channel: bool, batch_size: int, return_label: bool, adaptive_patching: bool, separate_channels: bool, patch_size: int, fixed_length: int, twoD: bool, _dataset: str, return_qdt: bool) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_channels = num_channels
        self.single_channel = single_channel
        if self.single_channel:
            self.num_buffers = num_channels
        else:
            self.num_buffers = 1
        self.batch_size = batch_size
        self.return_label = return_label
        self.adaptive_patching = adaptive_patching
        self.separate_channels = separate_channels
        self.patch_size = patch_size
        self.twoD = twoD
        self._dataset = _dataset
        self.return_qdt = return_qdt
        if self.adaptive_patching:
            if self.single_channel:
                if self.twoD:
                    self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, dataset=self._dataset)
                else:
                    self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, dataset=self._dataset)
            else:
                if self.separate_channels:
                    if self.twoD:
                        self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, dataset=self._dataset)
                    else:
                        self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, dataset=self._dataset)
                else:
                    if self.twoD:
                        self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self._dataset)
                    else:
                        self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self._dataset)

    def __iter__(self):
        yield_x_list = []
        yield_var_list = []
        if self.return_label:
            yield_label_list = []
        #Make a buffer for each channel
        for i in range(self.num_buffers):
            yield_x_list.append([])
            yield_var_list.append([])
            if self.return_label:
                yield_label_list.append([])

        for x in self.dataset:
            for i in range(self.num_buffers):
                if self.single_channel:
                    if self.return_label:
                        yield_x_list[i].append(x[0][i])
                        yield_label_list[i].append(x[1])
                        yield_var_list[i].append(x[2][i])
                    else:
                        yield_x_list[i].append(x[0][i])
                        yield_var_list[i].append(x[1][i])
                else:
                    if self.return_label:
                        yield_x_list[i].append(x[0])
                        yield_label_list[i].append(x[1])
                        yield_var_list[i].append(x[2])
                    else:
                        yield_x_list[i].append(x[0])
                        yield_var_list[i].append(x[1])
                  
                if len(yield_x_list[i]) == self.batch_size:
                    while yield_x_list[i]:
                        if self.return_label:
                            if self.adaptive_patching:
                                np_image = yield_x_list[i].pop()
                                if self.single_channel:
                                    seq_image, seq_size, seq_pos, qdt = self.patchify(np.expand_dims(np_image,axis=-1))
                                    if self._dataset != "imagenet":
                                        np_label = yield_label_list[i].pop()
                                        if self._dataset == "basic_ct":
                                            np_label = np.expand_dims(np_label,axis=0)
                                        seq_label_list = []
                                        for j in range(np_label.shape[0]):
                                            if self.twoD:
                                                if self._dataset == "basic_ct":
                                                    seq_label, _, _ = qdt.serialize_labels(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.num_channels))
                                                    seq_label = np.asarray(seq_label)
                                                    seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size, -1, self.num_channels])
                                                else:
                                                    seq_label, _, _ = qdt.serialize(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.num_channels))
                                                    seq_label = np.asarray(seq_label, dtype=np.float32)
                                                    if self.num_channels > 1:
                                                        seq_label = np.reshape(seq_label, [self.num_channels, -1, self.patch_size*self.patch_size])
                                                    else:
                                                        seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size])
                                            else:
                                                if self._dataset == "basic_ct":
                                                    seq_label, _, _ = qdt.serialize_labels(np_label[j], size=(self.patch_size,self.patch_size,self.patch_size))
                                                    seq_label = np.asarray(seq_label)
                                                    seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size*self.patch_size, -1, self.num_channels])
                                                else:
                                                    seq_label, _, _ = qdt.serialize(np_label[j], size=(self.patch_size,self.patch_size,self.patch_size))
                                                    seq_label = np.asarray(seq_label, dtype=np.float32)
                                                    assert self.num_channels <=1, "num_channels >1 not implemented for 3D yet"
                                                    if self.num_channels > 1:
                                                        seq_label = np.reshape(seq_label, [self.num_channels, -1, self.patch_size*self.patch_size*self.patch_size])
                                                    else:
                                                        seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size*self.patch_size])
                                            seq_label_list.append(seq_label)
                                else:
                                    if self.separate_channels:
                                        seq_image_list = []
                                        seq_size_list = []
                                        seq_pos_list = []
                                        qdt_list = []
                                        for j in range(self.num_channels):
                                            seq_image, seq_size, seq_pos, qdt = self.patchify(np.expand_dims(np_image[j],axis=-1))
                                            seq_image_list.append(seq_image)
                                            seq_size_list.append(seq_size)
                                            seq_pos_list.append(seq_pos)
                                            qdt_list.append(qdt)
                                        seq_image = np.stack([seq_image_list[k] for k in range(len(seq_image_list))])
                                        seq_size = np.stack([seq_size_list[k] for k in range(len(seq_size_list))])
                                        seq_pos = np.stack([seq_pos_list[k] for k in range(len(seq_pos_list))])
                                        qdt = qdt_list

                                    else:
                                        seq_image, seq_size, seq_pos, qdt = self.patchify(np.moveaxis(np_image,0,-1))

                                    if self._dataset != "imagenet":
                                        np_label = yield_label_list[i].pop()
                                        if self._dataset == "basic_ct":
                                            np_label = np.expand_dims(np_label,axis=0)

                                        #TODO: If separate_channel=True, which qdt from qdt_list to use? Default to using the first in the list for now
                                        if self.separate_channels:
                                            qdt_ = qdt[0]
                                        else:
                                            qdt_ = qdt

                                        seq_label_list = []
                                        for j in range(np_label.shape[0]):
                                            if self.twoD:
                                                if self._dataset == "basic_ct":
                                                    seq_label, _, _ = qdt_.serialize_labels(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.num_channels))
                                                    seq_label = np.asarray(seq_label)
                                                    seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size, -1, self.num_channels])
                                                else:
                                                    seq_label, _, _ = qdt_.serialize(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.num_channels))
                                                    seq_label = np.asarray(seq_label, dtype=np.float32)
                                                    if self.num_channels > 1:
                                                        seq_label = np.reshape(seq_label, [self.num_channels, -1, self.patch_size*self.patch_size])
                                                    else:
                                                        seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size])
                                            else:
                                                if self._dataset == "basic_ct":
                                                    seq_label, _, _ = qdt_.serialize_labels(np_label[j], size=(self.patch_size,self.patch_size,self.patch_size))
                                                    seq_label = np.asarray(seq_label)
                                                    seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size*self.patch_size, -1, self.num_channels])
                                                else:
                                                    seq_label, _, _ = qdt_.serialize(np_label[j], size=(self.patch_size,self.patch_size,self.patch_size))
                                                    seq_label = np.asarray(seq_label, dtype=np.float32)
                                                    assert self.num_channels <=1, "num_channels >1 not implemented for 3D yet"
                                                    if self.num_channels > 1:
                                                        seq_label = np.reshape(seq_label, [self.num_channels, -1, self.patch_size*self.patch_size*self.patch_size])
                                                    else:
                                                        seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size*self.patch_size])
                                            seq_label_list.append(seq_label)

                                if self._dataset == "imagenet":
                                    if self.return_qdt:
                                        yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_label_list[i].pop(), yield_var_list[i].pop(), qdt
                                    else:
                                        yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_label_list[i].pop(), yield_var_list[i].pop()
                                else:
                                    if self._dataset == "basic_ct":
                                        if self.return_qdt:
                                            yield np_image, seq_image, seq_size, seq_pos, np.asarray(np_label,dtype=np.uint8), seq_label_list, yield_var_list[i].pop(), qdt
                                        else:
                                            yield np_image, seq_image, seq_size, seq_pos, np.asarray(np_label,dtype=np.uint8), seq_label_list, yield_var_list[i].pop()
                                    else:
                                        if self.return_qdt:
                                            yield np_image, seq_image, seq_size, seq_pos, np_label, seq_label_list, yield_var_list[i].pop(), qdt
                                        else:
                                            yield np_image, seq_image, seq_size, seq_pos, np_label, seq_label_list, yield_var_list[i].pop()
                            else:
                                if self._dataset == "imagenet":
                                    np_image = yield_x_list[i].pop()
                                    yield np.asarray(np_image,dtype=np.float32), yield_label_list[i].pop(), yield_var_list[i].pop()
                                else:
                                    yield yield_x_list[i].pop(), yield_label_list[i].pop(), yield_var_list[i].pop()

                        else:
                            if self.adaptive_patching:
                                np_image = yield_x_list[i].pop()
                                if self.single_channel:
                                    seq_image, seq_size, seq_pos, _ = self.patchify(np.expand_dims(np_image,axis=-1))
                                else:
                                    if self.separate_channels:
                                        seq_image_list = []
                                        seq_size_list = []
                                        seq_pos_list = []
                                        qdt_list = []
                                        for j in range(self.num_channels):
                                            seq_image, seq_size, seq_pos, _ = self.patchify(np.expand_dims(np_image[j],axis=-1))
                                            seq_image_list.append(seq_image)
                                            seq_size_list.append(seq_size)
                                            seq_pos_list.append(seq_pos)
                                            qdt_list.append(qdt)
                                        seq_image = np.stack([seq_image_list[k] for k in range(len(seq_image_list))])
                                        seq_size = np.stack([seq_size_list[k] for k in range(len(seq_size_list))])
                                        seq_pos = np.stack([seq_pos_list[k] for k in range(len(seq_pos_list))])
                                        qdt = qdt_list

                                    else:
                                        seq_image, seq_size, seq_pos, _ = self.patchify(np.moveaxis(np_image,0,-1))
                                if self._dataset == "imagenet":
                                    if self.return_qdt:
                                        yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_var_list[i].pop(), qdt
                                    else:
                                        yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_var_list[i].pop()
                                else:
                                    if self.return_qdt:
                                        yield np_image, seq_image, seq_size, seq_pos, yield_var_list[i].pop(), qdt 
                                    else:
                                        yield np_image, seq_image, seq_size, seq_pos, yield_var_list[i].pop()
                            else:
                                if self._dataset == "imagenet":
                                    np_image = yield_x_list[i].pop()
                                    yield np.asarray(np_image,dtype=np.float32), yield_var_list[i].pop()
                                else:
                                    yield yield_x_list[i].pop(), yield_var_list[i].pop()
