import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import nibabel as nib

from .transform import Patchify
from PIL import Image
import cv2 as cv

class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        label_list,
        num_channels_available,
        num_channels_used,
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
    ) -> None:
        super().__init__()
        self.num_channels_available = num_channels_available
        self.num_channels_used = num_channels_used
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = file_list
        label_list = label_list[start_idx:end_idx]
        self.label_list = label_list
        self.multi_dataset_training = multi_dataset_training
        self.data_par_size = data_par_size
        self.twoD = twoD
        self.return_label = return_label
        self.variables = variables
        self.gx = gx
        self.keys_to_add = keys_to_add
        self.ddp_group = ddp_group

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

        #print ("global rank %d: ddp rank %d iter_start,iter_end = %d %d"%(torch.distributed.get_rank(), ddp_rank,iter_start, iter_end))
        for m in range(self.keys_to_add):
            start_it = iter_start + m*int(len(self.file_list)/self.keys_to_add)
            end_it = iter_end + m*int(len(self.file_list)/self.keys_to_add)
            for idx in range(iter_start, iter_end):
                path = self.file_list[idx]
                data = Image.open(path).convert("RGB")
                data = np.array(data) 
                data = cv.resize(data, dsize=[256,256])
                data = np.moveaxis(data,-1,0)




                #if self.num_channels_used == 1:
                #    if self.npy_files:
                #        data = np.load(path)
                #    else:
                #        data = nib.load(path)
                #        data = np.array(data.dataobj).astype(np.float32)
                #    if not self.norm_stats:
                #        data = (data-data.min())/(data.max()-data.min())
                #    else:
                #        minimum = self.norm_stats[path].item()['0']['min']
                #        maximum = self.norm_stats[path].item()['0']['max']
                #        data = (data-minimum)/(maximum-minimum)

                #if self.return_label:
                #    assert self.norm_stats, "No labelling currently setup for basic_ct, so can't run segmentation"
                #    label_suffix = self.norm_stats[path].item()['0']['label_suffix']
                #    assert label_suffix != "NONE", "No labels available, can't run segementation on this dataset"
                #    data_path = Path(path)
                #    path2 = data_path.parent.absolute()
                #    path3 = path2.parent.absolute()
                #    path4= os.path.join(path3,'labelsTr', data_path.stem+label_suffix)
                #    label = nib.load(path4)
                #    label = np.array(label.dataobj).astype(np.int64)


                if self.return_label:
                    yield data, self.label_list[idx], self.variables
                else:
                    yield data, self.variables

                #if self.twoD:
                #    for i in range(data.shape[-1]):
                #        if self.num_channels_used == 1:
                #            if self.return_label:
                #                yield np.expand_dims(data[:,:,i],axis=0), label, self.variables
                #            else:
                #                yield np.expand_dims(data[:,:,i],axis=0), self.variables
                #        else:
                #            if self.return_label:
                #                yield data[:,:,:,i], label, self.variables
                #            else:
                #                yield data[:,:,:,i], self.variables
                #else:
                #    if self.num_channels_used == 1:
                #        if self.return_label:
                #            yield np.expand_dims(data,axis=0), label, self.variables
                #        else:
                #            yield np.expand_dims(data,axis=0), self.variables
                #    else:
                #        if self.return_label:
                #            yield data, label, self.variables
                #        else:
                #            yield data, self.variables

class ImageBlockDataIter(IterableDataset):
    def __init__(
        self, dataset: NpyReader, num_channels_available: int = 1, num_channels_used: int = 1, tile_size_x: int = 64, tile_size_y: int = 64, tile_size_z: int = None, twoD: bool = True, return_label: bool = False, tile_overlap: float = 0.0, use_all_data: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_channels_available = num_channels_available
        self.num_channels_used = num_channels_used
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
                            #yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                            yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables
                        else:
                            if self.tile_size_x+ii*x_step_size > (datalen_x-1):
                                if self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                    #xy
                                    #yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], label[datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], variables
                                    yield data[:, datalen_x-self.tile_size_x:datalen_x, datalen_y-self.tile_size_y:datalen_y], label, variables
                                else:
                                #x
                                    #yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                                    yield data[:, datalen_x-self.tile_size_x:datalen_x, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables
                            elif self.tile_size_y+jj*y_step_size > (datalen_y-1):
                                #y
                                #yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], variables
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, datalen_y-self.tile_size_y:datalen_y], label, variables
                            else:
                                #yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label[ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], variables
                                yield data[:, ii*x_step_size:self.tile_size_x+ii*x_step_size, jj*y_step_size:self.tile_size_y+jj*y_step_size], label, variables

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
    def __init__(self, dataset, num_channels: int, single_channel: bool, batch_size: int, return_label: bool, adaptive_patching: bool, separate_channels: bool, patch_size: int, fixed_length: int, gauss_filter_order: int) -> None:
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
        self.gauss_filter_order = gauss_filter_order
        if self.adaptive_patching:
            if self.single_channel:
                self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, sths=[self.gauss_filter_order])
            else:
                if self.separate_channels:
                    self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=1, sths=[self.gauss_filter_order])
                else:
                    self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, sths=[self.gauss_filter_order])

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
                        yield_label_list[i].append(x[1][i])
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
                                    seq_image, seq_size, seq_pos = self.patchify(np.expand_dims(np_image,axis=-1))
                                else:
                                    if self.separate_channels:
                                        seq_image_list = []
                                        seq_size_list = []
                                        seq_pos_list = []
                                        for j in range(self.num_channels):
                                            seq_image, seq_size, seq_pos = self.patchify(np.expand_dims(np_image[j],axis=-1))
                                            seq_image_list.append(seq_image)
                                            seq_size_list.append(seq_size)
                                            seq_pos_list.append(seq_pos)
                                        seq_image = np.stack([seq_image_list[k] for k in range(len(seq_image_list))])
                                        seq_size = np.stack([seq_size_list[k] for k in range(len(seq_size_list))])
                                        seq_pos = np.stack([seq_pos_list[k] for k in range(len(seq_pos_list))])

                                    else:
                                        seq_image, seq_size, seq_pos = self.patchify(np.moveaxis(np_image,0,-1))
                                yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_label_list[i].pop(), yield_var_list[i].pop()
                            else:
                                np_image = yield_x_list[i].pop()
                                #yield yield_x_list[i].pop(), yield_label_list[i].pop(), yield_var_list[i].pop()
                                yield np.asarray(np_image,dtype=np.float32), yield_label_list[i].pop(), yield_var_list[i].pop()

                        else:
                            if self.adaptive_patching:
                                np_image = yield_x_list[i].pop()
                                if self.single_channel:
                                    seq_image, seq_size, seq_pos = self.patchify(np.expand_dims(np_image,axis=-1))
                                else:
                                    if self.separate_channels:
                                        seq_image_list = []
                                        seq_size_list = []
                                        seq_pos_list = []
                                        for j in range(self.num_channels):
                                            seq_image, seq_size, seq_pos = self.patchify(np.expand_dims(np_image[j],axis=-1))
                                            seq_image_list.append(seq_image)
                                            seq_size_list.append(seq_size)
                                            seq_pos_list.append(seq_pos)
                                        seq_image = np.stack([seq_image_list[k] for k in range(len(seq_image_list))])
                                        seq_size = np.stack([seq_size_list[k] for k in range(len(seq_size_list))])
                                        seq_pos = np.stack([seq_pos_list[k] for k in range(len(seq_pos_list))])

                                    else:
                                        seq_image, seq_size, seq_pos = self.patchify(np.moveaxis(np_image,0,-1))
                                yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_var_list[i].pop()
                            else:
                                np_image = yield_x_list[i].pop()
                                #yield yield_x_list[i].pop(), yield_var_list[i].pop()
                                yield np.asarray(np_image,dtype=np.float32), yield_var_list[i].pop()
