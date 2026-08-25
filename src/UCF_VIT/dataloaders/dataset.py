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

from UCF_VIT.utils.misc import calculate_tile_bounds, calculate_tile_overlap

class FileReader(IterableDataset):
    """Iterable dataset that reads and preprocesses raw data files, sharded across DDP ranks and dataloader workers.

    Slices `file_list` to the `[start_idx, end_idx)` fraction requested, then in
    `__iter__` further splits the remaining files across the data-parallel group
    (identified via `gx`) and dataloader workers so each worker yields a disjoint
    shard.
    """

    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        gx,
        ddp_group,
        data_par_size: int = 1,
        twoD: bool = False,
        return_label: bool = False,
        keys_to_add: int = 1,
        dataset: str = "imagenet",
        resize: Optional[list] = None,
    ) -> None:
        """Initializes the reader over the `[start_idx, end_idx)` fraction of `file_list`.

        Args:
            file_list: Full list of file paths for this dataset.
            start_idx: Fraction (0.0-1.0) of `file_list` to start reading from.
            end_idx: Fraction (0.0-1.0) of `file_list` to stop reading at.
            variables: Variable/channel labels to attach to each yielded sample.
            gx: Colon-separated string of data-parallel GPU counts per dataset,
                used in `__iter__` to determine this rank's shard.
            ddp_group: Process group used to determine this rank's position within
                `gx`.
            data_par_size: Total number of data-parallel ranks; overridden to 1 if
                `torch.distributed` is not initialized.
            twoD: Whether the data is 2D or 3D; stored but not directly used here.
            return_label: Whether to read and yield a label alongside the data.
            keys_to_add: Number of times to repeat iteration over the (sharded)
                file list per epoch, used to balance dataset sizes.
            dataset: Dataset name, e.g. "imagenet" or "basic_ct"; determines how
                files are read in `read_process_file`.
            resize: `[height, width]` to resize images to, only used for
                `dataset == "imagenet"`. If None, images are left at their
                native size (every file under this dataset key must then
                already share the same size, since samples in a batch
                must have matching shapes).
        """
        super().__init__()
        self.num_channels_available = len(variables)
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = file_list
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
            self.resize = resize

    def read_process_file(self, path):
        """Reads and preprocesses a single data file according to `self.dataset`.

        For "imagenet", loads and resizes an RGB image and, if `return_label`,
        derives the class label from the parent directory name. For "basic_ct",
        loads a NIfTI volume, min-max normalizes it, and if `return_label`, loads
        the corresponding label volume from the sibling "labelsTr" directory.

        Args:
            path: Path to the file to read.

        Returns:
            `data` (channel-first array), or `(data, label)` if `self.return_label`
            is True.
        """
        if self.dataset == "imagenet":
            data = Image.open(path).convert("RGB")
            data = np.array(data)
            if self.resize is not None:
                # dsize=[self.resize[0], self.resize[1]] unchanged from
                # before resize became optional -- cv2's dsize is (width,
                # height), so despite datamodule.py's own resize docstring
                # claiming [height, width], the convention this call has
                # actually always used (untouched here) is [width, height].
                # detect_img_size's output matches this actual convention,
                # not the docstring -- see its own docstring.
                data = cv.resize(data, dsize=[self.resize[0], self.resize[1]])
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
        """Yields preprocessed samples for this worker's shard of `self.file_list`.

        Determines this dataloader worker's contiguous shard of `self.file_list`
        based on the data-parallel rank (via `gx`/`ddp_group`) and the worker's
        index among `torch.utils.data.get_worker_info()`, then repeats iteration
        over that shard `self.keys_to_add` times, calling `read_process_file` on
        each file.

        Yields:
            `(data, label, self.variables)` if `self.return_label` is True,
            otherwise `(data, self.variables)`.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # No DataLoader multiprocessing workers (num_workers=0) -- stand
            # in with a single worker per DDP rank so this still shards by
            # DDP rank via gx/ddp_group below. Previously this branch skipped
            # that sharding entirely (iter_start=0, iter_end=len(file_list)),
            # so every DDP rank silently read the *entire* file_list with
            # num_workers=0 instead of its own shard.
            assert torch.distributed.is_initialized()
            class dummy:
                num_workers = 1
                id = 0
            worker_info = dummy()

        if not torch.distributed.is_initialized():
            ddp_rank = 0
            self.data_par_size = 1
        else:
            if self.ddp_group == None:
                ddp_rank = torch.distributed.get_rank()
            else:
                ddp_rank = torch.distributed.get_rank(group=self.ddp_group)

        num_workers_per_ddp = worker_info.num_workers
        group_list = list(map(lambda x: int(x), self.gx.split(":")))
        group_id = np.where(np.cumsum(group_list) > ddp_rank)[0][0]
        group_size = group_list[group_id]
        group_rank = ddp_rank - ([0] + np.cumsum(group_list).tolist())[group_id]
        num_shards = group_size * num_workers_per_ddp
        rank = group_rank

        per_worker = int(math.floor(len(self.file_list)/ float(self.keys_to_add) / float(num_shards)))
        assert per_worker > 0, "Each worker doesn't have at least one file, run utils/load_balance.py to diagnose the issue"
        worker_id = rank * num_workers_per_ddp + worker_info.id
        iter_start = worker_id * per_worker
        iter_end = iter_start + per_worker

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

class TileDataIter(IterableDataset):
    """Iterable dataset that splits each sample from an upstream dataset into overlapping tiles.

    Wraps a `FileReader` (or similarly-shaped iterable) and, for each yielded
    sample, iterates over a `div x div[ x div]` grid of tiles (optionally slicing
    3D volumes into 2D slices first), yielding one tile at a time.
    """

    def __init__(
        self, dataset: FileReader, tile_size: tuple[int, ...] = (64, 64), twoD: bool = True, return_label: bool = False, div: int = 1, tile_overlap: tuple[int,...] = (0,0), classification: bool = False,
    ) -> None:
        """Initializes the tiling parameters.

        Args:
            dataset: Upstream iterable dataset yielding `(data, [label,] variables)`.
            tile_size: Target tile size (including overlap), one entry per spatial
                dimension.
            twoD: If the upstream data is 3D, whether to slice it into 2D tiles
                along the z-axis (True) or tile it fully in 3D (False). Also
                controls how many spatial dims `tile_size` has.
            return_label: Whether upstream samples include a label to also tile
                (for segmentation) or pass through unchanged (for classification).
            div: Number of tiles to divide each spatial dimension into.
            tile_overlap: Total overlap amount per dimension between adjacent
                tiles, used to compute `start_overlap`/`end_overlap` via
                `calculate_tile_overlap`.
            classification: If True, labels are per-sample (not tiled) and passed
                through unchanged for every tile; if False, labels are tiled the
                same way as the data (segmentation).
        """
        super().__init__()
        self.dataset = dataset
        self.tile_size = tile_size
        self.twoD = twoD
        self.return_label = return_label
        self.div = div
        self.start_overlap, self.end_overlap = calculate_tile_overlap(tile_overlap)
        self.tile_size_no_overlap = []
        # Only the dims with an overlap value (x, y) are ever div'd into
        # tiles -- `tile_size` can have an extra untiled z entry (3D data
        # sliced into 2D z-planes; see parse.py's tile_size computation), so
        # this must iterate over tile_overlap's length, not tile_size's, or
        # it would index past the end of start_overlap/end_overlap.
        for i in range(len(tile_overlap)):
            self.tile_size_no_overlap.append(self.tile_size[i] - (self.start_overlap[i] + self.end_overlap[i]))

        self.classification = classification

    def __iter__(self):
        """Yields one tile at a time from every sample produced by `self.dataset`.

        For 3D data with `self.twoD=True`, first iterates over z-slices and then
        the 2D tile grid within each slice; for 3D data with `self.twoD=False`,
        iterates over the full 3D tile grid; for 2D data, iterates over the 2D tile
        grid directly.

        Yields:
            `(tile, label, variables)` if `self.return_label` is True (with `label`
            tiled too unless `self.classification` is True), otherwise `(tile,
            variables)`.
        """

        if len(self.tile_size) == 3: #Data is 3D
            if self.return_label:
                for (data,label,variables) in self.dataset:
                    if self.twoD: #Loop through slices of 3D data
                        #The current implementation slices on the z dimension but, could do x or y as well
                        #TODO: Add an option on which dimension to slice
                        # self.tile_size[2] is the raw (untiled) z-axis size -- was
                        # data.shape[2], which is actually the y-axis size, not z
                        # (data is channel-first (C, X, Y, Z)); only happened to be
                        # harmless for basic_ct's cubic 256x256x256 volumes.
                        for z_idx in range(self.tile_size[2]):
                            for x_idx in range(self.div):
                                for y_idx in range(self.div):
                                    start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                                    start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                                    if self.classification:
                                        yield data[:, start_x:end_x, start_y:end_y, z_idx], label, variables
                                    else:
                                        yield data[:, start_x:end_x, start_y:end_y, z_idx], label[start_x:end_x, start_y:end_y, z_idx], variables
                                    
                    else: #Loop through full 3D
                        for x_idx in range(self.div):
                            for y_idx in range(self.div):
                                for z_idx in range(self.div):
                                    start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                                    start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                                    start_z, end_z = calculate_tile_bounds(z_idx, self.div, self.tile_size_no_overlap[2], self.start_overlap[2], self.end_overlap[2])
                                    if self.classification:
                                        yield data[:, start_x:end_x, start_y:end_y, start_z:end_z], label, variables
                                    else:
                                        yield data[:, start_x:end_x, start_y:end_y, start_z:end_z], label[start_x:end_x, start_y:end_y, start_z:end_z], variables

            else:
                for (data,variables) in self.dataset:
                    if self.twoD: #Loop through slices of 3D data
                        #The current implementation slices on the z dimension but, could do x or y as well
                        #TODO: Add an option on which dimension to slice
                        # self.tile_size[2] is the raw (untiled) z-axis size -- was
                        # data.shape[2], which is actually the y-axis size, not z
                        # (data is channel-first (C, X, Y, Z)); only happened to be
                        # harmless for basic_ct's cubic 256x256x256 volumes.
                        for z_idx in range(self.tile_size[2]):
                            for x_idx in range(self.div):
                                for y_idx in range(self.div):
                                    start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                                    start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                                    yield data[:, start_x:end_x, start_y:end_y, z_idx], variables

                    else: #Loop through full 3D
                        for x_idx in range(self.div):
                            for y_idx in range(self.div):
                                for z_idx in range(self.div):
                                    start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                                    start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                                    start_z, end_z = calculate_tile_bounds(z_idx, self.div, self.tile_size_no_overlap[2], self.start_overlap[2], self.end_overlap[2])
                                    yield data[:, start_x:end_x, start_y:end_y, start_z:end_z], variables

        else: #Data is 2D -- imagenet only in practice (the only
              #iterative_dataloader dataset with a 2D img_size); catsdogs
              #never reaches TileDataIter at all (no tiling capability).
              #img_size/tile_size are stored [width, height] (matching
              #cv2's own dsize convention -- see dataset.py's
              #FileReader.read_process_file), but imagenet's real array
              #here is (C, H, W): cv.resize(dsize=(W, H)) returns a
              #(H, W, C) array (OpenCV's own convention), then
              #np.moveaxis(-1, 0) makes it channel-first without touching
              #H/W order. So tile_size[0] (width-derived) must bound dim 1
              #(H) via start_x, and tile_size[1] (height-derived) must
              #bound dim 2 (W) via start_y -- indices 0/1 swapped relative
              #to tile_size_no_overlap/start_overlap/end_overlap's own
              #storage order. basic_ct's 3D branch above needs no such
              #swap (no resize step, so img_size's axes already match the
              #array's axes directly) -- this reversal is specific to the
              #resize+moveaxis path.
            if self.return_label:
                for (data,label,variables) in self.dataset:
                    for x_idx in range(self.div):
                        for y_idx in range(self.div):
                            start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                            start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                            if self.classification:
                                yield data[:, start_x:end_x, start_y:end_y], label, variables
                            else:
                                yield data[:, start_x:end_x, start_y:end_y], label[start_x:end_x, start_y:end_y], variables

            else:
                for (data,variables) in self.dataset:
                    for x_idx in range(self.div):
                        for y_idx in range(self.div):
                            start_x, end_x = calculate_tile_bounds(x_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
                            start_y, end_y = calculate_tile_bounds(y_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
                            yield data[:, start_x:end_x, start_y:end_y], variables

class ShuffleIterableDataset(IterableDataset):
    """Iterable dataset that approximately shuffles an upstream iterable using a fixed-size reservoir buffer."""

    def __init__(self, dataset, buffer_size: int) -> None:
        """Initializes the shuffle buffer over an upstream dataset.

        Args:
            dataset: Upstream iterable dataset to shuffle.
            buffer_size: Size of the reservoir buffer; must be greater than 0.
                Larger values give better shuffling at the cost of memory.
        """
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        """Yields samples from `self.dataset` in reservoir-shuffled order.

        Fills a buffer of `self.buffer_size` samples, then for each new sample
        swaps it with a random buffer slot and yields the evicted sample; once the
        upstream dataset is exhausted, shuffles and drains the remaining buffer.

        Yields:
            Samples from `self.dataset`, in shuffled order.
        """
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
    """Iterable dataset that batches samples from an upstream dataset and optionally adaptively patchifies them.

    Buffers `batch_size` upstream samples (tiles/images and, if present, labels)
    before draining them one at a time, applying `Patchify`/`Patchify_3D` per sample
    when `adaptive_patching` is enabled (including patchifying segmentation labels
    with the same quadtree/octree used for the image).
    """

    def __init__(self, dataset, num_channels: int, batch_size: int, return_label: bool, adaptive_patching: bool, separate_channels: bool, patch_size: int, fixed_length: int, twoD: bool, _dataset: str, return_qdt: bool) -> None:
        """Initializes the batching buffer and, if needed, the adaptive-patching transform.

        Args:
            dataset: Upstream iterable dataset yielding `(data, [label,]
                variables)`.
            num_channels: Number of channels in each sample, used to configure the
                adaptive-patching transform.
            batch_size: Number of upstream samples to buffer before yielding.
            return_label: Whether upstream samples include a label.
            adaptive_patching: Whether to adaptively patchify each sample (and its
                label, for segmentation datasets) via a quadtree/octree.
            separate_channels: Whether adaptive patching is done independently per
                channel (True, using a `num_channels=1` patcher applied per
                channel) or jointly across all channels (False).
            patch_size: Leaf patch size used by the adaptive-patching transform.
            fixed_length: Fixed output sequence length for adaptive patching.
            twoD: Whether samples are 2D (`Patchify`) or 3D (`Patchify_3D`).
            _dataset: Dataset name, e.g. "imagenet" or "basic_ct"; determines label
                handling and the patchifier's edge-detection behavior.
            return_qdt: Whether to also yield the quadtree/octree object(s) used
                for each sample.
        """
        super().__init__()
        self.dataset = dataset
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.return_label = return_label
        self.adaptive_patching = adaptive_patching
        self.separate_channels = separate_channels
        self.patch_size = patch_size
        self.twoD = twoD
        self._dataset = _dataset
        self.return_qdt = return_qdt
        if self.adaptive_patching:
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
        """Buffers `self.batch_size` upstream samples, then yields them one by one, patchified if configured.

        Yields:
            A tuple whose composition depends on `self.adaptive_patching`,
            `self.return_label`, `self._dataset`, and `self.return_qdt`; broadly:
            `(image, [seq_image, seq_size, seq_pos,] [label, [seq_label,]]
            variables, [qdt])`.
        """
        yield_x_list = []
        yield_var_list = []
        if self.return_label:
            yield_label_list = []

        for x in self.dataset:
            if self.return_label:
                yield_x_list.append(x[0])
                yield_label_list.append(x[1])
                yield_var_list.append(x[2])
            else:
                yield_x_list.append(x[0])
                yield_var_list.append(x[1])
            #TODO: Don't need these lists anymore
              
            if len(yield_x_list) == self.batch_size:
                while yield_x_list:
                    if self.return_label:
                        if self.adaptive_patching:
                            np_image = yield_x_list.pop()
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
                                np_label = yield_label_list.pop()
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
                                            seq_label, _, _ = qdt_.serialize_labels(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,1))
                                            seq_label = np.asarray(seq_label)
                                            seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size, -1, 1])
                                        else:
                                            seq_label, _, _ = qdt_.serialize(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,1))
                                            seq_label = np.asarray(seq_label, dtype=np.float32)
                                            seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size])
                                    else:
                                        if self._dataset == "basic_ct":
                                            seq_label, _, _ = qdt_.serialize_labels(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.patch_size, 1))
                                            seq_label = np.asarray(seq_label)
                                            seq_label = np.reshape(seq_label, [self.patch_size*self.patch_size*self.patch_size, -1, 1])
                                        else:
                                            seq_label, _, _ = qdt_.serialize(np.expand_dims(np_label[j],axis=-1), size=(self.patch_size,self.patch_size,self.patch_size, 1))
                                            seq_label = np.asarray(seq_label, dtype=np.float32)
                                            seq_label = np.reshape(seq_label, [-1, self.patch_size*self.patch_size*self.patch_size])
                                    seq_label_list.append(seq_label)

                            if self._dataset == "imagenet":
                                if self.return_qdt:
                                    yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_label_list.pop(), yield_var_list.pop(), qdt
                                else:
                                    yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_label_list.pop(), yield_var_list.pop()
                            else:
                                if self._dataset == "basic_ct":
                                    if self.return_qdt:
                                        yield np_image, seq_image, seq_size, seq_pos, np.asarray(np_label,dtype=np.uint8), seq_label_list, yield_var_list.pop(), qdt
                                    else:
                                        yield np_image, seq_image, seq_size, seq_pos, np.asarray(np_label,dtype=np.uint8), seq_label_list, yield_var_list.pop()
                                else:
                                    if self.return_qdt:
                                        yield np_image, seq_image, seq_size, seq_pos, np_label, seq_label_list, yield_var_list.pop(), qdt
                                    else:
                                        yield np_image, seq_image, seq_size, seq_pos, np_label, seq_label_list, yield_var_list.pop()
                        else:
                            if self._dataset == "imagenet":
                                np_image = yield_x_list.pop()
                                yield np.asarray(np_image,dtype=np.float32), yield_label_list.pop(), yield_var_list.pop()
                            else:
                                yield yield_x_list.pop(), yield_label_list.pop(), yield_var_list.pop()

                    else:
                        if self.adaptive_patching:
                            np_image = yield_x_list.pop()
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
                            if self._dataset == "imagenet":
                                if self.return_qdt:
                                    yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_var_list.pop(), qdt
                                else:
                                    yield np.asarray(np_image,dtype=np.float32), seq_image, seq_size, seq_pos, yield_var_list.pop()
                            else:
                                if self.return_qdt:
                                    yield np_image, seq_image, seq_size, seq_pos, yield_var_list.pop(), qdt
                                else:
                                    yield np_image, seq_image, seq_size, seq_pos, yield_var_list.pop()
                        else:
                            if self._dataset == "imagenet":
                                np_image = yield_x_list.pop()
                                yield np.asarray(np_image,dtype=np.float32), yield_var_list.pop()
                            else:
                                yield yield_x_list.pop(), yield_var_list.pop()
