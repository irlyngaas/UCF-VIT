from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D
from UCF_VIT.utils.misc import calculate_tile_bounds, calculate_tile_overlap

def CatsDogsCollate(batch, adaptive_patching, return_label):
    """Collate function for `CatsDogsDataset`, stacking per-sample numpy arrays into batched tensors.

    Args:
        batch: List of samples as returned by `CatsDogsDataset.__getitem__`.
        adaptive_patching: Whether each sample includes adaptive-patching sequence
            data (patch sequence, size, and position) in addition to the raw image.
        return_label: Whether to include the label in the returned tuple.

    Returns:
        A tuple of batched tensors/values. With `adaptive_patching=True`:
        `(inp, seq, size, pos, [label,] variables, dict_key)`. Otherwise:
        `(inp, [label,] variables, dict_key)`. The label is included only when
        `return_label` is True.
    """
    if adaptive_patching:
        inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
        seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
        size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
        pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])
        label = torch.stack([torch.tensor(batch[i][4]) for i in range(len(batch))])
        variables = batch[0][5]
        dict_key = batch[0][6]
        if return_label:
            return (inp, seq, size, pos, label, variables, dict_key)
        else:
            return (inp, seq, size, pos, variables, dict_key)
    else:
        inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
        label = torch.stack([torch.tensor(batch[i][1]) for i in range(len(batch))])
        variables = batch[0][2]
        dict_key = batch[0][3]
        if return_label:
            return (inp, label, variables, dict_key)
        else:
            return (inp, variables, dict_key)


class CatsDogsDataset(Dataset):
    """Torch `Dataset` for the Kaggle cats-vs-dogs image classification dataset.

    Loads each image, optionally resizes it to `resize` (leaving it at native
    size if `resize` is None), optionally divides it into a `div x div` grid
    of tiles (mirroring `UCF_VIT.dataloaders.dataset.TileDataIter`'s tiling
    for `imagenet`/`basic_ct`, adapted for this class's map-style
    `Dataset`/`__getitem__` interface rather than `TileDataIter`'s
    `IterableDataset` on-the-fly expansion), derives a binary label from the
    filename, and optionally adaptively patchifies the (possibly tiled)
    image.
    """

    def __init__(self, file_list, variables, tile_size, twoD = True, adaptive_patching = False, fixed_length=196, patch_size=16, num_channels=3, dataset="catsdogs", resize=None, div=1, tile_overlap=(0, 0)):
        """Initializes the dataset over a list of image file paths.

        Args:
            file_list: List of image file paths, each named like ".../cat.123.jpg"
                or ".../dog.123.jpg".
            variables: Variable/channel labels to attach to each returned sample.
            tile_size: `(width, height)` size tiling/patch-size math is based
                on. When `div == 1` (no tiling), this is the size the whole
                image actually is once `resize` (if any) has been applied.
                When `div > 1`, this is the *per-tile* size (matches
                `parse.py`'s own `tile_size` computation:
                `effective_size[i] // div + tile_overlap[i]`) -- already
                divided by `div`, not divided again internally here.
            twoD: Whether to use the 2D (`Patchify`) or 3D (`Patchify_3D`) adaptive
                patcher when `adaptive_patching` is True.
            adaptive_patching: Whether to also compute an adaptive-patching sequence
                for each (possibly tiled) image.
            fixed_length: Fixed output sequence length for adaptive patching.
            patch_size: Patch size used by the adaptive patcher.
            num_channels: Number of image channels.
            dataset: Dataset name to attach to each returned sample.
            resize: `(width, height)` to resize each image to (matches cv2's own
                `dsize` convention), or None to leave images at their native size
                (every file must then already share the same size, since samples
                in a batch must have matching shapes).
            div: Number of tiles to divide each image into per axis (`div * div`
                tiles total). `1` (the default) means no tiling.
            tile_overlap: `(width overlap, height overlap)` total overlap between
                adjacent tiles per axis, only used when `div > 1`.
        """
        self.file_list = file_list
        self.variables = variables
        self.tile_size = tile_size
        self.adaptive_patching = adaptive_patching
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.twoD = twoD
        self.resize = resize
        self.div = div

        # start_overlap/end_overlap and tile_size_no_overlap use the same
        # [width, height]-ordered convention as tile_size/tile_overlap
        # themselves (see calculate_tile_bounds's own docstring for how
        # this is combined with div/tile_idx into per-tile bounds) --
        # matches TileDataIter.__init__'s identical computation exactly.
        self.start_overlap, self.end_overlap = calculate_tile_overlap(tile_overlap)
        self.tile_size_no_overlap = [
            self.tile_size[i] - (self.start_overlap[i] + self.end_overlap[i]) for i in range(2)
        ]

        if self.adaptive_patching:
            if self.twoD:
                self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)
            else:
                self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)

    def __len__(self):
        """Returns the number of (file, tile) samples in the dataset.

        Returns:
            `len(self.file_list) * self.div * self.div` -- `self.div == 1`
            (no tiling) reduces this to the plain file count.
        """
        self.filelength = len(self.file_list) * self.div * self.div
        return self.filelength

    def __getitem__(self, idx):
        """Loads, resizes, optionally tiles, and labels the image at `idx`, optionally adaptively patchifying it.

        Args:
            idx: Flat index over every (file, tile) combination -- decomposed
                below into `file_idx` (which file) and `w_idx`/`h_idx` (which
                tile of that file's `div x div` grid; both `0` when
                `self.div == 1`).

        Returns:
            If `self.adaptive_patching` is True: `(image, seq_img, seq_size, seq_pos,
            label, self.variables, self.dataset)`. Otherwise: `(image, label,
            self.variables, self.dataset)`. `image` is a channel-first numpy array
            and `label` is 1 for "dog" and 0 for "cat" -- the same label for
            every tile of one image, since `catsdogs` is classification-only
            (no per-tile/segmentation label to also tile, unlike
            `TileDataIter`'s general case).
        """
        file_idx, tile_idx = divmod(idx, self.div * self.div)
        w_idx, h_idx = divmod(tile_idx, self.div)

        img_path = self.file_list[file_idx]
        img = Image.open(img_path)
        img = np.array(img)
        if self.resize is not None:
            img = cv.resize(img, dsize=[self.resize[0], self.resize[1]])

        if self.div > 1:
            # img is still channel-last (H, W[, C]) here -- Patchify's own
            # docstring documents that as its expected input, and this
            # matches catsdogs.py's existing pipeline order (moveaxis to
            # channel-first only happens at the return lines below).
            # tile_size/tile_overlap are [width, height]-ordered (matching
            # cv2's own dsize convention -- see this class's docstring), so
            # tile_size_no_overlap[0]/start_overlap[0]/end_overlap[0] (width)
            # bound dim 1 (W) and index 1 (height) bound dim 0 (H) --
            # deliberately explicit start_h/start_w naming (not x/y) to
            # avoid repeating the axis-order bug this exact swap fixed in
            # TileDataIter's own 2D branch.
            start_h, end_h = calculate_tile_bounds(h_idx, self.div, self.tile_size_no_overlap[1], self.start_overlap[1], self.end_overlap[1])
            start_w, end_w = calculate_tile_bounds(w_idx, self.div, self.tile_size_no_overlap[0], self.start_overlap[0], self.end_overlap[0])
            img = img[start_h:end_h, start_w:end_w]

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0


        if self.adaptive_patching:
            seq_img, seq_size, seq_pos, qdt = self.patchify(img)
            return np.moveaxis(img,-1,0), seq_img, seq_size, seq_pos, label, self.variables, self.dataset
        else:
            return np.moveaxis(img,-1,0), label, self.variables, self.dataset
