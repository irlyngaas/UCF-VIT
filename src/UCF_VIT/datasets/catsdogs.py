from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D

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
    size if `resize` is None), derives a binary label from the filename, and
    optionally adaptively patchifies the image.
    """

    def __init__(self, file_list, variables, tile_size, twoD = True, adaptive_patching = False, fixed_length=196, patch_size=16, num_channels=3, dataset="catsdogs", resize=None):
        """Initializes the dataset over a list of image file paths.

        Args:
            file_list: List of image file paths, each named like ".../cat.123.jpg"
                or ".../dog.123.jpg".
            variables: Variable/channel labels to attach to each returned sample.
            tile_size: `(width, height)` tiling/patch-size math is based on --
                the size the data actually is once `resize` (if any) has been
                applied, i.e. `resize` when set, else the real native size.
            twoD: Whether to use the 2D (`Patchify`) or 3D (`Patchify_3D`) adaptive
                patcher when `adaptive_patching` is True.
            adaptive_patching: Whether to also compute an adaptive-patching sequence
                for each image.
            fixed_length: Fixed output sequence length for adaptive patching.
            patch_size: Patch size used by the adaptive patcher.
            num_channels: Number of image channels.
            dataset: Dataset name to attach to each returned sample.
            resize: `(width, height)` to resize each image to (matches cv2's own
                `dsize` convention), or None to leave images at their native size
                (every file must then already share the same size, since samples
                in a batch must have matching shapes).
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

        if self.adaptive_patching:
            if self.twoD:
                self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)
            else:
                self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)

    def __len__(self):
        """Returns the number of images in the dataset.

        Returns:
            Number of files in `self.file_list`.
        """
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        """Loads, resizes, and labels the image at `idx`, optionally adaptively patchifying it.

        Args:
            idx: Index into `self.file_list`.

        Returns:
            If `self.adaptive_patching` is True: `(image, seq_img, seq_size, seq_pos,
            label, self.variables, self.dataset)`. Otherwise: `(image, label,
            self.variables, self.dataset)`. `image` is a channel-first numpy array
            and `label` is 1 for "dog" and 0 for "cat".
        """
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img = np.array(img)
        if self.resize is not None:
            img = cv.resize(img, dsize=[self.resize[0], self.resize[1]])
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0


        if self.adaptive_patching:
            seq_img, seq_size, seq_pos, qdt = self.patchify(img)
            return np.moveaxis(img,-1,0), seq_img, seq_size, seq_pos, label, self.variables, self.dataset
        else:
            return np.moveaxis(img,-1,0), label, self.variables, self.dataset
