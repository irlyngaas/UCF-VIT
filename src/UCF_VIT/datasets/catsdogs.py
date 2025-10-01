from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv
import numpy as np
import torch
from UCF_VIT.dataloaders.transform import Patchify, Patchify_3D

def CatsDogsCollate(batch, adaptive_patching):
    if adaptive_patching:
        inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
        seq = torch.stack([torch.from_numpy(batch[i][1]) for i in range(len(batch))])
        size = torch.stack([torch.from_numpy(np.expand_dims(batch[i][2],axis=0)) for i in range(len(batch))])
        pos = torch.stack([torch.from_numpy(np.expand_dims(batch[i][3],axis=0)) for i in range(len(batch))])
        label = torch.stack([torch.tensor(batch[i][4]) for i in range(len(batch))])
        variables = batch[0][5]
        return (inp, seq, size, pos, label, variables)
    else:
        inp = torch.stack([torch.from_numpy(batch[i][0]) for i in range(len(batch))])
        label = torch.stack([torch.tensor(batch[i][1]) for i in range(len(batch))])
        variables = batch[0][2]
        return (inp, label, variables)


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, variables, tile_size, twoD = True, adaptive_patching = False, fixed_length=196, patch_size=16, num_channels=3, dataset="catsdogs"):
        self.file_list = file_list
        self.variables = variables
        self.tile_size = tile_size
        self.adaptive_patching = adaptive_patching
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.twoD = twoD

        if self.adaptive_patching:
            if self.twoD:
                self.patchify = Patchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)
            else:
                self.patchify = Patchify_3D(fixed_length=fixed_length, patch_size=patch_size, num_channels=num_channels, dataset=self.dataset)

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img = np.array(img)
        img = cv.resize(img, dsize=[self.tile_size[0],self.tile_size[1]])
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0


        if self.adaptive_patching:
            seq_img, seq_size, seq_pos, qdt = self.patchify(img)
            return np.moveaxis(img,-1,0), seq_img, seq_size, seq_pos, label, self.variables
        else:
            return np.moveaxis(img,-1,0), label, self.variables
