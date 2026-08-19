# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision
import torch.nn.functional as torchF
        
def masked_mse(pred, y, mask):
    """Computes the mean squared error over only the masked (e.g. patch-level) positions.

    Args:
        pred: Predicted values.
        y: Target values, same shape as `pred`.
        mask: Mask indicating which positions along the batch/sequence dimension to
            include in the loss (1 = include, 0 = exclude); broadcastable against the
            per-position mean of `(pred - y) ** 2`.

    Returns:
        Scalar tensor with the mean squared error averaged over the masked positions.
    """

    loss = (pred - y) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss*mask).sum() / mask.sum()

    return loss

def adaptive_patching_mse(output, y, size, pos, patch_size, twoD):
    """Computes MSE loss between adaptively-sized predicted patches and the target image.

    Each predicted patch is resized (bicubic) from `patch_size` back up to its
    original adaptive-patch size, then compared against the corresponding region of
    the target image `y` located via `pos`/`size`. Patches with size 0 (unused/padded
    patches) are skipped, and the final loss is averaged over the number of patches
    actually compared.

    Args:
        output: Predicted patch values, shape (Batch, Channel, Seq_Length,
            Patch_Size*Patch_Size) or (Batch, Channel, Seq_Length,
            Patch_Size*Patch_Size*Patch_Size) for 3D.
        y: Target image, shape (Batch, Channel, H, W) for 2D or (Batch, Channel, H,
            W, D) for 3D.
        size: Side length of each adaptive patch, shape (Batch, Channel, Seq_Length, 1).
        pos: Center coordinates of each adaptive patch, shape (Batch, Channel,
            Seq_Length, (x_center, y_center)) or with a z_center added for 3D.
        patch_size: Fixed spatial size that predicted patches are stored at before
            being resized to their adaptive size.
        twoD: Whether the data is 2D (True) or 3D (False).

    Returns:
        Scalar tensor with the MSE loss averaged over all non-empty patches.
    """
    #output (Batch, Channel, Seq_Length, Patch_Size*Patch_Size)
    #data (Batch, Channel, Tile_Size_x, Tile_Size_y)
    #size (Batch, Channel, Seq_Length, 1)
    #pos (Batch, Channel, Seq_Length, (x_center,y_center))
    batch_size, num_channels, seq_len = size.shape
    if twoD:
        batch_size_y, num_channels_y, H, W = y.shape
    else:
        batch_size_y, num_channels_y, H, W, D = y.shape

    if num_channels_y > 1:
        #output = output.reshape(batch_size, num_channels_y, seq_len_y, patch_size*patch_size) 
        if twoD:
            output = output.reshape(batch_size, seq_len, num_channels_y, patch_size*patch_size) 
        else:
            output = output.reshape(batch_size, seq_len, num_channels_y, patch_size*patch_size*patch_size) 

    loss = 0.0 
    patch_counter = 0
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(num_channels_y):
                if num_channels == 1: #Adaptive Patching was done across all channels
                    p_center = np.asarray(pos[i,0,j])
                    p_size = np.asarray(size[i,0,j])
                else: #Adaptive Patching was done for each channel individually
                    p_center = np.asarray(pos[i,k,j])
                    p_size = np.asarray(size[i,k,j])
                    
                if p_size == 0:
                    continue
                else:
                    patch_counter += 1

                if p_size == 1:
                    x_start = int(p_center[0])
                    x_end = int(p_center[0])+1
                    y_start = int(p_center[1])
                    y_end = int(p_center[1])+1
                    if not twoD:
                        z_start = int(p_center[2])
                        z_end = int(p_center[2])+1
                else:
                    x_start = int(p_center[0] - p_size/2)
                    x_end = int(p_center[0] + p_size/2)
                    y_start = int(p_center[1] - p_size/2)
                    y_end = int(p_center[1] + p_size/2)
                    if not twoD:
                        z_start = int(p_center[2] - p_size/2)
                        z_end = int(p_center[2] + p_size/2)

                if num_channels_y == 1:
                    if twoD:
                        resize_output = F.resize(output[i,j].reshape(patch_size,patch_size,1), (int(p_size),int(p_size)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                        diff = (resize_output.cpu() - y[i,0,x_start:x_end,y_start:y_end])**2
                        diff = diff.mean()
                    else:
                        resize_output = F.resize(output[i,j].reshape(patch_size,patch_size,patch_size, 1), (int(p_size),int(p_size),int(p_size)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                        diff = (resize_output.cpu() - y[i,0,x_start:x_end,y_start:y_end,z_start:z_end])**2
                        diff = diff.mean()
                    loss += diff
                else: #Need to recover channel information from output
                    if twoD:
                        resize_output = F.resize(output[i,j,k].reshape(patch_size,patch_size,1), (int(p_size),int(p_size)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                        diff = (resize_output.cpu() - y[i,k,x_start:x_end,y_start:y_end])**2
                        diff = diff.mean()
                    else:
                        resize_output = F.resize(output[i,j,k].reshape(patch_size,patch_size,patch_size,1), (int(p_size),int(p_size),int(p_size)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
                        diff = (resize_output.cpu() - y[i,k,x_start:x_end,y_start:y_end,z_start:z_end])**2
                        diff = diff.mean()
                    loss += diff

    loss = loss/patch_counter
    return loss

class DiceBLoss(nn.Module):
    """Combined Dice loss and binary cross-entropy loss for segmentation.

    Computes a weighted sum of binary cross-entropy and soft Dice loss, both
    excluding the first (background) class channel.
    """

    def __init__(self, weight=0.5, num_class=2, size_average=True):
        """Initializes the loss weighting between BCE and Dice loss.

        Args:
            weight: Weight given to the BCE term; the Dice term is weighted by
                `1 - weight`.
            num_class: Number of segmentation classes. Unused directly but stored
                for reference.
            size_average: Unused; kept for interface compatibility.
        """
        super(DiceBLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, inputs, targets, smooth=1, act=True):
        """Computes the combined Dice + BCE loss.

        Args:
            inputs: Predicted logits (or probabilities if `act=False`), shape
                (Batch, Class, ...).
            targets: Ground-truth one-hot/probability targets, same shape as
                `inputs`.
            smooth: Smoothing constant added to numerator/denominator of the Dice
                coefficient to avoid division by zero.
            act: If True, apply a sigmoid activation to `inputs` before computing
                the loss.

        Returns:
            Scalar tensor with `weight * BCE + (1 - weight) * dice_loss`.
        """
        #comment out if your model contains a sigmoid or equivalent activation layer
        if act:
            inputs = torchF.sigmoid(inputs)    
    
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
    
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
    
        intersection = (pred * true).sum()
        coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)    
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)  
        BCE = torchF.binary_cross_entropy(pred, true, reduction='mean')
        dice_bce = self.weight*BCE + (1-self.weight)*dice_loss
        # dice_bce = dice_loss 
    
        return dice_bce
