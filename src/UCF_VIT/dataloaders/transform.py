import numpy as np
import cv2 as cv
import torch
import random
from scipy.ndimage import gaussian_filter, sobel
from .quadtree import FixedQuadTree
from .octree import FixedOctTree

class Patchify(torch.nn.Module):
    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3, dataset="imagenet") -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset
        
    def forward(self, img):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        # self.smooth_factor = 0
        if self.smooth_factor ==0 :
            if self.dataset == "imagenet":
                edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            else:
                edges = np.random.uniform(low=np.min(img),high=np.max(img),size=(img.shape[0],img.shape[1]))
        else:
            if self.dataset == "imagenet":
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
            else:
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                edges = cv.Canny((grey_img*255).astype(np.uint8), self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.patch_size,self.patch_size,self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)

        #seq_img = np.reshape(seq_img, [self.patch_size*self.patch_size, -1, 3])
        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size])

        seq_pos = np.asarray(seq_pos)
        return seq_img, seq_size, seq_pos

class Patchify_3D(torch.nn.Module):
    #TODO: Pass dtype for preferred return dtype
    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3, dataset="basic_ct") -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset

    def forward(self, img):  # we assume inputs are always structured like this
        img = np.squeeze(img,axis=None)
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        grey_img = gaussian_filter(img, sigma=self.smooth_factor)

        gradient_magnitude = np.zeros_like(grey_img)
        gradient_direction = np.zeros_like(grey_img)

        for i in range(grey_img.shape[0]):
            sobelx = cv.Sobel(grey_img[i, :, :], cv.CV_64F, 1, 0, ksize=5)
            sobely = cv.Sobel(grey_img[i, :, :], cv.CV_64F, 0, 1, ksize=5)
            gradient_magnitude[i, :, :] = np.sqrt(sobelx**2 + sobely**2)
            gradient_direction[i, :, :] = np.arctan2(sobely, sobelx)

        edges_combined = np.zeros(grey_img.shape, dtype=bool)

        for i in range(grey_img.shape[0]):
            canny_edges = cv.Canny(grey_img[i, :, :].astype(np.uint8), self.canny[0], self.canny[1])
            edges_combined[i, :, :] != (canny_edges > 0)

        edge_direction_data = np.zeros_like(gradient_direction)
        edge_direction_data[edges_combined] = gradient_direction[edges_combined]
        
        edge_data_normalized = (edge_direction_data - edge_direction_data.min()) / (edge_direction_data.max() - edge_direction_data.min())
        #TODO: Add parameter for this threshold
        threshold = 0.5
        binary_edges = (edge_data_normalized > threshold).astype(np.uint8) * 255
        edges = binary_edges

        octtree = FixedOctTree(domain=edges, fixed_length=self.fixed_length)

        seq_img, seq_size, seq_pos = octtree.serialize(img, size=(self.patch_size,self.patch_size,self.patch_size))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)
        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size*self.patch_size])

        seq_pos = np.asarray(seq_pos)
        return seq_img, seq_size, seq_pos
