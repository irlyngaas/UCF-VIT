import numpy as np
import cv2 as cv
import torch
import random
from scipy.ndimage import gaussian_filter, sobel
from .quadtree import FixedQuadTree
from .octree import FixedOctTree

class Patchify(torch.nn.Module):
    """Adaptive (quadtree-based) patchification transform for 2D images.

    Detects edges with a randomly smoothed Canny filter, builds a `FixedQuadTree`
    over the edge map, and serializes the image into a fixed-length sequence of
    variable-sized patches concentrated around detected edges.
    """

    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3, dataset="imagenet", return_edges=False) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing kernel sizes to randomly choose from
                before edge detection (0 = no smoothing, use uniform random noise as
                the edge map instead).
            fixed_length: Fixed number of patches the image is serialized into.
            cannys: `[low, high)` range of Canny lower thresholds to randomly choose
                from; the corresponding upper threshold is `low + 50`.
            patch_size: Side length of the (square) leaf patches.
            num_channels: Number of image channels.
            dataset: Dataset name; controls how edges are computed/normalized
                ("imagenet"/"catsdogs" vs. other datasets).
            return_edges: If True, also return the computed edge map from `forward`.
        """
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges
        
    def forward(self, img):  # we assume inputs are always structured like this
        """Computes an edge map for `img` and adaptively patchifies it via a quadtree.

        Args:
            img: Input 2D image array, shape (H, W[, C]).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos, qdt)`.
            If True: `(seq_img, seq_size, seq_pos, qdt, edges)`. `seq_img` is the
            flattened patch sequence, `seq_size` the per-patch side length,
            `seq_pos` the per-patch center position, `qdt` the `FixedQuadTree`
            instance, and `edges` the computed edge map.
        """
        # Do some transformations. Here, we're just passing though the input

        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        if self.smooth_factor == 0:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            else:
                edges = np.random.uniform(low=np.min(img),high=np.max(img),size=(img.shape[0],img.shape[1]))
        else:
            if self.dataset == "imagenet" or self.dataset == "catsdogs":
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
            else:
                grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
                edges = cv.Canny((grey_img*255).astype(np.uint8), self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.patch_size,self.patch_size,self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)

        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size])

        seq_pos = np.asarray(seq_pos)
        if self.return_edges:
            return seq_img, seq_size, seq_pos, qdt, edges
        else:
            return seq_img, seq_size, seq_pos, qdt

class Patchify_3D(torch.nn.Module):
    """Adaptive (octree-based) patchification transform for 3D volumes.

    Detects edges per-slice with a Gaussian-smoothed Sobel/Canny pipeline, combines
    them into a 3D binary edge volume, builds a `FixedOctTree` over it, and
    serializes the volume into a fixed-length sequence of variable-sized patches
    concentrated around detected edges.
    """

    #TODO: Pass dtype for preferred return dtype
    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3, dataset="basic_ct", return_edges=False) -> None:
        """Initializes the randomization ranges and patch parameters for the transform.

        Args:
            sths: Candidate Gaussian smoothing sigmas to randomly choose from before
                edge detection.
            fixed_length: Fixed number of patches the volume is serialized into.
            cannys: `[low, high)` range of Canny lower thresholds to randomly choose
                from; the corresponding upper threshold is `low + 50`.
            patch_size: Side length of the (cubic) leaf patches.
            num_channels: Number of volume channels.
            dataset: Dataset name; kept for interface compatibility with `Patchify`.
            return_edges: If True, also return the computed edge volume from
                `forward`.
        """
        super().__init__()

        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.dataset = dataset
        self.return_edges = return_edges

    def forward(self, img):  # we assume inputs are always structured like this
        """Computes a 3D edge volume for `img` and adaptively patchifies it via an octree.

        Args:
            img: Input 3D volume array, shape (D, H, W, C).

        Returns:
            If `self.return_edges` is False: `(seq_img, seq_size, seq_pos,
            octtree)`. If True: `(seq_img, seq_size, seq_pos, octtree, edges)`.
            `seq_img` is the flattened patch sequence, `seq_size` the per-patch side
            length, `seq_pos` the per-patch center position, `octtree` the
            `FixedOctTree` instance, and `edges` the computed edge volume.
        """
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        grey_img = gaussian_filter(img, sigma=(self.smooth_factor,self.smooth_factor,self.smooth_factor,0))

        gradient_magnitude = np.zeros_like(grey_img[:,:,:,0])
        gradient_direction = np.zeros_like(grey_img[:,:,:,0])
        for i in range(grey_img.shape[0]):
            for j in range(self.num_channels):
                if j == 0:
                    sobelx = cv.Sobel(grey_img[i, :, :, j], cv.CV_64F, 1, 0, ksize=5)
                    sobely = cv.Sobel(grey_img[i, :, :, j], cv.CV_64F, 0, 1, ksize=5)
                    g_mag = np.sqrt(sobelx**2 + sobely**2)
                else:
                    sx = cv.Sobel(grey_img[i, :, :, j], cv.CV_64F, 1, 0, ksize=5)
                    sy = cv.Sobel(grey_img[i, :, :, j], cv.CV_64F, 0, 1, ksize=5)
                    if np.mean(np.sqrt(sx**2 + sy**2)) > np.mean(g_mag):
                        sobelx = sx
                    if np.mean(sy) > np.mean(sobely):
                        sobely = sy
            gradient_magnitude[i, :, :] = g_mag
            gradient_direction[i, :, :] = np.arctan2(sobely, sobelx)
        edges_combined = np.zeros_like(grey_img[:,:,:,0], dtype=bool)
        edges_combined_counter = np.zeros_like(grey_img[:,:,:,0], dtype=np.uint8)

        for i in range(grey_img.shape[0]):
            for j in range(self.num_channels):
                if j == 0:
                    canny_edges = cv.Canny((grey_img[i, :, :, j]*255).astype(np.uint8), self.canny[0], self.canny[1])
                    cond1 = canny_edges >0
                    edges_combined_counter[i,:,:] = edges_combined_counter[i,:,:] + cond1.astype(np.uint8)
                else:
                    canny = cv.Canny((grey_img[i, :, :, j]*255).astype(np.uint8), self.canny[0], self.canny[1])
                    canny_edges = canny_edges + canny
                    cond1 = canny >0
                    edges_combined_counter[i,:,:] = edges_combined_counter[i,:,:] + cond1.astype(np.uint8)
            edges_combined[i, :, :] = (canny_edges > 0)

        edge_direction_data = np.zeros_like(gradient_direction)
        edge_direction_data[edges_combined] = gradient_direction[edges_combined]
        
        edge_data_normalized = (edge_direction_data - edge_direction_data.min()) / (edge_direction_data.max() - edge_direction_data.min())
        #TODO: Add parameter for this threshold
        threshold = 0.5
        norm_factor = int(255/self.num_channels)
        binary_edges = (edge_data_normalized > threshold).astype(np.uint8) * (edges_combined_counter*norm_factor)
        edges = binary_edges

        octtree = FixedOctTree(domain=edges, fixed_length=self.fixed_length, norm_factor=norm_factor)

        seq_img, seq_size, seq_pos = octtree.serialize(img, size=(self.patch_size,self.patch_size,self.patch_size, self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img, dtype=np.float32)
        if self.num_channels > 1:
            seq_img = np.reshape(seq_img, [self.num_channels, -1, self.patch_size*self.patch_size*self.patch_size])
        else:
            seq_img = np.reshape(seq_img, [-1, self.patch_size*self.patch_size*self.patch_size])

        seq_pos = np.asarray(seq_pos)
        if self.return_edges:
            return seq_img, seq_size, seq_pos, octtree, edges
        else:
            return seq_img, seq_size, seq_pos, octtree
