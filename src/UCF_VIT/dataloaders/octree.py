import heapq
import itertools

import numpy as np
import torch
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator

class Cube:
    """An axis-aligned cubic (or cuboid) region of a 3D volume, used as an octree node."""

    def __init__(self, x1, x2, y1, y2, z1, z2) -> None:
        """Initializes the cube's bounding coordinates.

        Args:
            x1: Lower bound along the x axis.
            x2: Upper bound along the x axis; must be >= `x1`.
            y1: Lower bound along the y axis.
            y2: Upper bound along the y axis; must be >= `y1`.
            z1: Lower bound along the z axis.
            z2: Upper bound along the z axis; must be >= `z1`.
        """
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
        assert z1<=z2, 'z1 > z2, wrong coordinate.'
    
    def contains(self, domain, score_fn="canny"):
        """Computes a split-priority score for this cube's region of `domain`.

        Deliberately not normalized by any scale factor -- see `Rect.contains`
        (`quadtree.py`)'s own docstring: this score is only ever consumed by
        `FixedOctTree._build_tree`'s own `max(self.nodes, key=lambda x:x[1])`
        to pick which node to split next, a pure relative comparison
        invariant to a uniform positive scale applied to every candidate.

        Args:
            domain: `score_fn="canny"`: 3D edge-intensity volume, shape
                (Z, Y, X). `score_fn="variance"`: the raw volume itself,
                shape (Z, Y, X) or (Z, Y, X, C) -- see `Rect.contains`'s
                own docstring for why variance needs the real voxels rather
                than a precomputed per-voxel value.
            score_fn: `"canny"` (default -- sum of `domain` within this
                cube) or `"variance"` (this cube's own SSE -- `np.var(patch)`
                scaled by voxel count, summed across channels if `domain`
                has a trailing channel axis).

        Returns:
            Split-priority score for this cube's region -- an int (summed
            intensity) for `"canny"`, a float (SSE) for `"variance"`.
        """
        patch = domain[self.z1:self.z2, self.y1:self.y2, self.x1:self.x2]
        if score_fn == "variance":
            return float(np.var(patch, axis=(0, 1, 2)).sum() * patch.shape[0] * patch.shape[1] * patch.shape[2])
        return int(np.sum(patch))

    def get_area(self, img):
        """Extracts this cube's region from a 4D (Z, Y, X, Channel) image volume.

        Args:
            img: Image volume, shape (Z, Y, X, Channel).

        Returns:
            The sub-volume within this cube's bounds.
        """
        return img[self.z1:self.z2, self.y1:self.y2, self.x1:self.x2, :]

    def set_area(self, mask, patch, num_channels):
        """Resizes `patch` to this cube's size and writes it into `mask` at this cube's location.

        Uses multilinear (`RegularGridInterpolator`) interpolation to resize the
        cubic `patch` from its native size to this cube's actual size before
        writing it in place.

        Args:
            mask: Output volume to write into, in place, shape (Z, Y, X, Channel).
            patch: Cubic patch to resize and place, shape (h1, h1, h1, num_channels).
            num_channels: Number of channels in `patch`.

        Returns:
            `mask`, with this cube's region overwritten by the resized `patch`.
        """
        # import pdb
        # pdb.set_trace()
        patch_size = self.get_size()
        h1, w1, d1, c1 = patch.shape

        h1_ = np.linspace(0,h1,h1)
        w1_ = np.linspace(0,w1,w1)
        d1_ = np.linspace(0,d1,d1)
        #4 to 8 -> (0,1,2,3,4) 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
        #2 to 4 -> (0,2) to (0, .667, 1.3667, 2)
        #_SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}

        interp_fct_list = []
        for j in range(c1):
            interp_fct_list.append(RegularGridInterpolator(points=[h1_,w1_,d1_], values=patch[:,:,:,j]))
        patch = np.zeros([int(patch_size[0]),int(patch_size[1]),int(patch_size[2]),c1])
        h2_ = np.linspace(0,h1,int(patch_size[0]))
        w2_ = np.linspace(0,w1,int(patch_size[1]))
        d2_ = np.linspace(0,d1,int(patch_size[2]))
        H2_, W2_, D2_ = np.meshgrid(h2_, w2_, d2_, indexing='ij')
        query_points = np.vstack([H2_.ravel(),W2_.ravel(),D2_.ravel()]).T
        for j in range(c1):
            patch[:,:,:,j] = interp_fct_list[j](query_points).reshape(H2_.shape)
        mask[self.z1:self.z2, self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask

    def get_coord(self):
        """Returns this cube's bounding coordinates.

        Returns:
            Tuple `(x1, x2, y1, y2, z1, z2)`.
        """
        return self.x1,self.x2,self.y1,self.y2,self.z1,self.z2

    def get_size(self):
        """Returns this cube's side lengths.

        Returns:
            Tuple `(x2-x1, y2-y1, z2-z1)`.
        """
        return self.x2-self.x1, self.y2-self.y1, self.z2-self.z1

    def get_center(self):
        """Returns this cube's center coordinates.

        Returns:
            Tuple `(x_center, y_center, z_center)`.
        """
        return (self.x2+self.x1)/2, (self.y2+self.y1)/2, (self.z2+self.z1)/2

class FixedOctTree:
    """An octree over a 3D edge-intensity volume with a fixed number of leaf nodes.

    Recursively subdivides the volume with `Cube`s, always splitting the node with
    the highest edge-density score into 8 octants, until exactly `fixed_length`
    nodes exist (or a node can no longer be halved). This concentrates small
    (high-resolution) patches around edges and leaves large patches over flat
    regions.
    """

    def __init__(self, domain, fixed_length=128, score_fn="canny", min_size=2) -> None:
        """Builds the octree over `domain`.

        Args:
            domain: 3D edge-intensity volume, shape (Z, Y, X), to subdivide
                -- or, when `score_fn="variance"`, the raw volume itself,
                shape (Z, Y, X) or (Z, Y, X, C) (see `Cube.contains`'s own
                docstring).
            fixed_length: Target number of leaf nodes to subdivide into.
            score_fn: `"canny"` (default) or `"variance"` -- see `Cube.
                contains`'s own docstring for what each means.
            min_size: Smallest leaf side length `_build_tree` will ever
                produce -- see `FixedQuadTree.__init__`'s identical
                `min_size` entry (same shared-with-the-GPU-path role).
        """
        self.domain = domain
        self.fixed_length = fixed_length
        self.score_fn = score_fn
        self.min_size = min_size
        self._build_tree()

    def _build_tree(self):
        """Iteratively splits the highest edge-density node into 8 octants until `fixed_length` nodes exist.

        Populates `self.nodes` as a list of `[Cube, edge_density_score]` pairs.
        Stops early if the highest-scoring node's side length has shrunk to
        `self.min_size` -- it's never split smaller than that, regardless of
        how much it still dominates every other candidate's score.

        Uses a heap (keyed on the negated score, so the max is always
        `heap[0]`) to pick which node to split each iteration, rather than
        `max()`/`.index()` over a plain list -- those are each an O(N) scan
        of the whole current node list, repeated every one of the O(fixed_length)
        iterations (an O(N) list-rebuild via slicing was a third such scan,
        no longer needed either): O(fixed_length^2) overall, confirmed
        empirically (doubling fixed_length cost ~3.3-4x the wall time, not
        the ~2x linear scaling would predict). `heapq`'s O(log N) push/pop
        gets this to O(fixed_length log fixed_length) -- confirmed empirically
        too (~2.0-2.2x per doubling on a prototype, ~23x faster than the
        original at fixed_length=28673). A plain int counter breaks ties
        between equal scores so heapq never needs to compare `Cube` objects
        directly (no `__lt__` needed on `Cube`).
        """
        #channel, height, width, depth = self.domain.shape
        h, w, d = self.domain.shape[:3]  # domain may carry a trailing channel axis under score_fn="variance"
        assert h>0 and w >0 and d>0, "Wrong img size."
        # Cube's own indexing (contains/get_area: domain[z1:z2, y1:y2, x1:x2],
        # matching this class's own (Z, Y, X) docstring convention) requires
        # z <-> axis0 (h), y <-> axis1 (w), x <-> axis2 (d) -- the full
        # reverse of the (h, w, d) unpack order above, not the same order.
        # Getting this wrong silently truncates/misindexes non-cuboid
        # domains (h != d) without ever raising -- see
        # tests/dataloaders/test_octree.py's own non-cuboid regression test.
        root = Cube(0,d,0,w,0,h)
        tiebreak = itertools.count()
        heap = [(-root.contains(self.domain, self.score_fn), next(tiebreak), root)]
        count = 1
        while count < self.fixed_length:
            neg_value, _, bbox = heap[0]
            if bbox.get_size()[0] <= self.min_size:
                break
            heapq.heappop(heap)

            x1,x2,y1,y2,z1,z2 = bbox.get_coord()
            n1 = Cube(x1, int((x1+x2)/2), y1, int((y1+y2)/2), z1, int((z1+z2)/2))
            n2 = Cube(int((x1+x2)/2), x2, y1, int((y1+y2)/2), z1, int((z1+z2)/2))
            n3 = Cube(x1, int((x1+x2)/2), int((y1+y2)/2), y2, z1, int((z1+z2)/2))
            n4 = Cube(int((x1+x2)/2), x2, int((y1+y2)/2), y2, z1, int((z1+z2)/2))
            n5 = Cube(x1, int((x1+x2)/2), y1, int((y1+y2)/2), int((z1+z2)/2), z2)
            n6 = Cube(int((x1+x2)/2), x2, y1, int((y1+y2)/2), int((z1+z2)/2), z2)
            n7 = Cube(x1, int((x1+x2)/2), int((y1+y2)/2), y2, int((z1+z2)/2), z2)
            n8 = Cube(int((x1+x2)/2), x2, int((y1+y2)/2), y2, int((z1+z2)/2), z2)

            for child in (n1, n2, n3, n4, n5, n6, n7, n8):
                heapq.heappush(heap, (-child.contains(self.domain, self.score_fn), next(tiebreak), child))
            count += 7  # -1 (popped parent) + 8 (pushed children)

        self.nodes = [[bbox, -neg_value] for neg_value, _, bbox in heap]

    def serialize(self, img, size=(8,8,8,1)):
        """Extracts and resizes each leaf node's patch from `img` into a fixed-length sequence.

        Each node's variable-sized region is extracted from `img` and resized
        (multilinear interpolation) to `size`. Pads with zero patches if the
        tree has fewer than `fixed_length` nodes. The native region need not
        be cubic -- `RegularGridInterpolator` handles any source grid shape
        -- e.g. a non-cuboid `domain`/`img` (axes of different sizes) makes
        every leaf inherit that same aspect ratio (every split bisects all
        three axes together), which resizes to `size` exactly the same as a
        cubic leaf would.

        Args:
            img: Image volume to extract patches from, shape (Z, Y, X, Channel).
            size: Target `(h, w, d, channel)` size for every patch.

        Returns:
            A tuple `(seq_patch, seq_size, seq_pos)`: `seq_patch` is a list of
            `fixed_length` resized patches, `seq_size` a list of each node's
            original side length (0 for padding), and `seq_pos` a list of each
            node's center coordinates (`(-1, -1, -1)` for padding).
        """
        seq_patch = []
        seq_size = []
        seq_pos = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        h2,w2,d2,c2 = size
        
        for i in range(len(seq_patch)):
            h1, w1, d1, c1 = seq_patch[i].shape
            h1_ = np.linspace(0,h1,h1)
            w1_ = np.linspace(0,w1,w1)
            d1_ = np.linspace(0,d1,d1)
            #4 to 8 -> (0,1,2,3,4) 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
            #2 to 4 -> (0,2) to (0, .667, 1.3667, 2)
            #_SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}

            interp_fct_list = []
            for j in range(c2):
                interp_fct_list.append(RegularGridInterpolator(points=[h1_,w1_,d1_], values=seq_patch[i][:,:,:,j]))

            patch_ = np.zeros([h2,w2,d2,c2])
            h2_ = np.linspace(0,h1,h2)
            w2_ = np.linspace(0,w1,w2)
            d2_ = np.linspace(0,d1,d2)
            H2_, W2_, D2_ = np.meshgrid(h2_, w2_, d2_, indexing='ij')
            query_points = np.vstack([H2_.ravel(),W2_.ravel(),D2_.ravel()]).T
            for j in range(c2):
                patch_[:,:,:,j] = interp_fct_list[j](query_points).reshape(H2_.shape)
            seq_patch[i] = patch_

        if len(seq_patch)<self.fixed_length:
            # import pdb
            # pdb.set_trace()
            seq_patch += [np.zeros(shape=(h2,w2,d2,c2))] * (self.fixed_length-len(seq_patch))
            seq_size += [0]*(self.fixed_length-len(seq_size))
            seq_pos += [tuple([-1,-1,-1])]*(self.fixed_length-len(seq_pos))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
        assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
        return seq_patch, seq_size, seq_pos

    def deserialize(self, seq, patch_size, channel):
        """Reassembles a flat sequence of predicted patches back into a full-size volume.

        Inverse of `serialize`: reshapes `seq` into per-node patches and writes each
        one into its node's location via `Cube.set_area` (which resizes it back up
        to the node's actual size).

        Args:
            seq: Flat array of predicted patch values, reshaped to (fixed_length,
                patch_size, patch_size, patch_size, channel).
            patch_size: Side length each patch is stored at.
            channel: Number of channels.

        Returns:
            Reconstructed volume, shape matching `self.domain` with `channel`
            channels appended.
        """

        H,W,D = self.domain.shape[:3]  # domain may carry a trailing channel axis under score_fn="variance"
        seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, patch_size, channel))
        #seq = seq.astype(int)
        mask = np.zeros(shape=(H, W, D, channel))
        #print("demask:", mask.shape)
        
        # mask = np.expand_dims(mask, axis=-1)
        for idx,(bbox,value) in enumerate(self.nodes):
            pred_mask = seq[idx, ...]
            mask = bbox.set_area(mask, pred_mask, channel)
        return mask
