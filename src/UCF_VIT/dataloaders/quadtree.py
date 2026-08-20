import numpy as np
import torch
import cv2 as cv
from matplotlib import pyplot as plt

class Rect:
    """An axis-aligned rectangular region of a 2D image, used as a quadtree node."""

    def __init__(self, x1, x2, y1, y2) -> None:
        """Initializes the rectangle's bounding coordinates.

        Args:
            x1: Lower bound along the x axis.
            x2: Upper bound along the x axis; must be >= `x1`.
            y1: Lower bound along the y axis.
            y2: Upper bound along the y axis; must be >= `y1`.
        """
        # *q
        # p*
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        assert x1<=x2, 'x1 > x2, wrong coordinate.'
        assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        """Computes a normalized edge-density score for this rectangle's region of `domain`.

        Args:
            domain: 2D edge-intensity image, shape (H, W).

        Returns:
            Integer edge-density score for this rectangle's region (summed
            intensity divided by 255).
        """
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)

    def get_area(self, img):
        """Extracts this rectangle's region from a 3D (H, W, Channel) image.

        Args:
            img: Image, shape (H, W, Channel).

        Returns:
            The sub-image within this rectangle's bounds.
        """
        return img[self.y1:self.y2, self.x1:self.x2, :]

    def set_area(self, mask, patch):
        """Resizes `patch` to this rectangle's size and writes it into `mask` at this rectangle's location.

        Uses bicubic interpolation to resize the square `patch` from its native
        size to this rectangle's actual size before writing it in place.

        Args:
            mask: Output image to write into, in place, shape (H, W, Channel).
            patch: Square patch to resize and place.

        Returns:
            `mask`, with this rectangle's region overwritten by the resized
            `patch`.
        """
        # import pdb
        # pdb.set_trace()
        patch_size = self.get_size()
        # patch = np.resize(patch, patch_size)
        patch = patch.astype('float32')
        patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
        # patch = np.expand_dims(patch, axis=-1)
        # import pdb
        # pdb.set_trace()
        mask[self.y1:self.y2, self.x1:self.x2, :] = patch
        return mask
    
    def get_coord(self):
        """Returns this rectangle's bounding coordinates.

        Returns:
            Tuple `(x1, x2, y1, y2)`.
        """
        return self.x1,self.x2,self.y1,self.y2

    def get_size(self):
        """Returns this rectangle's side lengths.

        Returns:
            Tuple `(x2-x1, y2-y1)`.
        """
        return self.x2-self.x1, self.y2-self.y1

    def get_center(self):
        """Returns this rectangle's center coordinates.

        Returns:
            Tuple `(x_center, y_center)`.
        """
        return (self.x2+self.x1)/2, (self.y2+self.y1)/2

    def draw(self, ax, c='grey', lw=0.5, **kwargs):
        """Draws this rectangle's outline (no fill) onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Unused; kept for interface compatibility with `draw_area`.
            lw: Line width of the outline.
            **kwargs: Unused.
        """
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1),
                                 width=self.x2-self.x1,
                                 height=self.y2-self.y1,
                                 linewidth=lw, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

    def draw_area(self, ax, c='green', lw=0.5, **kwargs):
        """Draws this rectangle filled with color `c` onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Fill color.
            lw: Line width of the outline.
            **kwargs: Unused.
        """
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1),
                                 width=self.x2-self.x1,
                                 height=self.y2-self.y1,
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)

    def draw_rescale(self, ax, c='green', lw=0.5, **kwargs):
        """Draws a fixed 16x16 filled rectangle at this rectangle's top-left corner.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Fill color.
            lw: Line width of the outline.
            **kwargs: Unused.
        """
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1),
                                 width=16,
                                 height=16,
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)

    def draw_zorder(self, ax, c='red', lw=0.5, **kwargs):
        """Draws a fixed 16x16 filled rectangle at this rectangle's top-left corner.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Fill color.
            lw: Line width of the outline.
            **kwargs: Unused.
        """
        # Create a Rectangle patch
        import matplotlib.patches as patches
        rect = patches.Rectangle((self.x1, self.y1),
                                 width=16,
                                 height=16,
                                 linewidth=lw, edgecolor='w', facecolor=c)
        ax.add_patch(rect)
    
                 
class FixedQuadTree:
    """A quadtree over a 2D edge-intensity image with a fixed number of leaf nodes.

    Recursively subdivides the image with `Rect`s, always splitting the node with
    the highest edge-density score into 4 quadrants, until exactly `fixed_length`
    nodes exist (or a node can no longer be halved). This concentrates small
    (high-resolution) patches around edges and leaves large patches over flat
    regions.
    """

    def __init__(self, domain, fixed_length=128, build_from_info=False, meta_info=None) -> None:
        """Builds the quadtree over `domain`, or reconstructs it from saved metadata.

        Args:
            domain: 2D edge-intensity image, shape (H, W), to subdivide.
            fixed_length: Target number of leaf nodes to subdivide into.
            build_from_info: If True, reconstruct `self.nodes` from `meta_info`
                instead of building the tree from scratch.
            meta_info: Node bounding-box list as returned by `encode_nodes`, used
                only when `build_from_info` is True.
        """
        self.domain = domain
        self.fixed_length = fixed_length
        if build_from_info:
            self.nodes = self.decoder_nodes(meta_info=meta_info)
        else:
            self._build_tree()

    def nodes_value(self):
        """Computes a normalized size value for each leaf node.

        Returns:
            List of single-element lists `[size/8]`, one per node, where `size` is
            the node's x side length.
        """
        meta_value = []
        for rect,v in self.nodes:
            size,_ = rect.get_size()
            meta_value += [[size/8]]
        return meta_value

    def encode_nodes(self):
        """Serializes each leaf node's bounding box, for later reconstruction via `decoder_nodes`.

        Returns:
            List of `[x1, x2, y1, y2]` lists, one per node.
        """
        meta_info = []
        for rect,v in self.nodes:
            meta_info += [[rect.x1,rect.x2,rect.y1,rect.y2]]
        return meta_info

    def decoder_nodes(self, meta_info):
        """Reconstructs `[Rect, edge_density_score]` node pairs from saved bounding boxes.

        Args:
            meta_info: List of `[x1, x2, y1, y2]` bounding boxes, as returned by
                `encode_nodes`.

        Returns:
            List of `[Rect, value]` pairs, with `value` recomputed via
            `Rect.contains(self.domain)`.
        """
        nodes = []
        for info in meta_info:
            x1,x2,y1,y2 = info
            n = Rect(x1, x2, y1, y2)
            v = n.contains(self.domain)
            nodes +=  [[n,v]] 
        return nodes
            
    def _build_tree(self):
        """Iteratively splits the highest edge-density node into 4 quadrants until `fixed_length` nodes exist.

        Populates `self.nodes` as a list of `[Rect, edge_density_score]` pairs.
        Stops early if the highest-scoring node's side length has shrunk to 2 (it
        can't be evenly halved further).
        """

        h,w = self.domain.shape
        assert h>0 and w >0, "Wrong img size."
        root = Rect(0,w,0,h)
        self.nodes = [[root, root.contains(self.domain)]]
        while len(self.nodes)<self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x:x[1])
            idx = self.nodes.index([bbox, value])
            if bbox.get_size()[0] == 2:
                break

            x1,x2,y1,y2 = bbox.get_coord()
            lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
            v1 = lt.contains(self.domain)
            rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
            v2 = rt.contains(self.domain)
            lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
            v3 = lb.contains(self.domain)
            rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
            v4 = rb.contains(self.domain)
            
            self.nodes = self.nodes[:idx] + [[lt,v1], [rt,v2], [lb,v3], [rb,v4]] +  self.nodes[idx+1:]

            # print([v for _,v in self.nodes])
            
    def count_patches(self):
        """Returns the current number of leaf nodes in the tree.

        Returns:
            Number of nodes in `self.nodes`.
        """
        return len(self.nodes)

    def serialize(self, img, size=(8,8,3)):
        """Extracts and resizes each leaf node's patch from `img` into a fixed-length sequence.

        Each node's variable-sized square region is extracted from `img` and
        resized (bicubic interpolation) to `size`. Pads with zero patches if the
        tree has fewer than `fixed_length` nodes.

        Args:
            img: Image to extract patches from, shape (H, W, Channel).
            size: Target `(h, w, channel)` size for every patch.

        Returns:
            A tuple `(seq_patch, seq_size, seq_pos)`: `seq_patch` is a list of
            `fixed_length` resized patches, `seq_size` a list of each node's
            original side length (0 for padding), and `seq_pos` a list of each
            node's center coordinates (`(-1, -1)` for padding).
        """

        seq_patch = []
        seq_size = []
        seq_pos = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        h2,w2,c2 = size
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            assert h1==w1, "Need squared input."
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_CUBIC)
            # assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
        if len(seq_patch)<self.fixed_length:
            # import pdb
            # pdb.set_trace()
            if c2 > 1:
                seq_patch += [np.zeros(shape=(h2,w2,c2))] * (self.fixed_length-len(seq_patch))
            else:
                seq_patch += [np.zeros(shape=(h2,w2))] * (self.fixed_length-len(seq_patch))
            seq_size += [0]*(self.fixed_length-len(seq_size))
            seq_pos += [tuple([-1,-1])]*(self.fixed_length-len(seq_pos))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
        assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
        return seq_patch, seq_size, seq_pos

    def serialize_labels(self, img, size=(8,8,3)):
        """Like `serialize`, but resizes each patch with nearest-neighbor interpolation.

        Intended for label/segmentation-mask images, where nearest-neighbor
        resizing avoids introducing invalid interpolated class values.

        Args:
            img: Label image to extract patches from, shape (H, W, Channel).
            size: Target `(h, w, channel)` size for every patch.

        Returns:
            A tuple `(seq_patch, seq_size, seq_pos)`, as in `serialize`.
        """

        seq_patch = []
        seq_size = []
        seq_pos = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        h2,w2,c2 = size
        
        for i in range(len(seq_patch)):
            h1, w1, c1 = seq_patch[i].shape
            assert h1==w1, "Need squared input."
            seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_NEAREST)
            # assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
        if len(seq_patch)<self.fixed_length:
            # import pdb
            # pdb.set_trace()
            if c2 > 1:
                seq_patch += [np.zeros(shape=(h2,w2,c2))] * (self.fixed_length-len(seq_patch))
            else:
                seq_patch += [np.zeros(shape=(h2,w2))] * (self.fixed_length-len(seq_patch))
            seq_size += [0]*(self.fixed_length-len(seq_size))
            seq_pos += [tuple([-1,-1])]*(self.fixed_length-len(seq_pos))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
        assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
        return seq_patch, seq_size, seq_pos
    
    def deserialize(self, seq, patch_size, channel):
        """Reassembles a flat sequence of predicted patches back into a full-size image.

        Inverse of `serialize`: reshapes `seq` into per-node patches and writes each
        one into its node's location via `Rect.set_area` (which resizes it back up
        to the node's actual size).

        Args:
            seq: Flat array of predicted patch values, reshaped to (fixed_length,
                patch_size, patch_size, channel).
            patch_size: Side length each patch is stored at.
            channel: Number of channels.

        Returns:
            Reconstructed image, shape matching `self.domain` with `channel`
            channels appended.
        """

        H,W = self.domain.shape
        seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, channel))
        seq = seq.astype(int)
        mask = np.zeros(shape=(H, W, channel))
        print("demask:", mask.shape)
        
        # mask = np.expand_dims(mask, axis=-1)
        for idx,(bbox,value) in enumerate(self.nodes):
            pred_mask = seq[idx, ...]
            mask = bbox.set_area(mask, pred_mask)
        return mask
    
    def draw(self, ax, c='grey', lw=1):
        """Draws the outline of every leaf node's rectangle onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Unused; kept for interface compatibility.
            lw: Unused; kept for interface compatibility.
        """
        for bbox,value in self.nodes:
            bbox.draw(ax=ax)

    def draw_area(self, ax, c='green', lw=1):
        """Draws every leaf node's rectangle filled with color `c` onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Fill color.
            lw: Line width of each rectangle's outline.
        """
        for bbox,value in self.nodes:
            bbox.draw_area(ax=ax, c=c, lw=lw)

    def draw_rescale(self, ax, c='green', lw=1):
        """Draws a fixed-size marker at every leaf node's top-left corner onto a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Fill color.
            lw: Line width of each marker's outline.
        """
        for bbox,value in self.nodes:
            bbox.draw_rescale(ax=ax, c=c, lw=lw)

    def draw_zorder(self, ax, c='red', lw=1):
        """Plots a line connecting the centers of every leaf node, in node order.

        Args:
            ax: Matplotlib axes to draw onto.
            c: Unused; the line is always drawn in red.
            lw: Unused; the line width is always 1.
        """
        xs = []
        ys = []
        for bbox,value in self.nodes:
            x,y = bbox.get_center()
            xs += [x]
            ys += [y]
        ax.plot(xs, ys, color='red', linewidth=1)
        
