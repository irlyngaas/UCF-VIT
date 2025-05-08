import numpy as np
import torch
import cv2 as cv
from scipy.interpolate import RegularGridInterpolator

class Cube:
    def __init__(self, x1, x2, y1, y2, z1, z2) -> None:
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
    
    def contains(self, domain):
        patch = domain[self.z1:self.z2, self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
        
    def get_area(self, img):
        return img[self.z1:self.z2, self.y1:self.y2, self.x1:self.x2]

    def get_coord(self):
        return self.x1,self.x2,self.y1,self.y2,self.z1,self.z2

    def get_size(self):
        return self.x2-self.x1, self.y2-self.y1, self.z2-self.z1
    
    def get_center(self):
        return (self.x2+self.x1)/2, (self.y2+self.y1)/2, (self.z2+self.z1)/2

class FixedOctTree:
    def __init__(self, domain, fixed_length=128,) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()

    def _build_tree(self):
        #channel, height, width, depth = self.domain.shape
        h, w, d = self.domain.shape
        assert h>0 and w >0 and d>0, "Wrong img size."
        root = Cube(0,h,0,w,0,d)
        self.nodes = [[root, root.contains(self.domain)]]
        while len(self.nodes) < self.fixed_length:
            bbox, value = max(self.nodes, key=lambda x:x[1])
            idx = self.nodes.index([bbox, value])
            if bbox.get_size()[0] == 2:
                break

            x1,x2,y1,y2,z1,z2 = bbox.get_coord()
            n1 = Cube(x1, int((x1+x2)/2), y1, int((y1+y2)/2), z1, int((z1+z2)/2))
            v1 = n1.contains(self.domain)
            n2 = Cube(int((x1+x2)/2), x2, y1, int((y1+y2)/2), z1, int((z1+z2)/2))
            v2 = n2.contains(self.domain)
            n3 = Cube(x1, int((x1+x2)/2), int((y1+y2)/2), y2, z1, int((z1+z2)/2))
            v3 = n3.contains(self.domain)
            n4 = Cube(int((x1+x2)/2), x2, int((y1+y2)/2), y2, z1, int((z1+z2)/2))
            v4 = n4.contains(self.domain)
            n5 = Cube(x1, int((x1+x2)/2), y1, int((y1+y2)/2), int((z1+z2)/2), z2)
            v5 = n5.contains(self.domain)
            n6 = Cube(int((x1+x2)/2), x2, y1, int((y1+y2)/2), int((z1+z2)/2), z2)
            v6 = n6.contains(self.domain)
            n7 = Cube(x1, int((x1+x2)/2), int((y1+y2)/2), y2, int((z1+z2)/2), z2)
            v7 = n7.contains(self.domain)
            n8 = Cube(int((x1+x2)/2), x2, int((y1+y2)/2), y2, int((z1+z2)/2), z2)
            v8 = n8.contains(self.domain)

            self.nodes = self.nodes[:idx] + [[n1,v1], [n2,v2], [n3,v3], [n4,v4],[n5,v5], [n6,v6], [n7,v7], [n8,v8]] +  self.nodes[idx+1:]

    def serialize(self, img, size=(8,8,8)):
        seq_patch = []
        seq_size = []
        seq_pos = []
        for bbox,value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        h2,w2,d2 = size
        
        for i in range(len(seq_patch)):
            h1, w1, d1 = seq_patch[i].shape
            assert h1==w1==d1, "Need squared input."

            h1_ = np.linspace(0,h1,h1)
            w1_ = np.linspace(0,w1,w1)
            d1_ = np.linspace(0,d1,d1)
            #4 to 8 -> (0,1,2,3,4) 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
            #2 to 4 -> (0,2) to (0, .667, 1.3667, 2)
            #_SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3}
            interp_fct = RegularGridInterpolator((h1_, w1_, d1_), seq_patch[i])
            #interp_fct = RegularGridInterpolator((h1_, w1_, d1_), seq_patch[i], method='cubic')
            patch_ = np.zeros([h2,w2,d2])
            h2_ = np.linspace(0,h1,h2)
            w2_ = np.linspace(0,w1,w2)
            d2_ = np.linspace(0,d1,d2)
            for m in range (len(h2_)):
                for n in range (len(w2_)):
                    for o in range (len(d2_)):
                        patch_[m,n,o] = interp_fct([h2_[m], w2_[n], d2_[o]])
            seq_patch[i] = patch_
            # assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
        if len(seq_patch)<self.fixed_length:
            # import pdb
            # pdb.set_trace()
            #if c2 > 1:
            #    seq_patch += [np.zeros(shape=(h2,w2,d2,c2))] * (self.fixed_length-len(seq_patch))
            #else:
            seq_patch += [np.zeros(shape=(h2,w2,d2))] * (self.fixed_length-len(seq_patch))
            seq_size += [0]*(self.fixed_length-len(seq_size))
            seq_pos += [tuple([-1,-1,-1])]*(self.fixed_length-len(seq_pos))
        elif len(seq_patch)>self.fixed_length:
            pass
            # random_drop
        assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
        assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
        return seq_patch, seq_size, seq_pos
