import matplotlib.pyplot as plt
from typing import List
from einops import rearrange
import numpy as np
import math,os

def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10,1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x[:,:,16,:], 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()

def plotLoss(lossVec):
    fig,ax=plt.subplots(1,1,facecolor='w')
    ax.plot(np.array(lossVec)[:,0],'-k',label='train')
    ax.plot(np.array(lossVec)[:,1],'-r',label='val')
    ax.set_yscale('log')
    plt.legend()
    ax.set_ylim([np.array(lossVec).min()/2,1])
    fig.savefig('loss.png',format='png',dpi=150)
    plt.close(fig)


def plotExamples(x,c,filename=None):

    N = int(math.floor((x.shape[0])**0.5))

    #3d
    if len(x.shape)==5:
        cmap = plt.get_cmap('viridis')
        fig,ax = plt.subplots(N,N,figsize=(12,12),facecolor='w',subplot_kw={'projection':'3d'})
        ax=ax.ravel()
        for i in range(ax.shape[0]):
            x_slice = x[i,0,...][x[i,0,...].shape[0]//2, :, :]
            y_slice = x[i,0,...][:, x[i,0,...].shape[1]//2, :]
            z_slice = x[i,0,...][:, :, x[i,0,...].shape[2]//2]
            plot_quadrants(ax[i], x_slice, 'x', x[i,0,...].shape[0]//2, cmap, x[i,0,...].shape, -2, 2)
            plot_quadrants(ax[i], y_slice, 'y', x[i,0,...].shape[1]//2, cmap, x[i,0,...].shape, -2, 2)
            plot_quadrants(ax[i], z_slice, 'z', x[i,0,...].shape[2]//2, cmap, x[i,0,...].shape, -2, 2)
            ax[i].set_title('class %i' %c[i,0],pad=-30)
            ax[i].tick_params(axis='both', which='major', labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        plt.subplots_adjust(left=0.01,right=0.99,hspace=-0.05,wspace=-0.05,top=0.99,bottom=0.01)
        if filename is None: filename='examples_3d.png'
        fig.savefig(os.path.join('figures',filename),format='png',dpi=300)
        plt.close(fig)
    else:
        cmap = plt.get_cmap('viridis')
        fig,ax = plt.subplots(N,N,figsize=(12,12),facecolor='w',subplot_kw={'projection':'3d'})
        ax=ax.ravel()
        for i in range(ax.shape[0]):
            ax[i].imshow(x[i,0,...],interpolation='none',cmap=cmap,vmin=-2,vmax=2)
            ax[i].set_axis_off()
        plt.subplots_adjust(left=0.01,right=0.99,hspace=-0.05,wspace=-0.05,top=0.99,bottom=0.01)
        if filename is None: filename='examples_2d.png'
        fig.savefig(os.path.join('figures',filename),format='png',dpi=300)
        plt.close(fig)

# https://github.com/matplotlib/matplotlib/issues/3919

def plot_quadrants(ax, slice_data, fixed_coord, coord_val, colormap, shape, min_val, max_val):
    half_shape = (shape[1] // 2, shape[2] // 2)
    quadrants = [
        slice_data[:half_shape[0], :half_shape[1]],
        slice_data[:half_shape[0], half_shape[1]:],
        slice_data[half_shape[0]:, :half_shape[1]],
        slice_data[half_shape[0]:, half_shape[1]:]
    ]
    
    for i, quad in enumerate(quadrants):
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:half_shape[0], 0:half_shape[1]]
            X = coord_val * np.ones_like(Y)
            Y_offset = (i // 2) * half_shape[0]
            Z_offset = (i % 2) * half_shape[1]
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1, facecolors=colormap((quad-min_val)/(max_val-min_val)), shade=False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:half_shape[0], 0:half_shape[1]]
            Y = coord_val * np.ones_like(X)
            X_offset = (i // 2) * half_shape[0]
            Z_offset = (i % 2) * half_shape[1]
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1, facecolors=colormap((quad-min_val)/(max_val-min_val)), shade=False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:half_shape[0], 0:half_shape[1]]
            Z = coord_val * np.ones_like(X)
            X_offset = (i // 2) * half_shape[0]
            Y_offset = (i % 2) * half_shape[1]
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1, facecolors=colormap((quad-min_val)/(max_val-min_val)), shade=False)

def plot_3D_array_slices(arrays,filename='3Dslices.png'):
    # arrays is a list with N entries (corresponding to N reverse diffusion time steps)
    # array[j] is a [B,C,H,W,D] array corresponding to the generated modality at time step t 

    colormap = plt.cm.jet
    fig,ax = plt.subplots(arrays[0].shape[0],len(arrays),figsize=(10,10),facecolor='w',subplot_kw={'projection':'3d'})

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            x_slice = arrays[j][i,...,0][arrays[j][i,...,0].shape[0]//2, :, :]
            y_slice = arrays[j][i,...,0][:, arrays[j][i,...,0].shape[1]//2, :]
            z_slice = arrays[j][i,...,0][:, :, arrays[j][i,...,0].shape[2]//2]
            plot_quadrants(ax[i][j], x_slice, 'x', arrays[j][i,...,0].shape[0]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            plot_quadrants(ax[i][j], y_slice, 'y', arrays[j][i,...,0].shape[1]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            plot_quadrants(ax[i][j], z_slice, 'z', arrays[j][i,...,0].shape[2]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            ax[i][j].set_axis_off()

    plt.subplots_adjust(left=0.01,right=0.99,hspace=-0.05,wspace=-0.05,top=0.99,bottom=0.01)
    fig.savefig(filename,format='png',dpi=300)
    plt.close(fig)


def plot_3D_array_center_slices(arrays,filename='3D_center_slices.png'):
    # arrays is a list with N entries (corresponding to N reverse diffusion time steps)
    # array[j] is a [B,C,H,W,D] array corresponding to the generated modality at time step t 

    colormap = "gray"#plt.cm.jet
    fig,ax = plt.subplots(arrays[0].shape[0],len(arrays),figsize=(10,10),facecolor='w',subplot_kw={'projection':'3d'})

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            x_slice = arrays[j][i,...,0][arrays[j][i,...,0].shape[0]//2, :, :]
            # y_slice = arrays[j][i,...,0][:, arrays[j][i,...,0].shape[1]//2, :]
            # z_slice = arrays[j][i,...,0][:, :, arrays[j][i,...,0].shape[2]//2]
            ax[i][j].imshow(x_slice,cmap=colormap)
            # plot_quadrants(ax[i][j], x_slice, 'x', arrays[j][i,...,0].shape[0]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            # plot_quadrants(ax[i][j], y_slice, 'y', arrays[j][i,...,0].shape[1]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            # plot_quadrants(ax[i][j], z_slice, 'z', arrays[j][i,...,0].shape[2]//2, colormap, arrays[j][i,...,0].shape, -2, 2)
            ax[i][j].set_axis_off()

    plt.subplots_adjust(left=0.01,right=0.99,hspace=-0.05,wspace=-0.05,top=0.99,bottom=0.01)
    fig.savefig(filename,format='png',dpi=300)
    plt.close(fig)



def plot_2D_array_slices(arrays,filename='2D_slices.png'):
    # arrays is a list with N entries (corresponding to N reverse diffusion time steps)
    # array[j] is a [B,C,H,W,D] array corresponding to the generated modality at time step t 

    colormap = "gray"#plt.cm.jet
    fig,ax = plt.subplots(arrays[0].shape[0],len(arrays),figsize=(10,10),facecolor='w',subplot_kw={'projection':'3d'})

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].imshow(arrays[j][i,:,:,0],cmap=colormap)
            ax[i][j].set_axis_off()

    plt.subplots_adjust(left=0.01,right=0.99,hspace=-0.05,wspace=-0.05,top=0.99,bottom=0.01)
    fig.savefig(filename,format='png',dpi=300)
    plt.close(fig)
