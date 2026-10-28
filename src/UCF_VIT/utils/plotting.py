"""Standalone visualization helpers, ported from the old SST branch (dev
scripts, not part of the core training/dataloading pipeline) -- no
dependency on model/dataloader code, safe to use with any dataset.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(train_loss_history, val_loss_history=None):
    """Plots training (and optionally validation) loss history on a log-scale y-axis.

    Args:
        train_loss_history: Sequence of per-epoch training loss values.
        val_loss_history: Optional sequence of per-epoch validation loss values.
    """
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 18})
    plt.title('Learning curve')
    plt.plot(train_loss_history, label='training')
    if val_loss_history:
        plt.plot(val_loss_history, label='validation', alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(r'Loss ($mse$)')
    plt.legend(frameon=False)


def get_1Dgrid(Lh, nx, nxoffset, nxsl, nxskip):
    """Computes real-space 1D grid coordinates for a sub-domain/tile.

    Args:
        Lh: Physical length of the full domain along this axis.
        nx: Number of grid points in the full domain along this axis.
        nxoffset: Starting grid-point index of the sub-domain along this axis.
        nxsl: Number of grid points in the sub-domain along this axis.
        nxskip: Stride (in grid points) between consecutive sub-domain samples.

    Returns:
        A length-`nxsl` array of real-space coordinates for the sub-domain.
    """
    dx = Lh / nx
    xin = 0 + (dx * nxoffset)
    xfi = xin + dx * nxsl * nxskip
    return np.linspace(xin, xfi, nxsl)


def plot_contour_box(x, y, z, datacube, gravity):
    """Plots a 3D contour box: filled contours on 3 faces of a volume, plus its wireframe edges.

    Args:
        x: 1D grid in the x-direction.
        y: 1D grid in the y-direction.
        z: 1D grid in the z-direction.
        datacube: 3D array of values over the (x, y, z) grid.
        gravity: Direction of gravity, `"y"` or `"z"` -- determines which
            faces are plotted and how axes are labeled/oriented.

    Returns:
        The `matplotlib` 3D axis the plot was drawn on.

    Raises:
        ValueError: If `gravity` is neither `"y"` nor `"z"`.
    """
    plt.clf()
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 10})
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=20., azim=-35)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nxsl, nysl, nzsl = X.shape
    clevels = np.linspace(datacube.min() * 0.5, datacube.max() * 0.5, 101)
    kw = {
        'vmin': clevels.min(),
        'vmax': clevels.max(),
        'levels': clevels,
        'cmap': 'RdBu_r',
        'extend': 'both',
    }

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    edges_kw = dict(color='0.8', linewidth=0.5, zorder=1e3)

    if gravity == 'z':
        ax.contourf(X[:, 0, :], datacube[:, 0, :], Z[:, 0, :], zdir='y', offset=Y.min(), **kw)
        ax.contourf(X[:, :, -1], Y[:, :, -1], datacube[:, :, -1], zdir='z', offset=Z.max(), **kw)
        ax.contourf(datacube[-1, :, :], Y[-1, :, :], Z[-1, :, :], zdir='x', offset=X.max(), **kw)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], zmin, **edges_kw)
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
        ax.set(xlabel='X', ylabel='Y', zlabel='Z')
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        ax.set_box_aspect([int(nxsl / nzsl), int(nysl / nzsl), 1], zoom=1)
    elif gravity == 'y':
        ax.contourf(X[:, -1, :], Z[:, -1, :], datacube[:, -1, :], zdir='z', offset=Y.max(), **kw)
        ax.contourf(X[:, :, 0], datacube[:, :, 0], Y[:, :, 0], zdir='y', offset=Z.min(), **kw)
        ax.contourf(datacube[-1, :, :], Z[-1, :, :], Y[-1, :, :], zdir='x', offset=X.max(), **kw)
        ax.plot([xmax, xmax], [zmin, zmax], ymin, **edges_kw)
        ax.plot([xmax, xmax], [zmin, zmax], ymax, **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], ymin, **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], ymax, **edges_kw)
        ax.plot([xmax, xmax], [zmin, zmin], [ymin, ymax], **edges_kw)
        ax.set(xlabel='X', ylabel='Z', zlabel='Y')
        ax.set(xlim=[xmin, xmax], ylim=[zmin, zmax], zlim=[ymin, ymax])
        ax.set_box_aspect([int(nxsl / nysl), int(nzsl / nysl), 1], zoom=1)
    else:
        raise ValueError(f"Invalid gravity {gravity!r} -- choose 'y' or 'z'")

    ax.view_init(20, -45)
    ax.grid(False)
    fig.tight_layout()

    return ax
