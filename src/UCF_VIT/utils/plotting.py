import numpy as np
import matplotlib.pyplot as plt

# # Function to plot learning curve(s)
def plot_learning_curve(train_loss_history, val_loss_history=None):
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 18})
    plt.title('Learning curve')
    plt.plot(train_loss_history, label='training')
    if val_loss_history: plt.plot(val_loss_history, label='validation',alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel(r'Loss ($mse$)')
    plt.legend(frameon=False);

# # Function to compute grid coordinates for subdomain/box
def get_1Dgrid(Lh, nx, nxoffset, nxsl, nxskip):
    dx = Lh/nx
    xin = 0 + (dx*nxoffset)
    xfi = xin + dx*nxsl*nxskip
    x = np.linspace(xin, xfi, nxsl)
    return x

# # Function for box plot of a given variable
def plot_contour_box(x, y, z, datacube, gravity):
    # Plot contour box
    plt.clf()
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 10})
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=20., azim=-35)
    xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
    clevels = np.linspace(0.5*datacube.min(), 0.5*datacube.max(), 101)
    kw = {
        'vmin': clevels.min(),
        'vmax': clevels.max(),
        'levels': clevels,
        'cmap': 'RdBu_r',
        'extend': 'both',
        'alpha': 0.8
    }
    # Plot contour surfaces
    A = ax.contourf(
        xx[:, -1, :], zz[:, -1, :], datacube[:, -1, :],
        zdir='z', offset=yy.max(), **kw
    )
    B = ax.contourf(
        xx[:, :, 0], datacube[:, :, 0], yy[:, :, 0],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        datacube[-1, :, :], zz[-1, :, :], yy[-1, :, :],
        zdir='x', offset=xx.max(), **kw
    )
    # Set limits of the plot from coord limits
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()
    zmin, zmax = zz.min(), zz.max()
    ax.set(xlim=[xmin, xmax], zlim=[ymin, ymax], ylim=[zmin, zmax])
    # Plot edges
    edges_kw = dict(color='0.5', linewidth=0.5, zorder=1e3)
    ax.plot([xmax, xmax], [zmin, zmax], ymin, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmax], ymax, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymin, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymax, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmin], [ymin, ymax], **edges_kw)
    # Set labels and zticks
    ax.set(
        xlabel='x',
        ylabel='z',
        zlabel='y',
    )
    # Set zoom and angle view
    ax.view_init(20, -45)
    nx = len(x)
    ny = len(y)
    nz = len(z)
    if gravity == 'z': aspectratio_x, aspectratio_z, aspectratio_y = int(nx/nz), 1             , int(nx/nz)
    if gravity == 'y': aspectratio_x, aspectratio_z, aspectratio_y = int(nx/ny), int(nx/ny), 1
    ax.set_box_aspect([aspectratio_x, aspectratio_z , aspectratio_y], zoom=1)
    ax.grid(False);

    return ax
