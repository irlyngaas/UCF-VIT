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
    """
    Plot contour box.
    
    Args:
        x: 1D grid in x-direction
        y: 1D grid in y-direction
        z: 1D grid in z-direction
        datacube: 3D subdomain from raw data
        gravity: String denoting direction of gravity. 'y' or 'z'
    Returns:
        ax: Figure axis handle
    """
    plt.clf()
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 10})
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=20., azim=-35)
    # Get meshgrid and set cmap
    X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
    nxsl, nysl, nzsl = X.shape
    clevels = np.linspace(datacube.min()*0.5, datacube.max()*0.5, 101)
    # clevels = np.linspace(-0.2, 0.2, 101) # For P1F4R32 pressure
    kw = {
        'vmin': clevels.min(),
        'vmax': clevels.max(),
        'levels': clevels,
        'cmap': 'RdBu_r', #'Reds', #'RdBu_r',
        'extend': 'both', #'max', #'both'
        # 'alpha': 0.8,
    }
    
    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()    
    # edge args for box
    edges_kw = dict(color='0.8', linewidth=0.5, zorder=1e3)

    if gravity == 'z':
        # Plot contour surfaces
        A = ax.contourf(
        X[:, 0, :], datacube[:, 0, :], Z[:, 0, :],
        zdir='y', offset=Y.min(), **kw
        )
        B = ax.contourf(
            X[:, :, -1], Y[:, :, -1], datacube[:, :, -1],
            zdir='z', offset=Z.max(), **kw
        )
        C = ax.contourf(
            datacube[-1, :, :], Y[-1, :, :], Z[-1, :, :],
            zdir='x', offset=X.max(), **kw
        )
        # Plot edges
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
        ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], zmin, **edges_kw)
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
        ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
        # Set labels and zticks
        ax.set(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        aspectratio_x, aspectratio_z, aspectratio_y = int(nxsl/nzsl), 1, int(nysl/nzsl)
        ax.set_box_aspect([aspectratio_x, aspectratio_y, aspectratio_z], zoom=1)
    elif gravity == 'y':
        # Plot contour surfaces
        A = ax.contourf(
        X[:, -1, :], Z[:, -1, :], datacube[:, -1, :],
        zdir='z', offset=Y.max(), **kw
        )
        B = ax.contourf(
            X[:, :, 0], datacube[:, :, 0], Y[:, :, 0],
            zdir='y', offset=Z.min(), **kw
        )
        C = ax.contourf(
            datacube[-1, :, :], Z[-1, :, :], Y[-1, :, :],
            zdir='x', offset=X.max(), **kw
        )
        # Plot edges
        ax.plot([xmax, xmax], [zmin, zmax], ymin, **edges_kw)
        ax.plot([xmax, xmax], [zmin, zmax], ymax, **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], ymin, **edges_kw)
        ax.plot([xmin, xmax], [zmin, zmin], ymax, **edges_kw)
        ax.plot([xmax, xmax], [zmin, zmin], [ymin, ymax], **edges_kw)
        # Set labels and zticks
        ax.set(
            xlabel='X',
            ylabel='Z',
            zlabel='Y',
        )
        ax.set(xlim=[xmin, xmax], ylim=[zmin, zmax], zlim=[ymin, ymax])
        aspectratio_x, aspectratio_z, aspectratio_y = int(nxsl/nysl), int(nzsl/nysl), 1
        ax.set_box_aspect([aspectratio_x, aspectratio_z, aspectratio_y], zoom=1)
    else:
        raise Exception("Invalide gravity. Choose 'y' or 'z'")
    
    # cbar = fig.colorbar(A, ax=ax, orientation='horizontal', pad=-0.1, fraction=0.04)
    # cbar.set_label(r"$d\rho_t/dy$")  # or whatever label is appropriate
    # Set zoom and angle view
    ax.view_init(20, -45)
    ax.grid(False)
    # ax.axis("equal");
    # ax.axis("off");
    fig.tight_layout()

    return ax
