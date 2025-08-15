import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics import FrechetInceptionDistance
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler
from UCF_VIT.utils.misc import  unpatchify
from einops import rearrange
import torch.distributed as dist
## plots
def plotLoss(lossVec, save_path='./'):
    loss_array = np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lossVec])

    fig, ax = plt.subplots(1, 1, facecolor='w')
    ax.plot(loss_array, '-k', label='train')
    ax.set_yscale('log')
    plt.legend()
    ax.set_ylim([min(loss_array), 1])
    fig.savefig(save_path, format='png', dpi=150)
    plt.close(fig)



def plot_2D_array_slices(arrays,filename='2D_slices.png'):

    colormap = "gray"
    B = arrays[0].shape[0]         # number of samples
    T = len(arrays)                # number of time steps

    fig, ax = plt.subplots(B, T, figsize=(T * 2, B * 2))

    for i in range(B):
        for j in range(T):
            image_slice = arrays[j][i, ..., 0]  # shape: (W, D)

            # # Take center slice along the first axis (X direction)
            # center_slice = vol[ :, :]  # (W, D)

            if B > 1 and T > 1:
                ax_ij = ax[i, j]
            elif B == 1:
                ax_ij = ax[j]        # Single row
            else:
                ax_ij = ax[i]        # Single column

            ax_ij.imshow(image_slice, cmap=colormap)
            ax_ij.axis('off')
            if i == B - 1:
                print(f'for the last sampl, at time step {j}, the max/min values are: [{np.max(image_slice),np.min(image_slice)}]')
            
    plt.subplots_adjust(left=0.01, right=0.99, hspace=0.05, wspace=0.05, top=0.99, bottom=0.01)
    fig.savefig(filename, format='png', dpi=300)
    plt.close(fig)
    
    

def plot_3D_array_center_slices_up(arrays, filename='3D_center_slices.png'):
    """
    Plot center X-slices (axis 0) for each sample and timestep.

    arrays: list of arrays, each with shape [B, H, W, D, C] for a diffusion step.
    """
    colormap = "gray"
    B = arrays[0].shape[0]         # number of samples
    T = len(arrays)                # number of time steps

    fig, ax = plt.subplots(B, T, figsize=(T * 2, B * 2))

    for i in range(B):
        for j in range(T):
            vol = arrays[j][i, ..., 0]  # shape: (H, W, D)

            # Take center slice along the first axis (X direction)
            center_slice = vol[vol.shape[0] // 2, :, :]  # (W, D)

            if B > 1 and T > 1:
                ax_ij = ax[i, j]
            elif B == 1:
                ax_ij = ax[j]        # Single row
            else:
                ax_ij = ax[i]        # Single column

            ax_ij.imshow(center_slice, cmap=colormap)
            ax_ij.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, hspace=0.05, wspace=0.05, top=0.99, bottom=0.01)
    fig.savefig(filename, format='png', dpi=300)
    plt.close(fig)



def plot_3D_array_center_slices(arrays,filename='3D_center_slices.png'):
    # arrays is a list with N entries (corresponding to N reverse diffusion time steps)
    # array[j] is a [B,C,H,W,D] array corresponding to the generated modality at time step t 

    colormap = "gray"#plt.cm.jet
    fig,ax = plt.subplots(arrays[0].shape[0],len(arrays),figsize=(10,10))#,facecolor='w',subplot_kw={'projection':'3d'})
    
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



# def sample_images(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):

#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
#     times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

#     images = []
#     if not twoD:
#         z = torch.randn(num_samples, 1, res[0], res[1], res[2])
#     else:
#         z = torch.randn(num_samples, 1, res[1], res[2])

#     with torch.no_grad():
#         for t in reversed(range(1, num_time_steps)):
#             if not twoD:
#                 e = torch.randn(num_samples, 1, res[0], res[1], res[2])
#             else:
#                 e = torch.randn(num_samples, 1, res[1], res[2])

#             t = [t]
#             temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
#             predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var]) 
#             predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#             z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*predicted_noise.cpu())
#             if t[0] in times:
#                 images.append(z)
#             z = z + (e*torch.sqrt(scheduler.beta[t]))

#         temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])))
#         predicted_noise = model(z.to(precision_dt).to(device), torch.tensor([0]).to(device), [var])
#         predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#         x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*predicted_noise.cpu())
#         images.append(x)

#     if not twoD:
#         images = [rearrange(img.squeeze(0), 'b c h w d -> b h w d c').detach().numpy() for img in images]
#     else:
#         images = [rearrange(img.squeeze(0), 'b c h w -> b h w c').detach().numpy() for img in images]
    
#     # images = np.array(images)
#     # images = images.astype('float32')

#     if not twoD:
#         plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#         # # plot_3D_array_center_slices(images, filename=os.path.join(save_path, '3D_gen_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#         # images = np.array(images)
#         # images = images.astype('float32')
#         # np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_%irank%i.npz' %(var, epoch, res[0], res[1], res[2], dist.get_rank())),images)
#     else:
#         plot_2D_array_slices(images, filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())))
#         # np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_rank%i.npz' %(var, epoch, res[1], res[2], dist.get_rank())),images)


## correct sample images (supposedly!)

def sample_images(
    model,
    var,                       # kept for API compatibility (unused here)
    device,
    res,
    precision_dt,
    patch_size,
    epoch=0,
    num_samples=10,
    twoD=False,
    save_path="figures",
    num_time_steps=1000,
):
    """
    Minimal fixes for dtype/device:
      - Keep tensors on one device.
      - Allow bf16 compute but cast to fp32 right before numpy conversion.
    """

    os.makedirs(save_path, exist_ok=True)

    # model to eval
    inner = model.module if hasattr(model, "module") else model
    inner.eval()

    # scheduler on CPU; we'll index on CPU and move values to `device`
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    # shapes
    if twoD:
        if isinstance(res, (tuple, list)):
            H, W = int(res[0]), int(res[1])
        else:
            H = W = int(res)
        z = torch.randn(num_samples, 1, H, W, device=device, dtype=precision_dt)
    else:
        if isinstance(res, (tuple, list)):
            D, H, W = int(res[0]), int(res[1]), int(res[2])
        else:
            D = H = W = int(res)
        z = torch.randn(num_samples, 1, D, H, W, device=device, dtype=precision_dt)

    images = []
    times_to_save = {0, 15, 50, 100, 200, 300, 400, 550, 700,num_time_steps - 1}

    with torch.no_grad():
        for t_scalar in reversed(range(1, num_time_steps)):
            # CPU indices for CPU scheduler tensors, then move gathered values to GPU with same dtype as z
            t_idx  = torch.full((num_samples,), t_scalar, dtype=torch.long, device=scheduler.beta.device)
            beta_t = scheduler.beta[t_idx].to(device=device, dtype=z.dtype)     # β_t
            abar_t = scheduler.alpha[t_idx].to(device=device, dtype=z.dtype)    # \bar{α}_t
            alpha_t = (1.0 - beta_t)                                            # α_t

            view_shape = (num_samples,) + (1,) * (z.ndim - 1)
            beta_t_b  = beta_t.view(view_shape)
            abar_t_b  = abar_t.view(view_shape)
            alpha_t_b = alpha_t.view(view_shape)

            # model time tensor on same device as z
            t_model = t_idx.to(device)

            # predict noise ε̂
            eps_hat = model(z, t_model, [var])
            eps_hat = unpatchify(eps_hat, z,patch_size, twoD)

            # # μ_t = 1/√α_t * ( x_t − (β_t / √(1 − \bar{α}_t)) * ε̂ )
            # mu = (z - (beta_t_b / torch.sqrt(1.0 - abar_t_b)) * eps_hat) / torch.sqrt(alpha_t_b)
            
            # μ_t = 1/√α_t * ( x_t − (β_t / √(1 − \bar{α}_t)) * ε̂ )
            denom = torch.sqrt(torch.clamp(1.0 - abar_t_b, min=1e-12))
            mu = (z - (beta_t_b / denom) * eps_hat) / torch.sqrt(alpha_t_b)


            if t_scalar > 1:
                e = torch.randn_like(z, device=device, dtype=z.dtype)
                z = mu + torch.sqrt(beta_t_b) * e
            else:
                z = mu

            if t_scalar in times_to_save:
                images.append(z.clone())

        images.append(z.clone())  # final

    # Cast to float32 right before numpy to avoid "Unsupported ScalarType BFloat16"
    if not twoD:
        images = [rearrange(img.float().detach().cpu(), "b c d h w -> b d h w c").numpy() for img in images]
        plot_3D_array_center_slices_up(
            images,
            filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png'
                                  % (var, epoch, (res[0] if isinstance(res,(tuple,list)) else res),
                                     (res[1] if isinstance(res,(tuple,list)) else res),
                                     (res[2] if isinstance(res,(tuple,list)) else res),
                                     dist.get_rank()))
        )
    else:
        images = [rearrange(img.float().detach().cpu(), "b c h w -> b h w c").numpy() for img in images]
        plot_2D_array_slices(
            images,
            filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png'
                                  % (var, epoch,
                                     (res[0] if isinstance(res,(tuple,list)) else res),
                                     (res[1] if isinstance(res,(tuple,list)) else res),
                                     dist.get_rank()))
        )

    #  # to numpy for plotting
    # if twoD:
    #     images_np = [rearrange(img, "b c h w -> b h w c").detach().cpu().numpy() for img in images]
    #     plot_2D_array_slices(images_np, filename=os.path.join(save_path, f"2D_gen_{epoch}_{H}_{W}.png"))
    # else:
    #     images_np = [rearrange(img, "b c d h w -> b d h w c").detach().cpu().numpy() for img in images]
    #     plot_3D_array_center_slices_up(images_np, filename=os.path.join(save_path, f"3D_GEN_centerSlice_{epoch}_{D}_{H}_{W}.png"))

# def sample_images(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):

#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
#     times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

#     images = []
#     if not twoD:
#         z = torch.randn(num_samples, 1, res[0], res[1], res[2])
#     else:
#         z = torch.randn(num_samples, 1, res[1], res[2])

#     with torch.no_grad():
#         for t in reversed(range(1, num_time_steps)):
#             if not twoD:
#                 e = torch.randn(num_samples, 1, res[0], res[1], res[2])
#             else:
#                 e = torch.randn(num_samples, 1, res[1], res[2])

#             t = [t]
#             temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
#             predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var]) 
#             predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#             z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*predicted_noise.cpu())
#             if t[0] in times:
#                 images.append(z)
#             z = z + (e*torch.sqrt(scheduler.beta[t]))

#         temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])))
#         predicted_noise = model(z.to(precision_dt).to(device), torch.tensor([0]).to(device), [var])
#         predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#         x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*predicted_noise.cpu())
#         images.append(x)

#     if not twoD:
#         images = [rearrange(img.squeeze(0), 'b c h w d -> b h w d c').detach().numpy() for img in images]
#     else:
#         images = [rearrange(img.squeeze(0), 'b c h w -> b h w c').detach().numpy() for img in images]

#     if save_path is not None:
#         if not twoD:
#             plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#         else:
#             plot_2D_array_slices(images, filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())))
#             # fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10),facecolor='w')
#             # for i in range(images[0].shape[0]):
#             #     for j in range(len(times)):
#             #         ax[i,j].axis('off')
#             #         ax[i,j].imshow(images[j][i,:,:,0], interpolation='none')
#             # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
#             # fig.savefig(os.path.join(save_path,'2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())), format='png', dpi=300)
#             # plt.close(fig)


def save_intermediate_data(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

    images = []
    if not twoD:
        z = torch.randn(num_samples, 1, res[0], res[1], res[2])
    else:
        z = torch.randn(num_samples, 1, res[1], res[2])

    with torch.no_grad():
        for t in reversed(range(1, num_time_steps)):
            if not twoD:
                e = torch.randn(num_samples, 1, res[0], res[1], res[2])
            else:
                e = torch.randn(num_samples, 1, res[1], res[2])

            t = [t]
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
            predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var]) 
            predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
            z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*predicted_noise.cpu())
            if t[0] in times:
                images.append(z)
            z = z + (e*torch.sqrt(scheduler.beta[t]))

        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])))
        predicted_noise = model(z.to(precision_dt).to(device), torch.tensor([0]).to(device), [var])
        predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*predicted_noise.cpu())
        images.append(x)

    if not twoD:
        images = [rearrange(img.squeeze(0), 'b c h w d -> b h w d c').detach().numpy() for img in images]
    else:
        images = [rearrange(img.squeeze(0), 'b c h w -> b h w c').detach().numpy() for img in images]
    
    # images = np.array(images)
    # images = images.astype('float32')

    if not twoD:
        plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
        # plot_3D_array_center_slices(images, filename=os.path.join(save_path, '3D_gen_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
        images = np.array(images)
        images = images.astype('float32')
        np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_%irank%i.npz' %(var, epoch, res[0], res[1], res[2], dist.get_rank())),images)
    else:
        plot_2D_array_slices(images, filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())))
        np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_rank%i.npz' %(var, epoch, res[1], res[2], dist.get_rank())),images)



def extract_rgb_like_slices(volume, num_samples=5):
    """
    Converts a 3D volume (1, D, H, W) to multiple (3, H, W) pseudo-RGB slices.
    """
    _, D, H, W = volume.shape
    indices = torch.linspace(1, D - 2, steps=num_samples).long()
    slices = []

    for i in indices:
        s = torch.cat([
            volume[:, i - 1, :, :],
            volume[:, i, :, :],
            volume[:, i + 1, :, :]
        ], dim=0)  # (3, H, W)
        slices.append(s)

    return torch.stack(slices)  # (num_samples, 3, H, W)


def save_intermediate_data_with_fid(model, var, device, res, precision_dt, patch_size,
                                    epoch=0, num_samples=10, twoD=False,
                                    save_path='figures', num_time_steps=1000,
                                    test_volume_path=None,downscale=1):
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

    images = []

    # Initial noise input
    if not twoD:
        z = torch.randn(num_samples, 1, res[0], res[1], res[2])
    else:
        z = torch.randn(num_samples, 1, res[1], res[2])

    with torch.no_grad():
        for t in reversed(range(1, num_time_steps)):
            if not twoD:
                e = torch.randn(num_samples, 1, res[0], res[1], res[2])
            else:
                e = torch.randn(num_samples, 1, res[1], res[2])

            t = [t]
            temp = (scheduler.beta[t] / ((torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
            predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var])
            predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
            z = (1 / torch.sqrt(1 - scheduler.beta[t])) * z - (temp * predicted_noise.cpu())
            if t[0] in times:
                images.append(z)
            z = z + (e * torch.sqrt(scheduler.beta[t]))

        # Final step at t = 0
        temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
        predicted_noise = model(z.to(precision_dt).to(device), torch.tensor([0]).to(device), [var])
        predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
        x = (1 / torch.sqrt(1 - scheduler.beta[0])) * z - (temp * predicted_noise.cpu())
        images.append(x)

    # Rearrange and convert to numpy
    if not twoD:
        images = [rearrange(img, 'b c h w d -> b h w d c').detach().numpy() for img in images]
    else:
        images = [rearrange(img, 'b c h w -> b h w c').detach().numpy() for img in images]
    
    # images = np.array(images).astype('float32')
    # images = images.astype('float32')

    # Save visuals and arrays
    if not twoD:
        print("writing the center slice!")
        plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_FID_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
    else:
        fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10),facecolor='w')
        for i in range(images[0].shape[0]):
            for j in range(len(times)):
                ax[i,j].axis('off')
                ax[i,j].imshow(images[j][i,:,:,0], interpolation='none')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
        fig.savefig(os.path.join(save_path,'2D_gen_FID_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())), format='png', dpi=300)
        plt.close(fig)
        
    # ----- FID Calculation -----
    if test_volume_path is not None and dist.get_rank() == 0:
        try:
            idx = random.randint(0, num_samples - 1)
            gen_vol = images[-1][idx, ..., 0]  # (H, W, D)

            # Load GT volume
            gt_vol = np.load(test_volume_path)
            if gt_vol.ndim != 3:
                raise ValueError("Expected 3D test volume (H, W, D)")
            gt_vol = gt_vol.astype('float32')
            if downscale>1:
                gt_vol = gt_vol[::downscale,::downscale,::downscale]

            # Extract RGB-style 2D slices
            gen_slices = extract_rgb_slices(gen_vol)
            gt_slices = extract_rgb_slices(gt_vol)

            # Convert to torch tensors on device
            gen_tensor = torch.tensor(gen_slices).float().to(device)
            gt_tensor = torch.tensor(gt_slices).float().to(device)

            # Compute FID
            fid_metric = FrechetInceptionDistance().to(device)
            fid_metric.update(gt_tensor, real=True)
            fid_metric.update(gen_tensor, real=False)
            fid = fid_metric.compute()

            print(f"[Epoch {epoch}] FID (1-of-N sample vs. test volume): {fid.item():.4f}", flush=True)
            return fid.item()

        except Exception as e:
            print(f"[Epoch {epoch}] FID calculation failed: {e}", flush=True)
            return None

    return None

