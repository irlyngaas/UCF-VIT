import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# from torcheval.metrics import FrechetInceptionDistance
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler
from UCF_VIT.utils.misc import  unpatchify
from einops import rearrange
import torch.distributed as dist
import math
## plots
def plotLoss(lossVec, save_path='./'):
    loss_array = np.array([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lossVec])

    fig, ax = plt.subplots(1, 1, facecolor='w')
    ax.plot(loss_array, '-k', label='train')
    ax.set_yscale('log')
    plt.legend()
    ax.set_ylim([min(loss_array), 1])
    if save_path.__contains__("rank"):
        plt.title("rank_" + save_path.split("rank")[1].split(".png")[0])
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



def sample_images(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):

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
        # # plot_3D_array_center_slices(images, filename=os.path.join(save_path, '3D_gen_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
        # images = np.array(images)
        # images = images.astype('float32')
        # np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_%irank%i.npz' %(var, epoch, res[0], res[1], res[2], dist.get_rank())),images)
    else:
        plot_2D_array_slices(images, filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())))
        # np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_rank%i.npz' %(var, epoch, res[1], res[2], dist.get_rank())),images)

# ## correct sample images (supposedly!)
# def sample_images(
#     model,
#     var,                       # kept for API compatibility (unused here)
#     device,
#     res,
#     precision_dt,
#     patch_size,
#     epoch=0,
#     num_samples=10,
#     twoD=False,
#     save_path="figures",
#     num_time_steps=1000,
# ):
#     """
#     Minimal fixes for dtype/device:
#       - Keep tensors on one device.
#       - Allow bf16 compute but cast to fp32 right before numpy conversion.
#     """

#     os.makedirs(save_path, exist_ok=True)

#     # model to eval
#     inner = model.module if hasattr(model, "module") else model
#     inner.eval()

#     # scheduler on CPU; we'll index on CPU and move values to `device`
#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

#     # shapes
#     if twoD:
#         if isinstance(res, (tuple, list)):
#             H, W = int(res[0]), int(res[1])
#         else:
#             H = W = int(res)
#         z = torch.randn(num_samples, 1, H, W, device=device, dtype=precision_dt)
#     else:
#         if isinstance(res, (tuple, list)):
#             D, H, W = int(res[0]), int(res[1]), int(res[2])
#         else:
#             D = H = W = int(res)
#         z = torch.randn(num_samples, 1, D, H, W, device=device, dtype=precision_dt)

#     images = []
#     times_to_save = {0, 15, 50, 100, 200, 300, 400, 550, 700, num_time_steps - 1}#0

#     with torch.no_grad():
#         for t_scalar in reversed(range(1, num_time_steps)):
#             # CPU indices for CPU scheduler tensors, then move gathered values to GPU with same dtype as z
#             t_idx  = torch.full((num_samples,), t_scalar, dtype=torch.long, device=scheduler.beta.device)
#             beta_t = scheduler.beta[t_idx].to(device=device, dtype=z.dtype)     # β_t
#             abar_t = scheduler.alpha[t_idx].to(device=device, dtype=z.dtype)    # \bar{α}_t
#             alpha_t = (1.0 - beta_t)                                            # α_t

#             view_shape = (num_samples,) + (1,) * (z.ndim - 1)
#             beta_t_b  = beta_t.view(view_shape)
#             abar_t_b  = abar_t.view(view_shape)
#             alpha_t_b = alpha_t.view(view_shape)

#             # model time tensor on same device as z
#             t_model = t_idx.to(device)

#             # predict noise ε̂
#             eps_hat = model(z, t_model, [var])
#             eps_hat = unpatchify(eps_hat, z,patch_size, twoD)

#             # μ_t =  ( x_t − (β_t / √(1 − \bar{α}_t)) * ε̂ ) * 1/√α_t 
#             # mu = (z - (beta_t_b / torch.sqrt(1.0 - abar_t_b)) * eps_hat) / torch.sqrt(alpha_t_b)
            
#             denom = torch.sqrt(torch.clamp(1.0 - abar_t_b, min=1e-12))
#             denom_2 = torch.sqrt(torch.clamp(alpha_t_b, min=1e-12))
#             mu = (z - (beta_t_b / denom) * eps_hat) / denom_2


#             if t_scalar > 1:
#                 e = torch.randn_like(z, device=device, dtype=z.dtype)
#                 z = mu + torch.sqrt(beta_t_b) * e
#             else:
#                 z = mu

#             if t_scalar in times_to_save:
#                 images.append(z.clone())

#         images.append(z.clone())  # final

#     # Cast to float32 right before numpy to avoid "Unsupported ScalarType BFloat16"
#     if not twoD:
#         images = [rearrange(img.float().detach().cpu(), "b c d h w -> b d h w c").numpy() for img in images]
#         plot_3D_array_center_slices_up(
#             images,
#             filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png'
#                                   % (var, epoch, (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      (res[2] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )
#     else:
#         images = [rearrange(img.float().detach().cpu(), "b c h w -> b h w c").numpy() for img in images]
#         plot_2D_array_slices(
#             images,
#             filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png'
#                                   % (var, epoch,
#                                      (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )


# ##minimalChanges!
# def sample_images(
#     model,
#     var,                       # kept for API compatibility (unused here)
#     device,
#     res,
#     precision_dt,
#     patch_size,
#     epoch=0,
#     num_samples=10,
#     twoD=False,
#     save_path='figures',
#     num_time_steps=1000,
# ):
#     """
#     Your original sampler, with minimal safety tweaks:
#       - clamp denominators before sqrt to avoid NaNs near t=0
#       - keep arithmetic on CPU (as in your code), only move a copy to GPU for the model
#       - explicit t=0 update retained
#       - append snapshot BEFORE adding next noise (your original ordering)
#     """
#     import os, torch
#     from einops import rearrange

#     os.makedirs(save_path, exist_ok=True)

#     # Leave scheduler on CPU; we index it on CPU (matches your original)
#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

#     # Convenience: safe sqrt to avoid tiny negative args from rounding
#     def safe_sqrt(x, eps=1e-12):
#         return torch.sqrt(torch.clamp(x, min=eps))

#     # Set up latents on CPU (matches your original)
#     if twoD:
#         if isinstance(res, (tuple, list)):
#             H, W = int(res[0]), int(res[1])
#         else:
#             H = W = int(res)
#         z = torch.randn(num_samples, 1, H, W)                 # CPU
#     else:
#         if isinstance(res, (tuple, list)):
#             D, H, W = int(res[0]), int(res[1]), int(res[2])
#         else:
#             D = H = W = int(res)
#         z = torch.randn(num_samples, 1, D, H, W)              # CPU

#     times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
#     images = []

#     with torch.no_grad():
#         # t = T-1 ... 1
#         for t_scalar in reversed(range(1, num_time_steps)):
#             # noise on CPU (same dtype as z)
#             if twoD:
#                 e = torch.randn(num_samples, 1, H, W)
#             else:
#                 e = torch.randn(num_samples, 1, D, H, W)

#             # build model inputs on GPU
#             t_list = [t_scalar]  # keep your list-based indexing semantics
#             t_gpu  = torch.tensor(t_list, dtype=torch.long, device=device).repeat(num_samples)

#             z_gpu = z.to(device=device, dtype=precision_dt)
#             pred_eps = model(z_gpu, t_gpu, [var])                          # predict ε (patched space)
#             pred_eps = unpatchify(pred_eps, z_gpu, patch_size, twoD)       # back to image/volume shape
#             pred_eps_cpu = pred_eps.detach().to('cpu', copy=True)          # back to CPU to match z math

#             # scheduler pieces on CPU (indexed by list)
#             beta_t  = scheduler.beta[t_list]                                # CPU
#             abar_t  = scheduler.alpha[t_list]                               # CPU (cumulative \bar{α}_t)
#             inv_sqrt_1m_beta = 1.0 / safe_sqrt(1.0 - beta_t)                # CPU
#             temp = beta_t / (safe_sqrt(1.0 - abar_t) * safe_sqrt(1.0 - beta_t))  # CPU

#             # Broadcast to spatial on CPU
#             expand_spatial = (slice(None), None, None, None) if twoD else (slice(None), None, None, None, None)
#             inv_sqrt_1m_beta_b = inv_sqrt_1m_beta[:, None, *([None] * (z.ndim - 2))]   # [B,1,...]
#             temp_b = temp[:, None, *([None] * (z.ndim - 2))]

#             # Update step (your formula/order)
#             z = inv_sqrt_1m_beta_b * z - (temp_b * pred_eps_cpu)

#             # Save snapshot BEFORE adding noise (your original behavior)
#             if t_list[0] in times:
#                 images.append(z.clone())

#             # Add noise for next step
#             sqrt_beta_b = safe_sqrt(beta_t)[:, None, *([None] * (z.ndim - 2))]
#             z = z + e * sqrt_beta_b

#         # Explicit t=0 step (no fresh noise) — your original
#         t0_list = [0]
#         t0_gpu  = torch.tensor(t0_list, dtype=torch.long, device=device).repeat(num_samples)

#         z_gpu = z.to(device=device, dtype=precision_dt)
#         pred_eps0 = model(z_gpu, t0_gpu, [var])
#         pred_eps0 = unpatchify(pred_eps0, z_gpu, patch_size, twoD)
#         pred_eps0_cpu = pred_eps0.detach().to('cpu', copy=True)

#         beta0  = scheduler.beta[t0_list]
#         abar0  = scheduler.alpha[t0_list]
#         inv_sqrt_1m_beta0 = 1.0 / safe_sqrt(1.0 - beta0)
#         temp0 = beta0 / (safe_sqrt(1.0 - abar0) * safe_sqrt(1.0 - beta0))

#         inv_sqrt_1m_beta0_b = inv_sqrt_1m_beta0[:, None, *([None] * (z.ndim - 2))]
#         temp0_b = temp0[:, None, *([None] * (z.ndim - 2))]

#         x = inv_sqrt_1m_beta0_b * z - (temp0_b * pred_eps0_cpu)
#         images.append(x)

#     # To numpy for plotting
#     if not twoD:
#         images = [rearrange(img, 'b c d h w -> b d h w c').numpy() for img in images]
#         plot_3D_array_center_slices_up(
#             images,
#             filename=os.path.join(
#                 save_path,
#                 '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png' % (
#                     var, epoch, (res[0] if isinstance(res,(tuple,list)) else res),
#                     (res[1] if isinstance(res,(tuple,list)) else res),
#                     (res[2] if isinstance(res,(tuple,list)) else res),
#                     dist.get_rank()
#                 )
#             )
#         )
#     else:
#         images = [rearrange(img, 'b c h w -> b h w c').numpy() for img in images]
#         plot_2D_array_slices(
#             images,
#             filename=os.path.join(
#                 save_path,
#                 '2D_gen_%s_%i_%i_%i_rank%i.png' % (
#                     var, epoch,
#                     (res[0] if isinstance(res,(tuple,list)) else res),
#                     (res[1] if isinstance(res,(tuple,list)) else res),
#                     dist.get_rank()
#                 )
#             )
#         )


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


# def save_intermediate_data_with_fid(model, var, device, res, precision_dt, patch_size,
#                                     epoch=0, num_samples=10, twoD=False,
#                                     save_path='figures', num_time_steps=1000,
#                                     test_volume_path=None,downscale=1):
    
#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
#     times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]

#     images = []

#     # Initial noise input
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
#             temp = (scheduler.beta[t] / ((torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
#             predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var])
#             predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#             z = (1 / torch.sqrt(1 - scheduler.beta[t])) * z - (temp * predicted_noise.cpu())
#             if t[0] in times:
#                 images.append(z)
#             z = z + (e * torch.sqrt(scheduler.beta[t]))

#         # Final step at t = 0
#         temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
#         predicted_noise = model(z.to(precision_dt).to(device), torch.tensor([0]).to(device), [var])
#         predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#         x = (1 / torch.sqrt(1 - scheduler.beta[0])) * z - (temp * predicted_noise.cpu())
#         images.append(x)

#     # Rearrange and convert to numpy
#     if not twoD:
#         images = [rearrange(img, 'b c h w d -> b h w d c').detach().numpy() for img in images]
#     else:
#         images = [rearrange(img, 'b c h w -> b h w c').detach().numpy() for img in images]
    
#     # images = np.array(images).astype('float32')
#     # images = images.astype('float32')

#     # Save visuals and arrays
#     if not twoD:
#         print("writing the center slice!")
#         plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_FID_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#     else:
#         fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10),facecolor='w')
#         for i in range(images[0].shape[0]):
#             for j in range(len(times)):
#                 ax[i,j].axis('off')
#                 ax[i,j].imshow(images[j][i,:,:,0], interpolation='none')
#         plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
#         fig.savefig(os.path.join(save_path,'2D_gen_FID_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())), format='png', dpi=300)
#         plt.close(fig)
        
#     # ----- FID Calculation -----
#     if test_volume_path is not None and dist.get_rank() == 0:
#         try:
#             idx = random.randint(0, num_samples - 1)
#             gen_vol = images[-1][idx, ..., 0]  # (H, W, D)

#             # Load GT volume
#             gt_vol = np.load(test_volume_path)
#             if gt_vol.ndim != 3:
#                 raise ValueError("Expected 3D test volume (H, W, D)")
#             gt_vol = gt_vol.astype('float32')
#             if downscale>1:
#                 gt_vol = gt_vol[::downscale,::downscale,::downscale]

#             # Extract RGB-style 2D slices
#             gen_slices = extract_rgb_slices(gen_vol)
#             gt_slices = extract_rgb_slices(gt_vol)

#             # Convert to torch tensors on device
#             gen_tensor = torch.tensor(gen_slices).float().to(device)
#             gt_tensor = torch.tensor(gt_slices).float().to(device)

#             # Compute FID
#             fid_metric = FrechetInceptionDistance().to(device)
#             fid_metric.update(gt_tensor, real=True)
#             fid_metric.update(gen_tensor, real=False)
#             fid = fid_metric.compute()

#             print(f"[Epoch {epoch}] FID (1-of-N sample vs. test volume): {fid.item():.4f}", flush=True)
#             return fid.item()

#         except Exception as e:
#             print(f"[Epoch {epoch}] FID calculation failed: {e}", flush=True)
#             return None

#     return None




# def sample_images(
#     model,
#     var,                       # kept for API compatibility (unused by math here)
#     device,
#     res,
#     precision_dt,
#     patch_size,
#     epoch=0,
#     num_samples=10,
#     twoD=False,
#     save_path="figures",
#     num_time_steps=1000,
# ):
#     os.makedirs(save_path, exist_ok=True)

#     inner = model.module if hasattr(model, "module") else model
#     inner.eval()

#     # Scheduler lives on CPU; we will index with CPU indices, then move the gathered values to `device`
#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

#     # Build latent z on device with desired precision
#     if twoD:
#         if isinstance(res, (tuple, list)):
#             H, W = int(res[0]), int(res[1])
#         else:
#             H = W = int(res)
#         z = torch.randn(num_samples, 1, H, W, device=device, dtype=precision_dt)
#     else:
#         if isinstance(res, (tuple, list)):
#             D, H, W = int(res[0]), int(res[1]), int(res[2])
#         else:
#             D = H = W = int(res)
#         z = torch.randn(num_samples, 1, D, H, W, device=device, dtype=precision_dt)

#     # Optional: initial x0 = zeros (not used in this variant; matches WORKS)
#     # x0 = torch.zeros_like(z)

#     # Which time snapshots to save (same spirit as your list)
#     times_to_save = {0, 15, 50, 100, 200, 300, 400, 550, 700, num_time_steps - 1}
#     images = []

#     with torch.no_grad():
#         # t: T-1 .. 1
#         for t_scalar in reversed(range(1, num_time_steps)):
#             # Build a CPU time index for scheduler, then move gathered β_t, ᾱ_t to device with z.dtype
#             t_idx = torch.full((num_samples,), t_scalar, dtype=torch.long, device=scheduler.beta.device)

#             beta_t  = scheduler.beta[t_idx].to(device=device, dtype=z.dtype)    # β_t
#             abar_t  = scheduler.alpha[t_idx].to(device=device, dtype=z.dtype)   # ᾱ_t (cumulative)
#             one_m_abar = torch.clamp(1.0 - abar_t, min=1e-12)                   # avoid /0
#             one_m_beta = torch.clamp(1.0 - beta_t, min=1e-12)                   # avoid /0

#             # Broadcast to spatial dims
#             view_shape = (num_samples,) + (1,) * (z.ndim - 1)
#             beta_t_b     = beta_t.view(view_shape)
#             sqrt_beta_t  = torch.sqrt(beta_t_b)
#             inv_sqrt_1mb = 1.0 / torch.sqrt(one_m_beta.view(view_shape))
#             denom = (torch.sqrt(one_m_abar).view(view_shape) * torch.sqrt(one_m_beta.view(view_shape)))  # sqrt(1-ᾱ_t)*sqrt(1-β_t)
#             denom = torch.clamp(denom, min=torch.finfo(z.dtype).eps)  # extra safety

#             # Model time tensor on device (batch-shaped)
#             t_model = t_idx.to(device)

#             # Predict ε̂ and unpatchify to z-shape
#             eps_hat = model(z, t_model, [var])
#             eps_hat = unpatchify(eps_hat, z, patch_size, twoD)

#             # WORKS update:
#             # x_{t-1} = (1/sqrt(1-β_t)) * x_t - [β_t / (sqrt(1-ᾱ_t)*sqrt(1-β_t))] * ε̂ + 1_{t>1} * sqrt(β_t) * e
#             z = inv_sqrt_1mb * z - (beta_t_b / denom) * eps_hat

#             if t_scalar > 1:
#                 e = torch.randn_like(z, device=device, dtype=z.dtype)
#                 z = z + sqrt_beta_t * e  # add noise except for the last step

#             # Save AFTER the update
#             if t_scalar in times_to_save:
#                 images.append(z.clone())

#         # At this point, z ≡ x_0; if you also want to *guarantee* t=0 frame saved:
#         if 0 not in times_to_save:
#             images.append(z.clone())
    
#     # Cast to float32 right before numpy to avoid "Unsupported ScalarType BFloat16"
#     if not twoD:
#         images = [rearrange(img.float().detach().cpu(), "b c d h w -> b d h w c").numpy() for img in images]
#         plot_3D_array_center_slices_up(
#             images,
#             filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png'
#                                   % (var, epoch, (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      (res[2] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )
#     else:
#         images = [rearrange(img.float().detach().cpu(), "b c h w -> b h w c").numpy() for img in images]
#         plot_2D_array_slices(
#             images,
#             filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png'
#                                   % (var, epoch,
#                                      (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )



# ## correct sample images (supposedly!)
# def sample_images(
#     model,
#     var,                       # kept for API compatibility (unused here)
#     device,
#     res,
#     precision_dt,
#     patch_size,
#     epoch=0,
#     num_samples=10,
#     twoD=False,
#     save_path="./figures",
#     num_time_steps=1000,
# ):

#     os.makedirs(save_path, exist_ok=True)

#     # eval mode
#     inner = model.module if hasattr(model, "module") else model
#     inner.eval()

#     # scheduler stays on CPU; we index on CPU, then move gathered values to `device`
#     scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

#     # shapes
#     if twoD:
#         if isinstance(res, (tuple, list)):
#             H, W = int(res[0]), int(res[1])
#         else:
#             H = W = int(res)
#         z = torch.randn(num_samples, 1, H, W, device=device, dtype=precision_dt)
#     else:
#         if isinstance(res, (tuple, list)):
#             D, H, W = int(res[0]), int(res[1]), int(res[2])
#         else:
#             D = H = W = int(res)
#         z = torch.randn(num_samples, 1, D, H, W, device=device, dtype=precision_dt)

#     images = []
#     # We'll save AFTER the update to x_{t-1}, so we check (t_scalar - 1)
#     times_to_save = {0, 15, 50, 100, 200, 300, 400, 550, 700, num_time_steps - 1}

#     with torch.no_grad():
#         for t_scalar in reversed(range(1, num_time_steps)):
#             # Gather β_t and \bar{α}_t on CPU, then move to GPU with same dtype as z
#             t_idx  = torch.full((num_samples,), t_scalar, dtype=torch.long, device=scheduler.beta.device)
#             beta_t = scheduler.beta[t_idx].to(device=device, dtype=z.dtype)      # β_t
#             abar_t = scheduler.alpha[t_idx].to(device=device, dtype=z.dtype)     # \bar{α}_t (cumulative)
#             alpha_t = (1.0 - beta_t)                                             # α_t

#             view_shape = (num_samples,) + (1,) * (z.ndim - 1)
#             beta_t_b  = beta_t.view(view_shape)
#             abar_t_b  = abar_t.view(view_shape)
#             alpha_t_b = alpha_t.view(view_shape)

#             # model time tensor on the same device as z
#             t_model = t_idx.to(device)

#             # predict ε̂, then unpatchify back to z-shape
#             eps_hat = model(z, t_model, [var])
#             eps_hat = unpatchify(eps_hat, z, patch_size, twoD)

#             # μ_t = 1/√α_t * ( x_t − (β_t / √(1 − \bar{α}_t)) * ε̂ )
#             denom = torch.sqrt(torch.clamp(1.0 - abar_t_b, min=1e-12))
#             mu = (z - (beta_t_b / denom) * eps_hat) / torch.sqrt(alpha_t_b)

#             # Sample x_{t-1}:
#             if t_scalar > 1:
#                 e = torch.randn_like(z, device=device, dtype=z.dtype)
#                 z_next = mu + torch.sqrt(beta_t_b) * e
#             else:
#                 # t=1 → 0 : no noise added
#                 z_next = mu

#             # Save the *post-update* state for (t_scalar - 1)
#             if (t_scalar - 1) in times_to_save:
#                 images.append(z_next.clone())

#             # move forward
#             z = z_next

#     # Cast to float32 right before numpy to avoid "Unsupported ScalarType BFloat16"
#     if not twoD:
#         images = [rearrange(img.float().detach().cpu(), "b c d h w -> b d h w c").numpy() for img in images]
#         plot_3D_array_center_slices_up(
#             images,
#             filename=os.path.join(save_path, '3D_GEN_centerSlice_%s_%i_%i_%i_%irank%i.png'
#                                   % (var, epoch, (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      (res[2] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )
#     else:
#         images = [rearrange(img.float().detach().cpu(), "b c h w -> b h w c").numpy() for img in images]
#         plot_2D_array_slices(
#             images,
#             filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_rank%i.png'
#                                   % (var, epoch,
#                                      (res[0] if isinstance(res,(tuple,list)) else res),
#                                      (res[1] if isinstance(res,(tuple,list)) else res),
#                                      dist.get_rank()))
#         )



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

 
# def plotPerformance(model,device,it_loader,
#                     num_time_steps,modality,scheduler,epoch,savefol,
#                     Ntimes=9):
    
#     print('-------- assessing performance of denoising epoch %i --------' %(epoch))
 
#     model.eval()
 
#     fol = os.path.join(savefol,'performance')
#     os.makedirs(fol,exist_ok=True)
 
#     with torch.no_grad():
 
 
#         # i forget how many things their loader outputs - change here
#         # i think its like data, vars, etc..
#         x, var, _ = next(it_loader)
 
#         Nsamples = x.shape[0]
    
#         dt = int(num_time_steps/(Ntimes-1))
#         times = [1] + \
#                 [int(i) for i in np.linspace(1+dt,num_time_steps-1-dt,Ntimes-2)] + \
#                 [num_time_steps-1]
 
#         Nsubplots = int(math.ceil(Ntimes**0.5))
#         fig,ax = plt.subplots(Nsubplots,Nsubplots,figsize=(Nsubplots*3,Nsubplots*3),facecolor='w')
#         ax=ax.ravel()
#         for i,ti in enumerate(times):
#             t = torch.Tensor([ti]).repeat(Nsamples).to(torch.int64)
#             e = torch.randn_like(x, requires_grad=False)
#             f not twoD:
#                 a = scheduler.alpha[t].view(y.shape[0],1,1,1,1).to(device)
#             else:
#                 a = scheduler.alpha[t].view(y.shape[0],1,1,1).to(device)
#             x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
#             # output = model(x, y,t,c)
#             output = model(x.to(precision_dt), torch.tensor(t).to(device), [var])
 
#             mu = e[:,0,::4,::4].detach().to('cpu').flatten().numpy().mean()
#             R2 = 1-(np.sum((output[:,0,::4,::4].detach().to('cpu').flatten().numpy()\
#                          -e[:,0,::4,::4].detach().to('cpu').flatten().numpy())**2)/\
#                    np.sum((e[:,0,::4,::4].detach().to('cpu').flatten().numpy()-mu)**2))
 
#             for j in range(e.shape[1]):
#                 ax[i].plot(e[:,j,::4,::4].detach().to('cpu').flatten(),
#                         output[:,j,::4,::4].detach().to('cpu').flatten(),'o',alpha=0.1)
#             ax[i].xaxis.set_ticks([])
#             ax[i].yaxis.set_ticks([])
#             ax[i].set_xlim([-4,4])
#             ax[i].set_ylim([-4,4])
#             ax[i].text(0.95,0.05,ti,ha='right',va='bottom',transform=ax[i].transAxes)
#             ax[i].text(0.05,0.95,'R2=%1.3f' %R2,ha='left',va='top',transform=ax[i].transAxes)
        
#         plt.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,wspace=0,hspace=0)
#         plt.savefig(os.path.join(fol,'performance_%i.jpg' %epoch),dpi=150,format='jpg')
#         plt.close(fig)
 
#     model.train()


# def sample_images(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):
 
def plotPerformance(model, device, x, variables, patch_size,
                    num_time_steps, twoD, scheduler, epoch, savefol,
                    precision_dt=torch.float32, Ntimes=9):


    model.eval()
    fol = os.path.join(savefol, 'performance')
    os.makedirs(fol, exist_ok=True)

    # small helper
    def safe_sqrt(v, eps=1e-12):
        return torch.sqrt(torch.clamp(v, min=eps))

    with torch.no_grad():
        B = x.shape[0]
        # choose times across the chain
        dt = int(num_time_steps / max(Ntimes - 1, 1))
        times = [1] + [int(i) for i in np.linspace(1 + dt, num_time_steps - 1 - dt, max(Ntimes - 2, 0))] + [num_time_steps - 1]
        times = [t for t in times if 0 <= t < num_time_steps]

        # create a clean copy each time so steps are independent
        x_clean = x.detach().clone().to(device)

        Nsub = int(math.ceil(len(times) ** 0.5))
        fig, ax = plt.subplots(Nsub, Nsub, figsize=(Nsub * 3, Nsub * 3), facecolor='w')
        ax = ax.ravel()

        for i, ti in enumerate(times):
            # noise
            e = torch.randn_like(x_clean, device=device)
            t = torch.full((B,), ti, dtype=torch.long, device=device)

            # a_bar dim shaping
            if twoD:
                a_bar = scheduler.alpha[t.cpu()].view(B, 1, 1, 1).to(device)
            else:
                a_bar = scheduler.alpha[t.cpu()].view(B, 1, 1, 1, 1).to(device)

            x_t = (safe_sqrt(a_bar) * x_clean) + (safe_sqrt(1.0 - a_bar) * e)

            # predict ε̂
            eps_hat = model(x_t.to(dtype=precision_dt), t, variables)
            eps_hat = unpatchify(eps_hat, x_t, patch_size, twoD)
            # predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var])
            # predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
            ## scatter on a subsample grid to keep it light
            if twoD:
                ref = e[:, 0, ::4, ::4].detach().float().cpu().flatten().numpy()
                out = eps_hat[:, 0, ::4, ::4].detach().float().cpu().flatten().numpy()
            else:
                ref = e[:, 0, ::4, ::4, ::4].detach().float().cpu().flatten().numpy()
                out = eps_hat[:, 0, ::4, ::4, ::4].detach().float().cpu().flatten().numpy()

            mu = ref.mean() if ref.size else 0.0
            denom = np.sum((ref - mu) ** 2) + 1e-12
            R2 = 1.0 - (np.sum((out - ref) ** 2) / denom)

            ax[i].plot(ref, out, 'o', alpha=0.1, markersize=2)
            ax[i].set_xlim([-4, 4]); ax[i].set_ylim([-4, 4])
            ax[i].set_xticks([]); ax[i].set_yticks([])
            ax[i].text(0.95, 0.05, f"t={ti}", ha='right', va='bottom', transform=ax[i].transAxes)
            ax[i].text(0.05, 0.95, f"R²={R2:.3f}", ha='left', va='top', transform=ax[i].transAxes)

        # hide any leftover axes
        for k in range(len(times), len(ax)):
            ax[k].axis('off')

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0, hspace=0)
        plt.savefig(os.path.join(fol, f'performance_{epoch}.jpg'), dpi=150)
        plt.close(fig)

    model.train()

def plotPerformanceImgs(
    model,
    device,
    x,                  # [B, C, H, W] or [B, C, D, H, W] in same dtype/device as training
    variables,          # pass exactly what you feed the model during training (no extra list)
    patch_size,
    num_time_steps,
    twoD,
    scheduler,          # DDPM_Scheduler
    epoch,
    savefol,
    precision_dt=torch.float32,
    Ntimes=9,
):

    model.eval()
    fol = os.path.join(savefol, "performance")
    os.makedirs(fol, exist_ok=True)

    def safe_sqrt(v, eps=1e-12):
        return torch.sqrt(torch.clamp(v, min=eps))

    # small helper to select a 2D slice for 3D volumes
    def to_2d(img3d):  # img3d: [B, C, D, H, W] -> take center slice in D
        B, C, D, H, W = img3d.shape
        d0 = D // 2
        return img3d[:, :, d0, :, :]  # [B, C, H, W]

    with torch.no_grad():
        x = x.to(device=device, dtype=precision_dt)
        B = x.shape[0]

        # choose times across the chain
        dt = max(int(num_time_steps / max(Ntimes - 1, 1)), 1)
        # make sure we don’t include t=0 (we’re evaluating denoising at nonzero t)
        times = [1] + [int(i) for i in np.linspace(1 + dt, num_time_steps - 1 - dt, max(Ntimes - 2, 0))] + [num_time_steps - 1]
        # de-dup & clamp
        times = sorted(set(int(t) for t in times if 0 < t < num_time_steps))

        # figure: B rows, (len(times)+1) cols (left column = clean input)
        ncols = len(times) + 1
        figsize = (max(3 * ncols, 6), max(3 * B, 3))
        fig, ax = plt.subplots(B, ncols, figsize=figsize, facecolor="w")
        if B == 1:
            ax = np.expand_dims(ax, axis=0)  # ensure 2D indexing
        if ncols == 1:
            ax = np.expand_dims(ax, axis=1)

        # show the clean x in the first column
        if twoD:
            x_show = x  # [B, C, H, W]
        else:
            # center slice in depth
            x_show = to_2d(x)  # [B, C, H, W]

        x_show_cpu = x_show[:, 0].float().detach().cpu().numpy()  # [B, H, W]
        for b in range(B):
            if b<2:
                print(f"max and min values are: {np.min(x_show_cpu)}, {np.max(x_show_cpu)}!")
            ax[b, 0].imshow(x_show_cpu[b], interpolation="none", cmap="gray",vmin=0,vmax=1)
            if b == 0:
                ax[b, 0].set_title("clean x")
            ax[b, 0].set_xticks([]); ax[b, 0].set_yticks([])

        # compute errors at selected timesteps
        for j, ti in enumerate(times, start=1):
            # fresh copy of x each time -> evaluate denoising at exactly t=ti
            x_clean = x.clone()

            # noise
            e = torch.randn_like(x_clean, device=device)

            # a_bar(ti) on GPU (index on CPU)
            if twoD:
                a_bar = scheduler.alpha[torch.tensor(ti)].to(device=device, dtype=precision_dt)
                a_bar = a_bar.view(1, 1, 1, 1).expand(B, 1, x.shape[-2], x.shape[-1])
            else:
                a_bar = scheduler.alpha[torch.tensor(ti)].to(device=device, dtype=precision_dt)
                a_bar = a_bar.view(1, 1, 1, 1, 1).expand(B, 1, x.shape[-3], x.shape[-2], x.shape[-1])

            x_t = safe_sqrt(a_bar) * x_clean + safe_sqrt(1.0 - a_bar) * e

            # predict ε̂
            t_batch = torch.full((B,), ti, dtype=torch.long, device=device)
            eps_hat = model(x_t, t_batch, variables)      # same signature as training
            eps_hat = unpatchify(eps_hat, x_t, patch_size, twoD)  # returns same shape as x_t

            # get a 2D view for plotting
            if twoD:
                ref = e[:, 0]           # [B, H, W]
                out = eps_hat[:, 0]     # [B, H, W]
            else:
                ref = to_2d(e)[:, 0]           # [B, H, W]
                out = to_2d(eps_hat)[:, 0]     # [B, H, W]

            # robust scaling for error map
            err = (ref - out).float().detach().cpu().numpy()  # [B, H, W]
            abs_err = np.abs(err)
            # vmax = np.percentile(abs_err, 99.5) if np.isfinite(abs_err).all() else 1.0
            vmax = 3#1#max(vmax, 1e-6)
            vmin = -3#0

            for b in range(B):
                ax[b, j].imshow(abs_err[b], interpolation="none", cmap="magma", vmin=vmin, vmax=vmax)
                if b == 0:
                    ax[b, j].set_title(f"t={ti}")
                ax[b, j].set_xticks([]); ax[b, j].set_yticks([])

        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(fol, f"performanceImgs_{epoch}.png"), dpi=100)
        plt.close(fig)

    model.train()

# def plotPerformanceImgs(model, device, x, variables, patch_size,
#                     num_time_steps, twoD, scheduler, epoch, savefol,
#                     precision_dt=torch.float32, Ntimes=9):
 
 
#     model.eval()
#     fol = os.path.join(savefol, 'performance')
#     os.makedirs(fol, exist_ok=True)
 
#     # small helper
#     def safe_sqrt(v, eps=1e-12):
#         return torch.sqrt(torch.clamp(v, min=eps))
 
#     with torch.no_grad():
#         B = x.shape[0]
#         # choose times across the chain
#         dt = int(num_time_steps / max(Ntimes - 1, 1))
#         times = [1] + [int(i) for i in np.linspace(1 + dt, num_time_steps - 1 - dt, max(Ntimes - 2, 0))]
#         times = [t for t in times if 0 <= t < num_time_steps]
 
#         # create a clean copy each time so steps are independent
#         x_clean = x.detach().clone().to(device)
 
#         fig, ax = plt.subplots(B, Ntimes, figsize=(B, Ntimes), facecolor='w')
 
#         for i in range(x.shape[0]):
#             ax[i,0].imshow(x[i,0,:,:].to('cpu').detach(),interpolation='none')
 
#         for i, ti in enumerate(times):
#             # noise
#             e = torch.randn_like(x_clean, device=device)
#             t = torch.full((B,), ti, dtype=torch.long, device=device)
 
#             # a_bar dim shaping
#             if twoD:
#                 a_bar = scheduler.alpha[t.cpu()].view(B, 1, 1, 1).to(device)
#             else:
#                 a_bar = scheduler.alpha[t.cpu()].view(B, 1, 1, 1, 1).to(device)
 
#             x_t = (safe_sqrt(a_bar) * x_clean) + (safe_sqrt(1.0 - a_bar) * e)
 
#             # predict ε̂
#             eps_hat = model(x_t.to(dtype=precision_dt), t, [variables])
#             eps_hat = unpatchify(eps_hat, x_t, patch_size, twoD)
#             # predicted_noise = model(z.to(precision_dt).to(device), torch.tensor(t).to(device), [var])
#             # predicted_noise = unpatchify(predicted_noise, z.to(precision_dt).to(device), patch_size, twoD)
#             ## scatter on a subsample grid to keep it light
#             if twoD:
#                 ref = e.detach().float().cpu().numpy()
#                 out = eps_hat.detach().float().cpu().numpy()
#             else:
#                 ref = e.detach().float().cpu().numpy()
#                 out = eps_hat.detach().float().cpu().numpy()
 
#             for k in range(x.shape[0]):
#                 ax[k,i+1].imshow(abs((ref-out)[k,0,:,:]),interpolation='none',vmin=0,vmax=1)
#                 ax[0,i+1].set_title(ti)
 
#         for i in range(ax.shape[0]):
#             for j in range(ax.shape[1]):
#                 ax[i,j].xaxis.set_ticks([])
#                 ax[i,j].yaxis.set_ticks([])
        
#         # plt.subplots_adjust(left=0.01,right=0.99,top=0.975,bottom=0.01,wspace=0,hspace=0)
#         # plt.savefig(os.path.join(fol,'performanceImgs_%i.png' %epoch),dpi=300,format='png')
#         plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0, hspace=0)
#         plt.savefig(os.path.join(fol, f'performance_{epoch}.png'), dpi=150)
#         plt.close(fig)
 
#     model.train()

# for f in files:
#     x=np.load(os.path.join(inp_dir,f))
#     print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
#     x[x>0.279]=1
#     x[x<0.1]=0
#     x[(x>0) &(x<1)]=0.5
#     np.save(os.path.join(out_dir,f),x)
#     print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")


##"/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth_3level_0to1/"
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.26489898562431335
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.91424560546875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.20930016040802002
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.6151123046875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.26887625455856323
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.926971435546875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.12737822532653809
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.16168212890625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28751885890960693
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.99627685546875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2734057307243347
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.937408447265625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2338986098766327
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.74090576171875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.23043465614318848
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.724517822265625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2867593765258789
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.994873046875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.18725918233394623
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.498260498046875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28837552666664124
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.99847412109375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27667760848999023
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.958587646484375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2696969509124756
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.94537353515625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2782647907733917
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.977142333984375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.24371814727783203
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.801513671875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2652035653591156
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.91790771484375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2742580771446228
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.956634521484375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2215355485677719
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.693206787109375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.27572041749954224
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.95745849609375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2709299325942993
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.9466552734375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2688066065311432
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.929351806640625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2877730429172516
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.996856689453125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2637060582637787
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.8876953125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.23584577441215515
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.75872802734375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.276294469833374
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.97430419921875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2076147049665451
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.607208251953125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2888902425765991
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.999267578125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.24342721700668335
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.804473876953125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2652119994163513
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.902252197265625
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2869330048561096
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.996673583984375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.18383803963661194
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.477386474609375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2809602618217468
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.97393798828125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27943509817123413
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.97406005859375
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.19880732893943787
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.559600830078125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2880808711051941
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.99676513671875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.24054840207099915
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.78729248046875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2715320587158203
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.955841064453125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2699772119522095
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.945770263671875
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.20460376143455505
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.5860595703125
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.27778300642967224
# min,max,mean for ConcA_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.961639404296875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.23339678347110748
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.75238037109375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2746146023273468
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.950927734375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.27927616238594055
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.972869873046875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2838507294654846
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.981414794921875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2714443802833557
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.93988037109375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.20901702344417572
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.61761474609375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.25170931220054626
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.838897705078125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2737923264503479
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.9447021484375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2759864628314972
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9622802734375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.214835062623024
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.652862548828125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.25290974974632263
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.846099853515625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2419416457414627
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.7999267578125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2326856553554535
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.744384765625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,0.28999999165534973,0.281645804643631
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.98187255859375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.23396876454353333
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.757659912109375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27658355236053467
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.95147705078125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.281592458486557
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.97869873046875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.27521559596061707
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.9732666015625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.278595507144928
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.97186279296875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.24783605337142944
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.8294677734375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.22988082468509674
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.730377197265625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.27967292070388794
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.97625732421875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2725115120410919
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.94317626953125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2877500653266907
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.998931884765625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2521504759788513
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.843048095703125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2543463408946991
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.853851318359375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2662901282310486
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.92236328125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.23538276553153992
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.755035400390625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27686306834220886
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.960296630859375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28248628973960876
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.988433837890625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2793805003166199
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9798583984375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2678773105144501
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.917724609375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2751903533935547
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.95721435546875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.251421183347702
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.844329833984375
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.21332567930221558
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.6416015625
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2782740294933319
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.974151611328125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2669435739517212
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.927001953125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2749031186103821
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.96173095703125
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.270068883895874
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.92547607421875
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2786855101585388
# min,max,mean for ConcA_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.980987548828125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.27240341901779175
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.958648681640625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27306699752807617
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.962432861328125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2822200655937195
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.973663330078125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.279025673866272
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.97479248046875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27618420124053955
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9683837890625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28233668208122253
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.985321044921875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2809077501296997
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.969329833984375
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2764902710914612
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.970794677734375
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28607508540153503
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.988983154296875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2781723737716675
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.976287841796875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2745088040828705
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.95556640625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2813907265663147
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.978668212890625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28939929604530334
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.999267578125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28363659977912903
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.986083984375
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2659567594528198
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9259033203125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28650256991386414
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9937744140625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.28100037574768066
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.98046875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2872527241706848
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99383544921875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2741459608078003
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.964141845703125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27884435653686523
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.9814453125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27218902111053467
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.94537353515625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.28002288937568665
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.981719970703125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2766600251197815
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.974853515625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2840973138809204
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.993896484375
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.288544237613678
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.99932861328125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2804434895515442
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.985107421875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.28670287132263184
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.9969482421875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.28920093178749084
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.9986572265625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27490395307540894
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.964691162109375
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2798476219177246
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.975494384765625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.27949807047843933
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9747314453125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2828653156757355
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.989654541015625
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2792414128780365
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.974151611328125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2886453866958618
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.995819091796875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2714630365371704
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.960662841796875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2801077961921692
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.975341796875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27547121047973633
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.9578857421875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2889935076236725
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.9981689453125
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28344160318374634
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.98699951171875
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.27000001072883606,0.28999999165534973,0.2898809611797333
# min,max,mean for ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2864331305027008
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.993011474609375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2791385054588318
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.969512939453125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.27000001072883606,0.28999999165534973,0.2897680699825287
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28455859422683716
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.996826171875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2797469198703766
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.974273681640625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.27000001072883606,0.28999999165534973,0.2880706787109375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2836655378341675
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.992431640625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2788849174976349
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.978363037109375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2843421697616577
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.986083984375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27177953720092773
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.9442138671875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2879464030265808
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.9959716796875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.27000001072883606,0.28999999165534973,0.28995785117149353
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2895907461643219
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.99908447265625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2806216776371002
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.987823486328125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.26727548241615295
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.932037353515625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28015321493148804
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.978973388671875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2762245535850525
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.97149658203125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.27576565742492676
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.965179443359375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2851572334766388
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.992431640625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.28302091360092163
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.988983154296875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27741217613220215
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.98114013671875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2864046096801758
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.991363525390625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.26823049783706665
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.931243896484375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.26954036951065063
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.939849853515625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2767086923122406
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.97711181640625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28420141339302063
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.985107421875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2896469831466675
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.9991455078125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2836189568042755
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.99468994140625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2851492166519165
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99969482421875
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2871606945991516
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.999114990234375
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.27000001072883606,0.28999999165534973,0.2899554371833801
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27684298157691956
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.957916259765625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27795615792274475
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.95751953125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.27114713191986084
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.95025634765625
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2775201201438904
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.973419189453125
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.27000001072883606,0.28999999165534973,0.2889569103717804
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.27000001072883606,0.28999999165534973,0.28976622223854065
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.27000001072883606,0.28999999165534973,0.289658784866333
# min,max,mean for ConcA_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28482696413993835
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99603271484375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2779983878135681
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.982666015625
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2764674723148346
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.97210693359375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2852783501148224
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.993621826171875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,0.28999999165534973,0.28206247091293335
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.99267578125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27765482664108276
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.979339599609375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2805357575416565
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.977294921875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.28079259395599365
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.986328125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2876622974872589
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.99444580078125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.272344172000885
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.955535888671875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28987598419189453
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27706024050712585
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.98016357421875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2829012870788574
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.98687744140625
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.27809035778045654
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.977264404296875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2772044539451599
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.977294921875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27814656496047974
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.977935791015625
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.27000001072883606,0.28999999165534973,0.28995180130004883
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2749726176261902
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.970489501953125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.28167399764060974
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.984832763671875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.28842246532440186
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.995635986328125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27654629945755005
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.97528076171875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2782975435256958
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.982391357421875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2789149582386017
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.976043701171875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2808748185634613
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.99163818359375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28131580352783203
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.987396240234375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2842535674571991
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.988739013671875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.28405997157096863
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.99139404296875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2738812565803528
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.9603271484375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2851679027080536
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99090576171875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28322964906692505
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.983489990234375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2897653877735138
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.999755859375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2763110399246216
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9749755859375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27175742387771606
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.943756103515625
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27236366271972656
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.9588623046875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28622353076934814
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.998260498046875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.28887131810188293
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2847818434238434
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.987548828125
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27909916639328003
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.970428466796875
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.27759864926338196
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.984222412109375
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2855372130870819
# min,max,mean for ConcA_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.9927978515625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2829959988594055
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.992156982421875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2888329327106476
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.99749755859375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28586333990097046
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.98870849609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.258165568113327
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.86907958984375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.24883098900318146
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.82281494140625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2818446159362793
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.986083984375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28246310353279114
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.982330322265625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.25651776790618896
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.88079833984375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2683640122413635
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.921630859375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27569806575775146
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.969573974609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,0.28999999165534973,0.2577335238456726
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.88525390625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2694171369075775
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.934417724609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27996864914894104
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.983001708984375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28302842378616333
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.984619140625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.26870471239089966
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.926025390625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.25468480587005615
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.849609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2750566601753235
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.946502685546875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2838602066040039
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.992095947265625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2497132122516632
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.842315673828125
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.24450993537902832
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.80780029296875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2527412474155426
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.849090576171875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.282453715801239
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.989410400390625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2758880853652954
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.96588134765625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2216140627861023
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.68963623046875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2803829312324524
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.983154296875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.24885597825050354
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.823699951171875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2752608060836792
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.95745849609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.268827885389328
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.916748046875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27826786041259766
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.963714599609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28525763750076294
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99151611328125
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.26363736391067505
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.894989013671875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2851017117500305
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.99371337890625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27017325162887573
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.92974853515625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27445292472839355
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.94488525390625
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2650327682495117
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.91009521484375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.23288775980472565
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.74261474609375
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.26294058561325073
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.88641357421875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28592854738235474
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.99285888671875
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.25071293115615845
# min,max,mean for ConcA_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.850799560546875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28674712777137756
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99786376953125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.27000001072883606,0.28999999165534973,0.2898889183998108
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2845068573951721
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.995452880859375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28306135535240173
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.98394775390625
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2779087722301483
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.960479736328125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2813516855239868
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.976287841796875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2846163809299469
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.996307373046875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27319538593292236
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.94158935546875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2815967798233032
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.992462158203125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.27000001072883606,0.28999999165534973,0.2897875905036926
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.28342241048812866
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.98126220703125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28452402353286743
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.994903564453125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.27869680523872375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.981292724609375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2893013060092926
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.999847412109375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2863459885120392
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.99591064453125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2830607295036316
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99737548828125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.28372329473495483
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.994964599609375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.28775081038475037
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.997833251953125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2837226390838623
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.9952392578125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.281508207321167
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.986907958984375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.28630322217941284
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9970703125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2774835228919983
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9798583984375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2877109944820404
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.99859619140625
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2868499159812927
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.99688720703125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27955251932144165
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.983123779296875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2822614312171936
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.977203369140625
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2819407284259796
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99005126953125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.26453593373298645
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.91009521484375
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.27000001072883606,0.28999999165534973,0.28982236981391907
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.28670734167099
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.996246337890625
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,0.28999999165534973,0.27509334683418274
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.967193603515625
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2887248396873474
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28171414136886597
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.988983154296875
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.279822438955307
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.98175048828125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.27000001072883606,0.28999999165534973,0.28997373580932617
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2817933261394501
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.9854736328125
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.27000001072883606,0.28999999165534973,0.2891296446323395
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.28039175271987915
# min,max,mean for ConcA_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.98583984375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2889070510864258
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99957275390625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.277091383934021
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.975006103515625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2774887681007385
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.9613037109375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2726564407348633
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.941925048828125
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.282697856426239
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.982421875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28994011878967285
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.99993896484375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2860148847103119
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.99560546875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2719496488571167
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.95257568359375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2813915014266968
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.984344482421875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.27000001072883606,0.28999999165534973,0.2899859547615051
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.27000001072883606,0.28999999165534973,0.2899487316608429
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.283550888299942
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.98583984375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2728259265422821
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.94384765625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2664121389389038
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.918304443359375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2808053493499756
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.98870849609375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2786518931388855
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.98187255859375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.289616197347641
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99951171875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2865314781665802
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99951171875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.28366661071777344
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.994415283203125
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2872481048107147
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.99774169921875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2655699849128723
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.915985107421875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2807258665561676
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.976654052734375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.27000001072883606,0.28999999165534973,0.28691160678863525
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2859139144420624
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.993194580078125
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28558939695358276
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.997650146484375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28349021077156067
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.988922119140625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.28216350078582764
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.98577880859375
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27419716119766235
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.95703125
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.27000001072883606,0.28999999165534973,0.28945738077163696
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27957507967948914
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.9901123046875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.28183287382125854
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.98046875
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2781028747558594
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.959259033203125
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2740539312362671
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.953277587890625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2838367223739624
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.99517822265625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.27000001072883606,0.28999999165534973,0.2898583710193634
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28286826610565186
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.9801025390625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2604433596134186
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.892822265625
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.27000001072883606,0.28999999165534973,0.28798097372055054
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.28044357895851135
# min,max,mean for ConcA_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.972686767578125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2784169018268585
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.983978271484375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27726101875305176
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.962127685546875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.27809393405914307
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.95794677734375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28385746479034424
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.99029541015625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2789481282234192
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.980560302734375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27062803506851196
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.939239501953125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27999362349510193
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.984161376953125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2770845890045166
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.96484375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28057539463043213
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.986114501953125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2763676941394806
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.974853515625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2824004888534546
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.986236572265625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2819456458091736
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.976959228515625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27250877022743225
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.9501953125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2817528247833252
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.9888916015625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.282259464263916
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.97308349609375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28355512022972107
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.99127197265625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2691701650619507
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.935882568359375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.27430668473243713
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.965545654296875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.27835339307785034
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.9696044921875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2843722403049469
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.99298095703125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2758494019508362
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.957366943359375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2846700847148895
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.99542236328125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2806604504585266
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.98785400390625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2822713255882263
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.98797607421875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.27992987632751465
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.98541259765625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.279498815536499
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.990447998046875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2751502990722656
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.957733154296875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27488964796066284
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.946746826171875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.282588928937912
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99066162109375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.26223137974739075
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.89984130859375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2807457447052002
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9840087890625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.28090018033981323
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.98834228515625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2850819528102875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.993743896484375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27745288610458374
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.9754638671875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27930906414985657
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.984405517578125
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2769829034805298
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.969635009765625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2847539782524109
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.99078369140625
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.288494735956192
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.9974365234375
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28061917424201965
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.990203857421875
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2820413112640381
# min,max,mean for ConcA_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.98260498046875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.27000001072883606,0.28999999165534973,0.28999876976013184
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.27000001072883606,0.28999999165534973,0.2883685231208801
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.27000001072883606,0.28999999165534973,0.28979307413101196
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.289067804813385
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.99920654296875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28954264521598816
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.99981689453125
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2894067168235779
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.9993896484375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.27000001072883606,0.28999999165534973,0.28983762860298157
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2863043546676636
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.998504638671875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.28457850217819214
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.9935302734375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28981515765190125
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.999786376953125
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27483272552490234
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.95904541015625
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2818063497543335
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.983551025390625
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.27000001072883606,0.28999999165534973,0.28976988792419434
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28185367584228516
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.985076904296875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.27000001072883606,0.28999999165534973,0.2896130084991455
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.28276726603507996
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99017333984375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2815176248550415
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.987762451171875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2825731635093689
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.985565185546875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27493733167648315
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.964508056640625
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.27000001072883606,0.28999999165534973,0.2888702154159546
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.27000001072883606,0.28999999165534973,0.2871472239494324
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2886592745780945
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.997802734375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2851904034614563
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.987457275390625
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28585249185562134
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99310302734375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.27000001072883606,0.28999999165534973,0.287299782037735
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2898460626602173
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28305694460868835
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.97894287109375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2787759304046631
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.977386474609375
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.27000001072883606,0.28999999165534973,0.2890533208847046
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.28724414110183716
# min,max,mean for ConcA_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28079554438591003
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.9874267578125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27762141823768616
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.972198486328125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28105440735816956
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.976837158203125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2814689874649048
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.994384765625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2802443504333496
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.979156494140625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27802062034606934
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.988677978515625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2800861597061157
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.9854736328125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.27855193614959717
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.98260498046875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27913060784339905
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.980224609375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2781088054180145
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.973114013671875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2783755660057068
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.973785400390625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2849632501602173
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.999267578125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2799789011478424
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.982269287109375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2767753601074219
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.978790283203125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27787497639656067
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9788818359375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28001725673675537
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.98858642578125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2809517979621887
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.993865966796875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2778354585170746
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.98388671875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2767578959465027
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.96588134765625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27437907457351685
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.96380615234375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2850207984447479
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.99395751953125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2792899012565613
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9832763671875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2778090536594391
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.974884033203125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.28691160678863525
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.9957275390625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.27709442377090454
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.97406005859375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.27733299136161804
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.966064453125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,0.28999999165534973,0.27733325958251953
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.9808349609375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27972927689552307
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.983489990234375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,0.28999999165534973,0.2761313021183014
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.976715087890625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2784199118614197
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.978759765625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2728596329689026
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.959228515625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27759265899658203
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.982177734375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27546146512031555
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.968597412109375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27483507990837097
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.962646484375
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2814551591873169
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.993682861328125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.28257763385772705
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.99462890625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2766586244106293
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.97467041015625
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2810348570346832
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.99237060546875
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28108078241348267
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.9864501953125
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.27578210830688477
# min,max,mean for ConcA_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.96728515625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.27679091691970825
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.966278076171875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2720228135585785
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.9464111328125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.27885568141937256
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.986968994140625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27916476130485535
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.970123291015625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27880099415779114
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.979888916015625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2693805396556854
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.92193603515625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2804855704307556
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.975616455078125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.276366651058197
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.96343994140625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28653356432914734
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9954833984375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2755943536758423
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.9638671875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.27893829345703125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.97833251953125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2753137946128845
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.962554931640625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27958011627197266
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.989349365234375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28865230083465576
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.99993896484375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27491724491119385
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.962127685546875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27902135252952576
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.985107421875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2838676869869232
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.9876708984375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2768395245075226
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.96881103515625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.28073567152023315
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.983428955078125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27562859654426575
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.96575927734375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.275592565536499
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.963592529296875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.28361302614212036
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9942626953125
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2697393298149109
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.93951416015625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.27671942114830017
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.95916748046875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.27767178416252136
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.979217529296875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28035369515419006
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.98199462890625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2751600742340088
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.96966552734375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2758007049560547
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.972747802734375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27828070521354675
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.97869873046875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.277293860912323
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.962005615234375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2769607901573181
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.972900390625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27729102969169617
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.975677490234375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2796269655227661
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.976409912109375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27437978982925415
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.962371826171875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2774277329444885
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.9759521484375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2729913592338562
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.951507568359375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28364062309265137
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.9852294921875
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2790956497192383
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.977935791015625
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2709818482398987
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.941497802734375
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.28021004796028137
# min,max,mean for ConcA_round_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.991241455078125
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2853599190711975
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.99554443359375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28392499685287476
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.98760986328125
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.27000001072883606,0.28999999165534973,0.28933775424957275
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28412503004074097
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.990203857421875
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.27000001072883606,0.28999999165534973,0.289895623922348
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28710120916366577
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.997314453125
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2892463803291321
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.99957275390625
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.27000001072883606,0.28999999165534973,0.28852784633636475
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.284480482339859
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.987396240234375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.286757230758667
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.996673583984375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,0.28999999165534973,0.2678273320198059
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.92376708984375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2834372818470001
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.983001708984375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.27000001072883606,0.28999999165534973,0.28996825218200684
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.27000001072883606,0.28999999165534973,0.2897271513938904
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.28646954894065857
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.992218017578125
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.27000001072883606,0.28999999165534973,0.2899231016635895
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2875972390174866
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99774169921875
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.27000001072883606,0.28999999165534973,0.2898571491241455
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,0.28999999165534973,0.28425857424736023
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.984466552734375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28929603099823
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.99853515625
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.28849121928215027
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.9945068359375
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.27000001072883606,0.28999999165534973,0.28975099325180054
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.27000001072883606,0.28999999165534973,0.2881537079811096
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.28963765501976013
# min,max,mean for ConcA_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.999908447265625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28747573494911194
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.999237060546875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27217286825180054
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.9554443359375
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2763299345970154
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.977081298828125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.27000001072883606,0.28999999165534973,0.2893475294113159
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27060383558273315
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.944366455078125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.27000001072883606,0.28999999165534973,0.28998899459838867
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28789064288139343
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.997283935546875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2779250144958496
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.974273681640625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27349400520324707
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9576416015625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,0.28999999165534973,0.2758435904979706
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.971954345703125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28014132380485535
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.979888916015625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.27000001072883606,0.28999999165534973,0.28999143838882446
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2667505443096161
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.92437744140625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2798105776309967
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.984222412109375
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.27000001072883606,0.28999999165534973,0.2898791432380676
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28182554244995117
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9884033203125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2846587896347046
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.997833251953125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2664914131164551
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.922515869140625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.27000001072883606,0.28999999165534973,0.28980588912963867
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2757243514060974
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.9718017578125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.27000001072883606,0.28999999165534973,0.2871600389480591
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2876720428466797
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.998382568359375
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.27614107728004456
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.972564697265625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2886003851890564
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.99871826171875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.27051907777786255
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.94293212890625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28887617588043213
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.99920654296875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.275032639503479
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.963958740234375
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.286196768283844
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.993438720703125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28533539175987244
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99420166015625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.26457473635673523
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.912567138671875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27850309014320374
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.969451904296875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27496519684791565
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.967041015625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2754303812980652
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.971099853515625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27462995052337646
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.96136474609375
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.27644190192222595
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.9727783203125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28873369097709656
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.99957275390625
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2829195261001587
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.9959716796875
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2886964976787567
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.999298095703125
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2738250195980072
# min,max,mean for ConcA_round_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.95904541015625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2657009959220886
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.91851806640625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2660427391529083
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.9215087890625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.276885449886322
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.97137451171875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27722299098968506
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.964996337890625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27848854660987854
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.968017578125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27563971281051636
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.96282958984375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27700275182724
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.97271728515625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.27012044191360474
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.940093994140625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2836951017379761
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.98297119140625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.26642802357673645
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.91253662109375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28908538818359375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.99908447265625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2764270305633545
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.974090576171875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2844313383102417
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.989532470703125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28724491596221924
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.99517822265625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28007060289382935
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.96563720703125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2761828601360321
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9658203125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2820022702217102
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.98504638671875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.28436124324798584
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.98565673828125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2648349702358246
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.910980224609375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27526551485061646
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.958831787109375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.28503289818763733
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.99200439453125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2858361005783081
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.986175537109375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2749168872833252
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9644775390625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2852635681629181
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.992279052734375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,0.28999999165534973,0.2643013000488281
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.92364501953125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2726062536239624
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.95733642578125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2749936580657959
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.965576171875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.28828996419906616
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.999237060546875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2886016368865967
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.998443603515625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2871028482913971
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99713134765625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.27863818407058716
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.98095703125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27267181873321533
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.950592041015625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27061522006988525
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.942901611328125
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28400760889053345
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.987518310546875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28463008999824524
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.996002197265625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.24906012415885925
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.82977294921875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2666964828968048
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.924957275390625
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.28593116998672485
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.993988037109375
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28039297461509705
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.979461669921875
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2787413001060486
# min,max,mean for ConcA_round_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.973907470703125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2838088870048523
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99176025390625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2807289958000183
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.9862060546875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28158730268478394
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.977508544921875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28904032707214355
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.99920654296875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.26773715019226074
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.928619384765625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2829042375087738
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.98248291015625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27894559502601624
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.98077392578125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2864248752593994
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.99481201171875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2834060788154602
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.991485595703125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27863776683807373
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.975982666015625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.27000001072883606,0.28999999165534973,0.28743040561676025
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.26919057965278625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.931243896484375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2769445776939392
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.960662841796875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2871512770652771
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.996429443359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27958548069000244
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.98095703125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28515398502349854
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.99713134765625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2854457199573517
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99163818359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2810448408126831
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.98529052734375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2841871380805969
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.986083984375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.27000001072883606,0.28999999165534973,0.2892962694168091
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.27000001072883606,0.28999999165534973,0.289017915725708
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.27000001072883606,0.28999999165534973,0.2893310785293579
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2808574438095093
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.984344482421875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2841063141822815
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.9931640625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28270357847213745
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.98382568359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2668185532093048
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.92584228515625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.281636506319046
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.992523193359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.28128278255462646
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.989776611328125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28233468532562256
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.979034423828125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28205615282058716
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.98828125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.27000001072883606,0.28999999165534973,0.2858605980873108
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.28192028403282166
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.98486328125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.279753714799881
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.98016357421875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28090476989746094
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.98681640625
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28531569242477417
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.9964599609375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2838295102119446
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.988250732421875
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28012099862098694
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.972991943359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.28857940435409546
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.99786376953125
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28152787685394287
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.982757568359375
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.27000001072883606,0.28999999165534973,0.2899548411369324
# min,max,mean for ConcA_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28635871410369873
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.993072509765625
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27858710289001465
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.96466064453125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2869056761264801
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.99591064453125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28744202852249146
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.990570068359375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2832275629043579
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.991790771484375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28820791840553284
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.99688720703125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28776586055755615
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.992706298828125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.28992459177970886
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.99969482421875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28880637884140015
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9996337890625
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.28392457962036133
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.990234375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2896895706653595
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.999969482421875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28226667642593384
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.980010986328125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2874516248703003
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.997711181640625
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2887941300868988
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.99652099609375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2758829891681671
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.95404052734375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.27000001072883606,0.28999999165534973,0.289933443069458
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.27603763341903687
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.958221435546875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2863745093345642
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.9932861328125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2862577438354492
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.99615478515625
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2790547013282776
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.972381591796875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.27000001072883606,0.28999999165534973,0.2898125946521759
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2879124879837036
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.992889404296875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,0.28999999165534973,0.2891503572463989
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.9970703125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2853931188583374
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99395751953125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2763966917991638
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.954132080078125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.27000001072883606,0.28999999165534973,0.28999876976013184
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28860098123550415
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.995330810546875
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.27000001072883606,0.28999999165534973,0.2896704077720642
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2860957980155945
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.995635986328125
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2870350480079651
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.993743896484375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2889975905418396
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.99798583984375
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2892909348011017
# min,max,mean for ConcA_semi_angular_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.99835205078125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.26585763692855835
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.917388916015625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.26889753341674805
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.944000244140625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2716837525367737
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.95587158203125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27672576904296875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.974334716796875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27539166808128357
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.97174072265625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27059078216552734
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.944122314453125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2627488374710083
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.90582275390625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2827487587928772
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.985107421875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2763609290122986
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.967803955078125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27422186732292175
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.96832275390625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2776123285293579
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.964324951171875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27819228172302246
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.982147216796875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27064430713653564
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.94970703125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.27604806423187256
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.959503173828125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2804146409034729
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.979766845703125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2772131860256195
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.97869873046875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2759741246700287
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.97314453125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2744044065475464
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.966552734375
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2796759605407715
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.99334716796875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2777138352394104
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.9825439453125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.28114327788352966
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.983917236328125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.27772557735443115
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.960357666015625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2789572477340698
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.983551025390625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.27760687470436096
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.98602294921875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2721940875053406
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.960205078125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2776681184768677
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.980438232421875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27469179034233093
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.966766357421875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2774249315261841
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.97314453125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2782340347766876
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.985443115234375
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27304136753082275
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.9591064453125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.28332728147506714
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9935302734375
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2773022949695587
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.97686767578125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27457696199417114
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.96881103515625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27629896998405457
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.974212646484375
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.26835817098617554
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.9267578125
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.28047671914100647
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.978973388671875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27125656604766846
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.942779541015625
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27804291248321533
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.97283935546875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2688826024532318
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.938201904296875
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.26855871081352234
# min,max,mean for ConcA_semi_angular_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.927734375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2822068929672241
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.979736328125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.28098756074905396
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.989013671875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.27000001072883606,0.28999999165534973,0.2898632884025574
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27131158113479614
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.94073486328125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28621360659599304
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.995361328125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28038710355758667
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.983978271484375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2751502990722656
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.963226318359375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.28824087977409363
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.99810791015625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2825138568878174
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.98638916015625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.28079044818878174
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.985687255859375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2808750569820404
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.97723388671875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.28213152289390564
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.98675537109375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27708807587623596
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.97137451171875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2825009226799011
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.98870849609375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28946956992149353
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.999664306640625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2836619019508362
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.99127197265625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2852931618690491
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99285888671875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2713968753814697
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.94427490234375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2830508053302765
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.987152099609375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.28262901306152344
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.988037109375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2864897847175598
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.99566650390625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2876800298690796
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.995391845703125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.28201594948768616
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.991119384765625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2855423390865326
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.9954833984375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28108832240104675
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.98480224609375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.27000001072883606,0.28999999165534973,0.2892358601093292
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2615860402584076
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.899658203125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.28395286202430725
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.992584228515625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2855744957923889
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99713134765625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2810894846916199
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.985076904296875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.28037816286087036
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.976226806640625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.27668654918670654
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.964447021484375
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.27000001072883606,0.28999999165534973,0.2899768054485321
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28467756509780884
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.996490478515625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2879186272621155
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.99871826171875
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2809531092643738
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.985992431640625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28979673981666565
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.999664306640625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2833313047885895
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.99267578125
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.279001921415329
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.972076416015625
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2766649127006531
# min,max,mean for ConcA_semi_angular_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.977294921875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28611722588539124
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99298095703125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2821238338947296
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.98089599609375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28227293491363525
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.986572265625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27722784876823425
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.96649169921875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2829248011112213
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9835205078125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28009819984436035
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.97735595703125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2821720838546753
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.976348876953125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2667364180088043
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.915435791015625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27434009313583374
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9473876953125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.28405386209487915
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.98773193359375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.284676730632782
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.990478515625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2801133096218109
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.979949951171875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28242504596710205
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.9891357421875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.28542855381965637
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.989410400390625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2797147035598755
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.987213134765625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.28488779067993164
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9921875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.28270769119262695
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.98248291015625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.27232858538627625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.951568603515625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2699984610080719
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.92401123046875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2764754891395569
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.970672607421875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.28006458282470703
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.968475341796875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2750737965106964
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.96148681640625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.26884156465530396
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.925872802734375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2727384567260742
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.94793701171875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2777721881866455
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.963043212890625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2825953960418701
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.98370361328125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2699476182460785
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.92626953125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.275717169046402
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.9615478515625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28205394744873047
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.985137939453125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27705758810043335
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.956451416015625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2850521206855774
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9886474609375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,0.28999999165534973,0.27521324157714844
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.970489501953125
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27842503786087036
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.9710693359375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28282293677330017
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.98541259765625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2788594961166382
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.9788818359375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.27507805824279785
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.95599365234375
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.26352155208587646
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.90887451171875
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27959930896759033
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.97222900390625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28374576568603516
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.98785400390625
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2729239761829376
# min,max,mean for ConcA_semi_angular_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.937835693359375
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28881824016571045
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.996826171875
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.27000001072883606,0.28999999165534973,0.28955748677253723
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.28968411684036255
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.999969482421875
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2881928086280823
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.99578857421875
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.27000001072883606,0.28999999165534973,0.2891186475753784
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.27000001072883606,0.28999999165534973,0.2897436320781708
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.27000001072883606,0.28999999165534973,0.28996336460113525
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.27000001072883606,0.28999999165534973,0.28997802734375
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.27000001072883606,0.28999999165534973,0.28994321823120117
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.27000001072883606,0.28999999165534973,0.2897247076034546
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.27000001072883606,0.28999999165534973,0.28960752487182617
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.27000001072883606,0.28999999165534973,0.28999388217926025
# min,max,mean for ConcA_semi_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2757255733013153
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.9644775390625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.28007447719573975
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.984130859375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.27497369050979614
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.96392822265625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28382429480552673
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.995880126953125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2800788879394531
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.988861083984375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2742539644241333
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.95281982421875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27615121006965637
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.963775634765625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2765617072582245
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.97357177734375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2710079252719879
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9495849609375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.277640700340271
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.973785400390625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2742944359779358
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.969482421875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2827257215976715
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.993499755859375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2778085470199585
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.975616455078125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.27867037057876587
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.97955322265625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2850987911224365
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.99261474609375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27104538679122925
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9443359375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2778366208076477
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.974395751953125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2794346809387207
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.9766845703125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.27497372031211853
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.968536376953125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2789214253425598
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.988006591796875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27745476365089417
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.97735595703125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.27670642733573914
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9730224609375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.27777206897735596
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.972076416015625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.27894163131713867
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.985015869140625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2778767943382263
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.97210693359375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2795789837837219
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.9842529296875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27602753043174744
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.96795654296875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2760063707828522
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.969696044921875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.274740993976593
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.961029052734375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2729445695877075
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.9532470703125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2722558379173279
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.949615478515625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.275332510471344
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9620361328125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27547603845596313
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.966705322265625
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28190457820892334
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.978607177734375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27864551544189453
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.978851318359375
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.277526319026947
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.976593017578125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27428826689720154
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.95721435546875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27562904357910156
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.966522216796875
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,0.28999999165534973,0.27682214975357056
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.96533203125
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2741294503211975
# min,max,mean for ConcA_semi_angular_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.95831298828125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.27712929248809814
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.970794677734375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27970704436302185
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.979583740234375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.279776930809021
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.977447509765625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2842788100242615
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.9910888671875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28445398807525635
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9923095703125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.28796687722206116
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.99810791015625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28324243426322937
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.9862060546875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.27710551023483276
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.974456787109375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2752813398838043
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.970977783203125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2766694128513336
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.977203369140625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2815232276916504
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.98211669921875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27995914220809937
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.97991943359375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28545600175857544
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.99212646484375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2730571925640106
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.95843505859375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27516475319862366
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.958160400390625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2824368476867676
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.98748779296875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.280475378036499
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.987396240234375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2877867817878723
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99993896484375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.27958017587661743
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.980255126953125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2748904824256897
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.968902587890625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2809563875198364
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.979248046875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.28172510862350464
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.980438232421875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.28083574771881104
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.98370361328125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2810932993888855
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.980926513671875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2732734978199005
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.95257568359375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28155338764190674
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.986907958984375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.281777024269104
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.97528076171875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2832871675491333
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.98883056640625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2769087851047516
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.969390869140625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27685847878456116
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.97613525390625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.28286999464035034
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9903564453125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2809900641441345
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.98114013671875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27804040908813477
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.983154296875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2732478976249695
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.962677001953125
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27692723274230957
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.972900390625
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2737153470516205
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.9647216796875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.27000001072883606,0.28999999165534973,0.28931334614753723
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.28399473428726196
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.98779296875
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2760554552078247
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.977874755859375
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2799256443977356
# min,max,mean for ConcA_semi_angular_mix_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.968048095703125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2812003791332245
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.97967529296875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27962028980255127
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.98040771484375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.26798054575920105
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.926666259765625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.27856823801994324
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.982421875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.26904433965682983
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.930877685546875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.27000001072883606,0.28999999165534973,0.2891162037849426
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28024351596832275
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.9847412109375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.278813898563385
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.97418212890625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.27000001072883606,0.28999999165534973,0.28942808508872986
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2721770107746124
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.944671630859375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2861614227294922
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.99871826171875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.287036269903183
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.998291015625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.26655489206314087
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.920684814453125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2776173949241638
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.977935791015625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.280361533164978
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.971221923828125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.27000001072883606,0.28999999165534973,0.2895098924636841
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2835271656513214
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.99200439453125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2848862409591675
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.998260498046875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2695332169532776
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.938201904296875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2831788957118988
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.97882080078125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2811134457588196
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.99163818359375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2680962085723877
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.931854248046875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2801671028137207
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9754638671875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2720414698123932
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.953765869140625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2812122106552124
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.989349365234375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.27967143058776855
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.977508544921875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27779239416122437
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.97808837890625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27664533257484436
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.95648193359375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,0.28999999165534973,0.28855836391448975
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.9951171875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27821046113967896
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.974090576171875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.27820461988449097
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.976776123046875
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2821059226989746
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.991973876953125
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.28020715713500977
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.988037109375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2706402838230133
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.938079833984375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.27345532178878784
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.943359375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2778320908546448
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.97747802734375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27300897240638733
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.95416259765625
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2803674638271332
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.980804443359375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.27620840072631836
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.959320068359375
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.26390525698661804
# min,max,mean for ConcA_semi_angular_sph_sph_disks_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.905914306640625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.274891197681427
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.961700439453125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.274855375289917
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.966156005859375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2731502056121826
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.95074462890625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2748004198074341
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.9627685546875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2755706310272217
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.96636962890625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2703125476837158
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.94818115234375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.276982843875885
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.97052001953125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2803399860858917
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.989959716796875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27224624156951904
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.946136474609375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.26719358563423157
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.9261474609375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.27224451303482056
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.9544677734375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2730623781681061
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.9556884765625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.277762770652771
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.96612548828125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2808411717414856
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.988800048828125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28814780712127686
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9990234375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2871008515357971
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.996673583984375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2748214602470398
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.96368408203125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.27725839614868164
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.979644775390625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.27863869071006775
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.982391357421875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2756899893283844
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.96856689453125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27416306734085083
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.947662353515625
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2724505066871643
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.945343017578125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2752556800842285
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.969757080078125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2812531590461731
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.982696533203125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2813888192176819
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.983489990234375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2751500606536865
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.952606201171875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27430132031440735
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.95684814453125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2788504660129547
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.9637451171875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2793610095977783
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.98419189453125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.279872328042984
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.979736328125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2812075614929199
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.98504638671875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.275412380695343
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.972015380859375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,0.28999999165534973,0.28397297859191895
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.988067626953125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.27358943223953247
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.959991455078125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2835172712802887
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.99066162109375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.28144967555999756
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.98486328125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27535897493362427
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.963165283203125
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.274429589509964
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.960906982421875
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2847711145877838
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.993896484375
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2736166715621948
# min,max,mean for ConcA_semi_angular_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.964019775390625
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28831687569618225
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.9959716796875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.28528159856796265
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.993804931640625
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.28635457158088684
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.994415283203125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,0.28999999165534973,0.26867860555648804
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.918182373046875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2746780812740326
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9661865234375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28721189498901367
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.993896484375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.27000001072883606,0.28999999165534973,0.288839727640152
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.27000001072883606,0.28999999165534973,0.2899792194366455
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28471365571022034
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.98956298828125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.27000001072883606,0.28999999165534973,0.2898828089237213
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27517589926719666
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.951904296875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2796209454536438
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.985015869140625
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28839007019996643
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.99652099609375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2849881649017334
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.985443115234375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,0.28999999165534973,0.27782681584358215
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.977020263671875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.27000001072883606,0.28999999165534973,0.2899053692817688
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2878987789154053
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.996246337890625
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2790437638759613
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.98187255859375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.26973235607147217
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.94354248046875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2858702540397644
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.991058349609375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28197646141052246
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.987030029296875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27178627252578735
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.941162109375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2875277101993561
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99658203125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27383187413215637
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.961944580078125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2854885160923004
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.9971923828125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.27000001072883606,0.28999999165534973,0.2897234857082367
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2809803783893585
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.986419677734375
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28372544050216675
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.980621337890625
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2710922360420227
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.94415283203125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.28961843252182007
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.998992919921875
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.28719353675842285
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.99444580078125
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.272402286529541
# min,max,mean for ConcA_very_round_elong100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.94500732421875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.27000001072883606,0.28999999165534973,0.2895098924636841
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27958691120147705
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.976318359375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2698485851287842
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.93951416015625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.283275842666626
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.986968994140625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.27549242973327637
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9754638671875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2829021215438843
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.985565185546875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28662341833114624
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.99560546875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2856634557247162
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.9945068359375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28007182478904724
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.981781005859375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27871835231781006
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.980865478515625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,0.28999999165534973,0.2639797031879425
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.907257080078125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.28238409757614136
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.98602294921875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27432048320770264
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.96441650390625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.27712374925613403
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.976226806640625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2746964395046234
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.947601318359375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27669191360473633
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.969482421875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.277784526348114
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.981475830078125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2881122827529907
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.9976806640625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2799701392650604
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.982025146484375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.27000001072883606,0.28999999165534973,0.2897491455078125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2845604419708252
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.987396240234375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.27965015172958374
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9742431640625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2750873863697052
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.971710205078125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.276439368724823
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.973602294921875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2821402847766876
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.991546630859375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2749181389808655
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.96331787109375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,0.28999999165534973,0.27428993582725525
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.960296630859375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.26816216111183167
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.92852783203125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28622499108314514
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.9932861328125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,0.28999999165534973,0.2778589129447937
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.978973388671875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2771538496017456
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.97735595703125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2789607644081116
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.987701416015625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2835211157798767
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.9884033203125
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28089162707328796
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.984954833984375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2776762545108795
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.975738525390625
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2855605483055115
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.9959716796875
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2756844758987427
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.972503662109375
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.27000001072883606,0.28999999165534973,0.2899676561355591
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.27000001072883606,0.28999999165534973,0.28999513387680054
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.27410435676574707
# min,max,mean for ConcA_very_round_elong33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.95947265625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2842552065849304
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.986724853515625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.27262452244758606
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.952392578125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2754438817501068
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.965667724609375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2818964123725891
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.988739013671875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2851055860519409
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.9896240234375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2843111753463745
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.988372802734375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.28220582008361816
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.9822998046875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2826280891895294
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.994110107421875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2803362309932709
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.98406982421875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2833201289176941
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.98883056640625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28009793162345886
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.987152099609375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2812102437019348
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.981292724609375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28306078910827637
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.9925537109375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2857286334037781
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.999267578125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.2838215231895447
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9970703125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2718738913536072
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.944580078125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.28502988815307617
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.998779296875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.28227758407592773
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.9954833984375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2864261269569397
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.999755859375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.28692832589149475
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.996246337890625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2835833430290222
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.996002197265625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2809447646141052
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.99688720703125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.28601551055908203
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9986572265625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.28073787689208984
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.988677978515625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28085553646087646
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.984283447265625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.27316969633102417
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.952239990234375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2852763831615448
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.998382568359375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2839445471763611
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.990997314453125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28715044260025024
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.99896240234375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28226137161254883
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.9884033203125
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2846788763999939
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.991058349609375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.28264522552490234
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9951171875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2837471067905426
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.996185302734375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28018537163734436
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.98699951171875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,0.28999999165534973,0.27200978994369507
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.949310302734375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.27000001072883606,0.28999999165534973,0.2897528111934662
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.27714598178863525
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.97802734375
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27967333793640137
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.9840087890625
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2776433527469635
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.972137451171875
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.27689415216445923
# min,max,mean for ConcA_very_round_elong66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.975830078125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2844512164592743
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.99566650390625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2845851480960846
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.98675537109375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2781108319759369
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.983062744140625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2845277190208435
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.9842529296875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2776893079280853
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.974151611328125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2792077958583832
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.986907958984375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27871209383010864
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.98651123046875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2806550860404968
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.980377197265625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28276461362838745
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.983551025390625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.28041061758995056
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.973297119140625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28132230043411255
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.991424560546875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2816542983055115
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.988037109375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.27850067615509033
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.97161865234375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2834736108779907
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.993682861328125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.283050537109375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9859619140625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2810012996196747
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.988433837890625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.27905839681625366
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.97210693359375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.28374433517456055
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99676513671875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.28132882714271545
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.975372314453125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27920788526535034
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.968109130859375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,0.28999999165534973,0.27768200635910034
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.9700927734375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.28050169348716736
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.970703125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.27679675817489624
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.971527099609375
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2794698178768158
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.986480712890625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2856171131134033
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.99639892578125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2822767198085785
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.99462890625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2851589322090149
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.9959716796875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,0.28999999165534973,0.28143006563186646
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.982391357421875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2836361527442932
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.996307373046875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.28171470761299133
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.9776611328125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2773401737213135
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.972320556640625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.282570481300354
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.98016357421875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.28177332878112793
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.9752197265625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2744818925857544
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.95489501953125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28152143955230713
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.992218017578125
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2811514437198639
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.98992919921875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2825402319431305
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.98406982421875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.28483086824417114
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.99566650390625
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2783511281013489
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.97039794921875
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2808019518852234
# min,max,mean for ConcA_very_round_flack100_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.98773193359375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.28050658106803894
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.9786376953125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.28386932611465454
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.99298095703125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2883933186531067
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.9991455078125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2834860682487488
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.9930419921875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2734353840351105
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.949981689453125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2896959185600281
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.999267578125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.2826823592185974
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.9833984375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.26695024967193604
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.92388916015625
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.28345340490341187
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.989959716796875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.27000001072883606,0.28999999165534973,0.2899945080280304
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2872960567474365
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.995330810546875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.2878865599632263
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.999298095703125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2889111042022705
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.9981689453125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.27000001072883606,0.28999999165534973,0.2899828851222992
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.28112685680389404
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.977264404296875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.27000001072883606,0.28999999165534973,0.2898761034011841
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.28337395191192627
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.98712158203125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.28861504793167114
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99871826171875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.27000001072883606,0.28999999165534973,0.2893396019935608
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 1.0,1.0,1.0
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27376803755760193
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.960357666015625
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.284471720457077
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.993255615234375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.26930493116378784
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.9305419921875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2737026512622833
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9617919921875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2723245620727539
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.958251953125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.28616493940353394
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.990936279296875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.28868067264556885
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.997802734375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27371376752853394
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.9600830078125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2866203784942627
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.995330810546875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.28950485587120056
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.9998779296875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2879323959350586
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.999969482421875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2857476770877838
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9932861328125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.28111642599105835
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.986083984375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28340327739715576
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.991119384765625
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.28429165482521057
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.98870849609375
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28236740827560425
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.9901123046875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2720454931259155
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.948028564453125
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,0.28999999165534973,0.2817528545856476
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.989715576171875
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2856927514076233
# min,max,mean for ConcA_very_round_flack66_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.9945068359375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2746043801307678
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.964813232421875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2830999791622162
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.984130859375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.271792471408844
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.950714111328125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28541019558906555
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.994842529296875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2751506567001343
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.967529296875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.27851176261901855
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.977508544921875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27988576889038086
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.988616943359375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2707821726799011
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.94903564453125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.2793407738208771
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.965484619140625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.27471327781677246
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.958953857421875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.27701476216316223
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.9725341796875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27679258584976196
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.9627685546875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2731558084487915
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.957244873046875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2720510959625244
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.95086669921875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27630728483200073
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.973358154296875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2816895842552185
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.9903564453125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.27152639627456665
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.94921875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2773682773113251
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.9683837890625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2739853858947754
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.958587646484375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.2735886871814728
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.953155517578125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.2740117907524109
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.964263916015625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.2753429710865021
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.96820068359375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2787162959575653
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.9693603515625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.2739369869232178
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.956573486328125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.27342432737350464
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.9508056640625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.27429407835006714
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.9617919921875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.27237367630004883
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.953338623046875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.28076261281967163
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.97528076171875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27164122462272644
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.9530029296875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.27571144700050354
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.964691162109375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2746586799621582
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.965362548828125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2777670621871948
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.971343994140625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27288302779197693
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.948150634765625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.28432998061180115
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.990081787109375
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2727906107902527
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.951568603515625
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,0.28999999165534973,0.2751806080341339
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.96453857421875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2783052921295166
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.975830078125
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.2741397023200989
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.957244873046875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.27474653720855713
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.96746826171875
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2732791006565094
# min,max,mean for ConcA_very_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.95697021484375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0989999994635582,0.28999999165534973,0.2763565480709076
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.0,1.0,0.978057861328125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0989999994635582,0.28999999165534973,0.2756917178630829
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.0,1.0,0.97088623046875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0989999994635582,0.28999999165534973,0.2806634306907654
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.989898681640625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.2750502824783325
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.9739990234375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.2769118845462799
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.983489990234375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0989999994635582,0.28999999165534973,0.2803398668766022
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.0,1.0,0.989837646484375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0989999994635582,0.28999999165534973,0.27552151679992676
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.97100830078125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.27400341629981995
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.9677734375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27208447456359863
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.9580078125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2734810709953308
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.9593505859375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.2772887051105499
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.97607421875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27199262380599976
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.953765869140625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.2703474164009094
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.94580078125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.2732166349887848
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.958282470703125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.27910923957824707
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.9820556640625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.2778659164905548
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.98199462890625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.2759557366371155
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.977020263671875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2766278386116028
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.972869873046875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.271633505821228
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.952301025390625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.283534973859787
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.99493408203125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.27563971281051636
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.96588134765625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0989999994635582,0.28999999165534973,0.27668023109436035
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.0,1.0,0.97430419921875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.2758762538433075
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.964691162109375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.27269768714904785
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.95355224609375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2733538746833801
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.9652099609375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.2726353108882904
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.95953369140625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.28013908863067627
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.99566650390625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.27144360542297363
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.9464111328125
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.27172666788101196
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.957275390625
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2738918662071228
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.970062255859375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2706034481525421
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.953704833984375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2842337489128113
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.994873046875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.27625685930252075
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.971771240234375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.2762734889984131
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.980438232421875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.2806583642959595
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.98577880859375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0989999994635582,0.28999999165534973,0.2752619981765747
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.972381591796875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.28121548891067505
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.991424560546875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0989999994635582,0.28999999165534973,0.27892357110977173
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.973388671875
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.27772319316864014
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.98089599609375
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.27300816774368286
# min,max,mean for Fuller_p5_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.9637451171875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 0.27000001072883606,0.28999999165534973,0.28974059224128723
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_1.npy are: 1.0,1.0,1.0
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 0.27000001072883606,0.28999999165534973,0.2895910441875458
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_10.npy are: 1.0,1.0,1.0
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,0.28999999165534973,0.26679378747940063
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_11.npy are: 0.0,1.0,0.929107666015625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0989999994635582,0.28999999165534973,0.28497636318206787
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_12.npy are: 0.0,1.0,0.99407958984375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0989999994635582,0.28999999165534973,0.28226524591445923
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_13.npy are: 0.0,1.0,0.986480712890625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 0.27000001072883606,0.28999999165534973,0.2899322509765625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_14.npy are: 1.0,1.0,1.0
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,0.28999999165534973,0.2821654677391052
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_15.npy are: 0.0,1.0,0.989654541015625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0989999994635582,0.28999999165534973,0.2831318974494934
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_16.npy are: 0.0,1.0,0.98797607421875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0989999994635582,0.28999999165534973,0.27253103256225586
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_17.npy are: 0.0,1.0,0.954437255859375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0989999994635582,0.28999999165534973,0.2775841951370239
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_18.npy are: 0.0,1.0,0.977081298828125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0989999994635582,0.28999999165534973,0.28475385904312134
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_19.npy are: 0.0,1.0,0.991607666015625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0989999994635582,0.28999999165534973,0.27411603927612305
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_2.npy are: 0.0,1.0,0.96319580078125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0989999994635582,0.28999999165534973,0.28057238459587097
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_20.npy are: 0.0,1.0,0.99072265625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0989999994635582,0.28999999165534973,0.278402715921402
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_21.npy are: 0.0,1.0,0.97802734375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0989999994635582,0.28999999165534973,0.26741212606430054
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_22.npy are: 0.0,1.0,0.926483154296875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0989999994635582,0.28999999165534973,0.27602213621139526
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_23.npy are: 0.0,1.0,0.97174072265625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0989999994635582,0.28999999165534973,0.27653253078460693
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_24.npy are: 0.0,1.0,0.96453857421875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0989999994635582,0.28999999165534973,0.2858278155326843
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_25.npy are: 0.0,1.0,0.99591064453125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0989999994635582,0.28999999165534973,0.2876622974872589
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_26.npy are: 0.0,1.0,0.99688720703125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0989999994635582,0.28999999165534973,0.27335625886917114
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_27.npy are: 0.0,1.0,0.96270751953125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0989999994635582,0.28999999165534973,0.28082436323165894
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_28.npy are: 0.0,1.0,0.99853515625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 0.27000001072883606,0.28999999165534973,0.28975218534469604
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_29.npy are: 1.0,1.0,1.0
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0989999994635582,0.28999999165534973,0.27181294560432434
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_3.npy are: 0.0,1.0,0.948028564453125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0989999994635582,0.28999999165534973,0.279368132352829
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_30.npy are: 0.0,1.0,0.976348876953125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0989999994635582,0.28999999165534973,0.2797771692276001
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_31.npy are: 0.0,1.0,0.98956298828125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0989999994635582,0.28999999165534973,0.27780312299728394
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_32.npy are: 0.0,1.0,0.971405029296875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0989999994635582,0.28999999165534973,0.2668779194355011
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_33.npy are: 0.0,1.0,0.920806884765625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0989999994635582,0.28999999165534973,0.2835269570350647
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_34.npy are: 0.0,1.0,0.997589111328125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0989999994635582,0.28999999165534973,0.2810734510421753
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_35.npy are: 0.0,1.0,0.98394775390625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0989999994635582,0.28999999165534973,0.2847927212715149
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_36.npy are: 0.0,1.0,0.99658203125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0989999994635582,0.28999999165534973,0.2766348123550415
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_37.npy are: 0.0,1.0,0.976776123046875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0989999994635582,0.28999999165534973,0.2759268879890442
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_38.npy are: 0.0,1.0,0.9698486328125
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0989999994635582,0.28999999165534973,0.2887493968009949
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_39.npy are: 0.0,1.0,0.9993896484375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0989999994635582,0.28999999165534973,0.274971067905426
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_4.npy are: 0.0,1.0,0.9564208984375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0989999994635582,0.28999999165534973,0.280674010515213
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_40.npy are: 0.0,1.0,0.99407958984375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,0.28999999165534973,0.2864837050437927
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_5.npy are: 0.0,1.0,0.99896240234375
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0989999994635582,0.28999999165534973,0.2742166221141815
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_6.npy are: 0.0,1.0,0.955474853515625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,0.28999999165534973,0.27405646443367004
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_7.npy are: 0.0,1.0,0.96246337890625
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0989999994635582,0.28999999165534973,0.2844160497188568
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_8.npy are: 0.0,1.0,0.994354248046875
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0989999994635582,0.28999999165534973,0.2857721149921417
# min,max,mean for Fuller_p6_round_sph_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_SimulVol_Pair_9.npy are: 0.0,1.0,0.9951171875


# import os
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# import numpy as np
# from math import sqrt
# from tqdm import tqdm


# def scan_chunk(indices, inp_dir, files):
#     # Return tuple: (n, sum, sumsq)
#     n_total = 0
#     s_total = 0.0
#     ss_total = 0.0
#     for i in indices:
#         arr = np.load(os.path.join(inp_dir, files[i]), mmap_mode='r')
#         # Summations in float64 for numerical stability
#         s = arr.sum(dtype=np.float32)
#         ss = np.square(arr, dtype=np.float32).sum(dtype=np.float32)
#         n = arr.size
#         n_total += n
#         s_total += s
#         ss_total += ss
#     return n_total, s_total, ss_total

# def global_mean_std(inp_dir, files, num_chunks=20, max_workers=None):
#     # Build (almost) equal chunks of indices
#     N = len(files)
#     if N == 0:
#         raise ValueError("No files provided.")
#     chunk_size = max(1, N // num_chunks)
#     chunks = [list(range(i, min(i + chunk_size, N))) for i in range(0, N, chunk_size)]
#     # Parallel scan
#     worker = partial(scan_chunk, inp_dir=inp_dir, files=files)
#     results = []
#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         for out in tqdm(ex.map(worker, chunks), total=len(chunks)):
#             results.append(out)
#     # Combine totals
#     n_total = sum(r[0] for r in results)
#     s_total = sum(r[1] for r in results)
#     ss_total = sum(r[2] for r in results)
#     mean = s_total / n_total
#     # population variance: ss_total/n_total - mean^2
#     # convert to unbiased sample variance with Bessel's correction:
#     var = (ss_total - n_total * mean * mean) / (n_total - 1)
#     std = sqrt(var) if var > 0 else 0.0
#     return mean, std

# inp_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files'
# out_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files_standardized'
# os.makedirs(out_dir,exist_ok=True)
# files = [ f for f in sorted(os.listdir(inp_dir)) if f.endswith('npy')]
# mean_val, std_val = global_mean_std(inp_dir, files, num_chunks=20, max_workers=8)
# print(mean_val, std_val) # (np.float64(15202.856814677318), 5063.825531473543)
# for f in files:
#     x=np.load(os.path.join(inp_dir,f))
#     print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
#     x = (x-mean_val)/std_val
#     np.save(os.path.join(out_dir,f),x)
#     print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")

## inp_dir =  '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth/'
## out_dir = '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth_standardized/'
## # #0.27842160105071867 0.03345132856836802
