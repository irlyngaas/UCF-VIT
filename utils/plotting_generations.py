import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics import FrechetInceptionDistance
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
            eps_hat = model(x_t.to(dtype=precision_dt), t, [variables])
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