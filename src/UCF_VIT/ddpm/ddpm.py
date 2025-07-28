import os
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
import torch.distributed as dist
import numpy as np

# import sys
# sys.path.insert(0, '/home/ziabariak/git/UCF-VIT/src')

# from torcheval.metrics import FrechetInceptionDistance
# from UCF_VIT.utils.plotting import plot_3D_array_slices, plot_3D_array_center_slices,plot_3D_array_center_slices_up
from UCF_VIT.utils.misc import unpatchify

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)
        self.num_time_steps = num_time_steps

    def forward(self, t):
        return self.beta[t], self.alpha[t]

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
#             # plot_3D_array_slices(images, filename=os.path.join(save_path, '3D_gen_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#             plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#             # fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10))#,facecolor='w')
#             # for i in range(num_samples):
#             #     for j in range(len(times)):
#             #         ax[i,j].axis('off')
#             #         print(f"shape is {images[j].shape}")
#             #         sh_1 = images[j].shape[1]
#             #         ax[i,j].imshow(images[j][i,sh_1//2,:,:,0], interpolation='none')
#             # # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
#             # plt.tight_layout()
#             # plt.show()
#             # fig.savefig(os.path.join(save_path,'GOH_gen_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())), format='png', dpi=300)
#             # plt.close(fig)
        
#         else:
#             fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10),facecolor='w')
#             for i in range(images[0].shape[0]):
#                 for j in range(len(times)):
#                     ax[i,j].axis('off')
#                     ax[i,j].imshow(images[j][i,:,:,0], interpolation='none')
#             plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
#             fig.savefig(os.path.join(save_path,'2D_gen_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())), format='png', dpi=300)
#             plt.close(fig)


# def save_intermediate_data(model, var, device, res, precision_dt, patch_size, epoch=0, num_samples=10, twoD=False, save_path='figures', num_time_steps=1000):

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
    
#     images = np.array(images)
#     images = images.astype('float32')

#     if not twoD:
#         plot_3D_array_center_slices(images, filename=os.path.join(save_path, '3D_gen_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#         np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_rank%i.npz' %(var, epoch, res[1], res[2], dist.get_rank())),images)
#     else:
#         plot_2D_array_slices(images, filename=os.path.join(save_path, '2D_gen_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())))
#         np.savez(os.path.join(save_path,'Output_gen_%s_%i_%i_%i_%i_rank%i.npz' %(var, epoch, res[0], res[1], res[2], dist.get_rank())),images)



# def extract_rgb_like_slices(volume, num_samples=5):
#     """
#     Converts a 3D volume (1, D, H, W) to multiple (3, H, W) pseudo-RGB slices.
#     """
#     _, D, H, W = volume.shape
#     indices = torch.linspace(1, D - 2, steps=num_samples).long()
#     slices = []

#     for i in indices:
#         s = torch.cat([
#             volume[:, i - 1, :, :],
#             volume[:, i, :, :],
#             volume[:, i + 1, :, :]
#         ], dim=0)  # (3, H, W)
#         slices.append(s)

#     return torch.stack(slices)  # (num_samples, 3, H, W)


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
#     images = images.astype('float32')

#     # Save visuals and arrays
#     if not twoD:
#         print("writing the center slice!")
#         plot_3D_array_center_slices_up(images, filename=os.path.join(save_path, '3D_GEN_FID_centerSlice_%s_%i_%i_%i_%irank%i.png' %(var, epoch, res[0], res[1], res[2], dist.get_rank())))
#         # plot_3D_array_center_slices(images, filename=os.path.join(
#         #     save_path, f'3D_gen_centerSlice_{var}_{epoch}_{res[0]}_{res[1]}_{res[2]}rank{dist.get_rank()}.png'))
#         # np.savez(os.path.join(
#         #     save_path, f'Output_gen_{var}_{epoch}_{res[1]}_{res[2]}_rank{dist.get_rank()}.npz'), images)
#     else:
#         fig,ax = plt.subplots(num_samples,ncols = len(times),figsize=(10,10),facecolor='w')
#         for i in range(images[0].shape[0]):
#             for j in range(len(times)):
#                 ax[i,j].axis('off')
#                 ax[i,j].imshow(images[j][i,:,:,0], interpolation='none')
#         plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.025, wspace=0.025)
#         fig.savefig(os.path.join(save_path,'2D_gen_FID_%s_%i_%i_%i_rank%i.png' %(var, epoch, res[1], res[2], dist.get_rank())), format='png', dpi=300)
#         plt.close(fig)
#         # plot_2D_array_slices(images, filename=os.path.join(
#         #     save_path, f'2D_gen_{var}_{epoch}_{res[1]}_{res[2]}_{res[0]}rank{dist.get_rank()}.png'))
#         # np.savez(os.path.join(
#         #     save_path, f'Output_gen_{var}_{epoch}_{res[0]}_{res[1]}_{res[2]}_rank{dist.get_rank()}.npz'), images)

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
