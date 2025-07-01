## Table of Contents
- [UCF-VIT](#ucf-vit)
- [Install](#install)
- [Innovations](#Innovations)
- [Supported Model Architectures](#supported-model-architectures)
1. [Vision Transformer](#vision-transformer-vit)
2. [Masked Autoencoder](#masked-autoencoder-mae)
3. [Unet Transformer](#unet-transformer-unetr)
4. [Symmetric Adaptive Patching](#symmetric-adaptive-patching-sap)
5. [Diffusion Vision Transformer](#diffusion-vision-transformer-diffusionvit)
- [Dataloader](#dataloader)
- [Dataset Integration](#dataset-integration)
- [Load Balancing](#load-balancing)
- [Training Scripts](#training-scripts)

## UCF-VIT
UCF-VIT is a **Uniform Coding Framework (UCF)** for training large scale **Vision Transformer (VIT)** based models. The framework brings together a host of functionalities for the purpose of training large models with large input data in a scalable and efficient manner. It consists of advanced parallelism schemes, State of the Art techniques for efficient computing, and custom dataloader utilities that are integrated with the aforementioned schemes and techniques to allow for training on a multitude of different datasets.


The intention is to provide the building blocks and utilities for using these different parallelism techniques in fashion that they can be easily integrated to use with various types of scientific data. We provide various different end to end examples for different computer vision tasks using two example datasets. We provide various different options so that the integration of new datasets can be done with a number of different strategies. We also provide various advanced techniques that we have been developed for the specific use case of efficient computing with large scientific datasets.

## Why use UCF-VIT?

# Install
Installation instruction are provide for running on Frontier and an NVIDIA DGX Cluster

## Frontier
There are two options available for creating software environments on Frontier 1) creating Conda environment from scratch or 2) using an apptainer container. Creating a Conda environment from scratch is recommended as the Apptainer containers currently only work in limited scenarios due to missing ROCM packages in the base apptainer image.

### Conda
Create Conda Environment from Scratch. Example below using options from the corresponding Apptainer definition files
```
PYTHON_VERSION=3.11
conda create -n vit python=${PYTHON_VERSION} -y
conda activate vit 
ROCM_VERSION=6.2.4
TORCH_URL="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
TORCH_VERSION=2.7.0+rocm6.2.4
TORCHVISION_VERSION=0.22.0
TORCHAUDIO_VERSION=2.7.0

pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_URL}
pip install xformers --extra-index-url=https://download.pytorch.org/whl/rocm${ROCM_VERSION}
pip install timm \
 monai \
 nibabel \
 torchdata==0.9.0 \
 einops \
 opencv-python-headless \
 matplotlib \
 scipy

cd UCF-VIT
pip install -e .
```

### Apptainer
Use Apptainer container definition files
```
cd apptainter
apptainer build frontier-ubuntu-gnu-rocm624.sif frontier-ubuntu-gnu-rocm624.def
apptainer build frontier-ubuntu-gnu-rocm624-vit.sif frontier-ubuntu-gnu-rocm624-vit.def
mkdir lib/8.1.31
cd lib/8.1.31
ln -s $CRAY_MPICH_DIR/lib/libfmpich.so libmpicxx.so
```

Various example scripts for launching jobs are in the launch folder. Those identified with `_apptainer` in the filename are for running with the Apptainer container

## NVIDIA DGX
To run on NVIDIA DGX systems we rely on Pytorch Docker containers maintained by NVIDIA. The following instructions give commands to build a docker container with our codebase.

```
cd Docker
docker build -t ucf-vit:25.05 .
docker tag ucf-vit:25.05 [DOCKER_USERNAME]/[DOCKER_REPO]/ucf-vit:25.05
docker push [DOCKER_USERNAME]/[DOCKER_REPO]/ucf-vit:25.05
```
Various example scripts for launching jobs are in the launch folder. Those identified with `_dgx` in the filename are for running with the Docker container

# Innovations
Advanced Parallelism & Efficient Computing

## Hybrid-STOP 
Hybrid Sharded Tensor-Data Orthogonal Parallelism (Hybrid-STOP) [[1]](#1) is a novel parallelism algorithm that combines tensor parallelism and Fully Sharded Data Parallelism (FSDP). It avoids the peak memory use probelm in FSDP and leads to better memory reduction capability by keeping parameters sharded throughout training. 
### Usage
The Hybrid-STOP algorithm is available when using our fsdp [parallelism mode](#parallelism-modes). The following example shows how to initialize and do the forward pass of a [MAE](#masked-autoencoder-mae) model using this algorithm for different number of simple_ddp, fsdp, and tensor parallel ranks. Our custom [dataloader](#dataloader) with the [Imagenet](#imagenet) dataset is used to facilitate proper dataloading when tensor parallelism is > 1 (In this case each tensor parallel rank needs the same batch of input data).

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import timedelta
from UCF_VIT.utils.misc import init_par_groups
from UCF_VIT.fsdp.arch import MAE
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule

os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
os.environ['MASTER_PORT'] = "29500"
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['RANK'] = os.environ['SLURM_PROCID']

world_size = int(os.environ['SLURM_NTASKS'])
world_rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])

torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

#Assume we have 8 GPUs for training. Change these variables in any way desired such that fsdp_size*simple_ddp_size*tensor_par_size=8
fsdp_size = 2
simple_ddp_size = 2
tensor_par_size = 2
seq_par_size = 1 


data_par_size = fsdp_size * simple_ddp_size
assert seq_par_size == 1, "Sequence parallelism not implemented"
assert (data_par_size * seq_par_size * tensor_par_size)==world_size, "DATA_PAR_SIZE * SEQ_PAR_SIZE * TENSOR_PAR_SIZE must equal to world_size"
num_heads = 12
assert (num_heads % tensor_par_size) == 0, "model heads % tensor parallel size must be 0"
decoder_num_heads = 8
assert (decoder_num_heads % tensor_par_size) == 0, "decoder model heads % tensor parallel size must be 0"

seq_par_group, ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(world_rank = world_rank, data_par_size = data_par_size, tensor_par_size = tensor_par_size, seq_par_size = seq_par_size, fsdp_size = fsdp_size, simple_ddp_size = simple_ddp_size)

model = MAE(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=num_heads,
        linear_decoder=False,
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        drop_path_rate=0.1,
        mask_ratio=0.1,
        twoD=True,
        mlp_ratio_decoder=4,
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        tensor_par_size=tensor_par_size,
        tensor_par_group=tensor_par_group,
        class_token=False,
        weight_init='skip',
    )

if world_rank==0:
    #save initial model weights and distribute to all GPUs in the tensor parallel group to synchronize model weights that do not belong to the training block
    init_model_dict = {k: v for k, v in model.state_dict().items() if ('attn' not in  k and 'mlp' not in k and 'var_agg' not in k)}
    torch.save(init_model_dict,
        'initial_'+str(dist.get_rank())+'.pth')
    del init_model_dict

dist.barrier()

if world_rank!=0 and world_rank <tensor_par_size:
   #load initial model weights and synchronize model weights that are not in the training block among tensor parallel GPUs
   src_rank = dist.get_rank() - dist.get_rank(group=tensor_par_group)

   map_location = 'cpu'
   model.load_state_dict(torch.load('initial_'+str(0)+'.pth',map_location=map_location),strict=False)

dist.barrier()

precision_dt = torch.float32
if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, forward_prefetch=True, limit_all_gathers = False )
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, forward_prefetch=True, limit_all_gathers = False )
else:
    model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, forward_prefetch=True, limit_all_gathers = False )


if dist.get_rank(tensor_par_group) == 0:
    data_module = NativePytorchDataModule(dict_root_dirs={'imagenet': '~/imagenet/train',},
            dict_start_idx={'imagenet': 0},
            dict_end_idx={'imagenet': 1},
            dict_buffer_sizes={'imagenet': 100},
            dict_in_variables={'imagenet': ["red", "green", "blue"]},
            num_channels_available = {'imagenet': 3},
            num_channels_used = {'imagenet': 3},
            batch_size=32,
            num_workers=1,
            pin_memory=False,
            patch_size = 16,
            tile_size_x = 256,
            tile_size_y = 256,
            tile_size_z = None,
            twoD = True,
            single_channel = False,
            return_label = False,
            dataset_group_list = '1:1:1:1', #Calculate from running utils/preprocess_load_balancing.py
            batches_per_rank_epoch = {'imagenet':9945}, #Calculated from running utils/preprocess_load_balancing.py
            tile_overlap = 0.0,
            use_all_data = False,
            adaptive_patching = False,
            fixed_length = None,
            separate_channels = False,
            data_par_size = data_par_size,
            ddp_group = ddp_group,
            dataset = 'imagenet',
            imagenet_resize = {'imagenet':[256,256]},
        ).to(device)

    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    it_loader = iter(train_dataloader)

counter = 0
while counter < 9945:
    if tensor_par_size > 1:
        if dist.get_rank(tensor_par_group) == 0:
            data, variables, _ = next(it_loader)
            data = data.to(precision_dt)
            data = data.to(device)
        else:
            data = torch.zeros(32, 3, 256, 256, dtype=precision_dt).to(device)
            variables = [None] * 3
        dist.broadcast(data, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)
        dist.broadcast_object_list(variables, src=(dist.get_rank()//tensor_par_size*tensor_par_size), group=tensor_par_group)

    else: #Avoid unnecesary broadcasts if not using tensor parallelism
        data, variables, _ = next(it_loader)
        data = data.to(precision_dt)
        data = data.to(device)
    model_output = model.forward(data, variables)

```

### Implementation Details 
In order to properly implement the H-STOP algorithm, specifically the tensor parallelism aspects, the architecture code requires modifications to correctly split up and communicate the tensor calculations amongst the tensor parallel ranks. Currently tensor parallelism is only enacted over the attention and mlp calculations, which are the costliest components. The below code snippet shows how the tensor parallel implementation is implemented in the FSDP parallelism mode, corresponding modifications are built into the attention mechanism in building_blocks.py. Tensor parallelism is not implemented in simple parallelism mode, so the tensor_par_group communicator group is not required to be passed to the neural network architecture for that case.  

```python
if self.tensor_par_size > 1:
    src_rank = dist.get_rank() - dist.get_rank(group=self.tensor_par_group)
    dist.broadcast(x, src_rank, group=self.tensor_par_group)

x = self.blocks(x)
x = self.norm(x)

if self.tensor_par_size > 1:
    x = F_Identity_B_Broadcast(x, src_rank, group=self.tensor_par_group)

```

## Lower Precision Support
The ability to implement lower precision support is facilitated by using a MixedPrecision Policy which is passed as an argument to the FSDP wrapper

### Usage 
Add the following code and replace the FSDP Wrapper calls in the [full example](#Hybrid-STOP) to apply mixed precision training (specifically bfloat16 in this case). 
```python
precision_dt = torch.bfloat16
mixedPrecisionPolicy = MixedPrecision(
    param_dtype=precision_dt,
    reduce_dtype=precision_dt,
    buffer_dtype=precision_dt,
)

if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers = False )
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers = False )
else:
    model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers = False )
```

## Layer Wrapping
TEXT TO DESCRIBE
### Usage 
Add the following code and replace the FSDP Wrapper calls in the [full example](#Hybrid-STOP) to apply layer wrapping
```python
from UCF_VIT.fsdp.building_blocks import Block
from torch.nn import Sequential
import functools

my_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        Block, Sequential   # < ---- Your Transformer layer class
    },

if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group= (fsdp_group,simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy = my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers = False )
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group= fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy = my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers = False )
else:
    model = FSDP(model, device_id=local_rank, process_group= simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy = my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers = False )
)
```
## Activation Checkpointing
TEXT TO DESCRIBE
### Usage
Add the following code after the FSDP Wrapper
```python
from UCF_VIT.fsdp.building_blocks import Block
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   apply_activation_checkpointing,
)

check_fn = lambda submodule: isinstance(submodule, Block)
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
)

```
## Composable Kernels

## Advanced Features
## Adaptive Patching
## Variable Aggregation

# Supported Model Architectures
Currently we provide 5 different architecutres **(VIT, MAE, UNETR, SAP, VIT-DIFFUSION)**, all of which use the same VIT encoder, but a different decoder architecture depending on the task being trained. All code for the different architectures inherit the ecnoder from VIT in order to facilitate using the same encoder.

## Vision Transformer (VIT)
VIT based on [1]. Code adapted and slimmed down from (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L425) to only contain basic options for VIT Training and options to use some of the [innovations](#innovations). Task: Image Classification. Input: Image or Image Tile (A tile is a subset portion of a full image).

### Usage
```python
import torch
from UCF_VIT.simple.arch import VIT
from UCF_VIT.utils.fused_attn import FusedAttn

model = VIT(
    img_size = [256,256],
    patch_size = 16,
    num_classes = 1000,
    in_chans = 3,
    embed_dim = 768,
    depth = 12,
    num_heads = 12,
    mlp_ratio = 4,
    drop_path_rate = 0.1,
    drop_rate = 0.1,
    twoD = True, # set False if 3D
    use_varemb = False,
    default_vars = ["red", "green", "blue"]
    single_channel = False,
    adaptive_patching = False,
    fixed_length = None,
    FusedAttn_option = FusedAttn.DEFAULT,
)

img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 1000)
```

### Parameters

- `img_size`: int, Tuple[int,int], or Tuple[int,int,int].  
Image size. If a single int is given the input image is assumed to be the same in each dimension and the dimension of the image is set via the **twoD** variable. If **adaptive_patching** is set to True this variable is ignored, since input has already been patched in an adaptive fashion in the dataloader.

- `patch_size`: int.  
Size of patches. `image_size` must be divisible by `patch_size` if **adaptive_patching** is set to True.

- `num_classes`: int.  
Number of classes to classify.

- `in_chans`: int.  
Number of input channels contained in each input image. E.g, a JPEG image has 3 color channels [R,G,B] at each pixel

- `embed_dim`: int.  
Embedding dimension for Transformer Inputs

- `depth`: int.
Number of Transformer blocks.

- `num_heads`: int.  
Number of heads in Multi-head Attention layer.

- `mlp_ratio`: int.
Ratio of MLP hidden dimension to embedding dimension, used to set the dimension of the MLP (FeedForward) layer.

- `drop_path_rate`: float (0,1).
Stochastic depth dropout rate for dropping random layers during training

- `drop_rate`: float (0,1).
Dropout rate for classification head

- `twoD`: bool.
Variable for indicating two or three dimensionsal input, if False, three dimensional input. Needed to do correct patching. Used in coordination with the [dataloader module](#dataloader) to correctly set up the data for the given architecture

- `use_varemb`: bool.
Variable for indicating whether to use variable embedding tokens. When using variable embedding tokens each input channel is tokenized separately. In order to feed these tokens into the model, the separate variable tokens are fed through [variable aggregation](#variable-aggregation) to compress multiple input into a single aggregated input channel via an attention mechanism

- `default_vars`: list[str].
List of different potential modalities to be used as input. When **use_varemb** is set to true, this list contains the possible variable tokenizations available.

- `single_channel`: bool.
Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality

- `adaptive_patching`: bool.
Variable for indicating whether to use adaptive patching. See [Adaptive Patching](#adaptive-patching)

- `fixed_length`: int.
How many adaptive patches used to tokenize the input image. Only used if **adaptive_patching** is set to true

- `FusedAttn_option`: [FusedAttn.CK, FusedAttn.DEFAULT, FusedAttn.NONE]
Which option to use for fused attention. CK - [ComposableKernels](#composable-kernels), DEFAULT - torch implementaion, or None - No fused-attention used

## Masked Autoencoder (MAE)
Masked Autoencoder pre-training based on [2]. Task: Masked Image Prediction. Input: Image or Image Tile

### Usage
```python
import torch
from UCF_VIT.simple.arch import MAE
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.utils.misc import unpatchify

model = MAE(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        linear_decoder=False,
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.1,
        mask_ratio=0.1,
        twoD=True,
        mlp_ratio_decoder=4,
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 16*16, 16*16*3)

#Move masked image prediction from patched space back to original image space
pred_img = unpatchify(preds, img, 16, True) # (1, 3, 256, 256) 
```

### Parameters

- `linear_decoder`: bool.
Variable to indicate whether to use a linear decoder. If False, a Transformer decoder is used to predict the mask prediction output

- `decoder_depth`: int.
Number of Transformer blocks to use in the decoder. Not used if **linear_decoder** is set to True

- `decoder_embed_dim`: int.  
Embedding dimension for Inputs to the Transformer decoder. Not used if **linear_decoder** is set to True

- `decoder_num_heads`: int.  
Number of heads in Multi-head Attention layer for the Transformer decoder. Not used if **linear_decoder** is set to True

- `mlp_ratio_decoder`: int.
Ratio of MLP hidden dimension to embedding dimension, used to set the dimension of the MLP (FeedForward) layer in the Transfomer decoder. Not used if **linear_decoder** is set to True

- `mask_ratio`: float (0, 1).
Amount of tokens to mask out in the Transformer encoder

- `class_token`: bool.
Whether to append a class token to the tokenized data. Set to false unless using VIT

- `weight_init`: ['' or 'skip'].
Whether to skip the weight_init in the VIT parent class. Set to 'skip' unless using VIT

## UNet Transformer (UNETR)
Image segmentation architecture based on [3]. Task: Image Segmentation. Input: Image or Image Tile

### Usage
```python
import torch
from UCF_VIT.simple.arch import UNETR
from UCF_VIT.utils.fused_attn import FusedAttn

model = UNETR(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        twoD=True,
        default_vars=["red", "green", "blue"],
        linear_decoder=False,
        skip_connection=True,
        feature_size=16,
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 4, 256, 256)
```

### Parameters

- `num_classes`: int.  
Number of classes to predict from at each image pixel.

- `linear_decoder`: bool.
Variable to indicate whether to use a linear decoder. If False, a convolutional decoder is used to predict the class of each pixel

- `skip_connection`: bool.
Variable to indicate whether to use skip connection in the convolutional decoder. The skip connection uses intermediate output from the Trasnformer encoder blocks

- `feature_size`: int.
Variable to set the how embedding features are expanded through the UNETR convolutional blocks

## Symmetric Adaptive Patching (SAP)
Image segmentation architecture for adaptively patched input based on [4]. Task: Image Segmentation. Input: Adaptively Patching Image or Image Tile

### Usage
```python
import torch
from UCF_VIT.simple.arch import SAP
from UCF_VIT.utils.fused_attn import FusedAttn

model = SAP(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        twoD=True,
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=True,
        fixed_length=4096,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 4, sqrt(4096)*16, sqrt(4096)*16)
```

### Diffusion Vision Transformer (DiffusionVIT)
Diffusion model training based on [5]. Task: Generate Image via noise that matches distribution of data trained on. Input: Noise

### Usage
```python
import torch
from UCF_VIT.simple.arch import DiffusionVIT
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler

model = DiffusionVIT(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.1,
        twoD=True,
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        time_steps=1000,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

t = torch.randint(0,num_time_steps,(1,))
e = torch.randn_like(data, requires_grad=False)
ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
a = ddpm_scheduler.alpha[t].view(batch_size,1,1,1)
data = (torch.sqrt(a)*data) + (torch.sqrt(1-a)*e)
output = net.forward(data, t, variables) # (1, 16*16, 16*16*3)

#Move generated image from patched space back to original image space
output = unpatchify(output, data, patch_size, twoD) #(1, 3, 256, 256)
```
### Parameters

- `time_steps`: int.
Number of time steps in the diffusion process

## Dataloader
The dataloader we provide is a custom native pytorch iterative dataloader. For simplicity, we assume that we are receiving raw data files and we leave it to the user to normalize the data properly within in the dataloader module for training. The reasons for making this assumption of raw data file as input is 1) we intend this repo to be used on very large datasets, thus preprocessing and storing all of the data before training can quickly take up a massive amount of storage, and 2) it removes the need for further data preprocessing scripts to be included in this repo. If performing preprocessing during the dataloading phase is too computationally intensive, we recommend doing it offline and properly storing it in a manner that the dataloader module can handle. 

The dataloader is built in a fashion such that it can handle multiple different dataset directories at the same time. A dataset directory contains one or more raw data file (with all raw data files having the same dimension or able to be resized so that they have the same dimension). The purpose of being able to handle multiple dataset directories is 1) it provides flexible training where you can easily remove and add different datasets for the purposes of running experiments and 2) it allows for the integration of identifying properties from the different datasets that can potentially used for improved learning via our advanced features. For instance, with data that has multiple channels, e.g. images with (R,G,B) channels, we are able to pass along the information on what variable the channel is from and use that information during network training. We then could utilize [variable aggregration](#variable-aggregation) to tokenize each channel separately.

This dataloader provides the flexibility to add a plethora of different options for customizing how the data is broken up for training. Since we are using a VIT, at least 2D data is expected. However, we have capability for both 2D and 3D spatial data currently. If desired, we have the utilities implemented to break given raw data into smaller tiled chunks. Also, we have a number of different options for how to tile this raw data, e.g. tile overlapping.

### Usage
```python
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
import torch.distributed as dist

data_module = NativePytorchDataModule(dict_root_dirs={'imagenet': '~/imagenet/train',},
        dict_start_idx={'imagenet': 0},
        dict_end_idx={'imagenet': 1},
        dict_buffer_sizes={'imagenet': 100},
        dict_in_variables={'imagenet': ["red", "green", "blue"]},
        num_channels_available = {'imagenet': 3},
        num_channels_used = {'imagenet': 3},
        batch_size=32,
        num_workers=1,
        pin_memory=False,
        patch_size = 16,
        tile_size_x = 256,
        tile_size_y = 256,
        tile_size_z = None,
        twoD = True,
        single_channel = False,
        return_label = False,
        dataset_group_list = '8',
        batches_per_rank_epoch = {'imagenet':4935},
        tile_overlap = 0.0,
        use_all_data = False,
        adaptive_patching = False,
        fixed_length = None,
        separate_channels = False,
        data_par_size = dist.get_world_size(),
        dataset = 'imagenet',
        imagenet_resize = [256,256],
    ).to(device)

    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    for batch_idx, batch in enumerate(train_dataloader):
        data, variables = batch
```

### Parameters

- `dict_root_dirs`: Dictionary of paths.
Paths to directories with raw input data

- `dict_start_idx`: Dictionary of floats (0,1).
Starting indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_end_idx`: Dictionary of floats (0,1).
Ending indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_buffer_sizes`: Dictionary of ints.
Buffer Size to use when filling iterative dataloader with prospective tiles for creation of batches

- `num_channels_available`: Dictionary of ints.
Number of Channels each dataset consists of

- `num_channels_used`: Dictionary of ints.
Number of Channels to use during training, currently no control of choosing modalities, but will cycle through the channels in order

- `dict_in_variables`: Diction of Lists of strings.
Variables corresponding to the different channels in the dataset, used in the dataloader to find corresponding correct values in the default_var_list. Needs to be in the correct order of the raw data files

- `batch_size`: Int.
Per GPU batch size

- `num_workers`: Int.
Number of data loader workers, should be set at 1 for now

- `pin_memory`: Bool.
Variable whether to use pinned memory on GPU for dataloading

- `patch_size`: Int.
Patch Size to use when creating patch Embeddings Sequences for the network input

- `tile_size_[x,y,z]`: Int.
Desired tile size to generate from raw input. If tile_size smaller than raw input files, multiple tiles will be created from each raw data file

- `twoD`: Bool. Variable for indicating two or three dimensionsal input, if False, three-dimensional data will be created from the dataloader. If the raw dataloader is three-dimensional and twoD is set to True, two-dimensional slices will be created from the three-dimensional data by iterating over the final spatial dimension of the data

- `single_channel`: Bool.
Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality

- `return_label`: Bool.
Whether to return back a corresponding label to each tile when loading data

- `auto_load_balancing`: Bool
Whether to allow [Load Balancing](#load-balancing) to be done automatically in the training script. If True, then both both dataset_group_list and batches_per_rank_epoch do not need to specified in the config file.

- `dataset_group_list`: Str of colon separated ints.
How to split available GPUs amongst the available datasets. Run "python utils/load_balance.py [CONFIG_FILE] [NUM_GPUS]" to obtain. See [Load Balancing](#load-balancing) for more details

- `batches_per_rank_epoch`: Dictionary of Ints.
How many batches per rank per epoch for a given dataset. Used to get a full epoch from the Dataset with largest value. If a dataset has less than the maximum. Reuse data to obtain enough data to run until the largest data has been fully trained on. Run "python utils/load_balance.py [CONFIG_FILE] [NUM_GPUS]" to obtain. See [Load Balancing](#load-balancing) for more details

- `tile_overlap`: Float (0,1).
Amount of tile overlapping to use, multiplies tile_size by tile_overlap to determine step size. Use 0.0 for no overlapping

- `use_all_data`: Bool. 
Whether or not to use all data in dataloading. Including if tile size doesn't evenly split images. If tile size splits an image unevenly on last tile of a dimension go from last pixel backwards to get a full tile

- `adaptive_patching`: Bool.
Variable for indicating whether to use adaptive patching. If set to True, Adaptive Patching is done within the dataloader before being fed into a model. See [Adaptive Patching](#adaptive-patching)

- `fixed_length`: Int.
How many adaptive patches used to tokenize the input image. Only used if **adaptive_patching** is set to true

- `separate_channels`: Bool.
Whether or not to separate channels and adaptively patch with different quadtrees/octtrees rather than a single one for all input channels

- `data_par_size`: Int.
The amount of data parallel training ranks being used

- `dataset`: String. 
-Variable for telling dataloader how to handle raw data and how to break up root directories into files within source code (Each dataset potentially needs it's own code to do this depending on the data type and layout of files). See [Datset Integration](#dataset-integration)

- `imagenet_resize`: List of Ints.
-Optional argument specific to the imagenet datset which tells the dataloader what size to resize all images to so that the same input size is used.

## Dataset Integration
For Examples, see the XCT-Diffusion, SST, and S8D branches
1. Name your dataset and use it in place of the dataset option of the config file
2. Write code to process file keys for the different datasets
- Add a new branch to if/else in the process_root_dirs function of the NativePytorchDataModule in `src/UCF_VIT/dataloaders/datamodule.py`, to process datafile paths from each dataset into a corresponding dictionary
3. Write code that uses appropriate iterative dataloader functions from `src/UCF_VIT/dataloaders/dataset.py` to handle the raw data files
- Add a new branch to if/else in the set_iterative_dataloader function of the NativePytorchDataModule class in `src/UCF_VIT/dataloaders/datamodule.py`, using the correct Tile Iterator (ImageBlockDataIter_2D or ImageBlockDataIter_3D) depending on the dimension of your data
4. Write code to appropriately read and process (including normalization) raw data files
- Add a new branch to if/else in the read_process_file function of the FileReader class in `src/UCF_VIT/dataloaders/dataset.py`, using an appropriate python function to read the raw data files depending on the type
5. Write code to appropriately load balance data files across the computing hardware
- Add a new branch to if/else in the process_root_dirs function of `src/UCF_VIT/utils/misc.py` (similar to step 2)
- Add a new bracnh to if/else in the read_process_file function of `src/UCF_VIT/utils/misc.py` (similar to step 4)

## Load Balancing
In order for the dataloader to handle multiple datasets at the same time, the data needs to be spread out amongst the GPUs evenly. In the case where different datasets have different amounts and/or different sizes of images, it's difficult to evenly spread this data amongst the GPUs evenly. We provide example load balancing scripts that for a given setting in a config file determines how the data should be split amongst a given set of N GPUs, in order to evenly balance the data amongst the compute resources. The output from this script gives the necessary information to the dataloader in order to do this in a proper fashion. If you want this load_balancing to be done automatically set `auto_load_balancing` to True in your config file. If you want to do the load balancing manually to check for correct implementation run `python utils/load_balance.py [CONFIG_FILE] [NUM_GPUS]` and use the output from this script to add to the load balancing portion of the config file.

## Parallelism Modes
All of these architectures exist in 2 independent sub-folders, simple and fsdp, for which we separate the network architecture code into what we call modes. The choice of mode to be used will depend on the types of advanced parallelism and computing techniques needed for the model being trained.  The first `src/UCF_VIT/simple`, provides a simplified version for training in Distributed Data Parallel (DDP) fashion only. The second `src/UCF_VIT/fsdp`, provides a more complex version with different parallel training techniques. This includes options for training with a combination of Sharded Data Parallelism , DDP, and Tensor Parallelism. These parallelisms are all integrated via Hybrid Sharded Tensor-Data Parallelism (Hybrid-STOP) from [6,7]. Both modes can be used with the same data loading module and load balancing scripts provided. While the training done within the simple mode can be done with the correct choice of options in the fsdp mode, the purpose of keeping the simple mode is 1) to provide an entry point for new users and developers to add new architectures without the intricacies of the advanced features and 2) to provide a simple reference point to compare with when new innovations are added in order to test how they interact with the more complex parallelism methods.

Thus far the examples we have given have only used the **simple** DDP mode. Minimal changes are needed to run with the FSDP mode

### Building Blocks
The main building blocks for the VIT based archictectures are in the **Attention** and **Feed-forward** functions, provided in the Attention class and MLP class in `src/UCF_VIT/simple/building_blocks.py` and `src/UCF_VIT/fsdp/building_blocks.py`. We ask that you use these functions as is and do not modify them, as these common building blocks will be used across the different network architectures.

## Training Scripts
We provide several training scripts. These include all of the necessary things for running the main training loop, including utilities such as checkpoint loading and saving. We leave it to the user to implement their own validation and testing routines, based off the existing training examples, in order to more closely fit their needs. Training scripts are provided for each of the training architectures for the simple mode. To convert these scripts to use fsdp mode, look at the code changes made to go from `training_scripts/train_masked_simple.py` to `training_scripts/train_masked_fsdp.py`

### Usage
1. If using a new dataset, modify dataloader module accordingly
- Follow [Dataset Integration](#dataset-integration)

2. Find training script of interest from training_scripts/

3. Modify training script for particular use case. (Adding validation, testing, inferencing, etc. as needed)

4. Create/Modify config file for your training

5. Modify Launch Script
- Change project allocation to one you have access to `#SBATCH -A PROJXXX`.
- Set number of nodes you want to run with `#SBATCH --nodes=N`

6. Run Load Balancing Script
- `python utils/preprocess_load_balancing.py [CONFIG_FILE] [NUM_GPUS]`

7. Modify Config File with the output from load balancing output
- dataset_group_list
- batches_per_rank_epoch

8. Launch job `sbatch launch/[DATASET]/train_[MODEL]_[MODE].sh`
- [DATASET] is the particular dataset you want to use. The examples use (imagenet or basic_ct)
- [MODEL] is the type of model you want to use chose from those in [Model Architectures](#Model-Architectures)
- [MODE] is the training mode you want to use. There are two options (simple and fsdp). See [Parallelism Modes](#Parallelism-Modes) for a more detailed description

### Config Arguments
We store the arguments for each individual run in a yaml file. This config file holds all of the arguments for defining the specific training, dataloading, parallelism, and checkpointing options

1. Trainer
- max_epochs: Max number of epochs to train
- data_type: Data type to be used during training
- checkpoint_path: Directory to save checkpoint to
- checkpoint_filename: File prefix to save checkpoint to, "even" or "odd" will be appended to the for checkpoint file robustness
- checkpoint_filename_for_loading: Prefix of file to use when starting from checkpoint, append with "even" or "odd"
- resume_from_checkpoint: Whether to start from checkpoint or from scratch

2. Parallelism (only used in fsdp mode and load balancing script)
- fsdp_size: Number of Fully Sharded Data Parallel ranks to use, for sharding model states
- simple_ddp_size: Number of Data Parallel ranks to use, for distributing different data to ranks
- tensor_par_size: Number of Tensor Parallel ranks to use, for distributing tensor across multiple ranks
- seq_par_size: Number of Sequence Parallel ranks to use, for distributing input sequence across multiple ranks (NOT IMPLEMENTED YET)

3. Model Optimizer and Scheduler
- lr: Initial learning rate for optimizer
- beta_1,beta_2: Beta coefficients for Adam optimizer
- warmup_steps: Number of warmup steps for learnining rate scheduler
- max_steps: Maximum number of warmup steps for learning rate scheduler
- warmup_start_lr: Learning rate to use for warm up steps

## Example Datasets
### Imagenet
Download data at `https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php`

Directory consists of 1000 sub-folders of 2D JPEG images, each folder corresponding to different classification labels. In order to use data in this format in data distributed fashion across N GPUs, we make individual datasets within the dataloader code, each with 1000/N of these subfolders. Since ImageNet has images of all different sizes, all input images are resized to a standard size, chosen by the imagenet_resize argument in the config file.
### Basic_CT
Download data at

Directory consists of 3D synthetic CT images of concrete including corresponding labels for segmentation.

## Citations
### [1]
```bibtex
@inproceedings{wang2024orbit,
  title={Orbit: Oak ridge base foundation model for earth system predictability},
  author={Wang, Xiao and Liu, Siyan and Tsaris, Aristeidis and Choi, Jong-Youl and Aji, Ashwin M and Fan, Ming and Zhang, Wei and Yin, Junqi and Ashfaq, Moetasim and Lu, Dan and others},
  booktitle={SC24: International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--11},
  year={2024},
  organization={IEEE}
}
```
