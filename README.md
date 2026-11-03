## Table of Contents
- [UCF-VIT](#ucf-vit)
- [Motivation](#motivation)
- [Install](#install)
- [Features](#features)
- [Model Architectures](#model-architectures)
1. [Vision Transformer](#vision-transformer-vit)
2. [Masked Autoencoder](#masked-autoencoder-mae)
3. [Unet Transformer](#unet-transformer-unetr)
4. [Symmetric Adaptive Patching](#symmetric-adaptive-patching-sap)
5. [Diffusion Vision Transformer](#diffusion-vision-transformer-diffusionvit)
- [Dataloader](#dataloader)
- [Dataset Integration](#dataset-integration)
- [Normalization](#normalization)
- [Load Balancing](#load-balancing)
- [Parallelism Modes](#parallelism-modes)
- [Training Scripts](#training-scripts)
- [Config Files](#config-files)
- [Datasets](#datasets)

## UCF-VIT
This repository provides a unified coding framework for scalable Vision Transformer (ViT) model training, designed to support both extreme-scale model training on leadership-class supercomputers and efficient deployment on smaller systems such as DGX nodes. The framework integrates several cutting-edge innovations in efficient computing, including optimized attention mechanisms, advanced parallelism strategies, and mixed-precision acceleration, enabling the training of vision models with hundreds of billions of parameters and achieving sustained ExaFLOP-scale performance on systems like the Frontier supercomputer.

At the same time, the codebase is modular and flexible, allowing researchers and practitioners to run the same models on smaller compute environments without sacrificing performance or usability. This unified approach lowers the barrier to entry for next-generation large-scale vision model research while ensuring reproducibility and scalability across diverse computing platforms.

The repository is open-source and maintained on GitHub to foster collaboration and accelerate innovation in ViT research at scale.

## Motivation
The purpose of this framework is to provide capabilities to train large scale ViT models with state-of-the-art efficient computing techniques while still being usable and adaptable for a general audience looking to train on various different types of data and downstream tasks. While frameworks for scalable computing with transformers currently exist they are 1) mostly focused on training with 1D text data with generative tasks, 2) provide inefficient implementations for scaling to more accelerator resources when targetting large models and inputs, and 3) are difficult for users to adapt the code to their desired data and downstream tasks.

### Target Audience
The target audience for this framework is users who have either very large datasets and/or datasets with large volume data. In the case of users with very large datasets, transformer models often need to scale to large parameter counts (on the order of tens to hundreds of billions of parameters) in order to accurately capture features across the entire training dataset. In the case of users with large volume data, typically resulting from high-resolution scientific simulations or image acquistion, vision transfomer models either become intractable due to resultant long input sequences being combined with the quadratic scaling of the attention mechanism in the transformer architecture, or become inaccurate from the necessary coarsening of the patch embeddings in order to even be able to train with the data. While the target audience are users that are eventually looking to train models with 1000's of GPUs, the framework is also built such that it can be used on smaller scale systems in order to test and develop new methodologies. 

### Why use UCF-VIT?
Training large ViT models at scale remains a major technical challenge in both AI research and deployment, with most existing codebases narrowly optimized for either small-scale experiments or vendor-specific hardware, creating barriers to portability, scalability, and reproducibility. A unified coding framework is essential to streamline development, maximize resource efficiency, and enable seamless scaling from small machines to the world’s largest supercomputers. This repository provides such a unified solution—offering robust support for both NVIDIA and AMD GPUs, across systems ranging from single-node DGX machines to leadership-class clusters like the Frontier supercomputer. The framework incorporates our in-house hybrid-op extreme-scale parallel computing technique, which won the HPCWire Supercomputing Achievement Award, to unlock scalable and efficient ViT model training. It also introduces our adaptive patching method to reduce the computational complexity of ViT attention, alongside variable aggregation for handling large-channel ViTs. Additionally, it integrates widely used techniques such as lower-precision training, layer wrapping, fused attention, and FlashAttention for further efficiency gains. The framework is compatible with a variety of ViT architectural variants, making it a powerful, flexible tool for researchers and practitioners aiming to push the frontier of scalable vision model development and deployment.

# Install
Installation instruction are provide for running on systems with both AMD and NVIDIA GPU hardware. Instructions are included for systems ranging from a local DGX cluster to the Frontier Supercomputer.

Every platform-independent package we depend on (`monai`, `nibabel`, `torchdata`, `einops`, `opencv-python-headless`, `matplotlib`, `scipy`, `scikit-image`, `SimpleITK`, `timm`) is declared in `pyproject.toml` and gets installed automatically by `pip install -e .`. `torch` and `xformers` are deliberately left out of `pyproject.toml` since the correct build depends on your GPU vendor (ROCm for AMD, CUDA for NVIDIA) -- install those manually first with the platform-specific commands below, then run `pip install -e .` to pull in everything else.

## Systems with AMD GPUs
There are two options available for creating software environments for systems with AMD GPUs 1) creating Conda environment from scratch or 2) using an apptainer container (the installation instructions for this are currently limited to use on the Frontier supercomputer). Creating a Conda environment from scratch is recommended as the Apptainer containers currently only work in limited scenarios due to missing ROCM packages in the base apptainer image.

### Conda
Create Conda Environment from Scratch. Example below uses similar options from the corresponding Apptainer definition files
```
PYTHON_VERSION=3.11
conda create -n UCF-rocm7.13 python=${PYTHON_VERSION} -y
conda activate UCF-rocm7.13

python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx90a/ \
  "torch==2.10.0+rocm7.13.0" \
  "torchvision==0.25.0+rocm7.13.0" \
  "torchaudio==2.10.0+rocm7.13.0"
pip3 install -U xformers --index-url https://download.pytorch.org/whl/rocm7.1

#If your system has an existing MPI installed use the proper mpi4py installation for your sytsem
#Default install mpi4py 
MPI_DIST=mpich
conda install -c conda-forge mpi4py ${MPI_DIST}

#mpi4py install on Frontier
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

cd UCF-VIT
pip install -e .
```

#### Optional: `cupy` for `region_backend="cupyx"`
`cupy` is not installed by the steps above -- it's an opt-in dependency, only needed if you want `UCF_VIT.model.gpu_adaptive_patching.GPUPatchify2D`/`GPUPatchify3D`'s `region_backend="cupyx"` option (GPU-native connected-components labeling, vs. the default `"scipy"`; currently test-only, see `launch/tests/run_cupyx_smoke.sh`). Build from source against your ROCm install:
```
conda activate UCF-rocm7.13
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/7.13
export ROCM_HOME=${ROCM_PATH}

export CUPY_INSTALL_USE_HIP=1
export HCC_AMDGPU_TARGET=gfx90a   # MI250X's ISA

pip install cupy --no-cache-dir -v
```
`ROCM_HOME` also needs to be set at runtime (`module load rocm/7.13; export ROCM_HOME=${ROCM_PATH}` in any job script), and `cupy`'s JIT kernel cache should be redirected off `$HOME` to node-local scratch:
```
export CUPY_CACHE_DIR=/tmp/$JOBID/cupy_kernel_cache
mkdir -p $CUPY_CACHE_DIR
```
See `launch/tests/run_cupyx_smoke.sh` for a working example job.

### Apptainer (on Frontier)
Use Apptainer container definition files (Only use this on Frontier)
```
cd apptainer
apptainer build frontier-ubuntu-gnu-rocm624.sif frontier-ubuntu-gnu-rocm624.def
apptainer build frontier-ubuntu-gnu-rocm624-vit.sif frontier-ubuntu-gnu-rocm624-vit.def
mkdir lib/8.1.31
cd lib/8.1.31
ln -s $CRAY_MPICH_DIR/lib/libfmpich.so libmpicxx.so
```


## Local DGX System
To run on a local NVIDIA system we provide instructions for creating a Conda environment from scratch

```
conda create -n vit python=3.11 -y
conda activate vit
CUDA_DRIVER=cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_DRIVER}
pip install xformers

#If your system has an existing MPI installed use the proper mpi4py installation for your sytsem
#Default install mpi4py 
MPI_DIST=mpich
conda install -c conda-forge mpi4py ${MPI_DIST}

cd UCF-VIT
pip install -e .
```

To run on a local DGX system using one of our training scripts invoke the following command `mpirun -n [NUM_GPUS] python -u [TRAINING_SCRIPT] [CONFIG_FILE] MPI` or `python [TRAINING_SCRIPT] [CONFIG_FILE] MPI` to run on a single GPU. If you are running on a shared resource machine and you want to use a subset of the available GPUS be sure to set the visible devices before running, e.g. `os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'`.

# Features
In this codebase, we provide various Advanced Parallelism & Efficient Computing techniques that we have used to explore larger model and input sizes with VITs than has been previously possible. These techniques range from novel methods to the utilization of several techniques provided by external libraries that are integrated with these novel techniques.

## Hybrid-OP 
Hybrid Sharded Tensor-Data Orthogonal Parallelism (Hybrid-OP) [[1]](#1),[[11]](#11) is a novel parallelism algorithm that combines tensor parallelism and Fully Sharded Data Parallelism (FSDP). It avoids the peak memory use problem in FSDP by working in coordination with Tensor Parallel ranks.
### Usage
Hybrid-OP is driven entirely by three config values -- `parallelism.fsdp_size`, `parallelism.simple_ddp_size`, and `parallelism.tensor_par_size`. The snippet below is excerpted from `get_model` (`src/UCF_VIT/model/utils.py`), which every training/eval script uses to build its model:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from UCF_VIT.utils.misc import init_par_groups
from UCF_VIT.model.arch import MAE  # or VIT/UNETR/SAP/DiffusionVIT
from UCF_VIT.utils.fused_attn import FusedAttn

# Assume we have 8 GPUs for training. Change these in any way desired such
# that fsdp_size * simple_ddp_size * tensor_par_size == world_size.
fsdp_size = 2
simple_ddp_size = 2
tensor_par_size = 2
data_par_size = fsdp_size * simple_ddp_size
assert (data_par_size * tensor_par_size) == world_size, "DATA_PAR_SIZE * TENSOR_PAR_SIZE must equal world_size"

ddp_group, tensor_par_group, data_seq_ort_group, fsdp_group, simple_ddp_group = init_par_groups(
    world_rank=world_rank, data_par_size=data_par_size, tensor_par_size=tensor_par_size,
    fsdp_size=fsdp_size, simple_ddp_size=simple_ddp_size,
)

model = MAE(
    img_size=[256, 256], patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
    linear_decoder=False, decoder_depth=8, decoder_embed_dim=512, decoder_num_heads=8, decoder_mlp_ratio=4,
    mlp_ratio=4, drop_path_rate=0.1, mask_ratio=0.1, twoD=True,
    default_vars=["red", "green", "blue"], use_varemb=False,
    adaptive_patching=False, fixed_length=None,
    FusedAttn_option=FusedAttn.DEFAULT,
    tensor_par_size=tensor_par_size, tensor_par_group=tensor_par_group,
    class_token=False, weight_init='skip',
).to(device)

if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group=(fsdp_group, simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, forward_prefetch=True, limit_all_gathers=False)
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group=fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, forward_prefetch=True, limit_all_gathers=False)
else:
    model = FSDP(model, device_id=local_rank, process_group=simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, forward_prefetch=True, limit_all_gathers=False)
```

For the full training loop (distributed init, dataloader construction, etc.), see `training_scripts/train.py` directly.

### Implementation Details 
Tensor parallelism is enacted over the attention and MLP calculations (the costliest components of transformer architectures), implemented in `src/UCF_VIT/model/building_blocks.py`'s `Attention`/`Mlp` classes. `self.tensor_par_group` is only meaningful when `tensor_par_size > 1`; every architecture always receives it (see `get_model`), but every sharding guard checks `self.tensor_par_size > 1` first. Here `self.blocks` is the `nn.Sequential` of transformer blocks shared by every architecture:

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
The ability to implement lower precision support is facilitated by using a MixedPrecision Policy which is passed as an argument to the FSDP wrapper. Using lower precision data types such as bfloat16 reduces storage size to allow for large model sizes and increased throughput due to the faster computing that is possible with lower precision. In the real code this is driven by `trainer.data_type` in the config (`"float32"` or `"bfloat16"`) -- see `get_model` in `src/UCF_VIT/model/utils.py`.

### Usage 
Add the following code and replace the FSDP Wrapper calls in the [Hybrid-OP example](#hybrid-op) to apply mixed precision training (specifically bfloat16 in this case).
```python
from torch.distributed.fsdp import MixedPrecision

precision_dt = torch.bfloat16
mixedPrecisionPolicy = MixedPrecision(
    param_dtype=precision_dt,
    reduce_dtype=precision_dt,
    buffer_dtype=precision_dt,
)

if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group=(fsdp_group, simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers=False)
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group=fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers=False)
else:
    model = FSDP(model, device_id=local_rank, process_group=simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, mixed_precision=mixedPrecisionPolicy, forward_prefetch=True, limit_all_gathers=False)
```

## Layer Wrapping
The ability to use layer wrapping is facilitated by using a custom autowrap policy which is passed as an argument to the FSDP wrapper. The purpose of this wrapping is to control how and when the parameters of each layer are sharded and gathered during the forward and backward passes. By wrapping layers in the `Block` submodule (which contains the transformer computational layers) peak GPU memory usage is reduced, and communication and computations have improved overlapping. This is exactly what `get_model` (`src/UCF_VIT/model/utils.py`) does for every architecture:
### Usage 
```python
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Sequential
from UCF_VIT.model.building_blocks import Block

my_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Block, Sequential},
)

if fsdp_size > 1 and simple_ddp_size > 1:
    model = FSDP(model, device_id=local_rank, process_group=(fsdp_group, simple_ddp_group), sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.HYBRID_SHARD, auto_wrap_policy=my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers=False)
elif fsdp_size > 1 and simple_ddp_size == 1:
    model = FSDP(model, device_id=local_rank, process_group=fsdp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD, auto_wrap_policy=my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers=False)
else:
    model = FSDP(model, device_id=local_rank, process_group=simple_ddp_group, sync_module_states=True, sharding_strategy=dist.fsdp.ShardingStrategy.NO_SHARD, auto_wrap_policy=my_auto_wrap_policy, forward_prefetch=True, limit_all_gathers=False)
```

## Activation Checkpointing
The ability to invoke activation checkpointing to the model is provided through FSDP's `apply_activation_checkpointing` function. In our cases we apply activation checkpointing only to the `Block` submodule containing the transformer computational layers to reduce the memory storage required by that component of the model training. With activation checkpointing, the gradients are no longer stored for that component to be used during the backward pass of optimization. Rather the gradients are recomputed from previously stored states when necessary during the backwards pass, significantly reducing the amount of GPU memory used during runtime.
### Usage
Add the following code after the FSDP Wrapper (again, this is the real code from `get_model`):
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   apply_activation_checkpointing,
)
from UCF_VIT.model.building_blocks import Block

check_fn = lambda submodule: isinstance(submodule, Block)
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
)
```
## Fused Attention (Flash Attention)
In order to use computationally efficient implementations of the fused multi-head attention (FMHA) kernel on different GPU accelerated hardware, we provide several different options. One option is to use the native implementation of FlashAttention [[4]](#4),[[5]](#5) provided through Pytorch. Otherwise we can use flash attention implementations provided by the XFormers [[2]](#2) libary. The XFormers library provides an interface to memory efficient implementations of FMHA for various different hardwares. On AMD GPUs we use the ComposableKernel (CK)[[3]](#3) implementation of FMHA, and on NVIDIA GPUs we again use the FlashAttention implementation of FMHA, similar to the implementation provided through Pytorch. The XFormer implementations only have compatibility with bfloat16 operations, thus with float32 data types we use the default torch FMHA implementation.

### Usage 
We provide an enumerated class with which to choose the FMHA implementation to use in the architecture class.
```python
import torch
from UCF_VIT.model.arch import VIT
from UCF_VIT.utils.fused_attn import FusedAttn

# Choose from these options
FusedAttn_option = FusedAttn.DEFAULT  # Default torch implementation of FMHA, compatible with float32 and bfloat16
# FusedAttn_option = FusedAttn.FLASH  # FlashAttention implementation of FMHA, bfloat16 + NVIDIA GPUs
# FusedAttn_option = FusedAttn.CK     # ComposableKernel implementation of FMHA, bfloat16 + AMD GPUs
# FusedAttn_option = FusedAttn.NONE   # Basic Python implementation of FMHA

model = VIT(
    img_size=[256, 256],
    patch_size=16,
    num_classes=1000,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_path_rate=0.1,
    drop_rate=0.1,
    twoD=True,  # set False if 3D
    use_varemb=False,
    default_vars=["red", "green", "blue"],
    adaptive_patching=False,
    fixed_length=None,
    FusedAttn_option=FusedAttn_option,
)
```

## Adaptive Patching
A recent innovation for efficient computing with VITs that we have implemented within this codebase is adaptive patching [[6]](#6). Traditionally in VITs, input images (or data) are separated into groups of individual nonoverlapping pixels (or individual datapoints) called tokens of size patch_size x patch_size in 2D (or patch_size x patch_size x patch_size in 3D), these tokens are then flattened to be used as input to the ViT. When input image sizes become too large, the sequence length of tokens being fed into the network can become intractably large except for at very large patch sizes which can hinder the ability to train more accurate models. Often it is the case that there are large portions of regions of input that are largely homogenous, thus it is wasteful to consider all of the input tokens equally. Adaptive patching is an approach inspired by Adaptive Mesh Refinement (AMR) techniques in which a tree is used to break down input data into adaptively sized regions based on some quantity such as the magnitude of the gradient to indicate how much variation is in each spatial region. In our case we use a Canny Edge detection method (or, alternatively, a variance-based score -- see below) to break down the images with a quadtree (2D) or octree (3D) into smaller regions adaptively depending on the amount of edges in certain regions of the image. To control the amount of regions the image is broken into we use the integer variable `fixed_length`, to tell adaptive patching to stop splitting the image. Each adaptively sized region is then resized (via `interp_size`, not `patch_size`) into tokens of size `interp_size` x `interp_size` (or `interp_size`^3 in 3D) to be in a form suitable for being input into a VIT. This approach can drastically reduce the sequence length of tokens and consequently significantly reduce the amount of compute time required to feed through the network. **Not currently supported for `DiffusionVIT`** -- `parse.py` asserts `ap.do_ap` is `False` whenever `model.type: DiffusionVIT`.

### Usage
Adaptive patching can run in one of two places:
- **Dataloader-side** (`ap.do_ap: True`, `ap.do_gpu_ap: False`, the original/default): the dataloader itself returns an already-patchified sequence instead of a raw image -- `model.forward` receives that sequence directly rather than a dense image tensor.
- **GPU-side** (`ap.do_gpu_ap: True`): the dataloader still returns the full raw image, and patchification (`UCF_VIT.model.gpu_adaptive_patching.GPUPatchify2D`/`GPUPatchify3D`) runs on-device as the first step of the model's own forward pass instead -- useful to avoid doing this CPU-bound work in a dataloader worker.

Either way, `fixed_length` must be chosen such that `3n+1 = fixed_length` (2D) or `7n+1 = fixed_length` (3D) for some integer `n`, to satisfy the underlying quadtree/octree's leaf-count math, and each spatial dimension of the input must be a power of 2 (32, 64, ...). `interp_size` (not `patch_size`) is required whenever `adaptive_patching=True` -- see `VIT.effective_patch_size`.

## Variable Aggregation
Another advanced technique that we have introduced into this codebase particularly for the purpose of foundational model training is the capability to incorporate variable aggregation [[12]](#12). Variable aggregation is a technique where instead of tokenizing multi-channel inputs all at once and transforming them into the latent embedding dimension space, each channel is tokenized individually and transformed into the latent embedding dimension space based on the type of channel data, given some direction via the `variables` argument in the forward pass. Each of these channel embedding vectors are then fed through an additional attention mechanism to compress all of the input into a single dimension, a process we call variable aggregation. The reason it is important that these input channels be tokenized and embedded separately is because it allows the flexibility to use a pre-trained foundational model in a flexible manner. Variable aggregation allows for the ability to use different types of data during training, e.g. data that does not contain all of the input channels contained in other datasets that the model is being trained with.

### Usage 
In order to use variable aggregation set `use_varemb` to True. In the case where `use_varemb=False` multi-channel tokenization will be performed and thus any further training or finetuning with that model will require data to match the specific number of channels as the original data used with that model. Variable aggregation is controlled by sending a list of variable identifiers to the forward pass of the model architecture, i.e. `["red","green","blue"]` for the case of RGB images. This list of variables must correspond correctly with the order that the data is formatted in, and this process is facilitated through our [dataloader](#dataloader) via passing in the identifier list to the `dict_in_variables` argument of the config file. `default_vars` controls the types of input channels that the model will allow for ingestion. Thus every variable in `dict_in_variables` must be in `default_vars`, however not every input channel is necessary when passing through the model.

# Model Architectures
Currently we provide 5 different model architectures **(VIT, MAE, UNETR, SAP, DiffusionVIT)**, all of which use the same `VIT` encoder (`src/UCF_VIT/model/arch.py`), but a different decoder architecture depending on the task being trained. All code for the different architectures inherit the encoder from the `VIT` architecture class in order to facilitate using the same encoder. In the following sections we provide working examples for executing a forward pass with each of these architectures that can be run on a single CPU. For real multi-GPU training runs, use `training_scripts/train.py` (generic across every architecture, driven entirely by a config file) rather than hand-assembling the pieces below yourself -- see [Training Scripts](#training-scripts).

## Vision Transformer (VIT)
VIT based on [[7]](#7). Code slimmed down from (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L425) to only contain basic options for VIT Training and adapted to integrate our [innovations](#features). Task: Image Classification. Input: Image or Image Tile (A tile is a contiguous subset of a full image).

### Usage
```python
import torch
from UCF_VIT.model.arch import VIT
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
    default_vars = ["red", "green", "blue"],
    adaptive_patching = False,
    fixed_length = None,
    FusedAttn_option = FusedAttn.DEFAULT,
)

img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # torch.Size([1, 1000])
```

### Parameters

- `img_size`: Int, Tuple[int,int], or Tuple[int,int,int].  
Image size. If a single int is given the input image is assumed to be the same in each dimension and the dimension of the image is set via the **twoD** variable. If **adaptive_patching** is set to True this variable is ignored, since input has already been patched in an adaptive fashion in the dataloader.

- `patch_size`: Int.  
Size of patches. `image_size` must be divisible by `patch_size` if **adaptive_patching** is set to True.

- `num_classes`: Int.  
Number of classes to classify.

- `in_chans`: Int.  
Number of input channels contained in each input image. E.g, a JPEG image has 3 color channels [R,G,B] at each pixel

- `embed_dim`: Int.  
Embedding dimension for Transformer Inputs

- `depth`: Int.
Number of Transformer blocks.

- `num_heads`: Int.  
Number of heads in Multi-head Attention layer.

- `mlp_ratio`: Int.
Ratio of MLP hidden dimension to embedding dimension, used to set the dimension of the MLP (FeedForward) layer.

- `drop_path_rate`: Float (0,1).
Stochastic depth dropout rate for dropping random layers during training

- `drop_rate`: Float (0,1).
Dropout rate for classification head

- `twoD`: Bool.
Variable for indicating two or three dimensionsal input, if False, three dimensional input. Needed to do correct tokenizing. Used in coordination with the [dataloader module](#dataloader) to correctly set up the data for the given architecture

- `use_varemb`: Bool.
Variable for indicating whether to use variable embedding tokens. When using variable embedding tokens each input channel is tokenized separately. In order to feed these tokens into the model, the separate variable tokens are fed through [variable aggregation](#variable-aggregation) to compress multiple input into a single aggregated input channel via an attention mechanism

- `default_vars`: List[str].
List of different potential modalities to be used as input. When **use_varemb** is set to true, this list contains the available input channels.

- `adaptive_patching`: Bool.
Variable for indicating whether to use adaptive patching. See [Adaptive Patching](#adaptive-patching) for more details. 

- `fixed_length`: Int.
How many adaptive patches used to tokenize the input image. Only used if **adaptive_patching** is set to true

- `FusedAttn_option`: [FusedAttn.FLASH, FusedAttn.CK, FusedAttn.DEFAULT, FusedAttn.NONE]
Which option to use for fused attention See [Fused Attention](#Fused-Attention-via-XFormers) for more details.

## Masked Autoencoder (MAE)
Masked Autoencoder pre-training based on [[8]](#8). Task: Masked Image Prediction. Input: Image or Image Tile

### Usage
```python
import torch
from UCF_VIT.model.arch import MAE
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
        decoder_mlp_ratio=4,
        default_vars=["red", "green", "blue"],
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds, mask = model.forward(img, variables) # preds: torch.Size([1, 256, 768]) -> (batch_size, num_tokens_x * num_tokens_y, patch_size * patch_size * num_channels); mask: torch.Size([1, 256])

#Move masked image prediction from patched space back to original image space
pred_img = unpatchify(preds, img, 16, True) # torch.Size([1, 3, 256, 256])
```

### Parameters (if not listed here see descriptions in the architectures above)

- `linear_decoder`: Bool.
Variable to indicate whether to use a linear decoder. If False, a Transformer decoder is used to predict the mask prediction output

- `decoder_depth`: Int.
Number of Transformer blocks to use in the decoder. Not used if **linear_decoder** is set to True

- `decoder_embed_dim`: Int.  
Embedding dimension for Inputs to the Transformer decoder. Not used if **linear_decoder** is set to True

- `decoder_num_heads`: Int.  
Number of heads in Multi-head Attention layer for the Transformer decoder. Not used if **linear_decoder** is set to True

- `decoder_mlp_ratio`: Int.
Ratio of MLP hidden dimension to embedding dimension, used to set the dimension of the MLP (FeedForward) layer in the Transfomer decoder. Not used if **linear_decoder** is set to True

- `mask_ratio`: Float in (0, 1).
Amount of tokens to mask out in the Transformer encoder

- `class_token`: Bool.
Whether to append a class token to the tokenized data. Set to false unless using VIT

- `weight_init`: Str from ['' or 'skip'].
Whether to skip the weight_init in the VIT parent class. Set to 'skip' unless using VIT

## UNet Transformer (UNETR)
Image segmentation architecture based on [[9]](#9). Task: Image Segmentation. Input: Image or Image Tile

### Usage
```python
import torch
from UCF_VIT.model.arch import UNETR
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
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # torch.Size([1, 4, 256, 256]) -> (batch_size, num_classes, tile_size_x, tile_size_y)
```

### Parameters (if not listed here see descriptions in the architectures above)

- `num_classes`: int.  
Number of classes to predict from at each image pixel.

- `linear_decoder`: bool.
Variable to indicate whether to use a linear decoder. If False, a convolutional decoder is used to predict the class of each pixel

- `skip_connection`: bool.
Variable to indicate whether to use skip connection in the convolutional decoder. The skip connection uses intermediate output from the Trasnformer encoder blocks

- `feature_size`: int.
Variable to set how the embedding features are expanded through the UNETR convolutional blocks

## Symmetric Adaptive Patching (SAP)
Image segmentation architecture for adaptively patched input based on [[6]](#6). Task: Image Segmentation. Input: Adaptively Patching Image or Image Tile

### Usage
Adaptive patching needs `interp_size` (not `patch_size`, and no `sqrt_len` -- see [Adaptive Patching](#adaptive-patching)), and `Patchify` (the dataloader-side transform that actually does the patching) takes `interp_size`, not `patch_size`, either:
```python
import torch
import numpy as np
from UCF_VIT.model.arch import SAP
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.dataloaders.transform import Patchify

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
        use_varemb=False,
        adaptive_patching=True,
        interp_size=16,
        fixed_length=64,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = np.random.randn(1, 3, 256, 256).astype(np.float32) # (batch_size, num_channels, tile_size_x, tile_size_y)
patchify = Patchify(fixed_length=64, interp_size=16, num_channels=3, dataset="imagenet")
seq_img, seq_size, seq_pos, qdt = patchify(np.moveaxis(np.squeeze(img), 0, -1)) # seq_img: (num_channels, fixed_length, interp_size*interp_size) == (3, 64, 256)
adaptive_patch_img = torch.from_numpy(np.expand_dims(seq_img, axis=0).astype(np.float32)) # Add back batch dimension

variables = ["red", "green", "blue"]

preds = model.forward(adaptive_patch_img, variables) # torch.Size([1, 64, 4, 16, 16]) -> (batch_size, fixed_length, num_classes, interp_size, interp_size); one prediction per adaptive token, not a reassembled dense image -- see mask_head's own docstring for why
```

## Diffusion Vision Transformer (DiffusionVIT)
Diffusion model training based on [[10]](#10). Task: Generate Image via noise that matches distribution of data trained on. Input: Noise

### Usage
`DiffusionVIT`'s per-block timestep conditioning (`Block_diffusion`) is a different block class from every other architecture's plain `Block` -- it must be passed explicitly as `block_fn`, unlike the other architectures above which all use the default:
```python
import torch
from UCF_VIT.model.arch import DiffusionVIT
from UCF_VIT.model.building_blocks import Block_diffusion
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler
from UCF_VIT.utils.misc import unpatchify

num_time_steps = 1000
model = DiffusionVIT(
        img_size=[256,256],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_depth=8,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.1,
        linear_decoder=True,
        twoD=True,
        default_vars=["red", "green", "blue"],
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        num_time_steps=num_time_steps,
        block_fn=Block_diffusion,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

# One forward-noising step, matching training_scripts/train.py's own DiffusionVIT
# forward_step formula -- this predicts the noise for ONE step, it does not
# generate a sample. For the real reverse-diffusion sampling loop (pure noise
# -> a generated sample), see training_scripts/inference_diffusion.py below.
t = torch.randint(0,num_time_steps,(1,))
e = torch.randn_like(img, requires_grad=False)
ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
a = ddpm_scheduler.alpha[t].view(1,1,1,1)
data = (torch.sqrt(a)*img) + (torch.sqrt(1-a)*e)
output = model.forward(data, t, variables) # torch.Size([1, 256, 768]) -> (batch_size, num_tokens_x * num_tokens_y, patch_size * patch_size * num_channels)

#Move predicted noise from patched space back to original image space
output_img = unpatchify(output, data, 16, True) # torch.Size([1, 3, 256, 256])
```

### Generating samples
The snippet above only predicts noise for a single step -- it's the same thing training does, not generation. To actually generate a sample (pure noise -> a plausible output, via the full reverse-diffusion chain), use `training_scripts/inference_diffusion.py`, which loads a trained checkpoint through the same `get_model`/`parse_config` pipeline every other script uses and runs `UCF_VIT.ddpm.ddpm.diffusion_reverse_step` `num_time_steps` times:
```
cd launch/eval
sbatch inference_diffusion.sh ../../configs/basic_ct/diffusion/base_config.yaml --num_samples 4 --seed 1234
```
See that script's own docstring for the full set of flags (`--dict_key`, `--variance_mode`, `--output_dir`, etc.).

### Parameters (if not listed here see descriptions in the architectures above)

- `num_time_steps`: int.
Number of time steps in the diffusion process

- `block_fn`: Type[nn.Module], must be `UCF_VIT.model.building_blocks.Block_diffusion` for this architecture.
Adds a per-block sinusoidal timestep embedding re-injected before every block's attention, instead of `VIT`'s plain `Block`. `get_model` (`src/UCF_VIT/model/utils.py`) always passes this automatically for `model.type: DiffusionVIT` -- only needed explicitly when constructing the architecture class directly, as above.

## Dataloader
The dataloader we provide is a custom native pytorch iterative dataloader. For simplicity, we assume that we are receiving unprocessed data files and we leave it to the user to normalize the data properly within in the dataloader module for training. The reasons for making this assumption of unprocessed data files as input is 1) we intend this repo to be used on very large datasets, thus preprocessing and storing all of the data before training can quickly take up a massive amount of storage, and 2) it removes the need for further data preprocessing scripts to be included in this repo. If performing preprocessing during the dataloading phase is too computationally intensive, we recommend doing it offline and properly storing it in a manner that the dataloader module can handle. 

The dataloader is built in a fashion such that it can handle multiple different dataset directories at the same time. A dataset directory contains one or more data file (with all data files having the same dimension or able to be resized so that they have the same dimension). The purpose of being able to handle multiple dataset directories is 1) it provides flexible training where you can easily remove and add different datasets for the purposes of running experiments and 2) it allows for the integration of identifying properties from the different datasets that can potentially used for improved learning via our advanced features. For instance, with data that has multiple channels, e.g. images with (R,G,B) channels, we are able to pass along the information on what variable the channel is from and use that information during network training. We then could utilize [variable aggregration](#variable-aggregation) to tokenize each channel separately.

This dataloader provides the flexibility to add a plethora of different options for customizing how the data is broken up for training. Since we are using a VIT, at least 2D data is expected. However, we have capability for both 2D and 3D spatial data currently. If desired, we have the utilities implemented to break given data into smaller tiled chunks. Also, we have a number of different options for how to tile this data, e.g. tile overlapping.

### Usage
In practice, every real script (`train.py`/`val.py`/`test.py`/`inference_diffusion.py`) builds this from a parsed YAML config rather than hand-specifying every kwarg -- see [Config Files](#config-files). This is the real construction (from `training_scripts/train.py`), shown directly rather than a hand-maintained duplicate:
```python
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
from UCF_VIT.utils.misc import calculate_load_balancing_on_the_fly
from UCF_VIT.parse import parse_config

conf = parse_config(args)  # args.config is a path to a YAML file -- see Config Files
batches_per_rank_epoch, dataset_group_list = calculate_load_balancing_on_the_fly(conf)

data_module = NativePytorchDataModule(
    dict_root_dirs=conf["data"]["dict_root_dirs"],
    dict_start_idx=conf["dataloader"]["dict_start_idx"],
    dict_end_idx=conf["dataloader"]["dict_end_idx"],
    dict_buffer_sizes=conf["dataloader"]["dict_buffer_sizes"],
    dict_in_variables=conf["data"]["dict_in_variables"],
    num_channels_used=conf["data"]["num_channels"],
    batch_size=conf["dataloader"]["batch_size"],
    num_workers=conf["dataloader"]["num_workers"],
    pin_memory=conf["dataloader"]["pin_memory"],
    interp_size=conf["data"]["interp_size"],
    tile_size=conf["data"]["tile_size"],
    twoD=conf["data"]["twoD"],
    return_label=conf["dataloader"]["return_label"],
    dataset_group_list=dataset_group_list,
    batches_per_rank_epoch=batches_per_rank_epoch,
    div=conf["tiling"]["div"],
    tile_overlap=conf["tiling"]["tile_overlap"],
    adaptive_patching=conf["ap"]["do_ap"],
    fixed_length=conf["ap"]["fixed_length"],
    separate_channels=conf["ap"]["separate_channels"],
    data_par_size=conf["parallelism"]["data_par_size"],
    dataset=conf["data"]["dataset"],
    resize=conf["dataset_options"]["resize"],
)
data_module.setup()
train_dataloader = data_module.train_dataloader()

for batch_idx, batch in enumerate(train_dataloader):
    data, variables, dict_key = batch
```

### Parameters

- `dict_root_dirs`: Dictionary of paths.
Paths to directories with input data files

- `dict_start_idx`: Dictionary of floats (0,1).
Starting indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_end_idx`: Dictionary of floats (0,1).
Ending indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_val_root_dirs` / `dict_test_root_dirs`: Dictionary of paths, optional.
Paths to directories with dedicated validation/test data files, for datasets that have their own real split. Any key present in `dict_root_dirs` but absent here gets an automatic split instead (see below).

- `dict_val_start_idx` / `dict_val_end_idx` / `dict_test_start_idx` / `dict_test_end_idx`: Dictionary of floats (0,1), optional.
Same idea as `dict_start_idx`/`dict_end_idx`, but for trimming a *separate* `dict_val_root_dirs`/`dict_test_root_dirs` directory.

- `val_split_ratio` / `test_split_ratio`: Float (0,1), default `0.1` each.
For any dataset key with **no** separate `dict_val_root_dirs`/`dict_test_root_dirs` entry, this fraction of that key's own `[dict_start_idx, dict_end_idx)` window is automatically carved out for validation/testing instead -- `train.py` only trains on the remaining share. Set both to `0` to use the full range for training. See [Training Scripts](#training-scripts) for `val.py`/`test.py`, which consume whichever split ends up resolved.

- `dict_buffer_sizes`: Dictionary of ints.
Buffer Size to use when filling iterative dataloader with prospective tiles for creation of batches

- `num_channels_used`: Dictionary of ints.
Number of Channels to use during training, currently no control of choosing modalities, but will cycle through the channels in order

- `dict_in_variables`: Dictionary of Lists of strings.
Variables corresponding to the different channels in the dataset, used in the dataloader to find corresponding correct values in `default_vars`. Needs to be in the correct order of the data files

- `batch_size`: Int.
Per GPU batch size

- `num_workers`: Int.
Number of data loader workers

- `pin_memory`: Bool.
Variable whether to use pinned memory on GPU for dataloading

- `interp_size`: Int.
Side length each adaptive (quadtree/octree) leaf patch is interpolated to -- the dataloader's own name for the same concept `VIT`'s `interp_size` parameter uses. Only meaningful when `adaptive_patching=True`.

- `tile_size`: Tuple[int, ...].
Desired tile size (one entry per spatial dimension) to generate from the input data files. If smaller than the full input file, multiple tiles will be created per file -- see `div`/`tile_overlap` below and [Load Balancing](#load-balancing).

- `twoD`: Bool. Variable for indicating two or three dimensional input, if False, three-dimensional data will be created from the dataloader. If the dataloader is three-dimensional and twoD is set to True, two-dimensional slices will be created from the three-dimensional data by iterating over the final spatial dimension of the data

- `return_label`: Bool.
Whether to return back a corresponding label to each tile when loading data

- `dataset_group_list`: Str of colon separated ints.
How to split available GPUs amongst the available datasets. Run `python utils/load_balance.py [CONFIG_FILE]` to obtain, or set `auto_load_balancing: True` in the config to have the training script compute it automatically instead. See [Load Balancing](#load-balancing) for more details

- `batches_per_rank_epoch`: Dictionary of Ints.
How many batches per rank per epoch for a given dataset. Used to get a full epoch from the Dataset with largest value. If a dataset has less than the maximum, data is reused to obtain enough to run until the largest dataset has been fully trained on. See [Load Balancing](#load-balancing) for more details

- `div`: Int.
Number of tiles to divide each spatial dimension of the full input into (a `div x div[ x div]` grid). `1` means no real tiling -- `tile_size` equals the full input size.

- `tile_overlap`: Tuple[int, ...].
Per-dimension overlap amount (in the same units as `tile_size`) between adjacent tiles, used to compute `start_overlap`/`end_overlap`. `(0, 0[, 0])` for no overlap.

- `adaptive_patching`: Bool.
Variable for indicating whether to use adaptive patching. If set to True, Adaptive Patching is done within the dataloader before being fed into a model. See [Adaptive Patching](#adaptive-patching)

- `fixed_length`: Int.
How many adaptive patches used to tokenize the input image. Only used if **adaptive_patching** is set to true

- `separate_channels`: Bool.
Whether or not to separate channels and adaptively patch with different quadtrees/octrees rather than a single one for all input channels

- `data_par_size`: Int.
The amount of data parallel training ranks being used

- `allow_file_reuse`: Bool, default `False`.
If a dataset key has fewer files than the DDP ranks/dataloader workers assigned to it, startup normally fails. Set this `True` to instead reuse (duplicate) files round-robin across ranks/workers as needed. Only applies to `train.py` -- `val.py`/`test.py` always evaluate with it off, so results stay directly comparable across runs.

- `bucket_shuffle_seed`: Int or `null`, default `42`. Imagenet only.
Imagenet's files arrive sorted class-by-class; this seeds a deterministic shuffle applied before splitting them into per-rank buckets, so each bucket gets a representative mix of classes instead of a narrow range. Set to `null` to keep the original ordering instead.

- `dataset`: String.
Variable for telling dataloader how to handle data and how to break up root directories into files within source code (Each dataset potentially needs its own code to do this depending on the data type and layout of files). One of `"imagenet"`, `"catsdogs"`, `"basic_ct"`, `"sst"` on `main`. See [Dataset Integration](#dataset-integration)

- `resize`: Dict, optional.
Maps a dataset key to the `[height, width]` size to resize that key's inputs to -- e.g. imagenet's images are all different sizes and need a standard size (`data.dict_in_variables`'s own real shape). Not specific to any one dataset name.

## Dataset Integration
Adding a new dataset means giving it a name, then teaching four places in the code how to find, read, and normalize its files.

1. Pick a name for your dataset and add it to `parse.py`'s dataset allow-list (`assert dataset in [...]`), and use that name as `data.dataset` in your config.
2. **List files**: add a branch to `UCF_VIT.utils.misc.process_root_dirs` (and, if your layout needs a representative file picked out separately, `_find_representative_file`) to list your dataset's real files under each `dict_root_dirs` entry.
3. **Read and process files**: add a branch to `FileReader.read_process_file` in `src/UCF_VIT/dataloaders/dataset.py` -- read each file with whatever library fits its format (`nibabel` for NIfTI, `np.load` for `.npy`, etc.), apply any fixed axis/layout conventions your format needs, and run it through the existing `zscore_normalize`/`normalize_stats` call (or deliberately skip that, if your data is already normalized upstream -- document why, as `xct`'s config does). See [Normalization](#normalization) for precomputing the stats this call needs.
4. **Detect size/channels for config auto-detection**: add a branch to `UCF_VIT.utils.misc.detect_num_channels`/`detect_img_size` so `parse_config` can infer `data.img_size`/`data.num_channels` from a real file instead of requiring them to be hand-specified.
5. If your dataset can't pair with every model type (e.g. no labels, so no `VIT`/`SAP`/`UNETR`), add the rejection to `parse.py`'s `return_label`-compatibility dispatch.

The shared `TileDataIter` (`src/UCF_VIT/dataloaders/dataset.py`) handles tiling for every dataset uniformly once your `read_process_file` branch hands it a correctly-shaped array -- you don't need your own tiling code.

## Normalization
Per-(dataset key, variable) z-score stats (mean/std) are not computed automatically. Precompute them once with `utils/compute_normalization_stats.py`:
```
python utils/compute_normalization_stats.py configs/basic_ct/sap/base_config.yaml
```
This reads the config's training split, writes the stats to a YAML file (default: mirrored under `stats/`), and edits the config file itself in place to set `data.normalize_stats_path` to that file, so the config is ready to train with immediately -- no separate manual edit step. It scales across multiple SLURM ranks automatically -- see `launch/utils/run_compute_normalization_stats.sh` for a launchable example.

If `data.normalize_stats_path` isn't set, training still runs, just on raw (unnormalized) data -- this is never a hard requirement, but skipping it can hurt training stability/accuracy for data whose raw scale isn't already well-behaved.

## Load Balancing
In order for the dataloader to handle multiple datasets at the same time, the data needs to be spread out amongst the GPUs evenly. In the case where different datasets have different amounts and/or different sizes of images, it's difficult to evenly spread this data amongst the GPUs evenly. `UCF_VIT.utils.misc.calculate_load_balancing_on_the_fly` computes, for a given config, how many DDP ranks each dataset key should get (proportional to its real file count, with every key guaranteed at least one rank) and how many batches-per-rank-epoch that implies. If you want this done automatically, set `trainer.auto_load_balancing: True` in your config and `train.py` computes it itself at startup -- nothing else to do. If you want to check the result manually (or you're using a dataloader type that doesn't support the automatic path), run it offline instead: `python utils/load_balance.py [CONFIG_FILE]`, then copy its `dataset_group_list`/`batches_per_rank_epoch` output into the config's load-balancing fields.

## Parallelism Modes
There is a single, unified architecture implementation (`src/UCF_VIT/model/`). Every architecture goes through the same `get_model` (`src/UCF_VIT/model/utils.py`), and parallelism is controlled purely by config (`parallelism.fsdp_size`/`simple_ddp_size`/`tensor_par_size`). Set `tensor_par_size: 1` to disable tensor/model parallelism entirely. See [Hybrid-OP](#hybrid-op) for how these three sizes combine.

### Building Blocks
The main building blocks for the VIT based architectures are in the **Attention** and **Feed-forward** functions, provided in the `Attention`/`Mlp` classes in `src/UCF_VIT/model/building_blocks.py` (also home to `Block`/`Block_diffusion`, the per-architecture block classes -- see [DiffusionVIT](#diffusion-vision-transformer-diffusionvit)). We ask that you use these functions as is and do not modify them, as these common building blocks are used across every architecture.

## Training Scripts
`training_scripts/train.py` is the single, generic training entry point for every architecture -- which model/dataset/parallelism settings it uses is entirely config-driven (see [Config Files](#config-files)), there's no separate script per architecture or per parallelism mode to pick between.

Alongside `train.py`, we also provide `val.py` and `test.py` -- forward-only counterparts that evaluate an already-trained checkpoint against a validation/test split (no backward pass, no optimizer/scheduler, `model.eval()` + `torch.no_grad()` throughout). Both take the *same* config file as the training run being evaluated (same `checkpoint_path`/`checkpoint_filename`) -- just invoked as `python val.py [CONFIG_FILE]`/`python test.py [CONFIG_FILE]` instead of `train.py`. Which data they evaluate against is controlled by `dict_val_root_dirs`/`dict_test_root_dirs` (and the `val_split_ratio`/`test_split_ratio` auto-split fallback) in the [Dataloader](#dataloader) config section -- see that section's own parameter descriptions for the full train/val/test split behavior.

For `model.type: DiffusionVIT` specifically, there's also `inference_diffusion.py` -- unlike the three scripts above, it doesn't evaluate against real data at all; it loads a trained checkpoint and runs the actual reverse-diffusion sampling loop to generate new samples from pure noise. See [DiffusionVIT](#diffusion-vision-transformer-diffusionvit)'s "Generating samples" section.

The training scripts have the capability to launch in different modes for compatibility with different systems. Each training script takes two arguments the first is the config file containing all the different parameters for the particular run and an argument for specifying the specific launching mechanism used to work across the hardware on the system. Currently we provide two launch modes: MPI and SLURM. 

For a local DGX system specify `MPI` as the LAUNCHER, then we use the `mpi4py` library to instantiate an MPI communicator and launch jobs via `mpirun -n [NUM_GPUS] -u python [TRAINING_SCRIPT] [CONFIG_FILE] MPI' spawning enough tasks to correspond to the number of GPUs to be used. In this case each task will have it's own GPU for computing. In the case where a single GPU was to be used for training the mpirun can be omitted and a simple invocation of `python [TRAINING_SCRIPT] [CONFIG_FILE] MPI` can be used (be aware to set the device to be used via `CUDA_VISIBLE_DEVICES` environment variable).

For systems that use resource schedulers to use multiple nodes acrossed a system, such as DGX Clusters or Frontier, specify `SLURM` as the launcher. In this case, SLURM environment variables are used to set up the distributed training across the resources. See some of the various examples in the `launch/` folder to see how this is done. Again in this case we assume that there this is one task per every GPU utilized for computing.

### Writing your own script
Config files store paths like `trainer.checkpoint_path`/`inference_output.output_dir` as relative paths (e.g. `"checkpoint/basic_ct/unetr/base"`), meant to be relative to the repo root. `parse_config` (`src/UCF_VIT/parse.py`) resolves these to absolute paths itself via `UCF_VIT.utils.misc.find_repo_root()`, so if you write your own script around `parse_config`, you get this for free.


## Config Files
We store the arguments for each individual run in a yaml file. This config file holds all of the arguments for defining the specific training, dataloading, parallelism, and checkpointing options. Below are a number of arguments used in these config files that weren't listed in the example files in the [Model Architectures](#model-architectures) section. In addition to these arguments, the config files also are store information for running the architectures through stored variables.

1. Trainer
- max_epochs: Max number of epochs to train
- data_type: Data type to be used during training (`"float32"` or `"bfloat16"`)
- checkpoint_path: Directory to save checkpoints to (resolved relative to the repo root -- see [Writing your own script](#writing-your-own-script))
- checkpoint_filename: File prefix to save checkpoints to -- real files are written as `<checkpoint_path>/<checkpoint_filename>_rank_<N>.ckpt`, one per tensor-parallel rank
- resume_from_checkpoint: Whether to start from checkpoint or from scratch
- auto_load_balancing: Whether to have `train.py` compute [Load Balancing](#load-balancing) itself at startup, instead of requiring `dataset_group_list`/`batches_per_rank_epoch` to already be filled in by hand

2. Parallelism (used by every script, and by `utils/load_balance.py`'s offline precompute)
- fsdp_size: Number of Fully Sharded Data Parallel ranks to use, for sharding model states
- simple_ddp_size: Number of Data Parallel ranks to use, for distributing different data to ranks. Set to the literal string `"auto"` to derive this from however many ranks the current launch actually has instead of a fixed number (`simple_ddp_size = world_size / (fsdp_size * tensor_par_size)`) -- useful so the same config works whether a run uses more or fewer nodes. Not available to `utils/load_balance.py`'s offline precompute step -- use an explicit integer there.
- tensor_par_size: Number of Tensor Parallel ranks to use, for distributing tensor across multiple ranks. `1` disables model/tensor parallelism entirely -- see [Parallelism Modes](#parallelism-modes).

3. Model Optimizer and Scheduler
- lr: Initial learning rate for optimizer
- beta_1,beta_2: Beta coefficients for Adam optimizer
- warmup_steps: Number of warmup steps for learnining rate scheduler
- max_steps: Maximum number of warmup steps for learning rate scheduler
- warmup_start_lr: Learning rate to use for warm up steps

### Usage
1. If using a new dataset, modify dataloader module accordingly
- If using a new dataset, follow [Dataset Integration](#dataset-integration) intstructions
- If using our example datasets, see [datasets](#datasets) for downloading the data and properly set your root_dir paths based on the location you download the data to.

2. Find training script of interest from the `training_scripts/` directory

3. Modify training script for the particular use case. (Adding validation, testing, inferencing, etc. as needed)

4. Create/Modify config file for your training

5. Create/Modify Launch Script
- Find/Create a proper launch script for the run you want to do
- If on Frontier Change project allocation to one you have access to `#SBATCH -A PROJXXX`.
- Set number of nodes you want to run with `#SBATCH --nodes=N`

6. Run Load Balancing Script (or set `trainer.auto_load_balancing: True` in the config and skip to step 8)
- `python utils/load_balance.py [CONFIG_FILE]`

7. Modify Config File with the output from load balancing output
- dataset_group_list
- batches_per_rank_epoch

8. Launch job `sbatch launch/[DATASET]/[MODEL].sh`
- Existing launch scripts follow `launch/[DATASET]/[MODEL].sh`, e.g. `launch/basic_ct/unetr.sh`, `launch/imagenet/diffusion.sh`
- [DATASET] is the particular dataset you want to use (examples use `imagenet`/`basic_ct`)
- [MODEL] is the model/task from [Model Architectures](#model-architectures) (lowercased, e.g. `unetr`, `mae`, `diffusion`)
- Unlike the eval scripts in step 9 below, these training launch scripts don't take the config path as a `sbatch` argument -- each one hardcodes its own config path at the bottom. Edit that line to point it at your own config (this is the "Create/Modify Launch Script" step above).
- There's no separate parallelism-"mode" launch script anymore -- see [Parallelism Modes](#parallelism-modes); the same script works for any `fsdp_size`/`simple_ddp_size`/`tensor_par_size` combination in the config

9. Evaluate a checkpoint with `training_scripts/val.py`/`test.py`
- Unlike the training launch scripts above, `launch/eval/val.sh` and `launch/eval/test.sh` are generic across every dataset/model -- pass the config path as an argument instead of hardcoding it per-launch-script. They live one directory down (`launch/eval/`), matching the training launch scripts' own depth (`launch/[DATASET]/`), so the config path argument uses the same `../../` convention either way:
  - `cd launch/eval && sbatch val.sh ../../configs/basic_ct/unetr/base_config.yaml`
  - `cd launch/eval && sbatch test.sh ../../configs/basic_ct/unetr/base_config.yaml`
- Use the same config as the training run being evaluated (same `checkpoint_path`/`checkpoint_filename`) -- `val.py`/`test.py` force `trainer.resume_from_checkpoint: True` in code regardless of what the config says.


## Datasets
`data.dataset` (in [Config Files](#config-files)) selects one of these on `main`: `"imagenet"`, `"catsdogs"`, `"basic_ct"`, `"sst"`. See [Dataset Integration](#dataset-integration) for adding your own.
### Imagenet
Download data at `https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php`

Directory consists of 1000 sub-folders of 2D JPEG images, each folder corresponding to a different classification label. Since ImageNet has images of all different sizes, all input images are resized to a standard size via the `resize` argument in the config file (`data.resize` with an `"imagenet"` key -- see [Dataloader](#dataloader)).
### Catsdogs
Binary cats-vs-dogs 2D image classification/diffusion dataset -- see `configs/catsdogs/` for real shipped configs (classification, MAE, diffusion). This dataset is intended as a worked example of plugging in a simple, map-style PyTorch `Dataset` + `DistributedSampler` (`dataloader.type: dataloader`), as opposed to the iterative dataloader (`dataloader.type: iterative_dataloader`) used by imagenet/basic_ct/sst -- see `UCF_VIT.datasets.catsdogs` and `train.py`'s own branching on `dataloader.type`.
### Basic_CT
3D synthetic CT volumes of concrete, with corresponding segmentation labels. We don't have a public download link for this one documented anywhere in the repo currently -- ask a maintainer/collaborator for access if you need it.
### SST
3D CFD (computational fluid dynamics) volumes with multiple fields (density `r`, velocity components `u`/`v`/`w`, pressure `p`), used for both MAE pretraining and UNETR regression (predicting `p` from the others) -- see `configs/sst/` for real shipped configs, and their header comments for the real data layout (large volumes, chunked and tiled via `tiling.div`). The one dataset that makes real use of `data.time_offsets`/`use_timeemb`'s time-stepping conditioning -- see [Variable Aggregation](#variable-aggregation).

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
### [2]
```bibtex
@Misc{xFormers2022,
  author =       {Benjamin Lefaudeux and Francisco Massa and Diana Liskovich and Wenhan Xiong and Vittorio Caggiano and Sean Naren and Min Xu and Jieru Hu and Marta Tintore and Susan Zhang and Patrick Labatut and Daniel Haziza and Luca Wehrstedt and Jeremy Reizenstein and Grigory Sizov},
  title =        {xFormers: A modular and hackable Transformer modelling library},
  howpublished = {\url{https://github.com/facebookresearch/xformers}},
  year =         {2022}
}
```

### [3]
```bibtex
@software{Liu_Composable_Kernel,
author = {Liu, Chao and Zhang, Jing and Qin, Letao and Zhang, Qianfeng and Huang, Liang and Wang, Shaojie and Chang, Anthony and Lai, Chunyu and Silin, Illia and Osewski, Adam and Chen, Poyen and Geyyer, Rosty and Chen, Hanwen and Shah, Tejash and Zhou, Xiaoyan and Yan, Jianfeng},
license = {MIT},
title = {{Composable Kernel}},
url = {https://github.com/ROCm/composable_kernel}
}

```

### [4]
```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

### [5]
```bibtex
@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

### [6]
```bibtex
@inproceedings{zhang2024adaptive,
  title={Adaptive Patching for High-resolution Image Segmentation with Transformers},
  author={Zhang, Enzhi and Lyngaas, Isaac and Chen, Peng and Wang, Xiao and Igarashi, Jun and Huo, Yuankai and Munetomo, Masaharu and Wahib, Mohamed},
  booktitle={SC24: International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--16},
  year={2024},
  organization={IEEE}
}
```

### [7]
```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

### [8]
```bibtex
@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16000--16009},
  year={2022}
}
```

### [9]
```bibtex
@inproceedings{hatamizadeh2022unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={574--584},
  year={2022}
}

```

### [10]
```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

### [11]
```bibtex
@article{wang2025orbit,
  title={ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling},
  author={Wang, Xiao and Choi, Jong-Youl and Kurihaya, Takuya and Lyngaas, Isaac and Yoon, Hong-Jun and Fan, Ming and Nafi, Nasik Muhammad and Tsaris, Aristeidis and Aji, Ashwin M and Hossain, Maliha and others},
  journal={arXiv preprint arXiv:2505.04802},
  year={2025}
}
```

### [12]
```bibtex
@article{tsaris2025distributed,
  title={Distributed Cross-Channel Hierarchical Aggregation for Foundation Models},
  author={Tsaris, Aristeidis and Lyngaas, Isaac and Lagregren, John and Wahib, Mohamed and York, Larry and Balaprakash, Prasanna and Lu, Dan and Wang, Feiyi and Wang, Xiao},
  journal={arXiv preprint arXiv:2506.21411},
  year={2025}
}
```
