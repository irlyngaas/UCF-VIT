## Table of Contents
- [UCF-VIT](#ucf-vit)
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
- [Load Balancing](#load-balancing)
- [Parallelism Modes](#parallelism-modes)
- [Training Scripts](#training-scripts)
- [Config Files](#config-files)
- [Datasets](#datasets)

## UCF-VIT
This repository provides a unified coding framework for scalable Vision Transformer (ViT) models, designed to support both extreme-scale model training on leadership-class supercomputers and efficient deployment on smaller systems such as DGX nodes. The framework integrates several cutting-edge innovations in efficient computing, including optimized attention mechanisms, advanced parallelism strategies (data, tensor, sequence, and pipeline parallelism), and mixed-precision acceleration, enabling the training of vision models with hundreds of billions of parameters and achieving sustained ExaFLOP-scale performance on systems like the Frontier supercomputer.

At the same time, the codebase is modular and flexible, allowing researchers and practitioners to run the same models on smaller compute environments without sacrificing performance or usability. This unified approach lowers the barrier to entry for next-generation large-scale vision model research while ensuring reproducibility and scalability across diverse computing platforms.

The repository is open-source and maintained on GitLab to foster collaboration and accelerate innovation in vision transformer research at scale.

## Why use UCF-VIT?
Training large Vision Transformer (ViT) models at scale remains a major technical challenge in both AI research and deployment, with most existing codebases narrowly optimized for either small-scale experiments or vendor-specific hardware, creating barriers to portability, scalability, and reproducibility. A unified coding framework is essential to streamline development, maximize resource efficiency, and enable seamless scaling from small machines to the world’s largest supercomputers. This repository provides such a unified solution—offering robust support for both NVIDIA and AMD GPUs, across systems ranging from single-node DGX machines to leadership-class clusters like the Frontier supercomputer. The framework incorporates our in-house hybrid-stop extreme-scale parallel computing technique, which won the HPCWire Supercomputing Achievement Award, to unlock scalable and efficient ViT model training. It also introduces our adaptive patching method to reduce the computational complexity of ViT attention, alongside variable aggregation for handling large-channel ViTs. Additionally, it integrates widely used techniques such as lower-precision training, layer wrapping, fused attention, and FlashAttention for further efficiency gains. The framework is compatible with a variety of ViT architectural variants, making it a powerful, flexible tool for researchers and practitioners aiming to push the frontier of scalable vision model development and deployment.

# Install
Installation instruction are provide for running on systems with both AMD and NVIDIA GPU hardware. Instructions are included for systems ranging from a local DGX cluster to the Frontier Supercomputer.

## Systems with AMD GPUs
There are two options available for creating software environments for systems with AMD GPUs 1) creating Conda environment from scratch or 2) using an apptainer container (the installation instructions for this are currently limited to use on the Frontier supercomputer). Creating a Conda environment from scratch is recommended as the Apptainer containers currently only work in limited scenarios due to missing ROCM packages in the base apptainer image.

### Conda
Create Conda Environment from Scratch. Example below uses similar options from the corresponding Apptainer definition files
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
pip install xformers==0.0.30 --extra-index-url=https://download.pytorch.org/whl/rocm${ROCM_VERSION}
pip install timm \
 monai \
 nibabel \
 torchdata==0.9.0 \
 einops \
 opencv-python-headless \
 matplotlib \
 scipy

#If your system has an existing MPI installed use the proper mpi4py installation for your sytsem
#Default install mpi4py 
MPI_DIST=mpich
conda install -c conda-forge mpi4py ${MPI_DIST}
#mpi4py install on Frontier
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

cd UCF-VIT
pip install -e .
```

### Apptainer (on Frontier)
Use Apptainer container definition files (Only use this on Frontier)
```
cd apptainter
apptainer build frontier-ubuntu-gnu-rocm624.sif frontier-ubuntu-gnu-rocm624.def
apptainer build frontier-ubuntu-gnu-rocm624-vit.sif frontier-ubuntu-gnu-rocm624-vit.def
mkdir lib/8.1.31
cd lib/8.1.31
ln -s $CRAY_MPICH_DIR/lib/libfmpich.so libmpicxx.so
```

Various example scripts for launching jobs are in the launch folder. Those identified with `_apptainer` in the filename are for running with the Apptainer container

## Local DGX System
To run on a local NVIDIA system we provide instructions for creating a Conda environment from scratch

```
conda create -n vit python=3.11 -y
conda activate vit
CUDA_DRIVER=cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_DRIVER}
pip install xformers \
timm \
monai==1.4.0 \
nibabel \
torchdata==0.9.0 \
einops \
opencv-python-headless==4.11.0.86 \
matplotlib \
scipy 

#If your system has an existing MPI installed use the proper mpi4py installation for your sytsem
#Default install mpi4py 
MPI_DIST=mpich
conda install -c conda-forge mpi4py ${MPI_DIST}

cd UCF-VIT
pip install -e .
```

To run on a local DGX system using one of our training scripts invoke the following command `mpirun -n [NUM_GPUS] python -u [TRAINING_SCRIPT] [CONFIG_FILE] MPI` or `python [TRAINING_SCRIPT] [CONFIG_FILE] MPI` to run on a single GPU. If you are running on a shared resource machine and you want to use a subset of the available GPUS be sure to set the visible devices before running, e.g. `os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'`.

## DGX Cluster System
To run on NVIDIA DGX cluster systems we rely on Pytorch Docker containers maintained by NVIDIA. The following instructions give commands to build a docker container with our codebase.

```
cd Docker
docker build -t ucf-vit:25.05 .
docker tag ucf-vit:25.05 [DOCKER_USERNAME]/ucf-vit:25.05
docker push [DOCKER_USERNAME]/ucf-vit:25.05
```

or alternatively pull an already created image from the public dockerhub repo with

```
docker pull irlyngaas/ucf-vit:25.05-upd2
```

Various example scripts for launching jobs are in the launch folder. Those identified with `_dgx` in the filename are for running with a Docker container

# Features
In this codebase, we provide various Advanced Parallelism & Efficient Computing techniques that we have used to explore larger model and input sizes with VITs than has been previously possible. These techniques range from novel methods to the utilization of several techniques provided by external libraries that are integrated with these novel techniques.

## Hybrid-STOP 
Hybrid Sharded Tensor-Data Orthogonal Parallelism (Hybrid-STOP) [[1]](#1),[[11]](#11) is a novel parallelism algorithm that combines tensor parallelism and Fully Sharded Data Parallelism (FSDP). It avoids the peak memory use probelm in FSDP and leads to better memory reduction capability by keeping parameters sharded throughout training. 
### Usage
The Hybrid-STOP algorithm is available when using our fsdp [parallelism mode](#parallelism-modes). The following example shows how to initialize and do the forward pass of a [MAE](#masked-autoencoder-mae) model using this algorithm for different number of simple_ddp, fsdp, and tensor parallel ranks (see scripts in the training_script folder full end-to-end training examples). Our custom [dataloader](#dataloader) with the [Imagenet](#imagenet) dataset is used to facilitate proper dataloading when tensor parallelism is > 1 (In this case each tensor parallel rank needs the same batch of input data). 

This example is meant to be run on a system that uses a slurm resource scheduler or with an installed MPI library. To run with a single node on a system with a slurm scheduler use the following command `srun -n [NUM_TASKS] -c [NUM_CPUS_PER_TASK] --gpus [NUM_GPUS] python test-hstop.py SLURM`. To run on a local system via MPI use the following command `mpirun -n [NUM_GPUS] python -u test-hstop.py MPI`

```python
#test-hstop.py

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datetime import timedelta
from UCF_VIT.utils.misc import init_par_groups
from UCF_VIT.fsdp.arch import MAE
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule

LAUNCHER = sys.argv[1]

if LAUNCHER == "SLURM":

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

elif LAUNCHER == "MPI":
    from mpi4py import MPI
    import socket 

    num_gpus_per_node = torch.cuda.device_count()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(num_gpus_per_node) if num_gpus_per_node>0 else 0 # local_rank and device are 0 when using 1 GPU per task
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    master_addr = None
    if rank == 0:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        master_addr = ip_address
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr

    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

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
    ).to(device)

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
            dataset_group_list = '1:1:1:1', #Calculated from running utils/load_balance.py with a corresponding config file, these values will change if data_par_size changes
            batches_per_rank_epoch = {'imagenet':9945}, #Calculated from running utils/load_balance.py with a corresponding config file, these values will change if data_par_size changes
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
In order to properly implement the H-STOP algorithm, specifically the tensor parallelism aspects, the architecture code requires modifications (from that of the code for the ([simple architecture mode](#parallelism-modes)) to correctly split up and communicate the tensor calculations amongst the tensor parallel ranks. Currently tensor parallelism is only enacted over the attention and mlp calculations, which are the costliest components. The below code snippet shows how the tensor parallel implementation is implemented in the FSDP parallelism mode, corresponding modifications are built into the attention mechanism in `src/UCF-VIT/fsdp/building_blocks.py`. Tensor parallelism is not implemented in simple parallelism mode, so the tensor_par_group communicator group is not required to be passed to the neural network architecture for that case. Here `self.blocks` corresponds to the attention and multi-layer perceptron (mlp) layers that consist of the bulk of the computing in each of the VIT architectures.  

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
The ability to implement lower precision support is facilitated by using a MixedPrecision Policy which is passed as an argument to the FSDP wrapper. Using lower precision data types such as bfloat16 reduces storage size to allow for large model sizes and increased throughput due to the faster computing that is possible with lower precision.

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
The ability to use layer wrapping is facilitated by using a custom autowrap policy which is passed as an argument to the FSDP wrapper. The purpose of this wrapping is to control how and when the parameters of each layer are sharded and gathered during the forward and backward passes. By wrapping layers in the Block submodule (which contains the transformer computational layers) peak GPU memory usage is reduced, and communication and computations have improved overlapping.
### Usage 
Add the following code and replace the FSDP Wrapper calls in the [full example](#Hybrid-STOP) to apply layer wrapping
```python
from UCF_VIT.fsdp.building_blocks import Block
from torch.nn import Sequential
import functools

my_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        Block, Sequential
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
The ability to invoke activation checkpointing to the model is provided through FSDP's `apply_activation_checkpointing` function. In our cases we apply activation checkpointing only to the `Block` submodule containing the transformer computational layers to reduce the memory storage required by that component of the model training. With activation checkpointing, the gradients are no longer stored for that component to be used during the backward pass of optimatization. Rather the gradients are recomputed from previously stored states when necessary during the backwards pass, significantly reducing the amount of GPU memory used during runtime.
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
## Fused Attention via XFormers
In order to integrate computationally efficient kernels across different GPU accelerated hardware, we rely on the XFormers [[2]](#2) libary. The XFormers library provides an interface to memory efficient implementations of fused multi-head attention (FMHA) for various different hardwares. On AMD GPUs we use the ComposableKernel (CK)[[3]](#3) implementation of FMHA and on NVIDIA GPUs we use the FlashAttention [[4]](#4),[[5]](#5) implementation of FMHA. These two implementations only have compatibility with bfloat16 operations, thus with float32 data types we use the default torch FMHA kernel implementation.

### Usage 
We provide an enumerated class with which to choose the FMHA implementation to use in the architecture class.
```python
from UCF_VIT.utils.fused_attn import FusedAttn

#Choose from these options
FusedAttn_option = FusedAttn.DEFAULT #Default torch implementation of FMHA compatible with float32 datatype
#FusedAttn_option = FusedAttn.FLASH #Flash Attention implementation of FMHA comatible with bloat16 datatype and NVIDIA GPUs
#FusedAttn_option = FusedAttn.CK #Composable Kernel implementation of FMHA comatible with bloat16 datatype and AMD GPUs
#FusedAttn_option = FusedAttn.None #Basic Python implementation of FMHA

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
    FusedAttn_option = FusedAttn_option,
)

```

## Adaptive Patching
A recent innovation for efficient computing with VITs that we have implemented within this codebase is adaptive patching [[6]](#6). Traditionally in VITs, input images (or data) are separated in to groups of individual nonoverlapping pixels (or individual datapoints) called tokens of size patch_size x patch_size in 2D (or patch_size x patch_size x patch_size in 3D), these tokens are then flattened to be used as input to the ViT. When input image sizes become too large, the sequence length of tokens being fed into the network can become intractably large except for at very large patch sizes which can hinder the ability to train more accurate models. Often it is the case that there are large portions of regions of input that are largely homogenous, thus it is wasteful to consider all of the input tokens equally. Adaptive patching is an approach inspired by Adaptive Mesh Refinement (AMR) techniques in which a tree is used to break down input data into adaptively sized regions based on some quantity such as the magnitude of the gradient to indicate how much variation is in each spatial region. In our case we use a Canny Edge detection method to break down the images with a quadtree (2D) or octtree (3D) into a smaller regions adaptively depending on the amount of edges in certain regions of the image. To control the amount of regions the image is broken into we use the integer variable `fixed_length`, to tell adaptive patching to stop splitting the image. Each adaptively sized region is then resized into tokens of size patch_size x patch_size or (patch_size x patch_size x patch_size in 3D) to be in a form suitable for being input into a VIT. This approach can drastrically reduce the sequence length of tokens and consequently significantly reduced the amount of compute time required to feed through the network.

### Usage
In our implementation, adaptive patching is currently handled during dataloading time. Therefore if `adaptive_patching` is set to True rather then the dataloader passing back a batch of input images, instead a batch of adaptively patched input images are passed through the dataloader. If adaptive patching is being used, an integer fixed length needs to be defined and it must be chosen such that 3n+1 = fixed_length where n is some integer if the input is 2D or such that 7n+1 = fixed_length where n is some integer if the input is 3D, to satisify the requirements for the underlying quadtree/octtree. Also it is a requirement that the each dimension of the input images be of a size that is a power of 2, i.e. 32, 64, ..., in order to properly split the data. 

## Variable Aggregation
Another advanced technique that we have introduced into this codebase particularly for the purpose of foundational model training is the capability to incorporate variable aggregation [[12]](#12). Variable aggregation is a technique where instead of tokenizing multi-channel inputs all at once and transforming them into the latent embedding dimension space, each individual channel is tokenize individually  and transformed into the latent embedding dimensions space based on the type of channel data, given some direction via the `variables` argument in the forward pass. Each of these channel embedding vectors are then fed through an additional attention mechanism to compress all of the input into a single dimension, a process we call variable aggregation. The reason it is important that these input channels be tokenized and embedded separately is because it allows the flexibility to use a pre-trained foundational model in a flexible manner. Variable aggregation allows for the ability to use different types of data during training, e.g. data that does not contain all of the input channels contained in other datasets that the model is being trained with.

### Usage 
In order to use variable aggregation set the `use_varemb` to True. In the case where `use_varemb=False` multi-channel tokenization will be performed and thus any further training or finetuning with that model will require data to match the specific number of channels as the original data used with that model. Variable aggregation is controlled by sending a list of variables identifiers to the forward pass of the model architecture, i.e. `["red","green","blue"]` for the case of RGB images. This list of variables must correspond correctly with the order that the data is formatted in and this process is facilitated through our [dataloader](#dataloader) via passing in the identifier list to the `dict_in_variables` argument of the config file. `default_vars` controls the type of types of input channels that the model will allow for ingestion. Thus every variable in `dict_in_variables` must be in `default_vars`, however not every input channel is necessary when passing through the model. 

# Model Architectures
Currently we provide 5 different model architecutres **(VIT, MAE, UNETR, SAP, VIT-DIFFUSION)**, all of which use the same VIT encoder, but a different decoder architecture depending on the task being trained. All code for the different architectures inherit the encoder from the VIT architecture class in order to facilitate using the same encoder. In the following sections we provide working examples for exectuing a forward pass with each of these architectures that can be ran on a single CPU. For more complex full training runs on multiple GPUs look to the example scripts in the `training_scripts/` directory.

## Vision Transformer (VIT)
VIT based on [[7]](#7). Code slimmed down from (https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L425) to only contain basic options for VIT Training and adapted to integrate our [innovations](#innovations). Task: Image Classification. Input: Image or Image Tile (A tile is a subset portion of a full image).

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
    default_vars = ["red", "green", "blue"],
    single_channel = False,
    adaptive_patching = False,
    fixed_length = None,
    FusedAttn_option = FusedAttn.DEFAULT,
)

img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 1000)
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

- `single_channel`: Bool.
Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality

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


img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds, _ = model.forward(img, variables) # (1, 16*16, 16*16*3) -> (batch_size, num_tokens_x * num_tokens_y, patch_size * patch_size * num_channels)

#Move masked image prediction from patched space back to original image space
pred_img = unpatchify(preds, img, 16, True) # (1, 3, 256, 256) 
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

- `mlp_ratio_decoder`: Int.
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


img = torch.randn(1, 3, 256, 256) # (batch_size, num_channels, tile_size_x, tile_size_y)
variables = ["red", "green", "blue"]

preds = model.forward(img, variables) # (1, 4, 256, 256) # (batch_size, num_classes, tile_size_x, tile_size_y)
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
```python
import torch
import numpy as np
from UCF_VIT.simple.arch import SAP
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
        single_channel=False,
        use_varemb=False,
        adaptive_patching=True,
        fixed_length=64,
        sqrt_len=int(np.sqrt(64)),
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = np.random.randn(1, 3, 256, 256).astype(np.uint8) # (batch_size, num_channels, tile_size_x, tile_size_y)
patchify = Patchify(fixed_length=64, patch_size=16, num_channels=3)
adaptive_patch_img, _, _, _ = patchify(np.moveaxis(np.squeeze(img),0,-1)) # -> (num_channels, fixed_length, patch_size*patch_size)
adaptive_patch_img = np.expand_dims(adaptive_patch_img,axis=0).astype(np.float32) # Add back batch dimension
adaptive_patch_img = torch.from_numpy(adaptive_patch_img)


variables = ["red", "green", "blue"]

preds = model.forward(adaptive_patch_img, variables) # (1, 4, sqrt(64)*16, sqrt(64)*16)
```

## Diffusion Vision Transformer (DiffusionVIT)
Diffusion model training based on [[10]](#10). Task: Generate Image via noise that matches distribution of data trained on. Input: Noise

### Usage
```python
import torch
from UCF_VIT.simple.arch import DiffusionVIT
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.ddpm.ddpm import DDPM_Scheduler
from UCF_VIT.utils.misc import unpatchify

num_time_steps = 1000
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
        linear_decoder=True,
        twoD=True,
        mlp_ratio_decoder=4,
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        time_steps=num_time_steps,
        FusedAttn_option=FusedAttn.DEFAULT,
        class_token=False,
        weight_init='skip',
    )


img = torch.randn(1, 3, 256, 256)
variables = ["red", "green", "blue"]

t = torch.randint(0,num_time_steps,(1,))
e = torch.randn_like(img, requires_grad=False)
ddpm_scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
a = ddpm_scheduler.alpha[t].view(1,1,1,1)
data = (torch.sqrt(a)*img) + (torch.sqrt(1-a)*e)
output = model.forward(data, t, variables) # (1, 16*16, 16*16*3) -> (batch_size, num_tokens_x * num_tokens_y, patch_size * patch_size * num_channels)

#Move generated image from patched space back to original image space
output = unpatchify(output, data, 16, True) #(1, 3, 256, 256)
```
### Parameters (if not listed here see descriptions in the architectures above)

- `time_steps`: int.
Number of time steps in the diffusion process

## Dataloader
The dataloader we provide is a custom native pytorch iterative dataloader. For simplicity, we assume that we are receiving unprocessed data files and we leave it to the user to normalize the data properly within in the dataloader module for training. The reasons for making this assumption of unprocessed data files as input is 1) we intend this repo to be used on very large datasets, thus preprocessing and storing all of the data before training can quickly take up a massive amount of storage, and 2) it removes the need for further data preprocessing scripts to be included in this repo. If performing preprocessing during the dataloading phase is too computationally intensive, we recommend doing it offline and properly storing it in a manner that the dataloader module can handle. 

The dataloader is built in a fashion such that it can handle multiple different dataset directories at the same time. A dataset directory contains one or more data file (with all data files having the same dimension or able to be resized so that they have the same dimension). The purpose of being able to handle multiple dataset directories is 1) it provides flexible training where you can easily remove and add different datasets for the purposes of running experiments and 2) it allows for the integration of identifying properties from the different datasets that can potentially used for improved learning via our advanced features. For instance, with data that has multiple channels, e.g. images with (R,G,B) channels, we are able to pass along the information on what variable the channel is from and use that information during network training. We then could utilize [variable aggregration](#variable-aggregation) to tokenize each channel separately.

This dataloader provides the flexibility to add a plethora of different options for customizing how the data is broken up for training. Since we are using a VIT, at least 2D data is expected. However, we have capability for both 2D and 3D spatial data currently. If desired, we have the utilities implemented to break given data into smaller tiled chunks. Also, we have a number of different options for how to tile this data, e.g. tile overlapping.

### Usage
This example is meant to be run on a system that uses a slurm resource scheduler or with an installed MPI library. To run with a single node on a system with a slurm scheduler use the following command `srun -n [NUM_TASKS] -c [NUM_CPUS_PER_TASK] --gpus [NUM_GPUS] python test-hstop.py SLURM`. To run on a local system via MPI use the following command `mpirun -n [NUM_GPUS] python -u test-hstop.py MPI`
```python
#test_dataloader.py
from UCF_VIT.dataloaders.datamodule import NativePytorchDataModule
import torch.distributed as dist
import sys

LAUNCHER = sys.argv[1]

if LAUNCHER == "SLURM":

    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

elif LAUNCHER == "MPI":
    from mpi4py import MPI
    import socket 

    num_gpus_per_node = torch.cuda.device_count()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(num_gpus_per_node) if num_gpus_per_node>0 else 0 # local_rank and device are 0 when using 1 GPU per task
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)

    master_addr = None
    if rank == 0:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        master_addr = ip_address
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr

    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), rank=world_rank, world_size=world_size)

data_module = NativePytorchDataModule(dict_root_dirs={'imagenet': '~/imagenet/train',},
        dict_start_idx={'imagenet': 0},
        dict_end_idx={'imagenet': 1},
        dict_buffer_sizes={'imagenet': 100},
        dict_in_variables={'imagenet': ["red", "green", "blue"]},
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
        dataset_group_list = '1:1:1:1:1:1:1:1', #Calculated from running utils/load_balance.py with a corresponding config file, these values will change if data_par_size changes
        batches_per_rank_epoch = {'imagenet':4935}, #Calculated from running utils/load_balance.py with a corresponding config file, these values will change if data_par_size changes
        tile_overlap = 0.0,
        use_all_data = False,
        adaptive_patching = False,
        fixed_length = None,
        separate_channels = False,
        data_par_size = dist.get_world_size(),
        dataset = 'imagenet',
        imagenet_resize = {'imagenet':[256,256]},
    ).to(device)

data_module.setup()

train_dataloader = data_module.train_dataloader()

for batch_idx, batch in enumerate(train_dataloader):
    data, variables = batch
```

### Parameters

- `dict_root_dirs`: Dictionary of paths.
Paths to directories with input data files

- `dict_start_idx`: Dictionary of floats (0,1).
Starting indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_end_idx`: Dictionary of floats (0,1).
Ending indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use

- `dict_buffer_sizes`: Dictionary of ints.
Buffer Size to use when filling iterative dataloader with prospective tiles for creation of batches

- `num_channels_used`: Dictionary of ints.
Number of Channels to use during training, currently no control of choosing modalities, but will cycle through the channels in order

- `dict_in_variables`: Diction of Lists of strings.
Variables corresponding to the different channels in the dataset, used in the dataloader to find corresponding correct values in the default_var_list. Needs to be in the correct order of the data files

- `batch_size`: Int.
Per GPU batch size

- `num_workers`: Int.
Number of data loader workers, should be set at 1 for now

- `pin_memory`: Bool.
Variable whether to use pinned memory on GPU for dataloading

- `patch_size`: Int.
Patch Size to use when creating patch Embeddings Sequences for the network input

- `tile_size_[x,y,z]`: Int.
Desired tile size to generate from the input data files. If tile_size is smaller than the size of the input files, multiple tiles will be created from each data file

- `twoD`: Bool. Variable for indicating two or three dimensionsal input, if False, three-dimensional data will be created from the dataloader. If the dataloader is three-dimensional and twoD is set to True, two-dimensional slices will be created from the three-dimensional data by iterating over the final spatial dimension of the data

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
-Variable for telling dataloader how to handle data and how to break up root directories into files within source code (Each dataset potentially needs it's own code to do this depending on the data type and layout of files). See [Datset Integration](#dataset-integration)

- `imagenet_resize`: List of Ints.
-Optional argument specific to the imagenet datset which tells the dataloader what size to resize all images to so that the same input size is used.

## Dataset Integration
For Examples, see the XCT-Diffusion, SST, and S8D branches
1. Name your dataset and use it in place of the dataset option of the config file
2. Write code to process file keys for the different datasets
- Add a new branch to if/else in the process_root_dirs function of the NativePytorchDataModule in `src/UCF_VIT/dataloaders/datamodule.py`, to process datafile paths from each dataset into a corresponding dictionary
3. Write code that uses appropriate iterative dataloader functions from `src/UCF_VIT/dataloaders/dataset.py` to handle the data files
- Add a new branch to if/else in the set_iterative_dataloader function of the NativePytorchDataModule class in `src/UCF_VIT/dataloaders/datamodule.py`, using the correct Tile Iterator (ImageBlockDataIter_2D or ImageBlockDataIter_3D) depending on the dimension of your data
4. Write code to appropriately read and process (including normalization) data files
- Add a new branch to if/else in the read_process_file function of the FileReader class in `src/UCF_VIT/dataloaders/dataset.py`, using an appropriate python function to read the data files depending on the type
5. Write code to appropriately load balance data files across the computing hardware
- Add a new branch to if/else in the process_root_dirs function of `src/UCF_VIT/utils/misc.py` (similar to step 2)
- Add a new bracnh to if/else in the read_process_file function of `src/UCF_VIT/utils/misc.py` (similar to step 4)

## Load Balancing
In order for the dataloader to handle multiple datasets at the same time, the data needs to be spread out amongst the GPUs evenly. In the case where different datasets have different amounts and/or different sizes of images, it's difficult to evenly spread this data amongst the GPUs evenly. We provide example load balancing scripts that for a given setting in a config file determines how the data should be split amongst a given set of N GPUs, in order to evenly balance the data amongst the compute resources. The output from this script gives the necessary information to the dataloader in order to do this in a proper fashion. If you want this load_balancing to be done automatically set `auto_load_balancing` to True in your config file. If you want to do the load balancing manually to check for correct implementation run `python utils/load_balance.py [CONFIG_FILE] [NUM_GPUS]` and use the output from this script to add to the load balancing portion of the config file.

## Parallelism Modes
All of the currently existing architectures exist in 2 independent sub-folders, simple and fsdp, for which we separate the network architecture code into what we call modes. The choice of mode to be used will depend on the types of advanced parallelism and computing techniques needed for the model being trained.  The first `src/UCF_VIT/simple`, provides a simplified version for training in Distributed Data Parallel (DDP) fashion only. The second `src/UCF_VIT/fsdp`, provides a more complex version with different parallel and efficient computing training techniques. This includes options for training with a combination of Sharded Data Parallelism , DDP, and Tensor Parallelism. These parallelisms are all integrated via the [Hybrid-Stop](#hybrid-stop) algorithm. Both modes can be used with the same data loading module and load balancing scripts provided. While the training done within the simple mode can be done with the correct choice of options in the fsdp mode, the purpose of keeping the simple mode is 1) to provide an entry point for new users and developers to add new architectures without the intricacies of the advanced features and 2) to provide a simple reference point to compare with when new innovations are added in order to test how they interact with the more complex parallelism methods.

### Building Blocks
The main building blocks for the VIT based archictectures are in the **Attention** and **Feed-forward** functions, provided in the Attention class and MLP class in `src/UCF_VIT/simple/building_blocks.py` and `src/UCF_VIT/fsdp/building_blocks.py`. We ask that you use these functions as is and do not modify them, as these common building blocks will be used across the different network architectures.

## Training Scripts
We provide several example training scripts. These include all of the necessary things for running the main training loop, including utilities such as checkpoint loading and saving and mechanisms for launching across hardware for different systems. We leave it to the user to implement their own validation and testing routines in order to more closely fit their needs. Training scripts are provided for each of the training architectures for the simple mode. We also have several scripts to train architectures in the fdsp mode. To convert the simple scripts to use fsdp mode, implement the code changes that were done for, e.g. `training_scripts/train_masked_simple.py` to `training_scripts/train_masked_fsdp.py`. 

The training scripts have the capability to launch in different modes for compatibility with different systems. Each training script takes two arguments the first is the config file containing all the different parameters for the particular run and an argument for specifying the specific launching mechanism used to work across the hardware on the system. Currently we provide two launch modes: MPI and SLURM. 

For a local DGX system specify `MPI` as the LAUNCHER, then we use the `mpi4py` library to instantiate an MPI communicator and launch jobs via `mpirun -n [NUM_GPUS] -u python [TRAINING_SCRIPT] [CONFIG_FILE] MPI' spawning enough tasks to correspond to the number of GPUs to be used. In this case each task will have it's own GPU for computing. In the case where a single GPU was to be used for training the mpirun can be omitted and a simple invocation of `python [TRAINING_SCRIPT] [CONFIG_FILE] MPI` can be used (be aware to set the device to be used via `CUDA_VISIBLE_DEVICES` environment variable).

For systems that use resource schedulers to use multiple nodes acrossed a system, such as DGX Clusters or Frontier, specify `SLURM` as the launcher. In this case, SLURM environment variables are used to set up the distributed training across the resources. See some of the various examples in the `launch/` folder to see how this is done. Again in this case we assume that there this is one task per every GPU utilized for computing.


## Config Files
We store the arguments for each individual run in a yaml file. This config file holds all of the arguments for defining the specific training, dataloading, parallelism, and checkpointing options. Below are a number of arguments used in these config files that weren't listed in the example files in the [Model Architectures](#model-architectures) section. In addition to these arguments, the config files also are store information for running the architectures through stored variables.

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

6. Run Load Balancing Script (or set `auto_load_balancing` to True in config file and skip to step 8)
- `python utils/load_balance.py [CONFIG_FILE] [NUM_GPUS]`

7. Modify Config File with the output from load balancing output
- dataset_group_list
- batches_per_rank_epoch

8. Launch job `sbatch launch/[DATASET]/train_[MODEL]_[MODE]_[OPTIONAL].sh`
- Existing launch scripts follow the following naming conventions
- [DATASET] is the particular dataset you want to use. The examples use (imagenet or basic_ct)
- [MODEL] is the type of model you want to use chose from those in [Model Architectures](#Model-Architectures)
- [MODE] is the training mode you want to use. There are two options (simple and fsdp). See [Parallelism Modes](#Parallelism-Modes) for a more detailed description
- [OPTIONAL] is an optional keyword for launch scripts using the apptainer method for installing containers on Frontier or the docker method for installing containers for running on DGX machines


## Datasets
We use two example datasets to test functionality of our code, their descriptions and instructions for downloading are below.
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
