# UCF-VIT
# Goal
The purpose of this codebase is to provide a **Uniform Coding Framework (UCF)** for the development of efficient computing and parallelism techniques for use in large scale **Vision Transformer (VIT)** based models. The intention is to provide the building blocks and utilities for using these different parallelism techniques in fashion that they can be easily integrated to use with various types of scientific data. We provide various different end to end examples for different computer vision tasks using two example datasets. We provide various different options so that the integration of new datasets can be done with a number of different strategies. We also provide various advanced techniques that we have developed for the specific use case of efficient computing with large scientific datasets.

## Architectures
Currently we provide 5 different architecutres **(VIT, MAE, UNETR, SAP, VIT-DIFFUSION)**, all of which use the same VIT encoder, but a different decoder architecture depending on the task being trained.

### VIT
### MAE
### UNETR
### SAP
### VIT-DIFFUSION

## Parallelism Modes
All of these architectures exist in 2 independent sub-folders, simple and fsdp, for which we separate the network architecture code into what we call modes. The choice of mode to be used will depend on the types of advanced parallelism and computing techniques needed for the model being trained.  The first `src/UCF_VIT/simple`, provides a simplified version for training in Distributed Data Parallel (DDP) fashion only. The second `src/UCF_VIT/fsdp`, provides a more complex version with different parallel training techniques. This includes options for training with a combination of Sharded Data Parallelism , DDP, and Tensor Parallelism. These parallelisms are all integrated via the HSTOP method from [cite]. Both modes can be used with the same data loading module and load balancing scripts provided. While the training done within the simple mode can be done with the correct choice of options in the fsdp mode, the purpose of keeping the simple mode is 1) to provide an entry point for new users and developers to add new architectures without the intricacies of the advanced features and 2) to provide a simple reference point to compare with when new innovations are added in order to test how they interact with the more complex parallelism methods.

### Building Blocks
The main building blocks for the VIT based archictectures are in the **Attention** and **Feed-forward** functions, provided in the Attention class and MLP class in `src/UCF_VIT/simple/building_blocks.py` and `src/UCF_VIT/fsdp/building_blocks.py`. We ask that you use these functions as is and do not modify them, as these common building blocks will be used across the different network architectures.

## Training Scripts
We provide several training scripts. These include all of the necessary things for running the main training loop, including utilities such as checkpoint loading and saving. We leave it to the user to implement their own validation and testing routines, based off the existing training examples, in order to more closely fit their needs. Training scripts are provided for each of the training architectures for the simple mode. To convert these scripts to use fsdp mode, look at the code changes made to go from `training_scripts/train_masked_simple.py` to `training_scripts/train_masked_fsdp.py`

## Data Loader
The dataloader we provide is a custom native pytorch iterative dataloader. For simplicity, we assume that we are receiving raw data files and we leave it to the user to normalize the data properly within in the dataloader module for training. The purpose of making this assumption of raw data file as input is 1) it removes the need for further data preprocessing scripts to be included in this repo, and 2) we want to provide an end-to-end worfklow from raw data to model prediction. If performing preprocessing during the dataloading phase is to computationally intensive, we recommend doing it offline and properly storing it in a manner that the dataloader module can handle and then removing the data normalization source code.

The dataloader is built in a fashion such that it can handle multiple different dataset directories at the same time. A dataset directory contains one or more raw data file. The purpose of being able to handle multiple dataset directories is 1) it provides flexible training where you can easily remove and add different datasets for the purposes of running experiments and 2) it allows for the integration of identifying properties from the different datasets that can potentially used for improved learning. For instance, with data that has multiple channels, e.g. images with (R,G,B), we are able to pass along the information on what variable the channel is from and use that information for more advanced network training. We then could utilize variable aggregration, described in a following section, to tokenize each channel separately.

This dataloader provides the flexibility to add a plethora of different options for customizing how the data is broken up for training. Since we are using a VIT, at least 2D data is expected, however, we have capability for both 2D and 3D spatial data. If desired, we have the utilities implemented to break given raw data into smaller tiled chunks. Also, we have a number of different options for how to tile this raw data.

## Load Balancing
In order for the dataloader to handle multiple datasets at the same time, the data needs to be spread out amongst the GPUs evenly. In the case where different datasets have different amounts and/or different sizes of images, it's very difficult to evenly spread this data amongst the GPUs evenly. We provide example load balancing scripts that for a given setting in a config file determines how the data should be split amongst a given set of N GPUs, in order to evenly balance the resources amongst the compute resources. The output from this script gives the necessary information to the dataloader in order to do this in a proper fashion.

## Config Arguments
We store the arguments for each individual run in a yaml file. This config file holds all of the arguments for defining the specific training, dataloading, parallelism, and checkpointing options for a particular architecture

### Required Arguments
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

4. Model Network
- default_vars: List of potential modalities available to be used as Variable Embeddings
- tile_size: Tile size to be used as input to the network
- patch_size: Patch Size to use when creating patch Embeddings Sequences for the network input
- embed_dim: Transformer Encoder Embedding Dimension
- depth: Number of Layers in Transformer Encoder
- num_heads: Number of Attention Heads for the Transformer Encoder Block in each Layer
- mlp_ratio: Ratio of mlp hidden dimension to embedding dimension in the Tranformer Encoder Blocks
- drop_path: Stochastic depth drop rate for layers in Transformer Blocks
- twoD: Variable for indicating two or three dimensionsal input, if False, three dimensional input. When tile_size is 3D, tiling creates 2D slices using the third dimension as the slicing dimension
- use_varemb: Whether to use Variable Embedding Tokens
- adaptive_patching: Whether to use adaptive patching, if False use traditional patching
- fixed_length: Fixed Length to make adaptive patching sequences
- separate_channels: Whether or not to separate channels and adaptively patch with different quadtrees

5. Data
- dataset: Variable for telling dataloader how to handle raw data and how to break up root directories into files within source code (Each dataset potentially needs it's own code to do this depending on the data type and layout of files)
- dict_root_dirs: Paths to directories with raw input data
- dict_start_idx: Starting indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use
- dict_end_idx: Ending indices ratio (between 0.0 and 1.0) to determine amount of files in directory to use
- dict_buffer_sizes: Buffer Size to use when filling iterative dataloader with prospective tiles for creation of batches
- num_channels_available: Number of Channels each dataset consists of
- num_channels_used: Number of Channels to use during training, currently no control of choosing modalities, but will cycle through the channels in order
- dict_in_variables: Variables corresponding to the different channels in the dataset, used in the dataloader to find corresponding correct values in the default_var_list. Needs to be in the correct order of the raw data files
- batch_size: Per GPU batch size
- num_workers: Number of data loader workers, should be set at 1 for now
- pin_memory: Variable to use pinned memory on GPU for dataloading
- single_channel: Variable for indicating that multiple modalities will be used, but the model will be fed with modalities separated into batches only containing a single modality
- tile_overlap: Amount of tile overlapping to use, takes decimal values, multiples tile_size by tile_overlap to determine step size. Use 0.0 for no overlapping
- use_all_data: Whether or not to use all data in dataloading. Including if tile size doesn't evenly split images. If tile size splits an image unevenly on last tile of a dimension go from last pixel backwards to get a full tile

6. Load Balancing
- batches_per_rank_epoch: How many batches per rank per epoch for a given dataset. Used to get a full epoch from the Dataset with largest value. If a dataset has less than the maximum. Reuse data to obtain enough data to run until the largest data has been fully trained on. Run "python utils/DATASET/preprocess_load_balancing.py CONFIG_FILE NUM_GPUS" to obtain
- dataset_group_list: How to split available GPUs amongst the available datasets. Run "python utils/DATASET/preprocess_load_balancing.py CONFIG_FILE NUM_GPUS" to obtain

### Optional Arguments (Depending on the Network Architecture)

## Advanced Features
### Adaptive Patching
### Variable Aggregation
### Composable Kernels

## Example Datasets
### Imagenet
### Basic_CT

## Integrating a new Dataset with the Dataloader Module
1. Name your dataset and use it in place of the dataset option the yaml config file
2. Write code to process file keys for the different datasets
- Add a new else if branch to 
### Setting up a new load balancing script


# Installation
Currently installation instructions are limited to Frontier only (the system we use for development and testing). Installation instructions and launch scripts for other machines will be added as they are implemented.

## Frontier
There are two options available for creating software environments. The Apptainer environment creation currently works only when adaptive_patching=True in the config file. Also UNETR won't work. Issues with missing ROCM packages. To be fixed later.
### Conda
Create Conda Environment from Scratch. Example below using options from the corresponding Apptainer definition files
```
PYTHON_VERSION=3.11
conda create -n vit python=${PYTHON_VERSION} -y
conda activate vit
ROCM_VERSION=6.2.4
TORCH_URL="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
TORCH_VERSION=2.6.0+rocm6.2.4
TORCHVISION_VERSION=0.21.0
TORCHAUDIO_VERSION=2.6.0

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

# Usage
1. Setup dataset for dataloader
2. Modify Launch Script
- Change project allocation to one you have access to `#SBATCH -A PROJXXX`.
- Set number of nodes you want to run with `#SBATCH --nodes=N`

3. Run Load Balancing Script

4. Modify Config Script with numbers for load balancing

5. Launch job `sbatch launch/DATASET/train_TASK_MODE.sh`
