#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J quanto_4bit_catsdogs
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00  # 30 minutes for fast testing
#SBATCH -q debug
#SBATCH -o quanto_4bit_catsdogs-%j.out
#SBATCH -e quanto_4bit_catsdogs-%j.out

# QUANTO 4-BIT QUANTIZED CATSDOGS TESTING
# Frontier Supercomputer - AMD MI250X Optimized
# Target: Fast testing with quanto 4-bit quantization

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

echo "Quanto 4-bit Quantized CatsDogs Testing Starting..."
echo "Job ID: $JOBID"
echo "Nodes: $JOBSIZE"
echo "Total GPUs: $((JOBSIZE*8))"
echo "Target: Fast testing with quanto 4-bit on $(($JOBSIZE*8)) AMD MI250X GPUs"

# Load Frontier environment optimized for quanto quantization
#eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
#conda activate forge-vit
. ~/.bashrc_075

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

# ROCm optimizations for quanto quantization
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# Performance optimizations for testing
export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Enable ROCm quanto optimizations
export ROCM_QUANTIZATION_ENABLED=1
export HIP_LAUNCH_BLOCKING=0
export HSA_ENABLE_SDMA=1

# Memory optimizations for testing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1

echo "Environment loaded - ROCm quanto optimizations enabled"
echo "Starting Quanto 4-bit quantized CatsDogs testing on $((SLURM_JOB_NUM_NODES*8)) GPUs..."

# Launch quanto quantized training with testing configuration
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train_class_simple_torchDataloader.py \
--quantization \
--quantization-bits 4 \
--quantize-weights \
--quantize-activations \
--rocm-optimizations \
--performance-mode extreme_scale \
../../configs/catsdogs/classification/quanto_4bit_config.yaml

echo "Quanto 4-bit quantized CatsDogs testing completed!"
echo "Next: Try 2-bit quantization for maximum compression!"
