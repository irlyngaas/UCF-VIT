#!/bin/bash
#SBATCH -A stf006
#SBATCH -J quanto_8bit_vit
#SBATCH --nodes=64
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -o quanto_8bit-%j.out
#SBATCH -e quanto_8bit-%j.out

# 🚀 QUANTO 8-BIT QUANTIZED VISION TRANSFORMER 🚀
# Frontier Supercomputer - AMD MI250X Optimized
# Target: Peak FLOPS with Quanto 8-bit Vision Transformer Quantization

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

echo "🎯 Quanto 8-bit Quantized ViT Training Starting..."
echo "Job ID: $JOBID"
echo "Nodes: $JOBSIZE"
echo "Total GPUs: $((JOBSIZE*8))"
echo "Target: Quanto 8-bit QAT on $(($JOBSIZE*8)) AMD MI250X GPUs"

# Load Frontier environment optimized for quanto quantization
eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
conda activate forge-vit

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

# ROCm optimizations for quanto quantization
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# Performance optimizations for Gordon Bell
export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# Enable ROCm quanto optimizations
export ROCM_QUANTIZATION_ENABLED=1
export HIP_LAUNCH_BLOCKING=0
export HSA_ENABLE_SDMA=1

# Memory optimizations for large-scale quanto training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TORCH_CUDNN_V8_API_ENABLED=1

echo "🔥 Environment loaded - ROCm quanto optimizations enabled"
echo "Starting Quanto 8-bit quantized training on $((SLURM_JOB_NUM_NODES*8)) GPUs..."

# Launch quanto quantized training with maximum performance
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train_class_simple.py \
--quantization \
--quantization-bits 8 \
--quantize-weights \
--quantize-activations \
--rocm-optimizations \
--performance-mode extreme_scale \
../../configs/imagenet/classification/quanto_8bit_config.yaml

echo "✅ Quanto 8-bit quantized training completed!"
echo "Next target: 4-bit → 2-bit → 1-bit quantization for ultimate performance"
