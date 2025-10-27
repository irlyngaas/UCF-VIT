#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J rocblas_int4_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o rocblas_int4_gpu-%j.out
#SBATCH -e rocblas_int4_gpu-%j.out

# INT4 GPU UNPACKING TEST
# Frontier Supercomputer - AMD MI250X
# Testing INT4 unpacking on GPU + MFMA INT8

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID

echo "INT4 GPU Unpacking Test Starting..."
echo "Job ID: $JOBID"
echo "Node: $SLURM_NODEID"
echo "GPU: 1 AMD MI250X"
echo "Purpose: Test INT4 unpacking on GPU + MFMA INT8"

# Load Frontier environment
. ~/.bashrc_075

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

# Detect ROCm path
if [ -d "/opt/rocm-6.2.4" ]; then
    export ROCM_PATH=/opt/rocm-6.2.4
elif [ -d "/opt/rocm" ]; then
    export ROCM_PATH=/opt/rocm
else
    echo "ROCm not found!"
    exit 1
fi

export HIP_PLATFORM=amd
export HIP_VISIBLE_DEVICES=0

# ROCm optimizations
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

export OMP_NUM_THREADS=7

# Use module's hipcc instead of ROCm's
export PATH=$ROCM_PATH/bin:$PATH

echo "Environment loaded - ROCm path: $ROCM_PATH"
echo "Building INT4 GPU unpacking test..."

# Build the test
make clean
make int4-gpu ROCM_PATH=$ROCM_PATH

if [ $? -eq 0 ]; then
    echo "Build successful! Running INT4 GPU unpacking test..."
    echo ""
    
    # Run the INT4 GPU test
    time ./rocblas_int4_gpu_unpack
    
    echo ""
    echo "INT4 GPU unpacking test completed!"
    echo "Key metrics to look for:"
    echo "1. Overhead vs INT8 should be < 20% for success"
    echo "2. Memory reduction: 2x vs INT8, 8x vs FP32"
    echo "3. Compare basic vs optimized kernel performance"
else
    echo "Build failed!"
    exit 1
fi