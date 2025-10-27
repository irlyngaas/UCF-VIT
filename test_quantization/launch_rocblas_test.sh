#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J rocblas_int8_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o rocblas_int8-%j.out
#SBATCH -e rocblas_int8-%j.out

# ROCBLAS INT8 DIRECT TEST
# Frontier Supercomputer - AMD MI250X
# Testing rocBLAS gemm_ex with native INT8

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID

echo "rocBLAS INT8 Direct Test Starting..."
echo "Job ID: $JOBID"
echo "Node: $SLURM_NODEID"
echo "GPU: 1 AMD MI250X"
echo "Purpose: Test rocBLAS gemm_ex native INT8 acceleration"

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
echo "Building rocBLAS INT8 test..."

# Build the test (we're already in the right directory)
make clean
make ROCM_PATH=$ROCM_PATH

if [ $? -eq 0 ]; then
    echo "Build successful! Running test..."
    echo ""
    
    # Run the test
    time ./rocblas_int8_test
    
    echo ""
    echo "rocBLAS INT8 test completed!"
    echo "Check output for native INT8 acceleration results."
else
    echo "Build failed!"
    exit 1
fi