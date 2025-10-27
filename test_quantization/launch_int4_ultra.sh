#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J rocblas_int4_ultra
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -q debug
#SBATCH -o rocblas_int4_ultra-%j.out
#SBATCH -e rocblas_int4_ultra-%j.out

# INT4 ULTRA-OPTIMIZED TEST
# Frontier Supercomputer - AMD MI250X
# Target: <10% overhead vs INT8

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID

echo "INT4 Ultra-Optimized Test Starting..."
echo "Job ID: $JOBID"
echo "Node: $SLURM_NODEID"
echo "GPU: 1 AMD MI250X"
echo "Target: <10% overhead vs INT8 baseline"

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

# Use module's hipcc
export PATH=$ROCM_PATH/bin:$PATH

echo "Environment loaded - ROCm path: $ROCM_PATH"
echo "Building INT4 ultra-optimized test..."

# Build the test
make clean
make int4-ultra ROCM_PATH=$ROCM_PATH

if [ $? -eq 0 ]; then
    echo "Build successful! Running INT4 ultra-optimized test..."
    echo ""
    echo "Testing 3 optimization techniques:"
    echo "1. Vectorized register unpacking (128-bit loads)"
    echo "2. Warp-cooperative unpacking (shared memory)"
    echo "3. Template-based compile-time optimization"
    echo ""
    
    # Run the ultra test
    time ./rocblas_int4_ultra_optimized
    
    echo ""
    echo "INT4 ultra-optimization test completed!"
    echo "Success criteria: Any kernel with <10% overhead"
else
    echo "Build failed!"
    exit 1
fi