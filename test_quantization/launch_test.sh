#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J test_quantization
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00  # 10 minutes should be enough for testing
#SBATCH -q debug
#SBATCH -o test_quantization-%j.out
#SBATCH -e test_quantization-%j.out

# CUSTOM INT8 QUANTIZATION TEST
# Frontier Supercomputer - AMD MI250X
# Testing basic quantization functionality

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID

echo "Custom INT8 Quantization Test Starting..."
echo "Job ID: $JOBID"
echo "Node: $SLURM_NODEID"
echo "GPU: 1 AMD MI250X"
echo "Purpose: Test basic quantization functions"

# Load Frontier environment
. ~/.bashrc_075

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

# ROCm optimizations
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# Suppress MIOpen warnings
export MIOPEN_LOG_LEVEL=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_DISABLE_LOGGING=1
export HIP_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Environment loaded - ROCm optimizations enabled"
echo "Starting custom INT8 quantization test..."

# Run our custom quantization test
time python test_int8_basic.py

echo "Custom INT8 quantization test completed!"
echo "Check output for test results and performance metrics."