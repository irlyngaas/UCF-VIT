#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J native_int8_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -o native_int8-%j.out
#SBATCH -e native_int8-%j.out

# NATIVE INT8 ACCELERATION TEST
# Frontier Supercomputer - AMD MI250X
# Testing true hardware INT8 acceleration

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID

echo "Native INT8 Acceleration Test Starting..."
echo "Job ID: $JOBID"
echo "Node: $SLURM_NODEID"
echo "GPU: 1 AMD MI250X"
echo "Purpose: Test true INT8 hardware acceleration"

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
echo "Starting native INT8 acceleration test..."

# Run native INT8 test
time python test_native_int8.py

echo "Native INT8 acceleration test completed!"
echo "Check output for true hardware acceleration results."