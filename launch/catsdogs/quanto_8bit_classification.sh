#!/bin/bash
#SBATCH -A lrn075
#SBATCH -J quanto_8bit_catsdogs
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00  # 30 minutes for fast testing
#SBATCH -q debug
#SBATCH -o quanto_8bit_catsdogs-%j.out
#SBATCH -e quanto_8bit_catsdogs-%j.out

# QUANTO 8-BIT QUANTIZED CATSDOGS TESTING
# Frontier Supercomputer - AMD MI250X Optimized
# Target: Fast testing with quanto 8-bit quantization

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

echo "Quanto 8-bit Quantized CatsDogs Testing Starting..."
echo "Job ID: $JOBID"
echo "Nodes: $JOBSIZE"
echo "Total GPUs: $((JOBSIZE*8))"
echo "Target: Fast testing with quanto 8-bit on $(($JOBSIZE*8)) AMD MI250X GPUs"

# Load Frontier environment optimized for quanto quantization
#eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
#conda activate forge-vit
. ~/.bashrc_075

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

# ROCm optimizations (same as classification_simple.sh)
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

# Suppress MIOpen warnings completely
export MIOPEN_LOG_LEVEL=0
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_DISABLE_LOGGING=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_LAUNCH_BLOCKING=0

export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Environment loaded - ROCm quanto optimizations enabled"
echo "Starting Quanto 8-bit quantized CatsDogs testing on $((SLURM_JOB_NUM_NODES*8)) GPUs..."

# Launch quanto quantized training with testing configuration
# Redirect MIOpen warnings to /dev/null
time srun -n $((SLURM_JOB_NUM_NODES*8)) \
python ../../training_scripts/train_class_simple_torchDataloader.py \
--quantization \
--quantization-bits 8 \
--quantize-weights \
--rocm-optimizations \
--performance-mode normal \
../../configs/catsdogs/classification/quanto_8bit_config.yaml 2> >(grep -v "MIOpen(HIP): Warning" >&2)

echo "Quanto 8-bit quantized CatsDogs testing completed!"
echo "Next: Try 4-bit quantization for even more compression!"
