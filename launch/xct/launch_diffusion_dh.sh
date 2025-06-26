#!/bin/bash
#SBATCH -A LRN036
#SBATCH -J dh
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -o dh-%j.out
#####SBATCH -e flash-%j.error
#SBATCH -q debug

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

source /lustre/orion/proj-shared/stf006/irl1/conda/bin/activate
conda activate /lustre/orion/stf006/proj-shared/irl1/vit-2.7

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/6.2.4

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
#export PYTHONPATH=$PWD/../src:$PYTHONPATH

#export ORBIT_USE_DDSTORE=0

#export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

export NNODES=$SLURM_JOB_NUM_NODES
export NTOTGPUS=$(( $NNODES * 8 ))
export NGPUS_PER_TRAINING=8
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRAINING ))
export OMP_NUM_THREADS=4

export TRAINING_SCRIPT="../../dev_scripts/train_diffusion_fsdp_dh.py"
export CONFIG_FILE="../../configs/xct/diffusion/base_config_dh.yaml"
export WALLTIME="1:00:00"
export MACHINE="FRONTIER"

export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR
export DEEPHYPER_DB_HOST=$HOST

sleep 5
echo "Doing something"
python ../../dev_scripts/main_dh_centralized.py 
