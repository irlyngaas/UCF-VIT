#!/bin/bash
#SBATCH -A STF006
#SBATCH -J inference_pred_2d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -p batch
##SBATCH -p extended
#SBATCH -o logs/inference_pred_2d-%j.out
#SBATCH -e logs/inference_pred_2d-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


#eval "$(/lustre/orion/stf006/proj-shared/nafi/miniforge3/bin/conda shell.bash hook)"
eval "$(/lustre/orion/stf006/proj-shared/irl1/MINI_CLEAN/bin/conda shell.bash hook)"
#conda activate forge-vit
conda activate sst

#module load PrgEnv-gnu
#module load gcc/12.2.0

#module load rocm/6.2.4
module load rocm/7.13.0

export MIOPEN_DISABLE_CACHE=1
#export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

# srun python ../../dev_scripts/inference_pred_fsdp.py ../../configs/sst/pred/base_config.yaml 28.040000 z
#srun python ../../dev_scripts/inference_pred_fsdp_2d.py ../../configs/sst/pred/base_config_2d.yaml 15.000000 z
srun python ../../dev_scripts/inference_pred_fsdp_2d.py ../../configs/sst/pred/base_config_2d_uniform.yaml 15.000000 z
