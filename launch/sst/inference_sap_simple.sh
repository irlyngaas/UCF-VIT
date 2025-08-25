#!/bin/bash
#SBATCH -A stf006
#SBATCH -J inference_sap
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -o inference_sap-%j.out
#SBATCH -e inference_sap-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536


#source /lustre/orion/proj-shared/stf006/irl1/conda/bin/activate
#conda activate /lustre/orion/stf006/proj-shared/irl1/vit-2.7-sst
eval "$(/lustre/orion/stf006/proj-shared/irl1/miniforge3/bin/conda shell.bash hook)"
conda activate forge-vit

module load PrgEnv-gnu
module load gcc/12.2.0

module load rocm/6.2.4

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ../../dev_scripts/train_pred_fsdp.py ../../configs/sst/pred/base_config.yaml

#srun -n 1 --ntasks-per-node=1 -c 1 python ../../dev_scripts/inference_pred_fsdp.py ../../configs/sst/pred/base_config.yaml 28.040000 z UNETR
srun python ../../dev_scripts/inference_sap_simple.py ../../configs/sst/sap/base_config3D.yaml 28.040000 z
