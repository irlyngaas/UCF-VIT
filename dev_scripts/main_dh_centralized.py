import logging
import os
import subprocess
import sys
import dh_utils
#import parse
from deephyper.evaluator import profile

NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NGPUS_PER_TRAINING = int(os.environ["NGPUS_PER_TRAINING"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
DEEPHYPER_DB_HOST = os.environ["DEEPHYPER_DB_HOST"]


@profile
def run_distributed_training(job, dequed=None):
    
    #python_exe = sys.executable
    #python_script = os.path.join(os.path.dirname(__file__), "train.py")

    params = job.parameters

    MACHINE = os.environ["MACHINE"]
    
    if MACHINE == "FRONTIER":
        prefix = "".join([f"HOME=/tmp srun",
                          f" -u -N {NGPUS_PER_TRAINING//8} -n {NGPUS_PER_TRAINING}" ,
                          f" --ntasks-per-node=8",                      
                          ])
    else: #MACHINE == "DGX"
        #prefix = "".join([f"HOME=/tmp srun",
        prefix = "".join([f"srun",
                          f" -u -N {NGPUS_PER_TRAINING//8} -n {NGPUS_PER_TRAINING}" ,
                          f" --ntasks-per-node=8",                      
                          f" --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/lyngaasir/sqsh-files/0698614322576143+ucf-vit+25.05-upd2.sqsh ",
                          ])
#srun --mpi=pmix --container-mounts /lustre/fs0 --container-mount-home --container-image /lustre/fs0/scratch/lyngaasir/sqsh-files/0698614322576143+ucf-vit+25.05-upd2.sqsh python $HOME/UCF-VIT/dev_scripts/train_diffusion_fsdp.py $HOME/UCF-VIT/configs/xct/diffusion/base_config_dgx.yaml
    command = dh_utils.create_launch_command(prefix, params, job.id, dequed,DEEPHYPER_LOG_DIR) 
    print("Command = ", command, flush=True)

    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, text=True)
    print(result)
    output = result.stdout #.decode('utf-8')
    output = output.strip().split(" ")
    loss = output[output.index('MUST-0')+1]
        
    objective = float(loss)
 
    print(" objective", command, objective, flush=True)
    
    metadata = {
        "loss": float(loss),
    }

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":
    from deephyper.evaluator import Evaluator, ProcessPoolEvaluator, queued
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from dh_utils import read_node_list
    
    logging.basicConfig(
        filename=os.path.join(DEEPHYPER_LOG_DIR, f"deephyper.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - "
        + "%(message)s",
        force=True,
    )

    # Define the problem
    problem = HpProblem()
    problem.add_hyperparameter((1e-8, 1e-2, "log-uniform"), "lr", default_value=1e-6)
    problem.add_hyperparameter((0.1, 1., "log-uniform"), "beta_1", default_value=0.9)
    problem.add_hyperparameter((0.1, 1., "log-uniform"), "beta_2", default_value=0.9999)
    problem.add_hyperparameter((1e-5, 10, "log-uniform"), "weight_decay", default_value=1e-5)    
    problem.add_hyperparameter((1e-12, 1e-1, "log-uniform"), "eta_min", default_value=1e-9)
    problem.add_hyperparameter((1, 8, "log-uniform"), "batch_size", default_value=8)    

    # Create the node queue
    queue, _ = read_node_list()
    print("The queue:", queue, len(queue))
    print(NTOTGPUS, NGPUS_PER_TRAINING, NTOTGPUS // NGPUS_PER_TRAINING, len(queue))
    evaluator = queued(ProcessPoolEvaluator)(
        run_distributed_training,
        num_workers = NTOTGPUS // NGPUS_PER_TRAINING,
        queue = queue,
        queue_pop_per_task=1 #CHANGE THIS MANUALLY TO MATCH #NODES PER TRAINING        
    )
    
    search = CBO(
        problem,
        evaluator,
        acq_func="UCB",
        multi_point_strategy="qUCB",
        random_state=42,
        log_dir=DEEPHYPER_LOG_DIR,
        n_jobs=OMP_NUM_THREADS,
        initial_points=[problem.default_configuration],
    )

    results = search.search(max_evals=1000, timeout=14400)
