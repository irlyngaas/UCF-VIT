import os
import subprocess

def master_from_host(host):
    MACHINE = os.environ["MACHINE"]
    if MACHINE == "FRONTIER":
        get_master = "ssh " + host + " hostname -I"
        master_addr = subprocess.check_output(get_master, shell=True)
        master_addr = master_addr.decode().split()[0]
    else: # MACHINE == "DGX"
        get_master = "getent hosts " + host
        master_addr = subprocess.check_output(get_master, shell=True)
        master_addr = master_addr.decode().split()[0]
    print("master address =", master_addr)
    return master_addr

def read_node_list():
    MACHINE = os.environ["MACHINE"]
    node_list = os.environ['SLURM_NODELIST']
    nodes = []
    if MACHINE == "FRONTIER":
        node_subsets = node_list[9:-1].split(",")
        for subset in node_subsets:
            if "-" in subset:
                start, end = subset.split("-")
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    leading_zeros = "".join(["0"] * (5 - len(str(i))))
                    nodes.append(f"frontier{leading_zeros}{i}")
            else:
                nodes.append(f"frontier{subset}")
        nodes_string = ",".join(nodes)
    elif MACHINE == "DGX":
        node_subsets = node_list[4:-1].split(",")
        for subset in node_subsets:
            if "-" in subset:
                start, end = subset.split("-")
                start, end = int(start), int(end)
                for i in range(start, end + 1):
                    leading_zeros = "".join(["0"] * (3 - len(str(i))))
                    nodes.append(f"gpu{leading_zeros}{i}")
            else:
                nodes.append(f"gpu{subset}")
        nodes_string = ",".join(nodes)
    return nodes, nodes_string


def read_job_node_list(job_id, job_nodes=None):
    if job_nodes is None:
        nodes, nodes_string = read_node_list()
        print(nodes, nodes_string)
        total_nodes = len(nodes)
        start = (job_id * 4) % total_nodes
        end = start + 4

        job_nodes_string = ",".join(nodes[start : end])
        job_master = master_from_host(nodes[start])
    else:
        job_master = master_from_host(job_nodes[0])
        job_nodes_string = ",".join(job_nodes)

    return job_nodes_string, job_master

def create_launch_command(prefix, params, job_id, job_nodes=None, DEEPHYPER_LOG_DIR="."):

    print("job_id original:", job_id)   
    job_id = int(job_id.split(".")[1])

    YAML_FILE = " ".join([os.environ["CONFIG_FILE"]])
    
    lr = float(params["lr"])
    beta_1 = float(params["beta_1"])
    beta_2 = float(params["beta_2"])
    weight_decay = float(params["weight_decay"])
    eta_min = float(params["eta_min"])
    batch_size = int(params["batch_size"])    

    ARGS = " ".join([
        f"--lr {lr}",
        f"--beta_1 {beta_1}",
        f"--beta_2 {beta_2}",
        f"--weight_decay {weight_decay}",
        f"--eta_min {eta_min}",
        f"--batch_size {batch_size}",
        
    ])

    python_exe = "python"
    python_script = os.environ["TRAINING_SCRIPT"]
    MACHINE = os.environ["MACHINE"]
    walltime = os.environ["WALLTIME"]

    print("job_id =", job_id)
    job_nodes, job_master = read_job_node_list(job_id, job_nodes)
    SLURM_ARGS = f"--nodelist {job_nodes} -t {walltime} --error {DEEPHYPER_LOG_DIR}/error_{job_id}.txt"
    ORBIT_ARGS = YAML_FILE + " " + ARGS + f" --master_addr {job_master} --job_id {job_id}"
    
    command = f"{prefix} {SLURM_ARGS} {python_exe} {python_script} {ORBIT_ARGS}"
    return command
