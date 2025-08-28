
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from math import sqrt
from tqdm import tqdm


def scan_chunk(indices, inp_dir, files):
    # Return tuple: (n, sum, sumsq)
    n_total = 0
    s_total = 0.0
    ss_total = 0.0
    for i in indices:
        arr = np.load(os.path.join(inp_dir, files[i]), mmap_mode='r')
        # Summations in float64 for numerical stability
        s = arr.sum(dtype=np.float32)
        ss = np.square(arr, dtype=np.float32).sum(dtype=np.float32)
        n = arr.size
        n_total += n
        s_total += s
        ss_total += ss
    return n_total, s_total, ss_total

def global_mean_std(inp_dir, files, num_chunks=20, max_workers=None):
    # Build (almost) equal chunks of indices
    N = len(files)
    if N == 0:
        raise ValueError("No files provided.")
    chunk_size = max(1, N // num_chunks)
    chunks = [list(range(i, min(i + chunk_size, N))) for i in range(0, N, chunk_size)]
    # Parallel scan
    worker = partial(scan_chunk, inp_dir=inp_dir, files=files)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for out in tqdm(ex.map(worker, chunks), total=len(chunks)):
            results.append(out)
    # Combine totals
    n_total = sum(r[0] for r in results)
    s_total = sum(r[1] for r in results)
    ss_total = sum(r[2] for r in results)
    mean = s_total / n_total
    # population variance: ss_total/n_total - mean^2
    # convert to unbiased sample variance with Bessel's correction:
    var = (ss_total - n_total * mean * mean) / (n_total - 1)
    std = sqrt(var) if var > 0 else 0.0
    return mean, std

Base_fldr ="/lustre/fs0/scratch/ziabariak/data_LDRD/"
inp_dirs  = ["NeutronCT_Concrete_256x256x256","XCT_Concrete_256x256","TRISO_Data/Triso_Data_256x256x256","Breaker_data/256x256x256","AM_ScattCorrected_data/AM_ScatterCorrected_Data_256x256x256"]
all_means = []#8787.113,5714.432,7081.9893,7925.5483,17900.906 
all_stds = []#2271.0200351383955,838.6252142643935,2228.4333061592847,3702.5231937153344,4606.49975577987
for inpf in inp_dirs[:1]:
    inp_dir = os.path.join(Base_fldr,inpf)
    files = [ f for f in sorted(os.listdir(inp_dir)) if f.endswith('npy')]
    out_dir = inp_dir +"_standardized"
    os.makedirs(out_dir,exist_ok=True)
    mean_val, std_val = global_mean_std(inp_dir, files, num_chunks=20, max_workers=8)
    print(mean_val, std_val) 
    all_means.append(mean_val)
    all_stds.append(std_val)
    for f in files:
        x=np.load(os.path.join(inp_dir,f))
        # print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
        x = (x-mean_val)/std_val
        np.save(os.path.join(out_dir,f),x)
        # print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
print(f"mean_vales are:{all_means}; and std_values are: {all_stds}")
## inp_dir =  '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth/'
## out_dir = '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth_standardized/'
## # #0.27842160105071867 0.03345132856836802

# inp_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files'
# out_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files_standardized'
# os.makedirs(out_dir,exist_ok=True)
# print(mean_val, std_val) # (np.float64(15202.856814677318), 5063.825531473543)
