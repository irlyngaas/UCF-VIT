
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import numpy as np
from math import sqrt
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from math import ceil
from typing import Iterable, List

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

def _standardize_many(fnames: List[str], inp_dir: str, out_dir: str,
                      mean_val: float, std_val: float,
                      out_dtype=np.float32):
    if std_val == 0:
        raise ValueError("std_val is zero; cannot standardize.")
    inp = Path(inp_dir); outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for fname in fnames:
        # 256x256x256 float32 (~64 MB) -> process in one go
        src = np.load(inp / fname, mmap_mode='r')
        dst = np.lib.format.open_memmap(outp / fname, mode='w+',
                                        dtype=out_dtype, shape=src.shape)
        dst[...] = ((src - mean_val) / std_val).astype(out_dtype, copy=False)
        del dst  # flush to disk
    return len(fnames)

def _batched(iterable: Iterable[str], n: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk: break
        yield chunk

def standardize_files_massive(files: List[str],
                              inp_dir: str, out_dir: str,
                              mean_val: float, std_val: float,
                              max_workers: int = 32,
                              files_per_task: int = 8,
                              max_inflight_tasks: int = None):
    if max_inflight_tasks is None:
        max_inflight_tasks = 4 * max_workers  # keep queue bounded

    batches = _batched(files, files_per_task)
    total_tasks = ceil(len(files) / files_per_task)
    inflight = []

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Prime queue
        while len(inflight) < max_inflight_tasks:
            try:
                inflight.append(ex.submit(_standardize_many, next(batches),
                                          inp_dir, out_dir, mean_val, std_val))
            except StopIteration:
                break

        pbar = tqdm(total=total_tasks, desc="Standardizing (batched)")
        while inflight:
            # Wait for the first to complete (avoids busy-wait)
            fut = inflight.pop(0)
            _ = fut.result()  # raises on error
            completed += 1
            pbar.update(1)
            # Top up queue
            while len(inflight) < max_inflight_tasks:
                try:
                    inflight.append(ex.submit(_standardize_many, next(batches),
                                              inp_dir, out_dir, mean_val, std_val))
                except StopIteration:
                    break
        pbar.close()
    return completed, total_tasks



def _standardize_one(fname, inp_dir, out_dir, mean_val, std_val,
                     out_dtype=np.float32, block_elems=8_000_000):
    """
    Standardize one .npy file: (x - mean) / std
    - Reads with mmap
    - Writes with open_memmap
    - Processes in blocks along axis 0
    """
    in_path  = Path(inp_dir) / fname
    out_path = Path(out_dir) / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if std_val == 0:
        raise ValueError("std_val is zero; cannot standardize.")

    src = np.load(in_path, mmap_mode='r')         # read-only, zero-copy mapping
    shape = src.shape
    if len(shape) == 0:
        # Scalar .npy
        out = np.lib.format.open_memmap(out_path, mode='w+', dtype=out_dtype, shape=())
        out[...] = (float(src) - mean_val) / std_val
        del out
        return fname

    # Create output memmap with same shape
    dst = np.lib.format.open_memmap(out_path, mode='w+', dtype=out_dtype, shape=shape)

    # Process in blocks along axis 0
    elems_per_row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    rows_per_block = max(1, block_elems // max(1, elems_per_row))

    for i in range(0, shape[0], rows_per_block):
        j = min(i + rows_per_block, shape[0])
        blk = src[i:j]  # still a view backed by mmap
        # compute -> cast (avoid extra copies)
        dst[i:j] = ((blk - mean_val) / std_val).astype(out_dtype, copy=False)

    # Ensure data is flushed
    del dst
    return fname

def standardize_files_parallel(inp_dir, out_dir, files, mean_val, std_val,
                               max_workers=100, out_dtype=np.float32, block_elems=8_000_000):
    """
    Standardize many .npy files in parallel (up to max_workers concurrently).
    """
    worker = partial(
        _standardize_one,
        inp_dir=inp_dir, out_dir=out_dir,
        mean_val=mean_val, std_val=std_val,
        out_dtype=out_dtype, block_elems=block_elems,
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Standardizing"):
            _ = fut.result()  # will raise if any worker failed
            completed += 1
    return completed



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

Base_fldr ="/lustre/fs0/scratch/ziabariak/"
inp_dirs  = ["ALL_256x256x256_data_Original"] #13988.486 6382.325281588208
# inp_dirs  = ["ALL_256x256x256_data"] #6.7299716e-08 1.0000006556508776
all_means = []  
all_stds = []

# Base_fldr ="/lustre/fs0/scratch/ziabariak/data_LDRD/"
# inp_dirs  = ["NeutronCT_Concrete_256x256x256","XCT_Concrete_256x256","TRISO_Data/Triso_Data_256x256x256","Breaker_data/256x256x256",
#              "AM_ScattCorrected_data/AM_ScatterCorrected_Data_256x256x256","XCT_Concrete_256x256_Z00730_XYZformat","Hex_Flower_FuelNozzle/256x256x256"]
# all_means = []#8787.113,5714.432,7081.9893,7925.5483,17900.906,15202.856,7518.2866  
# all_stds = []#2271.0200351383955,838.6252142643935,2228.4333061592847,3702.5231937153344,4606.49975577987,5063.825233951108,2537.3006325620936
for inpf in inp_dirs:
    inp_dir = os.path.join(Base_fldr,inpf)
    files = [ f for f in sorted(os.listdir(inp_dir)) if f.endswith('npy')]
    out_dir = inp_dir +"_Standardized"
    os.makedirs(out_dir,exist_ok=True)
    mean_val, std_val = global_mean_std(inp_dir, files, num_chunks=2800, max_workers=8)
    print(mean_val, std_val) 
    all_means.append(mean_val)#
    all_stds.append(std_val)
    completed, total = standardize_files_massive(
    files, inp_dir, out_dir, mean_val, std_val,
    max_workers=32,         # <= main throttle
    files_per_task=8,      # amortize overhead
    max_inflight_tasks=128  # ~4x workers
    )
    print(f"Finished {completed}/{total} tasks.")
    # n_done = standardize_files_parallel(inp_dir, out_dir, files, mean_val, std_val,
    #                                     max_workers=100, out_dtype=np.float32, block_elems=8_000_000)
    # print(f"Standardized {n_done} files.")
    # for f in files:
    #     x=np.load(os.path.join(inp_dir,f))
    #     # print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
    #     x = (x-mean_val)/std_val
    #     np.save(os.path.join(out_dir,f),x)
    #     # print(f"min,max,mean for {f} are: {np.min(x)},{np.max(x)},{np.mean(x)}")
print(f"mean_vales are:{all_means}; and std_values are: {all_stds}")


## inp_dir =  '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth/'
## out_dir = '/lustre/fs0/scratch/ziabariak/data_LDRD/XCT_NCT_Synth/Downsampled_128x128_128_beforeCrop/XCT_Concrete_32x32x32_Synth_standardized/'
## # #0.27842160105071867 0.03345132856836802

# inp_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files'
# out_dir = '/lustre/fs0/scratch/lyngaasir/DiffusiveINR_Data/XCT_Concrete_Z00730_650Files_standardized'
# os.makedirs(out_dir,exist_ok=True)
# print(mean_val, std_val) # (np.float64(15202.856814677318), 5063.825531473543)


# --- usage ---
# mean_val, std_val = ...  # from your previous global stats
# n_done = standardize_files_parallel(inp_dir, out_dir, files, mean_val, std_val,
#                                     max_workers=100, out_dtype=np.float32, block_elems=8_000_000)
# print(f"Standardized {n_done} files.")


# completed, total = standardize_files_massive(
#     files, inp_dir, out_dir, mean_val, std_val,
#     max_workers=48,         # <= main throttle
#     files_per_task=16,      # amortize overhead
#     max_inflight_tasks=192  # ~4x workers
# )
# print(f"Finished {completed}/{total} tasks.")