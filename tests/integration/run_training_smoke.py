"""Tier 3 integration smoke test: runs a tiny, fast version of every shipped
config through the real training_scripts/train.py entry point, against real
data, including a checkpoint-save-then-resume cycle.

This deliberately does NOT use pytest. Nesting a real multi-process
torch.distributed training launch inside a pytest process that's already
running under its own srun launch (as tests/distributed/ does for Tier 2)
would race the same "process group already initialized" class of bug that
was hit and fixed there. Instead, this script itself runs as a single,
plain process (see launch/tests/run_training_smoke.sh, which does NOT put it
under srun) and spawns its own independent `srun` subprocess per training
run -- each one a fully separate multi-process job step, with no nesting.

For each config under configs/**/base_config.yaml, this script:
  1. Writes a single smoke-test config to a job-scoped scratch dir: same data
     paths, tiling, and adaptive-patching settings as the real config (so the
     real data pipeline runs unmodified), but a tiny model (embed_dim=32,
     num_heads=2, depth=4 -- not 1, since UNETR's skip-connection indexing
     does depth // 4 and degenerates to duplicate indices below that),
     max_epochs=1, save_frequency=1, resume_from_checkpoint=False, a scratch
     checkpoint_path, and -- for configs using the iterative dataloader
     (dataloader.type == "iterative_dataloader", e.g. basic_ct/imagenet, but
     not catsdogs, which globs every file directly with no trimming knob) --
     narrowed dict_start_idx/dict_end_idx so only a small, fixed number of
     real files get read regardless of how large the real dataset actually
     is. This is computed dynamically (see compute_narrow_dict_idx) by
     calling the same UCF_VIT.utils.misc.process_root_dirs the real pipeline
     uses to get actual file counts, rather than guessing a fixed fraction
     that could round to 0 files for a modest dataset or barely help for a
     huge one like full ImageNet.
  2. Runs it via `srun -n <ntasks> python training_scripts/train.py <config>`
     and checks it exits 0 and actually wrote a rank-0 checkpoint file.
  3. Edits that *same* config file in place -- resume_from_checkpoint=True,
     checkpoint_filename="epoch_0" (the file the fresh run actually produced;
     see src/UCF_VIT/training.py's save_checkpoint vs.
     src/UCF_VIT/model/utils.py's get_model resume-loading logic, which is
     why checkpoint_filename is a manual "which epoch to resume from"
     selector rather than an auto-detected name), max_epochs=2 -- mirroring
     how a real user actually resumes: manually flipping fields in their one
     config, not maintaining a separate "resume" config.
  4. Runs that and checks it exits 0 too -- exercising get_model's and
     load_optimizer_scheduler_from_checkpoint's resume path against a real
     checkpoint, not just the group-membership-level Tier 2 tests.
  5. Sets resume_from_checkpoint back to False in the config file (the
     scratch dir gets deleted afterward regardless, but this leaves the file
     itself in a clean, reusable state if inspected before cleanup).

Usage (from anywhere, inside an sbatch job with GPUs already allocated):
    python tests/integration/run_training_smoke.py [--ntasks N] [--timeout SECONDS] [--min-files N] [config ...]

With no config arguments, runs every configs/**/base_config.yaml. Prints a
per-config, per-stage PASS/FAIL/TIMEOUT summary at the end and exits nonzero
if anything failed.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "training_scripts", "train.py")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from UCF_VIT.utils.misc import process_root_dirs  # noqa: E402 (needs REPO_ROOT/src on sys.path first)

TINY_MODEL_OVERRIDES = {
    # embed_dim must be divisible by 12: get_2d_sincos_pos_embed needs
    # embed_dim % 4 == 0 (it halves embed_dim, then get_1d_sincos_pos_embed_
    # from_grid needs that half to itself be even) and get_3d_sincos_pos_embed
    # needs embed_dim % 6 == 0 (same halving logic, split three ways instead
    # of two) -- LCM(4, 6) = 12. Also needs to be divisible by num_heads.
    "embed_dim": 24,
    "num_heads": 2,
    "depth": 4,
}

# Default target number of real files to keep per dataset key (or, for
# imagenet, per data-parallel bucket) after narrowing dict_start_idx/
# dict_end_idx -- see compute_narrow_dict_idx. Comfortably above a typical
# data_par_size (8) and typical batch_size (2-32 across the shipped
# configs), so at least one real batch per rank should still be possible.
DEFAULT_MIN_FILES = 64


def discover_configs():
    return sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "**", "base_config.yaml"), recursive=True))


def config_slug(config_path):
    # e.g. configs/basic_ct/unetr/base_config.yaml -> basic_ct-unetr
    rel = os.path.relpath(config_path, os.path.join(REPO_ROOT, "configs"))
    return os.path.dirname(rel).replace(os.sep, "-")


class NoRealDataFoundError(Exception):
    """Raised when process_root_dirs finds zero real files for a dataset key.

    Almost always means dict_root_dirs points at a stale/wrong/empty path in
    the config -- not a bug in this driver or in UCF_VIT. Raised early
    (before ever launching srun) specifically so callers can fail fast with
    a clear message instead of burning GPU allocation time on a training run
    that's guaranteed to crash confusingly deep inside
    calculate_load_balancing_on_the_fly with a ZeroDivisionError.
    """


def compute_narrow_dict_idx(conf, min_files):
    """Computes tight dict_start_idx/dict_end_idx overrides from real file counts.

    Only meaningful for conf["dataloader"]["type"] == "iterative_dataloader"
    (e.g. basic_ct, imagenet) -- "dataloader"-type datasets like catsdogs glob
    every file directly in train.py's main(), with no start/end-idx trimming
    mechanism, so this is a no-op for those.

    Calls the same UCF_VIT.utils.misc.process_root_dirs the real pipeline
    uses to find out how many real files actually exist per dataset key,
    rather than guessing a fixed fraction: a fixed fraction either rounds to
    0 files for a modest-sized dataset, or barely reduces anything for a huge
    one like full ImageNet.

    Args:
        conf: A parsed (real, un-tinied) config dict, as loaded from YAML.
            Needs conf["data"]["dataset"]/["dict_root_dirs"] and
            conf["parallelism"]["fsdp_size"]/["simple_ddp_size"] (to derive
            data_par_size the same way process_root_dirs' caller does).
        min_files: Target number of real files to keep per dataset key (or,
            for imagenet, per data-parallel bucket -- process_root_dirs
            partitions imagenet into one bucket per data-parallel rank, and
            dict_start_idx/dict_end_idx["imagenet"] applies identically to
            every bucket, so this sizes against the *smallest* bucket to
            guarantee every bucket keeps at least min_files).

    Returns:
        A dict {key: fraction in (0, 1]} to use as both dict_end_idx and (with
        every value replaced by 0.0) dict_start_idx, or None if this config
        doesn't use the iterative dataloader.

    Raises:
        NoRealDataFoundError: If process_root_dirs finds zero real files for
            any dataset key (or none at all) -- almost always a stale/wrong
            dict_root_dirs path in the config, not a code bug.
    """
    if conf["dataloader"]["type"] != "iterative_dataloader":
        return None

    dataset = conf["data"]["dataset"]
    dict_root_dirs = conf["data"]["dict_root_dirs"]
    data_par_size = conf["parallelism"]["fsdp_size"] * conf["parallelism"]["simple_ddp_size"]

    dict_lister_trains = process_root_dirs(dataset, dict_root_dirs, data_par_size)

    empty_keys = [k for k, v in dict_lister_trains.items() if len(v) == 0]
    if not dict_lister_trains or empty_keys:
        bad_keys = empty_keys or list(dict_root_dirs.keys())
        paths = {k: dict_root_dirs.get(k, "<no matching dict_root_dirs entry>") for k in bad_keys}
        raise NoRealDataFoundError(
            f"No real files found for dataset={dataset!r}, key(s) {bad_keys} "
            f"under dict_root_dirs={paths}"
        )

    def frac_for(count):
        return min(1.0, min_files / count) if count > 0 else 1.0

    if dataset == "imagenet":
        smallest = min(len(v) for v in dict_lister_trains.values())
        return {"imagenet": frac_for(smallest)}

    return {k: frac_for(len(v)) for k, v in dict_lister_trains.items()}


def make_smoke_config(base_config_path, scratch_dir, min_files=DEFAULT_MIN_FILES):
    """Loads a real config and writes a tiny/fast, freshly-training smoke-test variant.

    Only overrides model size, epoch count, checkpoint location,
    resume_from_checkpoint, and (see compute_narrow_dict_idx)
    dict_start_idx/dict_end_idx. Everything else -- data paths, tiling,
    adaptive patching, batch size, load-balancing settings -- is left exactly
    as in the real config.

    Args:
        base_config_path: Path to the real config YAML to base this on.
        scratch_dir: Job-scoped scratch directory to write the checkpoint and
            this generated config file into.
        min_files: Passed through to compute_narrow_dict_idx.

    Returns:
        Path to the written smoke-test config file.

    Raises:
        NoRealDataFoundError: See compute_narrow_dict_idx.
    """
    with open(base_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf["model"].update(TINY_MODEL_OVERRIDES)

    conf["trainer"]["checkpoint_path"] = scratch_dir
    conf["trainer"]["save_frequency"] = 1
    conf["trainer"]["max_epochs"] = 1
    conf["trainer"]["resume_from_checkpoint"] = False

    narrow_end_idx = compute_narrow_dict_idx(conf, min_files)
    if narrow_end_idx is not None:
        conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
        conf["dataloader"]["dict_end_idx"] = narrow_end_idx

    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, "smoke.yaml")
    with open(out_path, "w") as f:
        yaml.dump(conf, f)
    return out_path


def set_resume(config_path, resume):
    """Edits an existing smoke-test config in place to toggle resume_from_checkpoint.

    Mirrors how a real user actually resumes training: manually flipping a
    field in their one config file, rather than a script silently maintaining
    a separate "resume" config alongside it.

    Args:
        config_path: Path to the smoke-test config file (as written by
            `make_smoke_config`) to edit in place.
        resume: If True, sets resume_from_checkpoint=True, checkpoint_filename
            to "epoch_0" (what the fresh run actually produced), and bumps
            max_epochs to 2 so at least one more epoch runs after resuming.
            If False, sets resume_from_checkpoint back to False ("changing it
            back"); checkpoint_filename/max_epochs are left as they were,
            since they're harmless once resume_from_checkpoint is off.
    """
    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf["trainer"]["resume_from_checkpoint"] = resume
    if resume:
        conf["trainer"]["checkpoint_filename"] = "epoch_0"
        conf["trainer"]["max_epochs"] = 2

    with open(config_path, "w") as f:
        yaml.dump(conf, f)


def run_training(config_path, ntasks, timeout):
    """Runs training_scripts/train.py against `config_path` as an srun subprocess.

    Returns:
        A dict with "returncode" (None on timeout) and "log" (combined
        stdout+stderr, tail-truncated).
    """
    cmd = ["srun", "-n", str(ntasks), "python", TRAIN_SCRIPT, config_path, "--launcher", "slurm"]
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
        log = result.stdout + result.stderr
        return {"returncode": result.returncode, "log": log}
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "")
        return {"returncode": None, "log": log}


def rank0_checkpoint_exists(scratch_dir):
    return os.path.isfile(os.path.join(scratch_dir, "epoch_0_rank_0.ckpt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="Specific config file(s) to test; default is every configs/**/base_config.yaml")
    parser.add_argument("--ntasks", type=int, default=8, help="Number of srun tasks per training run (default 8, matching launch/*/*.sh)")
    parser.add_argument("--timeout", type=int, default=150, help="Per-run timeout in seconds (default 150)")
    parser.add_argument("--min-files", type=int, default=DEFAULT_MIN_FILES, help=f"Target real files to keep per dataset key after narrowing dict_start_idx/dict_end_idx (default {DEFAULT_MIN_FILES})")
    args = parser.parse_args()

    config_paths = args.configs if args.configs else discover_configs()
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_root = f"/tmp/{job_id}/checkpoint_smoke_test"

    results = []
    for config_path in config_paths:
        slug = config_slug(config_path)
        scratch_dir = os.path.join(scratch_root, slug)
        print(f"\n{'='*80}\n{slug} ({config_path})\n{'='*80}", flush=True)

        try:
            smoke_config = make_smoke_config(config_path, scratch_dir, min_files=args.min_files)
        except NoRealDataFoundError as e:
            # Fail fast, before ever spending GPU allocation time on a run
            # that's guaranteed to crash confusingly deep inside
            # calculate_load_balancing_on_the_fly with a ZeroDivisionError.
            fresh_status = "FAIL (no real data found)"
            fresh_elapsed = 0.0
            print(f"[{slug}] {e}", flush=True)
            print(f"[{slug}] fresh run: {fresh_status} (srun never launched)", flush=True)
            results.append({
                "slug": slug, "config": config_path,
                "fresh_status": fresh_status, "fresh_elapsed": fresh_elapsed,
                "resume_status": "SKIPPED (fresh run didn't produce a checkpoint)", "resume_elapsed": None,
            })
            shutil.rmtree(scratch_dir, ignore_errors=True)
            continue

        print(f"[{slug}] fresh run: srun -n {args.ntasks} python train.py {smoke_config}", flush=True)
        t0 = time.time()
        fresh = run_training(smoke_config, args.ntasks, args.timeout)
        fresh_elapsed = time.time() - t0

        fresh_status = "TIMEOUT" if fresh["returncode"] is None else ("PASS" if fresh["returncode"] == 0 else "FAIL")
        if fresh_status != "PASS":
            print(fresh["log"][-4000:], flush=True)
        elif not rank0_checkpoint_exists(scratch_dir):
            fresh_status = "FAIL (no checkpoint written)"
            print(fresh["log"][-4000:], flush=True)
        print(f"[{slug}] fresh run: {fresh_status} ({fresh_elapsed:.0f}s)", flush=True)

        resume_status = "SKIPPED (fresh run didn't produce a checkpoint)"
        resume_elapsed = None
        if fresh_status == "PASS":
            set_resume(smoke_config, resume=True)
            print(f"[{slug}] resume run: srun -n {args.ntasks} python train.py {smoke_config}", flush=True)
            t0 = time.time()
            resume = run_training(smoke_config, args.ntasks, args.timeout)
            resume_elapsed = time.time() - t0

            resume_status = "TIMEOUT" if resume["returncode"] is None else ("PASS" if resume["returncode"] == 0 else "FAIL")
            if resume_status != "PASS":
                print(resume["log"][-4000:], flush=True)
            print(f"[{slug}] resume run: {resume_status} ({resume_elapsed:.0f}s)", flush=True)

            set_resume(smoke_config, resume=False)

        results.append({
            "slug": slug,
            "config": config_path,
            "fresh_status": fresh_status,
            "fresh_elapsed": fresh_elapsed,
            "resume_status": resume_status,
            "resume_elapsed": resume_elapsed,
        })

        shutil.rmtree(scratch_dir, ignore_errors=True)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    ok = True
    for r in results:
        fresh_time = f"{r['fresh_elapsed']:.0f}s" if r["fresh_elapsed"] is not None else "-"
        resume_time = f"{r['resume_elapsed']:.0f}s" if r["resume_elapsed"] is not None else "-"
        print(f"{r['slug']:30s} fresh={r['fresh_status']:8s}({fresh_time:>6s})  resume={r['resume_status']:8s}({resume_time:>6s})")
        if r["fresh_status"] != "PASS" or r["resume_status"] != "PASS":
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
