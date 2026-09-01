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
     real data pipeline runs unmodified), but a tiny model (embed_dim=24,
     num_heads=2, depth=4 -- not 1, since UNETR's skip-connection indexing
     does depth // 4 and degenerates to duplicate indices below that),
     max_epochs=1, save_frequency=1, resume_from_checkpoint=False, a scratch
     checkpoint_path, and one of two real-data-narrowing mechanisms so only a
     small, fixed number of real files get read regardless of how large the
     real dataset actually is:
       - For configs using the iterative dataloader (dataloader.type ==
         "iterative_dataloader", e.g. basic_ct/imagenet): narrowed
         dict_start_idx/dict_end_idx, computed dynamically (see
         compute_narrow_dict_idx) by calling the same
         UCF_VIT.utils.misc.process_root_dirs the real pipeline uses to get
         actual file counts, rather than guessing a fixed fraction that
         could round to 0 files for a modest dataset or barely help for a
         huge one like full ImageNet.
       - For configs using the plain dataloader (dataloader.type ==
         "dataloader", e.g. catsdogs): train.py globs every real file
         directly with no config-level trimming knob at all, so instead (see
         create_narrow_catsdogs_dir) this globs the real directory itself,
         then points dict_root_dirs at a scratch directory of *symlinks* to
         a subset of the real files -- same "real data, just less of it"
         principle, since there's no index-based slicing to hook into here.
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

make_smoke_config's `extra_overrides` parameter and the
deep_merge_config_overrides helper (both otherwise unused here -- every call
in this file's own main() passes extra_overrides=None) exist for
tests/integration/run_feature_matrix_smoke.py, the sibling Tier 3b driver
that turns individual advanced features (adaptive patching, tiling, twoD,
tensor parallelism) back on one at a time against real shipped configs,
reusing this file's real-data-narrowing and fresh/resume-run machinery
(run_fresh_phase/run_resume_phase in particular) rather than duplicating it.
"""

import argparse
import glob
import math
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

# Default (ceiling) target number of real files to keep per dataset key (or,
# for imagenet, per data-parallel bucket) after narrowing dict_start_idx/
# dict_end_idx -- see compute_narrow_dict_idx. Sized for the shared
# batch_size=32 most configs use: batch_size(32) * a typical
# data_par_size(8) = 256, comfortably under Tr8_Training's real 852 file
# pairs -- basic_ct's baseline config (do_tiling=False, twoD=False,
# do_ap=False -- do_ap=True only for the sap exception, which doesn't affect
# sample count) no longer multiplies each real file into many tiles/
# z-slices, so min_files must directly cover a full batch per rank.
# make_smoke_config further caps this down to each *individual* config's own
# batch_size * data_par_size, so a smaller-batch_size exception (currently
# basic_ct-unetr, batch_size=4) doesn't end up running many more
# batches/epoch than it needs -- see the comment there.
DEFAULT_MIN_FILES = 256

# Default per-run timeout in seconds. Previously 300s left basic_ct-unetr's
# real 227s fresh run little margin, but that run predated this baseline --
# basic_ct no longer multiplies each real file into up to 4096
# tiles/z-slices (do_tiling=False, twoD=False), so real runs should now be
# far cheaper across all basic_ct configs; kept at 300s as a still-generous
# ceiling rather than re-tuned down, pending real confirmation.
DEFAULT_TIMEOUT = 300


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


def inflate_min_files_for_train_split(conf, min_files):
    """Scales up a `min_files` narrowing target to survive the automatic train/val/test split.

    `compute_narrow_dict_idx`/`create_narrow_catsdogs_dir` narrow a dataset
    down to (approximately) `min_files` real files -- but `parse_config`'s
    own `dataloader.val_split_ratio`/`test_split_ratio` (default 0.1/0.1
    each) then automatically carves *train's own* share down further, within
    whatever `dict_start_idx`/`dict_end_idx` window narrowing already
    produced (see `UCF_VIT.parse._resolve_dataset_splits`). Narrowing
    straight to `min_files` and then losing 20% of it to that split can push
    an already-tight target (e.g. exactly `batch_size * data_par_size`, "just
    enough for one batch/rank") below one batch/rank in *training* -- a real
    `ZeroDivisionError` this caught on Frontier (`calculate_load_balancing_
    on_the_fly` asserts against it now, but callers narrowing real data for
    a training run should still request enough to not hit that assert in the
    first place). Callers that want `min_files` to land in *training specifically*
    (not just somewhere in the narrowed pre-split total) should pass their
    target through this before calling `compute_narrow_dict_idx`/
    `create_narrow_catsdogs_dir`.

    Args:
        conf: A parsed (real, un-tinied) config dict, as loaded from YAML --
            reads `conf["dataloader"].get("val_split_ratio"/"test_split_ratio", 0.1)`,
            matching parse.py's own defaults.
        min_files: The number of files wanted in training specifically.

    Returns:
        `min_files` scaled up by `1 / (1 - val_split_ratio - test_split_ratio)`,
        rounded up.
    """
    train_share = 1.0 - conf["dataloader"].get("val_split_ratio", 0.1) - conf["dataloader"].get("test_split_ratio", 0.1)
    return math.ceil(min_files / train_share)


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
        min_files: Target number of real files to keep per dataset key. For
            imagenet specifically, this narrows the *whole* dataset (every
            class combined) to approximately min_files total -- process_root_dirs
            no longer buckets imagenet by data-parallel rank before slicing
            (see its own docstring), so unlike before, this is NOT "min_files
            per bucket"; NativePytorchDataModule/calculate_load_balancing_on_
            the_fly divide whatever this narrows to across data_par_size
            buckets afterward. Callers wanting a specific per-bucket count
            should multiply min_files by data_par_size before calling this.

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

    try:
        dict_lister_trains = process_root_dirs(dataset, dict_root_dirs, data_par_size)
    except FileNotFoundError as e:
        # process_root_dirs' imagenet branch does a bare os.listdir(root_dir);
        # a root_dir that doesn't exist at all (as opposed to existing but
        # empty, the case already handled below) raises FileNotFoundError
        # directly rather than returning an empty listing -- normalize both
        # into the same NoRealDataFoundError so callers get one consistent,
        # always-gracefully-skippable failure mode.
        raise NoRealDataFoundError(
            f"No real files found for dataset={dataset!r} under dict_root_dirs={dict_root_dirs}: {e}"
        ) from e

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


def deep_merge_config_overrides(conf, overrides):
    """Recursively merges `overrides` into `conf` in place, returning `conf`.

    For each key in `overrides`: if both `conf[key]` and `overrides[key]` are
    dicts, recurses (so e.g. {"ap": {"do_ap": True}} only touches
    conf["ap"]["do_ap"], leaving conf["ap"]["fixed_length"]/etc. untouched).
    Otherwise, `conf[key]` is replaced wholesale with `overrides[key]` --
    this is a *replace*, not an element-wise merge, deliberately: it applies
    to lists/tuples too, not just scalars.

    Used by make_smoke_config's `extra_overrides` parameter (see
    run_feature_matrix_smoke.py) to flip individual advanced-feature flags
    (ap.do_ap, tiling.do_tiling, data.twoD, parallelism.tensor_par_size, ...)
    on top of a real shipped config, without hand-writing a full copy of
    that config's every field for each variant.

    Gotcha this deliberately does NOT protect against: parse_config
    (parse.py's tiling-overlap handling) only recognizes an
    already-multi-dimensional tiling.tile_overlap if it's a Python *tuple*
    -- YAML has no tuple literal, so writing e.g.
    {"tiling": {"tile_overlap": [0, 0]}} here loads back (and round-trips
    through yaml.dump/yaml.load) as a *list*, which parse_config silently
    mishandles (wraps the whole list twice instead of treating it as
    already-2D, then fails a downstream int-only assert). Every shipped
    config avoids this by using a bare scalar int (e.g. `tile_overlap: 0`),
    which parse_config *does* correctly expand to a 2- or 3-tuple based on
    twoD -- do the same in any override that touches tile_overlap.

    Args:
        conf: Config dict to merge into, modified in place.
        overrides: Dict of values to merge on top of `conf`.

    Returns:
        `conf`, for convenience chaining.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(conf.get(key), dict):
            deep_merge_config_overrides(conf[key], value)
        else:
            conf[key] = value
    return conf


def create_narrow_catsdogs_dir(conf, scratch_dir, min_files):
    """Narrows a "dataloader"-type config's real data by symlinking a subset of real files.

    Only meaningful for conf["dataloader"]["type"] == "dataloader" (e.g.
    catsdogs). train.py's handling of this type globs every *.jpg file
    directly from dict_root_dirs with no config-level trimming knob at all
    (unlike the iterative dataloader's dict_start_idx/dict_end_idx), so this
    globs the real directory itself (mirroring train.py's own
    `glob.glob(os.path.join(dict_root_dirs[dkey], '*.jpg'))`, and takes the
    same first dict_root_dirs key train.py's "dataloader_type == 'dataloader'"
    branch does), then creates a scratch directory of symlinks to a subset of
    the real files -- same "real data, just less of it" principle as
    compute_narrow_dict_idx.

    The subset size is max(min_files, batch_size * data_par_size), not just
    min_files: train.py wraps this dataset in a DataLoader with
    drop_last=True, so after DistributedSampler splits files across
    data_par_size ranks, each rank needs at least batch_size files or its
    entire (undersized) batch gets silently dropped -- giving an empty
    dataloader (0 iterations/epoch) rather than an error, which would make
    the smoke test report PASS despite zero actual training happening.

    Args:
        conf: A parsed (real, un-tinied) config dict. Needs
            conf["data"]["dict_root_dirs"]/["dataset"],
            conf["dataloader"]["batch_size"], and
            conf["parallelism"]["fsdp_size"]/["simple_ddp_size"].
        scratch_dir: Job-scoped scratch directory to create the symlink
            subdirectory under.
        min_files: Target (minimum) number of real files to symlink.

    Returns:
        A tuple (dkey, narrowed_dir): dkey is the dict_root_dirs key train.py
        actually uses for this dataloader type, and narrowed_dir is the
        scratch directory of symlinks to point dict_root_dirs[dkey] at
        instead of the real directory.

    Raises:
        NoRealDataFoundError: If the real directory has zero *.jpg files.
    """
    dkey_train = list(conf["data"]["dict_root_dirs"])[0]
    real_dir = conf["data"]["dict_root_dirs"][dkey_train]
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.jpg")))

    if not real_files:
        raise NoRealDataFoundError(
            f"No *.jpg files found for dataset={conf['data']['dataset']!r}, key {dkey_train!r} "
            f"under dict_root_dirs={{{dkey_train!r}: {real_dir!r}}}"
        )

    data_par_size = conf["parallelism"]["fsdp_size"] * conf["parallelism"]["simple_ddp_size"]
    needed = conf["dataloader"]["batch_size"] * data_par_size
    count = min(len(real_files), max(min_files, needed))

    narrowed_dir = os.path.join(scratch_dir, "narrowed_data")
    os.makedirs(narrowed_dir, exist_ok=True)
    for f in real_files[:count]:
        os.symlink(f, os.path.join(narrowed_dir, os.path.basename(f)))

    return dkey_train, narrowed_dir


def make_smoke_config(base_config_path, scratch_dir, min_files=DEFAULT_MIN_FILES, extra_overrides=None):
    """Loads a real config and writes a tiny/fast, freshly-training smoke-test variant.

    Only overrides model size, epoch count, checkpoint location,
    resume_from_checkpoint, (see compute_narrow_dict_idx /
    create_narrow_catsdogs_dir) the real-data-narrowing fields appropriate to
    this config's dataloader type, and (if given) `extra_overrides`.
    Everything else -- data paths (beyond narrowing), tiling, adaptive
    patching, batch size, load-balancing settings -- is left exactly as in
    the real config.

    Args:
        base_config_path: Path to the real config YAML to base this on.
        scratch_dir: Job-scoped scratch directory to write the checkpoint,
            any narrowed-data symlinks, and this generated config file into.
        min_files: Passed through to compute_narrow_dict_idx /
            create_narrow_catsdogs_dir.
        extra_overrides: Optional dict merged onto the loaded real config via
            deep_merge_config_overrides, applied immediately after loading
            it and before anything else -- in particular, before
            data_par_size is computed for the min_files cap below, so a
            parallelism.tensor_par_size/fsdp_size/simple_ddp_size override
            here correctly affects that cap. Used by
            run_feature_matrix_smoke.py to flip individual advanced-feature
            flags on top of a real shipped config; run_training_smoke.py's
            own main() never passes this (None).

    Returns:
        Path to the written smoke-test config file.

    Raises:
        NoRealDataFoundError: See compute_narrow_dict_idx /
            create_narrow_catsdogs_dir.
    """
    with open(base_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    if extra_overrides:
        deep_merge_config_overrides(conf, extra_overrides)

    # DEFAULT_MIN_FILES (or an explicit --min-files) is a ceiling, not a
    # target: it's sized for the shared batch_size=32 most configs use, but
    # basic_ct-unetr's batch_size=4 exception (see its config comment) means
    # 256 real files there works out to 8 batches/rank/epoch instead of 1 --
    # plenty of real data for correctness, but the extra iterations through
    # UNETR's inherently expensive full-resolution conv decoder cost real
    # wall-clock time and were timing out the smoke test. Capping min_files
    # to this config's own batch_size * data_par_size keeps every config to
    # ~1 batch/rank regardless of how small its batch_size is, without
    # needing a per-config override.
    data_par_size = conf["parallelism"]["fsdp_size"] * conf["parallelism"]["simple_ddp_size"]
    # The cap targets ~1 batch/rank in *training* -- inflate_min_files_for_
    # train_split accounts for parse_config's own automatic train/val/test
    # split (dataloader.val_split_ratio/test_split_ratio) narrowing train's
    # own share further, so this many files still land in training itself,
    # not 0 (see that function's own docstring for the real Frontier
    # ZeroDivisionError this fixes).
    min_files = min(min_files, inflate_min_files_for_train_split(conf, conf["dataloader"]["batch_size"] * data_par_size))

    conf["model"].update(TINY_MODEL_OVERRIDES)

    conf["trainer"]["checkpoint_path"] = scratch_dir
    conf["trainer"]["save_frequency"] = 1
    conf["trainer"]["max_epochs"] = 1
    conf["trainer"]["resume_from_checkpoint"] = False

    os.makedirs(scratch_dir, exist_ok=True)

    if conf["dataloader"]["type"] == "iterative_dataloader":
        narrow_end_idx = compute_narrow_dict_idx(conf, min_files)
        conf["dataloader"]["dict_start_idx"] = {k: 0.0 for k in narrow_end_idx}
        conf["dataloader"]["dict_end_idx"] = narrow_end_idx
    elif conf["dataloader"]["type"] == "dataloader":
        dkey, narrowed_dir = create_narrow_catsdogs_dir(conf, scratch_dir, min_files)
        conf["data"]["dict_root_dirs"][dkey] = narrowed_dir

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


def run_training(config_path, ntasks, timeout, pretrained_config_path=None):
    """Runs training_scripts/train.py against `config_path` as an srun subprocess.

    Args:
        config_path: Path to the config to train.
        ntasks: Number of srun tasks.
        timeout: Timeout in seconds.
        pretrained_config_path: If given, passed through as train.py's own
            --pretrained_config argument (used by run_pretrained_smoke.py's
            pretrained-loading phase; every other caller leaves this None).

    Returns:
        A dict with "returncode" (None on timeout) and "log" (combined
        stdout+stderr, tail-truncated).
    """
    cmd = ["srun", "-n", str(ntasks), "python", TRAIN_SCRIPT, config_path, "--launcher", "slurm"]
    if pretrained_config_path is not None:
        cmd += ["--pretrained_config", pretrained_config_path]
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


def run_fresh_phase(smoke_config, slug, ntasks, timeout, scratch_dir):
    """Runs one fresh training run against `smoke_config` and classifies the result.

    Factored out of main()'s loop so run_feature_matrix_smoke.py's own loop
    can reuse the exact same PASS/FAIL/TIMEOUT/checkpoint-written
    classification and logging, instead of a second, independently
    maintained copy of the same logic.

    Args:
        smoke_config: Path to a smoke-test config, as written by
            make_smoke_config.
        slug: Short label for this run, used only in printed output.
        ntasks: Number of srun tasks (passed through to run_training).
        timeout: Per-run timeout in seconds (passed through to run_training).
        scratch_dir: The same scratch_dir make_smoke_config wrote this
            config's checkpoint_path to, used to check whether a rank-0
            checkpoint was actually produced.

    Returns:
        A dict: {"status": one of "PASS"/"FAIL"/"TIMEOUT"/
        "FAIL (no checkpoint written)", "elapsed": float, "log": str}.
    """
    print(f"[{slug}] fresh run: srun -n {ntasks} python train.py {smoke_config}", flush=True)
    t0 = time.time()
    fresh = run_training(smoke_config, ntasks, timeout)
    elapsed = time.time() - t0

    status = "TIMEOUT" if fresh["returncode"] is None else ("PASS" if fresh["returncode"] == 0 else "FAIL")
    if status != "PASS":
        print(fresh["log"][-4000:], flush=True)
    elif not rank0_checkpoint_exists(scratch_dir):
        status = "FAIL (no checkpoint written)"
        print(fresh["log"][-4000:], flush=True)
    print(f"[{slug}] fresh run: {status} ({elapsed:.0f}s)", flush=True)

    return {"status": status, "elapsed": elapsed, "log": fresh["log"]}


def run_resume_phase(smoke_config, slug, ntasks, timeout):
    """Runs the resume half of a fresh->resume cycle against an existing checkpoint.

    Factored out of main()'s loop for the same reason as run_fresh_phase --
    reused by run_feature_matrix_smoke.py for the one tensor-parallel cell
    that also exercises resume (see that file's module docstring for why
    just that one cell).

    Precondition: the fresh run smoke_config came from already passed and
    produced a checkpoint (callers are responsible for checking this, same
    as main() does today).

    Args:
        smoke_config: Path to the smoke-test config that already ran (and
            passed) a fresh run.
        slug: Short label for this run, used only in printed output.
        ntasks: Number of srun tasks (passed through to run_training).
        timeout: Per-run timeout in seconds (passed through to run_training).

    Returns:
        A dict: {"status": one of "PASS"/"FAIL"/"TIMEOUT", "elapsed": float,
        "log": str}.
    """
    set_resume(smoke_config, resume=True)
    print(f"[{slug}] resume run: srun -n {ntasks} python train.py {smoke_config}", flush=True)
    t0 = time.time()
    resume = run_training(smoke_config, ntasks, timeout)
    elapsed = time.time() - t0

    status = "TIMEOUT" if resume["returncode"] is None else ("PASS" if resume["returncode"] == 0 else "FAIL")
    if status != "PASS":
        print(resume["log"][-4000:], flush=True)
    print(f"[{slug}] resume run: {status} ({elapsed:.0f}s)", flush=True)

    set_resume(smoke_config, resume=False)
    return {"status": status, "elapsed": elapsed, "log": resume["log"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="Specific config file(s) to test; default is every configs/**/base_config.yaml")
    parser.add_argument("--ntasks", type=int, default=8, help="Number of srun tasks per training run (default 8, matching launch/*/*.sh)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Per-run timeout in seconds (default {DEFAULT_TIMEOUT})")
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

        min_files = args.min_files
        timeout = args.timeout

        try:
            smoke_config = make_smoke_config(config_path, scratch_dir, min_files=min_files)
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

        fresh = run_fresh_phase(smoke_config, slug, args.ntasks, timeout, scratch_dir)
        fresh_status = fresh["status"]
        fresh_elapsed = fresh["elapsed"]

        resume_status = "SKIPPED (fresh run didn't produce a checkpoint)"
        resume_elapsed = None
        if fresh_status == "PASS":
            resume = run_resume_phase(smoke_config, slug, args.ntasks, timeout)
            resume_status = resume["status"]
            resume_elapsed = resume["elapsed"]

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
