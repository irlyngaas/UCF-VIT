"""Tier 3 integration smoke test: real end-to-end pretrained-checkpoint loading.

No config ships with use_pretrained_model:True, and neither
run_training_smoke.py nor run_feature_matrix_smoke.py exercises it -- this is
the first real, end-to-end proof of the actual workflow the pretrained-
loading generalization work was for: train once, then fine-tune from that
checkpoint at a *different* resolution.

Reuses run_training_smoke.py's real infrastructure directly (same pattern
run_feature_matrix_smoke.py already uses): make_smoke_config, run_fresh_phase,
run_training, DEFAULT_MIN_FILES, config_slug -- not reimplemented here.

Two phases, both against configs/catsdogs/classification/base_config.yaml
(chosen for its "dataloader"-type real-data narrowing, the simplest/fastest
of the two mechanisms make_smoke_config supports):

  1. A real, unmodified fresh training run -- produces a real
     epoch_0_rank_0.ckpt via the real training loop (not a hand-built
     checkpoint fixture).
  2. A second run against the *same* base config, but with
     dataset_options.resize.catsdogs overridden to a different, non-square
     size than the shipped [256,256] (deliberately independent-ratio, same
     shape as tests/model/test_pretrained_loading.py's Tier 1 cases) --
     resize, not data.img_size, is what actually controls both the real
     data pipeline's resize step and tile_size's computation for catsdogs
     (see parse.py's effective_size = resize_conf.get(dataset, img_size)),
     so overriding it keeps the two consistent automatically. trainer.
     use_pretrained_model:True and pretrained_checkpoint_filename:"epoch_0"
     point it at phase 1's checkpoint via train.py's own --pretrained_config
     argument (phase 1's config path).

Usage (from anywhere, inside an sbatch job with GPUs already allocated):
    python tests/integration/run_pretrained_smoke.py [--ntasks N] [--timeout SECONDS]
"""

import argparse
import os
import shutil
import sys
import time

from run_training_smoke import (
    DEFAULT_MIN_FILES,
    NoRealDataFoundError,
    config_slug,
    make_smoke_config,
    rank0_checkpoint_exists,
    run_fresh_phase,
    run_training,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")
BASE_CONFIG = os.path.join(CONFIGS_DIR, "catsdogs", "classification", "base_config.yaml")

# [height, width]. Deliberately non-square, independently different from
# the shipped [256, 256] on both axes -- not a uniform rescale.
NEW_RESIZE = [128, 192]

DEFAULT_TIMEOUT = 600


def run_pretrained_phase(smoke_config, pretrained_config_path, slug, ntasks, timeout, scratch_dir):
    """Runs the pretrained-loading phase against an already-completed fresh run.

    Mirrors run_training_smoke.py's own run_resume_phase, but points at a
    *different* config (phase 1's, via train.py's --pretrained_config
    argument) instead of resuming smoke_config's own checkpoint.

    Args:
        smoke_config: Path to phase 2's own smoke-test config (as written by
            make_smoke_config, with use_pretrained_model:True already set).
        pretrained_config_path: Path to phase 1's config file (the one whose
            checkpoint smoke_config is fine-tuning from).
        slug: Short label for this run, used only in printed output.
        ntasks: Number of srun tasks (passed through to run_training).
        timeout: Per-run timeout in seconds (passed through to run_training).
        scratch_dir: The same scratch_dir make_smoke_config wrote this
            config's checkpoint_path to, used to check whether it wrote its
            own rank-0 checkpoint after fine-tuning.

    Returns:
        A dict: {"status": one of "PASS"/"FAIL"/"TIMEOUT"/
        "FAIL (no checkpoint written)", "elapsed": float, "log": str}.
    """
    print(f"[{slug}] pretrained run: srun -n {ntasks} python train.py {smoke_config} --pretrained_config {pretrained_config_path}", flush=True)
    t0 = time.time()
    result = run_training(smoke_config, ntasks, timeout, pretrained_config_path=pretrained_config_path)
    elapsed = time.time() - t0

    status = "TIMEOUT" if result["returncode"] is None else ("PASS" if result["returncode"] == 0 else "FAIL")
    if status != "PASS":
        print(result["log"][-4000:], flush=True)
    elif not rank0_checkpoint_exists(scratch_dir):
        status = "FAIL (no checkpoint written)"
        print(result["log"][-4000:], flush=True)
    print(f"[{slug}] pretrained run: {status} ({elapsed:.0f}s)", flush=True)

    return {"status": status, "elapsed": elapsed, "log": result["log"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntasks", type=int, default=8, help="Number of srun tasks per training run (default 8, matching launch/*/*.sh)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Per-run timeout in seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--min-files", type=int, default=DEFAULT_MIN_FILES, help=f"Target real files to keep per dataset key after narrowing (default {DEFAULT_MIN_FILES})")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_root = f"/tmp/{job_id}/checkpoint_pretrained_smoke"
    slug = config_slug(BASE_CONFIG)

    pretrained_scratch_dir = os.path.join(scratch_root, "pretrained")
    new_scratch_dir = os.path.join(scratch_root, "new")

    print(f"\n{'='*80}\n{slug} phase 1: fresh (produces the checkpoint phase 2 fine-tunes from)\n{'='*80}", flush=True)
    try:
        pretrained_config = make_smoke_config(BASE_CONFIG, pretrained_scratch_dir, min_files=args.min_files)
    except NoRealDataFoundError as e:
        sys.exit(f"[{slug}] {e}")

    fresh = run_fresh_phase(pretrained_config, f"{slug}+phase1", args.ntasks, args.timeout, pretrained_scratch_dir)

    pretrained_status = "-"
    pretrained_elapsed = None
    if fresh["status"] == "PASS":
        print(f"\n{'='*80}\n{slug} phase 2: pretrained (resize.catsdogs -> {NEW_RESIZE}, a different, non-square ratio)\n{'='*80}", flush=True)
        new_config = make_smoke_config(
            BASE_CONFIG, new_scratch_dir, min_files=args.min_files,
            extra_overrides={
                "dataset_options": {"resize": {"catsdogs": NEW_RESIZE}},
                "trainer": {"use_pretrained_model": True, "pretrained_checkpoint_filename": "epoch_0"},
            },
        )
        pretrained = run_pretrained_phase(new_config, pretrained_config, f"{slug}+phase2", args.ntasks, args.timeout, new_scratch_dir)
        pretrained_status = pretrained["status"]
        pretrained_elapsed = pretrained["elapsed"]
    else:
        pretrained_status = "SKIPPED (phase 1 didn't produce a checkpoint)"

    shutil.rmtree(scratch_root, ignore_errors=True)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    fresh_time = f"{fresh['elapsed']:.0f}s"
    pretrained_time = f"{pretrained_elapsed:.0f}s" if pretrained_elapsed is not None else "-"
    print(f"{slug:38s} phase1(fresh)={fresh['status']:8s}({fresh_time:>6s})  phase2(pretrained)={pretrained_status:8s}({pretrained_time:>6s})")

    ok = fresh["status"] == "PASS" and pretrained_status == "PASS"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
