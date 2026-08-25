"""Tier 3b feature-matrix smoke test: for one representative (model, dataset)
combination per "advanced feature" this session's baseline reconfiguration
turned off by default -- tensor parallelism, adaptive patching, tiling, and
(basic_ct only) twoD -- turns that feature back ON and runs a tiny, fast
training run through the real training_scripts/train.py entry point against
real Frontier data, to confirm the feature still works now that it's opt-in
rather than the shipped default.

This is deliberately a *curated* matrix, not a full combinatorial sweep of
every feature x every shipped config: the sharding/tiling/patching logic
under test is generic and orthogonal to which dataset it's applied to, so
one representative cell per mechanism is enough to catch a real regression,
at a fraction of full combinatorial cost. It also includes a handful of
realistic *multi*-feature cells (see the "multi-feature combinations"
section of FEATURE_MATRIX below) -- picked because they're combinations
someone would plausibly actually configure together (two of them mirror
this repo's own pre-baseline shipped configs almost exactly), not an
exhaustive sweep of every possible pairing. If a real need for some other
specific combination ever comes up, add a cell for it rather than expanding
this matrix generally.

This deliberately does NOT use pytest, for the same reason
run_training_smoke.py doesn't -- see that file's module docstring. This
script also spawns its own srun subprocess per run and is meant to be
launched as a single unwrapped process (see
launch/tests/run_feature_matrix_smoke.sh), not nested inside a pytest
process already under its own srun.

Reuses run_training_smoke.py's infrastructure directly: TINY_MODEL_OVERRIDES,
DEFAULT_MIN_FILES, make_smoke_config (via its extra_overrides parameter --
see that function's docstring for the deep-merge helper and its
tile_overlap-must-be-a-bare-int gotcha), run_fresh_phase/run_resume_phase,
set_resume, config_slug, NoRealDataFoundError, rank0_checkpoint_exists. Does
NOT use run_training_smoke.py's main()/discover_configs() -- this script has
its own FEATURE_MATRIX of (base config, feature label, config overrides)
cells rather than looping over every configs/**/base_config.yaml.

Every cell runs fresh-only (proving forward/backward/checkpoint-save with
the feature on) EXCEPT one tensor-parallel cell, which also runs a resume
cycle: baseline Tier 3's own fresh+resume runs already prove
resume_from_checkpoint's mechanics work for do_ap/do_tiling/twoD (that logic
doesn't depend on which advanced feature is on), but tensor_par_size has its
own per-tensor-parallel-rank checkpoint file loop (parse.py's pre-flight
existence check, model/utils.py's resume loading) that no *shipped* config
exercises at all (every one ships tensor_par_size:1) -- so nothing in Tier 1,
2, or baseline Tier 3 has ever run that loop with tensor_par_size > 1. One
TP cell closes that gap; the other two stay fresh-only to keep total cost
down, per the same "curated, not combinatorial" principle above.

Usage (from anywhere, inside an sbatch job with GPUs already allocated):
    python tests/integration/run_feature_matrix_smoke.py [--ntasks N] [--timeout SECONDS] [label ...]

With no label arguments, runs every FEATURE_MATRIX cell. Prints a per-cell
PASS/FAIL/TIMEOUT summary at the end and exits nonzero if anything failed.
"""

import argparse
import os
import shutil
import sys
from typing import NamedTuple, Optional

from run_training_smoke import (
    DEFAULT_MIN_FILES,
    NoRealDataFoundError,
    config_slug,
    make_smoke_config,
    run_fresh_phase,
    run_resume_phase,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

# Wider default than run_training_smoke.py's 300s: every cell here is new,
# unproven territory (a feature that's off by default in every shipped
# config), not the already-tuned baseline. Cells known from this session's
# own history to need more (see FEATURE_MATRIX's basic_ct-unetr+twoD cell)
# override this per-cell rather than raising the shared default for
# everything.
DEFAULT_TIMEOUT = 600


class FeatureMatrixCell(NamedTuple):
    base_config_relpath: str  # relative to configs/, e.g. "basic_ct/unetr/base_config.yaml"
    label: str  # unique short name, e.g. "basic_ct-unetr+do_ap"
    overrides: dict  # deep-merged onto the real config; see make_smoke_config's extra_overrides
    min_files_override: Optional[int] = None  # None => use DEFAULT_MIN_FILES
    timeout_override: Optional[int] = None  # None => use this file's DEFAULT_TIMEOUT
    test_resume: bool = False  # if True, also runs a resume cycle after a passing fresh run


FEATURE_MATRIX = [
    # --- ap.do_ap:True -- one cell per free-choice model type (UNETR, MAE,
    # VIT), plus both dataloader.type code paths ("iterative_dataloader" via
    # imagenet, "dataloader" via catsdogs). SAP is excluded: parse.py
    # hard-requires do_ap:True for SAP already (not a free choice), and its
    # baseline config already proves this combination on real Frontier data.
    # DiffusionVIT is excluded: parse.py hard-requires do_ap:False for it.
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+do_ap",
        {"ap": {"do_ap": True}},
        # fixed_length:512 already valid (512 % 7 == 1, octree; cube root 8
        # is a whole number, satisfying UNETR's extra sqrt_len constraint) --
        # no override needed.
    ),
    FeatureMatrixCell(
        "basic_ct/mae/base_config.yaml", "basic_ct-mae+do_ap",
        {"ap": {"do_ap": True, "fixed_length": 512}},
        # shipped fixed_length:196 is invalid for the octree check
        # (196 % 7 == 0, not 1) now that mae is twoD:False -- override to
        # 512, the same value basic_ct/sap's shipped config already proves
        # works on real Frontier data for this exact twoD:False/octree case.
    ),
    FeatureMatrixCell(
        "imagenet/classification/base_config.yaml", "imagenet-classification+do_ap",
        {"ap": {"do_ap": True}},
        # fixed_length:196 already valid (196 % 3 == 1, quadtree) -- no
        # override needed.
    ),
    FeatureMatrixCell(
        "catsdogs/classification/base_config.yaml", "catsdogs-classification+do_ap",
        {"ap": {"do_ap": True}},
        # Also the only cell exercising do_ap:True through the "dataloader"
        # (not "iterative_dataloader") code path -- CatsDogsDataset's own
        # adaptive-patching branch, not TileDataIter/ProcessChannels.
    ),

    # --- tiling.do_tiling:True -- catsdogs excluded: dataloader.type:
    # "dataloader" never invokes TileDataIter at all (train.py routes it
    # straight to CatsDogsDataset), and CatsDogsDataset's own tile_size is
    # purely a cv.resize target, not a tiling grid -- "tiling" isn't a
    # meaningful feature to exercise there at all.
    # min_files_override is deliberately small: do_tiling multiplies each
    # real file into div**N samples, which DEFAULT_MIN_FILES/
    # make_smoke_config's batch_size*data_par_size cap doesn't account for
    # (it's sized in units of files, not resulting samples) -- see this
    # file's module docstring.
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+do_tiling",
        # tile_overlap MUST be a bare int, not a list -- see
        # deep_merge_config_overrides's docstring for why.
        {"tiling": {"do_tiling": True, "div": 4, "tile_overlap": 0}},
        min_files_override=16,
        # div=4 with twoD:False was already real-Frontier-verified working
        # earlier this session, before the baseline reconfiguration --
        # tile_size becomes 64**3, still divisible by unetr's patch_size:32.
    ),
    FeatureMatrixCell(
        "imagenet/classification/base_config.yaml", "imagenet-classification+do_tiling",
        {"tiling": {"do_tiling": True, "div": 4, "tile_overlap": 0}},
        min_files_override=16,
        # tile_size becomes 64x64, still divisible by patch_size:16.
    ),

    # --- data.twoD:True -- basic_ct only; imagenet/catsdogs already always
    # run twoD:True (parse.py forces it whenever img_size has 2 entries,
    # unconditionally -- data.twoD in their YAML is never even read).
    # Highest-risk cell in this matrix: with do_tiling:False forced (so x/y
    # stay untiled) but twoD:True, TileDataIter still walks the *entire*
    # z-axis one index at a time, so each real file yields up to 256
    # samples -- the exact mechanism behind this session's original
    # basic_ct-mae timeout. min_files is pinned to the floor (8 -- the
    # minimum for FileReader's per-rank sharding to give every one of
    # data_par_size(8) ranks at least 1 file), and timeout is raised well
    # past the shared default, mirroring the 1800s this session already
    # needed for the analogous basic_ct-sap z-slice-multiplication timeout.
    # Expect this cell specifically to need live tuning against the first
    # real Frontier run, the same way the original baseline effort did.
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+twoD",
        {"data": {"twoD": True}},
        min_files_override=8,
        timeout_override=1800,
    ),

    # --- parallelism.tensor_par_size:2 -- representative subset (3 of 10),
    # per the user's confirmed scope decision: the sharding logic is generic/
    # orthogonal to dataset specifics. fsdp_size:1/simple_ddp_size:4 keeps
    # data_par_size*tensor_par_size == world_size == 8 (this script's default
    # --ntasks), matching parse.py's hard assertion. TINY_MODEL_OVERRIDES's
    # num_heads:2/embed_dim:24 already divide cleanly by tensor_par_size:2
    # (num_heads % tensor_par_size / embed_dim % tensor_par_size are not
    # enforced anywhere in code -- only documented in README.md -- so this
    # isn't a blocker here, but raising tensor_par_size further would need
    # to keep this in mind).
    FeatureMatrixCell(
        "imagenet/classification/base_config.yaml", "imagenet-classification+tensor_par",
        {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        test_resume=True,
        # The one cell that also resumes: parse.py's pre-flight checkpoint
        # check and model/utils.py's resume loading both loop per
        # tensor-parallel rank (range(tensor_par_size)) -- code no shipped
        # config (all tensor_par_size:1) or any other tier has ever
        # exercised with tensor_par_size > 1. See this file's module
        # docstring.
    ),
    FeatureMatrixCell(
        "basic_ct/mae/base_config.yaml", "basic_ct-mae+tensor_par",
        {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        # Exercises MAE's TP-specific noise-mask broadcast (arch.py) -- a
        # code path unique to MAE among all 5 model types, only fires when
        # tensor_par_size > 1.
    ),
    FeatureMatrixCell(
        "catsdogs/classification/base_config.yaml", "catsdogs-classification+tensor_par",
        {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        # Exercises tensor parallelism through the distinct "dataloader"-type
        # (DistributedSampler, not FileReader) code path.
    ),

    # --- A few realistic multi-feature combinations, not exhaustive (that
    # would be a large combinatorial blowup this session deliberately scoped
    # out) -- picked because someone would plausibly actually configure them
    # together, not just because they're technically legal. The first two
    # aren't hypothetical: they mirror the *actual* pre-baseline shipped
    # basic_ct/unetr and basic_ct/mae configs (before this session's
    # baseline reconfiguration) almost exactly, so passing here is a real
    # regression check against configurations that genuinely existed and
    # worked in this repo's history.
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+do_ap+do_tiling",
        {"ap": {"do_ap": True}, "tiling": {"do_tiling": True, "div": 4, "tile_overlap": 0}},
        min_files_override=16,
        # Mirrors basic_ct/unetr's original pre-baseline config almost
        # exactly (do_ap:True, do_tiling:True, div:4, twoD:False) -- that
        # exact combination already passed real Frontier runs earlier this
        # session (see tests/README.md's "Real runs on Frontier so far" for
        # Tier 3), so this is a real regression check, not new territory.
        # tile_size becomes 64**3 (power of two -- required for do_ap) with
        # fixed_length:512 (cube root 8, unaffected by tile_size's actual
        # value). min_files matches the do_tiling-alone cell's reasoning --
        # do_tiling still multiplies each real file into div**3=64 samples.
    ),
    FeatureMatrixCell(
        "basic_ct/mae/base_config.yaml", "basic_ct-mae+twoD+do_tiling",
        {"data": {"twoD": True}, "tiling": {"do_tiling": True, "div": 2, "tile_overlap": 0}},
        min_files_override=8,
        timeout_override=1800,
        # Mirrors basic_ct/mae's original pre-baseline config (twoD:True,
        # do_tiling:True) -- the exact combination behind this session's
        # original basic_ct-mae timeout that started the whole baseline
        # reconfiguration effort. Deliberately exercising it again, but
        # controlled this time: div:2 (not the original div:4) keeps x/y
        # tiling from also multiplying on top of twoD's already-expensive
        # z-slicing (each real file still yields up to div**2 * img_size[2]
        # = 4 * 256 = 1024 samples -- div:4 would have been 16 * 256 =
        # 4096, the same order of magnitude that caused the original
        # timeout even after PER_CONFIG_OVERRIDES tuning). min_files pinned
        # to the floor (8) and timeout raised to 1800s for the same reason
        # as basic_ct-unetr+twoD above. tile_size becomes (128,128,256);
        # do_ap stays False (mae's baseline), so only x/y need
        # tile_size % patch_size == 0 when twoD:True (parse.py's
        # checkDims = 2 if twoD else 3) -- 128 % 32 == 0, fine.
    ),
    FeatureMatrixCell(
        "imagenet/classification/base_config.yaml", "imagenet-classification+do_ap+tensor_par",
        {"ap": {"do_ap": True}, "parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        # Neither do_ap nor tensor_par_size multiplies sample count, so no
        # min_files/timeout override needed beyond the shared defaults.
    ),

    # --- Completing pairwise coverage across all 4 axes: of the 6 possible
    # pairs among {do_ap, do_tiling, twoD, tensor_par_size}, the 3 above
    # cover do_ap+do_tiling, do_tiling+twoD, and do_ap+tensor_par -- these 3
    # cover the remaining do_ap+twoD, do_tiling+tensor_par, and
    # twoD+tensor_par. Each still needs only a single fixed value per
    # feature (one tensor_par_size, one div), not a sweep, so this stays as
    # cheap as every other cell here.
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+do_ap+twoD",
        {"ap": {"do_ap": True, "fixed_length": 196}, "data": {"twoD": True}},
        min_files_override=8,
        timeout_override=1800,
        # fixed_length:512 (unetr's baseline, valid for the twoD:False/
        # octree case) does NOT work once twoD:True routes this through the
        # quadtree check instead (fixed_length % 3 == 1) and a *square*
        # (not cube) root requirement for UNETR's sqrt_len -- 512 % 3 == 2
        # and sqrt(512) isn't whole. fixed_length:196 (already used
        # elsewhere in this matrix/the shipped imagenet/catsdogs configs)
        # satisfies both: 196 % 3 == 1, sqrt(196) == 14. twoD:True still
        # brings its usual z-slice sample multiplication cost regardless of
        # do_ap, so min_files/timeout match basic_ct-unetr+twoD above.
    ),
    FeatureMatrixCell(
        "imagenet/classification/base_config.yaml", "imagenet-classification+do_tiling+tensor_par",
        {
            "tiling": {"do_tiling": True, "div": 4, "tile_overlap": 0},
            "parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2},
        },
        min_files_override=16,
        # Same div:4/min_files:16 as imagenet-classification+do_tiling above
        # -- do_tiling still multiplies each real file into div**2=16
        # samples regardless of tensor_par_size.
    ),
    FeatureMatrixCell(
        "basic_ct/unetr/base_config.yaml", "basic_ct-unetr+twoD+tensor_par",
        {
            "data": {"twoD": True},
            "parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2},
        },
        min_files_override=8,
        timeout_override=1800,
        # Same min_files:8/timeout:1800 as basic_ct-unetr+twoD above --
        # twoD's z-slice multiplication cost is unaffected by
        # tensor_par_size. Exercises whether training.py's process_batch
        # TP-rank-0-pulls-then-broadcasts logic still behaves correctly
        # when twoD inflates the per-rank sample count.
    ),

    # --- Completing model-type coverage: every cell above uses UNETR, MAE,
    # or VIT -- SAP and DiffusionVIT never appear anywhere in this matrix.
    # Neither has a free do_ap choice (SAP is always do_ap:True, DiffusionVIT
    # always do_ap:False -- both hard-required by parse.py, not config
    # choices), so neither needed a do_ap cell, but that's not true of
    # tensor_par_size: it's orthogonal to model type, and both SAP and
    # DiffusionVIT have their own model-specific broadcast code in
    # training.py's process_batch (SAP: seq_label; DiffusionVIT: the t/e
    # diffusion noise terms) that no cell above has ever exercised with a
    # real forward+loss. One tensor_par_size cell each closes that gap
    # without re-testing do_tiling/twoD, which are generic dataloader-level
    # mechanics already proven independent of model type above.
    FeatureMatrixCell(
        "basic_ct/sap/base_config.yaml", "basic_ct-sap+tensor_par",
        {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        timeout_override=1800,
        # SAP's baseline already keeps patch_size:4 (its own decoder-memory
        # exception), so tensor_par_size doesn't compound with any known
        # memory risk here. Exercises training.py's seq_label broadcast
        # (only taken for UNETR/SAP) under tensor_par_size > 1 for the
        # first time -- and, since SAP is the only model requiring
        # do_ap:True, also the first time the do_ap:True Segmentation
        # `label` placeholder branch runs under tensor_par_size > 1 at all.
        # timeout:1800 (no min_files override -- keeps DEFAULT_MIN_FILES:256)
        # is a safety margin for job 5339608's dtype-mismatch broadcast fix
        # (training.py) getting its first real-Frontier verification, not a
        # known cost driver like the twoD cells. min_files_override=8 was
        # tried here once (mirroring basic_ct-unetr+twoD) and was wrong:
        # unlike twoD, SAP has no tiles_per_image multiplier (twoD:False,
        # do_tiling:False), so 8 files split across data_par_size:4 ranks
        # gives only 2 images/rank -- far below the baseline's
        # batch_size:32 -- which makes
        # calculate_load_balancing_on_the_fly's batches_per_rank floor to 0
        # and datamodule.py's setup() crash with ZeroDivisionError (job
        # 5340104). DEFAULT_MIN_FILES:256 gives 64 images/rank, comfortably
        # above batch_size:32.
    ),
    FeatureMatrixCell(
        "catsdogs/diffusion/base_config.yaml", "catsdogs-diffusion+tensor_par",
        {"parallelism": {"fsdp_size": 1, "simple_ddp_size": 4, "tensor_par_size": 2}},
        # catsdogs is otherwise underrepresented in this matrix (2 cells,
        # both VIT) -- pairing DiffusionVIT with catsdogs here covers a new
        # (dataset, model) combination too, not just a new model type.
        # Exercises training.py's t/e diffusion-noise broadcast (only taken
        # for DiffusionVIT, which is always do_ap:False) under
        # tensor_par_size > 1 for the first time.
    ),
]


def resolve_base_config(relpath):
    path = os.path.join(CONFIGS_DIR, relpath)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FEATURE_MATRIX entry points at a nonexistent config: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="*", help="Specific FEATURE_MATRIX label(s) to run; default is every cell")
    parser.add_argument("--ntasks", type=int, default=8, help="Number of srun tasks per training run (default 8, matching launch/*/*.sh)")
    parser.add_argument("--timeout", type=int, default=None, help=f"Per-run timeout in seconds, overriding both DEFAULT_TIMEOUT ({DEFAULT_TIMEOUT}) and any cell's own timeout_override")
    args = parser.parse_args()

    all_labels = {cell.label for cell in FEATURE_MATRIX}
    if args.labels:
        unknown = set(args.labels) - all_labels
        if unknown:
            sys.exit(f"Unknown FEATURE_MATRIX label(s): {sorted(unknown)}. Known labels: {sorted(all_labels)}")
        cells = [c for c in FEATURE_MATRIX if c.label in args.labels]
    else:
        cells = FEATURE_MATRIX

    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    scratch_root = f"/tmp/{job_id}/checkpoint_feature_matrix_smoke"

    results = []
    for cell in cells:
        base_config = resolve_base_config(cell.base_config_relpath)
        scratch_dir = os.path.join(scratch_root, cell.label)
        min_files = cell.min_files_override if cell.min_files_override is not None else DEFAULT_MIN_FILES
        timeout = args.timeout if args.timeout is not None else (cell.timeout_override or DEFAULT_TIMEOUT)
        print(f"\n{'='*80}\n{cell.label} ({cell.base_config_relpath}, overrides={cell.overrides})\n{'='*80}", flush=True)

        try:
            smoke_config = make_smoke_config(base_config, scratch_dir, min_files=min_files, extra_overrides=cell.overrides)
        except NoRealDataFoundError as e:
            # Fail fast, before ever spending GPU allocation time on a run
            # that's guaranteed to crash confusingly deep inside
            # calculate_load_balancing_on_the_fly with a ZeroDivisionError.
            fresh_status = "FAIL (no real data found)"
            print(f"[{cell.label}] {e}", flush=True)
            print(f"[{cell.label}] fresh run: {fresh_status} (srun never launched)", flush=True)
            results.append({
                "label": cell.label, "fresh_status": fresh_status, "fresh_elapsed": 0.0,
                "resume_status": "-", "resume_elapsed": None,
            })
            shutil.rmtree(scratch_dir, ignore_errors=True)
            continue

        fresh = run_fresh_phase(smoke_config, cell.label, args.ntasks, timeout, scratch_dir)

        resume_status = "-"
        resume_elapsed = None
        if cell.test_resume:
            if fresh["status"] == "PASS":
                resume = run_resume_phase(smoke_config, cell.label, args.ntasks, timeout)
                resume_status = resume["status"]
                resume_elapsed = resume["elapsed"]
            else:
                resume_status = "SKIPPED (fresh run didn't produce a checkpoint)"

        results.append({
            "label": cell.label,
            "fresh_status": fresh["status"],
            "fresh_elapsed": fresh["elapsed"],
            "resume_status": resume_status,
            "resume_elapsed": resume_elapsed,
        })

        shutil.rmtree(scratch_dir, ignore_errors=True)

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    ok = True
    for r in results:
        fresh_time = f"{r['fresh_elapsed']:.0f}s" if r["fresh_elapsed"] is not None else "-"
        resume_time = f"{r['resume_elapsed']:.0f}s" if r["resume_elapsed"] is not None else "-"
        print(f"{r['label']:38s} fresh={r['fresh_status']:8s}({fresh_time:>6s})  resume={r['resume_status']:8s}({resume_time:>6s})")
        if r["fresh_status"] != "PASS" or r["resume_status"] not in ("PASS", "-"):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
