# Running the test suite

This project uses [pytest](https://docs.pytest.org/). Tests live under `tests/`,
mirroring the layout of `src/UCF_VIT/`.

## Three tiers

- **Tier 1** (`tests/*.py`, `tests/dataloaders/`, `tests/utils/`): fast,
  single-process tests that don't need a GPU or a SLURM allocation. Runs
  anywhere in a couple of seconds.
- **Tier 2** (`tests/distributed/`): multi-process/multi-GPU tests for the
  code that genuinely needs a live `torch.distributed` process group.
  Launched via `sbatch launch/tests/run_distributed_tests.sh` on Frontier —
  see "Running the distributed (Tier 2) tests" below. These are meaningless
  as a single local process, so `tests/distributed/conftest.py` skips the
  whole directory (with a clear reason) if not actually launched under `srun`.
- **Tier 3** (`tests/integration/`): a real (tiny, tiny-model) training run
  against your actual data on Frontier, through the real
  `training_scripts/train.py` entry point, including a checkpoint
  save-then-resume cycle. Launched via
  `sbatch launch/tests/run_training_smoke.sh` — see "Running the training
  smoke test (Tier 3)" below. Not pytest-based, and not a single local
  process either — see that section for why.

## Setup

From the repo root, with your environment activated (e.g. the `forge-vit`
conda env on Frontier, or any environment that already has the project's
dependencies installed):

```bash
pip install -e ".[test]"
```

This installs `pytest` on top of the project's normal dependencies. If you
just want to run the Tier 1 tests somewhere lighter (no `monai`/`timm`/
`xformers`), you only need: `torch`, `torchvision`, `opencv-python-headless`,
`scipy`, `matplotlib`, `nibabel`, `torchdata==0.9.0`, `pyyaml`, and `pytest`.
Model-level tests (`UCF_VIT.model.*`) aren't in the suite yet for exactly this
reason — see "What isn't covered yet" below.

No editable install is required just to run tests: `pyproject.toml` adds
`src/` (and `utils/`) to pytest's `pythonpath`, so `import UCF_VIT` and
`import validate_config` work out of the box from the repo root.

## Running tests

```bash
# everything
pytest

# one file
pytest tests/dataloaders/test_quadtree.py

# one test
pytest tests/utils/test_misc.py::test_is_power_of_two

# verbose, stop at the first failure
pytest -v -x

# only the shipped-config regression checks
pytest tests/test_config_validation.py -v
```

## What's covered today

| File | Covers |
| --- | --- |
| `tests/dataloaders/test_quadtree.py` | `Rect` geometry, `FixedQuadTree` subdivision, node-value/encode-decode |
| `tests/dataloaders/test_octree.py` | `Cube` geometry, `FixedOctTree` subdivision |
| `tests/dataloaders/test_dataset.py` | `TileDataIter` (2D, 3D-full, and 3D-twoD-sliced tiling, with/without labels, overlap), `ShuffleIterableDataset` (no data loss/duplication across buffer sizes), `ProcessChannels` (batching, adaptive-patching wiring), `FileReader` (DDP-rank + dataloader-worker sharding disjointness/coverage up to `num_workers=7`, `keys_to_add` replication) |
| `tests/utils/test_misc.py` | `is_power_of_two`, `calculate_tile_overlap`, `patchify`/`unpatchify` roundtrips |
| `tests/utils/test_pos_embed.py` | 1D/2D/3D sin-cos position embeddings, `SinusoidalEmbeddings` |
| `tests/utils/test_lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` warmup/annealing shape |
| `tests/utils/test_metrics.py` | `masked_mse`, `DiceBLoss` |
| `tests/test_config_validation.py` | Every YAML under `configs/` actually parses via `parse_config` |

`tests/dataloaders/test_dataset.py`'s `TileDataIter` coverage is deliberately
thorough: that class is where a real, live bug was found and fixed this
session (a 3D `basic_ct` volume being twoD-sliced into 2D z-planes was
silently treated as genuinely-2D data, leaving the whole z-axis attached to
every tile and producing a 5D batch several layers downstream — see "Real
runs on Frontier so far" below). The `twoD=True`/`twoD=False` 3D tests there
are regression tests for exactly that bug.

Writing `test_filereader_*` also surfaced and fixed a second real bug (not
found via a real run, so not in the Frontier findings log below), more
serious than it first looked: `FileReader.__iter__`'s `num_workers=0`
branch (`torch.utils.data.get_worker_info()` returns `None` whenever
`num_workers=0`) never applied DDP-rank sharding at all — it unconditionally
set `iter_start=0, iter_end=len(file_list)`, so *every* DDP rank read the
*entire* file list instead of its own shard. This was live today, not a
landmine: `basic_ct/sap` and `basic_ct/unetr` both ship with
`num_workers: 0` and `simple_ddp_size: 8`. A related symptom of the same
broken branch: combined with `keys_to_add > 1` (dataset-balancing
replication across multiple, differently-sized `dict_root_dirs` keys), it
would walk past the end of `file_list` and raise `IndexError` — that part
alone doesn't hit any shipped config today (every shipped `basic_ct` config
has exactly one key), but the missing DDP-rank sharding did. Fixed by
routing `num_workers=0` through the same DDP-rank/`gx`-based sharding math
as `num_workers >= 1`, instead of duplicating (and getting wrong) a separate
code path for it. `test_filereader_num_workers_zero_shards_by_ddp_rank` and
`test_filereader_shards_combine_ddp_rank_and_dataloader_workers` (the
latter parametrized up to `num_workers=7`, matching real Frontier node
core counts) are the regression tests.

`tests/dataloaders/test_dataset_speed.py` has informational-only throughput
measurements (buffer_size, num_workers) for the same pipeline — no
pass/fail threshold, and not run by default; see "Running the dataloader
speed tests" below.

`tests/conftest.py` provides a session-wide, autouse fixture that initializes
a single-process (`world_size=1`, `gloo` backend) `torch.distributed` group.
This exists because several functions (notably `parse_config`) call
`dist.get_rank()`/`dist.get_world_size()` even outside of a real multi-process
launch; the fixture lets those calls succeed locally without any SLURM
allocation. You don't need to do anything to use it — it's automatic.

## Running the dataloader speed tests

`tests/dataloaders/test_dataset_speed.py` measures throughput of
`ShuffleIterableDataset` across `buffer_size` values and of a `DataLoader`
wrapping a synthetic (artificially delayed) source across `num_workers`
values. These are excluded from the default `pytest` run (see `addopts` in
`pyproject.toml`) since they take longer and, being timing measurements,
aren't meaningfully pass/fail — run them explicitly and read the printed
numbers:

```bash
pytest -m dataloader_speed -s
```

## What isn't covered yet

- `UCF_VIT.model.arch` / `UCF_VIT.model.building_blocks` (needs `timm`,
  `monai`, and `xformers` — the last is GPU/build-toolchain-sensitive, so
  forward-pass tests for these are better verified directly in the
  `forge-vit` env on Frontier than guessed at in a generic dev environment).
- `training.py`'s `process_batch` tensor-parallel broadcasts specifically
  (as opposed to the rest of a training run, which Tier 3 now covers
  end-to-end) — would need its own live distributed setup like Tier 2's.
- `UCF_VIT.parse` itself only has indirect coverage today, through
  `test_config_validation.py` running real shipped configs end to end
  (rather than unit tests of individual branches) — this is what caught the
  config vs. parser mismatches below.

## Running the distributed (Tier 2) tests

```bash
cd launch/tests
sbatch run_distributed_tests.sh
```

This requests 1 node / 8 GPUs (matching every other script in `launch/`) and
runs `srun -n 8 python -m pytest ../../tests/distributed/ -v` — each of the 8
tasks is an independent pytest process, one per rank, all running the same
test files. `tests/distributed/conftest.py` initializes an NCCL process group
per-process from `SLURM_PROCID`/`SLURM_NTASKS`/`SLURM_LOCALID` (the same env
vars `training_scripts/train.py`'s `init_dist` uses), so no `torchrun` wrapper
is needed. Output lands in `pytest-distributed-<jobid>.out` in that directory.

**Covered today:**

| File | Covers |
| --- | --- |
| `tests/distributed/test_smoke.py` | Basic connectivity: rank/world_size sanity, a plain `all_reduce`. Check this first if anything else fails — it isolates launch/environment problems from actual `UCF_VIT` bugs. |
| `tests/distributed/test_init_par_groups.py` | `init_par_groups`'s process-group membership (world size and this rank's local rank within each of the 5 returned groups), across several `tensor_par_size`/`fsdp_size`/`simple_ddp_size` splits of the job's actual world size. |
| `tests/distributed/test_dist_functions.py` | Forward (and, for `all_reduce`/`broadcast`, backward) correctness of the collective autograd ops most exercised by tensor parallelism: `all_reduce`, `broadcast`, `all_gather`, `gather`, `F_Identity_B_AllReduce`, `F_AllReduce_B_Identity`. |

**Important constraint if you add more tests here**: `init_par_groups` and the
`dist_functions.py` ops all make *collective* calls (`dist.new_group`,
`dist.broadcast`, etc.), which every process in the job must call in the same
order. Since each rank runs its own independent pytest process (there's no
cross-process pytest coordination here), any test skip/parametrize decision
must depend only on values that are identical across ranks and available
before any process has connected — i.e. `SLURM_NTASKS`/world size, never this
rank's own `SLURM_PROCID`. Both existing test files follow this pattern
(see the module docstrings); breaking it will hang or crash the job, not just
fail a test.

Verified with a real 1-node/8-GPU run on Frontier (job 5314922): all 14
tests passed on all 8 ranks. (An earlier run, job 5314802, caught a real bug
first — `tests/conftest.py`'s autouse fixture was also firing for
`tests/distributed/` since it's a parent directory, racing this directory's
own multi-process init and making every test error out at setup with
"trying to initialize the default process group twice!"; fixed in
`tests/conftest.py`, see its docstring.) If something fails after a future
change, `test_smoke.py`'s result tells you whether to look at the
environment or at the specific collective's logic.

## Running the training smoke test (Tier 3)

```bash
cd launch/tests
sbatch run_training_smoke.sh
```

Unlike Tier 2, this does **not** run under a top-level `srun` — `sbatch`
launches `tests/integration/run_training_smoke.py` as a single plain
process, and that script spawns its own `srun -n 8 python
training_scripts/train.py <config>` subprocess per training run. Nesting a
real training launch inside a pytest process already running under its own
`srun` (the way Tier 2 works) would hit the same "process group already
initialized twice" class of bug fixed for Tier 2 — running the driver as an
unwrapped single process and letting *it* own each `srun` call sidesteps
that entirely. Output lands in `training-smoke-<jobid>.out`.

For each of the 10 shipped configs, in order:

1. Writes a smoke-test config to `/tmp/$SLURM_JOB_ID/checkpoint_smoke_test/<config>/smoke.yaml`:
   real data paths, tiling, and adaptive-patching settings, unmodified; only
   the model is shrunk (`embed_dim=24, num_heads=2, depth=4` — `embed_dim`
   must be divisible by `LCM(4, 6)=12` for the 2D/3D sincos position
   embeddings and by `num_heads`), `max_epochs=1`,
   `resume_from_checkpoint=False`, `checkpoint_path` pointed at that scratch
   directory, and one of two real-data-narrowing mechanisms depending on
   `dataloader.type`, so only a small, fixed number of real files get read
   regardless of how large the real dataset actually is:
   - For the iterative dataloader (`basic_ct`/`imagenet`):
     `dict_start_idx`/`dict_end_idx` narrowed so only ~64 real files get read
     per dataset key. This is computed dynamically
     (`compute_narrow_dict_idx`) by calling the same
     `UCF_VIT.utils.misc.process_root_dirs` the real pipeline uses to get
     actual file counts on Frontier's filesystem, rather than guessing a
     fixed fraction that could round to 0 files for a modest dataset or
     barely help for one as huge as full ImageNet.
   - For the plain dataloader (`catsdogs`), which globs every real file
     directly in `train.py` with no config-level trimming knob at all:
     `create_narrow_catsdogs_dir` globs the real directory itself the same
     way `train.py` does, then points `dict_root_dirs` at a scratch
     directory of *symlinks* to a subset of the real files — `max(min_files,
     batch_size * data_par_size)` of them, not just `min_files`, since
     `train.py` wraps this dataset in a `DataLoader` with `drop_last=True`:
     with too few files, `DistributedSampler` would give some rank an
     undersized batch that gets silently dropped entirely, yielding 0
     iterations/epoch (and a false `PASS`) instead of an error.

   `--min-files` overrides the target for both mechanisms (default 64 —
   comfortably above `data_par_size` and every shipped config's
   `batch_size`, so at least one real batch per rank should still be
   possible; if a real run shows too few/many files, adjust this rather than
   editing either mechanism directly). If no real files are found at all —
   almost always a stale/wrong `dict_root_dirs` path in the config, not a
   code bug — this step raises `NoRealDataFoundError` with the offending key
   and path, and that config is marked `FAIL (no real data found)` **without
   ever launching `srun`**, rather than burning GPU allocation on a run
   that's guaranteed to crash confusingly deep inside
   `calculate_load_balancing_on_the_fly` with a bare `ZeroDivisionError`.
2. Runs it, and checks it exits 0 and actually wrote a rank-0 checkpoint.
3. Edits that *same* config file in place — `resume_from_checkpoint=True`,
   `checkpoint_filename="epoch_0"` (the file the fresh run actually
   produced), `max_epochs=2` — the way a real user resumes: flipping fields
   in their one config, not maintaining a separate "resume" file.
4. Runs it again and checks it exits 0 — this is the first real exercise of
   `get_model`'s and `load_optimizer_scheduler_from_checkpoint`'s actual
   resume path against a real checkpoint (Tier 2 only covers the
   process-group primitives underneath it).
5. Sets `resume_from_checkpoint` back to `False` in the file, then deletes
   the whole scratch directory for that config.

Prints a per-config, per-stage PASS/FAIL/TIMEOUT table at the end and exits
nonzero if anything failed. Defaults: 8 `srun` tasks and a 300s timeout per
run (basic_ct-unetr's fresh run alone measured 227s against real data); both
are CLI flags (`--ntasks`, `--timeout`, and `--min-files` — see above) if a
config needs more (or you want to fail faster). You can also run it directly
against one config instead of all 10:

```bash
python tests/integration/run_training_smoke.py configs/basic_ct/unetr/base_config.yaml
```

For iterating on a single failing config without paying for (or waiting on)
the other 9 every time, `run_training_smoke_single.sh` wraps this in its own
sbatch script with a shorter time limit. It shares the same 300s default
per-run timeout as `run_training_smoke.sh` (it doesn't pass `--timeout`
itself), so the two scripts can't drift out of sync — pass `--timeout`
explicitly to override it for just one run:

```bash
cd launch/tests
sbatch run_training_smoke_single.sh ../../configs/basic_ct/unetr/base_config.yaml
# extra flags pass straight through to run_training_smoke.py:
sbatch run_training_smoke_single.sh ../../configs/basic_ct/unetr/base_config.yaml --timeout 600 --min-files 128
```

**Writing this surfaced a real, serious bug before it ever touched
Frontier**: `parse_config`'s pre-flight checkpoint-existence check (used
whenever `resume_from_checkpoint: True`) looked for a bare
`<checkpoint_path>/<checkpoint_filename>` file — but `save_checkpoint`
always writes `<checkpoint_path>/epoch_<N>_rank_<R>.ckpt`, and the code that
actually loads a checkpoint looks for
`<checkpoint_path>/<checkpoint_filename>_rank_<N>.ckpt`. The pre-check's
filename never matched anything that gets created, so it would
`sys.exit("Checkpoint file does not exist")` on *every* resume attempt, for
any config, checkpoint present or not — `resume_from_checkpoint` could never
have worked in production before this. Fixed in `parse.py` to check every
tensor-parallel rank's actual checkpoint file (fully resolving a `#TODO`
left in that code for the multi-rank case); verified locally against real
`parse_config` calls with both a complete and a partially-missing checkpoint
set, for `tensor_par_size` of 1 and 2.

`compute_narrow_dict_idx` was verified against fabricated local directory
structures mimicking both the `basic_ct` (`imagesTr/`) and `imagenet`
(class-subdirectory) layouts, in both the "fewer real files than
`--min-files`" (kept as-is) and "more" (correctly trimmed) cases, plus a
real `parse_config` call end to end against those fixtures.

**First two real runs on Frontier:**

1. The `UCF_VIT` package resolved to a stale, unrelated checkout
   (`.../DUMMY_DATASET/src/UCF_VIT/...`) instead of this one, even though
   `training_scripts/train.py` itself was read from the right place —
   `PYTHONPATH=$PWD:$PYTHONPATH` (copied from every other `launch/*/*.sh`
   script) only helps if `$PWD` happens to contain `UCF_VIT`, which it
   doesn't from `launch/tests/`; the actual import was resolving via a
   separately-installed editable package. Not a bug in this repo's code —
   fixed by correcting the environment directly. Worth checking whether the
   *other* `launch/*/*.sh` scripts have silently had the same issue.
2. With that fixed, all 7 `basic_ct`/`imagenet` configs still failed, but for
   an unrelated reason confirmed by direct filesystem inspection: their
   `dict_root_dirs` paths are themselves stale (only `catsdogs`'s data path
   is currently valid). This is what motivated the `NoRealDataFoundError`
   fail-fast check described above — it turns a ~20s confusing
   `ZeroDivisionError` deep in a training subprocess into an immediate,
   clear "no real files found at `<path>`" before `srun` is even invoked.
3. All 3 `catsdogs` configs (the only ones with currently-valid data) hit
   the 150s per-run timeout rather than failing outright — `catsdogs` uses
   `dataloader.type: "dataloader"`, which globs every real file directly in
   `train.py` with no `dict_start_idx`/`dict_end_idx` trimming mechanism
   available, so this is plausibly just real data loading taking longer than
   150s rather than an actual hang. Not yet root-caused; try a longer
   `--timeout` (e.g. via `run_training_smoke_single.sh`, which defaults to
   300s) next.
4. With one `dict_root_dirs` path fixed, `basic_ct-unetr` (a 3D config —
   `twoD: False`) got past data loading and failed inside `get_model` with
   `assert embed_dim % 3 == 0` in `get_3d_sincos_pos_embed` — a bug in
   `TINY_MODEL_OVERRIDES` itself, not the library: `embed_dim=32` isn't
   divisible by 3. Tracing the exact assert chain: `get_2d_sincos_pos_embed`
   needs `embed_dim % 4 == 0` (halves it, then the halved value must itself
   be even) and `get_3d_sincos_pos_embed` needs `embed_dim % 6 == 0` (same
   halving, split three ways) — since the override applies to both 2D and 3D
   configs uniformly, fixed to `embed_dim=24` (divisible by `LCM(4,6)=12`,
   and by `num_heads=2`). Verified against the real `pos_embed.py` functions
   locally before pushing back.

That fixture testing also surfaced (but didn't need fixing to proceed) a
narrower, real edge case in `process_root_dirs`: when an `imagenet`-format
dataset has `<= data_par_size` classes, `classes_to_combine` is only assigned
inside an `if len(classes) > data_par_size:` block, so it's referenced
unassigned (`UnboundLocalError`) right after. Real ImageNet-1k (1000 classes)
is far above any realistic `data_par_size`, so this shouldn't affect the
configs actually shipped here — flagging it in case a future config points
at a small custom classification dataset in `imagenet` format.

## Validating a config file by hand

`utils/validate_config.py` is a standalone utility — the same one
`tests/test_config_validation.py` uses under the hood — for checking a single
config file without going through pytest:

```bash
python utils/validate_config.py configs/basic_ct/unetr/base_config.yaml

# also validate a pretrained-model config
python utils/validate_config.py configs/basic_ct/unetr/base_config.yaml \
    --pretrained-config configs/basic_ct/mae/base_config.yaml
```

It initializes its own single-process `torch.distributed` group, so it runs
standalone from a login node or laptop — no SLURM allocation needed. On
success it prints a short summary (model type, dataset, parallelism sizes);
on failure it prints the exact exception instead of a raw traceback.

All 10 shipped configs currently pass. This test caught three real,
previously-undiscovered bugs when it was first added (all since fixed):
`tiling.tile_overlap` given as a YAML float instead of an int (now guarded
against in `parse.py` directly, not just in the affected configs);
`dataset_options.imagenet_resize` never renamed to `resize` in the three
`configs/imagenet/*` files after the code was; and
`configs/basic_ct/sap/base_config.yaml`'s `ap.fixed_length` not satisfying
the constraints 2D adaptive patching requires. If this test starts failing
again after a config or `parse.py` change, that's a real regression, not a
flaky test.
