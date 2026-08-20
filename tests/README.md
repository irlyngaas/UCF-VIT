# Running the test suite

This project uses [pytest](https://docs.pytest.org/). Tests live under `tests/`,
mirroring the layout of `src/UCF_VIT/`.

## Two tiers

- **Tier 1** (`tests/*.py`, `tests/dataloaders/`, `tests/utils/`): fast,
  single-process tests that don't need a GPU or a SLURM allocation. Runs
  anywhere in a couple of seconds.
- **Tier 2** (`tests/distributed/`): multi-process/multi-GPU tests for the
  code that genuinely needs a live `torch.distributed` process group.
  Launched via `sbatch launch/tests/run_distributed_tests.sh` on Frontier —
  see "Running the distributed (Tier 2) tests" below. These are meaningless
  as a single local process, so `tests/distributed/conftest.py` skips the
  whole directory (with a clear reason) if not actually launched under `srun`.

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
| `tests/utils/test_misc.py` | `is_power_of_two`, `calculate_tile_overlap`, `patchify`/`unpatchify` roundtrips |
| `tests/utils/test_pos_embed.py` | 1D/2D/3D sin-cos position embeddings, `SinusoidalEmbeddings` |
| `tests/utils/test_lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` warmup/annealing shape |
| `tests/utils/test_metrics.py` | `masked_mse`, `DiceBLoss` |
| `tests/test_config_validation.py` | Every YAML under `configs/` actually parses via `parse_config` |

`tests/conftest.py` provides a session-wide, autouse fixture that initializes
a single-process (`world_size=1`, `gloo` backend) `torch.distributed` group.
This exists because several functions (notably `parse_config`) call
`dist.get_rank()`/`dist.get_world_size()` even outside of a real multi-process
launch; the fixture lets those calls succeed locally without any SLURM
allocation. You don't need to do anything to use it — it's automatic.

## What isn't covered yet

- `UCF_VIT.model.arch` / `UCF_VIT.model.building_blocks` (needs `timm`,
  `monai`, and `xformers` — the last is GPU/build-toolchain-sensitive, so
  forward-pass tests for these are better verified directly in the
  `forge-vit` env on Frontier than guessed at in a generic dev environment).
- `get_model`'s FSDP wrap/checkpoint-resume path, and `training.py`'s
  `process_batch` tensor-parallel broadcasts. Both need a live distributed
  setup like Tier 2's, but weren't added in this first pass — a reasonable
  next step once `tests/distributed/`'s current tests are confirmed working.
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

I wasn't able to execute these against a live multi-GPU job myself (no
Frontier access in this environment) — they're written from a careful,
by-hand trace of `init_par_groups`'/`dist_functions.py`'s exact source, but
the first real run on Frontier is the actual verification. If something
fails, `test_smoke.py`'s result tells you whether to look at the environment
or at the specific collective's logic.

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
