# Running the test suite

This project uses [pytest](https://docs.pytest.org/). Tests live under `tests/`,
mirroring the layout of `src/UCF_VIT/`.

## Two tiers

- **Tier 1 (this directory)**: fast, single-process tests that don't need a
  GPU or a SLURM allocation. Runs anywhere in a couple of seconds.
- **Tier 2 (not yet added)**: multi-process/multi-GPU tests for the code that
  genuinely needs a live `torch.distributed` process group (`init_par_groups`,
  `dist_functions.py`'s custom collectives, `get_model`'s FSDP wrap and
  checkpoint-resume path, `process_batch`'s tensor-parallel broadcasts). These
  will be launched via a dedicated `sbatch` script on Frontier, modeled on the
  existing scripts in `launch/`, once added.

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
- `UCF_VIT.utils.dist_functions`, `init_par_groups`, `get_model`,
  `training.py` — need a real (or multi-process-simulated) distributed
  setup; this is what Tier 2 is for.
- `UCF_VIT.parse` itself only has indirect coverage today, through
  `test_config_validation.py` running real shipped configs end to end
  (rather than unit tests of individual branches) — this is what caught the
  config vs. parser mismatches below.

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
