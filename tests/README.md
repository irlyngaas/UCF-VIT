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
- **Tier 3b** (`tests/integration/run_feature_matrix_smoke.py`): same
  mechanics as Tier 3, but for a different purpose — Tier 3 proves the 10
  shipped configs' shared baseline (advanced features off by default) still
  works; Tier 3b proves those advanced features (adaptive patching, tiling,
  `twoD`, tensor parallelism) still work when turned back on, one
  representative config per feature. Launched via
  `sbatch launch/tests/run_feature_matrix_smoke.sh` — see "Running the
  feature-matrix smoke test (Tier 3b)" below.

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
| `tests/dataloaders/test_dataset.py` | `TileDataIter` (2D, 3D-full, and 3D-twoD-sliced tiling, with/without labels, overlap), `ShuffleIterableDataset` (no data loss/duplication across buffer sizes), `ProcessChannels` (batching, adaptive-patching wiring, `separate_channels`), `FileReader` (DDP-rank + dataloader-worker sharding disjointness/coverage up to `num_workers=7`, `keys_to_add` replication) |
| `tests/dataloaders/test_datamodule.py` | `collate_fn` across `adaptive_patching`/`return_label`/`separate_channels`/`return_qdt`/dataset-type combinations, built from real `ProcessChannels` output rather than hand-fabricated tuples |
| `tests/datasets/test_catsdogs.py` | `CatsDogsDataset` (label-from-filename, resize/channel-first conversion, adaptive-patching shapes) and `CatsDogsCollate`, against small real JPEG files written to a temp dir — `catsdogs` is the only shipped dataset using `dataloader.type: "dataloader"` (a plain `Dataset` + `DistributedSampler`, not the `iterative_dataloader` stack the two rows above cover) |
| `tests/utils/test_misc.py` | `is_power_of_two`, `calculate_tile_overlap`, `patchify`/`unpatchify` roundtrips, `process_root_dirs` (`imagenet` per-class bucketing — evenly/non-evenly-divisible `> data_par_size`, `<= data_par_size`, bucket-content correctness — and the non-`imagenet` branch), `shard_mlp_state_dict`/`shard_attention_state_dict` (weight-slice reconstruction, `fc2.bias`/`proj.bias` summing back to the original exactly, `qk_norm` rejection) |
| `tests/utils/test_pos_embed.py` | 1D/2D/3D sin-cos position embeddings, `SinusoidalEmbeddings` |
| `tests/utils/test_lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` warmup/annealing shape |
| `tests/utils/test_metrics.py` | `masked_mse`, `DiceBLoss` |
| `tests/test_config_validation.py` | Every YAML under `configs/` actually parses via `parse_config` |
| `tests/integration/test_run_training_smoke_helpers.py` | `run_training_smoke.py`'s `compute_narrow_dict_idx` (real-data-found narrowing, empty-but-existing-dir and nonexistent-dir both raising `NoRealDataFoundError`, no-op for non-`iterative_dataloader` configs) and `deep_merge_config_overrides` (nested-key merge, wholesale-replace of non-dict values, new-key insertion, multiple independent sections) |
| `tests/integration/test_feature_matrix_smoke_helpers.py` | `run_feature_matrix_smoke.py`'s `FEATURE_MATRIX` well-formedness (unique labels, real base-config paths, no accidental list-valued `tile_overlap` overrides) and — the most valuable check — every cell's tiny-model config surviving a real `parse_config` call |

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
*entire* file list instead of its own shard. This was live at the time,
against `basic_ct/sap` and `basic_ct/unetr`'s then-shipped `num_workers: 0`
+ `simple_ddp_size: 8` (both now ship `num_workers: 1` as part of this
session's config-baseline reconfiguration, but the bug and its regression
test stand regardless of what any shipped config currently uses). A related
symptom of the same
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

`tests/dataloaders/test_datamodule.py`'s `collate_fn` tests surfaced and
fixed four more real bugs, all in the `return_label=False` +
`adaptive_patching=True` corner, which had drifted out of sync with its
(correct) `return_label=True` sibling — exactly the territory the code's own
`# TODO: Finish and Test separate_channels implementation` comments flagged
as unfinished:
1. `ProcessChannels.__iter__` raised `UnboundLocalError` for
   `separate_channels=True`: the per-channel patchify loop discarded the
   quadtree object into `_` instead of `qdt`, then referenced the
   never-assigned `qdt` name a few lines later.
2. The same discard-into-`_` mistake in the `separate_channels=False`
   sibling, but only triggered by `return_qdt=True` — unreachable in
   production today (`return_qdt` defaults to `False`, nothing sets it
   `True`).
3. `collate_fn`'s `seq` computation for `dataset="basic_ct"` +
   `separate_channels=True` produced a wrong, spurious extra dimension — the
   `return_label=True` branch applied an `expand_dims` meant only for
   `separate_channels=False`, without checking `separate_channels` first
   (unlike `size`/`pos` right below it, which already did).
4. `collate_fn`'s `return_label=False` branch never added the channel
   dimension `basic_ct`'s (typically single-channel, un-separated) `seq`
   needs at all, producing a 3D tensor where the model's
   `rearrange(x, 'b c s p -> b s (p c)')` needs 4D.

None of these are hit by any shipped config today — all four need either
`separate_channels: True` (every shipped config uses `False`), `return_qdt:
True` (nothing sets it), or a `basic_ct` MAE/DiffusionVIT config with
`do_ap: True` (both ship with `do_ap: False`) — but all four are real, and
all four are fixed. See `tests/dataloaders/test_datamodule.py`'s module
docstring for the full detail on each, and
`test_dataset.py::test_processchannels_separate_channels_does_not_crash` for
the `ProcessChannels`-level regression test on #1.

Writing `tests/datasets/test_catsdogs.py`'s `adaptive_patching=True` tests
surfaced a real bug too, this time in `training_scripts/train.py` rather
than in `catsdogs.py` itself: it constructed `CatsDogsDataset` with
`num_channels=conf["data"]["num_channels"]` — the whole `{key: count}`
dict — instead of the per-key int
`conf["data"]["num_channels"][dkey_train]`. Harmless when
`adaptive_patching` is `False` (`num_channels` goes unused then), but with
`adaptive_patching: True` it's stored as `Patchify.num_channels` and
immediately compared with `if self.num_channels > 1`, raising `TypeError:
'>' not supported between instances of 'dict' and 'int'`. The shipped
`catsdogs` config ships with `ap.do_ap: False`, so this was dormant, not an
active failure — confirmed fixed by constructing `CatsDogsDataset` and
`Patchify` directly (bypassing `train.py`) with real fabricated JPEGs and
`adaptive_patching=True` end to end.

`tests/utils/test_misc.py`'s new `process_root_dirs` coverage fixed a real,
if long-dormant, bug: for `imagenet`-format datasets with `<= data_par_size`
classes, `classes_to_combine` was only assigned inside an `if len(classes) >
data_par_size:` block, so `UnboundLocalError` on the very next line. Never
hit by any shipped config (real ImageNet-1k's 1000 classes are far above any
realistic `data_par_size`), but a real crash waiting for the first small
custom classification dataset in `imagenet` format. Fixed to combine 1 class
per bucket in that case (`len(classes)` buckets rather than `data_par_size`
— matches the function's own "`data_par_size` (or fewer) buckets"
docstring). The new tests also document, without fixing, a separate,
pre-existing, self-flagged limitation in the `> data_par_size` branch: when
`len(classes)` doesn't divide evenly by `data_par_size`, the leftover
classes past `data_par_size * classes_to_combine` are silently dropped —
already called out by its own `# TODO: Add shuffling for data_par_size if it
doesn't divide 1000 equally` comment, so left as-is here rather than folded
into this fix.

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

Runs anywhere (no GPU/SLURM needed) — that's just the local invocation. To
run it on a real Frontier compute node instead of a login node (mainly so
the `num_workers` cases have real, uncontended cores behind them):

```bash
cd launch/tests
sbatch run_dataloader_speed.sh
```

A single plain process (no `srun`, unlike `run_distributed_tests.sh`) — the
tests themselves are CPU-only and don't touch `torch.distributed` or the
GPUs. Output lands in `pytest-dataloader-speed-<jobid>.out`.

`tests/dataloaders/test_dataset_speed_real_data.py` replaces
`test_dataset_speed.py`'s made-up `time.sleep()` delay with genuine
NIfTI/JPEG decode, for `basic_ct/unetr`, `imagenet/classification`, and
`catsdogs/classification` — the same real construction
`tests/distributed/test_dataloader_real_pipeline.py` uses
(`parse_config`/`calculate_load_balancing_on_the_fly`/
`NativePytorchDataModule` for the first two; `CatsDogsDataset` +
`DistributedSampler` + `DataLoader` for `catsdogs`), just single-process —
`parse_config(..., load_balance_offline=True)` skips the
`data_par_size * tensor_par_size == world_size` assertion the real shipped
configs (`data_par_size=8`) would otherwise fail under a lone process, and
`calculate_load_balancing_on_the_fly`/`NativePytorchDataModule` both derive
sharding from the config's stated `data_par_size` rather than the real
world size regardless, so this process still ends up decoding a real,
correctly-sharded ~1/8th slice — genuine work, not a full-scale
measurement. Sweeps `num_workers` up to 7 (matching real Frontier node core
counts) for all three; needs real Frontier data to mean anything, so (like
`tests/distributed/`'s real-data files) it skips gracefully rather than
failing when the real paths aren't reachable — `run_dataloader_speed.sh`
above is what actually runs it against real data. See the module docstring
for a caveat on `num_workers > 0`: worker-subprocess startup cost is
included in the timed region, so it can dominate the measurement at small
batch counts, especially at higher worker
counts.

Writing this surfaced a real gap in `compute_narrow_dict_idx` (shared with
`test_dataloader_real_pipeline.py` and Tier 3's smoke test): it already
handled a `dict_root_dirs` path that exists but is empty (raising
`NoRealDataFoundError`, caught and skipped), but a path that doesn't exist
*at all* let a raw `FileNotFoundError` (from `process_root_dirs`'
`os.listdir`/`FileLister` calls) propagate uncaught instead — this only
surfaced now because this file is the first real-data test that isn't
gated behind a real SLURM launch (`test_dataloader_real_pipeline.py` lives
in `tests/distributed/`, so it's skipped entirely before ever reaching this
code path when run without `srun`). Fixed by normalizing both cases to the
same `NoRealDataFoundError`; `tests/integration/test_run_training_smoke_
helpers.py` (new — `run_training_smoke.py`'s narrowing helpers had no unit
tests of their own before this) is the regression test.

## What isn't covered yet

- `UCF_VIT.model.arch` — the full `VIT` forward pass (patch embedding,
  positional embedding, `Block` stacking) has no dedicated test yet, only
  indirect coverage via Tier 3/3b's real training runs.
  `UCF_VIT.model.building_blocks`'s `Mlp`/`Attention` tensor-parallel
  forward correctness and `Block`-level FSDP (`sharding_strategy=
  FULL_SHARD`) forward correctness *are* now covered — see
  `tests/distributed/test_tensor_parallel_correctness.py` and
  `tests/distributed/test_fsdp_correctness.py` below (needs `timm`,
  `monai`, and `xformers` — the last is GPU/build-toolchain-sensitive, so
  both files `importorskip` cleanly rather than erroring at collection
  when they're not installed). Combined `fsdp_size > 1` +
  `tensor_par_size > 1` (production's `HYBRID_SHARD` branch) is not yet
  covered — deferred follow-up once both of the above are proven.
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
| `tests/distributed/test_tensor_parallel_correctness.py` | Numerical correctness of `Mlp`/`Attention`'s real `tensor_par_size > 1` forward **and backward** pass: given weights sliced from an identical `tensor_par_size=1` reference module (via `UCF_VIT.utils.misc.shard_mlp_state_dict`/`shard_attention_state_dict`) and an identical input, checks the sharded forward pass (across real 2/4/8-rank tensor-parallel groups, using real `F_Identity_B_AllReduce`/`F_AllReduce_B_Identity`/`dist.all_reduce` collectives) matches the reference's output within `float32` tolerance. The backward tests additionally check, after a non-uniform-weighted `.sum().backward()`: the input's gradient (should already be fully all-reduced by `F_Identity_B_AllReduce`'s backward); each sharded parameter's gradient against the corresponding slice of the reference's full gradient (reusing the same, already-verified `shard_*_state_dict` slicing on the gradient tensors); and the unsharded `fc2.bias`/`proj.bias` gradient (same value on every rank, by linearity of the forward all-reduce-sum). Model-level only (not the full training loop/FSDP/checkpointing pipeline) — deliberately targets the exact class of bug found earlier this session in `training.py`'s `process_batch` and `arch.py`'s `_pos_embed`, which only a real multi-rank launch can exercise. |
| `tests/distributed/test_fsdp_correctness.py` | Numerical correctness of a small stack of real `Block`s wrapped in PyTorch's own FSDP with `sharding_strategy=FULL_SHARD` (`fsdp_size > 1`, `tensor_par_size=1`) against an identically-seeded, unwrapped reference, forward **and backward** — mirrors `model/utils.py`'s `get_model` `FULL_SHARD` branch exactly (same `FSDP(...)` call shape, `transformer_auto_wrap_policy` targeting `Block`), but with a `float32` `MixedPrecision` policy for a tight tolerance instead of production's `bfloat16`. The backward test checks the input's gradient directly (never sharded, no FSDP-specific handling needed) and every parameter's gradient via `FSDP.summon_full_params(..., with_grads=True)` (the documented way to materialize FSDP's internally-sharded gradients for inspection) against the reference. Combined `fsdp_size > 1` + `tensor_par_size > 1` (`HYBRID_SHARD`) is a deferred follow-up. |
| `tests/distributed/test_dataloader_real_data.py` | `FileReader`'s DDP-rank sharding and `ShuffleIterableDataset`'s no-loss/no-duplication guarantee, against real `basic_ct` and `imagenet` file lists on Frontier and `torch.distributed.get_rank()` for real (not simulated) across all `world_size` ranks, across `num_workers` (0/1/4) and `buffer_size` (1/20/100) — the real-scale counterpart to `tests/dataloaders/test_dataset.py`'s simulated-rank coverage of the same `num_workers=0` fix. File I/O itself is stubbed out (`FileReader.read_process_file` monkeypatched to a no-op) so this stays fast and focused on correctness, not decode speed. |
| `tests/distributed/test_catsdogs_real_data.py` | The real production `DistributedSampler` + `DataLoader` + `CatsDogsDataset`/`CatsDogsCollate` wiring, against real CatsDogs JPEGs and real ranks — disjoint/complete file sharding across `num_workers` (0/1/4), and `adaptive_patching=True` against real photo content (not synthetic random-noise JPEGs, unlike `tests/datasets/test_catsdogs.py`), which actually exercises Canny edge detection on real image structure. Unlike the row above, file I/O is *not* stubbed — `CatsDogsDataset.__getitem__` has no meaningful decode-free path. |
| `tests/distributed/test_dataloader_real_pipeline.py` | The full real pipeline — decode, tile, (for `basic_ct`) adaptive patch, collate — for `basic_ct`/`unetr`, `imagenet`/`classification`, and `catsdogs`/`classification`, each built via the exact real construction `train.py` itself uses for that dataloader type (`parse_config` + `calculate_load_balancing_on_the_fly` + `NativePytorchDataModule` for `basic_ct`/`imagenet`'s `iterative_dataloader`; a plain `CatsDogsDataset` + `DistributedSampler` + `DataLoader` for `catsdogs`'s `dataloader` type, which never touches the other two calls in production either). No stubbing anywhere; checks the actual decoded/collated batch (shapes, finite values, normalized ranges, valid label ranges, one-hot `seq_label` correctness for `basic_ct`'s real segmentation masks) rather than just sharding math. |

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

`test_dataloader_real_data.py` (added later, alongside the `FileReader`
`num_workers=0` DDP-sharding fix) bumps the sbatch time limit to 10 minutes
(from 5) since it's 2 datasets x 3 `num_workers` x 3 `buffer_size` = 18
parametrized cases. It covers both `basic_ct` (where the `num_workers=0` fix
specifically mattered — `basic_ct/sap` and `basic_ct/unetr` shipped with
`num_workers: 0` at the time; both now ship `num_workers: 1` as part of this
session's config-baseline reconfiguration) and `imagenet` (a meaningfully different code path in
`process_root_dirs`, per-class bucketing rather than a flat file listing,
though every shipped `imagenet` config uses `num_workers: 1` so wasn't
actually broken); `imagenet`'s real directory can be on the order of a
million files, so its file list here is deliberately narrowed to the first
few real class subdirectories (`IMAGENET_MAX_CLASSES` in the test file) —
same "real data, just less of it" principle as Tier 3's
`create_narrow_catsdogs_dir`. `num_workers > 0` combines a real DDP rank
with a *simulated* per-worker split (monkeypatched
`torch.utils.data.get_worker_info`, same technique as Tier 1), rather than
spawning real DataLoader multiprocessing workers — real ranks are what Tier
1 could only simulate and is the part that was actually broken; real
multiprocess workers under an already-NCCL-initialized process add real
complexity (CUDA-after-fork concerns) without testing anything Tier 1 didn't
already cover for the per-worker-split math itself. Each combination also
round-trips this rank's real shard through `ShuffleIterableDataset` at the
given `buffer_size` and checks the result is set-identical to the
unshuffled shard (no loss/duplication), reusing Tier 1's synthetic-data
invariant against real files instead. If a given `(num_workers,
buffer_size)` combination needs more real files than are actually available,
that combination is skipped rather than failed — see the test's docstring
(in practice this doesn't trigger: `Tr8_Training` turned out to have 852
real file pairs, not the ~8 its name suggests).

`test_catsdogs_real_data.py` is a different kind of check: `catsdogs` is the
only shipped dataset using
`dataloader.type: "dataloader"`, sharded by PyTorch's own
`DistributedSampler` rather than any UCF_VIT-custom logic like
`FileReader`'s — so there's no known bug to regression-test the way there
was for `test_dataloader_real_data.py`. Its value is confirming the real
production wiring (`train.py`'s exact
`DistributedSampler(..., num_replicas=data_par_size, rank=world_rank)` +
`DataLoader(..., num_workers=...)` + `CatsDogsCollate` construction) works
end to end against real files, real ranks, and — for `adaptive_patching`
specifically — real photo content, since Canny edge detection on an actual
photo exercises real image structure that
`tests/datasets/test_catsdogs.py`'s synthetic random-noise JPEGs can't.
Narrows the real `CatsDogs` directory to an exact multiple of `world_size`
real files (`FILES_PER_RANK * world_size`) so `DistributedSampler`'s default
`drop_last=False` padding (which repeats samples to round up to a multiple
of `num_replicas`) never kicks in, keeping the disjointness check
unambiguous. File reads are *not* stubbed here, unlike
`test_dataloader_real_data.py` — `CatsDogsDataset.__getitem__` has no
meaningful decode-free path, and real decode against a handful of narrowed
files is fast enough not to need it.

Both files verified with a real 1-node/8-GPU run on Frontier (job 5321217):
all 36 tests passed on all 8 ranks (the original 14 plus these two files'
18 + 4 new parametrized cases) in 28s.

`test_dataloader_real_pipeline.py` (added later, not yet run on Frontier)
closes what was, as of this session, the biggest remaining gap in
dataloader test coverage: every other real-data test above deliberately
stubs out file I/O to stay fast and focus on sharding correctness, so
nothing previously decoded a real file and checked what came out the other
end of `TileDataIter`/`ProcessChannels`/`collate_fn`. This file does exactly
that, with no stubbing, for `basic_ct/unetr` (real NIfTI decode, the
baseline config -- no tiling/adaptive patching -- and real segmentation
labels -- the other gap flagged this session, since
`test_dataloader_real_data.py` only ever globs `imagesTr`, never
`labelsTr`), `imagenet/classification` (real JPEG decode +
resize, real classification labels), and `catsdogs/classification` (also
real JPEG decode + resize + classification labels, but a completely
different production code path — `dataloader.type: "dataloader"`, not
`"iterative_dataloader"`). Rather than hand-assembling
`FileReader`/`TileDataIter`/`ProcessChannels`/`collate_fn` (or, for
`catsdogs`, `CatsDogsDataset`/`CatsDogsCollate`) a third or fourth time, each
config is built via the exact real construction `train.py` itself makes for
its dataloader type: `parse_config` + `calculate_load_balancing_on_the_fly`
+ `NativePytorchDataModule` for `basic_ct`/`imagenet` — skipping only the
model/optimizer/training-loop parts this file has no need for — or, for
`catsdogs`, a plain `CatsDogsDataset` + `DistributedSampler` + `DataLoader`
straight from `parse_config`'s real output, since production never calls
`calculate_load_balancing_on_the_fly`/`NativePytorchDataModule` at all for
`dataloader.type: "dataloader"`. Narrowing is reused rather than
reimplemented too: Tier 3's `compute_narrow_dict_idx` for
`basic_ct`/`imagenet`, its `create_narrow_catsdogs_dir` for `catsdogs`. All
three configs' real `parallelism` settings (`data_par_size=8`,
`tensor_par_size=1`) already exactly match this file's real 8-rank launch,
so unlike Tier 3's smoke test, no parallelism overrides are needed anywhere.
Building `catsdogs`'s version through real `parse_config` output (rather
than `test_catsdogs_real_data.py`'s hand-picked constants) also gives free
regression coverage for the `num_channels` dict-vs-int `train.py` wiring bug
fixed earlier this session (see `tests/datasets/test_catsdogs.py`'s module
docstring) — a wrong argument there surfaces here as a real crash.

`compute_narrow_dict_idx`'s `min_files` means something different across the
`iterative_dataloader` datasets, which the test accounts for with separate
constants: for `basic_ct` (one `dict_root_dirs` key) it's a total file count
that `FileReader`'s own sharding then divides across ranks; with the
baseline config's `do_tiling=False`/`twoD=False`, each real file is exactly
one sample (no tiling/z-slice multiplication), so `min_files` must directly
cover `batch_size * NUM_BATCHES_TO_CHECK * data_par_size` real files total
(`BASIC_CT_MIN_FILES = 512`), the same reasoning as `catsdogs` below. For
`imagenet`, `process_root_dirs` already buckets real files one bucket per
rank *before* `min_files` narrows within each bucket — so `min_files` there
must directly cover `batch_size * NUM_BATCHES_TO_CHECK` real images per rank
(`IMAGENET_MIN_FILES = 100`, comfortably above `32 * 2 = 64`), not just per
dataset. `catsdogs`'s
`create_narrow_catsdogs_dir` has its own, different floor
(`max(min_files, batch_size * data_par_size)`), so `CATSDOGS_MIN_FILES` is
set explicitly to `batch_size * NUM_BATCHES_TO_CHECK * data_par_size` rather
than relying on that floor to happen to be enough. Getting any of these
wrong doesn't error loudly — it just silently returns fewer batches than
expected, caught here by asserting the exact batch count pulled.

Verified locally against fabricated NIfTI/JPEG files structured to match
the real directories (single-process, `parallelism` scaled to
`world_size=1` so `parse_config`'s real-world-size assertion holds without
needing 8 real ranks) before ever touching real Frontier data — all three
configs' full chain (decode through collate) produces correctly-shaped,
finite, correctly-ranged batches.

`test_tensor_parallel_correctness.py` and `test_fsdp_correctness.py` (added
later, real multi-rank numerical-correctness checks for `tensor_par_size >
1` and `fsdp_size > 1` respectively — see their own module docstrings and
their `tests/utils/test_misc.py` weight-slicing-helper coverage above). Both
`importorskip` cleanly in any environment without `timm`/`monai`/`xformers`
installed, confirmed locally. Run `run_distributed_tests.sh` to exercise
them for real; if either file's tolerance (`rtol`/`atol`) turns out too
tight, retune and rerun.

First real run (job 5339980) found a real bug in `test_fsdp_correctness.py`
itself, not production code: `init_par_groups` was called with
`data_par_size=fsdp_size` for every `fsdp_size` value, but `data_par_size`
must span the *entire* world when `tensor_par_size=1` (production's own
`parse.py` asserts `data_par_size*tensor_par_size == world_size`) — passing
just `fsdp_size` (2 or 4, both < `WORLD_SIZE:8`) left every rank outside
`range(fsdp_size)` with `fsdp_group=None`, so those ranks' `FSDP(...,
process_group=None, ...)` call silently fell back to the world-default
8-rank group instead — a real size mismatch against the `fsdp_size`-rank
group the other ranks got, for the *same* nominal wrap call. This aborted
Frontier's real run with a fatal `Aborted` signal inside FSDP's own
`_move_states_to_device`, mid-`fsdp_size=2` for most ranks (2 of 8 ranks —
the ones inside the real subgroup — did print `PASSED` first). Fixed by
passing `data_par_size=WORLD_SIZE` and `simple_ddp_size=WORLD_SIZE //
fsdp_size` (not `1`), which makes `init_par_groups` actually partition the
full 8-rank world into `WORLD_SIZE // fsdp_size` independent, symmetric
`fsdp_size`-rank FULL_SHARD groups. `test_tensor_parallel_correctness.py`'s
own `init_par_groups` call already derived `data_par_size = WORLD_SIZE //
tensor_par_size` correctly and needed no change. Neither
`test_init_par_groups.py` nor `test_tensor_parallel_correctness.py` (later
alphabetically) got to run at all in that job — the whole `srun` step was
cancelled once one rank aborted. Fixed locally (full Tier 1 suite green,
153 passed) — verified against real Frontier data by job 5341031 below
(confirmed working: no more abort, all `fsdp_size` values pass).

Second real run (job 5341031), after that fix: `test_fsdp_correctness.py`
passed for real (all `fsdp_size` values, no more abort), and
`test_mlp_tensor_parallel_forward_matches_reference` passed too, but
`test_attention_tensor_parallel_forward_matches_reference` failed for
every `tensor_par_size` (2, 4, 8) with ~99.9% of elements mismatched
(`Greatest relative difference: 719...`) — not a tolerance problem, a real
correctness bug. Found in `UCF_VIT.utils.misc.shard_attention_state_dict`
itself (not production `Attention` code, which never needs to convert a
`tensor_par_size=1` reference's weights into a sharded set — only this
test's helper does): `qkv`'s output (dim 0, size `dim * 3`) is 3
contiguous `dim`-sized blocks (Q, K, V), each internally split into
`num_heads` head-groups, and `Attention.forward` reshapes each rank's own
`qkv` output as `(3, num_heads // tensor_par_size, head_dim)` — meaning
every rank's shard must be the *same* head range from each of Q, K, and V
independently. `shard_attention_state_dict` instead took one flat
contiguous row-slice of the whole `dim * 3`-sized output, which — for
`tensor_par_size:2`, `dim:64` — gives rank 0 all of Q plus the first half
of K, and rank 1 the second half of K plus all of V: a completely
different (and wrong) partition. `proj`'s column-slice (dim 1) was
already correct as a plain contiguous chunk, since head ranges are
assigned to ranks in contiguous order and `proj`'s input columns are
head-major, so the two coincide there — only `qkv` needed the fix. Fixed
by adding a required `num_heads` parameter (not derivable from the state
dict's shapes alone) and slicing each of the 3 Q/K/V blocks by head range
before re-concatenating. `tests/utils/test_misc.py`'s
`test_shard_attention_state_dict_reconstructs_full_weights` had not
caught this: concatenating any contiguous partition back together always
reconstructs the original tensor via `torch.cat`, regardless of whether
the partition boundaries have the right *semantic* meaning, so it exercised
none of the actual bug. Replaced with
`test_shard_attention_state_dict_slices_qkv_by_head_range_not_flat_chunk`,
which reimplements the expected head-range reshape independently and
checks each shard against it directly, and
`test_shard_attention_state_dict_reconstructs_full_proj_weight` (the old
test's still-valid `proj.weight` assertion, split out on its own since
`proj`'s slicing didn't need the same treatment). Fixed locally (full
Tier 1 suite green, 156 passed).

Third real run (job 5341143), after the `shard_attention_state_dict` fix:
**all 8 ranks report 49 passed, 0 failed** — `test_fsdp_correctness.py`,
`test_tensor_parallel_correctness.py` (`Mlp` and now `Attention` too, all
of `tensor_par_size` 2/4/8), and every other file in `tests/distributed/`
all pass cleanly on real Frontier data. Both new correctness test files
are now fully verified for real, closing out this round of Tier 2 work.

### Major finding while scoping a combined `fsdp_size > 1` + `tensor_par_size > 1` test

While investigating how to build that deferred test (which needs to
construct a real `Block`/`VIT`-family model with `tensor_par_size > 1` the
same way production does, to wrap in `HYBRID_SHARD` FSDP), found that
**`model/utils.py`'s `get_model` never actually wires `tensor_par_size`/
`tensor_par_group` into the model it builds.** `model_arch(...)`'s
constructor call (and the `use_pretrained_model` path's
`pretrained_model_arch(...)` call) only passes `**conf['model']['kwargs']`
plus a fixed list of other kwargs — never `tensor_par_size`/
`tensor_par_group` — and `conf['model']['kwargs']` (built by `parse.py`'s
`get_kwargs`) never sets them either. So every `VIT`/`SAP`/`MAE`/`UNETR`/
`DiffusionVIT` instance built by `get_model` fell back to the class
defaults (`tensor_par_size=1, tensor_par_group=None`) **regardless of
`conf["parallelism"]["tensor_par_size"]`.**

Practical effect: every `if self.tensor_par_size > 1:` guard throughout
`arch.py`/`building_blocks.py` (the real `Attention`/`Mlp` sharding, MAE's
noise-mask broadcast, etc.) never fired in production. The model was built
at full, unsharded size on every rank — `training.py`'s `process_batch`
still correctly distributed data according to the real `tensor_par_group`
(wired directly from `train.py`, not through `get_model`), so
`tensor_par_size > 1` configs ran to completion with no crash — which is
exactly why every `+tensor_par` cell in the Tier 3b feature-matrix passed
— but did **zero actual model-parallel sharding**: pure redundant compute
plus broadcast overhead, not real tensor parallelism. This has presumably
been true since tensor parallelism was first added to this repo, and is
unrelated to any of this session's other fixes.

Fixed by passing `tensor_par_size=conf["parallelism"]["tensor_par_size"]`
and `tensor_par_group=tensor_par_group` (already an existing `get_model`
parameter) to both `model_arch(...)` call sites. Fixed locally (full Tier
1 suite green, 156 passed) — **not yet verified against real Frontier
data.** This is a much bigger behavioral change than the earlier fixes in
this file (those were data-plumbing bugs; this changes whether the model
itself shards at all for the first time), so every `tensor_par_size > 1`
config — including every already-passing `+tensor_par` Tier 3b cell and
the whole Tier 2 distributed suite — needs a fresh real run to check for
fallout: real sharded `Attention`/`Mlp` layers, actually exercised for the
first time by the full `train.py` pipeline, may surface new bugs that
running full-size unsharded models never could.

### Backward-pass coverage added to both correctness test files

Both `test_tensor_parallel_correctness.py` and `test_fsdp_correctness.py`
were forward-pass-only until now (every model call wrapped in
`torch.no_grad()`) — correct forward output says nothing about whether
gradients are synchronized correctly across the parallel group during
backprop, which is what `F_Identity_B_AllReduce`/`F_AllReduce_B_Identity`
(tensor parallelism's hand-rolled autograd functions) and FSDP's own
gradient reduce-scatter actually exist to do; a bug there wouldn't crash
training, just silently corrupt gradients. Added
`test_mlp_tensor_parallel_backward_matches_reference`,
`test_attention_tensor_parallel_backward_matches_reference`, and
`test_fsdp_full_shard_backward_matches_reference` — see the updated Tier 2
table entries above and each test's own docstring for exactly what's
compared and why. Fixed/added locally (full Tier 1 suite green, 156
passed unchanged — these are all-new Tier 2 files that skip cleanly via
`importorskip` locally); **not yet run against real Frontier data.**

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

   `--min-files` overrides the target for both mechanisms (default 256 —
   comfortably above `data_par_size` (8) and every shipped config's now
   shared `batch_size` (32), so at least one real batch per rank should
   still be possible; if a real run shows too few/many files, adjust this
   rather than editing either mechanism directly). If no real files are found at all —
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
run; both are CLI flags (`--ntasks`, `--timeout`, and `--min-files` — see
above) if a config needs more (or you want to fail faster). All 10 shipped
configs share the same data-pipeline baseline (tensor parallelism, adaptive
patching, tiling, and — for `basic_ct` — `twoD` all off by default), with
three documented, architecture-driven exceptions for `basic_ct`: `sap` keeps
`ap.do_ap: True` (`parse.py` hard-requires it for `SAP`) and `patch_size: 4`
(its decoder is a single `ConvTranspose3d(kernel=stride=patch_size)`, whose
memory scales as `patch_size**3`); `unetr` keeps `batch_size: 4` (its
skip-connection decoder runs a plain `Conv3d` on the full-resolution volume
regardless of `patch_size`/`embed_dim`, so memory scales with `batch_size`
directly). No config currently needs a different `--min-files`/`--timeout`
CLI value than the shared default. You can also run it directly against one
config instead of all 10:

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

**Real runs on Frontier so far:**

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
5. With that fixed, `basic_ct-unetr`'s **fresh run passed end to end for the
   first time** — real data loading, model construction, training, and
   checkpoint save all completed (227s) — but its resume run failed
   immediately with `RuntimeError: No backend type associated with device
   type cpu` inside `get_model`'s checkpoint-resume rank-sync broadcast
   (added earlier this session): `torch.tensor(epoch_start, ...)` created a
   CPU tensor by default, but `dist.broadcast` on an NCCL process group
   needs a GPU tensor. Fixed by passing `device=device` (already used
   elsewhere in `get_model` for the same purpose) to that call, and
   proactively to the following `dist.broadcast_object_list` call too, since
   it would have hit the identical failure right after — PyTorch added a
   `device` parameter to that function specifically for this NCCL
   requirement.
6. With both fixed, **`basic_ct-unetr` passed fully — fresh and resume —
   against real data**, the first shipped config to do so end to end. The
   first genuine, from-scratch validation of this session's `get_model`
   checkpoint-resume consistency fix, the `parse_config` checkpoint-existence
   fix, and everything else these logs surfaced along the way.
7. Root-caused the `catsdogs` timeout from item 3: `dataloader.type:
   "dataloader"` has no index-based trimming knob, so with no narrowing at
   all, every real file in `dict_root_dirs` gets globbed and decoded each
   epoch — for a real `catsdogs` directory of any real size, comfortably
   over 150s (or 300s). Fixed by adding `create_narrow_catsdogs_dir` (see
   above): globs the real directory the same way `train.py` does, then
   symlinks a `max(min_files, batch_size * data_par_size)`-sized subset into
   a scratch directory and points `dict_root_dirs` at that instead.
   **Verified: `catsdogs-classification` now passes against real data** with
   this fix in place.
8. A full `run_training_smoke.sh` run (job 5317650) against real data: all 3
   `catsdogs` and all 3 `imagenet` configs **passed fully, fresh and
   resume** — 6 of 10 shipped configs now fully verified end to end.
   `basic_ct-unetr` and `basic_ct-diffusion` both hit `run_training_smoke.sh`'s
   then-default 150s timeout — not a regression: `basic_ct-unetr`'s fresh run
   had already measured 227s in item 6, comfortably over 150s but under
   `run_training_smoke_single.sh`'s separate 300s default, so the two
   scripts' timeouts had silently drifted apart. Fixed by making 300s the one
   shared default both scripts use (see "Running the training smoke test"
   above) — `run_training_smoke_single.sh` no longer hardcodes its own
   `--timeout`.

   `basic_ct-mae` and `basic_ct-sap` both failed for real, and for the same
   root cause: `UCF_VIT.parse.parse_config` computes `data.tile_size` as a
   plain 2-tuple `(x, y)` whenever `twoD` is `True` — which conflates two
   different situations that both set `twoD=True`: genuinely 2D data
   (`imagenet`/`catsdogs`, where `img_size` itself has 2 entries and a
   2-tuple `tile_size` is correct) and 3D data being *sliced* into 2D
   z-planes (`basic_ct` with `data.twoD: True`, where `img_size` has 3
   entries but `tile_size` collapsed to 2D anyway). `TileDataIter.__iter__`
   uses `len(self.tile_size) == 3` to decide whether the *raw* incoming data
   is 3D; with a collapsed 2-tuple, `basic_ct-mae`/`basic_ct-sap` silently
   took the plain-2D tiling branch, which only slices the x/y axes and
   leaves the entire untouched z-axis (256) tacked onto every tile —
   producing a 5D batch (`(32, 1, 64, 64, 256)`) by the time it reached
   `PatchEmbed` (`ValueError: too many values to unpack (expected 4)` for
   `basic_ct-mae`) or the quadtree serializer (`...expected 3` for
   `basic_ct-sap`). `basic_ct-unetr` (`twoD: False`, full 3D tiling) never
   exercised this branch, and `imagenet`/`catsdogs` never exercised it
   incorrectly (their `img_size` genuinely is 2D), which is why this went
   undetected until the first real run of a `twoD: True` `basic_ct` config.
   Fixed in three places:
   - `parse.py`: `tile_size` is now a 3-tuple whenever `img_size` is 3D,
     regardless of `twoD` — for `twoD: True` the z entry is the raw,
     undivided depth (`img_size[2]`, not tiled — z-planes are walked one
     index at a time, not chunked), restoring `len(tile_size) == 3` as an
     accurate "raw data is 3D" signal.
   - `TileDataIter.__init__`: `tile_size_no_overlap` now loops over
     `len(tile_overlap)` instead of `len(tile_size)`, since `tile_size` can
     now have an extra z entry with no corresponding overlap value (would
     otherwise `IndexError`).
   - `TileDataIter.__iter__`: its z-slice loop bound was `data.shape[2]`,
     which is actually the *y*-axis size for channel-first `(C, X, Y, Z)`
     data, not z — silently harmless only because `basic_ct`'s volumes are
     cubic (256×256×256). Changed to `self.tile_size[2]` (the now-correct z
     size threaded through from `parse.py`), which is right regardless of
     whether the volume is cubic.
   Verified locally: a synthetic `(1, 256, 256, 256)` sample through the
   fixed `TileDataIter` now yields exactly `(1, 64, 64)` tiles (was `(1, 64,
   64, 256)`), and `parse_config` against the real `basic_ct-mae`/
   `basic_ct-sap`/`basic_ct-unetr` configs now reports `tile_size` of
   `(64, 64, 256)`, `(64, 64, 256)`, and `(64, 64, 64)` respectively. Not yet
   verified against a real Frontier run — and since `basic_ct-mae`/
   `basic_ct-sap` will now actually iterate instead of crashing in ~30-45s,
   watch for a *new* timeout: each real volume yields `div² × 256` z-sliced
   tiles (4096 for `div=4`) versus `basic_ct-unetr`'s `div³` full-3D tiles
   (64), so an epoch over even a handful of real files could turn out to
   need much more than 300s.
9. Confirmed: a rerun had **7 of 10 configs pass** (all 3 `catsdogs`, all 3
   `imagenet`, and now `basic_ct-unetr` too), leaving `basic_ct-diffusion`,
   `basic_ct-mae`, and `basic_ct-sap` timing out — as predicted in item 8 for
   `mae`/`sap`. Root-caused with real numbers this time: `Tr8_Training` has
   852 real file pairs (not ~8 as its name suggests), so the shared
   `min_files=64` default narrows to ~64 files total, ~8/rank across
   `simple_ddp_size=8`. For `mae`/`sap` (`twoD: True`, `tiling.div: 4`),
   each of those real files yields `div² × 256 = 4096` z-sliced tiles (not
   `div³ = 64` the way `basic_ct-unetr`'s `twoD: False` does), so 8
   files/rank is ~32768 tiles/rank/epoch. `basic_ct-diffusion` is
   `twoD: False` with the same tile-count formula, `batch_size`, and
   `fixed_length` as `basic_ct-unetr` (whose own fresh run measured 227s,
   only ~73s under the then-300s default) — no tile-count explanation, so
   plausibly just real Frontier I/O/scheduling variance pushing it over that
   thin margin.

   Added `PER_CONFIG_OVERRIDES` in `run_training_smoke.py` — a slug-keyed
   dict of `min_files`/`timeout` overrides applied under the shared
   `--min-files`/`--timeout` CLI defaults but under an explicit CLI flag
   (which always wins over both): `basic_ct-mae`: `min_files=16` (~2
   files/rank — chosen with margin above the `simple_ddp_size=8` floor,
   against `int()` truncation in `FileReader`'s start/end-idx narrowing,
   rather than cutting to the bare 1-file/rank minimum), bringing it to
   ~256 iterations/epoch, close to `basic_ct-unetr`'s own real iteration
   count. `basic_ct-sap`: same `min_files=16`, but its `batch_size=2` (vs
   `mae`'s 32) means that's still ~4096 iterations/epoch — `min_files` is
   already near its practical floor (can't drop below ~8 total without
   going under 1 file/rank and hitting `FileReader`'s own `per_worker > 0`
   assertion), so `sap` additionally gets `timeout=1800` as the real lever;
   if that's still not enough, the next lever would be lowering `tiling.div`
   for the smoke test specifically (fewer spatial tiles per z-slice), not
   `min_files`. `basic_ct-diffusion`: `timeout=900`, no `min_files` change
   (no tile-count case for it). Bumped both launch scripts' sbatch time
   limits for the new worst-case totals
   (`run_training_smoke.sh`: 90min → 2h; `run_training_smoke_single.sh`:
   20min → 75min, since `sap` alone could now need up to 30min for a single
   run). Not yet verified against a real Frontier run — the `min_files=16`
   estimates in particular are rough (based on iteration-count reasoning,
   not a measured per-iteration cost), so treat these as a first attempt to
   tune from, not a guaranteed fix.

10. Rather than keep tuning `PER_CONFIG_OVERRIDES` around `basic_ct-mae`'s
    persistent timeout, reconfigured all 4 `basic_ct` configs
    (`unetr`/`mae`/`sap`/`diffusion`) to share one baseline: `ap.do_ap`,
    `tiling.do_tiling`, and `data.twoD` all `False` (removing the
    `twoD: True` + `do_tiling: True` z-slice/tile explosion that motivated
    the overrides in the first place -- each real volume now yields exactly
    one sample), `tiling.div`/`tiling.tile_overlap` set to `1`/`0`
    (cosmetic -- `parse.py` already force-overrides these whenever
    `do_tiling: False`), `data.patch_size: 32` (512 tokens/sample, the same
    order of magnitude as `imagenet`/`catsdogs`'s 256-token baseline), and
    `dataloader.batch_size`/`num_workers`/`dict_buffer_sizes.ct1` unified to
    `32`/`1`/`100`, matching `imagenet`/`catsdogs`. One exception, found by
    running `tests/test_config_validation.py` against the edited configs:
    `parse.py` (`get_kwargs`) hard-asserts `SAP` requires `do_ap: True` --
    architectural, not a cost tradeoff -- so `basic_ct-sap` keeps
    `do_ap: True` with `fixed_length: 512` (needed a value satisfying both
    "cube root is a whole number" and "`fixed_length % 7 == 1`" now that
    `twoD: False` routes it through the octree check instead of the quadtree
    one its old `twoD: True` + `fixed_length: 196` combination satisfied).
    Confirmed all 10 shipped configs still parse (`pytest
    tests/test_config_validation.py -v`) and manually verified `tile_size`/
    `twoD`/`patch_size`/`do_ap`/`fixed_length` come out as intended for all
    4 edited configs. Updated
    `tests/distributed/test_dataloader_real_pipeline.py`'s
    `test_real_pipeline_basic_ct_unetr` for the new non-adaptive 4-tuple
    batch shape (`inp, label, variables, dict_key`, mirroring
    `test_real_pipeline_imagenet_classification`) -- including that
    non-adaptive `basic_ct` labels come straight from `FileReader` as
    `int64`, not the `uint8` the adaptive-patching branch explicitly casts
    to, a real dtype difference between the two code paths this test now
    asserts on. `BASIC_CT_MIN_FILES` in both that file and
    `tests/dataloaders/test_dataset_speed_real_data.py` raised from 16 to
    `batch_size(32) * N * data_par_size(8)` (512 and 1024 respectively) --
    with tiling/z-slicing no longer multiplying each file into many
    samples, `min_files` must now directly cover a full batch per rank, the
    same reasoning `catsdogs` already used. Removed `PER_CONFIG_OVERRIDES`
    from `run_training_smoke.py` entirely (all 4 `basic_ct` configs now
    share the same defaults as everything else) and raised
    `DEFAULT_MIN_FILES` from 64 to 256 for the same batch-coverage reason.
    Confirmed real basic_ct file count is comfortably in the hundreds+
    (consistent with `Tr8_Training`'s known 852 real file pairs from item 9
    above), so `batch_size: 32` is achievable. Full local `pytest -q` (Tier
    1) passes (113 passed, 36 skipped -- the real-data tests needing
    Frontier mounts). **Not yet verified against a real Frontier run** --
    next step is rerunning `run_training_smoke.sh` (Tier 3) and
    `run_distributed_tests.sh` (Tier 2, for the updated real-pipeline test)
    to confirm all 10 configs now pass with the simplified defaults, and to
    get a real "real runs on Frontier" measurement to replace item 9's
    now-obsolete tile-count-based numbers.
11. Real run (job 5322693) against item 10's baseline: **`basic_ct-diffusion`
    passed fully, fresh and resume** (177s/102s) -- confirming the
    tiling/`twoD` fix actually resolves the timeout it was meant to.
    `basic_ct-mae`, `basic_ct-sap`, and `basic_ct-unetr` all failed, each for
    a different, genuine reason -- not the same root cause repeating:
    - `basic_ct-mae`: `AssertionError: embed_dim % 3 == 0` in
      `get_3d_sincos_pos_embed`, called from `MAE.init_weights` for the
      *decoder* pos embed. `decoder_embed_dim: 512` was never touched by
      this session's baseline changes (it's a per-model decoder setting, not
      an "advanced feature"), but flipping `mae`'s `twoD` to `False` routed
      it through the 3D sincos path (`embed_dim % 3 == 0`) instead of the 2D
      one (`% 4 == 0`) it satisfied before -- 512 divides by 4 but not by 3.
      Fixed: `decoder_embed_dim: 480` (divisible by both 3 and
      `decoder_num_heads=16`).
    - `basic_ct-sap`: `torch.OutOfMemoryError: HIP out of memory. Tried to
      allocate 512.00 GiB` inside `SAP.mask_head`'s `neck` -- a single
      `nn.ConvTranspose3d(embed_dim, 256, kernel_size=patch_size,
      stride=patch_size)` (`arch.py:685-693`). Its memory scales as
      `patch_size**3`; the requested 512 GiB is *exactly*
      `(32/4)**3 = 512` times whatever `patch_size=4` needed -- unambiguous
      confirmation of the cubic relationship. Unlike `do_ap`, this isn't a
      config-parsing assertion, so nothing caught it before a real GPU run.
      Fixed: reverted `sap`'s `patch_size` to its original `4` (an
      architecture-driven exception, documented alongside its `do_ap: True`
      exception) -- confirmed via `training.py`'s loss computation
      (`einops.rearrange` using `conf["data"]["patch_size"]` dynamically for
      both the model output and the reshaped `seq_label`) that this doesn't
      introduce a shape mismatch: output and label resolution are always
      `sqrt_len * patch_size` by construction, self-consistent for any
      `patch_size` value, not tied to the literal `tile_size`.
    - `basic_ct-unetr`: `torch.OutOfMemoryError: HIP out of memory. Tried to
      allocate 40.00 GiB` inside MONAI's `dynunet_block.py`, not
      `PatchEmbed` -- `UNETR.encoder1` (`arch.py:1135-1143`) is a plain
      `Conv3d` that runs directly on the **full 256^3-resolution raw
      volume**, with `feature_size=16` output channels, entirely
      independent of `patch_size`/`embed_dim`. At `batch_size=32` that one
      activation alone is `32 * 16 * 256**3 * 4 bytes` ~= 34 GB, comfortably
      explaining the OOM on a 64 GB GPU. This is a structural property of
      full-resolution conv segmentation decoders, not something the
      tiny-model smoke-test override (`embed_dim`/`depth`/`num_heads`) could
      mask, since none of those touch `feature_size` or `batch_size`.
      Fixed: `unetr`'s `batch_size` reverted to `4` (another
      architecture-driven exception; `patch_size` stays `32` since the OOM
      was batch_size-driven, not patch_size-driven). `BASIC_CT_MIN_FILES` in
      `test_dataloader_real_pipeline.py`/`test_dataset_speed_real_data.py`
      (which target `basic_ct/unetr`'s config) lowered from the
      `batch_size=32` formula to `batch_size=4` accordingly.

    So `basic_ct` now has three documented, architecture-driven exceptions
    on top of the unified baseline: `sap` (`do_ap: True`, `patch_size: 4`)
    and `unetr` (`batch_size: 4`) -- not a regression back to
    `PER_CONFIG_OVERRIDES`-style tuning (those were workarounds for a single
    shared root cause; these are three separate, real architectural
    constraints specific to each model's decoder). Full local `pytest -q`
    passes (113 passed, 36 skipped) and all 10 configs still parse with
    these changes. **Not yet verified against a real Frontier run** -- next
    step is rerunning `run_training_smoke.sh` again to confirm all three
    fixes actually work (the `sap`/`unetr` memory numbers in particular are
    reasoned from the failure messages, not yet measured after the fix).
12. Reran after item 11's fixes: `basic_ct-mae`'s and `basic_ct-sap`'s
    failures are gone, but `basic_ct-unetr` now **times out** instead of
    OOMing (log not saved, but reported directly). Root cause: `min_files`
    (the target real-file count `make_smoke_config` narrows to) was a flat
    `DEFAULT_MIN_FILES=256` shared by every config, sized for the shared
    `batch_size=32` most configs use (256 = 32 * data_par_size(8), ~1
    batch/rank). `basic_ct-unetr`'s `batch_size=4` exception (item 11) meant
    the same 256 real files worked out to 32 files/rank / 4 = 8
    batches/rank/epoch instead of 1 -- 8x the iterations through UNETR's
    inherently expensive full-resolution conv decoder (`encoder1` runs a
    plain `Conv3d` on the raw 256^3 volume regardless of `patch_size`/
    `embed_dim`, so every iteration is costly no matter what else changed).
    This was flagged as a risk when `batch_size:4` was applied ("generous
    ... but that's harmless" -- true for correctness, false for wall-clock
    time). Fixed structurally rather than with another single-config
    override: `make_smoke_config` now caps whatever `min_files` it's given
    down to *that config's own* `batch_size * data_par_size`, so every
    config gets ~1 batch/rank regardless of its `batch_size` -- automatically
    right-sizing itself for any future batch_size exception too, not just
    `unetr`'s current one. Doesn't change behavior for any config that
    already had `batch_size=32` (256 already equalled that cap). Full local
    `pytest -q` still passes (113 passed, 36 skipped). **Not yet verified
    against a real Frontier run.**
13. Reran after item 12's fix (job 5322916): **all 10 shipped configs now
    pass Tier 3 fully, fresh and resume** --
    `basic_ct-{diffusion,mae,sap,unetr}` (145s/94s, 81s/68s, 212s/173s,
    220s/43s), all 3 `imagenet` configs, and all 3 `catsdogs` configs. This
    closes out the config-baseline reconfiguration effort: the original
    `basic_ct-mae` timeout that started it is gone (structural fix, not a
    tuned-around workaround), and the three follow-on issues it
    surfaced -- `mae`'s decoder pos-embed divisibility, `sap`'s
    patch_size-cubed decoder memory, `unetr`'s batch_size-linear decoder
    memory and its knock-on min_files/iteration-count timeout -- are all
    fixed and verified for real. `basic_ct` now ships with three documented,
    architecture-driven exceptions to the shared baseline (`sap`:
    `do_ap: True` + `patch_size: 4`; `unetr`: `batch_size: 4`); everything
    else (tensor parallelism, tiling, `twoD`, `num_workers`,
    `dict_buffer_sizes`) is uniform across all 4 `basic_ct` configs, matching
    `imagenet`/`catsdogs`.
14. Tier 2 rerun (job 5323450) confirms the rest: **all 39 tests pass on all
    8 ranks** (74.72s), including `test_real_pipeline_basic_ct_unetr` against
    its updated non-adaptive 4-tuple batch shape. No failures anywhere in the
    log. This closes out the config-baseline reconfiguration effort
    end to end -- both Tier 3 (all 10 configs, item 13) and Tier 2 (all 39
    tests, real data, real 8-rank launch) are green against the new
    `basic_ct` baseline.

That fixture testing also surfaced a narrower, real edge case in
`process_root_dirs`: when an `imagenet`-format dataset has `<= data_par_size`
classes, `classes_to_combine` was only assigned inside an `if len(classes) >
data_par_size:` block, so it was referenced unassigned (`UnboundLocalError`)
right after. Real ImageNet-1k (1000 classes) is far above any realistic
`data_par_size`, so this never affected the configs actually shipped here —
now fixed anyway (`classes_to_combine = 1` in that case, giving one bucket
per class — `len(classes)` buckets rather than `data_par_size`, matching the
function's own "`data_par_size` (or fewer) buckets" docstring), with new
`tests/utils/test_misc.py` coverage for `process_root_dirs`'s bucketing
generally: the evenly- and non-evenly-divisible `> data_par_size` cases
(including documenting, not fixing, the pre-existing "leftover classes past
`data_par_size * classes_to_combine` are silently dropped" behavior — flagged
by its own `# TODO: Add shuffling for data_par_size if it doesn't divide
1000 equally` comment), the `<= data_par_size` regression case parametrized
across several class counts, bucket-content correctness (not just counts),
and the non-`imagenet` (`basic_ct`-style) branch.

## Running the feature-matrix smoke test (Tier 3b)

```bash
cd launch/tests
sbatch run_feature_matrix_smoke.sh
```

Tier 3 verifies all 10 shipped configs still work under this session's
shared baseline (tensor parallelism, adaptive patching, tiling, and — for
`basic_ct` — `twoD` all off by default, with `basic_ct/sap`'s and
`basic_ct/unetr`'s three documented architecture-driven exceptions). Tier 3b
verifies the advanced features *themselves* still work now that they're
opt-in: for one representative (model, dataset) combination per feature —
not a full combinatorial sweep, see
`tests/integration/run_feature_matrix_smoke.py`'s module docstring for why —
it flips that one feature back on and runs the same kind of tiny/fast
real-data training run Tier 3 does.

Same "not run under a top-level `srun`" structure as Tier 3, for the same
reason (see Tier 3's section above) — this script also spawns its own
`srun` subprocess per run. Output lands in
`feature-matrix-smoke-<jobid>.out`.

**Covered today** (see `run_feature_matrix_smoke.py`'s `FEATURE_MATRIX` for
the exact overrides and reasoning behind each cell):

| Feature | Cell(s) |
| --- | --- |
| `ap.do_ap:True` | `basic_ct/unetr`, `basic_ct/mae`, `imagenet/classification`, `catsdogs/classification` |
| `tiling.do_tiling:True` | `basic_ct/unetr`, `imagenet/classification` (`catsdogs` excluded — its `dataloader.type: "dataloader"` never invokes `TileDataIter`; `tile_size` there is purely a resize target, not a tiling grid) |
| `data.twoD:True` | `basic_ct/unetr` only (`imagenet`/`catsdogs` always run `twoD:True` already — `parse.py` forces it whenever `img_size` has 2 entries) |
| `parallelism.tensor_par_size:2` | `imagenet/classification` (+ resume cycle), `basic_ct/mae`, `catsdogs/classification`, `basic_ct/sap`, `catsdogs/diffusion` |
| Multi-feature combinations | `basic_ct/unetr` (`do_ap`+`do_tiling`, `do_ap`+`twoD`, `twoD`+`tensor_par_size`), `basic_ct/mae` (`twoD`+`do_tiling`), `imagenet/classification` (`do_ap`+`tensor_par_size`, `do_tiling`+`tensor_par_size`) |

Every dataset (`basic_ct`/`imagenet`/`catsdogs`) and every model type
(`VIT`/`MAE`/`SAP`/`UNETR`/`DiffusionVIT`) appears in at least one cell.
`SAP`/`DiffusionVIT` only get a `tensor_par_size` cell each (neither has a
free `do_ap` choice — `parse.py` hard-requires `do_ap:True` for `SAP` and
`do_ap:False` for `DiffusionVIT`, and `SAP`'s own shipped baseline already
proves its case), chosen because both have their own model-specific
broadcast code in `training.py`'s `process_batch` (`SAP`: `seq_label`;
`DiffusionVIT`: the diffusion noise terms `t`/`e`) that no other cell
exercises.

Every cell runs fresh-only except `imagenet/classification`'s solo
tensor-parallel cell, which also resumes — closing a real, otherwise
permanent coverage gap: no shipped config ships `tensor_par_size > 1`, so
nothing else in this suite (Tier 1, 2, or baseline Tier 3) ever exercises
the per-tensor-parallel-rank checkpoint file loop in `parse.py`'s
pre-flight check or `model/utils.py`'s resume loading.

The 6 multi-feature cells cover every pairwise combination among the 4 axes
where it's architecturally meaningful (`do_ap`+`do_tiling`, `do_tiling`+
`twoD`, `do_ap`+`tensor_par_size`, `do_ap`+`twoD`, `do_tiling`+
`tensor_par_size`, `twoD`+`tensor_par_size`), each with a single fixed value
per feature (one `tensor_par_size`, one `div`) rather than a sweep, so this
stays as cheap as every single-feature cell. Two of them aren't
hypothetical: `basic_ct/unetr`'s `do_ap`+`do_tiling` cell mirrors that
config's actual pre-baseline settings (`do_ap:True, do_tiling:True, div:4,
twoD:False`) — a combination that already passed real Frontier runs earlier
this session, so it's a real regression check, not new territory.
`basic_ct/mae`'s `twoD`+`do_tiling` cell mirrors the exact combination
behind this session's original `basic_ct-mae` timeout that started the
whole baseline reconfiguration effort — deliberately re-exercised here, but
controlled (`div:2` instead of the original `div:4`, `min_files` pinned to
the floor, `timeout:1800s`) rather than accidental.

For iterating on a single failing cell:

```bash
cd launch/tests
sbatch run_feature_matrix_smoke_single.sh basic_ct-unetr+twoD
```

`tests/integration/test_feature_matrix_smoke_helpers.py` sanity-checks
`FEATURE_MATRIX` itself before any of this ever touches Frontier: unique
labels, every `base_config_relpath` resolves to a real file, no
accidentally-empty `overrides`, no accidentally-list-valued
`tiling.tile_overlap` override (a real gotcha in
`deep_merge_config_overrides` — see its docstring in
`run_training_smoke.py`), and — the most valuable check — every cell's
tiny-model config (real base config + `TINY_MODEL_OVERRIDES` + the cell's
own overrides, exactly as `make_smoke_config` would build it) is fed
through the real `UCF_VIT.parse.parse_config` with
`load_balance_offline=True` (skips the `data_par_size*tensor_par_size==
world_size` assertion, so even the `tensor_par_size:2` cells validate with
zero real GPUs). All 18 cells pass this locally before ever touching real
Frontier data.

**Real runs on Frontier so far:**

1. `basic_ct-unetr+twoD` (job 5337762), run single-cell first via
   `run_feature_matrix_smoke_single.sh` as the deliberately highest-risk
   cell (its `min_files:8`/`timeout:1800` were reasoned from historical
   numbers, not measured): **PASS (280s)** — comfortably under its 1800s
   timeout, no retuning needed. Confirms the z-slice-multiplication cost
   estimate for this cell was accurate.
2. Running the full matrix surfaced a real, previously-dormant bug in
   `src/UCF_VIT/training.py`'s `train_step`, in the `MAE` + `do_ap:True`
   branch — the exact code path `basic_ct-mae+do_ap` exercises for the
   first time ever (no shipped config used `do_ap:True` for MAE before
   Tier 3b). Two bugs stacked in one line:
   `target = rearrange(seq, 'b c s p -> b s (p c)')` — `rearrange` was
   never imported bare (the file only does `import einops`, not
   `from einops import rearrange`, unlike `model/arch.py` which does), and
   `seq` was never defined in this branch at all (the actual sequence data
   is `batch["seq"]`, as used one line above to compute `output`). Fixed to
   `target = einops.rearrange(batch["seq"], 'b c s p -> b s (p c)')`. A
   real regression this session's Tier 1 tests couldn't catch (they don't
   exercise `train_step` against real batch dicts), only findable by
   actually running MAE with `do_ap:True` end to end — exactly what Tier 3b
   is for.
3. The same full-matrix run surfaced five more real, previously-dormant
   bugs, all in code paths no shipped config ever exercised before (every
   one requires either `do_ap:True` on `VIT`, or `tensor_par_size > 1`):
   - `src/UCF_VIT/training.py`'s `train_step`, `VIT` branch: unconditionally
     called `model.forward(batch["data"], ...)`, never checking
     `conf["ap"]["do_ap"]` at all (unlike the `SAP`/`UNETR`/`MAE` branches,
     which all correctly switch to `batch["seq"]` when adaptive patching is
     on). `model/arch.py`'s `VIT.__init__` even documents
     `#ASSUMES INPUT HAS ALREADY BEEN ADAPTIVELY PATCHED` when
     `adaptive_patching` is set — feeding it raw image data instead produced
     a token-count mismatch (`RuntimeError: The size of tensor a (257) must
     match the size of tensor b (196)`) in `_pos_embed`, hit by both
     `imagenet-classification+do_ap` and `catsdogs-classification+do_ap`.
     Fixed to mirror the other branches: `batch["seq"]` when `do_ap:True`,
     `batch["data"]` otherwise.
   - `training_scripts/train.py`: `NativePytorchDataModule` was never passed
     `ddp_group=ddp_group`, so it silently fell back to the *global* world
     rank instead of this rank's position within its data-parallel
     replica group — `NativePytorchDataModule` already supported a
     `ddp_group` parameter for exactly this, it just was never wired up.
     Broke as `IndexError: index 0 is out of bounds for axis 0 with size 0`
     inside `datamodule.py`'s `train_dataloader()` for any rank whose world
     rank exceeded `data_par_size` (e.g. ranks 4/6 with
     `tensor_par_size:2`/`data_par_size:4`) — `basic_ct-mae+tensor_par` and
     `basic_ct-unetr+twoD+tensor_par`.
   - `training_scripts/train.py`: the `"dataloader"`-type (`catsdogs`)
     branch's `DistributedSampler(..., rank=world_rank)` has the same bug --
     `world_rank` only equals the correct data-parallel-relative rank when
     `tensor_par_size:1`. Broke as `ValueError: Invalid rank 6, rank should
     be in the interval [0, 3]` for `catsdogs-classification+tensor_par`.
     Fixed both to `dist.get_rank(ddp_group)`, which reduces to `world_rank`
     exactly when `tensor_par_size:1` (so no behavior change for any
     existing config) but gives the correct per-replica rank otherwise.
   - `training_scripts/train.py`: `train_dataloader`/`data_module` are only
     ever constructed on `tensor_par_group`-rank-0 (by design --
     `UCF_VIT.training.process_batch`'s docstring: only that rank reads
     real data, then broadcasts it to the rest of its tensor-parallel
     group), but `train_epoch(...)` references `train_dataloader`
     unconditionally for *every* rank, and the per-epoch dataloader reset
     did too. With `tensor_par_size:1` every rank trivially satisfies
     `tensor_par_group`-rank-0, so this was never exercised. Broke as
     `UnboundLocalError: cannot access local variable 'train_dataloader'`
     on every non-tensor_par_group-rank-0 process, across every
     `+tensor_par` cell. Fixed by explicitly binding
     `train_dataloader = None` (and `data_module = None`) on the `else`
     branch in both dataloader-type cases, and gating the per-epoch reset
     the same way.
   - `training_scripts/train.py`: for the `"dataloader"` type specifically,
     `iterations_per_epoch = len(train_dataloader)` has the same
     unconditional-reference problem, but unlike `train_dataloader` itself
     (never touched by non-rank-0 processes inside `process_batch`), every
     rank *does* need a valid `iterations_per_epoch` — it's the training
     loop's iteration count, and `process_batch`'s per-tensor-parallel-group
     broadcasts require every rank in a group to call it the same number of
     times. Fixed by broadcasting it from each tensor-parallel group's
     rank-0 to the rest of that group, reusing the exact
     `dist.broadcast(..., src=(dist.get_rank()//tensor_par_size*
     tensor_par_size), group=tensor_par_group)` idiom `process_batch`
     already uses for batch tensors.

   All five fixed locally (syntax + full Tier 1 suite, `pytest -q`); none
   are exercisable without a real multi-rank `tensor_par_size > 1` launch,
   so — like the `VIT`+`do_ap` fix above — not yet re-verified against a
   real Frontier run.
4. Auditing the matrix afterward found a real coverage gap: every cell used
   `UNETR`, `MAE`, or `VIT` — `SAP` and `DiffusionVIT` never appeared
   anywhere. Added one `tensor_par_size:2` cell each
   (`basic_ct-sap+tensor_par`, `catsdogs-diffusion+tensor_par`) — the axis
   most likely to matter for them specifically, since both have their own
   model-specific broadcast code in `process_batch` (`SAP`'s `seq_label`,
   `DiffusionVIT`'s `t`/`e` noise terms) that no other cell exercises;
   `do_ap` doesn't apply to either (both hard-required to a fixed value by
   `parse.py`). Now every dataset and every model type appears in at least
   one of the matrix's 18 cells. Both new cells pass the local
   `parse_config` dry-run; not yet run against real Frontier data.
5. Reran the full matrix (job 5338382) against all the item-3 fixes: **8
   PASS, 8 FAIL** — real progress (`basic_ct-mae+do_ap`,
   `basic_ct-mae+tensor_par`, and `catsdogs-diffusion+tensor_par` now pass),
   but surfaced a whole second wave of bugs, every one only reachable once
   the item-3 fixes cleared the way to exercise it for the first time:
   - `model/arch.py`'s `_pos_embed`: the item-3 `VIT`+`do_ap` fix was
     necessary but not sufficient — `self.adaptive_pos_dep_emb(seq_ps)`
     computes one position embedding per real patch (no cls-token row),
     but `x` gets the cls token prepended before the add, so `VIT`
     (the only model type with `class_token=True` — `UNETR`/`MAE`/`SAP`/
     `DiffusionVIT` all construct with `class_token=False`, per
     `model/utils.py`'s `get_model`) is one token short:
     `RuntimeError: size of tensor a (197) must match size of tensor b
     (196)`. Fixed by prepending a zero row to the adaptive `pos_embed` for
     the cls token, mirroring `get_2d_sincos_pos_embed`/
     `get_3d_sincos_pos_embed`'s own documented convention for the
     non-adaptive case ("prepend a zero embedding row for a class token").
   - `training.py`'s `process_batch`: the `VIT` classification `label`
     placeholder (4 occurrences, `do_ap` x `twoD`) was
     `torch.zeros(batch_size, 1, dtype=precision_dt)` -- shape
     `(batch_size, 1)` float, but the real label (from `get_batch`) is
     `(batch_size,)` int64 (a flat class index). `dist.broadcast` fills
     values into the existing placeholder without reshaping/recasting it,
     so non-rank-0 processes ended up with a 2D float target:
     `RuntimeError: 0D or 1D target tensor expected, multi-target not
     supported` from `nn.CrossEntropyLoss`. Fixed all 4 to
     `torch.zeros(batch_size, dtype=torch.int64)`.
   - `training.py`'s `process_batch`: `tile_size` was only assigned in the
     `do_ap:False` branch of the setup code, but the `do_ap:True` branch's
     placeholder construction still needs it to shape `data` (the raw,
     pre-patchification image, needed regardless of `do_ap`) --
     `UnboundLocalError: cannot access local variable 'tile_size'`. Fixed
     by moving the assignment out of the `if`/`else` so it's unconditional.
   - `training.py`'s `process_batch`: the `dict_key` broadcast passed a
     bare Python `str` (e.g. `"ct1"`) as `dist.broadcast_object_list`'s
     `object_list` argument on the source rank, while every receiver
     correctly pre-allocated a real `list`. That API requires an actual
     list on every rank -- the type mismatch corrupted the pickle framing:
     `_pickle.UnpicklingError: invalid load key` and (on a different rank)
     `torch.OutOfMemoryError: HIP out of memory. Tried to allocate more
     than 1EB memory` (a garbage size read from the corrupted buffer).
     Fixed by using `list(dict_key)` on the source rank too, matching the
     receivers' `[None] * dict_key_len` placeholder exactly.
   - `training.py`'s `process_batch`: `seq_size`/`seq_pos` were read
     straight off `batch["seq_size"]`/`batch["seq_pos"]` with no
     `.to(precision_dt).to(device)` on the source rank, unlike every other
     broadcast field (`data`, `seq`, `label`) -- since NCCL has no CPU
     backend, broadcasting a CPU tensor into GPU-resident receivers failed
     outright: `RuntimeError: No backend type associated with device type
     cpu`. Fixed to match `data`/`seq`'s existing pattern.
   - `training.py`'s `process_batch`: found by inspection, not yet hit by
     any real run -- the final "convert `seq_size`/`seq_pos` into `seq_ps`"
     block (used for adaptive position embeddings) unconditionally read
     `batch["seq_size"]`/`batch["seq_pos"]`, but `batch` is only ever
     assigned on `tensor_par_group`-rank-0 when `tensor_par_size > 1` --
     every other rank would hit `UnboundLocalError` here, even though the
     correct, already-broadcast values were sitting right there in the
     local `seq_size`/`seq_pos` variables. Fixed to read those locals
     instead (and assigned them as locals in the `tensor_par_size == 1`
     branch too, for consistency, since previously only the dict entries
     existed there).
   - `training.py`'s `process_batch`: also found by inspection --
     `DiffusionVIT`'s `t` placeholder used `dtype=torch.int` (int32), but
     `torch.randint`'s real default dtype is `int64` -- a byte-size
     mismatch that should break an NCCL broadcast, though
     `catsdogs-diffusion+tensor_par` passed anyway in job 5338382 (possibly
     silent corruption rather than a hard failure, or NCCL tolerating it
     for this particular tensor shape -- not fully understood). Fixed
     defensively to `int64` regardless, since the mismatch is real either
     way.

   All fixed locally (syntax + full Tier 1 suite, `pytest -q`, 140 passed);
   none exercisable without a real multi-rank `tensor_par_size > 1` launch
   (or, for the `_pos_embed` fix, `VIT`+`do_ap:True`), so not yet
   re-verified against a real Frontier run. Given how many of these were
   only reachable after clearing an earlier one, expect this to need at
   least one more real-run/fix cycle.
6. Reran the full matrix (job 5339018) against item 5's fixes: **15 PASS, 3
   FAIL** — the biggest jump yet, including `imagenet-classification+
   tensor_par`'s resume cycle passing for the first time. The 3 remaining
   failures reduced to two distinct, well-understood bugs:
   - `training.py`'s `process_batch`: the `separate_channels=False`
     placeholder for `seq_pos` was missing its trailing coordinate
     dimension entirely (`torch.zeros(batch_size, 1, fixed_length, 1, 1,
     ...)` instead of `torch.zeros(batch_size, 1, fixed_length, 2, ...)`
     for `twoD`, `3` for 3D) — confirmed against the real per-sample
     construction in `datamodule.py` (`np.expand_dims(batch[i][3],
     axis=0)`, where the raw per-patch position is a 2- or 3-element
     coordinate, not a scalar). The `seq_size` placeholder had the
     opposite problem, one spurious extra trailing dim
     (`(batch_size, 1, fixed_length, 1)` instead of `(batch_size, 1,
     fixed_length)`, `seq_size` being a scalar per patch with no
     coordinate dim at all). Broke as `RuntimeError: Tensors must have
     same number of dimensions: got 3 and 2` in the final `seq_ps =
     torch.concat([seq_size, seq_pos], dim=-1)` — hit by
     `imagenet-classification+do_ap+tensor_par` and
     `basic_ct-sap+tensor_par`. Fixed both placeholder blocks (`twoD` and
     3D) to the correct shapes.
   - `training.py`'s `process_batch`: all four `dist.broadcast_object_list`
     calls (`dict_key_list` x2, `variables` x2) never passed a `device`
     argument. `broadcast_object_list`'s own docs warn that for NCCL
     groups its internal object-size/pickled-bytes tensors must live on
     this rank's GPU, and without an explicit `device` it falls back to
     the global `torch.cuda.current_device()` — relying on that
     implicitly, rather than the `device` `process_batch` already has in
     hand as a parameter, is exactly the kind of thing that can produce
     the `torch.OutOfMemoryError: Tried to allocate more than 1EB memory`
     corruption `basic_ct-unetr+twoD+tensor_par` hit here (the same
     symptom as an earlier, different bug this session — that one was a
     genuine `str`-vs-`list` type mismatch, already fixed; this is a
     second, distinct cause producing the identical crash signature).
     Fixed by passing `device=device` explicitly to all four calls.

   Both fixed locally (full Tier 1 suite green, 153 passed); not yet
   re-verified against a real Frontier run.

7. Reran the full matrix (job 5339608) against item 6's fixes: **16 PASS, 2
   FAIL** — both remaining failures reduced to real, previously-dormant
   bugs, both now fixed (not yet re-verified against a real Frontier run):
   - `basic_ct-unetr+twoD+tensor_par`: `KeyError: ''` at
     `num_channels[dict_key]` — the 1EB-OOM crash from item 6 was gone
     (confirming that fix helped), but `dict_key` still arrived empty on
     the receiving rank. Root cause: the previous `dict_key` broadcast
     scheme (broadcast its length as a separate tensor, then broadcast a
     `[None]*length` list of individual characters via `list(dict_key)`,
     then `''.join(...)` to reconstruct) was fragile — even with the
     `str`-vs-`list` and missing-`device` bugs already fixed, it still
     occasionally produced an empty string for reasons never fully
     isolated. Replaced entirely with a much simpler design in both the
     `do_ap:True` and `do_ap:False` branches: broadcast `dict_key` itself
     as a single-element list (`dict_key_holder = [dict_key] if
     dist.get_rank(tensor_par_group) == 0 else [None]`) via one
     `broadcast_object_list` call — `broadcast_object_list` already
     natively supports a variable-length pickled object directly, so the
     separate length-broadcast-then-characters dance was never needed.
   - `basic_ct-sap+tensor_par`: `TIMEOUT (600s)`, no crash. Traced to a
     real sender/receiver `dtype` mismatch on the `label` broadcast in the
     Segmentation (`UNETR`/`SAP`) branch of `process_batch`: the real
     `batch["label"]` is `uint8` (do_ap:True, basic_ct — see `dataset.py`'s
     `np.asarray(np_label, dtype=np.uint8)`) or `int64` (do_ap:False — see
     `dataset.py`'s `np.array(label.dataobj).astype(np.int64)`), never
     cast to `precision_dt` on the sender side, but the non-rank-0
     placeholder declared `dtype=precision_dt` (a float type) in all four
     spots (`do_ap:True`/`twoD`, `do_ap:True`/3D, `do_ap:False`/`twoD`,
     `do_ap:False`/3D). NCCL requires identical dtype/byte-size across
     ranks for a collective to complete (the same reasoning already
     documented next to the `DiffusionVIT` `t`/`e` placeholders); a
     mismatch here manifests as a hang until the watchdog times out,
     rather than an immediate crash — consistent with the observed
     `TIMEOUT` instead of a `RuntimeError`. This exact code path
     (do_ap:True + Segmentation + `tensor_par_size > 1`) had never been
     exercised before `basic_ct-sap+tensor_par` (`SAP` is the only model
     requiring `do_ap:True`), and `basic_ct-unetr+twoD+tensor_par`
     (do_ap:False + Segmentation) never reached this broadcast either,
     since it was crashing earlier on the `dict_key` bug above. Fixed all
     four placeholders to the correct integer dtype (`torch.uint8` for
     do_ap:True, `torch.int64` for do_ap:False). While auditing this,
     also fixed a related latent bug one step earlier: the rank-0 sender's
     `seq_label = batch["seq_label"].to(device)` was missing a
     `.to(precision_dt)` cast (its receiver placeholder is
     `dtype=precision_dt`) — harmless today only because `basic_ct/sap`'s
     baseline happens to use `data_type: float32` already (matching
     `seq_label`'s native `float32` from `datamodule.py`'s
     `seq_mask.permute(2, 0, 1).float()`), but would hit the identical
     hang under a `bfloat16` config. `run_feature_matrix_smoke.py`'s
     `basic_ct-sap+tensor_par` cell also picked up `min_files_override=8`/
     `timeout_override=1800` (mirroring `basic_ct-unetr+twoD`'s existing
     margin) as a safety buffer while this fix gets its first real run —
     not because SAP's baseline has any known sample-count multiplication
     cost (`twoD` stays `False`).

   Both fixed locally (full Tier 1 suite green, 153 passed); not yet
   re-verified against a real Frontier run.

8. Reran the full matrix (job 5340104) against item 7's fixes: **17 PASS, 1
   FAIL**. Both item 7 fixes verified working on real Frontier data:
   `basic_ct-unetr+twoD+tensor_par` now passes (`dict_key` broadcast
   redesign fixed), and `basic_ct-sap+tensor_par` no longer times out —
   it now fails fast at 22s instead of hanging for 600s, confirming the
   `label`-dtype broadcast fix resolved the hang. But it still failed, with
   a new, different error: `ZeroDivisionError: division by zero` at
   `datamodule.py`'s `setup()` (`keys_to_add =
   int(np.ceil(self.max_balance/self.batches_per_rank_epoch[k]))`), coming
   from `calculate_load_balancing_on_the_fly` (`misc.py`) computing
   `batches_per_rank_epoch["ct1"] = 0`. Root cause was a mistake in item
   7's own `run_feature_matrix_smoke.py` change, not `training.py`:
   `basic_ct-sap+tensor_par` had copied `min_files_override=8` from
   `basic_ct-unetr+twoD`, but that value only works there because
   `twoD:True` multiplies each real file into up to `div*div*img_size[2]`
   tiles — SAP has `twoD:False` and no tiling, so `tiles_per_image` is `1`.
   With only 8 real files split across `data_par_size:4` ranks, each rank
   gets just 2 images (`tiles_per_image:1`, so 2 samples) — far below
   the baseline's `batch_size:32` — so `calculate_load_balancing_on_the_fly`
   floors `batches_per_rank` to `0`. This cell previously ran fine at
   `DEFAULT_MIN_FILES:256` (64 images/rank, well above `batch_size:32`);
   the fix is simply to drop the `min_files_override` for this cell
   (kept `timeout_override=1800` as a harmless safety margin — the actual
   failure now surfaces in 22s, well under even the shared 600s default).

   Fixed locally (full Tier 1 suite green, 153 passed); not yet
   re-verified against a real Frontier run.

9. Reran the full matrix (job 5341029) against item 8's fix: **17 PASS, 1
   FAIL**. The `min_files` fix worked — `basic_ct-sap+tensor_par` got well
   past `setup()` this time (180s in, well into real training) — but still
   failed, with yet another new, different error:
   `einops.EinopsError: Shape mismatch, 512 != 64` in `training.py`'s
   `train_step` (`seq_label = einops.rearrange(batch["seq_label"], 'b c
   (ps1 ps2 ps3) (s1 s2 s3)-> ...')`), on a non-rank-0 process, with
   `batch["seq_label"].shape == (32, 4, 512, 64)` where the pattern expected
   `(..., 64, 512)` (`ps1*ps2*ps3=4*4*4=64`, `s1*s2*s3=8*8*8=512`) — the
   last two dims were transposed. Traced to yet another placeholder-shape
   bug in `process_batch`, in the exact same never-before-exercised
   do_ap:True + Segmentation + `tensor_par_size>1` code path as items 7/8:
   the `seq_label` placeholder (both `twoD` and 3D variants) declared shape
   `(batch_size, num_classes, fixed_length, patch_size**N)`, but the real
   per-sample construction (`dataset.py`'s `np.reshape(seq_label,
   [patch_size**N, -1, 1])` composed with `datamodule.py`'s
   `seq_mask.permute(2, 0, 1)` stacking) actually produces
   `(batch_size, num_classes, patch_size**N, fixed_length)` — the two
   trailing dims swapped. Since both orderings have the same total element
   count, `dist.broadcast` (which fills a tensor's existing memory without
   reshaping) copied the bytes without erroring, and the corruption only
   surfaced later, downstream in `train_step`'s `einops.rearrange` (which
   is why this bug survived items 7/8's local Tier 1 checks and the earlier
   `label`/`seq_label` dtype fixes — dtype matched, only the shape was
   wrong). Fixed both placeholders (`twoD`: `patch_size*patch_size,
   fixed_length`; 3D: `patch_size*patch_size*patch_size, fixed_length`) to
   match the real dim order.

   Fixed locally (full Tier 1 suite green, 153 passed); not yet
   re-verified against a real Frontier run.

10. Reran the full matrix (job 5341245), now against `get_model`'s
    `tensor_par_size`/`tensor_par_group` wiring fix (see "Major finding"
    above) actually shipping: **16 PASS, 2 FAIL** — `imagenet-
    classification+tensor_par`, `basic_ct-mae+tensor_par`, `catsdogs-
    classification+tensor_par`, `imagenet-classification+do_ap+tensor_par`,
    `imagenet-classification+do_tiling+tensor_par`, and `catsdogs-
    diffusion+tensor_par` all still passed with real sharding actually
    active for the first time — a good sign for the sharding logic itself
    (consistent with `test_tensor_parallel_correctness.py`/
    `test_fsdp_correctness.py` already passing on real data). But two
    cells that were previously passing under the old (never-sharded)
    behavior broke: `basic_ct-unetr+twoD+tensor_par` (regression) and
    `basic_ct-sap+tensor_par` (still unresolved), both with the identical
    error: `ValueError: Tensors must be contiguous` in `arch.py`'s
    `forward_features`, at `dist.broadcast(x, src_rank,
    group=self.tensor_par_group)` right after `_pos_embed`/`patch_drop`.

    Root cause: `_pos_embed`'s `torch.cat` step (prepending the class
    token), which happens to also produce a fresh contiguous tensor as a
    side effect, only runs when `self.cls_token is not None` — i.e. only
    for `model_type == "VIT"` (`get_model`'s `class_token=True if
    conf["model"]["type"] == "VIT" else False`). For every other model
    type (`UNETR`, `SAP`, `MAE`, `DiffusionVIT`), that step is skipped, so
    `x`'s contiguity depends entirely on `self.token_embeds(x)`'s raw
    output — typically `PatchEmbed`'s own flatten+transpose, which is
    non-contiguous. This code path was dead until item "Major finding"'s
    `get_model` fix, so it had never been exercised by any real run
    before. One instance of this exact bug (`DiffusionVIT.forward_features`)
    had apparently already been found and fixed at some earlier point
    (`dist.broadcast(x.contiguous(), ...)`), but the identical pattern was
    never applied to the other four analogous broadcasts.

    Audited every `dist.broadcast(x, ...)`/`dist.broadcast(x.contiguous(),
    ...)` call in `arch.py` (6 total) rather than patching only the two
    confirmed failures, checking each one's actual upstream contiguity:
    - `VIT.forward_features` (shared by `VIT`/`SAP`/`UNETR`) and
      `UNETR.forward_intermediates`: same `_pos_embed` → `patch_drop` →
      broadcast shape with no cleanup step in between — fixed.
    - `DiffusionVIT.forward_features`: `x = x + time_emb` right before the
      broadcast does *not* reliably produce a contiguous result (an
      elementwise op can preserve a non-contiguous input's memory layout)
      — already had `.contiguous()`, but as an unassigned inline copy (see
      below) — fixed properly.
    - `MAE.random_masking`'s noise broadcast, `MAE.mask_head`'s decoder
      broadcast, and `DiffusionVIT.forward_head`'s decoder broadcast: `x`
      immediately before each is the output of `torch.rand`/`torch.cat`/
      `torch.gather` (always produces a fresh, contiguous tensor) or a
      full-range slice of an already-contiguous tensor — confirmed safe
      both by this reasoning and empirically (both `basic_ct-mae+
      tensor_par` and `catsdogs-diffusion+tensor_par` passed in this same
      job with `linear_decoder: False`, i.e. their decoder-broadcast
      branches were genuinely exercised). Left unchanged.

    Also fixed a latent correctness bug while doing this, present in both
    the already-shipped `DiffusionVIT` instance and my own first pass at
    the `VIT`/`UNETR` fixes: `dist.broadcast` fills its tensor argument in
    place, but `.contiguous()` returns a *new* tensor whenever the input
    isn't already contiguous — passing `x.contiguous()` inline without
    reassigning `x = x.contiguous()` first would broadcast-fill that
    throwaway copy while leaving the original `x` variable (used by the
    very next line) silently un-updated on every non-src rank. This never
    caused a visible failure only because every rank already computes an
    (almost) bit-identical `x` independently from identical
    data/weights, so the broadcast's result and each rank's own local
    value coincide in practice — but it defeats the broadcast's actual
    purpose (forcing bit-identical values before the sharded blocks) and
    is a real latent bug if that assumption ever doesn't hold exactly
    (e.g. kernel-level floating-point non-determinism). All fixed
    instances now do `x = x.contiguous()` then `dist.broadcast(x, ...)`.

    Fixed locally (full Tier 1 suite green, 156 passed); not yet
    re-verified against a real Frontier run.

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
