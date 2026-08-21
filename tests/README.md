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
| `tests/dataloaders/test_dataset.py` | `TileDataIter` (2D, 3D-full, and 3D-twoD-sliced tiling, with/without labels, overlap), `ShuffleIterableDataset` (no data loss/duplication across buffer sizes), `ProcessChannels` (batching, adaptive-patching wiring, `separate_channels`), `FileReader` (DDP-rank + dataloader-worker sharding disjointness/coverage up to `num_workers=7`, `keys_to_add` replication) |
| `tests/dataloaders/test_datamodule.py` | `collate_fn` across `adaptive_patching`/`return_label`/`separate_channels`/`return_qdt`/dataset-type combinations, built from real `ProcessChannels` output rather than hand-fabricated tuples |
| `tests/datasets/test_catsdogs.py` | `CatsDogsDataset` (label-from-filename, resize/channel-first conversion, adaptive-patching shapes) and `CatsDogsCollate`, against small real JPEG files written to a temp dir — `catsdogs` is the only shipped dataset using `dataloader.type: "dataloader"` (a plain `Dataset` + `DistributedSampler`, not the `iterative_dataloader` stack the two rows above cover) |
| `tests/utils/test_misc.py` | `is_power_of_two`, `calculate_tile_overlap`, `patchify`/`unpatchify` roundtrips, `process_root_dirs` (`imagenet` per-class bucketing — evenly/non-evenly-divisible `> data_par_size`, `<= data_par_size`, bucket-content correctness — and the non-`imagenet` branch) |
| `tests/utils/test_pos_embed.py` | 1D/2D/3D sin-cos position embeddings, `SinusoidalEmbeddings` |
| `tests/utils/test_lr_scheduler.py` | `LinearWarmupCosineAnnealingLR` warmup/annealing shape |
| `tests/utils/test_metrics.py` | `masked_mse`, `DiceBLoss` |
| `tests/test_config_validation.py` | Every YAML under `configs/` actually parses via `parse_config` |
| `tests/integration/test_run_training_smoke_helpers.py` | `run_training_smoke.py`'s `compute_narrow_dict_idx` (real-data-found narrowing, empty-but-existing-dir and nonexistent-dir both raising `NoRealDataFoundError`, no-op for non-`iterative_dataloader` configs) |

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
