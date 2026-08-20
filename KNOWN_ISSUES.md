# Known Issues

Notes from a code review pass on 2026-08-19. Not exhaustive, and line numbers
may drift as the code changes.

All crash-level bugs found in that pass have been fixed. What remains is
lower-severity.

## Lower-severity / worth knowing about

1. **Doc/code drift**: `NativePytorchDataModule`'s class docstring
   (pre-existing) documents only about half its actual constructor
   parameters — missing `patch_size`, `adaptive_patching`, `fixed_length`,
   `separate_channels`, `dataset`, `return_qdt`, `ddp_group`, `num_classes`,
   `resize`, `batches_per_rank_epoch`.
2. `yaml.load(open(path), Loader=yaml.FullLoader)` (`parse.py` x2,
   `visualize_adaptive.py`) never closes the file handle. `FullLoader`
   avoids arbitrary code execution, so this isn't a security issue for
   trusted config files, just a minor resource-leak/style nit — could be
   `with open(...) as f: yaml.load(f, ...)`.
3. ~50 `TODO` comments scattered across `parse.py` (18), `model/utils.py`
   (7), `datamodule.py` (5), `dataset.py`/`training.py` (4 each) — mostly
   self-flagged missing input validation, not active bugs, but a signal of
   how much of the config-parsing path is unfinished/untested.
