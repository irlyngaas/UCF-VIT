# Known Issues

Notes from a code review pass on 2026-08-19. Not exhaustive, and line numbers
may drift as the code changes.

All crash-level bugs found in that pass have been fixed (see git log), as
have both actionable lower-severity items (doc drift on
`NativePytorchDataModule`, unclosed `yaml.load` file handles — fixed in
commit `3765a91`). What remains isn't a bug to fix in place:

- ~50 `TODO` comments scattered across `parse.py` (18), `model/utils.py`
  (7), `datamodule.py` (5), `dataset.py`/`training.py` (4 each) — mostly
  self-flagged missing input validation, not active bugs, but a signal of
  how much of the config-parsing path is unfinished/untested. Worth a
  closer look if/when validation gaps start biting, but not a single
  actionable fix.
