"""Per-variable z-score normalization/denormalization, shared by every dataset's own loader.

Every dataset's raw-loading path (`UCF_VIT.dataloaders.dataset.FileReader.
read_process_file` for imagenet/basic_ct/sst, `UCF_VIT.datasets.catsdogs.
CatsDogsDataset.__getitem__`) calls `zscore_normalize` right after loading,
before any tiling/adaptive-patching runs -- so the actual model-visible data
(including adaptive patching's own resized leaf patches, which are built
from whatever the source array already contains) is normalized, not just a
separately-tracked raw copy. `zscore_denormalize` is the exact inverse, used
only at inference-output time (`UCF_VIT.utils.inference_output.
save_inference_batch`) -- training loss is computed entirely in normalized
space, by design (see this module's own tests/README.md entry for why).

Stats are per (dataset key, variable name), precomputed once by `utils/
compute_normalization_stats.py` over the training split only -- never
per-image, and never including val/test data (both would leak information
the model isn't supposed to have and would make val/test inconsistent with
what the model was actually trained against). See `config.data.
normalize_stats_path`/`UCF_VIT.parse`'s own handling of that config key for
how a stats file reaches these functions.
"""

import numpy as np

MIN_STD = 1e-6


def zscore_normalize(data, variable_names, stats, channel_axis=0):
    """Applies per-channel z-score normalization: `(x - mean) / std`.

    Args:
        data: A single array (channel along `channel_axis`) or a list of
            arrays (one per channel, no shared `channel_axis` -- "sst"'s
            own `read_process_file` convention, each variable is its own
            separately-shaped array).
        variable_names: Channel names, in the same order as `data`'s
            channel axis (or list order) -- an "sst" `(name, offset)` pair
            is resolved to its plain `name` for the stats lookup (stats are
            physical-variable-level, not per-timestep).
        stats: `{variable_name: {"mean": float, "std": float}}`. A channel
            whose name isn't a key here is passed through unchanged --
            lets normalization be configured incrementally (or not at all,
            the default), rather than being all-or-nothing.
        channel_axis: Which axis of each array in `data` is the channel
            axis, when `data` is a single array (ignored for a list, where
            each list element is already one whole channel). `0` (default)
            matches `FileReader`'s own channel-first convention;
            `CatsDogsDataset` passes `-1` (still channel-last at the point
            it normalizes, before its own `moveaxis`).

    Returns:
        Same structure as `data` (array or list of arrays), same dtype
        family (cast to `float32`).
    """
    return _apply_per_channel(data, variable_names, stats, channel_axis, _normalize_one)


def zscore_denormalize(data, variable_names, stats, channel_axis=0):
    """Inverse of `zscore_normalize`: `x * std + mean`. Same args/contract."""
    return _apply_per_channel(data, variable_names, stats, channel_axis, _denormalize_one)


def _normalize_one(channel, s):
    return ((channel - s["mean"]) / max(s["std"], MIN_STD)).astype(np.float32)


def _denormalize_one(channel, s):
    return (channel * max(s["std"], MIN_STD) + s["mean"]).astype(np.float32)


def _var_name(entry):
    """Resolves one `variable_names` entry to a plain string -- "sst"'s own `(name, offset)` pairs included."""
    return entry[0] if isinstance(entry, tuple) else entry


def _apply_per_channel(data, variable_names, stats, channel_axis, fn):
    stats = stats or {}
    names = [_var_name(v) for v in variable_names]

    if isinstance(data, list):
        assert len(data) == len(names), f"got {len(data)} channels but {len(names)} variable_names"
        return [fn(c, stats[n]) if n in stats else np.asarray(c, dtype=np.float32) for c, n in zip(data, names)]

    data = np.asarray(data, dtype=np.float32)
    assert data.shape[channel_axis] == len(names), (
        f"data has {data.shape[channel_axis]} channels along axis {channel_axis} "
        f"but {len(names)} variable_names were given"
    )
    if not any(n in stats for n in names):
        return data
    out = np.moveaxis(data, channel_axis, 0).copy()
    for i, n in enumerate(names):
        if n in stats:
            out[i] = fn(out[i], stats[n])
    return np.moveaxis(out, 0, channel_axis)
