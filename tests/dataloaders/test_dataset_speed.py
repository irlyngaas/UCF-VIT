"""Informational throughput measurements for the dataloader pipeline.

These are not correctness tests (see test_dataset.py for those) and assert
no wall-clock thresholds -- absolute timing varies too much machine to
machine (laptop vs. CI runner vs. a Frontier login node) for a hardcoded
bound to be anything but flaky. Each test measures one factor (buffer_size
or num_workers) across a few settings and prints a small report; read the
printed numbers by eye (run with `pytest -s`) to judge whether a change had
the effect you expected.

Not run by default -- excluded from the default `pytest` invocation via
`addopts` in pyproject.toml (`-m "not dataloader_speed"`). Run explicitly:

    pytest -m dataloader_speed -s
"""

import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from UCF_VIT.dataloaders.dataset import ShuffleIterableDataset

pytestmark = pytest.mark.dataloader_speed


class _InMemorySource(IterableDataset):
    """Yields pre-built in-memory items with no artificial delay -- used to
    isolate ShuffleIterableDataset's own reservoir-buffer overhead from any
    I/O cost.
    """

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        yield from self.items


class _SlowSource(IterableDataset):
    """Synthetic source with an artificial per-item delay standing in for
    real disk I/O (e.g. FileReader's NIfTI/JPEG decode cost), sharded across
    DataLoader workers the same way FileReader shards across them -- so
    num_workers has something real to parallelize.
    """

    def __init__(self, num_items, delay_seconds, item_shape=(1, 8, 8)):
        self.num_items = num_items
        self.delay_seconds = delay_seconds
        self.item_shape = item_shape

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self.num_items
        else:
            per_worker = self.num_items // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else self.num_items
        for i in range(start, end):
            time.sleep(self.delay_seconds)
            yield np.full(self.item_shape, float(i), dtype=np.float32), ("v0",)


def test_shuffle_iterable_dataset_throughput_vs_buffer_size():
    """Reports ShuffleIterableDataset's own overhead as buffer_size grows.

    No I/O here -- items are already in memory -- so this isolates the cost
    of the reservoir-shuffle bookkeeping itself (a randint + swap per item)
    from any upstream dataloading cost.
    """
    num_items = 20_000
    items = [np.float32(i) for i in range(num_items)]

    print(f"\nShuffleIterableDataset throughput ({num_items} in-memory items):")
    for buffer_size in (1, 10, 100, 1_000, 10_000):
        start = time.perf_counter()
        drained = list(ShuffleIterableDataset(_InMemorySource(items), buffer_size=buffer_size))
        elapsed = time.perf_counter() - start
        assert len(drained) == num_items
        rate = num_items / elapsed if elapsed > 0 else float("inf")
        print(f"  buffer_size={buffer_size:>6}: {elapsed:.4f}s ({rate:,.0f} items/s)")


@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_dataloader_throughput_vs_num_workers(num_workers):
    """Reports one epoch's wall time through a DataLoader with a simulated
    per-item I/O delay, for a few num_workers settings.

    A meaningful speedup from num_workers > 0 depends on the delay being
    large enough to dominate multiprocessing overhead -- 5ms/item here
    (~1s/epoch at num_workers=0 for 200 items) is a deliberately generous
    stand-in for real decode cost so the effect is visible without a slow
    test. If you're investigating a specific config's real num_workers
    setting, prefer running that config directly (Tier 3) over extrapolating
    from this synthetic delay.
    """
    num_items = 200
    delay_seconds = 0.005
    dataset = _SlowSource(num_items, delay_seconds)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    start = time.perf_counter()
    count = sum(1 for _ in loader)
    elapsed = time.perf_counter() - start

    assert count == num_items
    print(
        f"\nnum_workers={num_workers}: {elapsed:.3f}s for {num_items} items "
        f"(delay={delay_seconds * 1000:.0f}ms/item, single-process lower bound "
        f"{num_items * delay_seconds:.3f}s)"
    )
