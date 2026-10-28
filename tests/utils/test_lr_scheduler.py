import pytest
import torch

from UCF_VIT.utils.lr_scheduler import LinearWarmupCosineAnnealingLR


def _run_schedule(warmup_epochs, max_epochs, base_lr=1.0, warmup_start_lr=0.0, eta_min=0.0, total_steps=None):
    """Steps the scheduler `total_steps` times, returning the LR observed at each step (including step 0)."""
    total_steps = total_steps if total_steps is not None else max_epochs
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=base_lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr, eta_min=eta_min,
    )

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_warmup_start_and_end_values():
    lrs = _run_schedule(warmup_epochs=5, max_epochs=20, base_lr=1.0, warmup_start_lr=0.0)
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[5] == pytest.approx(1.0)  # last_epoch == warmup_epochs -> exactly base_lr


def test_warmup_is_monotonically_increasing():
    lrs = _run_schedule(warmup_epochs=5, max_epochs=20, base_lr=1.0, warmup_start_lr=0.0)
    warmup_lrs = lrs[: 5 + 1]  # last_epoch == warmup_epochs still returns base_lr, so <= not <
    assert all(a <= b for a, b in zip(warmup_lrs, warmup_lrs[1:]))


def test_cosine_annealing_reaches_eta_min_at_max_epochs():
    lrs = _run_schedule(warmup_epochs=5, max_epochs=20, base_lr=1.0, eta_min=0.1)
    assert lrs[20] == pytest.approx(0.1, abs=1e-4)


def test_cosine_annealing_is_monotonically_decreasing():
    lrs = _run_schedule(warmup_epochs=5, max_epochs=20, base_lr=1.0, eta_min=0.0)
    annealing_lrs = lrs[5:21]  # from end-of-warmup through max_epochs
    assert all(a >= b for a, b in zip(annealing_lrs, annealing_lrs[1:]))


def test_lr_stays_within_bounds_throughout():
    # the cosine phase descends toward eta_min, which can be below warmup_start_lr
    lrs = _run_schedule(warmup_epochs=5, max_epochs=20, base_lr=1.0, warmup_start_lr=0.2, eta_min=0.1)
    assert all(0.1 - 1e-6 <= lr <= 1.0 + 1e-6 for lr in lrs)
