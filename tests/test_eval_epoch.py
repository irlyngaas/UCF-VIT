"""Tests for UCF_VIT.training.eval_epoch's forward-only batch loop.

eval_epoch is val.py/test.py's counterpart to train_epoch -- it mirrors
train_epoch's per-iteration process_batch -> forward_step -> print logic
exactly (reusing both functions unchanged), but wraps everything in
torch.no_grad() and never calls backward()/optimizer.step()/scheduler.step()/
save_checkpoint at all (val.py/test.py build no optimizer/scheduler in the
first place -- see val.py's own docstring).

Uses a fake, always-iterable dataloader and a fake MAE model (mirroring
test_forward_step.py's _FakeMAEModel) rather than a real DataLoader/model --
eval_epoch builds one real iterator (iter(eval_dataloader), once per call,
not once per batch -- see process_batch's own it_loader docstring entry for
why that matters) and process_batch only ever calls next() on it, and
forward_step only ever calls model.forward(...), so neither the dataloader
nor the model needs to be real. tensor_par_group is unused entirely at
tensor_par_size==1 (only read inside process_batch's tensor_par_size>1
branch), so None is a safe placeholder.
"""

import torch

from UCF_VIT.training import eval_epoch
from UCF_VIT.utils.misc import patchify

PATCH_SIZE = 2
# Same fixture shape as test_forward_step.py's own MSE non-do_ap case (verified
# there): a 4x4 image, patch_size=2 -> 4 patches, patch_dim=4. Output off by a
# constant 2 everywhere -> per-iteration "MSE" loss (mean over every patch) is
# exactly 4.0.
DATA = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
TARGET = patchify(DATA, PATCH_SIZE, True)  # (1, 4, 4)
OUTPUT_OFF_BY_2 = TARGET + 2.0
PER_ITERATION_MSE = 4.0


class _FakeIterableDataloader:
    """Always yields the same fixed (data, variables, dict_key) 3-tuple that
    get_batch's MAE/do_ap:False branch unpacks -- __iter__ returns self (like
    a real DataLoader's own __iter__ would return a real, freshly-constructed
    iterator) so eval_epoch's one iter(eval_dataloader) call works the same
    way a real one does.
    """

    def __iter__(self):
        return self

    def __next__(self):
        return DATA, ["v0"], "ct1"


class _FakeMAEModel:
    """Stub standing in for a real MAE model. Records whether autograd was
    enabled at the moment of each forward call (to verify eval_epoch's
    torch.no_grad() wrapping) and returns a fixed (output, mask).
    """

    def __init__(self, output, mask):
        self._output = output
        self._mask = mask
        self.grad_enabled_during_calls = []

    def forward(self, x, variables, seq_ps):
        self.grad_enabled_during_calls.append(torch.is_grad_enabled())
        return self._output, self._mask


def _conf(loss_fn="MSE"):
    return {
        "model": {"type": "MAE", "loss_fn": loss_fn},
        "ap": {"do_ap": False},
        "data": {"patch_size": PATCH_SIZE, "twoD": True, "dataset": "basic_ct", "num_channels": {"ct1": 1}, "tile_size": (4, 4)},
        "dataloader": {"return_label": False, "batch_size": 1},
        "parallelism": {"tensor_par_size": 1},
        "trainer": {"data_type": "float32"},
    }


def test_eval_epoch_runs_forward_under_no_grad():
    mask = torch.zeros(1, 4)
    model = _FakeMAEModel(TARGET, mask)

    eval_epoch(_conf(), model, _FakeIterableDataloader(), epoch=0, iterations_per_epoch=3,
               device=torch.device("cpu"), tensor_par_group=None, ddpm_scheduler=None)

    assert len(model.grad_enabled_during_calls) == 3
    assert all(enabled is False for enabled in model.grad_enabled_during_calls)


def test_eval_epoch_sums_loss_over_every_iteration():
    mask = torch.zeros(1, 4)
    model = _FakeMAEModel(OUTPUT_OFF_BY_2, mask)

    epoch_loss, epoch_accuracy = eval_epoch(_conf(), model, _FakeIterableDataloader(), epoch=0, iterations_per_epoch=4,
                                             device=torch.device("cpu"), tensor_par_group=None, ddpm_scheduler=None)

    # Summed (not averaged) over 4 iterations, matching train_epoch's own
    # epoch_loss accumulation semantics.
    assert epoch_loss.item() == PER_ITERATION_MSE * 4
    assert epoch_accuracy.item() == 0.0  # MAE never computes a per-batch accuracy/Dice metric


def test_eval_epoch_returns_zero_iterations_untouched():
    model = _FakeMAEModel(TARGET, torch.zeros(1, 4))

    epoch_loss, epoch_accuracy = eval_epoch(_conf(), model, _FakeIterableDataloader(), epoch=0, iterations_per_epoch=0,
                                             device=torch.device("cpu"), tensor_par_group=None, ddpm_scheduler=None)
    assert len(model.grad_enabled_during_calls) == 0
    assert epoch_loss.item() == 0.0


def test_eval_epoch_reuses_a_given_it_loader_instead_of_building_a_fresh_one():
    """The actual fix this test locks in: an already-iter()'d it_loader (val.py/
    test.py's own pre-CUDA-init warm-up iterator, real callers) is used
    directly, not discarded in favor of a fresh iter(eval_dataloader) call --
    verified here by passing an eval_dataloader that would raise if anything
    ever actually tried to iterate it, and a real, independent iterator (not
    the self-returning _FakeIterableDataloader trick the other tests use) as
    it_loader.
    """
    class _ExplodingDataloaderIfIterated:
        def __iter__(self):
            raise AssertionError("eval_epoch should not have called iter(eval_dataloader) -- it_loader was given")

    mask = torch.zeros(1, 4)
    model = _FakeMAEModel(TARGET, mask)
    it_loader = iter([(DATA, ["v0"], "ct1")] * 3)

    eval_epoch(_conf(), model, _ExplodingDataloaderIfIterated(), epoch=0, iterations_per_epoch=3,
               device=torch.device("cpu"), tensor_par_group=None, ddpm_scheduler=None, it_loader=it_loader)

    assert len(model.grad_enabled_during_calls) == 3
