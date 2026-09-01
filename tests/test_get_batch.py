"""Tests for UCF_VIT.training.get_batch's DataLoader-worker-crash error wrapping.

basic_ct-sap+tensor_par intermittently segfaulted on real Frontier runs (job
5390076) -- a DataLoader worker (dataloader.num_workers > 0) forked after
CUDA/NCCL was already initialized in the parent process, a documented hazard
(see configs/basic_ct/sap/base_config.yaml's own num_workers comment and
tests/README.md for the full story, including why multiprocessing_context:
"spawn" isn't a safe blanket fix either). Fixed for that config with
num_workers:0, but any other config combining num_workers>0 with a similar
risky combination (tensor_par_size>1 + adaptive patching on 3D data) could
still hit the same PyTorch RuntimeError -- get_batch now catches PyTorch's own
"... exited unexpectedly" RuntimeError (the outer, catchable exception
next(it_loader) actually raises -- confirmed against the installed torch's own
dataloader.py source) and re-raises with a pointer to try num_workers:0.

Uses a fake it_loader (no real DataLoader/model/timm/monai/xformers needed --
get_batch only ever calls next(it_loader) and reads conf) rather than
reproducing a real worker crash, which isn't reproducible outside a real
multi-rank Frontier/CUDA/NCCL environment in the first place.
"""

import pytest

from UCF_VIT.training import get_batch


def _conf(num_workers, model_type="MAE", do_ap=False):
    return {
        "model": {"type": model_type},
        "ap": {"do_ap": do_ap},
        "dataloader": {"num_workers": num_workers, "return_label": False},
    }


class _RaisingIter:
    def __init__(self, exc):
        self._exc = exc

    def __next__(self):
        raise self._exc


def _real_worker_crash_error():
    # Mirrors PyTorch's own chaining: signal_handling.py's handler raises the
    # inner "is killed by signal" RuntimeError while dataloader.py's
    # _try_get_data is already handling a timeout, so it ends up chained as
    # the outer "exited unexpectedly" RuntimeError's __cause__ -- exactly the
    # shape seen in every real Frontier log this was diagnosed from.
    try:
        try:
            raise RuntimeError("DataLoader worker (pid 12345) is killed by signal: Segmentation fault. ")
        except RuntimeError as inner:
            raise RuntimeError("DataLoader worker (pid(s) 12345) exited unexpectedly") from inner
    except RuntimeError as outer:
        return outer


def test_get_batch_augments_worker_crash_message_when_num_workers_positive():
    it = _RaisingIter(_real_worker_crash_error())
    with pytest.raises(RuntimeError) as excinfo:
        get_batch(_conf(num_workers=1), it)
    assert "num_workers" in str(excinfo.value)
    assert "exited unexpectedly" in str(excinfo.value)
    # Original PyTorch exception chain preserved as the cause, not swallowed:
    # excinfo.value.__cause__ is the caught "exited unexpectedly" RuntimeError,
    # whose own __cause__ (PyTorch's own chaining) is the "is killed by signal" one.
    assert "exited unexpectedly" in str(excinfo.value.__cause__)
    assert "is killed by signal" in str(excinfo.value.__cause__.__cause__)


def test_get_batch_does_not_augment_when_num_workers_zero():
    # No worker process exists to crash when num_workers:0 -- augmenting here
    # would be a misleading false positive, so the original message passes
    # through unchanged.
    it = _RaisingIter(_real_worker_crash_error())
    with pytest.raises(RuntimeError) as excinfo:
        get_batch(_conf(num_workers=0), it)
    assert "Try setting dataloader.num_workers" not in str(excinfo.value)
    assert "exited unexpectedly" in str(excinfo.value)


def test_get_batch_does_not_augment_unrelated_runtime_errors():
    it = _RaisingIter(RuntimeError("some unrelated dataloader problem"))
    with pytest.raises(RuntimeError) as excinfo:
        get_batch(_conf(num_workers=1), it)
    assert "Try setting dataloader.num_workers" not in str(excinfo.value)
    assert str(excinfo.value) == "some unrelated dataloader problem"
