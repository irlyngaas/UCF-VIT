"""Tests for UCF_VIT.training.load_optimizer_scheduler_from_checkpoint's
trainer.reset_scheduler_on_resume behavior.

Real motivating scenario: a run's trainer.max_epochs turns out to be too
small (loss hadn't converged), and the fix is to bump max_epochs way up
(e.g. 200 -> 1000) and keep training from the existing checkpoint. Before
this option existed, resuming always called scheduler.load_state_dict on the
checkpointed scheduler -- which restores its OLD max_epochs (200) along with
everything else, so the LR stays pinned wherever the old, already-mostly-
decayed-to-eta_min schedule left it, rather than running a real schedule
over the new, larger budget.

reset_scheduler_on_resume:True skips that load and instead rebuilds the
scheduler fresh, from the *current* config's scheduler_conf (whose max_epochs
is always derived from trainer.max_epochs -- see parse_config) -- attached to
the optimizer *after* optimizer.load_state_dict has already restored its
state, so the fresh scheduler's own construction-time LR write is what
actually wins over the just-restored (near-eta_min) lr, not something this
test needs to special-case around.

Default (reset_scheduler_on_resume:False, or omitted) behavior -- loading the
checkpointed scheduler state as-is -- is also covered here, as a regression
check that adding the new branch didn't change the existing path.
"""

import os

import torch

from UCF_VIT.training import load_optimizer_scheduler_from_checkpoint
from UCF_VIT.utils.misc import configure_scheduler


def _make_optimizer_and_scheduler(max_epochs):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler_kwargs = {
        "warmup_epochs": 10,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
        "max_epochs": max_epochs,
    }
    scheduler = configure_scheduler(optimizer, "linear-warmup-cosine-annealing", scheduler_kwargs)
    return optimizer, scheduler, scheduler_kwargs


def _write_checkpoint(tmp_path, filename, optimizer, scheduler, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": torch.nn.Linear(2, 2).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_list": [1.0] * (epoch + 1),
    }, os.path.join(str(tmp_path), f"{filename}_rank_0.ckpt"))


def _base_conf(tmp_path, scheduler_kwargs, reset_scheduler_on_resume):
    return {
        "trainer": {
            "checkpoint_path": str(tmp_path),
            "checkpoint_filename": "epoch_182",
            "scheduler_type": "linear-warmup-cosine-annealing",
            "reset_scheduler_on_resume": reset_scheduler_on_resume,
        },
        "parallelism": {"tensor_par_size": 1},
        "scheduler": scheduler_kwargs,
    }


def test_reset_scheduler_on_resume_rebuilds_fresh_instead_of_loading_old_state(tmp_path):
    # Old run: max_epochs=200, stepped to near its end (epoch 182) -- LR
    # already decayed close to eta_min.
    old_optimizer, old_scheduler, _ = _make_optimizer_and_scheduler(max_epochs=200)
    for _ in range(182):
        old_scheduler.step()
    old_lr_near_end = old_optimizer.param_groups[0]["lr"]
    assert old_lr_near_end < 1e-4  # sanity: genuinely decayed, not still near the start
    _write_checkpoint(tmp_path, "epoch_182", old_optimizer, old_scheduler, epoch=182)

    # New run: same checkpoint, but max_epochs bumped to 1000 and
    # reset_scheduler_on_resume:True -- configure_scheduler builds the fresh
    # scheduler/optimizer pair main() would, exactly as it does before calling
    # load_optimizer_scheduler_from_checkpoint for real.
    new_optimizer, new_scheduler, new_scheduler_kwargs = _make_optimizer_and_scheduler(max_epochs=1000)
    conf = _base_conf(tmp_path, new_scheduler_kwargs, reset_scheduler_on_resume=True)

    optimizer, scheduler, loss_list, epoch_start = load_optimizer_scheduler_from_checkpoint(
        conf, new_optimizer, new_scheduler, data_seq_ort_group=None, device="cpu",
    )

    assert epoch_start == 183  # still resumes at the checkpointed epoch, only the schedule resets
    assert loss_list == [1.0] * 183
    assert scheduler.max_epochs == 1000  # not 200 -- the old checkpointed schedule's shape
    # Fresh epoch-0 LR (warmup_start_lr), not wherever the old near-eta_min
    # schedule had decayed to.
    assert optimizer.param_groups[0]["lr"] == new_scheduler_kwargs["warmup_start_lr"]


def test_reset_scheduler_on_resume_false_still_loads_checkpointed_state(tmp_path):
    old_optimizer, old_scheduler, old_scheduler_kwargs = _make_optimizer_and_scheduler(max_epochs=200)
    for _ in range(182):
        old_scheduler.step()
    old_lr_near_end = old_optimizer.param_groups[0]["lr"]
    _write_checkpoint(tmp_path, "epoch_182", old_optimizer, old_scheduler, epoch=182)

    new_optimizer, new_scheduler, _ = _make_optimizer_and_scheduler(max_epochs=1000)
    conf = _base_conf(tmp_path, old_scheduler_kwargs, reset_scheduler_on_resume=False)

    optimizer, scheduler, loss_list, epoch_start = load_optimizer_scheduler_from_checkpoint(
        conf, new_optimizer, new_scheduler, data_seq_ort_group=None, device="cpu",
    )

    assert epoch_start == 183
    assert scheduler.max_epochs == 200  # restored from the checkpoint, not new_scheduler's fresh 1000
    assert optimizer.param_groups[0]["lr"] == old_lr_near_end
