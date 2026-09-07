import os
import tempfile
import unittest
from pathlib import Path

import torch

from train_diffusion_fsdp_wFixedFID_2D_singMod import (
    latest_complete_best_checkpoint,
    rebase_optimizer_and_scheduler_lr,
    update_loss_degradation_streak,
)
from UCF_VIT.utils.misc import configure_scheduler


class NumericalRecoveryTest(unittest.TestCase):
    def test_rebases_optimizer_and_scheduler(self):
        parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        optimizer = torch.optim.AdamW([
            {'params': [parameters[0]], 'lr': 0.0049},
            {'params': [parameters[1]], 'lr': 0.0049},
        ])
        scheduler = configure_scheduler(
            optimizer,
            warmup_steps=10,
            max_steps=200,
            warmup_start_lr=1e-8,
            eta_min=1e-8,
        )
        for _ in range(40):
            optimizer.step()
            scheduler.step()

        rebase_optimizer_and_scheduler_lr(optimizer, scheduler, 0.001)

        self.assertTrue(
            all(0.0 < g['lr'] <= 0.001 for g in optimizer.param_groups)
        )
        self.assertEqual(
            [g['initial_lr'] for g in optimizer.param_groups], [0.001] * 2
        )
        self.assertEqual(scheduler.base_lrs, [0.001] * 2)
        self.assertTrue(all(0.0 < lr <= 0.001 for lr in scheduler._last_lr))
        optimizer.step()
        scheduler.step()
        self.assertTrue(all(0.0 < lr <= 0.001 for lr in scheduler.get_last_lr()))

    def test_selects_latest_checkpoint_complete_across_tp_ranks(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            for epoch, ranks in ((8, (0, 1)), (9, (0,))):
                for rank in ranks:
                    path = os.path.join(
                        checkpoint_path,
                        f'model_BEST_{epoch}_rank_{rank}.ckpt',
                    )
                    Path(path).touch()

            selected = latest_complete_best_checkpoint(
                checkpoint_path, 'model', tensor_par_size=2
            )

        self.assertEqual(selected, 'model_BEST_8')

    def test_finite_loss_degradation_requires_consecutive_epochs(self):
        streak, should_recover = update_loss_degradation_streak(
            current_loss=0.13,
            best_loss=0.06,
            factor=2.0,
            streak=0,
            patience=2,
        )
        self.assertEqual(streak, 1)
        self.assertFalse(should_recover)

        streak, should_recover = update_loss_degradation_streak(
            current_loss=0.47,
            best_loss=0.06,
            factor=2.0,
            streak=streak,
            patience=2,
        )
        self.assertEqual(streak, 2)
        self.assertTrue(should_recover)

    def test_healthy_loss_resets_degradation_streak(self):
        streak, should_recover = update_loss_degradation_streak(
            current_loss=0.07,
            best_loss=0.06,
            factor=2.0,
            streak=1,
            patience=2,
        )
        self.assertEqual(streak, 0)
        self.assertFalse(should_recover)


if __name__ == '__main__':
    unittest.main()
