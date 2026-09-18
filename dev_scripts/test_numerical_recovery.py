import os
import tempfile
import unittest
from pathlib import Path

import torch

from train_diffusion_fsdp_wFixedFID_2D_singMod import (
    completed_epoch_improves_best,
    configure_optimizer_with_residual_lr,
    continuation_epoch_fields,
    latest_complete_best_checkpoint,
    initialize_ema_parameters,
    load_model_checkpoint,
    modality_metric_spec,
    normalize_dataset_key,
    rebase_optimizer_and_scheduler_lr,
    restart_optimizer_and_scheduler_lr,
    should_generate_preview,
    should_generate_resume_preview,
    should_save_periodic_checkpoint,
    update_ema_degradation,
    update_ema_parameters,
    update_loss_degradation_streak,
)
from UCF_VIT.utils.misc import configure_scheduler


class NumericalRecoveryTest(unittest.TestCase):
    def test_residual_optimizer_uses_separate_learning_rate(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 4)
                self.decoder_residual = torch.nn.Sequential(
                    torch.nn.Linear(4, 8),
                    torch.nn.GELU(),
                    torch.nn.Linear(8, 2),
                )

        model = Model()
        optimizer, counts = configure_optimizer_with_residual_lr(
            model, base_lr=5e-7, residual_lr=1e-4,
            beta_1=0.9, beta_2=0.95, weight_decay=1e-5,
        )
        group_lrs = {
            group['group_name']: group['lr']
            for group in optimizer.param_groups
        }
        self.assertEqual(group_lrs['backbone_decay'], 5e-7)
        self.assertEqual(group_lrs['residual_decay'], 1e-4)
        self.assertGreater(counts['residual_decay'], 0)

        scheduler = configure_scheduler(
            optimizer, warmup_steps=1, max_steps=100,
            warmup_start_lr=5e-7, eta_min=1e-8,
        )
        restarted = restart_optimizer_and_scheduler_lr(
            optimizer, scheduler, resume_lr=5e-7,
            max_steps=100, eta_min=1e-8,
        )
        self.assertEqual(restarted, [5e-7, 1e-4])

    def test_partial_resume_allows_only_new_residual_decoder(self):
        class RefinedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.decoder_residual = torch.nn.Linear(2, 2)

        model = RefinedModel()
        legacy_state = {
            key: value for key, value in model.state_dict().items()
            if not key.startswith('decoder_residual.')
        }
        missing, unexpected = load_model_checkpoint(
            model, legacy_state, allow_residual_decoder_init=True
        )
        self.assertEqual(unexpected, [])
        self.assertTrue(missing)
        self.assertTrue(all(
            key.startswith('decoder_residual.') for key in missing
        ))

    def test_partial_resume_allows_new_conv3d_residual_decoder(self):
        class RefinedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.decoder_residual_conv3d = torch.nn.Conv3d(1, 1, 3)

        model = RefinedModel()
        legacy_state = {
            key: value for key, value in model.state_dict().items()
            if not key.startswith('decoder_residual_conv3d.')
        }
        missing, unexpected = load_model_checkpoint(
            model, legacy_state, allow_residual_decoder_init=True
        )
        self.assertEqual(unexpected, [])
        self.assertTrue(missing)
        self.assertTrue(all(
            key.startswith('decoder_residual_conv3d.') for key in missing
        ))

    def test_partial_resume_allows_new_feature_conv3d_residual_decoder(self):
        class RefinedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)
                self.decoder_residual_feature_proj = torch.nn.Linear(2, 4)
                self.decoder_residual_feature_conv3d = torch.nn.Conv3d(
                    1, 1, 3
                )

        model = RefinedModel()
        legacy_state = {
            key: value for key, value in model.state_dict().items()
            if not key.startswith('decoder_residual_feature_')
        }
        missing, unexpected = load_model_checkpoint(
            model, legacy_state, allow_residual_decoder_init=True
        )
        self.assertEqual(unexpected, [])
        self.assertTrue(missing)
        self.assertTrue(all(
            key.startswith('decoder_residual_feature_') for key in missing
        ))

    def test_ema_parameter_update(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema = initialize_ema_parameters(model)
        with torch.no_grad():
            model.weight.fill_(4.0)
        update_ema_parameters(model, ema, decay=0.75)
        torch.testing.assert_close(ema[0], torch.full_like(ema[0], 2.5))

    def test_partial_walltime_checkpoint_replays_only_interrupted_epoch(self):
        self.assertEqual(
            continuation_epoch_fields(epoch=26, epoch_completed=False),
            {'epoch': 26, 'next_epoch': 26},
        )

    def test_completed_walltime_checkpoint_advances_one_epoch(self):
        self.assertEqual(
            continuation_epoch_fields(epoch=26, epoch_completed=True),
            {'epoch': 26, 'next_epoch': 27},
        )

    def test_only_completed_epoch_can_replace_best_checkpoint(self):
        self.assertTrue(completed_epoch_improves_best(True, 0.04, 0.05))
        self.assertFalse(completed_epoch_improves_best(False, 0.04, 0.05))
        self.assertFalse(completed_epoch_improves_best(True, 0.06, 0.05))

    def test_preview_runs_on_new_best_and_global_period(self):
        self.assertTrue(should_generate_preview(
            35, 35, True, 50, epoch_improved=True
        ))
        self.assertTrue(should_generate_preview(50, 44, True, 50))
        self.assertFalse(should_generate_preview(44, 44, False, 50))
        self.assertFalse(should_generate_preview(36, 35, True, 50))

    def test_resume_preview_requires_a_restored_best_state(self):
        self.assertTrue(should_generate_resume_preview(True, True))
        self.assertFalse(should_generate_resume_preview(False, True))
        self.assertFalse(should_generate_resume_preview(True, False))

    def test_periodic_checkpoint_requires_completed_period_epoch(self):
        self.assertTrue(should_save_periodic_checkpoint(50, True, 50))
        self.assertFalse(should_save_periodic_checkpoint(50, False, 50))
        self.assertFalse(should_save_periodic_checkpoint(49, True, 50))

    def test_ema_degradation_requires_sustained_relative_regression(self):
        ema = None
        best = float('inf')
        streak = 0
        triggered = False
        for loss in (1.0, 1.3, 1.3, 1.3):
            ema, best, streak, triggered = update_ema_degradation(
                loss, ema, best, alpha=1.0, factor=1.2,
                streak=streak, patience=3,
            )
        self.assertEqual(best, 1.0)
        self.assertEqual(streak, 3)
        self.assertTrue(triggered)

    def test_ema_degradation_resets_when_loss_recovers(self):
        ema, best, streak, triggered = update_ema_degradation(
            1.05, 1.3, 1.0, alpha=1.0, factor=1.2,
            streak=2, patience=3,
        )
        self.assertEqual((ema, best, streak), (1.05, 1.0, 0))
        self.assertFalse(triggered)

    def test_modality_metrics_follow_arbitrary_configuration(self):
        spec = modality_metric_spec(
            {'xray_source': '/x', 'neutron_source': '/n', 'mri_source': '/m'},
            {
                'xray_source': ['xct'],
                'neutron_source': ['nct'],
                'mri_source': ['t1', 't2'],
            },
            dataset='xct',
        )
        self.assertEqual(spec, [
            ('xray_source', 'xct'),
            ('neutron_source', 'nct'),
            ('mri_source', 't1+t2'),
        ])
        self.assertEqual(normalize_dataset_key(['mri', '_source']), 'mri_source')

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

    def test_restarts_optimizer_and_scheduler_at_requested_lr(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.004)
        scheduler = configure_scheduler(
            optimizer, warmup_steps=10, max_steps=200,
            warmup_start_lr=1e-8, eta_min=1e-8,
        )
        for _ in range(40):
            optimizer.step()
            scheduler.step()

        current_lrs = restart_optimizer_and_scheduler_lr(
            optimizer, scheduler, resume_lr=1e-5,
            max_steps=100, eta_min=1e-8,
        )

        self.assertEqual(current_lrs, [1e-5])
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(scheduler.max_epochs, 100)
        optimizer.step()
        scheduler.step()
        self.assertLess(scheduler.get_last_lr()[0], 1e-5)
        self.assertGreater(scheduler.get_last_lr()[0], 1e-8)

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

    def test_prefers_stable_best_checkpoint_complete_across_tp_ranks(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            for rank in (0, 1):
                Path(os.path.join(
                    checkpoint_path,
                    f'model_BEST_rank_{rank}.ckpt',
                )).touch()
            for rank in (0, 1):
                Path(os.path.join(
                    checkpoint_path,
                    f'model_BEST_12_rank_{rank}.ckpt',
                )).touch()

            selected = latest_complete_best_checkpoint(
                checkpoint_path, 'model', tensor_par_size=2
            )

        self.assertEqual(selected, 'model_BEST')

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
