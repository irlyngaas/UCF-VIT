#!/usr/bin/env python3
"""Regression tests for opt-in diffusion decoder refinements."""

import unittest

import torch

from UCF_VIT.fsdp.arch import DiffusionVIT
from UCF_VIT.utils.fused_attn import FusedAttn
from UCF_VIT.utils.misc import patchify, unpatchify


def _small_model(**decoder_options):
    options = dict(
        img_size=[8, 8, 8],
        patch_size=4,
        in_chans=1,
        embed_dim=48,
        depth=1,
        num_heads=6,
        decoder_depth=1,
        decoder_embed_dim=48,
        decoder_num_heads=6,
        mlp_ratio=2,
        drop_path_rate=0,
        linear_decoder=True,
        twoD=False,
        mlp_ratio_decoder=2,
        default_vars=['nct'],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=196,
        tensor_par_size=1,
        tensor_par_group=None,
        FusedAttn_option=FusedAttn.DEFAULT,
        time_steps=1000,
        class_token=False,
        weight_init='skip',
    )
    options.update(decoder_options)
    return DiffusionVIT(**options)


class DiffusionDecoderModesTest(unittest.TestCase):
    def test_legacy_linear_model_has_no_residual_parameters(self):
        model = _small_model()
        self.assertFalse(any(
            name.startswith('decoder_residual')
            for name in model.state_dict()
        ))

    def test_conv3d_token_layout_matches_shared_patch_helpers(self):
        model = _small_model(residual_conv3d_decoder=True)
        tokens = torch.randn(2, model.num_patches, model.patch_dim)
        reference_shape = torch.empty(2, 1, 8, 8, 8)

        volume = model._tokens_to_3d_volume(tokens)
        torch.testing.assert_close(
            volume,
            unpatchify(tokens, reference_shape, patch_size=4, twoD=False),
        )
        torch.testing.assert_close(
            model._volume_to_3d_tokens(volume),
            patchify(volume, patch_size=4, twoD=False),
        )
        torch.testing.assert_close(
            model._volume_to_3d_tokens(volume), tokens
        )

    def test_conv3d_step_zero_matches_legacy_linear_decoder(self):
        legacy = _small_model()
        refined = _small_model(residual_conv3d_decoder=True)
        incompatible = refined.load_state_dict(legacy.state_dict(), strict=False)

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith('decoder_residual_conv3d.')
            for key in incompatible.missing_keys
        ))
        features = torch.randn(2, legacy.num_patches, legacy.embed_dim)
        torch.testing.assert_close(
            refined.forward_head(features),
            legacy.forward_head(features),
            atol=0,
            rtol=0,
        )

    def test_conv3d_residual_receives_gradient(self):
        model = _small_model(residual_conv3d_decoder=True)
        features = torch.randn(
            2, model.num_patches, model.embed_dim, requires_grad=True
        )
        target = torch.randn(2, model.num_patches, model.patch_dim)
        torch.nn.functional.mse_loss(model.forward_head(features), target).backward()

        output_conv = model.decoder_residual_conv3d[-1]
        self.assertIsNotNone(output_conv.weight.grad)
        self.assertGreater(output_conv.weight.grad.abs().sum().item(), 0)

    def test_feature_conv3d_step_zero_matches_legacy_linear_decoder(self):
        legacy = _small_model()
        refined = _small_model(residual_feature_conv3d_decoder=True)
        incompatible = refined.load_state_dict(legacy.state_dict(), strict=False)

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            key.startswith('decoder_residual_feature_')
            for key in incompatible.missing_keys
        ))
        features = torch.randn(2, legacy.num_patches, legacy.embed_dim)
        torch.testing.assert_close(
            refined.forward_head(features),
            legacy.forward_head(features),
            atol=0,
            rtol=0,
        )

    def test_feature_conv3d_uses_multichannel_spatial_layout(self):
        model = _small_model(
            residual_feature_conv3d_decoder=True,
            residual_feature_conv3d_channels=3,
        )
        tokens = torch.randn(2, model.num_patches, 4 ** 3 * 3)
        volume = model._tokens_to_3d_volume(tokens, num_channels=3)
        self.assertEqual(volume.shape, (2, 3, 8, 8, 8))
        torch.testing.assert_close(
            model._volume_to_3d_tokens(volume), tokens
        )

    def test_feature_conv3d_projection_and_output_receive_gradients(self):
        model = _small_model(residual_feature_conv3d_decoder=True)
        with torch.no_grad():
            model.decoder_residual_feature_conv3d[-1].weight.normal_(
                std=1e-3
            )
        features = torch.randn(
            2, model.num_patches, model.embed_dim, requires_grad=True
        )
        target = torch.randn(2, model.num_patches, model.patch_dim)
        torch.nn.functional.mse_loss(model.forward_head(features), target).backward()

        projection = model.decoder_residual_feature_proj[1]
        output_conv = model.decoder_residual_feature_conv3d[-1]
        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(output_conv.weight.grad.abs().sum().item(), 0)

    def test_spatial_feature_conv3d_replaces_patchwise_linear_head(self):
        model = _small_model(
            spatial_feature_conv3d_decoder=True,
            spatial_feature_conv3d_channels=3,
        )
        features = torch.randn(
            2, model.num_patches, model.embed_dim, requires_grad=True
        )
        target = torch.randn(2, model.num_patches, model.patch_dim)
        prediction = model.forward_head(features)

        self.assertEqual(prediction.shape, target.shape)
        torch.nn.functional.mse_loss(prediction, target).backward()
        self.assertFalse(hasattr(model, 'decoder_pred'))
        projection = model.decoder_spatial_feature_proj[1]
        output_conv = model.decoder_spatial_feature_conv3d[-1]
        self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(output_conv.weight.grad.abs().sum().item(), 0)

    def test_spatial_feature_conv3d_is_mutually_exclusive_with_residual(self):
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            _small_model(
                spatial_feature_conv3d_decoder=True,
                residual_feature_conv3d_decoder=True,
            )

    def test_spatial_feature_conv2d_layout_and_gradients(self):
        model = _small_model(
            twoD=True,
            spatial_feature_conv2d_decoder=True,
            spatial_feature_conv2d_channels=3,
        )
        tokens = torch.randn(2, model.num_patches, 4 ** 2 * 3)
        image = model._tokens_to_2d_image(tokens, num_channels=3)
        self.assertEqual(image.shape, (2, 3, 8, 8))
        torch.testing.assert_close(model._image_to_2d_tokens(image), tokens)

        features = torch.randn(
            2, model.num_patches, model.embed_dim, requires_grad=True
        )
        target = torch.randn(2, model.num_patches, model.patch_dim)
        prediction = model.forward_head(features)
        self.assertEqual(prediction.shape, target.shape)
        self.assertFalse(hasattr(model, 'decoder_pred'))
        torch.nn.functional.mse_loss(prediction, target).backward()
        self.assertGreater(
            model.decoder_spatial_feature_proj[1].weight.grad.abs().sum(), 0
        )
        self.assertGreater(
            model.decoder_spatial_feature_conv2d[-1].weight.grad.abs().sum(), 0
        )

    def test_spatial_feature_conv2d_rejects_3d_model(self):
        with self.assertRaisesRegex(ValueError, 'requires a 2D model'):
            _small_model(spatial_feature_conv2d_decoder=True)

    def test_residual_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'only one residual decoder'):
            _small_model(
                residual_mlp_decoder=True,
                residual_conv3d_decoder=True,
            )
        with self.assertRaisesRegex(ValueError, 'only one residual decoder'):
            _small_model(
                residual_conv3d_decoder=True,
                residual_feature_conv3d_decoder=True,
            )


if __name__ == '__main__':
    unittest.main()
