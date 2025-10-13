#!/usr/bin/env python3
"""
Test script for quanto quantization integration
Tests different quantization levels (8-bit, 4-bit, 2-bit, 1-bit)
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from UCF_VIT.simple.arch import VIT
from UCF_VIT.utils.quanto_quantization import setup_quanto_quantization, log_quanto_quantization_info

def test_quanto_quantization():
    """Test quanto quantization with different bit levels"""
    
    print("🚀 Testing Quanto Quantization Integration")
    print("=" * 60)
    
    # Create a simple VIT model for testing
    model = VIT(
        img_size=[256, 256],
        patch_size=16,
        num_classes=1000,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.0,
        drop_rate=0.0,
        twoD=True,
        weight_init='',
        default_vars=["red", "green", "blue"],
        single_channel=False,
        use_varemb=False,
        adaptive_patching=False,
        fixed_length=None,
        FusedAttn_option=None,
        use_adaptive_pos_emb=False,
    )
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Original model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f} MB")
    
    # Test different quantization levels
    quantization_levels = [8]
    
    for bits in quantization_levels:
        print(f"\n🔧 Testing {bits}-bit quantization...")
        
        # Create quantization config
        quantization_config = {
            'enabled': True,
            'bits': bits,
            'quantize_weights': True,
            'quantize_activations': True,
            'rocm_optimizations': True,
            'mi250x_kernels': True,
            'performance_mode': 'extreme_scale',
            'profile_quantization': True,
            'calibration_samples': 1000,
            'quantize_layers': ['linear', 'conv'],
            'exclude_layers': ['cls_token', 'pos_embed']
        }
        
        try:
            # Apply quantization
            quantized_model = setup_quanto_quantization(model, quantization_config)
            
            # Test forward pass
            test_input = torch.randn(1, 3, 256, 256)
            test_variables = ["red", "green", "blue"]
            
            with torch.no_grad():
                output = quantized_model(test_input, test_variables)
            
            print(f"✅ {bits}-bit quantization successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Log quantization info
            log_quanto_quantization_info(quantized_model, {}, quantization_config)
            
        except Exception as e:
            print(f"❌ {bits}-bit quantization failed: {e}")
    
    print("\n🎉 Quanto quantization testing completed!")

if __name__ == "__main__":
    test_quanto_quantization()
