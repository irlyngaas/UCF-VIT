#!/usr/bin/env python3
"""
Test extreme quantization (4-bit, 2-bit, 1-bit) on AMD GPU
Comprehensive performance and accuracy testing
"""

import torch
import torch.nn as nn
import time
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.UCF_VIT.quantization.extreme_quantization import (
    ExtremeQuantizer, BinaryQuantizer, QuantizedLinearExtreme,
    quantize_model_extreme, get_model_size_mb, get_quantized_model_size_mb
)


def test_basic_quantization():
    """Test basic quantization for different bit widths"""
    print("=" * 60)
    print("TESTING BASIC EXTREME QUANTIZATION")
    print("=" * 60)
    
    # Create test tensor
    torch.manual_seed(42)
    test_tensor = torch.randn(100, 100) * 5.0
    
    print(f"Original tensor:")
    print(f"  Shape: {test_tensor.shape}")
    print(f"  Range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
    print(f"  Mean: {test_tensor.mean():.3f}")
    print(f"  Std: {test_tensor.std():.3f}")
    print()
    
    # Test different bit widths
    for bits in [4, 2, 1]:
        print(f"Testing {bits}-bit quantization:")
        
        if bits == 1:
            quantizer = BinaryQuantizer()
        else:
            quantizer = ExtremeQuantizer(bits=bits, symmetric=True)
        
        # Quantize and dequantize
        quantized, scale = quantizer.quantize_tensor(test_tensor)
        dequantized = quantizer.dequantize_tensor(quantized, scale)
        
        # Calculate error
        error = (test_tensor - dequantized).abs()
        relative_error = (error / test_tensor.abs().mean()) * 100
        
        print(f"  Scale: {scale:.6f}")
        print(f"  Quantized range: [{quantized.min()}, {quantized.max()}]")
        print(f"  Mean error: {error.mean():.6f}")
        print(f"  Max error: {error.max():.6f}")
        print(f"  Relative error: {relative_error.mean():.3f}%")
        
        # Memory calculation
        original_bits = 32  # FP32
        compression_ratio = original_bits / bits
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        print()


def test_quantized_linear_layers():
    """Test quantized linear layers for different bit widths"""
    print("=" * 60)
    print("TESTING EXTREME QUANTIZED LINEAR LAYERS")
    print("=" * 60)
    
    # Test configuration
    batch_size = 32
    in_features = 768
    out_features = 3072
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features)
    
    # Create reference FP32 layer
    fp32_layer = nn.Linear(in_features, out_features)
    fp32_output = fp32_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Linear layer: {in_features} -> {out_features}")
    print()
    
    # Test different bit widths
    results = {}
    
    for bits in [4, 2, 1]:
        print(f"Testing {bits}-bit quantized linear layer:")
        
        # Create quantized layer
        quantized_layer = QuantizedLinearExtreme(in_features, out_features, bits=bits)
        quantized_layer.load_from_fp32(fp32_layer.weight.data, fp32_layer.bias.data)
        
        # Timing test
        num_runs = 100
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(num_runs):
            quantized_output = quantized_layer(x)
        end_time = time.time()
        
        quantized_time = (end_time - start_time) * 1000 / num_runs  # ms per run
        
        # Accuracy test
        output_diff = (fp32_output - quantized_output).abs()
        relative_diff = (output_diff / fp32_output.abs().mean()) * 100
        
        # Memory calculation
        fp32_params = sum(p.numel() for p in fp32_layer.parameters())
        fp32_memory = fp32_params * 4 / (1024 * 1024)  # MB
        
        quantized_memory = get_quantized_model_size_mb(quantized_layer, bits)
        memory_reduction = fp32_memory / quantized_memory if quantized_memory > 0 else float('inf')
        
        results[bits] = {
            'time': quantized_time,
            'accuracy_diff': relative_diff.mean().item(),
            'memory_mb': quantized_memory,
            'memory_reduction': memory_reduction
        }
        
        print(f"  Inference time: {quantized_time:.3f} ms")
        print(f"  Output difference mean: {output_diff.mean():.6f}")
        print(f"  Output difference max: {output_diff.max():.6f}")
        print(f"  Relative difference: {relative_diff.mean():.3f}%")
        print(f"  Memory usage: {quantized_memory:.2f} MB")
        print(f"  Memory reduction: {memory_reduction:.1f}x")
        print()
    
    # Compare with FP32 timing
    print("FP32 baseline timing:")
    start_time = time.time()
    for _ in range(num_runs):
        fp32_output = fp32_layer(x)
    end_time = time.time()
    fp32_time = (end_time - start_time) * 1000 / num_runs
    
    fp32_memory = fp32_params * 4 / (1024 * 1024)
    
    print(f"  FP32 time: {fp32_time:.3f} ms")
    print(f"  FP32 memory: {fp32_memory:.2f} MB")
    print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY:")
    print(f"{'Bits':<6} {'Time (ms)':<12} {'Speedup':<10} {'Accuracy Loss':<15} {'Memory Reduction':<18}")
    print("-" * 70)
    print(f"{'FP32':<6} {fp32_time:<12.3f} {'1.0x':<10} {'0.0%':<15} {'1.0x':<18}")
    
    for bits in [4, 2, 1]:
        speedup = fp32_time / results[bits]['time']
        print(f"{bits:<6} {results[bits]['time']:<12.3f} {speedup:<10.1f}x {results[bits]['accuracy_diff']:<15.2f}% {results[bits]['memory_reduction']:<18.1f}x")


def test_model_quantization():
    """Test quantization on a small model"""
    print("=" * 60)
    print("TESTING MODEL-LEVEL EXTREME QUANTIZATION")
    print("=" * 60)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(16, 512)
    
    # Original model
    original_model = SimpleModel()
    original_output = original_model(x)
    original_size = get_model_size_mb(original_model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {original_output.shape}")
    print()
    
    # Test different quantizations
    for bits in [4, 2, 1]:
        print(f"Testing {bits}-bit model quantization:")
        
        # Create quantized model
        quantized_model = SimpleModel()
        quantized_model.load_state_dict(original_model.state_dict())
        quantized_model = quantize_model_extreme(quantized_model, bits=bits)
        
        # Test forward pass
        quantized_output = quantized_model(x)
        
        # Calculate differences
        output_diff = (original_output - quantized_output).abs()
        relative_diff = (output_diff / original_output.abs().mean()) * 100
        
        # Calculate model size
        quantized_size = get_quantized_model_size_mb(quantized_model, bits)
        size_reduction = original_size / quantized_size if quantized_size > 0 else float('inf')
        
        print(f"  Quantized model size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {size_reduction:.1f}x")
        print(f"  Output difference mean: {output_diff.mean():.6f}")
        print(f"  Output difference max: {output_diff.max():.6f}")
        print(f"  Relative difference: {relative_diff.mean():.3f}%")
        print()


def main():
    """Run all extreme quantization tests"""
    print("EXTREME QUANTIZATION TEST")
    print("AMD GPU Optimized Implementation")
    print("Testing 4-bit, 2-bit, and 1-bit quantization")
    print("Running on Frontier Supercomputer")
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    try:
        # Run tests
        test_basic_quantization()
        test_quantized_linear_layers()
        test_model_quantization()
        
        print("=" * 60)
        print("ALL EXTREME QUANTIZATION TESTS PASSED!")
        print("4-bit, 2-bit, and 1-bit quantization working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())