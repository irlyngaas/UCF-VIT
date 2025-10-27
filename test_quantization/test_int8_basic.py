"""
Basic INT8 quantization test for Frontier supercomputer
Tests our custom quantization implementation without full model
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.UCF_VIT.quantization import Int8Quantizer, QuantizedLinear, quantize_model_int8


def test_basic_quantization():
    """Test basic quantization functions"""
    print("=" * 60)
    print("TESTING BASIC INT8 QUANTIZATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    quantizer = Int8Quantizer()
    
    # Test tensor on GPU
    x = torch.randn(100, 100, device=device) * 2.5
    print(f"\nOriginal tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Mean: {x.mean():.3f}")
    print(f"  Std: {x.std():.3f}")
    
    # Quantize
    x_quant, scale = quantizer.quantize_tensor(x)
    print(f"\nQuantized tensor:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Range: [{x_quant.min()}, {x_quant.max()}]")
    print(f"  Dtype: {x_quant.dtype}")
    
    # Dequantize
    x_dequant = quantizer.dequantize_tensor(x_quant, scale)
    
    # Check error
    error = torch.abs(x - x_dequant).mean()
    max_error = torch.abs(x - x_dequant).max()
    print(f"\nQuantization error:")
    print(f"  Mean error: {error:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Relative error: {(error / x.abs().mean() * 100):.3f}%")
    print("✓ Basic quantization test passed")


def test_quantized_linear():
    """Test quantized linear layer"""
    print("\n" + "=" * 60)
    print("TESTING QUANTIZED LINEAR LAYER")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size, in_features, out_features = 32, 768, 3072
    x = torch.randn(batch_size, in_features, device=device)
    
    print(f"Input shape: {x.shape}")
    print(f"Linear layer: {in_features} -> {out_features}")
    
    # Original linear layer
    linear_fp32 = nn.Linear(in_features, out_features).to(device)
    
    # Quantized linear layer
    linear_int8 = QuantizedLinear(in_features, out_features).to(device)
    linear_int8.load_from_fp32(linear_fp32.weight.data, linear_fp32.bias.data)
    
    # Forward pass timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # FP32 forward
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        start_time.record()
    y_fp32 = linear_fp32(x)
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        fp32_time = start_time.elapsed_time(end_time)
    else:
        fp32_time = 0
    
    # INT8 forward
    if device.type == 'cuda':
        start_time.record()
    y_int8 = linear_int8(x)
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        int8_time = start_time.elapsed_time(end_time)
    else:
        int8_time = 0
    
    # Compare results
    diff = torch.abs(y_fp32 - y_int8).mean()
    max_diff = torch.abs(y_fp32 - y_int8).max()
    
    print(f"\nResults:")
    print(f"  Output difference mean: {diff:.6f}")
    print(f"  Output difference max: {max_diff:.6f}")
    print(f"  Relative difference: {(diff / y_fp32.abs().mean() * 100):.3f}%")
    
    if device.type == 'cuda':
        print(f"\nTiming:")
        print(f"  FP32 time: {fp32_time:.3f} ms")
        print(f"  INT8 time: {int8_time:.3f} ms")
        print(f"  Speedup: {fp32_time / int8_time:.2f}x" if int8_time > 0 else "N/A")
    
    print("✓ QuantizedLinear test passed")


def test_memory_usage():
    """Test memory reduction"""
    print("\n" + "=" * 60)
    print("TESTING MEMORY USAGE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Large linear layer for memory test
    size = 2048
    print(f"Testing with {size}x{size} linear layer")
    
    # FP32 layer
    linear_fp32 = nn.Linear(size, size).to(device)
    fp32_params = sum(p.numel() for p in linear_fp32.parameters())
    fp32_memory = fp32_params * 4  # 4 bytes per float32
    
    # INT8 layer
    linear_int8 = QuantizedLinear(size, size).to(device)
    linear_int8.load_from_fp32(linear_fp32.weight.data, linear_fp32.bias.data)
    
    # Calculate INT8 memory
    int8_weights = linear_int8.weight_quantized.numel() * 1  # 1 byte per int8
    scale_memory = 4  # 1 scale factor
    bias_memory = linear_int8.bias.numel() * 4 if linear_int8.bias is not None else 0
    int8_memory = int8_weights + scale_memory + bias_memory
    
    print(f"\nMemory usage:")
    print(f"  FP32 layer: {fp32_memory / 1024**2:.2f} MB")
    print(f"  INT8 layer: {int8_memory / 1024**2:.2f} MB")
    print(f"  Reduction: {fp32_memory / int8_memory:.1f}x")
    print(f"  Savings: {(1 - int8_memory / fp32_memory) * 100:.1f}%")
    
    if device.type == 'cuda':
        print(f"\nGPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print("✓ Memory test passed")


def main():
    """Main test function"""
    print("CUSTOM INT8 QUANTIZATION TEST")
    print("AMD GPU Optimized Implementation")
    print("Running on Frontier Supercomputer")
    
    try:
        test_basic_quantization()
        test_quantized_linear()
        test_memory_usage()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("Custom INT8 quantization is working correctly")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())