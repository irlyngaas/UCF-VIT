"""
Extreme Quantization for AMD GPU
4-bit, 2-bit, and 1-bit quantization without external dependencies
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal
import math


class ExtremeQuantizer:
    """Extreme quantization: 4-bit, 2-bit, 1-bit"""
    
    def __init__(self, bits: Literal[4, 2, 1] = 4, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        self.max_val = (2 ** (bits - 1)) - 1 if symmetric else (2 ** bits) - 1
        self.min_val = -(2 ** (bits - 1)) if symmetric else 0
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to specified bit width
        
        Returns:
            quantized_tensor: quantized tensor (stored as int8 for simplicity)
            scale: FP32 scale factor
        """
        if self.symmetric:
            # Symmetric quantization
            abs_max = tensor.abs().max()
            scale = abs_max / self.max_val
            scale = torch.clamp(scale, min=1e-8)
            
            quantized = torch.round(tensor / scale)
            quantized = torch.clamp(quantized, min=self.min_val, max=self.max_val)
            
        else:
            # Asymmetric quantization
            min_val = tensor.min()
            max_val = tensor.max()
            
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_val / scale)
            zero_point = torch.clamp(zero_point, min=0, max=2 ** self.bits - 1)
            
            quantized = torch.round(tensor / scale + zero_point)
            quantized = torch.clamp(quantized, min=0, max=2 ** self.bits - 1)
        
        return quantized.to(torch.int8), scale
    
    def dequantize_tensor(self, quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to FP32"""
        return quantized_tensor.float() * scale


class BinaryQuantizer:
    """1-bit (binary) quantization with sign-magnitude representation"""
    
    def __init__(self):
        pass
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binary quantization: -1 or +1
        
        Returns:
            quantized_tensor: binary tensor (-1, +1) stored as int8
            scale: FP32 scale factor (mean absolute value)
        """
        # Scale is the mean absolute value
        scale = tensor.abs().mean()
        scale = torch.clamp(scale, min=1e-8)
        
        # Binary quantization: sign of the input
        quantized = torch.sign(tensor)
        quantized = torch.where(quantized == 0, torch.ones_like(quantized), quantized)
        
        return quantized.to(torch.int8), scale
    
    def dequantize_tensor(self, quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize binary tensor back to FP32"""
        return quantized_tensor.float() * scale


class QuantizedLinearExtreme(nn.Module):
    """Extreme quantized linear layer (4-bit, 2-bit, 1-bit)"""
    
    def __init__(self, in_features: int, out_features: int, bits: Literal[4, 2, 1] = 4, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Store quantized weights
        self.register_buffer('weight_quantized', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Choose quantizer based on bit width
        if bits == 1:
            self.quantizer = BinaryQuantizer()
        else:
            self.quantizer = ExtremeQuantizer(bits=bits, symmetric=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using normal distribution"""
        # Create temporary FP32 weight for initialization
        temp_weight = torch.randn(self.out_features, self.in_features)
        temp_weight *= math.sqrt(2.0 / self.in_features)  # He initialization
        
        # Quantize and store
        weight_q, scale = self.quantizer.quantize_tensor(temp_weight)
        self.weight_quantized.copy_(weight_q)
        self.weight_scale.copy_(scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dequantization"""
        # Dequantize weights for computation
        weight_fp32 = self.quantizer.dequantize_tensor(self.weight_quantized, self.weight_scale)
        
        # Standard linear operation
        output = torch.nn.functional.linear(x, weight_fp32, self.bias)
        return output
    
    def load_from_fp32(self, fp32_weight: torch.Tensor, fp32_bias: Optional[torch.Tensor] = None):
        """Load weights from existing FP32 linear layer"""
        weight_q, scale = self.quantizer.quantize_tensor(fp32_weight)
        self.weight_quantized.copy_(weight_q)
        self.weight_scale.copy_(scale)
        
        if fp32_bias is not None and self.bias is not None:
            self.bias.data.copy_(fp32_bias)


def replace_linear_with_extreme_quantized(model: nn.Module, bits: Literal[4, 2, 1] = 4) -> nn.Module:
    """Replace nn.Linear layers with extreme quantized versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create extreme quantized replacement
            quantized_layer = QuantizedLinearExtreme(
                module.in_features, 
                module.out_features, 
                bits=bits,
                bias=module.bias is not None
            )
            
            # Copy weights
            quantized_layer.load_from_fp32(
                module.weight.data,
                module.bias.data if module.bias is not None else None
            )
            
            # Replace in model
            setattr(model, name, quantized_layer)
            
        else:
            # Recursively apply to submodules
            replace_linear_with_extreme_quantized(module, bits)
    
    return model


def quantize_model_extreme(model: nn.Module, bits: Literal[4, 2, 1] = 4) -> nn.Module:
    """
    Apply extreme quantization to model
    
    Args:
        model: PyTorch model to quantize
        bits: Number of bits (4, 2, or 1)
    
    Returns:
        Quantized model
    """
    model = replace_linear_with_extreme_quantized(model, bits=bits)
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    # Assume 4 bytes per parameter for FP32
    total_size_mb = (total_params * 4) / (1024 * 1024)
    return total_size_mb


def get_quantized_model_size_mb(model: nn.Module, bits: int) -> float:
    """Calculate quantized model size in MB"""
    total_params = 0
    for module in model.modules():
        if hasattr(module, 'weight_quantized'):
            total_params += module.weight_quantized.numel()
        elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            total_params += module.weight.numel()
    
    # Calculate size based on bit width
    bits_per_param = bits if bits > 1 else 1
    total_size_mb = (total_params * bits_per_param / 8) / (1024 * 1024)
    return total_size_mb