"""
Custom INT8 Quantization for AMD GPU
Weights-only quantization without external dependencies
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class Int8Quantizer:
    """Simple INT8 quantization for weights"""
    
    def __init__(self, symmetric: bool = True):
        self.symmetric = symmetric
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8
        
        Returns:
            quantized_tensor: INT8 tensor
            scale: FP32 scale factor
        """
        if self.symmetric:
            # Symmetric quantization: -127 to 127
            abs_max = tensor.abs().max()
            scale = abs_max / 127.0
            scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
            
            quantized = torch.round(tensor / scale)
            quantized = torch.clamp(quantized, min=-127, max=127)
            
        else:
            # Asymmetric quantization: -128 to 127
            min_val = tensor.min()
            max_val = tensor.max()
            
            scale = (max_val - min_val) / 255.0
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_val / scale - 128)
            zero_point = torch.clamp(zero_point, min=-128, max=127)
            
            quantized = torch.round(tensor / scale + zero_point)
            quantized = torch.clamp(quantized, min=-128, max=127)
        
        return quantized.to(torch.int8), scale
    
    def dequantize_tensor(self, quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor back to FP32"""
        return quantized_tensor.float() * scale


class QuantizedLinear(nn.Module):
    """INT8 quantized linear layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store original FP32 weights for initialization
        self.register_buffer('weight_quantized', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.quantizer = Int8Quantizer(symmetric=True)
        
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


def replace_linear_with_quantized(model: nn.Module, quantize_layers: list = ['linear']) -> nn.Module:
    """Replace nn.Linear layers with QuantizedLinear"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create quantized replacement
            quantized_layer = QuantizedLinear(
                module.in_features, 
                module.out_features, 
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
            replace_linear_with_quantized(module, quantize_layers)
    
    return model


def quantize_model_int8(model: nn.Module, weights_only: bool = True) -> nn.Module:
    """
    Apply INT8 quantization to model
    
    Args:
        model: PyTorch model to quantize
        weights_only: If True, only quantize weights (not activations)
    
    Returns:
        Quantized model
    """
    if weights_only:
        # Only replace linear layers with quantized versions
        model = replace_linear_with_quantized(model)
    else:
        raise NotImplementedError("Activation quantization not yet implemented")
    
    return model